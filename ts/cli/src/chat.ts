import readline from "node:readline";
import type { Readable, Writable } from "node:stream";
import type { CliArgs } from "./args.js";
import type { AlloySettings } from "./settings.js";
import { RpcClient, type SessionEvent } from "./rpc-client.js";
import { textFromEvent } from "./render.js";

type CreateResult = { session_id: string };
type ResumeResult = { session_id: string; event_count?: number };

export interface ChatRpcClient {
  call(method: string, params?: Record<string, unknown>): Promise<unknown>;
  callStreaming(method: string, params: Record<string, unknown>, onEvent: (event: SessionEvent) => void): Promise<unknown>;
  close(): void;
}

export type ChatTurnOptions = {
  output?: Writable;
  errorOutput?: Writable;
  waitingInitialMs?: number;
  waitingRepeatMs?: number;
};

export type ChatOptions = ChatTurnOptions & {
  connect?: (socket: string) => Promise<ChatRpcClient>;
  input?: Readable;
};

function finalAssistantText(value: unknown, sessionId: string): string {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid session.send final response");
  const result = value as Record<string, unknown>;
  if (result.ok !== true) throw new Error("session.send did not report a successful final response");
  if (result.session_id !== sessionId) throw new Error("session.send final response does not match the active session");
  if (typeof result.assistant_text !== "string" || !result.assistant_text.trim()) throw new Error("session.send completed without assistant text");
  if (!Number.isSafeInteger(result.input_tokens) || (result.input_tokens as number) < 0) throw new Error("session.send returned invalid input_tokens");
  return result.assistant_text;
}

/** One submitted turn; status stays on stderr and never invents an engine stage. */
export async function sendChatTurn(
  client: Pick<ChatRpcClient, "callStreaming">,
  sessionId: string,
  content: string,
  options: ChatTurnOptions = {},
): Promise<number> {
  const output = options.output ?? process.stdout;
  const errorOutput = options.errorOutput ?? process.stderr;
  const firstMs = options.waitingInitialMs ?? 5000;
  const repeatMs = options.waitingRepeatMs ?? 15_000;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let waiting = true;
  let settled = false;
  let failed = false;
  let streamed = "";
  const report = (text: string): void => { errorOutput.write(`${text}\n`); };
  const stopWaiting = (): void => { waiting = false; clearTimeout(timer); timer = undefined; };
  const notice = (): void => {
    if (!waiting || settled) return;
    report("Waiting for the first response…");
    if (waiting && !settled) { timer = setTimeout(notice, repeatMs); timer.unref(); }
  };
  try {
    if (typeof sessionId !== "string" || !sessionId) throw new Error("Cannot send a turn without a valid session_id");
    if (![firstMs, repeatMs].every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error("Waiting notice intervals must be positive milliseconds");
    timer = setTimeout(notice, firstMs);
    timer.unref();
    const result = await client.callStreaming("session.send", { session_id: sessionId, content }, (event) => {
      // A connection may carry unrelated session notifications. They must not
      // print text, stop this turn's notice, or change its terminal outcome.
      if (settled || failed || event.session_id !== sessionId) return;
      if (event.event === "request_failed") {
        failed = true;
        stopWaiting();
        report(`request_failed: ${JSON.stringify(event.body)}`);
        return;
      }
      if (event.event === "runtime_warning") {
        report(`runtime_warning: ${JSON.stringify(event.body)}`);
        return;
      }
      if (event.event === "generation_completed") {
        stopWaiting();
        return;
      }
      const text = textFromEvent({ method: "session.event", params: event });
      if (text !== null && text.length > 0) {
        stopWaiting();
        streamed += text;
        output.write(text);
      }
    });
    stopWaiting();
    if (failed) return 3;
    const finalText = finalAssistantText(result, sessionId);
    if (streamed.length === 0) {
      report("Received the final response without streamed text.");
      output.write(finalText);
    } else if (streamed !== finalText) {
      throw new Error("session.send final assistant_text does not match the streamed response");
    }
    output.write("\n");
    return 0;
  } catch (error) {
    report(error instanceof Error ? error.message : String(error));
    return 2;
  } finally {
    settled = true;
    stopWaiting();
  }
}

export async function runChat(
  settings: AlloySettings,
  args: CliArgs,
  options: ChatOptions = {},
): Promise<number> {
  const errorOutput = options.errorOutput ?? process.stderr;
  const report = (text: string): void => { errorOutput.write(`${text}\n`); };
  let client: ChatRpcClient;
  try {
    client = await (options.connect ?? RpcClient.connect)(settings.socket);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    report(`alloy: cannot connect to alloyd at ${settings.socket}: ${message}`);
    return 1;
  }

  try {
    let sessionId: string;
    if (args.resume) {
      const resumed = (await client.call("sessions.resume", {
        session_id: args.resume,
      })) as ResumeResult;
      sessionId = resumed.session_id;
      report(
        `resumed ${sessionId} (events=${resumed.event_count ?? "?"})`,
      );
    } else {
      const created = (await client.call("session.create", {
        profile: settings.profile,
      })) as CreateResult;
      sessionId = created.session_id;
      report(`session ${sessionId} profile=${settings.profile}`);
    }

    const sendOnce = (content: string): Promise<number> => sendChatTurn(client, sessionId, content, options);

    if (args.message !== undefined) {
      return await sendOnce(args.message);
    }

    const rl = readline.createInterface({
      input: options.input ?? process.stdin,
      output: errorOutput,
      terminal: true,
    });
    rl.setPrompt("> ");
    rl.prompt();

    try {
      for await (const line of rl) {
        const content = line.trim();
        if (!content) {
          rl.prompt();
          continue;
        }
        const code = await sendOnce(content);
        if (code !== 0) return code;
        rl.prompt();
      }
      return 0;
    } finally {
      rl.close();
    }
  } finally {
    client.close();
  }
}
