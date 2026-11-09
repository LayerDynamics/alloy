import readline from "node:readline";
import { askApproval, formatToolEvent } from "./approval.js";
import type { Readable, Writable } from "node:stream";
import type { CliArgs } from "./args.js";
import type { AlloySettings } from "./settings.js";
import { RpcClient, type SessionEvent } from "./rpc-client.js";
import { formatRequestFailed, textFromEvent } from "./render.js";

type CreateResult = { session_id: string };
type ResumeResult = { session_id: string; event_count?: number };

export interface ChatRpcClient {
  call(method: string, params?: Record<string, unknown>): Promise<unknown>;
  callStreaming(method: string, params: Record<string, unknown>, onEvent: (event: SessionEvent) => void): Promise<unknown>;
  close(): void;
}

export type ChatTurnOptions = {
  control?: Pick<ChatRpcClient, "call">;
  approval?: (body: Record<string, unknown>, signal: AbortSignal) => Promise<boolean | null>;
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
  let controlFailure: Error | undefined;
  let decisions = Promise.resolve();
  const prompts = new AbortController();
  const seenApprovals = new Set<string>();
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
      if (["approval_requested", "tool_requested", "tool_call_started", "tool_result"].includes(event.event)) {
        stopWaiting();
        report(formatToolEvent(event.event, event.body));
        if (event.event === "approval_requested") {
          const callId = event.body.call_id;
          if (typeof callId !== "string" || !callId || seenApprovals.has(callId)) return;
          seenApprovals.add(callId);
          decisions = decisions.then(async () => {
            if (settled || prompts.signal.aborted) return;
            if (!options.control) throw new Error("Approval requires a separate control connection");
            const allow = options.approval ? await options.approval(event.body, prompts.signal) : null;
            if (prompts.signal.aborted) return;
            if (allow === null) {
              report("No explicit approval decision; cancelling the turn.");
              await options.control.call("session.pause", { session_id: sessionId });
            } else {
              await options.control.call("session.approve", { session_id: sessionId, call_id: callId, allow });
            }
          }).catch(async (error: unknown) => {
            controlFailure = error instanceof Error ? error : new Error(String(error));
            report(controlFailure.message);
            await options.control?.call("session.pause", { session_id: sessionId }).catch(() => {});
          });
        }
        return;
      }
      if (event.event === "request_failed") {
        failed = true;
        stopWaiting();
        prompts.abort();
        report(formatRequestFailed(event.body));
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
    prompts.abort();
    await decisions;
    if (failed || controlFailure) return 3;
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
    // A request_failed notification is terminal and already reported. Its
    // matching final RPC rejection must not print the same failure again.
    if (failed) return 3;
    report(error instanceof Error ? error.message : String(error));
    return 2;
  } finally {
    settled = true;
    prompts.abort();
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

  let control: ChatRpcClient | undefined;
  let rl: readline.Interface | undefined;
  let askingApproval = false;
  let sending = false;
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

    const input = options.input ?? process.stdin;
    const interactive = (input as Readable & { isTTY?: boolean }).isTTY === true || typeof (input as Readable & { setRawMode?: unknown }).setRawMode === "function";
    if (interactive) {
      rl = readline.createInterface({ input, output: errorOutput, terminal: true });
      rl.on("SIGINT", () => {
        if (askingApproval) return;
        if (sending) void control?.call("session.pause", { session_id: sessionId }).catch((error: unknown) => report(String(error)));
        rl?.close();
      });
      rl.on("close", () => {
        if (sending && !askingApproval) void control?.call("session.pause", { session_id: sessionId }).catch((error: unknown) => report(String(error)));
      });
    }
    const sendOnce = async (content: string): Promise<number> => {
      try {
      control ??= await (options.connect ?? RpcClient.connect)(settings.socket);
      sending = true;
      return await sendChatTurn(client, sessionId, content, { ...options, control,
        approval: options.approval ?? (interactive && rl ? async (_body, signal) => {
          askingApproval = true;
          try { return await askApproval(rl!, signal, report); }
          finally { askingApproval = false; }
        } : async () => null),
      });
      } catch (error) {
        report(error instanceof Error ? error.message : String(error)); return 2;
      } finally { sending = false; }
    };

    if (args.message !== undefined) {
      return await sendOnce(args.message);
    }

    rl ??= readline.createInterface({
      input: options.input ?? process.stdin,
      output: errorOutput,
      terminal: interactive,
    });
    const terminal = rl;
    const queued: string[] = [];
    let ended = false;
    let waiter: ((value: string | null) => void) | undefined;
    terminal.on("line", (line: string) => {
      if (askingApproval) return;
      if (waiter) { const done = waiter; waiter = undefined; done(line); }
      else queued.push(line);
    });
    terminal.once("close", () => { ended = true; waiter?.(null); waiter = undefined; });
    const nextLine = (): Promise<string | null> => {
      if (queued.length) return Promise.resolve(queued.shift()!);
      if (ended) return Promise.resolve(null);
      return new Promise((resolve) => { waiter = resolve; terminal.setPrompt("> "); terminal.prompt(); });
    };
    while (true) {
      const line = await nextLine();
      if (line === null) return 0;
      const content = line.trim();
      if (!content) continue;
      const code = await sendOnce(content);
      if (code !== 0) return code;
    }
  } finally {
    rl?.close();
    if (control !== client) control?.close();
    client.close();
  }
}
