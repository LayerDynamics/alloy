import readline from "node:readline";
import type { CliArgs } from "./args.js";
import type { AlloySettings } from "./settings.js";
import { RpcClient, type SessionEvent } from "./rpc-client.js";
import { textFromEvent } from "./render.js";

type CreateResult = { session_id: string };
type ResumeResult = { session_id: string; event_count?: number };

export async function runChat(
  settings: AlloySettings,
  args: CliArgs,
): Promise<number> {
  let client: RpcClient;
  try {
    client = await RpcClient.connect(settings.socket);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`alloy: cannot connect to alloyd at ${settings.socket}: ${message}`);
    return 1;
  }

  try {
    let sessionId: string;
    if (args.resume) {
      const resumed = (await client.call("sessions.resume", {
        session_id: args.resume,
      })) as ResumeResult;
      sessionId = resumed.session_id;
      console.error(
        `resumed ${sessionId} (events=${resumed.event_count ?? "?"})`,
      );
    } else {
      const created = (await client.call("session.create", {
        profile: settings.profile,
      })) as CreateResult;
      sessionId = created.session_id;
      console.error(`session ${sessionId} profile=${settings.profile}`);
    }

    const sendOnce = async (content: string): Promise<number> => {
      let failed = false;
      try {
        await client.callStreaming(
          "session.send",
          { session_id: sessionId, content },
          (event: SessionEvent) => {
            if (event.event === "request_failed") {
              failed = true;
              console.error(`\nrequest_failed: ${JSON.stringify(event.body)}`);
              return;
            }
            const text = textFromEvent({
              method: "session.event",
              params: event,
            });
            if (text !== null) {
              process.stdout.write(text);
            }
          },
        );
      } catch (error) {
        console.error(error instanceof Error ? error.message : String(error));
        return 2;
      }
      process.stdout.write("\n");
      return failed ? 3 : 0;
    };

    if (args.message !== undefined) {
      return await sendOnce(args.message);
    }

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stderr,
      terminal: true,
    });
    rl.setPrompt("> ");
    rl.prompt();

    for await (const line of rl) {
      const content = line.trim();
      if (!content) {
        rl.prompt();
        continue;
      }
      const code = await sendOnce(content);
      if (code !== 0) {
        rl.close();
        return code;
      }
      rl.prompt();
    }
    return 0;
  } finally {
    client.close();
  }
}
