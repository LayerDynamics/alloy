import type readline from "node:readline";

/** Only explicit yes/no is a decision. EOF, interrupt and abort cancel. */
export function askApproval(rl: readline.Interface, signal: AbortSignal, report: (text: string) => void): Promise<boolean | null> {
  return new Promise((resolve) => {
    const finish = (decision: boolean | null): void => {
      rl.off("line", onLine); rl.off("close", onClose); rl.off("SIGINT", onClose);
      signal.removeEventListener("abort", onClose); resolve(decision);
    };
    const onClose = (): void => finish(null);
    const onLine = (line: string): void => {
      const answer = line.trim().toLowerCase();
      if (answer === "y" || answer === "yes") finish(true);
      else if (answer === "n" || answer === "no") finish(false);
      else report("Type yes to allow, no to deny, or Ctrl-C to cancel.");
    };
    if (signal.aborted || (rl as readline.Interface & { closed?: boolean }).closed) return finish(null);
    rl.on("line", onLine); rl.once("close", onClose); rl.once("SIGINT", onClose);
    signal.addEventListener("abort", onClose, { once: true });
    report("Allow this exact tool call? [yes/no] ");
  });
}
/** Quote terminal controls, including C1 and bidi controls left raw by JSON. */
function preview(value: unknown, limit: number): string {
  const encoded = (JSON.stringify(value) ?? "null").replace(
    /[\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g,
    (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
  return encoded.length > limit ? encoded.slice(0, limit) + "… [preview truncated]" : encoded;
}

/** Keep invocation identity visible even when untrusted arguments fill the preview. */
export function formatToolEvent(event: string, body: Record<string, unknown>): string {
  if (event === "approval_requested" || event === "tool_requested" || event === "tool_call_started") {
    return `${event}: tool_id=${preview(body.tool_id, 512)} call_id=${preview(body.call_id, 512)} effect=${preview(body.effect, 128)} summary=${preview(body.summary, 1024)} arguments=${preview(body.arguments, 4096)}`;
  }
  if (event === "tool_result") {
    const result = body.result !== null && typeof body.result === "object" ? body.result as Record<string, unknown> : body;
    return `${event}: call_id=${preview(body.call_id, 512)} is_error=${preview(body.is_error ?? result.is_error, 16)} result=${preview(result, 4096)} summary=${preview(body.summary, 1024)} hooks=${preview(body.hooks, 2048)}`;
  }
  return `${event}: ${preview(body, 8192)}`;
}
