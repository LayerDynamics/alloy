import test from "node:test";
import assert from "node:assert/strict";
import readline from "node:readline";
import { PassThrough, Writable } from "node:stream";
import { askApproval, formatToolEvent } from "../src/approval.js";
import { sendChatTurn } from "../src/chat.js";
import type { SessionEvent } from "../src/rpc-client.js";
const capture = () => { let text = ""; return { stream: new Writable({ write(c,_e,cb) { text += c.toString(); cb(); } }), text: () => text }; };

test("cli_renders_approval_and_tool_result", async () => {
  for (const allow of [true, false, null]) {
    const output = capture(), errors = capture();
    const calls: Array<[string, Record<string, unknown>]> = [];
    let emit!: (e: SessionEvent) => void;
    let finish!: (v: unknown) => void;
    const done = new Promise<unknown>((r) => { finish = r; });
    const pending = sendChatTurn({ callStreaming: async (_m,_p,onEvent) => { emit = onEvent; return done; } }, "s", "use tool", {
      output: output.stream, errorOutput: errors.stream,
      approval: async () => allow,
      control: { call: async (m,p = {}) => {
        calls.push([m,p]);
        if (allow === null) { emit({ session_id: "s", event: "request_failed", body: { message: "cancelled" } }); finish({}); }
        else {
          emit({ session_id: "s", event: "tool_result", body: { call_id: "c", result: { is_error: !allow, content: "untrusted\u001b[31m" } } });
          emit({ session_id: "s", event: "text_delta", body: { text: "done" } });
          finish({ ok: true, session_id: "s", assistant_text: "done", input_tokens: 1 });
        }
        return { ok: true };
      } },
    });
    emit({ session_id: "s", event: "approval_requested", body: { call_id: "c", tool_id: "shell.run", effect: "destructive", arguments: { executable: "/bin/echo", args: ["literal"] } } });
    assert.equal(await pending, allow === null ? 3 : 0);
    assert.deepEqual(calls, allow === null ? [["session.pause", { session_id: "s" }]] : [["session.approve", { session_id: "s", call_id: "c", allow }]]);
    assert.match(errors.text(), /approval_requested.*shell.run/);
    if (allow !== null) assert.match(errors.text(), /tool_result.*untrusted\\u001b/);
    assert.doesNotMatch(output.text(), /untrusted|shell.run/);
  }
});
test("approval requires real yes or no; EOF, abort and Ctrl-C cancel", async () => {
  for (const mode of ["yes", "no", "eof", "abort", "interrupt"]) {
    const input = new PassThrough(); const rl = readline.createInterface({ input });
    const cancel = new AbortController(); const messages: string[] = [];
    const pending = askApproval(rl, cancel.signal, (s) => messages.push(s));
    input.write("maybe\n");
    if (mode === "eof") input.end();
    else if (mode === "abort") cancel.abort();
    else if (mode === "interrupt") rl.emit("SIGINT");
    else input.write(mode + "\n");
    assert.equal(await pending, mode === "yes" ? true : mode === "no" ? false : null);
    assert.match(messages.join("\n"), /Type yes/); rl.close();
  }
});
test("tool previews quote escapes and bound output", () => {
  assert.doesNotMatch(formatToolEvent("tool_result", { text: "\u001b[2J" }), /\u001b/);
  assert.ok(formatToolEvent("tool_result", { text: "x".repeat(100000) }).length < 8300);
});
test("noninteractive input cancels without inventing an approval", async () => {
  let emit!: (e: SessionEvent) => void; let finish!: (v: unknown) => void;
  const calls: string[] = []; const done = new Promise((r) => { finish = r; });
  const pending = sendChatTurn({ callStreaming: async (_m,_p,f) => { emit = f; return done; } }, "s", "tool", {
    output: capture().stream, errorOutput: capture().stream,
    control: { call: async (method) => { calls.push(method); emit({ session_id: "s", event: "request_failed", body: { message: "cancelled" } }); finish({}); } },
  });
  emit({ session_id: "s", event: "approval_requested", body: { call_id: "c" } });
  assert.equal(await pending, 3); assert.deepEqual(calls, ["session.pause"]);
});

test("CLI sends the real readline decision over a second socket while send waits", { timeout: 10000 }, async () => {
  const net = await import("node:net");
  const fs = await import("node:fs/promises");
  const os = await import("node:os");
  const path = await import("node:path");
  const { runChat } = await import("../src/chat.js");
  const { parseArgv } = await import("../src/args.js");
  const { defaultSettings } = await import("../src/settings.js");
  for (const answer of ["yes", "no", "eof"]) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "alloy-w6-"));
    const socket = path.join(dir, "rpc.sock");
    const peers = new Set<import("node:net").Socket>();
    let sendPeer: import("node:net").Socket | undefined;
    let sendId: unknown;
    let connections = 0;
    const calls: string[] = [];
    const reply = (peer: import("node:net").Socket, value: unknown): void => { peer.write(JSON.stringify(value) + "\n"); };
    const event = (name: string, body: object): void => reply(sendPeer!, { jsonrpc: "2.0", method: "session.event", params: { session_id: "s", event: name, body } });
    const server = net.createServer((peer) => {
      connections++; peers.add(peer); let buffer = "";
      peer.on("data", (chunk) => {
        buffer += chunk.toString();
        let end: number;
        while ((end = buffer.indexOf("\n")) >= 0) {
          const message = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
          const result = (value: unknown): void => reply(peer, { jsonrpc: "2.0", id: message.id, result: value });
          if (message.method === "session.create") result({ session_id: "s" });
          else if (message.method === "session.send") {
            sendPeer = peer; sendId = message.id;
            event("approval_requested", { call_id: "c", tool_id: "shell.run", arguments: { executable: "/bin/echo", args: ["literal"] } });
          } else if (message.method === "session.approve") {
            assert.notEqual(peer, sendPeer); assert.equal(message.params.allow, answer === "yes"); calls.push(message.method);
            result({ ok: true }); event("tool_result", { call_id: "c", result: { is_error: !message.params.allow, content: "result" } });
            event("text_delta", { text: "finished" }); event("generation_completed", {});
            reply(sendPeer!, { jsonrpc: "2.0", id: sendId, result: { ok: true, session_id: "s", assistant_text: "finished", input_tokens: 1 } });
          } else if (message.method === "session.pause") {
            calls.push(message.method); result({ ok: true }); event("request_failed", { message: "cancelled" });
            reply(sendPeer!, { jsonrpc: "2.0", id: sendId, error: { code: -32000, message: "cancelled" } });
          }
        }
      });
    });
    await new Promise<void>((resolve) => server.listen(socket, resolve));
    const input = new PassThrough() as PassThrough & { isTTY: boolean; setRawMode(value: boolean): PassThrough };
    input.isTTY = true; input.setRawMode = () => input;
    let decided = false;
    const errors = new Writable({ write(chunk,_encoding,callback) {
      if (!decided && chunk.toString().includes("[yes/no]")) {
        decided = true; queueMicrotask(() => { if (answer === "eof") input.end(); else input.write(answer + "\n"); });
      }
      callback();
    } });
    try {
      const code = await runChat({ ...defaultSettings(), socket }, parseArgv(["node", "alloy", "-m", "tool"]), { input, output: capture().stream, errorOutput: errors });
      assert.equal(code, answer === "eof" ? 3 : 0); assert.equal(connections, 2);
      assert.deepEqual(calls, [answer === "eof" ? "session.pause" : "session.approve"]);
    } finally {
      input.destroy(); for (const peer of peers) peer.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve())); await fs.rm(dir, { recursive: true, force: true });
    }
  }
});

test("approval identity stays visible with oversized arguments and terminal controls stay quoted", () => {
  const rendered = formatToolEvent("approval_requested", {
    arguments: { args: ["x".repeat(32_000)], executable: "/bin/echo" },
    call_id: "exact-call-id", effect: "destructive", tool_id: "shell.run", summary: "exact invocation",
  });
  assert.match(rendered, /tool_id="shell.run" call_id="exact-call-id" effect="destructive"/);
  assert.match(rendered, /preview truncated/);
  assert.ok(rendered.length < 8300);
  const control = formatToolEvent("tool_result", { text: "\u009b2J\u202e\u2066untrusted" });
  assert.doesNotMatch(control, /[\u007f-\u009f\u202a-\u202e\u2066-\u2069]/);
  assert.match(control, /\\u009b2J\\u202e\\u2066untrusted/);
  const artifact = formatToolEvent("tool_result", {
    call_id: "exact-call-id", is_error: true, hooks: [{ annotations: "x".repeat(32_000) }],
    result: { call_id: "exact-call-id", artifact_ref: "artifact-visible", is_error: true },
  });
  assert.match(artifact, /is_error=true result=.*artifact-visible/);
  assert.ok(artifact.length < 8300);
});
