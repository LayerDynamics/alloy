import test from "node:test";
import assert from "node:assert/strict";
import { PassThrough, Writable } from "node:stream";
import { setImmediate as nextTurn } from "node:timers/promises";
import { runChat, sendChatTurn, type ChatRpcClient } from "../src/chat.js";
import { parseArgv } from "../src/args.js";
import { defaultSettings } from "../src/settings.js";
import { RpcError, type SessionEvent } from "../src/rpc-client.js";

const SESSION = "sess-test";
const final = (text: string) => ({ ok: true, session_id: SESSION, assistant_text: text, input_tokens: 3 });
function capture() {
  let text = "";
  return { stream: new Writable({ write(chunk, _encoding, callback) { text += chunk.toString(); callback(); } }), text: () => text };
}
class PendingTurn {
  onEvent!: (event: SessionEvent) => void;
  resolve!: (value: unknown) => void;
  reject!: (error: Error) => void;
  readonly promise = new Promise<unknown>((resolve, reject) => { this.resolve = resolve; this.reject = reject; });
  callStreaming = (_method: string, _params: Record<string, unknown>, onEvent: (event: SessionEvent) => void): Promise<unknown> => {
    this.onEvent = onEvent;
    return this.promise;
  };
  emit(event: string, body: Record<string, unknown> = {}, session_id = SESSION): void { this.onEvent({ session_id, event, body }); }
}

test("waiting notices start at five seconds, repeat every fifteen, and stop on first text or settlement", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  t.mock.timers.tick(4999);
  assert.equal(errors.text(), "");
  t.mock.timers.tick(1);
  assert.equal(errors.text().match(/Waiting for the first response/g)?.length, 1);
  t.mock.timers.tick(14_999);
  assert.equal(errors.text().match(/Waiting for the first response/g)?.length, 1);
  t.mock.timers.tick(1);
  assert.equal(errors.text().match(/Waiting for the first response/g)?.length, 2);
  turn.emit("text_delta", { text: "Hello" });
  const before = errors.text();
  t.mock.timers.tick(60_000);
  assert.equal(errors.text(), before);
  turn.resolve(final("Hello"));
  assert.equal(await pending, 0);
  turn.emit("text_delta", { text: "late duplicate" });
  turn.emit("runtime_warning", { message: "late warning" });
  t.mock.timers.tick(60_000);
  assert.equal(output.text(), "Hello\n");
  assert.equal(errors.text(), before);
});

test("runtime warnings remain visible on stderr while successful streamed text is printed exactly once", async () => {
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  turn.emit("runtime_warning", { code: "fixture_warning", message: "warning details" });
  turn.emit("text_delta", { text: "Hello " });
  turn.emit("text_delta", { text: "world" });
  turn.resolve(final("Hello world"));
  assert.equal(await pending, 0);
  assert.equal(output.text(), "Hello world\n");
  assert.match(errors.text(), /runtime_warning.*fixture_warning.*warning details/);
  assert.doesNotMatch(output.text(), /runtime_warning|Waiting|session/);
});

test("unrelated session text, warning and failure cannot affect the selected turn or its waiting notice", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  turn.emit("text_delta", { text: "unrelated text" }, "another-session");
  turn.emit("runtime_warning", { message: "unrelated warning" }, "another-session");
  turn.emit("request_failed", { message: "unrelated failure" }, "another-session");
  assert.equal(output.text(), "");
  assert.equal(errors.text(), "");
  t.mock.timers.tick(5000);
  assert.match(errors.text(), /Waiting for the first response/);
  turn.emit("text_delta", { text: "Selected answer" });
  turn.resolve(final("Selected answer"));
  assert.equal(await pending, 0);
  assert.equal(output.text(), "Selected answer\n");
  assert.doesNotMatch(errors.text(), /unrelated/);
});

test("an engine RPC error cancels waiting and cannot produce a blank successful answer", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  t.mock.timers.tick(5000);
  turn.reject(new Error("RPC -32000 (engine_failed): timed out waiting for engine future"));
  assert.equal(await pending, 2);
  assert.equal(output.text(), "");
  assert.match(errors.text(), /engine_failed.*timed out waiting for engine future/);
  const before = errors.text();
  t.mock.timers.tick(60_000);
  assert.equal(errors.text(), before);
});

test("request_failed stops notices before RPC settlement and suppresses later text or a contradictory success", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  turn.emit("request_failed", { message: "execution cancelled" });
  turn.emit("text_delta", { text: "must not appear" });
  t.mock.timers.tick(60_000);
  assert.doesNotMatch(errors.text(), /Waiting/);
  turn.resolve(final("must not appear"));
  assert.equal(await pending, 3);
  assert.equal(output.text(), "");
  assert.match(errors.text(), /request_failed.*execution cancelled/);
});

test("a structured terminal event and matching final RPC failure are reported once without retry", async () => {
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const diagnostic = { kind: "protected_resources", message: "model A remains pinned", operation: "generation_startup", resources: [{ protection: "model_pin", artifact_id: "model-a", instance_id: "9", handles: 1 }] };
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  turn.emit("text_delta", { text: "partial" });
  turn.emit("request_failed", { message: diagnostic.message, engine_error: diagnostic, partial_usage: { input_tokens: 1, output_tokens: 1 } });
  turn.reject(new RpcError({ code: -32000, message: diagnostic.message, data: { kind: "engine_failed", engine_error: diagnostic } }));
  assert.equal(await pending, 3);
  assert.equal(output.text(), "partial");
  assert.equal(errors.text().match(/model A remains pinned/g)?.length, 1);
  assert.equal(errors.text().match(/caller holding these model pins/g)?.length, 1);
  assert.doesNotMatch(errors.text(), /Wait for the active request/);
});

test("an immediate startup rejection keeps its typed guidance and cannot produce a successful answer", async () => {
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  const diagnostic = { kind: "memory_limit_exceeded", message: "startup requires 200 bytes; 100 available", reason: "host_preflight", required_bytes: 200, available_bytes: 100 };
  turn.reject(new RpcError({ code: -32000, message: diagnostic.message, data: { kind: "engine_failed", engine_error: diagnostic } }));
  assert.equal(await pending, 2);
  assert.equal(output.text(), "");
  assert.match(errors.text(), /planned startup exceeds the current safe memory ceiling/);
  assert.match(errors.text(), /requires 200 bytes; 100 available/);
});

test("generation completion stops waiting and a valid final-only answer is shown once with an explanatory notice", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const turn = new PendingTurn();
  const output = capture(), errors = capture();
  const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
  turn.emit("generation_completed");
  t.mock.timers.tick(60_000);
  assert.equal(errors.text(), "");
  turn.resolve(final("Actual final text"));
  assert.equal(await pending, 0);
  assert.equal(output.text(), "Actual final text\n");
  assert.match(errors.text(), /final response without streamed text/);
});

test("empty, wrong-session, malformed and inconsistent final results fail instead of returning a successful prompt", async () => {
  const cases = [
    { result: final(""), streamed: "", reason: /without assistant text/ },
    { result: final(" \n"), streamed: "", reason: /without assistant text/ },
    { result: { ...final("answer"), session_id: "other" }, streamed: "", reason: /active session/ },
    { result: final("different answer"), streamed: "streamed answer", reason: /does not match the streamed/ },
    { result: { ...final("answer"), input_tokens: -1 }, streamed: "", reason: /invalid input_tokens/ },
    { result: null, streamed: "", reason: /Invalid session.send/ },
  ];
  for (const example of cases) {
    const turn = new PendingTurn();
    const output = capture(), errors = capture();
    const pending = sendChatTurn(turn, SESSION, "hello", { output: output.stream, errorOutput: errors.stream });
    if (example.streamed) turn.emit("text_delta", { text: example.streamed });
    turn.resolve(example.result);
    assert.equal(await pending, 2);
    assert.equal(output.text(), example.streamed);
    assert.match(errors.text(), example.reason);
  }
});

test("one-shot chat keeps stdout exclusively model text and closes its connection", async () => {
  const output = capture(), errors = capture();
  let closed = 0;
  const client: ChatRpcClient = {
    call: async () => ({ session_id: SESSION }),
    callStreaming: async (_method, _params, event) => {
      event({ session_id: SESSION, event: "text_delta", body: { text: "Actual answer" } });
      return final("Actual answer");
    },
    close: () => { closed++; },
  };
  const result = await runChat(defaultSettings(), parseArgv(["node", "alloy", "-m", "hello"]), {
    connect: async () => client, output: output.stream, errorOutput: errors.stream,
  });
  assert.equal(result, 0);
  assert.equal(output.text(), "Actual answer\n");
  assert.match(errors.text(), /session sess-test profile=/);
  assert.equal(closed, 1);
});

test("interactive requests remain serialized and an engine error closes readline and restores raw mode", async (t) => {
  const input = new PassThrough() as PassThrough & { isRaw: boolean; setRawMode(value: boolean): PassThrough };
  input.isRaw = false;
  const raw: boolean[] = [];
  input.setRawMode = (value) => { input.isRaw = value; raw.push(value); return input; };
  t.after(() => input.destroy());
  const output = capture(), errors = capture();
  const first = new PendingTurn(), second = new PendingTurn();
  const sent: string[] = [];
  let closed = 0;
  const client: ChatRpcClient = {
    call: async () => ({ session_id: SESSION }),
    callStreaming: (method, params, event) => {
      sent.push(params.content as string);
      return (sent.length === 1 ? first : second).callStreaming(method, params, event);
    },
    close: () => { closed++; },
  };
  const pending = runChat(defaultSettings(), parseArgv(["node", "alloy", "chat"]), {
    connect: async () => client, input, output: output.stream, errorOutput: errors.stream,
  });
  await nextTurn();
  input.write("first\rsecond\r");
  await nextTurn();
  assert.deepEqual(sent, ["first"]);
  first.emit("text_delta", { text: "First answer" });
  first.resolve(final("First answer"));
  await nextTurn();
  assert.deepEqual(sent, ["first", "second"]);
  second.reject(new Error("RPC -32000 (engine_failed): engine timeout"));
  assert.equal(await pending, 2);
  assert.equal(output.text(), "First answer\n");
  assert.match(errors.text(), /engine_failed.*engine timeout/);
  assert.deepEqual(raw, [true, false]);
  assert.equal(input.isRaw, false);
  assert.equal(closed, 1);
});
