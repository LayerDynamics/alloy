import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import net from "node:net";
import { RpcClient, RpcError } from "../src/rpc-client.js";

class FakeSocket extends EventEmitter {
  destroyed = false;
  writes: string[] = [];
  setEncoding(): this { return this; }
  write(value: string, callback: (error?: Error) => void): boolean { this.writes.push(value); callback(); return true; }
  destroy(): this { this.destroyed = true; queueMicrotask(() => this.emit("close")); return this; }
}

test("connection timeout destroys the connecting socket and removes listeners", async (t) => {
  const socket = new FakeSocket();
  t.mock.method(net, "createConnection", () => socket as unknown as net.Socket);
  await assert.rejects(RpcClient.connect("/test.sock", { timeoutMs: 20 }), /timed out/);
  assert.equal(socket.destroyed, true);
  assert.equal(socket.listenerCount("connect"), 0);
  assert.equal(socket.listenerCount("error"), 0);
});

test("RPC failures retain code and structured data without retrying or rewriting unknown causes", async (t) => {
  const socket = new FakeSocket();
  t.mock.method(net, "createConnection", () => socket as unknown as net.Socket);
  const connected = RpcClient.connect("/test.sock");
  socket.emit("connect");
  const client = await connected;
  try {
    for (const engine_error of [
      { kind: "protected_resources", message: "pinned model", operation: "generation_startup", resources: [{ protection: "model_pin", artifact_id: "model-a", instance_id: "7", handles: 1 }] },
      { kind: "future_server_kind", message: "future server explanation", future_field: { retained: true } },
    ]) {
      const error = { code: -32000, message: engine_error.message, data: { kind: "engine_failed", engine_error } };
      const pending = client.call("session.send", { session_id: "s", content: "hello" });
      const request = JSON.parse(socket.writes.at(-1)!);
      const rejected = assert.rejects(pending, (caught: unknown) => {
        assert.ok(caught instanceof RpcError);
        assert.equal(caught.code, error.code);
        assert.deepEqual(caught.data, error.data);
        assert.ok(caught.message.includes(error.message));
        return true;
      });
      socket.emit("data", JSON.stringify({ jsonrpc: "2.0", id: request.id, error }) + "\n");
      await rejected;
    }
    assert.equal(socket.writes.length, 2, "each user request produces one RPC without automatic retry");
  } finally {
    client.close();
  }
});

test("aborting a pending connection destroys it without waiting for the deadline", async (t) => {
  const socket = new FakeSocket();
  t.mock.method(net, "createConnection", () => socket as unknown as net.Socket);
  const abort = new AbortController();
  const pending = RpcClient.connect("/test.sock", { timeoutMs: 1000, signal: abort.signal });
  abort.abort();
  await assert.rejects(pending, /cancelled/);
  assert.equal(socket.destroyed, true);
  assert.equal(socket.listenerCount("connect"), 0);
});

test("already-aborted connection requests do not create a socket", async (t) => {
  const create = t.mock.method(net, "createConnection", () => { throw new Error("must not connect"); });
  const abort = new AbortController();
  abort.abort();
  await assert.rejects(RpcClient.connect("/test.sock", { signal: abort.signal }), /cancelled/);
  assert.equal(create.mock.callCount(), 0);
});

test("existing untimed connections still resolve and hand ownership to RpcClient", async (t) => {
  const socket = new FakeSocket();
  t.mock.method(net, "createConnection", () => socket as unknown as net.Socket);
  const pending = RpcClient.connect("/test.sock");
  socket.emit("connect");
  const client = await pending;
  assert.equal(socket.destroyed, false);
  client.close();
  assert.equal(socket.destroyed, true);
});
