import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import net from "node:net";
import { RpcClient } from "../src/rpc-client.js";

class FakeSocket extends EventEmitter {
  destroyed = false;
  setEncoding(): this { return this; }
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
