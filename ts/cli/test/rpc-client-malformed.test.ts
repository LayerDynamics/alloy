import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import net from "node:net";
import { RpcClient } from "../src/rpc-client.js";

class ResponseSocket extends EventEmitter {
  lastRequestId = 0;
  setEncoding(): this { return this; }
  write(line: string, callback: (error?: Error | null) => void): boolean {
    this.lastRequestId = (JSON.parse(line) as { id: number }).id;
    queueMicrotask(() => { callback(null); });
    return true;
  }
  destroy(): this {
    queueMicrotask(() => { this.emit("close"); });
    return this;
  }
}

for (const root of ["null", "42", '"text"', "true", "[]"]) {
  test(`RPC root ${root} rejects pending reads without throwing from the socket handler`, async (t) => {
    const socket = new ResponseSocket();
    t.mock.method(net, "createConnection", () => socket as unknown as net.Socket);
    const connected = RpcClient.connect("/test.sock");
    socket.emit("connect");
    const client = await connected;
    try {
      const pending = client.call("daemon.metrics");
      const rejected = assert.rejects(pending, /malformed JSON-RPC line: expected an object/);
      assert.doesNotThrow(() => { socket.emit("data", `${root}\n`); });
      await rejected;
    } finally {
      client.close();
    }
  });
}

test("invalid roots reject streaming calls and release their stream ownership", async (t) => {
  const socket = new ResponseSocket();
  t.mock.method(net, "createConnection", () => socket as unknown as net.Socket);
  const connected = RpcClient.connect("/test.sock");
  socket.emit("connect");
  const client = await connected;
  try {
    const failed = client.callStreaming("session.send", {}, () => {});
    const rejected = assert.rejects(failed, /expected an object/);
    assert.doesNotThrow(() => { socket.emit("data", "null\n"); });
    await rejected;
    const next = client.callStreaming("session.send", {}, () => {});
    socket.emit("data", JSON.stringify({ jsonrpc: "2.0", id: socket.lastRequestId, result: { ok: true } }) + "\n");
    assert.deepEqual(await next, { ok: true });
  } finally {
    client.close();
  }
});
