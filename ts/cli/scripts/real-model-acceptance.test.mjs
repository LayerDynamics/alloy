import assert from "node:assert/strict";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { assertCacheRestore, assertCacheSpill, assertCandidateCheckout, assertRecoveredHistory, assertStreamingTurn, recordProxy, rpc } from "./real-model-acceptance.mjs";

const SESSION = "sess-7";
const PROMPT = "Remember copper-orchid-47.";
const TEXT = "café marker saved";
const clone = (value) => structuredClone(value);
const event = (kind, body) => ({ jsonrpc: "2.0", method: "session.event", params: { session_id: SESSION, event: kind, body } });
const final = () => ({ jsonrpc: "2.0", id: 1, result: { ok: true, session_id: SESSION, assistant_text: TEXT, input_tokens: 12 } });
const responseMessages = () => [
  event("text_delta", { text: "café " }),
  event("text_delta", { text: "marker saved" }),
  event("generation_completed", { finish_reason: "end_of_sequence", usage: { input_tokens: 12, output_tokens: 5 } }),
  final(),
];
function stream() {
  const messages = [{ jsonrpc: "2.0", api_version: 1, id: 1, method: "session.send", params: { session_id: SESSION, content: PROMPT } }, ...responseMessages()];
  return messages.map((message, i) => ({ sequence: i + 1, at_ms: 100 + i * 5, connection_id: 7, direction: i === 0 ? "request" : "response", message }));
}
const stdoutChunks = () => [
  { at_ms: 106, text: "café " },
  { at_ms: 111, text: "marker saved" },
  { at_ms: 121, text: "\n" },
];
function history() {
  return { session_id: SESSION, profile_id: "qwen3-4b-4bit", event_count: 4, events: [
    { seq: 1, kind: "session.created", payload: {} },
    { seq: 2, kind: "user.message", payload: { role: "user", content: PROMPT } },
    { seq: 3, kind: "assistant.message", payload: { role: "assistant", content: TEXT } },
    { seq: 4, kind: "run.completed", payload: { input_tokens: 12 } },
  ] };
}
const turn = { session_id: SESSION, prompt: PROMPT, assistant_text: TEXT };
const cacheMetrics = () => ({ admission: { reserved_bytes: 128 * 1024 * 1024 }, kv_cache: { enabled: true, disk_enabled: true, spill_count: 1, restore_count: 1, restored_tokens: 12, cold_entries: 1, hot_limit_bytes: 0, hot_reserved_bytes: 0, protected_reserved_bytes: 0, transfer_reserved_bytes: 64 * 1024 * 1024, last_error: null } });

test("acceptance requires a clean full commit and an exact optional expected commit", () => {
  const candidate = { commit: "1".repeat(40), status: "" };
  assert.deepEqual(assertCandidateCheckout(candidate), candidate);
  assert.deepEqual(assertCandidateCheckout(candidate, candidate.commit), candidate);
  assert.throws(() => assertCandidateCheckout({ ...candidate, commit: "1234567" }), /full 40-character commit SHA/);
  assert.throws(() => assertCandidateCheckout(candidate, "1234567"), /ALLOY_E2E_EXPECTED_COMMIT must be a full/);
  assert.throws(() => assertCandidateCheckout(candidate, "2".repeat(40)), /does not match ALLOY_E2E_EXPECTED_COMMIT/);
  for (const status of [" M src/lib.rs\n", "M  src/lib.rs\n", "?? untracked-source.mjs\n", " m vendor/mlx-c\n"]) {
    assert.throws(() => assertCandidateCheckout({ ...candidate, status }), /clean committed checkout/);
  }
});

test("build completion cannot certify a changed HEAD or changed source status", () => {
  const before = { commit: "1".repeat(40), status: "" };
  assert.deepEqual(assertCandidateCheckout(clone(before), undefined, before), before);
  assert.throws(() => assertCandidateCheckout({ ...before, commit: "2".repeat(40) }, undefined, before), /HEAD changed while building/);
  assert.throws(() => assertCandidateCheckout({ ...before, status: "?? generated-source.rs\n" }, undefined, before), /source status changed while building/);
});

test("stream evidence requires incremental frames and visible CLI chunks before completion", () => {
  const proof = assertStreamingTurn(stream(), `${TEXT}\n`, stdoutChunks(), 2);
  assert.equal(proof.delta_count, 2);
  assert.equal(proof.stdout_text_chunk_count, 2);
  assert.equal(proof.stdout_chunks_before_terminal, 2);
  assert.equal(proof.first_stdout_ms, 6);
  assert.equal(proof.terminal_ms, 15);
  assert.equal(proof.assistant_text, TEXT);
});

test("a CLI that buffers correct streamed text until the terminal cannot pass", () => {
  // These retain valid wire deltas and identical final stdout. Only the observed
  // stdout ordering distinguishes a streaming CLI from a buffered one.
  for (const at_ms of [115, 120]) {
    const buffered = [{ at_ms, text: TEXT }, { at_ms: 121, text: "\n" }];
    assert.throws(() => assertStreamingTurn(stream(), `${TEXT}\n`, buffered), /buffered all assistant text until generation_completed/);
  }
  const oneLiveChunk = [{ at_ms: 106, text: "café " }, { at_ms: 120, text: "marker saved\n" }];
  assert.equal(assertStreamingTurn(stream(), `${TEXT}\n`, oneLiveChunk).stdout_chunks_before_terminal, 1);
  assert.throws(() => assertStreamingTurn(stream(), `${TEXT}\n`, oneLiveChunk, 2), /at least 2 CLI stdout chunks before generation_completed/);
  assert.throws(() => assertStreamingTurn(stream(), `${TEXT}\n`, []), /chunk observations are missing/);
  assert.throws(() => assertStreamingTurn(stream(), `${TEXT}\n`, [{ at_ms: 106, text: TEXT }]), /chunk evidence differs/);
});

test("final-only, reordered, failed, cross-session, and duplicated streams cannot pass", async (t) => {
  const cases = [
    ["final-only reply", (rows) => rows.filter((r) => r.message.params?.event !== "text_delta"), /multiple incremental/],
    ["one buffered delta", (rows) => { rows[1].message.params.body.text = TEXT; return rows.filter((_, i) => i !== 2); }, /multiple incremental/],
    ["missing terminal", (rows) => rows.filter((r) => r.message.params?.event !== "generation_completed"), /exactly one generation_completed/],
    ["duplicate terminal", (rows) => [...rows, { ...clone(rows[3]), sequence: 6 }], /exactly one generation_completed/],
    ["post-terminal delta", (rows) => { rows[2].sequence = 6; return rows; }, /after the terminal/],
    ["early final", (rows) => { rows[4].sequence = 3; return rows; }, /before the terminal/],
    ["cross-session frame", (rows) => { rows[1].message.params.session_id = "sess-999"; return rows; }, /cross-session/],
    ["request failure", (rows) => [...rows, { ...clone(rows[3]), sequence: 6, message: event("request_failed", { message: "native failure" }) }], /request_failed/],
    ["final text mismatch", (rows) => { rows[4].message.result.assistant_text = "different"; return rows; }, /differs from incremental/],
    ["duplicate final", (rows) => [...rows, { ...clone(rows[4]), sequence: 6 }], /exactly one final/],
  ];
  for (const [name, mutate, expected] of cases) {
    await t.test(name, () => assert.throws(() => assertStreamingTurn(mutate(stream()), `${TEXT}\n`, stdoutChunks()), expected));
  }
  assert.throws(() => assertStreamingTurn(stream(), "pong\n", stdoutChunks()), /CLI stdout differs/);
});

test("recovery compares prior message contents and the entire ordered event prefix", () => {
  const before = history();
  assert.deepEqual(assertRecoveredHistory(before, clone(before), turn), { event_count: 4, preserved_event_count: 4 });
  const changed = clone(before);
  changed.events[1].payload.content = "rewritten";
  assert.throws(() => assertRecoveredHistory(before, changed, turn), /prior conversation event contents/);
  const missing = { ...clone(before), events: [], event_count: 0 };
  assert.throws(() => assertRecoveredHistory(before, missing, turn), /lost prior events/);
  const noConversation = clone(before);
  noConversation.events[1].kind = "run.started";
  assert.throws(() => assertRecoveredHistory(noConversation, clone(noConversation), turn), /user message is missing/);
  const gap = clone(before);
  gap.events[3].seq = 6;
  assert.throws(() => assertRecoveredHistory(gap, clone(gap), turn), /not gapless/);
});

test("a resumed turn may append events but cannot rewrite the recovered history", () => {
  const before = history();
  const after = clone(before);
  after.events.push({ seq: 5, kind: "user.message", payload: { content: "continue" } });
  after.event_count++;
  assert.throws(() => assertRecoveredHistory(before, after, turn), /changed the stored event count/);
  assert.equal(assertRecoveredHistory(before, after, turn, true).preserved_event_count, 4);
  after.events[2].payload.content = "lost assistant message";
  assert.throws(() => assertRecoveredHistory(before, after, turn, true), /prior conversation event contents/);
});

test("KV restart proof requires completed spill, zero hot retention, and restored tokens", () => {
  assert.equal(assertCacheSpill(cacheMetrics()).cold_entries, 1);
  assert.equal(assertCacheRestore(cacheMetrics()).restored_tokens, 12);
  for (const [field, value] of [["enabled", false], ["disk_enabled", false], ["spill_count", 0], ["cold_entries", 0], ["hot_limit_bytes", 1], ["hot_reserved_bytes", 1], ["protected_reserved_bytes", 1], ["transfer_reserved_bytes", 1], ["last_error", "disk write failed"], ["restore_count", 0], ["restored_tokens", 0]]) {
    const metrics = cacheMetrics();
    metrics.kv_cache[field] = value;
    assert.throws(() => assertCacheRestore(metrics), undefined, field);
  }
  const uncharged = cacheMetrics();
  uncharged.admission.reserved_bytes = 0;
  assert.throws(() => assertCacheSpill(uncharged), /admission ledger/);
  assert.throws(() => assertCacheRestore({}), /real KV cache evidence/);
});

async function unixServer(t, handler) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-wire-"));
  const socketPath = path.join(directory, "daemon.sock");
  const sockets = new Set();
  const server = net.createServer((socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
    socket.on("error", () => {});
    handler(socket);
  });
  t.after(async () => {
    for (const socket of sockets) socket.destroy();
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(directory, { recursive: true, force: true });
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(socketPath, resolve); });
  return { directory, socketPath };
}

test("wire recorder preserves split UTF-8, ordered frames, and the actual CLI connection", { timeout: 5000 }, async (t) => {
  const { directory, socketPath } = await unixServer(t, (socket) => {
    socket.once("data", () => {
      const bytes = Buffer.from(`${responseMessages().map((message) => JSON.stringify(message)).join("\n")}\n`);
      const split = bytes.indexOf(Buffer.from("é")) + 1;
      socket.write(bytes.subarray(0, split));
      setTimeout(() => socket.end(bytes.subarray(split)), 5);
    });
  });
  const evidence = path.join(directory, "wire.ndjson");
  const relay = await recordProxy(path.join(directory, "proxy.sock"), socketPath, evidence);
  try {
    const result = await rpc(path.join(directory, "proxy.sock"), "session.send", { session_id: SESSION, content: PROMPT }, 1000);
    assert.equal(result.assistant_text, TEXT);
    assert.deepEqual(relay.errors, []);
    assert.deepEqual(relay.records.filter((record) => record.direction === "response").map((record) => record.message), responseMessages());
    assert.deepEqual(fs.readFileSync(evidence, "utf8").trim().split("\n").map(JSON.parse), relay.records);
  } finally { await relay.close(); }
});

test("RPC readiness probes time out and reject peers that close before an answer", { timeout: 5000 }, async (t) => {
  const hung = await unixServer(t, () => {});
  await assert.rejects(rpc(hung.socketPath, "daemon.status", {}, 40), /timed out/);
  const closed = await unixServer(t, (socket) => socket.once("data", () => socket.end()));
  await assert.rejects(rpc(closed.socketPath, "daemon.status", {}, 500), /closed before/);
});
