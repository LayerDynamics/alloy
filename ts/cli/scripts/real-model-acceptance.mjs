#!/usr/bin/env node
/** Real Qwen3 acceptance: candidate build, owned process, recorded CLI wire stream. */
import assert from "node:assert/strict";
import { spawnOwned } from "../../../scripts/homebrew/group-supervisor.mjs";
import { createHash } from "node:crypto";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { StringDecoder } from "node:string_decoder";
import { fileURLToPath } from "node:url";

const SCRIPT = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(SCRIPT), "../../..");
const PROFILE = "qwen3-4b-4bit";
const NODE_VERSION = "v26.13.0";
const now = () => Number(process.hrtime.bigint() / 1000n) / 1000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const jsonFile = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
const appendJson = (file, value) => fs.appendFileSync(file, `${JSON.stringify(value)}\n`);

/** Keep reviewable provenance/metrics in Actions logs; full evidence stays on disk. */
function printEvidenceSummary(prefix, value, maxBytes = 16384) {
  let summary = JSON.stringify(value, (key, item) => {
    if (["prompt", "assistant_text", "text", "events", "files"].includes(key)) return undefined;
    return typeof item === "string" && item.length > 1024 ? `${item.slice(0, 1024)}…` : item;
  });
  if (Buffer.byteLength(summary) > maxBytes) {
    summary = JSON.stringify({ test: value.test, status: value.status, event: value.event, sequence: value.sequence, commit: value.commit ?? value.data?.commit_sha, summary_truncated: true, detail: "Full result/provenance/metrics retained in the acceptance artifact." });
  }
  console.log(`${prefix} ${summary}`);
}

function printM9Evidence(file) {
  try {
    const selected = new Map();
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      if (!line.trim()) continue;
      let record;
      try { record = JSON.parse(line); } catch { continue; }
      const event = record.event ?? "";
      if (["acceptance_started", "acceptance_passed", "acceptance_failed", "budget_planned", "native_limits_configured"].includes(event) || (event.endsWith("_verified") && event !== "artifact_file_verified")) {
        selected.set(`${event}:${record.data?.artifact_id ?? ""}`, record);
      }
    }
    const records = [...selected.values()];
    // Final outcome first, then bounded provenance and memory checks. At most
    // 16 records of 4 KiB each; token streams and per-file hashes stay in artifacts.
    const terminal = records.filter((record) => ["acceptance_passed", "acceptance_failed"].includes(record.event));
    const other = records.filter((record) => !terminal.includes(record));
    for (const record of [...terminal, ...other].slice(0, 16)) printEvidenceSummary("ALLOY_M9_EVIDENCE", record, 4096);
    if (records.length === 0) printEvidenceSummary("ALLOY_M9_EVIDENCE", { event: "evidence_unavailable", file, reason: "No complete acceptance summary records; inspect driver.log." });
  } catch (error) {
    printEvidenceSummary("ALLOY_M9_EVIDENCE", { event: "evidence_unavailable", file, error: error.message });
  }
}

/** Acceptance belongs to one committed candidate, never an unrecorded source edit. */
export function assertCandidateCheckout(snapshot, expectedCommit, beforeBuild) {
  assert.match(snapshot.commit, /^[0-9a-f]{40}$/, "candidate HEAD must be a full 40-character commit SHA");
  assert.equal(typeof snapshot.status, "string", "candidate status must be captured from git");
  if (expectedCommit !== undefined) {
    assert.match(expectedCommit, /^[0-9a-f]{40}$/, "ALLOY_E2E_EXPECTED_COMMIT must be a full 40-character commit SHA");
    assert.equal(snapshot.commit, expectedCommit, "candidate HEAD does not match ALLOY_E2E_EXPECTED_COMMIT");
  }
  if (beforeBuild !== undefined) {
    assert.equal(snapshot.commit, beforeBuild.commit, "candidate HEAD changed while building the daemon and CLI");
    assert.equal(snapshot.status, beforeBuild.status, "candidate source status changed while building the daemon and CLI");
  }
  assert.equal(snapshot.status, "", "acceptance requires a clean committed checkout, including tracked files, untracked source, and submodules; commit the candidate before running (ignored build/evidence output is allowed)");
  return snapshot;
}

/** Reject final-only replies, multiple terminals, cross-session frames, and text drift. */
export function assertStreamingTurn(records, stdout, stdoutChunks, minStdoutChunksBeforeTerminal = 1) {
  const sends = records.filter((r) => r.direction === "request" && r.message?.method === "session.send");
  assert.equal(sends.length, 1, "expected exactly one CLI session.send request");
  const sent = sends[0];
  const sessionId = sent.message.params.session_id;
  const responses = records.filter((r) => r.direction === "response" && r.connection_id === sent.connection_id);
  const events = responses.filter((r) => r.message?.method === "session.event");
  assert(events.every((r) => r.message.params?.session_id === sessionId), "received a cross-session stream event");
  assert(!events.some((r) => r.message.params.event === "request_failed"), "generation emitted request_failed");
  const deltas = events.filter((r) => r.message.params.event === "text_delta" && typeof r.message.params.body?.text === "string" && r.message.params.body.text.length > 0);
  assert(deltas.length >= 2, `expected multiple incremental text_delta frames; received ${deltas.length}`);
  const terminals = events.filter((r) => r.message.params.event === "generation_completed");
  assert.equal(terminals.length, 1, "expected exactly one generation_completed event");
  const terminal = terminals[0];
  assert(deltas.every((r) => r.sequence < terminal.sequence), "text arrived after the terminal event");
  assert(events.every((r) => r.sequence <= terminal.sequence), "received an event after completion");
  assert(["end_of_sequence", "output_limit"].includes(terminal.message.params.body?.finish_reason), "invalid terminal finish reason");
  const finals = responses.filter((r) => r.message?.id === sent.message.id);
  assert.equal(finals.length, 1, "expected exactly one final session.send response");
  assert(finals[0].sequence > terminal.sequence, "final RPC response arrived before the terminal event");
  const result = finals[0].message.result;
  assert.equal(result?.ok, true, `session.send failed: ${JSON.stringify(finals[0].message)}`);
  assert.equal(result.session_id, sessionId);
  assert(Number.isSafeInteger(result.input_tokens) && result.input_tokens > 0, "expected real input token usage");
  const text = deltas.map((r) => r.message.params.body.text).join("");
  assert(/\p{L}/u.test(text), "expected decoded assistant text");
  assert.equal(result.assistant_text, text, "final assistant_text differs from incremental deltas");
  assert.equal(stdout, `${text}\n`, "CLI stdout differs from the actual streamed text");
  assert(Array.isArray(stdoutChunks) && stdoutChunks.length > 0, "CLI stdout chunk observations are missing");
  assert(stdoutChunks.every((chunk) => Number.isFinite(chunk.at_ms) && typeof chunk.text === "string"), "invalid CLI stdout chunk observation");
  assert.equal(stdoutChunks.map((chunk) => chunk.text).join(""), stdout, "CLI stdout chunk evidence differs from captured stdout");
  assert(stdoutChunks.every((chunk, i) => i === 0 || chunk.at_ms >= stdoutChunks[i - 1].at_ms), "CLI stdout chunk observations are out of order");
  const textChunks = stdoutChunks.filter((chunk) => /\S/u.test(chunk.text));
  assert(textChunks.length > 0, "CLI emitted no visible assistant text");
  assert(textChunks[0].at_ms >= deltas[0].at_ms, "CLI emitted text before the first streamed delta");
  // Both observations use this process's monotonic clock. Require visible CLI
  // output while generation is active, with no arbitrary elapsed-time threshold.
  const liveChunks = textChunks.filter((chunk) => chunk.at_ms < terminal.at_ms);
  assert(liveChunks.length > 0, "CLI buffered all assistant text until generation_completed");
  assert(liveChunks.length >= minStdoutChunksBeforeTerminal, `expected at least ${minStdoutChunksBeforeTerminal} CLI stdout chunks before generation_completed; observed ${liveChunks.length}`);
  return {
    session_id: sessionId, prompt: sent.message.params.content, assistant_text: text,
    delta_count: deltas.length, first_delta_ms: deltas[0].at_ms - sent.at_ms,
    stdout_text_chunk_count: textChunks.length, stdout_chunks_before_terminal: liveChunks.length,
    first_stdout_ms: textChunks[0].at_ms - sent.at_ms,
    terminal_ms: terminal.at_ms - sent.at_ms, input_tokens: result.input_tokens,
    finish_reason: terminal.message.params.body.finish_reason,
    usage: terminal.message.params.body.usage,
  };
}

/** Compare durable event contents, not merely a repeated session identifier. */
export function assertRecoveredHistory(before, recovered, turn, allowAdditional = false) {
  assert.equal(recovered?.session_id, turn.session_id, "recovery returned another session");
  assert.equal(recovered.profile_id, PROFILE);
  assert(Array.isArray(before?.events) && Array.isArray(recovered.events), "resume must return stored events");
  assert.equal(recovered.event_count, recovered.events.length);
  assert(recovered.events.length >= before.events.length, "recovery lost prior events");
  if (!allowAdditional) assert.equal(recovered.events.length, before.events.length, "restart changed the stored event count");
  assert.deepEqual(recovered.events.slice(0, before.events.length), before.events, "restart changed prior conversation event contents");
  recovered.events.forEach((event, index) => assert.equal(event.seq, index + 1, "stored event sequence is not gapless"));
  assert(recovered.events.some((e) => e.kind === "user.message" && e.payload?.content === turn.prompt), "recovered user message is missing or changed");
  assert(recovered.events.some((e) => e.kind === "assistant.message" && e.payload?.content === turn.assistant_text), "recovered assistant message is missing or changed");
  return { event_count: recovered.event_count, preserved_event_count: before.events.length };
}

/** A cold entry must survive after all active KV ownership has drained. */
export function assertCacheSpill(metrics) {
  const cache = metrics?.kv_cache;
  assert(cache, "daemon.metrics must expose real KV cache evidence");
  assert.equal(cache.enabled, true, "tiered KV retention is disabled");
  assert.equal(cache.disk_enabled, true, "the SSD KV tier is disabled");
  assert(cache.spill_count > 0, "the completed turn was not persisted to the SSD cache");
  assert(cache.cold_entries > 0, "the SSD cache has no persisted entry");
  assert.equal(cache.hot_limit_bytes, 0, "acceptance must force idle KV out of the hot cache");
  assert.equal(cache.hot_reserved_bytes, 0, "idle KV is still retained in RAM");
  assert.equal(cache.protected_reserved_bytes, 0, "active KV ownership has not drained");
  // The storage owner keeps this admitted allowance throughout its lifetime;
  // zero hot/protected bytes proves native idle KV drained, not the transfer pool.
  assert.equal(cache.transfer_reserved_bytes, 64 * 1024 * 1024, "expected the daemon's owned 64 MiB transfer allowance");
  assert(metrics.admission?.reserved_bytes >= cache.transfer_reserved_bytes, "the live admission ledger does not include the KV transfer allowance");
  assert.equal(cache.last_error, null, "SSD cache reported an error");
  return cache;
}

export function assertCacheRestore(metrics) {
  const cache = assertCacheSpill(metrics);
  assert(cache.restore_count > 0, "the new daemon never restored an SSD KV entry");
  assert(cache.restored_tokens > 0, "the resumed request reused no persisted KV tokens");
  return cache;
}

function milliseconds(name, fallback) {
  const value = Number(process.env[name] ?? fallback);
  assert(Number.isSafeInteger(value) && value > 0 && value <= 3_600_000, `${name} must be a positive integer <= 3600000`);
  return value;
}

/** One bounded NDJSON RPC; every call uses a new connection. */
export function rpc(socketPath, method, params = {}, timeoutMs = 3000) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ path: socketPath });
    let buffered = "";
    let settled = false;
    const timer = setTimeout(() => finish(new Error(`${method} timed out after ${timeoutMs} ms`)), timeoutMs);
    function finish(error, result) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.destroy();
      if (error) reject(error); else resolve(result);
    }
    socket.setEncoding("utf8");
    socket.once("connect", () => socket.write(`${JSON.stringify({ jsonrpc: "2.0", api_version: 1, id: 1, method, params })}\n`));
    socket.on("error", (error) => finish(error));
    socket.on("close", () => finish(new Error(`${method}: socket closed before response`)));
    socket.on("data", (chunk) => {
      buffered += chunk;
      let newline;
      while ((newline = buffered.indexOf("\n")) >= 0) {
        const raw = buffered.slice(0, newline);
        buffered = buffered.slice(newline + 1);
        if (!raw.trim()) continue;
        try {
          const message = JSON.parse(raw);
          if (message.id !== 1) continue;
          if (message.error) finish(new Error(`${method}: ${JSON.stringify(message.error)}`));
          else finish(null, message.result);
        } catch (error) { finish(error); }
      }
    });
  });
}

/** A transparent relay records the frames used by the real, unmodified CLI. */
export async function recordProxy(socketPath, daemonPath, evidenceFile) {
  const records = [];
  const errors = [];
  const connections = new Set();
  let nextConnection = 0;
  const server = net.createServer((downstream) => {
    const connectionId = ++nextConnection;
    const upstream = net.createConnection({ path: daemonPath });
    connections.add(downstream);
    connections.add(upstream);
    const observe = (socket, direction) => {
      const decoder = new StringDecoder("utf8");
      let buffered = "";
      socket.on("data", (chunk) => {
        buffered += decoder.write(chunk);
        let newline;
        while ((newline = buffered.indexOf("\n")) >= 0) {
          const raw = buffered.slice(0, newline);
          buffered = buffered.slice(newline + 1);
          if (!raw.trim()) continue;
          try {
            const record = { sequence: records.length + 1, at_ms: now(), connection_id: connectionId, direction, message: JSON.parse(raw) };
            records.push(record);
            appendJson(evidenceFile, record);
          } catch (error) { errors.push(`${direction}: ${error.message}`); }
        }
      });
      socket.on("close", () => connections.delete(socket));
      socket.on("error", (error) => { errors.push(error.message); downstream.destroy(); upstream.destroy(); });
    };
    observe(downstream, "request");
    observe(upstream, "response");
    downstream.pipe(upstream);
    upstream.pipe(downstream);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(socketPath, resolve);
  });
  return {
    records, errors,
    async close() {
      for (const socket of connections) socket.destroy();
      await new Promise((resolve) => server.close(resolve));
      fs.rmSync(socketPath, { force: true });
    },
  };
}

export class Acceptance {
  owned = new Set();
  aborted = false;
  constructor(testName) {
    this.testName = testName;
    const base = path.resolve(process.env.ALLOY_ACCEPTANCE_OUTPUT_DIR || path.join(ROOT, "target/phase-1-acceptance"));
    fs.mkdirSync(base, { recursive: true });
    this.output = fs.mkdtempSync(path.join(base, `${testName}-`));
    // Darwin's UDS path limit is shorter than many runner checkout/TMPDIR paths.
    this.runtime = fs.mkdtempSync("/tmp/alloy-e2e-");
    this.socket = path.join(this.runtime, "daemon.sock");
    this.db = path.join(this.output, "sessions.sqlite");
    this.cache = path.join(this.output, "kv-cache");
    this.artifact = path.resolve(process.env.ALLOY_QWEN3_ARTIFACT || path.join(os.homedir(), "alloy-omlx-models/alloy-qwen3-4b-4bit"));
    this.corpus = path.join(ROOT, "bench/corpus/models.json");
    this.expectedCommit = process.env.ALLOY_E2E_EXPECTED_COMMIT;
    this.metadata = { test: testName, status: "running", started_at: new Date().toISOString(), candidate_checkout: ROOT, expected_commit: this.expectedCommit, node: process.version, artifact_path: this.artifact, database: this.db, ssd_cache: this.cache };
    jsonFile(path.join(this.output, "result.json"), this.metadata);
    console.log(`${testName}: evidence ${this.output}`);
  }

  start(command, args, label, cwd = ROOT) {
    assert(!this.aborted, "acceptance interrupted");
    // A fresh process group contains only this invocation's child and its
    // descendants (including Cargo/rustc). Never signal unrelated daemons.
    const processOwner = spawnOwned(command, args, { cwd, env: process.env });
    const child = processOwner.child;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    const owned = { child, processOwner, alive: true, pid: null, stdout: "", stdoutChunks: [], stderr: "" };
    owned.ready = processOwner.ready.then((pid) => { owned.pid = pid; return pid; });
    this.owned.add(owned);
    owned.ready.then((pid) => appendJson(path.join(this.output, "processes.ndjson"), { command, args, cwd, pid, process_group: pid, at: new Date().toISOString() }));
    const stdoutPath = path.join(this.output, `${label}.stdout.log`);
    const stderrPath = path.join(this.output, `${label}.stderr.log`);
    fs.writeFileSync(stdoutPath, "");
    fs.writeFileSync(stderrPath, "");
    child.stdout.on("data", (chunk) => {
      const observation = { at_ms: now(), text: chunk.toString() };
      if (label.startsWith("cli-")) {
        owned.stdoutChunks.push(observation);
        appendJson(path.join(this.output, `${label}.stdout-chunks.ndjson`), observation);
      }
      fs.appendFileSync(stdoutPath, chunk);
      // Cargo output is read from its retained JSON log after completion.
      if (label !== "build") owned.stdout += observation.text;
    });
    child.stderr.on("data", (chunk) => {
      fs.appendFileSync(stderrPath, chunk);
      owned.stderr = (owned.stderr + chunk.toString()).slice(-16000);
    });
    owned.done = processOwner.done.then(({ code, signal, error }) => {
      owned.alive = false;
      this.owned.delete(owned);
      return { code, signal, error: error ? new Error(error) : null };
    });
    return owned;
  }

  signal(owned, signal) {
    if (!owned?.alive) return;
    owned.processOwner.request(signal);
  }

  async stop(owned) {
    if (!owned?.alive) return;
    this.signal(owned, "SIGTERM");
    let timer;
    await Promise.race([owned.done, new Promise((resolve) => { timer = setTimeout(resolve, 5000); })]);
    clearTimeout(timer);
    if (owned.alive) {
      this.signal(owned, "SIGKILL");
      await Promise.race([owned.done, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`owned PID ${owned.pid} did not exit after SIGKILL`)), 5000); })]);
      clearTimeout(timer);
    }
  }

  async command(command, args, label, timeoutMs, cwd = ROOT) {
    const owned = this.start(command, args, label, cwd);
    let timer;
    try {
      const exit = await Promise.race([owned.done, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs); })]);
      assert.equal(exit.code, 0, `${label} failed (${exit.error?.message ?? exit.signal ?? exit.code}); see ${label}.stderr.log\n${owned.stderr}`);
      return owned;
    } finally { clearTimeout(timer); await this.stop(owned); }
  }

  async candidateSnapshot(label, beforeBuild) {
    const revision = await this.command("git", ["rev-parse", "--verify", "HEAD^{commit}"], `${label}-revision`, 10000);
    const status = await this.command("git", ["status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"], `${label}-status`, 10000);
    const snapshot = { commit: revision.stdout.trim(), status: status.stdout };
    jsonFile(path.join(this.output, `${label}.json`), snapshot);
    return assertCandidateCheckout(snapshot, this.expectedCommit, beforeBuild);
  }

  async prepare() {
    this.buildTimeout = milliseconds("ALLOY_E2E_BUILD_TIMEOUT_MS", 1_200_000);
    this.startTimeout = milliseconds("ALLOY_E2E_START_TIMEOUT_MS", 240_000);
    this.turnTimeout = milliseconds("ALLOY_E2E_TURN_TIMEOUT_MS", 180_000);
    this.cacheTimeout = milliseconds("ALLOY_E2E_CACHE_TIMEOUT_MS", 30_000);
    assert.equal(process.version, NODE_VERSION, `Use the repository's pinned Node ${NODE_VERSION.slice(1)} runtime`);
    assert.equal(process.platform, "darwin", "real-model acceptance requires the Apple Silicon macOS/Metal lane");
    assert.equal(process.arch, "arm64", "real-model acceptance requires Apple Silicon arm64");
    if (process.env.ALLOY_ENGINE_ROOT) assert.equal(fs.realpathSync(process.env.ALLOY_ENGINE_ROOT), fs.realpathSync(ROOT), "ALLOY_ENGINE_ROOT may only name this candidate checkout; another worktree is not acceptance evidence");
    const beforeBuild = await this.candidateSnapshot("candidate-before-build");
    this.metadata.commit = beforeBuild.commit;
    const profile = JSON.parse(fs.readFileSync(this.corpus, "utf8")).profiles.find((p) => p.id === PROFILE);
    assert(profile?.artifact?.sha256 && profile.artifact.files?.length > 0, "corpus must declare the complete frozen model identity");
    this.metadata.model = profile.artifact;
    this.metadata.corpus_sha256 = createHash("sha256").update(fs.readFileSync(this.corpus)).digest("hex");
    jsonFile(path.join(this.output, "model-pin.json"), profile);
    if (!fs.existsSync(path.join(this.artifact, "config.json"))) {
      throw new Error(`Pinned model missing at ${this.artifact}. Set ALLOY_QWEN3_ARTIFACT to a flat local copy of ${profile.artifact.repository}@${profile.artifact.revision} (aggregate SHA-256 ${profile.artifact.sha256}); all files in bench/corpus/models.json are required. This test never downloads or skips a missing model.`);
    }
    await this.command("cargo", ["build", "--locked", "--manifest-path", path.join(ROOT, "Cargo.toml"), "-p", "alloyd", "--release", "--message-format=json-render-diagnostics"], "build", this.buildTimeout);
    const artifacts = fs.readFileSync(path.join(this.output, "build.stdout.log"), "utf8").split("\n").flatMap((line) => {
      try { const event = JSON.parse(line); return event.reason === "compiler-artifact" && event.target?.name === "alloyd" && event.executable && path.resolve(event.manifest_path) === path.join(ROOT, "crates/alloyd/Cargo.toml") ? [event] : []; } catch { return []; }
    });
    assert(artifacts.length > 0, "Cargo did not report the candidate alloyd executable");
    this.binary = fs.realpathSync(artifacts.at(-1).executable);
    this.metadata.binary = this.binary;
    this.metadata.binary_sha256 = createHash("sha256").update(fs.readFileSync(this.binary)).digest("hex");
    if (this.testName === "phase2_tool_loop_real_model") {
      await this.command("cargo", ["build", "--locked", "--manifest-path", path.join(ROOT, "Cargo.toml"), "-p", "harness-core", "--bins", "--release"], "plugin-helper-build", this.buildTimeout);
      this.metadata.plugin_helper = fs.realpathSync(path.join(path.dirname(this.binary), "alloy-plugin-pipe-helper"));
      this.metadata.plugin_helper_sha256 = createHash("sha256").update(fs.readFileSync(this.metadata.plugin_helper)).digest("hex");
      this.metadata.plugin_exec_helper = fs.realpathSync(path.join(path.dirname(this.binary), "alloy-plugin-exec-helper"));
      this.metadata.plugin_exec_helper_sha256 = createHash("sha256").update(fs.readFileSync(this.metadata.plugin_exec_helper)).digest("hex");
    }
    await this.command("npm", ["run", "build"], "cli-build", this.buildTimeout, path.join(ROOT, "ts/cli"));
    await this.candidateSnapshot("candidate-after-build", beforeBuild);
    this.metadata.source_verified_after_build = true;
    jsonFile(path.join(this.output, "candidate.json"), this.metadata);
  }

  async call(method, params = {}, timeout = 5000) {
    const result = await rpc(this.socket, method, params, timeout);
    appendJson(path.join(this.output, "checks.ndjson"), { at_ms: now(), method, params, result });
    return result;
  }

  async waitForCache(label, check) {
    const deadline = Date.now() + this.cacheTimeout;
    let lastError = "no cache metrics";
    while (Date.now() < deadline) {
      assert(!this.aborted, "acceptance interrupted");
      try {
        const metrics = await this.call("daemon.metrics", {}, Math.min(5000, Math.max(1, deadline - Date.now())));
        const cache = check(metrics);
        jsonFile(path.join(this.output, `${label}.cache.json`), metrics);
        return cache;
      } catch (error) { lastError = error.message; }
      await sleep(Math.min(250, Math.max(1, deadline - Date.now())));
    }
    throw new Error(`${label}: KV cache evidence did not satisfy acceptance within ${this.cacheTimeout} ms: ${lastError}`);
  }

  async startDaemon(label, extraArgs = []) {
    const args = ["--artifact", this.artifact, "--profile", PROFILE, "--socket", this.socket, "--db", this.db, "--corpus-models", this.corpus, "--metal", "--paged-ssd-cache-dir", this.cache, "--hot-cache-max-size", "0", "--max-output-tokens", "256", "--telemetry-endpoint", "https://127.0.0.1:1", ...extraArgs];
    // The daemon boot path verifies every pinned corpus file before readiness.
    const owned = this.start(this.binary, args, label);
    await owned.ready;
    const deadline = Date.now() + this.startTimeout;
    let lastError = "no API response";
    while (Date.now() < deadline) {
      assert(!this.aborted, "acceptance interrupted");
      assert(owned.alive && owned.pid, `${label} exited before API readiness; see ${label}.stderr.log\n${owned.stderr}`);
      try {
        const status = await rpc(this.socket, "daemon.status", {}, Math.min(2000, Math.max(1, deadline - Date.now())));
        const metrics = await rpc(this.socket, "daemon.metrics", {}, Math.min(2000, Math.max(1, deadline - Date.now())));
        assert.equal(status.api_version, 1, "daemon API version mismatch");
        // Legacy daemon.status reports configured boot metadata; weights load
        // lazily. Live residency is checked only after actual generation below.
        assert(status.models_resident > 0, "daemon boot configured no model");
        assert.equal(metrics.process_id, owned.pid, "API listener does not belong to the launched daemon PID");
        assert.equal(metrics.residency?.worker_alive, true, "native worker is not alive");
        assert(owned.alive, "daemon exited during readiness verification");
        const ready = { pid: owned.pid, status, metrics, at: new Date().toISOString() };
        jsonFile(path.join(this.output, `${label}.ready.json`), ready);
        return owned;
      } catch (error) {
        lastError = error.message;
        appendJson(path.join(this.output, `${label}.readiness.ndjson`), { at: new Date().toISOString(), error: lastError });
      }
      await sleep(Math.min(250, Math.max(1, deadline - Date.now())));
    }
    throw new Error(`${label} was not API/PID-ready within ${this.startTimeout} ms: ${lastError}; see ${label}.stderr.log`);
  }

  async turn(label, prompt, resume) {
    const proxy = await recordProxy(path.join(this.runtime, `${label}.sock`), this.socket, path.join(this.output, `${label}.rpc.ndjson`));
    try {
      const args = [path.join(ROOT, "ts/cli/bin/alloy.js"), "chat", "--socket", path.join(this.runtime, `${label}.sock`), "--profile", PROFILE, "--artifact", this.artifact, "--db", this.db, "--corpus-models", this.corpus];
      if (resume) args.push("--resume", resume);
      args.push("-m", prompt);
      const cli = await this.command(process.execPath, args, `cli-${label}`, this.turnTimeout);
      assert.equal(proxy.errors.length, 0, `RPC recording failed: ${proxy.errors.join("; ")}`);
      assert(!cli.stderr.includes("without streamed text"), "CLI used final-response fallback");
      const turn = assertStreamingTurn(proxy.records, cli.stdout, cli.stdoutChunks, this.testName === "qwen3_end_to_end_stream" ? 2 : 1);
      jsonFile(path.join(this.output, `${label}.turn.json`), turn);
      return turn;
    } finally { await proxy.close(); }
  }

  async run() {
    await this.prepare();
    const first = await this.startDaemon("daemon-before");
    if (this.testName === "qwen3_end_to_end_stream") {
      const turn = await this.turn("stream", "Reply with exactly these eight words separated by spaces: amber birch cobalt dune ember fern grove harbor. /no_think");
      this.metadata.stream = turn;
    } else {
      const firstTurn = await this.turn("before", "Remember this conversation marker: copper-orchid-47. Reply with the sentence: I have stored the conversation marker. /no_think");
      const before = await this.call("sessions.resume", { session_id: firstTurn.session_id });
      assertRecoveredHistory(before, before, firstTurn);
      jsonFile(path.join(this.output, "history-before.json"), before);
      const spilled = await this.waitForCache("before", assertCacheSpill);
      const firstPid = first.pid;
      await this.stop(first);
      assert(!first.alive, "first daemon is still running");
      const second = await this.startDaemon("daemon-after");
      assert.notEqual(second.pid, firstPid, "test did not launch a distinct daemon process");
      const recovered = await this.call("sessions.resume", { session_id: firstTurn.session_id });
      const recovery = assertRecoveredHistory(before, recovered, firstTurn);
      jsonFile(path.join(this.output, "history-recovered.json"), recovered);
      const resumed = await this.turn("after", "What was the conversation marker in my previous message? Reply with the marker only. /no_think", firstTurn.session_id);
      assert.equal(resumed.session_id, firstTurn.session_id);
      assert(resumed.assistant_text.includes("copper-orchid-47"), "resumed generation did not recover the prior conversation marker");
      const after = await this.call("sessions.resume", { session_id: firstTurn.session_id });
      assertRecoveredHistory(before, after, firstTurn, true);
      assert(after.events.length > before.events.length, "resumed turn appended no events");
      assert(after.events.some((e) => e.kind === "user.message" && e.payload?.content === resumed.prompt));
      assert(after.events.some((e) => e.kind === "assistant.message" && e.payload?.content === resumed.assistant_text));
      jsonFile(path.join(this.output, "history-after.json"), after);
      const restored = await this.waitForCache("after", assertCacheRestore);
      this.metadata.restart = { before_pid: firstPid, after_pid: second.pid, ...recovery, events_after_resumed_turn: after.event_count, first_turn: firstTurn, resumed_turn: resumed, cache_before_restart: spilled, cache_after_restart: restored };
    }
    this.metadata.final_metrics = await this.call("daemon.metrics");
    const residency = this.metadata.final_metrics.residency;
    assert(residency?.models_resident > 0, "real generation left no live loaded-model evidence");
    assert(residency.models?.some((model) => model.artifact_id === PROFILE && model.weights_reserved_bytes > 0), "live metrics do not show the generated Qwen3 model's admitted weights");
  }

  async cleanup() {
    const results = await Promise.allSettled([...this.owned].map((owned) => this.stop(owned)));
    fs.rmSync(this.runtime, { recursive: true, force: true });
    const failure = results.find((result) => result.status === "rejected");
    if (failure) throw failure.reason;
  }
}

export async function main(testName) {
  assert(["qwen3_end_to_end_stream", "session_restart_real_model"].includes(testName), "expected a named Phase 1 real-model acceptance test");
  const acceptance = new Acceptance(testName);
  const interrupt = () => {
    acceptance.aborted = true;
    for (const owned of acceptance.owned) acceptance.signal(owned, "SIGTERM");
  };
  process.on("SIGINT", interrupt);
  process.on("SIGTERM", interrupt);
  let error;
  try { await acceptance.run(); }
  catch (failure) { error = failure; }
  finally {
    try { await acceptance.cleanup(); } catch (failure) { error ??= failure; }
    process.off("SIGINT", interrupt);
    process.off("SIGTERM", interrupt);
  }
  acceptance.metadata.status = error ? "failed" : "passed";
  acceptance.metadata.finished_at = new Date().toISOString();
  if (error) acceptance.metadata.error = error.stack ?? String(error);
  jsonFile(path.join(acceptance.output, "result.json"), acceptance.metadata);
  printEvidenceSummary("ALLOY_ACCEPTANCE_RESULT", acceptance.metadata);
  console.log(`${testName} ${error ? "FAIL" : "PASS"}; evidence retained at ${acceptance.output}`);
  if (error) throw error;
}

if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT) {
  if (process.argv[2] === "--summarize-m9") printM9Evidence(process.argv[3]);
  else main(process.argv[2]).catch((error) => { console.error(error.stack ?? String(error)); process.exitCode = 1; });
}
