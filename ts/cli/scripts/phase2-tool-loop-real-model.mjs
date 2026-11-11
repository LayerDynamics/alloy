#!/usr/bin/env node
/** Phase 2 deployed-daemon acceptance. Only synthetic fixtures are used. */
import assert from "node:assert/strict";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Acceptance, rpc } from "./real-model-acceptance.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const save = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);

async function sendTurn(a, session, prompt, allow, marker, label) {
  const records = [];
  const socket = net.createConnection({ path: a.socket });
  const timer = setTimeout(() => socket.destroy(new Error("session.send timed out")), a.turnTimeout);
  let buffer = "";
  let pending = Promise.resolve();
  let approvalCount = 0;
  let settled = false;
  try {
    const final = await new Promise((resolve, reject) => {
      socket.setEncoding("utf8");
      socket.on("connect", () => {
        const request = { jsonrpc: "2.0", api_version: 1, id: 1, method: "session.send", params: { session_id: session, content: prompt } };
        records.push({ direction: "request", message: request });
        socket.write(`${JSON.stringify(request)}\n`);
      });
      socket.on("error", reject);
      socket.on("close", () => { if (!settled) reject(new Error("send socket closed before final response")); });
      socket.on("data", (chunk) => {
        buffer += chunk;
        let end;
        while ((end = buffer.indexOf("\n")) !== -1) {
          const raw = buffer.slice(0, end); buffer = buffer.slice(end + 1);
          if (!raw.trim()) continue;
          pending = pending.then(async () => {
            const message = JSON.parse(raw);
            records.push({ direction: "response", message });
            if (message.params?.event === "approval_requested") {
              approvalCount++;
              assert.equal(message.params.body.tool_id, "shell.run");
              const existsBeforeDecision = fs.existsSync(marker);
              records.push({ direction: "observation", point: "before_decision", marker_exists: existsBeforeDecision, call_id: message.params.body.call_id });
              assert.equal(existsBeforeDecision, false, "marker exists before decision");
              const decision = { session_id: session, call_id: message.params.body.call_id, allow };
              const result = await rpc(a.socket, "session.approve", decision);
              records.push({ direction: "control", method: "session.approve", params: decision, result });
            }
            if (message.id === 1) {
              assert.equal(message.error, undefined, JSON.stringify(message.error));
              settled = true;
              resolve(message.result);
            }
          }).catch(reject);
        }
      });
    });
    assert.equal(approvalCount, 1, "model must request one shell approval");
    const existsAfterTurn = fs.existsSync(marker);
    records.push({ direction: "observation", point: "after_turn", marker_exists: existsAfterTurn, decision: allow });
    assert.equal(existsAfterTurn, allow, "marker state does not match explicit decision");
    const history = await rpc(a.socket, "sessions.resume", { session_id: session });
    save(path.join(a.output, `${label}-history.json`), history);
    save(path.join(a.output, `${label}-rpc.json`), records);
    assert.equal(final.ok, true);
    const events = records.filter((r) => r.message?.method === "session.event").map((r) => r.message.params);
    assert.equal(events.filter((e) => e.event === "generation_completed").length, 1);
    assert.equal(events.filter((e) => e.event === "tool_requested" && e.body.tool_id === "fs.read").length, 1);
    assert.equal(events.filter((e) => e.event === "tool_requested" && e.body.tool_id === "shell.run").length, 1);
    assert.equal(events.filter((e) => e.event === "tool_call_started" && e.body.tool_id === "shell.run").length, Number(allow));
    assert(final.assistant_text?.includes("indigo-saffron-83"), "final answer must consume the fixture result");
    if (allow) {
      const shellCall = events.find((e) => e.event === "tool_requested" && e.body.tool_id === "shell.run");
      const shellResult = events.find((e) => e.event === "tool_result" && e.body.call_id === shellCall.body.call_id);
      const chunks = shellResult.body.result.content.ndjson.trim().split("\n").map(JSON.parse);
      const stdout = chunks.filter((chunk) => chunk.stream === "stdout").flatMap((chunk) => chunk.data);
      assert.equal(Buffer.from(stdout).toString("utf8"), "violet-signal-61\n");
      assert(final.assistant_text.includes("violet-signal-61") || final.assistant_text.replace(/\s/g, "").includes(stdout.join(",")), "final answer must consume the unseen shell output, either as text or bytes");
    }
    return { final, events, history };
  } finally {
    clearTimeout(timer);
    socket.destroy();
    // Keep incomplete streams when any assertion fails.
    save(path.join(a.output, `${label}-rpc.json`), records);
  }
}

async function main() {
  const a = new Acceptance("phase2_tool_loop_real_model");
  let failure;
  try {
    await a.prepare();
    const work = fs.realpathSync(a.runtime);
    const marker = path.join(work, "marker");
    const fixture = path.join(work, "fixture.txt");
    fs.writeFileSync(fixture, "indigo-saffron-83\n");
    fs.writeFileSync(path.join(work, "shell-result.txt"), "violet-signal-61\n");
    fs.writeFileSync(path.join(work, "fixture.sh"), "/bin/cat shell-result.txt\n/usr/bin/touch marker\n");
    const executable = fs.realpathSync("/bin/sh");
    const python = fs.realpathSync("/usr/bin/python3");
    const hookLog = path.join(a.output, "plugin-hooks.ndjson");
    const readArgs = { path: "fixture.txt", offset: 0, length: 128 };
    const shellArgs = { executable, args: ["fixture.sh"], cwd: work };
    const config = {
      read_root: work, write_root: work, shell_executables: [executable], shell_cwd_roots: [work],
      http_destinations: [{ origin: "http://fixture.invalid/", address: "127.0.0.1" }], policy: { rules: [
        { tool_id: "fs.read", allowed_arguments: [readArgs], require_approval: false, timeout_ms: 10000, max_output_bytes: 65536 },
        { tool_id: "shell.run", allowed_arguments: [shellArgs], require_approval: true, timeout_ms: 10000, max_output_bytes: 65536 },
      ] }, plugins: [{ id: "w7", version: "1.0.0", program: python,
        args: [path.join(root, "ts/cli/scripts/fixtures/phase2-hook.py"), hookLog], cwd: work, sandbox: { mode: "trusted" } }],
    };
    const configPath = path.join(a.output, "tools.json");
    save(configPath, config);
    await a.startDaemon("daemon-phase2", ["--tools-config", configPath, "--hot-cache-only"]);
    const prompt = `Use the tools. First call fs.read with exactly ${JSON.stringify(readArgs)}. Then call shell.run with exactly ${JSON.stringify(shellArgs)}. After receiving both results, reply with the fixture text and the shell result. /no_think`;
    const startedCalls = [];
    for (const [label, allow] of [["denied", false], ["approved", true]]) {
      const created = await rpc(a.socket, "session.create", { profile: "qwen3-4b-4bit" });
      const turn = await sendTurn(a, created.session_id, prompt, allow, marker, label);
      startedCalls.push(...turn.events.filter((event) => event.event === "tool_call_started").map((event) => ({
        session_id: created.session_id, call_id: event.body.call_id, tool_id: event.body.tool_id,
      })));
    }
    const hooks = fs.readFileSync(hookLog, "utf8").trim().split("\n").map(JSON.parse);
    const before = hooks.filter((h) => h.method === "tool.before_execute");
    const after = hooks.filter((h) => h.method === "tool.after_execute");
    assert.equal(before.length, 3, "both reads and the approved shell require before hooks");
    assert.equal(after.length, 3, "both reads and the approved shell require after hooks");
    assert.deepEqual(hooks.filter((h) => h.method !== "initialize").map((h) => h.method), Array(3).fill(["tool.before_execute", "tool.after_execute"]).flat());
    assert.deepEqual(startedCalls.map((call) => call.tool_id), ["fs.read", "fs.read", "shell.run"]);
    for (const [index, call] of startedCalls.entries()) {
      const b = before[index].params.invocation;
      const a = after[index].params.invocation;
      for (const invocation of [b, a]) {
        assert.equal(invocation.session_id, call.session_id);
        assert.equal(invocation.call_id, call.call_id);
        assert.equal(invocation.definition.id, call.tool_id);
      }
      assert.equal(b.invocation_id, a.invocation_id, "hook pair must describe the same invocation");
    }
    a.metadata.status = "passed";
  } catch (error) {
    failure = error;
    a.metadata.status = "failed";
    a.metadata.error = error.stack ?? String(error);
  } finally {
    try { await a.cleanup(); } catch (error) { failure ??= error; a.metadata.status = "failed"; a.metadata.error = error.stack; }
    a.metadata.finished_at = new Date().toISOString();
    save(path.join(a.output, "result.json"), a.metadata);
  }
  console.log(`phase2_tool_loop_real_model ${a.metadata.status}; evidence ${a.output}`);
  if (failure) throw failure;
}
main().catch((error) => { console.error(error.stack ?? String(error)); process.exitCode = 1; });
