import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { OwnedProcess, REQUIRED_FILES, assertNativeCheck, assertOwnedDaemonPid, assertOwnedPluginPid, installedPluginSource, parseArguments, processTable, runSmoke, verifyInstallation } from "./smoke.mjs";

const COMMIT = "c".repeat(40);
const hash = (value) => createHash("sha256").update(value).digest("hex");
const argumentsFor = (prefix, output) => ["--prefix", prefix, "--expected-commit", COMMIT, "--output", output];

test("installed plugin fixture exits on EOF before and after its real handshake", async (t) => {
  const root = directory(t);
  const script = path.join(root, "fixture.cjs");
  fs.writeFileSync(script, installedPluginSource());
  for (const handshake of [false, true]) {
    const marker = path.join(root, `marker-${handshake}`);
    const child = spawn(process.execPath, [script, marker], { stdio: ["pipe", "pipe", "pipe"] });
    let output = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", chunk => { output += chunk; });
    const exited = new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("close", (code, signal) => resolve({ code, signal }));
    });
    if (handshake) child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "initialize", id: 42 }) + "\n");
    child.stdin.end();
    const exit = await Promise.race([exited, new Promise((_, reject) => {
      const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new Error("fixture did not exit on EOF")); }, 5000);
      exited.then(() => clearTimeout(timer), () => clearTimeout(timer));
    })]);
    assert.deepEqual(exit, { code: 0, signal: null });
    if (handshake) {
      assert.equal(JSON.parse(output).id, 42);
      assert.equal(JSON.parse(fs.readFileSync(marker, "utf8")).pid, child.pid);
    } else assert.equal(fs.existsSync(marker), false);
  }
});

function directory(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-installed-smoke-test-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

function packageFixture(t) {
  const root = directory(t);
  const prefix = path.join(root, "installed prefix");
  const files = REQUIRED_FILES.map((name) => {
    const content = Buffer.from(`installed fixture: ${name}\n`);
    const filename = path.join(prefix, name);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, content);
    return { path: name, sha256: hash(content) };
  });
  const manifest = { schema_version: 1, source_commit: COMMIT, version: "0.1.0", node_version: "26.13.0", files };
  const save = () => fs.writeFileSync(path.join(prefix, "libexec/install.json"), JSON.stringify(manifest));
  save();
  return { root, prefix, manifest, save };
}

test("argument validation refuses ambiguous provenance, unsafe output locations, and incomplete options", () => {
  const parsed = parseArguments([...argumentsFor("/tmp/install", "/tmp/evidence"), "--artifact", "/models/pinned"]);
  assert.deepEqual(parsed, { prefix: "/tmp/install", expectedCommit: COMMIT, output: "/tmp/evidence", artifact: "/models/pinned" });
  for (const args of [
    argumentsFor("relative", "/tmp/evidence"), argumentsFor("/tmp/install", "relative"),
    argumentsFor("/tmp/install", "/tmp/install/evidence"),
    [...argumentsFor("/tmp/install", "/tmp/evidence"), "--prefix", "/another"],
    [...argumentsFor("/tmp/install", "/tmp/evidence"), "--artifact"],
    [...argumentsFor("/tmp/install", "/tmp/evidence"), "--artifact", "relative"],
    ["--prefix", "/tmp/install", "--expected-commit", "short", "--output", "/tmp/evidence"],
    [...argumentsFor("/tmp/install", "/tmp/evidence"), "--skip-metal"],
  ]) assert.throws(() => parseArguments(args));
});

test("existing acceptance evidence is never overwritten", async (t) => {
  const root = directory(t);
  const output = path.join(root, "evidence");
  fs.mkdirSync(output);
  fs.writeFileSync(path.join(output, "result.json"), "earlier evidence");
  await assert.rejects(runSmoke(parseArguments(argumentsFor(path.join(root, "prefix"), output))), /EEXIST/);
  assert.equal(fs.readFileSync(path.join(output, "result.json"), "utf8"), "earlier evidence");
});

test("installed hashes bind the candidate while allowing Homebrew-owned receipts outside runtime roots", async (t) => {
  const fixture = packageFixture(t);
  fs.mkdirSync(path.join(fixture.prefix, ".brew"));
  fs.writeFileSync(path.join(fixture.prefix, ".brew/alloy.rb"), "Homebrew formula metadata");
  fs.writeFileSync(path.join(fixture.prefix, "INSTALL_RECEIPT.json"), "{}");
  const verified = await verifyInstallation(fixture.prefix, COMMIT);
  assert.equal(verified.files.length, REQUIRED_FILES.length);
  assert.equal(verified.manifest.source_commit, COMMIT);
  await assert.rejects(verifyInstallation(fixture.prefix, "d".repeat(40)), /Installed source commit/);
  fs.appendFileSync(path.join(fixture.prefix, "libexec/native/alloyd"), " changed");
  await assert.rejects(verifyInstallation(fixture.prefix, COMMIT), /Installed checksum mismatch/);
});

test("manifest coverage rejects unexpected runtime files, missing assets, traversal, and external runtime links", async (t) => {
  const extra = packageFixture(t);
  fs.writeFileSync(path.join(extra.prefix, "bin/unrecorded-helper"), "unrecorded");
  await assert.rejects(verifyInstallation(extra.prefix, COMMIT), /not covered by the manifest/);
  const missing = packageFixture(t);
  missing.manifest.files = missing.manifest.files.filter((entry) => entry.path !== "libexec/native/mlx.metallib");
  missing.save();
  await assert.rejects(verifyInstallation(missing.prefix, COMMIT), /Required installed asset/);
  const traversal = packageFixture(t);
  traversal.manifest.files[0].path = "../outside";
  traversal.save();
  await assert.rejects(verifyInstallation(traversal.prefix, COMMIT), /unsafe path/);
  const external = packageFixture(t);
  const installedNode = path.join(external.prefix, "libexec/node/bin/node");
  const outsideNode = path.join(external.root, "external-node");
  fs.renameSync(installedNode, outsideNode);
  fs.symlinkSync(outsideNode, installedNode);
  await assert.rejects(verifyInstallation(external.prefix, COMMIT), /escapes the prefix/);
});

test("CPU-only reports, fabricated Metal results, and foreign daemon PIDs cannot satisfy installed acceptance", () => {
  const report = { schema_version: 1, version: "0.1.0", platform: "macos", architecture: "aarch64", ok: true,
    cpu: { status: "passed", result: [0.5, -3, -6, -1] },
    metal: { required: true, status: "passed", result: [0.5, -3, -6, -1] },
    shared_arrays: { status: "passed", input_storage_unchanged: true, metal_output_cpu_readable: true,
      cpu_to_metal_result: [1.5, -5, -3, -5], metal_to_cpu_result: [1.5, -5, -3, -5] } };
  assertNativeCheck(report, "0.1.0");
  assert.throws(() => assertNativeCheck({ ...report, metal: { required: false, status: "unsupported" } }, "0.1.0"));
  assert.throws(() => assertNativeCheck({ ...report, metal: { ...report.metal, result: [0, 0, 0, 0] } }, "0.1.0"));
  assert.throws(() => assertNativeCheck({ ...report, shared_arrays: { ...report.shared_arrays, input_storage_unchanged: false } }, "0.1.0"));
  const group = process.pid + 10000;
  const daemon = group + 1;
  assertOwnedDaemonPid(group, daemon, [{ pid: daemon, parent: group, group, state: "S" }]);
  assert.throws(() => assertOwnedDaemonPid(group, daemon, [{ pid: daemon, parent: 1, group: group + 2, state: "S" }]));
  assert.throws(() => assertOwnedDaemonPid(group, daemon, [{ pid: daemon, parent: group, group, state: "Z" }]));
});

test("cleanup stops orphaned descendants and a hung child while preserving another owned process group", { timeout: 15000 }, async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-installed-process-test-"));
  const options = (label) => ({ cwd: root, env: process.env, output: root, label });
  const orphanProgram = `
    const { spawn } = require('node:child_process');
    const child = spawn(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); process.stdout.write("ready"); setInterval(() => {}, 1000);'], { stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.once('data', () => {
      process.stdout.write(JSON.stringify({ orphan: child.pid }), () => {
        child.unref(); child.stdout.destroy(); child.stderr.destroy(); process.exit(0);
      });
    });
  `;
  const orphan = new OwnedProcess(process.execPath, ["-e", orphanProgram], options("orphan-parent"));
  const unrelated = new OwnedProcess(process.execPath, ["-e", "process.stdout.write('ready\\n'); setInterval(() => {}, 1000);"], options("unrelated"));
  t.after(async () => {
    try { await Promise.all([orphan.stop(500), unrelated.stop(500)]); }
    finally { fs.rmSync(root, { recursive: true, force: true }); }
  });
  const finished = await orphan.wait(4000);
  assert.equal(finished.code, 0);
  const descendant = JSON.parse(orphan.stdout.trim()).orphan;
  assert(finished.live_before_cleanup.includes(descendant), "owner must observe the orphan before cleanup");
  assert(!(await processTable()).some((entry) => entry.pid === descendant && !entry.state.startsWith("Z")));
  await orphan.stop(500);
  assert.equal((await orphan.liveMembers()).length, 0);
  assert((await unrelated.liveMembers()).length > 0, "cleanup must not signal another group");
  await assert.rejects(unrelated.wait(30), /timed out/);
  await unrelated.stop(500);
  assert.equal((await unrelated.liveMembers()).length, 0);
});

test("smoke cancellation interrupts its wait and removes TERM-ignoring groups within the outer six-second grace", { timeout: 20000 }, async (t) => {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "alloy-smoke-cancellation-test-")));
  const options = (label) => ({ cwd: root, env: process.env, output: root, label });
  const runners = [];
  const targetGroups = new Set();
  const unrelated = new OwnedProcess(process.execPath, ["-e", "process.stdout.write('ready\\n'); setInterval(() => {}, 1000);"], options("unrelated"));
  t.after(async () => {
    try {
      await Promise.all([...runners, unrelated].map((child) => child.stop(500)));
    } finally { fs.rmSync(root, { recursive: true, force: true }); }
  });
  for (const signal of ["SIGINT", "SIGTERM"]) {
    const output = path.join(root, signal);
    fs.mkdirSync(output);
    const program = `
      import assert from 'node:assert/strict';
      import { OwnedProcess, SmokeCancellation } from ${JSON.stringify(new URL("./smoke.mjs", import.meta.url).href)};
      const owned = [];
      const cancellation = new SmokeCancellation(owned);
      const target = new OwnedProcess(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); process.stdout.write("ready"); setInterval(() => {}, 1000);'],
        { cwd: ${JSON.stringify(root)}, env: process.env, output: ${JSON.stringify(output)}, label: 'target' });
      owned.push(target);
      const deadline = Date.now() + 4000;
      while (!target.stdout.includes('ready')) {
        assert(Date.now() < deadline && !target.closed, 'target did not become ready');
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      process.stdout.write(JSON.stringify({ target_group: target.group }) + '\\n');
      let interruption;
      try { await target.wait(60000, cancellation.signal); }
      catch (error) { interruption = error; }
      assert.equal(interruption, cancellation.signal.reason, 'pending wait must reject with the interruption');
      assert(cancellation.signal.aborted && !target.closed, 'wait must abort before the TERM-ignoring process exits');
      const cleanup = await cancellation.cleanup;
      cancellation.dispose();
      assert(cleanup.every(item => item.status === 'fulfilled'), JSON.stringify(cleanup));
      process.stdout.write(JSON.stringify({ cleanup: cleanup.map(item => item.value), wait_aborted_before_close: true }) + '\\n');
      process.exitCode = 143;
    `;
    const runner = new OwnedProcess(process.execPath, ["--input-type=module", "-e", program], options(`runner-${signal.toLowerCase()}`));
    runners.push(runner);
    const readyDeadline = Date.now() + 5000;
    while (!runner.stdout.includes("\n")) {
      assert(Date.now() < readyDeadline && !runner.closed, `smoke subprocess did not become ready: ${runner.stderr}`);
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    const targetGroup = JSON.parse(runner.stdout.split("\n")[0]).target_group;
    assert(Number.isSafeInteger(targetGroup) && targetGroup > 1);
    targetGroups.add(targetGroup);
    const started = performance.now();
    assert(runner.ownedProcess.request(signal, true)); // Signal the smoke owner, not its detached target.
    const exit = await runner.wait(5500);
    assert.equal(exit.code, 143, runner.stderr);
    assert(performance.now() - started < 6000, "smoke cleanup must finish before its outer owner escalates");
    const report = JSON.parse(runner.stdout.trim().split("\n").at(-1));
    assert.equal(report.wait_aborted_before_close, true);
    assert.equal(report.cleanup.length, 1);
    assert.equal(report.cleanup[0].escalated, true);
    assert.deepEqual(report.cleanup[0].remaining_live_pids, []);
    assert(!(await processTable()).some((entry) => entry.group === targetGroup && !entry.state.startsWith("Z")), "owned target survived smoke cancellation");
    targetGroups.delete(targetGroup);
    assert((await unrelated.liveMembers()).length > 0, "smoke cancellation signalled an unrelated group");
  }
});


test("installed plugin ownership requires a live isolated supervisor and private groups", () => {
  const rows = [
    { pid: 100, parent: 1, group: 100, state: "S" },
    { pid: 101, parent: 100, group: 101, state: "S" },
    { pid: 102, parent: 101, group: 102, state: "S" },
  ];
  assert.equal(assertOwnedPluginPid(100, 102, rows), 101);
  for (const change of [
    { index: 2, parent: 100 }, { index: 1, parent: 99 },
    { index: 1, state: "Z" }, { index: 2, state: "Z" },
    { index: 1, group: 100 }, { index: 2, group: 101 },
  ]) {
    const altered = rows.map(row => ({ ...row }));
    Object.assign(altered[change.index], change);
    assert.throws(() => assertOwnedPluginPid(100, 102, altered));
  }
  assert.throws(() => assertOwnedPluginPid(100, 102, rows.filter(row => row.pid !== 101)));
});


test("unestablished supervisor cleanup never becomes a successful smoke cleanup", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-failed-owner-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const owned = new OwnedProcess(path.join(root, "missing-command"), [], { cwd: root, env: process.env, output: root, label: "missing" });
  const outcome = await owned.wait(5000);
  assert(outcome.error, "missing executable must report supervisor failure");
  await assert.rejects(owned.stop(500), /cleanup was not established/);
});
