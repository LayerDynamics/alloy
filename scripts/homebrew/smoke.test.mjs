import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { OwnedProcess, REQUIRED_FILES, assertNativeCheck, assertOwnedDaemonPid, parseArguments, processTable, runSmoke, verifyInstallation } from "./smoke.mjs";

const COMMIT = "c".repeat(40);
const hash = (value) => createHash("sha256").update(value).digest("hex");
const argumentsFor = (prefix, output) => ["--prefix", prefix, "--expected-commit", COMMIT, "--output", output];

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
  assert.equal((await orphan.wait(4000)).code, 0);
  const descendant = JSON.parse(orphan.stdout.trim()).orphan;
  assert((await processTable()).some((entry) => entry.pid === descendant && entry.group === orphan.group && !entry.state.startsWith("Z")));
  const cleaned = await orphan.stop(500);
  assert.equal(cleaned.escalated, true, "SIGTERM-ignoring descendant requires bounded escalation");
  assert.equal((await orphan.liveMembers()).length, 0);
  assert((await unrelated.liveMembers()).length > 0, "cleanup must not signal another group");
  await assert.rejects(unrelated.wait(30), /timed out/);
  await unrelated.stop(500);
  assert.equal((await unrelated.liveMembers()).length, 0);
});
