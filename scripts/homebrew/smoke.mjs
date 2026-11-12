#!/usr/bin/env node
/** Acceptance of an installed package; this driver never builds or provisions a model. */
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { spawnOwned } from "./group-supervisor.mjs";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { assertStreamingTurn, recordProxy, rpc } from "../../ts/cli/scripts/real-model-acceptance.mjs";

const SCRIPT = fileURLToPath(import.meta.url);
const SOURCE_ROOT = path.resolve(path.dirname(SCRIPT), "../..");
const PROFILE = "qwen3-4b-4bit";
const SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin";
const OUTPUT_LIMIT = 8 * 1024 * 1024;
const execute = promisify(execFile);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Match the wire recorder's monotonic clock, so stdout/terminal ordering is exact.
const now = () => Number(process.hrtime.bigint() / 1000n) / 1000;
const jsonFile = (name, value) => fs.writeFileSync(name, `${JSON.stringify(value, null, 2)}\n`);
export const installedPluginSource = () => `const fs = require("node:fs");
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("end", () => process.stdout.end());
process.stdin.on("data", chunk => {
  input += chunk;
  const end = input.indexOf("\\n");
  if (end < 0) return;
  const request = JSON.parse(input.slice(0, end));
  if (request.method !== "initialize") process.exit(3);
  fs.writeFileSync(process.argv[2], JSON.stringify({ pid: process.pid, method: request.method }));
  process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: request.id, result: {
    protocol_version: 1, plugin_id: "installed-smoke", plugin_version: "1.0.0",
    hooks: ["tool.before_execute", "tool.after_execute"]
  } }) + "\\n");
  input = input.slice(end + 1);
});
`;
const appendJson = (name, value) => fs.appendFileSync(name, `${JSON.stringify(value)}\n`);

export const REQUIRED_FILES = [
  "bin/alloy", "bin/alloyd", "libexec/node/bin/node", "libexec/native/alloyd", "libexec/native/alloy-plugin-pipe-helper", "libexec/native/alloy-plugin-exec-helper", "libexec/native/mlx.metallib",
  "libexec/cli/package.json", "libexec/cli/bin/alloy.js", "libexec/cli/models/starter-model.json",
  "libexec/cli/dist/src/main.js", "libexec/cli/dist/src/version.js", "libexec/cli/dist/native/alloy-metrics",
];

function within(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

export function parseArguments(argv) {
  const options = {};
  const names = new Map([["--prefix", "prefix"], ["--expected-commit", "expectedCommit"], ["--output", "output"], ["--artifact", "artifact"]]);
  for (let i = 0; i < argv.length; i++) {
    const key = names.get(argv[i]);
    assert(key, `Unknown argument: ${argv[i]}`);
    assert(options[key] === undefined, `Duplicate argument: ${argv[i]}`);
    const value = argv[++i];
    assert(typeof value === "string" && value.length > 0 && !value.startsWith("--") && !/[\u0000-\u001f\u007f]/.test(value), `${argv[i - 1]} requires a value`);
    options[key] = value;
  }
  for (const name of ["prefix", "output"]) assert(options[name] && path.isAbsolute(options[name]), `--${name} must be an absolute path`);
  assert.match(options.expectedCommit ?? "", /^[0-9a-f]{40}$/, "--expected-commit must be a full lowercase commit SHA");
  if (options.artifact !== undefined) assert(path.isAbsolute(options.artifact), "--artifact must be an absolute local model path");
  options.prefix = path.resolve(options.prefix);
  options.output = path.resolve(options.output);
  assert(!within(options.prefix, options.output), "--output must be outside the installed prefix");
  return options;
}

async function fileDigest(filename) {
  const before = fs.statSync(filename);
  assert(before.isFile(), `Expected an installed file: ${filename}`);
  const hash = createHash("sha256");
  for await (const chunk of fs.createReadStream(filename)) hash.update(chunk);
  const after = fs.statSync(filename);
  for (const key of ["dev", "ino", "size", "mtimeMs", "ctimeMs"]) assert.equal(after[key], before[key], `Installed file changed while hashing: ${filename}`);
  return { sha256: hash.digest("hex"), bytes: before.size };
}

/** Bind every installed runtime file to the candidate, including the copied Node runtime. */
export async function verifyInstallation(prefix, expectedCommit) {
  const realPrefix = fs.realpathSync(prefix);
  const manifestPath = path.join(realPrefix, "libexec/install.json");
  assert(within(realPrefix, fs.realpathSync(manifestPath)), "Install manifest escapes the prefix");
  assert(fs.statSync(manifestPath).size <= 1024 * 1024, "Install manifest exceeds its size bound");
  const raw = fs.readFileSync(manifestPath);
  const manifest = JSON.parse(raw);
  assert.equal(manifest.schema_version, 1, "Unsupported install manifest schema");
  assert.equal(manifest.source_commit, expectedCommit, "Installed source commit differs from --expected-commit");
  assert.match(manifest.version ?? "", /^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/, "Invalid installed version");
  assert.match(manifest.node_version ?? "", /^v?[0-9]+\.[0-9]+\.[0-9]+$/, "Invalid installed Node version");
  assert(Array.isArray(manifest.files) && manifest.files.length > 0 && manifest.files.length <= 4096, "Invalid installed file inventory");
  const entries = new Map();
  const files = [];
  for (const item of manifest.files) {
    assert(typeof item.path === "string" && item.path !== "." && !path.isAbsolute(item.path)
      && path.posix.normalize(item.path) === item.path && !item.path.startsWith("../")
      && !/[\\\u0000-\u001f\u007f]/.test(item.path), "Install manifest contains an unsafe path");
    assert(!entries.has(item.path), `Duplicate installed path: ${item.path}`);
    assert.match(item.sha256 ?? "", /^[0-9a-f]{64}$/, `Invalid installed checksum: ${item.path}`);
    const filename = fs.realpathSync(path.join(realPrefix, item.path));
    assert(within(realPrefix, filename), `Installed file escapes the prefix: ${item.path}`);
    const measured = await fileDigest(filename);
    assert.equal(measured.sha256, item.sha256, `Installed checksum mismatch: ${item.path}`);
    entries.set(item.path, measured);
    files.push({ path: item.path, ...measured });
  }
  for (const required of REQUIRED_FILES) assert(entries.has(required), `Required installed asset is absent from manifest: ${required}`);
  const found = new Set();
  function inventory(directory, ancestors = new Set()) {
    const real = fs.realpathSync(directory);
    assert(within(realPrefix, real) && !ancestors.has(real), "Installed directory escapes the prefix or contains a cycle");
    const next = new Set(ancestors).add(real);
    for (const name of fs.readdirSync(directory)) {
      const filename = path.join(directory, name);
      const relative = path.relative(realPrefix, filename).split(path.sep).join("/");
      assert(within(realPrefix, fs.realpathSync(filename)), `Installed path escapes the prefix: ${relative}`);
      const info = fs.statSync(filename);
      if (info.isDirectory()) inventory(filename, next);
      else {
        assert(info.isFile(), `Unexpected installed file type: ${relative}`);
        if (relative === "libexec/install.json") continue;
        assert(entries.has(relative), `Installed file is not covered by the manifest: ${relative}`);
        found.add(relative);
        assert(found.size <= 4096, "Installed inventory exceeds its file bound");
      }
    }
  }
  // Homebrew adds its own receipt and .brew metadata after installation. Only
  // Alloy's runtime/documentation roots belong to this package's inventory.
  for (const relative of ["bin", "libexec", "share/doc/alloy"]) {
    const directory = path.join(realPrefix, relative);
    if (fs.existsSync(directory)) inventory(directory);
  }
  assert.equal(found.size, entries.size, "Install manifest and actual inventory differ");
  return { prefix: realPrefix, manifest, manifest_sha256: createHash("sha256").update(raw).digest("hex"), files };
}

export async function processTable(timeoutMs = 3000) {
  const { stdout } = await execute("/bin/ps", ["-axo", "pid=,ppid=,pgid=,stat="], { encoding: "utf8", timeout: timeoutMs, maxBuffer: 4 * 1024 * 1024 });
  return stdout.split("\n").flatMap((line) => {
    const match = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)/.exec(line);
    return match ? [{ pid: Number(match[1]), parent: Number(match[2]), group: Number(match[3]), state: match[4] }] : [];
  });
}

export function assertOwnedDaemonPid(group, daemonPid, table) {
  assert(Number.isSafeInteger(group) && group > 1 && group !== process.pid, "Invalid owned process group");
  assert(Number.isSafeInteger(daemonPid) && daemonPid > 1 && daemonPid !== group, "Daemon did not report a distinct child PID");
  const row = table.find((entry) => entry.pid === daemonPid);
  assert(row && row.group === group && !row.state.startsWith("Z"), `Daemon PID ${daemonPid} is not a live descendant in owned process group ${group}`);
}

export function parseLaunchdJobs(stdout) {
  return stdout.split("\n").flatMap(line => {
    const match = /^\s*(\d+)\s+-?\d+\s+(com\.layerdynamics\.alloy\.cohort\.\d+\.\d+\.[1-9]\d*\.[0-9a-f]{32})\s*$/.exec(line);
    return match ? [{ pid: Number(match[1]), label: match[2] }] : [];
  });
}

/** Verify the installed launchd cleaner -> exclusive gate -> plugin chain. */
export function assertOwnedPluginPid(daemonPid, pluginPid, table, jobs, uid = process.getuid()) {
  assert(Number.isSafeInteger(pluginPid) && pluginPid > 1 && pluginPid !== daemonPid, "Invalid installed plugin PID");
  const plugin = table.find(entry => entry.pid === pluginPid);
  assert(plugin && plugin.group === pluginPid && !plugin.state.startsWith("Z"), "Installed plugin is not a live private-group leader");
  const supervisor = table.find(entry => entry.pid === plugin.parent);
  assert(supervisor && supervisor.pid !== daemonPid &&
    supervisor.group === supervisor.pid && !supervisor.state.startsWith("Z"),
  "Installed plugin has no live isolated gate");
  const cleaner = table.find(entry => entry.pid === supervisor.parent);
  assert(cleaner && cleaner.pid !== daemonPid && cleaner.parent === 1 && cleaner.group === cleaner.pid && !cleaner.state.startsWith("Z"),
    "Installed plugin has no live launchd cleaner");
  const job = jobs.find(entry => entry.pid === cleaner.pid &&
    entry.label.startsWith(`com.layerdynamics.alloy.cohort.${uid}.${daemonPid}.`) &&
    /^com\.layerdynamics\.alloy\.cohort\.\d+\.\d+\.[1-9]\d*\.[0-9a-f]{32}$/.test(entry.label));
  assert(job, "Installed cleaner has no launchd registration for this daemon");
  const holders = table.filter(entry => entry.parent === supervisor.pid && entry.pid !== pluginPid);
  assert(holders.length === 1 && holders[0].group === supervisor.pid && !holders[0].state.startsWith("Z"),
    "Installed gate has no unique live independent revoker");
  return { supervisor_pid: supervisor.pid, cleaner_pid: cleaner.pid, revoker_pid: holders[0].pid, cohort_job: job.label };
}

export async function waitForPluginCleanup(pluginPid, ownership, { read = processTable, timeoutMs = 5000 } = {}) {
  const owned = [pluginPid, ownership.supervisor_pid, ownership.cleaner_pid, ownership.revoker_pid];
  const until = Date.now() + timeoutMs;
  while ((await read()).some(entry => owned.includes(entry.pid))) {
    assert(Date.now() < until, "Installed plugin, gate, cleaner or revoker remained after daemon stop");
    await sleep(50);
  }
}

/** A fresh process group owns the complete child tree, even if its parent exits first. */
export class OwnedProcess {
  constructor(command, args, { cwd, env, output, label, recordChunks = false }) {
    assert.match(label, /^[a-z0-9-]+$/);
    this.label = label;
    this.stdout = "";
    this.stderr = "";
    this.stdoutChunks = [];
    this.closed = false;
    this.retired = false;
    this.error = null;
    this.output = output;
    const stdoutFile = path.join(output, `${label}.stdout.log`);
    const stderrFile = path.join(output, `${label}.stderr.log`);
    fs.writeFileSync(stdoutFile, "", { flag: "wx" });
    fs.writeFileSync(stderrFile, "", { flag: "wx" });
    this.ownedProcess = spawnOwned(command, args, { cwd, env });
    this.child = this.ownedProcess.child;
    this.group = undefined;
    this.ready = this.ownedProcess.ready.then(group => {
      if (group !== null) {
        this.group = group;
        try {
          appendJson(path.join(output, "processes.ndjson"), { command, args, cwd, pid: group, process_group: group, label, at: new Date().toISOString() });
        } catch (error) { this.error ??= error; this.signal("SIGKILL"); }
      }
      return group;
    });
    this.child.stdout.setEncoding("utf8");
    this.child.stderr.setEncoding("utf8");
    let bytes = 0;
    const consume = (stream, file, text) => {
      try {
        bytes += Buffer.byteLength(text);
        assert(bytes <= OUTPUT_LIMIT, `${label} exceeded the ${OUTPUT_LIMIT}-byte output bound`);
        const observation = { at_ms: now(), text };
        fs.appendFileSync(file, text);
        this[stream] += text;
        if (stream === "stdout" && recordChunks) {
          assert(this.stdoutChunks.length < 65536, "Too many CLI stdout chunks");
          this.stdoutChunks.push(observation);
          appendJson(path.join(output, `${label}.stdout-chunks.ndjson`), observation);
        }
      } catch (error) {
        this.error ??= error;
        this.signal("SIGKILL");
      }
    };
    this.child.stdout.on("data", (text) => consume("stdout", stdoutFile, text));
    this.child.stderr.on("data", (text) => consume("stderr", stderrFile, text));
    this.done = this.ownedProcess.done.then(({ code, signal, error, live_before_cleanup }) => {
      this.closed = true;
      this.retired = true;
      this.error ??= error ? new Error(error) : null;
      this.exit = { code, signal, error: this.error?.message ?? null, live_before_cleanup };
      return this.exit;
    });
  }

  signal(signal) {
    if (this.retired) return;
    this.ownedProcess.request(signal);
  }

  async wait(timeoutMs, signal) {
    let timer;
    let onAbort;
    try {
      signal?.throwIfAborted();
      const pending = [this.done, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${this.label} timed out after ${timeoutMs} ms`)), timeoutMs);
      })];
      if (signal) pending.push(new Promise((_, reject) => {
        onAbort = () => reject(signal.reason);
        signal.addEventListener("abort", onAbort, { once: true });
      }));
      return await Promise.race(pending);
    } finally {
      clearTimeout(timer);
      if (onAbort) signal.removeEventListener("abort", onAbort);
    }
  }

  async liveMembers(timeoutMs = 3000) {
    await this.ready;
    if (this.retired || this.group === undefined) return [];
    const members = (await processTable(timeoutMs)).filter((entry) => entry.group === this.group && !entry.state.startsWith("Z"));
    if (members.length === 0) this.retired = true;
    return members;
  }

  async stop(graceMs = 5000) {
    let members = await this.liveMembers();
    let escalated = false;
    if (members.length > 0) {
      this.signal("SIGTERM");
      for (const signal of ["SIGTERM", "SIGKILL"]) {
        if (signal === "SIGKILL") { escalated = true; this.signal(signal); }
        const deadline = Date.now() + graceMs;
        while (Date.now() < deadline && (members = await this.liveMembers()).length > 0) await sleep(50);
        if (members.length === 0) break;
      }
    }
    assert.equal((await this.liveMembers()).length, 0, `Owned process group ${this.group} survived cleanup`);
    const outcome = await this.wait(graceMs);
    assert.equal(outcome.error, null, `Owned process cleanup was not established: ${outcome.error}`);
    const result = { label: this.label, process_group: this.group, escalated, remaining_live_pids: [] };
    appendJson(path.join(this.output, "cleanup.ndjson"), result);
    return result;
  }

  async stopAfterInterrupt() {
    // All live groups use this same three-second budget concurrently. Escalation
    // has its own timer so a delayed process-table probe cannot postpone SIGKILL.
    const deadline = Date.now() + 3000;
    let escalated = false;
    let signalError;
    const kill = () => {
      if (this.retired) return;
      escalated = true;
      try { this.signal("SIGKILL"); } catch (error) { signalError ??= error; }
    };
    const escalation = setTimeout(kill, 1500);
    try {
      this.signal("SIGTERM");
      while (true) {
        const remaining = deadline - Date.now();
        assert(remaining > 0, `Owned process group ${this.group} exceeded cancellation cleanup deadline`);
        const members = await this.liveMembers(Math.min(500, remaining));
        if (members.length === 0) break;
        await sleep(Math.min(50, Math.max(1, deadline - Date.now())));
      }
      const outcome = await this.wait(Math.max(1, deadline - Date.now()));
      assert.equal(outcome.error, null, `Owned process cleanup was not established: ${outcome.error}`);
      if (signalError) throw signalError;
      const result = { label: this.label, process_group: this.group, escalated, remaining_live_pids: [] };
      appendJson(path.join(this.output, "cleanup.ndjson"), result);
      return result;
    } finally {
      clearTimeout(escalation);
      // A failed probe still must not leave a TERM-ignoring owned child alive.
      if (!this.retired) kill();
    }
  }
}

/** Signal handling shared by installed acceptance and its subprocess test. */
export class SmokeCancellation {
  constructor(owned) {
    this.owned = owned;
    this.controller = new AbortController();
    this.signal = this.controller.signal;
    this.cleanup = null;
    this.onInt = () => this.interrupt("SIGINT");
    this.onTerm = () => this.interrupt("SIGTERM");
    process.on("SIGINT", this.onInt);
    process.on("SIGTERM", this.onTerm);
  }

  interrupt(signal) {
    if (this.signal.aborted) return;
    // Begin every owned group's cleanup before unwinding a pending command.
    // Retired groups remain retired and receive no further signals.
    this.cleanup = Promise.allSettled(this.owned.filter((child) => !child.retired || !child.closed)
      .map((child) => child.stopAfterInterrupt()));
    this.controller.abort(new Error(`Installed acceptance interrupted by ${signal}`));
  }

  check() { this.signal.throwIfAborted(); }

  dispose() {
    process.off("SIGINT", this.onInt);
    process.off("SIGTERM", this.onTerm);
  }
}

export function assertNativeCheck(report, version) {
  assert.equal(report.schema_version, 1);
  assert.equal(report.version, version);
  assert.equal(report.platform, "macos");
  assert.equal(report.architecture, "aarch64");
  assert.equal(report.ok, true, report.error ?? "Installed native check failed");
  assert.equal(report.cpu?.status, "passed");
  assert.deepEqual(report.cpu?.result, [0.5, -3, -6, -1]);
  assert.equal(report.metal?.required, true, "Metal must be required on Apple Silicon");
  assert.equal(report.metal?.status, "passed", "Installed Metal execution is required");
  assert.deepEqual(report.metal?.result, report.cpu.result);
  assert.equal(report.shared_arrays?.status, "passed");
  assert.equal(report.shared_arrays?.input_storage_unchanged, true);
  assert.equal(report.shared_arrays?.metal_output_cpu_readable, true);
  assert.deepEqual(report.shared_arrays?.cpu_to_metal_result, [1.5, -5, -3, -5]);
  assert.deepEqual(report.shared_arrays?.metal_to_cpu_result, [1.5, -5, -3, -5]);
}

function isolatedEnvironment(config) {
  const env = { ...process.env, PATH: SYSTEM_PATH, ALLOY_CONFIG_DIR: config };
  for (const key of Object.keys(env)) {
    if (["ALLOYD", "NODE_PATH", "NODE_OPTIONS", "MLX_METAL_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD"].includes(key) || key.startsWith("DYLD_")) delete env[key];
  }
  return env;
}

function summary(result) {
  const value = { test: result.test, mode: result.mode, status: result.status, source_commit: result.source_commit,
    prefix: result.prefix, version: result.version, node_version: result.node_version,
    manifest_sha256: result.manifest_sha256, verified_files: result.verified_files, installation_unchanged: result.installation_unchanged,
    native: result.native, monitor: result.monitor, model: result.model, stream: result.stream,
    daemon_pid: result.daemon_pid, final_residency: result.final_residency, foreground_stop: result.foreground_stop,
    cleanup: result.cleanup, error: result.error?.slice(0, 1500), output: result.output };
  if (value.stream) { value.stream = { ...value.stream }; delete value.stream.prompt; delete value.stream.assistant_text; }
  let encoded = JSON.stringify(value, (_key, item) => typeof item === "string" && item.length > 1024 ? `${item.slice(0, 1024)}…` : item);
  if (Buffer.byteLength(encoded) > 16384) {
    encoded = JSON.stringify({ test: result.test, mode: result.mode, status: result.status, source_commit: result.source_commit,
      version: result.version, manifest_sha256: result.manifest_sha256, installation_unchanged: result.installation_unchanged,
      summary_truncated: true, detail: "Full provenance, metrics, and process evidence are retained in result.json." });
  }
  console.log(`ALLOY_INSTALLED_ACCEPTANCE_RESULT ${encoded}`);
}

export async function runSmoke(options) {
  fs.mkdirSync(path.dirname(options.output), { recursive: true });
  fs.mkdirSync(options.output); // Never overwrite an earlier acceptance run.
  const runtime = fs.mkdtempSync("/tmp/alloy-install-smoke-");
  const cwd = path.join(runtime, "cwd");
  fs.mkdirSync(cwd);
  const config = path.join(runtime, "config");
  const socket = path.join(runtime, "daemon.sock");
  const env = isolatedEnvironment(config);
  const owned = [];
  let pluginMarker;
  const result = { schema_version: 1, test: "homebrew_installed_package", mode: options.artifact === undefined ? "native_installation" : "real_model", status: "running", started_at: new Date().toISOString(),
    source_commit: options.expectedCommit, prefix: options.prefix, output: options.output,
    cwd, runtime_path: SYSTEM_PATH, artifact_requested: options.artifact ?? null, cleanup: [] };
  const resultFile = path.join(options.output, "result.json");
  jsonFile(resultFile, result);
  const cancellation = new SmokeCancellation(owned);
  const check = () => cancellation.check();
  const launch = (command, args, label, recordChunks = false) => {
    check();
    const process = new OwnedProcess(command, args, { cwd, env, output: options.output, label, recordChunks });
    owned.push(process);
    return process;
  };
  const command = async (file, args, label, { timeout = 30000, codes = [0], recordChunks = false } = {}) => {
    const process = launch(file, args, label, recordChunks);
    try {
      const exit = await process.wait(timeout, cancellation.signal);
      assert.equal(exit.error, null, `${label}: ${exit.error}`);
      assert(codes.includes(exit.code), `${label} failed (${exit.signal ?? exit.code}); see ${label}.stderr.log\n${process.stderr.slice(-2000)}`);
      return process;
    } finally {
      if (!cancellation.signal.aborted) {
        const stopped = await process.stop();
        if (!cancellation.signal.aborted) result.cleanup.push(stopped);
      }
    }
  };
  let failure;
  try {
    assert.equal(process.platform, "darwin", "Installed Homebrew acceptance requires real Apple Silicon macOS");
    assert.equal(process.arch, "arm64", "Installed Homebrew acceptance requires native arm64 Node");
    assert(typeof process.geteuid === "function" && process.geteuid() > 0, "Installed acceptance must run as an unprivileged user");
    assert(!within(SOURCE_ROOT, cwd), "Installed commands must run outside the source checkout");
    const installation = await verifyInstallation(options.prefix, options.expectedCommit);
    const { manifest } = installation;
    Object.assign(result, { version: manifest.version, node_version: manifest.node_version,
      manifest_sha256: installation.manifest_sha256, verified_files: installation.files.length });
    jsonFile(path.join(options.output, "installation.json"), installation);
    const cli = path.join(options.prefix, "bin/alloy");
    const daemon = path.join(options.prefix, "bin/alloyd");
    const node = path.join(options.prefix, "libexec/node/bin/node");
    const nodeVersion = await command(node, ["--version"], "node-version");
    assert.equal(nodeVersion.stdout.trim().replace(/^v/, ""), manifest.node_version.replace(/^v/, ""));
    assert.equal((await command(cli, ["--version"], "cli-version")).stdout, `alloy ${manifest.version}\n`);
    assert.equal((await command(cli, ["-V"], "cli-version-short")).stdout, `alloy ${manifest.version}\n`);
    assert.equal((await command(daemon, ["--version"], "daemon-version")).stdout, `alloyd ${manifest.version}\n`);
    assert.match((await command(cli, ["--help"], "cli-help")).stdout, /ALLOY_CONFIG_DIR/);
    const daemonHelp = await command(daemon, ["--help"], "daemon-help");
    assert.match(daemonHelp.stdout + daemonHelp.stderr, /--check-native/);
    const doctor = JSON.parse((await command(cli, ["doctor", "--json"], "doctor", { codes: [1] })).stdout);
    assert.equal(doctor.schema_version, 1);
    assert.equal(doctor.configuration?.status, "missing");
    assert.equal(doctor.launch?.kind, "binary", "Installed CLI attempted a source build or failed to find its daemon");
    assert.equal(fs.realpathSync(doctor.launch.command), fs.realpathSync(path.join(options.prefix, "libexec/native/alloyd")), "Installed CLI selected another daemon");
    const missing = path.join(runtime, "missing-daemon");
    const refused = JSON.parse((await command(cli, ["doctor", "--json", "--alloyd-bin", missing], "missing-override", { codes: [1] })).stdout);
    assert.equal(refused.launch?.kind, "unavailable", "Explicit missing daemon override was silently replaced");
    assert(refused.launch.error.includes(missing));
    assert(!fs.existsSync(config), "Read-only installed diagnostics created user configuration");
    jsonFile(path.join(options.output, "doctor.json"), doctor);
    result.native = JSON.parse((await command(daemon, ["--check-native"], "native-check", { timeout: 60000 })).stdout);
    assertNativeCheck(result.native, manifest.version);
    const monitor = JSON.parse((await command(cli, ["monitor", "--once", "--json"], "monitor")).stdout);
    jsonFile(path.join(options.output, "monitor.json"), monitor);
    assert.equal(monitor.schema_version, 1);
    assert.equal(monitor.host?.platform, "darwin");
    assert.equal(monitor.host?.architecture, "arm64");
    const hardware = monitor.host?.accelerators;
    assert.equal(hardware?.status, "ready", `Installed native sampler failed: ${hardware?.error}`);
    assert.equal(hardware.source, "ioreport");
    assert.equal(hardware.effective_uid, process.geteuid(), "Native sampler ran with a different effective user");
    assert(Number.isFinite(hardware.sample_duration_ms) && hardware.sample_duration_ms > 0);
    assert([hardware.cpu_power_mw, hardware.gpu_power_mw, hardware.ane_power_mw].some((value) => typeof value === "number" && Number.isFinite(value) && value >= 0), "Native sampler produced no real power counters");
    result.monitor = { source: hardware.source, status: hardware.status, effective_uid: hardware.effective_uid,
      sample_duration_ms: hardware.sample_duration_ms, cpu_power_mw: hardware.cpu_power_mw,
      gpu_power_mw: hardware.gpu_power_mw, ane_power_mw: hardware.ane_power_mw, warnings: hardware.warnings };
    if (options.artifact !== undefined) {
      const pins = JSON.parse(fs.readFileSync(path.join(options.prefix, "libexec/cli/models/starter-model.json"), "utf8"));
      const model = pins.profiles?.find((entry) => entry.id === PROFILE)?.artifact;
      assert(model?.revision && model?.sha256 && model.files?.length, "Installed starter pins are incomplete");
      result.model = { path: options.artifact, repository: model.repository, revision: model.revision, sha256: model.sha256 };
      assert(fs.existsSync(path.join(options.artifact, "config.json")), `Pinned model missing at ${options.artifact}; supply the complete local ${model.repository}@${model.revision} artifact (SHA-256 ${model.sha256}). This test does not download models.`);
      const owner = launch(cli, ["start", "--artifact", options.artifact, "--socket", socket, "--db", path.join(options.output, "sessions.sqlite")], "start");
      assert(await owner.ready, "Could not start foreground CLI supervisor");
      const deadline = Date.now() + 240000;
      let ready;
      let lastError = "no API response";
      while (Date.now() < deadline) {
        check();
        assert(!owner.closed, `Installed start exited before readiness; see start.stderr.log\n${owner.stderr.slice(-2000)}`);
        try {
          const status = await rpc(socket, "daemon.status", {}, 2000);
          const metrics = await rpc(socket, "daemon.metrics", {}, 2000);
          assert.equal(status.api_version, 1);
          assert.equal(metrics.schema_version, 1);
          assert.equal(metrics.residency?.worker_alive, true);
          assertOwnedDaemonPid(owner.group, metrics.process_id, await processTable());
          ready = { owner_pid: owner.group, daemon_pid: metrics.process_id, status, metrics };
          break;
        } catch (error) {
          lastError = error.message;
          appendJson(path.join(options.output, "readiness.ndjson"), { at: new Date().toISOString(), error: lastError });
        }
        await sleep(200);
      }
      assert(ready, `Installed daemon did not become API/PID-ready within 240000 ms: ${lastError}`);
      result.daemon_pid = ready.daemon_pid;
      jsonFile(path.join(options.output, "ready.json"), ready);
      const proxyPath = path.join(runtime, "record.sock");
      const proxy = await recordProxy(proxyPath, socket, path.join(options.output, "chat.rpc.ndjson"));
      try {
        const chat = await command(cli, ["chat", "--socket", proxyPath, "-m", "Reply with exactly these eight words separated by spaces: amber birch cobalt dune ember fern grove harbor. /no_think"], "chat", { timeout: 180000, recordChunks: true });
        assert.equal(proxy.errors.length, 0, `Stream recording failed: ${proxy.errors.join("; ")}`);
        assert(!chat.stderr.includes("without streamed text"), "Installed CLI used a final-only text fallback");
        result.stream = assertStreamingTurn(proxy.records, chat.stdout, chat.stdoutChunks, 2);
        jsonFile(path.join(options.output, "turn.json"), result.stream);
      } finally { await proxy.close(); }
      const metrics = await rpc(socket, "daemon.metrics");
      assert.equal(metrics.process_id, ready.daemon_pid);
      assert(metrics.residency?.models_resident > 0);
      assert(metrics.residency.models.some((item) => item.artifact_id === PROFILE && item.weights_reserved_bytes > 0));
      result.final_residency = metrics.residency;
      jsonFile(path.join(options.output, "final-metrics.json"), metrics);
      // Interrupt only the CLI owner here. Group cleanup below is a failure
      // fallback, so this gate proves that ordinary `alloy start` stops its child.
      assert(owner.ownedProcess.request("SIGINT", true), "Could not interrupt the owned foreground CLI");
      const stopped = await owner.wait(15000, cancellation.signal);
      assert.equal(stopped.error, null);
      assert.equal(stopped.code, 130, "Foreground start did not preserve Ctrl+C exit status");
      assert.deepEqual(stopped.live_before_cleanup, [], "Foreground CLI left a live daemon/descendant after interruption");
      await assert.rejects(rpc(socket, "daemon.status", {}, 1000), "Daemon API still responds after owner cleanup");
      result.foreground_stop = { owner_pid: owner.group, daemon_pid: ready.daemon_pid, exit: stopped, live_descendants: [] };
      jsonFile(path.join(options.output, "foreground-stop.json"), result.foreground_stop);

      // Exercise the packaged daemon's default sibling-helper discovery using
      // an actual JSON-RPC child, independently of model-generated tool calls.
      const pluginFile = path.join(runtime, "installed-plugin.cjs");
      pluginMarker = path.join(runtime, "installed-plugin.json");
      const pluginSocket = path.join(runtime, "plugin.sock");
      const toolsFile = path.join(runtime, "tools.json");
      fs.writeFileSync(pluginFile, installedPluginSource());
      jsonFile(toolsFile, {
        read_root: cwd, write_root: cwd, shell_executables: [fs.realpathSync("/bin/echo")],
        shell_cwd_roots: [cwd], http_destinations: [{ origin: "http://fixture.invalid/", address: "127.0.0.1" }],
        policy: { rules: [] },
        plugins: [{ id: "installed-smoke", version: "1.0.0", program: fs.realpathSync(node),
          args: [pluginFile, pluginMarker], cwd, sandbox: { mode: "trusted" } }]
      });
      const pluginOwner = launch(daemon, ["--artifact", options.artifact, "--socket", pluginSocket,
        "--db", path.join(runtime, "plugin.sqlite"), "--tools-config", toolsFile], "installed-plugin");
      const pluginDeadline = Date.now() + 240000;
      let pluginReady = false;
      let pluginError = "no API response";
      while (Date.now() < pluginDeadline) {
        check();
        assert(!pluginOwner.closed, `Installed plugin daemon exited before readiness: ${pluginOwner.stderr.slice(-2000)}`);
        try {
          const status = await rpc(pluginSocket, "daemon.status", {}, 2000);
          assert.equal(status.api_version, 1);
          assert(fs.existsSync(pluginMarker), "Plugin handshake marker was not written");
          pluginReady = true;
          break;
        } catch (error) { pluginError = error.message; }
        await sleep(200);
      }
      assert(pluginReady, `Installed plugin daemon did not become ready: ${pluginError}`);
      const pluginObservation = JSON.parse(fs.readFileSync(pluginMarker, "utf8"));
      assert.equal(pluginObservation.method, "initialize");
      assert(Number.isInteger(pluginObservation.pid) && pluginObservation.pid > 1);
      await pluginOwner.ready;
      const { stdout: registeredJobs } = await execute("/bin/launchctl", ["list"], { encoding: "utf8", timeout: 3000, maxBuffer: 4 * 1024 * 1024 });
      const ownership = assertOwnedPluginPid(pluginOwner.group, pluginObservation.pid, await processTable(), parseLaunchdJobs(registeredJobs));
      const pluginStop = await pluginOwner.stop();
      await waitForPluginCleanup(pluginObservation.pid, ownership);
      result.plugin_startup = { ...pluginObservation, daemon_pid: pluginOwner.group, ...ownership, cleanup: pluginStop };
      jsonFile(path.join(options.output, "installed-plugin.json"), result.plugin_startup);
    }
    check();
    const after = await verifyInstallation(options.prefix, options.expectedCommit);
    assert.equal(after.manifest_sha256, installation.manifest_sha256, "Install manifest changed during acceptance");
    assert.deepEqual(after.files, installation.files, "Installed files changed during acceptance");
    result.installation_unchanged = true;
  } catch (error) { failure = error; }
  finally {
    for (const child of owned) {
      if (cancellation.signal.aborted) break;
      try { if (!child.retired || !child.closed) result.cleanup.push(await child.stop()); }
      catch (error) { failure ??= error; }
    }
    if (cancellation.signal.aborted) {
      failure ??= cancellation.signal.reason;
      for (const cleanup of await cancellation.cleanup) {
        if (cleanup.status === "fulfilled") result.cleanup.push(cleanup.value);
        else failure ??= cleanup.reason;
      }
    }
    cancellation.dispose();
    try {
      // The fixture exits on stdin EOF. Its private process group is distinct
      // from the daemon's, so observe it separately on success and failure.
      if (pluginMarker && fs.existsSync(pluginMarker)) {
        const { pid } = JSON.parse(fs.readFileSync(pluginMarker, "utf8"));
        assert(Number.isInteger(pid) && pid > 1, "Invalid installed plugin PID");
        const deadline = Date.now() + 5000;
        while ((await processTable()).some(entry => entry.pid === pid && !entry.state.startsWith("Z"))) {
          assert(Date.now() < deadline, `Installed plugin ${pid} survived daemon cleanup`);
          await sleep(50);
        }
      }
    } catch (error) { failure ??= error; }
    try {
      const settingsFile = path.join(config, "settings.json");
      if (fs.existsSync(settingsFile)) fs.copyFileSync(settingsFile, path.join(options.output, "settings.json"));
    } catch (error) { failure ??= error; }
    try { fs.rmSync(runtime, { recursive: true, force: true }); }
    catch (error) { failure ??= error; }
  }
  result.status = failure ? "failed" : "passed";
  result.finished_at = new Date().toISOString();
  if (failure) result.error = failure.stack ?? String(failure);
  jsonFile(resultFile, result);
  summary(result);
  if (failure) throw failure;
  return result;
}

if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT) {
  if (process.argv.length === 3 && ["--help", "-h"].includes(process.argv[2])) {
    console.log("Usage: node scripts/homebrew/smoke.mjs --prefix ABSOLUTE --expected-commit SHA --output NEW_DIRECTORY [--artifact ABSOLUTE_PINNED_QWEN3_4B]\nRequires unprivileged Apple Silicon macOS. Evidence is retained on success or failure; no builds or downloads run.");
  } else {
    Promise.resolve().then(() => runSmoke(parseArguments(process.argv.slice(2)))).catch((error) => {
      console.error(error.stack ?? String(error));
      process.exitCode = 1;
    });
  }
}
