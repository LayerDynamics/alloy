import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import { Writable } from "node:stream";
import { setTimeout as delay, setImmediate as nextTurn } from "node:timers/promises";
import { defaultSettings } from "../src/settings.js";
import { STARTER_CORPUS_PATH } from "../src/model.js";
import { probeDaemon, resolveDaemonLaunch, startDaemon, type DaemonProbe, type DaemonProbeOptions } from "../src/daemon.js";

const status = { api_version: 1 as const, models_resident: 0, memory_reserved: 1024, sessions_active: 0 };
const ready = (): DaemonProbe => ({ status: "ready", error: null, daemon_status: status, can_start: false });
const absent = (): DaemonProbe => ({ status: "unavailable", error: "connect ECONNREFUSED", daemon_status: null, can_start: true });

function workspace(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-start-"));
  t.after(() => { fs.rmSync(root, { recursive: true, force: true }); });
  return root;
}

function script(root: string, name: string, body: string): string {
  const file = path.join(root, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `#!${process.execPath}\n${body}\n`, { mode: 0o755 });
  return file;
}

function capture(): { output: Writable; text: () => string } {
  let text = "";
  return { output: new Writable({ write(chunk, _encoding, callback) { text += String(chunk); callback(); } }), text: () => text };
}

async function until(predicate: () => boolean, timeoutMs = 3000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("test helper did not become ready");
    await delay(10);
  }
}

function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ESRCH") return false;
    throw error;
  }
}

async function stop(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const closed = new Promise<void>((resolve) => { child.once("close", () => { resolve(); }); });
  child.kill("SIGTERM");
  await closed;
}

const runningBody = `
const fs = require('node:fs');
process.on('SIGINT', () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));
fs.writeFileSync(process.env.START_RECORD, JSON.stringify({pid:process.pid,args:process.argv.slice(2)}));
setInterval(() => {}, 1000);
`;

test("readiness validates API-v1 status and closes its read-only connection", async () => {
  const methods: string[] = [];
  let closed = false;
  const result = await probeDaemon("/test.sock", { connect: async () => ({
    call: async (method) => { methods.push(method); return status; }, close: () => { closed = true; },
  }) });
  assert.deepEqual(result, ready());
  assert.deepEqual(methods, ["daemon.status"]);
  assert.equal(closed, true);
  for (const value of [null, {}, { ...status, api_version: 2 }, { ...status, memory_reserved: -1 }, { ...status, models_resident: Number.NaN }]) {
    const failed = await probeDaemon("/test.sock", { connect: async () => ({ call: async () => value, close: () => {} }) });
    assert.equal(failed.status, "incompatible");
    assert.equal(failed.can_start, false);
    assert.ok(failed.error);
  }
});

test("only missing or refused sockets permit a new launch", async () => {
  for (const code of ["ENOENT", "ECONNREFUSED", "EACCES", "EPERM", "ETIMEDOUT"]) {
    const result = await probeDaemon("/test.sock", { connect: async () => { throw Object.assign(new Error(code), { code }); } });
    assert.equal(result.status, "unavailable");
    assert.equal(result.can_start, code === "ENOENT" || code === "ECONNREFUSED");
    assert.equal(result.error, code);
  }
});

test("hung and late readiness connections are bounded and closed", async () => {
  let closed = false;
  const reading = await probeDaemon("/test.sock", { timeoutMs: 20, connect: async () => ({
    call: () => new Promise(() => {}), close: () => { closed = true; },
  }) });
  assert.equal(reading.status, "incompatible");
  assert.equal(reading.can_start, false);
  assert.match(reading.error!, /timed out/);
  assert.equal(closed, true);

  let connect!: (value: { call: () => Promise<unknown>; close: () => void }) => void;
  closed = false;
  const late = await probeDaemon("/test.sock", { timeoutMs: 20, connect: () => new Promise((resolve) => { connect = resolve; }) });
  assert.equal(late.status, "unavailable");
  assert.equal(late.can_start, false);
  connect({ call: async () => { throw new Error("late connector must not make requests"); }, close: () => { closed = true; } });
  await nextTurn();
  assert.equal(closed, true);
});

test("new process readiness also verifies the responding daemon PID", async () => {
  for (const pid of [42, 99]) {
    const methods: string[] = [];
    const result = await probeDaemon("/test.sock", { expectedPid: 42, connect: async () => ({
      call: async (method) => { methods.push(method); return method === "daemon.status" ? status : { schema_version: 1, process_id: pid }; }, close: () => {},
    }) });
    assert.deepEqual(methods, ["daemon.status", "daemon.metrics"]);
    assert.equal(result.status, pid === 42 ? "ready" : "incompatible");
    assert.equal(result.can_start, false);
  }
});

test("launch resolution is read-only and respects explicit binaries and the current source tree", async (t) => {
  const root = workspace(t);
  const marker = path.join(root, "executed");
  const binary = script(root, "configured-daemon", `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'bad');`);
  const cargo = script(root, "cargo", "process.exit(0);");
  fs.mkdirSync(path.join(root, "crates/alloyd"), { recursive: true });
  fs.writeFileSync(path.join(root, "Cargo.toml"), "[workspace]\n");
  fs.writeFileSync(path.join(root, "crates/alloyd/Cargo.toml"), "[package]\nname='alloyd'\n");
  const settings = defaultSettings(root);
  const env = { PATH: root, ALLOYD: binary };
  assert.deepEqual(await resolveDaemonLaunch(settings, { env, sourceRoot: root }), { kind: "binary", command: binary });
  settings.alloyd_bin = path.join(root, "missing");
  assert.equal((await resolveDaemonLaunch(settings, { env, sourceRoot: root })).kind, "unavailable");
  settings.alloyd_bin = "alloyd";
  assert.deepEqual(await resolveDaemonLaunch(settings, { env: { PATH: root }, sourceRoot: root }), { kind: "source", command: cargo, cwd: root });
  assert.equal(fs.existsSync(marker), false);
});

test("ready or incompatible listeners never cause a second process to start", async (t) => {
  const root = workspace(t);
  const marker = path.join(root, "executed");
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model");
  settings.alloyd_bin = script(root, "alloyd", `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'bad');`);
  const output = capture();
  const logs: string[] = [];
  assert.equal(await startDaemon(settings, { probe: async () => ready(), output: output.output, errorOutput: output.output, log: (text) => { logs.push(text); } }), 0);
  assert.match(logs.join("\n"), /Existing daemon configuration was not changed/);
  assert.equal(await startDaemon(settings, { probe: async () => ({ status: "incompatible", error: "wrong service", daemon_status: null, can_start: false }), output: output.output, errorOutput: output.output, log: () => {} }), 1);
  assert.equal(fs.existsSync(marker), false);
});

test("owned foreground startup waits for its PID, passes configuration literally, and cleans up on cancellation", { timeout: 5000 }, async (t) => {
  const root = workspace(t);
  const record = path.join(root, "native.json");
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model with spaces $(literal)");
  settings.alloyd_bin = script(root, "alloyd", runningBody);
  const abort = new AbortController();
  const output = capture();
  const logs: string[] = [];
  const listeners = [process.listenerCount("SIGINT"), process.listenerCount("SIGTERM")];
  const probe = async (_socket: string, options: DaemonProbeOptions): Promise<DaemonProbe> => {
    if (options.expectedPid === undefined || !fs.existsSync(record)) return absent();
    const actual = JSON.parse(fs.readFileSync(record, "utf8")) as { pid: number };
    assert.equal(options.expectedPid, actual.pid);
    return ready();
  };
  const result = await startDaemon(settings, {
    env: { ...process.env, START_RECORD: record }, signal: abort.signal, probe, pollIntervalMs: 10,
    output: output.output, errorOutput: output.output,
    log: (text) => { logs.push(text); if (text.startsWith("READY")) abort.abort(); },
  });
  assert.equal(result, 130);
  assert.ok(logs.some((text) => text.startsWith("READY")));
  const actual = JSON.parse(fs.readFileSync(record, "utf8")) as { pid: number; args: string[] };
  assert.equal(alive(actual.pid), false);
  assert.equal(actual.args[actual.args.indexOf("--artifact") + 1], settings.artifact);
  assert.equal(actual.args[actual.args.indexOf("--corpus-models") + 1], STARTER_CORPUS_PATH);
  assert.equal(actual.args.includes("--cpu"), false);
  assert.equal(actual.args.includes("--metal"), false);
  assert.deepEqual([process.listenerCount("SIGINT"), process.listenerCount("SIGTERM")], listeners);
});

test("startup failure retains the child's error and nonzero outcome", { timeout: 5000 }, async (t) => {
  const root = workspace(t);
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model");
  settings.alloyd_bin = script(root, "alloyd", "console.error('fixture fatal: artifact rejected'); process.exit(7);");
  const output = capture();
  const logs: string[] = [];
  const code = await startDaemon(settings, { probe: async () => absent(), pollIntervalMs: 10,
    output: output.output, errorOutput: output.output, log: (text) => { logs.push(text); } });
  assert.equal(code, 7);
  assert.match(output.text(), /fixture fatal: artifact rejected/);
  assert.match(logs.join("\n"), /exited before API readiness/);
  assert.ok(!logs.some((text) => text.startsWith("READY")));
});

test("a readiness PID mismatch stops only the launched process", { timeout: 5000 }, async (t) => {
  const root = workspace(t);
  const unrelated = spawn(process.execPath, ["-e", "setInterval(()=>{},1000)"], { stdio: "ignore" });
  t.after(async () => { await stop(unrelated); });
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model");
  settings.alloyd_bin = script(root, "alloyd", "setInterval(()=>{},1000);");
  let ownedPid: number | undefined;
  const output = capture();
  const code = await startDaemon(settings, { output: output.output, errorOutput: output.output, log: () => {}, probe: async (socket, options) => {
    if (options.expectedPid === undefined) return absent();
    ownedPid = options.expectedPid;
    return probeDaemon(socket, { ...options, connect: async () => ({
      call: async (method) => method === "daemon.status" ? status : { schema_version: 1, process_id: unrelated.pid }, close: () => {},
    }) });
  } });
  assert.equal(code, 1);
  assert.ok(ownedPid);
  assert.equal(alive(ownedPid), false);
  assert.equal(alive(unrelated.pid!), true);
});

test("an unready child cannot survive the readiness deadline", { timeout: 5000 }, async (t) => {
  const root = workspace(t);
  const record = path.join(root, "native.json");
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model");
  settings.alloyd_bin = script(root, "alloyd", runningBody);
  const output = capture();
  const logs: string[] = [];
  let ownedPid: number | undefined;
  const code = await startDaemon(settings, { env: { ...process.env, START_RECORD: record }, readinessTimeoutMs: 100, pollIntervalMs: 10,
    output: output.output, errorOutput: output.output, log: (text) => { logs.push(text); }, probe: async (_socket, options) => { ownedPid = options.expectedPid ?? ownedPid; return absent(); } });
  assert.equal(code, 1);
  assert.ok(ownedPid);
  assert.equal(alive(ownedPid), false);
  assert.match(logs.join("\n"), /did not become API-ready/);
});

test("source builds launch Cargo's reported executable and honor a nondefault target directory", { timeout: 5000 }, async (t) => {
  const root = workspace(t);
  const sourceDirectory = path.join(root, "source");
  const source = path.join(root, "source-alias");
  fs.mkdirSync(path.join(sourceDirectory, "crates/alloyd"), { recursive: true });
  fs.symlinkSync(sourceDirectory, source, "dir");
  fs.writeFileSync(path.join(source, "Cargo.toml"), "[workspace]\n");
  fs.writeFileSync(path.join(source, "crates/alloyd/Cargo.toml"), "[package]\nname='alloyd'\n");
  const built = script(root, "custom target/debug/actual-daemon", runningBody);
  const record = path.join(root, "native.json");
  const buildRecord = path.join(root, "build.json");
  script(root, "cargo", `
    require('node:fs').writeFileSync(process.env.BUILD_RECORD, JSON.stringify({args:process.argv.slice(2),cwd:process.cwd(),target:process.env.CARGO_TARGET_DIR}));
    process.stdout.write(JSON.stringify({reason:'compiler-artifact',target:{name:'alloyd',kind:['bin']},executable:process.env.BUILT_DAEMON})+'\\n');
  `);
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model");
  settings.corpus_models = path.join(root, "custom corpus.json");
  const output = capture();
  const abort = new AbortController();
  const env = { ...process.env, PATH: root, START_RECORD: record, BUILD_RECORD: buildRecord, BUILT_DAEMON: built, CARGO_TARGET_DIR: path.join(root, "custom target") };
  const code = await startDaemon(settings, { env, sourceRoot: source, signal: abort.signal, output: output.output, errorOutput: output.output, pollIntervalMs: 10,
    probe: async (_socket, options) => options.expectedPid !== undefined && fs.existsSync(record) ? ready() : absent(),
    log: (text) => { if (text.startsWith("READY")) abort.abort(); },
  });
  assert.equal(code, 130);
  const build = JSON.parse(fs.readFileSync(buildRecord, "utf8")) as { args: string[]; cwd: string; target: string };
  assert.deepEqual(build.args, ["build", "--locked", "-p", "alloyd", "--message-format=json-render-diagnostics"]);
  // Child cwd resolves aliases, including macOS's /var -> /private/var.
  assert.equal(fs.realpathSync(build.cwd), fs.realpathSync(source));
  assert.equal(build.target, env.CARGO_TARGET_DIR);
  const native = JSON.parse(fs.readFileSync(record, "utf8")) as { pid: number; args: string[] };
  assert.equal(native.args[native.args.indexOf("--corpus-models") + 1], settings.corpus_models);
  assert.equal(alive(native.pid), false);
});

test("cancelling a source build stops its owned compiler descendants without touching another process", { timeout: 5000 }, async (t) => {
  const root = workspace(t);
  fs.mkdirSync(path.join(root, "crates/alloyd"), { recursive: true });
  fs.writeFileSync(path.join(root, "Cargo.toml"), "[workspace]\n");
  fs.writeFileSync(path.join(root, "crates/alloyd/Cargo.toml"), "[package]\nname='alloyd'\n");
  const record = path.join(root, "compiler.pid");
  const compiler = `require('node:fs').writeFileSync(process.env.COMPILER_RECORD,String(process.pid)); setInterval(()=>{},1000);`;
  script(root, "cargo", `
    const child = require('node:child_process').spawn(process.execPath,['-e',${JSON.stringify(compiler)}],{stdio:'inherit'});
    process.on('SIGINT',()=>{}); process.on('SIGTERM',()=>{});
    child.on('exit',()=>process.exit(0));
  `);
  const unrelated = spawn(process.execPath, ["-e", "setInterval(()=>{},1000)"], { stdio: "ignore" });
  t.after(async () => { await stop(unrelated); });
  const settings = defaultSettings(root);
  settings.artifact = path.join(root, "model");
  const output = capture();
  const abort = new AbortController();
  const pending = startDaemon(settings, { sourceRoot: root, env: { ...process.env, PATH: root, COMPILER_RECORD: record },
    signal: abort.signal, probe: async () => absent(), output: output.output, errorOutput: output.output, log: () => {}, shutdownGraceMs: 100 });
  await until(() => fs.existsSync(record));
  const compilerPid = Number(fs.readFileSync(record, "utf8"));
  abort.abort();
  assert.equal(await pending, 130);
  assert.equal(alive(compilerPid), false);
  assert.equal(alive(unrelated.pid!), true);
});
