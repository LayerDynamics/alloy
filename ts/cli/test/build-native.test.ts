import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

type BuildOptions = {
  platform?: string;
  architecture?: string;
  packageRoot?: string;
  compiler?: string;
  compilerPrefix?: string[];
  timeoutMs?: number;
  signal?: AbortSignal;
  report?: (text: string) => void;
};
const { buildNative } = await import(new URL("../../scripts/build-native.mjs", import.meta.url).href) as {
  buildNative(options?: BuildOptions): Promise<{ status: "built" | "skipped"; executable?: string }>;
};

async function fixture(t: TestContext, compiler: string) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "alloy native build "));
  t.after(async () => { await fs.rm(root, { recursive: true, force: true }); });
  await fs.mkdir(path.join(root, "native"));
  await fs.mkdir(path.join(root, "dist", "native"), { recursive: true });
  const source = path.join(root, "native", "macos-metrics.m");
  const executable = path.join(root, "dist", "native", "alloy-metrics");
  const record = path.join(root, "compile.json");
  const script = path.join(root, "compiler.cjs");
  await fs.writeFile(source, "original source");
  await fs.writeFile(executable, "previous complete helper");
  await fs.writeFile(script, `const fs = require('node:fs');
const args = process.argv.slice(2);
const output = args[args.indexOf('-o') + 1];
const record = ${JSON.stringify(record)};
${compiler}
`);
  const options: BuildOptions = {
    platform: "darwin", architecture: "arm64", packageRoot: root,
    compiler: process.execPath, compilerPrefix: [script], timeoutMs: 3000,
    report: () => {},
  };
  return { root, source, executable, record, options };
}

async function assertNoTemporaryBuild(root: string) {
  assert.deepEqual(await fs.readdir(path.join(root, "dist", "native")), ["alloy-metrics"]);
}

test("native build skips unsupported platforms without touching package files or launching a compiler", async () => {
  const messages: string[] = [];
  const result = await buildNative({
    platform: "linux", architecture: "arm64", packageRoot: "/does/not/exist",
    compiler: "/must/not/be/executed", report: (text) => { messages.push(text); },
  });
  assert.equal(result.status, "skipped");
  assert.match(messages.join(""), /skipping.*linux\/arm64/);
});

test("native compilation passes literal package paths and atomically publishes only a complete successful helper", { timeout: 10_000 }, async (t) => {
  const f = await fixture(t, `
fs.writeFileSync(record, JSON.stringify({args, cwd: process.cwd(), pid: process.pid}));
fs.writeFileSync(output, 'new complete helper');
process.once('SIGUSR1', () => process.exit(0));
process.stdout.write('COMPILER_READY\\n');
setInterval(() => {}, 1000);`);
  let ready!: () => void;
  const started = new Promise<void>((resolve) => { ready = resolve; });
  const abort = new AbortController();
  const pending = buildNative({ ...f.options, signal: abort.signal, report: (text) => { if (text.includes("COMPILER_READY")) ready(); } });
  try {
    await Promise.race([started, pending.then(() => { throw new Error("Compiler finished before release"); })]);
    const record = JSON.parse(await fs.readFile(f.record, "utf8")) as { args: string[]; cwd: string; pid: number };
    assert.equal(await fs.readFile(f.executable, "utf8"), "previous complete helper");
    assert.equal(await fs.realpath(record.cwd), await fs.realpath(f.root));
    assert.deepEqual(record.args.slice(0, 9), ["-O2", "-fobjc-arc", "-arch", "arm64", "-mmacosx-version-min=14.0", "-framework", "Foundation", "-framework", "IOKit"]);
    assert.equal(record.args[9], f.source);
    assert.equal(record.args[10], "-o");
    assert.ok(record.args[11]!.startsWith(path.join(f.root, "dist", "native", ".alloy-metrics-build-")));
    process.kill(record.pid, "SIGUSR1");
    const result = await pending;
    assert.equal(result.status, "built");
    assert.equal(result.executable, f.executable);
    assert.equal(await fs.readFile(f.executable, "utf8"), "new complete helper");
    assert.equal((await fs.stat(f.executable)).mode & 0o777, 0o755);
    await assertNoTemporaryBuild(f.root);
  } finally { abort.abort(); await pending.catch(() => {}); }
});

test("compiler failure is nonzero and cannot replace the previously complete helper with partial output", async (t) => {
  const f = await fixture(t, "fs.writeFileSync(output, 'partial'); process.exit(7);");
  await assert.rejects(buildNative(f.options), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /compiler failed.*exit 7/);
    assert.equal((error as Error & { exitCode: number }).exitCode, 7);
    return true;
  });
  assert.equal(await fs.readFile(f.executable, "utf8"), "previous complete helper");
  await assertNoTemporaryBuild(f.root);
});

test("a source change during compilation prevents publishing an obsolete helper", async (t) => {
  const f = await fixture(t, `
const source = args[args.indexOf('-o') - 1];
fs.writeFileSync(output, 'compiled old source');
fs.writeFileSync(source, 'changed source');`);
  await assert.rejects(buildNative(f.options), /source changed during compilation/);
  assert.equal(await fs.readFile(f.executable, "utf8"), "previous complete helper");
  await assertNoTemporaryBuild(f.root);
});

for (const mode of ["abort", "timeout"] as const) {
  test(`native build ${mode} stops the owned compiler descendants and removes partial output`, { timeout: 10_000 }, async (t) => {
    const descendant = `process.on('SIGTERM', () => {}); process.stdout.write('DESCENDANT_READY\\n'); setInterval(() => {}, 1000);`;
    const f = await fixture(t, `
const child = require('node:child_process').spawn(process.execPath, ['-e', ${JSON.stringify(descendant)}], { stdio: ['ignore', 'inherit', 'inherit'] });
fs.writeFileSync(record, JSON.stringify({child: child.pid}));
fs.writeFileSync(output, 'partial');
process.on('SIGTERM', () => process.exit(0));
setInterval(() => {}, 1000);`);
    let ready!: () => void;
    const started = new Promise<void>((resolve) => { ready = resolve; });
    const abort = new AbortController();
    const pending = buildNative({
      ...f.options, signal: abort.signal, timeoutMs: mode === "timeout" ? 2000 : 3000,
      report: (text) => { if (text.includes("DESCENDANT_READY")) ready(); },
    });
    try {
      await Promise.race([started, pending.then(() => { throw new Error("Compiler finished before cancellation"); })]);
      const record = JSON.parse(await fs.readFile(f.record, "utf8")) as { child: number };
      if (mode === "abort") abort.abort(new Error("fixture cancellation"));
      await assert.rejects(pending, mode === "abort" ? /fixture cancellation/ : /timed out/);
      // An orphaned zombie may remain until the host reaps it; it cannot run
      // or retain the compiler pipe. A live descendant is a real leak.
      const state = spawnSync("/bin/ps", ["-o", "stat=", "-p", String(record.child)], { encoding: "utf8" });
      assert.ok(state.status === 1 || state.stdout.trim().startsWith("Z"), `Compiler descendant still running: ${state.stdout}`);
      assert.equal(await fs.readFile(f.executable, "utf8"), "previous complete helper");
      await assertNoTemporaryBuild(f.root);
    } finally { abort.abort(); await pending.catch(() => {}); }
  });
}
