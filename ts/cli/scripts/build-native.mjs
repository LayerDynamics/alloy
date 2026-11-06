import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const OUTPUT_LIMIT = 1024 * 1024;

class NativeBuildError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}

function checkAbort(signal) {
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new NativeBuildError("Build cancelled", 130);
  }
}

/** Own the compiler and its descendants, including on interruption or timeout. */
async function compile(command, args, { cwd, timeoutMs, signal, report }) {
  checkAbort(signal);
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    let spawnError = null;
    let stopError = null;
    let outputBytes = 0;
    let escalation;
    const killGroup = (name) => {
      if (child.pid === undefined) return;
      try { process.kill(-child.pid, name); }
      catch (error) { if (error.code !== "ESRCH") throw error; }
    };
    const stop = (error) => {
      if (stopError) return;
      stopError = error;
      try { killGroup("SIGTERM"); }
      catch (killError) { stopError = killError; }
      escalation = setTimeout(() => {
        try { killGroup("SIGKILL"); }
        catch (killError) { stopError = killError; }
      }, 250);
    };
    const onAbort = () => stop(signal.reason instanceof Error ? signal.reason : new NativeBuildError("Build cancelled", 130));
    const deadline = setTimeout(() => stop(new NativeBuildError(`Native monitor compilation timed out after ${timeoutMs} ms`)), timeoutMs);
    const consume = (chunk) => {
      outputBytes += Buffer.byteLength(chunk);
      if (outputBytes > OUTPUT_LIMIT) {
        stop(new NativeBuildError("Native monitor compiler output exceeded its limit"));
        return;
      }
      try { report(chunk.toString()); }
      catch (error) { stop(error); }
    };
    child.stdout.on("data", consume);
    child.stderr.on("data", consume);
    child.once("error", (error) => { spawnError = error; });
    child.once("close", (code, name) => {
      clearTimeout(deadline);
      clearTimeout(escalation);
      signal?.removeEventListener("abort", onAbort);
      // A compiler wrapper must not leave descendants behind after exiting.
      try { killGroup("SIGKILL"); }
      catch (error) { stopError ??= error; }
      if (stopError) reject(stopError);
      else if (spawnError) reject(new NativeBuildError(`Cannot launch the native monitor compiler: ${spawnError.message}`));
      else if (code !== 0) reject(new NativeBuildError(`Native monitor compiler failed (${name ?? `exit ${code}`})`, code && code > 0 ? code : 1));
      else resolve();
    });
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}

/** Build-time only. Runtime monitoring never compiles or installs a helper. */
export async function buildNative(options = {}) {
  const platform = options.platform ?? process.platform;
  const architecture = options.architecture ?? process.arch;
  const report = options.report ?? ((text) => { process.stderr.write(text); });
  if (platform !== "darwin" || architecture !== "arm64") {
    report(`Native monitor: skipping Apple Silicon helper on ${platform}/${architecture}.\n`);
    return { status: "skipped" };
  }
  const root = options.packageRoot ?? packageRoot;
  const signal = options.signal;
  const timeoutMs = options.timeoutMs ?? 60_000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new NativeBuildError("Invalid native monitor build timeout");
  checkAbort(signal);
  const source = path.join(root, "native", "macos-metrics.m");
  const original = await fs.readFile(source);
  const outputDirectory = path.join(root, "dist", "native");
  const executable = path.join(outputDirectory, "alloy-metrics");
  await fs.mkdir(outputDirectory, { recursive: true });
  const temporary = await fs.mkdtemp(path.join(outputDirectory, ".alloy-metrics-build-"));
  const candidate = path.join(temporary, "alloy-metrics");
  try {
    report("Native monitor: building the unprivileged Apple Silicon sampler…\n");
    await compile(options.compiler ?? "/usr/bin/xcrun", [
      ...(options.compilerPrefix ?? ["clang"]),
      "-O2", "-fobjc-arc", "-arch", "arm64", "-mmacosx-version-min=14.0",
      "-framework", "Foundation", "-framework", "IOKit", source, "-o", candidate,
    ], { cwd: root, timeoutMs, signal, report });
    checkAbort(signal);
    if (!(await fs.readFile(source)).equals(original)) {
      throw new NativeBuildError("Native monitor source changed during compilation; run the build again");
    }
    const result = await fs.stat(candidate);
    if (!result.isFile() || result.size === 0) throw new NativeBuildError("Native monitor compiler did not produce an executable");
    await fs.chmod(candidate, 0o755);
    checkAbort(signal);
    // The prior complete build remains intact until the new build succeeds.
    await fs.rename(candidate, executable);
    report("Native monitor: unprivileged sampler ready.\n");
    return { status: "built", executable };
  } finally {
    await fs.rm(temporary, { recursive: true, force: true });
  }
}

async function main() {
  const abort = new AbortController();
  let outputFailed = false;
  const onInt = () => abort.abort(new NativeBuildError("Native monitor build interrupted", 130));
  const onTerm = () => abort.abort(new NativeBuildError("Native monitor build terminated", 143));
  const onOutputError = (error) => {
    outputFailed = true;
    abort.abort(new NativeBuildError(`Native monitor build output failed: ${error.message}`));
  };
  process.on("SIGINT", onInt);
  process.on("SIGTERM", onTerm);
  process.stderr.on("error", onOutputError);
  try {
    await buildNative({ signal: abort.signal });
  } catch (error) {
    process.exitCode = Number.isInteger(error.exitCode) ? error.exitCode : 1;
    if (!outputFailed && !process.stderr.destroyed) process.stderr.write(`Native monitor build failed: ${error.message}\n`);
  } finally {
    process.off("SIGINT", onInt);
    process.off("SIGTERM", onTerm);
    process.stderr.off("error", onOutputError);
  }
}

if (process.argv[1] && await fs.realpath(process.argv[1]).catch(() => "") === fileURLToPath(import.meta.url)) {
  await main();
}
