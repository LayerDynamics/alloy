#!/usr/bin/env node
/** Real Homebrew install/test in an owned, unlinked keg on the physical Mac runner. */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { renderFormula } from "./formula.mjs";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const owned = new Set();
let stopping = false;
function execute(command, args, { cwd = ROOT, env = process.env, log, allowed = [0], timeoutMs = 7_200_000, allowDuringShutdown = false } = {}) {
  if (stopping && !allowDuringShutdown) return Promise.reject(new Error("Homebrew acceptance cancelled"));
  return new Promise((resolve, reject) => {
    const file = log ? fs.openSync(log, "a") : null;
    const child = spawn(command, args, { cwd, env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "", error;
    let escalation;
    const signalGroup = (signal) => {
      if (!Number.isSafeInteger(child.pid) || child.pid <= 1) return;
      try { process.kill(-child.pid, signal); } catch (e) { if (e.code !== "ESRCH") error ??= e; }
    };
    const stop = (reason) => {
      error ??= new Error(reason);
      signalGroup("SIGTERM");
      // The smoke driver owns nested process groups and needs time to drain them.
      escalation ??= setTimeout(() => signalGroup("SIGKILL"), 20_000);
    };
    owned.add(stop);
    const timer = setTimeout(() => stop(`${command} timed out`), timeoutMs);
    child.once("error", (e) => { error = e; });
    const consume = (chunk, isError) => {
      if (file !== null) fs.writeSync(file, chunk);
      (isError ? process.stderr : process.stdout).write(chunk);
      if (isError) stderr = (stderr + chunk).slice(-256 * 1024); else stdout = (stdout + chunk).slice(-1024 * 1024);
    };
    child.stdout.on("data", (chunk) => consume(chunk, false));
    child.stderr.on("data", (chunk) => consume(chunk, true));
    child.once("close", (code, signal) => {
      // A command may exit before a descendant. Retire only its own process group.
      signalGroup("SIGKILL");
      clearTimeout(timer); clearTimeout(escalation); owned.delete(stop);
      if (file !== null) fs.closeSync(file);
      if (error) reject(error);
      else if (!allowed.includes(code)) reject(new Error(`${path.basename(command)} ${args.join(" ")} failed (${signal ?? code}): ${stderr.slice(-3000)}`));
      else resolve({ code, stdout, stderr });
    });
  });
}
function stopOwned() {
  stopping = true;
  for (const stop of owned) stop("Homebrew acceptance cancelled");
}

export function ciFormula(commit, name) {
  assert.match(name, /^alloy-ci-[0-9]+-[0-9]+$/);
  const className = name.split("-").map((part) => part[0].toUpperCase() + part.slice(1)).join("");
  return renderFormula({ commit }).replace("class Alloy < Formula", `class ${className} < Formula`).replace('  license "Apache-2.0"\n', '  license "Apache-2.0"\n  keg_only "isolated Alloy installation acceptance"\n');
}
export async function main() {
  assert(process.platform === "darwin" && process.arch === "arm64", "real Homebrew acceptance requires physical Apple Silicon with Metal");
  process.on("SIGINT", stopOwned); process.on("SIGTERM", stopOwned);
  const id = process.env.GITHUB_RUN_ID || String(Date.now());
  const attempt = process.env.GITHUB_RUN_ATTEMPT || "1";
  const name = `alloy-ci-${id}-${attempt}`;
  const tap = `layerdynamics/${name}`;
  const formula = `${tap}/${name}`;
  assert.match(name, /^alloy-ci-[0-9]+-[0-9]+$/);
  const commit = (await execute("git", ["rev-parse", "HEAD"])).stdout.trim();
  assert.match(commit, /^[0-9a-f]{40}$/);
  const base = path.resolve(process.env.ALLOY_ACCEPTANCE_OUTPUT_DIR || path.join(ROOT, "target/phase-1-acceptance"));
  fs.mkdirSync(base, { recursive: true });
  const output = fs.mkdtempSync(path.join(base, "homebrew-"));
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-brew-"));
  const brewTemp = path.join(temporary, "build");
  fs.mkdirSync(brewTemp);
  const localTap = path.join(temporary, "tap");
  fs.mkdirSync(path.join(localTap, "Formula"), { recursive: true });
  const rendered = ciFormula(commit, name);
  fs.writeFileSync(path.join(localTap, "Formula", `${name}.rb`), rendered);
  fs.writeFileSync(path.join(output, "tested-formula.rb"), rendered);
  const env = {
    ...process.env, HOMEBREW_NO_AUTO_UPDATE: "1", HOMEBREW_NO_INSTALL_CLEANUP: "1", HOMEBREW_NO_AUTOREMOVE: "1",
    HOMEBREW_NO_ANALYTICS: "1", HOMEBREW_TEMP: brewTemp, HOMEBREW_LOGS: path.join(output, "brew-logs"),
  };
  delete env.HOMEBREW_KEEP_TMP;
  const brew = (await execute("/usr/bin/which", ["brew"], { env })).stdout.trim();
  assert(path.isAbsolute(brew), "Homebrew is not installed on the configured Mac runner");
  const log = path.join(output, "brew.log");
  const result = { schema_version: 1, test: "homebrew_installed_acceptance", source_commit: commit, formula, status: "running", output };
  const save = () => fs.writeFileSync(path.join(output, "result.json"), `${JSON.stringify(result, null, 2)}\n`);
  save();
  let ownTap = false, installAttempted = false;
  try {
    result.brew_version = (await execute(brew, ["--version"], { env, log })).stdout.trim();
    const taps = (await execute(brew, ["tap"], { env, log })).stdout.split("\n").map((x) => x.trim().toLowerCase());
    assert(!taps.includes(tap), `refusing to reuse an existing tap ${tap}`);
    const installed = await execute(brew, ["list", "--formula", "--versions", name], { env, log, allowed: [0, 1] });
    assert.equal(installed.stdout.trim(), "", "refusing to replace an existing formula");
    await execute("git", ["init", "--initial-branch=main", localTap], { env, log });
    await execute("git", ["-C", localTap, "add", "Formula"], { env, log });
    await execute("git", ["-C", localTap, "-c", "user.name=Alloy CI", "-c", "user.email=ci@alloy.invalid", "commit", "-m", `Test installed source ${commit}`], { env, log });
    await execute(brew, ["tap", "--custom-remote", tap, localTap], { env, log });
    ownTap = true;
    if (stopping) throw new Error("Homebrew acceptance cancelled");
    installAttempted = true;
    await execute(brew, ["install", "--build-from-source", formula], { env, log });
    if (stopping) throw new Error("Homebrew acceptance cancelled");
    result.prefix = (await execute(brew, ["--prefix", formula], { env, log })).stdout.trim();
    assert(path.isAbsolute(result.prefix));
    // Everything built under our exclusive Homebrew temp path becomes inaccessible
    // at the embedded path before either installed native check or model execution.
    fs.renameSync(brewTemp, `${brewTemp}-unavailable`);
    fs.mkdirSync(brewTemp);
    result.original_build_tree_unavailable = true;
    save();
    await execute(brew, ["test", formula], { env, log, timeoutMs: 180_000 });
    const artifact = process.env.ALLOY_QWEN3_ARTIFACT || path.join(os.homedir(), "alloy-omlx-models/alloy-qwen3-4b-4bit");
    await execute(process.execPath, [path.join(ROOT, "scripts/homebrew/smoke.mjs"), "--prefix", result.prefix, "--expected-commit", commit, "--output", path.join(output, "installed"), "--artifact", artifact], { env, log, timeoutMs: 600_000 });
    if (stopping) throw new Error("Homebrew acceptance cancelled");
    result.status = "passed";
  } catch (error) {
    result.status = "failed"; result.error = error.message;
    throw error;
  } finally {
    // Only the unique test keg/tap belongs to this job. Never link it or autoremove dependencies.
    const cleanupErrors = [];
    if (installAttempted) {
      try {
        const found = await execute(brew, ["list", "--formula", "--versions", formula], { env, log, allowed: [0, 1], timeoutMs: 30_000, allowDuringShutdown: true });
        if (found.stdout.trim()) await execute(brew, ["uninstall", "--formula", formula], { env, log, timeoutMs: 120_000, allowDuringShutdown: true });
      } catch (error) { cleanupErrors.push(error.message); }
    }
    if (ownTap) {
      try { await execute(brew, ["untap", tap], { env, log, timeoutMs: 30_000, allowDuringShutdown: true }); }
      catch (error) { cleanupErrors.push(error.message); }
    }
    if (cleanupErrors.length === 0) fs.rmSync(temporary, { recursive: true, force: true });
    else { result.status = "failed"; result.cleanup_errors = cleanupErrors; }
    result.finished_at = new Date().toISOString(); save();
    process.stdout.write(`ALLOY_HOMEBREW_ACCEPTANCE ${JSON.stringify(result)}\n`);
    process.off("SIGINT", stopOwned); process.off("SIGTERM", stopOwned);
    if (cleanupErrors.length) throw new Error(`Homebrew acceptance cleanup failed: ${cleanupErrors.join("; ")}`);
  }
}
if (process.argv[1] && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
