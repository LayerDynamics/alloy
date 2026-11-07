#!/usr/bin/env node
/** Real Homebrew install/test in an owned, unlinked keg on the physical Mac runner. */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { renderFormula } from "./formula.mjs";
import { createSourceSnapshot } from "./source.mjs";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const owned = new Set();
const workAbort = new AbortController();
let stopping = false;
function execute(command, args, { cwd = ROOT, env = process.env, log, allowed = [0], timeoutMs = 7_200_000, killGraceMs = 3000, allowDuringShutdown = false } = {}) {
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
      // Stay inside Actions' roughly ten-second cancellation window. The smoke
      // driver receives an explicit six-second grace for its nested groups.
      // Cleanup can receive the runner's second signal late in that window.
      escalation ??= setTimeout(() => signalGroup("SIGKILL"), allowDuringShutdown ? Math.min(killGraceMs, 1000) : killGraceMs);
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
  workAbort.abort(new Error("Homebrew acceptance cancelled"));
  for (const stop of owned) stop("Homebrew acceptance cancelled");
}

const OWNER_MARKER = "alloy-ci-owner.json";
const fingerprint = (value) => createHash("sha256").update(value).digest("hex");
const json = (filename) => {
  assert(fs.lstatSync(filename).isFile(), `ownership evidence must be a regular file: ${filename}`);
  assert(fs.statSync(filename).size <= 65536, "ownership evidence exceeds its size bound");
  return JSON.parse(fs.readFileSync(filename, "utf8"));
};

export function ownershipContext(env, commit) {
  assert.match(env.GITHUB_RUN_ID ?? "", /^[1-9][0-9]*$/, "GITHUB_RUN_ID is required for owned Homebrew CI resources");
  assert.match(env.GITHUB_RUN_ATTEMPT ?? "", /^[1-9][0-9]*$/, "GITHUB_RUN_ATTEMPT is required for owned Homebrew CI resources");
  assert.match(commit, /^[0-9a-f]{40}$/, "cleanup requires the exact checked-out source commit");
  const base = path.resolve(env.ALLOY_ACCEPTANCE_OUTPUT_DIR || path.join(ROOT, "target/phase-1-acceptance"));
  return { run_id: env.GITHUB_RUN_ID, run_attempt: env.GITHUB_RUN_ATTEMPT, source_commit: commit,
    base: fs.existsSync(base) ? fs.realpathSync(base) : base, temporary_parent: fs.realpathSync(os.tmpdir()) };
}

export function ownershipFile(context) {
  return path.join(context.base, `homebrew-owner-${context.run_id}-${context.run_attempt}.json`);
}

function markerFor(owner) {
  const { schema_version, run_id, run_attempt, source_commit, token, name, tap, formula,
    output, temporary, local_tap, brew, formula_sha256 } = owner;
  return { schema_version, run_id, run_attempt, source_commit, token, name, tap, formula,
    output, temporary, local_tap, brew, formula_sha256 };
}

export function validateOwnership(owner, context) {
  assert.equal(owner.schema_version, 1, "unsupported Homebrew ownership schema");
  for (const key of ["run_id", "run_attempt", "source_commit"]) assert.equal(owner[key], context[key], `Homebrew ownership ${key} mismatch`);
  assert.match(owner.token ?? "", /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/, "invalid ownership token");
  const name = `alloy-ci-${context.run_id}-${context.run_attempt}`;
  assert.equal(owner.name, name);
  assert.equal(owner.tap, `layerdynamics/${name}`);
  assert.equal(owner.formula, `${owner.tap}/${name}`);
  assert.equal(owner.output, path.join(context.base, `homebrew-${context.run_id}-${context.run_attempt}-${owner.token}`));
  assert.equal(owner.temporary, path.join(context.temporary_parent, `alloy-brew-${context.run_id}-${context.run_attempt}-${owner.token}`));
  assert.equal(owner.local_tap, path.join(owner.temporary, "tap"));
  assert(path.isAbsolute(owner.brew), "owned Homebrew executable must be absolute");
  assert.match(owner.formula_sha256 ?? "", /^[0-9a-f]{64}$/);
  for (const key of ["preflight_verified", "tap_attempted", "install_attempted"]) assert.equal(typeof owner[key], "boolean", `invalid ${key}`);
  assert(!owner.tap_attempted || owner.preflight_verified, "tap mutation lacks an absence precheck");
  assert(!owner.install_attempted || owner.tap_attempted, "installation lacks owned tap intent");
  assert(["pending", "resources_removed", "complete"].includes(owner.cleanup_phase), "invalid cleanup phase");
  return owner;
}

export function createOwnership(context, brew, rendered, token = randomUUID()) {
  assert.match(token, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/, "invalid ownership token");
  const name = `alloy-ci-${context.run_id}-${context.run_attempt}`;
  const temporary = path.join(context.temporary_parent, `alloy-brew-${context.run_id}-${context.run_attempt}-${token}`);
  const owner = { schema_version: 1, run_id: context.run_id, run_attempt: context.run_attempt,
    source_commit: context.source_commit, token, name, tap: `layerdynamics/${name}`,
    formula: `layerdynamics/${name}/${name}`, output: path.join(context.base, `homebrew-${context.run_id}-${context.run_attempt}-${token}`),
    temporary, local_tap: path.join(temporary, "tap"), brew, formula_sha256: fingerprint(rendered),
    preflight_verified: false, tap_attempted: false, install_attempted: false, cleanup_phase: "pending" };
  validateOwnership(owner, context);
  const fd = fs.openSync(ownershipFile(context), "wx", 0o600);
  try { fs.writeFileSync(fd, `${JSON.stringify(owner, null, 2)}\n`); fs.fsyncSync(fd); }
  finally { fs.closeSync(fd); }
  return owner;
}

export function updateOwnership(context, owner, update) {
  assert(Object.keys(update).every((key) => ["preflight_verified", "tap_attempted", "install_attempted", "cleanup_phase"].includes(key)), "ownership identity is immutable");
  const filename = ownershipFile(context);
  assert.deepEqual(markerFor(validateOwnership(json(filename), context)), markerFor(owner), "ownership record changed before update");
  const next = validateOwnership({ ...owner, ...update }, context);
  const temporary = `${filename}.${owner.token}.${randomUUID()}.tmp`;
  const fd = fs.openSync(temporary, "wx", 0o600);
  try { fs.writeFileSync(fd, `${JSON.stringify(next, null, 2)}\n`); fs.fsyncSync(fd); }
  finally { fs.closeSync(fd); }
  fs.renameSync(temporary, filename);
  Object.assign(owner, next);
}

export function createTemporaryMarker(owner) {
  fs.mkdirSync(owner.temporary, { mode: 0o700 });
  fs.writeFileSync(path.join(owner.temporary, OWNER_MARKER), `${JSON.stringify(markerFor(owner), null, 2)}\n`, { flag: "wx", mode: 0o600 });
}

function verifyTemporaryRoot(owner) {
  assert(fs.lstatSync(owner.temporary).isDirectory(), "owned temporary directory was replaced");
  assert.equal(fs.realpathSync(owner.temporary), owner.temporary, "owned temporary directory must not be an alias");
}

function verifyTemporaryMarker(owner) {
  verifyTemporaryRoot(owner);
  assert.deepEqual(json(path.join(owner.temporary, OWNER_MARKER)), markerFor(owner), "temporary ownership marker mismatch");
}

function verifyTemporaryForCleanup(owner) {
  verifyTemporaryRoot(owner);
  if (fs.existsSync(path.join(owner.temporary, OWNER_MARKER))) verifyTemporaryMarker(owner);
  else {
    assert(owner.cleanup_phase === "resources_removed" || (!owner.tap_attempted && !owner.install_attempted), "temporary ownership marker is missing; refusing external cleanup");
    assert.equal(fs.readdirSync(owner.temporary).length, 0, "unmarked temporary directory contains files; refusing removal");
  }
}

function removeOwnedTemporary(owner) {
  verifyTemporaryForCleanup(owner);
  const marker = path.join(owner.temporary, OWNER_MARKER);
  if (fs.existsSync(marker)) {
    // Keep the proof through every potentially long recursive deletion. A hard
    // kill leaves a marked root that the separate cleanup step can resume.
    for (const name of fs.readdirSync(owner.temporary)) {
      if (name !== OWNER_MARKER) fs.rmSync(path.join(owner.temporary, name), { recursive: true, force: true });
    }
    verifyTemporaryMarker(owner);
    fs.unlinkSync(marker);
  }
  // The only unmarked deletion is an atomic empty-directory removal. This also
  // handles interruption between initial mkdir/marker or final unlink/rmdir.
  fs.rmdirSync(owner.temporary);
}

function brewEnvironment(owner, env = process.env) {
  const result = { ...env, HOMEBREW_NO_AUTO_UPDATE: "1", HOMEBREW_NO_INSTALL_CLEANUP: "1", HOMEBREW_NO_AUTOREMOVE: "1",
    HOMEBREW_NO_ANALYTICS: "1", HOMEBREW_TEMP: path.join(owner.temporary, "build"), HOMEBREW_LOGS: path.join(owner.output, "brew-logs") };
  delete result.HOMEBREW_KEEP_TMP;
  return result;
}

/**
 * Reconcile this run after normal exit or GitHub runner cancellation. Commands
 * stop their owned groups before the runner's forced owner deadline; this record
 * is resource ownership, not recovery from an isolated owner SIGKILL.
 */
export async function cleanupOwned(context, { run = execute, env = process.env } = {}) {
  const filename = ownershipFile(context);
  if (!fs.existsSync(filename)) return { status: "no_state", run_id: context.run_id, run_attempt: context.run_attempt };
  const owner = validateOwnership(json(filename), context);
  if (owner.cleanup_phase === "complete") return { status: "already_clean", formula: owner.formula, source_commit: owner.source_commit };
  if (fs.existsSync(owner.output)) {
    assert(fs.lstatSync(owner.output).isDirectory() && fs.realpathSync(owner.output) === owner.output, "ownership output directory was replaced");
  }
  const hasTemporary = fs.existsSync(owner.temporary);
  if (hasTemporary) verifyTemporaryForCleanup(owner);
  else assert(owner.cleanup_phase === "resources_removed" || (!owner.tap_attempted && !owner.install_attempted), "temporary ownership marker is missing; refusing external cleanup");
  const actions = [];
  if (owner.cleanup_phase === "pending" && (owner.tap_attempted || owner.install_attempted)) {
    const options = { env: brewEnvironment(owner, env), log: path.join(owner.output, "brew.log"), timeoutMs: 30000, allowDuringShutdown: true };
    const currentBrew = (await run("/usr/bin/which", ["brew"], options)).stdout.trim();
    assert.equal(fs.realpathSync(currentBrew), fs.realpathSync(owner.brew), "Homebrew executable changed since ownership was recorded");
    const taps = (await run(owner.brew, ["tap"], options)).stdout.split("\n").map((value) => value.trim().toLowerCase());
    const tapPresent = taps.includes(owner.tap);
    if (tapPresent) {
      const repository = (await run(owner.brew, ["--repository", owner.tap], options)).stdout.trim();
      const origin = (await run("git", ["-C", repository, "remote", "get-url", "origin"], options)).stdout.trim();
      const originPath = origin.startsWith("file://") ? fileURLToPath(origin) : origin;
      assert(path.isAbsolute(originPath) && fs.realpathSync(originPath) === fs.realpathSync(owner.local_tap), "tap origin is not the recorded owned repository");
      assert.equal(fingerprint(fs.readFileSync(path.join(repository, "Formula", `${owner.name}.rb`))), owner.formula_sha256, "tap formula differs from the recorded candidate");
    }
    if (owner.install_attempted) {
      const found = await run(owner.brew, ["list", "--formula", "--versions", owner.name], { ...options, allowed: [0, 1] });
      if (found.stdout.trim()) {
        assert.equal(found.stdout.trim().split(/\s+/)[0], owner.name, "Homebrew reported a different installed formula");
        // A missing tap cannot prove ownership of an unfinished keg. A complete
        // install can instead prove its exact source through its own manifest.
        if (!tapPresent) {
          const prefix = (await run(owner.brew, ["--prefix", owner.name], options)).stdout.trim();
          assert.equal(json(path.join(prefix, "libexec/install.json")).source_commit, owner.source_commit, "installed keg lacks matching source ownership");
        }
        await run(owner.brew, ["uninstall", "--formula", tapPresent ? owner.formula : owner.name], { ...options, timeoutMs: 90000 });
        actions.push({ action: "uninstall", formula: owner.formula });
      }
    }
    if (tapPresent) {
      await run(owner.brew, ["untap", owner.tap], options);
      actions.push({ action: "untap", tap: owner.tap });
    }
  }
  // Persist completion of external cleanup before removing its proof directory.
  // A hard kill between removal and the final write remains safe to resume.
  updateOwnership(context, owner, { cleanup_phase: "resources_removed" });
  if (fs.existsSync(owner.temporary)) {
    removeOwnedTemporary(owner);
    actions.push({ action: "remove_temporary", path: owner.temporary });
  }
  updateOwnership(context, owner, { cleanup_phase: "complete" });
  const result = { status: "cleaned", formula: owner.formula, source_commit: owner.source_commit, actions };
  if (fs.existsSync(owner.output)) fs.writeFileSync(path.join(owner.output, "cleanup.json"), `${JSON.stringify(result, null, 2)}\n`);
  return result;
}

export function ciFormula(commit, name, sourceUrl) {
  assert.match(name, /^alloy-ci-[0-9]+-[0-9]+$/);
  const className = name.split("-").map((part) => part[0].toUpperCase() + part.slice(1)).join("");
  let formula = renderFormula({ commit }).replace("class Alloy < Formula", `class ${className} < Formula`).replace('  license "Apache-2.0"\n', '  license "Apache-2.0"\n  keg_only "isolated Alloy installation acceptance"\n');
  if (sourceUrl !== undefined) {
    const url = new URL(sourceUrl);
    assert.equal(url.protocol, "file:", "CI source must be its owned local Git snapshot");
    assert.equal(url.hostname, "");
    assert(!/["\r\n\\]/.test(sourceUrl), "unsafe CI source URL");
    formula = formula.replace('  url "https://github.com/LayerDynamics/alloy.git",', `  url "${sourceUrl}", using: :git,`);
  }
  return formula;
}
export async function main() {
  assert(process.platform === "darwin" && process.arch === "arm64", "real Homebrew acceptance requires physical Apple Silicon with Metal");
  process.on("SIGINT", stopOwned); process.on("SIGTERM", stopOwned);
  const commit = (await execute("git", ["rev-parse", "HEAD"], { timeoutMs: 15000 })).stdout.trim();
  let context = ownershipContext(process.env, commit);
  fs.mkdirSync(context.base, { recursive: true });
  context = ownershipContext(process.env, commit);
  const name = `alloy-ci-${context.run_id}-${context.run_attempt}`;
  const token = randomUUID();
  const sourceDirectory = path.join(context.temporary_parent, `alloy-brew-${context.run_id}-${context.run_attempt}-${token}`, "source.git");
  const rendered = ciFormula(commit, name, pathToFileURL(sourceDirectory).href);
  const brew = (await execute("/usr/bin/which", ["brew"], { timeoutMs: 15000 })).stdout.trim();
  assert(path.isAbsolute(brew), "Homebrew is not installed on the configured Mac runner");
  // This exclusive, synced record precedes even creation of the owned temp root.
  const owner = createOwnership(context, brew, rendered, token);
  const { tap, formula, output, temporary, local_tap: localTap } = owner;
  const brewTemp = path.join(temporary, "build");
  const env = brewEnvironment(owner);
  const log = path.join(output, "brew.log");
  const result = { schema_version: 1, test: "homebrew_installed_acceptance", source_commit: commit, formula, status: "running", output,
    ownership_record: ownershipFile(context), run_id: context.run_id, run_attempt: context.run_attempt };
  const save = () => fs.writeFileSync(path.join(output, "result.json"), `${JSON.stringify(result, null, 2)}\n`);
  try {
    fs.mkdirSync(output);
    createTemporaryMarker(owner);
    fs.mkdirSync(brewTemp);
    fs.mkdirSync(path.join(localTap, "Formula"), { recursive: true });
    fs.writeFileSync(path.join(localTap, "Formula", `${name}.rb`), rendered);
    fs.writeFileSync(path.join(output, "tested-formula.rb"), rendered);
    save();
    // PR merge commits are not ordinary remote branches. Keep their exact tree
    // in an independent Git source that Homebrew can fetch by the same SHA.
    result.source_snapshot = await createSourceSnapshot({ sourceRoot: ROOT, destination: sourceDirectory,
      expectedCommit: commit, signal: workAbort.signal });
    save();
    result.brew_version = (await execute(brew, ["--version"], { env, log })).stdout.trim();
    result.host = {
      os: (await execute("/usr/bin/sw_vers", [], { env, log })).stdout.trim(),
      architecture: process.arch,
      xcode: (await execute("/usr/bin/xcodebuild", ["-version"], { env, log })).stdout.trim(),
      sdk_version: (await execute("/usr/bin/xcrun", ["--show-sdk-version"], { env, log })).stdout.trim(),
    };
    const taps = (await execute(brew, ["tap"], { env, log })).stdout.split("\n").map((x) => x.trim().toLowerCase());
    assert(!taps.includes(tap), `refusing to reuse an existing tap ${tap}`);
    const installed = await execute(brew, ["list", "--formula", "--versions", name], { env, log, allowed: [0, 1] });
    assert.equal(installed.stdout.trim(), "", "refusing to replace an existing formula");
    updateOwnership(context, owner, { preflight_verified: true });
    await execute("git", ["init", "--initial-branch=main", localTap], { env, log });
    await execute("git", ["-C", localTap, "add", "Formula"], { env, log });
    await execute("git", ["-C", localTap, "-c", "user.name=Alloy CI", "-c", "user.email=ci@alloy.invalid", "commit", "-m", `Test installed source ${commit}`], { env, log });
    updateOwnership(context, owner, { tap_attempted: true });
    await execute(brew, ["tap", "--custom-remote", tap, localTap], { env, log });
    if (stopping) throw new Error("Homebrew acceptance cancelled");
    updateOwnership(context, owner, { install_attempted: true });
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
    // Homebrew creates and removes its own test directory here and grants its
    // sandbox Unix-socket access there. Our deeper build root exceeds macOS's
    // socket path limit once Homebrew appends the test name and config path.
    await execute(brew, ["test", formula], {
      env: { ...env, HOMEBREW_TEMP: fs.realpathSync("/tmp") }, log, timeoutMs: 180_000,
    });
    const artifact = process.env.ALLOY_QWEN3_ARTIFACT || path.join(os.homedir(), "alloy-omlx-models/alloy-qwen3-4b-4bit");
    await execute(process.execPath, [path.join(ROOT, "scripts/homebrew/smoke.mjs"), "--prefix", result.prefix, "--expected-commit", commit, "--output", path.join(output, "installed"), "--artifact", artifact], { env, log, timeoutMs: 600_000, killGraceMs: 6000 });
    if (stopping) throw new Error("Homebrew acceptance cancelled");
    result.status = "passed";
  } catch (error) {
    result.status = "failed"; result.error = error.message;
    throw error;
  } finally {
    const cleanupErrors = [];
    try { if (fs.existsSync(output)) save(); result.cleanup = await cleanupOwned(context); }
    catch (error) { cleanupErrors.push(error.message); }
    if (cleanupErrors.length) { result.status = "failed"; result.cleanup_errors = cleanupErrors; }
    result.finished_at = new Date().toISOString(); save();
    process.stdout.write(`ALLOY_HOMEBREW_ACCEPTANCE ${JSON.stringify(result)}\n`);
    process.off("SIGINT", stopOwned); process.off("SIGTERM", stopOwned);
    if (cleanupErrors.length) throw new Error(`Homebrew acceptance cleanup failed: ${cleanupErrors.join("; ")}`);
  }
}

export async function cleanupCurrentRun() {
  process.on("SIGINT", stopOwned); process.on("SIGTERM", stopOwned);
  try {
    const commit = (await execute("git", ["rev-parse", "HEAD"], { timeoutMs: 15000, allowDuringShutdown: true })).stdout.trim();
    const context = ownershipContext(process.env, commit);
    const result = await cleanupOwned(context);
    if (fs.existsSync(ownershipFile(context))) {
      const owner = validateOwnership(json(ownershipFile(context)), context);
      const evidence = path.join(owner.output, "result.json");
      if (fs.existsSync(evidence)) {
        const previous = json(evidence);
        assert.equal(previous.source_commit, context.source_commit);
        if (previous.status === "running") {
          previous.status = "interrupted";
          previous.error = "The owning CI step ended before acceptance finalized; the separate cleanup step recovered its recorded resources.";
        }
        previous.cleanup = result;
        previous.finished_at ??= new Date().toISOString();
        fs.writeFileSync(evidence, `${JSON.stringify(previous, null, 2)}\n`);
      }
    }
    process.stdout.write(`ALLOY_HOMEBREW_CLEANUP ${JSON.stringify(result)}\n`);
    return result;
  } finally {
    process.off("SIGINT", stopOwned); process.off("SIGTERM", stopOwned);
  }
}
if (process.argv[1] && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const operation = args.length === 0 ? main : args.length === 1 && args[0] === "--cleanup-owned" ? cleanupCurrentRun : null;
  if (!operation) { console.error("Usage: node scripts/homebrew/ci.mjs [--cleanup-owned]"); process.exitCode = 1; }
  else operation().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
