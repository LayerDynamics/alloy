#!/usr/bin/env node
/** One-time recovery of the artifact-identified installer from run 42313536852. */
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const exec = promisify(execFile);
const PID = 33418;
const NAME = "alloy-ci-42313536852-1";
const TAP = `layerdynamics/${NAME}`;
const FORMULA = `${TAP}/${NAME}`;
const SOURCE = "d03f18a5734b28d21853e9ef7172b71731ac367f";
const HASH = "ea880c2fc3d5e86606e868b5f58dd4c8e64aedc45c79f41075371a87fa38bf3c";
const START = "Fri Nov 6 23:46:45 2026";
const COMMAND = `/opt/homebrew/Library/Homebrew/vendor/portable-ruby/current/bin/ruby -W1 --disable=gems,rubyopt /opt/homebrew/Library/Homebrew/brew.rb install --build-from-source ${FORMULA}`;
const TAP_PATH = `/opt/homebrew/Library/Taps/layerdynamics/homebrew-${NAME}`;
const CELLAR = `/opt/homebrew/Cellar/${NAME}`;
const LOCK = "/opt/homebrew/var/homebrew/locks/cmake.formula.lock";
const BREW = "/opt/homebrew/bin/brew";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const environment = () => ({ ...process.env, TZ: "UTC", LC_ALL: "C", LANG: "C",
  HOMEBREW_NO_AUTO_UPDATE: "1", HOMEBREW_NO_AUTOREMOVE: "1", HOMEBREW_NO_INSTALL_CLEANUP: "1", HOMEBREW_NO_ANALYTICS: "1" });

async function run(command, args, { allowed = [0], timeout = 5000 } = {}) {
  let result;
  try { result = { ...(await exec(command, args, { env: environment(), encoding: "utf8", timeout, killSignal: "SIGKILL", maxBuffer: 8 * 1024 * 1024 })), code: 0 }; }
  catch (error) {
    if (!Number.isInteger(error.code) || !allowed.includes(error.code)) throw error;
    result = { stdout: error.stdout ?? "", stderr: error.stderr ?? "", code: error.code };
  }
  assert(allowed.includes(result.code), `${command} returned ${result.code}`);
  return result;
}

export function parseProcessTable(stdout) {
  return stdout.split("\n").filter((line) => line.trim()).map((line) => {
    const m = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s+(\S+)\s+(.*)$/.exec(line);
    assert(m, "Cannot parse complete process identity; refusing recovery");
    return { pid: Number(m[1]), ppid: Number(m[2]), pgid: Number(m[3]), uid: Number(m[4]),
      start: m[5].replace(/\s+/g, " "), state: m[6], command: m[7].trim() };
  });
}

const processes = async () => parseProcessTable((await run("/bin/ps", ["-ww", "-axo", "pid=,ppid=,pgid=,uid=,lstart=,stat=,command="])).stdout);

export function validateLeader(leader, uid) {
  assert(leader, "Recorded installer is absent");
  assert.deepEqual({ pid: leader.pid, ppid: leader.ppid, pgid: leader.pgid, uid: leader.uid, start: leader.start, command: leader.command },
    { pid: PID, ppid: 1, pgid: PID, uid, start: START, command: COMMAND }, "Recorded installer identity changed; never adopt a reused PID");
}

export function validateInitialGroup(table, uid) {
  const leader = table.find((row) => row.pid === PID);
  validateLeader(leader, uid);
  assert(!leader.state.startsWith("Z"), "Recorded installer is no longer live");
  const byPid = new Map(table.map((row) => [row.pid, row]));
  const group = table.filter((row) => row.pgid === PID);
  assert(group.length > 0);
  for (const member of group) {
    assert.equal(member.uid, uid, "Process group contains a different user");
    assert(Date.parse(`${member.start} UTC`) >= Date.parse(`${START} UTC`), "Group member predates the recorded installer");
    let ancestor = member;
    const seen = new Set();
    while (ancestor.pid !== PID) {
      assert(!seen.has(ancestor.pid), "Process ancestry contains a cycle");
      seen.add(ancestor.pid);
      ancestor = byPid.get(ancestor.ppid);
      assert(ancestor && ancestor.pgid === PID && ancestor.uid === uid, "Cannot prove that every group member descends from the recorded installer");
    }
  }
  return new Map(group.map((row) => [row.pid, row]));
}

export function validateSurvivingGroup(table, original, uid) {
  const leader = table.find((row) => row.pid === PID);
  if (leader) validateLeader(leader, uid);
  const group = table.filter((row) => row.pgid === PID);
  for (const row of group) {
    const previous = original.get(row.pid);
    assert(previous && row.uid === uid && row.uid === previous.uid && row.start === previous.start,
      "Process group contains an unrecorded or changed identity; refusing further signals");
  }
  for (const previous of original.values()) {
    const current = table.find((row) => row.pid === previous.pid);
    if (current && current.uid === previous.uid && current.start === previous.start) {
      assert.equal(current.pgid, PID, "A recorded process left its group; refusing cleanup");
    }
  }
  return group;
}

async function lockHolders() {
  if (!fs.existsSync(LOCK)) return { exists: false, holders: [] };
  const result = await run("/usr/sbin/lsof", ["-nP", "-F", "pguftn", LOCK], { allowed: [0, 1] });
  assert.equal(result.stderr.trim(), "", "Lock-holder probe was incomplete");
  const holders = [];
  let holder;
  for (const line of result.stdout.split("\n")) {
    if (line.startsWith("p")) { holder = { pid: Number(line.slice(1)), names: [] }; holders.push(holder); }
    else if (holder && line.startsWith("u")) holder.uid = Number(line.slice(1));
    else if (holder && line.startsWith("g")) holder.pgid = Number(line.slice(1));
    else if (holder && line.startsWith("n")) holder.names.push(line.slice(1));
  }
  return { exists: true, holders, raw: result.stdout };
}

async function verifyTap() {
  if (!fs.existsSync(TAP_PATH)) return { exists: false };
  assert(fs.lstatSync(TAP_PATH).isDirectory() && fs.realpathSync(TAP_PATH) === TAP_PATH, "Recorded tap path was replaced");
  const expected = path.join(fs.realpathSync(os.tmpdir()), "alloy-brew-OwVXSa", "tap");
  const origin = (await run("/usr/bin/git", ["-C", TAP_PATH, "remote", "get-url", "origin"])).stdout.trim();
  const originPath = origin.startsWith("file://") ? fileURLToPath(origin) : origin;
  assert(path.isAbsolute(originPath), "Tap origin must be the absolute recorded local repository");
  assert.equal(fs.realpathSync(originPath), expected, "Tap origin path was replaced");
  const filename = path.join(TAP_PATH, "Formula", `${NAME}.rb`);
  assert(fs.lstatSync(filename).isFile() && fs.statSync(filename).size < 1024 * 1024, "Invalid recorded formula file");
  const sha256 = createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
  assert.equal(sha256, HASH, "Tap formula differs from the original digest-verified Actions artifact");
  return { exists: true, path: TAP_PATH, origin, formula_sha256: sha256, source_commit: SOURCE };
}

function verifiedKeg() {
  if (!fs.existsSync(CELLAR)) return null;
  assert(fs.lstatSync(CELLAR).isDirectory() && fs.realpathSync(CELLAR) === CELLAR, "Recorded Cellar path was replaced");
  const versions = fs.readdirSync(CELLAR);
  if (versions.length === 0) return null;
  assert.equal(versions.length, 1, "Refusing an ambiguous installed keg");
  const prefix = path.join(CELLAR, versions[0]);
  assert(fs.lstatSync(prefix).isDirectory() && fs.realpathSync(prefix) === prefix, "Installed keg is not an owned directory");
  const filename = path.join(prefix, "libexec/install.json");
  assert(fs.lstatSync(filename).isFile() && fs.statSync(filename).size < 1024 * 1024, "Installed keg lacks bounded source provenance");
  const manifest = JSON.parse(fs.readFileSync(filename, "utf8"));
  assert.equal(manifest.schema_version, 1);
  assert.equal(manifest.source_commit, SOURCE, "Refusing to uninstall a keg from a different source");
  return { prefix, source_commit: manifest.source_commit };
}

export async function main() {
  assert.equal(process.argv.length, 2, "This fixed one-time recovery accepts no arguments");
  assert.equal(process.platform, "darwin", "Recovery is restricted to the recorded macOS runner");
  assert.equal(process.arch, "arm64");
  const uid = process.getuid();
  assert(uid > 0 && process.geteuid() === uid, "Recovery must run unprivileged under its own UID");
  assert.equal(process.env.GITHUB_ACTIONS, "true");
  assert.equal(process.env.GITHUB_REPOSITORY, "LayerDynamics/alloy");
  assert.equal(process.env.GITHUB_EVENT_NAME, "push");
  assert.equal(process.env.GITHUB_REF, "refs/heads/ci/recover-homebrew-42313536852");
  for (const name of ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"]) assert.match(process.env[name] ?? "", /^[1-9][0-9]*$/);
  assert(path.isAbsolute(process.env.GITHUB_WORKSPACE ?? ""));
  assert.equal((await run("/usr/bin/id", ["-un"])).stdout.trim(), "ryanoboyle", "Runner user differs from the diagnostic artifact");
  const workspace = fs.realpathSync(process.env.GITHUB_WORKSPACE);
  const output = path.join(workspace, "target/homebrew-recovery", `${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.mkdirSync(output, { mode: 0o700 });
  const save = (name, value) => {
    const fd = fs.openSync(path.join(output, name), "w", 0o600);
    try { fs.writeFileSync(fd, `${JSON.stringify(value, null, 2)}\n`); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
  };
  const actions = [];
  const record = (action) => { actions.push({ at: new Date().toISOString(), ...action }); save("actions.json", actions); };
  const result = { schema_version: 1, recovery_for_run: "42313536852", source_commit: SOURCE, pid: PID, uid,
    run_id: process.env.GITHUB_RUN_ID, run_attempt: process.env.GITHUB_RUN_ATTEMPT, started_at: new Date().toISOString(), status: "running", output };
  save("result.json", result); save("actions.json", actions);
  save("before.json", { at: new Date().toISOString(), status: "probe_pending" });
  let original = new Map();
  try {
    const table = await processes();
    const leader = table.find((row) => row.pid === PID);
    const group = table.filter((row) => row.pgid === PID);
    const tap = await verifyTap();
    const lock = await lockHolders();
    save("before.json", { at: new Date().toISOString(), leader: leader ?? null, group, tap, lock, cellar_exists: fs.existsSync(CELLAR) });
    if (leader) {
      validateInitialGroup(table, uid);
      assert(tap.exists, "Cannot signal the installer without its verified original tap");
      assert(lock.holders.some((holder) => holder.pid === PID && holder.pgid === PID && holder.uid === uid && holder.names.includes(LOCK)),
        "Recorded installer is not the confirmed current holder of the recorded lock");
      // Fresh complete ancestry immediately before the first signal. After this
      // point only these recorded identities may receive further group signals.
      original = validateInitialGroup(await processes(), uid);
      save("pre-term-group.json", [...original.values()]);
      record({ action: "signal_intent", signal: "SIGTERM", process_group: PID, members: [...original.values()] });
      try { process.kill(-PID, "SIGTERM"); } catch (error) { if (error.code !== "ESRCH") throw error; }
      const termDeadline = Date.now() + 2000;
      let remaining = validateSurvivingGroup(await processes(), original, uid);
      while (remaining.length && Date.now() < termDeadline) { await sleep(100); remaining = validateSurvivingGroup(await processes(), original, uid); }
      if (remaining.length) {
        remaining = validateSurvivingGroup(await processes(), original, uid);
        if (remaining.length) {
          record({ action: "signal_intent", signal: "SIGKILL", process_group: PID, members: remaining });
          try { process.kill(-PID, "SIGKILL"); } catch (error) { if (error.code !== "ESRCH") throw error; }
        }
      }
      const killDeadline = Date.now() + 3000;
      remaining = validateSurvivingGroup(await processes(), original, uid);
      while (remaining.length && Date.now() < killDeadline) { await sleep(100); remaining = validateSurvivingGroup(await processes(), original, uid); }
      assert.equal(remaining.length, 0, "Recorded group still exists; refusing Homebrew cleanup");
    } else assert.equal(group.length, 0, "Leader is absent with unproven surviving group members; refusing recovery");

    const empty = await processes();
    assert(!empty.some((row) => row.pid === PID || row.pgid === PID), "Recorded process/group reappeared; refusing cleanup");
    const released = await lockHolders();
    assert(!released.holders.some((holder) => holder.pid === PID || original.has(holder.pid)), "Recorded process still holds the lock");
    save("released.json", { at: new Date().toISOString(), group: [], lock: released });
    const currentTap = await verifyTap();
    const keg = verifiedKeg();
    if (keg) {
      assert(currentTap.exists, "Cannot uninstall a keg without its verified original tap");
      record({ action: "uninstall_intent", formula: FORMULA, keg });
      save("uninstall.json", await run(BREW, ["uninstall", "--formula", FORMULA], { timeout: 30000 }));
      assert.equal(verifiedKeg(), null, "Recorded keg remains installed");
    }
    if (currentTap.exists) {
      await verifyTap();
      assert.equal(verifiedKeg(), null, "Refusing to untap while its keg exists");
      record({ action: "untap_intent", tap: TAP });
      save("untap.json", await run(BREW, ["untap", TAP], { timeout: 30000 }));
      assert(!fs.existsSync(TAP_PATH), "Recorded tap remains present");
    }
    result.status = actions.length === 0 ? "already_absent" : "recovered";
  } catch (error) { result.status = "refused_or_failed"; result.error = error.stack ?? String(error); }
  finally {
    try {
      const table = await processes();
      save("after.json", { at: new Date().toISOString(), group: table.filter((row) => row.pgid === PID),
        leader: table.find((row) => row.pid === PID) ?? null, lock: await lockHolders(), tap_exists: fs.existsSync(TAP_PATH), cellar_exists: fs.existsSync(CELLAR) });
    } catch (error) {
      result.after_probe_error = error.message; result.status = "refused_or_failed";
      save("after.json", { at: new Date().toISOString(), error: error.message });
    }
    result.finished_at = new Date().toISOString();
    save("result.json", result);
    console.log(`ALLOY_HOMEBREW_RECOVERY ${JSON.stringify(result)}`);
  }
  if (result.status === "refused_or_failed") throw new Error(result.error ?? result.after_probe_error);
  return result;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}

