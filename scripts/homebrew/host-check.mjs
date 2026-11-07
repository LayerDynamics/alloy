#!/usr/bin/env node
/** Read-only evidence for Homebrew locks on the persistent acceptance runner. */
import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
const exec = promisify(execFile);
const output = path.join(process.env.ALLOY_ACCEPTANCE_OUTPUT_DIR || "target/phase-1-acceptance", "homebrew-host");
fs.mkdirSync(output, { recursive: true });
const run = async (command, args) => {
  try { return await exec(command, args, { encoding: "utf8", timeout: 10000, maxBuffer: 4 * 1024 * 1024 }); }
  catch (error) { return { stdout: error.stdout || "", stderr: error.stderr || error.message, code: error.code }; }
};
const result = { schema_version: 1, read_only: true, run_id: process.env.GITHUB_RUN_ID,
  run_attempt: process.env.GITHUB_RUN_ATTEMPT, observed_at: new Date().toISOString() };
const brew = await run("brew", ["--prefix"]);
if (path.isAbsolute(brew.stdout.trim())) {
  result.homebrew_prefix = brew.stdout.trim();
  result.lock_path = path.join(result.homebrew_prefix, "var/homebrew/locks/cmake.formula.lock");
  const lock = await run("/usr/sbin/lsof", ["-nP", result.lock_path]);
  result.lock = { stdout: lock.stdout.slice(0, 12000), stderr: lock.stderr.slice(0, 2000), code: lock.code ?? 0 };
  const table = await run("/bin/ps", ["-axo", "pid=,ppid=,pgid=,lstart=,command="]);
  // Retain only this project's temporary installation commands, not unrelated
  // application arguments or the user's general process list.
  result.alloy_ci_processes = table.stdout.split("\n").filter((line) =>
    /layerdynamics\/alloy-ci-[0-9]+-[0-9]+|alloy-brew-[A-Za-z0-9-]+\//i.test(line)
  ).slice(0, 100).map((line) => line.slice(0, 2500));
  result.process_probe_error = table.code ? table.stderr.slice(0, 2000) : null;
} else result.error = brew.stderr.slice(0, 2000);
fs.writeFileSync(path.join(output, "diagnostic.json"), `${JSON.stringify(result, null, 2)}\n`);
console.log(`Homebrew host evidence: ${path.join(output, "diagnostic.json")}`);
