/** An independent Git source for the exact checked-out CI candidate, including merge refs. */
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

const MAX_OUTPUT = 4 * 1024 * 1024;
const KILL_GRACE_MS = 250;
const inside = (parent, child) => {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith(".." + path.sep) && relative !== ".." && !path.isAbsolute(relative));
};

function gitEnvironment(input) {
  const env = { ...input };
  // Repository selectors, alternates, hooks and user config must not redirect
  // these read-only source probes or the private clone's object storage.
  for (const key of Object.keys(env)) if (key.startsWith("GIT_")) delete env[key];
  return { ...env, GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: os.devNull,
    GIT_OPTIONAL_LOCKS: "0", GIT_TERMINAL_PROMPT: "0", GIT_NO_LAZY_FETCH: "1" };
}

function gitRunner({ env, signal, deadline }) {
  return (args) => new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new Error("Git source snapshot cancelled")); return; }
    const remaining = deadline - performance.now();
    if (remaining <= 0) { reject(new Error("Git source snapshot timed out")); return; }
    const child = spawn("git", args, { env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "", bytes = 0, failure, escalation;
    const signalGroup = (name) => {
      if (!Number.isSafeInteger(child.pid) || child.pid <= 1) return;
      try { process.kill(-child.pid, name); }
      catch (error) { if (error.code !== "ESRCH") failure ??= error; }
    };
    const stop = (error) => {
      failure ??= error;
      signalGroup("SIGTERM");
      escalation ??= setTimeout(() => signalGroup("SIGKILL"), KILL_GRACE_MS);
    };
    const aborted = () => stop(new Error("Git source snapshot cancelled"));
    const timer = setTimeout(() => stop(new Error("Git source snapshot timed out")), Math.ceil(remaining));
    signal?.addEventListener("abort", aborted, { once: true });
    // Cover cancellation between the initial check and listener registration.
    if (signal?.aborted) aborted();
    const consume = (chunk, isError) => {
      bytes += Buffer.byteLength(chunk);
      if (bytes > MAX_OUTPUT) { stop(new Error("Git source snapshot output exceeded its bound")); return; }
      if (isError) stderr += chunk;
      else stdout += chunk;
    };
    child.stdout.setEncoding("utf8").on("data", (chunk) => consume(chunk, false));
    child.stderr.setEncoding("utf8").on("data", (chunk) => consume(chunk, true));
    child.once("error", (error) => { failure ??= error; });
    child.once("close", (code, childSignal) => {
      // Retire only this invocation's group, including a surviving pack child.
      signalGroup("SIGKILL");
      clearTimeout(timer);
      clearTimeout(escalation);
      signal?.removeEventListener("abort", aborted);
      if (failure) reject(failure);
      else if (code !== 0) reject(new Error("git " + args[0] + " failed (" + (childSignal ?? code) + "): " + stderr.slice(-8192)));
      else resolve(stdout);
    });
  });
}

function verifyPrivateStorage(directory) {
  const pending = [directory];
  let objectFiles = 0;
  while (pending.length) {
    const parent = pending.pop();
    for (const name of fs.readdirSync(parent)) {
      const filename = path.join(parent, name);
      const stat = fs.lstatSync(filename);
      assert(!stat.isSymbolicLink(), "source snapshot cannot depend on filesystem symlinks");
      if (stat.isDirectory()) pending.push(filename);
      else {
        assert(stat.isFile(), "source snapshot contains a non-regular file");
        assert.equal(stat.nlink, 1, "source snapshot cannot share hard-linked files");
        const relative = path.relative(directory, filename).split(path.sep).join("/");
        assert(!["objects/info/alternates", "objects/info/http-alternates"].includes(relative),
          "source snapshot cannot borrow another object database");
        assert(!relative.endsWith(".promisor"), "partial Git objects cannot satisfy complete source acceptance");
        if (relative.startsWith("objects/")) objectFiles++;
      }
    }
  }
  return objectFiles;
}

/**
 * Create an owned, independently cloneable bare repository for a clean HEAD.
 * Only the new destination is mutated; callers own its eventual cleanup.
 */
export async function createSourceSnapshot({
  sourceRoot, destination, expectedCommit, timeoutMs = 120_000, signal, env = process.env,
}) {
  assert.match(expectedCommit ?? "", /^[0-9a-f]{40}$/, "expectedCommit must be a full lowercase Git SHA");
  assert(Number.isSafeInteger(timeoutMs) && timeoutMs > 0 && timeoutMs <= 600_000,
    "snapshot timeout must be between 1 and 600000 milliseconds");
  for (const [name, value] of [["sourceRoot", sourceRoot], ["destination", destination]]) {
    assert(typeof value === "string" && path.isAbsolute(value) && !/[\u0000-\u001f\u007f]/.test(value),
      name + " must be an absolute filesystem path without control characters");
  }
  const source = fs.realpathSync(sourceRoot);
  assert(fs.statSync(source).isDirectory(), "sourceRoot must be a directory");
  const directory = path.join(fs.realpathSync(path.dirname(destination)), path.basename(destination));
  assert(!inside(source, directory), "snapshot destination must be outside the source checkout");
  try {
    fs.lstatSync(directory);
    assert.fail("snapshot destination already exists");
  } catch (error) { if (error.code !== "ENOENT") throw error; }
  const git = gitRunner({ env: gitEnvironment(env), signal, deadline: performance.now() + timeoutMs });
  const sourceGit = (args) => git(["-C", source, ...args]);
  assert.equal(fs.realpathSync((await sourceGit(["rev-parse", "--show-toplevel"])).trim()), source,
    "sourceRoot must be the checkout root");
  const commit = (await sourceGit(["rev-parse", "--verify", "HEAD^{commit}"])).trim();
  assert.equal(commit, expectedCommit, "source HEAD differs from the expected candidate");
  const tree = (await sourceGit(["rev-parse", "--verify", "HEAD^{tree}"])).trim();
  assert.match(tree, /^[0-9a-f]{40}$/, "invalid source tree identity");
  assert.equal((await sourceGit(["status", "--porcelain=v1", "--untracked-files=normal"])).trim(), "",
    "source snapshot requires a clean committed checkout");
  const shallow = (await sourceGit(["rev-parse", "--is-shallow-repository"])).trim() === "true";
  // mkdir is the ownership boundary, including for an existing dangling symlink.
  fs.mkdirSync(directory, { mode: 0o700 });
  const owned = fs.lstatSync(directory);
  try {
    await git(["clone", "--bare", "--no-hardlinks", "--dissociate", "--template=", "--", source, directory]);
    const snapshotGit = (args) => git(["--git-dir", directory, ...args]);
    assert.equal((await snapshotGit(["rev-parse", "--verify", "HEAD^{commit}"])).trim(), commit,
      "cloned source HEAD changed");
    await snapshotGit(["update-ref", "refs/heads/candidate", commit]);
    await snapshotGit(["symbolic-ref", "HEAD", "refs/heads/candidate"]);
    // Fetching this snapshot must never consult the original checkout.
    for (const remote of (await snapshotGit(["remote"])).trim().split("\n").filter(Boolean)) {
      await snapshotGit(["remote", "remove", remote]);
    }
    const objectFiles = verifyPrivateStorage(directory);
    await snapshotGit(["fsck", "--full", "--strict", "--no-reflogs", "--no-dangling"]);
    const entries = (await snapshotGit(["ls-tree", "-r", "-z", "--full-tree", commit])).split("\0").filter(Boolean);
    assert(entries.every((entry) => /^(100644|100755|120000) blob [0-9a-f]{40}\t/.test(entry)),
      "source snapshot must contain all tracked files, without unresolved submodules");
    assert.equal((await snapshotGit(["rev-parse", "--verify", "HEAD^{tree}"])).trim(), tree,
      "cloned source tree changed");
    assert.equal((await snapshotGit(["symbolic-ref", "HEAD"])).trim(), "refs/heads/candidate");
    assert.equal((await snapshotGit(["rev-parse", "--is-shallow-repository"])).trim() === "true", shallow);
    assert.equal((await sourceGit(["rev-parse", "--verify", "HEAD^{commit}"])).trim(), commit,
      "source HEAD changed while snapshotting");
    assert.equal((await sourceGit(["status", "--porcelain=v1", "--untracked-files=normal"])).trim(), "",
      "source checkout changed while snapshotting");
    return { schema_version: 1, url: pathToFileURL(directory).href, directory,
      source_commit: commit, tree_id: tree, branch: "candidate", shallow,
      tracked_files: entries.length, object_files: objectFiles };
  } catch (error) {
    const current = fs.lstatSync(directory, { throwIfNoEntry: false });
    if (current?.isDirectory() && !current.isSymbolicLink() && current.dev === owned.dev && current.ino === owned.ino) {
      fs.rmSync(directory, { recursive: true, force: false });
    }
    throw error;
  }
}
