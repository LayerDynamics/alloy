import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { test } from "node:test";
import { createSourceSnapshot } from "./source.mjs";

const gitEnv = { ...process.env, GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: os.devNull, GIT_OPTIONAL_LOCKS: "0" };
function git(args, allowed = [0]) {
  const result = spawnSync("git", args, { env: gitEnv, encoding: "utf8", timeout: 15000, killSignal: "SIGKILL", maxBuffer: 4 * 1024 * 1024 });
  assert.ifError(result.error);
  assert(allowed.includes(result.status), "git " + args.join(" ") + ": " + result.stderr);
  return result.stdout.trim();
}
function fixture(t) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "alloy source ' spaced-")));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, "checkout");
  git(["init", "--initial-branch=main", source]);
  configure(source);
  write(source, "README.md", "frozen source\n");
  write(source, "nested/a spaced file.txt", "all content remains available\n");
  git(["-C", source, "add", "."]);
  git(["-C", source, "commit", "-m", "base"]);
  return { root, source, destination: path.join(root, "snapshot ' source.git"),
    expectedCommit: git(["-C", source, "rev-parse", "HEAD"]) };
}
function configure(source) {
  git(["-C", source, "config", "user.name", "Alloy fixture"]);
  git(["-C", source, "config", "user.email", "fixture@alloy.invalid"]);
  git(["-C", source, "config", "commit.gpgsign", "false"]);
  git(["-C", source, "config", "maintenance.auto", "false"]);
  git(["-C", source, "config", "gc.auto", "0"]);
}
function write(root, relative, text) {
  const filename = path.join(root, relative);
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  fs.writeFileSync(filename, text);
}
function gitMetadata(source) {
  const base = path.join(source, ".git");
  const files = {};
  function walk(directory) {
    for (const name of fs.readdirSync(directory).sort()) {
      const filename = path.join(directory, name);
      if (fs.lstatSync(filename).isDirectory()) walk(filename);
      else files[path.relative(base, filename)] = createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
    }
  }
  walk(base);
  return files;
}
function cloneCandidate(snapshot, destination, { depth } = {}) {
  git(["clone", "--no-checkout", ...(depth ? ["--depth", String(depth)] : []), snapshot.url, destination]);
  git(["-C", destination, "checkout", "--detach", snapshot.source_commit]);
  assert.equal(git(["-C", destination, "rev-parse", "HEAD"]), snapshot.source_commit);
  assert.equal(git(["-C", destination, "rev-parse", "HEAD^{tree}"]), snapshot.tree_id);
  assert.equal(fs.readFileSync(path.join(destination, "README.md"), "utf8"), "frozen source\n");
  assert.equal(fs.readFileSync(path.join(destination, "nested/a spaced file.txt"), "utf8"), "all content remains available\n");
}
function syntheticMerge(t) {
  const f = fixture(t);
  git(["-C", f.source, "switch", "-c", "feature"]);
  write(f.source, "feature.txt", "feature side\n");
  git(["-C", f.source, "add", "."]);
  git(["-C", f.source, "commit", "-m", "feature"]);
  git(["-C", f.source, "switch", "main"]);
  write(f.source, "main.txt", "main side\n");
  git(["-C", f.source, "add", "."]);
  git(["-C", f.source, "commit", "-m", "main"]);
  const upstream = path.join(f.root, "published.git");
  git(["clone", "--bare", "--no-hardlinks", f.source, upstream]);
  git(["-C", f.source, "checkout", "--detach"]);
  git(["-C", f.source, "merge", "--no-ff", "feature", "-m", "synthetic merge"]);
  f.expectedCommit = git(["-C", f.source, "rev-parse", "HEAD"]);
  const missing = spawnSync("git", ["--git-dir", upstream, "cat-file", "-e", f.expectedCommit + "^{commit}"],
    { env: gitEnv, encoding: "utf8", timeout: 15000 });
  assert.equal(missing.status, 128, "synthetic merge must not exist in the published remote");
  return { ...f, upstream };
}

test("owned snapshot advertises the exact source and remains cloneable after checkout removal", async (t) => {
  const f = fixture(t);
  const before = gitMetadata(f.source);
  const tree = git(["-C", f.source, "rev-parse", "HEAD^{tree}"]);
  const snapshot = await createSourceSnapshot({ sourceRoot: f.source, ...f });
  assert.deepEqual(gitMetadata(f.source), before, "snapshotting must not mutate source Git metadata");
  assert.equal(snapshot.tree_id, tree);
  assert.equal(snapshot.source_commit, f.expectedCommit);
  assert.equal(snapshot.branch, "candidate");
  assert.equal(snapshot.shallow, false);
  assert.equal(snapshot.tracked_files, 2);
  assert.match(snapshot.url, /^file:\/\//);
  assert.equal(git(["ls-remote", "--heads", snapshot.url, "candidate"]), f.expectedCommit + "\trefs/heads/candidate");
  assert.equal(git(["--git-dir", snapshot.directory, "remote"]), "");
  assert(!fs.existsSync(path.join(snapshot.directory, "objects/info/alternates")));
  fs.rmSync(f.source, { recursive: true });
  cloneCandidate(snapshot, path.join(f.root, "consumer"));
});

test("a detached synthetic merge absent from the published remote is served as candidate", async (t) => {
  const f = syntheticMerge(t);
  const before = gitMetadata(f.source);
  const snapshot = await createSourceSnapshot({ sourceRoot: f.source, ...f });
  assert.deepEqual(gitMetadata(f.source), before);
  fs.rmSync(f.source, { recursive: true });
  fs.rmSync(f.upstream, { recursive: true });
  const consumer = path.join(f.root, "merge-consumer");
  cloneCandidate(snapshot, consumer);
  assert.equal(fs.readFileSync(path.join(consumer, "feature.txt"), "utf8"), "feature side\n");
  assert.equal(fs.readFileSync(path.join(consumer, "main.txt"), "utf8"), "main side\n");
});

test("a shallow detached merge remains complete and cloneable without its original history", async (t) => {
  const f = syntheticMerge(t);
  const shallowSource = path.join(f.root, "shallow-checkout");
  git(["clone", "--depth", "1", pathToFileURL(f.source).href, shallowSource]);
  assert.equal(git(["-C", shallowSource, "rev-parse", "--is-shallow-repository"]), "true");
  assert.equal(git(["-C", shallowSource, "rev-parse", "HEAD"]), f.expectedCommit);
  const before = gitMetadata(shallowSource);
  const snapshot = await createSourceSnapshot({ ...f, sourceRoot: shallowSource });
  assert.equal(snapshot.shallow, true);
  assert.deepEqual(gitMetadata(shallowSource), before);
  fs.rmSync(shallowSource, { recursive: true });
  fs.rmSync(f.source, { recursive: true });
  fs.rmSync(f.upstream, { recursive: true });
  const consumer = path.join(f.root, "shallow-consumer");
  cloneCandidate(snapshot, consumer, { depth: 1 });
  assert.equal(fs.readFileSync(path.join(consumer, "feature.txt"), "utf8"), "feature side\n");
  assert.equal(fs.readFileSync(path.join(consumer, "main.txt"), "utf8"), "main side\n");
});

test("borrowed object storage is dissociated without hardlinks or surviving alternates", async (t) => {
  const f = fixture(t);
  const borrowed = path.join(f.root, "borrowed");
  git(["clone", "--shared", f.source, borrowed]);
  assert(fs.existsSync(path.join(borrowed, ".git/objects/info/alternates")));
  const before = gitMetadata(borrowed);
  const snapshot = await createSourceSnapshot({ ...f, sourceRoot: borrowed });
  assert.deepEqual(gitMetadata(borrowed), before);
  fs.rmSync(borrowed, { recursive: true });
  fs.rmSync(f.source, { recursive: true });
  cloneCandidate(snapshot, path.join(f.root, "independent-consumer"));
});

test("invalid identity, paths and existing destinations fail without replacing user files", async (t) => {
  const f = fixture(t);
  const options = { sourceRoot: f.source, ...f };
  await assert.rejects(createSourceSnapshot({ ...options, expectedCommit: "main" }), /full lowercase Git SHA/);
  await assert.rejects(createSourceSnapshot({ ...options, expectedCommit: "0".repeat(40) }), /differs from the expected/);
  await assert.rejects(createSourceSnapshot({ ...options, destination: "relative.git" }), /absolute filesystem path/);
  await assert.rejects(createSourceSnapshot({ ...options, destination: path.join(f.source, "snapshot.git") }), /outside the source/);
  await assert.rejects(createSourceSnapshot({ ...options, timeoutMs: 0 }), /snapshot timeout/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(createSourceSnapshot({ ...options, signal: controller.signal }), /cancelled/);
  assert(!fs.existsSync(f.destination));
  fs.mkdirSync(f.destination);
  write(f.destination, "keep", "existing data");
  await assert.rejects(createSourceSnapshot(options), /already exists/);
  assert.equal(fs.readFileSync(path.join(f.destination, "keep"), "utf8"), "existing data");
  const dangling = path.join(f.root, "dangling.git");
  fs.symlinkSync(path.join(f.root, "absent"), dangling);
  await assert.rejects(createSourceSnapshot({ ...options, destination: dangling }), /already exists/);
  assert(fs.lstatSync(dangling).isSymbolicLink());
});

test("uncommitted source changes cannot silently disappear from acceptance", async (t) => {
  const f = fixture(t);
  write(f.source, "README.md", "uncommitted change");
  const before = gitMetadata(f.source);
  await assert.rejects(createSourceSnapshot({ sourceRoot: f.source, ...f }), /clean committed checkout/);
  assert.deepEqual(gitMetadata(f.source), before);
  assert(!fs.existsSync(f.destination));
});

function blockedClone(f, { ignoreTerm = false } = {}) {
  const realGit = spawnSync("/bin/sh", ["-c", "command -v git"], { encoding: "utf8" }).stdout.trim();
  assert(path.isAbsolute(realGit));
  const shim = path.join(f.root, "git-shim");
  fs.mkdirSync(shim);
  const gate = path.join(f.root, "clone.pid");
  const body = "#!/bin/sh\nset -eu\nif [ \"$1\" = clone ]; then\n" +
    (ignoreTerm ? "  trap '' TERM\n" : "") +
    "  printf '%s\\n' \"$$\" > \"$ALLOY_SNAPSHOT_GATE\"\n  exec /bin/sleep 60\nfi\nexec \"$ALLOY_REAL_GIT\" \"$@\"\n";
  fs.writeFileSync(path.join(shim, "git"), body, { mode: 0o755 });
  return { gate, env: { ...gitEnv, PATH: shim + path.delimiter + process.env.PATH,
    ALLOY_REAL_GIT: realGit, ALLOY_SNAPSHOT_GATE: gate } };
}
async function waitForFile(filename) {
  const deadline = performance.now() + 5000;
  while (!fs.existsSync(filename)) {
    assert(performance.now() < deadline, "clone never entered its deterministic gate");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
function assertStopped(gate) {
  const pid = Number(fs.readFileSync(gate, "utf8").trim());
  assert(Number.isSafeInteger(pid) && pid > 1);
  assert.throws(() => process.kill(pid, 0), { code: "ESRCH" }, "owned clone process must have exited");
}

test("AbortSignal cancels an occupied clone and removes only its new destination", async (t) => {
  const f = fixture(t);
  const gate = blockedClone(f);
  const controller = new AbortController();
  const pending = createSourceSnapshot({ sourceRoot: f.source, ...f, env: gate.env, signal: controller.signal });
  pending.catch(() => {});
  await waitForFile(gate.gate);
  controller.abort();
  await assert.rejects(pending, /cancelled/);
  assertStopped(gate.gate);
  assert(!fs.existsSync(f.destination));
  assert.equal(git(["-C", f.source, "rev-parse", "HEAD"]), f.expectedCommit);
});

test("the overall deadline kills a clone that ignores TERM and removes its partial snapshot", async (t) => {
  const f = fixture(t);
  const gate = blockedClone(f, { ignoreTerm: true });
  const pending = createSourceSnapshot({ sourceRoot: f.source, ...f, env: gate.env, timeoutMs: 3000 });
  pending.catch(() => {});
  await waitForFile(gate.gate);
  await assert.rejects(pending, /timed out/);
  assertStopped(gate.gate);
  assert(!fs.existsSync(f.destination));
});
