import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { cleanupOwned, createOwnership, createTemporaryMarker, ownershipContext, ownershipFile, updateOwnership } from "./ci.mjs";

const COMMIT = "c".repeat(40);
const MARKER = "alloy-ci-owner.json";

function fixture(t, { marker = true, token } = {}) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "alloy-brew-cleanup-test-")));
  const base = path.join(root, "evidence");
  fs.mkdirSync(base);
  const env = { GITHUB_RUN_ID: "123456789", GITHUB_RUN_ATTEMPT: "2", ALLOY_ACCEPTANCE_OUTPUT_DIR: base };
  const context = ownershipContext(env, COMMIT);
  const brew = path.join(root, "brew");
  fs.writeFileSync(brew, "never executed by these tests", { mode: 0o755 });
  const rendered = `# pinned source ${COMMIT}\nclass OwnedFormula; end\n`;
  const owner = createOwnership(context, brew, rendered, token);
  t.after(() => {
    fs.rmSync(owner.temporary, { recursive: true, force: true });
    fs.rmSync(root, { recursive: true, force: true });
  });
  fs.mkdirSync(owner.output);
  if (marker) createTemporaryMarker(owner);
  const repository = path.join(root, "cloned-tap");
  const state = { tap: false, installed: false, origin: owner.local_tap, failUninstall: false };
  const calls = [];
  const prepare = () => {
    fs.mkdirSync(path.join(owner.local_tap, "Formula"), { recursive: true });
    fs.mkdirSync(path.join(repository, "Formula"), { recursive: true });
    fs.writeFileSync(path.join(repository, "Formula", `${owner.name}.rb`), rendered);
    updateOwnership(context, owner, { preflight_verified: true, tap_attempted: true, install_attempted: true });
    state.tap = true;
    state.installed = true;
  };
  const run = async (command, args, options) => {
    calls.push({ command, args });
    assert.equal(options.env.HOMEBREW_NO_AUTOREMOVE, "1");
    assert(options.timeoutMs > 0 && options.timeoutMs <= 90000);
    if (command === "/usr/bin/which") { assert.deepEqual(args, ["brew"]); return { stdout: `${brew}\n` }; }
    if (command === "git") { assert.deepEqual(args, ["-C", repository, "remote", "get-url", "origin"]); return { stdout: `${state.origin}\n` }; }
    assert.equal(command, brew);
    if (args[0] === "tap") { assert.deepEqual(args, ["tap"]); return { stdout: `unrelated/keep\n${state.tap ? `${owner.tap}\n` : ""}` }; }
    if (args[0] === "--repository") { assert.deepEqual(args, ["--repository", owner.tap]); return { stdout: `${repository}\n` }; }
    if (args[0] === "list") { assert.deepEqual(args, ["list", "--formula", "--versions", owner.name]); return { stdout: state.installed ? `${owner.name} 0.1.0\n` : "" }; }
    if (args[0] === "uninstall") {
      assert.deepEqual(args, ["uninstall", "--formula", owner.formula]);
      assert.equal(JSON.parse(fs.readFileSync(ownershipFile(context), "utf8")).install_attempted, true);
      if (state.failUninstall) throw new Error("simulated uninstall failure");
      state.installed = false;
      return { stdout: "" };
    }
    if (args[0] === "untap") { assert.deepEqual(args, ["untap", owner.tap]); state.tap = false; return { stdout: "" }; }
    throw new Error(`unexpected cleanup command ${command} ${args.join(" ")}`);
  };
  return { root, env, context, owner, repository, state, calls, prepare, run };
}

test("durable mutation intent recovers an interrupted owner and repeated cleanup is harmless", async (t) => {
  const token = randomUUID();
  const f = fixture(t, { token });
  assert.equal(f.owner.token, token);
  assert(f.owner.temporary.endsWith(token));
  f.prepare();
  fs.mkdirSync(path.join(f.owner.temporary, "build"));
  fs.writeFileSync(path.join(f.owner.temporary, "build/partial-output"), "unfinished native build");
  const result = await cleanupOwned(f.context, { run: f.run });
  assert.equal(result.status, "cleaned");
  assert.deepEqual(result.actions.map((item) => item.action), ["uninstall", "untap", "remove_temporary"]);
  assert.equal(f.state.installed, false);
  assert.equal(f.state.tap, false);
  assert.equal(fs.existsSync(f.owner.temporary), false);
  assert.equal(JSON.parse(fs.readFileSync(ownershipFile(f.context), "utf8")).cleanup_phase, "complete");
  const count = f.calls.length;
  assert.equal((await cleanupOwned(f.context, { run: f.run })).status, "already_clean");
  assert.equal(f.calls.length, count, "completed cleanup must execute no further Homebrew commands");
  assert(f.calls.every((call) => !call.args.includes("cleanup") && !call.args.includes("autoremove")));
});

test("cleanup never searches for or adopts an earlier run or attempt", async (t) => {
  const f = fixture(t);
  f.prepare();
  for (const identity of [{ ...f.env, GITHUB_RUN_ID: "123456788" }, { ...f.env, GITHUB_RUN_ATTEMPT: "3" }]) {
    const current = ownershipContext(identity, COMMIT);
    assert.equal((await cleanupOwned(current, { run: f.run })).status, "no_state");
  }
  assert.equal(f.calls.length, 0);
  assert(fs.existsSync(path.join(f.owner.temporary, MARKER)));
  assert.equal(f.state.installed, true);
});

test("wrong source, run, attempt, identifiers, or temporary path refuse all cleanup commands", async (t) => {
  const f = fixture(t);
  assert.throws(() => createOwnership(f.context, f.owner.brew, "unused", "../foreign"), /invalid ownership token/);
  f.prepare();
  const filename = ownershipFile(f.context);
  const original = fs.readFileSync(filename, "utf8");
  for (const change of [
    { source_commit: "d".repeat(40) }, { run_id: "123456788" }, { run_attempt: "3" },
    { formula: "unrelated/keep/tool" }, { temporary: f.root }, { local_tap: f.root },
  ]) {
    fs.writeFileSync(filename, JSON.stringify({ ...JSON.parse(original), ...change }));
    await assert.rejects(cleanupOwned(f.context, { run: f.run }));
    assert.equal(f.calls.length, 0);
  }
  fs.writeFileSync(filename, original);
  assert(fs.existsSync(path.join(f.owner.temporary, MARKER)));
});

test("marker and actual tap provenance must match before an installed keg can be removed", async (t) => {
  const marker = fixture(t);
  marker.prepare();
  const filename = path.join(marker.owner.temporary, MARKER);
  const data = JSON.parse(fs.readFileSync(filename, "utf8"));
  fs.writeFileSync(filename, JSON.stringify({ ...data, source_commit: "d".repeat(40) }));
  await assert.rejects(cleanupOwned(marker.context, { run: marker.run }), /marker mismatch/);
  assert.equal(marker.calls.length, 0);
  const origin = fixture(t);
  origin.prepare();
  origin.state.origin = origin.root;
  await assert.rejects(cleanupOwned(origin.context, { run: origin.run }), /tap origin/);
  assert(origin.calls.every((call) => !["uninstall", "untap"].includes(call.args[0])));
  const formula = fixture(t);
  formula.prepare();
  fs.appendFileSync(path.join(formula.repository, "Formula", `${formula.owner.name}.rb`), "changed source");
  await assert.rejects(cleanupOwned(formula.context, { run: formula.run }), /formula differs/);
  assert(formula.calls.every((call) => !["uninstall", "untap"].includes(call.args[0])));
});

test("a failed uninstall retains ownership proof and can be retried safely", async (t) => {
  const f = fixture(t);
  f.prepare();
  f.state.failUninstall = true;
  await assert.rejects(cleanupOwned(f.context, { run: f.run }), /simulated uninstall failure/);
  assert.equal(f.state.tap, true);
  assert(fs.existsSync(path.join(f.owner.temporary, MARKER)));
  assert.equal(JSON.parse(fs.readFileSync(ownershipFile(f.context), "utf8")).cleanup_phase, "pending");
  f.state.failUninstall = false;
  assert.equal((await cleanupOwned(f.context, { run: f.run })).status, "cleaned");
});

test("interruption before marker creation permits only an empty exact temporary root", async (t) => {
  const empty = fixture(t, { marker: false });
  fs.mkdirSync(empty.owner.temporary);
  assert.equal((await cleanupOwned(empty.context, { run: empty.run })).status, "cleaned");
  assert.equal(empty.calls.length, 0);
  const populated = fixture(t, { marker: false });
  fs.mkdirSync(populated.owner.temporary);
  fs.writeFileSync(path.join(populated.owner.temporary, "keep"), "unproven contents");
  await assert.rejects(cleanupOwned(populated.context, { run: populated.run }), /unmarked temporary directory contains files/);
  assert.equal(fs.readFileSync(path.join(populated.owner.temporary, "keep"), "utf8"), "unproven contents");
  assert.equal(populated.calls.length, 0);
});

test("interrupted temporary deletion resumes with its marker or after the final empty-root cut point", async (t) => {
  const marked = fixture(t);
  updateOwnership(marked.context, marked.owner, { cleanup_phase: "resources_removed" });
  fs.mkdirSync(path.join(marked.owner.temporary, "partially-removed-build"));
  fs.writeFileSync(path.join(marked.owner.temporary, "partially-removed-build/remaining"), "owned");
  assert.equal((await cleanupOwned(marked.context, { run: marked.run })).status, "cleaned");
  assert.equal(marked.calls.length, 0);
  const empty = fixture(t);
  updateOwnership(empty.context, empty.owner, { cleanup_phase: "resources_removed" });
  fs.unlinkSync(path.join(empty.owner.temporary, MARKER));
  assert.equal((await cleanupOwned(empty.context, { run: empty.run })).status, "cleaned");
  assert.equal(empty.calls.length, 0);
  const unmarked = fixture(t);
  updateOwnership(unmarked.context, unmarked.owner, { cleanup_phase: "resources_removed" });
  fs.unlinkSync(path.join(unmarked.owner.temporary, MARKER));
  fs.writeFileSync(path.join(unmarked.owner.temporary, "keep"), "unproven contents");
  await assert.rejects(cleanupOwned(unmarked.context, { run: unmarked.run }), /unmarked temporary directory contains files/);
  assert.equal(unmarked.calls.length, 0);
});
