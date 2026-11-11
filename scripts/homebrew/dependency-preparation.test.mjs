import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { prepareCargoDependencies } from "./ci.mjs";

function fixture(t, mode) {
  const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-prefetch-"));
  t.after(() => fs.rmSync(sourceRoot, { recursive: true, force: true }));
  fs.writeFileSync(path.join(sourceRoot, "Cargo.lock"), "locked fixture\n");
  fs.writeFileSync(path.join(sourceRoot, "rust-toolchain.toml"), '[toolchain]\nchannel = "1.99.1"\n');
  const calls = [];
  const run = async (command, args, options) => {
    calls.push({ command, args, options });
    if (command === "/selected/brew") {
      assert.deepEqual(args, ["--cache"]);
      return { code: 0, stdout: mode === "bad-cache" ? "relative\n" : "/dedicated/homebrew-cache\n" };
    }
    assert.equal(command, "cargo");
    assert.equal(options.cwd, sourceRoot);
    assert.equal(options.env.CARGO_HOME, "/dedicated/homebrew-cache/alloy-cargo");
    assert.equal(options.env.CARGO_TARGET_DIR, undefined);
    assert.equal(options.env.HOMEBREW_CACHE, "/dedicated/homebrew-cache");
    assert(options.timeoutMs > 0 && options.timeoutMs <= 120000);
    if (args[1] === "--version") return { code: 0, stdout: mode === "bad-version" ? "cargo 1.98.0 (wrong)" : "cargo 1.99.1 (pinned)" };
    assert.deepEqual(args.slice(0, 5), ["+1.99.1", "fetch", "--locked", "--target", "aarch64-apple-darwin"]);
    if (args.includes("--offline")) {
      assert.deepEqual(options.allowed, [0, 101]);
      if (mode === "mutated") fs.appendFileSync(path.join(sourceRoot, "Cargo.lock"), "changed\n");
      return { code: mode === "hit" || mode === "mutated" ? 0 : 101, stdout: "" };
    }
    if (mode === "failure") throw new Error("network failed");
    return { code: 0, stdout: "" };
  };
  return { calls, options: { brew: "/selected/brew", sourceRoot, env: { HOMEBREW_CACHE: "/dedicated/homebrew-cache" }, run } };
}

test("CI dependency cache hit avoids network and records pinned preparation", async (t) => {
  const f = fixture(t, "hit");
  const result = await prepareCargoDependencies(f.options);
  assert.equal(result.cache_hit, true);
  assert.match(result.lockfile_sha256, /^[0-9a-f]{64}$/);
  assert.equal(result.cargo_version, "cargo 1.99.1 (pinned)");
  assert.equal(result.commands.length, 1);
  assert.equal(f.calls.length, 3);
});
test("CI dependency miss performs only locked download preparation", async (t) => {
  const f = fixture(t, "miss");
  const result = await prepareCargoDependencies(f.options);
  assert.equal(result.cache_hit, false);
  assert.equal(result.commands.length, 2);
  assert.equal(f.calls.length, 4);
  assert(!f.calls.some(({ args }) => args.includes("build") || args.includes("install")));
});
for (const [mode, error] of [["failure", /network failed/], ["mutated", /changed Cargo.lock/], ["bad-cache", /invalid Homebrew cache/], ["bad-version", /Cargo version mismatch/]]) {
  test(`CI dependency preparation fails closed: ${mode}`, async (t) => {
    const f = fixture(t, mode);
    await assert.rejects(prepareCargoDependencies(f.options), error);
  });
}
