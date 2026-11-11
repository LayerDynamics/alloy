import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { renderFormula } from "./formula.mjs";

const formula = renderFormula({ commit: "1".repeat(40) });
const fetchMethod = formula.slice(formula.indexOf("  def fetch\n"), formula.indexOf("  def install\n"));

function runFetch(mode) {
  // Execute the rendered Ruby method; replace only external command execution.
  const ruby = `
require "json"
require "pathname"
class FetchFixture
  attr_reader :calls
  def initialize(mode); @mode = mode; @calls = []; end
  def prepare_toolchains; Pathname.new("/pinned-node"); end
  def quiet_system(*args)
    @calls << args.map(&:to_s)
    @mode == "hit"
  end
  def system(*args)
    @calls << args.map(&:to_s)
    raise "network failure" if @mode == "failure" && args.first == "cargo"
  end
${fetchMethod}
end
fixture = FetchFixture.new(ARGV.fetch(0))
begin
  fixture.fetch
  puts JSON.generate(calls: fixture.calls, ok: true)
rescue => error
  puts JSON.generate(calls: fixture.calls, ok: false, error: error.message)
end
`;
  const result = spawnSync("ruby", ["-e", ruby, mode], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
const offline = ["cargo", "fetch", "--locked", "--offline", "--target", "aarch64-apple-darwin"];
const online = ["cargo", "fetch", "--locked", "--target", "aarch64-apple-darwin"];

test("cached locked dependencies avoid registry access but still install CLI dependencies", () => {
  const result = runFetch("hit");
  assert.equal(result.ok, true);
  assert.deepEqual(result.calls[0], offline);
  assert.equal(result.calls.length, 2);
  assert.equal(result.calls[1][0], "/pinned-node/bin/node");
  assert(result.calls[1].includes("ci"));
});
test("cache miss fetches exact locked dependencies before continuing", () => {
  const result = runFetch("miss");
  assert.equal(result.ok, true);
  assert.deepEqual(result.calls.slice(0, 2), [offline, online]);
  assert.equal(result.calls.length, 3);
});
test("failed online fetch stops installation instead of accepting incomplete cache", () => {
  const result = runFetch("failure");
  assert.equal(result.ok, false);
  assert.deepEqual(result.calls, [offline, online]);
});
test("only dependencies persist while each installed native build remains fresh", () => {
  assert(formula.includes('ENV["CARGO_HOME"] = (HOMEBREW_CACHE/"alloy-cargo").to_s'));
  assert(formula.includes('ENV["CARGO_TARGET_DIR"] = (buildpath/"target/homebrew/cargo-target").to_s'));
  assert(formula.includes('ENV["CARGO_NET_RETRY"] = "3"'));
  assert(formula.includes('ENV["CARGO_HTTP_TIMEOUT"] = "30"'));
  const install = formula.slice(formula.indexOf("  def install\n"), formula.indexOf("  def caveats\n"));
  assert(install.includes('"--offline"'));
  assert(install.includes('"scripts/homebrew/package.mjs"'));
});
