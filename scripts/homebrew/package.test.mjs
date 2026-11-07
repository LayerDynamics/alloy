import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { installPackage, parseOptions, selectNativeArtifacts, sha256 } from "./package.mjs";
import { renderFormula } from "./formula.mjs";
const COMMIT = "bf95b93fe92f6ce0cdfc21ac8fecc0982ecc1b43";
function fixture(t) {
  // Compare actual installation identities even when TMPDIR contains a symlink.
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "alloy brew 'quoted-")));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const sourceRoot = path.join(root, "source");
  const prefix = path.join(root, "Cellar/alloy/0.1.0");
  const optPrefix = path.join(root, "opt/alloy");
  const nodeRuntime = path.join(root, "node");
  const write = (name, body, mode = 0o644) => { fs.mkdirSync(path.dirname(name), { recursive: true }); fs.writeFileSync(name, body, { mode }); };
  write(path.join(sourceRoot, "LICENSE"), "Alloy license");
  for (const relative of ["mlx/LICENSE", "mlx-c/LICENSE", "fmt/LICENSE", "json/LICENSE.MIT", "metal-cpp/LICENSE.txt"]) write(path.join(sourceRoot, "vendor", relative), relative);
  write(path.join(sourceRoot, "ts/cli/package.json"), JSON.stringify({ name: "alloy", type: "module", version: "0.1.0", engines: { node: process.versions.node } }));
  write(path.join(sourceRoot, "ts/cli/bin/alloy.js"), 'import "../dist/src/main.js";\n');
  write(path.join(sourceRoot, "ts/cli/dist/src/main.js"), 'process.stdout.write(JSON.stringify({argv:process.argv.slice(2),daemon:process.env.ALLOYD,node:process.execPath})+"\\n");\n');
  write(path.join(sourceRoot, "ts/cli/models/starter-model.json"), '{"fixture":true}\n');
  write(path.join(sourceRoot, "ts/cli/dist/native/alloy-metrics"), '#!/bin/sh\nprintf "fixture\\n"\n', 0o755);
  write(path.join(sourceRoot, "ts/cli/node_modules/should-not-ship"), "not a runtime dependency");
  write(path.join(sourceRoot, "ts/cli/src/should-not-ship.ts"), "source");
  write(path.join(nodeRuntime, "LICENSE"), "Node license");
  fs.mkdirSync(path.join(nodeRuntime, "bin"), { recursive: true });
  fs.symlinkSync(process.execPath, path.join(nodeRuntime, "bin/node"));
  const daemon = path.join(sourceRoot, "target/release/alloyd");
  const out = path.join(sourceRoot, "target/release/build/mlx-ffi/out");
  write(daemon, '#!/bin/sh\nprintf "%s\\n" "$@"\n', 0o755);
  write(path.join(out, "lib/mlx.metallib"), "candidate metallib");
  const metadata = { packages: [
    { name: "alloyd", id: "candidate-daemon", manifest_path: path.join(sourceRoot, "crates/alloyd/Cargo.toml"), version: "0.1.0" },
    { name: "mlx-ffi", id: "candidate-ffi", manifest_path: path.join(sourceRoot, "crates/mlx-ffi/Cargo.toml"), version: "0.1.0" },
  ] };
  const records = [
    { reason: "compiler-artifact", package_id: "candidate-daemon", target: { name: "alloyd", kind: ["bin"] }, profile: { test: false }, executable: daemon },
    { reason: "build-script-executed", package_id: "candidate-ffi", out_dir: out },
  ];
  const messages = records.map((r) => JSON.stringify(r)).join("\n");
  const native = selectNativeArtifacts(messages, metadata, sourceRoot);
  fs.mkdirSync(path.dirname(optPrefix), { recursive: true });
  fs.mkdirSync(prefix, { recursive: true });
  fs.symlinkSync(prefix, optPrefix);
  return { sourceRoot, prefix, optPrefix, nodeRuntime, sourceCommit: COMMIT, native, metadata, records, messages, write };
}
test("installed wrappers retain exact arguments and paired runtime outside the deleted checkout", (t) => {
  const f = fixture(t);
  const receipt = installPackage(f);
  fs.rmSync(f.sourceRoot, { recursive: true, force: true });
  fs.rmSync(f.nodeRuntime, { recursive: true, force: true });
  const literal = "a spaced value ' $() `not a command`";
  const run = (name, args, env = {}) => spawnSync(path.join(f.prefix, "bin", name), args, { cwd: os.tmpdir(), env: { PATH: "/usr/bin:/bin", ...env }, encoding: "utf8" });
  const cli = run("alloy", [literal, "--version"]);
  assert.equal(cli.status, 0, cli.stderr);
  const data = JSON.parse(cli.stdout);
  assert.deepEqual(data.argv, [literal, "--version"]);
  assert.equal(data.daemon, path.join(f.optPrefix, "libexec/native/alloyd"));
  assert.equal(fs.realpathSync(data.node), path.join(f.prefix, "libexec/node/bin/node"));
  const override = "/deliberate/custom daemon";
  assert.equal(JSON.parse(run("alloy", [], { ALLOYD: override }).stdout).daemon, override);
  assert.equal(run("alloyd", [literal]).stdout, `${literal}\n`);
  assert(!fs.existsSync(path.join(f.prefix, "libexec/cli/node_modules")));
  assert(!fs.existsSync(path.join(f.prefix, "libexec/cli/src")));
  for (const file of receipt.files) assert.equal(sha256(path.join(f.prefix, file.path)), file.sha256);
  assert(receipt.files.some((file) => file.path === "libexec/native/mlx.metallib"));
});
test("missing Metal kernels fail before a partial package is installed", (t) => {
  const f = fixture(t);
  fs.rmSync(f.native.metal_library);
  assert.throws(() => installPackage(f), /ENOENT/);
  assert(!fs.existsSync(path.join(f.prefix, "bin")));
  assert(!fs.existsSync(path.join(f.prefix, "libexec")));
});
test("package refuses mixed CLI/daemon versions and existing installed payloads", (t) => {
  const f = fixture(t);
  assert.throws(() => installPackage({ ...f, native: { ...f.native, version: "0.2.0" } }), /versions differ/);
  fs.mkdirSync(path.join(f.prefix, "bin"));
  f.write(path.join(f.prefix, "bin/keep"), "existing installation");
  assert.throws(() => installPackage(f), /refusing to overwrite/);
  assert.equal(fs.readFileSync(path.join(f.prefix, "bin/keep"), "utf8"), "existing installation");
});
test("Cargo selection rejects a foreign artifact and ambiguous native outputs", (t) => {
  const f = fixture(t);
  const foreign = f.records.map((r) => ({ ...r, package_id: "other-checkout" }));
  assert.throws(() => selectNativeArtifacts(foreign.map(JSON.stringify).join("\n"), f.metadata, f.sourceRoot), /candidate alloyd/);
  const other = path.join(f.sourceRoot, "target/other-ffi/out");
  f.write(path.join(other, "lib/mlx.metallib"), "stale cached Metal library");
  const ambiguous = [...f.records, { reason: "build-script-executed", package_id: "candidate-ffi", out_dir: other }];
  assert.throws(() => selectNativeArtifacts(ambiguous.map(JSON.stringify).join("\n"), f.metadata, f.sourceRoot), /MLX build directory/);
});
test("formula source pin cannot be a moving branch or injected Ruby", () => {
  for (const commit of ["main", "a4a17df", `${COMMIT}\";system(\"oops\")`, COMMIT.toUpperCase()]) assert.throws(() => renderFormula({ commit }));
  const formula = renderFormula({ commit: COMMIT, revision: 1 });
  assert(formula.includes(`revision: "${COMMIT}"`));
  assert(formula.includes("  revision 1\n"));
  assert(!formula.includes("@@"));
  assert(formula.includes('"--offline"'));
  assert.throws(() => renderFormula({ commit: COMMIT, revision: -1 }));
});
test("packaging requires explicit destination, toolchain, and immutable source identity", () => {
  assert.throws(() => parseOptions(["--prefix", "/tmp/p"]), /required/);
  assert.throws(() => parseOptions(["--prefix", "--offline"]), /requires a value/);
  assert.throws(() => parseOptions(["--unknown"]), /unknown/);
});
