#!/usr/bin/env node
/** Build and install one paired Homebrew payload; never provision models or launch a daemon. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const VENDOR_LICENSES = ["mlx/LICENSE", "mlx-c/LICENSE", "fmt/LICENSE", "json/LICENSE.MIT", "metal-cpp/LICENSE.txt"];
export function shellQuote(value) {
  assert.equal(typeof value, "string");
  assert(!/[\0\r\n]/.test(value), "installation paths cannot contain NUL or newlines");
  return `'${value.replaceAll("'", "'\\''")}'`;
}
export function sha256(file) { return createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }
function requiredFile(file) {
  assert(fs.statSync(file).isFile(), `required install file is missing: ${file}`);
  assert(fs.statSync(file).size > 0, `required install file is empty: ${file}`);
  return fs.realpathSync(file);
}
function unique(values, description) {
  const found = [...new Set(values.map((value) => fs.realpathSync(value)))];
  assert.equal(found.length, 1, `expected one ${description}, found ${found.length}`);
  return found[0];
}

/** Select this checkout's actual Cargo outputs, never the newest cached filename. */
export function selectNativeArtifacts(messages, metadata, sourceRoot) {
  const localPackage = (name, relative) => {
    const candidates = metadata.packages.filter((pkg) => pkg.name === name && path.resolve(pkg.manifest_path) === path.join(sourceRoot, relative));
    assert.equal(candidates.length, 1, `Cargo metadata did not identify this checkout's ${name}`);
    return candidates[0];
  };
  const daemonPackage = localPackage("alloyd", "crates/alloyd/Cargo.toml");
  const harnessPackage = localPackage("harness-core", "crates/harness-core/Cargo.toml");
  const ffiPackage = localPackage("mlx-ffi", "crates/mlx-ffi/Cargo.toml");
  const records = messages.split("\n").filter((line) => line.trim()).map((line) => JSON.parse(line));
  const daemon = unique(records.filter((r) => r.reason === "compiler-artifact" && r.package_id === daemonPackage.id && r.target?.name === "alloyd" && r.target.kind.includes("bin") && r.profile?.test === false && r.executable).map((r) => requiredFile(r.executable)), "candidate alloyd executable");
  const pipe_helper = unique(records.filter((r) => r.reason === "compiler-artifact" && r.package_id === harnessPackage.id && r.target?.name === "alloy-plugin-pipe-helper" && r.target.kind.includes("bin") && r.profile?.test === false && r.executable).map((r) => requiredFile(r.executable)), "candidate pipe helper");
  const exec_helper = unique(records.filter((r) => r.reason === "compiler-artifact" && r.package_id === harnessPackage.id && r.target?.name === "alloy-plugin-exec-helper" && r.target.kind.includes("bin") && r.profile?.test === false && r.executable).map((r) => requiredFile(r.executable)), "candidate exec helper");
  const out = unique(records.filter((r) => r.reason === "build-script-executed" && r.package_id === ffiPackage.id && r.out_dir).map((r) => r.out_dir), "candidate MLX build directory");
  const libraries = ["lib", "lib64"].map((dir) => path.join(out, dir, "mlx.metallib")).filter((file) => fs.existsSync(file));
  const metal = unique(libraries.map(requiredFile), "candidate mlx.metallib");
  return { daemon, pipe_helper, exec_helper, metal_library: metal, native_build_out_dir: out, version: daemonPackage.version };
}

function runtimeFiles(root) {
  return fs.readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const name = path.join(root, entry.name);
    assert(!entry.isSymbolicLink(), `installed payload contains an unexpected symlink: ${name}`);
    return entry.isDirectory() ? runtimeFiles(name) : [name];
  });
}

export function installPackage({ sourceRoot, prefix, optPrefix, nodeRuntime, sourceCommit, native }) {
  assert.match(sourceCommit, /^[0-9a-f]{40}$/, "source commit must be a full immutable Git SHA");
  for (const value of [sourceRoot, prefix, optPrefix, nodeRuntime]) {
    assert(path.isAbsolute(value), "installation paths must be absolute");
    shellQuote(value);
  }
  const cli = path.join(sourceRoot, "ts/cli");
  const manifest = JSON.parse(fs.readFileSync(path.join(cli, "package.json"), "utf8"));
  assert.equal(manifest.version, native.version, "CLI and daemon versions differ");
  const inputs = [
    [path.join(nodeRuntime, "bin/node"), "libexec/node/bin/node", true],
    [path.join(nodeRuntime, "LICENSE"), "libexec/node/LICENSE", false],
    [native.daemon, "libexec/native/alloyd", true],
    [native.pipe_helper, "libexec/native/alloy-plugin-pipe-helper", true],
    [native.exec_helper, "libexec/native/alloy-plugin-exec-helper", true],
    [native.metal_library, "libexec/native/mlx.metallib", false],
    [path.join(cli, "bin/alloy.js"), "libexec/cli/bin/alloy.js", true],
    [path.join(cli, "package.json"), "libexec/cli/package.json", false],
    [path.join(cli, "models/starter-model.json"), "libexec/cli/models/starter-model.json", false],
    [path.join(cli, "dist/native/alloy-metrics"), "libexec/cli/dist/native/alloy-metrics", true],
    [path.join(sourceRoot, "LICENSE"), "share/doc/alloy/LICENSE", false],
    ...VENDOR_LICENSES.map((relative) => [path.join(sourceRoot, "vendor", relative), `share/doc/alloy/licenses/${relative}`, false]),
  ];
  const jsRoot = path.join(cli, "dist/src");
  for (const file of runtimeFiles(jsRoot).filter((file) => file.endsWith(".js"))) inputs.push([file, `libexec/cli/dist/src/${path.relative(jsRoot, file)}`, false]);
  assert(inputs.some(([, file]) => file === "libexec/cli/dist/src/main.js"), "compiled CLI is missing");
  for (const [file] of inputs) requiredFile(file);
  // A new keg is Homebrew's transaction boundary. Never replace an existing payload.
  for (const relative of ["bin", "libexec", "share/doc/alloy"]) assert(!fs.existsSync(path.join(prefix, relative)), `refusing to overwrite existing install: ${relative}`);
  for (const [from, relative, executable] of inputs) {
    const dest = path.join(prefix, relative);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(from, dest, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(dest, executable ? 0o755 : 0o644);
  }
  const ownNode = shellQuote(path.join(optPrefix, "libexec/node/bin/node"));
  const ownCli = shellQuote(path.join(optPrefix, "libexec/cli/bin/alloy.js"));
  const ownDaemon = shellQuote(path.join(optPrefix, "libexec/native/alloyd"));
  const wrappers = {
    alloy: `#!/bin/sh\nset -eu\nif [ -z "\${ALLOYD:-}" ]; then\n  ALLOYD=${ownDaemon}\n  export ALLOYD\nfi\nexec ${ownNode} ${ownCli} "$@"\n`,
    alloyd: `#!/bin/sh\nset -eu\nexec ${ownDaemon} "$@"\n`,
  };
  fs.mkdirSync(path.join(prefix, "bin"), { recursive: true });
  for (const [name, body] of Object.entries(wrappers)) fs.writeFileSync(path.join(prefix, "bin", name), body, { flag: "wx", mode: 0o755 });
  // Homebrew adds its own receipts after install; attest only package-owned files.
  const installedPaths = [...inputs.map(([, relative]) => relative), ...Object.keys(wrappers).map((name) => `bin/${name}`)];
  const files = installedPaths.map((relative) => ({ path: relative.split(path.sep).join("/"), sha256: sha256(path.join(prefix, relative)) })).sort((a, b) => a.path.localeCompare(b.path));
  const receipt = { schema_version: 1, source_commit: sourceCommit, version: manifest.version, node_version: manifest.engines.node, files };
  fs.writeFileSync(path.join(prefix, "libexec/install.json"), `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx", mode: 0o644 });
  return receipt;
}

function command(program, args, options = {}) {
  const result = spawnSync(program, args, { cwd: ROOT, encoding: "utf8", stdio: "inherit", ...options });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `${path.basename(program)} ${args[0]} failed (${result.signal ?? result.status})`);
  return result.stdout;
}
export function parseOptions(args) {
  const options = {};
  const names = new Set(["--prefix", "--opt-prefix", "--node-runtime", "--source-commit", "--build-proof"]);
  for (let i = 0; i < args.length; i++) {
    const name = args[i];
    if (name === "--offline") { assert(!options.offline, "duplicate --offline"); options.offline = true; continue; }
    assert(names.has(name), `unknown argument ${name}`);
    assert(options[name] === undefined, `duplicate ${name}`);
    const value = args[++i];
    assert(value && !value.startsWith("--"), `${name} requires a value`);
    options[name] = value;
  }
  for (const name of ["--prefix", "--opt-prefix", "--node-runtime", "--source-commit"]) assert(options[name], `${name} is required`);
  assert.match(options["--source-commit"], /^[0-9a-f]{40}$/);
  return options;
}
export function main(args = process.argv.slice(2)) {
  assert(process.platform === "darwin" && process.arch === "arm64", "Homebrew packaging requires native Apple Silicon macOS");
  const options = parseOptions(args);
  const nodeRuntime = path.resolve(options["--node-runtime"]);
  const node = requiredFile(path.join(nodeRuntime, "bin/node"));
  const npm = requiredFile(path.join(nodeRuntime, "lib/node_modules/npm/bin/npm-cli.js"));
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, "ts/cli/package.json"), "utf8"));
  const version = command(node, ["--version"], { stdio: ["ignore", "pipe", "inherit"] }).trim();
  assert.equal(version, `v${pkg.engines.node}`, "build Node must match the repository pin");
  const rustPin = fs.readFileSync(path.join(ROOT, "rust-toolchain.toml"), "utf8").match(/channel\s*=\s*"([^"]+)"/)[1];
  const rustVersion = command(process.env.RUSTC || "rustc", ["--version"], { stdio: ["ignore", "pipe", "inherit"] });
  assert(rustVersion.startsWith(`rustc ${rustPin} `), `build Rust must match ${rustPin}`);
  if (fs.existsSync(path.join(ROOT, ".git"))) {
    assert.equal(command("git", ["rev-parse", "HEAD"], { stdio: ["ignore", "pipe", "inherit"] }).trim(), options["--source-commit"], "formula revision differs from checkout");
    assert.equal(command("git", ["status", "--porcelain"], { stdio: ["ignore", "pipe", "inherit"] }).trim(), "", "packaging requires a clean committed source tree");
  }
  const env = { ...process.env, PATH: `${path.join(nodeRuntime, "bin")}${path.delimiter}${process.env.PATH ?? ""}` };
  const offline = options.offline ? ["--offline"] : [];
  command(node, [npm, "ci", "--prefix", "ts/cli", "--ignore-scripts", "--no-audit", "--no-fund", ...offline], { env });
  command(node, [npm, "run", "build", "--prefix", "ts/cli"], { env });
  const metadata = JSON.parse(command("cargo", ["metadata", "--format-version", "1", "--no-deps", "--locked", ...offline], { env, stdio: ["ignore", "pipe", "inherit"], maxBuffer: 8 * 1024 * 1024 }));
  const logDir = path.join(ROOT, "target/homebrew");
  fs.mkdirSync(logDir, { recursive: true });
  const log = path.join(logDir, "cargo-build.ndjson");
  const fd = fs.openSync(log, "w");
  try { command("cargo", ["build", "--release", "--locked", ...offline, "-p", "alloyd", "-p", "harness-core", "--message-format=json-render-diagnostics"], { env, stdio: ["ignore", fd, "inherit"] }); }
  finally { fs.closeSync(fd); }
  const native = selectNativeArtifacts(fs.readFileSync(log, "utf8"), metadata, ROOT.replace(/\/$/, ""));
  const receipt = installPackage({ sourceRoot: ROOT, prefix: path.resolve(options["--prefix"]), optPrefix: path.resolve(options["--opt-prefix"]), nodeRuntime, sourceCommit: options["--source-commit"], native });
  if (options["--build-proof"]) fs.writeFileSync(path.resolve(options["--build-proof"]), `${JSON.stringify({ ...native, source_commit: receipt.source_commit }, null, 2)}\n`);
  process.stdout.write(`Installed Alloy ${receipt.version} from ${receipt.source_commit}\n`);
}
if (process.argv[1] && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(`Homebrew packaging: ${error.message}`); process.exitCode = 1; }
}
