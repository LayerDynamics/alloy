#!/usr/bin/env node
/** Render an immutable tap formula from source-controlled packaging and toolchain pins. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
export function renderFormula({ commit, revision = 0, root = ROOT }) {
  assert.match(commit, /^[0-9a-f]{40}$/, "source commit must be a full lowercase immutable Git SHA");
  assert(Number.isSafeInteger(revision) && revision >= 0, "formula revision must be a nonnegative integer");
  const pkg = JSON.parse(fs.readFileSync(path.join(root, "ts/cli/package.json"), "utf8"));
  assert.match(pkg.version, /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/, "package version must be semantic");
  const toolchains = JSON.parse(fs.readFileSync(path.join(root, "packaging/homebrew/toolchains.json"), "utf8"));
  const rustPin = fs.readFileSync(path.join(root, "rust-toolchain.toml"), "utf8").match(/channel\s*=\s*"([^"]+)"/)?.[1];
  assert.equal(toolchains.node.version, pkg.engines.node, "Homebrew Node pin differs from CLI pin");
  assert.equal(toolchains.rust.version, rustPin, "Homebrew Rust pin differs from native pin");
  for (const [name, resource] of Object.entries(toolchains)) {
    assert.match(resource.sha256, /^[0-9a-f]{64}$/, `invalid ${name} checksum`);
    const origin = name === "node" ? "https://nodejs.org/" : "https://static.rust-lang.org/";
    assert(resource.url.startsWith(origin) && resource.url.includes(resource.version), `unexpected ${name} source`);
    assert(!/[\r\n"\\]/.test(resource.url), `invalid ${name} URL`);
  }
  const replacements = {
    SOURCE_COMMIT: commit, VERSION: pkg.version, REVISION_LINE: revision === 0 ? "" : `  revision ${revision}\n`,
    NODE_URL: toolchains.node.url, NODE_SHA256: toolchains.node.sha256,
    RUST_URL: toolchains.rust.url, RUST_SHA256: toolchains.rust.sha256,
  };
  const template = fs.readFileSync(path.join(root, "packaging/homebrew/alloy.rb.in"), "utf8");
  return template.replace(/@@([A-Z0-9_]+)@@/g, (_, key) => {
    assert(Object.hasOwn(replacements, key), `unknown template field ${key}`);
    return replacements[key];
  });
}
export function main(args = process.argv.slice(2)) {
  const options = {};
  for (let i = 0; i < args.length; i++) {
    const key = args[i];
    assert(["--commit", "--revision", "--output"].includes(key), `unknown argument ${key}`);
    assert(options[key] === undefined, `duplicate ${key}`);
    const value = args[++i];
    assert(value && !value.startsWith("--"), `${key} requires a value`);
    options[key] = value;
  }
  const formula = renderFormula({ commit: options["--commit"], revision: Number(options["--revision"] ?? 0) });
  if (options["--output"]) fs.writeFileSync(path.resolve(options["--output"]), formula);
  else process.stdout.write(formula);
}
if (process.argv[1] && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(`Homebrew formula: ${error.message}`); process.exitCode = 1; }
}
