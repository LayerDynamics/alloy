import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const packageRoot = fileURLToPath(new URL("../../", import.meta.url));

function installedPackage(t: TestContext) {
  // Node resolves imported module URLs through macOS's /var -> /private/var alias.
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "alloy-installed-")));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const installed = path.join(root, "installed package");
  const cwd = path.join(root, "unrelated working directory");
  const tools = path.join(root, "tools");
  const config = path.join(root, "config");
  for (const directory of [cwd, tools, path.join(installed, "bin"), path.join(installed, "dist/src"), path.join(installed, "models")]) {
    fs.mkdirSync(directory, { recursive: true });
  }
  for (const filename of fs.readdirSync(path.join(packageRoot, "dist/src"))) {
    if (filename.endsWith(".js")) fs.copyFileSync(path.join(packageRoot, "dist/src", filename), path.join(installed, "dist/src", filename));
  }
  for (const filename of ["package.json", "bin/alloy.js", "models/starter-model.json"]) {
    fs.copyFileSync(path.join(packageRoot, filename), path.join(installed, filename));
  }
  // The fixture deliberately contains only installed runtime files, with no
  // source checkout, build scripts, node_modules, npm, or Cargo in PATH.
  const env: NodeJS.ProcessEnv = { ...process.env, PATH: tools, ALLOY_CONFIG_DIR: config,
    ALLOYD: undefined, NODE_PATH: undefined, NODE_OPTIONS: undefined };
  const node = (args: string[], overrides: NodeJS.ProcessEnv = {}) => {
    const result = spawnSync(process.execPath, args, { cwd, env: { ...env, ...overrides }, encoding: "utf8", timeout: 8000, maxBuffer: 1024 * 1024 });
    assert.ifError(result.error);
    assert.equal(result.signal, null, result.stderr);
    return result;
  };
  const cli = (args: string[], overrides: NodeJS.ProcessEnv = {}) => node([path.join(installed, "bin/alloy.js"), ...args], overrides);
  const moduleUrl = (name: string) => JSON.stringify(pathToFileURL(path.join(installed, "dist/src", `${name}.js`)).href);
  const executable = (name: string) => {
    const filename = path.join(tools, name);
    fs.writeFileSync(filename, `#!${process.execPath}\nrequire('node:fs').writeFileSync(${JSON.stringify(path.join(root, "unexpected-launch"))}, 'executed');\nprocess.exit(29);\n`, { mode: 0o755 });
    return filename;
  };
  return { root, installed, cwd, config, cli, node, moduleUrl, executable };
}

test("installed entry resolves packaged version and help without settings or build tools", (t) => {
  const fixture = installedPackage(t);
  const metadataPath = path.join(fixture.installed, "package.json");
  const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
  metadata.version = "0.0.0-installed.fixture";
  fs.writeFileSync(metadataPath, JSON.stringify(metadata));
  for (const flag of ["--version", "-V"]) {
    const result = fixture.cli([flag], { ALLOY_CONFIG_DIR: "invalid\nconfiguration" });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, "alloy 0.0.0-installed.fixture\n");
    assert.equal(result.stderr, "");
  }
  const help = fixture.cli(["--help"], { ALLOY_CONFIG_DIR: "invalid\nconfiguration" });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /alloy start/);
  assert.match(help.stdout, /ALLOY_CONFIG_DIR/);
  assert.match(help.stdout, /--version/);
  assert.equal(fs.existsSync(fixture.config), false);
  assert.equal(fs.existsSync(path.join(fixture.installed, "node_modules")), false);
  assert.equal(fs.existsSync(path.join(fixture.installed, "src")), false);
  metadata.version = "not-a-version";
  fs.writeFileSync(metadataPath, JSON.stringify(metadata));
  const invalid = fixture.cli(["--version"]);
  assert.equal(invalid.status, 1);
  assert.match(invalid.stderr, /package metadata has no valid version/);
});

test("installed modules find model pins and helper beside the package and honor isolated setup paths", (t) => {
  const fixture = installedPackage(t);
  const socket = path.join(fixture.root, "explicit.sock");
  const database = path.join(fixture.root, "explicit.sqlite");
  const result = fixture.node(["--input-type=module", "-e", `
    import { STARTER_MODEL, STARTER_CORPUS_PATH } from ${fixture.moduleUrl("model")};
    import { NATIVE_METRICS_HELPER } from ${fixture.moduleUrl("powermetrics")};
    import { defaultSettings, loadSettings, settingsPath } from ${fixture.moduleUrl("settings")};
    import { runSetup } from ${fixture.moduleUrl("setup")};
    const defaults = defaultSettings();
    const settings = await runSetup({ artifact: "/verified/model", socket: ${JSON.stringify(socket)}, db: ${JSON.stringify(database)} }, undefined, {
      interactive: false, report: () => {},
      inspect: async (directory) => ({ ok: true, path: directory, verified: true, errors: [] }),
      discover: async () => { throw new Error("unexpected discovery"); },
      install: async () => { throw new Error("unexpected download"); },
    });
    console.log(JSON.stringify({ corpus: STARTER_CORPUS_PATH, profile: STARTER_MODEL.profile,
      helper: NATIVE_METRICS_HELPER, defaults, file: settingsPath(), settings, loaded: loadSettings() }));
  `]);
  assert.equal(result.status, 0, result.stderr);
  const output = JSON.parse(result.stdout);
  assert.equal(output.corpus, path.join(fixture.installed, "models/starter-model.json"));
  assert.equal(output.profile, "qwen3-4b-4bit");
  assert.equal(output.helper, path.join(fixture.installed, "dist/native/alloy-metrics"));
  assert.equal(output.defaults.socket, path.join(fixture.config, "alloyd.sock"));
  assert.equal(output.defaults.db, path.join(fixture.config, "sessions.sqlite"));
  assert.equal(output.file, path.join(fixture.config, "settings.json"));
  assert.equal(output.settings.socket, socket);
  assert.equal(output.settings.db, database);
  assert.deepEqual(output.loaded, output.settings);
  assert.equal(fs.statSync(output.file).mode & 0o777, 0o600);
});

test("installed doctor resolves binaries without building and preserves CLI, saved, and environment precedence", (t) => {
  const fixture = installedPackage(t);
  const environmentBinary = fixture.executable("environment-daemon");
  const savedBinary = fixture.executable("saved-daemon");
  const commandBinary = fixture.executable("command-daemon");
  const doctor = (args: string[] = [], env: NodeJS.ProcessEnv = {}) => {
    const result = fixture.cli(["doctor", "--json", ...args], env);
    assert.equal(result.status, 1, "an unconfigured model is not generation-ready");
    assert.equal(result.stderr, "");
    return JSON.parse(result.stdout);
  };
  const initial = doctor([], { ALLOYD: environmentBinary });
  assert.deepEqual(initial.launch, { kind: "binary", command: environmentBinary });
  assert.equal(initial.daemon.socket, path.join(fixture.config, "alloyd.sock"));
  assert.equal(fs.existsSync(fixture.config), false, "doctor must not create settings");
  fs.mkdirSync(fixture.config);
  const saved = { version: 1, profile: "qwen3-4b-4bit", artifact: "", socket: path.join(fixture.root, "saved.sock"),
    db: path.join(fixture.root, "saved.sqlite"), alloyd_bin: savedBinary, corpus_models: null };
  const content = JSON.stringify(saved);
  const settingsFile = path.join(fixture.config, "settings.json");
  fs.writeFileSync(settingsFile, content);
  assert.deepEqual(doctor([], { ALLOYD: environmentBinary }).launch, { kind: "binary", command: savedBinary });
  const explicitSocket = path.join(fixture.root, "command.sock");
  const explicit = doctor(["--alloyd-bin", commandBinary, "--socket", explicitSocket], { ALLOYD: environmentBinary });
  assert.deepEqual(explicit.launch, { kind: "binary", command: commandBinary });
  assert.equal(explicit.daemon.socket, explicitSocket);
  const missing = path.join(fixture.root, "missing-daemon");
  const refused = doctor(["--alloyd-bin", missing], { ALLOYD: environmentBinary });
  assert.equal(refused.launch.kind, "unavailable");
  assert.ok(refused.launch.error.includes(missing));
  assert.equal(fs.readFileSync(settingsFile, "utf8"), content);
  assert.equal(fs.existsSync(path.join(fixture.root, "unexpected-launch")), false);
});
