import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  alloyHome,
  defaultSettings,
  loadSettings,
  saveSettings,
  settingsPath,
} from "../src/settings.js";

test("defaultSettings uses ~/.alloy paths", () => {
  const settings = defaultSettings("/tmp/home");
  assert.equal(settings.profile, "qwen3-4b-4bit");
  assert.equal(settings.socket, path.join("/tmp/home", ".alloy", "alloyd.sock"));
  assert.equal(settings.db, path.join("/tmp/home", ".alloy", "sessions.sqlite"));
});

test("saveSettings round-trips", () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-settings-"));
  const settings = defaultSettings(home);
  settings.artifact = "/models/qwen";
  saveSettings(settings, home);
  const loaded = loadSettings(home);
  assert.ok(loaded);
  assert.equal(loaded.artifact, "/models/qwen");
  assert.equal(settingsPath(home), path.join(home, ".alloy", "settings.json"));
  assert.ok(fs.existsSync(alloyHome(home)));
});

test("loadSettings returns null when missing", () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-empty-"));
  assert.equal(loadSettings(home), null);
});

test("ALLOY_CONFIG_DIR consistently isolates settings and default state without changing explicit paths", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-config-dir-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const home = path.join(root, "user-home");
  const directory = path.join(root, "configuration with spaces");
  const env = { ALLOY_CONFIG_DIR: directory };
  assert.equal(alloyHome(home, env), directory);
  assert.equal(settingsPath(home, env), path.join(directory, "settings.json"));
  assert.equal(loadSettings(home, env), null);
  const defaults = defaultSettings(home, env);
  assert.equal(defaults.socket, path.join(directory, "alloyd.sock"));
  assert.equal(defaults.db, path.join(directory, "sessions.sqlite"));
  const explicit = { ...defaults, socket: path.join(root, "explicit.sock"), db: path.join(root, "explicit.sqlite"),
    artifact: "/models/verified", alloyd_bin: "/custom/alloyd", corpus_models: "/custom/pins.json" };
  saveSettings(explicit, home, env);
  assert.deepEqual(loadSettings(home, env), explicit);
  assert.equal(fs.statSync(settingsPath(home, env)).mode & 0o777, 0o600);
  assert.equal(fs.existsSync(home), false, "configuration isolation must not create or change the user home");
  assert.equal(alloyHome(home, {}), path.join(home, ".alloy"));
});

test("ALLOY_CONFIG_DIR resolves home and relative paths while rejecting malformed values", () => {
  const home = path.resolve("/tmp/alloy-config-home");
  assert.equal(alloyHome(home, { ALLOY_CONFIG_DIR: "~/custom alloy" }), path.join(home, "custom alloy"));
  assert.equal(alloyHome(home, { ALLOY_CONFIG_DIR: "./custom-config" }), path.resolve("custom-config"));
  for (const value of ["", "  ", "bad\npath", "bad\u0000path", "bad\u007fpath"]) {
    assert.throws(() => defaultSettings(home, { ALLOY_CONFIG_DIR: value }), /ALLOY_CONFIG_DIR/);
  }
});

test("malformed configuration reports its path and recovery action without rewriting it", () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-invalid-settings-"));
  try {
    fs.mkdirSync(alloyHome(home));
    for (const content of ["{bad json", "null", JSON.stringify({ ...defaultSettings(home), version: 2 }),
      JSON.stringify({ ...defaultSettings(home), socket: 42 }),
      JSON.stringify({ ...defaultSettings(home), artifact: "bad\u0000path" })]) {
      fs.writeFileSync(settingsPath(home), content);
      assert.throws(() => loadSettings(home), (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.ok(error.message.includes(settingsPath(home)));
        assert.match(error.message, /alloy setup/);
        return true;
      });
      assert.equal(fs.readFileSync(settingsPath(home), "utf8"), content);
    }
  } finally { fs.rmSync(home, { recursive: true, force: true }); }
});

test("legacy optional corpus setting is normalized while future fields survive", () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-legacy-settings-"));
  try {
    const { corpus_models: _pins, ...legacy } = defaultSettings(home);
    fs.mkdirSync(alloyHome(home));
    fs.writeFileSync(settingsPath(home), JSON.stringify({ ...legacy, future_setting: "preserved" }));
    const loaded = loadSettings(home)!;
    assert.equal(loaded.corpus_models, null);
    saveSettings(loaded, home);
    assert.equal(JSON.parse(fs.readFileSync(settingsPath(home), "utf8")).future_setting, "preserved");
    assert.equal(fs.statSync(settingsPath(home)).mode & 0o777, 0o600);
  } finally { fs.rmSync(home, { recursive: true, force: true }); }
});

test("a failed atomic save preserves the previous configuration and removes its temporary file", (t) => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-save-failure-"));
  try {
    const initial = { ...defaultSettings(home), artifact: "/original/model" };
    saveSettings(initial, home);
    const before = fs.readFileSync(settingsPath(home), "utf8");
    t.mock.method(fs, "renameSync", () => { throw new Error("simulated publication failure"); });
    assert.throws(() => saveSettings({ ...initial, artifact: "/replacement/model" }, home), /publication failure/);
    assert.equal(fs.readFileSync(settingsPath(home), "utf8"), before);
    assert.deepEqual(fs.readdirSync(alloyHome(home)), ["settings.json"]);
  } finally { fs.rmSync(home, { recursive: true, force: true }); }
});
