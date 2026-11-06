import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { alloyHome, defaultSettings, loadSettings, saveSettings, settingsPath, } from "../src/settings.js";
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
//# sourceMappingURL=settings.test.js.map