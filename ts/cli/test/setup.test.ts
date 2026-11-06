import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { runSetup, SetupCancelledError, type SetupOptions } from "../src/setup.js";
import { defaultSettings, loadSettings, saveSettings, settingsPath } from "../src/settings.js";

function options(): SetupOptions {
  return { interactive: false, report: () => {}, discover: async () => [],
    inspect: async (directory, flags) => {
      assert.equal(flags?.verifyHashes, true, "setup must verify pinned bytes before saving");
      return { ok: true, path: path.resolve(directory), errors: [], verified: true };
    },
    install: async () => { throw new Error("unexpected download"); },
  };
}

test("setup preserves existing valid paths and saves only after full model verification", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-setup-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  const previous = { ...defaultSettings(testHome), artifact: "/existing/model", socket: "/custom/alloy.sock", db: "/custom/sessions.sqlite", alloyd_bin: "/custom/alloyd" };
  saveSettings(previous, testHome);
  const result = await runSetup({ artifact: "~/models/qwen" }, testHome, options());
  assert.equal(result.artifact, path.join(testHome, "models/qwen"));
  assert.equal(result.socket, previous.socket);
  assert.equal(result.db, previous.db);
  assert.equal(result.alloyd_bin, previous.alloyd_bin);
  assert.deepEqual(loadSettings(testHome), result);
  assert.equal(fs.statSync(settingsPath(testHome)).mode & 0o777, 0o600);
});

test("failed or incomplete verification cannot overwrite an existing configuration", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-setup-failure-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  saveSettings({ ...defaultSettings(testHome), artifact: "/previous/model" }, testHome);
  const before = fs.readFileSync(settingsPath(testHome), "utf8");
  for (const ok of [false, true]) {
    await assert.rejects(runSetup({ artifact: "/broken/model" }, testHome, { ...options(),
      inspect: async (directory) => ({ ok, path: directory, errors: ["missing weights"], verified: false }),
    }), /not a verified starter model/);
    assert.equal(fs.readFileSync(settingsPath(testHome), "utf8"), before);
  }
});

test("first-run discovery reuses an existing verified model without downloading", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-setup-discovery-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  let verified = 0;
  const result = await runSetup({}, testHome, { ...options(),
    discover: async () => [{ ok: true, path: "/existing/corpus/model", errors: [], verified: false }],
    inspect: async (directory, flags) => {
      assert.equal(flags?.verifyHashes, true);
      verified++;
      return { ok: true, path: directory, errors: [], verified: true };
    },
  });
  assert.equal(result.artifact, "/existing/corpus/model");
  assert.equal(verified, 1);
});

test("unconfigured noninteractive setup exits with a specific action and writes nothing", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-setup-empty-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  await assert.rejects(runSetup({}, testHome, options()), /alloy setup --download/);
  assert.equal(fs.existsSync(settingsPath(testHome)), false);
});

test("only an explicit download choice installs a model; failed installs preserve settings", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-setup-install-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  let installed = 0;
  const result = await runSetup({}, testHome, { ...options(), interactive: true, ask: async () => "download",
    install: async (_destination, flags) => {
      installed++;
      assert.equal(flags?.home, testHome);
      return { ok: true, path: path.join(testHome, "models/verified"), errors: [], verified: true };
    },
  });
  assert.equal(installed, 1);
  const before = fs.readFileSync(settingsPath(testHome), "utf8");
  await assert.rejects(runSetup({}, testHome, { ...options(), download: true,
    install: async () => { throw new Error("hash mismatch"); },
  }), /hash mismatch/);
  assert.equal(fs.readFileSync(settingsPath(testHome), "utf8"), before);
  assert.equal(loadSettings(testHome)?.artifact, result.artifact);
});

test("cancelled verification preserves configuration and releases signal handlers", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-setup-cancel-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  const before = [process.listenerCount("SIGINT"), process.listenerCount("SIGTERM")];
  const controller = new AbortController();
  await assert.rejects(runSetup({ artifact: "/candidate" }, testHome, { ...options(), signal: controller.signal,
    inspect: async (directory) => {
      controller.abort();
      return { ok: true, path: directory, errors: [], verified: true };
    },
  }), SetupCancelledError);
  assert.equal(fs.existsSync(settingsPath(testHome)), false);
  assert.deepEqual([process.listenerCount("SIGINT"), process.listenerCount("SIGTERM")], before);
});
