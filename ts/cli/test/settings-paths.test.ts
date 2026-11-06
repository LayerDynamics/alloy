import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { defaultSettings, loadSettings, normalizeSettingsPaths, resolveSettingsPath, type AlloySettings } from "../src/settings.js";
import { runSetup } from "../src/setup.js";
import { runCli, type CliDependencies } from "../src/main.js";
import { resolveDaemonLaunch } from "../src/daemon.js";
import { collectDoctorReport, MonitorReadiness, MONITOR_READINESS_REFRESH_MS } from "../src/doctor.js";

function shorthandSettings(): AlloySettings {
  return { ...defaultSettings(), socket: "~/.alloy/custom.sock", db: "~/state/sessions.sqlite",
    artifact: "~/models/starter", alloyd_bin: "~/bin/custom-alloyd", corpus_models: "~/pins/models.json" };
}

test("shared path normalization expands home and relative paths without changing PATH executable names or missing models", () => {
  const testHome = path.resolve("/tmp/alloy-path-home");
  const original = shorthandSettings();
  const normalized = normalizeSettingsPaths(original, testHome);
  assert.equal(normalized.socket, path.join(testHome, ".alloy/custom.sock"));
  assert.equal(normalized.db, path.join(testHome, "state/sessions.sqlite"));
  assert.equal(normalized.artifact, path.join(testHome, "models/starter"));
  assert.equal(normalized.alloyd_bin, path.join(testHome, "bin/custom-alloyd"));
  assert.equal(normalized.corpus_models, path.join(testHome, "pins/models.json"));
  assert.equal(original.socket, "~/.alloy/custom.sock");
  assert.equal(resolveSettingsPath("~", testHome), testHome);
  assert.equal(resolveSettingsPath("./relative.sock", testHome), path.resolve("relative.sock"));
  const unconfigured = normalizeSettingsPaths({ ...defaultSettings(testHome), alloyd_bin: "custom-alloyd" }, testHome);
  assert.equal(unconfigured.artifact, "");
  assert.equal(unconfigured.corpus_models, null);
  assert.equal(unconfigured.alloyd_bin, "custom-alloyd");
  assert.deepEqual(normalizeSettingsPaths(normalized, testHome), normalized);
});

test("setup saves the same normalized paths used by prechecks with an injected home", async (t) => {
  const testHome = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-path-setup-"));
  t.after(() => fs.rmSync(testHome, { recursive: true, force: true }));
  const supplied = { ...shorthandSettings(), corpus_models: "~/pins.json" };
  fs.writeFileSync(path.join(testHome, "pins.json"), "{}");
  const expected = normalizeSettingsPaths(supplied, testHome);
  const configured = await runSetup(supplied, testHome, {
    interactive: false, report: () => {},
    inspect: async (directory, flags) => {
      assert.equal(directory, expected.artifact);
      assert.equal(flags?.verifyHashes, true);
      return { ok: true, path: directory, verified: true, errors: [] };
    },
    discover: async () => { throw new Error("unexpected model discovery"); },
    install: async () => { throw new Error("unexpected model download"); },
  });
  assert.deepEqual(configured, expected);
  assert.deepEqual(loadSettings(testHome), expected);
});

test("start expands quoted-tilde socket and executable overrides before any launch precheck", async () => {
  // Use an existing executable only for read-only resolution. Nothing executes it.
  const executable = "~/" + path.relative(os.homedir(), process.execPath);
  const socket = "~/.alloy/path-precheck.sock";
  const expectedSocket = path.join(os.homedir(), ".alloy/path-precheck.sock");
  const events: string[] = [];
  let selected: AlloySettings | undefined;
  const code = await runCli(["node", "alloy", "start", "--socket", socket, "--alloyd-bin", executable], {
    loadSettings: () => defaultSettings(),
    probeDaemon: async (candidate) => {
      events.push("probe");
      assert.equal(candidate, expectedSocket);
      return { status: "unavailable", can_start: true, error: "connect ENOENT", daemon_status: null };
    },
    resolveDaemonLaunch: async (settings) => {
      events.push("resolve");
      assert.equal(settings.socket, expectedSocket);
      assert.equal(settings.alloyd_bin, process.execPath);
      const plan = await resolveDaemonLaunch(settings, { sourceRoot: null });
      assert.equal(plan.kind, "binary");
      selected = { ...settings, artifact: "/already-verified-model" };
      return plan;
    },
    runSetup: async () => { events.push("setup"); return selected!; },
    startDaemon: async (settings) => { events.push("start"); assert.equal(settings.socket, expectedSocket); return 0; },
  });
  assert.equal(code, 0);
  assert.deepEqual(events, ["probe", "resolve", "setup", "start"]);
});

test("doctor checks and reports normalized paths even when called directly", async () => {
  const settings = shorthandSettings();
  const expected = normalizeSettingsPaths(settings);
  const report = await collectDoctorReport(settings, {}, {
    inspectModel: async (directory) => {
      assert.equal(directory, expected.artifact);
      return { ok: true, path: directory, verified: false, errors: [] };
    },
    probeDaemon: async (socket) => {
      assert.equal(socket, expected.socket);
      return { status: "unavailable", can_start: true, error: "connect ENOENT", daemon_status: null };
    },
    resolveDaemonLaunch: async (candidate) => {
      assert.deepEqual(candidate, expected);
      return { kind: "binary", command: candidate.alloyd_bin };
    },
  });
  assert.equal(report.readiness, "start_required");
  assert.equal(report.model.path, expected.artifact);
  assert.equal(report.daemon.socket, expected.socket);
  assert.equal(settings.socket, "~/.alloy/custom.sock");
});

test("read-only CLI commands normalize explicit sockets while preserving malformed-settings recovery", async () => {
  for (const command of ["doctor", "monitor"]) {
    const socket = "~/.alloy/read-only.sock";
    const expected = path.join(os.homedir(), ".alloy/read-only.sock");
    let called = false;
    const deps: Partial<CliDependencies> = {
      loadSettings: () => { throw new Error("malformed settings"); },
      runSetup: async () => { throw new Error("unexpected setup"); },
      startDaemon: async () => { throw new Error("unexpected daemon launch"); },
    };
    if (command === "doctor") deps.runDoctor = async (settings, options) => {
      called = true;
      assert.equal(settings.socket, expected);
      assert.match(options!.configurationError!, /malformed settings/);
      return 0;
    };
    else deps.runMonitor = async (settings, _args, options) => {
      called = true;
      assert.equal(settings.socket, expected);
      assert.match(options!.configurationError!, /malformed settings/);
      return 0;
    };
    assert.equal(await runCli(["node", "alloy", command, "--socket", socket], deps), 0);
    assert.equal(called, true);
  }
});

test("disconnected monitor reloads normalized paths and retains a normalized explicit socket override", async () => {
  for (const explicit of [false, true]) {
    let saved = shorthandSettings();
    const inspected: string[] = [];
    const override = "~/.alloy/explicit.sock";
    const advice = new MonitorReadiness(defaultSettings(), {
      reloadSettings: () => saved,
      ...(explicit ? { socket: override } : {}),
    }, async (directory) => {
      inspected.push(directory);
      return { ok: true, path: directory, errors: [], verified: false };
    });
    const first = await advice.refresh("connect ENOENT", { now: 0 });
    assert.equal(first.settings.socket, resolveSettingsPath(explicit ? override : saved.socket));
    saved = { ...saved, socket: "~/.alloy/reconfigured.sock", artifact: "~/models/reconfigured", alloyd_bin: "~/bin/reconfigured" };
    const refreshed = await advice.refresh("connect ENOENT", { now: MONITOR_READINESS_REFRESH_MS });
    assert.equal(refreshed.settings.socket, resolveSettingsPath(explicit ? override : saved.socket));
    assert.equal(refreshed.settings.alloyd_bin, resolveSettingsPath(saved.alloyd_bin));
    assert.deepEqual(inspected, [resolveSettingsPath("~/models/starter"), resolveSettingsPath("~/models/reconfigured")]);
    if (explicit) assert.equal(refreshed.nextStep, `Next: alloy start --socket '${resolveSettingsPath(override)}'`);
    assert.equal(saved.socket, "~/.alloy/reconfigured.sock", "normalization must not rewrite the loaded object");
  }
});
