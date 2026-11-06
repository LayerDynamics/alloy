import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { collectDoctorReport, renderDoctorReport, MonitorReadiness, MONITOR_READINESS_REFRESH_MS } from "../src/doctor.js";
import { defaultSettings, type AlloySettings } from "../src/settings.js";
import type { ModelInspection } from "../src/model.js";
import { renderMonitorSnapshot, monitorTextWidth } from "../src/monitor-render.js";
import type { MonitorSnapshot } from "../src/monitor-types.js";
import { resolveDaemonLaunch } from "../src/daemon.js";

type Checks = NonNullable<Parameters<typeof collectDoctorReport>[2]>;
const validModel = (directory: string): ModelInspection => ({ ok: true, path: directory, errors: [], verified: false });
const configured = (): AlloySettings => ({ ...defaultSettings("/tmp/doctor-home"), artifact: "/models/starter" });

function checks(overrides: Partial<Checks> = {}): Checks {
  return {
    inspectModel: async (directory) => validModel(directory),
    probeDaemon: async () => ({ status: "unavailable", error: "connect ECONNREFUSED /tmp/alloyd.sock", daemon_status: null, can_start: true }),
    resolveDaemonLaunch: async () => ({ kind: "source", command: "/usr/bin/cargo", cwd: "/work/alloy" }),
    ...overrides,
  };
}

test("missing configuration recommends guided start without inspecting weights or executing a helper", async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-doctor-"));
  try {
    const marker = path.join(temp, "executed");
    const binary = path.join(temp, "alloyd");
    fs.writeFileSync(binary, `#!/bin/sh\ntouch '${marker}'\n`, { mode: 0o755 });
    const settings = { ...defaultSettings(temp), alloyd_bin: binary };
    const report = await collectDoctorReport(settings, {}, checks({
      inspectModel: async () => { throw new Error("missing artifact must not be inspected"); },
      resolveDaemonLaunch,
    }));
    assert.equal(report.configuration.status, "missing");
    assert.equal(report.model.status, "not_configured");
    assert.equal(report.next_command, "alloy start");
    assert.match(report.next_step, /guided setup/);
    assert.equal(report.launch.kind, "binary");
    assert.equal(fs.existsSync(marker), false, "doctor must not execute the discovered binary");
    assert.equal(fs.existsSync(path.join(temp, ".alloy")), false, "doctor must not create configuration or runtime directories");
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test("source availability is a launch plan, not proof of a successful native build or generation", async () => {
  let inspections = 0;
  const report = await collectDoctorReport(configured(), {}, checks({ inspectModel: async (directory, options) => {
    inspections++;
    assert.equal(options?.verifyHashes, false);
    return validModel(directory);
  } }));
  assert.equal(inspections, 1);
  assert.equal(report.readiness, "start_required");
  assert.equal(report.model.status, "preflight_passed");
  assert.equal(report.model.hashes_verified, false);
  assert.equal(report.next_command, "alloy start");
  const text = renderDoctorReport(report);
  assert.match(text, /source checkout \/work\/alloy/);
  assert.match(text, /first start will build alloyd/);
  assert.match(text, /compilation have not been tested/);
  assert.match(text, /generation, and M9 acceptance are not verified/);
});

test("an existing compatible daemon can be ready even without a local launch executable", async () => {
  const report = await collectDoctorReport(configured(), {}, checks({
    probeDaemon: async () => ({ status: "ready", error: null, can_start: false,
      daemon_status: { api_version: 1, models_resident: 0, memory_reserved: 0, sessions_active: 0 } }),
    resolveDaemonLaunch: async () => ({ kind: "unavailable", error: "alloyd executable not found" }),
  }));
  assert.equal(report.readiness, "ready");
  assert.equal(report.next_command, "alloy chat");
  assert.match(report.validation_scope, /not verified/);
  assert.equal(report.launch.kind, "unavailable");
});

test("a compatible live daemon breaks the missing-local-model setup/start loop", async () => {
  const report = await collectDoctorReport(defaultSettings("/tmp/unconfigured-home"), {}, checks({
    inspectModel: async () => { throw new Error("no local model is configured"); },
    probeDaemon: async () => ({ status: "ready", error: null, can_start: false,
      daemon_status: { api_version: 1, models_resident: 1, memory_reserved: 1024, sessions_active: 0 } }),
    resolveDaemonLaunch: async () => ({ kind: "unavailable", error: "cargo and alloyd are not installed locally" }),
  }));
  assert.equal(report.readiness, "ready");
  assert.equal(report.next_command, "alloy chat");
  assert.equal(report.configuration.status, "missing");
  assert.equal(report.model.status, "not_configured");
  assert.match(report.next_step, /compatible daemon is already running/);
});

test("unsupported profiles require the exact supported-profile setup command even with a ready daemon", async () => {
  const settings = { ...configured(), profile: "unsupported-profile" };
  const report = await collectDoctorReport(settings, {}, checks({
    inspectModel: async () => { throw new Error("unsupported profiles must not inspect a model"); },
    probeDaemon: async () => ({ status: "ready", error: null, can_start: false,
      daemon_status: { api_version: 1, models_resident: 1, memory_reserved: 1024, sessions_active: 0 } }),
  }));
  assert.equal(report.readiness, "setup_required");
  assert.equal(report.next_command, "alloy setup --profile qwen3-4b-4bit");
  assert.equal(report.configuration.status, "invalid");
  assert.equal(report.model.status, "not_checked");
  assert.match(report.configuration.error!, /Unsupported profile unsupported-profile/);
  const advice = new MonitorReadiness(settings, { reloadSettings: () => null }, async () => {
    throw new Error("unsupported profiles must not inspect a model");
  });
  assert.match((await advice.refresh("ECONNREFUSED")).nextStep, /^Next: alloy setup --profile qwen3-4b-4bit/);
});

test("socket and launch prerequisites take precedence over provisioning a missing model", async () => {
  const settings = defaultSettings("/tmp/unconfigured-home");
  const inaccessible = await collectDoctorReport(settings, {}, checks({
    probeDaemon: async () => ({ status: "unavailable", error: "connect EACCES", daemon_status: null, can_start: false }),
  }));
  assert.equal(inaccessible.readiness, "blocked");
  assert.equal(inaccessible.next_command, "alloy doctor");
  const incompatible = await collectDoctorReport(settings, {}, checks({
    probeDaemon: async () => ({ status: "incompatible", error: "unsupported API version", daemon_status: null, can_start: false }),
  }));
  assert.equal(incompatible.readiness, "blocked");
  assert.match(incompatible.next_step, /Stop the incompatible daemon/);
  const unavailableBuild = await collectDoctorReport(settings, {}, checks({
    resolveDaemonLaunch: async () => ({ kind: "unavailable", error: "Cargo not found; install Rust" }),
  }));
  assert.equal(unavailableBuild.readiness, "blocked");
  assert.equal(unavailableBuild.next_command, "alloy doctor");
  assert.match(unavailableBuild.next_step, /before provisioning a model: Cargo not found/);
  const advice = new MonitorReadiness(settings, { reloadSettings: () => null });
  assert.match((await advice.refresh("connect EACCES")).nextStep, /^Next: alloy doctor/);
});

test("invalid configuration and invalid model paths retain their reason and direct users to setup", async () => {
  const invalidConfig = await collectDoctorReport(configured(), { configurationError: "settings.json: unsupported version 2" }, checks({
    inspectModel: async () => { throw new Error("must not inspect with invalid configuration"); },
  }));
  assert.equal(invalidConfig.configuration.status, "invalid");
  assert.equal(invalidConfig.model.status, "not_checked");
  assert.equal(invalidConfig.next_command, "alloy setup");
  assert.match(renderDoctorReport(invalidConfig), /unsupported version 2/);
  const invalidModel = await collectDoctorReport(configured(), {}, checks({
    inspectModel: async (directory) => ({ ok: false, path: directory, verified: false, errors: ["missing tokenizer.json"] }),
  }));
  assert.equal(invalidModel.next_command, "alloy setup");
  assert.match(renderDoctorReport(invalidModel), /missing tokenizer\.json/);
});

test("incompatible daemons and socket access failures are reported without suggesting another blind launch", async () => {
  const incompatible = await collectDoctorReport(configured(), {}, checks({
    probeDaemon: async () => ({ status: "incompatible", error: "unsupported API version 2", daemon_status: null, can_start: false }),
  }));
  assert.equal(incompatible.readiness, "blocked");
  assert.match(incompatible.next_step, /Stop the incompatible daemon, update alloy and alloyd together/);
  const inaccessible = await collectDoctorReport(configured(), {}, checks({
    probeDaemon: async () => ({ status: "unavailable", error: "connect EACCES /private/alloyd.sock", daemon_status: null, can_start: false }),
  }));
  assert.equal(inaccessible.next_command, "alloy doctor");
  assert.match(renderDoctorReport(inaccessible), /EACCES \/private\/alloyd\.sock/);
});

test("monitor refreshes setup advice at a bounded interval and picks up settings saved elsewhere", async () => {
  let saved: AlloySettings | null = null;
  let reads = 0;
  let inspections = 0;
  const advice = new MonitorReadiness(defaultSettings("/tmp/monitor-home"), {
    reloadSettings: () => { reads++; return saved; },
  }, async (directory, options) => { inspections++; assert.equal(options?.verifyHashes, false); return validModel(directory); });
  assert.equal((await advice.refresh("connect ECONNREFUSED", { now: 0 })).nextStep, "Next: alloy start (guided setup)");
  assert.equal(inspections, 0);
  saved = { ...configured(), socket: "/tmp/newly-configured.sock" };
  assert.match((await advice.refresh("connect ECONNREFUSED", { now: 100 })).nextStep, /guided setup/);
  assert.equal(reads, 1);
  const refreshed = await advice.refresh("connect ECONNREFUSED", { now: MONITOR_READINESS_REFRESH_MS });
  assert.equal(reads, 2);
  assert.equal(inspections, 1);
  assert.equal(refreshed.settings.socket, saved.socket);
  assert.equal(refreshed.nextStep, "Next: alloy start");
});

test("monitor keeps explicit socket overrides and can recover from a malformed settings file", async () => {
  let malformed = true;
  const advice = new MonitorReadiness(configured(), {
    socket: "/tmp/explicit.sock",
    configurationError: "invalid settings JSON",
    reloadSettings: () => { if (malformed) throw new Error("invalid settings JSON"); return configured(); },
  }, async (directory) => validModel(directory));
  assert.match((await advice.refresh("ECONNREFUSED", { now: 0 })).nextStep, /Next: alloy setup.*invalid settings JSON/);
  malformed = false;
  const refreshed = await advice.refresh("ECONNREFUSED", { now: MONITOR_READINESS_REFRESH_MS });
  assert.equal(refreshed.settings.socket, "/tmp/explicit.sock");
  assert.equal(refreshed.nextStep, "Next: alloy start --socket '/tmp/explicit.sock'");
});

test("monitor reports invalid models and incompatible metrics while retaining transport-specific advice", async () => {
  const advice = new MonitorReadiness(configured(), { reloadSettings: () => null }, async (directory) => ({
    ok: false, path: directory, errors: ["missing model.safetensors"], verified: false,
  }));
  assert.match((await advice.refresh("ECONNREFUSED", { now: 0 })).nextStep, /alloy setup.*missing model\.safetensors/);
  assert.match((await advice.refresh("unsupported daemon.metrics schema", { now: 100 })).nextStep, /update alloy\/alloyd, stop the incompatible daemon/);
  const valid = new MonitorReadiness(configured(), { reloadSettings: () => null }, async (directory) => validModel(directory));
  assert.match((await valid.refresh("connect EACCES", { now: 0 })).nextStep, /alloy doctor.*socket access/);
});

test("cancelling monitor interrupts local preflight even if a filesystem operation has not completed", async () => {
  const abort = new AbortController();
  let started = false;
  const advice = new MonitorReadiness(configured(), { reloadSettings: () => null }, async () => {
    started = true;
    return await new Promise<ModelInspection>(() => {});
  });
  const pending = advice.refresh("ECONNREFUSED", { signal: abort.signal });
  await Promise.resolve();
  assert.equal(started, true);
  abort.abort();
  await assert.rejects(pending, /abort/i);
});

test("actionable monitor hints fit normal, short and narrow layouts alongside the actual transport error", () => {
  const sample: MonitorSnapshot = {
    schema_version: 1, sampled_at_unix_ms: Date.now(), host: null, host_sample_age_ms: null, host_error: null,
    alloy: { status: "unavailable", sample_age_ms: null, metrics: null, error: "connect ECONNREFUSED /tmp/alloyd.sock" },
  };
  for (const [columns, rows] of [[90, 17], [59, 23], [80, 9], [80, 23], [120, 23]] as const) {
    const text = renderMonitorSnapshot(sample, { columns, rows, nextStep: "Next: alloy start (guided setup)" });
    assert.match(text, /Next: alloy start \(guided setup\)/);
    assert.match(text, /ECONNREFUSED/);
    assert.match(text, /q quit/);
    assert.ok(text.split("\n").length <= rows);
    assert.ok(text.split("\n").every((row) => monitorTextWidth(row) <= columns));
  }
});

test("setup advice cannot hide emergency cancellation or cached-source freshness in a short terminal", () => {
  const sample: MonitorSnapshot = {
    schema_version: 1, sampled_at_unix_ms: Date.now(), host: null, host_sample_age_ms: null, host_error: null,
    alloy: { status: "unavailable", sample_age_ms: 5100, error: "connect ECONNREFUSED", metrics: {
      schema_version: 1, sampled_at_unix_ms: Date.now() - 5100, process_id: 42,
      admission: { budget_bytes: 100, reserved_bytes: 125, available_bytes: 0, active_reservations: 1, admissions_paused: true },
      execution: { active_request_id: "request-a", cancellation_requested: true },
      residency: { worker_alive: true, sample_age_ms: 100, models_resident: 0, models: [] },
      memory_guard: { tier: "balanced", pressure: "emergency", emergency: true, usage_bytes: 125,
        static_ceiling_bytes: null, dynamic_ceiling_bytes: null, metal_ceiling_bytes: null, hard_limit_bytes: 100,
        soft_bytes: null, hard_bytes: null, abort_limit_bytes: null, admit_line_bytes: null, binding: "dynamic", sample_age_ms: 100,
        last_error: null, cache_reclaim_requested: true },
    } },
  };
  for (const rows of [9, 23]) {
    const text = renderMonitorSnapshot(sample, { columns: 80, rows, nextStep: "Next: alloy start" });
    for (const label of ["EMERGENCY", "PAUSED", "cleanup pending", "5.1s old", "Next: alloy start", "ECONNREFUSED", "q quit"]) {
      assert.ok(text.includes(label), `${rows} rows: ${label}`);
    }
    assert.ok(text.split("\n").length <= rows);
  }
});
