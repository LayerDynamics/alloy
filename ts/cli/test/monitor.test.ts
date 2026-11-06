import test from "node:test";
import assert from "node:assert/strict";
import { Writable } from "node:stream";
import { setTimeout as delay, setImmediate as nextTurn } from "node:timers/promises";
import { DaemonMetricsReader, collectMonitorSnapshot, renderMonitorSnapshot, writeMonitorOutput, type MetricsRpcClient } from "../src/monitor.js";
import { parseDaemonMetrics, type DaemonMetrics, type MonitorSnapshot } from "../src/monitor-types.js";
import type { HostMetricsSnapshot } from "../src/host-metrics.js";
import { emptyPowerMetrics } from "../src/powermetrics.js";
import { monitorTextWidth, renderMonitorSparkline } from "../src/monitor-render.js";
import { stripVTControlCharacters } from "node:util";

function metrics(): DaemonMetrics {
  return {
    schema_version: 1, sampled_at_unix_ms: Date.now(), process_id: 42,
    admission: { budget_bytes: 8 * 1024 ** 3, reserved_bytes: 2 * 1024 ** 3, available_bytes: 6 * 1024 ** 3, active_reservations: 2, admissions_paused: true },
    execution: { active_request_id: "request-a", cancellation_requested: true },
    residency: { worker_alive: true, sample_age_ms: 100, models_resident: 1,
      models: [{ artifact_id: "qwen-a", weights_reserved_bytes: 1024 ** 3, kv_reserved_bytes: 1024 ** 2, pinned: true, leased: true }] },
    memory_guard: { tier: "balanced", pressure: "emergency", emergency: true, usage_bytes: 7 * 1024 ** 3,
      static_ceiling_bytes: null, dynamic_ceiling_bytes: null, metal_ceiling_bytes: null, hard_limit_bytes: 8 * 1024 ** 3,
      soft_bytes: null, hard_bytes: null, abort_limit_bytes: null, admit_line_bytes: null, binding: "dynamic", sample_age_ms: 1200,
      last_error: null, cache_reclaim_requested: true },
  };
}

function host(): HostMetricsSnapshot {
  return {
    sampled_at_unix_ms: Date.now(), platform: "darwin", architecture: "arm64", hostname: "test-mac",
    logical_cpu_count: 12, cpu_utilization_percent: 24.5, cpu_sample_ms: 1000, load_average: [1, 2, 3], uptime_seconds: 900,
    memory_total_bytes: 36 * 1024 ** 3, memory_free_bytes: 10 * 1024 ** 3, memory_used_bytes: 26 * 1024 ** 3,
    memory_available_bytes: null, swap_total_bytes: 4 * 1024 ** 3, swap_used_bytes: 0, swap_free_bytes: 4 * 1024 ** 3,
    wired_bytes: 1024 ** 3, compressed_bytes: 0, errors: [], accelerators: emptyPowerMetrics("disabled"),
  };
}

function snapshot(): MonitorSnapshot {
  return { schema_version: 1, sampled_at_unix_ms: Date.now(), host: host(), host_sample_age_ms: 0, host_error: null,
    alloy: { status: "connected", sample_age_ms: 0, error: null, metrics: metrics() } };
}

test("daemon metric validation preserves unknown/null readings and rejects fabricated data", () => {
  assert.equal(parseDaemonMetrics(metrics()).memory_guard.static_ceiling_bytes, null);
  assert.throws(() => parseDaemonMetrics({ schema_version: 0 }), /unsupported/);
  const broken = metrics();
  broken.admission.reserved_bytes = Number.NaN;
  assert.throws(() => parseDaemonMetrics(broken), /reserved_bytes/);
  assert.throws(() => parseDaemonMetrics({ ...metrics(), execution: {} }), /active_request_id/);
});

test("80x24 rendering keeps host, accelerators, runtime and exit instructions visible", () => {
  const sample = snapshot();
  sample.host!.errors.push("sample warning");
  sample.alloy.metrics!.memory_guard.last_error = "latest host probe failed";
  const text = renderMonitorSnapshot(sample, { columns: 80, rows: 23 });
  for (const label of ["HOST", "ACCELERATORS", "ALLOY RUNTIME", "connected", "q quit", "Host error", "Guard error", "cleanup pending"]) {
    assert.ok(text.includes(label), label);
  }
  assert.ok(text.split("\n").length <= 23);
  assert.ok(text.split("\n").every((line) => Array.from(line).length <= 80));
});

test("unavailable accelerator readings are distinct from reported zero", () => {
  const sample = snapshot();
  const unavailable = renderMonitorSnapshot(sample);
  assert.match(unavailable, /GPU active.*unavailable/);
  assert.match(unavailable, /Est\. power.*ANE unavailable/);
  sample.host!.accelerators.gpu_active_percent = 0;
  sample.host!.accelerators.ane_power_mw = 0;
  const zero = renderMonitorSnapshot(sample);
  assert.match(zero, /GPU active.*0\.0%/);
  assert.match(zero, /Est\. power.*ANE 0 mW/);
});

test("cached daemon data is visibly stale, and rendered values cannot inject terminal controls", () => {
  const sample = snapshot();
  sample.alloy.status = "unavailable";
  sample.alloy.error = "connection lost\u001b[2J\nforged header";
  sample.alloy.sample_age_ms = 5100;
  sample.alloy.metrics!.execution.active_request_id = "request\u001b[?25l";
  const text = renderMonitorSnapshot(sample);
  assert.match(text, /unavailable \(stale\).*5\.1s old/);
  assert.equal(text.includes("\u001b"), false);
});

test("responsive colored panels stay within terminal cells and preserve emergency state", () => {
  const sample = snapshot();
  sample.host!.hostname = "Mac-東京-🖥️-e\u0301".repeat(8);
  sample.alloy.metrics!.execution.active_request_id = "request-東京-\u001b[2J".repeat(20);
  sample.alloy.metrics!.residency.models[0]!.artifact_id = "model-\u001b]52;c;clipboard\u0007-東京".repeat(8);
  for (const [columns, rows] of [[60, 21], [80, 23], [108, 24], [120, 35], [50, 18], [20, 8], [1, 1]] as const) {
    const color = renderMonitorSnapshot(sample, { columns, rows, color: true });
    const text = stripVTControlCharacters(color);
    assert.ok(text.split("\n").length <= rows!, `${columns}x${rows}: height`);
    assert.ok(text.split("\n").every((row) => monitorTextWidth(row) <= columns!), `${columns}x${rows}: width`);
    // Only the renderer's SGR colors may reach the terminal, never caller escapes.
    assert.equal(color.replace(/\u001b\[[\d;]*m/g, "").includes("\u001b"), false);
    assert.equal(text, renderMonitorSnapshot(sample, { columns, rows, color: false }));
    if (columns! >= 60 && rows! >= 21) {
      for (const label of ["PAUSED", "EMERGENCY", "cleanup pending", "q quit"]) assert.ok(text.includes(label), label);
    }
  }
});

test("small terminals retain hardware freshness and failures next to cached values", () => {
  const sample = snapshot();
  Object.assign(sample.host!.accelerators, {
    status: "stale", age_ms: 30_000, gpu_active_percent: 25, ane_power_mw: 0,
    error: "native sampling failed",
  });
  const text = renderMonitorSnapshot(sample, { columns: 50, rows: 18 });
  assert.match(text, /ACCELERATORS stale.*30\.0s old/);
  assert.match(text, /GPU active.*25\.0%/);
  assert.match(text, /ANE power 0 mW/);
  assert.match(text, /native sampling failed/);
});

test("short and narrow windows retain bars and history with a refused daemon connection", () => {
  const sample = snapshot();
  sample.alloy = { status: "unavailable", sample_age_ms: null, metrics: null,
    error: "connect ECONNREFUSED /Users/ryanoboyle/.alloy/alloyd.sock" };
  Object.assign(sample.host!.accelerators, { status: "ready", age_ms: 0, gpu_active_percent: 18.7, ane_power_mw: 0 });
  const history = [{ at_unix_ms: sample.sampled_at_unix_ms, cpu_percent: 31, memory_percent: 95,
    gpu_percent: 18, ane_power_mw: 0, reserved_percent: null }];
  for (const [columns, rows] of [[90, 17], [90, 20], [59, 23]] as const) {
    const text = renderMonitorSnapshot(sample, { columns, rows, history });
    for (const label of ["CPU", "RAM non-free", "GPU active", "CPU history", "GPU history", "ANE power 0 mW", "ECONNREFUSED", "q quit"]) {
      assert.ok(text.includes(label), `${columns}x${rows}: ${label}`);
    }
    assert.match(text, /CPU\s+━/);
    assert.match(text, /GPU active\s+━/);
    assert.match(text, /CPU history.*▃/);
    assert.ok(text.split("\n").length <= rows);
    assert.ok(text.split("\n").every((row) => monitorTextWidth(row) <= columns));
  }
});

test("very short windows prioritize pressure, cancellation and freshness over optional readings", () => {
  const sample = snapshot();
  sample.alloy.status = "unavailable";
  sample.alloy.sample_age_ms = 5100;
  sample.host!.accelerators.status = "stale";
  sample.host!.accelerators.age_ms = 30_000;
  const text = renderMonitorSnapshot(sample, { columns: 80, rows: 9 });
  for (const label of ["PAUSED", "EMERGENCY", "cleanup pending", "ALLOY unavailable (stale)", "5.1s old", "ACCELERATORS stale", "30.0s old", "q quit"]) {
    assert.ok(text.includes(label), label);
  }
  assert.ok(text.split("\n").length <= 9);
});

test("history graphs preserve unknown samples, real zero, and a fixed percentage scale", () => {
  assert.equal(renderMonitorSparkline([null, 0, 50, 100], 4), "·▁▄█");
  assert.equal(renderMonitorSparkline([Number.NaN, -1, 250], 3), "··█");
  assert.equal(renderMonitorSparkline([0, 100], 0), "");
  const sample = snapshot();
  sample.alloy.metrics!.admission.reserved_bytes = 10 * 1024 ** 3;
  const text = renderMonitorSnapshot(sample, { columns: 120, rows: 35 });
  assert.match(text, /10\.0 GiB \/ 8\.0 GiB reserved/);
  assert.match(text, /RAM = non-free, including caches; not memory pressure/);
  assert.match(text, /ANE trend.*power, not occupancy/);
});

test("ANE history keeps its own power scale even when a shorter panel omits details", () => {
  const sample = snapshot();
  sample.host!.accelerators.cpu_power_mw = 5435;
  sample.host!.accelerators.ane_power_mw = 200;
  const text = renderMonitorSnapshot(sample, { columns: 120, rows: 24, history: [{
    at_unix_ms: sample.sampled_at_unix_ms, cpu_percent: 25, memory_percent: 70,
    gpu_percent: 20, ane_power_mw: 200, reserved_percent: 25,
  }] });
  assert.match(text, /ESTIMATED POWER.*full bar 5\.43 W/);
  assert.match(text, /ANE 0–200 mW.*█/);
});

test("host snapshots remain available when daemon metrics fail", async () => {
  const sample = await collectMonitorSnapshot({ sample: async () => host(), close: async () => {} }, {
    read: async () => { throw new Error("daemon offline"); },
  });
  assert.equal(sample.host?.hostname, "test-mac");
  assert.equal(sample.alloy.status, "unavailable");
  assert.match(sample.alloy.error!, /offline/);
  const captured = JSON.parse(JSON.stringify(sample)) as MonitorSnapshot;
  assert.equal(captured.host?.accelerators.gpu_active_percent, null);
});

test("failed host sample retains a dated prior sample with the new error", async () => {
  const previous = host();
  previous.sampled_at_unix_ms -= 3000;
  const sample = await collectMonitorSnapshot({ sample: async () => { throw new Error("host probe failed"); }, close: async () => {} }, {
    read: async () => ({ status: "connected", sample_age_ms: 0, error: null, metrics: metrics() }),
  }, previous);
  assert.equal(sample.host?.sampled_at_unix_ms, previous.sampled_at_unix_ms);
  assert.ok(sample.host_sample_age_ms! >= 3000);
  assert.equal(sample.host_error, "host probe failed");
});

test("publication recomputes hardware/daemon age and marks retained hardware stale", async (t) => {
  t.mock.method(Date, "now", () => 1_000_000);
  const previous = host();
  previous.sampled_at_unix_ms -= 3000;
  previous.accelerators.status = "ready";
  previous.accelerators.sampled_at_unix_ms = previous.sampled_at_unix_ms - 2000;
  previous.accelerators.age_ms = 0;
  const priorDaemon = metrics();
  priorDaemon.sampled_at_unix_ms -= 4000;
  const sample = await collectMonitorSnapshot({ sample: async () => { throw new Error("sampling failed"); }, close: async () => {} }, {
    read: async () => ({ status: "unavailable", sample_age_ms: 0, error: "offline", metrics: priorDaemon }),
  }, previous);
  assert.ok(sample.host!.accelerators.age_ms! >= 5000);
  assert.equal(sample.host!.accelerators.status, "stale");
  assert.equal(sample.host!.accelerators.error, "sampling failed");
  assert.ok(sample.alloy.sample_age_ms! >= 4000);
  const rendered = renderMonitorSnapshot(sample);
  assert.match(rendered, /Guard.*5\.2s old/);
  assert.match(rendered, /Residency 4\.1s old/);
});

test("output awaits write completion instead of accumulating frames behind a slow consumer", async () => {
  let complete!: (error?: Error | null) => void;
  const output = new Writable({ highWaterMark: 1, write(_chunk, _encoding, callback) { complete = callback; } });
  let finished = false;
  const pending = writeMonitorOutput(output, "frame\n", new AbortController().signal).then(() => { finished = true; });
  await delay(10);
  assert.equal(finished, false);
  complete();
  await pending;
  assert.equal(finished, true);
  output.destroy();
});

test("blocked output can be aborted or timed out", async () => {
  for (const cancelled of [true, false]) {
    let complete!: (error?: Error | null) => void;
    const output = new Writable({ write(_chunk, _encoding, callback) { complete = callback; } });
    const abort = new AbortController();
    const pending = writeMonitorOutput(output, "frame\n", abort.signal, 20);
    if (cancelled) abort.abort();
    await assert.rejects(pending, cancelled ? /stopped/ : /timed out/);
    complete();
    output.destroy();
  }
});

test("a closed output pipe rejects without an uncaught stream error", async () => {
  const output = new Writable({ write(_chunk, _encoding, callback) {
    callback(Object.assign(new Error("closed pipe"), { code: "EPIPE" }));
  } });
  await assert.rejects(writeMonitorOutput(output, "frame\n", new AbortController().signal), /closed pipe/);
  await nextTurn();
  output.destroy();
});

test("daemon reader only calls daemon.metrics and reconnects after failure", async () => {
  let connections = 0;
  let closed = 0;
  let calls = 0;
  const methods: string[] = [];
  const reader = new DaemonMetricsReader("/test.sock", 100, async () => {
    connections++;
    return { call: async (method) => {
      methods.push(method);
      if (++calls === 2) throw new Error("disconnected");
      return metrics();
    }, close: () => { closed++; } };
  });
  assert.equal((await reader.read()).status, "connected");
  const failed = await reader.read();
  assert.equal(failed.status, "unavailable");
  assert.ok(failed.metrics);
  assert.equal((await reader.read()).status, "connected");
  assert.equal(connections, 2);
  assert.equal(closed, 1);
  assert.deepEqual(methods, ["daemon.metrics", "daemon.metrics", "daemon.metrics"]);
  reader.close();
});

test("a blocked daemon call is bounded and its client is closed", async () => {
  let closed = false;
  const reader = new DaemonMetricsReader("/test.sock", 20, async () => ({
    call: () => new Promise(() => {}), close: () => { closed = true; },
  }));
  const result = await reader.read();
  assert.equal(result.status, "unavailable");
  assert.match(result.error!, /timed out/);
  assert.equal(closed, true);
  reader.close();
});

test("polls cannot overlap and closing cancels an in-flight read", async () => {
  let resolveCall!: (value: unknown) => void;
  const client: MetricsRpcClient = { call: () => new Promise((resolve) => { resolveCall = resolve; }), close: () => {} };
  const reader = new DaemonMetricsReader("/test.sock", 1000, async () => client);
  const first = reader.read();
  await assert.rejects(reader.read(), /overlapping/);
  // The first connection has settled by this microtask boundary.
  await Promise.resolve();
  reader.close();
  assert.equal((await first).status, "unavailable");
  resolveCall?.(metrics());
});


test("IORegistry GPU utilization remains visible when recorded power sampling is disabled or failed", () => {
  for (const powerStatus of ["disabled", "unavailable"] as const) {
    const sample = snapshot();
    sample.host!.gpu = {
      source: "ioreg", status: "ready", sampled_at_unix_ms: sample.sampled_at_unix_ms,
      age_ms: 0, utilization_percent: 37.5,
      devices: [{ id: "42", name: "Apple GPU", utilization_percent: 37.5 }], error: null,
    };
    sample.host!.accelerators = emptyPowerMetrics(powerStatus,
      powerStatus === "unavailable" ? "Native sampling unavailable" : null);
    for (const [columns, rows] of [[80, 24], [120, 35], [50, 18]] as const) {
      const text = renderMonitorSnapshot(sample, { columns, rows });
      assert.match(text, /GPU use.*37\.5%/);
      assert.match(text, /GPU ready.*0\.0s old/);
      assert.match(text, powerStatus === "disabled" ? /Power disabled/ : /Power unavailable/);
      assert.doesNotMatch(text, /ACCELERATORS unavailable|Hardware.*authorization/);
      assert.ok(text.split("\n").length <= rows);
      assert.ok(text.split("\n").every((row) => monitorTextWidth(row) <= columns));
    }
  }
});

test("native CPU GPU ANE power and thermal readings render together without enabling another sampler", () => {
  const sample = snapshot();
  sample.host!.gpu = {
    source: "ioreg", status: "ready", sampled_at_unix_ms: sample.sampled_at_unix_ms,
    age_ms: 0, utilization_percent: 37.5,
    devices: [{ id: "42", name: "Apple GPU", utilization_percent: 37.5 }], error: null,
  };
  sample.host!.accelerators = {
    ...emptyPowerMetrics("ready", null, "ioreport"), sampled_at_unix_ms: sample.sampled_at_unix_ms,
    age_ms: 0, sample_duration_ms: 1008, effective_uid: 501,
    cpu_power_mw: 3500, gpu_power_mw: 850, ane_power_mw: 0,
    gpu_active_percent: 31, gpu_frequency_mhz: 900, thermal_pressure: "Nominal",
    cpu_clusters: [{ name: "E-CPU", active_percent: 20, frequency_mhz: 1800 },
      { name: "P-CPU", active_percent: 12, frequency_mhz: 3500 }],
    power_sources: { cpu: "CLPC", gpu: "CLPC", ane: "Energy Model" }, warnings: [],
  };
  for (const [columns, rows] of [[80, 24], [140, 35], [50, 18]] as const) {
    const text = renderMonitorSnapshot(sample, { columns, rows });
    assert.match(text, /GPU use.*37\.5%/);
    assert.match(text, /3\.50 W/);
    assert.match(text, /850 mW/);
    assert.match(text, /ANE(?: power)? .*0 mW|ANE 0 mW/);
    assert.doesNotMatch(text, /sudo|authoriz|enable --powermetrics|Power optional/i);
    if (columns >= 80) {
      assert.match(text, /IOReport/);
      assert.match(text, /Thermal Nominal/);
      assert.match(text, /GPU state avg 900 MHz/);
    }
    assert.ok(text.split("\n").length <= rows);
    assert.ok(text.split("\n").every((row) => monitorTextWidth(row) <= columns));
  }
});

test("partial native counters retain measured power and explain the missing field without an authorization request", () => {
  const sample = snapshot();
  sample.host!.accelerators = {
    ...emptyPowerMetrics("ready", null, "ioreport"), sampled_at_unix_ms: sample.sampled_at_unix_ms,
    age_ms: 0, cpu_power_mw: 3500, gpu_power_mw: 850, ane_power_mw: null,
    gpu_active_percent: 31, gpu_frequency_mhz: 900, thermal_pressure: "Nominal",
    power_sources: { cpu: "CLPC", gpu: "CLPC", ane: null }, warnings: ["ANE channel absent"],
  };
  const text = renderMonitorSnapshot(sample, { columns: 160, rows: 35 });
  assert.match(text, /CPU .*3\.50 W/);
  assert.match(text, /GPU .*850 mW/);
  assert.match(text, /ANE .*unavailable/);
  assert.match(text, /ANE channel absent/);
  assert.match(text, /IOReport · active residency/);
  assert.doesNotMatch(text, /sudo|authoriz|enable --powermetrics|Power optional|powermetrics/i);
});

test("zero active residency displays idle without inventing a frequency", () => {
  const sample = snapshot();
  sample.host!.accelerators = {
    ...emptyPowerMetrics("ready", null, "ioreport"), sampled_at_unix_ms: sample.sampled_at_unix_ms,
    age_ms: 0, cpu_power_mw: 1100, gpu_power_mw: 0, ane_power_mw: 0,
    gpu_active_percent: 0, gpu_frequency_mhz: null, thermal_pressure: "Nominal",
    cpu_clusters: [{ name: "E-CPU", active_percent: 0, frequency_mhz: null }],
    power_sources: { cpu: "CLPC", gpu: "CLPC", ane: "Energy Model" }, warnings: [],
  };
  for (const [columns, rows] of [[80, 24], [140, 35], [50, 24]] as const) {
    const text = renderMonitorSnapshot(sample, { columns, rows });
    assert.match(text, /GPU idle/);
    assert.doesNotMatch(text, /0 MHz|GPU frequency unavailable|sudo|authoriz/i);
    if (columns === 140) assert.match(text, /E-CPU.*0\.0% · idle/);
  }
  sample.host!.accelerators.gpu_active_percent = null;
  assert.match(renderMonitorSnapshot(sample, { columns: 140, rows: 35 }), /GPU frequency unavailable/);
});

test("a stale IORegistry GPU reading keeps its own error and age while power remains fresh", () => {
  const sample = snapshot();
  sample.host!.gpu = {
    source: "ioreg", status: "stale", sampled_at_unix_ms: sample.sampled_at_unix_ms - 30_000,
    age_ms: 30_000, utilization_percent: 25,
    devices: [{ id: "42", name: "Apple GPU", utilization_percent: 25 }], error: "registry probe failed",
  };
  sample.host!.accelerators = {
    ...emptyPowerMetrics("ready"), sampled_at_unix_ms: sample.sampled_at_unix_ms, age_ms: 0,
    ane_power_mw: 0, gpu_active_percent: 99,
  };
  for (const [columns, rows] of [[120, 35], [80, 24], [50, 18]] as const) {
    const text = renderMonitorSnapshot(sample, { columns, rows });
    assert.match(text, /GPU stale.*30\.0s old/);
    assert.match(text, /GPU use.*25\.0%/);
    assert.match(text, /registry probe failed/);
    assert.doesNotMatch(text, /GPU active.*99\.0%/);
  }
});

test("multiple GPU utilization is explicitly the busiest reporting device", () => {
  const sample = snapshot();
  sample.host!.gpu = {
    source: "ioreg", status: "ready", sampled_at_unix_ms: sample.sampled_at_unix_ms,
    age_ms: 0, utilization_percent: 80,
    devices: [{ id: "42", name: "GPU A", utilization_percent: 20 }, { id: "43", name: "GPU B", utilization_percent: 80 }], error: null,
  };
  const text = renderMonitorSnapshot(sample, { columns: 120, rows: 35 });
  assert.match(text, /GPU busiest.*80\.0%/);
  assert.match(text, /busiest of 2 reporting GPUs/);
  sample.host!.gpu.devices[0]!.utilization_percent = null;
  assert.match(renderMonitorSnapshot(sample, { columns: 120, rows: 35 }), /busiest of 1 reporting GPU/);
});

test("missing ANE samples have no invented power scale or unavailable-per-second rates", () => {
  const sample = snapshot();
  const missing = renderMonitorSnapshot(sample, { columns: 120, rows: 35 });
  assert.match(missing, /ANE trend unavailable/);
  assert.match(missing, /ESTIMATED POWER · unavailable/);
  assert.doesNotMatch(missing, /ANE 0–|unavailable\/s|Disk R\/W|Network ↓\/↑/);
  const measuredZero = renderMonitorSnapshot(sample, { columns: 120, rows: 35, history: [{
    at_unix_ms: sample.sampled_at_unix_ms, cpu_percent: 0, memory_percent: 50,
    gpu_percent: 0, ane_power_mw: 0, reserved_percent: 0,
  }] });
  assert.match(measuredZero, /ANE 0–1 mW.*▁/);
  assert.doesNotMatch(measuredZero, /ANE trend unavailable/);
});
