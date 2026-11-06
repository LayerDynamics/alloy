import test from "node:test";
import assert from "node:assert/strict";
import { MonitorHistory, MONITOR_HISTORY_MAX_AGE_MS } from "../src/monitor-history.js";
import type { DaemonMetrics, MonitorSnapshot } from "../src/monitor-types.js";
import { emptyPowerMetrics } from "../src/powermetrics.js";

function frame(at = 10_000): MonitorSnapshot {
  const metrics: DaemonMetrics = {
    schema_version: 1, sampled_at_unix_ms: at, process_id: 42,
    admission: { budget_bytes: 100, reserved_bytes: 25, available_bytes: 75, active_reservations: 1, admissions_paused: false },
    execution: { active_request_id: null, cancellation_requested: false },
    residency: { worker_alive: true, sample_age_ms: 0, models_resident: 0, models: [] },
    memory_guard: {
      tier: "balanced", pressure: "ok", emergency: false, usage_bytes: null,
      static_ceiling_bytes: null, dynamic_ceiling_bytes: null, metal_ceiling_bytes: null,
      hard_limit_bytes: null, soft_bytes: null, hard_bytes: null, abort_limit_bytes: null,
      admit_line_bytes: null, binding: null, sample_age_ms: null, last_error: null, cache_reclaim_requested: false,
    },
  };
  return {
    schema_version: 1, sampled_at_unix_ms: at, host_sample_age_ms: 0, host_error: null,
    host: {
      sampled_at_unix_ms: at, platform: "darwin", architecture: "arm64", hostname: "mac-a",
      logical_cpu_count: 14, cpu_utilization_percent: 40, cpu_sample_ms: 1000,
      load_average: [1, 2, 3], uptime_seconds: 1000,
      memory_total_bytes: 100, memory_used_bytes: 50, memory_free_bytes: 50, memory_available_bytes: null,
      swap_total_bytes: 0, swap_used_bytes: 0, swap_free_bytes: 0, wired_bytes: null, compressed_bytes: null, errors: [],
      accelerators: {
        ...emptyPowerMetrics("ready"), sampled_at_unix_ms: at, age_ms: 0,
        gpu_active_percent: 20, ane_power_mw: 0,
      },
    },
    alloy: { status: "connected", sample_age_ms: 0, error: null, metrics },
  };
}

test("history stays bounded and returned frames cannot mutate retained values", () => {
  const history = new MonitorHistory();
  for (let i = 0; i < 65; i++) history.push(frame(10_000 + i * 1000));
  assert.equal(history.snapshot().length, 60);
  assert.equal(history.snapshot()[0]!.at_unix_ms, 15_000);
  const copy = history.snapshot();
  copy[0]!.cpu_percent = 99;
  assert.equal(history.snapshot()[0]!.cpu_percent, 40);
  const small = new MonitorHistory(2);
  small.push(frame(1000)); small.push(frame(2000)); small.push(frame(3000));
  assert.deepEqual(small.snapshot().map((sample) => sample.at_unix_ms), [2000, 3000]);
  for (const capacity of [0, -1, 1.5, 61, Number.NaN]) assert.throws(() => new MonitorHistory(capacity), /capacity/);
});

test("zero remains measured zero and missing or invalid values remain gaps", () => {
  const history = new MonitorHistory();
  const zero = frame();
  zero.host!.cpu_utilization_percent = 0;
  zero.host!.memory_used_bytes = 0;
  zero.host!.accelerators.gpu_active_percent = 0;
  zero.alloy.metrics!.admission.reserved_bytes = 0;
  history.push(zero);
  assert.deepEqual(history.snapshot()[0], {
    at_unix_ms: 10_000, cpu_percent: 0, memory_percent: 0, gpu_percent: 0, ane_power_mw: 0, reserved_percent: 0,
  });
  const absent = frame(11_000);
  absent.host!.cpu_utilization_percent = null;
  absent.host!.memory_total_bytes = 0;
  absent.host!.accelerators.gpu_active_percent = Number.NaN;
  absent.host!.accelerators.ane_power_mw = null;
  absent.alloy.metrics!.admission.budget_bytes = 0;
  history.push(absent);
  assert.deepEqual(history.snapshot()[1], {
    at_unix_ms: 11_000, cpu_percent: null, memory_percent: null, gpu_percent: null, ane_power_mw: null, reserved_percent: null,
  });
});

test("host errors, stale hardware and disconnected daemon produce independent gaps", () => {
  const history = new MonitorHistory();
  const hostFailure = frame();
  hostFailure.host_error = "host probe failed";
  history.push(hostFailure);
  assert.equal(history.snapshot()[0]!.cpu_percent, null);
  assert.equal(history.snapshot()[0]!.gpu_percent, null);
  assert.equal(history.snapshot()[0]!.reserved_percent, 25);
  const partial = frame(11_000);
  partial.host!.errors = ["swap probe unavailable"];
  partial.host!.accelerators.status = "stale";
  partial.alloy.status = "unavailable";
  history.push(partial);
  assert.equal(history.snapshot()[1]!.cpu_percent, 40);
  assert.equal(history.snapshot()[1]!.memory_percent, 50);
  assert.equal(history.snapshot()[1]!.gpu_percent, null);
  assert.equal(history.snapshot()[1]!.ane_power_mw, null);
  assert.equal(history.snapshot()[1]!.reserved_percent, null);
});

test("source timestamp deduplication is independent and repeated frames do not advance history", () => {
  const history = new MonitorHistory();
  const first = frame();
  history.push(first);
  history.push(first);
  assert.equal(history.snapshot().length, 1);
  const second = frame(11_000);
  second.host!.sampled_at_unix_ms = 10_000;
  second.host_sample_age_ms = 1000;
  second.alloy.metrics!.sampled_at_unix_ms = 10_000;
  second.alloy.sample_age_ms = 1000;
  history.push(second);
  assert.deepEqual(history.snapshot()[1], {
    at_unix_ms: 11_000, cpu_percent: null, memory_percent: null, gpu_percent: 20, ane_power_mw: 0, reserved_percent: null,
  });
  const third = frame(12_000);
  third.host!.accelerators.sampled_at_unix_ms = 11_000;
  third.host!.accelerators.age_ms = 1000;
  history.push(third);
  assert.equal(history.snapshot()[2]!.cpu_percent, 40);
  assert.equal(history.snapshot()[2]!.gpu_percent, null);
  assert.equal(history.snapshot()[2]!.ane_power_mw, null);
  assert.equal(history.snapshot()[2]!.reserved_percent, 25);
});

test("old, future and inaccurately reported freshness do not enter graphs", () => {
  const history = new MonitorHistory();
  const old = frame(20_000);
  old.host!.sampled_at_unix_ms -= MONITOR_HISTORY_MAX_AGE_MS + 1;
  old.host_sample_age_ms = 0; // A misleading age cannot override source time.
  old.alloy.sample_age_ms = MONITOR_HISTORY_MAX_AGE_MS + 1;
  history.push(old);
  assert.equal(history.snapshot()[0]!.cpu_percent, null);
  assert.equal(history.snapshot()[0]!.gpu_percent, null);
  assert.equal(history.snapshot()[0]!.reserved_percent, null);
  const future = frame(21_000);
  future.host!.accelerators.sampled_at_unix_ms = 30_000;
  future.alloy.metrics!.sampled_at_unix_ms = 30_000;
  history.push(future);
  assert.equal(history.snapshot()[1]!.gpu_percent, null);
  assert.equal(history.snapshot()[1]!.reserved_percent, null);
  history.push(frame(22_000));
  assert.equal(history.snapshot()[2]!.gpu_percent, 20);
  assert.equal(history.snapshot()[2]!.reserved_percent, 25);
});

test("a host identity change resets all histories", () => {
  const history = new MonitorHistory();
  history.push(frame());
  const changed = frame(11_000);
  changed.host!.hostname = "mac-b";
  history.push(changed);
  assert.equal(history.snapshot().length, 1);
  assert.equal(history.snapshot()[0]!.at_unix_ms, 11_000);
  assert.equal(history.snapshot()[0]!.reserved_percent, 25);
});

test("daemon restart clears only previous ledger values and permits a fresh new process sample", () => {
  const history = new MonitorHistory();
  history.push(frame());
  const restarted = frame(11_000);
  restarted.alloy.metrics!.process_id = 43;
  restarted.alloy.metrics!.sampled_at_unix_ms = 10_000;
  restarted.alloy.sample_age_ms = 1000;
  restarted.alloy.metrics!.admission.reserved_bytes = 80;
  history.push(restarted);
  assert.equal(history.snapshot().length, 2);
  assert.equal(history.snapshot()[0]!.cpu_percent, 40);
  assert.equal(history.snapshot()[0]!.reserved_percent, null);
  assert.equal(history.snapshot()[1]!.reserved_percent, 80);
});

test("a reduced live budget can legitimately put existing reservations above 100 percent", () => {
  const history = new MonitorHistory();
  const sample = frame();
  sample.alloy.metrics!.admission.reserved_bytes = 125;
  sample.alloy.metrics!.admission.admissions_paused = true;
  history.push(sample);
  assert.equal(history.snapshot()[0]!.reserved_percent, 125);
});


function withGpu(sample: MonitorSnapshot, utilization = 35): MonitorSnapshot {
  sample.host!.gpu = {
    source: "ioreg", status: "ready", sampled_at_unix_ms: sample.sampled_at_unix_ms,
    age_ms: 0, utilization_percent: utilization,
    devices: [{ id: "42", name: "Apple GPU", utilization_percent: utilization }], error: null,
  };
  return sample;
}

test("GPU utilization history continues through disabled, unauthorized and stale power samples", () => {
  const history = new MonitorHistory();
  for (const [index, status] of (["disabled", "unavailable", "stale"] as const).entries()) {
    const sample = withGpu(frame(10_000 + index * 1000), index === 0 ? 0 : 35);
    sample.host!.accelerators = {
      ...emptyPowerMetrics(status, status === "disabled" ? null : "power authorization unavailable"),
      sampled_at_unix_ms: status === "stale" ? 1000 : null, age_ms: status === "stale" ? 11_000 : null,
      ane_power_mw: status === "stale" ? 500 : null,
    };
    history.push(sample);
  }
  assert.deepEqual(history.snapshot().map((sample) => sample.gpu_percent), [0, 35, 35]);
  assert.deepEqual(history.snapshot().map((sample) => sample.ane_power_mw), [null, null, null]);
});

test("GPU and ANE history deduplicate independently and reject each source's stale or invalid data", () => {
  const history = new MonitorHistory();
  history.push(withGpu(frame()));
  const powerRepeated = withGpu(frame(11_000), 40);
  powerRepeated.host!.accelerators.sampled_at_unix_ms = 10_000;
  powerRepeated.host!.accelerators.age_ms = 1000;
  history.push(powerRepeated);
  const gpuRepeated = withGpu(frame(12_000), 45);
  gpuRepeated.host!.gpu!.sampled_at_unix_ms = 11_000;
  gpuRepeated.host!.gpu!.age_ms = 1000;
  history.push(gpuRepeated);
  const gpuFailed = withGpu(frame(13_000));
  gpuFailed.host!.gpu!.status = "stale";
  gpuFailed.host!.gpu!.error = "registry unavailable";
  history.push(gpuFailed);
  const gpuOld = withGpu(frame(14_000));
  gpuOld.host!.gpu!.sampled_at_unix_ms = 1000;
  gpuOld.host!.gpu!.age_ms = 0;
  history.push(gpuOld);
  const gpuFuture = withGpu(frame(15_000));
  gpuFuture.host!.gpu!.sampled_at_unix_ms = 30_000;
  history.push(gpuFuture);
  const gpuInvalid = withGpu(frame(16_000), Number.NaN);
  history.push(gpuInvalid);
  assert.deepEqual(history.snapshot().map((sample) => sample.gpu_percent), [35, 40, null, null, null, null, null]);
  assert.deepEqual(history.snapshot().map((sample) => sample.ane_power_mw), [0, null, 0, 0, 0, 0, 0]);
});

test("switching GPU sources or devices clears only the previous GPU trend", () => {
  const history = new MonitorHistory();
  history.push(frame());
  const registry = withGpu(frame(11_000));
  history.push(registry);
  assert.equal(history.snapshot()[0]!.gpu_percent, null);
  assert.equal(history.snapshot()[0]!.cpu_percent, 40);
  assert.equal(history.snapshot()[0]!.ane_power_mw, 0);
  assert.equal(history.snapshot()[1]!.gpu_percent, 35);
  const replacement = withGpu(frame(12_000), 50);
  replacement.host!.gpu!.devices[0]!.id = "another-gpu";
  history.push(replacement);
  assert.equal(history.snapshot()[1]!.gpu_percent, null);
  assert.equal(history.snapshot()[2]!.gpu_percent, 50);
  assert.equal(history.snapshot()[1]!.ane_power_mw, 0);
});
