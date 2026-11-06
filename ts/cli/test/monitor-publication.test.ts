import test from "node:test";
import assert from "node:assert/strict";
import { collectMonitorSnapshot, refreshMonitorAges } from "../src/monitor.js";
import { MonitorReadiness } from "../src/doctor.js";
import { MonitorHistory } from "../src/monitor-history.js";
import { defaultSettings } from "../src/settings.js";
import { emptyPowerMetrics } from "../src/powermetrics.js";
import type { HostMetricsSnapshot } from "../src/host-metrics.js";
import type { DaemonMetrics, MonitorSnapshot } from "../src/monitor-types.js";
import type { ModelInspection } from "../src/model.js";

test("delayed readiness advice advances publication and source ages without changing native relative ages", async (t) => {
  let now = 1_000_000;
  t.mock.method(Date, "now", () => now);
  const host: HostMetricsSnapshot = {
    sampled_at_unix_ms: now - 4000, platform: "darwin", architecture: "arm64", hostname: "mac",
    logical_cpu_count: 12, cpu_utilization_percent: 25, cpu_sample_ms: 1000, load_average: [1, 2, 3], uptime_seconds: 900,
    memory_total_bytes: 1000, memory_free_bytes: 500, memory_used_bytes: 500, memory_available_bytes: null,
    swap_total_bytes: 0, swap_used_bytes: 0, swap_free_bytes: 0, wired_bytes: null, compressed_bytes: null, errors: [],
    accelerators: { ...emptyPowerMetrics("ready"), sampled_at_unix_ms: now - 4500, age_ms: 0, gpu_active_percent: 20 },
  };
  const daemon: DaemonMetrics = {
    schema_version: 1, sampled_at_unix_ms: now - 3500, process_id: 42,
    admission: { budget_bytes: 100, reserved_bytes: 125, available_bytes: 0, active_reservations: 1, admissions_paused: true },
    execution: { active_request_id: "request-a", cancellation_requested: true },
    residency: { worker_alive: true, sample_age_ms: 100, models_resident: 0, models: [] },
    memory_guard: { tier: "balanced", pressure: "emergency", emergency: true, usage_bytes: 125,
      static_ceiling_bytes: null, dynamic_ceiling_bytes: null, metal_ceiling_bytes: null, hard_limit_bytes: 100,
      soft_bytes: null, hard_bytes: null, abort_limit_bytes: null, admit_line_bytes: null, binding: "dynamic", sample_age_ms: 600,
      last_error: null, cache_reclaim_requested: true },
  };
  const captured = await collectMonitorSnapshot({ sample: async () => host, close: async () => {} }, {
    read: async () => ({ status: "unavailable", sample_age_ms: 0, error: "ECONNREFUSED", metrics: daemon }),
  });
  let finishAdvice!: (model: ModelInspection) => void;
  const advice = new MonitorReadiness({ ...defaultSettings(), artifact: "/models/starter" }, { reloadSettings: () => null },
    async () => new Promise<ModelInspection>((resolve) => { finishAdvice = resolve; }));
  const pending = advice.refresh("ECONNREFUSED");
  await Promise.resolve();
  now += 1250;
  finishAdvice({ ok: true, path: "/models/starter", verified: false, errors: [] });
  await pending;
  const published = refreshMonitorAges(captured);
  assert.equal(published.sampled_at_unix_ms, now);
  assert.equal(published.host_sample_age_ms, 5250);
  assert.equal(published.host!.accelerators.age_ms, 5750);
  assert.equal(published.alloy.sample_age_ms, 4750);
  assert.equal(published.alloy.metrics, daemon);
  assert.equal(daemon.memory_guard.sample_age_ms, 600);
  assert.equal(daemon.residency.sample_age_ms, 100);
  assert.equal(published.host!.sampled_at_unix_ms, 996_000);
  assert.equal(published.host!.accelerators.sampled_at_unix_ms, 995_500);
  assert.equal(captured.sampled_at_unix_ms, 1_000_000, "publication must not mutate the collected frame");
  assert.equal(captured.host_sample_age_ms, 4000);
  const history = new MonitorHistory();
  history.push(published);
  assert.equal(history.snapshot()[0]!.cpu_percent, null, "advice delay cannot make old host data look fresh in history");
  assert.equal(history.snapshot()[0]!.gpu_percent, null);
});

test("publication leaves absent samples absent and retains collection errors", () => {
  const sample: MonitorSnapshot = {
    schema_version: 1, sampled_at_unix_ms: 1000, host: null, host_sample_age_ms: null, host_error: "host unavailable",
    alloy: { status: "unavailable", sample_age_ms: null, metrics: null, error: "connection refused" },
  };
  const published = refreshMonitorAges(sample, 2500);
  assert.equal(published.host_sample_age_ms, null);
  assert.equal(published.alloy.sample_age_ms, null);
  assert.equal(published.host, null);
  assert.equal(published.host_error, "host unavailable");
  assert.equal(published.alloy.error, "connection refused");
});
