import test from "node:test";
import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import {
  emptyPowerMetrics,
  NATIVE_METRICS_HELPER,
  PowerMetricsCollector,
  parseNativeMetricsOutput,
  parsePowerMetricsOutput,
  parsePowerMetricsPlist,
  runMetricCommand,
} from "../src/powermetrics.js";

// Synthetic values in the schema emitted by Apple's tool, checked against the
// original Mac Admins fixture and Wattson parser/test fixture (links in source).
const wrap = (body: string): string => `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>${body}</dict></plist>`;
const sample = wrap(`
<key>is_delta</key><true/>
<key>timestamp</key><date>2026-11-06T12:00:00Z</date>
<key>elapsed_ns</key><integer>1010000000</integer>
<key>processor</key><dict>
  <key>cpu_power</key><real>2450.25</real>
  <key>gpu_power</key><real>1100</real>
  <key>ane_power</key><real>0</real>
  <key>clusters</key><array><dict>
    <key>name</key><string>P-Cluster &amp; control</string>
    <key>freq_hz</key><real>3000000000</real>
    <key>idle_ratio</key><real>0.4</real>
    <key>down_ratio</key><real>0.2</real>
  </dict></array>
</dict>
<key>gpu</key><dict>
  <key>freq_hz</key><real>900000000</real>
  <key>idle_ratio</key><real>0.75</real>
</dict>
<key>thermal_pressure</key><string>Nominal</string>
<key>disk</key><dict>
  <key>rbytes_per_s</key><real>4096</real>
  <key>wbytes_per_s</key><real>0</real>
</dict>
<key>network</key><dict>
  <key>ibyte_rate</key><real>10000</real>
  <key>obyte_rate</key><real>512</real>
</dict>
`);

test("hardware sample preserves units, real zero ANE power, and active residency", () => {
  const metrics = parsePowerMetricsOutput(`${sample}\0`, 1234);
  assert.equal(metrics.status, "ready");
  assert.equal(metrics.cpu_power_mw, 2450.25);
  assert.equal(metrics.gpu_power_mw, 1100);
  assert.equal(metrics.ane_power_mw, 0);
  assert.equal(metrics.gpu_active_percent, 25);
  assert.equal(metrics.gpu_frequency_mhz, 900);
  assert.equal(metrics.sample_duration_ms, 1010);
  assert.equal(metrics.sampled_at_unix_ms, 1234);
  assert.equal(metrics.thermal_pressure, "Nominal");
  assert.equal(metrics.disk_write_bytes_per_second, 0);
  assert.equal(metrics.network_in_bytes_per_second, 10000);
  assert.equal(metrics.cpu_clusters[0]!.frequency_mhz, 3000);
  assert.equal(metrics.cpu_clusters[0]!.name, "P-Cluster & control");
  assert.ok(Math.abs(metrics.cpu_clusters[0]!.active_percent! - 40) < 0.000001);
});

test("GPU freq_hz accepts the recorded MHz encoding without changing CPU conversion", () => {
  // GPU value from the real Mac15,9/macOS27 capture linked in gpuFrequencyMhz.
  const metrics = parsePowerMetricsOutput(sample.replace("900000000", "749.789"));
  assert.equal(metrics.gpu_frequency_mhz, 749.789);
  assert.equal(metrics.cpu_clusters[0]!.frequency_mhz, 3000);
});

test("GPU frequency still accepts Hz and preserves zero and missing readings", () => {
  assert.equal(parsePowerMetricsOutput(sample.replace("900000000", "888000000")).gpu_frequency_mhz, 888);
  assert.equal(parsePowerMetricsOutput(sample.replace("900000000", "0")).gpu_frequency_mhz, 0);
  const missing = sample.replace("<key>freq_hz</key><real>900000000</real>", "");
  assert.equal(parsePowerMetricsOutput(missing).gpu_frequency_mhz, null);
});

test("missing hardware values stay null; energy is not mislabeled as power", () => {
  const metrics = parsePowerMetricsOutput(wrap(`
    <key>processor</key><dict>
      <key>cpu_power</key><real>0</real>
      <key>ane_energy</key><real>50</real>
    </dict>
    <key>gpu</key><dict><key>idle_ratio</key><real>nan</real></dict>
  `));
  assert.equal(metrics.cpu_power_mw, 0);
  assert.equal(metrics.ane_power_mw, null);
  assert.equal(metrics.gpu_active_percent, null);
  assert.equal(metrics.gpu_power_mw, null);
  assert.equal(metrics.thermal_pressure, null);
  assert.equal(metrics.disk_read_bytes_per_second, null);
  assert.equal(emptyPowerMetrics("disabled").ane_power_mw, null);
});

test("explicit zero residency wins; invalid counters do not become busy or idle", () => {
  const metrics = parsePowerMetricsOutput(wrap(`
    <key>processor</key><dict><key>cpu_power</key><real>1</real></dict>
    <key>gpu</key><dict><key>used_ratio</key><real>0</real><key>idle_ratio</key><real>0.2</real></dict>
  `));
  assert.equal(metrics.gpu_active_percent, 0);
  const invalid = parsePowerMetricsOutput(wrap(`
    <key>processor</key><dict><key>cpu_power</key><real>1</real></dict>
    <key>gpu</key><dict><key>idle_ratio</key><real>1.5</real></dict>
  `));
  assert.equal(invalid.gpu_active_percent, null);
});

test("NUL-separated output selects the last sample and bounds count and size", () => {
  const first = sample.replace("2450.25", "100");
  assert.equal(parsePowerMetricsOutput(`${first}\0${sample}\0`).cpu_power_mw, 2450.25);
  assert.throws(() => parsePowerMetricsOutput(""), /sample count/);
  assert.throws(() => parsePowerMetricsOutput(`${sample}\0`.repeat(5)), /sample count/);
  assert.throws(() => parsePowerMetricsOutput("x".repeat(1024 * 1024 + 1)), /too large/);
});

test("invalid flags and unknown schemas fail instead of reporting healthy zeros", () => {
  assert.throws(() => parsePowerMetricsOutput(wrap("<key>invalid</key><true/>")), /marked.*invalid/);
  assert.throws(() => parsePowerMetricsOutput(wrap("<key>processor</key><dict><key>invalid</key><true/></dict>")), /invalid/);
  assert.throws(() => parsePowerMetricsOutput(wrap("<key>is_delta</key><false/>")), /invalid/);
  assert.throws(() => parsePowerMetricsOutput(wrap("<key>processor</key><dict/>")), /no CPU\/GPU\/ANE/);
});

test("bounded plist reader rejects malformed nesting, duplicate keys and custom entities", () => {
  assert.throws(() => parsePowerMetricsPlist(wrap("<key>a</key><string>unterminated")), /unterminated/);
  assert.throws(() => parsePowerMetricsPlist(wrap("<key>a</key><integer>1</integer><key>a</key><integer>2</integer>")), /duplicate/);
  assert.throws(() => parsePowerMetricsPlist(wrap("<key>a</key><string>&external;</string>")), /entity/);
  assert.throws(() => parsePowerMetricsPlist(wrap(`<key>a</key>${"<array>".repeat(34)}<integer>1</integer>${"</array>".repeat(34)}`)), /parser limits/);
  assert.throws(() => parsePowerMetricsPlist(wrap("<key>a</key><real>not-a-number</real>")), /numeric/);
  assert.throws(() => parsePowerMetricsPlist(wrap("<key>a</key><integer>1</integer>") + "oops"), /trailing/);
});

const nativeFrame = () => ({
  schema_version: 1, source: "ioreport", effective_uid: 501,
  sampled_at_unix_ms: 1000, sample_duration_ms: 1008.25,
  cpu_power_mw: 2500.5, gpu_power_mw: 825.25, ane_power_mw: 0,
  gpu_active_percent: 12.5, gpu_frequency_mhz: 900,
  cpu_clusters: [{ name: "E-CPU", active_percent: 24.5, frequency_mhz: 1800 },
    { name: "P-CPU", active_percent: 16, frequency_mhz: 3500 }],
  thermal_pressure: "Nominal",
  disk_read_bytes_per_second: 4096, disk_write_bytes_per_second: 0,
  network_in_bytes_per_second: 8192, network_out_bytes_per_second: 512,
  power_sources: { cpu: "CLPC", gpu: "CLPC", ane: "Energy Model" }, warnings: [],
});

test("native hardware JSON preserves measured units, idle zero, source identity and actual elapsed time", () => {
  const metrics = parseNativeMetricsOutput(JSON.stringify(nativeFrame()), 1250);
  assert.equal(metrics.status, "ready");
  assert.equal(metrics.source, "ioreport");
  assert.equal(metrics.effective_uid, 501);
  assert.equal(metrics.cpu_power_mw, 2500.5);
  assert.equal(metrics.gpu_power_mw, 825.25);
  assert.equal(metrics.ane_power_mw, 0);
  assert.equal(metrics.gpu_frequency_mhz, 900);
  assert.equal(metrics.gpu_active_percent, 12.5);
  assert.equal(metrics.sample_duration_ms, 1008.25);
  assert.equal(metrics.sampled_at_unix_ms, 1000);
  assert.equal(metrics.age_ms, 250);
  assert.equal(metrics.cpu_clusters[1]!.frequency_mhz, 3500);
  assert.equal(metrics.thermal_pressure, "Nominal");
  assert.equal(metrics.disk_write_bytes_per_second, 0);
  assert.deepEqual(metrics.power_sources, nativeFrame().power_sources);
  assert.deepEqual(metrics.warnings, []);
});

test("missing native counters remain null and partial warnings do not invalidate measured power history", () => {
  const frame = { ...nativeFrame(), ane_power_mw: null,
    power_sources: { ...nativeFrame().power_sources, ane: null },
    warnings: ["ANE energy channel is unavailable"], network_in_bytes_per_second: null };
  const result = parseNativeMetricsOutput(JSON.stringify(frame), 1000);
  assert.equal(result.status, "ready");
  assert.equal(result.error, null);
  assert.equal(result.cpu_power_mw, 2500.5);
  assert.equal(result.ane_power_mw, null);
  assert.equal(result.network_in_bytes_per_second, null);
  assert.equal(result.power_sources!.ane, null);
  assert.deepEqual(result.warnings, ["ANE energy channel is unavailable"]);
});

test("native schema rejects malformed, missing or inconsistent fields instead of inventing valid readings", () => {
  for (const [field, value] of [
    ["schema_version", 2], ["source", "powermetrics"], ["effective_uid", -1],
    ["sampled_at_unix_ms", -1], ["sampled_at_unix_ms", 999_999],
    ["sample_duration_ms", 0], ["sample_duration_ms", null],
    ["cpu_power_mw", "3"], ["gpu_power_mw", -1], ["ane_power_mw", false],
    ["gpu_active_percent", 101], ["gpu_frequency_mhz", undefined],
    ["cpu_clusters", {}], ["cpu_clusters", [{ name: "CPU", active_percent: -1, frequency_mhz: 2000 }]],
    ["thermal_pressure", "maybe"], ["warnings", "warning"], ["warnings", [null]],
    ["power_sources", { cpu: null, gpu: "CLPC", ane: "Energy Model" }],
  ] as const) {
    const frame: Record<string, unknown> = nativeFrame();
    frame[field] = value;
    assert.throws(() => parseNativeMetricsOutput(JSON.stringify(frame), 1000), Error, String(field));
  }
  assert.throws(() => parseNativeMetricsOutput("null"), /sample/);
  assert.throws(() => parseNativeMetricsOutput("[]"), /sample/);
  assert.throws(() => parseNativeMetricsOutput(JSON.stringify(nativeFrame()).replace('"cpu_power_mw":2500.5', '"cpu_power_mw":1e999')), /cpu_power_mw/);
  assert.throws(() => parseNativeMetricsOutput("x".repeat(1024 * 1024 + 1)), /too large/);
  assert.throws(() => parseNativeMetricsOutput(sample), /JSON|Unexpected/);
});

test("unsupported platforms and already stopped native collection invoke no helper", async () => {
  let commands = 0;
  const runCommand = async () => { commands++; return JSON.stringify(nativeFrame()); };
  for (const [platform, architecture] of [["linux", "arm64"], ["darwin", "x64"]]) {
    const result = await new PowerMetricsCollector({ enabled: true }, {
      platform: () => platform!, architecture: () => architecture!, runCommand,
    }).sample(new AbortController().signal);
    assert.equal(result.status, "unavailable");
    assert.equal(result.source, "ioreport");
    assert.equal(result.ane_power_mw, null);
  }
  const stopped = new AbortController();
  stopped.abort();
  const result = await new PowerMetricsCollector({ enabled: true }, {
    platform: () => "darwin", architecture: () => "arm64", runCommand,
  }).sample(stopped.signal);
  assert.equal(result.status, "stopped");
  assert.equal(commands, 0);
});

test("native collection launches only its colocated finite helper and retains stale measurements after failure", async () => {
  let now = 1000;
  let nativeSamples = 0;
  const collector = new PowerMetricsCollector({ enabled: true, intervalMs: 60_000 }, {
    platform: () => "darwin", architecture: () => "arm64", now: () => now,
    runCommand: async (file, args, _signal, timeout) => {
      assert.equal(file, NATIVE_METRICS_HELPER);
      assert.match(file, /\/dist\/native\/alloy-metrics$/);
      assert.deepEqual(args, ["--interval-ms", "1000"]);
      assert.equal(timeout, 3500);
      return ++nativeSamples === 2 ? "malformed data" : JSON.stringify({ ...nativeFrame(), sampled_at_unix_ms: now });
    },
  });
  const first = await collector.sample(new AbortController().signal);
  assert.equal(first.status, "ready");
  assert.equal(first.sampled_at_unix_ms, 1000);
  now = 3500;
  const stale = await collector.sample(new AbortController().signal);
  assert.equal(stale.status, "stale");
  assert.equal(stale.age_ms, 2500);
  assert.equal(stale.sampled_at_unix_ms, 1000);
  assert.equal(stale.ane_power_mw, 0);
  assert.match(stale.error!, /JSON|Unexpected/);
  now = 5000;
  const recovered = await collector.sample(new AbortController().signal);
  assert.equal(recovered.status, "ready");
  assert.equal(recovered.sampled_at_unix_ms, now);
  assert.equal(recovered.error, null);
});

test("a missing native helper requests a CLI rebuild and never launches an alternate backend", async () => {
  const commands: string[] = [];
  const collector = new PowerMetricsCollector({ enabled: true }, {
    platform: () => "darwin", architecture: () => "arm64",
    runCommand: async (file) => {
      commands.push(file);
      throw new Error("Metric helper failed (ENOENT).");
    },
  });
  const result = await collector.sample(new AbortController().signal);
  assert.equal(result.status, "unavailable");
  assert.equal(result.ane_power_mw, null);
  assert.equal(result.sampled_at_unix_ms, null);
  assert.match(result.error!, /Rebuild.*npm --prefix ts\/cli run build/);
  assert.doesNotMatch(result.error!, /sudo|authoriz|password/i);
  assert.deepEqual(commands, [NATIVE_METRICS_HELPER]);
});

test("native samples share ownership and a result arriving after cancellation cannot become fresh", async () => {
  const controller = new AbortController();
  let release!: (output: string) => void;
  let calls = 0;
  const collector = new PowerMetricsCollector({ enabled: true, intervalMs: 1 }, {
    platform: () => "darwin", architecture: () => "arm64", now: () => 1000,
    runCommand: async (file, args, signal) => {
      calls++;
      assert.equal(file, NATIVE_METRICS_HELPER);
      assert.deepEqual(args, ["--interval-ms", "500"]);
      assert.equal(signal, controller.signal);
      return new Promise<string>((resolve) => { release = resolve; });
    },
  });
  const first = collector.sample(controller.signal);
  assert.equal(collector.sample(controller.signal), first);
  controller.abort();
  release(JSON.stringify(nativeFrame()));
  const result = await first;
  assert.equal(calls, 1);
  assert.equal(result.status, "stopped");
  assert.equal(result.sampled_at_unix_ms, null);
  assert.equal(result.cpu_power_mw, null);
});

test("Apple Silicon reports real CPU GPU ANE power frequency and thermal state without elevation", {
  skip: process.platform !== "darwin" || process.arch !== "arm64",
  timeout: 7000,
}, async (t) => {
  assert.equal(typeof process.geteuid, "function");
  assert.notEqual(process.geteuid!(), 0, "Run native hardware acceptance as a normal user.");
  const selfTest = await runMetricCommand(NATIVE_METRICS_HELPER, ["--self-test"], new AbortController().signal, 1500);
  assert.deepEqual(JSON.parse(selfTest), { self_test: true }, "Native unit conversion and frequency-state checks must pass.");
  const result = await new PowerMetricsCollector({ enabled: true, intervalMs: 500 }).sample(new AbortController().signal);
  assert.equal(result.status, "ready", result.error ?? "No native hardware sample.");
  assert.equal(result.source, "ioreport");
  assert.equal(result.effective_uid, process.geteuid!());
  assert.equal(result.error, null);
  for (const key of ["cpu", "gpu", "ane"] as const) {
    const value = result[`${key}_power_mw`];
    assert.ok(value !== null && Number.isFinite(value) && value >= 0, `${key} power missing: ${result.warnings?.join("; ")}`);
    assert.equal(typeof result.power_sources?.[key], "string", `${key} needs a real measured source, including idle zero`);
  }
  assert.ok(!(result.cpu_power_mw === 0 && result.warnings?.some((warning) =>
    /\bCPU\b/i.test(warning) && /macOS 27/i.test(warning) && /legacy|stall|frozen/i.test(warning))),
  `A potentially frozen legacy CPU energy counter cannot verify live power: ${result.warnings?.join("; ")}`);
  assert.ok(result.sample_duration_ms !== null && result.sample_duration_ms > 0);
  assert.ok(result.gpu_active_percent !== null, `GPU residency counter missing: ${result.warnings?.join("; ")}`);
  if (result.gpu_active_percent > 0) {
    assert.ok(result.gpu_frequency_mhz !== null && Number.isFinite(result.gpu_frequency_mhz) && result.gpu_frequency_mhz > 0,
      `Active GPU frequency missing: ${result.warnings?.join("; ")}`);
  }
  assert.ok(result.cpu_clusters.length > 0, "CPU frequency-state counters are missing.");
  for (const cluster of result.cpu_clusters) {
    assert.ok(cluster.active_percent !== null, `${cluster.name} residency counter missing: ${result.warnings?.join("; ")}`);
    if (cluster.active_percent > 0) {
      assert.ok(cluster.frequency_mhz !== null && Number.isFinite(cluster.frequency_mhz) && cluster.frequency_mhz > 0,
        `Active ${cluster.name} frequency missing: ${result.warnings?.join("; ")}`);
    }
  }
  assert.ok(result.thermal_pressure !== null, "Thermal pressure must be read from Foundation.");
  t.diagnostic(JSON.stringify({ effective_uid: result.effective_uid, cpu_power_mw: result.cpu_power_mw,
    gpu_power_mw: result.gpu_power_mw, ane_power_mw: result.ane_power_mw, power_sources: result.power_sources,
    gpu_active_percent: result.gpu_active_percent, gpu_frequency_state_average_mhz: result.gpu_frequency_mhz, cpu_frequency_states: result.cpu_clusters,
    thermal_pressure: result.thermal_pressure, sample_duration_ms: result.sample_duration_ms, warnings: result.warnings }));
});

test("bounded helper timeout terminates even a child ignoring SIGTERM", async () => {
  const started = performance.now();
  await assert.rejects(runMetricCommand(process.execPath, ["-e",
    "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000);",
  ], new AbortController().signal, 150), /timed out/);
  assert.ok(performance.now() - started < 3000);
});

test("aborting an active helper stops it without an interactive prompt", async () => {
  const controller = new AbortController();
  const running = runMetricCommand(process.execPath, ["-e", "setInterval(() => {}, 1000)"], controller.signal);
  setTimeout(() => { controller.abort(); }, 50);
  await assert.rejects(running, /stopped/);
  await assert.rejects(runMetricCommand(process.execPath, ["-e", "console.log('never')"], controller.signal), /stopped/);
});
