import test from "node:test";
import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import {
  emptyPowerMetrics,
  PowerMetricsCollector,
  parsePowerMetricsOutput,
  parsePowerMetricsPlist,
  runMetricCommand,
  supportedPowerSamplers,
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

test("sampler selection adapts to the Mac and does not assume ANE sampler support", () => {
  const older = "The following samplers are supported:\n    cpu_power cpu power\n    gpu_power gpu power\n    thermal pressure\n    all tasks,cpu_power,gpu_power\n";
  assert.deepEqual(supportedPowerSamplers(older), ["cpu_power", "gpu_power", "thermal"]);
  assert.deepEqual(supportedPowerSamplers(`${older}\nane_power neural engine\n network usage\n disk usage`),
    ["cpu_power", "gpu_power", "ane_power", "thermal", "disk", "network"]);
  assert.deepEqual(supportedPowerSamplers("sampler cpu_power is not supported"), []);
});

test("hardware collector returns unavailable without running helpers on unsupported platforms", async () => {
  let commands = 0;
  const collector = new PowerMetricsCollector({ enabled: true }, {
    platform: () => "linux",
    runCommand: async () => { commands++; return ""; },
  });
  const result = await collector.sample(new AbortController().signal);
  assert.equal(commands, 0);
  assert.equal(result.status, "unavailable");
  assert.equal(result.ane_power_mw, null);
});

test("failed sampler discovery never caches a partial set or starts a privileged helper", async () => {
  const commands: string[] = [];
  const collector = new PowerMetricsCollector({ enabled: true }, {
    platform: () => "darwin",
    runCommand: async (file) => { commands.push(file); return " cpu_power power\n ane_power power"; },
  });
  assert.equal((await collector.sample(new AbortController().signal)).status, "unavailable");
  assert.equal((await collector.sample(new AbortController().signal)).status, "unavailable");
  assert.deepEqual(commands, ["/usr/bin/powermetrics", "/usr/bin/powermetrics"]);
});

test("privileged sampler is noninteractive and finite; malformed later samples stay stale", async () => {
  let now = 1000;
  let nativeSamples = 0;
  const collector = new PowerMetricsCollector({ enabled: true, intervalMs: 60_000 }, {
    platform: () => "darwin", now: () => now,
    runCommand: async (file, args) => {
      if (file === "/usr/bin/powermetrics") return " cpu_power cpu\n gpu_power gpu\n ane_power ane\n thermal pressure";
      assert.equal(file, "/usr/bin/sudo");
      assert.deepEqual(args.slice(0, 2), ["-n", "/usr/bin/powermetrics"]);
      assert.deepEqual(args.slice(6, 10), ["-n", "1", "-i", "1000"]);
      assert.equal(args[args.length - 1], "cpu_power,gpu_power,ane_power,thermal");
      return ++nativeSamples === 1 ? sample : "malformed data";
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
  assert.match(stale.error!, /header/);
});

test("authorization failures are actionable and never manufacture an idle ANE", async () => {
  const collector = new PowerMetricsCollector({ enabled: true }, {
    platform: () => "darwin",
    runCommand: async (file) => {
      if (file === "/usr/bin/powermetrics") return "cpu_power cpu\ngpu_power gpu\n";
      throw new Error("sudo: a password is required");
    },
  });
  const result = await collector.sample(new AbortController().signal);
  assert.equal(result.status, "unavailable");
  assert.equal(result.ane_power_mw, null);
  assert.equal(result.sampled_at_unix_ms, null);
  assert.match(result.error!, /sudo -v/);
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
