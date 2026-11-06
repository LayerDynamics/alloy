import test from "node:test";
import assert from "node:assert/strict";
import { GpuMetricsCollector, emptyGpuMetrics, parseGpuMetricsOutput } from "../src/gpu-metrics.js";

// Synthetic IORegistry values in the emitted root-array schema; the live Mac
// test below additionally exercises the real driver without privilege elevation.
const wrap = (body: string): string => '<?xml version="1.0" encoding="UTF-8"?>\n'
  + '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
  + '<plist version="1.0"><array>' + body + '</array></plist>';
const entry = (statistics: string, identity = '<key>IORegistryEntryID</key><integer>4294968001</integer>'): string => '<dict>'
  + identity + '<key>IORegistryEntryName</key><string>AGXAccelerator</string>'
  + '<key>IOObjectClass</key><string>AGXAcceleratorG16X</string>'
  + '<key>PerformanceStatistics</key><dict>' + statistics + '</dict></dict>';
const sample = wrap(entry('<key>Device Utilization %</key><integer>37</integer>'));
const signal = (): AbortSignal => new AbortController().signal;

test("IORegistry GPU parsing preserves identities, zero, and native plist data without confusing renderer load", () => {
  const output = wrap('<dict><key>IORegistryEntryID</key><integer>4294968001</integer>'
    + '<key>IORegistryEntryName</key><string>AGX &amp; GPU</string>'
    + '<key>IOObjectClass</key><string>AGXAcceleratorG16X</string>'
    + '<key>vendor-id</key><data>awYAAA==</data><key>online</key><true/>'
    + '<key>PerformanceStatistics</key><dict><key>Device Utilization %</key><integer>0</integer>'
    + '<key>Renderer Utilization %</key><integer>70</integer><key>Tiler Utilization %</key><integer>35</integer>'
    + '</dict></dict>');
  const result = parseGpuMetricsOutput(output, 1234);
  assert.equal(result.source, "ioreg");
  assert.equal(result.status, "ready");
  assert.equal(result.sampled_at_unix_ms, 1234);
  assert.equal(result.age_ms, 0);
  assert.equal(result.utilization_percent, 0);
  assert.equal(result.error, null);
  assert.deepEqual(result.devices, [{ id: "4294968001", name: "AGX & GPU", utilization_percent: 0 }]);
  assert.equal(emptyGpuMetrics("starting").utilization_percent, null);
});

test("GPU device counters take precedence and alternate driver keys are used only when the primary key is absent", () => {
  assert.equal(parseGpuMetricsOutput(wrap(entry('<key>GPU Activity(%)</key><real>12.5</real>'))).utilization_percent, 12.5);
  assert.equal(parseGpuMetricsOutput(wrap(entry('<key>Device Utilization %</key><integer>0</integer>'
    + '<key>GPU Activity(%)</key><integer>80</integer>'))).utilization_percent, 0);
  const invalid = parseGpuMetricsOutput(wrap(entry('<key>Device Utilization %</key><string>bad</string>'
    + '<key>GPU Activity(%)</key><integer>80</integer>')));
  assert.equal(invalid.status, "unavailable");
  assert.equal(invalid.utilization_percent, null);
});

test("missing, invalid and component-only counters stay unavailable instead of manufacturing an idle or busy GPU", () => {
  for (const counter of ["", '<key>Renderer Utilization %</key><integer>80</integer>',
    '<key>Tiler Utilization %</key><integer>80</integer>',
    ...['<integer>-1</integer>', '<integer>101</integer>', '<real>nan</real>', '<real>inf</real>',
      '<string>20</string>', '<true/>'].map((value) => '<key>Device Utilization %</key>' + value)]) {
    const result = parseGpuMetricsOutput(wrap(entry(counter)));
    assert.equal(result.status, "unavailable");
    assert.equal(result.sampled_at_unix_ms, null);
    assert.equal(result.utilization_percent, null);
    assert.equal(result.devices[0]!.utilization_percent, null);
    assert.match(result.error!, /valid device utilization/);
  }
  const noDevices = parseGpuMetricsOutput(wrap(""));
  assert.equal(noDevices.status, "unavailable");
  assert.deepEqual(noDevices.devices, []);
  assert.match(noDevices.error!, /no GPU accelerator/);
});

test("multiple GPUs retain individual readings and report the busiest measured device rather than a sum or average", () => {
  const result = parseGpuMetricsOutput(wrap(
    entry('<key>Device Utilization %</key><integer>35</integer>', '<key>IORegistryEntryID</key><integer>1</integer>')
    + entry('<key>GPU Activity(%)</key><integer>80</integer>', '<key>IORegistryEntryID</key><string>0xabc</string>')
    + entry('<key>Tiler Utilization %</key><integer>90</integer>', '<key>IORegistryEntryID</key><integer>3</integer>')));
  assert.equal(result.status, "ready");
  assert.equal(result.utilization_percent, 80);
  assert.deepEqual(result.devices.map((device) => [device.id, device.utilization_percent]), [["1", 35], ["0xabc", 80], ["3", null]]);
});

test("registry identities never silently round unsafe IDs and class names provide an explicit fallback", () => {
  const result = parseGpuMetricsOutput(wrap('<dict><key>IORegistryEntryID</key><integer>9007199254740993</integer>'
    + '<key>IOObjectClass</key><string>AGXAcceleratorG16X</string><key>PerformanceStatistics</key><dict>'
    + '<key>Device Utilization %</key><integer>10</integer></dict></dict>'));
  assert.deepEqual(result.devices[0], { id: "fallback:0:AGXAcceleratorG16X", name: "AGXAcceleratorG16X", utilization_percent: 10 });
});

test("GPU parsing only reads matched accelerator roots and rejects malformed or oversized plist documents", () => {
  const nested = wrap('<dict><key>IORegistryEntryChildren</key><array>'
    + entry('<key>Device Utilization %</key><integer>99</integer>') + '</array></dict>');
  assert.equal(parseGpuMetricsOutput(nested).utilization_percent, null);
  assert.throws(() => parseGpuMetricsOutput('<plist version="1.0"><dict/></plist>'), /array/);
  assert.throws(() => parseGpuMetricsOutput(wrap('<string>wrong shape</string>')), /dictionary/);
  assert.throws(() => parseGpuMetricsOutput("not a plist"), /IORegistry/);
  assert.throws(() => parseGpuMetricsOutput("x".repeat(1024 * 1024 + 1)), /too large/);
});

test("GPU collector uses only the bounded unprivileged IORegistry command and freshens each real sample", async () => {
  let now = 1000;
  const calls: Array<{ file: string; args: string[]; timeout: number | undefined }> = [];
  const controller = new AbortController();
  const collector = new GpuMetricsCollector({
    platform: () => "darwin", now: () => now,
    runCommand: async (file, args, receivedSignal, timeout) => {
      calls.push({ file, args, timeout });
      assert.equal(receivedSignal, controller.signal);
      return sample;
    },
  });
  const first = await collector.sample(controller.signal);
  assert.equal(first.status, "ready");
  assert.equal(first.sampled_at_unix_ms, 1000);
  now = 2300;
  const second = await collector.sample(controller.signal);
  assert.equal(second.sampled_at_unix_ms, 2300);
  assert.equal(second.age_ms, 0);
  assert.deepEqual(calls, [0, 1].map(() => ({
    file: "/usr/sbin/ioreg", args: ["-a", "-r", "-d", "1", "-c", "IOAccelerator"], timeout: 2000,
  })));
});

test("unsupported platforms and already-aborted sampling never invoke a helper", async () => {
  let calls = 0;
  const runCommand = async (): Promise<string> => { calls++; return sample; };
  const unsupported = await new GpuMetricsCollector({ platform: () => "linux", runCommand }).sample(signal());
  assert.equal(unsupported.status, "unavailable");
  assert.equal(unsupported.utilization_percent, null);
  const controller = new AbortController();
  controller.abort();
  const stopped = await new GpuMetricsCollector({ platform: () => "darwin", runCommand }).sample(controller.signal);
  assert.equal(stopped.status, "stopped");
  assert.equal(stopped.sampled_at_unix_ms, null);
  assert.equal(calls, 0);
});

test("GPU failures preserve dated readings as stale and recover on a new valid sample", async () => {
  let now = 1000;
  let output: string | Error = new Error("initial read failed");
  const collector = new GpuMetricsCollector({ platform: () => "darwin", now: () => now,
    runCommand: async () => { if (output instanceof Error) throw output; return output; },
  });
  assert.equal((await collector.sample(signal())).status, "unavailable");
  output = sample;
  const ready = await collector.sample(signal());
  assert.equal(ready.utilization_percent, 37);
  now = 3000;
  output = wrap(entry(""));
  const stale = await collector.sample(signal());
  assert.equal(stale.status, "stale");
  assert.equal(stale.sampled_at_unix_ms, 1000);
  assert.equal(stale.age_ms, 2000);
  assert.equal(stale.utilization_percent, 37);
  assert.deepEqual(stale.devices, ready.devices);
  assert.match(stale.error!, /valid device utilization/);
  now = 4500;
  output = new Error("registry unavailable");
  assert.equal((await collector.sample(signal())).age_ms, 3500);
  now = 5000;
  output = sample.replace('<integer>37</integer>', '<integer>0</integer>');
  const recovered = await collector.sample(signal());
  assert.equal(recovered.status, "ready");
  assert.equal(recovered.utilization_percent, 0);
  assert.equal(recovered.sampled_at_unix_ms, 5000);
  assert.equal(recovered.age_ms, 0);
  assert.equal(recovered.error, null);
});

test("concurrent samples share one helper and cancellation settles it before allowing another sample", async () => {
  let calls = 0;
  let active = 0;
  let reject!: (error: Error) => void;
  const controller = new AbortController();
  const collector = new GpuMetricsCollector({ platform: () => "darwin",
    runCommand: async (_file, _args, receivedSignal) => {
      calls++;
      if (calls > 1) return sample;
      active++;
      try {
        return await new Promise<string>((_resolve, rejectPromise) => {
          reject = rejectPromise;
          receivedSignal.addEventListener("abort", () => reject(new Error("GPU helper stopped")), { once: true });
        });
      } finally { active--; }
    },
  });
  const first = collector.sample(controller.signal);
  assert.equal(collector.sample(controller.signal), first);
  assert.equal(active, 1);
  controller.abort();
  assert.equal((await first).status, "stopped");
  assert.equal(active, 0);
  assert.equal(calls, 1);
  const recovered = await collector.sample(signal());
  assert.equal(recovered.status, "ready");
  assert.equal(calls, 2);
});

test("a helper result arriving after cancellation cannot become a fresh GPU sample", async () => {
  const controller = new AbortController();
  const collector = new GpuMetricsCollector({ platform: () => "darwin",
    runCommand: async () => { controller.abort(); return sample; },
  });
  const result = await collector.sample(controller.signal);
  assert.equal(result.status, "stopped");
  assert.equal(result.sampled_at_unix_ms, null);
  assert.equal(result.utilization_percent, null);
});

test("Apple Silicon exposes a real GPU utilization sample without privilege elevation", {
  skip: process.platform !== "darwin" || process.arch !== "arm64",
}, async () => {
  assert.equal(typeof process.geteuid, "function");
  assert.notEqual(process.geteuid!(), 0, "The GPU acceptance probe must run as a normal user.");
  const collector = new GpuMetricsCollector();
  const result = await collector.sample(signal());
  assert.equal(result.status, "ready", result.error ?? "No GPU measurement was returned.");
  assert.equal(result.source, "ioreg");
  assert.equal(result.error, null);
  assert.ok(result.devices.length > 0);
  assert.ok(result.sampled_at_unix_ms !== null);
  assert.ok(result.utilization_percent !== null && Number.isFinite(result.utilization_percent));
  assert.ok(result.utilization_percent >= 0 && result.utilization_percent <= 100);
});
