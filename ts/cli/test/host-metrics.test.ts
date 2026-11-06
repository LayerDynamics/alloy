import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import {
  cpuCounters,
  cpuUtilization,
  createHostMetricsCollector,
  parseMacSwap,
  parseMacVm,
} from "../src/host-metrics.js";

test("CPU utilization is interval-based and normalized across all logical cores", () => {
  const before = cpuCounters([
    { times: { user: 100, nice: 0, sys: 50, idle: 850, irq: 0 } },
    { times: { user: 200, nice: 0, sys: 100, idle: 700, irq: 0 } },
  ]);
  const after = cpuCounters([
    { times: { user: 150, nice: 0, sys: 50, idle: 900, irq: 0 } },
    { times: { user: 200, nice: 0, sys: 100, idle: 800, irq: 0 } },
  ]);
  assert.equal(cpuUtilization(before, after), 25);
  assert.equal(cpuUtilization(before, before), null);
  assert.equal(cpuUtilization(after, before), null);
  assert.equal(cpuUtilization({ count: 1, total: 1000, idle: 500 }, after), null);
  assert.equal(cpuUtilization({ count: 0, total: 0, idle: 0 }, { count: 0, total: 0, idle: 0 }), null);
});

test("swap parser preserves reported zero and byte units", () => {
  assert.deepEqual(parseMacSwap("total = 2048.00M used = 512.25M free = 1535.75M (encrypted)"), {
    total: 2 * 1024 ** 3, used: 512.25 * 1024 ** 2, free: 1535.75 * 1024 ** 2,
  });
  assert.deepEqual(parseMacSwap("total = 0.00M used = 0.00M free = 0.00M (encrypted)"), { total: 0, used: 0, free: 0 });
  assert.throws(() => parseMacSwap("permission denied"), /Unrecognized/);
});

test("Mac VM parser uses actual page size and physical compressor footprint", () => {
  assert.deepEqual(parseMacVm(`Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages wired down: 100.
Pages stored in compressor: 900.
Pages occupied by compressor: 200.
`), { wired: 100 * 16384, compressed: 200 * 16384 });
  assert.deepEqual(parseMacVm("Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages wired down: 0.\n"),
    { wired: 0, compressed: null });
  assert.throws(() => parseMacVm("Pages wired down: 100"), /page size/);
});

test("host sampling works without privileged helpers and serializes concurrent polls", async () => {
  const collector = createHostMetricsCollector({ powermetrics: false });
  try {
    const first = collector.sample();
    assert.equal(collector.sample(), first);
    const result = await first;
    assert.ok(result.memory_total_bytes > 0);
    assert.ok(result.memory_free_bytes >= 0);
    assert.equal(result.memory_used_bytes + result.memory_free_bytes, result.memory_total_bytes);
    assert.ok(result.logical_cpu_count > 0);
    assert.ok(result.cpu_utilization_percent !== null && result.cpu_utilization_percent >= 0 && result.cpu_utilization_percent <= 100);
    assert.ok(result.cpu_sample_ms !== null && result.cpu_sample_ms >= 90);
    assert.equal(result.accelerators.source, "ioreport");
    if (process.platform === "darwin" && process.arch === "arm64") {
      assert.equal(result.accelerators.status, "ready", result.accelerators.error ?? "Native telemetry did not become ready");
      assert.ok(result.accelerators.cpu_power_mw !== null);
      assert.ok(result.accelerators.gpu_power_mw !== null);
      assert.ok(result.accelerators.ane_power_mw !== null);
    } else {
      assert.equal(result.accelerators.status, "unavailable");
      assert.equal(result.accelerators.ane_power_mw, null);
    }
  } finally {
    await collector.close();
  }
  await assert.rejects(collector.sample(), /closed/);
});

test("closing a collector interrupts a pending initial sample", async () => {
  const collector = createHostMetricsCollector({ powermetrics: false });
  const sample = collector.sample();
  const rejected = assert.rejects(sample, /abort/i);
  await collector.close();
  await rejected;
});

const gpuRegistry = `<?xml version="1.0"?><plist version="1.0"><array><dict>
<key>IORegistryEntryID</key><integer>42</integer>
<key>IORegistryEntryName</key><string>AGXAccelerator</string>
<key>PerformanceStatistics</key><dict><key>Device Utilization %</key><integer>37</integer></dict>
</dict></array></plist>`;

const nativeHelper = fileURLToPath(new URL("../native/alloy-metrics", import.meta.url));
function nativeFixture(): string {
  return JSON.stringify({
    schema_version: 1, source: "ioreport", effective_uid: 501,
    sampled_at_unix_ms: Date.now(), sample_duration_ms: 500,
    cpu_power_mw: 4200, gpu_power_mw: 750, ane_power_mw: 0,
    gpu_active_percent: 20, gpu_frequency_mhz: 700,
    cpu_clusters: [{ name: "ECPU", active_percent: 42, frequency_mhz: 2064 }, { name: "PCPU", active_percent: 12, frequency_mhz: 3500 }],
    thermal_pressure: "Nominal", disk_read_bytes_per_second: 0, disk_write_bytes_per_second: 0,
    network_in_bytes_per_second: 0, network_out_bytes_per_second: 0,
    power_sources: { cpu: "clpc", gpu: "clpc", ane: "energy_model" }, warnings: [],
  });
}

function memoryFixture(file: string): string | null {
  if (file === "/usr/sbin/sysctl") return "total = 0.00M used = 0.00M free = 0.00M";
  if (file === "/usr/bin/vm_stat") return "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages wired down: 0.\n";
  return null;
}

test("default Mac sampling publishes GPU, CPU/GPU/ANE power, frequencies and thermal data without elevation", async () => {
  const calls: string[] = [];
  const collector = createHostMetricsCollector({}, {
    platform: () => "darwin", architecture: () => "arm64",
    runCommand: async (file, args) => {
      calls.push(file);
      const memory = memoryFixture(file);
      if (memory !== null) return memory;
      if (file === nativeHelper) {
        assert.deepEqual(args, ["--interval-ms", "1000"]);
        return nativeFixture();
      }
      assert.equal(file, "/usr/sbin/ioreg");
      assert.deepEqual(args, ["-a", "-r", "-d", "1", "-c", "IOAccelerator"]);
      return gpuRegistry;
    },
  });
  try {
    const sample = await collector.sample();
    assert.equal(sample.gpu?.source, "ioreg");
    assert.equal(sample.gpu?.status, "ready");
    assert.equal(sample.gpu?.utilization_percent, 37);
    assert.equal(sample.accelerators.source, "ioreport");
    assert.equal(sample.accelerators.status, "ready");
    assert.equal(sample.accelerators.cpu_power_mw, 4200);
    assert.equal(sample.accelerators.gpu_power_mw, 750);
    assert.equal(sample.accelerators.ane_power_mw, 0);
    assert.equal(sample.accelerators.gpu_frequency_mhz, 700);
    assert.equal(sample.accelerators.cpu_clusters.length, 2);
    assert.equal(sample.accelerators.thermal_pressure, "Nominal");
    assert.deepEqual(calls.sort(), ["/usr/bin/vm_stat", "/usr/sbin/ioreg", "/usr/sbin/sysctl", nativeHelper].sort());
  } finally { await collector.close(); }
});

test("the old power flag is a compatibility alias and cannot select a privileged backend", async () => {
  for (const powermetrics of [false, true]) {
    const calls: string[] = [];
    const collector = createHostMetricsCollector({ powermetrics }, {
      platform: () => "darwin", architecture: () => "arm64",
      runCommand: async (file) => {
        calls.push(file);
        const memory = memoryFixture(file);
        if (memory !== null) return memory;
        if (file === "/usr/sbin/ioreg") return gpuRegistry;
        assert.equal(file, nativeHelper);
        return nativeFixture();
      },
    });
    try {
      const sample = await collector.sample();
      assert.equal(sample.accelerators.source, "ioreport");
      assert.equal(sample.accelerators.status, "ready");
      assert.ok(calls.includes(nativeHelper));
      assert.ok(calls.every((file) => file !== "/usr/bin/sudo" && file !== "/usr/bin/powermetrics"));
    } finally { await collector.close(); }
  }
});

test("a native telemetry failure cannot hide independent GPU utilization or request authorization", async () => {
  const collector = createHostMetricsCollector({ powermetrics: true }, {
    platform: () => "darwin", architecture: () => "arm64",
    runCommand: async (file) => {
      const memory = memoryFixture(file);
      if (memory !== null) return memory;
      if (file === "/usr/sbin/ioreg") return gpuRegistry;
      assert.equal(file, nativeHelper);
      throw new Error("IOReport counters unavailable");
    },
  });
  try {
    const sample = await collector.sample();
    assert.equal(sample.gpu?.status, "ready");
    assert.equal(sample.gpu?.utilization_percent, 37);
    assert.equal(sample.gpu?.error, null);
    assert.equal(sample.accelerators.status, "unavailable");
    assert.equal(sample.accelerators.gpu_power_mw, null);
    assert.match(sample.accelerators.error!, /IOReport counters unavailable/);
    assert.doesNotMatch(sample.accelerators.error!, /sudo|password|authoriz/i);
    assert.deepEqual(sample.errors, []);
  } finally { await collector.close(); }
});

test("closing the host collector waits for its GPU probe to stop and rejects the cancelled frame", async () => {
  let signalStarted!: () => void;
  const started = new Promise<void>((resolve) => { signalStarted = resolve; });
  let stopped = false;
  const collector = createHostMetricsCollector({ powermetrics: false }, {
    platform: () => "darwin", architecture: () => "arm64",
    runCommand: async (file, _args, signal) => {
      const memory = memoryFixture(file);
      if (memory !== null) return memory;
      if (file === nativeHelper) return nativeFixture();
      assert.equal(file, "/usr/sbin/ioreg");
      return new Promise<string>((_resolve, reject) => {
        signal.addEventListener("abort", () => {
          stopped = true;
          reject(new Error("GPU probe aborted"));
        }, { once: true });
        signalStarted();
      });
    },
  });
  const rejected = assert.rejects(collector.sample(), /abort/i);
  await started;
  await collector.close();
  await rejected;
  assert.equal(stopped, true);
});
