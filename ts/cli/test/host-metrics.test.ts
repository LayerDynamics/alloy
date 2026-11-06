import test from "node:test";
import assert from "node:assert/strict";
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
    assert.equal(result.accelerators.status, "disabled");
    assert.equal(result.accelerators.ane_power_mw, null);
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
