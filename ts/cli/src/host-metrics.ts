import os from "node:os";
import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { setTimeout as delay } from "node:timers/promises";
import { PowerMetricsCollector, runMetricCommand, type PowerMetricsSnapshot } from "./powermetrics.js";

export type HostMetricsSnapshot = {
  sampled_at_unix_ms: number;
  platform: string;
  architecture: string;
  hostname: string;
  logical_cpu_count: number;
  cpu_utilization_percent: number | null;
  cpu_sample_ms: number | null;
  load_average: number[];
  uptime_seconds: number;
  memory_total_bytes: number;
  memory_free_bytes: number;
  /** Total minus OS-reported free memory; includes caches, not pressure. */
  memory_used_bytes: number;
  memory_available_bytes: number | null;
  swap_total_bytes: number | null;
  swap_used_bytes: number | null;
  swap_free_bytes: number | null;
  wired_bytes: number | null;
  compressed_bytes: number | null;
  errors: string[];
  accelerators: PowerMetricsSnapshot;
};

type CpuCounters = { count: number; total: number; idle: number };

export function cpuCounters(cpus: ReadonlyArray<Pick<os.CpuInfo, "times">>): CpuCounters {
  return cpus.reduce((sum, { times }) => ({
    count: sum.count + 1,
    total: sum.total + times.user + times.nice + times.sys + times.idle + times.irq,
    idle: sum.idle + times.idle,
  }), { count: 0, total: 0, idle: 0 });
}

/** Aggregate all logical CPUs on a 0–100 scale, using interval deltas. */
export function cpuUtilization(previous: CpuCounters, current: CpuCounters): number | null {
  if (previous.count === 0 || previous.count !== current.count) return null;
  const total = current.total - previous.total;
  const idle = current.idle - previous.idle;
  if (!Number.isFinite(total) || !Number.isFinite(idle) || total <= 0 || idle < 0 || idle > total) return null;
  return (1 - idle / total) * 100;
}

export function parseMacSwap(text: string): { total: number; used: number; free: number } {
  const values: Partial<{ total: number; used: number; free: number }> = {};
  for (const key of ["total", "used", "free"] as const) {
    const match = new RegExp(`\\b${key}\\s*=\\s*(\\d+(?:\\.\\d+)?)\\s*([KMGT]?)B?\\b`, "i").exec(text);
    if (!match) throw new Error("Unrecognized vm.swapusage output.");
    const unit = match[2]!.toUpperCase();
    const value = Number(match[1]) * 1024 ** (unit ? "KMGT".indexOf(unit) + 1 : 0);
    if (!Number.isFinite(value) || value < 0) throw new Error("Invalid swap counter.");
    values[key] = Math.round(value);
  }
  return values as { total: number; used: number; free: number };
}

export function parseMacVm(text: string): { wired: number | null; compressed: number | null } {
  const match = /page size of (\d+) bytes/.exec(text);
  if (!match) throw new Error("vm_stat did not report its page size.");
  const pageSize = Number(match[1]);
  if (!Number.isSafeInteger(pageSize) || pageSize <= 0) throw new Error("Invalid VM page size.");
  const pages = (name: string): number | null => {
    const counter = new RegExp(`^${name}:\\s*(\\d+)\\.?\\s*$`, "m").exec(text);
    if (!counter) return null;
    const bytes = Number(counter[1]) * pageSize;
    return Number.isSafeInteger(bytes) ? bytes : null;
  };
  // Occupied compressor pages are physical bytes; stored pages are uncompressed
  // contents and must not be added to the physical memory total.
  return { wired: pages("Pages wired down"), compressed: pages("Pages occupied by compressor") };
}

export interface HostMetricsCollector {
  sample(): Promise<HostMetricsSnapshot>;
  close(): Promise<void>;
}

export function createHostMetricsCollector(options: {
  powermetrics: boolean;
  intervalMs?: number;
}): HostMetricsCollector {
  const abort = new AbortController();
  const power = new PowerMetricsCollector({
    enabled: options.powermetrics,
    ...(options.intervalMs === undefined ? {} : { intervalMs: options.intervalMs }),
  });
  let previousCpu = cpuCounters(os.cpus());
  let previousAt = performance.now();
  let pending: Promise<HostMetricsSnapshot> | null = null;

  async function collect(): Promise<HostMetricsSnapshot> {
    if (abort.signal.aborted) throw new Error("Host metrics collector is closed.");
    const errors: string[] = [];
    let swap: { total: number; used: number; free: number } | null = null;
    let vm: { wired: number | null; compressed: number | null } = { wired: null, compressed: null };
    let available: number | null = null;
    const platform = os.platform();
    const memoryProbe = async (): Promise<void> => {
      if (platform === "darwin") {
        const results = await Promise.allSettled([
          runMetricCommand("/usr/sbin/sysctl", ["-n", "vm.swapusage"], abort.signal).then(parseMacSwap),
          runMetricCommand("/usr/bin/vm_stat", [], abort.signal).then(parseMacVm),
        ]);
        const [swapResult, vmResult] = results;
        if (swapResult.status === "fulfilled") swap = swapResult.value;
        else errors.push(`Swap: ${String(swapResult.reason).slice(0, 200)}`);
        if (vmResult.status === "fulfilled") vm = vmResult.value;
        else errors.push(`VM: ${String(vmResult.reason).slice(0, 200)}`);
      } else if (platform === "linux") {
        try {
          const info = await readFile("/proc/meminfo", { encoding: "utf8", signal: abort.signal });
          const bytes = (key: string): number | null => {
            const match = new RegExp(`^${key}:\\s+(\\d+) kB$`, "m").exec(info);
            return match ? Number(match[1]) * 1024 : null;
          };
          const total = bytes("SwapTotal");
          const free = bytes("SwapFree");
          if (total !== null && free !== null && total >= free) swap = { total, free, used: total - free };
          available = bytes("MemAvailable");
        } catch (error) {
          errors.push(`Memory: ${String(error).slice(0, 200)}`);
        }
      }
    };
    // A first one-shot snapshot needs a real CPU interval, not an uptime average.
    const baselineWait = Math.max(0, 100 - (performance.now() - previousAt));
    // Settle every owned probe before close() resolves, even if the warm-up
    // delay aborts first. This prevents a stopped UI from leaving probe children.
    const [memoryResult, powerResult, baselineResult] = await Promise.allSettled([
      memoryProbe(), power.sample(abort.signal), delay(baselineWait, undefined, { signal: abort.signal }),
    ]);
    if (memoryResult.status === "rejected") throw memoryResult.reason;
    if (baselineResult.status === "rejected") throw baselineResult.reason;
    if (powerResult.status === "rejected") throw powerResult.reason;
    const accelerators = {
      ...powerResult.value,
      age_ms: powerResult.value.sampled_at_unix_ms === null ? null
        : Math.max(0, Date.now() - powerResult.value.sampled_at_unix_ms),
    };
    const now = performance.now();
    const currentCpu = cpuCounters(os.cpus());
    const utilization = cpuUtilization(previousCpu, currentCpu);
    const cpuSampleMs = utilization === null ? null : now - previousAt;
    previousCpu = currentCpu;
    previousAt = now;
    const total = os.totalmem();
    const free = os.freemem();
    // TypeScript cannot follow assignments inside an awaited callback.
    const sampledSwap = swap as { total: number; used: number; free: number } | null;
    return {
      sampled_at_unix_ms: Date.now(), platform, architecture: os.arch(), hostname: os.hostname(),
      logical_cpu_count: currentCpu.count,
      cpu_utilization_percent: utilization, cpu_sample_ms: cpuSampleMs,
      load_average: os.loadavg(), uptime_seconds: os.uptime(),
      memory_total_bytes: total, memory_free_bytes: free, memory_used_bytes: Math.max(0, total - free),
      memory_available_bytes: available,
      swap_total_bytes: sampledSwap?.total ?? null, swap_used_bytes: sampledSwap?.used ?? null,
      swap_free_bytes: sampledSwap?.free ?? null,
      wired_bytes: vm.wired, compressed_bytes: vm.compressed, errors, accelerators,
    };
  }

  return {
    sample() {
      // Even accidental concurrent callers share one sampling cycle.
      if (pending === null) {
        pending = collect().finally(() => { pending = null; });
      }
      return pending;
    },
    async close() {
      abort.abort();
      await pending?.catch(() => {});
    },
  };
}
