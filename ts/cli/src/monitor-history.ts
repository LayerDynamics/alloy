import type { MonitorSnapshot } from "./monitor-types.js";
import { monitorGpuReading } from "./monitor-gpu.js";

export const MONITOR_HISTORY_MAX_AGE_MS = 5000;
const MAX_SAMPLES = 60;

/** One display frame. A null is a gap, never an inferred zero or interpolation. */
export type MonitorHistorySample = {
  at_unix_ms: number;
  cpu_percent: number | null;
  /** Non-free RAM divided by physical RAM; this is not memory pressure. */
  memory_percent: number | null;
  gpu_percent: number | null;
  /** Estimated power, not ANE utilization. */
  ane_power_mw: number | null;
  /** May exceed 100 after a live budget reduction while resources remain owned. */
  reserved_percent: number | null;
};

function nonnegative(value: number | null): value is number {
  return value !== null && Number.isFinite(value) && value >= 0;
}

function percent(value: number | null): number | null {
  return nonnegative(value) && value <= 100 ? value : null;
}

function ratio(numerator: number, denominator: number): number | null {
  if (!nonnegative(numerator) || !nonnegative(denominator) || denominator === 0) return null;
  const result = numerator / denominator * 100;
  return Number.isFinite(result) ? result : null;
}

function validTimestamp(timestamp: number | null, now: number): timestamp is number {
  return timestamp !== null && Number.isSafeInteger(timestamp) && timestamp >= 0 && timestamp <= now;
}

function isNew(timestamp: number | null, previous: number | null, now: number): timestamp is number {
  return validTimestamp(timestamp, now) && (previous === null || timestamp > previous);
}

function isFresh(timestamp: number | null, reportedAge: number | null, now: number): boolean {
  return validTimestamp(timestamp, now)
    && now - timestamp <= MONITOR_HISTORY_MAX_AGE_MS
    && nonnegative(reportedAge) && reportedAge <= MONITOR_HISTORY_MAX_AGE_MS;
}

/** Bounded samples with independent freshness for host, GPU, power and daemon. */
export class MonitorHistory {
  #samples: MonitorHistorySample[] = [];
  #hostIdentity: string | null = null;
  #daemonPid: number | null = null;
  #hostTimestamp: number | null = null;
  #hardwareTimestamp: number | null = null;
  #gpuTimestamp: number | null = null;
  #gpuKey: string | null = null;
  #daemonTimestamp: number | null = null;
  #frameTimestamp: number | null = null;

  constructor(private readonly capacity = MAX_SAMPLES) {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > MAX_SAMPLES) {
      throw new RangeError("Monitor history capacity must be an integer from 1 to 60.");
    }
  }

  push(snapshot: MonitorSnapshot): void {
    const now = snapshot.sampled_at_unix_ms;
    if (!validTimestamp(now, now) || now === this.#frameTimestamp) return;
    const host = snapshot.host;
    if (host !== null) {
      const identity = JSON.stringify([host.hostname, host.platform, host.architecture]);
      if (this.#hostIdentity !== null && this.#hostIdentity !== identity) {
        this.#samples = [];
        this.#hostTimestamp = this.#hardwareTimestamp = this.#daemonTimestamp = null;
        this.#gpuTimestamp = this.#gpuKey = null;
        this.#daemonPid = null;
      }
      this.#hostIdentity = identity;
    }
    const metrics = snapshot.alloy.metrics;
    if (metrics !== null) {
      if (this.#daemonPid !== null && this.#daemonPid !== metrics.process_id) {
        // Host/accelerator trends remain useful; do not join two daemon ledgers.
        this.#samples = this.#samples.map((sample) => ({ ...sample, reserved_percent: null }));
        this.#daemonTimestamp = null;
      }
      this.#daemonPid = metrics.process_id;
    }
    const next: MonitorHistorySample = {
      at_unix_ms: now, cpu_percent: null, memory_percent: null,
      gpu_percent: null, ane_power_mw: null, reserved_percent: null,
    };

    if (host !== null) {
      const hostNew = isNew(host.sampled_at_unix_ms, this.#hostTimestamp, now);
      if (hostNew) this.#hostTimestamp = host.sampled_at_unix_ms;
      const hostFresh = snapshot.host_error === null
        && isFresh(host.sampled_at_unix_ms, snapshot.host_sample_age_ms, now);
      if (hostNew && hostFresh) {
        next.cpu_percent = percent(host.cpu_utilization_percent);
        next.memory_percent = percent(ratio(host.memory_used_bytes, host.memory_total_bytes));
      }
      const gpu = monitorGpuReading(host);
      if (this.#gpuKey !== gpu.history_key) {
        // Device utilization and active residency are different measurements.
        // A device/source change must not relabel the old graph as the new one.
        this.#samples = this.#samples.map((sample) => ({ ...sample, gpu_percent: null }));
        this.#gpuTimestamp = null;
        this.#gpuKey = gpu.history_key;
      }
      const gpuNew = isNew(gpu.sampled_at_unix_ms, this.#gpuTimestamp, now);
      if (gpuNew) this.#gpuTimestamp = gpu.sampled_at_unix_ms;
      if (hostFresh && gpuNew && gpu.status === "ready" && gpu.error === null
          && isFresh(gpu.sampled_at_unix_ms, gpu.age_ms, now)) {
        next.gpu_percent = percent(gpu.percent);
      }
      const hardware = host.accelerators;
      const hardwareNew = isNew(hardware.sampled_at_unix_ms, this.#hardwareTimestamp, now);
      if (hardwareNew) this.#hardwareTimestamp = hardware.sampled_at_unix_ms;
      if (hostFresh && hardwareNew && hardware.status === "ready" && hardware.error === null
          && isFresh(hardware.sampled_at_unix_ms, hardware.age_ms, now)) {
        next.ane_power_mw = nonnegative(hardware.ane_power_mw) ? hardware.ane_power_mw : null;
      }
    }
    if (metrics !== null) {
      const daemonNew = isNew(metrics.sampled_at_unix_ms, this.#daemonTimestamp, now);
      if (daemonNew) this.#daemonTimestamp = metrics.sampled_at_unix_ms;
      if (daemonNew && snapshot.alloy.status === "connected" && snapshot.alloy.error === null
          && isFresh(metrics.sampled_at_unix_ms, snapshot.alloy.sample_age_ms, now)) {
        next.reserved_percent = ratio(metrics.admission.reserved_bytes, metrics.admission.budget_bytes);
      }
    }
    this.#samples.push(next);
    if (this.#samples.length > this.capacity) this.#samples.shift();
    this.#frameTimestamp = now;
  }

  /** Defensive copies prevent a renderer from changing retained measurements. */
  snapshot(): readonly MonitorHistorySample[] {
    return this.#samples.map((sample) => ({ ...sample }));
  }
}
