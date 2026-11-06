import os from "node:os";
import { parsePowerMetricsPlist, runMetricCommand } from "./powermetrics.js";

export type GpuMetricsSnapshot = {
  source: "ioreg";
  status: "starting" | "ready" | "stale" | "unavailable" | "stopped";
  sampled_at_unix_ms: number | null;
  age_ms: number | null;
  /** Driver-reported device utilization; the busiest reporting GPU if several exist. */
  utilization_percent: number | null;
  devices: Array<{ id: string; name: string; utilization_percent: number | null }>;
  error: string | null;
};

export function emptyGpuMetrics(
  status: GpuMetricsSnapshot["status"],
  error: string | null = null,
): GpuMetricsSnapshot {
  return {
    source: "ioreg", status, sampled_at_unix_ms: null, age_ms: null,
    utilization_percent: null, devices: [], error,
  };
}

function dictionary(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function name(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function percent(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100 ? value : null;
}

/**
 * IORegistry driver counters are not a stable Apple telemetry API. These are
 * the device-level keys also read by Stats; renderer/tiler load is distinct:
 * https://github.com/exelban/stats/blob/master/Modules/GPU/reader.swift
 * Only roots returned by the targeted ioreg query are inspected. Descendant
 * per-process statistics must never be substituted for system GPU utilization.
 */
export function parseGpuMetricsOutput(output: string, sampledAt = Date.now()): GpuMetricsSnapshot {
  let root: unknown;
  try {
    // Reuse the bounded XML plist reader (including data/date support) rather
    // than spawning plutil or introducing an unbounded general XML parser.
    root = parsePowerMetricsPlist(output);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(detail.replaceAll("powermetrics", "IORegistry"));
  }
  if (!Array.isArray(root)) throw new Error("IORegistry GPU sample must contain an array of accelerator entries.");
  const devices = root.map((value: unknown, index: number) => {
    const entry = dictionary(value);
    if (entry === null) throw new Error("IORegistry GPU entry must be a dictionary.");
    const deviceName = name(entry.IORegistryEntryName) ?? name(entry.IOObjectClass) ?? name(entry.IOClass) ?? "GPU " + (index + 1);
    const registryId = entry.IORegistryEntryID;
    // A rounded 64-bit registry ID would falsely identify a device. Keep
    // exact textual/safe numeric identities, otherwise mark the fallback.
    const id = name(registryId) ?? (typeof registryId === "number" && Number.isSafeInteger(registryId) && registryId >= 0
      ? String(registryId) : "fallback:" + index + ":" + deviceName);
    const statistics = dictionary(entry.PerformanceStatistics);
    const utilization = statistics === null ? null : percent(Object.hasOwn(statistics, "Device Utilization %")
      ? statistics["Device Utilization %"] : statistics["GPU Activity(%)"]);
    return { id, name: deviceName, utilization_percent: utilization };
  });
  const measured = devices.flatMap((device) => device.utilization_percent === null ? [] : [device.utilization_percent]);
  if (measured.length === 0) {
    return {
      ...emptyGpuMetrics("unavailable", devices.length === 0
        ? "IORegistry reported no GPU accelerator entries."
        : "This GPU driver did not report a valid device utilization counter."),
      devices,
    };
  }
  return {
    ...emptyGpuMetrics("ready"), sampled_at_unix_ms: sampledAt, age_ms: 0,
    utilization_percent: measured.reduce((maximum, value) => Math.max(maximum, value), 0), devices,
  };
}

/** Unprivileged, bounded IORegistry reads; power authorization is independent. */
export class GpuMetricsCollector {
  #last = emptyGpuMetrics("starting");
  #pending: Promise<GpuMetricsSnapshot> | null = null;
  #platform: () => string;
  #runCommand: typeof runMetricCommand;
  #now: () => number;

  constructor(dependencies: {
    platform?: () => string;
    runCommand?: typeof runMetricCommand;
    now?: () => number;
  } = {}) {
    this.#platform = dependencies.platform ?? os.platform;
    this.#runCommand = dependencies.runCommand ?? runMetricCommand;
    this.#now = dependencies.now ?? Date.now;
  }

  sample(signal: AbortSignal): Promise<GpuMetricsSnapshot> {
    // Host polling normally serializes calls. Preserve that ownership even
    // when another caller requests the same sample before its helper exits.
    if (this.#pending === null) {
      this.#pending = this.#collect(signal).finally(() => { this.#pending = null; });
    }
    return this.#pending;
  }

  #failed(error: string, stopped: boolean): GpuMetricsSnapshot {
    this.#last = {
      ...this.#last,
      status: stopped ? "stopped" : this.#last.sampled_at_unix_ms === null ? "unavailable" : "stale",
      age_ms: this.#last.sampled_at_unix_ms === null ? null : Math.max(0, this.#now() - this.#last.sampled_at_unix_ms),
      error: error.slice(0, 400),
    };
    return this.#last;
  }

  async #collect(signal: AbortSignal): Promise<GpuMetricsSnapshot> {
    if (signal.aborted) return this.#failed("GPU sampling was stopped.", true);
    if (this.#platform() !== "darwin") {
      return emptyGpuMetrics("unavailable", "IORegistry GPU utilization is available only on macOS.");
    }
    try {
      // -c includes subclasses; -r roots the output at matching services;
      // -d 1 omits descendants and their per-process counters. No sudo/shell.
      const output = await this.#runCommand("/usr/sbin/ioreg", ["-a", "-r", "-d", "1", "-c", "IOAccelerator"], signal, 2000);
      if (signal.aborted) return this.#failed("GPU sampling was stopped.", true);
      const current = parseGpuMetricsOutput(output, this.#now());
      if (current.status !== "ready" && this.#last.sampled_at_unix_ms !== null) {
        return this.#failed(current.error ?? "GPU utilization is unavailable.", false);
      }
      this.#last = current;
      return this.#last;
    } catch (error) {
      return this.#failed(error instanceof Error ? error.message : String(error), signal.aborted);
    }
  }
}
