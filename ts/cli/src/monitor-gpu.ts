import type { HostMetricsSnapshot } from "./host-metrics.js";
import type { PowerMetricsSnapshot } from "./powermetrics.js";

/** The GPU graph has its own source and clock, independent of optional power. */
export type MonitorGpuReading = {
  source: "ioreg" | "powermetrics" | "ioreport" | null;
  history_key: string;
  label: string;
  description: string;
  status: PowerMetricsSnapshot["status"];
  sampled_at_unix_ms: number | null;
  age_ms: number | null;
  percent: number | null;
  error: string | null;
};

export function monitorGpuReading(host: HostMetricsSnapshot | null): MonitorGpuReading {
  if (host?.gpu !== undefined) {
    const gpu = host.gpu;
    const multiple = gpu.devices.length > 1;
    const reporting = gpu.devices.filter((device) => device.utilization_percent !== null
      && Number.isFinite(device.utilization_percent) && device.utilization_percent >= 0 && device.utilization_percent <= 100).length;
    return {
      source: "ioreg",
      history_key: `ioreg:${JSON.stringify(gpu.devices.map((device) => device.id).sort())}`,
      label: multiple ? "GPU busiest" : "GPU use",
      description: multiple ? `IORegistry · busiest of ${reporting} reporting GPU${reporting === 1 ? "" : "s"}` : "IORegistry · device utilization",
      status: gpu.status, sampled_at_unix_ms: gpu.sampled_at_unix_ms,
      age_ms: gpu.age_ms, percent: gpu.utilization_percent, error: gpu.error,
    };
  }
  // Older captures may omit the independent registry sample. Preserve the
  // source and residency meaning; never replace a present failed GPU source.
  const power = host?.accelerators;
  return {
    source: power?.source ?? null,
    history_key: `${power?.source ?? "unavailable"}:active-residency`,
    label: "GPU active", description: `${power?.source === "ioreport" ? "IOReport" : "recorded powermetrics"} · active residency`,
    status: power?.status ?? "unavailable",
    sampled_at_unix_ms: power?.sampled_at_unix_ms ?? null,
    age_ms: power?.age_ms ?? null, percent: power?.gpu_active_percent ?? null,
    error: power?.error ?? null,
  };
}
