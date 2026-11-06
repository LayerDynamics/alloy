import type { HostMetricsSnapshot } from "./host-metrics.js";

/** Additive API v1 daemon.metrics response. Byte counts share one unified budget. */
export type DaemonMetrics = {
  schema_version: 1;
  sampled_at_unix_ms: number;
  process_id: number;
  admission: {
    budget_bytes: number;
    reserved_bytes: number;
    available_bytes: number;
    active_reservations: number;
    admissions_paused: boolean;
  };
  execution: {
    active_request_id: string | null;
    cancellation_requested: boolean;
  };
  residency: {
    worker_alive: boolean;
    /** Age at sampled_at_unix_ms; add the enclosing snapshot age when displaying. */
    sample_age_ms: number;
    models_resident: number;
    models: Array<{
      artifact_id: string;
      weights_reserved_bytes: number;
      kv_reserved_bytes: number;
      pinned: boolean;
      leased: boolean;
    }>;
  };
  memory_guard: {
    tier: string;
    pressure: string;
    emergency: boolean;
    usage_bytes: number | null;
    static_ceiling_bytes: number | null;
    dynamic_ceiling_bytes: number | null;
    metal_ceiling_bytes: number | null;
    hard_limit_bytes: number | null;
    soft_bytes: number | null;
    hard_bytes: number | null;
    abort_limit_bytes: number | null;
    admit_line_bytes: number | null;
    binding: string | null;
    /** Age at sampled_at_unix_ms; raw source age is retained in JSON captures. */
    sample_age_ms: number | null;
    last_error: string | null;
    cache_reclaim_requested: boolean;
  };
};

export type AlloyMetricsSnapshot = {
  status: "connected" | "unavailable";
  sample_age_ms: number | null;
  error: string | null;
  /** Last successful sample is retained, with its age, when a connection fails. */
  metrics: DaemonMetrics | null;
};

/** JSON/NDJSON capture contract used by alloy monitor. */
export type MonitorSnapshot = {
  schema_version: 1;
  sampled_at_unix_ms: number;
  host: HostMetricsSnapshot | null;
  host_sample_age_ms: number | null;
  host_error: string | null;
  alloy: AlloyMetricsSnapshot;
};

function record(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`invalid daemon.metrics response: ${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function integer(value: unknown, path: string, nullable = false): void {
  if (nullable && value === null) return;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`invalid daemon.metrics response: ${path} must be a nonnegative integer${nullable ? " or null" : ""}`);
  }
}

function boolean(value: unknown, path: string): void {
  if (typeof value !== "boolean") {
    throw new Error(`invalid daemon.metrics response: ${path} must be a boolean`);
  }
}

function string(value: unknown, path: string, nullable = false): void {
  if (nullable && value === null) return;
  if (typeof value !== "string") {
    throw new Error(`invalid daemon.metrics response: ${path} must be a string${nullable ? " or null" : ""}`);
  }
}

/** Reject malformed/older payloads rather than inventing zero-valued metrics. */
export function parseDaemonMetrics(value: unknown): DaemonMetrics {
  const root = record(value, "result");
  if (root.schema_version !== 1) {
    throw new Error("unsupported daemon.metrics schema; update alloy and alloyd together");
  }
  integer(root.sampled_at_unix_ms, "sampled_at_unix_ms");
  integer(root.process_id, "process_id");
  const admission = record(root.admission, "admission");
  for (const key of ["budget_bytes", "reserved_bytes", "available_bytes", "active_reservations"]) {
    integer(admission[key], `admission.${key}`);
  }
  boolean(admission.admissions_paused, "admission.admissions_paused");
  const execution = record(root.execution, "execution");
  string(execution.active_request_id, "execution.active_request_id", true);
  boolean(execution.cancellation_requested, "execution.cancellation_requested");
  const residency = record(root.residency, "residency");
  boolean(residency.worker_alive, "residency.worker_alive");
  integer(residency.sample_age_ms, "residency.sample_age_ms");
  integer(residency.models_resident, "residency.models_resident");
  if (!Array.isArray(residency.models)) {
    throw new Error("invalid daemon.metrics response: residency.models must be an array");
  }
  for (const item of residency.models) {
    const model = record(item, "residency.models[]");
    string(model.artifact_id, "model.artifact_id");
    integer(model.weights_reserved_bytes, "model.weights_reserved_bytes");
    integer(model.kv_reserved_bytes, "model.kv_reserved_bytes");
    boolean(model.pinned, "model.pinned");
    boolean(model.leased, "model.leased");
  }
  const guard = record(root.memory_guard, "memory_guard");
  for (const key of ["tier", "pressure"]) string(guard[key], `memory_guard.${key}`);
  for (const key of ["emergency", "cache_reclaim_requested"]) boolean(guard[key], `memory_guard.${key}`);
  for (const key of ["usage_bytes", "static_ceiling_bytes", "dynamic_ceiling_bytes", "metal_ceiling_bytes",
    "hard_limit_bytes", "soft_bytes", "hard_bytes", "abort_limit_bytes", "admit_line_bytes", "sample_age_ms"]) {
    integer(guard[key], `memory_guard.${key}`, true);
  }
  string(guard.binding, "memory_guard.binding", true);
  string(guard.last_error, "memory_guard.last_error", true);
  return value as DaemonMetrics;
}
