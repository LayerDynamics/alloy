import { execFile } from "node:child_process";
import os from "node:os";
import { fileURLToPath } from "node:url";

const MAX_OUTPUT_BYTES = 1024 * 1024;

export type PowerMetricsSnapshot = {
  source: "powermetrics" | "ioreport";
  status: "disabled" | "starting" | "ready" | "stale" | "unavailable" | "stopped";
  sampled_at_unix_ms: number | null;
  age_ms: number | null;
  sample_duration_ms: number | null;
  cpu_power_mw: number | null;
  gpu_power_mw: number | null;
  ane_power_mw: number | null;
  gpu_active_percent: number | null;
  gpu_frequency_mhz: number | null;
  cpu_clusters: Array<{
    name: string;
    active_percent: number | null;
    frequency_mhz: number | null;
  }>;
  thermal_pressure: string | null;
  disk_read_bytes_per_second: number | null;
  disk_write_bytes_per_second: number | null;
  network_in_bytes_per_second: number | null;
  network_out_bytes_per_second: number | null;
  /** Present for native snapshots; legacy recorded samples predate these fields. */
  effective_uid?: number;
  power_sources?: { cpu: string | null; gpu: string | null; ane: string | null };
  warnings?: string[];
  error: string | null;
};

/** No shell, no interactive password prompt, and bounded child output/lifetime. */
export function runMetricCommand(
  file: string,
  args: string[],
  signal: AbortSignal,
  timeoutMs = 2000,
): Promise<string> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new Error("Metric sampling was stopped."));
      return;
    }
    let stopReason: string | null = null;
    let escalation: ReturnType<typeof setTimeout> | undefined;
    const child = execFile(file, args, {
      encoding: "utf8",
      maxBuffer: MAX_OUTPUT_BYTES,
      env: { ...process.env, LC_ALL: "C", LANG: "C" },
    }, (error, stdout, stderr) => {
      clearTimeout(deadline);
      clearTimeout(escalation);
      signal.removeEventListener("abort", onAbort);
      if (stopReason !== null) {
        reject(new Error(stopReason));
        return;
      }
      if (error) {
        const detail = stderr.trim().slice(0, 400) || `Metric helper failed (${error.code ?? error.signal ?? "unknown error"}).`;
        reject(new Error(detail));
      } else {
        resolve(stdout);
      }
    });
    const stop = (reason: string): void => {
      if (stopReason !== null) return;
      stopReason = reason;
      // Each owned helper is finite. Kill escalation also bounds shutdown if
      // a native probe stops responding to the initial termination signal.
      child.kill("SIGTERM");
      escalation = setTimeout(() => { child.kill("SIGKILL"); }, 250);
    };
    const onAbort = (): void => { stop("Metric sampling was stopped."); };
    const deadline = setTimeout(() => { stop("Metric sampling timed out."); }, timeoutMs);
    signal.addEventListener("abort", onAbort, { once: true });
    child.stdin?.end();
  });
}

export function emptyPowerMetrics(
  status: PowerMetricsSnapshot["status"],
  error: string | null = null,
  source: PowerMetricsSnapshot["source"] = "powermetrics",
): PowerMetricsSnapshot {
  return {
    source, status, error,
    sampled_at_unix_ms: null, age_ms: null, sample_duration_ms: null,
    cpu_power_mw: null, gpu_power_mw: null, ane_power_mw: null,
    gpu_active_percent: null, gpu_frequency_mhz: null, cpu_clusters: [],
    thermal_pressure: null,
    disk_read_bytes_per_second: null, disk_write_bytes_per_second: null,
    network_in_bytes_per_second: null, network_out_bytes_per_second: null,
  };
}

type Plist = null | string | number | boolean | Plist[] | { [key: string]: Plist };

/**
 * Parse the XML plist format emitted by Apple's tool. This bounded reader does
 * not fetch DTDs or resolve custom entities. Dates remain strings: plutil's JSON
 * conversion rejects dates, which can occur in powermetrics documents.
 */
export function parsePowerMetricsPlist(xml: string): Plist {
  if (Buffer.byteLength(xml) > MAX_OUTPUT_BYTES) throw new Error("powermetrics sample is too large");
  let source = xml.trim().replace(/^<\?xml[^?]*\?>\s*/, "");
  source = source.replace(/^<!DOCTYPE\s+plist\s+PUBLIC\s+"[^"]*"\s+"[^"]*"\s*>\s*/, "");
  if (source.includes("<!DOCTYPE") || source.includes("<!ENTITY")) {
    throw new Error("unsupported powermetrics plist declaration");
  }
  let offset = 0;
  let nodes = 0;
  const skip = (): void => { while (/\s/.test(source[offset] ?? "") && offset < source.length) offset++; };
  const consume = (text: string): void => {
    skip();
    if (!source.startsWith(text, offset)) throw new Error("malformed powermetrics plist");
    offset += text.length;
  };
  const decode = (text: string): string => text.replace(/&([^;]*);/g, (_match, entity: string) => {
    const named: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
    if (Object.hasOwn(named, entity)) return named[entity]!;
    if (/^#(?:[0-9]+|x[0-9a-fA-F]+)$/.test(entity)) {
      const value = entity.startsWith("#x") ? Number.parseInt(entity.slice(2), 16) : Number(entity.slice(1));
      if (value > 0 && value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff)) {
        return String.fromCodePoint(value);
      }
    }
    throw new Error("unsupported powermetrics XML entity");
  });
  const read = (depth: number): Plist => {
    if (depth > 32 || ++nodes > 30_000) throw new Error("powermetrics plist exceeds parser limits");
    skip();
    const tag = /^<(dict|array|string|key|integer|real|date|data|true|false)\s*(\/?)>/.exec(source.slice(offset));
    if (!tag) throw new Error("unsupported powermetrics plist value");
    offset += tag[0].length;
    const name = tag[1]!;
    const empty = tag[2] === "/";
    if (name === "dict" || name === "array") {
      const result: Plist[] | { [key: string]: Plist } = name === "array" ? [] : Object.create(null) as { [key: string]: Plist };
      if (empty) return result;
      for (;;) {
        skip();
        if (source.startsWith(`</${name}>`, offset)) break;
        if (Array.isArray(result)) {
          result.push(read(depth + 1));
        } else {
          if (!/^<key(?:\s*\/)?\s*>/.test(source.slice(offset))) throw new Error("plist dictionary key required");
          const key = read(depth + 1);
          if (typeof key !== "string" || Object.hasOwn(result, key)) throw new Error("invalid or duplicate plist key");
          result[key] = read(depth + 1);
        }
      }
      consume(`</${name}>`);
      return result;
    }
    if (name === "true" || name === "false") {
      if (!empty) consume(`</${name}>`);
      return name === "true";
    }
    if (empty) {
      if (name === "integer" || name === "real") throw new Error("empty numeric plist value");
      return "";
    }
    const end = source.indexOf(`</${name}>`, offset);
    if (end < 0) throw new Error("unterminated powermetrics plist value");
    const raw = source.slice(offset, end);
    if (raw.includes("<")) throw new Error("nested primitive plist value");
    offset = end + name.length + 3;
    if (name === "integer" || name === "real") {
      const value = raw.trim();
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
        if (name === "real" && /^[+-]?(?:nan|inf(?:inity)?)$/i.test(value)) return null;
        throw new Error("invalid numeric plist value");
      }
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }
    return decode(raw);
  };
  const opening = /^<plist\s+version="1\.0"\s*>/.exec(source);
  if (!opening) throw new Error("powermetrics XML plist header missing");
  offset = opening[0].length;
  const value = read(0);
  consume("</plist>");
  skip();
  if (offset !== source.length) throw new Error("trailing powermetrics plist content");
  return value;
}

function dict(value: Plist | undefined): { [key: string]: Plist } {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function number(value: Plist | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function activePercent(value: { [key: string]: Plist }): number | null {
  const explicit = number(value.used_ratio);
  if (explicit !== null) return explicit <= 1 ? explicit * 100 : null;
  const idle = number(value.idle_ratio);
  const down = value.down_ratio === undefined ? 0 : number(value.down_ratio);
  if (idle === null || down === null || idle > 1 || down > 1 || idle + down > 1.000001) return null;
  return Math.max(0, 1 - idle - down) * 100;
}

function frequencyMhz(value: { [key: string]: Plist }): number | null {
  const hz = number(value.freq_hz);
  if (hz !== null) return hz / 1_000_000;
  return number(value.freq); // The legacy `freq` field is explicitly MHz.
}

function gpuFrequencyMhz(value: { [key: string]: Plist }): number | null {
  // Some Macs report GPU MHz under `freq_hz`, while CPU frequencies remain Hz.
  // Recorded GPU value 749.789 MHz and regression evidence:
  // https://github.com/gavi/monmon/blob/main/tests/fixtures/mac15-9-m3max-macos27.plist
  // https://github.com/gavi/monmon/blob/main/tests/test_power.py
  const raw = number(value.freq_hz);
  if (raw !== null && raw < 100_000) return raw;
  return frequencyMhz(value);
}

/**
 * Map observed plist keys only. Missing and invalid counters never become zero.
 * Primary schema examples:
 * https://github.com/macadmins/osquery-extension/blob/main/tables/socpower/test_powermetrics_output.plist
 * https://github.com/fourlexboehm/wattson/blob/main/src/parse_plist.rs
 * Power is an estimate in mW; GPU active residency is not hardware occupancy.
 */
export function parsePowerMetricsOutput(output: string, sampledAt = Date.now()): PowerMetricsSnapshot {
  if (Buffer.byteLength(output) > MAX_OUTPUT_BYTES) throw new Error("powermetrics output is too large");
  const samples = output.split("\0").filter((part) => part.trim());
  if (samples.length === 0 || samples.length > 4) throw new Error("unexpected powermetrics sample count");
  const root = dict(parsePowerMetricsPlist(samples[samples.length - 1]!));
  const processor = dict(root.processor);
  if (root.invalid === true || processor.invalid === true || root.is_delta === false) {
    throw new Error("powermetrics marked its sample invalid");
  }
  const gpu = dict(root.gpu);
  const disk = dict(root.disk);
  const network = dict(root.network);
  const elapsed = number(root.elapsed_ns);
  const snapshot: PowerMetricsSnapshot = {
    ...emptyPowerMetrics("ready"),
    sampled_at_unix_ms: sampledAt, age_ms: 0,
    sample_duration_ms: elapsed === null ? null : elapsed / 1_000_000,
    cpu_power_mw: number(processor.cpu_power),
    gpu_power_mw: number(processor.gpu_power),
    ane_power_mw: number(processor.ane_power),
    gpu_active_percent: activePercent(gpu),
    gpu_frequency_mhz: gpuFrequencyMhz(gpu),
    thermal_pressure: typeof root.thermal_pressure === "string" ? root.thermal_pressure : null,
    disk_read_bytes_per_second: number(disk.rbytes_per_s ?? root.rbytes_per_s),
    disk_write_bytes_per_second: number(disk.wbytes_per_s ?? root.wbytes_per_s),
    network_in_bytes_per_second: number(network.ibyte_rate ?? network.bytes_received_per_s ?? root.ibyte_rate),
    network_out_bytes_per_second: number(network.obyte_rate ?? network.bytes_sent_per_s ?? root.obyte_rate),
  };
  if (Array.isArray(processor.clusters)) {
    snapshot.cpu_clusters = processor.clusters.slice(0, 64).map((raw, index) => {
      const cluster = dict(raw);
      return {
        name: typeof cluster.name === "string" ? cluster.name : `cluster ${index}`,
        active_percent: activePercent(cluster),
        frequency_mhz: frequencyMhz(cluster),
      };
    });
  }
  if (snapshot.cpu_power_mw === null && snapshot.gpu_power_mw === null
      && snapshot.ane_power_mw === null && snapshot.gpu_active_percent === null
      && snapshot.cpu_clusters.length === 0) {
    throw new Error("powermetrics reported no CPU/GPU/ANE counters");
  }
  return snapshot;
}

/** The colocated helper is prepared during the CLI build, never at monitor runtime. */
export const NATIVE_METRICS_HELPER = fileURLToPath(new URL("../native/alloy-metrics", import.meta.url));

/** Validate the native wire contract without treating absent counters as idle. */
export function parseNativeMetricsOutput(output: string, now = Date.now()): PowerMetricsSnapshot {
  if (Buffer.byteLength(output) > MAX_OUTPUT_BYTES) throw new Error("Native hardware sample is too large.");
  const parsed: unknown = JSON.parse(output);
  const object = (value: unknown, field: string): Record<string, unknown> => {
    if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`Invalid native hardware ${field}.`);
    return value as Record<string, unknown>;
  };
  const root = object(parsed, "sample");
  if (root.schema_version !== 1 || root.source !== "ioreport") throw new Error("Unsupported native hardware schema or source.");
  const counter = (value: unknown, field: string, maximum = Number.MAX_VALUE): number | null => {
    if (value === null) return null;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > maximum) {
      throw new Error(`Invalid native hardware ${field}.`);
    }
    return value;
  };
  const integer = (value: unknown, field: string): number => {
    if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error(`Invalid native hardware ${field}.`);
    return value as number;
  };
  const label = (value: unknown, field: string, maximum = 128): string => {
    if (typeof value !== "string" || !value.trim() || value.length > maximum) throw new Error(`Invalid native hardware ${field}.`);
    return value;
  };
  const sampledAt = integer(root.sampled_at_unix_ms, "sampled_at_unix_ms");
  if (sampledAt > now + 1000) throw new Error("Native hardware sample timestamp is in the future.");
  const elapsed = counter(root.sample_duration_ms, "sample_duration_ms", 60_000);
  if (elapsed === null || elapsed === 0) throw new Error("Native hardware sample duration must be positive.");
  if (!Array.isArray(root.cpu_clusters) || root.cpu_clusters.length > 64) throw new Error("Invalid native hardware cpu_clusters.");
  const clusters = root.cpu_clusters.map((value: unknown, index: number) => {
    const cluster = object(value, `cpu_clusters[${index}]`);
    return {
      name: label(cluster.name, "cluster name"),
      active_percent: counter(cluster.active_percent, "cluster active_percent", 100),
      frequency_mhz: counter(cluster.frequency_mhz, "cluster frequency_mhz"),
    };
  });
  const thermal = root.thermal_pressure;
  if (thermal !== null && !["Nominal", "Fair", "Serious", "Critical"].includes(thermal as string)) {
    throw new Error("Invalid native hardware thermal_pressure.");
  }
  if (!Array.isArray(root.warnings) || root.warnings.length > 32) throw new Error("Invalid native hardware warnings.");
  const warnings = root.warnings.map((value: unknown) => label(value, "warning", 400));
  const sourceFields = object(root.power_sources, "power_sources");
  const source = (key: string): string | null => sourceFields[key] === null ? null : label(sourceFields[key], `power_sources.${key}`);
  const sources = { cpu: source("cpu"), gpu: source("gpu"), ane: source("ane") };
  const snapshot: PowerMetricsSnapshot = {
    ...emptyPowerMetrics("ready", null, "ioreport"),
    sampled_at_unix_ms: sampledAt,
    age_ms: Math.max(0, now - sampledAt),
    sample_duration_ms: elapsed,
    effective_uid: integer(root.effective_uid, "effective_uid"),
    cpu_power_mw: counter(root.cpu_power_mw, "cpu_power_mw"),
    gpu_power_mw: counter(root.gpu_power_mw, "gpu_power_mw"),
    ane_power_mw: counter(root.ane_power_mw, "ane_power_mw"),
    gpu_active_percent: counter(root.gpu_active_percent, "gpu_active_percent", 100),
    gpu_frequency_mhz: counter(root.gpu_frequency_mhz, "gpu_frequency_mhz"),
    cpu_clusters: clusters,
    thermal_pressure: thermal as string | null,
    disk_read_bytes_per_second: counter(root.disk_read_bytes_per_second, "disk_read_bytes_per_second"),
    disk_write_bytes_per_second: counter(root.disk_write_bytes_per_second, "disk_write_bytes_per_second"),
    network_in_bytes_per_second: counter(root.network_in_bytes_per_second, "network_in_bytes_per_second"),
    network_out_bytes_per_second: counter(root.network_out_bytes_per_second, "network_out_bytes_per_second"),
    power_sources: sources,
    warnings,
  };
  for (const key of ["cpu", "gpu", "ane"] as const) {
    if (snapshot[`${key}_power_mw`] !== null && sources[key] === null) {
      throw new Error(`Native ${key.toUpperCase()} power has no measured source.`);
    }
  }
  if (snapshot.cpu_power_mw === null && snapshot.gpu_power_mw === null && snapshot.ane_power_mw === null
      && snapshot.gpu_active_percent === null && snapshot.gpu_frequency_mhz === null
      && snapshot.cpu_clusters.length === 0 && snapshot.thermal_pressure === null) {
    throw new Error("Native sampler reported no hardware readings.");
  }
  return snapshot;
}

/** Direct, unprivileged native sampling. No external diagnostic-tool fallback. */
export class PowerMetricsCollector {
  #enabled: boolean;
  #intervalMs: number;
  #last: PowerMetricsSnapshot;
  #pending: Promise<PowerMetricsSnapshot> | null = null;
  #platform: () => string;
  #architecture: () => string;
  #runCommand: typeof runMetricCommand;
  #now: () => number;

  constructor(
    options: { enabled: boolean; intervalMs?: number },
    dependencies: {
      platform?: () => string;
      architecture?: () => string;
      runCommand?: typeof runMetricCommand;
      now?: () => number;
    } = {},
  ) {
    this.#enabled = options.enabled;
    // Hardware measurement windows stay short even when the display is slow.
    this.#intervalMs = Math.min(1000, Math.max(500, options.intervalMs ?? 1000));
    this.#last = emptyPowerMetrics(options.enabled ? "starting" : "disabled", null, "ioreport");
    this.#platform = dependencies.platform ?? os.platform;
    this.#architecture = dependencies.architecture ?? os.arch;
    this.#runCommand = dependencies.runCommand ?? runMetricCommand;
    this.#now = dependencies.now ?? Date.now;
  }

  sample(signal: AbortSignal): Promise<PowerMetricsSnapshot> {
    if (this.#pending === null) {
      this.#pending = this.#collect(signal).finally(() => { this.#pending = null; });
    }
    return this.#pending;
  }

  #failed(detail: string, stopped: boolean): PowerMetricsSnapshot {
    this.#last = {
      ...this.#last,
      status: stopped ? "stopped" : this.#last.sampled_at_unix_ms === null ? "unavailable" : "stale",
      age_ms: this.#last.sampled_at_unix_ms === null ? null : Math.max(0, this.#now() - this.#last.sampled_at_unix_ms),
      error: detail.slice(0, 400),
    };
    return this.#last;
  }

  async #collect(signal: AbortSignal): Promise<PowerMetricsSnapshot> {
    if (!this.#enabled) return this.#last;
    if (signal.aborted) return this.#failed("Native hardware sampling was stopped.", true);
    if (this.#platform() !== "darwin" || this.#architecture() !== "arm64") {
      return emptyPowerMetrics("unavailable", "Native hardware counters require Apple Silicon macOS.", "ioreport");
    }
    try {
      const output = await this.#runCommand(NATIVE_METRICS_HELPER,
        ["--interval-ms", String(this.#intervalMs)], signal, this.#intervalMs + 2500);
      if (signal.aborted) return this.#failed("Native hardware sampling was stopped.", true);
      this.#last = parseNativeMetricsOutput(output, this.#now());
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      return this.#failed(/ENOENT/.test(detail)
        ? "Native hardware sampler is missing. Reinstall Alloy using your installation method; source checkout users can rebuild the CLI."
        : detail, signal.aborted);
    }
    return this.#last;
  }
}
