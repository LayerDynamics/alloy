import { setTimeout as delay } from "node:timers/promises";
import type { Writable } from "node:stream";
import type { CliArgs } from "./args.js";
import { createHostMetricsCollector, type HostMetricsCollector, type HostMetricsSnapshot } from "./host-metrics.js";
import { parseDaemonMetrics, type AlloyMetricsSnapshot, type DaemonMetrics, type MonitorSnapshot } from "./monitor-types.js";
import { RpcClient, type RpcConnectionOptions } from "./rpc-client.js";
import type { AlloySettings } from "./settings.js";

export interface MetricsRpcClient {
  call(method: string): Promise<unknown>;
  close(): void;
}

type Connect = (path: string, options: RpcConnectionOptions) => Promise<MetricsRpcClient>;

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function bounded<T>(promise: Promise<T>, timeoutMs: number, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    };
    const onAbort = (): void => { cleanup(); reject(new Error("monitor stopped")); };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`daemon.metrics timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then((value) => { cleanup(); resolve(value); }, (error: unknown) => { cleanup(); reject(error); });
    if (signal.aborted) onAbort();
  });
}

/** One connection, one bounded read at a time; reconnect after a failed read. */
export class DaemonMetricsReader {
  #client: MetricsRpcClient | null = null;
  #last: DaemonMetrics | null = null;
  #active: AbortController | null = null;
  #closed = false;

  constructor(
    private readonly socketPath: string,
    private readonly timeoutMs = 1500,
    private readonly connect: Connect = (path, options) => RpcClient.connect(path, options),
  ) {}

  async read(): Promise<AlloyMetricsSnapshot> {
    if (this.#active !== null) throw new Error("overlapping monitor polls are not supported");
    const abort = new AbortController();
    this.#active = abort;
    let error: string | null = null;
    try {
      if (this.#closed) throw new Error("monitor stopped");
      if (this.#client === null) {
        const pending = this.connect(this.socketPath, { timeoutMs: this.timeoutMs, signal: abort.signal });
        // A custom connector may complete after cancellation. Never leak it.
        void pending.then((client) => { if (abort.signal.aborted || this.#closed) client.close(); }, () => {});
        this.#client = await bounded(pending, this.timeoutMs, abort.signal);
      }
      const value = await bounded(this.#client.call("daemon.metrics"), this.timeoutMs, abort.signal);
      this.#last = parseDaemonMetrics(value);
    } catch (caught) {
      error = message(caught);
      abort.abort();
      this.#client?.close();
      this.#client = null;
    } finally {
      this.#active = null;
    }
    return {
      status: error === null ? "connected" : "unavailable",
      sample_age_ms: this.#last === null ? null : Math.max(0, Date.now() - this.#last.sampled_at_unix_ms),
      error,
      metrics: this.#last,
    };
  }

  close(): void {
    this.#closed = true;
    this.#active?.abort();
    this.#client?.close();
    this.#client = null;
  }
}

/** Fresh host data survives daemon errors, and retained samples carry their age. */
export async function collectMonitorSnapshot(
  collector: HostMetricsCollector,
  reader: Pick<DaemonMetricsReader, "read">,
  previousHost: HostMetricsSnapshot | null = null,
): Promise<MonitorSnapshot> {
  const [hostResult, alloyResult] = await Promise.allSettled([collector.sample(), reader.read()]);
  const now = Date.now();
  const sourceHost = hostResult.status === "fulfilled" ? hostResult.value : previousHost;
  const sourcePower = sourceHost?.accelerators;
  const host = sourceHost === null ? null : {
    ...sourceHost,
    accelerators: {
      ...sourceHost.accelerators,
      age_ms: sourcePower?.sampled_at_unix_ms === null || sourcePower?.sampled_at_unix_ms === undefined
        ? null : Math.max(0, now - sourcePower.sampled_at_unix_ms),
      status: hostResult.status === "rejected" && sourcePower?.sampled_at_unix_ms !== null && sourcePower?.status === "ready"
        ? "stale" as const : sourceHost.accelerators.status,
      error: hostResult.status === "rejected" ? message(hostResult.reason) : sourceHost.accelerators.error,
    },
  };
  const alloy = alloyResult.status === "fulfilled" ? {
    ...alloyResult.value,
    sample_age_ms: alloyResult.value.metrics === null ? null : Math.max(0, now - alloyResult.value.metrics.sampled_at_unix_ms),
  } : { status: "unavailable" as const, sample_age_ms: null, metrics: null, error: message(alloyResult.reason) };
  return {
    schema_version: 1,
    sampled_at_unix_ms: now,
    host,
    host_sample_age_ms: host === null ? null : Math.max(0, now - host.sampled_at_unix_ms),
    host_error: hostResult.status === "rejected" ? message(hostResult.reason) : null,
    alloy,
  };
}

/** Await each write so a slow JSON consumer cannot create an unbounded queue. */
export function writeMonitorOutput(
  output: Writable,
  text: string,
  signal: AbortSignal,
  timeoutMs = 5000,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error?: Error | null): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      // Writable invokes its callback before emitting the corresponding error.
      // Keep our listener through that emission even without a UI-level listener.
      setImmediate(() => output.off("error", onError));
      if (error) reject(error); else resolve();
    };
    const onError = (error: Error): void => { finish(error); };
    const onAbort = (): void => { finish(new Error("monitor stopped")); };
    const timer = setTimeout(() => { finish(new Error("monitor output timed out")); }, timeoutMs);
    output.on("error", onError);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) { onAbort(); return; }
    try { output.write(text, finish); } catch (error) { finish(error instanceof Error ? error : new Error(String(error))); }
  });
}

function brokenPipe(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "EPIPE";
}

export function formatBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes)) return "unavailable";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function percent(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "unavailable" : `${value.toFixed(1)}%`;
}

function power(value: number | null): string {
  return value === null ? "unavailable" : `${value.toFixed(0)} mW`;
}

function age(value: number | null): string {
  return value === null ? "no sample" : `${(value / 1000).toFixed(1)}s old`;
}

function uptime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  return `${hours >= 24 ? `${Math.floor(hours / 24)}d ` : ""}${hours % 24}h ${minutes % 60}m`;
}

function meter(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "";
  const fill = Math.round(Math.min(100, Math.max(0, value)) / 100 * 12);
  return "█".repeat(fill) + "░".repeat(12 - fill);
}

function clean(text: string): string {
  // IDs, hostnames and error messages must not inject terminal control sequences.
  return text.replace(/[\u0000-\u001f\u007f-\u009f]/g, " ");
}

function fit(text: string, width: number): string {
  const chars = Array.from(clean(text));
  return chars.length <= width ? chars.join("") : chars.slice(0, Math.max(0, width - 1)).join("") + "…";
}

export function renderMonitorSnapshot(
  snapshot: MonitorSnapshot,
  options: { columns?: number; rows?: number; intervalMs?: number; interactive?: boolean } = {},
): string {
  const width = Math.max(20, options.columns ?? 80);
  const height = Math.max(8, options.rows ?? 24);
  const lines = [`ALLOY MONITOR  ${new Date(snapshot.sampled_at_unix_ms).toISOString()}  read-only`];
  const host = snapshot.host;
  if (host !== null) {
    lines.push(`HOST  ${host.hostname}  ${host.platform}/${host.architecture}  ${age(snapshot.host_sample_age_ms)}`);
    lines.push(` CPU  ${meter(host.cpu_utilization_percent)} ${percent(host.cpu_utilization_percent)}  ${host.logical_cpu_count} logical cores`);
    lines.push(` Load 1/5/15m ${host.load_average.map((value) => value.toFixed(2)).join(" / ")}  Uptime ${uptime(host.uptime_seconds)}`);
    lines.push(` ${host.platform === "darwin" ? "Unified RAM" : "RAM"} non-free  ${formatBytes(host.memory_used_bytes)} / ${formatBytes(host.memory_total_bytes)} (incl. caches)`);
    lines.push(` Swap ${formatBytes(host.swap_used_bytes)} / ${formatBytes(host.swap_total_bytes)}  Wired ${formatBytes(host.wired_bytes)}  Compressed ${formatBytes(host.compressed_bytes)}`);
    if (snapshot.host_error || host.errors.length) lines.push(` Host error: ${snapshot.host_error ?? host.errors.join("; ")}`);
    const accelerator = host.accelerators;
    lines.push(`ACCELERATORS  ${accelerator.status}  ${age(accelerator.age_ms)}`);
    lines.push(` GPU active ${percent(accelerator.gpu_active_percent)}  Est. power ${power(accelerator.gpu_power_mw)}  ${accelerator.gpu_frequency_mhz === null ? "frequency unavailable" : `${accelerator.gpu_frequency_mhz.toFixed(0)} MHz`}`);
    lines.push(` Estimated power: ANE ${power(accelerator.ane_power_mw)}  CPU ${power(accelerator.cpu_power_mw)}`);
    lines.push(` Thermal ${accelerator.thermal_pressure ?? "unavailable"}  ${accelerator.error ?? (accelerator.status === "disabled" ? "Use --powermetrics for hardware sampling" : "source: powermetrics")}`);
  } else {
    lines.push("HOST  unavailable", ` ${snapshot.host_error ?? "No host sample"}`, "ACCELERATORS  unavailable");
  }
  const alloy = snapshot.alloy;
  const metrics = alloy.metrics;
  lines.push(`ALLOY  ${alloy.status}${metrics === null ? "" : `  PID ${metrics.process_id}`}  ${age(alloy.sample_age_ms)}${alloy.status === "unavailable" && metrics !== null ? " (stale)" : ""}`);
  if (alloy.error) lines.push(` Connection: ${alloy.error}`);
  if (metrics !== null) {
    const { admission, execution, residency, memory_guard: guard } = metrics;
    lines.push(` Ledger ${formatBytes(admission.reserved_bytes)} / ${formatBytes(admission.budget_bytes)}  Available ${formatBytes(admission.available_bytes)}`);
    lines.push(` Admission ${admission.admissions_paused ? "PAUSED" : "open"}  Reservations ${admission.active_reservations}  Pressure ${guard.pressure}${guard.emergency ? " EMERGENCY" : ""}`);
    const elapsed = alloy.sample_age_ms ?? 0;
    lines.push(` Guard ${guard.tier}: ${formatBytes(guard.usage_bytes)} / ${formatBytes(guard.hard_limit_bytes)}  ${guard.binding ?? "binding unavailable"}  ${age(guard.sample_age_ms === null ? null : guard.sample_age_ms + elapsed)}`);
    lines.push(` Request ${execution.active_request_id ?? "idle"}  Cancellation ${execution.cancellation_requested ? "requested; cleanup pending" : "not requested"}`);
    lines.push(` Models ${residency.models_resident}  Worker ${residency.worker_alive ? "alive" : "stopped"}  Residency ${age(residency.sample_age_ms + elapsed)}`);
    if (guard.last_error) lines.push(` Guard error: ${guard.last_error}`);
    if (guard.cache_reclaim_requested) lines.push(" Cache reclaim requested");
  } else {
    lines.push(" Host monitoring continues; reconnecting to the configured alloyd socket.");
  }
  const details: string[] = [];
  if (metrics !== null) {
    for (const model of metrics.residency.models) {
      details.push(` ${model.artifact_id}  weights ${formatBytes(model.weights_reserved_bytes)}  KV ${formatBytes(model.kv_reserved_bytes)}${model.pinned ? "  pinned" : ""}${model.leased ? "  leased" : ""}`);
    }
  }
  if (host !== null) {
    for (const cluster of host.accelerators.cpu_clusters) {
      details.push(` ${cluster.name} active ${percent(cluster.active_percent)}  ${cluster.frequency_mhz === null ? "frequency unavailable" : `${cluster.frequency_mhz.toFixed(0)} MHz`}`);
    }
    const a = host.accelerators;
    details.push(` Disk R/W ${formatBytes(a.disk_read_bytes_per_second)}/s / ${formatBytes(a.disk_write_bytes_per_second)}/s`);
    details.push(` Network in/out ${formatBytes(a.network_in_bytes_per_second)}/s / ${formatBytes(a.network_out_bytes_per_second)}/s`);
  }
  const footer = options.interactive === false
    ? "Read-only snapshot; unavailable values are not zero."
    : `q quit | Ctrl+C exit | refresh ${(options.intervalMs ?? 1000) / 1000}s | unavailable ≠ zero`;
  const budget = Math.max(0, height - 1);
  const visible = lines.slice(0, budget);
  visible.push(...details.slice(0, Math.max(0, budget - visible.length)));
  visible.push(footer);
  return visible.map((line) => fit(line, width)).join("\n");
}

/** Read-only monitor: no session creation, daemon launch, or runtime mutation. */
export async function runMonitor(settings: AlloySettings, args: CliArgs): Promise<number> {
  const collector = createHostMetricsCollector({ powermetrics: args.powermetrics, intervalMs: args.intervalMs });
  const reader = new DaemonMetricsReader(settings.socket, Math.min(2000, Math.max(500, args.intervalMs)));
  const abort = new AbortController();
  const interactive = !args.once && !args.json && Boolean(process.stdout.isTTY && process.stdin.isTTY);
  const once = args.once || (!args.json && !interactive);
  const wasRaw = process.stdin.isRaw;
  const wasFlowing = process.stdin.readableFlowing === true;
  let terminalEntered = false;
  let exitCode = 0;
  let previousHost: HostMetricsSnapshot | null = null;
  let lastSnapshot: MonitorSnapshot | null = null;
  const stop = (code: number): void => {
    exitCode = code;
    abort.abort();
    reader.close();
    void collector.close().catch(() => {});
  };
  const onInt = (): void => { stop(130); };
  const onTerm = (): void => { stop(143); };
  const onOutputError = (error: Error): void => { stop(brokenPipe(error) ? 0 : 1); };
  const onData = (data: Buffer | string): void => {
    const text = data.toString();
    if (text.includes("\u0003")) stop(130);
    else if (/[qQ]/.test(text)) stop(0);
  };
  process.on("SIGINT", onInt);
  process.on("SIGTERM", onTerm);
  process.stdout.on("error", onOutputError);
  try {
    if (interactive) {
      terminalEntered = true;
      await writeMonitorOutput(process.stdout, "\u001b[?1049h\u001b[?25l", abort.signal);
      process.stdin.setRawMode(true);
      process.stdin.on("data", onData);
      process.stdin.resume();
    }
    while (!abort.signal.aborted) {
      const started = Date.now();
      const snapshot = await collectMonitorSnapshot(collector, reader, previousHost);
      if (abort.signal.aborted) break;
      previousHost = snapshot.host;
      lastSnapshot = snapshot;
      if (args.json) {
        await writeMonitorOutput(process.stdout, JSON.stringify(snapshot) + "\n", abort.signal);
      } else {
        const text = renderMonitorSnapshot(snapshot, {
          columns: process.stdout.columns ?? 100,
          rows: interactive ? Math.max(8, (process.stdout.rows ?? 24) - 1) : 1000,
          intervalMs: args.intervalMs,
          interactive,
        });
        await writeMonitorOutput(process.stdout, (interactive ? "\u001b[H\u001b[2J" : "") + text + (interactive ? "" : "\n"), abort.signal);
      }
      if (once) break;
      try {
        await delay(Math.max(0, args.intervalMs - (Date.now() - started)), undefined, { signal: abort.signal });
      } catch (error) {
        if (!abort.signal.aborted) throw error;
      }
    }
  } catch (error) {
    if (brokenPipe(error)) stop(0);
    else if (!abort.signal.aborted) { stop(1); throw error; }
  } finally {
    reader.close();
    try { await collector.close(); } finally {
      process.off("SIGINT", onInt);
      process.off("SIGTERM", onTerm);
      if (terminalEntered) {
        process.stdin.off("data", onData);
        try {
          process.stdin.setRawMode(Boolean(wasRaw));
        } finally {
          if (!wasFlowing) process.stdin.pause();
          await writeMonitorOutput(process.stdout, "\u001b[?25h\u001b[?1049l", new AbortController().signal, 250).catch(() => {});
        }
      }
      if (!interactive && abort.signal.aborted) process.stdout.destroy();
      process.stdout.off("error", onOutputError);
    }
  }
  if (once && lastSnapshot !== null && lastSnapshot.host === null && lastSnapshot.alloy.metrics === null) return 1;
  return exitCode;
}
