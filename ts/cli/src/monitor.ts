import { setTimeout as delay } from "node:timers/promises";
import type { Writable } from "node:stream";
import type { CliArgs } from "./args.js";
import { createHostMetricsCollector, type HostMetricsCollector, type HostMetricsSnapshot } from "./host-metrics.js";
import { parseDaemonMetrics, type AlloyMetricsSnapshot, type DaemonMetrics, type MonitorSnapshot } from "./monitor-types.js";
import { RpcClient, type RpcConnectionOptions } from "./rpc-client.js";
import type { AlloySettings } from "./settings.js";
import { MonitorHistory } from "./monitor-history.js";
import { renderMonitorSnapshot } from "./monitor-render.js";
import { MonitorReadiness } from "./doctor.js";
export { formatBytes, renderMonitorSnapshot } from "./monitor-render.js";

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

/** Re-date the frame at publication; native samples retain their original time bases. */
export function refreshMonitorAges(snapshot: MonitorSnapshot, now = Date.now()): MonitorSnapshot {
  const host = snapshot.host;
  const powerTime = host?.accelerators.sampled_at_unix_ms;
  return {
    ...snapshot,
    sampled_at_unix_ms: now,
    host_sample_age_ms: host === null ? null : Math.max(0, now - host.sampled_at_unix_ms),
    host: host === null ? null : {
      ...host,
      accelerators: {
        ...host.accelerators,
        age_ms: powerTime === null || powerTime === undefined ? null : Math.max(0, now - powerTime),
      },
    },
    alloy: {
      ...snapshot.alloy,
      sample_age_ms: snapshot.alloy.metrics === null ? null : Math.max(0, now - snapshot.alloy.metrics.sampled_at_unix_ms),
    },
  };
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
      status: hostResult.status === "rejected" && sourcePower?.sampled_at_unix_ms !== null && sourcePower?.status === "ready"
        ? "stale" as const : sourceHost.accelerators.status,
      error: hostResult.status === "rejected" ? message(hostResult.reason) : sourceHost.accelerators.error,
    },
  };
  const alloy = alloyResult.status === "fulfilled" ? alloyResult.value
    : { status: "unavailable" as const, sample_age_ms: null, metrics: null, error: message(alloyResult.reason) };
  return refreshMonitorAges({
    schema_version: 1,
    sampled_at_unix_ms: now,
    host,
    host_sample_age_ms: null,
    host_error: hostResult.status === "rejected" ? message(hostResult.reason) : null,
    alloy,
  }, now);
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

/** Read-only monitor: no session creation, daemon launch, or runtime mutation. */
export async function runMonitor(settings: AlloySettings, args: CliArgs, options: {
  configurationError?: string;
  reloadSettings?: () => AlloySettings | null;
} = {}): Promise<number> {
  const collector = createHostMetricsCollector({ powermetrics: args.powermetrics, intervalMs: args.intervalMs });
  const timeoutMs = Math.min(2000, Math.max(500, args.intervalMs));
  let activeSettings = settings;
  let reader = new DaemonMetricsReader(settings.socket, timeoutMs);
  const readiness = new MonitorReadiness(settings, { ...options, ...(args.socket === undefined ? {} : { socket: args.socket }) });
  const abort = new AbortController();
  const interactive = !args.once && !args.json && process.env.TERM !== "dumb" && Boolean(process.stdout.isTTY && process.stdin.isTTY);
  const once = args.once || (!args.json && !interactive);
  const wasRaw = process.stdin.isRaw;
  const wasFlowing = process.stdin.readableFlowing === true;
  let terminalEntered = false;
  let exitCode = 0;
  let previousHost: HostMetricsSnapshot | null = null;
  let lastSnapshot: MonitorSnapshot | null = null;
  const history = new MonitorHistory();
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
      let snapshot = await collectMonitorSnapshot(collector, reader, previousHost);
      if (abort.signal.aborted) break;
      let nextStep: string | undefined;
      if (snapshot.alloy.status === "unavailable") {
        const advice = await readiness.refresh(snapshot.alloy.error, { signal: abort.signal });
        if (abort.signal.aborted) break;
        nextStep = advice.nextStep;
        // Setup may finish in another terminal. Keep an explicit --socket,
        // otherwise reconnect to the newly saved target on the next poll.
        if (advice.settings.socket !== activeSettings.socket) {
          reader.close();
          reader = new DaemonMetricsReader(advice.settings.socket, timeoutMs);
        }
        activeSettings = advice.settings;
      }
      // Local readiness work can take time. History, text and JSON must all
      // expose the age at publication, not the age before that work began.
      snapshot = refreshMonitorAges(snapshot);
      previousHost = snapshot.host;
      lastSnapshot = snapshot;
      if (!args.json) history.push(snapshot);
      if (args.json) {
        await writeMonitorOutput(process.stdout, JSON.stringify(snapshot) + "\n", abort.signal);
      } else {
        const text = renderMonitorSnapshot(snapshot, {
          columns: process.stdout.columns ?? 100,
          rows: interactive ? Math.max(1, (process.stdout.rows ?? 24) - 1) : 1000,
          intervalMs: args.intervalMs,
          interactive,
          color: interactive && process.env.NO_COLOR === undefined && process.env.TERM !== "dumb",
          history: history.snapshot(),
          ...(nextStep === undefined ? {} : { nextStep }),
        });
        // Clear each row's old tail and any removed rows without blanking the
        // whole screen first. Shorter errors and terminal resizes leave no debris.
        const frame = interactive ? `\u001b[H${text.replace(/\n/g, "\u001b[K\n")}\u001b[J` : `${text}\n`;
        await writeMonitorOutput(process.stdout, frame, abort.signal);
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
