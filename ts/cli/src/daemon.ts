import childProcess, { type ChildProcess } from "node:child_process";
import fs from "node:fs/promises";
import { constants } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import type { Writable } from "node:stream";
import type { AlloySettings } from "./settings.js";
import { RpcClient, type RpcConnectionOptions } from "./rpc-client.js";
import { STARTER_CORPUS_PATH } from "./model.js";

export type DaemonStatus = {
  api_version: 1;
  models_resident: number;
  memory_reserved: number;
  sessions_active: number;
};

export type DaemonProbe = {
  status: "ready" | "unavailable" | "incompatible";
  error: string | null;
  daemon_status: DaemonStatus | null;
  /** Only an absent/refused socket is evidence that a new listener may start. */
  can_start: boolean;
};

export interface DaemonRpcClient {
  call(method: string): Promise<unknown>;
  close(): void;
}

export type DaemonProbeOptions = {
  timeoutMs?: number;
  signal?: AbortSignal;
  /** New launches must prove that the responding daemon is the owned child. */
  expectedPid?: number;
  connect?: (socketPath: string, options: RpcConnectionOptions) => Promise<DaemonRpcClient>;
};

function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }

function abortReason(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error("Startup cancelled");
}

/** One total deadline, including both connection and response time. */
function waitFor<T>(promise: Promise<T>, signal: AbortSignal, timeoutMs?: number, description = "Operation"): Promise<T> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const cleanup = (): void => { clearTimeout(timer); signal.removeEventListener("abort", onAbort); };
    const finish = (error: Error | null, value?: T): void => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) reject(error); else resolve(value as T);
    };
    const onAbort = (): void => { finish(abortReason(signal)); };
    signal.addEventListener("abort", onAbort, { once: true });
    if (timeoutMs !== undefined) timer = setTimeout(() => { finish(new Error(`${description} timed out after ${timeoutMs} ms`)); }, timeoutMs);
    promise.then((value) => { finish(null, value); }, (error: unknown) => { finish(error instanceof Error ? error : new Error(String(error))); });
    if (signal.aborted) onAbort();
  });
}

function parseStatus(value: unknown): DaemonStatus {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("daemon.status returned an invalid response");
  const data = value as Record<string, unknown>;
  if (data.api_version !== 1) throw new Error(`Unsupported daemon API version ${String(data.api_version)}; expected 1`);
  for (const key of ["models_resident", "memory_reserved", "sessions_active"] as const) {
    if (!Number.isSafeInteger(data[key]) || (data[key] as number) < 0) throw new Error(`daemon.status returned invalid ${key}`);
  }
  return { api_version: 1, models_resident: data.models_resident as number, memory_reserved: data.memory_reserved as number, sessions_active: data.sessions_active as number };
}

/** Read-only API readiness; a reachable socket alone is not sufficient. */
export async function probeDaemon(socketPath: string, options: DaemonProbeOptions = {}): Promise<DaemonProbe> {
  const timeoutMs = options.timeoutMs ?? 1500;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("Daemon probe timeout must be positive");
  const abort = new AbortController();
  const signal = options.signal === undefined ? abort.signal : AbortSignal.any([abort.signal, options.signal]);
  let client: DaemonRpcClient | null = null;
  let connected = false;
  const operation = (async (): Promise<DaemonStatus> => {
    if (signal.aborted) throw abortReason(signal);
    const connect = options.connect ?? ((name, opts) => RpcClient.connect(name, opts));
    const pending = connect(socketPath, { timeoutMs, signal });
    // Custom connectors can resolve after the deadline; ownership still ends.
    void pending.then((late) => { if (signal.aborted) late.close(); }, () => {});
    client = await pending;
    if (signal.aborted) { client.close(); throw abortReason(signal); }
    connected = true;
    const status = parseStatus(await client.call("daemon.status"));
    if (signal.aborted) throw abortReason(signal);
    if (options.expectedPid !== undefined) {
      const metrics = await client.call("daemon.metrics");
      if (metrics === null || typeof metrics !== "object" || Array.isArray(metrics) ||
          (metrics as Record<string, unknown>).schema_version !== 1 ||
          (metrics as Record<string, unknown>).process_id !== options.expectedPid) {
        throw new Error(`The responding daemon does not match launched process ${options.expectedPid}; another listener may own the socket`);
      }
    }
    return status;
  })();
  try {
    const status = await waitFor(operation, signal, timeoutMs, "daemon.status readiness check");
    return { status: "ready", error: null, daemon_status: status, can_start: false };
  } catch (error) {
    const code = error !== null && typeof error === "object" && "code" in error ? error.code : null;
    return {
      status: connected ? "incompatible" : "unavailable", error: message(error), daemon_status: null,
      can_start: !signal.aborted && !connected && (code === "ENOENT" || code === "ECONNREFUSED"),
    };
  } finally {
    abort.abort();
    // Assignments in the async operation are not followed by TypeScript's CFA.
    (client as DaemonRpcClient | null)?.close();
  }
}

/** Chat may use an existing compatible daemon, but does not implicitly launch one. */
export async function ensureDaemon(settings: AlloySettings): Promise<void> {
  const probe = await probeDaemon(settings.socket);
  if (probe.status === "ready") return;
  const next = probe.can_start
    ? settings.artifact ? "Run alloy start in another terminal, then retry." : "Run alloy start to choose a model and start the daemon."
    : "Run alloy doctor to inspect the listener and configuration before restarting it.";
  throw new Error(`Alloy is not API-ready at ${settings.socket}: ${probe.error}. ${next}`);
}

export type DaemonLaunchPlan =
  | { kind: "source"; command: string; cwd: string }
  | { kind: "binary"; command: string }
  | { kind: "unavailable"; error: string };

export type DaemonLaunchOptions = {
  env?: NodeJS.ProcessEnv;
  /** Undefined discovers this CLI's checkout; null deliberately selects installed mode. */
  sourceRoot?: string | null;
};

async function executable(command: string, env: NodeJS.ProcessEnv): Promise<string | null> {
  const candidates = command.includes(path.sep)
    ? [path.resolve(command)] : (env.PATH ?? "").split(path.delimiter).map((dir) => path.resolve(dir || ".", command));
  for (const candidate of candidates) {
    try {
      if (!(await fs.stat(candidate)).isFile()) continue;
      await fs.access(candidate, constants.X_OK);
      return candidate;
    } catch { /* Try the remaining PATH entries. */ }
  }
  return null;
}

async function isSourceRoot(root: string): Promise<boolean> {
  try {
    return (await fs.stat(path.join(root, "Cargo.toml"))).isFile()
      && (await fs.stat(path.join(root, "crates", "alloyd", "Cargo.toml"))).isFile();
  } catch { return false; }
}

async function discoverSourceRoot(): Promise<string | null> {
  let root = path.dirname(fileURLToPath(import.meta.url));
  for (let depth = 0; depth < 8; depth++) {
    if (await isSourceRoot(root)) return root;
    const parent = path.dirname(root);
    if (parent === root) break;
    root = parent;
  }
  return null;
}

/** Resolving a launch plan does not build, spawn, download or mutate settings. */
export async function resolveDaemonLaunch(settings: AlloySettings, options: DaemonLaunchOptions = {}): Promise<DaemonLaunchPlan> {
  const env = options.env ?? process.env;
  const configured = settings.alloyd_bin && settings.alloyd_bin !== "alloyd" ? settings.alloyd_bin : env.ALLOYD;
  if (configured) {
    const command = await executable(configured, env);
    return command ? { kind: "binary", command } : { kind: "unavailable", error: `Configured alloyd executable is missing or not executable: ${configured}` };
  }
  const sourceRoot = options.sourceRoot === undefined ? await discoverSourceRoot() : options.sourceRoot;
  if (sourceRoot !== null) {
    if (!(await isSourceRoot(sourceRoot))) return { kind: "unavailable", error: `Source checkout is incomplete: ${sourceRoot}` };
    const cargo = await executable("cargo", env);
    return cargo ? { kind: "source", command: cargo, cwd: path.resolve(sourceRoot) }
      : { kind: "unavailable", error: "Cargo is not available in PATH. Install the repository's pinned Rust toolchain, or configure --alloyd-bin with a built daemon." };
  }
  const command = await executable("alloyd", env);
  return command ? { kind: "binary", command }
    : { kind: "unavailable", error: "No alloyd executable or source checkout was found. Configure --alloyd-bin with a built daemon, or run this CLI from the Alloy checkout." };
}

type ProcessResult = { code: number | null; signal: NodeJS.Signals | null; error: Error | null };

function exitCode(result: ProcessResult): number {
  if (result.error) return 1;
  if (result.code !== null) return result.code;
  return result.signal ? 128 + (os.constants.signals[result.signal] ?? 0) : 1;
}

class ProcessFailure extends Error {
  constructor(message: string, readonly code = 1) { super(message); }
}

type OwnedProcess = {
  child: ChildProcess;
  done: Promise<ProcessResult>;
  isDone(): boolean;
  tail(): string;
  stop(signal?: NodeJS.Signals): Promise<void>;
};

function launchOwned(command: string, args: string[], options: {
  cwd?: string;
  env: NodeJS.ProcessEnv;
  output: Writable;
  errorOutput: Writable;
  processGroup: boolean;
  shutdownGraceMs: number;
  onStdout?: (text: string) => void;
}): OwnedProcess {
  const child = childProcess.spawn(command, args, {
    ...(options.cwd === undefined ? {} : { cwd: options.cwd }),
    env: options.env, stdio: ["ignore", "pipe", "pipe"],
    // A private build process group permits stopping Cargo and its compiler
    // descendants. The supervisor remains attached and awaits the whole job.
    detached: options.processGroup,
  });
  let finished = false;
  let stderrTail = "";
  let spawnError: Error | null = null;
  const done = new Promise<ProcessResult>((resolve) => {
    child.once("error", (error) => { spawnError = error; });
    child.once("close", (code, signal) => { finished = true; resolve({ code, signal, error: spawnError }); });
  });
  if (options.onStdout) child.stdout?.setEncoding("utf8").on("data", options.onStdout);
  else child.stdout?.pipe(options.output, { end: false });
  child.stderr?.setEncoding("utf8").on("data", (text: string) => { stderrTail = (stderrTail + text).slice(-8192); });
  child.stderr?.pipe(options.errorOutput, { end: false });
  let stopping: Promise<void> | null = null;
  const send = (signal: NodeJS.Signals): void => {
    if (finished || child.pid === undefined) return;
    try {
      if (options.processGroup && process.platform !== "win32") process.kill(-child.pid, signal);
      else if (child.exitCode === null && child.signalCode === null) child.kill(signal);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
  };
  return {
    child, done, isDone: () => finished, tail: () => stderrTail.trim(),
    stop(signal = "SIGTERM") {
      if (stopping) return stopping;
      stopping = (async () => {
        if (finished) return;
        send(signal);
        let killError: Error | null = null;
        const timer = setTimeout(() => { try { send("SIGKILL"); } catch (error) { killError = error instanceof Error ? error : new Error(String(error)); } }, options.shutdownGraceMs);
        try {
          await waitFor(done, new AbortController().signal, options.shutdownGraceMs + 5000, "Stopping the owned process");
          if (killError) throw killError;
        } finally { clearTimeout(timer); }
      })();
      return stopping;
    },
  };
}

export type StartDaemonOptions = DaemonLaunchOptions & {
  signal?: AbortSignal;
  readinessTimeoutMs?: number;
  buildTimeoutMs?: number;
  probeTimeoutMs?: number;
  pollIntervalMs?: number;
  shutdownGraceMs?: number;
  output?: Writable;
  errorOutput?: Writable;
  log?: (text: string) => void;
  /** Allows transport-level verification without requiring a native socket in tests. */
  probe?: (socketPath: string, options: DaemonProbeOptions) => Promise<DaemonProbe>;
};

/** Explicit foreground start. Model validation/configuration is the caller's step. */
export async function startDaemon(settings: AlloySettings, options: StartDaemonOptions = {}): Promise<number> {
  const abort = new AbortController();
  const signal = options.signal === undefined ? abort.signal : AbortSignal.any([abort.signal, options.signal]);
  const env = options.env ?? process.env;
  const output = options.output ?? process.stdout;
  const errorOutput = options.errorOutput ?? process.stderr;
  const log = options.log ?? ((text: string) => { console.error(text); });
  const probe = options.probe ?? probeDaemon;
  const probeTimeoutMs = options.probeTimeoutMs ?? 1500;
  const shutdownGraceMs = options.shutdownGraceMs ?? 2000;
  let requestedExit = 130;
  let outputFailed = false;
  let owned: OwnedProcess | null = null;
  const onInt = (): void => { requestedExit = 130; abort.abort(new Error("Startup interrupted")); };
  const onTerm = (): void => { requestedExit = 143; abort.abort(new Error("Startup terminated")); };
  const onOutputError = (error: Error): void => { outputFailed = true; requestedExit = 1; abort.abort(error); };
  process.on("SIGINT", onInt);
  process.on("SIGTERM", onTerm);
  output.on("error", onOutputError);
  if (errorOutput !== output) errorOutput.on("error", onOutputError);
  const probeExisting = async (): Promise<"ready" | "start"> => {
    const result = await probe(settings.socket, { timeoutMs: probeTimeoutMs, signal });
    if (signal.aborted) throw abortReason(signal);
    if (result.status === "ready") {
      log(`Already running: API 1 is ready at ${settings.socket}. Existing daemon configuration was not changed.`);
      return "ready";
    }
    if (!result.can_start) throw new Error(`Cannot safely start at ${settings.socket}: ${result.error}. Run alloy doctor to inspect the existing listener.`);
    return "start";
  };
  try {
    if (signal.aborted) return requestedExit;
    if (await probeExisting() === "ready") return 0;
    if (!settings.artifact) throw new Error("No model is configured. Run alloy setup, then alloy start.");
    const plan = await resolveDaemonLaunch(settings, {
      env, ...(options.sourceRoot === undefined ? {} : { sourceRoot: options.sourceRoot }),
    });
    if (plan.kind === "unavailable") throw new Error(plan.error);
    let command = plan.command;
    if (plan.kind === "source") {
      log(`Building alloyd from ${plan.cwd} (the first native build can take several minutes)…`);
      let buffer = "";
      let builtExecutable: string | null = null;
      let parseError: Error | null = null;
      const consume = (text: string): void => {
        if (parseError) return;
        buffer += text;
        if (buffer.length > 1024 * 1024) {
          parseError = new Error("Cargo emitted an oversized output line");
          abort.abort(parseError); requestedExit = 1;
          return;
        }
        let newline: number;
        while ((newline = buffer.indexOf("\n")) !== -1) {
          const raw = buffer.slice(0, newline).trim(); buffer = buffer.slice(newline + 1);
          if (!raw) continue;
          let value: unknown;
          try { value = JSON.parse(raw); } catch { errorOutput.write(raw + "\n"); continue; }
          if (value === null || typeof value !== "object" || Array.isArray(value)) continue;
          const item = value as Record<string, unknown>;
          const target = item.target as Record<string, unknown> | undefined;
          if (item.reason === "compiler-artifact" && target?.name === "alloyd" && Array.isArray(target.kind) && target.kind.includes("bin") && typeof item.executable === "string") builtExecutable = item.executable;
        }
      };
      owned = launchOwned(plan.command, ["build", "--locked", "-p", "alloyd", "--message-format=json-render-diagnostics"], {
        cwd: plan.cwd, env, output, errorOutput, processGroup: process.platform !== "win32", shutdownGraceMs, onStdout: consume,
      });
      const result = await waitFor(owned.done, signal, options.buildTimeoutMs ?? 30 * 60_000, "Native build");
      if (parseError) throw parseError;
      if (result.error || exitCode(result) !== 0) throw new ProcessFailure(`Native build failed: ${result.error?.message ?? `exit ${exitCode(result)}`}${owned.tail() ? `\n${owned.tail()}` : ""}`, exitCode(result) || 1);
      if (buffer.trim()) consume("\n");
      if (!builtExecutable) throw new Error("Cargo completed without reporting an alloyd executable");
      command = await executable(builtExecutable, env) ?? "";
      if (!command) throw new Error(`Cargo's reported daemon is not executable: ${builtExecutable}`);
      owned = null;
    }
    // A daemon may have started while this checkout was building. Reuse it
    // rather than starting a second process or replacing its socket.
    if (await probeExisting() === "ready") return 0;
    if (signal.aborted) throw abortReason(signal);
    const args = ["--artifact", settings.artifact, "--profile", settings.profile, "--socket", settings.socket, "--db", settings.db,
      "--corpus-models", settings.corpus_models ?? STARTER_CORPUS_PATH];
    log(`Starting ${command}\nModel: ${settings.artifact}\nSocket: ${settings.socket}`);
    owned = launchOwned(command, args, { env, output, errorOutput, processGroup: false, shutdownGraceMs });
    const daemon = owned;
    if (daemon.child.pid === undefined) {
      const result = await waitFor(daemon.done, signal);
      throw new ProcessFailure(`Could not start alloyd: ${result.error?.message ?? "no child PID"}`, exitCode(result) || 1);
    }
    const readinessAbort = new AbortController();
    const readinessSignal = AbortSignal.any([signal, readinessAbort.signal]);
    const ready = async (): Promise<void> => {
      const deadline = Date.now() + (options.readinessTimeoutMs ?? 120_000);
      let lastError = "Waiting for the daemon to bind its socket";
      while (!readinessSignal.aborted) {
        const remaining = deadline - Date.now();
        if (remaining <= 0) throw new Error(`Daemon did not become API-ready within ${options.readinessTimeoutMs ?? 120_000} ms: ${lastError}`);
        const result = await probe(settings.socket, { timeoutMs: Math.min(probeTimeoutMs, remaining), signal: readinessSignal, expectedPid: daemon.child.pid! });
        if (readinessSignal.aborted) throw abortReason(readinessSignal);
        if (result.status === "ready") return;
        lastError = result.error ?? lastError;
        if (!result.can_start) throw new Error(`Daemon readiness failed: ${lastError}`);
        await delay(Math.min(options.pollIntervalMs ?? 250, Math.max(1, deadline - Date.now())), undefined, { signal: readinessSignal });
      }
      throw abortReason(readinessSignal);
    };
    try {
      await Promise.race([ready(), daemon.done.then((result) => {
        throw new ProcessFailure(`alloyd exited before API readiness: ${result.error?.message ?? `exit ${exitCode(result)}`}${daemon.tail() ? `\n${daemon.tail()}` : ""}`, exitCode(result) || 1);
      })]);
    } finally { readinessAbort.abort(); }
    if (daemon.isDone()) throw new Error("alloyd exited as its readiness check completed");
    log(`READY — Alloy API 1 at ${settings.socket} (PID ${daemon.child.pid}).\nKeep this terminal open. Use alloy monitor or alloy chat in another terminal; Ctrl+C stops this daemon.`);
    const result = await waitFor(daemon.done, signal);
    if (result.error || exitCode(result) !== 0) log(`alloyd exited: ${result.error?.message ?? `exit ${exitCode(result)}`}`);
    return exitCode(result);
  } catch (error) {
    if (signal.aborted) {
      if (requestedExit === 1 && !outputFailed) log(`alloy start: ${message(error)}`);
      return requestedExit;
    }
    if (!outputFailed) log(`alloy start: ${message(error)}`);
    return error instanceof ProcessFailure ? error.code : 1;
  } finally {
    try {
      if (owned !== null && !owned.isDone()) await owned.stop(requestedExit === 130 && signal.aborted ? "SIGINT" : "SIGTERM");
    } finally {
      process.off("SIGINT", onInt);
      process.off("SIGTERM", onTerm);
      // Keep listeners through any already-scheduled Writable error emission.
      setImmediate(() => { output.off("error", onOutputError); if (errorOutput !== output) errorOutput.off("error", onOutputError); });
    }
  }
}
