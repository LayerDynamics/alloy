import { probeDaemon, resolveDaemonLaunch, type DaemonLaunchPlan, type DaemonProbe } from "./daemon.js";
import { inspectModel, STARTER_MODEL, type ModelInspection } from "./model.js";
import { defaultSettings, loadSettings, normalizeSettingsPaths, type AlloySettings } from "./settings.js";

export type DoctorOptions = { json?: boolean; configurationError?: string };
export type DoctorReport = {
  schema_version: 1;
  readiness: "ready" | "setup_required" | "start_required" | "blocked";
  configuration: { status: "configured" | "missing" | "invalid"; error: string | null };
  model: { status: "preflight_passed" | "not_configured" | "not_checked" | "invalid"; path: string; validation: "preflight"; hashes_verified: false; errors: string[] };
  daemon: DaemonProbe & { socket: string };
  launch: DaemonLaunchPlan;
  next_command: string;
  next_step: string;
  validation_scope: string;
};

type DoctorDependencies = {
  inspectModel: typeof inspectModel;
  probeDaemon: typeof probeDaemon;
  resolveDaemonLaunch: typeof resolveDaemonLaunch;
};

const dependencies: DoctorDependencies = { inspectModel, probeDaemon, resolveDaemonLaunch };
const validationScope = "Readiness and model preflight only. Weight hashes, native model loading, generation, and M9 acceptance are not verified by doctor.";
const errorText = (error: unknown): string => error instanceof Error ? error.message : String(error);
const supportedProfileCommand = `alloy setup --profile ${STARTER_MODEL.profile}`;

function unsupportedProfile(settings: AlloySettings): string | null {
  return settings.profile === STARTER_MODEL.profile ? null : `Unsupported profile ${settings.profile}; supported profile is ${STARTER_MODEL.profile}`;
}

function preflight(path: string, inspect: typeof inspectModel, parent?: AbortSignal): Promise<ModelInspection> {
  const deadline = new AbortController();
  const signal = parent ? AbortSignal.any([parent, deadline.signal]) : deadline.signal;
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (model?: ModelInspection, error?: unknown): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      if (error !== undefined) reject(error); else resolve(model!);
    };
    const onAbort = (): void => { finish(undefined, signal.reason ?? new Error("model preflight cancelled")); };
    const timer = setTimeout(() => { deadline.abort(new Error("model preflight timed out")); }, 1500);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) { onAbort(); return; }
    Promise.resolve().then(() => inspect(path, { verifyHashes: false, signal }))
      .then((model) => { finish(model); }, (error: unknown) => { finish(undefined, error); });
  });
}

/** Read local metadata and a bounded status RPC. Never build, download, or start a daemon. */
export async function collectDoctorReport(
  settings: AlloySettings,
  options: DoctorOptions = {},
  checks: DoctorDependencies = dependencies,
): Promise<DoctorReport> {
  settings = normalizeSettingsPaths(settings);
  const configured = settings.artifact.trim().length > 0;
  const profileError = unsupportedProfile(settings);
  const configurationError = options.configurationError ?? profileError;
  const [model, daemon, launch] = await Promise.all([
    configured && !configurationError
      ? preflight(settings.artifact, checks.inspectModel).catch((error: unknown): ModelInspection => ({ ok: false, path: settings.artifact, errors: [errorText(error)], verified: false }))
      : Promise.resolve<ModelInspection>({ ok: false, path: settings.artifact, errors: [], verified: false }),
    checks.probeDaemon(settings.socket, { timeoutMs: 1500 }),
    checks.resolveDaemonLaunch(settings),
  ]);
  const report: DoctorReport = {
    schema_version: 1,
    readiness: "ready",
    configuration: {
      status: configurationError ? "invalid" : configured ? "configured" : "missing",
      error: configurationError,
    },
    model: {
      status: configurationError ? "not_checked" : !configured ? "not_configured" : model.ok ? "preflight_passed" : "invalid",
      path: model.path,
      validation: "preflight",
      hashes_verified: false,
      errors: model.errors,
    },
    daemon: { ...daemon, socket: settings.socket },
    launch,
    next_command: "alloy chat",
    next_step: "Run alloy chat to try generation with the ready daemon.",
    validation_scope: validationScope,
  };
  if (options.configurationError) {
    report.readiness = "setup_required";
    report.next_command = "alloy setup";
    report.next_step = "Repair the reported configuration with alloy setup.";
  } else if (profileError) {
    report.readiness = "setup_required";
    report.next_command = supportedProfileCommand;
    report.next_step = `${profileError}. Run ${supportedProfileCommand}.`;
  } else if (daemon.status === "ready") {
    // Chat can reuse a compatible live daemon without local model provisioning.
    // Keep local model/launch findings above for the next time a start is needed.
    report.next_step = "The compatible daemon is already running. Run alloy chat; local model setup is only needed to start a new daemon.";
  } else if (daemon.status === "incompatible") {
    report.readiness = "blocked";
    report.next_command = "alloy start";
    report.next_step = "Stop the incompatible daemon, update alloy and alloyd together, then run alloy start.";
  } else if (daemon.status === "unavailable" && !daemon.can_start) {
    report.readiness = "blocked";
    report.next_command = "alloy doctor";
    report.next_step = "Resolve the reported socket access or response error, then run alloy doctor again.";
  } else if (daemon.status === "unavailable" && launch.kind === "unavailable") {
    report.readiness = "blocked";
    report.next_command = "alloy doctor";
    report.next_step = `Fix the launch prerequisite before provisioning a model: ${launch.error}. Then run alloy doctor again.`;
  } else if (!configured) {
    report.readiness = "setup_required";
    report.next_command = "alloy start";
    report.next_step = "Run alloy start for guided setup, model selection, and daemon start.";
  } else if (!model.ok) {
    report.readiness = "setup_required";
    report.next_command = "alloy setup";
    report.next_step = "Select or prepare a supported model with alloy setup; the configured model failed preflight.";
  } else if (daemon.status === "unavailable") {
    report.readiness = "start_required";
    report.next_command = "alloy start";
    report.next_step = launch.kind === "source"
      ? "Run alloy start; the source checkout is available and the first start will build alloyd."
      : "Run alloy start to launch the configured daemon.";
  }
  return report;
}

function safeText(text: string): string {
  return text.replace(/[\u0000-\u001f\u007f-\u009f\u2028-\u202e\u2066-\u2069]/g, " ");
}

export function renderDoctorReport(report: DoctorReport): string {
  const lines = [
    `Alloy readiness: ${report.readiness.replaceAll("_", " ").toUpperCase()}`,
    `Configuration: ${report.configuration.status}${report.configuration.error ? ` — ${report.configuration.error}` : ""}`,
    `Model: ${report.model.status.replaceAll("_", " ")}${report.model.path ? ` — ${report.model.path}` : ""}`,
    report.model.status === "not_checked" || report.model.status === "not_configured"
      ? "Model check: not run until configuration selects a model; no weights were read."
      : "Model check: local preflight; full weight hashes were not read.",
    ...report.model.errors.map((error) => `  ${error}`),
    `Daemon: ${report.daemon.status} — ${report.daemon.socket}${report.daemon.error ? ` — ${report.daemon.error}` : ""}`,
  ];
  if (report.launch.kind === "source") {
    lines.push(`Launch plan: source checkout ${report.launch.cwd}; executable ${report.launch.command}`,
      "Build: available on start; native build prerequisites and compilation have not been tested by doctor.");
  } else if (report.launch.kind === "binary") {
    lines.push(`Launch plan: installed binary ${report.launch.command}`);
  } else {
    lines.push(`Launch plan: unavailable — ${report.launch.error}`);
  }
  lines.push(report.validation_scope, `Next: ${report.next_command}`, report.next_step);
  return lines.map(safeText).join("\n");
}

export async function runDoctor(settings: AlloySettings, options: DoctorOptions = {}): Promise<number> {
  const report = await collectDoctorReport(settings, options);
  console.log(options.json ? JSON.stringify(report, null, 2) : renderDoctorReport(report));
  return report.readiness === "ready" ? 0 : 1;
}

export const MONITOR_READINESS_REFRESH_MS = 10_000;
type ReadinessOptions = {
  configurationError?: string;
  socket?: string;
  reloadSettings?: () => AlloySettings | null;
};

/** Cached local setup advice. This does not probe native execution or change runtime state. */
export class MonitorReadiness {
  #settings: AlloySettings;
  #configurationError: string | null;
  #modelError: string | null = null;
  #checkedAt = -Infinity;
  #loadedSavedSettings = false;
  readonly #socketOverride: string | undefined;

  constructor(
    settings: AlloySettings,
    private readonly options: ReadinessOptions = {},
    private readonly inspect: typeof inspectModel = inspectModel,
  ) {
    this.#settings = normalizeSettingsPaths({ ...settings, ...(options.socket === undefined ? {} : { socket: options.socket }) });
    this.#socketOverride = options.socket === undefined ? undefined : this.#settings.socket;
    this.#configurationError = options.configurationError ?? null;
  }

  async refresh(daemonError: string | null, options: { now?: number; signal?: AbortSignal } = {}): Promise<{ settings: AlloySettings; nextStep: string }> {
    const now = options.now ?? Date.now();
    if (now - this.#checkedAt >= MONITOR_READINESS_REFRESH_MS) {
      this.#checkedAt = now;
      try {
        const saved = (this.options.reloadSettings ?? loadSettings)();
        if (saved !== null) {
          this.#loadedSavedSettings = true;
          this.#settings = normalizeSettingsPaths({ ...saved, socket: this.#socketOverride ?? saved.socket });
        } else if (this.#loadedSavedSettings) {
          const defaults = defaultSettings();
          this.#settings = normalizeSettingsPaths({ ...defaults, socket: this.#socketOverride ?? defaults.socket });
        }
        this.#configurationError = null;
      } catch (error) {
        this.#configurationError = errorText(error);
      }
      this.#modelError = null;
      if (!this.#configurationError && !unsupportedProfile(this.#settings) && this.#settings.artifact.trim()) {
        try {
          const model = await preflight(this.#settings.artifact, this.inspect, options.signal);
          if (!model.ok) this.#modelError = model.errors[0] ?? "model preflight failed";
        } catch (error) {
          if (options.signal?.aborted) throw error;
          this.#modelError = errorText(error);
        }
      }
    }
    const socketFlag = this.#socketOverride ? ` --socket '${this.#socketOverride.replaceAll("'", "'\\''")}'` : "";
    const start = `alloy start${socketFlag}`;
    let nextStep: string;
    if (this.#configurationError) nextStep = `Next: alloy setup · Configuration: ${this.#configurationError}`;
    else if (unsupportedProfile(this.#settings)) nextStep = `Next: ${supportedProfileCommand} · Unsupported profile ${this.#settings.profile}`;
    else if (daemonError && /unsupported daemon\.metrics|invalid daemon\.metrics|method.*not found|unknown method|API.VERSION.MISMATCH|api.version.*unsupported|unsupported.*api.version/i.test(daemonError)) {
      nextStep = `Next: update alloy/alloyd, stop the incompatible daemon, then ${start}`;
    } else if (daemonError && /EACCES|EPERM|timed out/i.test(daemonError)) nextStep = "Next: alloy doctor · Resolve the reported socket access/response error";
    else if (!this.#settings.artifact.trim()) nextStep = `Next: ${start} (guided setup)`;
    else if (this.#modelError) nextStep = `Next: alloy setup · Model: ${this.#modelError}`;
    else nextStep = `Next: ${start}`;
    return { settings: this.#settings, nextStep };
  }
}
