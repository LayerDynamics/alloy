import { parseArgv, type CliArgs } from "./args.js";
import {
  defaultSettings,
  loadSettings,
  normalizeSettingsPaths,
  type AlloySettings,
} from "./settings.js";
import { runSetup, SetupCancelledError, type SetupPartial } from "./setup.js";
import { ensureDaemon, probeDaemon, resolveDaemonLaunch, startDaemon } from "./daemon.js";
import { runChat } from "./chat.js";
import { runMonitor } from "./monitor.js";
import { runDoctor } from "./doctor.js";
import { STARTER_MODEL } from "./model.js";

function printHelp(): void {
  console.log(`Get started:
  alloy start                   guided setup, verify model, build/start alloyd
                                stays in this terminal; Ctrl+C stops owned daemon
  alloy setup                   configure a verified local model, then exit
  alloy setup --download        download and verify the pinned starter model
  alloy setup --artifact PATH   configure an existing Qwen3-4B-4bit directory
  alloy doctor [--json]          read-only configuration and startup readiness
  alloy --setup                 alias for alloy setup

Setup/start options:
  --artifact PATH               existing model, or destination with --download
  --download                    explicitly download the pinned starter model
  --socket PATH --db PATH       override saved daemon paths
  --alloyd-bin PATH             use a specific daemon instead of this checkout
  --corpus-models PATH          use an explicit corpus pin file

Usage: alloy [-m TEXT] [--profile NAME] [--socket PATH] [--resume ID]
  alloy           interactive chat (connects to existing alloyd)
  alloy -m TEXT   one-shot message
  alloy chat ...  alias for the same chat path

Usage: alloy monitor [--socket PATH] [--interval MS] [--powermetrics]
  alloy monitor                 live read-only host and Alloy dashboard
  alloy monitor --once --json   capture one JSON snapshot
  alloy monitor --json          continuous newline-delimited JSON
  --interval MS                 refresh interval, 500–60000 ms (default 1000)
  --powermetrics                compatibility alias; native hardware is sampled by default
  q / Ctrl+C                    exit the live dashboard

The monitor remains useful when alloyd is offline. Redirected text output takes
one snapshot; --json streams until interrupted unless --once is also supplied.`);
}

function applyOverrides(
  settings: AlloySettings,
  args: ReturnType<typeof parseArgv>,
): AlloySettings {
  let next = settings;
  if (args.profile) next = { ...next, profile: args.profile };
  if (args.socket) next = { ...next, socket: args.socket };
  if (args.db) next = { ...next, db: args.db };
  if (args.artifact) next = { ...next, artifact: args.artifact };
  if (args.alloydBin) next = { ...next, alloyd_bin: args.alloydBin };
  if (args.corpusModels) next = { ...next, corpus_models: args.corpusModels };
  return next;
}

function setupOverrides(args: CliArgs): SetupPartial {
  const partial: SetupPartial = {};
  if (args.profile !== undefined) partial.profile = args.profile;
  if (args.socket !== undefined) partial.socket = args.socket;
  if (args.db !== undefined) partial.db = args.db;
  if (args.artifact !== undefined) partial.artifact = args.artifact;
  if (args.alloydBin !== undefined) partial.alloyd_bin = args.alloydBin;
  if (args.corpusModels !== undefined) partial.corpus_models = args.corpusModels;
  return partial;
}

export type CliDependencies = {
  loadSettings: typeof loadSettings;
  runSetup: typeof runSetup;
  probeDaemon: typeof probeDaemon;
  resolveDaemonLaunch: typeof resolveDaemonLaunch;
  startDaemon: typeof startDaemon;
  runDoctor: typeof runDoctor;
  runMonitor: typeof runMonitor;
  ensureDaemon: typeof ensureDaemon;
  runChat: typeof runChat;
};

/** Explicit command boundaries keep setup/start out of read-only monitoring. */
export async function runCli(argv: string[], overrides: Partial<CliDependencies> = {}): Promise<number> {
  const dependencies: CliDependencies = { loadSettings, runSetup, probeDaemon, resolveDaemonLaunch, startDaemon, runDoctor, runMonitor, ensureDaemon, runChat, ...overrides };
  let args;
  try {
    args = parseArgv(argv);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    return 1;
  }

  if (args.help) {
    printHelp();
    return 0;
  }

  if (args.command === "setup") {
    await dependencies.runSetup(setupOverrides(args), undefined, { download: args.download });
    console.error("Setup complete. Next: alloy start");
    return 0;
  }

  let settings = defaultSettings();
  let configurationError: string | undefined;
  try {
    settings = dependencies.loadSettings() ?? settings;
  } catch (error) {
    configurationError = error instanceof Error ? error.message : String(error);
  }
  settings = normalizeSettingsPaths(applyOverrides(settings, args));
  const readinessOptions = configurationError === undefined ? {} : { configurationError };

  if (args.command === "monitor") {
    return dependencies.runMonitor(settings, args, readinessOptions);
  }
  if (args.command === "doctor") {
    return dependencies.runDoctor(settings, { ...readinessOptions, json: args.json });
  }
  if (args.command === "start") {
    if (configurationError !== undefined) throw new Error(configurationError);
    if (settings.profile !== STARTER_MODEL.profile) throw new Error(`Unsupported profile ${settings.profile}. Run alloy setup --profile ${STARTER_MODEL.profile}.`);
    const current = await dependencies.probeDaemon(settings.socket);
    if (current.status === "ready") return dependencies.startDaemon(settings);
    if (!current.can_start) throw new Error(`${current.error ?? "The selected socket is unavailable"}. Run alloy doctor before starting another daemon.`);
    const launch = await dependencies.resolveDaemonLaunch(settings);
    if (launch.kind === "unavailable") throw new Error(`${launch.error}. Run alloy doctor to check startup prerequisites.`);
    settings = await dependencies.runSetup(setupOverrides(args), undefined, { download: args.download });
    return dependencies.startDaemon(settings);
  }
  if (configurationError !== undefined) throw new Error(configurationError);
  if (!settings.artifact && Boolean(process.stdin.isTTY && process.stderr.isTTY)) {
    // Chat connects to an explicitly started daemon; guide first use rather than
    // launching an unowned background process or silently downloading a model.
    const current = await dependencies.probeDaemon(settings.socket);
    if (current.status !== "ready") {
      console.error("Alloy is not set up yet. Run alloy start for guided setup, then open chat in another terminal.");
      return 1;
    }
  }

  try {
    await dependencies.ensureDaemon(settings);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    return 1;
  }
  return dependencies.runChat(settings, args);
}

export async function main(): Promise<void> {
  try { process.exitCode = await runCli(process.argv); }
  catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = error instanceof SetupCancelledError ? error.exitCode : 1;
  }
}
