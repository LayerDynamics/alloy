import { parseArgv } from "./args.js";
import {
  defaultSettings,
  loadSettings,
  type AlloySettings,
} from "./settings.js";
import { runSetup } from "./setup.js";
import { ensureDaemon } from "./daemon.js";
import { runChat } from "./chat.js";
import { runMonitor } from "./monitor.js";

function printHelp(): void {
  console.log(`Usage: alloy [--setup] [-m TEXT] [--profile NAME] [--socket PATH] [--resume ID]
  alloy           interactive chat (connects to existing alloyd)
  alloy -m TEXT   one-shot message
  alloy --setup   write ~/.alloy/settings.json
  alloy chat ...  alias for the same chat path

Usage: alloy monitor [--socket PATH] [--interval MS] [--powermetrics]
  alloy monitor                 live read-only host and Alloy dashboard
  alloy monitor --once --json   capture one JSON snapshot
  alloy monitor --json          continuous newline-delimited JSON
  --interval MS                 refresh interval, 500–60000 ms (default 1000)
  --powermetrics                optional macOS GPU/ANE/power sampling via sudo -n
                                authorize sudo separately before use
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
  return next;
}

async function main(): Promise<void> {
  let args;
  try {
    args = parseArgv(process.argv);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
    return;
  }

  if (args.help) {
    printHelp();
    return;
  }

  let settings: AlloySettings;
  if (args.forceSetup) {
    const partial: Parameters<typeof runSetup>[0] = {};
    if (args.profile !== undefined) partial.profile = args.profile;
    if (args.socket !== undefined) partial.socket = args.socket;
    if (args.db !== undefined) partial.db = args.db;
    if (args.artifact !== undefined) partial.artifact = args.artifact;
    if (args.alloydBin !== undefined) partial.alloyd_bin = args.alloydBin;
    settings = await runSetup(partial);
  } else {
    settings = applyOverrides(loadSettings() ?? defaultSettings(), args);
  }

  if (args.command === "monitor") {
    process.exitCode = await runMonitor(settings, args);
    return;
  }

  try {
    await ensureDaemon(settings);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
    return;
  }

  process.exitCode = await runChat(settings, args);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
