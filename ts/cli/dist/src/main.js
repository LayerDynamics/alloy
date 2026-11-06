import { parseArgv } from "./args.js";
import { defaultSettings, loadSettings, } from "./settings.js";
import { runSetup } from "./setup.js";
import { ensureDaemon } from "./daemon.js";
import { runChat } from "./chat.js";
function printHelp() {
    console.log(`Usage: alloy [--setup] [-m TEXT] [--profile NAME] [--socket PATH] [--resume ID]
  alloy           interactive chat (connects to existing alloyd)
  alloy -m TEXT   one-shot message
  alloy --setup   write ~/.alloy/settings.json
  alloy chat ...  alias for the same chat path`);
}
function applyOverrides(settings, args) {
    let next = settings;
    if (args.profile)
        next = { ...next, profile: args.profile };
    if (args.socket)
        next = { ...next, socket: args.socket };
    if (args.db)
        next = { ...next, db: args.db };
    if (args.artifact)
        next = { ...next, artifact: args.artifact };
    if (args.alloydBin)
        next = { ...next, alloyd_bin: args.alloydBin };
    return next;
}
async function main() {
    let args;
    try {
        args = parseArgv(process.argv);
    }
    catch (error) {
        console.error(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
        return;
    }
    if (args.help) {
        printHelp();
        return;
    }
    let settings;
    if (args.forceSetup) {
        const partial = {};
        if (args.profile !== undefined)
            partial.profile = args.profile;
        if (args.socket !== undefined)
            partial.socket = args.socket;
        if (args.db !== undefined)
            partial.db = args.db;
        if (args.artifact !== undefined)
            partial.artifact = args.artifact;
        if (args.alloydBin !== undefined)
            partial.alloyd_bin = args.alloydBin;
        settings = await runSetup(partial);
    }
    else {
        settings = applyOverrides(loadSettings() ?? defaultSettings(), args);
    }
    try {
        await ensureDaemon(settings);
    }
    catch (error) {
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
//# sourceMappingURL=main.js.map