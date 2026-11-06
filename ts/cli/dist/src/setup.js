import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { defaultSettings, saveSettings, } from "./settings.js";
async function ask(rl, prompt, fallback) {
    const suffix = fallback !== undefined && fallback !== "" ? ` [${fallback}]` : "";
    return new Promise((resolve) => {
        rl.question(`${prompt}${suffix}: `, (answer) => {
            const trimmed = answer.trim();
            if (trimmed.length > 0) {
                resolve(trimmed);
                return;
            }
            resolve(fallback ?? "");
        });
    });
}
function expandHome(value) {
    if (value.startsWith("~/")) {
        return path.join(os.homedir(), value.slice(2));
    }
    if (value === "~") {
        return os.homedir();
    }
    return value;
}
export async function runSetup(partial = {}, home = os.homedir()) {
    const base = defaultSettings(home);
    if (partial.profile)
        base.profile = partial.profile;
    if (partial.socket)
        base.socket = partial.socket;
    if (partial.db)
        base.db = partial.db;
    if (partial.artifact)
        base.artifact = partial.artifact;
    if (partial.alloyd_bin)
        base.alloyd_bin = partial.alloyd_bin;
    if (partial.corpus_models !== undefined) {
        base.corpus_models = partial.corpus_models;
    }
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stderr,
    });
    try {
        console.error("alloy setup — saved to ~/.alloy/settings.json");
        let artifact = base.artifact;
        while (true) {
            artifact = expandHome(await ask(rl, "Artifact directory (Qwen3-4B-4bit)", artifact || undefined));
            if (!artifact) {
                console.error("artifact path is required");
                continue;
            }
            const configPath = path.join(artifact, "config.json");
            if (!fs.existsSync(configPath)) {
                console.error(`missing ${configPath}`);
                continue;
            }
            break;
        }
        const profile = (await ask(rl, "Profile", base.profile)) || base.profile;
        const socket = expandHome(await ask(rl, "Socket path", base.socket)) || base.socket;
        const db = expandHome(await ask(rl, "Session database", base.db)) || base.db;
        const alloydBin = (await ask(rl, "alloyd binary", process.env.ALLOYD ?? base.alloyd_bin)) || base.alloyd_bin;
        const corpusAnswer = await ask(rl, "Corpus models.json (optional, Enter to skip)", base.corpus_models ?? undefined);
        const corpusModels = corpusAnswer.length > 0 ? expandHome(corpusAnswer) : null;
        const settings = {
            version: 1,
            profile,
            socket,
            db,
            artifact,
            alloyd_bin: alloydBin,
            corpus_models: corpusModels,
        };
        saveSettings(settings, home);
        console.error(`wrote ${path.join(home, ".alloy", "settings.json")}`);
        return settings;
    }
    finally {
        rl.close();
    }
}
//# sourceMappingURL=setup.js.map