import fs from "node:fs";
import os from "node:os";
import path from "node:path";
export function alloyHome(home = os.homedir()) {
    return path.join(home, ".alloy");
}
export function settingsPath(home = os.homedir()) {
    return path.join(alloyHome(home), "settings.json");
}
export function defaultSettings(home = os.homedir()) {
    const root = alloyHome(home);
    return {
        version: 1,
        profile: "qwen3-4b-4bit",
        socket: path.join(root, "alloyd.sock"),
        db: path.join(root, "sessions.sqlite"),
        artifact: "",
        alloyd_bin: "alloyd",
        corpus_models: null,
    };
}
export function loadSettings(home = os.homedir()) {
    const filePath = settingsPath(home);
    if (!fs.existsSync(filePath)) {
        return null;
    }
    const raw = JSON.parse(fs.readFileSync(filePath, "utf8"));
    if (raw.version !== 1) {
        throw new Error(`unsupported settings version: ${String(raw.version)}`);
    }
    return raw;
}
export function saveSettings(settings, home = os.homedir()) {
    const root = alloyHome(home);
    fs.mkdirSync(root, { recursive: true, mode: 0o700 });
    const filePath = settingsPath(home);
    fs.writeFileSync(filePath, `${JSON.stringify(settings, null, 2)}\n`, {
        mode: 0o600,
    });
}
//# sourceMappingURL=settings.js.map