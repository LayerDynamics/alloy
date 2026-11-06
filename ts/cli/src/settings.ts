import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type AlloySettings = {
  version: 1;
  profile: string;
  socket: string;
  db: string;
  artifact: string;
  alloyd_bin: string;
  corpus_models: string | null;
};

export function alloyHome(home = os.homedir()): string {
  return path.join(home, ".alloy");
}

export function settingsPath(home = os.homedir()): string {
  return path.join(alloyHome(home), "settings.json");
}

export function defaultSettings(home = os.homedir()): AlloySettings {
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

export function loadSettings(home = os.homedir()): AlloySettings | null {
  const filePath = settingsPath(home);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  const raw = JSON.parse(fs.readFileSync(filePath, "utf8")) as AlloySettings;
  if (raw.version !== 1) {
    throw new Error(`unsupported settings version: ${String(raw.version)}`);
  }
  return raw;
}

export function saveSettings(settings: AlloySettings, home = os.homedir()): void {
  const root = alloyHome(home);
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  const filePath = settingsPath(home);
  fs.writeFileSync(filePath, `${JSON.stringify(settings, null, 2)}\n`, {
    mode: 0o600,
  });
}
