import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";

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

/** Expand the current user's home shorthand and resolve relative paths once. */
export function resolveSettingsPath(value: string, home = os.homedir()): string {
  const expanded = value === "~" ? home : value.startsWith("~/") ? path.join(home, value.slice(2)) : value;
  return path.resolve(expanded);
}

/** Keep setup, read-only checks and launch probes on the same effective paths. */
export function normalizeSettingsPaths(settings: AlloySettings, home = os.homedir()): AlloySettings {
  return {
    ...settings,
    socket: resolveSettingsPath(settings.socket, home),
    db: resolveSettingsPath(settings.db, home),
    artifact: settings.artifact ? resolveSettingsPath(settings.artifact, home) : "",
    // A bare executable name must still be looked up in PATH, not the cwd.
    alloyd_bin: settings.alloyd_bin.includes(path.sep) || settings.alloyd_bin.startsWith("~")
      ? resolveSettingsPath(settings.alloyd_bin, home) : settings.alloyd_bin,
    corpus_models: settings.corpus_models === null ? null : resolveSettingsPath(settings.corpus_models, home),
  };
}

export function loadSettings(home = os.homedir()): AlloySettings | null {
  const filePath = settingsPath(home);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return validateSettings(JSON.parse(fs.readFileSync(filePath, "utf8")));
  } catch (error) {
    throw new Error(`Invalid settings at ${filePath}: ${error instanceof Error ? error.message : String(error)}. Run alloy setup to repair them.`);
  }
}

/** Empty artifact is a valid unconfigured state; malformed settings are not. */
export function validateSettings(value: unknown): AlloySettings {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("expected a settings object");
  const raw = value as Record<string, unknown>;
  if (raw.version !== 1) throw new Error(`unsupported settings version: ${String(raw.version)}`);
  for (const key of ["profile", "socket", "db", "artifact", "alloyd_bin"] as const) {
    if (typeof raw[key] !== "string" || /[\u0000-\u001f\u007f]/.test(raw[key])) throw new Error(`${key} must be a string without control characters`);
    if (key !== "artifact" && !raw[key].trim()) throw new Error(`${key} must not be empty`);
  }
  if (raw.corpus_models !== undefined && raw.corpus_models !== null &&
      (typeof raw.corpus_models !== "string" || !raw.corpus_models.trim() || /[\u0000-\u001f\u007f]/.test(raw.corpus_models))) {
    throw new Error("corpus_models must be a nonempty path or null");
  }
  return { ...raw, corpus_models: raw.corpus_models ?? null } as AlloySettings;
}

export function saveSettings(settings: AlloySettings, home = os.homedir()): void {
  const validated = validateSettings(settings);
  const root = alloyHome(home);
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  const filePath = settingsPath(home);
  const temporary = path.join(root, `.settings-${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(validated, null, 2)}\n`, { mode: 0o600, flag: "wx" });
    fs.renameSync(temporary, filePath);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
}
