import fs from "node:fs";
import os from "node:os";
import readline from "node:readline/promises";
import { defaultSettings, loadSettings, normalizeSettingsPaths, resolveSettingsPath, saveSettings, settingsPath, validateSettings, type AlloySettings } from "./settings.js";
import { STARTER_MODEL, discoverModels, downloadModel, inspectModel } from "./model.js";

export type SetupPartial = Partial<Pick<AlloySettings, "profile" | "socket" | "db" | "artifact" | "alloyd_bin" | "corpus_models">>;
export type SetupOptions = {
  download?: boolean;
  interactive?: boolean;
  signal?: AbortSignal;
  report?: (message: string) => void;
  ask?: (prompt: string) => Promise<string>;
  inspect?: typeof inspectModel;
  discover?: typeof discoverModels;
  install?: typeof downloadModel;
};

export class SetupCancelledError extends Error {
  constructor(readonly exitCode = 130) { super("Setup cancelled; existing settings were not changed."); this.name = "SetupCancelledError"; }
}

/** Configure the pinned starter model without launching chat or the daemon. */
export async function runSetup(partial: SetupPartial = {}, home = os.homedir(), options: SetupOptions = {}): Promise<AlloySettings> {
  const report = options.report ?? ((message: string) => { console.error(message); });
  const inspect = options.inspect ?? inspectModel;
  const discover = options.discover ?? discoverModels;
  const install = options.install ?? downloadModel;
  const interactive = options.interactive ?? Boolean(process.stdin.isTTY && process.stderr.isTTY);
  const controller = new AbortController();
  let cancelCode = 130;
  const cancel = (): void => { controller.abort(); };
  const terminate = (): void => { cancelCode = 143; cancel(); };
  const signal = controller.signal;
  let input: readline.Interface | undefined;
  let closingInput = false;
  let saved = false;
  let existing: AlloySettings | null = null;
  try { existing = loadSettings(home); } catch (error) {
    report(error instanceof Error ? error.message : String(error));
    report("Setup will save a repaired configuration only after the selected model passes verification.");
  }
  const base = validateSettings({ ...(existing ?? defaultSettings(home)), ...partial });
  if (base.profile !== STARTER_MODEL.profile) throw new Error(`Supported starter profile: ${STARTER_MODEL.profile}; received ${base.profile}. Run alloy setup --profile ${STARTER_MODEL.profile}.`);
  const selected = normalizeSettingsPaths(base, home);
  if (selected.corpus_models !== null && !fs.statSync(selected.corpus_models, { throwIfNoEntry: false })?.isFile()) {
    if (partial.corpus_models) throw new Error(`Corpus pin file is missing: ${selected.corpus_models}. Supply a readable --corpus-models PATH.`);
    report(`Saved corpus pin file is missing: ${selected.corpus_models}. Using the bundled pins for the same starter model.`);
    selected.corpus_models = null;
  }
  options.signal?.addEventListener("abort", cancel, { once: true });
  if (options.signal?.aborted) cancel();
  process.on("SIGINT", cancel);
  process.on("SIGTERM", terminate);
  let lastProgress = "";
  let lastProgressAt = 0;
  const progress: NonNullable<NonNullable<Parameters<typeof inspectModel>[1]>["onProgress"]> = (event) => {
    const key = `${event.phase}:${event.file ?? ""}`;
    if (key === lastProgress && Date.now() - lastProgressAt < 1000) return;
    lastProgress = key;
    lastProgressAt = Date.now();
    report(`[model] ${event.phase}${event.file ? ` ${event.file}` : ""}${event.bytes ? ` · ${(event.bytes / 1024 ** 2).toFixed(0)} MiB` : ""}`);
  };
  const check = async (candidate: string): Promise<string | null> => {
    signal.throwIfAborted();
    report(`Verifying pinned model: ${candidate}`);
    const result = await inspect(candidate, { verifyHashes: true, signal, onProgress: progress });
    signal.throwIfAborted();
    if (result.ok && result.verified) return result.path;
    report(`Model is not ready: ${result.errors.slice(0, 3).join("; ") || "pinned runtime files were not verified"}`);
    return null;
  };
  const download = async (): Promise<string> => {
    report(`Downloading ${STARTER_MODEL.repository} at ${STARTER_MODEL.revision}. Model weights require a multi-GB download.`);
    const destination = partial.artifact ? resolveSettingsPath(partial.artifact, home) : undefined;
    const result = await install(destination, { home, signal, onProgress: progress });
    signal.throwIfAborted();
    if (!result.ok || !result.verified) throw new Error(`Model installation did not complete: ${result.errors.join("; ")}`);
    return result.path;
  };
  const ask = async (): Promise<string> => {
    signal.throwIfAborted();
    const prompt = "Model directory, 'download' for the pinned Qwen3-4B model, or 'cancel': ";
    if (options.ask) return options.ask(prompt);
    if (!input) {
      input = readline.createInterface({ input: process.stdin, output: process.stderr, terminal: true });
      input.on("SIGINT", cancel);
      input.once("close", () => { if (!closingInput) cancel(); });
    }
    return input.question(prompt, { signal });
  };
  try {
    signal.throwIfAborted();
    report(`Alloy setup · ${STARTER_MODEL.profile}`);
    let artifact: string | null = null;
    const attempted = new Set<string>();
    if (options.download) artifact = await download();
    else if (selected.artifact) {
      attempted.add(selected.artifact);
      artifact = await check(selected.artifact);
      if (artifact === null && partial.artifact) throw new Error("The supplied --artifact is not a verified starter model. Choose a complete local artifact or run alloy setup --download.");
    }
    if (artifact === null && !partial.artifact) {
      const candidates = await discover({ home, configured: selected.artifact });
      for (const candidate of candidates) {
        signal.throwIfAborted();
        if (!candidate.ok || attempted.has(candidate.path)) continue;
        attempted.add(candidate.path);
        artifact = await check(candidate.path);
        if (artifact !== null) break;
      }
    }
    while (artifact === null) {
      if (!interactive) throw new Error("No verified model is configured. Run alloy start in a terminal for guided setup, alloy setup --artifact PATH, or alloy setup --download.");
      const answer = (await ask()).trim();
      signal.throwIfAborted();
      if (/^(cancel|quit|q)$/i.test(answer)) throw new SetupCancelledError();
      if (/^(download|d)$/i.test(answer)) artifact = await download();
      else if (answer) {
        const candidate = resolveSettingsPath(answer, home);
        validateSettings({ ...selected, artifact: candidate });
        artifact = await check(candidate);
      } else report("Choose an existing model directory or type download; no download starts automatically.");
    }
    const settings = validateSettings({ ...selected, artifact });
    signal.throwIfAborted();
    if (JSON.stringify(settings) !== JSON.stringify(existing)) {
      saveSettings(settings, home);
      saved = true;
      report(`Saved configuration: ${settingsPath(home)}`);
    } else report(`Configuration ready: ${settingsPath(home)}`);
    report(`Pinned runtime files verified · ${settings.artifact}`);
    return settings;
  } catch (error) {
    if (!saved && (signal.aborted || error instanceof SetupCancelledError)) throw new SetupCancelledError(cancelCode);
    throw error;
  } finally {
    closingInput = true;
    input?.close();
    process.off("SIGINT", cancel);
    process.off("SIGTERM", terminate);
    options.signal?.removeEventListener("abort", cancel);
  }
}
