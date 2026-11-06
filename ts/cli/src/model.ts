import { constants, existsSync, readFileSync } from "node:fs";
import * as fs from "node:fs/promises";
import type { FileHandle } from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type PinnedModel = {
  profile: string;
  repository: string;
  revision: string;
  sha256: string;
  files: ReadonlyArray<{ path: string; sha256: string }>;
};

export type ModelInspection = {
  ok: boolean;
  /** Stable absolute user/managed path, including a managed root-directory alias. */
  path: string;
  errors: string[];
  /** All pinned runtime files were hashed; this does not prove generation. */
  verified: boolean;
};

export type ModelProgress = {
  phase: "inspect" | "verify" | "download";
  file?: string;
  completedFiles: number;
  totalFiles: number;
  /** Bytes processed in the current file. */
  bytes: number;
  totalBytes?: number;
};

type Progress = (progress: ModelProgress) => void;
export type ModelFetch = (url: string, init: { signal: AbortSignal }) => Promise<Response>;

export type InspectModelOptions = {
  verifyHashes?: boolean;
  signal?: AbortSignal;
  onProgress?: Progress;
  /** Dependency-injection seam; product callers use the bundled starter pins. */
  definition?: PinnedModel;
};

export type DownloadModelOptions = {
  home?: string;
  signal?: AbortSignal;
  onProgress?: Progress;
  /** Transport/identity seams for small deterministic tests, never CLI flags. */
  fetch?: ModelFetch;
  definition?: PinnedModel;
  timeoutMs?: number;
};

const MiB = 1024 * 1024;
const MAX_CONFIG_BYTES = 8 * MiB;
const MAX_TOKENIZER_BYTES = 16 * MiB;
const MAX_WEIGHT_BYTES = 8 * 1024 * MiB;
const MAX_TOTAL_BYTES = MAX_WEIGHT_BYTES + 64 * MiB;
const MAX_ENTRIES = 128;
const DECORATIVE_FILES = new Set(["README.md", ".gitattributes"]);
const sourceBundle = new URL("../models/starter-model.json", import.meta.url);
export const STARTER_CORPUS_PATH = fileURLToPath(existsSync(sourceBundle)
  ? sourceBundle : new URL("../../models/starter-model.json", import.meta.url));

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function validName(name: string): boolean {
  return name.length > 0 && name !== "." && name !== ".." && !/[\\/\u0000-\u001f]/.test(name);
}

function validateDefinition(definition: PinnedModel): void {
  if (!/^[A-Za-z0-9_.-]+$/.test(definition.profile) || definition.profile === "." || definition.profile === ".."
      || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(definition.repository)
      || !/^[a-f0-9]{40}$/.test(definition.revision)
      || !/^[a-f0-9]{64}$/.test(definition.sha256)
      || !Array.isArray(definition.files) || definition.files.length === 0 || definition.files.length > 64) {
    throw new Error("Invalid pinned model definition.");
  }
  const names = new Set<string>();
  const hash = createHash("sha256");
  for (const file of [...definition.files].sort((a, b) => Buffer.compare(Buffer.from(a.path), Buffer.from(b.path)))) {
    if (!validName(file.path) || !/^[a-f0-9]{64}$/.test(file.sha256) || names.has(file.path)) {
      throw new Error("Invalid or duplicate pinned model filename/hash.");
    }
    names.add(file.path);
    hash.update(file.path).update("\0").update(file.sha256).update("\0");
  }
  if (!names.has("config.json") || !names.has("tokenizer.json")
      || !definition.files.some((file) => file.path.endsWith(".safetensors"))
      || hash.digest("hex") !== definition.sha256) throw new Error("Pinned model manifest identity is inconsistent.");
}

function bundledModel(): PinnedModel {
  const corpus = object(JSON.parse(readFileSync(STARTER_CORPUS_PATH, "utf8")));
  const profile = Array.isArray(corpus?.profiles)
    ? corpus.profiles.map(object).find((entry) => entry?.id === "qwen3-4b-4bit") : null;
  const artifact = object(profile?.artifact);
  if (!artifact || !Array.isArray(artifact.files)) throw new Error("Bundled starter model pins are missing.");
  const definition: PinnedModel = {
    profile: "qwen3-4b-4bit", repository: String(artifact.repository), revision: String(artifact.revision),
    sha256: String(artifact.sha256), files: artifact.files.map((entry: unknown) => {
      const file = object(entry);
      return Object.freeze({ path: String(file?.path), sha256: String(file?.sha256) });
    }),
  };
  validateDefinition(definition);
  Object.freeze(definition.files);
  return Object.freeze(definition);
}

/** The same frozen identity used by the native corpus verifier. */
export const STARTER_MODEL = bundledModel();

function abortError(signal?: AbortSignal): Error {
  if (signal?.reason instanceof Error) return signal.reason;
  return new DOMException("Model operation cancelled.", "AbortError");
}

function checkAbort(signal?: AbortSignal): void { if (signal?.aborted) throw abortError(signal); }
function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function missing(error: unknown): boolean { return object(error)?.code === "ENOENT"; }

function absolute(directory: string, home = os.homedir()): string {
  if (directory === "~") return path.resolve(home);
  return path.resolve(directory.startsWith("~/") ? path.join(home, directory.slice(2)) : directory);
}

function fileLimit(name: string): number {
  return name.endsWith(".safetensors") ? MAX_WEIGHT_BYTES
    : name === "tokenizer.json" ? MAX_TOKENIZER_BYTES : MAX_CONFIG_BYTES;
}

function progressReporter(callback?: Progress): (value: ModelProgress, force?: boolean) => void {
  let last = -Infinity;
  return (value, force = false) => {
    const now = performance.now();
    if (callback && (force || now - last >= 250)) { last = now; callback(value); }
  };
}

async function component(root: string, name: string): Promise<{ file: FileHandle; size: number }> {
  let resolved: string;
  try { resolved = await fs.realpath(path.join(root, name)); }
  catch (error) {
    if (missing(error)) throw new Error(`Missing required model file: ${name}.`);
    throw error;
  }
  if (path.dirname(resolved) !== root) {
    throw new Error(`${name} resolves outside the model directory. Materialize flat files; HF cache blob symlinks cannot be loaded.`);
  }
  const before = await fs.stat(resolved);
  if (!before.isFile() || before.size <= 0 || before.size > fileLimit(name)) {
    throw new Error(`Invalid ${name}: expected a nonempty regular file within its size limit.`);
  }
  // O_NONBLOCK also prevents a replaced FIFO from blocking between stat/open.
  const file = await fs.open(resolved, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | constants.O_NONBLOCK);
  try {
    const stat = await file.stat();
    if (!stat.isFile() || stat.size <= 0 || stat.size > fileLimit(name)) {
      throw new Error(`Invalid ${name}: expected a nonempty regular file within its size limit.`);
    }
    return { file, size: stat.size };
  } catch (error) { await file.close(); throw error; }
}

async function readExactly(file: FileHandle, bytes: number, position: number, signal?: AbortSignal): Promise<Buffer> {
  const buffer = Buffer.alloc(bytes);
  let offset = 0;
  while (offset < bytes) {
    checkAbort(signal);
    const { bytesRead } = await file.read(buffer, offset, Math.min(MiB, bytes - offset), position + offset);
    if (bytesRead === 0) throw new Error("Model file was truncated while reading.");
    offset += bytesRead;
  }
  checkAbort(signal);
  return buffer;
}

async function jsonComponent(root: string, name: string, expected: string, signal?: AbortSignal): Promise<Record<string, unknown>> {
  const { file, size } = await component(root, name);
  try {
    const data = await readExactly(file, size, 0, signal);
    const extra = await file.read(Buffer.alloc(1), 0, 1, size);
    if (extra.bytesRead || createHash("sha256").update(data).digest("hex") !== expected) {
      throw new Error(`${name} does not match the pinned artifact (SHA-256 mismatch).`);
    }
    const parsed = object(JSON.parse(data.toString("utf8")));
    if (!parsed) throw new Error(`${name} must contain a JSON object.`);
    return parsed;
  } finally { await file.close(); }
}

async function inspectWeights(root: string, name: string, signal?: AbortSignal): Promise<void> {
  const { file, size } = await component(root, name);
  try {
    if (size < 10) throw new Error(`${name} is incomplete.`);
    const prefix = await readExactly(file, 8, 0, signal);
    if (prefix.toString("ascii") === "version ") throw new Error(`${name} is a Git LFS pointer, not downloaded model weights.`);
    const length = prefix.readBigUInt64LE();
    if (length < 2n || length > BigInt(MAX_CONFIG_BYTES) || length + 8n >= BigInt(size)) {
      throw new Error(`${name} has an invalid or incomplete safetensors header.`);
    }
    const header = object(JSON.parse((await readExactly(file, Number(length), 8, signal)).toString("utf8")));
    const tensors = Object.entries(header ?? {}).filter(([key]) => key !== "__metadata__");
    if (!tensors.length) throw new Error(`${name} contains no tensors.`);
    const payload = size - 8 - Number(length);
    for (const [key, raw] of tensors) {
      const tensor = object(raw);
      const offsets = tensor?.data_offsets;
      if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every((n) => Number.isSafeInteger(n) && n >= 0)
          || offsets[0] > offsets[1] || offsets[1] > payload || !Array.isArray(tensor?.shape)
          || !tensor.shape.every((n) => Number.isSafeInteger(n) && n >= 0) || typeof tensor.dtype !== "string") {
        throw new Error(`${name} has an invalid tensor header: ${key}.`);
      }
    }
  } finally { await file.close(); }
}

async function hashComponent(root: string, name: string, signal: AbortSignal | undefined, report: (bytes: number, total: number) => void): Promise<string> {
  const { file, size } = await component(root, name);
  try {
    const hash = createHash("sha256");
    const buffer = Buffer.alloc(MiB);
    let bytes = 0;
    while (true) {
      checkAbort(signal);
      const read = await file.read(buffer, 0, buffer.length, null);
      if (read.bytesRead === 0) break;
      bytes += read.bytesRead;
      if (bytes > size || bytes > fileLimit(name)) throw new Error(`${name} changed while verifying.`);
      hash.update(buffer.subarray(0, read.bytesRead));
      report(bytes, size);
    }
    checkAbort(signal);
    if (bytes !== size) throw new Error(`${name} changed while verifying.`);
    return hash.digest("hex");
  } finally { await file.close(); }
}

/** Cheap preflight reads only bounded metadata/header bytes; full hashing is explicit. */
export async function inspectModel(directory: string, options: InspectModelOptions = {}): Promise<ModelInspection> {
  const result: ModelInspection = { ok: false, path: directory ? absolute(directory) : "", errors: [], verified: false };
  const definition = options.definition ?? STARTER_MODEL;
  validateDefinition(definition);
  checkAbort(options.signal);
  if (!directory.trim()) { result.errors.push("No model artifact is configured. Run alloy setup."); return result; }
  const report = progressReporter(options.onProgress);
  const pins = new Map(definition.files.map((file) => [file.path, file.sha256]));
  try {
    const root = await fs.realpath(result.path);
    if (!(await fs.stat(root)).isDirectory()) throw new Error(`Model path is not a directory: ${result.path}.`);
    const present = new Set<string>();
    let entries = 0;
    const dir = await fs.opendir(root);
    for await (const entry of dir) {
      checkAbort(options.signal);
      if (++entries > MAX_ENTRIES) throw new Error("Model directory has too many entries; select the flat artifact directory.");
      present.add(entry.name);
      if (entry.name.startsWith(".") || entry.name === "README.md" || entry.isDirectory()) continue;
      const resolved = await fs.realpath(path.join(root, entry.name));
      if (path.dirname(resolved) !== root) throw new Error(`${entry.name} resolves outside the model directory. Materialize flat files; HF cache blob symlinks cannot be loaded.`);
      if (!pins.has(entry.name) && (entry.name.endsWith(".safetensors") || entry.name === "generation_config.json")) {
        throw new Error(`Unexpected runtime file ${entry.name}; this is not the pinned ${definition.profile} artifact.`);
      }
    }
    for (const pin of definition.files) {
      if (!DECORATIVE_FILES.has(pin.path) && !present.has(pin.path)) result.errors.push(`Missing required model file: ${pin.path}.`);
    }
    if (result.errors.length) return result;
    for (const pin of definition.files) {
      if (DECORATIVE_FILES.has(pin.path) && !present.has(pin.path)) continue;
      checkAbort(options.signal);
      const opened = await component(root, pin.path);
      await opened.file.close();
    }
    report({ phase: "inspect", completedFiles: 0, totalFiles: definition.files.length, bytes: 0 }, true);
    const config = await jsonComponent(root, "config.json", pins.get("config.json")!, options.signal);
    const quantization = object(config.quantization) ?? object(config.quantization_config);
    if (config.model_type !== "qwen3" || !Array.isArray(config.architectures)
        || config.architectures.length !== 1 || config.architectures[0] !== "Qwen3ForCausalLM"
        || quantization?.bits !== 4 || quantization.group_size !== 64) {
      throw new Error("The starter profile requires Qwen3ForCausalLM with 4-bit/group-64 MLX quantization.");
    }
    const tokenizer = await jsonComponent(root, "tokenizer.json", pins.get("tokenizer.json")!, options.signal);
    if (object(tokenizer.model)?.type !== "BPE" || object(tokenizer.decoder)?.type !== "ByteLevel") {
      throw new Error("The starter profile requires the Qwen3 BPE/ByteLevel tokenizer.");
    }
    const indexName = "model.safetensors.index.json";
    if (pins.has(indexName)) {
      const index = await jsonComponent(root, indexName, pins.get(indexName)!, options.signal);
      const map = object(index.weight_map);
      if (!map || Object.keys(map).length === 0 || Object.values(map).some((name) =>
        typeof name !== "string" || !validName(name) || !name.endsWith(".safetensors") || !pins.has(name) || !present.has(name))) {
        throw new Error("The safetensors index references missing or unpinned weight files.");
      }
    }
    for (const pin of definition.files.filter((file) => file.path.endsWith(".safetensors"))) {
      await inspectWeights(root, pin.path, options.signal);
    }
    if (options.verifyHashes) {
      const files = definition.files.filter((file) => !DECORATIVE_FILES.has(file.path) || present.has(file.path));
      let completed = 0;
      for (const pin of files) {
        report({ phase: "verify", file: pin.path, completedFiles: completed, totalFiles: files.length, bytes: 0 }, true);
        const actual = await hashComponent(root, pin.path, options.signal, (bytes, totalBytes) => {
          report({ phase: "verify", file: pin.path, completedFiles: completed, totalFiles: files.length, bytes, totalBytes });
        });
        if (actual !== pin.sha256) throw new Error(`${pin.path} does not match the pinned artifact (SHA-256 mismatch).`);
        completed++;
        report({ phase: "verify", file: pin.path, completedFiles: completed, totalFiles: files.length, bytes: 0 }, true);
      }
      result.verified = true;
    }
    checkAbort(options.signal);
    result.ok = true;
    return result;
  } catch (error) {
    checkAbort(options.signal);
    result.errors.push(missing(error) ? `Model directory does not exist or contains a broken link: ${result.path}.` : message(error));
    return result;
  }
}

/** Known locations only. Explicit configured/environment paths retain errors. */
export async function discoverModels(options: { home?: string; configured?: string; env?: NodeJS.ProcessEnv } = {}): Promise<ModelInspection[]> {
  const home = options.home ?? os.homedir();
  const env = options.env ?? process.env;
  const candidates = [
    { value: options.configured, explicit: true },
    { value: env.ALLOY_QWEN3_ARTIFACT, explicit: true },
    { value: path.join(home, ".alloy", "models", STARTER_MODEL.profile), explicit: false },
    { value: path.join(home, "alloy-omlx-models", `alloy-${STARTER_MODEL.profile}`), explicit: false },
  ];
  const seen = new Set<string>();
  const found: ModelInspection[] = [];
  for (const candidate of candidates) {
    if (!candidate.value?.trim()) continue;
    const selected = absolute(candidate.value, home);
    if (seen.has(selected)) continue;
    seen.add(selected);
    if (!candidate.explicit) {
      try { await fs.lstat(selected); } catch (error) { if (missing(error)) continue; }
    }
    found.push(await inspectModel(selected));
  }
  return found;
}

function bounded<T>(promise: Promise<T>, signal: AbortSignal, timeoutMs: number, controller: AbortController, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const cleanup = (): void => { clearTimeout(timer); signal.removeEventListener("abort", cancelled); };
    const cancelled = (): void => { cleanup(); reject(abortError(signal)); };
    const timer = setTimeout(() => { controller.abort(new Error(label)); }, timeoutMs);
    signal.addEventListener("abort", cancelled, { once: true });
    promise.then((value) => { cleanup(); resolve(value); }, (error: unknown) => { cleanup(); reject(error); });
    if (signal.aborted) cancelled();
  });
}

async function writeAll(file: FileHandle, bytes: Uint8Array): Promise<void> {
  let offset = 0;
  while (offset < bytes.length) {
    const written = await file.write(bytes, offset, bytes.length - offset);
    if (!written.bytesWritten) throw new Error("Model download could not write its output file.");
    offset += written.bytesWritten;
  }
}

/** Explicit provisioning only. Never overwrites an existing path or edits settings. */
export async function downloadModel(destination?: string, options: DownloadModelOptions = {}): Promise<ModelInspection> {
  const definition = options.definition ?? STARTER_MODEL;
  validateDefinition(definition);
  const home = options.home ?? os.homedir();
  const selected = absolute(destination ?? path.join(home, ".alloy", "models", definition.profile), home);
  checkAbort(options.signal);
  let exists = false;
  try { await fs.lstat(selected); exists = true; } catch (error) { if (!missing(error)) throw error; }
  if (exists) {
    const existing = await inspectModel(selected, { ...options, verifyHashes: true });
    if (existing.ok) return existing;
    throw new Error(`Destination already exists and was left unchanged: ${selected}. ${existing.errors.join(" ")} Choose another destination or repair it explicitly.`);
  }
  const timeoutMs = options.timeoutMs ?? 60 * 60 * 1000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 24 * 60 * 60 * 1000) throw new Error("Invalid model download timeout.");
  const controller = new AbortController();
  const cancelled = (): void => { controller.abort(abortError(options.signal)); };
  options.signal?.addEventListener("abort", cancelled, { once: true });
  if (options.signal?.aborted) cancelled();
  const timer = setTimeout(() => { controller.abort(new Error("Model download timed out.")); }, timeoutMs);
  const report = progressReporter(options.onProgress);
  const fetchModel: ModelFetch = options.fetch ?? ((url, init) => fetch(url, {
    ...init, redirect: "follow", headers: { "Accept-Encoding": "identity" },
  }));
  let storage: string | null = null;
  let published = false;
  try {
    checkAbort(controller.signal);
    await fs.mkdir(path.dirname(selected), { recursive: true, mode: 0o700 });
    storage = await fs.mkdtemp(path.join(path.dirname(selected), `.alloy-model-${definition.profile}-`));
    let total = 0;
    let completed = 0;
    for (const pin of definition.files) {
      checkAbort(controller.signal);
      const url = `https://huggingface.co/${definition.repository}/resolve/${definition.revision}/${encodeURIComponent(pin.path)}`;
      report({ phase: "download", file: pin.path, completedFiles: completed, totalFiles: definition.files.length, bytes: 0 }, true);
      const request = fetchModel(url, { signal: controller.signal });
      void request.then((response) => { if (controller.signal.aborted) void response.body?.cancel().catch(() => {}); }, () => {});
      const response = await bounded(request, controller.signal, 30_000, controller, `Download connection timed out for ${pin.path}.`);
      if (!response.ok || !response.body) {
        void response.body?.cancel().catch(() => {});
        throw new Error(`Download failed for ${pin.path}: HTTP ${response.status}.`);
      }
      // Fetch decodes Content-Encoding before exposing the stream. A server
      // that ignores identity may report a compressed, not decoded, length.
      const encoding = response.headers.get("content-encoding");
      const lengthHeader = !encoding || encoding.toLowerCase() === "identity"
        ? response.headers.get("content-length") : null;
      const length = lengthHeader === null ? null : Number(lengthHeader);
      if (length !== null && (!Number.isSafeInteger(length) || length < 1 || length > fileLimit(pin.path))) {
        void response.body.cancel().catch(() => {});
        throw new Error(`Download size is invalid or exceeds the limit for ${pin.path}.`);
      }
      const reader = response.body.getReader();
      let output: FileHandle | undefined;
      const hash = createHash("sha256");
      let bytes = 0;
      try {
        output = await fs.open(path.join(storage, pin.path), "wx", 0o600);
        while (true) {
          checkAbort(controller.signal);
          const chunk = await bounded(reader.read(), controller.signal, 60_000, controller, `Download stalled for ${pin.path}.`);
          if (chunk.done) break;
          if (chunk.value.byteLength > 4 * MiB) throw new Error(`Oversized download chunk for ${pin.path}.`);
          bytes += chunk.value.byteLength;
          total += chunk.value.byteLength;
          if (bytes > fileLimit(pin.path) || total > MAX_TOTAL_BYTES) throw new Error(`Download exceeded its size limit for ${pin.path}.`);
          const stableBytes = Buffer.from(chunk.value);
          await writeAll(output, stableBytes);
          hash.update(stableBytes);
          report({ phase: "download", file: pin.path, completedFiles: completed, totalFiles: definition.files.length, bytes,
            ...(length === null ? {} : { totalBytes: length }) });
        }
        checkAbort(controller.signal);
        if (length !== null && bytes !== length) throw new Error(`Incomplete download for ${pin.path}.`);
        if (hash.digest("hex") !== pin.sha256) throw new Error(`Downloaded ${pin.path} failed its pinned SHA-256 check.`);
        await output.sync();
      } finally {
        void reader.cancel().catch(() => {});
        await output?.close();
      }
      completed++;
      report({ phase: "download", file: pin.path, completedFiles: completed, totalFiles: definition.files.length, bytes }, true);
    }
    const inspection = await inspectModel(storage, { definition, signal: controller.signal });
    if (!inspection.ok) throw new Error(`Downloaded model failed preflight: ${inspection.errors.join(" ")}`);
    checkAbort(controller.signal);
    // Directory rename can replace an existing empty directory on POSIX. An
    // exclusive root alias atomically publishes complete storage without that
    // overwrite race. The native loader canonicalizes the artifact root; all
    // individual files remain flat regular files inside that canonical root.
    await fs.symlink(path.basename(storage), selected, "dir");
    published = true;
    return { ok: true, path: selected, errors: [], verified: true };
  } catch (error) {
    controller.abort(error);
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", cancelled);
    if (storage !== null && !published) await fs.rm(storage, { recursive: true, force: true });
  }
}
