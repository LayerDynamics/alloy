import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  STARTER_CORPUS_PATH, STARTER_MODEL, discoverModels, downloadModel, inspectModel,
  type ModelFetch, type PinnedModel,
} from "../src/model.js";

const digest = (bytes: Uint8Array | string): string => createHash("sha256").update(bytes).digest("hex");

/** Tiny transport/preflight fixtures are never accepted as production model pins. */
function fixture(): { files: Map<string, Buffer>; definition: PinnedModel } {
  const header = Buffer.from(JSON.stringify({ fixture: { dtype: "U8", shape: [4], data_offsets: [0, 4] } }));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  const files = new Map<string, Buffer>([
    [".gitattributes", Buffer.from("*.safetensors filter=lfs\n")],
    ["README.md", Buffer.from("Synthetic fixture for installer tests only.\n")],
    ["added_tokens.json", Buffer.from("{}\n")],
    ["config.json", Buffer.from(JSON.stringify({ model_type: "qwen3", architectures: ["Qwen3ForCausalLM"], quantization: { bits: 4, group_size: 64 } }))],
    ["merges.txt", Buffer.from("#version: 0.2\na b\n")],
    ["model.safetensors", Buffer.concat([length, header, Buffer.from([1, 2, 3, 4])])],
    ["model.safetensors.index.json", Buffer.from(JSON.stringify({ weight_map: { fixture: "model.safetensors" } }))],
    ["special_tokens_map.json", Buffer.from("{}\n")],
    ["tokenizer.json", Buffer.from(JSON.stringify({ model: { type: "BPE" }, decoder: { type: "ByteLevel" } }))],
    ["tokenizer_config.json", Buffer.from("{}\n")],
    ["vocab.json", Buffer.from('{"a":0}\n')],
  ]);
  return { files, definition: pins(files) };
}

function pins(files: ReadonlyMap<string, Buffer>): PinnedModel {
  const records = [...files].map(([name, bytes]) => ({ path: name, sha256: digest(bytes) }))
    .sort((a, b) => Buffer.compare(Buffer.from(a.path), Buffer.from(b.path)));
  const hash = createHash("sha256");
  for (const record of records) hash.update(record.path).update("\0").update(record.sha256).update("\0");
  return { profile: "qwen3-test", repository: "alloy-fixture/qwen3", revision: "a".repeat(40), sha256: hash.digest("hex"), files: records };
}

async function temporaryHome(t: TestContext): Promise<string> {
  const home = await fs.mkdtemp(path.join(os.tmpdir(), "alloy-model-test-"));
  t.after(async () => { await fs.rm(home, { recursive: true, force: true }); });
  return home;
}

async function materialize(directory: string, files: ReadonlyMap<string, Buffer>): Promise<void> {
  await fs.mkdir(directory, { recursive: true });
  for (const [name, bytes] of files) await fs.writeFile(path.join(directory, name), bytes);
}

function transport(files: ReadonlyMap<string, Buffer>, requests: string[] = []): ModelFetch {
  return async (url, init) => {
    assert.equal(init.signal.aborted, false);
    requests.push(url);
    const name = decodeURIComponent(new URL(url).pathname.split("/").at(-1)!);
    const data = files.get(name);
    assert.ok(data, `unexpected download: ${name}`);
    let offset = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (offset >= data.length) { controller.close(); return; }
        const next = Math.min(offset + 17, data.length);
        controller.enqueue(data.subarray(offset, next));
        offset = next;
      },
    });
    return new Response(stream, { headers: { "Content-Length": String(data.length) } });
  };
}

async function absent(directory: string): Promise<void> {
  await assert.rejects(fs.lstat(directory), (error: unknown) => (error as NodeJS.ErrnoException).code === "ENOENT");
}

async function assertNoStorage(parent: string): Promise<void> {
  assert.deepEqual((await fs.readdir(parent)).filter((entry) => entry.startsWith(".alloy-model-")), []);
}

test("bundled native corpus and exported starter identity exactly match the frozen repository pins", async () => {
  const corpus = JSON.parse(await fs.readFile(new URL("../../../../bench/corpus/models.json", import.meta.url), "utf8"));
  const original = corpus.profiles.find((profile: { id: string }) => profile.id === "qwen3-4b-4bit");
  const bundled = JSON.parse(await fs.readFile(STARTER_CORPUS_PATH, "utf8"));
  assert.equal(bundled.schema_version, 1);
  assert.equal(bundled.profiles.length, 1);
  assert.equal(bundled.profiles[0].id, original.id);
  assert.equal(bundled.profiles[0].family, original.family);
  assert.deepEqual(bundled.profiles[0].artifact, original.artifact);
  assert.deepEqual(STARTER_MODEL, {
    profile: original.id, repository: original.artifact.repository, revision: original.artifact.revision,
    sha256: original.artifact.sha256, files: original.artifact.files,
  });
  assert.equal(STARTER_MODEL.files.length, 11);
  assert.equal(Object.isFrozen(STARTER_MODEL), true);
  assert.equal(Object.isFrozen(STARTER_MODEL.files), true);
  assert.ok(path.isAbsolute(STARTER_CORPUS_PATH));
  assert.equal(fileURLToPath(new URL("../../models/starter-model.json", import.meta.url)), STARTER_CORPUS_PATH);
});

test("cheap preflight and full verification are separate, and fixtures require an explicit test identity", async (t) => {
  const home = await temporaryHome(t);
  const directory = path.join(home, "model");
  const { files, definition } = fixture();
  await materialize(directory, files);
  const quick = await inspectModel(directory, { definition });
  assert.deepEqual(quick, { ok: true, path: directory, errors: [], verified: false });
  const verified = await inspectModel(directory, { definition, verifyHashes: true });
  assert.deepEqual(verified, { ...quick, verified: true });
  const production = await inspectModel(directory);
  assert.equal(production.ok, false);
  assert.equal(production.verified, false);
  assert.match(production.errors.join(" "), /pinned artifact.*SHA-256 mismatch/);

  const weight = await fs.open(path.join(directory, "model.safetensors"), "r+");
  try { await weight.write(Buffer.from([9]), 0, 1, files.get("model.safetensors")!.length - 1); }
  finally { await weight.close(); }
  assert.equal((await inspectModel(directory, { definition })).ok, true, "cheap preflight must not claim full payload verification");
  const corrupt = await inspectModel(directory, { definition, verifyHashes: true });
  assert.equal(corrupt.ok, false);
  assert.equal(corrupt.verified, false);
  assert.match(corrupt.errors.join(" "), /model\.safetensors.*SHA-256 mismatch/);
});

test("missing model files are actionable and optional documentation is allowed only when absent or pinned", async (t) => {
  const home = await temporaryHome(t);
  const directory = path.join(home, "model");
  const { files, definition } = fixture();
  await materialize(directory, files);
  await fs.rm(path.join(directory, "README.md"));
  await fs.rm(path.join(directory, ".gitattributes"));
  const runtimeOnly = await inspectModel(directory, { definition, verifyHashes: true });
  assert.equal(runtimeOnly.ok, true);
  assert.equal(runtimeOnly.verified, true, "verified covers pinned runtime files when documentation is absent");
  await fs.writeFile(path.join(directory, "README.md"), "changed documentation");
  assert.match((await inspectModel(directory, { definition, verifyHashes: true })).errors.join(" "), /README\.md.*SHA-256 mismatch/);
  await fs.rm(path.join(directory, "tokenizer.json"));
  const missing = await inspectModel(directory, { definition });
  assert.equal(missing.ok, false);
  assert.deepEqual(missing.errors, ["Missing required model file: tokenizer.json."]);
  assert.match((await inspectModel("")).errors.join(" "), /No model artifact is configured.*alloy setup/);
  assert.match((await inspectModel(path.join(home, "absent"))).errors.join(" "), /does not exist/);
});

test("preflight rejects incomplete weights, wrong architecture, and unpinned runtime extras", async (t) => {
  const home = await temporaryHome(t);
  const directory = path.join(home, "model");
  const { files } = fixture();
  files.set("model.safetensors", Buffer.from("version https://git-lfs.github.com/spec/v1\noid sha256:fixture\n"));
  await materialize(directory, files);
  assert.match((await inspectModel(directory, { definition: pins(files) })).errors.join(" "), /Git LFS pointer/);
  files.set("model.safetensors", fixture().files.get("model.safetensors")!);
  files.set("config.json", Buffer.from(JSON.stringify({ model_type: "llama", architectures: ["LlamaForCausalLM"], quantization: { bits: 4, group_size: 64 } })));
  await materialize(directory, files);
  assert.match((await inspectModel(directory, { definition: pins(files) })).errors.join(" "), /requires Qwen3ForCausalLM/);
  const valid = fixture();
  await materialize(directory, valid.files);
  await fs.writeFile(path.join(directory, "generation_config.json"), "{}\n");
  assert.match((await inspectModel(directory, { definition: valid.definition })).errors.join(" "), /Unexpected runtime file generation_config\.json/);
});

test("a root alias is supported while component links outside the canonical artifact are rejected", async (t) => {
  const home = await temporaryHome(t);
  const directory = path.join(home, "model-storage");
  const alias = path.join(home, "model");
  const { files, definition } = fixture();
  await materialize(directory, files);
  await fs.symlink(path.basename(directory), alias, "dir");
  const selected = await inspectModel(alias, { definition, verifyHashes: true });
  assert.equal(selected.ok, true);
  assert.equal(selected.path, alias, "return the stable selected path, not a private storage path");
  const external = path.join(home, "external-tokenizer");
  await fs.writeFile(external, files.get("tokenizer.json")!);
  await fs.rm(path.join(directory, "tokenizer.json"));
  await fs.symlink(external, path.join(directory, "tokenizer.json"));
  assert.match((await inspectModel(alias, { definition })).errors.join(" "), /outside the model directory.*Materialize flat files/);
});

test("preflight rejects non-file required components before opening them", async (t) => {
  const home = await temporaryHome(t);
  const directory = path.join(home, "model");
  const { files, definition } = fixture();
  await materialize(directory, files);
  await fs.rm(path.join(directory, "added_tokens.json"));
  await fs.mkdir(path.join(directory, "added_tokens.json"));
  assert.match((await inspectModel(directory, { definition })).errors.join(" "), /Invalid added_tokens\.json.*regular file/);
});

test("discovery checks only bounded known paths, deduplicates them, and retains explicit missing-path diagnostics", async (t) => {
  const home = await temporaryHome(t);
  const known = path.join(home, "alloy-omlx-models", `alloy-${STARTER_MODEL.profile}`);
  await fs.mkdir(known, { recursive: true });
  await fs.mkdir(path.join(home, "unrelated", "nested", STARTER_MODEL.profile), { recursive: true });
  const configured = path.join(home, "configured-but-missing");
  const models = await discoverModels({ home, configured, env: { ALLOY_QWEN3_ARTIFACT: configured } });
  assert.deepEqual(models.map((model) => model.path), [configured, known]);
  assert.equal(models[0]?.ok, false);
  assert.match(models[0]!.errors.join(" "), /does not exist/);
  assert.equal(models.every((model) => !model.verified), true);
  assert.deepEqual(await discoverModels({ home: path.join(home, "empty-home"), env: {} }), []);
});

test("explicit download streams all exact-revision files and publishes a complete stable alias", async (t) => {
  const home = await temporaryHome(t);
  const { files, definition } = fixture();
  const requests: string[] = [];
  const model = await downloadModel(undefined, { home, definition, fetch: transport(files, requests) });
  const expected = path.join(home, ".alloy", "models", definition.profile);
  assert.deepEqual(model, { ok: true, path: expected, errors: [], verified: true });
  assert.equal((await fs.lstat(expected)).isSymbolicLink(), true);
  assert.equal((await inspectModel(expected, { definition, verifyHashes: true })).ok, true);
  assert.equal(requests.length, 11);
  assert.deepEqual(requests, definition.files.map((file) => `https://huggingface.co/${definition.repository}/resolve/${definition.revision}/${encodeURIComponent(file.path)}`));
  const filesOnDisk = await fs.readdir(expected);
  assert.deepEqual(filesOnDisk.sort(), [...files.keys()].sort());
  const reused = await downloadModel(expected, { definition, fetch: async () => { throw new Error("Existing valid models must not be downloaded again."); } });
  assert.equal(reused.verified, true);
  assert.equal(reused.path, expected);
});

test("a hash mismatch removes only owned partial storage and preserves settings and existing files", async (t) => {
  const home = await temporaryHome(t);
  const parent = path.join(home, ".alloy", "models");
  await fs.mkdir(parent, { recursive: true });
  const settings = path.join(home, ".alloy", "settings.json");
  const originalSettings = '{"artifact":"/existing/model","profile":"custom-profile","keep":"yes"}\n';
  await fs.writeFile(settings, originalSettings);
  const other = path.join(parent, "other-model");
  await fs.mkdir(other);
  await fs.writeFile(path.join(other, "keep"), "existing weights");
  const { files, definition } = fixture();
  files.set("model.safetensors", Buffer.concat([files.get("model.safetensors")!, Buffer.from([99])]));
  const destination = path.join(parent, "selected");
  await assert.rejects(downloadModel(destination, { definition, fetch: transport(files) }), /model\.safetensors failed its pinned SHA-256 check/);
  await absent(destination);
  await assertNoStorage(parent);
  assert.equal(await fs.readFile(settings, "utf8"), originalSettings);
  assert.equal(await fs.readFile(path.join(other, "keep"), "utf8"), "existing weights");
});

test("HTTP failure and truncated bodies leave no published model or partial files", async (t) => {
  const home = await temporaryHome(t);
  const { files, definition } = fixture();
  const validFetch = transport(files);
  const destination = path.join(home, "model");
  await assert.rejects(downloadModel(destination, { definition, fetch: async (url, init) => {
    if (url.endsWith("model.safetensors")) return new Response("unavailable", { status: 503 });
    return await validFetch(url, init);
  } }), /model\.safetensors: HTTP 503/);
  await absent(destination);
  await assertNoStorage(home);
  await assert.rejects(downloadModel(destination, { definition, fetch: async (url, init) => {
    if (url.endsWith("model.safetensors")) return new Response(new Uint8Array(files.get("model.safetensors")!.subarray(0, 10)), { headers: { "Content-Length": "1000" } });
    return await validFetch(url, init);
  } }), /Incomplete download for model\.safetensors/);
  await absent(destination);
  await assertNoStorage(home);
});

test("existing invalid destinations and publication races never overwrite the other owner's path", async (t) => {
  const home = await temporaryHome(t);
  const { files, definition } = fixture();
  const destination = path.join(home, "model");
  await fs.mkdir(destination);
  await fs.writeFile(path.join(destination, "keep"), "original");
  await assert.rejects(downloadModel(destination, { definition, fetch: async () => { throw new Error("must not fetch"); } }), /already exists and was left unchanged/);
  assert.equal(await fs.readFile(path.join(destination, "keep"), "utf8"), "original");
  const race = path.join(home, "created-during-download");
  const validFetch = transport(files);
  let created = false;
  await assert.rejects(downloadModel(race, { definition, fetch: async (url, init) => {
    if (!created) { created = true; await fs.mkdir(race); }
    return await validFetch(url, init);
  } }), (error: unknown) => (error as NodeJS.ErrnoException).code === "EEXIST");
  assert.equal((await fs.lstat(race)).isDirectory(), true, "even an empty destination must survive the publication race");
  assert.deepEqual(await fs.readdir(race), []);
  await assertNoStorage(home);
});

test("cancelling a pending download interrupts the stream and releases its unpublished storage", async (t) => {
  const home = await temporaryHome(t);
  const { files, definition } = fixture();
  const abort = new AbortController();
  const destination = path.join(home, "model");
  const validFetch = transport(files);
  let streamCancelled = false;
  const pending = downloadModel(destination, { definition, signal: abort.signal, fetch: async (url, init) => {
    if (!url.endsWith("model.safetensors")) return await validFetch(url, init);
    return new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(files.get("model.safetensors")!.subarray(0, 8)); },
      pull() { abort.abort(new DOMException("User cancelled setup.", "AbortError")); },
      cancel() { streamCancelled = true; },
    }));
  } });
  await assert.rejects(pending, /User cancelled setup/);
  await absent(destination);
  await assertNoStorage(home);
  assert.equal(streamCancelled, true);
});

test("cancellation before provisioning and an unresponsive transport cannot leave partial artifacts", async (t) => {
  const home = await temporaryHome(t);
  const { definition } = fixture();
  const alreadyAborted = new AbortController();
  alreadyAborted.abort();
  const missingParent = path.join(home, "not-created");
  await assert.rejects(downloadModel(path.join(missingParent, "model"), {
    definition, signal: alreadyAborted.signal, fetch: async () => { throw new Error("must not fetch"); },
  }), { name: "AbortError" });
  await absent(missingParent);
  const destination = path.join(home, "timeout-model");
  let transportSignal: AbortSignal | undefined;
  await assert.rejects(downloadModel(destination, { definition, timeoutMs: 50, fetch: async (_url, init) => {
    transportSignal = init.signal;
    return await new Promise<Response>(() => {});
  } }), /Model download timed out/);
  assert.equal(transportSignal?.aborted, true);
  await absent(destination);
  await assertNoStorage(home);
});

test("oversized responses are rejected before writing, and rejected body streams are cancelled", async (t) => {
  const home = await temporaryHome(t);
  const { definition } = fixture();
  let cancelled = false;
  const destination = path.join(home, "model");
  await assert.rejects(downloadModel(destination, { definition, fetch: async () => new Response(new ReadableStream<Uint8Array>({
    cancel() { cancelled = true; },
  }), { headers: { "Content-Length": "9000000000" } }) }), /Download size is invalid or exceeds the limit/);
  assert.equal(cancelled, true);
  await absent(destination);
  await assertNoStorage(home);
});

test("decoded responses use streamed pinned hashes instead of comparing compressed Content-Length", async (t) => {
  const home = await temporaryHome(t);
  const { files, definition } = fixture();
  const result = await downloadModel(path.join(home, "model"), { definition, fetch: async (url) => {
    const name = decodeURIComponent(new URL(url).pathname.split("/").at(-1)!);
    // Native fetch exposes the decompressed bytes, preserving encoded headers.
    return new Response(new Uint8Array(files.get(name)!), { headers: { "Content-Encoding": "gzip", "Content-Length": "1" } });
  } });
  assert.equal(result.verified, true);
});
