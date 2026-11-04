# Alloy Phase 0 Benchmark Corpus

Frozen benchmark corpus for SPEC-1 v1.9, Phase 0. This directory is
content-addressed: every scenario, fixture, and model artifact referenced by
`manifest.json` carries a SHA-256 recorded in the manifest and verified by
`validate.sh`.

## Status

**Frozen: 2026-11-03 (declared), 2026-11-04 (all comparison rows verified).**

The structural corpus was frozen on 2026-11-03 with the mlx-community
artifacts resolved and hashed. On 2026-11-04 the three comparison backends
were resolved and verified: llama.cpp (downloaded and hashed), Ollama
(manifest digests pinned), and oMLX (byte-verified loads). Every
comparison row in `models.json` is `verified: true`, and `models.json`
carries `status: frozen`.

## Contents

| Path | What it is |
| --- | --- |
| `manifest.json` | Machine-readable corpus definition (tiers, models, scenarios, metrics) |
| `manifest.schema.json` | JSON Schema the manifest is validated against |
| `models.json` | Five model profiles: pinned mlx-community artifacts, per-file and aggregate SHA-256, quantization recipe, comparison artifacts |
| `scenarios/code-qa.json` | 50 questions over the fixed repository snapshot |
| `scenarios/agentic-loops.json` | 20 multi-turn agentic tool-loop tasks |
| `scenarios/long-context.json` | Scenarios referencing the frozen 8k and 32k documents |
| `scenarios/prefill-heavy.json` | 25 large-prompt scenarios |
| `fixtures/repo-snapshot.tar.gz` | Repository snapshot at the pinned commit |
| `fixtures/repo-snapshot.sha256` | Hash record for the snapshot archive |
| `fixtures/documents/` | The 8k and 32k long-context documents, with `.sha256` companions |
| `resolve_ggufs.py` | One-time llama.cpp artifact verification run |
| `resolve_ollama.py` | One-time Ollama artifact resolution run |
| `resolve_omlx.py` | One-time oMLX load verification run |
| `validate.sh` | Structural validation (named tests 13-16) |

## Hardware tiers

- `mac-36gb`: the physical 36 GB Apple-silicon Mac.
- `simulated-8gb`: a software-budget simulation run on that Mac with an
  explicit 8,000,000,000-byte total memory budget. It is **not** a physical
  8 GB machine and does not reproduce that machine's memory pressure,
  bandwidth, GPU configuration, or thermals. Run reports must never label
  its measurements as physical 8 GB hardware results.

## Model artifact resolution run

One-time local run on the physical Mac, 2026-11-03, using `hf download`
against each pinned revision. All five mlx-community artifacts resolved and
hash-matched their recorded per-file and aggregate digests
(`download_hashes_verified: true` in `models.json`). The tokenizer used for
document token counting is pinned separately under each document entry.

### Pinned artifacts

| Profile | Artifact repository | Revision | Aggregate SHA-256 |
| --- | --- | --- | --- |
| `qwen3-4b-4bit` | `mlx-community/Qwen3-4B-4bit` | `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | `95d369e2b999429fd8f02efdd9c0916718e653cb0a62d58e7f6a2770fee447a5` |
| `qwen3-14b-4bit` | `mlx-community/Qwen3-14B-4bit` | `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` | `166bc507b0d8185c0f1aedf41eef136a6d8c7757387a59ff9adb5e7ecdf467d9` |
| `llama-3.1-8b-4bit` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `241a666dad6cb93c8ff213d39a7f34a36bf26db4` | `a820507ffdc50b15ccac9be35886d8047a8676f593c17016e1f06f86aa8b486c` |
| `llama-3.1-8b-8bit` | `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | `142d428004044c37c441272c91316251d9aecc58` | `f834b4ca26cce5d19a2a26b19bfaf84d6d5e4f614387aeedc7b5bb93b0acbdf0` |
| `gemma-3-12b-4bit` | `mlx-community/gemma-3-text-12b-it-4bit` | `83eda451975103fff6f11e97bf75cd0f1c61af2c` | `7051e534d1b4a77afc860761b32dfd8c3adeff40a17a42ca0eff5539210529c8` |

Source checkpoint revisions recorded in `models.json`:
`Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`,
`Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18`,
`meta-llama/Meta-Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`,
`google/gemma-3-12b-it` @ `96b6f1eccf38110c56df3a15bffe176da04bfd80`.

CI never downloads models. Any future re-verification of these artifacts is
another local run, recorded here.

## llama.cpp comparison verification run

One-time local run on the physical Mac, 2026-11-04, via
`python3 bench/corpus/resolve_ggufs.py`. Each pinned GGUF was downloaded at
its pinned revision and hashed from the downloaded bytes.

| Profile | GGUF repository @ revision | File | SHA-256 | Bytes |
| --- | --- | --- | --- | --- |
| `qwen3-4b-4bit` | `bartowski/Qwen_Qwen3-4B-GGUF` @ `cb76885dc66d50759b207c5a48c4e78dfa00c638` | `Qwen_Qwen3-4B-Q4_K_M.gguf` | `fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3` | 2,497,280,960 |
| `qwen3-14b-4bit` | `bartowski/Qwen_Qwen3-14B-GGUF` @ `bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2` | `Qwen_Qwen3-14B-Q4_K_M.gguf` | `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6` | 9,001,753,632 |
| `llama-3.1-8b-4bit` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` @ `bf5b95e96dac0462e2a09145ec66cae9a3f12067` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` | 4,920,739,232 |
| `llama-3.1-8b-8bit` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` @ `bf5b95e96dac0462e2a09145ec66cae9a3f12067` | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` | `9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283` | 8,540,775,840 |
| `gemma-3-12b-4bit` | `ggml-org/gemma-3-12b-it-GGUF` @ `ec0cbabd8dbff316f659876a50202295c3c4a314` | `gemma-3-12b-it-Q4_K_M.gguf` | `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5` | 7,300,574,976 |

### Checkpoint provenance

The GGUF files carry no `general.source` metadata, so the conversion input
cannot be confirmed from the files themselves. Each row's evidence is the
publisher's model card, which names the original model, plus the Hugging Face
model tree:

- bartowski's Qwen cards name `Qwen/Qwen3-4B` and `Qwen/Qwen3-14B` as the
  original models, quantized with llama.cpp release b5200 using imatrix
  calibration.
- bartowski's Llama card names `meta-llama/Meta-Llama-3.1-8B-Instruct` as the
  original model, quantized with llama.cpp release b3472 using imatrix.
- ggml-org's Gemma model tree runs `google/gemma-3-12b-pt` →
  `google/gemma-3-12b-it` → this GGUF; the recipe is not stated on the card.

All four match the pinned source repositories. The match is
**repository-level, not commit-level**: the converters did not record the
exact source commit, so the same-checkpoint claim rests on the publishers'
statements, recorded per row in `models.json` under `checkpoint_match` and
`checkpoint_evidence_url`.

One modality caveat: the MLX artifact `mlx-community/gemma-3-text-12b-it-4bit`
is the text-only extraction of the Gemma 3 12B IT checkpoint, while the
ggml-org GGUF is the full multimodal model. Both derive from
`google/gemma-3-12b-it`; the corpus scenarios are text-only, but any
cross-backend comparison involving this profile must note that the GGUF
carries the vision tower weights the MLX artifact omits.

## Ollama comparison verification run

One-time local run on the physical Mac, 2026-11-04, via
`python3 bench/corpus/resolve_ollama.py`. Each tag's manifest was fetched
from `registry.ollama.ai` and pinned by the SHA-256 of the manifest bytes;
the layer digests named in each manifest are recorded in `models.json`, so
every model is content-addressed without downloading the weights.

**Tag corrections applied by this run**: the originally recorded tags
`qwen3:4b-instruct-q4_K_M` and `qwen3:14b-instruct-q4_K_M` do not exist in
the Ollama registry. Both were corrected to the tags that carry the original
Qwen3 hybrid checkpoints at q4_K_M.

| Tag | Manifest SHA-256 | Layers | Bytes |
| --- | --- | --- | --- |
| `qwen3:4b-q4_K_M` | `2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0` | 4 | 2,620,787,773 |
| `qwen3:14b-q4_K_M` | `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8` | 4 | 9,276,198,077 |
| `llama3.1:8b-instruct-q4_K_M` | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` | 4 | 4,920,752,841 |
| `llama3.1:8b-instruct-q8_0` | `b158ded76fa05be6bce8a682099ce5df8c5571340a04cf63a2923464679db576` | 4 | 8,540,789,449 |
| `gemma3:12b-it-q4_K_M` | `b92b218582c521ba287f38f1d6057dcd3c6cbff036f13cac97b5d861f5e58ec3` | 5 | 8,164,039,587 |

### Checkpoint provenance

Ollama publishes no Hugging Face source commits for library models. The
evidence for these rows is weaker than the llama.cpp rows and is recorded
per row in `models.json` under `checkpoint_match`:

- The Qwen tags are the **non-suffixed, 40K-context** variants, which are the
  original Qwen3 hybrid checkpoints matching `Qwen/Qwen3-4B` and
  `Qwen/Qwen3-14B`. The 2507-suffixed tags — including Ollama's current
  default `qwen3:4b` — are the July-2025 Instruct/Thinking refreshes and are
  different checkpoints, deliberately avoided.
- The Llama and Gemma tags match the pinned families and sizes per the
  Ollama library pages; repository-level only.
- The pin is the **manifest digest**, not hashed weights. The manifest names
  every layer digest, so the models are content-addressed, but this is a
  weaker verification than the llama.cpp run, which hashed downloaded bytes.
  `verification_method` records this distinction in `models.json`.

The Gemma tag is the multimodal build (Text+Image); the same vision-tower
caveat as the GGUF applies.

## oMLX comparison verification run

One-time local run on the physical Mac, 2026-11-04, via
`python3 bench/corpus/resolve_omlx.py`, using oMLX 0.7.3. For each
profile, the pinned mlx-community artifact was materialized at its pinned
revision into `~/alloy-omlx-models/alloy-<profile-id>` (from the local HF
cache, no re-download), every per-file SHA-256 and the aggregate digest were
re-verified from disk, and oMLX served that exact directory. One chat
completion was sent per model.

| Profile | Served model id | Completion result |
| --- | --- | --- |
| `qwen3-4b-4bit` | `alloy-qwen3-4b-4bit` | ok, `finish_reason: length`, `reasoning_content` |
| `qwen3-14b-4bit` | `alloy-qwen3-14b-4bit` | ok, `finish_reason: length`, `reasoning_content` |
| `llama-3.1-8b-4bit` | `alloy-llama-3.1-8b-4bit` | ok, `finish_reason: stop`, `content` |
| `llama-3.1-8b-8bit` | `alloy-llama-3.1-8b-8bit` | ok, `finish_reason: stop`, `content` |
| `gemma-3-12b-4bit` | `alloy-gemma-3-12b-4bit` | ok, `finish_reason: stop`, `content` |

The two Qwen3 rows are thinking models: with a 64-token budget the model
spent the whole budget reasoning, so the response carried
`reasoning_content` and no `content`. That is accepted as proof of load; the
row records which field was produced.

This is the strongest verification of the three comparison backends: the
exact bytes were re-hashed from disk and oMLX served that same directory,
so what was verified is what was loaded. Note that oMLX merges the supplied
`--model-dir` with any existing local collection; the `alloy-*` model ids
exist only in the hash-verified directory, which is what the check keyed on.

## Comparison artifacts

Four backends are in the comparison set: `alloy`, `llama.cpp`, `ollama`,
and `omlx`. Alloy is the system under test. **oMLX and Ollama are different
backends and appear as separate rows below.**

| Backend | What it consumes | Status |
| --- | --- | --- |
| llama.cpp | Pinned GGUFs (see the verification run above) | **Verified.** Downloaded, hashed, revision-pinned, checkpoint confirmed at repository level |
| Ollama (`ollama`) | Pinned registry tags (see the verification run above) | **Verified.** Manifest digests pinned, layer digests recorded, checkpoint matched at repository level |
| oMLX (`omlx`) | The **same MLX artifact** already pinned above (`artifact_reference: artifact`) | **Verified.** Bytes re-hashed on disk, loaded and served through oMLX 0.7.3 from that exact directory |

Matching nominal bit width alone is not an equivalence criterion; the
per-row provenance records above are what the same-quantization claims rest
on.

## Long-context documents

| Document | Source | License | Target tokens | Actual tokens |
| --- | --- | --- | --- | --- |
| `document-8k.txt` | Project Gutenberg ebook 84, *Frankenstein* (Shelley) | Public domain | 8,192 | 8,192 |
| `document-32k.txt` | Project Gutenberg ebook 2701, *Moby-Dick* (Melville) | Public domain | 32,768 | 32,768 |

Both were prepared by removing Gutenberg headers and footers, normalizing
line endings to LF, and truncating to the target token count. Token counts
were measured with the pinned tokenizer
(`mlx-community/Qwen3-4B-4bit` @ `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`,
`tokenizer.json` SHA-256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`,
`add_special_tokens: false`). Other models may tokenize the same text
differently; the counts are reproducible against this tokenizer, not a
property of the text alone.

## Repository snapshot

`fixtures/repo-snapshot.tar.gz` is the `alloy` repository at commit
`290d95d8b602106b635eb464eaa45f87f285d7ae` (git tree
`55a1276b32ab8f21f3220c8c2716da12f2bbfb7a`), Apache-2.0. The 50 code-QA
questions reference files and symbols in this snapshot only.

## Validation

Run from this directory (requires `python3` with the `jsonschema` package):
