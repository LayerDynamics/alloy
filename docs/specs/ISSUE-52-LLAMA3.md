# Issue 52: Native Llama 3.x

## Required outcome

The frozen `llama-3.1-8b-4bit` and `llama-3.1-8b-8bit` profiles in
`bench/corpus/models.json` execute native MLX inference through the existing
engine, daemon, context, tool policy and cache pipeline. Runtime must not invoke
Python or convert weights. Unsupported configurations must fail explicitly.
The frozen artifact revisions and file digests remain unchanged.

## Implementation amendment

1. Extract the existing dense decoder config, weights, bounded verified loader,
   host reference and MLX execution into `crates/inference-infra/src/decoder/`
   (`mod.rs`, `config.rs`, `fixture.rs`, `loader.rs`, `model.rs`,
   `decode_impl.rs`, `mlx_model.rs`). Preserve `qwen3` module exports for callers.
   Use explicit family configuration: Qwen has per-head Q/K RMSNorm with epsilon
   1e-6; Llama omits Q/K norms and uses its validated RMS epsilon. Do not infer
   family identity from tensor resemblance.
2. Add `crates/families/llama3/{Cargo.toml,src/{lib.rs,config.rs,attention.rs,
   rope.rs,template.rs,register.rs},fixture/{chat_template.jinja,
   tool_chat_template.jinja},tests/{reference.rs,cache.rs,quantized_loading.rs}}`.
   Validate `LlamaForCausalLM` plus `model_type=llama`, head divisibility, SiLU,
   no projection bias or sliding attention, and full-head RoPE. Implement
   split-half RoPE with the Llama 3.1 wavelength thresholds and interpolation.
   Reject other scaling types. Distinguish message end, turn end and tool handoff.
3. Extend the shared packed matrix representation and loader for affine 8-bit
   alongside affine 4-bit, retaining packed U32 matrices and f32 scales/biases.
   Validate shape, group size, mode and bit-width before publication. Add custom
   frequency support to `crates/mlx-ffi/src/attention.rs`; reuse existing MLX
   quantized operations and transactional full KV cache.
4. Extend `crates/engine/src/backends/qwen/{backend,artifact,loading,memory,
   cache,output}.rs` as the shared dense adapter, with explicit Llama construction
   and family-specific configuration, templates, stop IDs, registration and cache
   identity. Preserve existing Qwen constructors. Shared rendering, streaming,
   admission, cancellation, tool policy and exact-prefix identity remain single
   implementations. Add `crates/engine/src/backends/llama.rs` as the public Llama
   entry point and update `backends/mod.rs`.
5. Extend `crates/alloyd/src/{boot.rs,main.rs}` to select the requested frozen
   profile and its pinned manifest. Keep existing Qwen boot entry points as
   compatibility wrappers. Reject unknown profiles and architecture mismatches.
6. Update workspace/dependency manifests and lockfile. Add integration coverage
   in the shared adapter and `crates/alloyd/tests/llama_real_model.rs`.
   Add an offline reference capture script under the Llama fixture directory;
   generated diagnostic output and live reference captures go under ~/.codex/.

## Acceptance and evidence

- `llama3_reference_logits`: independent eager reference; strict maximum absolute
  error <0.02 and maximum per-logit relative error <0.01, including near zero
  (user decision 2026-11-13). Relative error is abs(actual-reference)/abs(reference);
  equal zeros have zero error and a nonzero result against zero fails. No floor,
  additive tolerance or relative-L2 substitute is accepted. Report metrics instead of treating token equality as numerical proof.
- `llama3_rope_positions` and `llama31_rope_frequency_transitions`: zero, cached
  offsets, both wavelength boundaries and interpolation.
- `llama31_instruct_stop_tokens`: message-end remains context; turn-end and
  handoff stop generation correctly.
- `llama3_4bit_8bit_load`: both real immutable profiles;
  reject malformed or unsupported layouts before publication.
- `llama3_tool_stop_context_cache`: actual shared pipeline, bounded tool proposals,
  policy-gated execution, exact counting, streamed stops, cached versus cold,
  cancellation, long context, compaction continuation and restart.
- Preserve Qwen reference, loader hardening, engine, harness and daemon regressions.
  Run formatting and clippy. Record terminal results, reviewed source identity,
  dependency pins, artifact hashes, memory and validated context. Tiny fixtures
  and builds alone do not qualify a real profile. Any unmet gate remains open.

## Qualification interface

This document defines the implementation and acceptance protocol. Qualification
results must identify the tested source commit, frozen artifact hashes, dependency
versions and complete terminal receipts; the implementation document alone is
not evidence that a profile passed.

## Numerical protocol fixed before real-model comparison

Reference: mlx 0.32.4, mlx-lm 0.31.2, NumPy 2.4.2. The offline
`fixture/capture_real_reference.py` verifies every corpus file digest before
loading each frozen quantization via upstream mlx-lm. All floating parameters
are cast to f32, preserving the exact packed U32 weight codes. Eager execution
captures all 128256 logits at final prefill and two teacher-forced decode tokens
(19, 13) for each of two literal prompts in the script. Native runs consume
those exact token IDs; no sampling or generated-token alignment changes the
comparison. Every logit must be finite. Cached/cold comparisons are separate.
The tiny fixture uses independent NumPy f64 attention/RoPE over f32 weights.

The tool protocol is the bounded XML proposal grammar already implemented by
the shared parser, explicitly instructed in the Llama template. Calls end with
`<|eom_id|>` (128008); turns with `<|eot_id|>` (128009); end-of-text is 128001.
Header IDs 128006/128007 are never stops. The frozen tokenizer has no normalizer
and byte-level BPE; projected admission charges the complete rendered byte length
plus all bounded serialized result bytes (no subtraction of unchanged text).
Its exact tokenizer digest gates this bound. Tool authorization stays in the
existing supervisor. Llama daemon inputs must match the embedded frozen corpus;
unknown or missing model inputs and cross-profile hashes fail before loading.

Build checkpoints: shared decoder extraction `cargo check -p qwen3 -p engine`;
family math `cargo test -p llama3 --test reference`; regressions
`cargo test -p inference-infra -p qwen3`; real numerical checks separately for
both frozen quantizations; shared engine/harness/daemon conformance last.

## Long-context execution amendment

Llama evaluates each native transformer layer before constructing the next,
keeping one activation graph live. Staged KV buffers remain transactional and
all-layer old/new KV overlap is still reserved. Output projection preserves
eager-reference row order. The daemon uses 64-token Llama prefill chunks, reduced
to 32 beyond 16k capacity to fit the full-f32 cache and workspace on the 36 GiB
host. Qwen's execution schedule is unchanged.

The shared text supervisor gives prompts above 8192 tokens one absolute 600s
startup/first-token deadline. It preserves the 120s control and inter-token
waits, cooperative cancellation, and the tool loop's caller-owned deadline.
This is required because real 32k prefill exceeded the previous 120s wait.

The named real gate is split across `llama_real_model.rs` (tool policy, actual
stops, 8k/32k retrieval, streamed cancellation/recovery, 8 GiB admission) and
`compaction_real_model.rs` (both profiles selected by `ALLOY_REAL_PROFILE`,
compaction, cache/cold equivalence, canonical bytes, SSD process restart,
skill invalidation and 32k rendering). Memory measurements are native MLX
allocator observations, not process RSS or proof of an OS memory bound.

Cold native workers drain unused process-wide MLX allocator buffers before
loading new weights. The pool can outlive a prior worker; real qualification
exposed retained buffers alongside severe swapping during host/native loading.
This reclamation preserves live arrays and admission reservations and is tested
at the native loader boundary with a retained live array and a 64 MiB free pool.

Shared compaction places fixed schema and verification instructions in system
messages, with historical events and candidate checkpoints in separate user
messages explicitly labeled as untrusted data. The strict JSON parsers, source
verification, two-attempt limit and transactional history publication remain
mandatory. This makes Llama's checkpoint generation obey the output contract
without accepting prose or code-fenced JSON.
