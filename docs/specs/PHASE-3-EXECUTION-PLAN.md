# Phase 3 execution plan

Baseline: `4d158b9cd3cf67ceefdd836095167b9ddf60049d`. Phase 2 prerequisite #34 is closed. This plan maps work and required evidence; it does not claim that any gate has passed. Preserve Phase 1/2 regression gates and append-only events. Record changes to this file set and tests in the relevant focused change before implementing them.

## Order and exit gates

1. **#49 Context and admission.** Extend `crates/engine/src/{api.rs,rendering.rs,registry.rs,types/contracts.rs}`, `crates/harness-core/src/context/{builder.rs,tool_history.rs}` and `crates/harness-core/src/supervisor/{coordinator.rs,tool_round.rs,tool_loop.rs}`; adapt `crates/alloyd/src/{service.rs,tool_config.rs}` where configuration or RPC error propagation requires it. Immutable snapshots include all currently available model-facing inputs; skills and checkpoints are wired when their owners land, without falsely claiming unavailable inputs. Bind the exact counted render and its artifact/template/tokenizer/schema identity to generation; reject mismatches. Admit input plus output reserve, margin, and projected tool-result growth, or return an actionable overflow error until #50 supplies compaction. Keep history append-only.
   - Tests: `crates/engine/tests/generation.rs` owns `counted_render_is_used_for_generation_and_stale_identity_is_rejected` and `reused_snapshot_id_cannot_hide_changed_context_from_counted_render`; `crates/engine/tests/projected_context.rs` owns `projected_growth_reserves_output_and_margin_and_rejects_overflow`. `crates/harness-core/tests/canonical_tool_context.rs` checks registry/schema identity and unchanged durable results; `crates/harness-core/tests/tool_loop.rs` checks bounded result retention and structural spellings. `crates/alloyd/tests/rpc_surface.rs::rpc_turn_context_boundary_reserves_output_and_margin` and `crates/alloyd/tests/tool_rpc.rs::projected_batch_rpc_fit_executes_and_overflow_has_no_effect` exercise turn and batch boundaries. `crates/engine/examples/context_32k.rs` performs the release-mode pinned render/count measurement, with warmup, ten samples, exact identities and a <150 ms limit; retain its terminal receipt and candidate SHA.
2. **#50 Transactional compaction.** Extend `crates/harness-core/src/context/`, `crates/harness-core/src/store/`, and supervisor turn paths; add checkpoint schema and transaction/resume tests in `crates/harness-core/tests/`. Preserve every B.3 field, retry malformed output once, commit only with 20% restored headroom, and keep prior checkpoint and events on failure. Test before/after commit crashes and exactly-one checkpoint; run a real-model overflow and continuation.
3. **#51 Cache identity and correctness.** Extend `crates/engine/src/{cache.rs,runtime.rs,types/}` and shared cache infrastructure as necessary. Test B.1 canonical identity, edits/truncation, template/tokenizer/tool/skill changes, compaction, corruption and restart with property seeds. Compare native cached and uncached continuations and RAM/SSD round trips using identical decoding.
4. **#72 and #52, #73, #95–#99 Family implementation and qualification.** Extend `crates/families/`, `crates/inference-infra/`, `crates/engine/src/` registration/loading and `crates/alloyd/` profile selection; add family-specific tests and conformance fixtures. Before each family implementation, record exact new/changed family files, tensor and kernel requirements, artifact pins and named tests in an amendment. Implement actual Llama, Gemma 3/4, Qwen3.5/Ornith, Phi, SmolLM and Mistral/Ministral architectures, alongside Qwen3 text/Coder. No brand-only aliases. Include architecture-specific attention/state, tokenizer/template/stops and quantization; test numerical reference thresholds, tool syntax, context, compaction, cache, streaming, cancellation and restart. Pin source and MLX revisions, license, hashes and conversion recipe; record 8 GB simulated and 36 GB real measurements. A real local policy-gated agent task must pass for each claimed profile. Ornith itself is mandatory and cannot be replaced by upstream Qwen3.5.
5. **#53 M3 exit.** Map each Phase 3 exit statement to candidate-bound terminal CI, real-model results, artifact identities and measurements. Run all Phase 1/2 gates. Keep M3 open until every mandatory family, including Ornith, and context/compaction/cache behavior is directly evidenced.

## Review and evidence

Each focused change retains actual command exit status, candidate SHA, pinned dependencies and artifacts; report host fixtures separately from real-model and performance evidence. A plan, source review, mock, or successful build is not a runtime qualification. Fail closed on missing profile evidence. No later registry, scheduling or conversion milestone is a circular prerequisite for initial M3 native profiles.

### #53 / #72 qualification implementation amendment (2026-11-14)

Start from `ba00e92` with the already delivered context, compaction, cache and
Llama implementations intact. The first implementation batch adds
`crates/engine/src/conformance.rs` and
`crates/engine/tests/m3_profile_conformance.rs`, and extends
`crates/engine/src/{lib.rs,registry.rs}`. It implements machine-readable,
identity-bound qualification and registry rejection of missing, failed or stale
qualification for profiles that require it. Identity includes the candidate,
dependency/pin digest, family implementation, actual artifact manifest,
tokenizer, text/tool templates, profile settings and fixture suite. Numerical
results are calculated from paired finite vectors, not accepted as caller-supplied
pass flags. Enforce both strict FR-8 limits: maximum absolute error <0.02 and
maximum pointwise relative error <0.01, preserving the existing real Llama
qualification rule. A zero reference logit requires exact equality. Report
relative L2 error separately; neither L2 nor an additive absolute-plus-relative
tolerance substitutes for either FR-8 gate.

Named tests: `m3_profile_numerical_reference`,
`m3_profile_requires_every_declared_fixture`,
`m3_profile_identity_changes_invalidate_qualification`,
`m3_profile_failure_revokes_prior_qualification`,
`m3_profile_registry_rejects_missing_or_stale_qualification`, and
`m3_profile_report_roundtrip_preserves_failures`. Fixtures include tools/schema,
stops, exact counting, long context, compaction, cached/uncached execution,
streaming, cancellation, restart, memory tiers and a real policy-gated task.
The report evaluator validates evidence structure and identity; trusted native
fixture execution must supply the observations. Unit tests of the evaluator do
not qualify any real model. Production daemon adoption, native runner/report
issuance and family implementations remain required follow-on batches; neither
#72 nor #53 closes on this batch alone.

### #53 / #98 SmolLM3 native batch (2026-11-14)

Pin source `HuggingFaceTB/SmolLM3-3B` at
`a07cc9a04f16550a088caea529712d1d335b0ac1` and the Apache-2.0 MLX affine
4-bit/group-64 artifact `mlx-community/SmolLM3-3B-4bit` at
`d3a7e0594d6642dbcfb7d149bed8b0bdf49f95ce`. Record file hashes in the new
`crates/families/smollm/fixture/profile.json`; do not modify the frozen benchmark
corpus. The converted configuration omits `tie_word_embeddings`; the source
declares true, the converted weight index contains no `lm_head`, and the pinned
MLX implementation defaults to tied embeddings. Validate this explicit default
and the actual index/weight layout. The artifact tokenizer exceeds 16 MiB, so
increase the bounded metadata allowance for this profile only.

Add `crates/families/smollm/Cargo.toml`, `src/{lib.rs,config.rs,register.rs,template.rs}`,
`fixture/{profile.json,chat_template.jinja,tool_chat_template.jinja,gen_reference.py,reference_logits.json}`,
and `tests/{config.rs,numerical.rs,cache.rs,quantized_loading.rs}`. Extend
`Cargo.toml`, `Cargo.lock`, `crates/inference-infra/src/decoder/{config.rs,model.rs,mlx_model.rs,loader.rs}`,
`crates/families/llama3/src/config.rs`,
`crates/engine/{Cargo.toml,src/backends/qwen/artifact.rs,src/backends/qwen/backend.rs,src/backends/qwen/memory.rs,src/rendering.rs}`,
`crates/alloyd/{Cargo.toml,src/boot.rs,tests/llama_real_model.rs,tests/compaction_real_model.rs}`,
and configuration literals in existing decoder/engine tests if required by the
shared configuration field. Add the family to `.github/workflows/ci.yaml`'s
native verification and extend `crates/engine/tests/unit/rendering_tests.rs`
with pinned-tokenizer projected-result bounds and deterministic template tests.
Native attention retains full KV and causal masks;
only layers selected by the validated NoPE schedule omit RoPE. Explicit schedule
overrides interval derivation; entries must be exactly zero or one. The family
has its own version/cache identity and uses native packed weight loading. Pin
the deterministic non-thinking/XML-tool profile templates; do not claim Python
tools or arbitrary reasoning modes. Reference code is the installed pinned
`mlx-lm==0.31.2` `models/smollm3.py` and its Llama dependency, plus the publisher
configuration above.

Named checks: `smollm3_nope_schedule_override`,
`smollm3_invalid_config_rejected`, `smollm3_reference_logits`,
`smollm3_cached_matches_full_across_nope_layers`,
`smollm3_quantized_native_reference`, and actual shared daemon
tool/context/compaction/cache/stream/cancellation/restart checks. Record native
reference vectors independently before accepting logits. This batch implements
one required family; it does not waive the remaining families or Ornith.

The tiny reference retains the independent NumPy float64 oracle for an absolute
semantic check (maximum error below 0.00002) and captures the specified pinned
upstream eager float32 oracle for FR8's unchanged strict maximum absolute and
pointwise-relative thresholds. The initial float64 comparison had maximum
absolute error 0.00000687762 but relative error 0.07705 near zero; native versus
upstream eager float32 passed both limits on the same fixed inputs. The real
quantized reference remains independently captured, hash-bound and subject to
both strict FR8 thresholds.

For a real tool-response mismatch, extend the existing test-only
`crates/alloyd/tests/support/recording_engine.rs` to optionally emit the actual
generation token inputs. Reuse it in `tests/llama_real_model.rs`; replay those
inputs in the pinned upstream eager environment to distinguish native runtime
errors from model instruction following. Preserve exact task assertions and
record failed qualification; do not normalize model output to obtain a pass.

Because the real SmolLM agent fixture has not passed, normal `boot_native`,
`boot_qwen3`, and the context-override wrapper must reject this profile before
loading weights. Extend `crates/alloyd/src/boot.rs` with an explicitly named
`boot_native_for_qualification` entry point for native conformance runners and
route only the real tests in `tests/{llama_real_model.rs,compaction_real_model.rs}`
through it. Test `unqualified_smollm_cannot_boot_production_or_override_context`.
The daemon CLI must retain its normal boot route; this restriction stays until
current complete native qualification is available through #72.

### #53 / #95 native hybrid-attention operations (2026-11-14)

The mandatory Ornith artifact `ornith-ai/Ornith-1.5-9B-MLX-4bit` at
`a48173b246ac705be75c05bedf1a0666db522d53` declares 32 alternating linear/full
layers (full attention every fourth layer), 16 key heads and 32 value heads in
the recurrent block, 128-wide recurrent heads, and a four-tap depthwise causal
convolution. Full-attention heads are 256-wide with a 0.25 rotary fraction and
an output gate. This is discovery for implementation; source/license pins,
artifact qualification and the smaller upstream profile remain open.

First extend `crates/mlx-ffi/src/ops.rs` and add `src/ops/hybrid.rs` plus
`crates/mlx-ffi/tests/hybrid.rs`. Wrap the existing vendored
`mlx_fast_gated_delta_update` and `mlx_conv1d` with explicit streams, owned
results and validated float32 shape contracts. Add the required native
sigmoid/exp/softplus operations. No Python execution participates in inference.
The recurrence result and final state must match an independent scalar oracle;
every split must reproduce full-sequence state and outputs, including grouped
key/value heads and nonzero initial state. Test the depthwise convolution's
channel and tap ordering, stable softplus at extreme inputs, and invalid shapes
before entering C. Named tests: `gated_delta_matches_scalar_and_chunked_state`,
`hybrid_operations_reject_invalid_shapes`, and
`causal_depthwise_convolution_and_stable_gates`. CPU and Metal runs qualify
these operations only; the actual Ornith loader, model, state persistence and
agent conformance remain mandatory subsequent work.

## Family work packages: planned files and named tests

Each family uses `crates/families/<family>/Cargo.toml`, `src/{lib.rs,config.rs,model.rs,weights.rs,register.rs}`, `tests/{config.rs,numerical.rs,agent_conformance.rs}` and `fixture/` pin manifests; the exact `<family>` directories and special files follow. Shared changes are `Cargo.toml`, `Cargo.lock`, `crates/engine/src/{backends/mod.rs,registry.rs}`, `crates/alloyd/src/boot.rs`, and `bench/corpus/models.json`. These are planned paths, not claims that kernels, weights, or tests already exist. If architecture inspection shows a different decomposition, amend this list before implementing it.

| Issue | Family paths in addition to common set | Named tests and explicit architectural questions |
|---|---|---|
| #52 Llama 3 | `crates/families/llama3/src/{attention.rs,rope.rs,template.rs}`; `tests/{cache.rs,quantized_loading.rs}` | `llama3_reference_logits`, `llama3_rope_positions`, `llama31_rope_frequency_transitions`, `llama31_instruct_stop_tokens`, `llama3_4bit_8bit_load`, `llama3_tool_stop_context_cache` |
| #73 Gemma 3 | `crates/families/gemma3/src/{attention.rs,rope.rs,template.rs}`; `tests/{sliding_attention.rs,quantized_loading.rs}` | `gemma3_reference_logits`, `gemma3_sliding_full_attention`, `gemma3_layer_schedule_config_override`, `gemma3_template_stops`, `gemma3_agent_context_cache` |
| #95 Ornith/Qwen3.5 | `crates/families/qwen35/src/{attention.rs,linear_attention.rs,recurrent_state.rs,template.rs}`; `tests/{recurrent_state.rs,quantized_loading.rs}` | `qwen35_reference_logits`, `qwen35_linear_projection_layout`, `qwen35_recurrent_chunk_equivalence`, `ornith_pinned_native_agent`, `qwen35_recurrent_state_edit_restart`, `ornith_tool_context_compaction_cache` |
| #96 Gemma 4 | `crates/families/gemma4/src/{attention.rs,shared_kv.rs,embeddings.rs,template.rs}`; `tests/{shared_kv.rs,quantized_loading.rs}` | `gemma4_reference_logits`, `gemma4_shared_kv_sliding_full`, `gemma4_embedding_logit_behavior`, `gemma4_agent_context_cache` |
| #97 Phi | `crates/families/phi/src/{attention.rs,rope.rs,template.rs}`; `tests/{position_attention.rs,quantized_loading.rs}` | `phi4mini_reference_logits`, `phi4mini_fused_qkv_gate_layout`, `phi4mini_longrope_boundary`, `phi4mini_position_attention`, `phi4mini_template_stop_tool`, `phi4mini_agent_context_cache` |
| #98 SmolLM | `crates/families/smollm/src/{attention.rs,rope.rs,template.rs}`; `tests/{position_attention.rs,quantized_loading.rs}` | `smollm3_reference_logits`, `smollm3_nope_schedule_override`, `smollm3_attention_position`, `smollm3_reasoning_template_tools`, `smollm3_agent_context_cache` |
| #99 Mistral/Ministral | `crates/families/mistral/src/{attention.rs,rope.rs,template.rs,text_checkpoint.rs}`; `tests/{sliding_attention.rs,quantized_loading.rs}` | `ministral3_reference_logits`, `ministral3_text_checkpoint_loading`, `ministral3_position_sliding_attention`, `ministral3_agent_context_cache` |

The conformance owner #72 adds `crates/engine/tests/m3_profile_conformance.rs`, `crates/alloyd/tests/m3_agent_conformance.rs`, `scripts/m3-profile-conformance.sh`, and `docs/acceptance/phase-3-profile-conformance.md`; named gates are `m3_profile_numerical_reference`, `m3_profile_tool_schema_stop`, `m3_profile_long_context_compaction`, `m3_profile_cached_uncached`, `m3_profile_stream_cancel_restart`, `m3_profile_memory_tiers`, and `m3_profile_real_policy_gated_task`. Qwen3 text and Coder each need a pinned profile and the same applicable gates. Reference thresholds and artifact hashes are recorded with the test fixtures before qualification. Real execution evidence and candidate SHA are required; proposed names do not establish that a test or implementation exists.

### #49 projected tool-result admission design constraint

Admission must happen before `start_tool`, including every call in a batch and the already generated assistant call blocks. Bound the serialized inline JSON, artifact-reference and error representations, template framing, and cumulative earlier results, not merely the raw `inline_bytes` threshold. Obtain the profile's validated context/output/margin from the engine and use a tokenizer-specific, verified upper bound; reject unsupported tokenizer/template configurations rather than assuming that arbitrary UTF-8 byte counts are token counts. Reject before any tool starts if a whole batch cannot be admitted, or explicitly provide a safe per-call continuation policy with durable semantics. The implementation pins the tokenizer identity and bounds normalization, serialization and result framing; unsupported tokenizer identities fail closed. Preserve policy/approval and exactly-once behavior. The implemented regression gates are `projected_growth_reserves_output_and_margin_and_rejects_overflow`, `pinned_normalizer_canonical_decomposition_is_bounded_by_input_bytes`, `projected_budget_reduces_inline_output_without_losing_artifact_bytes`, `pending_projected_admission_honors_deadline_before_any_tool_intent`, `configured_tool_notifications_abort_before_approval_or_execution_on_overflow`, and `projected_batch_rpc_fit_executes_and_overflow_has_no_effect`. The real-model and timing evidence must be tied to the candidate being evaluated.

The frozen Qwen3-4B tokenizer (`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`) uses NFC normalization and a regex split followed by ByteLevel encoding. The tool template emits serialized `call_json` and wraps a JSON-serialized result; `tool_history.rs` first serializes the inline JSON content, then `tool_context.rs` serializes that string again. The model-facing bound must include both escaping stages and the normalizer's actual behavior, not only the raw tool output. Determine effective caps from the immutable local policy binding, account for rejected/denied/error results and artifact references, and evaluate the complete proposed batch before `admit_tool` so overflow cannot leave a partially executed batch. Keep this derivation tied to the pinned tokenizer, template and encoder versions.

### Family architecture discovery (2026-11-12)

The following publisher `main` configurations are discovery evidence only; each family change must resolve an immutable revision, inspect the full implementation and tensor layout, and amend its exact file set before coding. They are not artifact pins or proof of support.

- [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/config.json) declares `Qwen3_5ForConditionalGeneration`; its text configuration lists linear-attention and full-attention layers. The Qwen3.5 package and recurrent-state tests above are required, not a Qwen3 alias.
- [Gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json) declares `Gemma4ForConditionalGeneration` and a nested text configuration with sliding attention and separate head dimensions. The text-loading boundary must explicitly exclude unqualified audio and image capabilities.
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) declares `Phi3ForCausalLM`, partial rotary dimensions and long/short RoPE scaling. The planned position tests must cover both scaling regimes and actual tensor names without runtime remote code.
- [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/config.json) declares `SmolLM3ForCausalLM` and a `no_rope_layers` schedule. Test the layer-specific schedule rather than assuming all Llama-like layers share the same position behavior.

The upstream Transformers implementation at revision [`40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2`](https://github.com/huggingface/transformers/tree/40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2) adds testable architecture constraints, without substituting for publisher artifact access. Its [Llama converter](https://github.com/huggingface/transformers/blob/40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2/src/transformers/models/llama/convert_llama_weights_to_hf.py) specifies the Llama 3.1 frequency-scaling parameters and separate instruct stop tokens; `llama31_rope_frequency_transitions` must cover both frequency thresholds and interpolation, and `llama31_instruct_stop_tokens` must distinguish end-of-message from end-of-turn. Its [Gemma 3 configuration](https://github.com/huggingface/transformers/blob/40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2/src/transformers/models/gemma3/configuration_gemma3.py) derives a sliding/full layer schedule only when an explicit schedule is absent; `gemma3_layer_schedule_config_override` must cover both paths. Reference values and exact checkpoint behavior still require the selected immutable model artifacts.

The same pinned upstream revision supplies additional implementation constraints: [Qwen3.5 text configuration](https://github.com/huggingface/transformers/blob/40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2/src/transformers/models/qwen3_5/configuration_qwen3_5.py) identifies separate linear-attention projections and convolution dimensions; test projection layout and chunked-versus-unchunked recurrent state before qualifying Ornith. [Phi-3 configuration](https://github.com/huggingface/transformers/blob/40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2/src/transformers/models/phi3/configuration_phi3.py) identifies fused QKV and gate/up projections and the original position limit relevant to long scaling; the Phi-4-mini artifact must determine the exact dimensions and scaling parameters. [SmolLM3 configuration](https://github.com/huggingface/transformers/blob/40a1fb90aa125da3c9f90dc293cc1a951b9ca9e2/src/transformers/models/smollm3/configuration_smollm3.py) specifies that `no_rope_layers` entries of zero denote NoPE layers, with a derived schedule only if the explicit schedule is absent; test the explicit override and its layer indexing. These sources guide implementation tests but do not pin model artifacts.

The original publisher configurations for Llama 3.1 and Gemma 3 were not accessible in this check. The [mlx-community Gemma 3 converted configuration](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit/blob/main/config.json) declares a nested Gemma 3 text configuration and a sliding window; this is conversion-side discovery, not verification of the publisher source or a release pin. The publisher's separate [Ministral 3 3B BF16 configuration](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16/blob/main/config.json) declares `Mistral3ForConditionalGeneration`, nested `ministral3` text, YaRN RoPE scaling, and `sliding_window: null`. This BF16 variant informs the proposed tests but does not establish the exact original or MLX artifact. Obtain authorized publisher files and immutable pins before declaring these work packages verified. This discovery does not satisfy any M3 runtime gate.

## #51 prefix cache file set and named acceptance tests

The #51 work package extends `crates/engine/src/{runtime.rs,cache_identity.rs,rendering.rs,backends/qwen/backend.rs,tiered_kv.rs}`. It preserves the separate native RAM, canonical-publication and SSD restore paths, exact tokenizer and token-prefix compatibility, and safe prefill fallback. Family numerical/layout and engine API version rules remain in force.

Tests extend `crates/engine/tests/{generation.rs,support/state_lifecycle.rs,support/runtime_policies.rs,support/runtime_policies_tiered.rs,unit/rendering_tests.rs}` and `crates/alloyd/tests/compaction_real_model.rs`; the generated-test dependency is pinned in `crates/engine/Cargo.toml` and `Cargo.lock`. Named checks include `canonical_publication_rejects_mismatched_prefix_bytes`, `canonical_prefix_requires_matching_lossless_tokenizer_roundtrip`, `canonical_prefix_real_qwen_prompt_and_special_tokens_roundtrip`, `prefix_property_cached_equals_fresh`, `schema_property_cached_equals_fresh`, `tiered_property_restart_edits_and_corruption_match_cold`, `tiered_corrupt_block_or_manifest_rolls_back_partial_restore_and_prefills_safely`, `tiered_process_restart_restores_prefix_and_matches_fresh`, `real_model_generated_history_has_verified_canonical_bytes`, `real_model_ssd_prefix_survives_process_restart`, and `real_model_compaction_preserves_history_and_continues`. Generated tests cover extensions, edits, truncations, template/tokenizer and schema changes, runtime recreation and corruption, but not every combination. Listing tests is not a claim that they pass.

The trained-model numerical test is `real_prefix_numerical::real_prefix_restored_logits_match_cold` in `crates/engine/tests/support/real_prefix_numerical.rs`, included by `crates/engine/tests/generation.rs`. It wraps the actual pinned Qwen backend and its `LoadedModel::autoregressive` result with an observing `AutoregressiveModel`; it captures final prefill logits before first-token sampling and each subsequent decode token and logits without substituting results. It executes the same prompt and greedy options with verified SSD restore and fresh prefill, asserts restored-token and prefill counts, aligns final prefill and subsequent decode positions, rejects nonfinite values, and checks equal token IDs and the user-selected existing elementwise tolerance `abs(actual - expected) <= 0.02 + 0.01 * abs(expected)`. Maximum absolute, relative L2, and maximum pure relative errors are reported separately. Run with the pinned `ALLOY_REAL_MODEL` and native Metal; preserve candidate, artifact, dependency and terminal evidence.

The #51 follow-up adds `session.skill.load {session_id, name, instructions}` for explicitly supplied skill text. The daemon records a versioned, SHA-256-checked `skill.loaded` event; the latest event for each name is projected near the current turn in text and tool snapshots and survives compaction and restart. Names are bounded ASCII identifiers; individual instructions are limited to 64 KiB, with at most 32 skills and 256 KiB total. Skill content is treated as untrusted model context and cannot authorize tools. Existing `skills.list`/`skills.show` discovery remains Phase 4 work. `models.capabilities {artifact_id}` reports the verified loaded model's actual capabilities; Qwen prefix reuse requires a compatible implementation and an enabled retention path. This flag does not guarantee that quota or memory is sufficient to retain a particular cache or that a request will hit it. Tests cover event validation, replacement, compaction, RPC, runtime policy, and `real_skill_change_invalidates_stale_cache_and_matches_cold` / `real_rpc_skill_load_changes_next_generation` with the pinned model.

The #51 adversarial follow-up shares durable and anticipated tool-history projection so loaded skills are counted and serialized before a proposed batch receives execution authority. `harness-core/tests/tool_loop.rs::first_tool_batch_accounts_for_loaded_skills_before_execution` checks block and escaped-JSON rejection with no tool intent or execution, plus fitting controls. The trained compaction test records the actual first checkpoint-continuation request before generation, compares that exact request against a fresh cache-disabled engine, and bounds restored tokens by the matching prefixes of earlier generated prompts; its later reopened-session comparison remains a separate check.

## #50 compaction file set and named acceptance tests

The #50 work package extends the #49 baseline rather than replacing its counted-render identity or projected-batch checks. The implementation files are `crates/engine/src/api.rs`, `crates/engine/src/backends/qwen/backend.rs`, `crates/alloyd/src/{boot.rs,main.rs,service.rs}`, `crates/harness-core/src/context/{mod.rs,builder.rs,tool_history.rs,checkpoint.rs}`, `crates/harness-core/src/store/{session_store.rs,durable/receipts.rs}`, and `crates/harness-core/src/supervisor/{coordinator.rs,tool_round.rs,tool_loop.rs,compaction.rs}`. The bounded profile context is configurable through the boot API for real-model qualification; the production default remains the artifact limit.

Tests are `crates/harness-core/tests/{compaction_runtime.rs,canonical_tool_context.rs,tool_loop.rs,token_count.rs,pause_resume.rs,supervisor_turn.rs}`, `crates/harness-core/tests/supervisor_turn/{turn_boundary.rs,tool_loop_tool_round_tests.rs}`, `crates/harness-core/tests/support/scripted.rs`, `crates/alloyd/tests/{compaction_real_model.rs,engine_errors.rs,rpc_surface.rs}`, and the checkpoint crash tests in `crates/harness-core/src/store/session_store.rs`. Named checks include `checkpoint_requires_every_b3_field_and_supported_version`, `admission_counts_reserves_growth_and_exact_headroom_boundary`, `missing_profile_budget_rejects_before_generation`, `overflow_retries_bad_json_then_commits_and_continues`, `second_invalid_checkpoint_attempt_preserves_committed_checkpoint`, `low_headroom_rejects_without_checkpoint`, `checkpoint_control_spellings_are_escaped_in_context_but_preserved_in_history`, `tool_round_compacts_then_generates_from_checkpoint`, `oversized_projected_batch_compacts_before_tool_authority`, `oversized_batch_without_compactable_history_has_no_tool_effect`, `checkpoint_process_exit_before_and_after_commit_recovers_once`, and `real_model_compaction_preserves_history_and_continues`. Preserve #49's `counted_render_is_used_for_generation_and_stale_identity_is_rejected` and projected-batch tests. `real_model_render_32k_latency` measures five warmed release-mode renders of the pinned 32k corpus document, excluding boot, and records the slowest result.

Review hardening adds `rejected_source_verification_retries_then_preserves_history`, `summary_render_failure_retains_the_original_cause`, `summary_registration_failure_clears_the_registered_request`, `checkpoint_records_conservative_source_provenance`, and `checkpoint_rejects_unowned_durable_artifact_and_preserves_history`. The same session model checks each candidate against labeled source events and a negative or malformed verdict triggers the one stricter retry; original events remain authoritative. The check is a probabilistic model judgment, not a proof of semantic completeness or resistance to prompt injection. Source event identities and trust classes are recorded, and the entire generated checkpoint is treated as untrusted; tool authorization remains a separate policy and receipt decision. Payload provenance increments the event schema version, with older checkpoints treated conservatively as untrusted.

The review file set also includes `crates/engine/src/{error.rs,rendering.rs}` and `crates/engine/tests/unit/backends_qwen_shared_rendering_wiring.rs` for a typed context-render buffer refusal, plus `crates/alloyd/tests/rpc_surface.rs` for the output-reserve boundary. Additional gates are `summary_render_buffer_limit_tries_earlier_completed_boundary`, `schema_v3_checkpoint_migrates_without_rewriting_history`, and `checkpoint_instructions_do_not_authorize_a_tool_effect`.

### #95 native recurrent attention block integration (2026-11-14)

Implement the actual Qwen3.5/Ornith linear attention block in
`crates/inference-infra/src/decoder/hybrid.rs`, exported by `decoder/mod.rs`.
Reuse resident dense/affine projections through crate-visible helpers in
`decoder/mlx_model.rs`; do not dequantize packed matrices. Extend
`crates/mlx-ffi/src/ops/{hybrid.rs}` and `ops.rs` with explicit, shape-checked
broadcasting, scalar scaling, concatenation and unweighted RMS normalization
needed by the trained equations. Keep existing strict equal-shape operations.

The block owns QKV, Z, A, B, convolution, gated normalization and output weights.
Its cache carries both convolution history and recurrent state, bound to the
loaded block identity. Forward produces a new immutable state; cancellation or
failed evaluation must not publish it. Bound weights, state and per-chunk
payloads before constructing native arrays. Reset/rebuild is explicit; no
unsupported recurrent truncation is advertised.

Add `crates/inference-infra/tests/qwen35_hybrid.rs` and
`tests/fixtures/qwen35_hybrid/{generate.py,reference.json}`. The offline generator
uses the pinned mlx-lm 0.31.2 / MLX 0.32.4 reference GatedDeltaNet with deterministic
weights and input; it never runs in production. Named checks:
`qwen35_linear_projection_layout`, `qwen35_recurrent_chunk_equivalence`, and
`qwen35_recurrent_owner_limits_and_reset`. Compare every output and both state
components, every sequence split and token-at-a-time continuation; reject wrong
owners, shape/nonfinite weights and payload overflow without changing prior
state. Native CPU and Metal runs qualify this block, not the whole Ornith model.
The complete hybrid model, bounded real-artifact loader, engine integration,
real agent fixtures and persistent state/rebuild qualification remain required.

### #95 shared verified hybrid tensor loader (2026-11-14)

Extend `crates/inference-infra/src/decoder/loader.rs` only for this batch,
including its existing `hardening_unit_tests` module. Extract the existing
bounded shard/index inventory opening into a shared helper without changing
Qwen/Llama/Smol loading policy. Add a verified tensor-set API whose family
callback receives hash-verified configuration bytes and declares exact logical
matrix and floating tensor shapes. Support rank-three convolution weights and
rank-one decay/norm vectors alongside resident affine 4/8-bit matrices. Reuse
all existing header, source, decoded-payload, cancellation, same-descriptor and
whole-shard hashing rules; reject extras and missing/duplicate tensor requests.
The API returns nothing until every pinned component has verified. It does not
infer trust from a freshly computed artifact hash.

Named tests: `verified_tensor_set_preserves_hybrid_shapes_and_packed_weights`,
`verified_tensor_set_rejects_wrong_shapes_extras_and_limits`, and
`verified_tensor_set_rejects_digest_or_configuration_failure`. Preserve all
existing loader and native model tests. Full Ornith configuration/model/engine
integration and the small upstream profile remain subsequent required work.

### #95 complete native hybrid decoder core (2026-11-14)

Add `crates/inference-infra/src/decoder/qwen35/{mod.rs,config.rs,weights.rs,model.rs}`
and export it from `decoder/mod.rs`. Extend only the crate-visible reusable
projection/logit helpers in `decoder/mlx_model.rs` and validation/array visitation
in `decoder/hybrid.rs`. Parse the verified nested Qwen3.5 text configuration,
validate its explicit linear/full schedule, gated Q projection, partial RoPE,
norm/MLP geometry and quantization declarations. Reject unsupported MoE,
activation, bias or RoPE scaling. Use the shared verified tensor loader for the
exact `language_model` checkpoint names and already-sanitized convolution/norm
layout; never silently shift normalized converted weights a second time.

Execute embedding, alternating GatedDeltaNet and gated full attention, residual
RMS/SwiGLU layers and tied/untied logits in native MLX. Apply RoPE only to the
configured leading rotary channels and retain unrotated tails. Publish all
full-KV layers and recurrent/convolution states only after evaluated finite
logits and cancellation checks. Check model ownership, context/output/weight
payload bounds and measured native buffers. Edits reset/rebuild recurrent state;
no arbitrary recurrent truncation is claimed.

Add `crates/inference-infra/tests/qwen35_model.rs` and
`tests/fixtures/qwen35_model/{generate.py,config.json,model.safetensors,reference.json,manifest.json}`.
The tiny deterministic mixed-layer model is generated offline by pinned MLX
0.32.4 / mlx-lm 0.31.2, with all-position logits and fixed prompt/decode tokens.
Named tests: `qwen35_reference_logits`, `qwen35_partial_rope_cached_continuation`,
`qwen35_model_state_transaction_and_reset`, `qwen35_config_and_tensor_inventory`.
All logits require strict FR8 maximum absolute <0.02 and maximum pointwise
relative <0.01, with the zero-reference rule unchanged. Tiny-model qualification
does not replace real Ornith/upstream artifact tests, engine integration,
policy-gated tool tasks, memory tiers or persistent cache conformance.

The full-model diagnosis found two numerical contracts to preserve: compare
CPU/Metal against their corresponding pinned upstream device and compare each
chunk geometry with the same upstream geometry. The generator retains the
original GPU vectors unchanged and adds CPU/per-split vectors. FR8 thresholds
are unchanged; #51 cached/full equivalence additionally keeps its existing
0.02 + 0.01*abs(reference) bound, and cancellation recovery must equal a fresh
execution with identical chunking exactly.

One Metal chunk still fails strict FR8 because vendored `fast.gated_delta_update`
uses an ops fallback for small head counts and chunked kernels for some larger
sequences, whereas pinned mlx-lm uses a sequential SIMD recurrence. Extend
`crates/mlx-ffi/src/ops.rs`, `src/ops/hybrid.rs`, and add
`src/ops/sequential_delta.rs` for a native explicit-stream Metal implementation
of that recurrence with owned kernel/config/vector handles. Keep the existing
CPU native path. Preserve scalar-oracle, chunk/state and invalid-shape tests
in `crates/mlx-ffi/tests/hybrid.rs`; the unchanged complete Qwen3.5 fixed-vector
tests must pass before this numerical fix is considered verified. No Python
runtime or vendored MLX source mutation is part of this implementation.

### #95 pinned Ornith 9B numerical execution (2026-11-14)

Add `crates/inference-infra/tests/qwen35_quantized.rs` and
`tests/fixtures/ornith_1_5_9b_profile.json`; add `--artifact` / `--output` offline
capture mode to `tests/fixtures/qwen35_model/generate.py` without changing the
tiny fixture's fixed weights or numerical vectors. Pin publisher
`ornith-ai/Ornith-1.5-9B` revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53` and converted artifact
`ornith-ai/Ornith-1.5-9B-MLX-4bit` revision
`a48173b246ac705be75c05bedf1a0666db522d53`, with actual config/index/weight and
tokenizer/template checksums. Record the publisher card's MIT declaration
precisely; a standalone license file has not been found in that artifact.

Capture two fixed token prompts plus two fixed continuation tokens through
pinned eager MLX 0.32.4 / mlx-lm 0.31.2, with all floating weights promoted to
float32 and packed weights kept resident. Freeze the reference digest before
running `ornith_9b_quantized_native_reference` (explicit ignored real-artifact
test). Verify actual model layers, packed matrix residency, cache length, finite
complete vocabulary logits and strict unchanged FR8. Store large reference
output under the authorized Codex artifact directory, not in the checkout.
This numerical gate does not qualify tools, templates, long-context agent
behavior, memory tiers, persistent state or the separately required upstream
Qwen3.5 profile; all remain mandatory subsequent work.
