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

The first real Ornith run passed all decode rows but failed prefill strict
pointwise relative error at a near-zero value. The fixed upstream capture
projects all prompt positions before selecting its final row, while the native
test requested last-position-only projection. In `tests/qwen35_quantized.rs`,
compare the frozen reference using the same all-position projection geometry,
selecting the last returned vocabulary row. Retain a separate last-only versus
all-position cache/output equivalence check using the existing #51 bound.
Neither fixed reference values nor FR8 thresholds may change. Both projection
paths must advance identical cache lengths and preserve continuation behavior.

### #95 required small upstream Qwen3.5 numerical execution (2026-11-14)

Add `crates/inference-infra/tests/fixtures/qwen35_0_8b_profile.json` for
`Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`
and `mlx-community/Qwen3.5-0.8B-4bit` revision
`da28692b5f139cb0ec58a356b437486b7dac7462`. Verify downloaded artifacts,
record source/license evidence and exact hashes, and retain all packed weights.
Generalize the offline capture generator with an explicit profile argument
and the real-artifact test helper with profile-specific expected dimensions,
tied-head semantics and vocabulary. Add the explicitly ignored
`qwen35_0_8b_quantized_native_reference` test alongside the mandatory Ornith
test; neither can substitute for the other. Capture/freeze independent eager
reference vectors before native comparison, with identical projection shapes
and unchanged strict FR8. This remains a numerical gate only; both profiles
remain unqualified pending actual engine/agent conformance.

The upstream 0.8B converted artifact includes 153 `vision_tower` tensors.
The text loader must retain its exact inventory rejection. Add offline
`tests/fixtures/qwen35_model/extract_text.py` to verify all pinned source files
and derive a new single-shard text artifact by copying the language tensor
byte ranges unchanged, with a rebuilt exact index. Reject any source namespace
other than language_model/vision_tower, invalid offsets, or existing output.
Record both original and derived file hashes plus recipe digest in the fixed
small-model profile. Verify each retained tensor's dtype, shape and raw payload
against the source before freezing the derived artifact. The pinned eager
reference from the original artifact must also match the derived artifact
capture exactly. No vision runtime support or relaxed production loader is
implied.

### #95 shared hybrid admission calculations (2026-11-14)

Before wiring the engine adapter, expose the native hybrid decoder's checked
cache and step payload calculations so runtime admission cannot drift from
execution. Add `decoder/qwen35/planning.rs`, export it from `qwen35/mod.rs`,
and update `qwen35/model.rs` and `decoder/hybrid.rs` to consume the same
calculations. Expose recurrent state/step payload on LinearAttentionConfig;
Qwen35Config supplies full-KV plus recurrent overlap, output payload and
conservative layer work payload. Reject zero/out-of-context/overflowing inputs
before native allocation. These are named buffer payloads, not process RSS;
engine allocator/loading/output overhead remains separately admitted.
Extend `tests/qwen35_model.rs` with `qwen35_admission_matches_execution` to
check exact logical cache formula, one-byte-below output/work refusal without
state publication, and successful execution at calculated step bounds on
CPU/Metal. Preserve frozen strict numerical vectors and existing tests.

Extend the same shared planning batch in `qwen35/planning.rs` with declared
weight payload and largest-upload-vector bounds, computed from the exact
validated tensor layout and quantization groups. Enforce those declared matrix
representations during `qwen35/model.rs` construction, so a dense/mixed
checkpoint cannot silently exceed a packed load estimate. Extend
`qwen35_model.rs` admission test and `qwen35_quantized.rs` real tests to assert
planned weight payload equals retained payload, and reject representation
mismatch before publication. Numerical refs remain unchanged.

### #95 nested hybrid artifact registration (2026-11-14)

Update `crates/engine/src/registry.rs` to obtain the declared text context
from validated nested Qwen3.5 configuration when the registered architecture
is exactly Qwen3_5ForConditionalGeneration. Preserve flat-family context
handling and reject missing/invalid hybrid metadata before publication.
Add `crates/engine/tests/qwen35_registry.rs` with named tests
`qwen35_registry_reads_validated_nested_context` and
`qwen35_registry_rejects_malformed_hybrid_without_publication`. Registering
metadata does not grant production qualification or route weights through
the homogeneous Qwen3 model.

Registry boundary refinement: keep the shared registry family-independent.
Validate a declared `text_config` object and use its positive context value,
rejecting conflicting outer/nested declarations. Architecture equations stay
in the native family adapter; registration remains metadata inspection.
The named malformed-metadata test covers null/non-object/missing/zero text
context and conflicting outer values, while flat-family registration remains
covered by the existing registry suite.

Focused parallel conformance tests exposed temporary-directory ownership
failure: wall-clock names can collide, and the helper constructs its cleanup
owner before create_dir succeeds. Add a process-local atomic nonce to test
directories and construct cleanup ownership only after successful creation in
`tests/m3_profile_conformance.rs` and the new nested-registry tests. Verify the
normal parallel invocation, not only a serialized workaround. No production
qualification rules change.

### #95 shared engine hybrid execution adapter (2026-11-14)

Extend `crates/engine/src/backends/qwen/{backend.rs,artifact.rs,loading.rs,memory.rs}`
and add `architecture.rs`, `hybrid_loading.rs`, `ornith_text.jinja`,
`qwen35_text.jinja` within that directory. Use an adapter-local family/config
enum to keep actual hybrid configuration separate from the homogeneous decoder.
Add explicit Qwen3.5/Ornith text constructors and pinned selected templates.
Share existing verified metadata, tokenization, rendering identities, admission,
streaming, output bounds and runtime orchestration; load weights only through
MlxQwen35Model and its verified exact tensor loader. Use shared weight/cache/step
plans, measured buffers and explicit native streams. Cache wraps both state
kinds, rejects foreign owners/truncation, and exposes no partial-KV disk export.
Existing runtime reset/rebuild handles edits and restart honestly. Text templates
match each publisher's explicit non-thinking text path; tool mode must reject
until the trained function/parameter protocol is implemented in the next batch.
EOS includes only explicitly pinned configuration/tokenizer EOS declarations.

Add `crates/engine/tests/qwen35_engine.rs`: named tests
`qwen35_engine_native_cache_lifecycle`, `qwen35_engine_rejects_foreign_and_cancelled_state`,
`qwen35_engine_text_template_contract`, and ignored real-artifact
`qwen35_real_engine_streaming`. Verify plans/load/CPU+Metal prefill/decode through
public engine interfaces, exact rebuilt cache versus cold execution, template
history differences, cancellation and missing-tool support rejection. Preserve
Qwen3/Llama/SmolLM regression behavior. Production boot and agent qualification
remain subsequent gates; constructors do not declare either profile qualified.

Real adapter admission observation: the pinned Ornith startup plan requires
9,430,037,496 bytes at the bounded 512-token test context, exceeding simulated
8 GiB. Extend `qwen35_real_engine_streaming` to assert its structured 8 GiB
reservation refusal before native load, then run Ornith streaming at the
36 GiB host tier. Keep upstream 0.8B streaming at 8 GiB. This preserves the
required incompatible-tier rejection; it does not qualify host peak memory
or long-context performance from a reservation calculation alone.

### #95 bounded trained XML call conversion (2026-11-14)

Add `crates/engine/src/slab_parser/xml.rs`, export its bounded conversion API
from `slab_parser/mod.rs`, and factor the existing complete JSON call validator
in `slab_parser/parser.rs` for reuse. Convert complete trained
`<function=...><parameter=...>...</parameter></function>` bodies using only
the caller-admitted call/decoded scratch regions and existing JSON bookkeeping.
Resolve each parameter's declared JSON type through a caller-supplied schema
lookup; never infer numeric/boolean types from string content. Preserve one
template-added leading/trailing newline, escaping and Unicode correctly.
Reject unknown schema/type, malformed structure, duplicate parameters or
nested JSON keys, invalid/non-finite numbers, and capacity overflow before
publishing the proposal. No Python literal evaluation or growing JSON tree.

Add `tests/unit/qwen35_xml_tools.rs` with `xml_schema_types_preserve_values`,
`xml_structure_and_duplicate_keys_fail_closed`,
`xml_fixed_capacity_boundaries_release_reservations`, and
`xml_conversion_success_allocates_no_heap`. This is the bounded call grammar
building block; runtime schema transport, incremental reasoning separation,
trained tool templates and real agent validation remain required before the
hybrid tool-mode rejection is removed.

### #95 incremental XML and reasoning boundary (2026-11-14)

Add `crates/engine/src/slab_parser/xml_stream.rs`, export `XmlToolParser` from
`slab_parser/mod.rs`, and extend `tests/unit/qwen35_xml_tools.rs`. The parser
owns its admitted slab and schema resolver, handles arbitrary UTF-8 chunk
boundaries, routes reasoning to a separate sink, and treats tool-looking
reasoning as inert text. Publish retained proposals only at successful finish;
malformed later output, cancellation/drop, incomplete syntax, capacity refusal
or sink failure discards the whole batch. Reject non-whitespace text after a
tool call as required by the trained tool prompt, while permitting multiple
complete calls. Reuse the bounded converter, never a fallback JSON tree.
Named tests: `xml_stream_all_chunk_boundaries_separate_reasoning`,
`xml_stream_later_failure_discards_every_call`,
`xml_stream_limits_and_sink_failure_release_storage`, and
`xml_stream_success_allocates_no_heap`. Keep production hybrid tool mode
disabled until exact schema transport and publisher tool rendering are wired.

### #95 immutable schema transport to native decoders (2026-11-14)

Update `crates/engine/src/types/{contracts.rs,tool_context.rs}` so snapshots
share immutable ToolContext ownership through Arc. Preserve the public borrowed
accessor and existing schema hash/rendering behavior. Implement XmlToolSchema
against these exact definitions: explicit JSON primitive/container types and
single-type nullable unions; reject unknown tools/parameters, missing types,
ambiguous unions and unsupported composition/reference constructs. Full schema
constraints and tool authority remain downstream. Add Arc delegation in
`slab_parser/xml.rs` without allocating schema copies in the parser.

Extend `crates/engine/src/runtime.rs` with a default-compatible decoder hook
carrying the shared tool context. Validate its schema hash against the counted
render before admission, retain the same owner in Setup, and pass it to the
native worker's decoder construction. Existing JSON backends delegate through
their existing admitted-parser hook. Add
`tests/tool_rendering.rs::xml_tool_context_resolves_declared_types_and_rejects_ambiguity`
and extend the observing native backend in `tests/generation.rs` with
`decoder_receives_the_rendered_schema_owner_on_each_generation` to verify exact
owner/hash transport across changed schemas and removal while actual model
generation still executes. This does not yet remove hybrid tool-mode rejection.

### #95 hybrid tool rendering and decoder integration (2026-11-14)

Update `crates/engine/src/backends/qwen/{architecture.rs,backend.rs,output.rs}`
and add `hybrid_tools.jinja`. Pin separate Ornith/upstream template constants
around their actual assistant-history difference. Preserve the publisher's
function/parameter instructions and grouped tool responses. Extend shared
`rendering.rs` with opt-in Hugging Face JSON/value formatting filters and supply
structured `tools` values; extend `types/tool_context.rs` rendering projections
with structured historical calls without changing canonical input serialization
or schema hashes. Existing templates and JSON protocols keep their formatting.

Connect `XmlToolParser<Arc<ToolContext>>` through the admitted decoder hook;
preserve think markers only for that grammar, route reasoning away from visible
answer/tool proposals, and require the exact schema owner. Count the enlarged
decoder owner through the existing exact Layout plan. Remove hybrid tool-mode
rejection only with this wiring; retain missing slab/schema refusal. Adapt
decoder literals in `tests/unit/backends_qwen_output_{protocol_token,structured_output}.rs`.
Extend `tests/qwen35_engine.rs` with `qwen35_engine_tool_template_contract`,
`qwen35_engine_xml_decoder_uses_advertised_schema`, and ignored
`qwen35_real_engine_tool_proposals`. Compare rendered tools/history against the
pinned publisher templates using the pinned Transformers Jinja environment;
verify schema-typed proposals through real tokenizer IDs and real native
generation. Actual daemon policy-gated tool execution remains the next gate.
Check projected tool-result growth through the same grouped response delimiters
and conservative existing Qwen byte/token calculation before real agent use.

Freeze independent render expectations in
`tests/fixtures/qwen35_tool_render/{input.json,ornith.txt,upstream.txt,generate.py}`.
The offline generator verifies each publisher template against the frozen
artifact profile and renders using the pinned Transformers Jinja environment.
Include Unicode, quoted punctuation, scalar/container arguments, old assistant
history, multiple calls and grouped results. These are product test fixtures.
The existing Qwen projection helper has a Qwen3-only tokenizer pin and different
response-role framing, so add a separate verified hybrid projection bound in
`rendering.rs` and cover its pins/placeholder validation in the hybrid tests;
do not route hybrid projections through the incompatible Qwen3-only helper.

### #95 qualification daemon boot and actual tool authority (2026-11-14)

Update `crates/alloyd/src/boot.rs` to admit the frozen Ornith/Qwen3.5 profiles
only through `boot_native_for_qualification`. Normal boot must reject both
before artifact access until their agent qualification is complete. Use their
actual constructors, text templates, positive nested model context, tokenizer
bounds and immutable compiled artifact hashes; external manifests cannot
replace these pins. Keep hybrid prefill bounded to 32 tokens. Add named tests
`unqualified_hybrids_cannot_boot_production`,
`hybrid_profiles_ignore_external_pin_overrides`, and
`hybrid_context_requires_positive_consistent_metadata`.

Extend `crates/alloyd/tests/llama_real_model.rs::llama3_tool_stop_context_cache`
to run the same actual fs.write effect, exact terminal answer, rejected write,
stream events and capability assertions for both hybrid profiles. Preserve
failed model observations; do not normalize answers or grant production support
from engine-only proposals. Run mandatory actual Ornith first. This batch does
not claim compaction, restart, long-context or broad family qualification.

### #95 complete hybrid state rebuild and retention reporting (2026-11-14)

Update `crates/engine/src/runtime.rs` to retain whether the actual allocated
cache supports complete disk snapshots. Report prefix reuse only when the
configured hot retention or an actually supported disk path can retain state;
a disk directory alone must not advertise hybrid reuse with a zero hot limit.
Add `crates/engine/tests/qwen35_engine.rs::qwen35_engine_retention_capabilities`
covering hot-only, disk-only and no-retention configurations after native work.

Extend `crates/alloyd/tests/compaction_real_model.rs` shared real tests for the
explicit hybrid reset/rebuild contract. Existing homogeneous profiles must
still restore nonzero SSD state. Hybrid runs must show zero spills/restores,
unchanged actual outputs after process restart, exact changed-skill and
post-compaction cold equivalence, preserved append-only events and every B.3
checkpoint field. Update named tests `real_rpc_skill_load_changes_next_generation`,
`real_skill_change_invalidates_stale_cache_and_matches_cold`,
`real_model_ssd_prefix_survives_process_restart`, and
`real_model_compaction_preserves_history_and_continues`. Retain independent
`real_model_generated_history_has_verified_canonical_bytes` and long-context,
streaming/cancellation/memory tests. Rebuilding is an explicit architecture
strategy, not evidence of unsupported recurrent-state SSD restoration.
The new resident metadata also requires updating the existing private fixture
initializer in `crates/engine/tests/unit/runtime_model_pool_runtime.rs`; retain
its current model-pool tests and declare its actual cache persistence support.

### #49/#95 actionable bounded terminal tool errors (2026-11-14)

Actual Ornith completed the authorized write and exact DONE answer, but its
rejected follow-up received a truncated preview that omitted the policy reason.
Fix `crates/harness-core/src/context/tool_history.rs`: project trusted terminal
state and reason as bounded structured fields, preserving the existing error
context byte ceiling and every original durable event. Do not enlarge tool
output admission, normalize model answers or change the actual agent prompt.
Retain the generic completed-error preview behavior. Add
`terminal_error_projection_preserves_policy_reason_and_bounds` in the module
and `terminal_policy_reason_survives_context_without_changing_history` in
`crates/harness-core/tests/canonical_tool_context.rs`, covering long escaped
Unicode reasons, structural markup, no tool execution and unchanged history.
Then rerun the same actual Ornith policy-gated daemon task.

### #49/#95 terminal rejection semantics for actual agent recovery (2026-11-14)

Exact upstream replay reproduced Ornith's duplicate rejected call. A controlled
replay with explicit no-effect semantics instead returned an accurate rejection
and no tools (0784b9b4ebb3462f968ed451fb9463b5). Extend
`crates/harness-core/src/context/tool_history.rs` terminal projection with a
trusted human-readable message for policy rejections/denials, stating the
operation was not executed and must not be retried or bypassed. Keep other
terminal states accurate, especially unknown outcomes, and keep the 256-byte
ceiling with UTF-8-safe, explicitly marked reason truncation. Preserve exact
state/reason and all durable events. Extend the existing named projection and
canonical-history tests; amend `crates/alloyd/tests/llama_real_model.rs` to
require the rejected turn to report rejection, never a success marker, while
retaining the exact one-request/zero-start/no-file-effect assertions. Run the
unchanged real user prompts through Ornith and preserve earlier failed runs.

### #95 resumed compaction diagnostic capture (2026-11-14)

The actual Ornith checkpoint preserved all B.3 facts and passed source
verification, but the reopened follow-up claimed the codename was unavailable.
Extend only `crates/alloyd/tests/support/recording_engine.rs` so existing opt-in
`ALLOY_TEST_TRACE_REQUESTS` captures terminal output for every recorded request,
including partial output on failure. Preserve streaming events unchanged and
retain always-on checkpoint traces. Compile the shared real test targets, then
rerun `real_model_compaction_preserves_history_and_continues` with exact input
and output capture. Replay the failing request in pinned upstream before any
runtime/context change; do not weaken its history or continuation assertions.

### #51/#95 realistic retrieval fixtures and exact acceptance (2026-11-14)

Exact upstream replay d578d964e192420bbaed41794593cccb reproduced the native
resumed refusal. The captured first continuation explicitly treated 500 repeated
`current` words plus a codename request as injection; its refusal mentioned
COBALT, exposing a false positive in the substring assertion. Controlled upstream
10444f2f540142d5a5df19695ae1737f returned exact COBALT for both continuation and
follow-up when the current input used an ordinary document excerpt and asked for
the project name. The same control refused a skill described as secret and
answered its ordinary display-color equivalent. Preserve all original failures;
these controls do not qualify native operation or justify changing authority.

Update only `crates/alloyd/tests/compaction_real_model.rs`: use a fixed 2,000-
character excerpt from the existing pinned benchmark document as current-turn
content, preserving measured actual overflow, the 2,048-token window, every B.3
checkpoint field, durable event equality, real reopen, cold continuation equality
and actual cache-state assertions. Ask for the project name and require exact
COBALT for both continuations so a refusal mentioning it cannot pass. In
`real_rpc_skill_load_changes_next_generation` and
`real_skill_change_invalidates_stale_cache_and_matches_cold`, use ordinary display
colors, explicitly request only the color, and require exact COBALT then AMBER;
retain changed-render/prefix, cold equivalence and cache/capability checks. Route
the real skill fixtures through the existing observational recorder for opt-in
failure replay. Compile the target and rerun these three actual Ornith tests;
production context, policy and decoder behavior stay unchanged in this batch.

### #97 Phi-4-mini pinned configuration and position math (2026-11-14)

Publisher `microsoft/Phi-4-mini-instruct` is pinned at
`cfbefacb99257ffa30c83adab238a50856ac3083` (standalone MIT LICENSE verified).
Preconverted `mlx-community/Phi-4-mini-instruct-4bit` is pinned at
`ac1c269cb4222a4e136a3d09edad301056c1f36a`, affine4/group64. Its configuration
hash is `431eafdf55dc6d4a76dc3e9f22b0d06eac42d9ac6dc7cbd7ff8ca1e38f18cdb1`;
the published packed weight SHA-256 is
`9dcfcdc0a579494283f2d2317ad686123782ce461d1111690ede7c904ff3f87b`.
The real architecture has hidden3072/intermediate8192,32 layers,24 query and8
KV heads of width128,96 rotary dimensions,tied embeddings and vocabulary200064.
QKV and gate/up projections are fused; splitting trained matrix geometry is not
a brand alias or a reason to skip numerical validation.

First add `crates/inference-infra/src/decoder/phi/{mod.rs,config.rs}` and export
it from `src/decoder/mod.rs`. Add exact converted configuration fixture
`crates/inference-infra/tests/fixtures/phi4_mini_config.json` and tests
`crates/inference-infra/tests/phi_config.rs` named
`phi4mini_pinned_geometry_and_longrope_parameters`,
`phi4mini_rejects_unsupported_or_inconsistent_architecture`, and
`phi4mini_longrope_boundary_math_preserves_partial_dimensions`. Validate fused
projection widths, positive bounded dimensions, partial rotary width, both
factor arrays, norms/tied embeddings, full-attention window and affine layout
before weights. Compute short/long frequency denominators and rotary-only
amplitude with explicit regime selection, covering4095/4096/4097 positions and
unrotated tail against an independent scalar formula.

Position semantics need an explicit profile contract: pinned mlx-lm0.31.2
`SuScaledRoPE` uses long factors at every position, whereas current publisher
code has a short/long update route. The native preconverted profile must match
its pinned eager reference and bind the chosen regime to state identity; never
silently change frequencies inside retained KV. This math batch does not settle
or qualify engine profile selection, native model execution, loading, tool
syntax, memory or agent behavior. Record those exact files/tests in the next
amendment after validating the packed artifact and reference contract.

### #97 complete native Phi model and packed-reference batch (2026-11-14)

Extend `crates/inference-infra/src/decoder/phi/{mod.rs,config.rs}` and add
`{weights.rs,model.rs,planning.rs}` in that directory. Use the existing verified
streaming tensor loader, resident `MlxLinear`, transactional `MlxKvCache`, native
RMSNorm/SwiGLU/attention and the validated partial LongRoPE operation. Keep fused
QKV and gate/up matmuls and split their outputs with checked shapes. Support
actual tied/untied heads, explicit native streams, finite logits, cancellation,
owner-bound cache append/reset/truncation, output/work/cache/weight payload limits,
measured backing buffers and invalidation after native failures. Reject extra or
missing tensors, mismatched representations, foreign caches and incompatible
position regimes before publication; do not dequantize resident packed weights.

The first frozen converted Phi profile explicitly uses long-factor SuScaledRoPE
for all positions, matching pinned mlx-lm0.31.2. The loaded model's regime is
immutable and bound to its owner; future engine/cache metadata must include it.
Short-regime operations remain independently tested but are not silently selected
by chunk length or advertised as a qualified alternative profile. Current
publisher sequence-dependent semantics are not claimed by this fixed profile.

Add `crates/inference-infra/tests/{phi_model.rs,phi_quantized.rs}` and product
fixtures `tests/fixtures/phi_model/{config.json,model.safetensors,reference.json,
manifest.json,generate.py}` plus `tests/fixtures/phi4_mini_profile.json` and
`tests/fixtures/phi_model/capture_real_reference.py`. Offline pinned upstream
MLX0.32.4/mlx-lm0.31.2 captures eager float32 references, preserving packed matrices
for the real artifact. Freeze exact config/input/weight/reference identities.
Use the same all-position projection geometry for FR8 and report last-only
execution separately; require maxabsolute<0.02 AND maxpointwiserelative<0.01,
with exact equality for zero reference values. No Python runs in production.

Named tests: `phi4mini_native_reference`,
`phi4mini_native_cached_matches_cold_and_regime_is_owner_bound`,
`phi4mini_native_cancel_and_invalid_inputs_preserve_cache`,
`phi4mini_verified_loader_rejects_wrong_inventory_and_hashes`,
`phi4mini_payload_plans_match_declared_representation`, and ignored
`phi4mini_real_quantized_native_reference`. Run actual CPU/Metal tiny tests,
strict Clippy, affected shared-infrastructure regression and the frozen real
4-bit reference under the host lease. Engine/daemon/tool/agent integration and
production qualification remain the next batch, not claims of these tests.

The complete Phi batch also extends `crates/mlx-ffi/src/attention.rs` with
`longrope_frequencies`: compute exponentiation and factor multiplication on the
explicit native stream, matching the pinned upstream eager graph. Route Phi's
native position operation through it; retain the independent host formula for
validation. The initial scalar-frequency primitive passed its position tolerance,
but host `powf` must not be assumed bit-identical to native MLX for strict full-
model FR8 near zero. Existing Phi CPU/Metal position tests and the new full-model
references cover this addition; no existing Llama/Qwen operation is changed.

### #97 Phi engine adapter and fixed-position cache identity (2026-11-14)

The real packed Phi decoder passes strict FR8 on both fixed prompts and two
cached decode steps (receipt 0f890535ea5546d280bdd0843efec5ad); last-only
selection is bit-identical to the corresponding complete projection. Its eager
oracle is frozen at bf2d5d13e92610c08391c814b45c71505fbd5ebfdedd372f82079393ef6a4729.
This establishes numerical execution, not model/agent qualification.

Extend `crates/engine/src/backends/qwen/{architecture.rs,backend.rs,loading.rs}`
and add `phi_loading.rs` and `phi_text.jinja`. Add a distinct Phi backend/config
dispatch and native loaded model with the existing full-KV transfer adapter.
Bind the immutable long-factor regime in family/cache representation identity;
use Phi's checked packed-weight, full-output, attention-workspace and KV-overlap
plans for admission, including separate measured buffer padding/headroom.
Keep CPU/Metal explicit, cancellation checked during layers, foreign-cache
rejection, reset/truncate and actual block export/restore. Reuse policy and
ownership contracts, not another family's decoder equations. Freeze exact
publisher text framing and both trained end markers. Reject tool-mode selection
until the distinct Phi trained tool protocol is implemented in a subsequent
batch; this intermediate text adapter is not a production profile.

Add `crates/engine/tests/phi_engine.rs` with
`phi_engine_native_cache_lifecycle`,
`phi_engine_rejects_foreign_cancelled_and_unimplemented_tools`,
`phi_engine_text_template_and_position_identity`, and ignored
`phi_real_engine_streaming`. Tiny CPU/Metal tests must check real prefill/decode,
truncation/rebuild, exported/restored continuation and measured memory. The real
frozen artifact test checks exact counted input, streaming completion and equal
cold/repeated output through the runtime at the 8 GiB simulated budget. Run
strict Clippy and affected engine regressions. Daemon selection, trained tools,
long-context/compaction/restart and report issuance remain required next work.

### #97 bounded Phi function-list parser (2026-11-14)

Pinned vLLM v0.18.0 `vllm/tool_parsers/phi4mini_tool_parser.py` documents Phi's
`functools[...]` JSON list, with `name` and either `arguments` or `parameters`.
The publisher example alone produced ordinary code/refusal in upstream probe
b9c0b6f9c9ef4d25a11e9a71220f5ce7; preserve that observation. An explicit grammar
instruction control is queued, not a native tool qualification. Do not treat
the tokenizer's reserved tool tokens as proof of output grammar or promote
ordinary example code to executable calls.

Extend `crates/engine/src/slab_parser/{mod.rs,parser.rs}` and add `phi.rs` plus
`crates/engine/tests/phi_tools.rs`. Implement streaming Phi function lists in
the exact caller-admitted parser slab. Track JSON strings/escapes and balanced
containers so nested arrays and brackets inside strings cannot terminate the
list. Validate each complete object through the existing bounded JSON checker;
permit `parameters` only on this Phi path, with exactly one name and one object
argument member. Preserve duplicate-key, finite-number, UTF-8, depth/key/node,
call/proposal/count and total-output limits. A malformed later call, truncated
list, cancellation/drop or failing text sink must publish no executable batch.
Keep text streaming and proposal access restricted to successful finish.

Named tests: `phi_tools_all_utf8_splits_preserve_nested_arguments`,
`phi_tools_reject_malformed_or_ambiguous_lists_atomically`,
`phi_tools_enforce_every_storage_bound`, and
`phi_tools_success_uses_only_admitted_storage`. Verify no success-path heap
allocation after construction, then run strict Clippy and existing bounded
JSON/XML parser regressions. Native rendering/decoder/daemon wiring is a
separate next batch and must use the actual Phi protocol; parser fixtures do
not qualify a model's tool behavior.

### #97 Phi trained tool rendering and native decoder wiring (2026-11-14)

Literal-format upstream probe 71cbd0a8416042c9949cc2d7644fcd32 produced a valid
`functools` fs.read call with README.md and EOS; its weather output remained
nonconforming. Earlier minimal and prose-format failures remain evidence.
This authorizes implementation of the documented grammar, not a claim of
general instruction-following or native agent qualification.

Extend `crates/engine/src/backends/qwen/{architecture.rs,backend.rs,output.rs}`,
add `phi_tools.jinja`, and extend `crates/engine/src/rendering.rs`. Wire the
admitted Phi parser into the existing incremental ByteLevel output owner and
successful terminal proposal transfer. Remove the explicit intermediate Phi
tool rejection only when this path is wired. Keep other family parsers intact;
update owner literals in
`crates/engine/tests/unit/{backends_qwen_output_protocol_token.rs,backends_qwen_output_structured_output.rs}`.
The compiled profile supplies the literal function-list instruction, all
advertised JSON schemas, publisher role/end framing and deterministic typed
call/result history. Preserve all source messages and tool result linkage.
Bind the distinct tool template digest and existing schema identity to counting
and generation. Add a pinned Phi tokenizer byte-bound projection with actual
normalizer/pretokenizer/postprocessor evidence and overflow checks.

Extend `crates/engine/tests/phi_engine.rs` with
`phi_engine_trained_tool_template_and_admitted_decoder`,
`phi_engine_tool_projection_bounds`, and ignored
`phi_real_engine_tool_proposals`. Add
`crates/engine/tests/fixtures/phi_tool_render/{input.json,expected.txt,generate.py}`
with independently serialized Python/publisher-template expected bytes. Check
typed nested schemas/history, all decoder splits, missing/wrong slabs, malformed
later calls and successful atomic proposal transfer; compare pinned actual
token counts and Unicode/control-result growth bounds. Run real Phi cold/repeat
fs.read proposals, strict Clippy and affected output/rendering/parser tests.
The daemon policy-gated file task and full conformance remain subsequent work.

### #97 Phi daemon qualification and actual agent lifecycle (2026-11-14)

The actual frozen Phi artifact passed bounded tool projection, repeated text
generation and cold/repeated fs.read proposals (f4f721ad590e4d6488fa9e2677adf0f9,
three tests, exit zero). This does not establish daemon tool execution or full
profile qualification. Preserve prior upstream format failures.

Extend `crates/alloyd/src/boot.rs` to recognize `phi4-mini-instruct-4bit`, pin
the compiled Phi manifest regardless of external overrides, select its own text
template/native backend, and bound prefill at 64. Refuse Phi in all normal boot
routes while full qualification remains incomplete. Add named tests
`phi_profile_ignores_external_pin_overrides` and
`unqualified_phi_cannot_boot_production_or_override_context` covering immutable
pins, missing/modified files, and every normal boot route with/without tools.

Extend `crates/alloyd/tests/llama_real_model.rs` shared profile/capability
expectations for Phi's distinct family and fixed-long cache identity; retain
the unchanged policy-gated file task, exact DONE answer, denied-write behavior,
8k/32k stream/memory checks and cancellation recovery. Run its real tool task,
then the existing `compaction_real_model` actual overflow, skill changes,
canonical history and process-restart tests with Phi. Full-KV Phi must retain
SSD restore requirements; it is not a hybrid rebuild profile. Run strict
Clippy and focused boot regressions before real artifact tests. Retain failures
and diagnose exact inputs against the pinned oracle before speculative edits.

### #97 Phi function-list turn boundary and canonical stop coverage (2026-11-14)

Daemon receipt 190ff59eb02d49a8b23945050e58cdd6 wrote the authorized file once
but failed the exact answer: Phi continued beyond its complete function list
with premature DONE. Upstream replay 047995425e2147ae90507ef9c6875f73 reproduced
both turns exactly. Two stronger prose instructions in d084c67ca20b4294a85722b57d95581b
did not fix the boundary. Implement the protocol's complete nonempty function
list as an explicit profile end condition, before another model token is run.
Do not strip already-generated answer text or synthesize an EOS token.

Extend `crates/engine/src/slab_parser/phi.rs` with opt-in terminal-list mode,
completed-boundary observation and rejection of non-whitespace after that list
within the same token. Existing general parser behavior remains separately
tested. Extend `crates/engine/src/runtime.rs` TextDecoder with a default-false
profile-end observer and consult it after actual token execution/decoding.
Extend `crates/engine/src/backends/qwen/{backend.rs,output.rs,phi_tools.jinja}`
to select this only for Phi tools; bind the changed contract in template source
identity without changing rendered user content. EOS is the profile condition;
usage/cache must include only the actually executed closing token, not invented
tokens. Keep cancellation, final validation and atomic proposal publication.

Extend `crates/engine/tests/{phi_tools.rs,phi_engine.rs}` with
`phi_tools_terminal_list_boundary_is_atomic` and native decoder boundary checks:
all character splits, nested brackets, multiple calls in one list, malformed
later objects, whitespace at the boundary, same-token suffix rejection, no
boundary for ordinary text, no proposals before finish. Strengthen real Phi
tool tests to require empty ordinary text and actual closing-token usage.
Extend `crates/engine/tests/unit/runtime_decoder_contract.rs` to prove the new
default preserves legacy behavior. Run strict Clippy, focused parser/decoder
checks, full engine regression and the unchanged actual daemon file task.

The running canonical-history test also exposed omitted Phi `<|end|>` among
its candidate EOS spellings. Add that actual configured token to
`crates/alloyd/tests/compaction_real_model.rs`; retain the exact token/byte
roundtrip requirement. Invalid checkpoint arrays are a separate unresolved
failure, not permission to accept malformed compaction or qualify Phi.

### #97 Phi policy-result interpretation (2026-11-14)

Structural stopping passed all seven actual Phi adapter tests in
6fb7db153b9847e6a653e4d7ca34a362. Daemon rerun
98aaa7e43425417180188023b8cb61ec now writes once and returns exactly DONE after
the successful result; the forbidden write is rejected without execution or
retry. The remaining assertion fails because Phi attributes rejection to prior
user instructions. Exact upstream replay f51b1562711c4861917953c0cefb8cb5
reproduces it; an explicit general error-result instruction reports the actual
policy rejection. Do not change policy or the task assertions.

Extend `crates/engine/src/backends/qwen/phi_tools.jinja` and independently
serialized `crates/engine/tests/fixtures/phi_tool_render/{generate.py,expected.txt}`
with the tested instruction: interpret is_error, report the stated failure,
identify rejected/not-executed operations accurately, and do not retry or claim
success. Existing fixture contains both successful and failed results. Run
`phi_engine_trained_tool_template_and_admitted_decoder`, actual
`phi_real_engine_tool_proposals`, and the unchanged daemon
`llama3_tool_stop_context_cache` with Phi, plus strict Clippy. Keep the original
failed receipts and require all real task assertions before claiming that gate.

### #73 Gemma 3 native text core (2026-11-15)

The frozen 12B text artifact is already present and remains mandatory alongside
an additional small profile. Implement its actual decoupled head width, six
normalizations per block with (1 + weight), BF16-rounded embedding multiplier,
approximate GELU gating, explicit query scale, local/global RoPE and mixed
sliding/full KV. Do not route it through a Llama alias. This batch establishes
native numerical/cache behavior; daemon registration and both real agent
profiles remain required subsequent integration.

Exact files: `crates/mlx-ffi/src/{attention.rs,ops.rs}`;
`crates/inference-infra/src/decoder/mod.rs`;
`crates/inference-infra/src/decoder/gemma3/{mod.rs,config.rs,weights.rs,planning.rs,model.rs}`;
`crates/inference-infra/tests/{gemma3_config.rs,gemma3_model.rs}`;
`crates/inference-infra/tests/fixtures/gemma3_model/{generate.py,config.json,model.safetensors,reference.json,manifest.json}`.
Use pinned mlx-lm 0.31.2/MLX 0.32.4 as an independent offline tiny-model oracle,
with compile disabled and explicit CPU/Metal checks. Keep production native.
The existing frozen benchmark pins own the actual 12B source/artifact identity.

Named tests: `gemma3_layer_schedule_config_override`,
`gemma3_rejects_unsupported_equations`, `gemma3_reference_logits`,
`gemma3_sliding_full_attention`, `gemma3_cancel_and_foreign_cache_are_atomic`,
`gemma3_verified_loading_and_payload_bounds`,
`gemma3_native_gelu_and_sliding_attention`. Exercise every chunk split across
window boundaries, explicit schedule override, distinct query/head widths,
4/8-bit resident representation metadata, wrong shapes/hashes, cache ownership,
cancellation and output/work/weight bounds. Preserve fixed FR-8 thresholds
(absolute <2e-2 and relative <1e-2); no relaxed near-zero denominator. Run
strict Clippy and the affected shared native primitive regressions.

Phi diagnostic receipts e985c969c7074d5496e7cd9da5c60e4a and
1689ed097ed540fc8550e26730baed51 establish that the pinned 8-bit candidate
improves checkpoint shape/policy explanation but still fails exact skill output
and tool syntax. It is not integrated or qualified. The prior 4-bit profile and
all Phi agent gates remain open; these observations do not narrow M3 scope.

### #73 frozen Gemma 3 12B numerical comparison (2026-11-15)

The native core's seven initial CPU/Metal tests passed in
cf3aec18b6b243c98cb1e24ce901a373. The actual frozen benchmark artifact was
re-hashed against all corpus entries in e3a96cbd689d478a904247cf9387557b.
Add `crates/inference-infra/tests/gemma3_quantized.rs`,
`crates/inference-infra/tests/fixtures/gemma3_12b_profile.json`, and
`crates/inference-infra/tests/fixtures/gemma3_model/capture_real_reference.py`.
The profile must retain the existing source and MLX revision/hash identities,
exact tokenizer inputs, pinned oracle versions and captured reference digest.
It stays explicitly unqualified. Store the generated full-vocabulary reference
under the task's Codex artifact directory, not in the checkout.

Run `gemma3_12b_real_quantized_native_reference` against all 262208 logits for
two frozen prefill requests and two subsequent tokens each, with unchanged
FR-8 absolute/relative thresholds. Assert packed 4-bit resident matrices,
actual wrapped tensor layout/output head, measured native weight buffers,
and bit-identical full-versus-last-only projection. No text-only success or
small synthetic fixture substitutes for this real artifact comparison.
If a mismatch identifies an equation/operation-order issue, fix the existing
Gemma core or planned MLX primitive files and rerun affected tiny references.
Daemon tool/context/restart qualification and the additional small Gemma 3
profile remain open after a passing numerical comparison.

### #73 smaller Gemma 3 numerical gate (2026-11-15)

The frozen 12B numerical gate passed in 8a13e74f523542d1b0a1d3923d215c80:
six vocabulary rows, maximum absolute error 3.553e-15 and relative error
2.221e-16, with 7,985,675,264 measured resident weight-buffer bytes. These
results establish the core, not agent support.

Pin the smaller text artifact `mlx-community/gemma-3-text-4b-it-4bit` revision
`4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e` and retain source metadata
`google/gemma-3-4b-it` revision `093f9f388b31de276ce2de164bdc2081324b9767`.
Add `crates/inference-infra/tests/fixtures/gemma3_4b_profile.json` and extend
`crates/inference-infra/tests/gemma3_quantized.rs` with
`gemma3_4b_real_quantized_native_reference`, sharing the unchanged numerical
assertions and allowing only the corresponding explicit artifact/reference
inputs. Reuse the committed offline capture script; bind the output digest
before native execution. Verify 34 layers, hidden 2560, head width 256,
8 query heads/4 KV heads, five full-attention layers and packed-resident
4-bit/group-64 storage. Run strict FR-8 and full/last-only identity checks.
Both profiles remain unqualified until shared engine/daemon, agent fixtures,
context/cache lifecycle, and budget measurements pass.

### #73 Gemma 3 shared engine text integration (2026-11-16)

Both pinned real models pass strict native numerical comparison (receipt
8be84d0db03a4f9abadab044363b5f7f). Wire that implementation into the existing
engine worker and admission paths. This batch does not qualify tool/daemon
agent behavior; keep unsupported tools explicit and all issue gates open.

Exact files: `crates/engine/src/backends/qwen/{architecture.rs,backend.rs,artifact.rs,loading.rs,output.rs,gemma3_loading.rs,gemma3_text.jinja}`;
`crates/engine/src/types/contracts.rs`; `crates/alloyd/src/service.rs`;
`crates/engine/tests/gemma3_engine.rs`;
`crates/inference-infra/src/decoder/gemma3/model.rs`.

Implement pinned BPE Replace/ByteFallback/Fuse decoding with bounded incremental
UTF-8 handling, preserving the existing ByteLevel contract. Validate the exact
decoder sequence and fallback model setting, keep both EOS IDs and BOS framing,
and reject unsupported tool mode and reserved delimiter injection. Match trained
Gemma text rendering (system folded into first user turn, assistant as model).
Advertise mixed sliding/full cache identity explicitly, with no prefix reuse or
persistence claim until implemented. Count actual retained KV and allocator
headroom, reject foreign ownership and insufficient admission, and use native
Gemma model execution on the worker.

Named tests: `gemma3_engine_render_and_decode_contract`,
`gemma3_engine_rejects_invalid_tokenizer_and_tools`,
`gemma3_engine_native_cache_and_cancellation`,
`gemma3_real_4b_engine_streaming`, `gemma3_real_12b_engine_streaming`.
Check genuine tokenizer Unicode/fallback boundaries and special tokens against
the pinned tokenizer decoder, trained text rendering and token IDs, full model
streaming/repeat after cancellation, cache ownership/reset and conservative
memory rejection. Run focused CPU/Metal fixture tests, both explicitly supplied
real artifacts, affected engine regression suites, strict Clippy and daemon
compilation for the new cache capability variant. Real engine text success is
not evidence of the still-required tool/agent/context/memory qualification.

The first integration compile (ada23ac7cd004947b2d270ab027f94e8) found
two existing direct decoder initializers needing the explicit ByteLevel mode.
Include `crates/engine/tests/unit/backends_qwen_output_protocol_token.rs` and
`crates/engine/tests/unit/backends_qwen_output_structured_output.rs` in this batch;
preserve their existing protocol assertions and rerun them.

Real engine receipt efe3b02076f247eaa312462606d6352b exposed a registry
metadata mismatch before native loading: both frozen converted Gemma configs
omit max_position_embeddings. Add `crates/engine/src/registry.rs` to this batch
and use the existing validated Gemma3Config context (32768 for the pinned
MLX dialect) for this family only. Preserve nested/outer conflict rejection
and all other families' explicit context checks. Extend the existing Gemma
contract test with missing/invalid context cases and rerun both real gates.

### #73 Gemma 3 native tool handoff and result framing (2026-11-16)

Shared text integration is committed as 1d6f241. Both actual model engine gates
passed (489b112e5c8f48008b35c0995776e0ef), all 618 engine regressions passed
(d452f1889bc0456ea8cf975a1e651e40), and strict engine/native/daemon Clippy passed
(16100b9fcd0545faadf1ce563448bcd5). No agent qualification is implied.

Offline 4B probes dd4597606ecb48d29dd5767d1dd88890 and
01274ed58eb44de6b1e54676161e38d2 emitted premature DONE and falsely claimed
success after a rejected write. A trusted continuation instruction outside the
untrusted tool-result wrapper corrected rejection interpretation in
3d8e09e0a0a945e9a4d4546e4314a619. Explicit format instructions removed Markdown
fences in fb677b5bcea747f38527a20adfaf32e0; premature text after a complete
write proposal remains. The mandatory 12B protocol probe is separate and pending.

Exact files: `crates/engine/src/backends/qwen/{architecture.rs,backend.rs,output.rs,gemma3_tools.jinja}`;
`crates/engine/src/rendering.rs`; `crates/engine/src/slab_parser/parser.rs`;
`crates/engine/tests/unit/slab_parser_tests.rs`;
`crates/engine/tests/gemma3_engine.rs`;
`crates/engine/tests/fixtures/gemma3_tool_render/{input.json,generate.py,expected.txt}`.

Define one tool call per Gemma assistant turn. Its validated closing XML tag is
a structural sequence boundary through the existing engine completion hook,
not a synthetic EOS or stripped generated answer. Reject a non-whitespace
suffix within the same decoded token and mixed explanatory text/call turns;
keep proposals private until successful finalization. Other families retain
their existing parser behavior. Exercise all UTF-8 chunk splits, malformed
calls, suffix/prefix rejection, storage limits and atomic proposal publication.

Serialize tool schemas and system instructions into the initial user turn,
group adjacent same-role history, use trained model/user role framing, and
append the tested trusted continuation after each tool-result group. Preserve
argument values, result error flags, order and all history. The independent
fixture generator transforms structured history through the pinned publisher
chat template, rather than reading the product template. Reserve tool wrapper
syntax in untrusted data. Bind projected growth to the actual tokenizer digest:
space-to-metaspace replacement expands bytes by at most three; added turn
boundaries isolate changing result groups. Charge the exact skeleton plus
complete changing groups and added bytes without subtracting estimated tokens.

Named tests: `gemma3_terminal_tool_handoff_is_atomic`,
`gemma3_engine_tool_template_and_admitted_decoder`,
`gemma3_engine_tool_projection_bounds`,
`gemma3_real_4b_engine_tools`, `gemma3_real_12b_engine_tools`.
Require native read/write proposals with exact arguments, no premature success,
successful-result DONE, and accurate rejected-result explanation without retry.
Run full affected parser regressions, both real native profiles, and strict
Clippy. Daemon policy effects, session/context/restart and tier measurements
remain subsequent mandatory gates; a Python probe never substitutes for them.

The 12B offline probe ada35cf57b1c4e53b840350d4fb720aa produces standalone
XML fenced calls, correct exact arguments and accurate rejection explanation.
Support two explicit Gemma tool grammars: bare `<tool_call>...</tool_call>` and
exact ` ```xml\n<tool_call>...</tool_call>\n``` ` framing (without the spaces
shown here for readability). For fenced calls require the closing fence before
structural completion. This is a declared protocol alternative, not extraction
of a code block from explanatory text. Test fragmented fences, missing closing
fences, incomplete JSON, prose prefixes and same-token non-whitespace suffixes;
all must fail atomically unless the entire declared grammar is complete. Keep
the prompt's bare-format preference and measure actual native behavior of both
forms. No extra source files beyond the preceding exact batch are needed.

The independent prompt comparison (37ec90488a8f46999185db3d678df931)
exposed tool-version metadata and float spelling differences. Keep the measured
protocol's version-free schema projection and compact Python-compatible JSON
number spelling using an explicit template filter. Other families' JSON and
schema fingerprints retain their behavior. The unchanged independent fixture
and admitted decoder gate pass in 6edf2ca2b72942039f2dc3f722dc26e7.

Real native run 79f994ccd3854f829d134cad01d1e986 passed eight gates but
failed 4B rejection handling: it retried the write. The prior successful probe
used a name-first call envelope, while generic canonical JSON moved arguments
first. Pinned upstream replay ec4d010b74d34956b0c19e9f53e77ffd reproduces
that retry (599 input tokens). Changing only the envelope to name-first while
keeping sorted arguments returns the correct rejection explanation in
0fabb835bf20438d985def87c6e09d04 (598 tokens). Declare the name-first
envelope in the Gemma template and independent generator; preserve the strict
no-retry and exact-argument assertions. Print prompt hashes in real gates to
bind subsequent observations to exact inputs. Both model sizes must be rerun.

### #73 Gemma 3 daemon qualification wiring (2026-11-21)

After the native tool batch passes, wire both frozen profiles through the real
daemon qualification path. Exact files: `crates/alloyd/Cargo.toml`, `Cargo.lock`,
`crates/alloyd/src/boot.rs`, `crates/alloyd/tests/llama_real_model.rs`, and
`crates/alloyd/tests/compaction_real_model.rs`.

Use the native Gemma factory and trained text template. Add a direct
inference-infra dependency to reuse the validated Gemma3Config context instead
of inventing a daemon default. Permit the actual 33 MB tokenizer within a 40 MiB
limit, preserve pinned artifact verification without external overrides, and
use bounded 32-token prefill chunks. Both profiles remain refused by production
boot until the complete agent/report gates pass.

Named boot tests: `gemma_profiles_ignore_external_pin_overrides`,
`unqualified_gemma_cannot_boot_production_or_override_context`, and
`gemma_context_uses_validated_native_metadata`. Run `llama3_tool_stop_context_cache`
for each actual Gemma artifact: exactly one authorized COBALT file write, no
premature answer, exact DONE, rejected forbidden write with zero execution and
an accurate explanation, plus truthful mixed-cache capabilities. Extend the
existing lifecycle gates to require explicit rebuild and zero SSD restoration
for Gemma's currently non-persistent mixed cache; preserve exact output, full
event-history, cancellation, restart and compaction assertions.

For the existing long-context gate use the validated 32768-token Gemma ceiling.
The 32k document must still produce at least 32000 actual prompt tokens and fit
with its declared output/margin. If the supplied document exceeds that budget,
measure and retain the rejection, then create a separately specified document
fixture that fits the actual trained context; do not silently truncate input or
raise the model limit. Run both model sizes and retain actual 8 GiB admission
and larger-tier native memory measurements. Compile/check affected daemon
targets and strict Clippy after implementation. These gates do not imply
Gemma 4, Phi, SmolLM, Mistral/Ministral or full issue #53 acceptance.

Daemon boot checks pass 9/9 and strict Clippy passes in
1f499ef20bbe499585a16b6d7fda06f7; engine regressions are 620 passed,
0 failed, 13 ignored (actual Gemma gates run separately). Actual 4B daemon
tool qualification passes in 59f38c5a358646d9855302c9350e0b7f. That same
debug command then fails the 12B per-turn deadline after the authorized write.
The unchanged 12B test passes with production release optimization in
fe06c691ffc4485bb500d8a1434de4c0: exact COBALT publication, exact DONE,
no answer before the actual result, rejected forbidden write with zero
execution and no retry, accurate rejection explanation and mixed-cache
capabilities. The absolute 120-second per-user-turn deadline is unchanged;
the whole two-turn test takes 192.78 seconds including boot. Preserve the
debug timeout as a failed observation. Use the production build for subsequent
real lifecycle/latency gates and retain the build mode with the evidence.
Both profiles remain unqualified pending all remaining required gates.

### #73 Gemma grouped text history (2026-11-21)

Actual 4B release lifecycle run bcf3bc93b980428c8f485a51f32cd00d passes
3 tests and fails 3. The model generates a structurally correct checkpoint
with all required facts and validates it, but continuation fails rendering.
Skills and checkpoint summaries are deliberately untrusted `user` blocks in
ContextBuilder; placing them next to the user's request creates consecutive
user messages. Gemma's strict alternating text template rejects them before
generation. Do not change their authority or discard those blocks.

Exact files: `crates/engine/src/backends/qwen/gemma3_text.jinja`,
`crates/engine/tests/gemma3_engine.rs`, and
`crates/engine/tests/fixtures/gemma3_text_render/{input.json,generate.py,expected.txt}`.
Group adjacent same-role user/assistant blocks with newline separators inside
one trained Gemma turn. Preserve the existing initial system prefix, ordering,
ordinary content trimming, EOS/BOS framing and role rejection. Continue to
reject a leading assistant or a system message inside the turn history.
An independent generator must group structured messages and use the pinned
publisher template, without reading the product template. Test checkpoints,
loaded-skill blocks, consecutive assistant blocks and exact Unicode/identifier
preservation with `gemma3_engine_grouped_history_matches_publisher_template`.
Replace only the now-valid consecutive-user rejection case with an actually
unsupported mid-history system case. Run focused engine rendering/decoder
checks, both actual release lifecycle suites, and strict affected Clippy.
Preserve all exact answer, complete-event-history, restart and no-SSD-restore
assertions. Any remaining model-output failure stays open for investigation.

### #73 Gemma bounded 32k fixture (2026-11-21)

The pinned Gemma tokenizer measures the original 32k document's complete
retrieval prompt at 32601 tokens (receipt 3c4293bffa3a43c69fd9d6f8f024d884).
With the validated 512 output reserve and 64 safety margin this exceeds
32768. Preserve the frozen Phase 0 corpus unchanged. Add
`crates/alloyd/tests/fixtures/gemma3_long_context/{generate.py,document.txt,manifest.json}`
and update `crates/alloyd/tests/llama_real_model.rs` only for this bounded batch.
The generator must verify the original document and frozen Gemma tokenizer
hashes, select a complete-line prefix producing 32000..32100 tokens with the
existing retrieval prompt and midpoint needle, and retain source/fixture hashes,
byte length, token count and full context budget in the manifest. Selection
happens only when generating the fixed fixture, never during inference. Default
check mode regenerates in memory and rejects drift; explicit --write produces
the committed fixture and manifest. A named `gemma3_long_context_fixture_integrity`
test checks the source-prefix relationship, hashes and declared budget without
model loading. The real test selects that fixture only for Gemma 3, verifies
actual rendering against the recorded count, and asserts the actual profile
budget fits before generation. Keep minimum 32000 prompt tokens, streaming,
EOS, needle answer, memory admission, cancellation and recovery assertions.
Run the generator check, focused integrity test, strict affected Clippy, and
both frozen model sizes in release mode under the native host lease. Retain
any timeout/output/memory failure as an open gate.

Grouped-history lifecycle result: 4B reaches real generation but remains
3 passed / 3 failed on exact continuation and skill output (receipt
9a6390cb5b2f446bb02eca9416d501ba). 12B passes 5/6 including both skill tests,
canonical history, restart and rebuild, but compaction emits fenced JSON then
hits the unchanged 120-second deadline during retry (receipt
0ab45556a6164ccdab74b5b13ca5bf82). No checkpoint is accepted from that failed
run. Exact-output prefix/suffix offline probes fix skills but retain a newline
on 4B compaction continuation; neither policy has been applied to product code.

Bounded-fixture generator/integrity check df4e3caf08844956812dab67621ac073
passes; fixture SHA-256 is
9c249b067cf7600cdbabda90f5231b2a3ad2058835872026a1f83e32f008c8dd.
The unchanged native rendering counts 32095 tokens, leaving 97 tokens beyond
the required 512 output reserve and 64 margin. Strict Clippy for the changed
daemon integration target passes in c524f5938a1648878b5f5fdb6520da25.
Actual 4B release long-context suite cd267ed412ab4ecf94af988dd717a3d1
passes in 217.77 seconds: simulated 8 GiB admission executes; 8101-token
retrieval takes 23.99 seconds with peak MLX active allocation 4029708288 bytes;
32095-token retrieval takes 130.46 seconds with peak 5994756096 bytes.
The latter answer includes a newline, permitted by this existing retrieval
gate but not the still-failing exact lifecycle gate. Post-run MLX free cache
is 25669961735 bytes; do not mislabel the active-allocation peak as total
resident memory or the larger-host long-context run as physical 8 GiB proof.
Cancellation after streamed output and identical fresh-session recovery pass.
Mandatory 12B execution is pending in ddf2013866e84656a9dbf500496872e6.

### #73 Gemma duplicate KV reservation repair (2026-11-21)

12B native long-context receipt ddf2013866e84656a9dbf500496872e6 passes
8101-token retrieval (74.73 seconds, peak MLX active 11488525476 bytes), but
refuses 32095-token startup: required incremental 25372775548 bytes exceeds
21386565592 remaining from budget 29439329240. Simulated 8 GiB correctly
refuses startup requiring 14077934400 bytes. The queued upstream diagnostic
5bbf95f8404a45249226fd6d56d3d326 did not run after the failed lease.

Source diagnosis: `Gemma3Config::step_payload_plan` deliberately includes
three complete KV payloads for old/new/concatenation overlap. The adapter
separately reserves two payloads in its cache allowance, then reserves all
three again as temporary workspace. This counts five payloads, while the
native checked transition plan requires three. Runtime correctly excludes
reusable weights; this is an adapter KV accounting defect.

Exact files: `crates/engine/src/backends/qwen/architecture.rs` and
`crates/engine/tests/gemma3_engine.rs`. Subtract the same checked two-payload
cache allowance from the Gemma temporary-workspace portion before admission.
Keep the native three-payload transition bound, all attention/activation/output
allowances, allocator padding, headroom and memory ceilings unchanged. Native
execution and numerical/layout representations do not change. Add
`gemma3_engine_transition_memory_is_reserved_once`: independently derive the
three-payload transition and layer/attention/output terms for multiple
capacities, require combined cache/workspace to cover that entire plan exactly
plus existing overhead, and exercise admission at the exact bound and one byte
below it. Run focused engine tests, strict engine/alloyd Clippy, then both actual
release Gemma long-context/memory/cancellation gates. No successful admission
alone qualifies runtime safety; retain actual peak measurement and terminal
results. Compaction and exact-output failures remain separate open gates.

The corrected adapter passes six focused engine checks, including independent
geometry arithmetic around sliding-window/chunk boundaries and exact-budget
admission versus one-byte-short refusal (64ae5bcf3cbb419fa49a121270cb01db).
Strict engine/alloyd all-target Clippy passes (c4433e38abb5441e8934a88f9fb85f79).
Actual 12B release suite 0836a5a6292c4e189f9c294ba310e678 passes in 547.72 s:
8 GiB simulated startup correctly refuses (13146798912 required versus
8589934592 available); 8101-token retrieval passes in 55.82 s with peak MLX
active allocation 11488525476 bytes; 32095-token retrieval passes in 368.61 s
with peak 17823136928 bytes. Cancellation after streamed output and identical
fresh-session recovery pass. Post-long-run allocator free cache is
20431418766 bytes, separate from the active-allocation peak and not RSS.
The code-bearing patch only removes duplicated KV allowance; numerical model,
context, hard ceiling, output reserve and exact-output acceptance are unchanged.

Actual 4B release regression 58fadaa1e3d14088ab3fa5024a56a242 passes in
200.45 s: simulated 8 GiB admission executes; 8101-token retrieval takes
20.85 s with peak 4029708288 bytes; 32095-token retrieval takes 110.81 s
with peak 5994756096 bytes. Cancellation and identical fresh-session recovery
pass. Both native long-context suites now pass the corrected accounting.

The bounded upstream format-policy diagnostic b1f007986aac49029294312cd082f030
completed, but the candidate behavior fails: 4B emits smart quotes in the
checkpoint, fences the verifier JSON, and retains the continuation newline.
Both skill colors are exact. This confirms that the candidate instruction is
not a repair; it is discarded and no format policy is added to product code.
Further prompt-wording variants are not the next resolving diagnostic. Strict
checkpoint syntax/schema and exact continuation/skill outputs remain required.

### #73 Native bounded checkpoint syntax constraints (2026-11-21)

The real upstream model probe with flexible JSON token masks preserves all
checkpoint facts and produces a model-selected true verifier for both Gemma
sizes (70bb91c2cf6a4d048d13ffc713e2ae7d and
c7481faaf80344a4826041bd701271d9). The compact variant fails its content
assertion (215f025559a34045b51a0671e4d7a83d); it is not qualification.
llguidance's mandatory serde_json preserve_order would change workspace JSON
serialization. Instead of adding that dependency or its general grammar
allocator, implement the closed checkpoint schema subset directly as a bounded
incremental JSON recognizer. This is production token selection, not output
repair, Python integration, or a replacement for the model fact verifier.

Exact files: crates/engine/src/structured_output.rs (new),
crates/engine/src/lib.rs, crates/engine/src/types/contracts.rs,
crates/engine/src/runtime.rs, crates/engine/src/backends/qwen/backend.rs,
crates/engine/src/backends/qwen/constraints.rs (new),
crates/engine/tests/gemma3_engine.rs,
crates/harness-core/src/context/checkpoint.rs,
crates/harness-core/src/supervisor/compaction.rs,
crates/harness-core/tests/compaction_runtime.rs. Tests of the recognizer live
in its module. No dependency or template changes are planned.

Support required unique ASCII object keys in any order, string values,
arrays of strings, booleans (both true and false), and unsigned integer
constants. Reject additional/duplicate/missing fields, invalid UTF-8,
escapes/surrogates and trailing non-JSON text. Permit JSON whitespace. Cap
schema at 32 fields/64-byte names, vocabulary at 1048576 entries, token pieces
at 1024 bytes, combined raw vocabulary at 4 MiB, and response at 64 KiB.
Use an allocation-free copyable parser state to test candidate token bytes.
Store vocabulary in one fixed 4 MiB buffer and one 8-byte range per token;
reserve their maximum 12 MiB plus 64 KiB fixed metadata/scratch allowance
before construction. Add this fixed host workspace after adaptive prefill
selection and retain its reservation until constraint destruction. Check
cancellation during vocabulary construction and mask scanning. Unsupported
backends and tool-context/schema combinations must fail explicitly.

GenerationOptions carries an optional validated schema. Select the highest
logit among allowed tokens, preserve stable tie ordering, execute the actual
selected token through the native model, then advance the recognizer. EOS and
successful flush require a complete object; an output cap cannot publish a
partial checkpoint. Normal text generation is unchanged. Compaction requests
the checkpoint schema and a separate boolean verification schema; strict JSON
parsing, fact verification, transaction boundaries, exact continuation/skill
assertions and deadlines remain unchanged.

Validate parser transitions at every byte split, escaped Unicode and malformed
input, field-order/duplicate/extra-field behavior, both verifier outcomes,
bounded vocabulary/output rejection, mask selection and EOS, cancellation,
and partial-output failure. Exercise the actual tiny Gemma engine through
constrained generation, including output exhaustion and recovery. Check harness
requests carry the right schemas and keep rollback tests. Run focused suites
and strict Clippy, then actual release Gemma compaction/lifecycle gates for
both pinned sizes. Passing JSON syntax alone does not qualify fact retention
or exact continuation, and no production profile is promoted by this batch.

Implementation refinement: reserve 128 KiB metadata/scratch rather than 64 KiB
because the pinned tokenizer API may clone one bounded 64 KiB spelling before
rejecting an oversized transformed piece. The 12 MiB catalogue bound is
unchanged. Add crates/harness-core/tests/support/scripted.rs to the exact test
file set solely to record generation options for compaction-schema assertions;
scripted events remain test-only and cannot qualify native generation.

Native integration correction: the daemon permits 512 KiB token spellings, so
comparing that policy ceiling to the constraint scratch cap falsely rejects
Gemma before generation (d2f2ffdc82bd4d27b9723197d18158f2). Add
crates/engine/src/backends/qwen/artifact.rs to the exact file set: retain the
actual maximum spelling length from the existing pinned-vocabulary validation
pass and check that immutable measured bound before catalogue construction.
Do not enlarge the constraint scratch cap or alter daemon decoder limits.
The untrained tiny model chooses newlines; its runtime test therefore proves
bounded exhaustion, cleanup and re-entry. Success belongs to frozen-model
lifecycle evidence. JSON requests use the plain decoder so protocol spellings
inside string values remain data; an explicit decoder regression covers this.

Native constraint validation: receipt b879db8bb6714dad93febce970cee8a2
passes the tiny-model exhaustion/re-entry and one-byte-short admission checks,
13 harness compaction/rollback checks, and strict Clippy. The actual tokenizer
maximum correction is covered with the daemon's 512 KiB policy ceiling; JSON
strings containing tool syntax are preserved as data. Raw schema names are
stored as boxed strings to discard unbounded caller-provided spare capacity.

Receipt 7130ba96783c4c8ab339f54d9a68721c completes successfully: 630 engine
checks pass, 13 opt-in tests are ignored, and strict engine/harness/alloyd
Clippy passes. The frozen Gemma3-12B release lifecycle suite passes all six
checks in 353.14 seconds. Native checkpoint generation emits all required
fields and facts as valid JSON (1355 input/169 output tokens); separate model
verification returns valid:true (1404 input/6 output tokens). Continuation and
reopened follow-up both return exactly COBALT. The fixture records initial
1634 tokens, continued610, resumed640, original5 events and final10 events;
original history remains intact. Canonical token history, actual process
restart/rebuild and both exact skill-change checks pass. The existing deadline
and exact-output assertions are unchanged. No dependency, prompt-template,
output postprocessing or production-profile promotion was introduced.

The 4B release lifecycle rerun is pending in
0fba7626921c42b9a7ec02396d9b6f0e. Its earlier plain-answer conformance failures
remain open until current terminal evidence resolves them. Gemma4, other
unqualified families, formal conformance reports and candidate-bound CI remain
separate issue53 requirements.

4B terminal receipt 0fba7626921c42b9a7ec02396d9b6f0e completes with 3/6
lifecycle checks passing (63.39 seconds). Checkpoint JSON preserves all facts
and model verification returns true. Exact continuation still emits
`COBALT authentication`; both skill checks still fail on `COBALT\n`. Canonical
history, rendering and process restart/rebuild pass. The five bounded JSON
recognizer tests and all13 compaction transaction tests pass on final code.
The family remains unqualified on the small profile. A next resolving step is
a pinned upstream comparison of a higher-precision small Gemma artifact
against these exact prompts, before changing production profiles or kernels.
The mandatory frozen12B artifact stays unchanged.

The precision diagnostic is complete. Exact upstream replays of the unchanged
native token inputs reject 4B8bit (0/3, receipt
9d99da045bfa40bc8730763f31fb7292), 1B4bit (2/3,
1a8cc6d810284ceb92b9e65720144a0d), and 1B8bit (2/3,
39c55afa1dc14288ba8930c615012b24). The last continuation is
`refactor COBALT authentication`, not the required exact name. These are
diagnostic results, not native profile qualification; no profile, prompt,
assertion or output normalization is changed. The small Gemma gate remains open.

### #53 / #99 Ministral native position batch (2026-11-21)

Resolve the missing native position equations first. Public metadata is pinned
to `mistralai/Ministral-3-3B-Instruct-2512` revision
`b35d4dfe56c142746f54dbd64f579faab2744308`, publisher BF16 variant
`b6d637bef2393152b3da2b2fde72eecdee30557e`, and converted
`mlx-community/Ministral-3-3B-Instruct-2512-4bit` revision
`a962dcb09eee4169c890e544c9eb938f1113fdee`. Verified converted config SHA256 is
`cdd4bfa0a702d0241faaa3b80d15d481d7f8abf2659fba0958b1e7f24a28055f`.
Cards declare Apache-2.0 and conversion by mlx-vlm0.3.9; weight/tokenizer
verification and conversion provenance remain separate loading work.

Add `crates/mlx-ffi/src/attention/yarn.rs`, export through `attention.rs`,
and add `crates/inference-infra/src/decoder/ministral3/{mod.rs,config.rs,position.rs}`
with export from `decoder/mod.rs`. Add
`crates/inference-infra/tests/ministral3_position.rs` and
`tests/fixtures/ministral3_position/{config.json,generate.py,reference.json,manifest.json}`.
These contain validated nested Mistral3/Ministral3 text geometry, native
float32 YaRN frequency interpolation, rotary amplitude, and the absolute-position
query scale `1 + beta * log(1 + floor(position / original_context))`.
Preserve the pinned eager MLX arithmetic graph rather than host-libm frequency
approximations. Reject unsupported position variants, conflicting metadata,
nonfinite/overflowing dimensions, invalid shapes and out-of-range offsets.
The selected profile has full attention in every layer and no sliding window;
reject sliding configurations until their actual decoder/cache path exists.

Named checks: `ministral3_pinned_geometry_and_position_contract`,
`ministral3_rejects_unsupported_position_and_geometry`,
`ministral3_yarn_matches_pinned_reference`,
`ministral3_position_boundary_and_chunk_equivalence`, and
`ministral3_position_rejects_invalid_inputs`. Independently capture pinned
mlx-lm0.31.2/MLX0.32.4 eager references before native comparison; compare both
FR8 limits without weakening them, and independently check scalar mathematics.
Exercise CPU and Metal, offsets0,16383,16384,32767 and the final supported
position, both rotary-amplitude branches, query-only scaling and grouped heads.
This batch implements the real position operations, not complete Ministral
support. Packed text-only loading, full decoder integration, templates/tools,
actual artifact logits, context/cache/agent/memory gates and production
qualification remain required follow-on work.

Position evidence: pinned upstream capture d59a2bab1669412eaae4cbd3183436b8
completed on CPU and Metal. Native receipt7cc6d968b4544f95a401ccccf840cdee
passes all five position/geometry tests in20.43seconds: each of20reference
cases has maximum absolute and pointwise-relative error exactly0. All splits
reproduce Q/K tensors exactly across the16384boundary and through the final
supported position. Independent scalar rotation/query-scale checks and invalid
geometry/offset rejection pass. Public-field documentation was added after
this run; strictClippy remains pending. This evidence does not qualify the
complete trained model. The subsequent user-requested Cargo cleanup does not
change product source or these retained terminal results.

### #99 verified Ministral text loader

The pinned 830-entry shard index has 602 language tensors, 218 vision tensors
and 10 projector tensors. Add weights.rs under the existing ministral3 decoder
and export it in mod.rs. Reuse DecoderWeights and LayerWeights for owned
packed data. Extend decoder/loader.rs with a crate-private variant of the
verified tensor-set loader that accepts an explicit excluded-name predicate.
The existing entry point retains exact inventory checking. Ministral permits
only the two known non-text namespaces, streams every byte through the trusted
shard hash, rejects unknown text tensors and enforces the declared affine
representation. Add tests/ministral3_loading.rs with
ministral3_text_checkpoint_loading on the actual pinned file, exercising exact
resident payload and one-byte-under refusal. Exercise skipped-byte integrity
and unknown namespaces on a small artifact. No registry promotion in this batch.

### #99 native Ministral decoder and trained reference

Use the shared dense MLX decoder for bias-free SwiGLU, ordinary RMSNorm,
packed tied embeddings and full transactional KV. Add a Ministral3 variant
to decoder/config.rs carrying validated YaRN and query scale parameters.
Wire config conversion in ministral3/config.rs and reuse the rotation code
in ministral3/position.rs from mlx_model.rs on every full attention layer.
Reject the ordinary flat checkpoint path in loader.rs for this nested family.
Add fixtures/ministral3_position/capture_model.py, fixtures/ministral3_profile.json
and ministral3_quantized.rs under inference-infra/tests. Compare every vocabulary
logit for prefill and two cached decode steps on two tokenizer-bound prompts
against eager mlx-lm on the trained artifact. Freeze the reference digest
before running ministral3_real_quantized_native_reference. Preserve the strict
maximum absolute and pointwise relative thresholds. Engine integration and
agent tests follow native model execution.

Adding a dense decoder variant requires updating the exhaustive cache identity
mapping in engine/src/backends/qwen/{architecture.rs,backend.rs,loading.rs}.
Give Ministral its own full-KV representation, keeping the adapter's declared
family distinct from the decoder's mathematical configuration. Check the
engine crate together with the native model checks.

The real Ministral checkpoint passed native text loading and numerical
comparison. Its verified 2,745,210,934-byte shard produced 183 packed matrices
and 53 norms totaling 2,143,678,464 resident bytes. The one-byte-under budget
was refused. The trained prefill and four subsequent cached decode steps on
two prompts matched all 786,432 reference logits with maximum absolute error
1.78e-15 and maximum pointwise relative error 2.23e-16 or lower. Local
receipts 82e69b378b134433baf7596432a26e6e and 03b9838872fa4fe3b871c537dbcf62be
cover these checks. Strict Clippy for engine, inference-infra and mlx-ffi
passed, as did 61 shared infrastructure tests and five CPU/Metal position
tests (0ac5b4705f974207891a25813272c570 and 29ecc502466945a8a38a255565cfe7f5).
The daemon chat/tool path, long-context, agent and qualification gates remain open.

### #99 daemon chat integration

Extend crates/engine/src/backends/qwen/{architecture.rs,backend.rs,loading.rs}
with Ministral registration, pinned prompt rendering, the verified text loader,
and its own KV identity. Add ministral3_text.jinja for trained system/user/
assistant framing, including the publisher default system message. Reject the
pinned tokenizer's reserved delimiters in untrusted message content. Verify
BOS and EOS token identities and the expected ByteLevel tokenizer. Enable
the frozen profile in crates/alloyd/src/boot.rs for qualification and keep
production gated. Exercise the daemon via crates/alloyd/tests/ministral3_chat.rs
using real text generation and verify exact rendered token IDs. Add tests for
unqualified production rejection and tool requests before native protocol
integration. Tool parsing and policy execution require the trained
[TOOL_CALLS]name[ARGS]JSON representation in a subsequent batch.

The pinned Ministral daemon run passed both tests in 37.57 seconds. Rendered
bytes and every token ID matched the publisher on the 548-token default-system
prompt and 27-token explicit history. One SQLite-backed RPC session streamed
exactly `4` and then `8`. Reserved input delimiters, unqualified production
boot and requests for unsupported tools were rejected. Reference capture
8441ede3badb4e5f9f540e0f9535b522 and native test
1fe9ef51025245758cc99b242bb31ae3 retain the terminal output. Strict Clippy
for engine and alloyd, including all targets, passed
(b79aedea615849fbbeca8ca96681fd2f).

### #99 Ministral tools and long context

Run `llama3_long_context_stream_cancel_memory` against the pinned 3B artifact
with the existing full 8K/32K archive inputs, retrieval assertion, memory
admission and post-cancellation comparison. The reference tokenizer counts fit
the 40,960-token test limit. Record actual native counts and peak MLX usage.

Implement the trained special-token protocol in
`crates/engine/src/slab_parser/{mod.rs,ministral3.rs}`. Use the shared bounded
JSON validator and admitted proposal storage. Cover literal text delimiters,
sequential calls, malformed later calls, limits and missing EOS in the
module's `ministral_protocol_*` tests. In `crates/engine/src/runtime.rs`,
notify the decoder when the selected token is an actual EOS so the parser
can distinguish a complete tool batch from an output limit. Integrate in
`crates/engine/src/backends/qwen/{backend.rs,output.rs,architecture.rs,ministral3_text.jinja}`.
Match the publisher's tool declarations, grouped assistant calls and result
messages, verify the pinned special IDs, and bound projected tokens before
execution in `crates/engine/src/rendering.rs`. Enable tool mode for
qualification in `crates/alloyd/src/boot.rs`. Verify publisher tool prompt
parity in `crates/alloyd/tests/ministral3_chat.rs` and run the actual
`llama3_tool_stop_context_cache` in `crates/alloyd/tests/llama_real_model.rs`
with its existing allow/deny, terminal response and cache gates.

Initialize the new parser field in the two existing decoder test builders,
crates/engine/tests/unit/backends_qwen_output_{protocol_token,structured_output}.rs.

For real skill and checkpoint history the harness emits adjacent user-role
records. In the planned Ministral template, serialize adjacent user records
in one publisher user turn with a newline between records. Test equality of
rendered bytes and IDs with the equivalent combined user input in
ministral3_chat.rs. Keep the publisher's alternating assistant boundaries.
Run the existing compaction_real_model integration suite with the frozen
profile, including actual process restart, cold comparisons and event checks.

Native long-context receipt 19113f5801b940e995edc043dbe8f08a passed on
the text integration from 6dcbe4a. The 8 GiB simulated admission executed.
At 8,895 and 33,925 tokens the daemon streamed the complete archive code
COBALT-7392 and stopped at EOS. Prefill plus generation took 44.09 and 208.97
seconds, with MLX peaks of 6,214,250,496 and 16,885,970,772 bytes. Cancellation
after the first text delta returned a cancellation error. A fresh session
then matched the baseline response exactly.

Ministral tool qualification passed its real allow/deny daemon test in
71.22 seconds (a19670dd824f426e9a5eaaeab0b63819). A sampled control ID
started one fs.write call. The daemon wrote the six bytes COBALT into
marker.txt, then the model returned exactly DONE. On the following turn the
daemon rejected forbidden.txt before execution and the model reported the
rejection. The file was absent. Both final answers reached EOS.

Reference capture 8aa83d48e47a40a9883136d1a65d4342 and native run
70a4dd03f5724676840bd8dfb4bd5475 establish byte and token equality for the
55-token declaration prompt and the 112-token history with two calls and two
results. The native run also passed both ordinary chat cases and production
refusal. Four parser/decoder regressions passed (4d62657d024c4d3bab113e34a0df01e5),
including ordinary text with marker spellings, sampled control IDs, complete
EOS batches, malformed/truncated input and storage bounds. Thirteen shared
tool-rendering and Phi protocol regressions passed
(1b707e4cb804428b8c4ba430dd2e0797). Strict Clippy passed for engine and
alloyd on all targets (d2a5e5835c8a4bd48984afc651926b9e).

### #99 Ministral bounded overflow fixture

The first Ministral lifecycle run refused the first archive turn at the
2,048-token fixture limit. No history existed to compact. Independent pinned
tokenization (e9933eed5a374bf8b5326a6b69c9f3d4) counted 1,606 tokens on
that first turn, 530 of them from the publisher default system content.
With one assistant token the second input is already 2,100 tokens. Apply
the same 2,048-token fixture budget plus those 530 tokens for this profile in
crates/alloyd/tests/compaction_real_model.rs, for both warm and cold boots of
real_model_compaction_preserves_history_and_continues. Assert the first actual
render including the 512 output reserve and 64 margin fits and the second
overflows. Keep the original messages, output requirements, checkpoint fields,
event preservation, cache and cold-replay assertions.

The generated-history fixture in the same file tried nine possible endings,
omitting the pinned Ministral EOS. For real_model_generated_history_has_verified_canonical_bytes
use only the publisher's exact closing bytes `</s>` for this profile. Retain
the byte-to-token identity check and state lookup. The independent two-process
test already restored 628 of 629 input tokens and returned the exact cold answer.

With the bounded fixture the checkpoint verified, the first continuation was
exactly COBALT, and the reopened answer was **COBALT** including Markdown.
Compare reopened and fresh execution before asserting the unchanged literal
answer in real_model_compaction_preserves_history_and_continues. Keep the
original prompt, the exact-answer assertion and the restored-prefix checks.
This distinguishes a cache divergence from a repeated model-format failure.

The 4-bit lifecycle suite passed four of five cases
(dfc8964d2c65459daf81505338dbb6ce). Retention verified 548 executed tokens
from a 544-token prompt. The process-restart case restored 628 of 629 input
tokens with the exact cold answer. Replacing the loaded skill changed COBALT
to AMBER over RPC, and a warm run restored the unchanged 534-token prefix and
matched the cold AMBER result. The complete native generation regression suite
passed 101 tests with one opt-in test ignored
(82ccfc7c7ea84f8d8dbd8ebddcb093d4).

The separate compaction replay on ddbd766 (54adf9270b7d44e298d2f0aa072bd536)
failed the unchanged resumed-output assertion. The 2,163-token history plus
576 reserved tokens exceeded the 2,578-token limit. After compaction the
1,256-token request returned exactly COBALT, matching the cold transition.
All checkpoint fields and previous events survived. The reopened 1,279-token
request and cold execution both emitted the literal Markdown string
`**COBALT**`. Independent pinned eager MLX reproduced the same five output
IDs on all 1,279 original input IDs (1c21ac297e8f4d9dbb85a0339d2ce850).
A diagnostic request for plain text returned COBALT; it does not replace the
failing fixture. Keep the 4-bit agent profile unqualified.

The optimized actual 4-bit renderer counted 33,875 tokens. All five warm
samples matched token-for-token and took 16.84 to 17.28 ms, within the
150 ms threshold (bf7c9c4cab244a2d91c513febcec43b4).

### #99 Ministral 3B affine8 qualification

Keep the failed affine4 result. The same publisher's 8-bit conversion at
mlx-community/Ministral-3-3B-Instruct-2512-8bit revision
79ebb6ee5067d20a89765e54f43449cadccb66ff passed an independent diagnostic on
the unchanged 1,279-token input and five other captured prompts
(94faf1ecd5bd472882e6ccb1fd009b6e). Inputs verified against remote object IDs,
including the 4,468,021,687-byte shard (313a3182e7444249aa32c5393ccdea20).
This diagnostic has no native qualification status.

Add crates/inference-infra/tests/fixtures/ministral3_8bit_profile.json, freeze
the existing capture_model.py output digest before running native inference,
and parameterize the trained numerical and resident-byte checks in
crates/inference-infra/tests/{ministral3_quantized.rs,ministral3_loading.rs}.
Run ministral3_real_affine8_native_reference and ministral3_affine8_checkpoint_loading
against the verified shard. Update crates/engine/src/backends/qwen/backend.rs
to register the implemented affine8 representation and crates/alloyd/src/boot.rs
to pin both profiles in qualification mode. Test both production refusals in
crates/alloyd/tests/ministral3_chat.rs. Include the 8-bit profile in the
existing compaction and policy fixtures in
crates/alloyd/tests/{llama_real_model.rs,compaction_real_model.rs}. Measure
the unchanged native tool, lifecycle, long-context, memory and latency gates.

The independent eager 8-bit reference (061985dd04c0416fa44f3ea5377500a7)
was frozen at f6ce3ca. Native loader and numerical tests passed
(1d241f22087e408a968d2101cce063c7). The 183 packed matrices and 53 norms
occupy 3,858,100,224 resident bytes, with refusal one byte below that amount.
The two trained prompts matched six full vocabulary vectors (786,432 logits)
at prefill and cached decode. Strict Clippy passed on all three affected
crates (7e221749aac0475380ad6ac3e3f8de84). Both Ministral profiles remain
behind the production qualification check (6ca1672f03ee4d4eafa453a3e732c290).

The affine8 native lifecycle suite at 24ea5de passed overflow, canonical
history, process restart and optimized rendering. The overflow case preserved
the five original events, reopened ten events and matched cold replay with the
literal answer COBALT. The skill RPC returned Cobalt and then AMBER. The
first response also failed the cache-invalidation test's exact assertion
(0d15cdcc617d4d6296439f49b8e3ab60). Independent eager execution of the
same 641 and 662 token inputs returned the identical strings with EOS
(1f03af326eaa487188d234c31715deda). Keep both failed assertions.

### #99 native memory measurements for both Ministral representations

Add crates/inference-infra/tests/ministral3_memory.rs with the single ignored
Metal test ministral3_native_memory_tiers. Factor the pinned text-model loader
from crates/inference-infra/tests/ministral3_quantized.rs into
crates/inference-infra/tests/support/ministral3.rs. Retain both existing
trained-reference tests. Use the same verified weights, stream and decoder
as the numerical gate. In an isolated test process measure unique evaluated
weight buffers, actual provisioned KV buffers, and the MLX active-byte peak
above the synchronized weights-and-cache baseline during prefill and decode.
That increment includes temporary KV update overlap. Check the committed KV
size and synchronized active memory again after execution. Retain the native
allocator peak and its configured guideline separately from the byte budget.
Exercise a bounded 512-token prefill at the simulated 8 GiB budget and a
32,736-token prefill at the actual 36 GiB host budget with the daemon's chunk
sizes. Decode one more token using each live cache. The independent daemon
tests continue to cover admission, retrieval quality and policy. Record
actual positions and measured bytes for each tier and quantization.

### #96 Gemma 4 E4B attention geometry and native position operations

The publisher is google/gemma-4-E4B-it at
4ecc2ba63bcac23ba4a767fc87025b1450f84c9c (Apache-2.0). The MLX conversion
is mlx-community/gemma-4-e4b-it-4bit at
475b9088d29754a3379866cf5aeb6b41acd313c2. The verified configuration has
42 dense text layers. The first 24 own KV. Each of the final 18 reuses
the last preceding owner of its own attention type. Sliding layers use
256-wide heads, a 512-position causal window and full rotation. Global
layers use 512-wide heads and rotate one quarter of their channels as
split-half pairs, with exponents divided by the entire head width. Both
execute attention with scale 1.0. Full model qualification also needs the
per-layer embeddings, trained normalization, loader, decoder and agent path.

Implement the layer-to-owner mapping and text attention kernels in
crates/inference-infra/src/decoder/gemma4.rs and expose the module from
crates/inference-infra/src/decoder/mod.rs. Read the verified publisher
configuration in crates/inference-infra/tests/fixtures/gemma4_e4b_config.json.
Use crates/inference-infra/tests/gemma4_attention.rs for
gemma4_shared_kv_sliding_full, gemma4_partial_rotation and
gemma4_attention_masks_and_scale. Exercise malformed layouts, absolute
offsets, unrotated channel pairs, sliding-window eviction and causality
using real MLX arrays against independent scalar expectations. The
scope of this batch is the attention component of #96.

Gemma 4 E4B passed all three geometry, rotation and attention tests
(824829e069e143199336aa00b25962f7). The rotation and attention outputs
were checked on both MLX CPU and Metal. Strict Clippy passed
(c6743e2a54574aa2b41a4907f2af5ada). Gemma 4 decoder, trained weights
execution, daemon registration and agent qualification are still pending.

The Ministral affine8 daemon passed both long-context and policy tests
(ed121cdd67c54195ab28d0f4c4d9ddf2). It retrieved COBALT-7392 at 8,895
and 33,925 input tokens, recovered after streamed cancellation, executed
the allowed write and respected denial. Its 33,925-token MLX peak was
18,601,867,588 bytes. Both trained numerical checks and both native memory
tests passed (16fba106cada4b419891e9b5b660a3b6). Measurements below are
bytes. Transient is the peak above the larger of pre-inference and settled
active memory, including KV update overlap.

| Quantization | Budget | Executed positions | Weights | Committed KV | Transient | Peak active |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4-bit | simulated 8 GiB | 513 | 2,143,678,464 | 109,903,872 | 170,198,804 | 2,423,781,140 |
| 4-bit | real 36 GiB | 32,737 | 2,143,678,464 | 6,973,358,080 | 7,000,621,056 | 16,117,657,600 |
| 8-bit | simulated 8 GiB | 513 | 3,858,100,224 | 109,903,872 | 169,148,932 | 4,137,153,028 |
| 8-bit | real 36 GiB | 32,737 | 3,858,100,224 | 6,973,358,080 | 7,000,621,056 | 17,832,079,360 |

### #96 Gemma 4 E4B verified text loading

The pinned shard header and index identify 345 packed text matrices and 320
floating tensors (including the 42 learned layer scalars). The expected
charge for packed words and float32 scales, biases and other float tensors
is 4,666,273,960 bytes. Shared layers contain query and output projections
and no independent K, V or key norm. The per-layer token embedding's logical
width is 42 × 256. The converted checkpoint also carries audio, vision and
their input projectors.

Move the parsed attention geometry in crates/inference-infra/src/decoder/gemma4.rs
into an allocation-independent config used by the attention kernels and the
new crates/inference-infra/src/decoder/gemma4/weights.rs. Use the existing
verified selected-tensor loader. Derive exact shapes from dense text geometry,
validate tied embeddings, per-layer inputs, the MLP and affine4/group64
recipe. Reject missing and unexpected text tensors and permit only the four
named non-text prefixes. Freeze verified input digests in
crates/inference-infra/tests/fixtures/gemma4_e4b_profile.json. Add
crates/inference-infra/tests/gemma4_loading.rs with gemma4_e4b_tensor_inventory
and gemma4_e4b_verified_quantized_loading. Verify the complete immutable
shard and the one-byte-under denial, then measure the returned text payload.
The trainable weight buffers must stay packed. Run gemma4_attention again.

Verified the entire 5,146,800,534-byte E4B shard against SHA-256
932b8271fc3fe65adcc78b96c10c6268bbfb13e8f67d1358727c0d6ee97e1eff
(f43e65856f2d4d7cae41d8507c6b36e5). The native trained loader passed the
one-byte-under refusal and loaded 345 packed matrices plus 320 float tensors
using exactly 4,666,273,960 payload bytes. All three attention tests passed
again (24a9b14fd574419281522d300ef010c2), as did the independent inventory
test (8ac3740ba9a84ca6b4d39a40e7a884ca) and strict Clippy
(838ef380058a46a1affa0dd67644d541).

### #96 Gemma 4 trained decoder and fixed numerical reference

Add crates/inference-infra/tests/fixtures/gemma4_capture.py, and bind its
two fixed text prompts, decode inputs and eager full-vocabulary outputs in
crates/inference-infra/tests/fixtures/gemma4_e4b_profile.json before running
the Rust decoder. Use the verified conversion, the installed pinned MLX-LM
Gemma 4 text equations, packed matrices and F32 floating payloads. Record
the exact package versions and upstream model source digests. The converted
artifact omits unused K/V weights in shared layers. Only those inactive
upstream attributes may be omitted; verify all executed parameters match
the source inventory. No production Python dependency is introduced.

Implement the native text decoder in
crates/inference-infra/src/decoder/gemma4/model.rs and update
crates/inference-infra/src/decoder/gemma4.rs with validated norm/softcap
parameters. Use actual packed per-layer token embeddings, projection and
gating, learned layer scalars, normalized Q/K/V with scale-one attention,
owner-only transactional KV, final norm, tied head and tanh logit softcap.
Extend crates/mlx-ffi/src/ops.rs only for missing native operations.
Add crates/inference-infra/tests/gemma4_quantized.rs with
gemma4_native_trained_reference. Compare two real prefills and two cached
decode steps per prompt to all 262,144 reference logits per row under both
existing FR-8 limits. Check cache ownership, offsets and refusal without
state change. Daemon integration and complete agent conformance follow this
trained decoder gate.

Gemma 4's independently captured six eager rows were frozen at 3c82e62
before native execution (1ff0187aabe74e688e97ab5b0b541885). The trained
Metal decoder passed for all 1,572,864 logits, including four cached
decodes. Across the six rows, maximum absolute error was 3.553e-15 and
maximum pointwise relative error was 2.221e-16. The evaluated native weight
buffers occupied 4,667,650,216 bytes. The same test exercised cancellation
during the layer loop, retry, workspace and capacity refusals, and rejection
of another model instance's cache (c1098ab5f28c432787779944173b6137).
Strict Clippy passed for both affected crates across all targets
(a0470d3941aa4152a522c084823a6282). These checks cover trained decoder
execution. Daemon chat, model-specific tools, long-context validation and
profile qualification are outstanding.

### #96 Gemma 4 native daemon text sessions

After the frozen trained-decoder comparison, connect the E4B text model to
the shared worker, memory admission and session API. Use the verified
profile's exact text tensor layout for upload and resident payloads, count
only independent KV owners, and include transactional overlap and the
worker's prefill workspace. Keep cache representation and instance ownership
distinct, with reset/rebuild across eviction. Implement in
`crates/inference-infra/src/decoder/gemma4.rs`,
`crates/inference-infra/src/decoder/gemma4/{model,weights}.rs`,
`crates/engine/src/backends/qwen/{architecture,backend,loading,gemma4_loading,output}.rs`,
`crates/engine/src/backends/qwen/gemma4_text.jinja` and
`crates/alloyd/src/boot.rs`. Use publisher text turns, pinned control IDs and
EOS, including the standalone system turn. Limit boot to explicit
qualification until the real agent gates pass. Reject unsupported tool mode
until the native Gemma 4 tool protocol is implemented.

Run gemma4_e4b_memory_plan in crates/inference-infra/tests/gemma4_loading.rs,
gemma4_native_trained_reference in crates/inference-infra/tests/gemma4_quantized.rs,
and gemma4_daemon_text_session in crates/alloyd/tests/gemma4_chat.rs. Compare
rendered inputs to the pinned publisher tokenizer/template and exercise a
real two-turn Metal session. These checks supply text pipeline evidence for
#96. Tool, long-context, profile and M3 exit checks remain required.

All six focused release tests passed on the Mac with the pinned E4B model
(dd8e6cf6d2414922a268a5874149891f). The shared renderer reproduced all three
publisher prompt byte and token hashes (22, 43 and 30 tokens). One
DaemonService session answered exactly 4 and then 8, streaming both replies.
The loader, memory plan, production refusal and six trained reference rows
passed in the same run. Strict Clippy passed across all targets in engine,
alloyd and inference-infra (5ca3f74c52fd4c95b3f3316f658cb674).

### #96 Gemma 4 trained tool output, declarations and linked results

Implement the publisher's call and string delimiters as token IDs in
`crates/engine/src/slab_parser/{mod,gemma4}.rs`. Convert the bounded call
name and typed argument object into the existing admitted parser slab,
validate JSON structure and duplicate keys, and publish the batch only at
a valid end. Preserve stream text, multiple calls, Unicode, nested objects
and arrays. Reject malformed, truncated, unsolicited-control and oversized
output. Run gemma4_tool_fragments_and_typed_arguments and
gemma4_tool_rejects_invalid_batches. Wire the parser, byte fallback and EOS
in `crates/engine/src/backends/qwen/{architecture,backend,output}.rs`.
Use `crates/engine/src/backends/qwen/gemma4_tools.jinja` for declarations,
typed call history and linked results; validate structural spellings in raw
values in `crates/engine/src/types/tool_context.rs`. Implement conservative
result-token admission in `crates/engine/src/rendering.rs`. Match the frozen
publisher tokenization for declarations and linked results in
gemma4_publisher_tool_prompt_parity in `crates/alloyd/tests/gemma4_chat.rs`.
Run the existing real-model tool-policy and compaction fixtures using the
Gemma 4 profile, retaining observed failures. Only a passing real task
counts toward production support.

The decoder's per-token parser slot will dispatch either Ministral or Gemma 4.
Update the existing constructors in
`crates/engine/tests/unit/backends_qwen_output_protocol_token.rs` and
`crates/engine/tests/unit/backends_qwen_output_structured_output.rs` with the
same disabled protocol state, and rerun their output regression tests.

For the two real harnesses, add the exact pinned E4B ID and its declared
family/mixed cache representation in `crates/alloyd/tests/llama_real_model.rs`
(llama3_tool_stop_context_cache) and its reset/rebuild semantics in
`crates/alloyd/tests/compaction_real_model.rs`
(real_model_compaction_preserves_history_and_continues). Apply the same
tool, state and event assertions as the other mixed-cache profiles.

The real E4B model stopped immediately after a rejected tool result on the
716-token request, in both Alloy and the independent greedy reference
(1e7308d68b764568ad5aa91e07cfcd56, 51632c9fb0ea41f4ae3de278937e37ea).
Changing the result content to a typed object also produced immediate EOS.
Supplying the agent's general result-reporting and no-retry instruction in
the trusted system turn produced an accurate rejection explanation upstream
(a6cfa0d3742e4a9289fb954a4e86f6e4). Apply that contract to every Gemma 4
turn with advertised tools in the planned gemma4_tools.jinja, preserve the
caller's system text and freeze publisher render parity with the same system
content. Keep the actual write, deny and model-reply requirements intact.

The native daemon created `marker.txt` with the requested `COBALT` content
in one authorized call, returned the linked result and answered `DONE`.
For a denied write it recorded `ArgumentsNotAllowed`, executed zero calls
and generated “The operation was rejected by local policy.”
(7d6f4c3238ae4720a7c07c6044a258d3). All three daemon chat tests passed,
including publisher prompt parity and production-profile refusal
(7974af10d5d04606baaa0818c3340a1d).

The real model retrieved `COBALT-7392` at 8,101 and 32,601 prompt tokens
and recovered after streamed cancellation. Simulated 8 GiB daemon startup
refused admission (48caa12df9264432ba50e638a4852241). The compaction test
reopened the session, retained the original five events and matched the cold
reply after cache rebuild (3e28a06448d84846a18033e76cc92b37). The engine
library release suite passed 399 tests with two ignored
(c53665579c2346ccb8dc20b5f768e2e3). Strict Clippy passed across all targets
of engine, alloyd and inference-infra (d1391c4c05cd4f69bb446fd011638f37).

### #96 Gemma 4 measured memory and sliding-window replay

Share the verified E4B checkpoint loader between the frozen trained
reference in `crates/inference-infra/tests/gemma4_quantized.rs` and the
new isolated allocator test in `crates/inference-infra/tests/gemma4_memory.rs`
using `crates/inference-infra/tests/support/gemma4.rs`. Run
`gemma4_native_memory_tiers` on Metal at a simulated 8 GiB allocator limit
with 512 prompt tokens plus one decode, and on the physical 36 GiB Mac
with 32,736 plus one. Record native weight buffers, empty and committed
KV, active endpoints, load and generation peaks and transient bytes above
both endpoints. Compare actual retained lengths and backing sizes with
the pinned full/sliding owner layout. In the smaller case compare every
final logit and the greedy token with a single cold prefill of the same
513 tokens. Rerun `gemma4_native_trained_reference` with the original
independent reference. These measurements qualify the native decoder. The
shared daemon's budget admission and conformance checks remain separate.

On the pinned E4B decoder the empty cache measured zero bytes. Both tiers
loaded 4,667,650,216 bytes of native weight buffers. At 513 tokens, the
simulated 8 GiB run measured 59,473,920 bytes of committed KV, a
4,822,185,136-byte allocator peak and 95,059,976 transient bytes above
both active endpoints. Cold and cached execution chose token 900 with a
maximum absolute logit difference of 0.001145. At 32,737 tokens on the
36 GiB host, committed KV measured 1,114,783,744 bytes, allocator peak
6,957,151,580 and transient bytes 1,174,716,596. The shortest owner
retained 512 tokens on both. All six trained reference rows passed in the
same terminal run (08889a302e8e4bbf83f3085cb62f2b20).

### #96 Gemma 4 short requests under the 8 GiB daemon budget

The native decoder fits the short prompt but the daemon's simultaneous
load, KV, output and workspace reservations exceeded the 8 GiB budget by
35,708,904 bytes. In `crates/engine/src/backends/qwen/backend.rs`, select a smaller actual
prefill chunk for the E4B profile when its resolved admission budget is
at most 8 GiB. Bound vocabulary-wide workspace by running at most four
prompt rows per native step. Keep the normal 32-row setting on larger
budgets. Preserve the existing context and output limits, packed weights,
full load reservation and all runtime checks. In
`crates/alloyd/tests/gemma4_chat.rs` run `gemma4_small_budget_generation`:
require a real short reply at 8 GiB, a memory refusal on the same input
at 4 GiB, and a longer request whose 8 GiB estimate exceeds the budget.
Rerun `llama3_long_context_stream_cancel_memory` with the E4B profile.
Apply the cap when the backend receives the admission controller, after
the daemon has verified metadata and resolved the host ceiling. This also
exercises the same budget in direct engine users.

The full E4B lifecycle run passed compaction, warm rendering, independent
process restart and both skill changes. The canonical-history fixture
failed (810173e83f6d44c6b3e8e02b3615b2ab). Its list of suffix candidates
omits Gemma 4's publisher turn terminator. In
`crates/alloyd/tests/compaction_real_model.rs`, supply the pinned model's
three actual EOS spellings to `real_model_generated_history_has_verified_canonical_bytes`.
Retain failure reasons per candidate. Require successful publication and
lookup through the existing exact decode and re-encode checks, then rerun
the whole fixture set.

The 22-token question returned exactly 4 with an 8 GiB daemon admission
budget. At 4 GiB the same question required 8,566,156,032 bytes and was
refused before generation. A longer prompt at 8 GiB required 10,139,852,364
bytes and was also refused. The two-turn, production-guard, publisher tool
parity and small-budget tests passed (767b25e185094c61aeadc86722f02a77).
With the E4B terminators the canonical-history verifier published and
looked up a state for all 22 executed tokens, including the 18-token input
(be9fc8fe88474933a27c8103bb7f15f6).

### #72 export and evaluate actual numerical observations

The model-specific tests currently compare real logits but discard the
vectors needed by the shared report. Add explicit opt-in export to
`crates/inference-infra/tests/{gemma4_quantized,qwen35_quantized}.rs` via
`crates/inference-infra/tests/support/numerical_capture.rs`. Capture the
actual cumulative token IDs and native final-position logits for every
pinned prompt/decode row. Write only to a supplied external run directory,
refuse overwrite, and retain numerical failures. No reference values may
supply an observation. Extend the existing `gemma4_native_trained_reference`,
`ornith_9b_quantized_native_reference` and upstream
`qwen35_0_8b_quantized_native_reference` runs.

Add `crates/engine/examples/m3_numerical_report.rs` to read those rows
against independently pinned reference bytes, verify exact profile,
per-step input IDs, vocabulary and row inventory, and run the existing
`NumericalMetrics::compare` on actual vs expected values. Produce
machine-readable references, numerical observations and per-row metrics
compatible with the existing qualification evaluator. Test rejected
missing/swapped/non-finite rows and duplicate output, then execute with
real native captures. Fixture execution, candidate identity, production
admission and terminal CI are additional #72/#53 requirements.

Apply the workspace formatter to the recently changed small-tier
assignment in `crates/engine/src/backends/qwen/backend.rs` and rerun
`cargo fmt --all -- --check`.

Raw E4B, Ornith and upstream Qwen3.5 capture passed on 5765db5. Independent
readback found a one-bit change in some exported decimal values even though
all metrics passed (cdabc9744df0422cb4d7337a113b02f6). Preserve the exact
IEEE values across the entire report path. Enable the pinned serde_json
float_roundtrip feature in `crates/engine/Cargo.toml` and for the numerical
test targets in `crates/inference-infra/Cargo.toml`. Add
`m3_numerical_report_preserves_native_float_bits` in
`crates/engine/examples/m3_numerical_report.rs` using the observed real
logit. Reissue the three raw captures and report sections on the corrected
candidate, then compare every persisted reference and observation to the
independent frozen and raw arrays.

### #72 reproducible native numerical runner

Add `scripts/ci/m3_native_numerical.py` as the entry point for the verified
E4B, Ornith and upstream Qwen3.5 numerical suites. Require the candidate
commit, its clean checkout, profile, artifact directory, reference and a
new external output directory. Verify every manifest file size and hash.
Build the exact release test and report binaries through Cargo's JSON
artifact output, record their hashes and run the named ignored test under
the existing Mac CI lease. Reject zero-test or skipped runs. Verify the
captured observations with the Rust report evaluator and save the
candidate, pins, raw logs, output hashes and terminal status even on
failure. This issues numerical evidence for #72; behavioral fixture
results are required to construct a complete QualificationReport.
In `scripts/tests/test_m3_native_numerical.py`, test rejection of wrong
artifact bytes and missing/filtered execution. Exercise the normal runner
against all three pinned models on the Mac.

### #72 native behavioral observations for the shared pipeline

Add scripts/ci/m3_native_behavior.py and scripts/tests/test_m3_native_behavior.py.
Reuse the numerical runner's pinned profile, build, exact libtest execution,
log digest and candidate checks from scripts/ci/m3_native_numerical.py.
Extract common pinned-run setup and terminal receipt writing there. On
Gemma 4 E4B, Ornith 1.5 9B and upstream Qwen3.5 run the named native
llama3_tool_stop_context_cache, llama3_long_context_stream_cancel_memory,
real_model_compaction_preserves_history_and_continues,
real_model_ssd_prefix_survives_process_restart,
real_model_generated_history_has_verified_canonical_bytes,
real_skill_change_invalidates_stale_cache_and_matches_cold and
real_rpc_skill_load_changes_next_generation fixtures. Include E4B's
gemma4_daemon_text_session, gemma4_publisher_tool_prompt_parity,
gemma4_small_budget_generation and gemma4_native_memory_tiers. Use the
existing release Cargo build and Mac host lease. Bind each observation
to the exact test binary, profile, pinned inputs, source and terminal
logs. Scrub inherited test overrides, retain failed observations and run
the remaining independent fixtures. Exercise failed/skipped execution,
evidence digests and inventory through the Python unit tests. Run native
suites on the pinned Mac artifacts. This supplies traceable per-test
observations for the #53 compaction/family/Ornith exits; boot admission
still requires a complete trusted spec for the actual validated profile.

### #95 hybrid memory measurements for both pinned profiles

In crates/inference-infra/tests/support/qwen35.rs share the existing verified
checkpoint loader from crates/inference-infra/tests/qwen35_quantized.rs.
Add crates/inference-infra/tests/qwen35_memory.rs and its
qwen35_native_memory_tiers test. On Ornith and the independent small Qwen3.5,
measure real weights, full KV and recurrent state at 512 prompt tokens with
an 8 GiB MLX ceiling and 32736 prompt tokens on the physical 36 GiB Mac.
Include a real decode, finite logits, context length, allocator peaks and
transient memory; assert physical RAM and budget. Bound new-cache and forward
allocations with checked model plans. Daemon admission/refusal and retrieval
remain separately measured by llama3_long_context_stream_cancel_memory.
Register this mandatory fixture for both profiles in scripts/ci/m3_native_behavior.py
and extend scripts/tests/test_m3_native_behavior.py inventory checks.
Re-run both named native reference tests after sharing the loader. No new
dependencies. These are #95 and #72 memory observations for the #53 exit.

Expose the separately measured full-KV and recurrent backing bytes through
Qwen35Cache::component_buffer_bytes in crates/inference-infra/src/decoder/qwen35/model.rs.
Keep buffer_bytes as the checked sum used by existing admission callers.
Allocator padding can change after an update, so subtraction of initial KV
from final aggregate bytes would not measure recurrent storage accurately.

### #53 reproducible CI execution of the implemented native suites

Add scripts/ci/m3_native_matrix.py to run both native evidence producers
for every currently registered pinned numerical suite (Gemma 4, Ornith and
small Qwen3.5). Take paths from one external JSON inventory, bind every run
to the requested exact Git commit, continue independent executions on
individual input or fixture failures and exit nonzero on any failure.
Write an external index with the status and SHA-256 of each terminal receipt.
A matrix pass covers its listed suites only. No production admission occurs.
Use this same entry point in .github/workflows/m3-native-observations.yaml
on explicit dispatch, with an exact checkout and the existing Mac lease.
Retain logs and index on failure. The runner operator supplies the path
inventory outside the checkout. Add scripts/tests/test_m3_native_matrix.py
for complete inventory, mismatched/missing receipts and cross-profile
failure isolation. Add the native Python regression tests to the existing
Linux CLI job in .github/workflows/ci.yaml. Verify ci_partition_test.rb,
runner tests, YAML structure and real three-profile local execution.
No new dependency. Hosted terminal CI and the other M3 families still
require their own exits.

### #72 derive qualification identity from production profile construction

Factor native backend and ModelProfile preparation out of
crates/alloyd/src/boot.rs and reuse it for identity inspection. Use the
actual verified tokenizer, consumed artifact manifest, selected text/tool
templates, registered family version and profile settings. Implement the
identity constructor in crates/engine/src/backends/qwen/backend.rs.
Bind the tool parser to its reviewed source revision, family, advertised
template, configured parsing mode and allocation limits. Include the
backend execution device. No Python reconstruction of the Rust identities.
Add crates/alloyd/examples/m3_profile_identity.rs for the trusted runner
to inspect the selected native Metal tool-enabled configuration without
changing production eligibility. Extend scripts/ci/m3_native_behavior.py
to build and execute it at the shared long-context fixture's configured
40960 tokens, preserving the exact context/output/margin, output file hash
and environment digest (Cargo.lock, PINS.md, rustc, host OS/architecture).
This is the observed configuration, not a claim that the entire context
has passed retrieval. In crates/engine/tests/qwen35_engine.rs, add
qwen35_qualification_identity_tracks_native_settings for both renderers
and context / parser mode changes. Verify exact metadata on the three
pinned artifacts; reports and production promotion still require all
fixture and provenance checks. No new dependencies.

### #53 exact 40960-token profile boundary on the three pinned models

Add native_validated_context_boundary to crates/alloyd/tests/llama_real_model.rs
and require it in scripts/ci/m3_native_behavior.py and its inventory test in
scripts/tests/test_m3_native_behavior.py. At the same 40960-token configuration
inspected by #72, preserve 512 output and 64 safety tokens. Exercise both
40384-token retrieval and a fresh 40385-token input that must be refused
before generation. Use both frozen corpus documents, an archive record in
the middle, and a short fixed padding suffix. The publisher tokenizer/Jinja
render independently yielded the exact two input counts for all three
profiles (b5cf1fda3bd84072a33c1014dc8f9b7d). Pin the complete render and
token hashes at the boundary. Verify the actual generation request used
those tokens without truncation and returned the vault code. Preserve both
existing 8k/32k retrieval cases and the separate overflow-compaction test.
No change to supported product limits or model weights. Run all three
named native boundary tests. This supplies the advertised-context measurement
for a 40960-token #72 configuration, conditional on real retrieval.

### #72 assemble and evaluate complete candidate-bound native reports

Extend crates/engine/examples/m3_numerical_report.rs with a qualification
mode that constructs QualificationSpec and QualificationReport, using
the same frozen reference/hash and actual raw rows as numerical issuance.
Compute the suite digest and run QualificationSpec::evaluate. Persist the
full report, spec and verdict even when a valid observation records a
fixture failure. The usual numerical-only entry point remains supported.
In scripts/ci/m3_native_behavior.py define the reviewed coverage mapping
for each named fixture in the registered suites, with multiple independent
assertions in one real test sharing its native transcript. For hybrid
tool format include qwen35_real_engine_tool_proposals using the pinned
artifact and the correct Ornith/upstream selection. Compare the inventory
to the mandatory kinds in the engine evaluator. Add
scripts/ci/m3_native_qualification.py to verify terminal fixture identities,
executables, commands, stream digests and exact named pass/failure, and use
the actual profile identity and pinned numerical inputs from the same
fresh matrix directory. Call it from scripts/ci/m3_native_matrix.py on
both success and failure. Record per-profile report/verdict hashes and
keep a failed matrix exit for unsupported profiles. Extend
scripts/tests/test_m3_native_{matrix,behavior}.py and add
scripts/tests/test_m3_native_qualification.py for rejected stale/tampered
evidence, missing execution, named failures, and retained multi-kind
observations. Exercise the engine example, the real matrix, and the property
and report gates. Production admission separately needs a trusted deployed
report and a matching runtime identity. No new dependencies.

### #53 extend candidate-bound numerical coverage to Gemma 3, Phi and Ministral

Use the five existing pinned trained checkpoints and their fixed eager references
in crates/inference-infra/tests/{gemma3_quantized,phi_quantized,ministral3_quantized}.rs.
Capture the actual final vocabulary vector for each prefill and each cached
decode with the existing numerical_capture writer, preserving the original
full-vector comparisons and cache assertions. Register both Gemma 3 sizes,
Phi-4-mini 4-bit and both Ministral precisions in
scripts/ci/m3_native_numerical.py and crates/engine/examples/m3_numerical_report.rs.
The matrix should include all eight registered profiles. In
scripts/ci/m3_native_behavior.py explicitly dispatch the existing family
inventories so other architectures cannot inherit a hybrid fixture or receive
a verdict from the wrong tests. In scripts/tests/test_m3_native_behavior.py
exercise that distinction. No new dependencies.

Named Metal tests are gemma3_4b_real_quantized_native_reference,
gemma3_12b_real_quantized_native_reference, phi4mini_real_quantized_native_reference,
ministral3_real_quantized_native_reference and ministral3_real_affine8_native_reference.
Run each through the numerical CLI on a clean revision under the host lease.
Check the six captured rows per profile against their full pinned vocabularies,
both FR-8 thresholds and the terminal receipts. Run the matrix and report unit
tests, rustfmt and strict Clippy. This batch supplies direct numerical evidence
for the local-family exit. Each profile still needs its own behavioral inventory,
native context and memory measurements, complete report and production admission.
