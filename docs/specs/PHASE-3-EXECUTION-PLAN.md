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
