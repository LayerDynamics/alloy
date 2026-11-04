# Phase 1 Execution Plan — Vertical Slice

Spec: SPEC-1 v1.9, 4.1 Phase 1
Duration: weeks 2–4 of M1, 2026-11-10 → 2026-12-03 (M1 target)
Owner: solo
Fallback: none for this phase. The Q2 binding decision is closed (DR-0001);
mlx-ffi, the vendored build, and CI all exist from Phase 0.

## Goal

One model family, one session, streamed chat end to end: `alloy chat` (TS
client) → daemon local JSON-RPC API → engine → quantized Qwen3 generating
tokens in-process from an mlx-community artifact.

## Scope (from the spec)

- Shared inference-infra crate: KV cache, RoPE, attention, sampling.
- Qwen3 family crate over the infra crate, registered by
  `config.json` architectures (B.5 trait).
- HF tokenizers (official Rust crate) for exact counting; minijinja for
  chat-template rendering (FR-9 — renderer pinned, template-hash identity
  checks).
- Agent supervisor skeleton: the turn loop reduced to
  user-message → render-and-count → generate → stream; no tools, no policy,
  no skills, no MCP (Phase 2+).
- SQLite session store: B.4 event log schema v1, append-only.
- Daemon `alloyd`: UDS JSON-RPC server exposing the B.6 subset Phase 1
  needs (`sessions.*`, chat streaming), with apiversion negotiation.
- Prefix-cache **stub only** — the interface, not the implementation; the
  real single-session prefix cache lands in Phase 3 with its tests.
- Telemetry client (FR-35): in-memory anonymous-count collection, HTTPS/TLS
  flush, offline drop, failure-isolated, plus the initial event schema.

## Explicitly out of scope (do not build)

Tool registry, policy engine, MCP client, skills, plugins, compaction,
prefix-cache reuse logic, additional families, oMLX adapter, TUI, the TS SDK
as a published package (the CLI's internal client library is the seed for
it), the conversion CLI.

## File set (exact, everything Phase 1 creates or extends)

crates/mlx-ffi/ existing, unchanged except pin bumps
crates/inference-infra/ KV cache, RoPE, attention, sampling
src/{kv_cache,rope,attention,sampling}.rs
tests/infra_cpu.rs named tests, CPU-backend MLX
crates/families/qwen3/ Qwen3 family crate
src/{loader,forward,decode,registry}.rs
tests/{forward_cpu.rs,template.rs}
fixtures/tiny-qwen3/ synthetic small config + weights, CI-sized
crates/engine/ engine core
src/{api.rs,registry.rs,scheduler.rs,cache_stub.rs,admission.rs}
tests/{api_mock.rs,stream_events.rs}
crates/harness-core/ supervisor skeleton, context, session store
src/{supervisor.rs,context.rs,store.rs,telemetry.rs}
tests/{events.rs,resume.rs,telemetry.rs}
crates/alloyd/ daemon binary
src/{main.rs,rpc.rs}
tests/rpc_surface.rs
ts/cli/ TypeScript CLI, Node LTS pin from DR-0001
src/{main.ts,chat.ts,rpc-client.ts}
package.json
docs/specs/PHASE-1-EXECUTION-PLAN.md this document
docs/events/telemetry-v1.json initial event schema
.github/workflows/ci.yaml extend: unit + mocked lanes

text

## Named tests

| # | Test (as named in code) | Proves | Lane |
| --- | --- | --- | --- |
| 1 | `infra_cpu_ops` | KV cache, RoPE, attention, sampling correct on CPU-backend MLX | Linux CI |
| 2 | `qwen3_forward_cpu` | synthetic tiny-Qwen3 forward matches captured reference logits within FR-8 tolerances (2e-2 abs / 1e-2 rel) | Linux CI |
| 3 | `chat_template_render` | minijinja renders the Qwen3 chat template; template-hash identity is stable across renders | Linux CI |
| 4 | `token_count_exact` | render_and_count returns HF-tokenizer counts for fixed fixtures | Linux CI |
| 5 | `events_append_only` | B.4 schema v1: gapless per-session seq, monotonic ids, events survive reopen | Linux CI |
| 6 | `session_resume_restart` | a session appends, daemon restarts, session resumes with history intact | Linux CI |
| 7 | `rpc_surface_v1` | B.6 subset: apiversion negotiation, method dispatch, error kinds on bad requests | Linux CI |
| 8 | `stream_events_order` | mocked engine: text_delta stream ordering, generation_completed, cancel | Linux CI |
| 9 | `telemetry_schema_conforms` | emitted events validate against docs/events/telemetry-v1.json; anonymity invariants hold (no prompts, paths, or model identifiers) | Linux CI |
| 10 | `telemetry_offline_drop` | unreachable endpoint drops counts, never queues, never affects sessions | Linux CI |
| 11 | `admission_estimate` | estimate_memory within 20% of measured for the synthetic model | Linux CI |
| 12 | `cache_stub_interface` | prefix-cache stub exists behind the B.1 interface and reports unimplemented | Linux CI |
| 13 | `qwen3_end_to_end_stream` | `alloy chat` streams tokens from the pinned quantized Qwen3 corpus artifact, in-process | Mac |
| 14 | `session_restart_real_model` | exit criterion on real hardware: session survives restart with the real model | Mac |
| 15 | `engine_traits_documented` | rustdoc builds with zero missing-doc lints on engine public traits; engine API version constant present | Linux CI |

Test 11 is a computed here from the NFR "measured memory within 20% of
estimates"; the 20% band is calibrated in Phase 5 where admission lands
fully — Phase 1 only proves the estimate exists and is honest for one model.

## Exit criteria mapping (spec → deliverable)

| Spec exit criterion | Closed by |
| --- | --- |
| `alloy chat` streams tokens from quantized Qwen3, in-process | Test 13 |
| A session survives a process restart | Tests 6, 14 |
| Engine API traits documented and versioned | Test 15 + B.1 version constant |
| Vendored-build reproducibility verified twice | Phase 0 test 11 in CI — two green CI runs post-corpus-freeze satisfy this |
| Telemetry emits anonymous counts per the initial event schema | Tests 9, 10 + `docs/events/telemetry-v1.json` |

## Work items

| ID | Item | Budget |
| --- | --- | --- |
| W1 | `inference-infra` crate, test 1 | 3 d |
| W2 | Qwen3 family crate + tiny fixture + tests 2–3 | 5 d |
| W3 | Engine core: B.1 trait, registry, scheduler (single session), admission estimate, cache stub | 4 d |
| W4 | `harness-core`: context builder (render_and_count; inherit the existing minijinja workspace pin), session store, telemetry client + schema | 4 d |
| W5 | `alloyd`: UDS JSON-RPC server, B.6 subset, tests 5–10 | 3 d |
| W6 | `ts/cli`: `alloy chat` + internal RPC client | 2 d |
| W7 | CI extension: unit/mocked lanes for the new crates; Mac integration tests 13–14 | 2 d |
| W8 | Exit-criteria review against this plan; Phase 2 plan drafted | 1 d |

Total 24 working days against 16 calendar weeks-days remaining before
2026-12-03; the slack is deliberate — family-crate forward-pass work (W2) is
the item most likely to absorb it, per risk R3/R4 in the spec.

## Notes and boundaries

- The tiny-Qwen3 fixture keeps test 2 CI-sized; the real pinned artifact
  (corpus profile `qwen3-4b-4bit`) runs only on the Mac lane. Reference
  logit vectors are captured from eager-mode execution and committed with
  the fixture, per FR-8.
- Completed dependency prerequisite: minijinja is exact-pinned to
  `=2.12.0` in the root `Cargo.toml`, with version `2.12.0` recorded in
  `Cargo.lock`. Qwen3 was the first consumer; DR-0001 records the decision
  under issue #10. This is not pending harness-core work.
  When harness-core adopts the renderer, inherit it with
  `minijinja = { workspace = true }`; do not introduce another version
  or repeat the pinning task.
- The privacy policy is published before the first tagged 0.x pre-release
  that emits — it is not required for Phase 1 merge, but the schema
  (`docs/events/telemetry-v1.json`) is, and it must contain the anonymity
  invariants as machine-checkable constraints.
- No performance work. The corpus is frozen; Phase 5 does the optimizing,
  and premature tuning now would be unmeasured.
