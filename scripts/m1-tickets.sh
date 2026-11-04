#!/usr/bin/env bash
set -euo pipefail
REPO="LayerDynamics/alloy"

# labels (create-if-missing; failure means it already exists, which is fine)
for label in "epic:5319e7" "phase-1:0e8a16" "engine:c5def5" "harness:fef2c0" \
             "daemon:fbca04" "cli:1d76db" "ci:d4c5f9" "closeout:b60205" "pin:e99695"; do
  gh api "repos/$REPO/labels" -f name="${label%%:*}" -f color="${label##*:}" >/dev/null 2>&1 || true
done

# milestone (create if missing; gh matches by name, not number)
MS_TITLE="M1 — Vertical slice"
gh api "repos/$REPO/milestones" \
  -f title="$MS_TITLE" \
  -F due_on="2026-12-03T08:00:00Z" >/dev/null 2>&1 || true

new() { gh issue create --repo "$REPO" --milestone "$MS_TITLE" "$@"; }

new --label "epic,phase-1" \
  --title "Epic: Phase 1 vertical slice" \
  --body "Tracks tests 1-15 and the five Phase 1 exit criteria from docs/specs/PHASE-1-EXECUTION-PLAN.md. Closes only on the exit-review ticket. Sub-tickets: inference-infra, Qwen3 family crate, engine core, harness-core, alloyd daemon, ts/cli, CI lanes, exit review, minijinja pin."

new --label "phase-1,engine" \
  --title "inference-infra crate: KV cache, RoPE, attention, sampling" \
  --body "W1, 3 days. Blocks the family crate and engine core.

Build crates/inference-infra over mlx-ffi. Acceptance: test infra_cpu_ops green on Linux CPU-backend MLX; no Metal-specific code; capability flags stubbed for the family crate."

new --label "phase-1,engine" \
  --title "Qwen3 family crate with tiny fixture" \
  --body "W2, 5 days. Blocks engine core and the end-to-end test. Depends on inference-infra.

Build crates/families/qwen3: safetensors loader, forward/decode loops, B.5 registration (architectures, impl_version, capabilities). Acceptance: qwen3_forward_cpu matches committed reference logits within FR-8 tolerances (2e-2 abs / 1e-2 rel) on the synthetic fixture; chat_template_render proves minijinja template-hash identity.

If forward-pass work exceeds one week, comment here and record it as the R3 signal — do not silently absorb it."

new --label "phase-1,engine" \
  --title "Engine core: B.1 trait, registry, scheduler, admission, cache stub" \
  --body "W3, 4 days. Blocks harness-core and the daemon. Depends on inference-infra and the Qwen3 crate.

Build crates/engine: async render_and_count / estimate_memory / generate / cancel / capabilities per Appendix B.1, engine-API version constant, single-session scheduler, admission estimate, prefix-cache stub reporting unimplemented. Acceptance: stream_events_order (mocked), admission_estimate within 20 percent on the synthetic model, cache_stub_interface, engine_traits_documented (rustdoc, zero missing-doc lints)."

new --label "phase-1,harness" \
  --title "harness-core: context builder, session store, telemetry client" \
  --body "W4, 4 days. Blocks the daemon. Depends on engine core.

Build crates/harness-core: render_and_count over HF tokenizers plus minijinja (FR-9), B.4 event log schema v1 in SQLite, FR-35 telemetry client (in-memory counts, TLS flush, offline drop). Acceptance: token_count_exact, events_append_only, session_resume_restart, telemetry_schema_conforms against docs/events/telemetry-v1.json, telemetry_offline_drop."

new --label "phase-1,daemon" \
  --title "alloyd daemon: UDS JSON-RPC server, B.6 subset" \
  --body "W5, 3 days. Blocks ts/cli and the end-to-end test. Depends on engine core and harness-core.

Build crates/alloyd: UDS listener, apiversion negotiation, the chat/session/streaming subset of Appendix B.6, error kinds. Acceptance: rpc_surface_v1 green; methods match B.6 verbatim — if B.6 naming differs from sessions.*, follow B.6 and note the delta here."

new --label "phase-1,cli" \
  --title "ts/cli: alloy chat" \
  --body "W6, 2 days. Depends on the daemon ticket.

Build ts/cli on the Node LTS pin from DR-0001: alloy chat --profile qwen3-4b-4bit, internal JSON-RPC client over UDS, streaming render. Thin client per FR-30a — no policy, no harness logic. Acceptance: alloy chat streams from the daemon; the client contains no inference or policy code."

new --label "phase-1,ci" \
  --title "CI: unit and mocked lanes for the new crates" \
  --body "W7, 2 days. Lands incrementally as the crates land.

Extend .github/workflows/ci.yaml: Linux lane runs tests 1-12 (CPU MLX plus mocked engine), Mac lane runs 13-14 on the self-hosted runner on merge-to-main. Acceptance: a PR touching any new crate runs its lane; the corpus and Phase 0 gates are untouched."

new --label "phase-1,closeout" \
  --title "Exit review and Phase 2 plan" \
  --body "W8, 1 day. Closes the epic.

Run the five exit criteria: alloy chat end-to-end on the pinned corpus artifact (test 13), session survives restart with the real model (test 14), engine traits documented and versioned (test 15), vendored-build reproducibility verified twice (two green CI runs post-corpus-freeze), telemetry emits per schema (tests 9-10). Then draft PHASE-2-EXECUTION-PLAN.md in the same file-set/named-tests format."

new --label "phase-1,pin" \
  --title "Pin minijinja and amend DR-0001" \
  --body "No blockers; lands with harness-core.

When harness-core takes the minijinja dependency, pin the exact version in the root Cargo.toml and record it in a DR-0001 amendment with the locked version. Closes the last W2 pin item from Phase 0."

echo "Done. Milestone: $MS_TITLE"