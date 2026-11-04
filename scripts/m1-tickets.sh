#!/usr/bin/env bash
# Provision M1 using fully paginated lookups and a repository-wide lock.
# All cooperating invocations must use this script and the same lock ref.
# An interrupted run can leave the lock behind: verify no provisioner is
# active before manually deleting refs/heads/automation/m1-provision-lock.
set -euo pipefail
REPO="LayerDynamics/alloy"
MS_TITLE="M1 — Vertical slice"
LOCK_REF="refs/heads/automation/m1-provision-lock"

command -v gh >/dev/null || { echo 'gh is required' >&2; exit 1; }
command -v jq >/dev/null || { echo 'jq is required' >&2; exit 1; }
WORK="$(mktemp -d)"
LOCK_HELD=0
cleanup() {
  local status=$?
  trap - EXIT
  if [ "$LOCK_HELD" = 1 ]; then
    if ! gh api --method DELETE "repos/$REPO/git/refs/${LOCK_REF#refs/}" >/dev/null; then
      echo "Could not release $LOCK_REF; inspect it before the next run." >&2
      status=1
    fi
  fi
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

DEFAULT_BRANCH="$(gh api "repos/$REPO" --jq .default_branch)"
LOCK_SHA="$(gh api "repos/$REPO/commits/$DEFAULT_BRANCH" --jq .sha)"
# Ref creation is atomic. Never delete an existing lock to acquire it.
# Abort on every API failure, including an existing ref or ambiguous result.
if ! gh api --method POST "repos/$REPO/git/refs" \
  -f ref="$LOCK_REF" -f sha="$LOCK_SHA" >/dev/null; then
  echo "Lock acquisition failed. Another run may hold $LOCK_REF; no provisioning performed." >&2
  exit 1
fi
LOCK_HELD=1

# Materialize complete responses first: API errors must not look like absence.
gh api --paginate --slurp "repos/$REPO/milestones?state=all&per_page=100" \
  > "$WORK/milestones.json"
MS_NUMBER="$(jq -r --arg title "$MS_TITLE" '
  [.[][] | select(.title == $title)]
  | if length > 1 then error("duplicate milestone titles require manual cleanup")
    elif length == 1 then .[0].number else empty end
' "$WORK/milestones.json")"
if [ -z "$MS_NUMBER" ]; then
  MS_NUMBER="$(gh api --method POST "repos/$REPO/milestones" \
    -f title="$MS_TITLE" -f due_on="2026-12-03T08:00:00Z" --jq .number)"
  echo "milestone created: $MS_TITLE (#$MS_NUMBER)"
else
  echo "milestone exists: $MS_TITLE (#$MS_NUMBER)"
fi

for label in "epic:5319e7" "phase-1:0e8a16" "engine:c5def5" "harness:fef2c0" \
             "daemon:fbca04" "cli:1d76db" "ci:d4c5f9" "closeout:b60205" "pin:e99695"; do
  gh label create "${label%%:*}" --repo "$REPO" --color "${label##*:}" --force
done

ticket() {
  local labels="$1" title="$2" body="$3" count
  gh api --paginate --slurp \
    "repos/$REPO/issues?state=all&milestone=$MS_NUMBER&per_page=100" \
    > "$WORK/issues.json"
  count="$(jq --arg title "$title" \
    '[.[][] | select(.pull_request == null and .title == $title)] | length' \
    "$WORK/issues.json")"
  if [ "$count" -gt 1 ]; then
    echo "Duplicate existing tickets require manual cleanup: $title" >&2
    return 1
  fi
  if [ "$count" -eq 1 ]; then
    echo "skip (exists): $title"
    return 0
  fi
  # REST accepts the resolved milestone number, avoiding title ambiguity.
  jq -n --arg title "$title" --arg body "$body" \
    --arg labels "$labels" --argjson milestone "$MS_NUMBER" \
    '{title:$title, body:$body, labels:($labels | split(",")), milestone:$milestone}' \
    > "$WORK/new-issue.json"
  gh api --method POST "repos/$REPO/issues" --input "$WORK/new-issue.json" --jq .html_url
}

ticket "epic,phase-1" "Epic: Phase 1 vertical slice" 'Tracks tests 1-15 and the five Phase 1 exit criteria from docs/specs/PHASE-1-EXECUTION-PLAN.md. Closes only on the exit-review ticket. Sub-tickets: inference-infra, Qwen3 family crate, engine core, harness-core, alloyd daemon, ts/cli, CI lanes, exit review, minijinja pin.'

ticket "phase-1,engine" "inference-infra crate: KV cache, RoPE, attention, sampling" 'W1, 3 days. Blocks the family crate and engine core.

Build crates/inference-infra over mlx-ffi. Acceptance: test infra_cpu_ops green on Linux CPU-backend MLX; no Metal-specific code; capability flags stubbed for the family crate.'

ticket "phase-1,engine" "Qwen3 family crate with tiny fixture" 'W2, 5 days. Blocks engine core and the end-to-end test. Depends on inference-infra.

Build crates/families/qwen3: safetensors loader, forward/decode loops, B.5 registration (architectures, impl_version, capabilities). Acceptance: qwen3_forward_cpu matches committed reference logits within FR-8 tolerances (2e-2 abs / 1e-2 rel) on the synthetic fixture; chat_template_render proves minijinja template-hash identity.

If forward-pass work exceeds one week, comment here and record it as the R3 signal — do not silently absorb it.'

ticket "phase-1,engine" "Engine core: B.1 trait, registry, scheduler, admission, cache stub" 'W3, 4 days. Blocks harness-core and the daemon. Depends on inference-infra and the Qwen3 crate.

Build crates/engine: async render_and_count / estimate_memory / generate / cancel / capabilities per Appendix B.1, engine-API version constant, single-session scheduler, admission estimate, prefix-cache stub reporting unimplemented. Acceptance: stream_events_order (mocked), admission_estimate within 20 percent on the synthetic model, cache_stub_interface, engine_traits_documented (rustdoc, zero missing-doc lints).'

ticket "phase-1,harness" "harness-core: context builder, session store, telemetry client" 'W4, 4 days. Blocks the daemon. Depends on engine core.

Build crates/harness-core: render_and_count over HF tokenizers plus minijinja (FR-9), B.4 event log schema v1 in SQLite, FR-35 telemetry client (in-memory counts, TLS flush, offline drop). Acceptance: token_count_exact, events_appendonly, session_resume_restart, telemetry_schema_conforms against docs/events/telemetry-v1.json, telemetry_offline_drop.'

ticket "phase-1,daemon" "alloyd daemon: UDS JSON-RPC server, B.6 subset" 'W5, 3 days. Blocks ts/cli and the end-to-end test. Depends on engine core and harness-core.

Build crates/alloyd: UDS listener, apiversion negotiation, the chat/session/streaming subset of Appendix B.6, error kinds. Acceptance: rpc_surface_v1 green; methods match B.6 verbatim — if B.6 naming differs from sessions.*, follow B.6 and note the delta here.'

ticket "phase-1,cli" "ts/cli: alloy chat" 'W6, 2 days. Depends on the daemon ticket.

Build ts/cli on the Node LTS pin from DR-0001: alloy chat --profile qwen3-4b-4bit, internal JSON-RPC client over UDS, streaming render. Thin client per FR-30a — no policy, no harness logic. Acceptance: alloy chat streams from the daemon; the client contains no inference or policy code.'

ticket "phase-1,ci" "CI: unit and mocked lanes for the new crates" 'W7, 2 days. Lands incrementally as the crates land.

Extend .github/workflows/ci.yaml: Linux lane runs tests 1-12 (CPU MLX plus mocked engine), Mac lane runs 13-14 on the self-hosted runner on merge-to-main. Acceptance: a PR touching any new crate runs its lane; the corpus and Phase 0 gates are untouched.'

ticket "phase-1,closeout" "Exit review and Phase 2 plan" 'W8, 1 day. Closes the epic.

Run the five exit criteria: alloy chat end-to-end on the pinned corpus artifact (test 13), session survives restart with the real model (test 14), engine traits documented and versioned (test 15), vendored-build reproducibility verified twice (two green CI runs post-corpus-freeze), telemetry emits per schema (tests 9-10). Then draft PHASE-2-EXECUTION-PLAN.md in the same file-set/named-tests format.'

ticket "phase-1,pin" "Pin minijinja and amend DR-0001" 'No blockers; lands with harness-core.

When harness-core takes the minijinja dependency, pin the exact version in the root Cargo.toml and record it in a DR-0001 amendment with the locked version. Closes the last W2 pin item from Phase 0.'

echo "Done. Milestone: $MS_TITLE"
