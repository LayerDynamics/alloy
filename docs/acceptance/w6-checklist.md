## W6 implementation and acceptance checklist

Scope: issue #33, Phase 2 W6 only. W2/W3/W4/W5 are reusable dependencies;
W6 wires them together. An individual patch is not W6 completion. Keep #33 open
until P2-12/P2-13/P2-14 and the required supporting checks pass.

### 1. Turn identity and generation completion
- [ ] Derive bounded generation identities from durable turn identity and round, not token count; avoid collisions across sessions, equal-sized prompts, rounds, and restart.
- [ ] Require an observed successful engine terminal before consuming proposals; EOF, failure, cancellation, malformed later output, or callback failure cannot authorize a tool batch.
- [ ] Keep engine-round completion distinct from user-turn completion. Emit exactly one terminal user-turn outcome, including errors and cancellation.
- [ ] Enforce explicit round/call/output/deadline limits; no unbounded generate/tool recursion or automatic effect replay.

### 2. Canonical context and proposals
- [ ] Render exact registered schemas and ordered linked assistant-call/tool-result blocks through the existing Rust renderer/tokenizer before every generation.
- [ ] Preserve schema-sensitive cache identity and existing context/admission limits; no estimated token counts or silently inflated limits.
- [ ] Handle both owned slab-batch and supported legacy proposal APIs without leaking reservations, dropping calls, or granting model-issued identity/authority.
- [ ] Resolve names in the canonical registry, validate complete object arguments, and allocate daemon-scoped call identities bound to session/turn/generation.
- [ ] Reject malformed/unknown/schema-invalid proposals with canonical protocol results and zero execution; no raw-text fallback.

### 3. W3/W4/W5 lifecycle wiring
- [ ] Persist tool.requested before effects and retain exact immutable registration/arguments/policy/approval binding.
- [ ] Use existing W3 bound approval, claim, final-start, completion and artifact APIs; do not create bypass execution authority or a competing registry.
- [ ] After validation/approval, invoke ordered before hooks before tool.started. Veto/failure prevents execution and later before hooks.
- [ ] Execute the unchanged registered implementation once through W4 receipt ownership with cancellation, effective deadline and output ceilings.
- [ ] Invoke after hooks once for actual attempted execution, including error/timeout/cancellation; no after hook for a before veto.
- [ ] Persist original result/artifact provenance and ordered hook receipts/annotations; after failures are warnings, never result replacement or retry.
- [ ] Feed canonical linked results back to the next generation and finish on final assistant response or an explicit bounded stop.
- [ ] Restart preserves durable history and unknown started outcomes without replaying effects. Pause withdraws pending approval and cancels the exact owned operation.

### 4. Concurrency and responsive RPC
- [ ] Never retain SQLite/store, session-map or native execution locks across approval waits, hook IPC or external execution.
- [ ] Add session.approve { session_id, call_id, allow } on a separate responsive control connection; require an explicit boolean and exact binding.
- [ ] Reject missing/malformed/stale/duplicate/wrong-session/changed-binding decisions with zero unauthorized effects.
- [ ] Preserve one active turn per session; pause/cancel targets only its current generation/approval/hook/tool and does not cancel another session.
- [ ] Keep daemon status/metrics and unrelated sessions responsive during waits/execution; preserve existing wire/error compatibility or document a deliberate contract amendment.
- [ ] Wire actual daemon construction/configuration to the registry, scoped builtins, policy, artifacts, plugins and supervisor; tests must not bypass the deployed selection path.

### 5. CLI
- [ ] Add ts/cli/src/approval.ts and integrate chat/rpc-client/render/args for approval and tool events; policy decisions remain Rust-owned.
- [ ] Show tool identity, effect/arguments summary and result/error/artifact preview; do not display untrusted tool output as system instructions.
- [ ] Send explicit allow/deny from real user input. EOF/noninteractive input must not invent approval; handle denial, cancellation, disconnect and terminal events coherently.
- [ ] Verify the CLI can send approval/pause while session.send is waiting, without blocking its own control path.

### 6. Acceptance and evidence
- [ ] P2-12 canonical_tool_loop_persists_results: actual supervisor orchestration with scripted inference requests a real registered builtin, consumes its result, returns final text, preserves ordering/call IDs, enforces limits and does not replay after reopen.
- [ ] P2-13 approval_rpc_remains_responsive: real daemon service/socket path accepts approval/denial/pause from another connection while send waits, keeps unrelated RPC responsive, and emits one terminal outcome.
- [ ] P2-14 cli_renders_approval_and_tool_result: approval/tool rendering and real decision/control flow, including deny/cancel/noninteractive cases. Node 26.13.0 Linux + Mac.
- [ ] Discover and execute exact named gates; verify effects/markers/process observations, not zero-test or mock-only success claims.
- [ ] Run Linux and Mac native regressions, package all-target/doc checks, strict rustdoc, formatting, Clippy, and CLI tests without weakening previous phase gates.
- [ ] Record the final head/tested-merge SHA relationship, clean source provenance, commands, exact test results and retained logs/artifact URLs. Reconcile the checklist before closing #33.
- [ ] Keep trained-model full Phase 2 acceptance/exit review in W7; do not claim it from scripted inference tests.

### Patch ledger
- Patch 1: durable generation identity and fail-closed text-supervisor stream boundary. It does not enable tools, add approvals/RPC/CLI, or satisfy P2-12/13/14 by itself.
- Subsequent patches must implement the context/loop/lifecycle, responsive RPC/configuration, CLI, and integrated acceptance items above; their completion must be demonstrated, not inferred from patch 1.
