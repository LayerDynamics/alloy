# W3 implementation and acceptance checklist

Status: implementation delivered; verification review and final acceptance pending.
Reviewed implementation baseline: `9ee31cdc8c5f3352cf492a1de41684b596331752`.

A checked implementation item means the implementation or named test is delivered,
not that every branch has completed source review or that its test passed on every
required platform. Execution and acceptance are tracked separately below.
No result is inferred merely from a pushed commit.

## Evidence index

- [Frozen contract](../decisions/w3-approval-and-storage-contract.md) and
  [manifest](../decisions/w3-contract-v1.json).
- [API and internal-error review](w3-api-internal-error-review.md).
- [Qualified interface boundaries](../decisions/w3-qualified-interface-boundaries.md).
- [Dependency, freeze and fixture provenance](w3-dependency-freeze-provenance.md).
- [Durable storage slice](../decisions/w3-durable-storage-slice.md).
- [Artifact publication slice](../decisions/w3-artifact-publication-slice.md).
- [Artifact completion slice](../decisions/w3-artifact-completion-slice.md).
- [Finite failure/recovery matrix](w3-failure-recovery-matrix.md) and
  [machine-readable inventory](w3-failure-recovery-matrix.json).

Source/test paths below are relative to `crates/harness-core/` unless qualified.
These records identify implementation and test evidence; historical slice status
remains historical, not current acceptance.

## 0. Baseline and contract

- [x] Freeze document and manifest delivered; constants identify `alloy.w3/1` and `alloy-json-c14n/1`.
- [x] Implementation starting commit recorded: `58d7db53d8fb6f549cddba990dd9e83521989d92`.
- [x] Rust freeze verification implemented in `tests/w3_contract.rs::freeze_manifest_matches_normative_record`.
- [x] Rust canonical-vector reproduction implemented in `tests/w3_provenance.rs::frozen_vectors_are_reproduced_by_rust`; goldens are not overwritten.
- [x] Historical policy commit `a578f175c504e54719abb164bb1d638f328cfc5d` inspected: runtime promotion of `sha2 = "=0.10.9"`; its lock patch adds only the harness-core `tool-registry` edge.
- [ ] Review the full baseline-to-candidate dependency audit and exact passing freeze/provenance logs.
- [ ] Link passing baseline regressions; recording the starting SHA alone does not establish them.
- [ ] Confirm any frozen behavior or upper-bound changes through explicit contract review; do not silently revise the freeze to satisfy tests.

Reproduction proves current Rust reproducibility, not an undocumented historical
creation command. The design-time manifest status is not a runtime acceptance flag.

## 1. Files and public API boundaries

- [x] Required direct dependencies declared in `Cargo.toml`; `sha2` remains exactly pinned and `tool-registry` is a path dependency.
- [x] Required W3 lockfile dependency edge delivered; complete historical pin audit awaits reviewed execution evidence in section 0.
- [x] Reviewed module exports in `src/lib.rs`; active supervisor still described as text-only.
- [x] Internal-error review recorded without expanding the tool-result wire enum or adding unnecessary variants.
- [x] Schema-v2 integration delivered in `src/store/mod.rs`, with coordinator/migration/recovery in `src/store/durable/mod.rs`.
- [x] `src/policy.rs`, `src/approval.rs`, and `src/artifact.rs` delivered.
- [x] `tests/tool_policy.rs`, `tests/tool_artifacts.rs`, and `tests/w3_provenance.rs` delivered.
- [x] Registry-owned definition, invocation/context and result contracts reused.
- [x] Authorization/start handles have private fields and immutable accessors, with no unchecked public constructor, Clone, Copy or Deserialize.
- [x] Receipt compile-fail documentation delivered for cloning, reuse and deserialization.
- [x] W4/W5/W6 handoffs documented without claiming those integrations exist.
- [ ] Complete candidate-diff and W1/W2 regression verification of decoder/slab allocation and ownership preservation.

## 2. Canonicalization and immutable binding

Implementation: `src/approval.rs`, `src/policy.rs`, `src/store/durable/mod.rs`.
Tests: `tests/w3_contract.rs`, `tests/tool_policy.rs`, `tests/w3_provenance.rs`.

- [x] Registry/schema validation precedes invocation binding.
- [x] Upstream Qwen3 StrictJson recursively rejects duplicate decoded keys before returning argument objects; canonicalization cannot recover discarded duplicates.
- [x] Atomic/streaming duplicate-key split coverage delivered in `crates/families/qwen3/tests/tool_calls.rs::w3_boundary_duplicate_keys_fail_in_atomic_and_streaming_modes`.
- [x] Object keys sorted recursively by UTF-8 bytes; array order and strings preserved.
- [x] Pinned serde_json representation used, including represented `1` versus `1.0` distinction.
- [x] No coercion, default insertion, Unicode normalization, path rewriting or shell expansion in canonicalization.
- [x] Rust byte/digest fixtures and reproduction test delivered, including nested objects, arrays, escapes, Unicode and numeric distinctions.
- [x] Complete bounded binding snapshot retained and persisted, including epoch and session/turn/generation/call association.
- [x] Complete registered-definition digest and effective policy revision bound.
- [x] Exact `alloy.w3.invocation.v1` domain plus NUL separator implemented.
- [x] Lifetime `(session_id, call_id)` reservation delivered, including terminal calls.
- [x] Complete binding ceiling of 1 MiB enforced; changed invocation requires a new call identity.
- [x] Tests cover relevant argument/context/definition/policy/epoch changes and object insertion-order independence.
- [ ] Review exact passing canonicalization, binding and Rust reproduction logs for the final candidate.

## 3. Daemon-owned policy

Implementation: `src/policy.rs`; coverage: `tests/tool_policy.rs`.

- [x] Deny, AllowWithoutPrompt and RequireApproval implemented; no matching allow rule defaults to denial.
- [x] Typed configuration rejects malformed types/unknown fields and invalid or duplicate rules.
- [x] Policy strengthens registration approval requirements; destructive/always cases require prompting.
- [x] Human approval cannot make a policy-denied invocation runnable.
- [x] Approval-never still requires registry validation and exact-argument policy matching.
- [x] Effective timeout/output limits use minima and cannot widen registration.
- [x] Final start re-evaluates supplied policy/registry and compares binding identity.
- [x] Argument canonicalization kept distinct from operation-boundary resource authorization.
- [x] Authority-shaped input rejection tests and zero-execution counters delivered.
- [x] Five frozen-default policy fixture delivered: `w3_boundary_five_frozen_builtin_defaults_preserve_prompt_resolution`.
- [ ] Review passing boundary/policy logs; fixture delivery is not production builtin verification.

Actual resource enforcement and builtin registrations belong to W4. Raw policy
loading and authoritative live policy/registry publication belong to W6 (section 9).
Selected W3 safety floors are not an exhaustive deployed-default audit.

## 4. Approval state machine and single-use admission

Implementation: `src/approval.rs`, `src/store/durable/mod.rs`.
Coverage: contract graph, durable unit suite, P2-06 and Step C matrix.

- [x] Frozen active/terminal labels and permitted transitions implemented; terminal states cannot regain runnable authority.
- [x] First valid decision wins; duplicate/conflicting decisions add no approval event.
- [x] Unknown, wrong-session, expired and withdrawn decisions rejected.
- [x] Explicit denial recorded as tool.approved with allow:false; automatic authorization fabricates no human event.
- [x] Ready-to-Claimed consumption is atomic; final start consumes the opaque handle and commits tool.started before returning an execution receipt.
- [x] Final checks cover active session, deadline, supplied policy/registration, epoch and binding ownership.
- [x] Claim returns without retaining a SQLite transaction across the future hook gap.
- [x] Dropped/failed consumed handles do not restore Ready or authorize execution.
- [x] Decision/claim races and duplicate start/completion rejection coverage delivered.
- [x] 300-second monotonic lifetime, no renewal on allow, equality-as-expired and Claimed expiry/withdrawal implemented.
- [x] Pause/cancel withdraws pre-start authority transactionally; failed revocation is not acknowledged as success.
- [x] Started cancellation and failed completion never imply nonexecution or grant an execution retry.
- [x] No transactional exactly-once external-effect claim.
- [ ] Review final-candidate lifecycle and failure/recovery results, including real subprocess observations.

## 5. Durable storage, migration and recovery

Implementation: `src/store/mod.rs`, `src/store/durable/mod.rs`; test index: Step C matrix.

- [x] Transactional schema-v1-to-v2 migration and newer-schema rejection delivered.
- [x] Historical row preservation and migration rollback tests delivered.
- [x] Intent, artifact and runtime metadata tables delivered; new W3 payloads carry their contract marker.
- [x] Intent/request, decision/approval, claim consumption, start/event and completion/event transactions delivered.
- [x] Pre-execution rejection records no start; duplicate terminal results rejected.
- [x] Persistent sidecar ownership acquired before epoch recovery; competing daemon opens rejected.
- [x] Recovery withdraws unresolved pre-start authority, marks Started unknown, preserves terminal results and never replays effects.
- [x] Expiry, withdrawal and recovery reasons recorded durably.
- [x] Combined 32-per-session and 256-per-daemon pre-start admission limits implemented with terminal capacity release.
- [x] Limits are not claimed to bound all downstream running operations.
- [x] Statement/commit failures and lifecycle process-exit checkpoints delivered under test-only controls.
- [ ] Review final-candidate preservation, ownership, capacity and failure results.

## 6. Artifact implementation delivered; detailed branch review pending

Implementation: `src/artifact.rs`, metadata transaction in `src/store/mod.rs`,
completion association in `src/store/durable/mod.rs`. Tests: `tests/tool_artifacts.rs`
and artifact unit tests. Checked delivery does not close the detailed audit below.

- [x] Opaque art1_ references, private temporary creation and no-overwrite publication delivered.
- [x] Ownership metadata, length, digest, preview, timestamp and contract version stored.
- [x] Store-owned resolution, exact session/call lookup and complete integrity verification delivered.
- [x] Invalid/traversal/symlink references and missing/corrupt objects covered by rejection tests.
- [x] 16 MiB object, 4096 preview-source bytes, 5464 base64-character, 64 KiB copy buffer and eight-writer ceilings implemented.
- [x] 1 GiB/4096 managed quotas, shared reservations and tighter invocation limits implemented.
- [x] UTF-8/binary previews and over-limit/partial-input rejection tests delivered; truncation is not reported as complete output.
- [x] Streaming digest/length/preview, file synchronization, no-overwrite publication and directory synchronization pipeline delivered.
- [x] Metadata plus artifact.created commit transaction delivered; public reference returned only after successful registration.
- [x] Runtime streaming/synchronization split from metadata transaction; startup reconciliation exception documented.
- [x] Uncertain publication/commit handling and proof-before-cleanup reconciliation delivered; no automatic referenced-object eviction.
- [x] Artifact-backed completion validates committed session/call provenance and preserves existing ToolResult envelope/error categories.
- [x] Collision/quota/corruption/orphan/publication interruption and completion rejection/rollback coverage delivered.
- [ ] Finish untruncated review of allocation failure, reservation release, writer/PendingArtifact drop, synchronization and uncertain-commit branches.
- [ ] Finish review of ownership leases, full verification and proof-before-deletion ordering.
- [ ] Review final-candidate artifact publication, completion and interruption test results.

Hard-link publication conservatively reserves two entries and twice the bounded
maximum bytes until settlement. Reconciliation assumes trusted daemon-owned local
storage; it does not defend against malicious same-user directory replacement.
Filesystem publication and SQLite registration are not one atomic transaction.

## 7. Named gates and supporting coverage

### P2-06: destructive_effect_requires_bound_approval

- [x] Named counter-based fixture delivered in `tests/tool_policy.rs`.
- [x] Assertions cover no execution before approval, matching authorization, denied/wrong-session/changed requests and duplicate authority.
- [x] Supporting durable/Step C suites cover expiry, races, cancellation, restart, transaction failure and no replay.
- [ ] Review exact named test discovery and pass on Linux at the final candidate.
- [ ] Review exact named test discovery and pass on macOS at the final candidate.

### P2-08: oversized_tool_result_artifact_roundtrip

- [x] Named fixture delivered in `tests/tool_artifacts.rs`.
- [x] Assertions cover full accepted bytes, digest, length, reference, bounded preview, completion and unchanged history after reopen.
- [x] Supporting tests cover empty/exact/over limits, UTF-8/binary output, interruption, quotas, collisions, integrity, paths and cleanup.
- [x] Migration preservation and rollback coverage delivered in durable/Step C suites.
- [ ] Review exact named test discovery and pass on Linux at the final candidate.
- [ ] Review exact named test discovery and pass on macOS at the final candidate.

Reported earlier host run: 21 Step C, 25 durable, 9 artifact-unit, 34 artifact
integration, 15 contract and 24 policy tests passed. This is historical conversation
evidence, not a reviewed final-candidate report. The later builtin-default test
adds one policy test; the new provenance target adds one test and Qwen3 gains the
duplicate-key boundary test. Reconcile exact validator inventories; never accept
zero-test success or suppress discovery mismatches.

## 8. Final evidence and acceptance (open)

- [ ] Record exact final candidate SHA, clean checkout and toolchain/platform identity.
- [ ] Review full dependency/freeze/Rust-reproduction evidence and retain raw logs.
- [ ] Review formatting, locked workspace compilation and Clippy with warnings denied.
- [ ] Review harness-core all-targets/doc, registry, Qwen3 and required workspace tests.
- [ ] Review strict rustdoc results, including the repaired Qwen3 shape comments.
- [ ] Review required W1/W2/native regression evidence; no gate waived.
- [ ] Retain Linux/macOS logs, exact named test discovery, source hashes and CI/artifact links.
- [ ] Finish detailed artifact/failure branch review and resolve any findings.
- [ ] Record a scoped W3 acceptance decision; this checklist is not sign-off.
- [ ] Verify PR links issue #30 and identifies downstream boundaries.
- [x] Keep source delivery, executed validation and accepted results distinct.
- [x] Preserve finite SQL/process-exit fault model, trusted-storage assumptions and no-replay limitation.
- [x] Do not claim builtin execution, plugins, live approvals, the integrated tool loop or overall Phase 2 completion.

## 9. Downstream obligations (not missing W3 implementations)

- [ ] W4: verify actual builtin defaults and enforce resource scopes/limits at operation boundaries.
- [ ] W5: enforce permitted hook responses without mutating the approved invocation or original result.
- [ ] W6: bound/strictly parse raw policy input and fail closed for unavailable/malformed authoritative configuration.
- [ ] W6: serialize current policy/registry publication against final start and race-test stale snapshots.
- [ ] W4/W5/W6: verify caller-owned store/session/native locks are released across waits, hooks, streaming, verification and execution.
- [ ] W6: handle failed revocation/completion conservatively without false cancellation acknowledgment or external-effect replay.

These are explicit integration handoffs. Their implementation belongs to the
downstream packages, not a retroactive expansion of W3.
