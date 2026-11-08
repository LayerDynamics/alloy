# W3 implementation and acceptance checklist

Status: implementation review and evidence tracking; not a W3 acceptance decision.

A checked source-review item identifies inspected implementation/documentation,
not an automatically passing test or downstream integration. Test execution and
platform acceptance require their own candidate-specific evidence. Unchecked
items may need review or evidence; they do not automatically mean missing code.

mean missing implementation. See [API review](w3-api-internal-error-review.md).

## 0. Baseline and contract

- [x] Freeze document exists at `docs/decisions/w3-approval-and-storage-contract.md`.
- [ ] Verify `w3-contract-v1.json` matches the normative Markdown and its recorded SHA-256.
- [ ] Confirm contract version `alloy.w3/1` and canonicalization version `alloy-json-c14n/1`.
- [ ] Record the implementation starting commit and establish passing baseline regressions.
- [ ] Treat changes to the frozen behavior or upper bounds as explicit contract revisions—not silent implementation deviations.

The frozen document explicitly distinguishes design approval from runtime implementation and acceptance.

## 1. Files and public API boundaries

### Production files

- [ ] Update `crates/harness-core/Cargo.toml` with only the dependencies actually required.
- [ ] Update `Cargo.lock` for actual dependency changes; retain pinned versions.
- [x] Update `crates/harness-core/src/lib.rs` with reviewed exports.
- [x] Review internal failures without expanding the tool-result wire enum; retain existing variants with conservative operation-stage handling (see API review).
- [ ] Extend `crates/harness-core/src/store.rs` for schema v2, durable transitions, migration, and recovery.
- [ ] Add `crates/harness-core/src/policy.rs`.
- [x] Add `crates/harness-core/src/approval.rs`.
- [x] Add `crates/harness-core/src/artifact.rs`.

### Acceptance-test files

- [ ] Add `crates/harness-core/tests/tool_policy.rs`.
- [ ] Add `crates/harness-core/tests/tool_artifacts.rs`.

### API invariants

- [ ] Reuse the existing registry definition, call, context, result, and error contracts.
- [x] Prevent public unchecked construction of authorization handles.
- [x] Keep authorization handles non-`Clone`, non-`Copy`, and non-deserializable.
- [ ] Preserve W2 decoder/slab ownership; do not modify the parser’s allocation contract to implement approvals.
- [x] Document the handoff interfaces needed by W4/W5/W6 without implementing those packages.

These are the production boundaries and ownership constraints recorded in the freeze.

## 2. Canonicalization and immutable binding

### Canonical arguments

- [ ] Require structural/schema validation before canonicalization.
- [x] Source-reviewed W2 recursive duplicate-key rejection before returning argument objects; W3 canonicalization does not replace parsing. New atomic/streaming split test requires execution.
- [ ] Recursively sort object keys by UTF-8 byte order.
- [ ] Preserve array order and string contents.
- [ ] Use the pinned Rust `serde_json` representation for strings and numbers.
- [ ] Preserve represented numeric distinctions such as `1` versus `1.0`.
- [ ] Perform no coercion, default insertion, Unicode normalization, path rewriting, or shell expansion.
- [ ] Generate and verify golden bytes/digests using Rust—not Python’s JSON encoder.

### Invocation binding

- [ ] Persist every frozen binding field, including daemon epoch and session/turn/generation/call association.
- [ ] Bind the complete registered definition and effective daemon-policy revision.
- [ ] Retain the complete bounded invocation snapshot, not only hashes.
- [ ] Compute the binding digest with the exact domain prefix and NUL separator.
- [ ] Enforce lifetime uniqueness of `(session_id, call_id)`, including terminal calls.
- [ ] Reject bindings exceeding 1 MiB.
- [ ] Require a new call identity for any changed invocation.

### Fixtures

- [ ] Cover nested objects, array order, escapes, Unicode, and numeric distinctions.
- [ ] Verify that relevant changes to arguments, context, definition, policy, and epoch alter the binding.
- [ ] Verify that object-key insertion order alone does not alter canonical bytes.

All canonicalization and binding rules above are frozen in `alloy.w3/1`.

## 3. Daemon-owned policy

- [ ] Implement `Deny`, `AllowWithoutPrompt`, and `RequireApproval`.
- [ ] Default to denial when no allow rule matches.
- [x] Source-reviewed typed policy configuration validation; tests cover malformed typed input.
- [ ] W6: bound and strictly parse raw policy input; fail closed on unavailable/malformed authoritative configuration, with loader tests.
- [ ] Allow policy to strengthen—but never weaken—registration requirements.
- [ ] Ensure approval cannot override policy denial.
- [ ] Ensure `approval: never` bypasses neither validation nor resource-scope checks.
- [ ] Run the five-default policy fixture; W4 must verify actual builtin registrations and operation-boundary enforcement. Selected W3 safety floors are not exhaustive default verification.
- [ ] Bind effective timeout/output limits and prevent widening.
- [x] Source-reviewed final-start binding re-evaluation against supplied policy/registry.
- [ ] W6: enforce authoritative current configuration and serialize reload publication against final start; race-test stale snapshots.
- [ ] Keep resource authorization separate from argument canonicalization.
- [ ] Test that model/plugin-provided authority-shaped data grants no permission.
- [ ] Document that actual operation-boundary enforcement belongs to W4.

The policy must use trusted registration and daemon state, not model claims.

## 4. Approval state machine and single-use execution

### States and transitions

- [ ] Implement all frozen active and terminal states.
- [ ] Reject every transition not explicitly allowed.
- [ ] Prevent terminal states from becoming runnable.
- [ ] Implement first-valid-decision-wins behavior.
- [ ] Reject duplicate/conflicting decisions without another approval event.
- [ ] Reject unknown, wrong-session, expired, and withdrawn requests.
- [ ] Record explicit denial as `tool.approved` with `allow: false`.
- [ ] Do not fabricate a human approval event for automatic authorization.

### Claim and start boundary

- [ ] Implement atomic `Ready → Claimed` consumption.
- [ ] Run future before-hook work outside database/session locks.
- [ ] Recheck cancellation, expiry, policy revision, registration digest, epoch, and ownership before `Started`.
- [ ] Consume the opaque handle when committing `tool.started`.
- [ ] Never execute before that commit succeeds.
- [ ] Ensure dropping a handle neither executes nor restores authority.
- [ ] Test racing decisions, racing claims, and duplicate start attempts.
- [x] Avoid claiming transactional exactly-once external effects.

### Expiry and cancellation

- [ ] Enforce the 300-second lifetime from intent creation.
- [ ] Do not renew expiry on allow.
- [ ] Check expiry on decision, claim, and final start.
- [ ] Treat equality with the deadline as expired.
- [ ] Use monotonic deadlines during the live daemon epoch.
- [ ] Permit expiry/withdrawal while `Claimed`.
- [ ] Withdraw pre-start authority on pause/cancel.
- [ ] Keep post-start cancellation distinct from “nothing executed.”

These behaviors are part of the frozen state machine, including the final authorization check after hooks.

## 5. Durable storage, migration, and recovery

### Schema and transactions

- [ ] Implement transactional schema v1 → v2 migration.
- [ ] Preserve historical event IDs, sequences, payloads, and artifact references.
- [ ] Reject unsupported newer schemas before application mutation.
- [ ] Add intent, artifact, and runtime metadata storage.
- [ ] Mark only new W3 payloads with `alloy.w3/1`.
- [ ] Atomically persist intent plus `tool.requested`.
- [ ] Atomically persist decision plus `tool.approved`.
- [ ] Atomically consume the claim plus append `tool.started`.
- [ ] Atomically record completion plus `tool.completed`.
- [ ] Prevent duplicate terminal results.
- [ ] Record pre-execution rejection without `tool.started`.
- [x] Document short runtime store access and release-before-I/O handoff; startup reconciliation is exclusive and may hash while borrowing the store.
- [ ] W4/W5/W6: verify caller-owned locks/native slots are not retained across waits, hooks, streaming, verification or execution.

### Ownership and recovery

- [ ] Acquire store-scoped exclusive mutating daemon ownership before epoch recovery.
- [ ] Prevent a second daemon from invalidating the first daemon’s authority.
- [ ] Withdraw old `AwaitingApproval`, `Ready`, and `Claimed` states on restart.
- [ ] Recover unresolved `Started` operations as `OutcomeUnknown`.
- [ ] Preserve terminal states.
- [ ] Never automatically replay an interrupted operation.
- [ ] Record durable explanations for expiry, withdrawal, and recovery.
- [ ] Inject transaction failures and verify that no usable authority escapes.

### Admission limits

- [ ] Enforce the combined pending/ready/claimed limit of 32 per session.
- [ ] Enforce the combined limit of 256 per daemon.
- [ ] Release admission capacity correctly on terminal transitions.
- [ ] Do not present these limits as complete bounds on downstream running operations.

Migration, exclusive ownership, and restart invalidation are explicitly frozen storage requirements.

## 6. Artifact storage and publication

### Identity, metadata, and lookup

- [ ] Generate `art1_` references with exactly 32 lowercase hexadecimal suffix characters.
- [ ] Enforce collision-safe, no-overwrite creation/publication.
- [ ] Store all frozen ownership, length, digest, preview, timestamp, and version metadata.
- [ ] Resolve references exclusively through store-owned paths.
- [ ] Enforce permitted session association on lookup.
- [ ] Reject invalid references, traversal, and symlink redirection.
- [ ] Fail on missing/corrupted committed objects; never substitute another object.

### Bounded collection

- [ ] Enforce 16 MiB maximum payload per artifact.
- [ ] Enforce 4,096 preview source bytes and 5,464 base64 characters.
- [ ] Handle UTF-8 boundaries and binary previews correctly.
- [ ] Use at most the frozen 64 KiB copy buffer per writer.
- [ ] Enforce eight concurrent writers.
- [ ] Enforce the 1 GiB managed-byte and 4,096 managed-file quotas.
- [ ] Include temporary/orphan files and reservations without double-counting.
- [ ] Reserve capacity before writing.
- [ ] Respect tighter tool limits.
- [ ] Reject over-limit output without labeling truncation as a complete artifact.
- [ ] Do not automatically evict referenced artifacts.

### Publication and failure handling

- [ ] Create private temporary files exclusively.
- [ ] Stream length, digest, and preview incrementally.
- [ ] Finalize/synchronize before publication.
- [ ] Publish without overwrite and synchronize directories as required.
- [ ] Commit artifact metadata plus `artifact.created` atomically.
- [ ] Return the reference only after successful commit.
- [ ] Keep database transactions closed during streaming/synchronization.
- [ ] Handle uncertain commit outcomes without blind deletion.
- [ ] Reconcile temporary/orphan files under exclusive ownership.
- [ ] Delete only objects proven unreferenced.
- [ ] Validate complete length/digest before claiming integrity verification.
- [ ] Preserve the existing inline-or-artifact result envelope and error categories.

The artifact contract separates filesystem publication from database commit; it does not claim cross-resource atomicity.

## 7. Required acceptance and failure tests

### P2-06 — `destructive_effect_requires_bound_approval`

- [ ] Test exists in `tests/tool_policy.rs`.
- [ ] No execution before approval.
- [ ] Valid approval admits only the matching invocation.
- [ ] Denied, changed, wrong-session, stale, and duplicate decisions cause zero extra effects.
- [ ] Use an execution counter/marker—not error assertions alone.
- [ ] Cover expiry, races, cancellation, restart, and commit failure.
- [ ] Discover and run the exact named test on Linux.
- [ ] Discover and run the exact named test on Mac.

### P2-08 — `oversized_tool_result_artifact_roundtrip`

- [ ] Test exists in `tests/tool_artifacts.rs`.
- [ ] Full accepted bytes, length, digest, reference, and preview survive reopen.
- [ ] Artifact creation is represented in append-only history.
- [ ] Interrupted writes never become valid references.
- [ ] Cover empty/exact/one-over boundaries and binary/UTF-8 output.
- [ ] Inject failures at write/publication/commit boundaries.
- [ ] Cover concurrent quotas, cleanup, collisions, corruption, and path attacks.
- [ ] Verify migration preserves old rows.
- [ ] Discover and run the exact named test on Linux.
- [ ] Discover and run the exact named test on Mac.

Both named gates and the supporting failure cases are required by the freeze; documentation or zero-test success does not satisfy them.

## 8. Regression evidence and merge gate

- [ ] Formatting passes.
- [ ] Locked workspace compilation passes.
- [ ] Clippy passes with warnings denied.
- [ ] Harness-core target and documentation tests pass.
- [ ] Registry contract tests remain green.
- [ ] Existing W1/W2 and required workspace/native regressions remain green.
- [ ] Strict rustdoc gates pass.
- [ ] Record the exact candidate SHA and clean-tree state.
- [ ] Retain commands, toolchain versions, Linux/Mac results, and raw logs.
- [ ] Keep implementation status separate from test and acceptance status.
- [ ] Review new dependencies and public authorization APIs.
- [ ] Record a scoped W3 acceptance decision.
- [ ] PR links issue #30 and states remaining downstream boundaries.
- [ ] Do not claim builtin execution, plugin hosting, live approval RPC/CLI, the full tool loop, or overall Phase 2 completion.

## Recommended execution order

```text
1. Verify the freeze pair.
2. Add Rust canonicalization fixtures and state-transition tests.
3. Implement policy and immutable binding.
4. Implement migration, exclusive ownership, and durable approvals.
5. Pass P2-06.
6. Implement artifact storage and failure recovery.
7. Pass P2-08.
8. Run regressions and collect Linux/Mac evidence.
9. Review and merge scoped W3.
```

Current closeout task: review remaining unchecked requirements and collect committed-candidate Linux/macOS evidence; implementation delivery is not acceptance.

## Qualified boundary review

See [W3 qualified interface boundaries](../decisions/w3-qualified-interface-boundaries.md).
Checked source/documentation items do not assert executed tests or downstream acceptance.
