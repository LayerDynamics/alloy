# W3 API and internal-error review

Status: source-reviewed obligations; runtime acceptance pending.
This record does not declare W3 accepted or downstream integration implemented.

## Evidence levels

Checked checklist boxes mean source-reviewed implementation/documentation only.
Unchecked boxes are not automatically missing features: they require the linked
source/test or execution evidence to be reviewed. Historical slice records retain
their original baseline and candidate status; they are not current acceptance.
The reported Step C host run passed 21 new, 25 durable, 9 artifact-unit,
34 artifact-integration, 15 contract and 24 policy tests. This conversation report
is not a substitute for raw logs on a clean committed Linux/macOS candidate.

## API decisions and downstream obligations

| Responsibility | Source assessment | Evidence / handoff |
|---|---|---|
| Opaque receipts | AuthorizedInvocation and StartedInvocation have private binding fields, immutable accessors and no public constructor/Clone/Copy/Deserialize | src/store/durable.rs; StartedInvocation compile-fail rustdoc; run harness-core --doc |
| Single use | start_tool and complete_tool take receipts by value; database state/epoch/digest checks reject reconstructed duplicate authority | durable.rs; Step C duplicate_start_attempts_do_not_admit_a_second_effect and duplicate_completion_attempt_cannot_append_a_second_terminal_result |
| W4 immutable execution | binding() exposes shared references only; no executable is selected by model input | W4 must execute only the registered unchanged invocation after successful start commit, with bound resource scopes/limits |
| W5 immutable hooks | W3 provides immutable views, not a plugin enforcement implementation | W5 must send bounded copies and accept only veto/reason/annotations; never apply changes to arguments or original results |
| W6 authoritative revisions | start_tool re-evaluates the supplied registry/policy; it cannot determine that caller-supplied instances are current | W6 must serialize configuration publication and final start or use an equivalent revision fence; old snapshots are not authoritative |
| Hook gap | claim returns without retaining a SQL transaction; final start revalidates after hooks | Step C claimed_hook_gap_retains_no_sqlite_transaction |
| Streaming/verification | begin and lookup return values without SessionStore borrows; stream/finish/verify must run after caller releases locks | artifact.rs; completion validates database provenance only, not file bytes |
| Startup reconciliation | ArtifactStore::open takes &mut SessionStore and hashes committed objects during reconciliation | Run as exclusive startup work before session/native-slot locks or runtime admission; a blanket claim that all hashing occurs without any store borrow would be false |
| Cancellation failure | withdraw_tools/pause can fail before revocation commits | No successful cancellation acknowledgment on error; stop admission for affected operation and repair/withdraw before continuing; a failed revocation alone is not durable cancellation |
| Completion failure | complete_tool consumes receipt even on failure; no execution retry is returned | Preserve/inspect durable outcome; unresolved Started becomes OutcomeUnknown on restart; never rerun the effect |
| Unknown outcomes | Recovery withdraws pre-start intents, marks unresolved Started unknown, preserves Completed | Step C lifecycle crash matrix; no exactly-once external-effect guarantee |

## Internal-error conclusion

Retain the existing error variants for this W3 boundary. HarnessError::Invalid
represents invalid requests/state and Store represents persistence/artifact
failures; both must fail closed. CanonicalizationError and BindingError preserve
module-specific preparation failures. No new tool-result wire category is needed.

This conclusion is conditional on conservative handling, NOT typed recovery:
Store(String) cannot reliably distinguish disk full, quota, integrity failure,
lock contention or commit uncertainty. Do not branch on message substrings or
infer retry safety from Invalid versus Store. Operation stage and persisted state
control recovery. If W6 needs machine-readable differentiated responses, review a
separate internal error design before that integration; do not expand wire enums.
Existing W6 callers do not yet establish these tool-operation obligations.

## Evidence index

- docs/decisions/w3-approval-and-storage-contract.md and w3-contract-v1.json:
  normative contract; this review does not change ceilings or wire formats.
- src/approval.rs: canonical encoding, immutable binding, state graph.
- src/store/durable.rs: admission/decision/claim/start/completion/recovery.
- src/artifact.rs: bounded publication, lookup/verification, startup reconciliation.
- tests/tool_policy.rs: P2-06 destructive_effect_requires_bound_approval.
- tests/tool_artifacts.rs: P2-08 oversized_tool_result_artifact_roundtrip.
- docs/acceptance/w3-failure-recovery-matrix.md and .json: finite fault model and
  named Step C tests; not a hardware/power-loss guarantee.
- docs/decisions/w3-artifact-completion-slice.md: exact completion envelope and
  ownership/provenance boundary.

Source paths above are under crates/harness-core unless prefixed docs/.

## Remaining acceptance and review actions

- Review every still-unchecked original checklist line against exact source/test
  evidence, including dependency pins, freeze digest and upstream W2 boundaries.
- Collect clean committed-candidate Linux and macOS logs, exact named gate
  discovery, workspace/all-targets/doc/strict-rustdoc/Clippy/fmt results, and required
  W1/W2/native regressions. Preserve toolchain versions, source SHA and raw logs.
- Review the evidence and record a scoped acceptance decision. Do not auto-sign.
- Preserve trusted local daemon-owned storage assumptions, finite process-exit
  and SQL-failure model, conservative uncertain publication handling, and no replay.

No claim is made here that W4/W5/W6, native tests, cross-platform acceptance,
PR review, or W3 sign-off has completed.

Reviewed source checkout: `8f31fbc677a9888847294d3a70a700fea346344b`.
