# W3 Step C — failure/recovery coverage matrix

Baseline: `47068b96dd991d468aa60a7433e9301c6a883608`.
Contract: `alloy.w3/1`; schema remains v2; wire/result formats unchanged.
Status: implementation candidate. No box below asserts an unexecuted test passed.
Machine-readable inventory: `w3-failure-recovery-matrix.json`.

## Sequential gate

This patch closes the identified Step C coverage gaps, reusing—not replacing—the
already delivered approval, artifact and completion tests. Run and review Step C
before starting Step D. Step D owns final API/export/error review, checklist
reconciliation, candidate-bound Linux/macOS evidence, native regression links,
and the scoped W3 acceptance decision. A push is delivery, not acceptance.

## Fault model and actual observations

Statement failures use SQLite RAISE(ABORT) triggers after the corresponding state
update but before the transition can finish. Actual COMMIT failures use deferred
foreign-key violations: the SQL statements succeed, then SQLite rejects COMMIT.
Tests assert the intended injected cause, autocommit restoration, unchanged
history/projection, and absence of returned authority. A failed decision may be
explicitly retried after repair; a failed start consumes its handle and never
reissues Ready. A failed completion never grants another execution opportunity.

Cancellation/withdrawal failure does NOT falsely acknowledge that revocation
committed. Projection/status/history roll back together; tests retry withdrawal
or pause successfully before attempting final start. W6 must handle cancellation
storage errors rather than claiming a failed cancellation succeeded. Previously
valid authority is not described as durably revoked by a failed transaction.

Five lifecycle functions gain checkpoint statements immediately before and after
their actual tx.commit call, guarded individually with #[cfg(test)]. The helper
and entire new test module are also cfg(test). No production environment variable,
RPC, policy, public API, hook, dependency or failure-control path is added. A child
unit-test process exits with a strict sentinel code without running Drop at ten
commit boundaries. The parent reads raw persisted state before recovery, then
checks the recovered state, epoch change, historical row identity and exact JSON
bytes, artifact integrity, saved completion envelope, and durable effect marker.
The child is owned and has a 30-second timeout. Polling observes process completion;
no sleep/retry is used to make database recovery or authority checks pass.

These tests model process exit and specific SQL failures on the running local
filesystem. They do not promise arbitrary hardware/power-loss immunity or claim
that every possible SQLite/filesystem I/O error was simulated.

## New coverage

| Requirement | New test(s) in store::durable::failure_recovery |
|---|---|
| Intent + request atomicity; no escaped authority | intent_statement_and_commit_failures_publish_no_authority |
| Policy-denied intent + terminal event atomicity | denied_intent_failure_rolls_back_request_and_terminal_event |
| First valid allow/deny only after commit | allow_and_deny_statement_and_commit_failures_are_atomic |
| Ready -> Claimed rollback and no escaped handle | claim_statement_and_commit_failures_return_no_handle |
| No execution receipt before successful start commit | start_statement_and_commit_failures_return_no_execution_receipt |
| Completion rollback and no reexecution opportunity | completion_statement_and_commit_failures_never_regrant_execution |
| Expiry equals deadline, including Claimed; failed expiry cannot authorize | expire_failure_is_atomic_and_expired_authority_never_escapes |
| Failed withdrawal is not a successful revocation | withdrawal_failure_does_not_falsely_acknowledge_revocation |
| Pause status + withdrawal atomicity | pause_failure_rolls_back_status_and_withdrawal_together |
| Hook veto failure consumes claim, never restores Ready | veto_failure_consumes_claim_without_restoring_ready |
| Duplicate starts at Started and Completed | duplicate_start_attempts_do_not_admit_a_second_effect |
| Duplicate completion and unchanged terminal history | duplicate_completion_attempt_cannot_append_a_second_terminal_result |
| Final registration revalidation across eight definition changes | every_registered_definition_change_invalidates_final_start |
| Current policy denies after claim; zero observable effects | policy_change_rejection_has_zero_marker_effects |
| No SQLite transaction retained across the future hook gap | claimed_hook_gap_retains_no_sqlite_transaction |
| Failed epoch recovery preserves old epoch/history | recovery_statement_and_commit_failures_preserve_prior_epoch_and_history |
| Failed v1 -> v2 COMMIT rolls back DDL/version/raw history | migration_commit_failure_rolls_back_ddl_version_and_raw_history |
| Ten actual lifecycle process-exit boundaries and no replay | process_exit_before_and_after_each_commit_preserves_single_use_recovery; crash_child_probe |
| Post-start cancellation is not nonexecution | post_start_pause_does_not_claim_nonexecution_or_reissue_authority |
| Exact byte/file reservation rejection and release without leftovers | exact_file_and_byte_reservation_failures_release_capacity_without_orphans |

Duplicate-handle tests deliberately reconstruct equivalent PRIVATE bindings
inside cfg(test) so database single-use enforcement is tested independently of
Rust move semantics. No public unchecked constructor or Clone is introduced.
Existing compile-fail receipt tests remain required. All attempted effects use
one counter/append-only marker fixture only after a returned start receipt; no
real shell/network/filesystem builtin implementation is enabled.

## Lifecycle crash expectations

| Commit boundary | Persisted state before recovery | Expected after recovery | Effect marker |
|---|---|---|---|
| Intent before | no intent | no intent | absent |
| Intent after | AwaitingApproval | Withdrawn | absent |
| Decision before | AwaitingApproval | Withdrawn | absent |
| Decision after | Ready | Withdrawn | absent |
| Claim before | Ready | Withdrawn | absent |
| Claim after | Claimed | Withdrawn | absent |
| Start before | Claimed | Withdrawn | absent |
| Start after, before returned receipt | Started | OutcomeUnknown | absent; recovery must not infer nonexecution |
| Completion before | Started, artifact committed | OutcomeUnknown; artifact preserved | exactly once |
| Completion after, before caller returns | Completed | Completed; result preserved, no duplicate terminal | exactly once |

Unresolved Started always records unknown outcome; the marker is a test observation,
not authority for production recovery to infer whether an arbitrary effect ran.

## Existing coverage retained

The validator also requires the existing durable unit suite (migration/history,
ownership, races, expiry, pause and restart), all canonical/state contract tests,
the policy suite including P2-06, the 9 artifact unit tests (including six
publication/process-exit stages) and all 34 artifact integration tests (including
extended P2-08, corruption, collisions/quotas, orphan proof and Step B completion).
Those existing cases are not reimplemented here. Workspace, Clippy, docs and
format gates still run. No W1/W2/native gates are waived; Step D records their
candidate-specific results and required hardware evidence.

## Completion rule

The finite matrix is implemented by this patch. It becomes a passed Step C result
only after the validator records exact discovery/passing names, real subprocess
observations, source hashes, logs and the reviewed candidate on the actual host.
Use --candidate on a clean committed checkout for SHA-bound collection. The
validator never sets w3_complete=true or signs off Step D. Preserve the full logs,
including failed runs, and do not treat local validation as cross-platform proof.
