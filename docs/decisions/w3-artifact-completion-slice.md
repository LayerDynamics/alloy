# W3 Step B — artifact-backed durable completion

Baseline: 4f3560ef72152aec76748a07067bd253e6414c3b.
Contract: alloy.w3/1. Database schema remains 2; no DDL or new wire variant.
Status: implementation candidate; acceptance requires actual recorded test results.

## Completed implementation scope

SessionStore::complete_tool now accepts both existing ToolResult variants. It
checks the result call ID against the consumed StartedInvocation, verifies the
persisted Started state, daemon epoch and full binding digest, and performs the
result update plus exactly one tool.completed event in one immediate transaction.

For artifact results it also validates reference syntax, the committed
session/call-owned tool_artifacts row, bounded frozen metadata, the invocation
output limit and the unique matching artifact.created evidence with matching
artifact_refs. A visible file without committed metadata is insufficient.
Foreign-session, foreign-call, unknown, malformed, unregistered and inconsistent
references fail closed. Completion never creates an artifact registration.

The saved result_json and nested event result retain EXACTLY the existing result
wire envelope: call_id + artifact_ref + is_error, or call_id + content + is_error.
No preview/digest/length fields are inserted into ToolResult. Artifact completion
sets the existing event-row artifact_refs array to the completed reference; inline
completion retains []. No new tool error category or contract marker is invented.
The schema-v2 contract already froze both ToolResult variants and artifact_refs;
this patch populates that existing shape rather than reinterpreting old rows.

## Locking, integrity and handoffs

Publication remains begin -> stream -> finish -> metadata commit. All byte writes,
hashing, filesystem synchronization and read verification remain outside the
completion transaction. complete_tool validates DATABASE provenance, not on-disk
byte integrity at the instant of completion. ArtifactStore::lookup followed by
UnverifiedArtifact::verify remains the mandatory path before exposing bytes.
A missing/corrupt object is an integrity failure, never substituted data.

W4 must execute only the unchanged StartedInvocation, publish accepted output,
then submit the canonical result. W5 after-hook annotations must not mutate the
original result. W6 must supply current authoritative policy/registry at start,
release store locks during streaming/verification, and persist this result before
feeding it back to the model. These downstream packages are not implemented here.

Completion consumes the non-Clone/non-Copy/non-deserializable start receipt even
on validation/storage failure. It never returns Ready or creates an execution
retry. Unknown or failed completion leaves Started unless a commit succeeded;
restart conservatively records OutcomeUnknown for an unresolved Started. A
failed return is NOT proof that no effect occurred. Do not replay the operation.
Existing valid committed artifacts remain owned and readable after recovery.

## Test coverage and acceptance

P2-08 now retains the actual start receipt, publishes the oversized result,
completes with the canonical artifact ToolResult, verifies tool.completed and its
artifact_refs, and reopens with the call still Completed and the complete bytes,
length, digest, preview and history unchanged. The publication/interruption and
integrity tests from the artifact slice remain in place.

Seventeen new completion tests cover both error flags, exact envelopes, inline
regression, wrong session/call/result identity, unknown/malformed/unregistered
references, metadata ownership and creation-proof corruption, duplicate creation
proof, epoch/digest mismatch, terminal projection rejection, SQL update/event
rollback and conservative reopen without reexecution. Three compile-fail rustdoc
cases enforce no Clone, no duplicate use after completion, and no deserialization.

Normal runtime completion cannot duplicate a receipt. The terminal-projection
fixture intentionally corrupts test-only projection state to exercise rejection
of a late receipt. Corruption tests modify fixture SQLite directly; these are not
new supported production mutation APIs. Event and update failure injection is
not the entire Step C crash/commit-uncertainty matrix.

Run the attached Step B validator on this host and retain logs/source hashes.
Run again on a clean committed candidate for SHA-bound evidence and on both
Linux/macOS before acceptance. Preserve W1/W2, workspace, strict documentation and
required native gates. Neither this document nor a pushed commit claims W3,
Step C, Step D, or overall Phase 2 acceptance.
