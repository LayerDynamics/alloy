# W3 approval and storage contract

Status: design frozen for implementation; runtime not implemented or accepted by this record.
Contract: `alloy.w3/1`
Canonicalization: `alloy-json-c14n/1`
Issue: #30
Branch: `feat/30-policy-approvals-artifacts`
Starting commit: `58d7db53d8fb6f549cddba990dd9e83521989d92`

## Authority and scope

Source requirements: `../specs/PHASE-2-EXECUTION-PLAN.md`, W3 and P2-06/P2-08.
This record freezes new W3 implementation decisions; its numeric defaults, state
names, and schema choices are not quotations from SPEC-1. Markdown is normative;
the adjacent `w3-contract-v1.json` records machine-readable constants and required
fixture coverage, not executed tests or golden canonicalization outputs.

W3 owns policy, durable approvals, lifecycle storage, and artifact storage.
W4 owns actual builtin execution and operation-boundary resource enforcement.
W5 owns plugin hosting. W6 owns live approval RPC/CLI and tool-loop integration.
No live tool execution, sandbox, approval UI, or overall Phase 2 acceptance is
claimed by adding these documents. W2 parser/slab ownership remains unchanged.

## 1. Authorization boundary

Complete parse -> registry resolution -> schema validation -> immutable invocation
snapshot -> daemon policy -> explicit bound approval when required -> single-use
claim -> before hooks -> final authorization check -> durable tool.started ->
external execution.

- Model output supplies data, not executable authority.
- Registry metadata is not permission to access a resource.
- Approval cannot override policy denial.
- `approval: never` skips only the human prompt, not validation or scope checks.
- Execution uses exactly the validated snapshot evaluated and shown for approval.
- Changed invocations require a new request and call ID.
- Hooks cannot mutate the invocation, original result, policy, or approval.
- Authorization is not an operating-system sandbox or protection against
  arbitrary malicious code already running inside the daemon.

Policy outputs: Deny(reason), AllowWithoutPrompt(binding), RequireApproval(binding).
No matching allow rule means denial. Invalid/unavailable configuration fails
closed. Configuration may strengthen but never weaken registration requirements.
Policy and registration changes invalidate outstanding authorization under the
previous revision/digest. W4 must recheck resource scope at the actual operation
boundary; canonicalizing arguments does not solve path/symlink check/use races.

Builtin defaults remain:

| Tool | Effect | Approval |
| --- | --- | --- |
| fs.read | read | never |
| fs.list | read | never |
| fs.write | write | policy |
| shell.run | destructive | always |
| http.fetch | read | policy |

## 2. Immutable binding

Persist the complete bounded snapshot and its digests:

- contract_version, daemon_epoch
- session_id, turn_id, generation_id, call_id
- tool_id, registered_tool_version, registered_definition_digest
- canonical_arguments, arguments_digest
- policy_revision
- effective_effect, effective_approval_requirement
- effective_timeout_ms, effective_max_output_bytes
- created_at_unix_ms, expires_at_unix_ms

(session_id, call_id) is unique for the lifetime of the store, including terminal
calls. Session/turn/generation/call identity is daemon-owned. The registered
definition digest covers the complete definition including schema, origin,
effect, approval, and limits. Policy revision covers the complete effective
policy configuration. daemon_epoch identifies the current daemon instance.

Binding digest: SHA-256 of UTF-8 `alloy.w3.invocation.v1` followed by one NUL
byte followed by the canonical encoding of the complete binding. It is an
identity checksum, not a signature or bearer credential. A digest alone is
not sufficient retained evidence of what was approved.

## 3. Canonical arguments: alloy-json-c14n/1

1. Input has already passed structural and registered-schema validation.
2. Duplicate keys must be rejected before object construction; canonicalization
   must not hide ambiguous raw input.
3. Recursively order object keys by UTF-8 byte sequence; preserve array order.
4. Preserve string contents exactly. No Unicode normalization, whitespace
   trimming, path rewriting, shell expansion, default insertion, or type coercion.
5. Encode strings/numbers with pinned serde_json; no insignificant whitespace.
6. Preserve integer/floating distinctions where represented by pinned
   serde_json. Do not deliberately equate 1 and 1.0.
7. Encoder changes that alter output require a canonicalization-version change
   and updated golden fixtures. This is not claimed to implement RFC 8785.

Path resolution and other resource checks are separate operations. They may not
silently replace the approved argument object. Canonicalization fixtures must
be generated/verified using the pinned Rust serializer, not Python JSON output.

## 4. State machine

Active states: AwaitingApproval, Ready, Claimed, Started.
Terminal states: Denied, Expired, Withdrawn, Rejected, Completed, OutcomeUnknown.

| From | To | Requirement |
| --- | --- | --- |
| New validated intent | AwaitingApproval | Policy requires explicit decision |
| New validated intent | Ready | Policy permits without prompt |
| New validated intent | Rejected | Policy denies; no execution |
| AwaitingApproval | Ready | First valid matching allow |
| AwaitingApproval | Denied | First valid matching deny |
| AwaitingApproval / Ready / Claimed | Expired | Deadline reached before Started |
| AwaitingApproval / Ready / Claimed | Withdrawn | Pause/cancel before Started |
| Ready | Claimed | Atomic single-use claim |
| Claimed | Rejected | Before-hook veto/failure or final authorization failure |
| Claimed | Started | Final checks and tool.started commit |
| Started | Completed | Actual outcome commits |
| Started, unresolved after restart | OutcomeUnknown | Do not infer or replay effect |

Terminal states never become runnable. The expiry transition from Claimed is
explicit because the authorization deadline includes before-hook time.

The future session.approve request remains { session_id, call_id, allow }.
The daemon finds the persisted binding; clients cannot provide replacement
arguments, tool versions, or policy. First valid decision wins. Duplicate or
conflicting decisions return already-decided failure without another approval
event or execution opportunity. Unknown, wrong-session, expired, or withdrawn
requests cannot change state. `tool.approved` records explicit `allow`, including
false. Automatic authorization never fabricates a human allow event.

## 5. Single-use ownership

AuthorizedInvocation is an opaque internal handle with private fields, no public
unchecked constructor, and no Clone, Copy, or deserialization. It is bound to
the persisted invocation and current daemon epoch. The final transition consumes
it when recording tool.started; dropping it never executes, restores authority,
or silently makes a claim reusable. Database state remains authoritative.

Before hooks run while Claimed without database/session locks. After hooks return,
recheck cancellation, expiry, policy revision, registration digest, epoch, and
ownership before committing Started. A committed tool.started means execution
may be attempted; it does not prove that an external effect occurred. Durable
single-use admission is not transactional exactly-once external execution.

## 6. Expiry, cancellation, restart

Authorization lifetime is 300 seconds from intent creation, including approval
wait, scheduling, and before hooks. Allow does not renew it. Expiry is checked
on decision, claim, and final start; equality with the deadline is expired.
Use monotonic deadlines during the live epoch. Persisted Unix timestamps provide
audit/history, not authority to renew a grant after restart.

Pause/cancel withdraws AwaitingApproval, Ready, or Claimed authority. After
Started, cancellation targets the owned operation; do not claim non-execution.
On restart, AwaitingApproval/Ready/Claimed become Withdrawn with reason
`daemon_restarted`; unresolved Started becomes OutcomeUnknown. Preserve terminal
states. Do not auto-replay. New attempts require new identities/authorization.

## 7. Database schema, transactions, and migration

Database schema_version becomes 2; new W3 payloads carry contract `alloy.w3/1`.
This is an application schema revision, not a renaming of the SPEC appendix.

- Existing events: append-only historical evidence.
- tool_intents: immutable binding plus current lifecycle/decision state.
- tool_artifacts: committed metadata and ownership.
- Runtime metadata: store identity and current daemon epoch.

Mutable state/projections do not replace or rewrite the event log. Every durable
transition and corresponding lifecycle event commit together:

- Intent + tool.requested before publishing approval_requested.
- Decision + tool.approved with explicit allow.
- Claim consumption + tool.started before effects.
- Completion + tool.completed, preventing duplicate terminal results.
- Artifact metadata + artifact.created.

Pre-execution rejection records an error outcome without tool.started.
Expiry/withdrawal/recovery transitions must also leave a durable explanation
using the existing lifecycle vocabulary and versioned payloads. OutcomeUnknown
must not fabricate a successful completion or evidence of non-execution.

Transactions are short. No SQLite/store lock, session-map lock, or native slot
is held across approval waits, hooks, artifact streaming/synchronization, or
external execution. Policy evaluation uses an immutable revision snapshot;
revalidate that revision at the final start boundary.

Migration v1 -> v2 is transactional. Preserve all older event IDs, sequence
numbers, payloads, and artifact-reference fields. Add W3 contract markers only
to new payloads. Reject unsupported newer schemas before application mutation.
Enforce store-scoped exclusive mutating daemon ownership with an OS lock before
epoch recovery; a second daemon cannot withdraw the first daemon's authority.
Use durable SQLite settings for supported local storage; failed commits cannot
produce usable authorization.

## 8. Limits

These are initial defaults AND upper bounds for this revision. Configuration may
lower them. Increasing them requires an explicit contract revision.

| Limit | Value |
| --- | --- |
| Complete canonical invocation binding | 1 MiB |
| AwaitingApproval/Ready/Claimed per session | 32 combined |
| AwaitingApproval/Ready/Claimed per daemon | 256 combined |
| Authorization lifetime | 300000 ms |
| Artifact payload per object | 16 MiB |
| Preview source bytes | 4096 |
| Base64 preview characters | 5464 maximum (4 * ceil(4096 / 3)) |
| Streaming copy buffer per writer | 64 KiB |
| Concurrent artifact writers per store | 8 |
| Managed artifact byte quota | 1 GiB |
| Managed artifact file quota | 4096 |

Artifact accounting includes temporary/orphan files and active reservations
without double-counting their actual bytes. Reserve capacity before writing.
Limits do not override tighter tool collection limits. Full output means all
accepted bytes within collection/storage limits; never label truncated output
as a complete artifact. Quotas bound logical managed-file bytes, not filesystem
allocation overhead, SQLite/WAL size, or process RSS. No automatic eviction of
referenced artifacts in W3; full stores reject new admission. Started operations
remain governed by the execution/concurrency contracts of W4/W6; the pending
intent limit is not a complete bound on those downstream resources.

## 9. Artifact identity and metadata

Opaque reference syntax: `art1_` followed by exactly 32 lowercase hexadecimal
characters (128 random bits). Collision-safe exclusive publication is mandatory.

Metadata: artifact_ref, session_id, call_id, byte_length, sha256, preview_bytes,
preview_encoding, created_at_unix_ms, contract_version. Metadata binds ownership
and the originating invocation. References are not paths or bearer permissions;
lookup checks the requesting session's permitted association.

Preview: inspect at most 4096 source bytes. Use exact UTF-8 text if possible,
trimming only an incomplete trailing code point. Otherwise use bounded base64.
Preview is explicitly incomplete/untrusted. Base64 text is at most 5464 ASCII
characters, excluding JSON encoding/envelope overhead. Preview metadata must
state its encoding and must not masquerade as the complete result.

Preserve ToolResult wire variants: exactly call_id + content + is_error OR
call_id + artifact_ref + is_error. No new top-level preview/digest/length fields.
Preserve tool error categories protocol/execution/timeout/policy. Cancellation
is execution with a structured reason. is_error does not prove no effect ran.

## 10. Artifact publication

Use an explicitly configured daemon-owned root with temporary and published
objects on the same local filesystem. User/model reference strings never provide
pathname components. Prevent traversal, symlink redirection, and overwrite.

1. Reserve file-count and byte capacity.
2. Exclusively create a private temporary file.
3. Stream bounded bytes; incrementally compute length, SHA-256, and preview.
4. Finalize/synchronize the complete file.
5. Publish without overwriting another object.
6. Synchronize containing directories as required by supported storage.
7. Commit metadata and artifact.created atomically in SQLite.
8. Return the reference only after successful commit.

No database transaction remains open during streaming/synchronization.
Filesystem publication and database commit are separate operations.

Before publication: no valid reference; interrupted temporary files need cleanup.
After publication/before metadata commit: orphan object, not publicly retrievable.
Commit failure/uncertainty: do not report success or blindly delete; first
reconcile committed state. After commit: reference resolves to the complete
recorded bytes. Missing/corrupt committed objects are integrity/storage failures;
never substitute a different file or partial bytes. Reads must not report
integrity verification until complete length/digest validation succeeds; no
whole-artifact buffering requirement is introduced by verification.

Under exclusive ownership, startup cleanup removes managed temporary files and
objects proven unreferenced by committed state. If that proof is unavailable,
fail closed rather than delete. Account for leftovers before admitting writers.
Test process-crash guarantees on supported storage; do not promise immunity to
arbitrary hardware failure or a malicious same-user process controlling files.

## 11. Required fixture/test coverage

P2-06 destructive_effect_requires_bound_approval in
`crates/harness-core/tests/tool_policy.rs`:

- No execution before approval; deny/stale/wrong-session/changed/duplicate
  decisions produce zero extra effects, observed via counter/marker.
- Canonical nested values, key order, Unicode, escapes, and numeric distinctions.
- Binding changes for arguments, context, definition, revision, and epoch.
- Expiry boundaries; allow does not renew; expiry while Claimed.
- Racing decisions/claims, withdrawal, pre/post-start cancellation.
- Restart invalidation and unknown outcome; no automatic replay.
- Transaction failures cannot produce usable authority.

P2-08 oversized_tool_result_artifact_roundtrip in
`crates/harness-core/tests/tool_artifacts.rs`:

- Full accepted bytes, length, digest, opaque reference, bounded preview,
  append-only artifact event, and close/reopen roundtrip.
- Empty/exact/one-over limits, binary data, UTF-8 boundaries, bounded streaming.
- Failures at every write/publication/commit boundary, uncertain commit recovery.
- Concurrent quotas/reservations, orphans/temp cleanup, invalid references,
  collisions, missing/corrupt objects, traversal and symlink redirection.
- Transactional v1 migration preserves all older rows.

Both named gates require actual Linux and Mac execution evidence; a docs-only
freeze or zero-test success does not satisfy either gate. Preserve W1/W2,
workspace/native, and strict rustdoc regressions. Runtime acceptance requires
candidate SHA, commands, logs, and platform evidence.

## Implementation order

1. This decision record and machine-readable constants.
2. Rust canonicalization golden fixtures and state-transition tests.
3. Policy evaluation and transactional durable approval storage.
4. Bounded artifact publication, integrity, failure/recovery tests.
5. Candidate-bound Linux/Mac validation and scoped W3 acceptance.

Production files: harness-core Cargo.toml; src/lib.rs, error.rs, store.rs;
new src/policy.rs, approval.rs, artifact.rs. Cargo.lock changes only for actual
pinned dependencies. Live RPC/CLI, builtin execution, plugin hosting, and the
complete tool loop remain downstream work. Changes to this contract require an
explicit documented revision plus updated machine data and fixtures.
