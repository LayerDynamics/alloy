# W3 durable approval storage implementation slice

Baseline: a578f175c504e54719abb164bb1d638f328cfc5d
Contract: alloy.w3/1 (unchanged).

This slice adds transactional v1 -> v2 migration, exclusive store ownership,
new daemon epochs and conservative recovery, persisted intent/decision/claim/
start/inline completion, monotonic expiry, withdrawal and admission limits.
No external tool is executed by these APIs.

Implementation placement: store.rs owns the connection and OS file lock;
store/durable.rs holds the coordinator and its focused tests. This is a file
organization decision, not new W3 scope. Artifact publication and completion now
use tool_artifacts; see w3-artifact-publication-slice.md and
w3-artifact-completion-slice.md for those follow-up implementation records.

All mutating SessionStore opens acquire a nonblocking OS lock on a persistent
`<database-path>.alloy-owner.lock` sidecar before SQLite migration/recovery. The
sidecar is held for the entire store lifetime and is NEVER unlinked on drop or
replaced during normal operation. SQLite retains sole responsibility for locking
its database/WAL/SHM files. A second daemon fails before epoch recovery.

The previous database-inode lock was incorrect on macOS, where flock and fcntl
locks interact, and caused SQLite opens/migration to fail with database locked.
Ownership is now separate from SQLite's own lock domain. Canonical parent paths
and rejection of final database symlinks remain in SessionStore::open. On Unix,
database hardlinks are rejected rather than allowing pathname aliases to select
different ownership sidecars. Existing sidecar symlinks and hardlinks are rejected;
opened/named sidecar identities are compared. The daemon-owned directory is trusted:
this is not a defense against malicious same-user concurrent file replacement.

All mutating access must use SessionStore; direct SQLite connections do not acquire
the daemon sidecar. Existing ownership tests cover same-process/subprocess contention
and release; additional regressions cover SQLite operation while ownership is held,
persistent sidecar identity, database hardlink aliases, and sidecar symlinks.
Requires Rust >= 1.89 and supported Linux/macOS local storage. Do not remove sidecars
as a stale-lock cleanup technique: OS locks release when handles/processes close.
Future read-only tooling needs a separate non-recovering reader API.

Schema inspection rejects newer versions before application DDL, WAL selection,
or epoch recovery. Existing spec_version values and raw event rows are retained.
Schema_version is bumped to 2 in the same migration transaction as the new tables.
Only new lifecycle events carry the alloy.w3/1 contract marker. Claimed transitions
use a tool.requested lifecycle receipt with transition=Claimed; terminal rejection,
expiry, withdrawal and recovery use tool.completed with explicit state/reason and
is_error=true. OutcomeUnknown additionally records outcome_known=false and is not
a successful result or proof of non-execution. These are versioned internal event
payloads, not additional wire kinds.

The public admission API takes the live registry/policy and the requested
invocation; it never accepts a client-constructed grant. A denied request retains
its validated request and denial explanation, not a fabricated allow binding.
Every (session_id, call_id) remains reserved, including terminal calls. Registry
and policy validation errors do not create a validated intent.

Decision, claim and final start check live monotonic deadlines. Final start
re-evaluates the immutable invocation with the current daemon policy and registry,
compares the complete binding digest, checks session status, persisted epoch,
claim ownership and deadline, and returns a receipt only after commit. Before
hooks run between claim_tool and start_tool with no database transaction held.
Dropping a claim or losing a start receipt never makes authority reusable.
Transactions that fail return no execution receipt. Ambiguous external outcomes
remain Started until conservative restart recovery, never automatic replay.

The caller passes the CURRENT authoritative policy/registry to start_tool; stale
snapshots must not be presented as current by W6. W4 must enforce operation-boundary
resource scope and execute only the receipt's unchanged invocation. Completion
accepts the existing inline and artifact-backed ToolResult variants. Artifact
results require committed session/call-owned metadata and matching creation
evidence; see w3-artifact-completion-slice.md. The result envelope remains bounded
to 1 MiB; larger bytes belong in the bounded artifact store. Completion consumes
the receipt; failed/uncertain completion never returns permission to execute.
Unresolved Started is conservatively recovered as OutcomeUnknown.

Pausing or completing a session withdraws pre-start intents atomically with the
status update. Started operations remain W4/W6-owned cancellation responsibilities.
Per-session/daemon pending capacity is 32/256 across AwaitingApproval, Ready and
Claimed. Capacity is not a bound on Started operations.

Validation added: focused tests for migration/raw-history preservation, newer
schema rejection, same-process and subprocess ownership, decision and claim
single-use behavior, denial identity reservation, policy-change rejection, hook
veto, pause, monotonic expiry including Claimed, nonrenewing approval, dropped
claims, restart recovery, inline completion, session/daemon capacity, racing decisions/claims, migration rollback and transaction
failure injection at decision/start event insertion. The P2-06 counter fixture is
added to tests/tool_policy.rs. Test presence is not test evidence.

Remaining acceptance work: Rust compilation and lint fixes if required; actual
Linux/macOS runs; additional commit-uncertainty and crash/fault matrix; operation-boundary
W4 tests. Existing W1/W2/workspace/native gates remain required. P2-08 artifacts,
plugins, live RPC/CLI and the canonical model tool loop are not implemented here.
No complete W3 or overall Phase 2 sign-off is asserted by this patch.

## Explicit ownership release

The logical ownership guard explicitly calls File::unlock during Drop rather
than relying solely on closing one descriptor. Rust file locks may remain held
while duplicated or inherited descriptors are alive. This is a candidate fix for
the intermittent post-drop reopen failure observed in parallel tests; the prior
assertion discarded the actual reopen error, so that log alone does not prove
which operation failed. The reopen test now reports that error with expect.

SessionStore keeps conn before _ownership in declaration order so SQLite closes
before ownership releases. The wrapper is not Clone and never unlinks the sidecar.
A deterministic regression retains a duplicated descriptor while dropping the
logical owner and verifies immediate reacquisition and continued exclusion by
the replacement owner. Full normal-parallelism tests and repeated release tests
must pass before this slice is committed. No serial-test or sleep workaround is
introduced. Drop cannot propagate unlock errors; the descriptor still closes as
a fallback. Test presence is not Linux/macOS validation evidence.
