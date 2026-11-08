# W3 qualified interface boundaries

Status: reviewed boundary documentation and executable fixture candidate;
not runtime acceptance or W3 sign-off.

## Duplicate keys: existing W2 enforcement, not W3 canonicalization

Qwen3 tool_calls.rs parses with StrictJson. Its recursive visit_map checks
contains_key before inserting a decoded key; escaped equivalent keys also collide.
Atomic parsing and push_stream reuse parse_complete. The added test exercises
all two-chunk splits in both modes, including nested array objects and escaped keys.
Streaming proposals from any generation remain tentative until finish succeeds;
a later error invalidates earlier proposals. No parser production/allocation code changes.
W3 receives validated objects; canonical_json cannot recover discarded duplicates.

## Configuration

DaemonPolicy::new validates typed configuration; evaluate resolves schemas first.
Unknown fields/types and invalid/duplicate rules fail closed in existing tests.
No raw policy loader is added. W6 configuration integration must bound raw bytes,
reject recursive duplicate keys, and refuse new authorization on unavailable or
malformed authoritative configuration. It must not replace a failed load with an
allow policy or accidentally reuse obsolete authority. These loader tests remain
W6 acceptance obligations, not completed W3 runtime behavior.

## Frozen builtin defaults and safety floors

The fixture explicitly evaluates all five frozen defaults:
fs.read/read/never, fs.list/read/never, fs.write/write/policy,
shell.run/destructive/always, http.fetch/read/policy.
Policy can strengthen prompting; policy registrations resolve via the explicit
rule. This is counter-only coverage, not production builtin registration/execution.
Existing selected safety-floor checks remain unchanged. W4 must test actual
registrations against the entire default table and enforce resource boundaries.
Do not make fs.read/fs.list production floor checks stricter merely to tick a box.

## Locks and startup

Runtime begin/lookup use short store access. Release caller store/session locks
before stream/finish/verify, hooks, waits or external execution. complete_tool
checks database provenance, not filesystem integrity. Startup ArtifactStore::open
borrows the store and hashes committed files during reconciliation. Perform this
as exclusive startup work before runtime admission, session locks or native slots.
The API does not prove that a downstream caller releases its own enclosing locks.

## Authoritative policy and registry

start_tool re-evaluates the supplied instances and checks binding identity. W6
must coordinate reload publication with final start, e.g. a short configuration
revision fence retained through the start transaction, never through hooks or
execution. A reload ordered before start must be observed by start; a reload
ordered after committed start cannot retroactively imply the effect never ran.
Missing/malformed authoritative configuration blocks admission. Downstream tests
must race reload against start and verify no stale start is admitted. No W6 live
configuration manager or reload guarantee is implemented by this documentation.

## Evidence and limits

Run w3_boundary_duplicate_keys_fail_in_atomic_and_streaming_modes and
w3_boundary_five_frozen_builtin_defaults_preserve_prompt_resolution, the complete
qwen3/tool-policy suites, existing final policy/registration-change tests and
workspace/doc regressions. Preserve finite SQL/process-exit fault and trusted
local storage assumptions. No new dependency, wire enum or authorization handle.
Passing a host run does not establish Linux/macOS acceptance or W3 sign-off.


## W4-01.01 adapter placement refinement

The receipt-consuming dispatcher lives in harness-core/src/tool_execution.rs,
not in tool-registry. ReceiptExecutor registers the same implementation Arc in
W1 and a private harness dispatch index. The registry is exposed read-only; no
raw executable handle or client approval flag is introduced. This preserves the
existing dependency direction and does not reimplement W3 admission/approval.

StartedInvocation retains a process-local monotonic observation immediately
before the successful start transaction's state update/commit. This private
observation is not serialized or added to the frozen binding; it conservatively
charges transaction/handoff delay against the approved execution timeout.
ExecutionContext::from_start prevents renewing that deadline at dispatch.

Preparation checks the current Started row/session/epoch/binding and current
registered definition, consumes the receipt, and retains the existing exclusive
store ownership lease. Detached run owns no SessionStore borrow or SQLite
transaction. Completion consumes the original receipt through complete_tool;
failed completion does not regrant execution. A dropped/inconsistent job remains
unresolved and recovers conservatively. No other caller lock is proven released
by these signatures; W6 must still validate its enclosing lock/configuration
publication discipline.

This generic registry/receipt bridge does not complete builtin registrations,
resource-scope adapters, streaming artifact finalization, W5 hooks, W6 dispatch,
or P2-07/P2-09 acceptance. Their existing W4 tasks remain open. Inline/artifact
wire shapes and W3 provenance checks are unchanged.


## W4-01.03 complete-catalogue registration entrypoint

ReceiptExecutor::with_builtins uses the existing tool-registry builtin definition
factory and registers each definition with its corresponding daemon-supplied
implementation through the same W1/private-index registration path. The required
BuiltinImplementations bundle has five nonoptional fields and no Default or
placeholder implementation. Initialization returns the entire fresh executor or
an error; it does not partially mutate a live catalogue or execute a tool.
Effect/approval defaults and schema identities are not duplicated or lowered by
this constructor. Limits remain explicit trusted inputs, tightened later by W3.

The registration tests supply explicitly identified counter backends and run W3
admission/approval/start/receipt dispatch for all five entries. They establish
registration/default/selection and authorization-boundary behavior, not actual
filesystem/process/network resource enforcement or P2-07/P2-09 acceptance.
Operational setup must supply all five configured concrete adapters. Missing
HTTP implementation cannot be hidden by a successful four-entry catalogue or
an automatically substituted not-implemented handler. Per-builtin resource and
canonical output adapters remain the subsequent W4 tasks; W6 admission is not
activated by this constructor.


## W4-02.01 concrete scoped file-read adapter

FileReadTool in tool-registry/src/builtins/fs.rs owns the pinned ReadScope and a
positive daemon-configured inline ceiling. Requests cannot choose the root or
raise that ceiling. Context-aware execution parses the unchanged validated
path/offset/length and delegates to the existing no-follow scoped range reader.
Its legacy entrypoint fails closed; production callers use receipt-owned dispatch.

This candidate supports complete bounded UTF-8 inline ToolResult content and
mirrors that exact serialized content to the supplied bounded sink. Explicit
scope-operation and incomplete-output diagnostics distinguish rejection from
attempted I/O. Missing/oversized/binary output never becomes truncated or lossy
success. Large or non-UTF-8 ranges return artifact_output_required until the
separate W4-02.02 streaming artifact adapter is implemented. No artifact reference
is fabricated, and W4-02.02/full W4 acceptance remain open.

The new tests register the real file adapter (with a test-only invocation counter)
and exercise W3 admission/start/dispatch/completion against real local files.
They retain existing scope and trusted-filesystem assumptions. Configured adapter
identity remains part of trusted daemon setup; W6 must coordinate authoritative
configuration publication and cancellation without retaining execution locks.
