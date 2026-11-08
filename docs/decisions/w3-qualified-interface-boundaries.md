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
