# W3 policy and binding preparation slice

## Current implementation status (historical-slice clarification)

The text below records the preparation-only slice at its original delivery.
Its references to a future coordinator and pending P2-06/P2-08 describe that
historical stage, not the current implementation. The durable coordinator,
artifact store, artifact-backed completion, and finite recovery matrix have
since been delivered:

- [Durable storage and approvals](w3-durable-storage-slice.md).
- [Artifact publication](w3-artifact-publication-slice.md).
- [Artifact-backed completion](w3-artifact-completion-slice.md).
- [Failure/recovery matrix](../acceptance/w3-failure-recovery-matrix.md), with
  its [machine-readable inventory](../acceptance/w3-failure-recovery-matrix.json).

P2-06 and P2-08 are implemented fixtures; passing Linux/macOS results must still
be linked to a reviewed candidate. Source delivery is not runtime acceptance.
Raw configuration loading, builtin execution, plugins and live tool-loop
integration remain downstream responsibilities. No historical acceptance
status, contract version, bound, or fault-model claim is changed by this note.

## Original preparation-slice record

Status: implementation slice; durable authorization and acceptance pending.
Related issue: #30. Contract: alloy.w3/1.

This slice adds a strict, daemon-owned exact-argument allowlist policy mode,
`exact_arguments_v1`, as an initial conservative configuration representation.
It introduces no policy file loader, live RPC, executor, or resource sandbox.
Rules identify registry tools and exact canonical argument objects. There are
no wildcard/path-prefix/DNS rules. Unknown rules/arguments are denied.

For registered `always`, prompting is mandatory. For `policy`, the explicit
rule resolves prompting. For `never`, the rule may strengthen to prompting.
Destructive effects always require a prompt in this implementation. Builtin
shell/write/HTTP registrations cannot contradict frozen effect/approval floors.
Rule limits only tighten registration via minimums. An allow is preparation,
not approval or a capability. Before connecting W4, actual resource checks
must enforce the approved request at operation boundaries, including symlink,
working-directory, executable, destination and redirect behavior.

Policy identity hashes the complete normalized effective configuration: sorted
registry IDs, canonical sorted/deduplicated argument allowlists, explicit prompt
settings and limits, with contract and policy-mode identifiers. Syntactic order
and redundant allowlist duplicates do not change the effective revision.
Configuration identity encoding uses the existing bounded canonical encoder;
oversized configuration is rejected, not truncated. Its eventual loader must
bound raw input and reject duplicate keys before constructing JSON objects.

Evaluation uses ToolRegistry::resolve and returns a read-only owned invocation
binding after validation. All frozen binding fields are retained; the complete
canonical envelope must fit 1 MiB, not just arguments. Hashes are checksums, not
credentials. The binding has private fields, no public unchecked constructor,
and no Clone/Deserialize implementation. Canonical arguments are retained as
UTF-8 canonical JSON text in the snapshot.

Epoch/time are daemon inputs. This slice calculates the fixed expiry with
checked i64 arithmetic but DOES NOT enforce live monotonic expiry, reserve
unique calls, persist approval, consume authority, invalidate grants, acquire
store ownership, run hooks, or execute tools. The next durable-coordinator
slice must compare current epoch/policy/definition and every required bound
identity before atomically admitting a start.

Errors remain module-local typed Rust errors with sources, not new serialized
tool error categories. Promotion of sha2 from dev to runtime preserves its pin;
tool-registry becomes a direct harness-core dependency. Cargo.lock changes only
the harness-core dependency list; no registry version update is requested.

Tests in tests/tool_policy.rs cover policy decisions and binding preparation
with counter-only implementations. Even the shell-named fixture does not run
a shell and is not a deployed builtin contract. P2-06/P2-08 remain pending.
Retain the earlier 15 contract tests and all required regressions.
