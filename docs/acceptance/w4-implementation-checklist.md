# Phase 2 W4 — builtin tools and cancellation

Issue: [#31](https://github.com/LayerDynamics/alloy/issues/31)
Dependencies: W1, W3
Status: in progress

## Progress snapshot — 2026-11-08

Shared context and filesystem/shell primitives are delivered. W3 authorization, registration and result adapters remain incomplete; W4-01 through W4-05 are not complete.

Checked boxes mean the exact implementation action is delivered, not acceptance validation. Three of 23 actions are checked. Task IDs/descriptions, defaults and named gates are unchanged.

| Slice | Delivery commit |
|---|---|
| W4-01 | [030561bb1803](https://github.com/LayerDynamics/alloy/commit/030561bb18030d2a7bfdcff1b622f29b1f221369) |
| W4-02 | [20bfced20cac](https://github.com/LayerDynamics/alloy/commit/20bfced20cac167d9021db25e96a36544e2eca36) |
| W4-03 | [d42e988265fe](https://github.com/LayerDynamics/alloy/commit/d42e988265feda5781d8da460736600865488474) |
| W4-04 | [7b9300de60c7](https://github.com/LayerDynamics/alloy/commit/7b9300de60c7c7a2b9831741af2f6b2169168fe3) |
| W4-05 | [877b448398a9](https://github.com/LayerDynamics/alloy/commit/877b448398a9e405feb6c4e9f6e9c875cbb6be82) |

Commit links prove source delivery, not CI/test acceptance.

## Production files

- `crates/tool-registry/src/execution.rs`
- `crates/tool-registry/src/builtins/mod.rs`
- `crates/tool-registry/src/builtins/fs.rs`
- `crates/tool-registry/src/builtins/shell.rs`
- `crates/tool-registry/src/builtins/http.rs`

## Required defaults

| Tool | Effect | Approval |
|---|---|---|
| fs.read | read | never |
| fs.list | read | never |
| fs.write | write | policy |
| shell.run | destructive | always |
| http.fetch | read | policy |

## W4-01 — Shared execution contract

- [ ] `W4-01.01` Reuse W1 registry and W3 authorization; execute only the unchanged authorized invocation.
- [x] `W4-01.02` Extend execution context with cancellation, deadline and byte limits; preserve the existing result/error contracts.
- [ ] `W4-01.03` Register the five builtins with the exact defaults listed below.

Implementation status: in progress. W4-01.02 is implemented: cancellation, monotonic deadlines, non-widening limits and bounded output/abort contracts; wire result/error categories are unchanged. W4-01.01 remains open for unchanged W3 authorization/durable-start/completion handoff. W4-01.03 remains open: five schema factories preserve defaults, but factories are not five executable registrations. Live dispatch is disabled.

## W4-02 — fs.read

- [ ] `W4-02.01` Implement policy-scoped file/range reads with path and symlink checks.
- [ ] `W4-02.02` Enforce deadline and output-byte limits; return the existing inline/artifact result form.

Implementation status: in progress. Pinned descriptors, no-follow resolution, bounded binary range streaming and cooperative cancellation/deadlines are implemented. Both tasks remain open for their full stated boundary: W3 policy/authorized-invocation binding and the existing inline/artifact result adapter. Aborted output must not become complete results or trigger replay. No snapshot-isolation or universally interruptible kernel-I/O claim.

## W4-03 — fs.list

- [ ] `W4-03.01` Implement policy-scoped directory listing with path/symlink checks.
- [x] `W4-03.02` Return deterministic results with bounded entry count, output and execution time.

Implementation status: in progress. W4-03.02 is implemented: deterministic raw-byte ordering, bounded entry count/serialized output, context checks and independent cursors. Filename bytes are preserved; portable encoding tests are separate from filesystem support for non-UTF-8 fixtures. W4-03.01 remains open for W3 authorization/policy-scope handoff. Results observe a live directory, not a transactional snapshot.

## W4-04 — fs.write

- [ ] `W4-04.01` Validate the approved destination and contents; prevent path/symlink scope escape.
- [x] `W4-04.02` Enforce limits and cancellation; clean owned incomplete writes and report any effect honestly.

Implementation status: in progress. A separate writable root/input ceiling and create-only staged publication are implemented. W4-04.02 is implemented: limits, cancellation checkpoints, owned staging cleanup, publication-state reporting and cleanup diagnostics. W4-04.01 remains open for W3-approved destination/content binding. Existing destinations remain unchanged; overwrite semantics are unresolved. Published/unknown outcomes must not be retried automatically. Source documents trusted-directory and crash-durability limitations.

## W4-05 — shell.run

- [ ] `W4-05.01` Spawn the exact executable, argument array and working directory after authorization; no implicit shell interpolation.
- [ ] `W4-05.02` Bound stdout/stderr collection and enforce the approved deadline.
- [ ] `W4-05.03` On cancellation or timeout, terminate and reap owned processes without affecting unrelated work.

Implementation status: in progress. Exact executable/argv/cwd, configured scope, cleared environment, null stdin, bounded binary stream capture and context checks are implemented. W4-05.01 remains open: an executable allowlist is not destructive/always approval. W4-05.02 remains open for approved-limit/result handoff. W4-05.03 remains open at the full owned-work/session-isolation boundary: group termination and direct-child reaping have local coverage, but escaping descendants are not contained. Darwin zombie-group handling uses bounded observation, not blanket permission-error suppression.

## W4-06 — http.fetch

- [ ] `W4-06.01` Implement bounded GET/HEAD requests to policy-approved destinations.
- [ ] `W4-06.02` Check every redirect before following; reject destination-scope escapes.
- [ ] `W4-06.03` Enforce response-byte limits, deadlines and cancellation; close owned network resources.

Implementation status: not started. GET/HEAD execution, initial/redirect policy, response bounds, cancellation and network cleanup remain pending. An http.fetch schema/default is not an executable implementation.

## W4-07 — Cancellation and resource cleanup

- [ ] `W4-07.01` Apply operation-specific cancellation across filesystem, process and HTTP work.
- [ ] `W4-07.02` Release owned handles, processes and output resources on completion, failure, timeout and cancellation.
- [ ] `W4-07.03` Test session isolation and avoid holding store/session/native inference locks across tool execution.

Implementation status: in progress. Filesystem/process cancellation, output aborts, staging cleanup and child ownership primitives exist. HTTP cleanup, cross-tool/session isolation and integrated store/session/native-slot lock-release verification remain pending. Primitive signatures alone do not prove integrated behavior. All three tasks remain unchecked.

## W4-08 — Acceptance and closeout

- [ ] `W4-08.01` Implement and run P2-07 builtin_scope_and_output_limits, proving allowed operations and scope/limit rejection.
- [ ] `W4-08.02` Implement and run P2-09 tool_cancel_releases_owned_resources, proving owned cleanup and isolation; record any remaining cross-package integration coverage explicitly.
- [ ] `W4-08.03` Run both named gates on Linux and macOS; record the candidate SHA and result links.
- [ ] `W4-08.04` Run affected existing tests, formatting, Clippy and documentation checks; fix failures.
- [ ] `W4-08.05` Review the change, link the PR to #31, and update this tracker from recorded results.

Implementation status: in progress. Pasted macOS development output reports 17/17 w4_shell tests passing, including the child fixture. This is not either named acceptance gate. The latest pasted validation stopped at the echo fixture Clippy sliced_string_as_bytes diagnostic; a correction was supplied, but final passing Clippy/workspace/harness/strict-doc output has not been provided. A source push is not candidate-pinned validation. No Linux/named-gate result links were provided, and no PR was found for the branch. All five closeout tasks remain unchecked.

## Acceptance

| Gate | Test | File | Linux | macOS |
|---|---|---|---|---|
| P2-07 | `builtin_scope_and_output_limits` | `crates/tool-registry/tests/builtins.rs` | Not run | Not run |
| P2-09 | `tool_cancel_releases_owned_resources` | `crates/harness-core/tests/tool_cancellation.rs` | Not run | Not run |

Candidate SHA: `877b448398a9e405feb6c4e9f6e9c875cbb6be82` (implementation head; not acceptance-validated)
PR: — (none found for this branch at update time)
Evidence: source-delivery commits and pasted macOS development output only. Retained candidate-pinned acceptance/CI links remain pending.

## Statuses

Not started → in progress → implemented → validated.

## Out of scope

- Reimplementing W1/W3
- Plugin hosting (W5)
- Live approval RPC/CLI and canonical model/tool loop (W6)
- MCP, skills and overall Phase 2 acceptance
