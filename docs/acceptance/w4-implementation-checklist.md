# Phase 2 W4 — builtin tools and cancellation

Issue: [#31](https://github.com/LayerDynamics/alloy/issues/31)
Dependencies: W1, W3
Status: not started

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
- [ ] `W4-01.02` Extend execution context with cancellation, deadline and byte limits; preserve the existing result/error contracts.
- [ ] `W4-01.03` Register the five builtins with the exact defaults listed below.

## W4-02 — fs.read

- [ ] `W4-02.01` Implement policy-scoped file/range reads with path and symlink checks.
- [ ] `W4-02.02` Enforce deadline and output-byte limits; return the existing inline/artifact result form.

## W4-03 — fs.list

- [ ] `W4-03.01` Implement policy-scoped directory listing with path/symlink checks.
- [ ] `W4-03.02` Return deterministic results with bounded entry count, output and execution time.

## W4-04 — fs.write

- [ ] `W4-04.01` Validate the approved destination and contents; prevent path/symlink scope escape.
- [ ] `W4-04.02` Enforce limits and cancellation; clean owned incomplete writes and report any effect honestly.

## W4-05 — shell.run

- [ ] `W4-05.01` Spawn the exact executable, argument array and working directory after authorization; no implicit shell interpolation.
- [ ] `W4-05.02` Bound stdout/stderr collection and enforce the approved deadline.
- [ ] `W4-05.03` On cancellation or timeout, terminate and reap owned processes without affecting unrelated work.

## W4-06 — http.fetch

- [ ] `W4-06.01` Implement bounded GET/HEAD requests to policy-approved destinations.
- [ ] `W4-06.02` Check every redirect before following; reject destination-scope escapes.
- [ ] `W4-06.03` Enforce response-byte limits, deadlines and cancellation; close owned network resources.

## W4-07 — Cancellation and resource cleanup

- [ ] `W4-07.01` Apply operation-specific cancellation across filesystem, process and HTTP work.
- [ ] `W4-07.02` Release owned handles, processes and output resources on completion, failure, timeout and cancellation.
- [ ] `W4-07.03` Test session isolation and avoid holding store/session/native inference locks across tool execution.

## W4-08 — Acceptance and closeout

- [ ] `W4-08.01` Implement and run P2-07 builtin_scope_and_output_limits, proving allowed operations and scope/limit rejection.
- [ ] `W4-08.02` Implement and run P2-09 tool_cancel_releases_owned_resources, proving owned cleanup and isolation; record any remaining cross-package integration coverage explicitly.
- [ ] `W4-08.03` Run both named gates on Linux and macOS; record the candidate SHA and result links.
- [ ] `W4-08.04` Run affected existing tests, formatting, Clippy and documentation checks; fix failures.
- [ ] `W4-08.05` Review the change, link the PR to #31, and update this tracker from recorded results.

## Acceptance

| Gate | Test | File | Linux | macOS |
|---|---|---|---|---|
| P2-07 | `builtin_scope_and_output_limits` | `crates/tool-registry/tests/builtins.rs` | Not run | Not run |
| P2-09 | `tool_cancel_releases_owned_resources` | `crates/harness-core/tests/tool_cancellation.rs` | Not run | Not run |

Candidate SHA: —
PR: —
Evidence: —

## Statuses

Not started → in progress → implemented → validated.

## Out of scope

- Reimplementing W1/W3
- Plugin hosting (W5)
- Live approval RPC/CLI and canonical model/tool loop (W6)
- MCP, skills and overall Phase 2 acceptance
