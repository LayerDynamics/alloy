# Phase 2 W4 — builtin tools and cancellation

Issue: [#31](https://github.com/LayerDynamics/alloy/issues/31)
Dependencies: W1, W3
Status: in progress

## Progress snapshot

W4-01 through W4-06 implementations are delivered. Checkboxes record implementation actions, not acceptance validation. The original eight steps and 23 task IDs/descriptions are unchanged.

W4-07/08 tests and CI evidence collection are added by this candidate. Execution, platform evidence and review actions remain unchecked until actual results are retained.

Implementation baseline: `f3988af56ab51d4df6622b20c1826f5a10188a99`. Final test candidate: pending the commit containing this patch.

Review: [pending W4 exit review](w4-exit-review.md). PR linkage remains pending.

## Required defaults

| Tool | Effect | Approval |
|---|---|---|
| fs.read | read | never |
| fs.list | read | never |
| fs.write | write | policy |
| shell.run | destructive | always |
| http.fetch | read | policy |

## W4-01 — Shared execution contract

- [x] `W4-01.01` Reuse W1 registry and W3 authorization; execute only the unchanged authorized invocation.
- [x] `W4-01.02` Extend execution context with cancellation, deadline and byte limits; preserve the existing result/error contracts.
- [x] `W4-01.03` Register the five builtins with the exact defaults listed below.

Status: implemented.
Delivery commit: `576309d6514c7edf6a575c02f8468b8939b9abb4`; source delivery is not validation evidence.

## W4-02 — fs.read

- [x] `W4-02.01` Implement policy-scoped file/range reads with path and symlink checks.
- [x] `W4-02.02` Enforce deadline and output-byte limits; return the existing inline/artifact result form.

Status: implemented.
Delivery commit: `c34d97e032e911e13275f90a1a99848f245e2e72`; source delivery is not validation evidence.

## W4-03 — fs.list

- [x] `W4-03.01` Implement policy-scoped directory listing with path/symlink checks.
- [x] `W4-03.02` Return deterministic results with bounded entry count, output and execution time.

Status: implemented.
Delivery commit: `411f08928b51288b1d88ae602b70bc25189512cb`; source delivery is not validation evidence.

## W4-04 — fs.write

- [x] `W4-04.01` Validate the approved destination and contents; prevent path/symlink scope escape.
- [x] `W4-04.02` Enforce limits and cancellation; clean owned incomplete writes and report any effect honestly.

Status: implemented.
Delivery commit: `20cd26c82929fdcc1de6c4d68a8d451814797287`; source delivery is not validation evidence.

## W4-05 — shell.run

- [x] `W4-05.01` Spawn the exact executable, argument array and working directory after authorization; no implicit shell interpolation.
- [x] `W4-05.02` Bound stdout/stderr collection and enforce the approved deadline.
- [x] `W4-05.03` On cancellation or timeout, terminate and reap owned processes without affecting unrelated work.

Status: implemented.
Delivery commit: `a7a4361843c7efeeddd22b8208df0006ff6a5e01`; source delivery is not validation evidence.

## W4-06 — http.fetch

- [x] `W4-06.01` Implement bounded GET/HEAD requests to policy-approved destinations.
- [x] `W4-06.02` Check every redirect before following; reject destination-scope escapes.
- [x] `W4-06.03` Enforce response-byte limits, deadlines and cancellation; close owned network resources.

Status: implemented.
Delivery commit: `0e455efbe800599d85477cee352985cb9d057bd8`; source delivery is not validation evidence.

## W4-07 — Cancellation and resource cleanup

- [ ] `W4-07.01` Apply operation-specific cancellation across filesystem, process and HTTP work.
- [ ] `W4-07.02` Release owned handles, processes and output resources on completion, failure, timeout and cancellation.
- [ ] `W4-07.03` Test session isolation and avoid holding store/session/native inference locks across tool execution.

Status: in_progress.
New test/source delivery is separate from unrun candidate/platform evidence and final review.

## W4-08 — Acceptance and closeout

- [ ] `W4-08.01` Implement and run P2-07 builtin_scope_and_output_limits, proving allowed operations and scope/limit rejection.
- [ ] `W4-08.02` Implement and run P2-09 tool_cancel_releases_owned_resources, proving owned cleanup and isolation; record any remaining cross-package integration coverage explicitly.
- [ ] `W4-08.03` Run both named gates on Linux and macOS; record the candidate SHA and result links.
- [ ] `W4-08.04` Run affected existing tests, formatting, Clippy and documentation checks; fix failures.
- [ ] `W4-08.05` Review the change, link the PR to #31, and update this tracker from recorded results.

Status: in_progress.
New test/source delivery is separate from unrun candidate/platform evidence and final review.

## Acceptance

| Gate | Exact test | Linux | macOS |
|---|---|---|---|
| P2-07 | `builtin_scope_and_output_limits` | Not run | Not run |
| P2-09 | `tool_cancel_releases_owned_resources` | Not run | Not run |

P2-09 here covers W4 builtins only. W5/W6 generation, approval-wait and hook pause orchestration and external caller-lock integration remain pending. No full Phase 2 claim.

Use `scripts/w4_acceptance.sh` and the exit review for candidate/source provenance. Both platform records and required native/workspace regressions remain required.

## Out of scope

- Reimplementing W1/W3
- Plugin hosting (W5)
- Live approval RPC/CLI and canonical model/tool loop (W6)
- MCP, skills and overall Phase 2 acceptance
