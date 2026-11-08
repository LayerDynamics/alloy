# Phase 2 W4 — Formal execution plan

Issue: #31 — https://github.com/LayerDynamics/alloy/issues/31
Repository: LayerDynamics/alloy
Branch: feat/31-builtin-tools-cancellation
Dependencies: W1 and W3
Status: proposed plan; no W4 implementation or test result claimed
Tracker delivery commit: 28d8b32868557cb74b22516648b9bae7106454b9
Date: 2026-11-08

## 1. Objective and scope

Implement the five required builtins under one registry/execution contract, with
resource-scope enforcement, approved deadlines/byte limits, cancellation and owned
cleanup. This plan organizes the existing eight-step, 23-task tracker. It does not
add tasks, features, a custom acceptance framework or new wire error categories.

Source of scope: issue #31 and docs/specs/PHASE-2-EXECUTION-PLAN.md, sections 3–5.
Progress remains in the existing Markdown/JSON trackers under docs/acceptance/.

## 2. Deliverables

| Production file under crates/tool-registry/src/ | Responsibility |
|---|---|
| execution.rs | Shared execution context, cancellation/deadline/byte-limit contract |
| builtins/mod.rs | Registration and schemas for the five builtins |
| builtins/fs.rs | fs.read, fs.list and fs.write |
| builtins/shell.rs | Exact argv execution and owned-process cleanup |
| builtins/http.rs | Bounded GET/HEAD and redirect/destination enforcement |

Update existing registry/exports and manifest/lockfile only where required for
these deliverables. Keep dependencies pinned; no unrelated package upgrades.
Any necessary W3/W4 adapter belongs at the existing harness boundary, not a
reverse tool-registry dependency on harness-core. Record a file-placement
refinement if necessary; do not silently introduce an extra work package.

| Tool | Effect | Approval |
|---|---|---|
| fs.read | read | never |
| fs.list | read | never |
| fs.write | write | policy |
| shell.run | destructive | always |
| http.fetch | read | policy |

These defaults remain unchanged. Never bypasses only the human prompt, not
registry validation or resource-scope checks.

## 3. Execution design within W4-01

Reuse the W1 definition/invocation/result contracts and W3 authorization/artifact
storage. A resolved request or policy preparation result is not execution authority.
Production execution must consume the unchanged invocation permitted by the
successful W3 durable-start boundary and retain the receipt for completion.

The shared execution context carries the approved limits, a monotonic deadline,
operation-specific cancellation and bounded output handling. It must not permit
callers or model data to widen limits or mutate the approved invocation.

Before connecting effects, settle concrete schemas, resource-scope representation,
output threshold versus total accepted bytes, and child cleanup behavior as part
of W4-01. These are implementation decisions within the existing scope, not
additional tracked stages or invented frozen limits.

Filesystem operations check path/symlink scope at the actual operation boundary.
Shell receives an executable, exact argv and cwd with no implicit interpolation.
HTTP checks destination policy before the initial connection and each redirect.
Resource checks may narrow/reject an operation; they must not silently change the
approved invocation. Output uses the existing inline-or-artifact result envelope;
large output uses W3 committed artifacts without exceeding the approved ceiling.

## 4. Work sequence and completion conditions

| Tracker step | Work | Completion condition |
|---|---|---|
| W4-01, tasks .01–.03 | Shared contract and registrations | Five registrations have exact defaults; execution/cancellation/deadline/output interfaces integrate with W1/W3 without bypasses |
| W4-02, tasks .01–.02 | fs.read | Authorized bounded file/range reads; path/symlink rejection; correct inline/artifact result |
| W4-03, tasks .01–.02 | fs.list | Authorized deterministic bounded directory results |
| W4-04, tasks .01–.02 | fs.write | Exact approved destination/content; scope/limit/cancel enforcement and owned incomplete-write cleanup |
| W4-05, tasks .01–.03 | shell.run | Exact argv/cwd spawn; bounded stdout/stderr; timeout/cancel terminates and reaps owned work |
| W4-06, tasks .01–.03 | http.fetch | Bounded GET/HEAD; checked redirects; deadline/cancel closes owned network resources |
| W4-07, tasks .01–.03 | Cancellation and cleanup | All operation kinds release owned resources and remain isolated across sessions; no store/session/native inference lock retained across execution |
| W4-08, tasks .01–.05 | Tests and closeout | Named gates pass on Linux/macOS, affected regressions pass, results recorded, change reviewed and PR linked to #31 |

Start cancellation/cleanup support with W4-01 and apply it to each builtin; W4-07
is the cross-tool verification step, not permission to defer safe cleanup until
all tools are written. Implement filesystem tools before shell and HTTP to reuse
the shared output/limit behavior. Shell and HTTP may then proceed independently.

Suggested commit order: shared contract/registrations; filesystem tools; shell;
HTTP; cross-tool cancellation verification; acceptance and tracker updates.
This groups the existing work; it does not create additional scope.

## 5. Tests and validation

Required acceptance gates:

| ID | Exact test | File | Required platforms |
|---|---|---|---|
| P2-07 | builtin_scope_and_output_limits | crates/tool-registry/tests/builtins.rs | Linux and macOS |
| P2-09 | tool_cancel_releases_owned_resources | crates/harness-core/tests/tool_cancellation.rs | Linux and macOS |

P2-07 must show allowed operations plus scope/limit rejection using controlled
filesystem/process/local HTTP fixtures. P2-09 must show owned-resource cleanup and
session isolation on cancellation. Use effect markers/counters and process/handle
observations where relevant; error assertions alone do not establish no effects.
Do not present W4-only cancellation cases as already-tested W5/W6 integration.
Keep the named requirement intact and record remaining cross-package coverage.

Run registry and harness all-targets/doc tests, affected existing W1/W2/W3 and
workspace/native regressions, formatting, Clippy with warnings denied and strict
documentation checks. Reuse existing CI where possible; this plan does not require
a new runner project. Required discovery must not succeed with zero matching tests.

Record the candidate SHA, platform, commands and retained result links in the
existing trackers. Keep implementation delivery distinct from validation; update
checkboxes only for the stated completed action. Linux and macOS results must
refer to the same reviewed candidate or have an explicit reviewed explanation.

## 6. Failure behavior and completion boundary

Denial or failed pre-start authorization permits no effect. After an operation
starts, cancellation, timeout or persistence failure must not claim nothing ran.
Clean owned resources, preserve the existing result/error categories, and submit
completion through W3 without replaying external effects on completion failure.
Do not hold store/session/native inference locks while executing or streaming.

W4 is complete when all five builtins meet the stated contract, both named gates
pass on Linux/macOS, affected checks pass, and the reviewed PR closes #31 with
tracker evidence updated. A planning document, source push or reported green icon
is not itself that result.

Out of scope: rebuilding W1/W3, plugin hosting (W5), live approval RPC/CLI and the
canonical tool loop (W6), MCP, skills and overall Phase 2 sign-off.

## References

- docs/specs/PHASE-2-EXECUTION-PLAN.md
- docs/acceptance/w4-implementation-checklist.md
- docs/acceptance/w4-implementation-checklist.json
- docs/decisions/w3-qualified-interface-boundaries.md
