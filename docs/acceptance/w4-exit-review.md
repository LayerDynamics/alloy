# W4 exit review — pending evidence and review

Issue: #31. Decision: pending; no platform pass or final acceptance is asserted.
Preparation baseline: `f3988af56ab51d4df6622b20c1826f5a10188a99`.
The final candidate is the commit containing this patch; use retained context.txt.

## Added checks

- P2-07 `builtin_scope_and_output_limits`: actual scoped read/list/write, process,
  and local HTTP effects; path/redirect rejection, limits and direct-child reap.
- P2-09 `tool_cancel_releases_owned_resources`: pre-start and in-output cancellation
  of all five concrete adapters, then successful work in another session through
  the same store/executor. No partial output or replay; published writes stay honest.
- Active shell execution allows a second session and an independent SQLite writer.
- Dropping a prepared read releases private output and the retained database lease.

Every concrete callback probes an independent SQLite write transaction. This
proves no retained SQL write transaction, not absence of every possible Rust
mutex. Detached W4 jobs retain leases, not a SessionStore borrow. External callers'
session-map/native inference locks and live pause orchestration remain W6 work.
Generation, human-approval waits and hook cancellation remain W5/W6 coverage;
this W4-scoped P2-09 entrypoint is not full Phase 2 P2-09 acceptance.

## Required evidence

- [ ] Clean final candidate SHA, toolchain and W4 source manifests.
- [ ] Exact discovery/execution of P2-07 on Linux and macOS.
- [ ] Exact discovery/execution of W4-scoped P2-09 on Linux and macOS.
- [ ] Supporting cancellation tests and existing builtin/adapter regressions pass.
- [ ] Required native/workspace, formatting, strict Clippy and docs checks pass.
- [ ] Platform candidate identities/source manifests and artifact links reconciled.
- [ ] Source/lock/failure-path review and limitations recorded by a reviewer.
- [ ] PR references #31 and records a scoped W4 decision.

Run `bash scripts/builtin_acceptance.sh` from a clean committed checkout. It retains
candidate/platform/toolchain, W4 source hashes, exact discovery and run logs under
`target/w4-acceptance/<platform>/`. CI reuses the existing Linux native container
and Mac lane, preserves their regressions and always uploads W4 evidence.
Linux copied-checkout SHA is CI-supplied; compare its source manifest to macOS.
Unrun steps or missing artifacts are not passes. This record must be completed
with actual run/artifact URLs; test-code delivery is not platform evidence.

Create-only writes, process-group containment limits, cooperative filesystem I/O,
explicit HTTP origin/address pins and exchange budgets remain boundaries.
Documentation-only follow-up commits must identify the actual tested code SHA;
do not relabel earlier results as a different candidate's evidence.
