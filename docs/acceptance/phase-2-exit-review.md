# Phase 2 exit review

**Decision: accepted for implementation candidate `9faba79f9ed27d11bcf1d8e7e7776826517587d0` on 2026-11-11.** All P2-01 through P2-16 requirements and the preserved Phase 1 regressions passed. Independent implementation, integration and acceptance reviews found no remaining blocker after the ownership and pause fixes. This sign-off is documentation-only; the final PR head must also have passing required checks before merge.

## Candidate and retained evidence

- [Complete Linux/Mac CI run](https://github.com/LayerDynamics/alloy/actions/runs/43072447930): every job passed, including CLI, Linux foundation/core/engine/reproducibility, Mac core/Metal, approval/storage, real-model and installed Homebrew acceptance.
- [Supplemental approval/storage run](https://github.com/LayerDynamics/alloy/actions/runs/43072447906): Linux evidence passed. Its standalone Mac job was intentionally skipped because the complete Mac evidence is collected and required by the main run.
- [Real-model, latency, Phase 1 and Homebrew artifact](https://github.com/LayerDynamics/alloy/actions/runs/43072447930/artifacts/13880006792).
- W4 artifacts: [Linux](https://github.com/LayerDynamics/alloy/actions/runs/43072447930/artifacts/13880246011), [Mac](https://github.com/LayerDynamics/alloy/actions/runs/43072447930/artifacts/13879131457).
- W5 artifacts: [Linux](https://github.com/LayerDynamics/alloy/actions/runs/43072447930/artifacts/13880131107), [Mac](https://github.com/LayerDynamics/alloy/actions/runs/43072447930/artifacts/13877579288).
- [Mac approval/storage artifact](https://github.com/LayerDynamics/alloy/actions/runs/43072447930/artifacts/13878243478).

Mac tested the candidate SHA directly. Linux tested PR merge commit `a5b0fa9d5459d18b25b0955b13b1fb33ea312989`; both retained the identical tracked source tree `ad597e206b4ecc7ad9d27e72a2ea6d20451eab03`. The W4 source manifests match across platforms. W5 before/after working-directory snapshots match within each lane; those lane-specific working-directory hashes are not claimed to match each other.

Collectors retain automated evidence and deliberately do not grant acceptance themselves. Their `accepted: false` / `source_review_complete: false` defaults are preserved. This exit review supplies the explicit source/design and requirement sign-off, after inspection of the raw results and independent reviews; it does not rewrite collector output or claim an exhaustive security audit.

## Final requirement matrix

| Requirement | Result | Candidate evidence |
| --- | --- | --- |
| P2-01 registry/schema normalization | PASS, Linux + Mac | Tool-registry all-targets and named registration/validation gates |
| P2-02 arguments rejected before execution | PASS, Linux + Mac | Invalid/missing/extra/type-invalid arguments and unknown IDs have zero effects |
| P2-03 Qwen chunk boundaries | PASS, Linux + Mac | Qwen parser chunk/split regressions |
| P2-04 malformed tool calls never execute | PASS, Linux + Mac | Malformed, ambiguous and truncated parser/dispatch regressions |
| P2-05 exact rendering/count/identity | PASS, Linux + Mac | Engine tool-rendering and compatibility gates |
| P2-06 bound destructive approval | PASS, Linux + Mac | Named policy gate, durable approval/storage and real-model denial/approval |
| P2-07 builtin scope/output/deadline limits | PASS, Linux + Mac | W4 named gates and complete builtin suites |
| P2-08 durable oversized artifacts | PASS, Linux + Mac | Named artifact roundtrip and interruption/provenance regressions |
| P2-09 pause/cancellation/resource ownership | PASS, Linux + Mac | Named cancellation gate plus real RPC pause during generation, approval, shell and plugin hook; owned PID disappears, one terminal outcome, durable paused status, unaffected session history and subsequent turn |
| P2-10 real hook order/veto/annotations | PASS, Linux + Mac | W5 named gates and actual P2-15 correlated hook requests |
| P2-11 protocol/timeout/sandbox | PASS, Linux + Mac protocol; Mac sandbox | W5 negative cases, allowed/denied resources, fork/network/descriptor/artifact hardening |
| P2-12 canonical loop/history/reopen | PASS, Linux + Mac | Named loop gate and no-replay/reopen regressions |
| P2-13 responsive approval/control RPC | PASS, Linux + Mac | Seven tool-RPC tests including active-hook and active-shell pause |
| P2-14 CLI approval/result behavior | PASS, Linux + Mac, Node 26.13.0 | CLI and monitor suites on both platforms |
| P2-15 real deployed model tool loop | PASS, Mac/Metal | Candidate-built daemon and matching helpers; pinned Qwen3; retained RPC, history, marker and correlated hook evidence |
| P2-16 harness roundtrip under 50 ms | PASS, Mac | 32 recorded samples; every sample below 50 ms |

## Three exit statements

1. **Real governed tool use:** Qwen3 read the synthetic fixture, requested the shell tool, and waited for explicit bound approval. The marker was absent before either decision, remained absent after denial, and appeared after approval. Durable execution records show one approved shell start. The final answer consumed the file content and previously unseen shell output.
2. **Invalid input has zero effects:** parser, registry, validation, builtin and policy tests reject malformed or unauthorized calls without fallback execution; execution counters and filesystem/network markers establish the zero-effect assertions.
3. **Actual out-of-process hooks:** the deployed plugin received correlated before/after requests. Separate W5 gates prove ordering, veto, bounded annotations, protocol/version handling, limits, fail-closed before-hook behavior, after-hook warning behavior and Mac sandbox restrictions.

## Ownership and pause review

Shell targets run beneath shared isolated native owners. Linux pins its supervisor with a pidfd and returns the target's actual exit/signal status. Completion framing and launch acknowledgment reads remain nonblocking and cancellation-aware. Failed exec is distinguished from a program exiting 127.

On Mac, a private Mach/fileport broker exclusively reaps shell supervisors. The daemon never numerically waits for those supervisors or signals shell target/group IDs. Atomic close-on-exec fileport imports preserve descriptor isolation. Abandoned requests publish STOP before releasing their mapping, per-request reply ports prevent reply crossover, and the bootstrap receiver survives cancellation or a caller timeout. The supervisor retains its target unreaped until process-group cleanup is complete.

Regression coverage includes SIGCHLD auto-reaping, a real competing `waitpid(-1)` loop that collects unrelated child statuses, exact exit/signal status, exited-leader descendant cleanup, cancellation/bootstrap recovery, closed host standard descriptors, large argv, byte limits and cold standalone Mac builds without cached harness helpers. Generation completion clears cancellation identity before proposal handling. Pause tolerates only NotFound for the exact copied request ID and propagates unrelated errors.

The Mac broker is one host-lifetime infrastructure process. It drains its supervisors when the host exits, including an unreaped zombie host. Unexpected broker death fails closed until host restart; the host does not automatically respawn or numerically reap it. A fatal broker failure can retain that bounded process resource until host exit. This explicit abnormal-failure boundary is accepted; normal per-call cancellation does not disable other sessions or leave a target running.

Trusted commands can deliberately escape a process group with setsid/setpgid; arbitrary escaped-descendant containment requires a stronger OS sandbox. The reviewed less-trusted Mac plugin sandbox continues to deny subprocess creation. Trusted plugin subprocess support and the existing plugin reaper-ownership API remain intact.

## Model, latency and workflow measurements

- Model: `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, aggregate SHA-256 `95d369e2b999429fd8f02efdd9c0916718e653cb0a62d58e7f6a2770fee447a5`.
- Corpus SHA-256: `dc2eac48c5522fa53360855950eba4527dc6c21dc943f1d1c9282bd8535b2534`.
- P2-15 daemon SHA-256: `22375a5d1883cf630eeba0f93bb2c877a32cfd8bce8ef76bffd6c3a1fcddcbdc`; the retained artifact also records both helper identities and the verified clean candidate before/after build.
- P2-16: 32 samples, minimum **7.303 ms**, median **8.891 ms**, p95 **9.491 ms**, maximum **10.488 ms**, mean **8.691 ms**. Boundary: `run_tool_turn(mock_inference,fs.read,no_approval,no_client_transport)`; human approval and client transport are excluded. All samples are retained.
- Preserved Phase 1 real-model stream and restart/SSD-cache tests passed on the same implementation candidate. Installed Homebrew CPU/Metal and real-model acceptance passed.
- Main CI took **9m41s** from first job start to final job completion and **9m44s** from workflow creation, including queue time. This run meets the ten-minute target with one Mac runner; it is an observation, not a guarantee for future queues, cold caches or host contention.

## Historical failures and superseded evidence

Candidate `1b767a2285d0f2fe7473f74796924d37e85f2390` had green CI and real-model results, but did not establish shell ownership under a competing reaper or integrated pause during active hooks/processes. The retained SIGCHLD regression subsequently failed on Linux and Mac with lost child ownership. No actual PID-wrap collision or unrelated signal was observed. Those earlier results were not used to close the defect.

The shared-owner implementation replaced the direct-child observation-then-numeric-signal path rather than weakening the regression or changing a process-wide SIGCHLD disposition. Independent review then identified supervisor-status reuse, blocking partial frames, cwd-audit conflation, helper provisioning and cancellation-isolation issues. The final candidate fixes those issues and adds their corresponding tests. Earlier real-model timing and workflow observations remain historical; the measurements above identify the accepted implementation.

Issues [#31](https://github.com/LayerDynamics/alloy/issues/31) and [#34](https://github.com/LayerDynamics/alloy/issues/34) may close when PR #101's final documentation sign-off head also passes its required checks and merges. The W5 tracking issue [#32](https://github.com/LayerDynamics/alloy/issues/32) retains the integration evidence and documented boundaries.
