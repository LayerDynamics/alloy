# Phase 2 exit review

**Review record.** Candidate `1b767a2285d0f2fe7473f74796924d37e85f2390`
passed the complete [Linux and Mac CI run](https://github.com/LayerDynamics/alloy/actions/runs/43018696590)
and the [supplemental approval/storage run](https://github.com/LayerDynamics/alloy/actions/runs/43018696710).
The final reviewed candidate and closure decision are recorded on
[issue #34](https://github.com/LayerDynamics/alloy/issues/34). A later code or
test change requires fresh applicable acceptance evidence.

## Required evidence

| Requirement | Evidence needed on the final candidate |
| --- | --- |
| Real model reads a file and requests a policy-governed shell tool; execution waits for bound approval | P2-06, P2-12–15. The Mac `phase2_tool_loop_real_model` driver records the candidate, pinned model, daemon identity, RPC stream, durable history, denial and approval runs. It checks that the marker is absent before either decision, absent after denial, and present after approval. The model must consume file and shell output. |
| Malformed calls and invalid arguments never execute | P2-02–04 and P2-07 on Linux and Mac, including the zero-effect assertions. |
| Real plugin receives before/after hooks and obeys veto, annotation, ordering and sandbox contracts | P2-10–11 on Linux and Mac and P2-15 on Mac. The real-model driver retains the plugin's ordered hook log; the negative and sandbox tests remain required separately. |
| Supporting invariants | P2-01 and P2-05–14, including rendering, durable approval and artifacts, cancellation, history, RPC responsiveness and CLI behavior. Preserve the existing Phase 1 gates. |
| Harness overhead | P2-16 on Mac, with mocked inference and no approval wait or client transport. The test logs the count, all samples, min, median, p95, max, mean and timed boundary; every sampled round must be below 50 ms. |

The automatic Mac lane retains the Phase 2 output alongside the Phase 1
acceptance artifact under `always()`. Acceptance must name the exact candidate,
corpus hash, model revision and artifact digest recorded by that lane. A skipped
or failed named test cannot be replaced by a local mock or by W5 acceptance.

## Candidate evidence and exit statements

- P2-01 through P2-14 passed in the full Linux and Mac suites. The retained
  W5 Linux and Mac artifacts each contain passing P2-10/P2-11 results and
  identical before/after source snapshots. Their checkout commits differ
  because the Linux lane tested the PR merge commit, but their source-tree
  identities match the Mac candidate tree.
- The retained Mac `phase-1-acceptance` artifact contains P2-15's candidate and
  binary/helper identities, pinned corpus, RPC/history, denial/approval
  observations, and actual ordered before/after plugin requests. Before both
  decisions the marker was absent; it remained absent after denial and appeared
  after approval. The final answer used both the read result and the unseen
  shell output. These observations satisfy the real-model exit statement and
  supply integrated plugin evidence.
- Invalid-call and zero-effect gates P2-02–04/07 passed in the full suites;
  plugin veto, annotation, ordering and sandbox gates P2-10/11 passed on their
  required lanes. Together with P2-15 they support the other two exit statements.
- P2-16 recorded 32 samples: 6.773 ms minimum, 8.118 ms median, 9.016 ms p95,
  and 12.573 ms maximum. The boundary is `run_tool_turn` with mocked inference
  and a real scoped file read; human approval and client transport are excluded.
- The pinned model is `mlx-community/Qwen3-4B-4bit` at revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, aggregate SHA-256
  `95d369e2b999429fd8f02efdd9c0916718e653cb0a62d58e7f6a2770fee447a5`;
  the corpus file SHA-256 is
  `dc2eac48c5522fa53360855950eba4527dc6c21dc943f1d1c9282bd8535b2534`.
- The main workflow took 8m56s from the first job start to the last aggregate,
  and 10m27s from workflow creation through completion including queue time.
  The latter exceeds ten minutes; this run does not establish a queue-inclusive
  ten-minute guarantee. One self-hosted Mac serializes the Mac lanes.

## Preliminary observations

On 2026-11-11, local candidate `d8bb02c806397c9aab15ccaa278a860bc88e2bbb`
passed the real-model acceptance, both Phase 1 real-model streaming/restart
checks, and the standalone latency test on macOS 26.4. The 32 latency samples
had median 7.919 ms and maximum 15.540 ms. This local candidate is historical
if the source or test tree changes; final candidate CI must be reviewed again.

The first deployed run exposed a missing tool-parser selection and later a
workspace admission failure. The candidate enables explicitly bounded parser
allocation only for configured tools and uses a 64-token prefill chunk for
those daemons, keeping the memory admission guard enabled. Text-only boot keeps
its prior parser and prefill configuration. These changes need final CI and
regression review, including the existing native memory and real-model gates.

## W5 boundaries carried forward

[PR #100](https://github.com/LayerDynamics/alloy/pull/100) records W5 component
acceptance and its constraints: trusted descendants deliberately leaving their
original group and unexpected supervisor death are outside the cleanup
contract; Linux requires the stated kernel/procfs support; macOS private
identity interfaces were qualified on the tested macOS 26.4 host. The full list
and platform evidence are in that PR and the plugin security review. Do not
infer broader sandbox or platform certification from this exit review.
