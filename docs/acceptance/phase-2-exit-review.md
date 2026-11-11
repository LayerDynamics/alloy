# Phase 2 exit review

**Status: pending final candidate CI and review.** Passing local tests are
preliminary evidence; they do not close Phase 2. The final decision, candidate
SHA, CI run and artifact links belong on [issue #34](https://github.com/LayerDynamics/alloy/issues/34).

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
