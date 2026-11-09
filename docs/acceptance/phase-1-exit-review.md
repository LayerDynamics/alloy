# Phase 1 exit review

**Decision: accepted 2026-11-06 for merged main `a4a17df5fb0c4d1ac734a98980139a5bc046352d`.** The final resource-error candidate passed fresh native, CLI and real-model acceptance. The merged Git tree exactly matches both tested candidates. [Issue #9](https://github.com/LayerDynamics/alloy/issues/9) records the final review; it and epic [#1](https://github.com/LayerDynamics/alloy/issues/1) are closed as completed. This document reconciles the earlier pending narrative with that decision and retains the failed-run history.

This review implements the exit mapping in the [Phase 1 execution plan](../specs/PHASE-1-EXECUTION-PLAN.md) and [SPEC-1 Phase 1](../specs/SPEC-1-mlx-agent-harness.md.md#phase-1-vertical-slice-mlx-c-ffi--first-family--harness-skeleton). The memory-parity pull-forward is governed by the [oMLX memory design](../designs/omlx-memory-method-parity.md). The next-phase draft is the [Phase 2 execution plan](../specs/PHASE-2-EXECUTION-PLAN.md).

## Candidate and evidence index

| Field | Value |
| --- | --- |
| Candidate branch | `feat/phase1-memory-closeout` |
| First tested implementation | `14b471598ad60b6906e24169113c27da0033c83f`; failed M9 preflight and subsequent repairs are recorded below. |
| Pull request | [#25 — native tensor ops, tiered KV storage, and Phase 1 acceptance](https://github.com/LayerDynamics/alloy/pull/25). |
| First candidate CI run | [42289710014](https://github.com/LayerDynamics/alloy/actions/runs/42289710014): Linux and CLI passed; Mac failed only the M9 large-model preflight. |
| First Mac acceptance artifact | [phase-1-acceptance-42289710014-1](https://github.com/LayerDynamics/alloy/actions/runs/42289710014/artifacts/13539817427); ZIP SHA-256 `77ec8a31bb30289fcfa7466038ec007f43fe11e5f8c9ef4a7c097bb7684a5bd8`. |
| First Linux native/infrastructure job | [128139616905](https://github.com/LayerDynamics/alloy/actions/runs/42289710014/job/128139616905), passed. |
| First CLI job | [128139617362](https://github.com/LayerDynamics/alloy/actions/runs/42289710014/job/128139617362), passed, including all 20 acceptance-helper tests. |
| Earlier accepted repair candidate | `75a960cad88062bb3b5eae6484ef706dfc9a61c5`; [push run 42299723238](https://github.com/LayerDynamics/alloy/actions/runs/42299723238) and [PR run 42299729975](https://github.com/LayerDynamics/alloy/actions/runs/42299729975), all jobs completed successfully. |
| Earlier repair Mac artifact | [phase-1-acceptance-42299723238-1](https://github.com/LayerDynamics/alloy/actions/runs/42299723238/artifacts/13543413413); downloaded and verified ZIP SHA-256 `8f26857e14512b6346d79e5ca921d80f9ee85743a6a3fcc14e6222a50c3d30ee`. |
| Final tested source head | `bf95b93fe92f6ce0cdfc21ac8fecc0982ecc1b43`; [push run 42305914377](https://github.com/LayerDynamics/alloy/actions/runs/42305914377), all three jobs successful. |
| Final tested PR merge candidate | `0d005c9e712074ccaf279d67bc330d6a49cd3a7f`; [PR run 42305920807](https://github.com/LayerDynamics/alloy/actions/runs/42305920807), all three jobs successful. |
| Merged main and source equality | `a4a17df5fb0c4d1ac734a98980139a5bc046352d`; its Git tree `bca536fa8199d717d495d304c5840d5c13bc7d41` equals both final tested trees. |
| Final inspected Mac artifact | [phase-1-acceptance-42305914377-1](https://github.com/LayerDynamics/alloy/actions/runs/42305914377/artifacts/13546233143); ZIP SHA-256 `57f1431a5cdeaaf0641341692fbf751a06f397ac4ca73e67a71904e33cda8e2a`. |
| Two post-freeze reproducible native builds | **Passed:** [run 42276463422](https://github.com/LayerDynamics/alloy/actions/runs/42276463422/job/128093939290) at `5b3b08dd219763e14c2c097c227ed71857903c54`, and [run 42279057333](https://github.com/LayerDynamics/alloy/actions/runs/42279057333/job/128102908050) at `a76d97bd2d91ffb75074fea9641d9b38992b5da8`. Logs inspected 2026-11-06. |
| Reviewer and date | ChatGPT, 2026-11-06: final implementation, requirements, source-tree and retained-artifact review; approved in [issue #9](https://github.com/LayerDynamics/alloy/issues/9). |

A prior branch's binary, fixture-only success, skipped model test, or an artifact from unrelated source cannot satisfy a real-model row. The final push and PR runs completed before approval; source-tree equality explicitly binds them to the merge. The separate post-merge main run was still in progress at review time and is not claimed as passing evidence here. Failed artifacts remain linked separately.

## Required exit gates

| Gate | Required evidence | Status |
| --- | --- | --- |
| Quantized Qwen3 streams through `alloy chat` | Test 13 `qwen3_end_to_end_stream`: current-checkout build; verified corpus artifact; API and owned-PID readiness; at least two nonempty `text_delta` frames and two visible CLI stdout chunks before one terminal event; final result and CLI stdout equal the streamed text. | **Passed:** 12 deltas and 12 visible chunks on the final candidate. |
| Session survives a real process restart | Test 14 `session_restart_real_model`: distinct owned daemon PIDs using the same SQLite DB; exact prior event prefix, user content, and assistant content recovered; resumed CLI generation recalls the stored marker and appends a new turn. | **Passed:** PIDs 71272 → 71275; all 5 prior events preserved, 9 after resume. |
| Cold SSD KV survives restart | Test 14 before/after cache snapshots: forced zero hot-cache limit; completed spill and native hot/protected ownership drained before stop; the 64 MiB transfer allowance remains charged in the live admission ledger; new process reports positive restore count and restored tokens after resumed generation; no cache error. | **Passed:** 37 native KV tokens restored in the new process. |
| Durable event history and pause/cancel | Phase 1 tests 5, 6, 7, 8 and their current harness/daemon regression suites; no changed or missing earlier event content. | **Passed** in final Linux/Mac suites and real restart evidence. |
| Engine contract documented/versioned | Test 15: engine rustdoc with `-D warnings`, documented public traits, explicit API version, and engine checks/tests. | **Passed** on the final candidate. |
| Vendored native build is reproducible | Phase 0 test 11: two successful post-freeze runs with recorded pins/corpus identity. This is distinct from Phase 1 test 11, `admission_estimate`. | **Passed** in the two linked post-freeze runs; unchanged vendor pins. |
| Telemetry ships with its privacy/schema contract | Phase 1 tests 9 and 10; schema/anonymity, bounded flush, offline drop, and failure-isolation tests; published event schema and privacy policy before any tagged emitting pre-release. | **Phase 1 passed:** named tests and worker/deadline regressions passed. Hosted endpoint and published privacy policy remain release prerequisites. |
| Live memory guard and residency/cache pull-forward | M1–M10 named acceptance from the [memory design](../designs/omlx-memory-method-parity.md), including live native counters and protected-resource behavior. | **Passed:** native CPU/Metal, exact protected ownership, emergency cancellation, and hot/SSD KV gates. |
| Large pinned model and clean admission rejection | `scripts/large_model_acceptance.sh`: candidate-bound pinned large-model load/generation, measured memory evidence, and separately demonstrated bounded rejection under an insufficient budget. | **Passed:** real 4B/14B Metal execution, typed protected refusal, separate `ReservationBudget` refusal, and empty shutdown ledger. |
| Existing native and CLI regression gates | All required jobs in [ci.yaml](../../.github/workflows/ci.yaml); no gate silently skipped due to absent model files. | **Passed** in final push and PR runs. |

## First candidate results and required repairs

The [Mac job](https://github.com/LayerDynamics/alloy/actions/runs/42289710014/job/128139617418) ran on an **Apple M4 Max, Mac16,6, 36 GiB unified RAM**, macOS 26.4/build 25E246. Its source and build provenance both identify `14b471598ad60b6906e24169113c27da0033c83f`; the real daemon binary SHA-256 is `1d313831ab12521feb148b9f772c2570d11dd005b75d03e23c7006eb1df90e24`.

| Observed gate | Recorded result at the first candidate |
| --- | --- |
| Native Mac suites | MLX CPU/Metal/shared-array, infrastructure, Qwen3, engine, harness-core, and daemon suites passed. Infrastructure includes 23 native KV tests, 13 existing kernel tests, and 8 native Tensor tests. |
| Test 13 | Passed: 12 nonempty deltas, 12 visible stdout chunks before the terminal result, 39 input tokens, 13 output tokens, one successful EOS result. |
| Test 14 history | Passed: daemon PID 6671 stopped, new PID 6709; all 5 prior events preserved; 9 events after the resumed turn. |
| Test 14 native KV | Passed: one spill before restart; one restore and 37 restored tokens afterward; zero hot/protected reservations at both completion observations, 67,108,864-byte transfer allowance retained, no cache error. |
| Frozen models | Both 4B and 14B files and aggregate hashes verified. This establishes model identity, not large-model execution. |
| M9 large-model execution | **Failed before any native load.** The 14B startup reservation was 33,813,526,874 bytes; balanced ceiling was 15,470,674,688 bytes; reported Metal cap was 30,064,771,072 bytes. Native model handle count remained zero. |

The first M9 run exposed unnecessary dense vocabulary residency and an over-large loading peak. The original loader held all checkpoint shards during host decoding; native upload then expanded vocabulary embeddings/output weights to float32. The plan also summed source, complete decoded weights, and complete native weights as though all three phases overlapped. Follow-up work keeps vocabulary matrices packed, streams and hashes bounded source chunks, and derives the loading allowance from actual ownership phases. The guard and the required 4B/14B execution test remained enforced. The later successful hardware runs below satisfy M9; the original preflight rejection did not.

The requirements review also found synchronous end-of-turn telemetry without an idle schedule or explicit network deadline. The follow-up isolates flushes on one bounded worker, adds scheduled flushing and finite HTTPS I/O deadlines, and preserves offline-drop semantics; see the [telemetry design](../designs/telemetry.md). Two direct M10 scenarios now cover positive-cap RAM reuse with SSD disabled and real second-block/manifest write failures after staging begins. Both pass locally, including cleanup, preserved native ownership, and successful recovery.

Local follow-up checks pass for all MLX FFI, inference-infrastructure, and Qwen targets, 423 engine tests, and all 16 harness tests. The loader tests cover bounded body buffering, cross-shard F16/BF16/U32 conversion, exact budgets, full-file digests, and cancellation. Telemetry coverage includes a real stalled loopback TLS connection and two persisted turns while its transport remains blocked. Local daemon compilation and the two live-metrics tests pass; four Unix-socket RPC cases encounter this workspace's `EPERM` bind restriction and remain mandatory in CI. These local results do not substitute for the new candidate's Mac CPU/Metal and real 4B/14B acceptance.

The Phase 2 plan was reviewed against SPEC-1 Phase 2, FR-20/21/24, the builtin policies, and B.2/B.4/B.6. Its seven work packages, exact file mapping, 16 named tests, and three exit mappings cover the defined scope. Its Phase 1 entry gate is now satisfied. A published privacy policy and an operational collection endpoint are release prerequisites; this review does not claim either from a mock transport test or the current `telemetry.invalid` development default.

## Loader repair and protected-pin acceptance correction

The [repair run at `d09fdd1ff0900633eb800b50d97bab1484eae3cf`](https://github.com/LayerDynamics/alloy/actions/runs/42296372344) passed Linux, CLI, all Mac native/engine/harness/daemon suites, real streaming, and real daemon history/SSD-KV restart. Linux logs record 575 passing top-level Rust tests and two pre-existing process-global FFI tests ignored. The [Mac artifact](https://github.com/LayerDynamics/alloy/actions/runs/42296372344/artifacts/13542229011) has ZIP SHA-256 `9f8566005cab53d6454913a123076177b874595ee66e3216191e9732f557708b`.

The repaired 14B startup plan is **14,104,910,369 bytes**, below the freshly measured **15,682,454,248-byte** balanced ceiling. Real 4B Metal generation completed with 28 input and 8 output tokens. Its retained weight reservation was 2,660,373,249 bytes. M9 then stopped at an acceptance-driver classification mismatch, before 14B loading: the configured startup pressure policy correctly returned `Busy` with `ProtectedResources` and no active request, while the driver incorrectly required `MemoryLimitExceeded` for that pinned-owner case. Artifact sequence 62 records the exact cause; A remained pinned and charged, the guard reported acceptable pressure/open admission, and B's loader was never entered.

At `75a960c`, the corrected driver retained the full error and required that specific protected-pin rejection. Generic scheduler Busy, paused admission, and unrelated memory failures could not satisfy it. It separately held the ledger's budget before any model was loaded and required a real `MemoryLimitExceeded` response, no native loader entry, no leaked scheduler/grant ownership, and release of that explicit test reservation. This budget hold was not represented as a physical memory allocation. The subsequent pinned/unpinned sequence required real 4B/14B generation, A's actual destruction before B entered its loader, and an empty shutdown ledger. The successful hardware result is recorded below.

The independent full-budget regression exposed a production observation bug: an empty model pool was marked protected solely because it had no eligible eviction victim. The runtime was changed to require an actual resident model before reporting protected model ownership. At `75a960c`, genuine pins still produced the historical protected-owner Busy response; an empty runtime with an exhausted ledger produced `MemoryLimitExceeded`. Both rejection paths were checked for native ownership, reservation cleanup, and successful recovery. The follow-up deliberately replaces that overloaded Busy classification as described below.

## Completed hardware acceptance at `75a960c`

The [Mac job](https://github.com/LayerDynamics/alloy/actions/runs/42299723238/job/128174049467) and its downloaded artifact identify the exact clean checkout and compiled commit `75a960cad88062bb3b5eae6484ef706dfc9a61c5`. The machine remains the Apple M4 Max, Mac16,6, with 36 GiB unified RAM. All native CPU/Metal/shared-memory, engine, harness, and daemon suites passed. The real daemon binary SHA-256 was `a9a0cfb98d60c8b2b4532684ad823220fb51824dfeb5e2a0af68b39f0ad3cc18`.

| Observed gate | Recorded result at `75a960c` |
| --- | --- |
| Test 13 incremental streaming | 12 nonempty deltas and 12 visible stdout chunks before one successful terminal result; 39 input tokens and 13 output tokens. |
| Test 14 durable history | PID 19839 stopped and PID 19864 restarted; all 5 prior events recovered, with 9 total events after resumed generation. |
| Test 14 actual SSD KV | One successful restore and 37 restored tokens in the new process; zero hot/protected KV at completion, 67,108,864 transfer bytes still reserved, and no cache error. |
| M9 host and plan | Planned 14B startup 14,104,910,369 bytes; freshly measured balanced ceiling 16,045,425,384 bytes. Both frozen artifacts verified. |
| M9 protected refusal | The original pinned 4B instance and its 2,660,373,249-byte weight reservation survived; the 14B loader was not entered. |
| M9 exhausted ledger | A separate full-budget hold produced the expected refusal, no loader call, no leaked grant/scheduler ownership, and successful recovery after releasing the hold. |
| M9 real replacement | Both models performed real Metal prefill/decode. A's actual destruction at sequence 77 preceded B's loader entry at sequence 81; maximum native model handles was one. |
| M9 measured native memory | MLX reported peak 9,371,885,656 bytes, active 9,233,276,928 bytes at the B-idle observation, and 268,457,540 cached bytes. The peak is a native allocator measurement, distinct from the 14,104,910,369-byte startup plan. |
| M9 shutdown | Zero native model handles, zero reserved bytes, zero active reservations, and zero MLX active bytes. Allocator cache remained separately reported. Final sequence 108 was `acceptance_passed`. |

These results resolve the original preflight failure and subsequent driver classification failure for `75a960c`. The later resource-error implementation has its own fresh evidence below.

## Typed resource-error follow-up

The [resource-error design](../designs/engine-resource-errors.md) defines the merged production contract. `Busy` denotes active/pending execution or scheduler ownership. `ProtectedResources` carries operation-scoped model pins and session-state leases with exact artifact/instance/state identities and positive observed handle counts. A retained diagnostic does not retain a protection handle. `MemoryLimitExceeded` records a typed reason, including the difference between unavailable capacity and a pressure-driven admission pause.

The M9 preflight emits an actual `HostPreflight` engine cause with the planned startup and complete observed ceiling breakdown in failure evidence. Its pin assertion matches the original instance and typed pin directly; its full-ledger assertion specifically requires `ReservationBudget`. Session RPC responses and streamed failures preserve the captured engine cause, and CLI handling reports one terminal failure. Fresh engine, daemon, CLI, and real-model acceptance for these changes passed at `bf95b93` and the equivalent PR merge tree.

## Final acceptance at `bf95b93`, merged as `a4a17df`

The final [Mac job](https://github.com/LayerDynamics/alloy/actions/runs/42305914377/job/128195038977), [Linux job](https://github.com/LayerDynamics/alloy/actions/runs/42305914377/job/128195039176), and [CLI job](https://github.com/LayerDynamics/alloy/actions/runs/42305914377/job/128195039159) all completed successfully. The independent [PR run](https://github.com/LayerDynamics/alloy/actions/runs/42305920807) passed on an equal source tree. The inspected push artifact is identified in the evidence index; the values below come from that artifact, not from the earlier `75a960c` run.

| Final observation | Recorded result |
| --- | --- |
| Hardware | Apple M4 Max, Mac16,6, 36 GiB unified RAM, macOS 26.4/build 25E246. |
| Test 13 | 12 nonempty deltas and 12 visible stdout chunks before one successful terminal outcome. |
| Test 14 | PIDs 71272 → 71275; all 5 prior events and contents preserved, 9 total after resume; 37 actual native KV tokens restored. |
| M9 budget | 14,104,910,369-byte startup/shared budget below the freshly measured 24,349,491,944-byte safe ceiling. |
| Separate capacity refusal | Sequence 39 requires `MemoryLimitExceeded` with `ReservationBudget`, zero native loader calls, idle scheduler, released probe hold and zero leaked reservations. The ledger hold is not represented as physical allocation. |
| Protected ownership | Sequence 69 records exact `ProtectedResources` for the original pinned 4B instance; its 2,660,373,249-byte reservation and identity survive, with zero B loader calls. |
| Real model replacement | Both models perform real Metal prefill/decode. A's actual destruction at sequence 75 precedes B's loader at sequence 80; maximum native model handles is one. |
| Native memory and shutdown | MLX peak 9,371,885,656 bytes; zero native handles, active bytes, reserved bytes and active reservations at shutdown. Final sequence 106 is `acceptance_passed`. |
| Native regressions | Mac passes unified CPU/Metal/shared-array gates, 443 engine tests, 17 harness tests and 10 daemon integration tests, including required pooled-KV pressure coverage. |
| Linux and documentation | 597 top-level Rust passes, zero failures; two existing process-global MLX tests intentionally ignored. `engine_traits_documented`, rustdoc `-D warnings`, named telemetry tests, and `vendored_build_reproducible` pass. |

The final push and PR Linux jobs each also perform two matching clean native builds. All named Phase 1 tests 1–15, the memory pull-forward gates, and the written Phase 2 plan were reviewed before approval. This closes Phase 1 runtime acceptance; Homebrew distribution and the remaining release prerequisites are separate work.

## Existing reproducibility evidence

Both linked Linux jobs report `ok: vendored_build_reproducible` after two
clean builds each. Their installed-library SHA-256 manifests agree:

| Library | SHA-256 in both builds of both runs |
| --- | --- |
| `libmlx.a` | `f0a599bb10c14a7820bf9825f0510aacea3f473a88a195065ec7f85dc17ff0ad` |
| `libmlxc.a` | `1e3c31ce28ca597daad04dc26b11e3dc9fe4d8169c2e39fe8c14a570c0a8824e` |

The corpus freeze is `06e086b909cfee9a8d74f0c2d7b47c0348811060`. Both
runs are later than that freeze. The current change preserves the MLX core
pin `36969c094354086e15d2a327a9c50feb20b9cf11` and mlx-c pin
`a26f2d584a14ec619ae8db9a09be03becd47598a`; see [PINS.md](../../PINS.md).
These results satisfy the two-run reproducibility requirement. They do not
stand in for new native-operation, SSD-KV, or real-model acceptance results.

## Reproduce tests 13 and 14

Use the repository's pinned **Node 26.13.0** on the Apple Silicon macOS runner. The scripts invoke Cargo against this checkout's manifest and use the `alloyd` executable reported by that build, including when `CARGO_TARGET_DIR` is configured. `ALLOY_ENGINE_ROOT`, if set, must resolve to this same checkout. The scripts do not select another worktree. Tracked, untracked, and submodule source must be clean before and after building. Optional `ALLOY_E2E_EXPECTED_COMMIT` must exactly match the full checkout HEAD. Put custom evidence output outside the checkout or in an ignored directory.

The 4B artifact is a pre-provisioned flat directory containing every file pinned under `qwen3-4b-4bit` in [models.json](../../bench/corpus/models.json):

| Identity | Frozen value |
| --- | --- |
| Repository | `mlx-community/Qwen3-4B-4bit` |
| Revision | `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` |
| Aggregate SHA-256 | `95d369e2b999429fd8f02efdd9c0916718e653cb0a62d58e7f6a2770fee447a5` |

Set `ALLOY_QWEN3_ARTIFACT` to the local directory. Without an override, the scripts use `alloy-omlx-models/alloy-qwen3-4b-4bit` under the current user's home directory. They never modify `HOME`, download model files, or skip an absent artifact. Actual daemon boot verifies the complete frozen corpus digest before the API can become ready; a present `config.json` alone is not acceptance.

```bash
ALLOY_QWEN3_ARTIFACT=/absolute/path/to/alloy-qwen3-4b-4bit \
ALLOY_ACCEPTANCE_OUTPUT_DIR=/absolute/path/to/phase-1-evidence \
  bash ts/cli/scripts/qwen3_end_to_end_stream.sh

ALLOY_QWEN3_ARTIFACT=/absolute/path/to/alloy-qwen3-4b-4bit \
ALLOY_ACCEPTANCE_OUTPUT_DIR=/absolute/path/to/phase-1-evidence \
  bash ts/cli/scripts/session_restart_real_model.sh
```

Each invocation creates a unique retained subdirectory. The default base is `target/phase-1-acceptance`. Temporary Unix sockets use a short isolated directory; cleanup signals only the isolated process groups created by this invocation (including their Cargo/CLI descendants) and removes only that temporary socket directory. Daemon state, raw CLI output, wire records, and error diagnostics remain available after failure. Test 14 uses an isolated `kv-cache` directory and `--hot-cache-max-size 0` to require disk restore rather than retaining idle KV in RAM. It checks that native hot/protected KV reservations drain to zero. The storage owner's separately admitted **64 MiB transfer allowance remains reserved for the daemon's lifetime**, including while idle, so persistence can run while new admissions are paused. The test requires that allowance in the live admission ledger; it is released when the runtime/daemon drops.

| Timeout control | Default | Applies to |
| --- | --- | --- |
| `ALLOY_E2E_BUILD_TIMEOUT_MS` | `1200000` | Each candidate/CLI build command. |
| `ALLOY_E2E_START_TIMEOUT_MS` | `240000` | Each daemon's bounded API/PID readiness loop. |
| `ALLOY_E2E_TURN_TIMEOUT_MS` | `180000` | Each real CLI turn. |
| `ALLOY_E2E_CACHE_TIMEOUT_MS` | `30000` | Cache ownership/persistence completion checks. |

Timeouts are positive integers, at most `3600000` ms. A timeout fails the test and retains its evidence. Increase one only after recording why the default was insufficient on the target hardware.

### Files to inspect in the artifact

| Evidence | What it proves |
| --- | --- |
| `result.json` | Named test, pass/fail/error, timestamps, candidate/model/binary provenance, summarized assertions, final live metrics. |
| `candidate.json`, `candidate-before-build.json`, `candidate-after-build.json`, their `*-revision.stdout.log` / `*-status.stdout.log` files, and `model-pin.json` | Checkout commit/status, current build output, binary SHA-256, corpus manifest SHA-256 and frozen model identity. |
| `build.stdout.log`, `build.stderr.log`, `cli-build.stdout.log`, `cli-build.stderr.log` | Candidate Cargo compiler-artifact output and CLI build diagnostics. |
| `processes.ndjson`, `daemon-*.ready.json`, `daemon-*.readiness.ndjson`, `daemon-*.stdout.log`, `daemon-*.stderr.log` | Owned PID/command identity; API/metrics readiness; load and validation errors. |
| `*.rpc.ndjson`, `cli-*.stdout-chunks.ndjson`, `cli-*.stdout.log`, `cli-*.stderr.log`, `*.turn.json` | Timestamped request/response/notification records from the real CLI connection, incremental text and terminal order, stdout consistency. |
| `history-before.json`, `history-recovered.json`, `history-after.json`, `sessions.sqlite` | Exact durable conversation content before stop, after process restart, and after the resumed turn. |
| `before.cache.json`, `after.cache.json`, `kv-cache/`, `checks.ndjson` | Real spill/restore metrics, persisted cache state, and read-only verification RPC results. |

The automatic Mac job runs both named tests and the M9 large-model driver after native checks. Native KV transfer unit tests run on both Linux and macOS; the Mac tests include CPU/Metal transfer coverage. The job retains `target/phase-1-acceptance/` with `actions/upload-artifact@v4` under `always()`. The optional repository variable `ALLOY_QWEN3_ARTIFACT` selects the runner's pre-provisioned 4B path; `ALLOY_M9_MODEL_B_DIR` selects its pre-provisioned 14B path. Missing or mismatched files are actionable failures, not skipped acceptance. Wire/history logs contain only the fixed synthetic acceptance prompts and generated outputs; do not replace them with private user conversations.

## Reproduce the M9 large-model gate

Run [large_model_acceptance.sh](../../scripts/large_model_acceptance.sh) from a clean committed candidate checkout on native Apple Silicon. It validates both distinct frozen profiles (`qwen3-4b-4bit` and `qwen3-14b-4bit`) against [models.json](../../bench/corpus/models.json), records the exact compiled checkout/commit, and requires real Metal prefill/decode. A low-memory preflight or missing model fails; the driver does not substitute a tiny or CPU-only model.

```bash
mkdir -p /absolute/path/to/phase-1-evidence
ALLOY_M9_MODEL_A_DIR=/absolute/path/to/alloy-qwen3-4b-4bit \
ALLOY_M9_MODEL_B_DIR=/absolute/path/to/alloy-qwen3-14b-4bit \
ALLOY_M9_OUTPUT=/absolute/path/to/phase-1-evidence/m9-acceptance.jsonl \
  bash scripts/large_model_acceptance.sh
```

`ALLOY_M9_OUTPUT` must name a new file; existing evidence is not overwritten. `ALLOY_M9_EXPECTED_COMMIT`, when supplied, must exactly equal the checkout's full HEAD SHA. `ALLOY_M9_TIMEOUT_SECONDS` defaults to 1800 with a 120-second cleanup grace; its supported range is 60–7200. CI stores the machine-readable `acceptance.jsonl` and the complete `driver.log` together in a unique `m9-large-model-*` artifact subdirectory, including early preflight/build failures. Require the final `acceptance_passed` record and inspect the measured peaks, admission budgets, and rejection records; exit status alone is not the full review evidence.

## Review record

1. **Observed results:** All required gates passed for `bf95b93` and the equivalent PR merge tree. The final acceptance section records actual streaming, process/history, restored KV, protected ownership, memory and shutdown observations; the evidence index identifies the downloaded artifact and successful jobs.
2. **Failures and repairs:** The `14b4715` preflight failure, `d09fdd1` driver classification failure, loader repair, typed resource-error follow-up and successful reruns remain recorded above. No failed run is relabeled as passing.
3. **Outstanding Phase 1 requirements:** None identified in the final review. Homebrew distribution, a published telemetry privacy policy, an operational collection endpoint and Phase 2 implementation retain their separate scope; this approval is not a public-release claim.
4. **Decision:** Approved 2026-11-06 for merged main `a4a17df` after source-tree equality and all required evidence were verified. [Issue #9](https://github.com/LayerDynamics/alloy/issues/9) and [epic #1](https://github.com/LayerDynamics/alloy/issues/1) are closed as completed. The Phase 2 plan's entry gate is satisfied.
