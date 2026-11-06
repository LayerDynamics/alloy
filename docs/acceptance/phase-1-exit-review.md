# Phase 1 exit review

**Decision: pending.** The candidate has acceptance tooling and an automatic macOS lane. Phase 1 is not signed off until the required results below are linked and reviewed for the candidate commit.

This review implements the exit mapping in the [Phase 1 execution plan](../specs/PHASE-1-EXECUTION-PLAN.md) and [SPEC-1 Phase 1](../specs/SPEC-1-mlx-agent-harness.md.md#phase-1-vertical-slice-mlx-c-ffi--first-family--harness-skeleton). The memory-parity pull-forward is governed by the [oMLX memory design](../designs/omlx-memory-method-parity.md). The next-phase draft is the [Phase 2 execution plan](../specs/PHASE-2-EXECUTION-PLAN.md).

## Candidate and evidence index

| Field | Value |
| --- | --- |
| Candidate branch | `feat/phase1-memory-closeout` |
| Candidate commit SHA | Pending — fill from the reviewed CI run and each `candidate.json`. |
| Pull request | Pending — link the single closeout PR after publication. |
| Latest CI run | Pending — link the run for the reviewed candidate. |
| Latest Mac acceptance artifact | Pending — link `phase-1-acceptance-<run_id>-<run_attempt>` from that run. |
| Linux native/infrastructure job | Pending — link its successful job. |
| CLI job | Pending — link its successful job. |
| Two post-freeze reproducible native builds | **Passed:** [run 42276463422](https://github.com/LayerDynamics/alloy/actions/runs/42276463422/job/128093939290) at `5b3b08dd219763e14c2c097c227ed71857903c54`, and [run 42279057333](https://github.com/LayerDynamics/alloy/actions/runs/42279057333/job/128102908050) at `a76d97bd2d91ffb75074fea9641d9b38992b5da8`. Logs inspected 2026-11-06. |
| Reviewer and date | Pending. |

A prior branch's binary, fixture-only success, skipped model test, or an artifact from a different commit cannot satisfy a real-model row. Preserve a failed run's artifact and link the succeeding rerun separately. Do not overwrite the evidence index with a claimed pass before the job completes.

## Required exit gates

| Gate | Required evidence | Status |
| --- | --- | --- |
| Quantized Qwen3 streams through `alloy chat` | Test 13 `qwen3_end_to_end_stream`: current-checkout build; verified corpus artifact; API and owned-PID readiness; at least two nonempty `text_delta` frames and two visible CLI stdout chunks before one terminal event; final result and CLI stdout equal the streamed text. | Pending |
| Session survives a real process restart | Test 14 `session_restart_real_model`: distinct owned daemon PIDs using the same SQLite DB; exact prior event prefix, user content, and assistant content recovered; resumed CLI generation recalls the stored marker and appends a new turn. | Pending |
| Cold SSD KV survives restart | Test 14 before/after cache snapshots: forced zero hot-cache limit; completed spill and native hot/protected ownership drained before stop; the 64 MiB transfer allowance remains charged in the live admission ledger; new process reports positive restore count and restored tokens after resumed generation; no cache error. | Pending |
| Durable event history and pause/cancel | Phase 1 tests 5, 6, 7, 8 and their current harness/daemon regression suites; no changed or missing earlier event content. | Pending |
| Engine contract documented/versioned | Test 15: engine rustdoc with `-D warnings`, documented public traits, explicit API version, and engine checks/tests. | Pending |
| Vendored native build is reproducible | Phase 0 test 11: two successful post-freeze runs with recorded pins/corpus identity. This is distinct from Phase 1 test 11, `admission_estimate`. | **Passed** in the two linked post-freeze runs; unchanged vendor pins. |
| Telemetry ships with its privacy/schema contract | Phase 1 tests 9 and 10; schema/anonymity, bounded flush, offline drop, and failure-isolation tests; published event schema and privacy policy before any tagged emitting pre-release. | Pending |
| Live memory guard and residency/cache pull-forward | M1–M10 named acceptance from the [memory design](../designs/omlx-memory-method-parity.md), including live native counters and protected-resource behavior. | Pending |
| Large pinned model and clean admission rejection | `scripts/m9_large_model_acceptance.sh`: candidate-bound pinned large-model load/generation, measured memory evidence, and separately demonstrated bounded rejection under an insufficient budget. | Pending |
| Existing native and CLI regression gates | All required jobs in [ci.yaml](../../.github/workflows/ci.yaml); no gate silently skipped due to absent model files. | Pending |

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

Run [m9_large_model_acceptance.sh](../../scripts/m9_large_model_acceptance.sh) from a clean committed candidate checkout on native Apple Silicon. It validates both distinct frozen profiles (`qwen3-4b-4bit` and `qwen3-14b-4bit`) against [models.json](../../bench/corpus/models.json), records the exact compiled checkout/commit, and requires real Metal prefill/decode. A low-memory preflight or missing model fails; the driver does not substitute a tiny or CPU-only model.

```bash
mkdir -p /absolute/path/to/phase-1-evidence
ALLOY_M9_MODEL_A_DIR=/absolute/path/to/alloy-qwen3-4b-4bit \
ALLOY_M9_MODEL_B_DIR=/absolute/path/to/alloy-qwen3-14b-4bit \
ALLOY_M9_OUTPUT=/absolute/path/to/phase-1-evidence/m9-acceptance.jsonl \
  bash scripts/m9_large_model_acceptance.sh
```

`ALLOY_M9_OUTPUT` must name a new file; existing evidence is not overwritten. `ALLOY_M9_EXPECTED_COMMIT`, when supplied, must exactly equal the checkout's full HEAD SHA. `ALLOY_M9_TIMEOUT_SECONDS` defaults to 1800 with a 120-second cleanup grace; its supported range is 60–7200. CI stores the machine-readable `acceptance.jsonl` and the complete `driver.log` together in a unique `m9-large-model-*` artifact subdirectory, including early preflight/build failures. Require the final `acceptance_passed` record and inspect the measured peaks, admission budgets, and rejection records; exit status alone is not the full review evidence.

## Review record

Fill this section after examining the linked candidate evidence:

1. **Observed results:** Pending — record named test results, actual PIDs/event counts/delta counts, spill/restore counters, large-model peak/budget measurements, and reproducible-build run links from the artifacts.
2. **Failures and repairs:** Pending — link failed artifacts and explain fixes and verified reruns. A retained failure is part of the record.
3. **Outstanding requirements:** Pending — enumerate any required gate without passing evidence. Phase 1 remains open while this list is nonempty.
4. **Decision:** Pending — reviewer records approval only after all required gates pass for the reviewed candidate and the Phase 2 plan has been reviewed against the current SPEC.
