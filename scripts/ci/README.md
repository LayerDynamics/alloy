# Alloy CI scripts

## System Overview

The M3 native matrix runs 14 pinned model profiles on one Mac. It runs one profile at a time, in numerical, behavioral, then qualification order. The GitHub workflow acquires the Mac lease around the entire matrix. For a direct matrix, numerical, or behavioral run, acquire the same lease with the wrapper. [Native observations workflow](../../.github/workflows/m3-native-observations.yaml) · [m3_native_matrix.py:run](./m3_native_matrix.py)

## Watching a run

1. **Waiting for the Mac.** The wrapper waits silently for the host lock before launching the step. During this wait the workflow has yet to create the run directory. When the wrapper launches the step, the step process inherits the lock descriptor and can hold the lock after the wrapper dies. Check the host process tree to distinguish a waiting wrapper, a build, and a running fixture. [mac_locked_step.py:main](./mac_locked_step.py) · [Native observations workflow](../../.github/workflows/m3-native-observations.yaml)
2. **Following a fixture.** The workflow writes to `~/.codex/artifacts/alloy/m3-native-ci/<run-id>-<attempt>/`. Under that directory, open `<fixture>.stdout` and `<fixture>.stderr` inside `<profile>-behavior/` for a behavioral test. For a numerical test, open `native-test.stdout` and `native-test.stderr` inside `<profile>-numerical/`. The behavior runner prints a fixture result when it finishes. A quiet log alone tells you nothing about whether its process is still alive. [m3_native_behavior.py:run_suite](./m3_native_behavior.py) · [m3_native_numerical.py:execute](./m3_native_numerical.py)
3. **Reading the result.** In the run directory, read `index.json` for the overall matrix status and its per-profile errors. A scope that reaches finalization writes `receipt.pending` before renaming it to `receipt.json`. Invalid profile path fields can fail both scopes before either starts, leaving qualification without input receipts. The terminal index can therefore contain three failures for that profile and no receipt references. An interruption can leave a receipt pending or absent. If the matrix never publishes its final index, the overall result is unconfirmed. [m3_native_matrix.py:run_one](./m3_native_matrix.py) · [m3_native_matrix.py:qualify_profile](./m3_native_matrix.py) · [m3_native_numerical.py:pinned_run](./m3_native_numerical.py) · [m3_native_matrix.py:run](./m3_native_matrix.py)

## Local Interface & Invariants

From the repository root, invoke a CLI entry point with `python3 -B scripts/ci/<script>.py`. Both single-profile runners require `--profile ID --candidate SHA --artifact MODEL_DIR --reference REF_FILE --output NEW_DIR`. Use the full Git commit ID for `SHA`.

| Entry point | What the caller can rely on |
| --- | --- |
| [mac_locked_step.py:main](./mac_locked_step.py) | `[--cleanup] STEP_SCRIPT` runs a Bash step with error and pipe-failure handling. `[--cleanup] -- COMMAND [ARG ...]` runs a command directly. The wrapper returns the child status or 128 plus the signal number on interruption. |
| [m3_native_matrix.py:run](./m3_native_matrix.py) | `--candidate SHA --inputs INVENTORY --output NEW_DIR [--package]`. The external JSON inventory maps profile IDs to objects with exactly two absolute paths, `artifact` and `reference`. An unknown top-level profile ID rejects the inventory before any profile runs. An omitted profile, a missing or extra field in its entry, or a relative path fails both scopes for that profile. The matrix continues with the other profiles. Exit 0 requires every scope and qualification to pass and any requested package build to succeed. |
| [m3_native_numerical.py:run](./m3_native_numerical.py) | Runs the profile’s pinned numerical fixture, retains raw output, and evaluates it against the pinned reference. |
| [m3_native_behavior.py:run_suite](./m3_native_behavior.py) | Runs shared and family-specific fixtures. Each gets its own recorded result. The suite continues past individual assertion failures and fails if any fixture failed. |
| [m3_native_qualification.py:issue](./m3_native_qualification.py) | Called by the matrix with both scope receipts. Checks their identities, retained evidence, executable and Metal library, then calls the Rust evaluator if the fixture evidence and coverage rows agree. |
| [m3_native_package.py:publish](./m3_native_package.py) | Called by the matrix with `--package`. Builds a local release `alloyd` for profiles whose numerical, behavior and qualification results passed. A package may exist even when the overall matrix failed. Check its catalog for the included profiles. |
| [m3_artifact_archive.py:main](./m3_artifact_archive.py) | `archive MATRIX_DIR` compresses large numerical JSON files after a terminal run. `restore MATRIX_DIR` verifies the archive and reconstructs the archived JSON payloads. |

## Subdirectories & System Boundaries

Manually dispatch **M3 native observations** with `profile_paths` set to the absolute path of the inventory on the self-hosted arm64 Mac. The job checks out the selected SHA and has a 360-minute timeout. Once the step exports `M3_OUTPUT` to the workflow environment, the workflow attempts to upload evidence even after failure. It excludes the unpacked `package/` directory and, when packaging produced one, archives its contents as `native-package.tar.gz` to preserve file modes. An uploaded artifact can contain partial evidence. Check the workflow conclusion and the matrix index when interpreting it. [Native observations workflow](../../.github/workflows/m3-native-observations.yaml)

## Global Invariants

- A successful pinned run starts and ends at the requested clean Git revision. The runner checks untracked files and submodules, validates manifest-listed model files and the reference digest, and uses an external Cargo target keyed by checkout path and candidate. For matrix and single-profile runs, the resolved output directory must be new, outside the checkout, and have an existing parent. [m3_native_numerical.py:verify_candidate](./m3_native_numerical.py) · [m3_native_numerical.py:pinned_run](./m3_native_numerical.py) · [m3_native_numerical.py:cargo_target](./m3_native_numerical.py)
- A passing native fixture must exit successfully and report exactly one passing test with the requested name. A filtered run that executes zero tests fails this check. Qualification rechecks the recorded command, executable digest and output. [m3_native_numerical.py:require_single_pass](./m3_native_numerical.py) · [m3_native_qualification.py:native_pass](./m3_native_qualification.py)
- The lease serializes cooperating steps on the same Mac. Mac acceptance workflows use `--verify-clean-failure`. An ordinary failed step keeps its original nonzero exit code and red check while the wrapper releases the host after two process snapshots verify no new surviving runner-account processes. Signals, reserved timeout statuses, failed inspection and surviving processes leave the lease dirty. Direct wrapper callers opt in explicitly. Strict `--cleanup` cannot clear a dirty lease. [mac_locked_step.py:main](./mac_locked_step.py) · [mac_host_processes.py:check_quiescence](./mac_host_processes.py)

### Reproducing a profile

Numerical fixtures inherit the caller’s environment and override the selected artifact, reference, profile and raw-output settings. Behavioral fixtures discard inherited `ALLOY_*` variables and apply the explicitly selected settings, including the hybrid-family selection for Ornith and Qwen3.5. The identity inspection requests a context of 32,768 tokens for Gemma 3 and 40,960 for other registered profiles. Individual fixtures determine their own workloads. [m3_native_numerical.py:run](./m3_native_numerical.py) · [m3_native_behavior.py:fixture_env](./m3_native_behavior.py) · [m3_native_behavior.py:inspect_profile](./m3_native_behavior.py)

Qualification verifies every fixture in the profile inventory and emits one behavioral evaluator row per coverage category. The required `qwen3_coder_native_registration` fixture has zero categories. If that fixture alone fails, the behavior receipt says failed while every emitted behavioral row passes. Qualification stops at the status mismatch before invoking the evaluator, so this case produces no evaluator verdict. [m3_native_behavior.py:COVERAGE](./m3_native_behavior.py) · [m3_native_qualification.py:fixture_rows](./m3_native_qualification.py) · [m3_native_qualification.py:issue](./m3_native_qualification.py)

### Reusing a build

For repeat local matrices, set `CARGO_TARGET_DIR` to the same external base to reuse eligible Cargo outputs for the same checkout path and commit. Without that setting, each matrix uses `<matrix-output>/native-builds` as its base, so a fresh output directory gets a separate build tree. The workflow shares `~/.codex/artifacts/alloy/m3-native-ci/cargo-target` across attempts. Under either base, the runner builds locked release targets in `m3-native/<checkout-path-hash>/<candidate>`, requires one matching executable, and rejects conflicting Metal libraries. [m3_native_numerical.py:cargo_target](./m3_native_numerical.py) · [m3_native_numerical.py:build](./m3_native_numerical.py) · [m3_native_numerical.py:metal_library](./m3_native_numerical.py) · [Native observations workflow](../../.github/workflows/m3-native-observations.yaml)

## Global Side Effects

**Host lease.** The wrapper marks `/tmp/alloy-ci-mac-host.lock` dirty before launching a child. Outside GitHub Actions, lock tests can supply their own lock path. [mac_locked_step.py:main](./mac_locked_step.py)

**Packaging.** Request `--package` on the matrix invocation that collects the evidence. The packager consumes that process’s matrix, checks the clean candidate, receipt identities, compiler target and dependency inputs, and requires a usable verdict with no failures for each included profile. At least one profile must pass, included profiles must agree on the Metal library, and the compiled daemon must match that kernel. The packager stages the daemon, Metal library and qualification catalog before publishing a new `package/` directory. The archive restore command only reconstructs payloads. There is no CLI to rebuild a package from a historical index. [m3_native_matrix.py:run](./m3_native_matrix.py) · [m3_native_package.py:package_profile](./m3_native_package.py) · [m3_native_package.py:publish](./m3_native_package.py)

**Archiving deletes verified originals.** Archive and restore take the same exclusive lock on the matrix directory. A second operation on that matrix exits before changing any files. Wait for the first operation to finish before retrying. A fresh archive requires a terminal matrix, no root-level `*.pending` file, and at least one eligible JSON file of 8 MiB or larger directly inside a numerical or qualification `raw/` or `report/` directory. It refuses to overwrite either `numerical-payloads.tar.gz` or `numerical-payloads.manifest.json`. After verifying the archive and publishing both files, it rechecks and deletes each original individually. An interruption can leave only some originals on disk. [m3_artifact_archive.py:archive](./m3_artifact_archive.py)

**If publication stops between the two renames**, the matrix contains `numerical-payloads.tar.gz` and `numerical-payloads.manifest.json.pending`, the final manifest is absent, and the originals are still present. Rerun `python3 -B scripts/ci/m3_artifact_archive.py archive MATRIX_DIR`. The retry compares the pending manifest with the current source inventory and archive digest, verifies every archive member, then publishes the manifest and removes the verified originals. If validation fails, preserve the files for inspection. [m3_artifact_archive.py:archive_locked](./m3_artifact_archive.py)

**Restoring or retrying.** Keep the archive and manifest together. Restore verifies the archive and its members, requires existing destination parents inside the matrix, skips files whose bytes already match, and stops on a differing file. For a missing payload it creates `<filename>.pending` exclusively, verifies the restored bytes, then renames the file. An interrupted restore can leave that pending file in place and block a retry for that payload. [m3_artifact_archive.py:restore](./m3_artifact_archive.py)

## System Verification

From the repository root, run these focused checks for the Python scripts. They exercise matrix accounting, evidence verification, packaging, archive round trips and host-lease behavior using test fixtures.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_m3_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_mac_ci_lock.py'
```

The Rust workspace test command is `cargo test`. Real-model qualification has its own entry point and requires pinned model and reference inputs on the Mac. [m3_native_matrix.py:main](./m3_native_matrix.py)

## Reserved-window scheduling trial

Use a clean checkout of the native build's candidate and a trusted local behavior build receipt. The operator reserves the Mac outside PR service. The trial runs one profile and one test binary under the same host lease as CI. It writes fresh fixture logs and timing receipts in a new external directory.

    python3 -B scripts/ci/m3_reserved_window.py --claim /absolute/path/claim.json --output /absolute/new/attempt

The version 1 JSON claim records `candidate`, `profile`, `package`, `target`, the ordered `fixtures` array, and `behavior_receipt` with its `behavior_receipt_sha256`. On PR arrival, create `STOP` in the attempt directory. The running fixture finishes and later fixtures are recorded as unstarted. Follow `events.jsonl` for fixture progress. Capture the arrival and stop times separately. A successful launcher binds the terminal receipt digest in `finished.json`. The scheduling result grants no qualification credit.

Each instrumented Mac step logs `alloy-ci-lease` with its request and acquisition wall times in nanoseconds, monotonic wait duration, and whether another owner held the lock. Use the first contended PR step to measure lease wait. Workflow queue time and total time to the final required result have different endpoints.
