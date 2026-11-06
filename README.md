# alloy

An Apple-Silicon-native agent harness with a pure-Rust MLX engine.

`alloyd` is a single Rust daemon that runs the harness and the MLX inference
engine in one process, with inference and data processing on-device. The
TypeScript client (`alloy` CLI) is thin and talks to the daemon over a local
JSON-RPC API on a Unix domain socket (`api_version=1`).

**Status:** Phase 1 vertical slice in progress (SPEC-1). Phase 0 (vendored
MLX bindgen, corpus, CI) is complete. Phase 1 delivers one family (Qwen3),
one session path, and streamed chat: `alloy` → `alloyd` → engine → quantized
Qwen3. See `docs/specs/PHASE-1-EXECUTION-PLAN.md`.

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Spec: `docs/specs/SPEC-1-mlx-agent-harness.md.md` (v1.9 on `main` /
  `7-ts-cli`; v1.10 oMLX memory amend on `memory-omlx-parity`)
- Phase 1 plan: `docs/specs/PHASE-1-EXECUTION-PLAN.md`
- Decisions: `docs/decisions/` (DR-numbered)
- Config directory: `~/.alloy/`

## What exists today

| Piece | Location | Notes |
| --- | --- | --- |
| MLX RAII bindings | `crates/mlx-ffi/` | Vendored mlx-c; arrays, streams, quantize ops |
| Shared infra | `crates/inference-infra/` | KV, RoPE, attention, sampling |
| Qwen3 family | `crates/families/qwen3/` | Loader, CPU reference, MLX forward |
| Engine core | `crates/engine/` | Admission, scheduler, FamilyRuntime, Qwen backend |
| Harness core | `crates/harness-core/` | Supervisor, SQLite store, telemetry client |
| Daemon | `crates/alloyd/` | UDS NDJSON JSON-RPC B.6 Phase-1 subset |
| CLI | `ts/cli/` | Guided `start`, standalone `setup`, readiness `doctor`, chat and monitor; Node `26.13.0` |
| Corpus | `bench/corpus/` | Frozen models/scenarios; telemetry schema in `docs/events/` |

Merged foundations include the W6 CLI (#18), memory parity (#19), and M9
residency/emergency cancellation with unified CPU/GPU validation (#22).
M9 still requires the representative two-large-model constrained-budget
acceptance in `docs/designs/omlx-memory-method-parity.md`.

Mac E2E tests 13–14 (real Qwen3-4B-4bit stream / session restart) are not
green yet; they need the memory-parity stack and a local artifact.

## Repo map

| Path | Contents |
| --- | --- |
| `crates/mlx-ffi/` | RAII bindgen over vendored mlx-c |
| `crates/inference-infra/` | Shared KV / RoPE / attention / sampling |
| `crates/families/qwen3/` | Qwen3 family crate |
| `crates/engine/` | In-process engine core + Qwen3 MLX backend |
| `crates/harness-core/` | Supervisor, session store, telemetry |
| `crates/alloyd/` | Daemon binary (`alloyd`) |
| `ts/cli/` | Thin TypeScript CLI (`alloy`) |
| `vendor/` | Vendored `mlx-c` + MLX sources (see [PINS.md](PINS.md)) |
| `scripts/` | Vendoring, coverage, CI helpers |
| `bench/corpus/` | Frozen benchmark corpus |
| `docs/` | Specs, designs, decision records |

## First run from a checkout

Requires Node `26.13.0`, the pinned Rust toolchain (`rust-toolchain.toml`),
CMake ≥ 3.16, and a C++20 compiler. From the repository root:

```bash
npm --prefix ts/cli ci
npm --prefix ts/cli run build
node ts/cli/bin/alloy.js start
```

`start` checks for a compatible running daemon first. When a new daemon is
needed, it checks that a launch command is available, verifies the configured
model or looks in the known local artifact locations, and guides setup if no
valid model is found. Enter a local directory, `download`, or `cancel` at the
prompt. Downloading requires an explicit choice and uses the frozen
Qwen3-4B-4bit revision and file hashes. Existing model directories are never
overwritten. Noninteractive invocations require a valid existing model or an
explicit `--download`; they do not wait for input.

After verification, settings are saved atomically in `~/.alloy/settings.json`.
In a source checkout, `start` builds the current daemon with Cargo, then runs
the executable Cargo reports, including with a custom `CARGO_TARGET_DIR`.
The first build compiles vendored MLX; later builds are incremental. An explicit
`--alloyd-bin /path/to/alloyd` selects a prebuilt daemon instead.

Wait for **`READY — Alloy API 1`**, then keep that terminal open. In another
terminal, from the repository root:

```bash
node ts/cli/bin/alloy.js monitor
# Or connect a chat client:
node ts/cli/bin/alloy.js chat
```

Ctrl+C in the `start` terminal stops its owned daemon. An already-running
compatible daemon is reused without changing or stopping it. API readiness
confirms the listener and protocol; real model loading and generation still
need to succeed on the first chat request. Native Apple-Silicon builds keep
their existing shared CPU/Metal default; no `--metal` flag is required.

These commands handle configuration and diagnostics independently:

```bash
node ts/cli/bin/alloy.js setup --artifact /path/to/qwen3-4b-4bit
node ts/cli/bin/alloy.js setup --download
node ts/cli/bin/alloy.js doctor
node ts/cli/bin/alloy.js doctor --json
```

Choose the setup command for your source: a local artifact or an explicit
download. `setup` (also `--setup`) verifies and saves configuration, then exits;
it does not open chat. `doctor` is read-only and reports configuration, model
preflight, daemon API and launch readiness with a next action. It does not
hash all weights, build the daemon, or prove generation. See
[the startup contract](docs/designs/startup.md) for discovery paths, failure
handling and validation limits.

### Local pre-push checks

Enable the repository's tracked hook once per checkout:

```bash
git config --local core.hooksPath scripts/hooks
```

On macOS, each push first checks the registered GitHub Actions runner through
`~/.local/bin/alloy-runner`. It installs the runner's background service if
missing, starts it when stopped, and requires a live process before continuing.
A running runner is left available for its current job. The wrapper must change
into the existing runner installation directory before invoking `svc.sh`.

The hook then checks `docker info`, opens Docker Desktop if needed, and waits
for the daemon before running the existing native MLX, Qwen3, engine, harness,
and daemon checks. A startup or test failure blocks the push. Docker startup
does not move the native Mac CPU/GPU tests into containers.

The default startup polling window is 120 seconds per service; Docker API
probes are individually limited to five seconds. Override the polling window
with `ALLOY_PRE_PUSH_START_TIMEOUT_SECONDS` or the runner wrapper's absolute
path with `ALLOY_RUNNER_COMMAND`. Runner process readiness is a local check;
GitHub connectivity is still reported by the runner and Actions UI.

Run the same checks without pushing:

```bash
bash scripts/hooks/pre-push </dev/null
```

### Live system and Alloy monitor

After building the CLI, open a separate terminal while `start` is running.
From the repository root:

```bash
node ts/cli/bin/alloy.js monitor
```

The monitor has a colored, responsive TUI with separate system, accelerator
and Alloy runtime panels. Usage bars and up to 60 recent samples show CPU,
GPU, unified RAM and reservation trends; the larger view also graphs ANE
power. Admission pauses, emergency pressure and cancellation awaiting cleanup
have explicit labels. A wider terminal (around 120 columns) places system and
hardware panels side by side; an 80×24 terminal keeps the core readings visible.
`NO_COLOR=1` disables color. Small terminals retain usage bars and history in
a compact view; very short windows prioritize pressure, pending cleanup and
source freshness. Full panels need at least 60 columns and 22 terminal rows.
`TERM=dumb` prints one plain snapshot.

The runtime panel shows the live admission budget, reserved memory, pressure,
request and cached model-residency metadata. The monitor stays usable when
`alloyd` is unavailable and shows the next setup/start or connection-repair
step. It periodically reloads saved configuration while disconnected, so it
can follow setup in another terminal; an explicit `--socket` remains fixed.
An older daemon without `daemon.metrics` must be stopped and restarted with
the current build. Monitoring itself never starts a daemon or downloads a model.

For supported Apple CPU/GPU/Neural Engine power and GPU activity counters,
authorize Apple's sampler in the same terminal, then enable it:

```bash
sudo -v
node ts/cli/bin/alloy.js monitor --powermetrics
```

Only the bounded `powermetrics` child uses `sudo -n`. The CLI remains a normal
user process and does not collect a password or change sudoers. Without
permission, the monitor explains the unavailable accelerator measurements
and continues showing the other data. ANE power is an estimated component
power reading, not ANE utilization percentage or proof that Alloy is using
ANE. Alloy retains its existing unified CPU/Metal execution path.

Press `q` or Ctrl+C to exit. Use `--interval 2000` for a two-second refresh
or `--socket /path/to/alloyd.sock` for another daemon socket. Capture one
snapshot, or record a stream of JSON snapshots for comparison:

```bash
node ts/cli/bin/alloy.js monitor --once --json
node ts/cli/bin/alloy.js monitor --powermetrics --json > /tmp/alloy-monitor.ndjson
```

Missing measurements are `null`, distinct from zero; stale readings carry
their age and errors. CPU/GPU share one physical memory capacity. Non-free
system RAM includes reclaimable pages and is not an app-memory total.
The monitor keeps these samples local. See
[`docs/designs/live-monitor.md`](docs/designs/live-monitor.md) for the sources,
units and validation contract. CLI tests run automatically on repository
pushes and pull requests; the Mac lane also runs them on the native host.

For direct native-daemon development, the existing manual path remains:

```bash
cargo run --locked -p alloyd -- --artifact /path/to/qwen3-4b-4bit --profile qwen3-4b-4bit --corpus-models bench/corpus/models.json
```

One-shot chat connects to an already-running daemon:

```bash
node ts/cli/bin/alloy.js -m "hello"
```

`alloyd` also accepts `--memory-guard
{safe|balanced|aggressive|custom}` (default `balanced`).
