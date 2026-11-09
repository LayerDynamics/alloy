# alloy

An Apple-Silicon-native agent harness with a pure-Rust MLX engine.

`alloyd` is a single Rust daemon that runs the harness and the MLX inference
engine in one process, with inference and data processing on-device. The
TypeScript client (`alloy` CLI) is thin and talks to the daemon over a local
JSON-RPC API on a Unix domain socket (`api_version=1`).

**Status:** Phase 1 is accepted, including real Qwen3 streaming, process
restart, unified CPU/Metal execution, and the memory-parity acceptance gates.
See [the final exit review](docs/acceptance/phase-1-exit-review.md) and
[issue #9](https://github.com/LayerDynamics/alloy/issues/9). Phase 2 tools
and policy are planned; the first Homebrew installation is being prepared.

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Spec: `docs/specs/SPEC-1-mlx-agent-harness.md.md` (v1.10)
- Phase 1 plan: `docs/specs/PHASE-1-EXECUTION-PLAN.md`
- Decisions: `docs/decisions/` (DR-numbered)
- Config directory: `~/.alloy/` by default; `ALLOY_CONFIG_DIR` selects another directory

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
PR [#25](https://github.com/LayerDynamics/alloy/pull/25) completed real 4B/14B
constrained-budget residency, incremental Qwen3 streaming, and daemon restart
with durable history plus restored native SSD KV. The Mac workflow runs these
gates from its checked-out commit and retains the evidence. The
[Phase 1 exit review](docs/acceptance/phase-1-exit-review.md) records the
completed runs and their source identities.

The daemon supports bounded hot RAM and persistent safetensors KV. See
[the M10 storage contract](docs/designs/tiered-kv.md) for configuration,
reservation ownership, corruption handling, and restart semantics.

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

## Install with Homebrew

Homebrew is the recommended installation on **Apple Silicon, macOS 14 or
newer**. The first formula builds from a pinned source commit and requires
Xcode 15 or newer with its command-line and Metal compiler tools. The first
native build can take several minutes. Homebrew shows build stages; add
`--verbose` to the install command for detailed compiler output.

**Availability:** this source change and the matching
[Homebrew tap](https://github.com/LayerDynamics/homebrew-alloy) formula must
both merge before the following install command is available. This is a
development preview; no prebuilt bottle or tagged binary release is claimed.

```bash
brew tap LayerDynamics/alloy
brew trust --formula layerdynamics/alloy/alloy
brew install LayerDynamics/alloy/alloy
alloy --version
alloyd --check-native
alloy start
```

Use the fully qualified formula name to select this project's tap. It installs
the `alloy` CLI, its matching `alloyd`, the Metal kernel library, and the
unprivileged monitor together. Homebrew manages a private pinned Node runtime;
your shell's Node version does not need to change. Installation downloads no
model and starts no daemon or background service.

`alloyd --check-native` checks both MLX CPU and Metal execution without a model.
It does not establish model-generation acceptance. If another package already
owns an `alloy` command, resolve that conflict before linking this formula;
do not overwrite the other command blindly. See
[the Homebrew installation contract](docs/designs/homebrew.md).

### Start and use Alloy

`start` checks for a compatible running daemon first. When a new daemon is
needed, it checks that a launch command is available, verifies the configured
model or looks in the known local artifact locations, and guides setup if no
valid model is found. Enter a local directory, `download`, or `cancel` at the
prompt. Downloading requires an explicit choice and uses the frozen
Qwen3-4B-4bit revision and file hashes. Existing model directories are never
overwritten. Noninteractive invocations require a valid existing model or an
explicit `--download`; they do not wait for input.

After verification, settings are saved atomically in the configuration
directory. The Homebrew CLI selects its packaged daemon; it does not compile
Rust during startup. An explicit `--alloyd-bin /path/to/alloyd` selects another
daemon. Existing configuration, model files, sessions and KV cache remain
outside Homebrew's Cellar, normally under `~/.alloy/`.

Wait for **`READY — Alloy API 1`**, then keep that terminal open. In another
terminal:

```bash
alloy monitor
# Or connect a chat client:
alloy chat
```

Ctrl+C in the `start` terminal stops its owned daemon. An already-running
compatible daemon is reused without changing or stopping it. API readiness
confirms the listener and protocol; real model loading and generation still
need to succeed on the first chat request. Native Apple-Silicon builds keep
their existing shared CPU/Metal default; no `--metal` flag is required.
Run one `start` command for the selected socket and wait for `READY` before
connecting clients. Chat uses a text prompt; `monitor` opens the dashboard.
Chat reports when it is still waiting for the first response and displays
runtime warnings on stderr; one-shot stdout remains model text.

These commands handle configuration and diagnostics independently:

```bash
alloy setup --artifact /path/to/qwen3-4b-4bit
alloy setup --download
alloy doctor
alloy doctor --json
```

Choose the setup command for your source: a local artifact or an explicit
download. `setup` (also `--setup`) verifies and saves configuration, then exits;
it does not open chat. `doctor` is read-only and reports configuration, model
preflight, daemon API and launch readiness with a next action. It does not
hash all weights, build the daemon, or prove generation. See
[the startup contract](docs/designs/startup.md) for discovery paths, failure
handling and validation limits.

### Upgrade

```bash
brew update
brew upgrade LayerDynamics/alloy/alloy
```

An upgrade does not replace a running daemon. Stop the existing foreground
`start` with Ctrl+C, then run `alloy start` to use the updated daemon. A
compatible listener is otherwise reused, even after the installed files
change. Model files, configuration, sessions and cache are preserved. Use the
same `ALLOY_CONFIG_DIR`, if set, in every Alloy terminal.

## Contributor checkout

For source development, install Node `26.13.0`, the pinned Rust toolchain
(`rust-toolchain.toml`), CMake ≥ 3.16, a C++20 compiler, and the native Mac
Metal build tools. From the repository root:

```bash
npm --prefix ts/cli ci
npm --prefix ts/cli run build
node ts/cli/bin/alloy.js start
```

Use `node ts/cli/bin/alloy.js` in place of `alloy` for the commands above.
This source launcher builds the checkout's daemon with Cargo and runs the
executable Cargo reports, including with a custom `CARGO_TARGET_DIR`. The
first build compiles vendored MLX; later builds are incremental. An explicit
`--alloyd-bin /path/to/alloyd` selects a prebuilt daemon instead. Development
builds optimize SHA-256 verification, including the ARM SHA backend on Apple
Silicon, while still checking every expected digest.

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

## Live system and Alloy monitor

Open a separate terminal while `alloy start` is running:

```bash
alloy monitor
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

On Apple Silicon, the default monitor reads GPU utilization, CPU/GPU/ANE
power estimates, frequency states and thermal pressure as the current user.
All hardware sampling is unprivileged and enabled by default. Homebrew
installs the native sampler with the CLI; a contributor build compiles it
with Xcode Command Line Tools. Monitoring never builds or installs a helper
at runtime. GPU utilization and native power samples keep independent status
and timestamps.

The runtime panel shows the live admission budget, reserved memory, pressure,
request and cached model-residency metadata. The monitor stays usable when
`alloyd` is unavailable and shows the next setup/start or connection-repair
step. It periodically reloads saved configuration while disconnected, so it
can follow setup in another terminal; an explicit `--socket` remains fixed.
An older daemon without `daemon.metrics` must be stopped and restarted with
the current build. Monitoring itself never starts a daemon or downloads a model.

The monitor never invokes `sudo` or Apple's `powermetrics` command, installs
a privileged service, or requests authorization. The old `--powermetrics`
flag remains a compatibility alias for the same default native collector.
ANE power is an estimated component reading in watts/milliwatts, distinct
from ANE utilization or evidence that Alloy executes on ANE. Frequency
readings are averages of reported frequency states; they are not a direct
measurement of throttled hardware clocks. Unsupported counters remain
explicitly unavailable with their source diagnostics.

Press `q` or Ctrl+C to exit. Use `--interval 2000` for a two-second refresh
or `--socket /path/to/alloyd.sock` for another daemon socket. Capture one
snapshot, or record a stream of JSON snapshots for comparison:

```bash
alloy monitor --once --json
alloy monitor --json > /tmp/alloy-monitor.ndjson
```

Missing measurements are `null`, distinct from zero; stale readings carry
their age and errors. CPU/GPU share one physical memory capacity. Non-free
system RAM includes reclaimable pages and is not an app-memory total.
The monitor keeps these samples local. See
[`docs/designs/live-monitor.md`](docs/designs/live-monitor.md) for the sources,
units and validation contract. CLI tests run automatically on pull requests
and pushes to `main`; the Mac lane also runs them on the native host.

For direct native-daemon development, the existing manual path remains:

```bash
cargo run --locked -p alloyd -- --artifact /path/to/qwen3-4b-4bit --profile qwen3-4b-4bit --corpus-models bench/corpus/models.json
```

One-shot chat connects to an already-running daemon:

```bash
alloy -m "hello"
```

`alloyd` also accepts `--memory-guard
{safe|balanced|aggressive|custom}` (default `balanced`).
