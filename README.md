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
| CLI | `ts/cli/` | `alloy` / `alloy chat`; Node `26.13.0` |
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

## Building

Requires a pinned Rust toolchain (`rust-toolchain.toml`), CMake ≥ 3.16, and
a C++20 compiler. The first build compiles MLX from vendored sources and is
slow; later builds are incremental.

```bash
cargo build -p alloyd
cd ts/cli && npm ci && npm run build
```

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

Build the CLI, then open a separate terminal while running Alloy:

```bash
cd ts/cli
npm ci
npm run build
node bin/alloy.js monitor
```

The monitor has a colored, responsive TUI with separate system, accelerator
and Alloy runtime panels. Usage bars and up to 60 recent samples show CPU,
GPU, unified RAM and reservation trends; the larger view also graphs ANE
power. Admission pauses, emergency pressure and cancellation awaiting cleanup
have explicit labels. A wider terminal (around 120 columns) places system and
hardware panels side by side; an 80×24 terminal keeps the core readings visible.
`NO_COLOR=1` disables color. Small terminals use a compact view, and `TERM=dumb`
prints one plain snapshot.

The runtime panel shows the live admission budget, reserved memory, pressure,
request and cached model-residency metadata. The monitor stays usable when
`alloyd` is unavailable. Rebuild
and restart `alloyd` to expose the new `daemon.metrics` method; a running
older daemon will report that the method is unavailable.

For supported Apple CPU/GPU/Neural Engine power and GPU activity counters,
authorize Apple's sampler in the same terminal, then enable it:

```bash
sudo -v
node bin/alloy.js monitor --powermetrics
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
node bin/alloy.js monitor --once --json
node bin/alloy.js monitor --powermetrics --json > /tmp/alloy-monitor.ndjson
```

Missing measurements are `null`, distinct from zero; stale readings carry
their age and errors. CPU/GPU share one physical memory capacity. Non-free
system RAM includes reclaimable pages and is not an app-memory total.
The monitor keeps these samples local. See
[`docs/designs/live-monitor.md`](docs/designs/live-monitor.md) for the sources,
units and validation contract. CLI tests run automatically on repository
pushes and pull requests; the Mac lane also runs them on the native host.

Daemon (production profile, local artifact directory):

```bash
alloyd --artifact /path/to/qwen3-4b-4bit --profile qwen3-4b-4bit
```

CLI (connects to an already-running daemon; does not spawn it):

```bash
cd ts/cli && node bin/alloy.js -m "hello"
# or interactive: node bin/alloy.js
```

On the memory-parity branch, `alloyd` also accepts `--memory-guard
{safe|balanced|aggressive|custom}` (default `balanced`).
