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

Open pull requests (as of 2026-11-05):

- [#18](https://github.com/LayerDynamics/alloy/pull/18) — W6 `ts/cli`
- [#19](https://github.com/LayerDynamics/alloy/pull/19) — oMLX-method memory parity (live guard, Affine4-resident weights, prefill admit). Design: `docs/designs/omlx-memory-method-parity.md` (on that branch).

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
