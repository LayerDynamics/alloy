# alloy

An Apple-Silicon-native agent harness with a pure-Rust MLX engine.

`alloyd` is a single Rust daemon that runs the whole harness — tools, MCP,
skills, durable sessions, context management with compaction, automations —
and the MLX inference engine in one process, with all inference and data
processing on-device. TypeScript clients (`alloy` CLI, TUI, SDK) are thin
and talk to the daemon over a local JSON-RPC API on a Unix domain socket.

**Status:** Phase 0 (decision spike). This repository currently contains
the vendored-MLX bindgen spike, the frozen benchmark corpus, and CI. It is
not yet usable as a product. See `docs/specs/PHASE-0-EXECUTION-PLAN.md`
for what exists today and what comes next.

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Spec: `docs/specs/SPEC-1-mlx-agent-harness.md` (v1.9)
- Decisions: `docs/decisions/` (DR-numbered)
- Config directory: `~/.alloy/`

## Repo map

| Path | Contents |
| --- | --- |
| `crates/mlx-ffi/` | RAII bindgen layer over vendored mlx-c (arrays, devices, streams, closures, `mlx_compile`) |
| `vendor/` | Vendored `mlx-c` + MLX core sources, pinned by commit and hash (see [PINS.md](PINS.md)) |
| `scripts/` | Vendoring, coverage inventory, and CI check scripts |
| `bench/corpus/` | Frozen benchmark corpus (tiers, models, scenarios, fixtures) |
| `docs/` | Specs and decision records |

## Building

Requires a pinned Rust toolchain (picked up automatically from
`rust-toolchain.toml`), CMake ≥ 3.16, and a C++20 compiler. The first
build compiles MLX from the vendored sources and is slow; subsequent
builds are incremental.
