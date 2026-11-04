# DR-0001: mlx-c pin and bindgen outcome

Status: draft. Not accepted until the blank rows are filled from the repo and the name-reservation commands are run.

Date: 2026-11-03
Spec: SPEC-1 v1.9, Phase 0

## Decision

Build the FFI with our own vendored bindgen layer. Do not adopt mlx-c-rust. The Q2 fallback was not triggered: tests 1–8 passed on the Mac before the 1.5-week gate.

The infra-crate API remains the boundary between family crates and this binding layer. Do not reopen the binding choice in Phase 1.

## Pins

Copy the commit and content digest for each vendored tree from `PINS.md`. Do not retype them from memory.

| Pin | Value | Source |
| --- | --- | --- |
| Rust | 1.99.1, `58cf8f9c3`, 2026-10-13 | `rust-toolchain.toml`; builder log |
| mlx-c | v0.8.0 | `PINS.md` |
| MLX | v0.32.4 | `PINS.md` |
| fmt | 12.1.0 | builder log |
| nlohmann/json | vendored | `PINS.md` |
| metal-cpp | vendored, macOS only | `PINS.md` |
| gguflib | not vendored | `MLX_BUILD_GGUF=OFF` |
| Node.js | 26.13.0 Active LTS, promoted 2026-10-28 | nodejs.org release |
| minijinja | not pinned | not in `Cargo.lock` |

Node is the Active LTS line for the TypeScript client. Confirm its end-of-life date on the Node release table before treating any date as final. Pin minijinja only when a crate depends on it, and pin the version that `cargo update -p minijinja --precise` locks.

## Builders

Two images. Do not merge them.

| Image | Dockerfile | Role |
| --- | --- | --- |
| `alloy-builder:phase0` | `scripts/docker/Dockerfile` | Bookworm, libclang `1:14.0.6-12`, test 10 |
| `alloy-native-builder:phase0` | `scripts/docker/Dockerfile.repro` | Debian Trixie, GCC 14, test 11 |

Recorded base prefixes from the image build log:

- `rust:1.99.1-bookworm` begins `sha256:9181b56c2a21ab1d5e73cffe04fc762426d30aa7fac3c91141dd033f49a7`
- `debian:trixie-slim` begins `sha256:f6b08b09a05725f9431502d3b5714f5c4576e82c1a84fa1debe0c24a7e0c3ec4`

Replace those prefixes with the full digests from `docker image inspect` before accepting this record. The Trixie apt packages are not exact-version-pinned.

## Outcome

Local evidence, not a CI result:

- Test 10 passed after `FILE` and the Darwin/glibc stdio types were blocklisted and replaced with an opaque `FILE`. Checked-in `bindings.rs` matched the arm64 container regeneration, and two regenerations matched each other.
- Tests 8 and 9 passed: inventory complete, and both drift injections were rejected.
- Tests 1–7 passed on the Mac, including test 6. Pointer stability and unchanged allocator accounting do not prove CPU/GPU physical-buffer aliasing. That limitation stays open.
- Test 11 passed in a later Colima session after GGUF was disabled. Copy the `libmlx.a` and `libmlxc.a` hashes from that session’s `target/repro-check.*` manifest into this file. Those hashes are not repeated here because the preserved manifest was not in the files available for this draft.
- Corpus tests 13–16 passed locally on 2026-11-03. Comparison artifacts in `models.json` are still unverified, so the corpus is not frozen.

## File-set amendments

Beyond the original Phase 0 file list:

- root `Cargo.toml`
- `scripts/docker/Dockerfile.repro`
- `scripts/bindgencheck.sh`
- `scripts/reprocheck.sh`
- `scripts/pinscheck.sh`
- `bench/corpus/manifest.schema.json`
- GGUF disabled in `crates/mlx-ffi/build.rs` instead of vendoring gguflib

## Names

Unverified. Do not mark these reserved until the command output is pasted under each row.

| Name | Check |
| --- | --- |
| GitHub `LayerDynamics/alloy` | repo exists |
| crates.io `alloyd` | `cargo owner --add` is not the check; publish the 0.0.0 placeholder or show the crate page |
| npm `alloy-ml` | publish the 0.0.0 placeholder |
| PyPI `alloy-convert` | publish the 0.0.0 placeholder |
| Homebrew tap | `homebrew-alloy` repo with stub `Formula/alloy.rb` |

## Not accepted yet

- CI has a workflow file and no passing run.
- No self-hosted Mac runner is registered.
- Name rows above are empty.
- minijinja is unpinned.
- Test 11 hashes are not copied into this record.
- Test 6 does not prove physical aliasing.
