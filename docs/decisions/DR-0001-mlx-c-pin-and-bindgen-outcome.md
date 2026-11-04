# DR-0001: mlx-c pin and bindgen outcome

Status: accepted
Date: 2026-11-04
Spec: SPEC-1 v1.9, Phase 0

## Decision

Build the FFI with our own vendored bindgen layer. Do not adopt mlx-c-rust.
The Q2 fallback was not triggered: tests 1–8 passed on the Mac before the
1.5-week gate.

The infra-crate API remains the boundary between family crates and this
binding layer. Do not reopen the binding choice in Phase 1.

## Pins

Copy the commit and content digest for each vendored tree from `PINS.md`;
do not retype them from memory.

| Pin | Value | Source |
| --- | --- | --- |
| Rust | 1.99.1, `58cf8f9c3`, 2026-10-13 | `rust-toolchain.toml`; builder log |
| bindgen | `=0.73.2`, exact pin | root `Cargo.toml` |
| mlx-c | v0.8.0 | `PINS.md` |
| MLX | v0.32.4 | `PINS.md` |
| fmt | 12.1.0 | builder log |
| nlohmann/json | vendored | `PINS.md` |
| metal-cpp | vendored, macOS only | `PINS.md` |
| gguflib | not vendored | `MLX_BUILD_GGUF=OFF` |
| Node.js | 26.13.0, Active LTS, promoted 2026-10-28 | nodejs.org release |
| minijinja | intentionally unpinned | not yet a workspace dependency |

Node is the Active LTS line for the TypeScript client. minijinja is pinned
in the root `Cargo.toml` only when a crate depends on it; pin the version
`cargo update -p minijinja --precise` locks at that time. The bindgen pin
is exact because its generated code must be byte-identical across
regenerations (named test 10).

## Builders

Two images. Do not merge them.

| Image | Dockerfile | Role |
| --- | --- | --- |
| `alloy-builder:phase0` | `scripts/docker/Dockerfile` | Bookworm, libclang `1:14.0.6-12`, test 10 |
| `alloy-native-builder:phase0` | `scripts/docker/Dockerfile.repro` | Debian Trixie, GCC 14, test 11 |

The native-builder digest from the passing CI run is
`sha256:1dc832eb4f433578ee15c3ce64f9275ba48e7ea9bcb3af80e3611de2d5ff00fc`.
The bindgen-builder digest was not captured from the CI log; record it from
the "bindgen image and test 10" step of the run before archiving this
document. The Trixie apt packages are not exact-version-pinned.

## Outcome

### Named tests

- Tests 1–7 passed on the physical Mac, including test 6.
- Test 6 limitation: pointer stability and unchanged allocator accounting
  do not prove CPU/GPU physical-buffer aliasing. This remains open
  evidence, documented rather than hidden.
- Tests 8–9 passed: inventory complete, both drift injections rejected.
- Test 10 passed after `FILE` and the Darwin/glibc stdio types were
  blocklisted and replaced with an opaque `FILE`. The checked-in
  `bindings.rs` matched the arm64 container regeneration, and two
  regenerations matched each other.
- Test 11 passed in CI: two clean builds in
  `alloy-native-builder:phase0` produced identical static-library
  manifests — `libmlx.a`
  `f0a599bb10c14a7820bf9825f0510aacea3f473a88a195065ec7f85dc17ff0ad`,
  `libmlxc.a`
  `1e3c31ce28ca597daad04dc26b11e3dc9fe4d8169c2e39fe8c14a570c0a8824e`.
- Tests 12–16 passed locally and in CI.

Reproducibility is per-image: the earlier Colima-session hashes differ
because the images were built from different apt snapshots. That is why the
image digest is recorded next to the library hashes.

### CI

The `ci` workflow (`.github/workflows/ci.yaml`) is green on
[run 41812314281](https://github.com/LayerDynamics/alloy/actions/runs/41812314281)
and on the corpus-freeze commit `06e086b` ("corpus finished"). The Linux
arm64 lane runs on `ubuntu-24.04-arm` and executes tests 12–16, the bindgen
check, the inventory gate, and the native reproducibility check. The Mac
lane runs on the self-hosted runner (`Ryans-MacBook-Pro`, labels
`self-hosted, macOS, ARM64`) on manual dispatch.

### Corpus

Frozen 2026-11-04, committed as `06e086b`. `models.json` carries
`status: frozen`: the mlx-community artifacts resolved and hashed
(2026-11-03), the llama.cpp GGUFs downloaded and hashed (2026-11-04), the
Ollama rows pinned by registry manifest digest (2026-11-04), and the oMLX
row closed by byte-verified loads through oMLX 0.7.3 (2026-11-04). The
freeze record and all run evidence live in `bench/corpus/README.md`.

## File-set amendments

Beyond the original Phase 0 file list:

- root `Cargo.toml`
- `scripts/docker/Dockerfile.repro`
- `scripts/bindgen_check.sh`
- `scripts/repro_check.sh`
- `scripts/pins_check.sh`
- `bench/corpus/manifest.schema.json`
- `bench/corpus/resolve_ggufs.py`, `resolve_ollama.py`, `resolve_omlx.py`
- GGUF disabled in `crates/mlx-ffi/build.rs` instead of vendoring gguflib

## Names

All reserved, 2026-11-04, as 0.0.0 placeholders:

| Name | Evidence |
| --- | --- |
| GitHub `LayerDynamics/alloy` | public repo |
| crates.io `alloyd` | v0.0.0 published; owner Ryan O'Boyle; links to the repo |
| npm `alloyml` | v0.0.0, confirmed via `npm view alloyml version` |
| PyPI `alloy-convert` | 0.0.0, maintainer `layerdynamics`, Apache-2.0 |
| Homebrew tap | `LayerDynamics/homebrew-alloy`, stub `Formula/alloy.rb` at `826777f`; `brew tap LayerDynamics/alloy` verified |

## Fallback wording

The fallback was never triggered; it was not removed from the spec. DR-0001
records only the pin and the own-layer outcome.
