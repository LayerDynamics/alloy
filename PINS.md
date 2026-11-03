# PINS.md — vendored pins

Updated (UTC): 2026-11-03 by `scripts/vendor.sh`. Do not hand-edit the
recorded values; re-run the script instead.

The git tree hashes below are provenance metadata from the upstream
repositories — they cannot be recomputed from the export. The content
digests are the enforceable gate: `scripts/pins_check.sh` (named test
`pins_hash_matches`) recomputes them using the `dir_hash` algorithm
defined in `scripts/vendor.sh`, and compares against this file and the
`vendor/*.sha256` companions.

This file is scoped to vendored sources and the Rust channel. Toolchain
pins beyond Rust (minijinja, Node LTS) are recorded in
`docs/decisions/DR-0001`, not here.

## mlx-c (vendor/mlx-c)

- Repository: https://github.com/ml-explore/mlx-c
- Ref: v0.8.0
- Commit: a26f2d584a14ec619ae8db9a09be03becd47598a
- Git tree hash: 2fb759102487016db122ebff2eb37e0f659b815d (provenance only)
- Content digest (sha256): ba5eaa0d604cfe7e75342027b8fcc6df6ec92c43644fbdf48d8e8b3becee80ac (mirrored in vendor/mlx-c.sha256)

## MLX core (vendor/mlx)

- Repository: https://github.com/ml-explore/mlx
- Ref: v0.32.4 (derived from mlx-c's CMakeLists.txt FetchContent pin)
- Commit: 36969c094354086e15d2a327a9c50feb20b9cf11
- Git tree hash: 0b3dbfacac35a0b64a7c3d9a778ef20db788f565 (provenance only)
- Content digest (sha256): 80f22f4519fb21d03f68f58c7caa155d4f0f74237b1438bdda26074daf8ad47c (mirrored in vendor/mlx.sha256)

## MLX fetch-content dependencies

Pins derived from MLX's own CMakeLists.txt at the MLX commit above.
Without these vendored trees, MLX's FetchContent declarations reach the
network at CMake configure time.

### nlohmann/json (vendor/json)

- URL: https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
- Content digest (sha256): 1ed2e7337686013fe6e835fb0124de257d55feaa45efab44feccec17da7eeee1 (mirrored in vendor/json.sha256)

### fmt (vendor/fmt)

- Repository: https://github.com/fmtlib/fmt
- Tag: 12.1.0
- Commit: 407c905e45ad75fc29bf0f9bb7c5c2fd3475976f
- Git tree hash: 07e6ac840f6697f4a83181d249a4c6aaf23b374e (provenance only)
- Content digest (sha256): 93728bba1fd081d628f7384cc6d5557b2ef3f2ca8df0985682640a331653e815 (mirrored in vendor/fmt.sha256)

### metal-cpp (vendor/metal-cpp)

- URL: https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
- Content digest (sha256): b4171a0acc88376b21860450ff521399880fe6716c50601fbb8754fd02fa6b2e (mirrored in vendor/metal-cpp.sha256)

## Hermeticity mapping

crates/mlx-ffi/build.rs passes these CMake defines, which neutralize
every network FetchContent in the mlx-c + MLX build:

- `-DFETCHCONTENT_SOURCE_DIR_MLX=` vendor/mlx
- `-DFETCHCONTENT_SOURCE_DIR_JSON=` vendor/json
- `-DFETCHCONTENT_SOURCE_DIR_FMT=` vendor/fmt
- `-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=` vendor/metal-cpp (macOS only)

## Toolchain

- Rust channel: 1.99.1 (rust-toolchain.toml)
