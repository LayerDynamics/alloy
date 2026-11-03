#!/usr/bin/env bash
#
# scripts/repro_check.sh
#
# Named test 11: vendored_build_reproducible
#
# Run two clean native builds in separate containers using the same image,
# source snapshot, source path, and target path. Compare SHA-256 manifests
# of the installed static libraries.
#
# Usage:
#   scripts/repro_check.sh [--check]
#   scripts/repro_check.sh --test
#
# --test is an alias for --check: the check itself performs both builds.
#
# Environment:
#   ALLOY_BUILDER_IMAGE
#       Default: alloy-native-builder:phase0
#
#   ALLOY_BUILDER_PLATFORM
#       Default: linux/arm64
#
#   ALLOY_REPRO_JOBS
#       Default: 4
#
#   SOURCE_DATE_EPOCH
#       Default: 0
#
# Results and logs remain under target/repro-check.XXXXXX/.
#
# Scope:
#   Checks installed native static-library archives, not Rust artifacts,
#   executables, Metal libraries, or reproducibility across different
#   images, architectures, source paths, or compiler versions.
#
# Exit codes:
#   0: archive manifests match
#   1: archive manifests differ
#   2: environment, setup, or build failure

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ALLOY_BUILDER_IMAGE:-alloy-native-builder:phase0}"
PLATFORM="${ALLOY_BUILDER_PLATFORM:-linux/arm64}"
JOBS="${ALLOY_REPRO_JOBS:-4}"
EPOCH="${SOURCE_DATE_EPOCH:-0}"
MODE="${1:---check}"

die() {
  echo "repro_check.sh: error: $*" >&2
  exit 2
}

[ "$#" -le 1 ] || die "expected at most one option"

case "$MODE" in
  --check|--test)
    ;;
  -h|--help)
    sed -n '2,42p' "$0"
    exit 0
    ;;
  *)
    die "unknown option: $MODE (try --help)"
    ;;
esac

case "$JOBS" in
  ""|*[!0-9]*)
    die "ALLOY_REPRO_JOBS must be a positive integer"
    ;;
esac

[ "$JOBS" -gt 0 ] || die "ALLOY_REPRO_JOBS must be greater than zero"

case "$EPOCH" in
  ""|*[!0-9]*)
    die "SOURCE_DATE_EPOCH must be a nonnegative integer"
    ;;
esac

[ "$PLATFORM" = "linux/arm64" ] \
  || die "this check currently requires linux/arm64"

command -v docker >/dev/null 2>&1 \
  || die "docker is required"

docker info >/dev/null 2>&1 \
  || die "the docker daemon is not reachable"

docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "builder image '$IMAGE' missing; build scripts/docker/Dockerfile.repro first"

for file in \
  "$REPO_ROOT/Cargo.lock" \
  "$REPO_ROOT/crates/mlx-ffi/src/bindings.rs" \
  "$REPO_ROOT/vendor/mlx-c/CMakeLists.txt"
do
  [ -f "$file" ] || die "required file missing: $file"
done

mkdir -p "$REPO_ROOT/target"
WORK="$(mktemp -d "$REPO_ROOT/target/repro-check.XXXXXX")"

# Resolve the tag once so both builds use the same local image,
# even if the tag is moved during this check.
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"

printf '%s\n' "$IMAGE_ID" > "$WORK/image-id.txt"

echo "Results: $WORK"
echo "Image:   $IMAGE_ID"
echo "Jobs:    $JOBS"

# Create one source snapshot for both runs. Exclude target to avoid
# including these results or any existing build products.
echo "Creating source snapshot..."

tar -C "$REPO_ROOT" \
  --exclude=./target \
  --exclude=./.git \
  -cf "$WORK/source.tar" . \
  || die "could not create source snapshot"

cat > "$WORK/build.sh" <<'EOS'
set -euo pipefail

export LC_ALL=C
export LANG=C
export TZ=UTC
export CARGO_INCREMENTAL=0

umask 022

# Neither regeneration nor bindgen-only mode belongs in a native build.
unset MLX_FFI_REGENERATE
unset MLX_FFI_BINDGEN_ONLY
unset MLX_FFI_ASAN
unset BINDGEN_EXTRA_CLANG_ARGS
unset RUSTFLAGS
unset CARGO_ENCODED_RUSTFLAGS
unset CFLAGS
unset CXXFLAGS

# Preserve the compiler selections supplied by the image.
export CC="${CC:-cc}"
export CXX="${CXX:-c++}"

command -v "$CC" >/dev/null 2>&1 || {
  echo "Selected C compiler not found: $CC" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || {
  echo "Selected C++ compiler not found: $CXX" >&2
  exit 1
}

mkdir -p /work /out
tar -C /work -xf /input/source.tar
cd /work

{
  echo "architecture: $(uname -m)"
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "CC=$CC"
  echo "CXX=$CXX"
  "$CC" --version
  "$CXX" --version
  cmake --version
  ar --version
  echo "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"
  echo "jobs=$BUILD_JOBS"
} > /out/toolchain.txt

# Every container starts with a fresh /tmp/target. Only downloaded Cargo
# registry data is shared; compilation artifacts are not shared.
cargo build \
  --locked \
  --release \
  -p mlx-ffi \
  --jobs "$BUILD_JOBS" \
  --target-dir /tmp/target

# Locate the CMake installation produced by the mlx-ffi build script.
mapfile -t roots < <(
  find /tmp/target/release/build \
    -mindepth 2 \
    -maxdepth 2 \
    -type d \
    -path '*/mlx-ffi-*/out' \
    | sort
)

if [ "${#roots[@]}" -ne 1 ]; then
  echo "Expected exactly one mlx-ffi native installation; found ${#roots[@]}." >&2
  printf '%s\n' "${roots[@]}" >&2
  exit 1
fi

native_root="${roots[0]}"

cd "$native_root"

mapfile -t archives < <(
  find . -type f \
    \( -path './lib/*.a' -o -path './lib64/*.a' \) \
    | sort
)

if [ "${#archives[@]}" -eq 0 ]; then
  echo "No installed static libraries found under $native_root." >&2
  exit 1
fi

have_mlxc=0
have_mlx=0

for archive in "${archives[@]}"; do
  case "$archive" in
    ./lib/libmlxc.a|./lib64/libmlxc.a)
      have_mlxc=1
      ;;
    ./lib/libmlx.a|./lib64/libmlx.a)
      have_mlx=1
      ;;
  esac
done

if [ "$have_mlxc" -ne 1 ] || [ "$have_mlx" -ne 1 ]; then
  echo "Required archives libmlxc.a and libmlx.a were not both installed." >&2
  printf '%s\n' "${archives[@]}" >&2
  exit 1
fi

# Relative paths keep the manifest independent of Cargo's build-script
# directory name. Compare every installed archive, not only mlx/mlxc.
sha256sum "${archives[@]}" > /out/archives.sha256

for archive in "${archives[@]}"; do
  printf '\n### %s\n' "$archive"
  ar t "$archive"
done > /out/archive-members.txt

cat /out/archives.sha256
EOS

run_build() {
  local label="$1"
  local out="$WORK/$label"

  mkdir -p "$out" || die "cannot create $out"

  echo
  echo "Starting $label..."

  if docker run --rm \
    --platform "$PLATFORM" \
    -v "$WORK/source.tar":/input/source.tar:ro \
    -v "$WORK/build.sh":/build.sh:ro \
    -v "$out":/out \
    -v alloy-cargo-registry:/usr/local/cargo/registry \
    -e BUILD_JOBS="$JOBS" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
    -e SOURCE_DATE_EPOCH="$EPOCH" \
    "$IMAGE_ID" bash /build.sh \
    > "$out/build.log" 2>&1
  then
    [ -s "$out/toolchain.txt" ] \
      || die "$label produced no toolchain metadata"

    [ -s "$out/archives.sha256" ] \
      || die "$label produced no archive manifest"

    echo "$label finished:"
    cat "$out/archives.sha256"
  else
    echo "FAIL: $label failed. Last 100 log lines:" >&2
    tail -n 100 "$out/build.log" >&2
    die "native build failed; full log: $out/build.log"
  fi
}

run_build run1
run_build run2

if ! diff -u \
  "$WORK/run1/toolchain.txt" \
  "$WORK/run2/toolchain.txt" \
  > "$WORK/toolchain.diff"
then
  cat "$WORK/toolchain.diff" >&2
  die "the two runs reported different toolchain metadata"
fi

if diff -u \
  "$WORK/run1/archives.sha256" \
  "$WORK/run2/archives.sha256" \
  > "$WORK/archives.diff"
then
  echo
  echo "ok: vendored_build_reproducible"
  echo "Two clean builds produced identical installed static-library manifests."
  echo "Evidence: $WORK"
else
  echo >&2
  echo "FAIL: vendored_build_reproducible" >&2
  cat "$WORK/archives.diff" >&2
  echo "Logs and archive-member lists: $WORK" >&2
  exit 1
fi