#!/usr/bin/env bash
#
# Regenerate mlx-c bindings inside the configured builder image.
#
# Usage:
#   scripts/bindgen_check.sh [--check]
#       Compare container-generated bindings with the checked-in file.
#
#   scripts/bindgen_check.sh --update
#       Replace the checked-in file with container-generated bindings.
#
#   scripts/bindgen_check.sh --test
#       Run --check, then require a second regeneration to match the first.
#
#   scripts/bindgen_check.sh --build-image
#       Build or rebuild the configured builder image.
#
# Environment:
#   ALLOY_BUILDER_IMAGE
#       Default: alloy-builder:phase0
#
#   ALLOY_BUILDER_PLATFORM
#       Default: linux/arm64. This script currently requires that platform.
#
#   ALLOY_RUST_IMAGE
#       Optional Dockerfile RUST_IMAGE override.
#
# Exit codes:
#   0: success
#   1: comparison failure
#   2: usage, environment, or regeneration failure

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ALLOY_BUILDER_IMAGE:-alloy-builder:phase0}"
PLATFORM="${ALLOY_BUILDER_PLATFORM:-linux/arm64}"
BINDINGS="$REPO_ROOT/crates/mlx-ffi/src/bindings.rs"

die() {
  echo "bindgen_check.sh: error: $*" >&2
  exit 2
}

MODE="${1:---check}"

[ "$#" -le 1 ] || die "expected at most one option"

case "$MODE" in
  --check|--update|--test|--build-image)
    ;;
  -h|--help)
    sed -n '2,35p' "$0"
    exit 0
    ;;
  *)
    die "unknown option: $MODE (try --help)"
    ;;
esac

[ "$PLATFORM" = "linux/arm64" ] \
  || die "this script's Clang flags require ALLOY_BUILDER_PLATFORM=linux/arm64"

command -v docker >/dev/null 2>&1 \
  || die "docker is required"

docker info >/dev/null 2>&1 \
  || die "the docker daemon is not reachable"

if [ "$MODE" = "--check" ] || [ "$MODE" = "--test" ]; then
  [ -s "$BINDINGS" ] \
    || die "missing or empty $BINDINGS; use --update to generate it"
fi

# Use a unique directory to avoid deleting another invocation's scratch files.
mkdir -p "$REPO_ROOT/target"
WORK="$(mktemp -d "$REPO_ROOT/target/bindgen-check.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

build_image() {
  local args=()

  if [ -n "${ALLOY_RUST_IMAGE:-}" ]; then
    args+=(--build-arg "RUST_IMAGE=$ALLOY_RUST_IMAGE")
  fi

  docker build \
    --platform "$PLATFORM" \
    -t "$IMAGE" \
    ${args[@]+"${args[@]}"} \
    -f "$REPO_ROOT/scripts/docker/Dockerfile" \
    "$REPO_ROOT/scripts/docker" \
    || die "builder image build failed"
}

ensure_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    build_image
  fi
}

cat > "$WORK/regen.sh" <<'EOS'
set -euo pipefail

echo "builder: $(uname -m) | $(rustc --version) | $(dpkg-query -W -f='${Package} ${Version}' libclang1-14)" >&2

mkdir -p /work

tar -C /src \
  --exclude=./target \
  --exclude=./.git \
  -cf - . \
  | tar -C /work -xf -

cd /work

# Remove the copied baseline so a failed or skipped regeneration cannot
# accidentally be mistaken for newly generated output.
rm -f crates/mlx-ffi/src/bindings.rs

cargo build \
  --locked \
  -p mlx-ffi \
  --features regenerate-bindings \
  --target-dir /tmp/target

test -s crates/mlx-ffi/src/bindings.rs

cp crates/mlx-ffi/src/bindings.rs /out/bindings.rs
chown "$HOST_UID:$HOST_GID" /out/bindings.rs
EOS

regenerate() {
  local out="$1"

  mkdir -p "$out" || die "cannot create output directory: $out"

  docker run --rm \
    --platform "$PLATFORM" \
    -v "$REPO_ROOT":/src:ro \
    -v "$out":/out \
    -v "$WORK/regen.sh":/regen.sh:ro \
    -v alloy-cargo-registry:/usr/local/cargo/registry \
    -e MLX_FFI_REGENERATE=1 \
    -e MLX_FFI_BINDGEN_ONLY=1 \
    -e BINDGEN_EXTRA_CLANG_ARGS="--target=aarch64-unknown-linux-gnu -march=armv8.6-a+bf16" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "$IMAGE" bash /regen.sh \
    || die "container regeneration failed"

  [ -s "$out/bindings.rs" ] \
    || die "regeneration produced no bindings.rs"
}

cmd_check() {
  ensure_image
  regenerate "$WORK/run1"

  if cmp -s "$WORK/run1/bindings.rs" "$BINDINGS"; then
    echo "ok: bindgen_output_stable (checked-in bindings equal container regeneration)"
    return 0
  fi

  diff -u "$BINDINGS" "$WORK/run1/bindings.rs" \
    > "$WORK/diff.txt" || true

  echo "FAIL: bindgen_output_stable: checked-in bindings differ from container regeneration." >&2
  echo "First 60 diff lines:" >&2
  sed -n '1,60p' "$WORK/diff.txt" >&2
  echo "Review the differences before adopting them with --update." >&2

  return 1
}

cmd_update() {
  ensure_image
  regenerate "$WORK/run1"

  cp "$WORK/run1/bindings.rs" "$BINDINGS" \
    || die "cannot update $BINDINGS"

  echo "wrote $BINDINGS"
  echo "Next: run --test, the coverage inventory gate, and the native Mac tests."
}

cmd_test() {
  cmd_check || return 1
  regenerate "$WORK/run2"

  if cmp -s "$WORK/run1/bindings.rs" "$WORK/run2/bindings.rs"; then
    echo "ok: bindgen_output_stable (two regenerations are byte-identical)"
    return 0
  fi

  echo "FAIL: bindgen_output_stable: two regenerations differ." >&2

  diff -u "$WORK/run1/bindings.rs" "$WORK/run2/bindings.rs" \
    > "$WORK/determinism.diff" || true

  sed -n '1,60p' "$WORK/determinism.diff" >&2
  return 1
}

case "$MODE" in
  --check)
    cmd_check || exit 1
    ;;
  --update)
    cmd_update
    ;;
  --test)
    cmd_test || exit 1
    ;;
  --build-image)
    build_image
    ;;
esac