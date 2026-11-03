#!/usr/bin/env bash
#
# scripts/pins_check.sh — verify vendored trees against PINS.md.
#
# Usage:
#   scripts/pins_check.sh [--check]   named test 12, pins_hash_matches
#   scripts/pins_check.sh --test      --check, plus proof that it gates:
#                                     tampering with a tree, a .sha256
#                                     mirror, or PINS.md is rejected
#
# For each vendored tree (mlx-c mlx json fmt metal-cpp) the content digest
# is recomputed from the working tree and must equal BOTH
#   - vendor/<tree>.sha256            (the mirror)
#   - the "Content digest" line in PINS.md for that tree
# The Rust channel recorded in PINS.md must also equal rust-toolchain.toml.
#
# Git tree hashes in PINS.md are provenance only (they cannot be recomputed
# from an export without .git) and are NOT checked here.
#
# dir_hash below is a verbatim copy of the function in scripts/vendor.sh.
# KEEP THE TWO IN SYNC: a change to either must change both, and every
# digest must then be re-recorded by running scripts/vendor.sh.
#
# Overrides (used by --test): VENDOR_DIR, PINS_FILE, TOOLCHAIN_FILE,
# PINS_TREES, CHECK_TOOLCHAIN (0 skips the Rust channel check).
#
# Exit codes: 0 ok, 1 check failed, 2 usage or environment error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO_ROOT/scripts/pins_check.sh"

VENDOR_DIR="${VENDOR_DIR:-$REPO_ROOT/vendor}"
PINS_FILE="${PINS_FILE:-$REPO_ROOT/PINS.md}"
TOOLCHAIN_FILE="${TOOLCHAIN_FILE:-$REPO_ROOT/rust-toolchain.toml}"
PINS_TREES="${PINS_TREES:-mlx-c mlx json fmt metal-cpp}"
CHECK_TOOLCHAIN="${CHECK_TOOLCHAIN:-1}"

die() { echo "pins_check.sh: error: $*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }
need awk; need find; need sort; need mktemp

if command -v sha256sum >/dev/null 2>&1; then
  SHA256() { sha256sum; }
elif command -v shasum >/dev/null 2>&1; then
  SHA256() { shasum -a 256; }
else
  die "no sha256 tool found (need sha256sum or shasum)"
fi

hash_file() { SHA256 < "$1" | awk '{print $1}'; }

# --- verbatim copy of dir_hash from scripts/vendor.sh -----------------------
# Deterministic content digest of a directory: every regular file (sorted
# by path under LC_ALL=C, .DS_Store excluded) contributes
# "relpath\0sha256(content)\0" to a final sha256. Content-only: mode bits
# are ignored so exports are identical across umasks and filesystems.
dir_hash() {
  local dir="$1"
  (
    cd "$dir"
    LC_ALL=C find . -type f ! -name '.DS_Store' -print0 \
      | LC_ALL=C sort -z \
      | while IFS= read -r -d '' f; do
          printf '%s\0%s\0' "$f" "$(hash_file "$f")"
        done
  ) | SHA256 | awk '{print $1}'
}
# --- end verbatim copy ------------------------------------------------------

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# cmd_check: reports every problem it finds, returns 1 if there were any.
cmd_check() {
  local failures=0 name dir computed mirrored recorded

  if [ ! -f "$PINS_FILE" ]; then
    echo "FAIL: $PINS_FILE not found" >&2
    return 1
  fi

  for name in $PINS_TREES; do
    dir="$VENDOR_DIR/$name"
    if [ ! -d "$dir" ]; then
      echo "FAIL: vendored tree missing: $dir (run scripts/vendor.sh)" >&2
      failures=$((failures + 1))
      continue
    fi

    computed="$(dir_hash "$dir")"

    mirrored=""
    if [ -f "$VENDOR_DIR/$name.sha256" ]; then
      mirrored="$(tr -d '[:space:]' < "$VENDOR_DIR/$name.sha256")"
    fi

    # "- Content digest (sha256): <hex> (mirrored in vendor/<name>.sha256)"
    recorded="$(awk -v tag="vendor/$name.sha256)" \
      'index($0, "Content digest") && index($0, tag) { print $5; exit }' \
      "$PINS_FILE")"

    if [ -z "$mirrored" ]; then
      echo "FAIL: $name: vendor/$name.sha256 missing or empty" >&2
      failures=$((failures + 1))
    elif [ "$computed" != "$mirrored" ]; then
      echo "FAIL: $name: digest mismatch: tree=$computed vendor/$name.sha256=$mirrored" >&2
      failures=$((failures + 1))
    fi

    if [ -z "$recorded" ]; then
      echo "FAIL: $name: no Content digest line for it in $PINS_FILE" >&2
      failures=$((failures + 1))
    elif [ "$computed" != "$recorded" ]; then
      echo "FAIL: $name: digest mismatch: tree=$computed PINS.md=$recorded" >&2
      failures=$((failures + 1))
    fi
  done

  if [ "$CHECK_TOOLCHAIN" = "1" ] && [ -f "$TOOLCHAIN_FILE" ]; then
    local want have
    want="$(awk -F'"' '/^channel/ { print $2; exit }' "$TOOLCHAIN_FILE")"
    have="$(awk '/^- Rust channel:/ { print $4; exit }' "$PINS_FILE")"
    if [ "$want" != "$have" ]; then
      echo "FAIL: Rust channel mismatch: rust-toolchain.toml=$want PINS.md=$have" >&2
      failures=$((failures + 1))
    fi
  fi

  if [ "$failures" -gt 0 ]; then
    echo "pins_hash_matches: $failures problem(s)" >&2
    return 1
  fi
  echo "ok: pins_hash_matches ($(echo $PINS_TREES | wc -w | tr -d ' ') trees match PINS.md and vendor/*.sha256)"
}

# --- --test: the check passes on the real repo AND fails on tampering --------
#
# NOTE: runs under "cmd_test || exit 1", which disables errexit inside it,
# so every outcome that matters is tested explicitly.

cmd_test() {
  local fail=0

  # 1. The real repo must pass.
  if bash "$SELF" --check > "$TMP/real.out" 2> "$TMP/real.err"; then
    echo "ok   pins_hash_matches"
  else
    echo "FAIL pins_hash_matches"; cat "$TMP/real.err"
    echo "SKIP tamper checks: they need a green baseline"
    return 1
  fi

  # 2. A synthetic one-tree fixture, tampered with in four ways.
  local fix="$TMP/fixture"
  mkdir -p "$fix/vendor/demo/sub"
  printf 'hello\n' > "$fix/vendor/demo/a.txt"
  printf 'world\n' > "$fix/vendor/demo/sub/b.txt"
  local digest zeros
  digest="$(dir_hash "$fix/vendor/demo")"
  zeros="0000000000000000000000000000000000000000000000000000000000000000"

  write_mirror() { printf '%s\n' "$1" > "$fix/vendor/demo.sha256"; }
  write_pins() {
    printf '## demo\n\n- Content digest (sha256): %s (mirrored in vendor/demo.sha256)\n' "$1" \
      > "$fix/PINS.md"
  }
  run_fixture() {
    VENDOR_DIR="$fix/vendor" PINS_FILE="$fix/PINS.md" PINS_TREES=demo \
      CHECK_TOOLCHAIN=0 bash "$SELF" --check
  }
  expect_reject() {   # $1 = label, $2 = regex the error must match
    if run_fixture > /dev/null 2> "$TMP/rej.err"; then
      echo "FAIL pins_hash_matches: accepted tampering ($1)"; fail=1
    elif ! grep -Eq "$2" "$TMP/rej.err"; then
      echo "FAIL pins_hash_matches: rejected ($1), but for the wrong reason:"
      cat "$TMP/rej.err"; fail=1
    else
      echo "ok   pins_hash_matches rejects: $1"
    fi
  }

  write_mirror "$digest"; write_pins "$digest"
  if ! run_fixture > /dev/null 2> "$TMP/base.err"; then
    echo "FAIL pins_hash_matches: fixture baseline is not green"; cat "$TMP/base.err"
    return 1
  fi

  printf 'tampered\n' >> "$fix/vendor/demo/sub/b.txt"
  expect_reject "a file edited inside the tree" "digest mismatch"
  printf 'world\n' > "$fix/vendor/demo/sub/b.txt"

  write_pins "$zeros"
  expect_reject "a wrong digest recorded in PINS.md" "PINS.md="
  write_pins "$digest"

  write_mirror "$zeros"
  expect_reject "a wrong digest in the .sha256 mirror" "demo.sha256="
  write_mirror "$digest"

  rm -rf "$fix/vendor/demo"
  expect_reject "a vendored tree deleted" "tree missing"

  return "$fail"
}

case "${1:---check}" in
  --check)   cmd_check || exit 1 ;;
  --test)    cmd_test || exit 1 ;;
  -h|--help) sed -n '2,28p' "$0" ;;
  *) die "unknown option: $1 (try --help)" ;;
esac