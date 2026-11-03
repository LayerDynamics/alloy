#!/usr/bin/env bash
#
# scripts/vendor.sh — update the vendored mlx-c / MLX pin set.
#
# Usage:
#   scripts/vendor.sh                    re-vendor the pins recorded in PINS.md
#   scripts/vendor.sh <mlx-c-ref>        vendor mlx-c at a tag/branch/commit
#   scripts/vendor.sh <mlx-c-ref> <mlx-ref>   ... and override the MLX pin
#
# What it does:
#   1. Resolves the mlx-c ref to an immutable commit SHA (tags are peeled
#      to their commit) and exports the tree (no .git) into vendor/mlx-c/.
#   2. Derives the MLX core pin from mlx-c's CMakeLists.txt (FetchContent
#      GIT_TAG) and exports it into vendor/mlx/.
#   3. Derives MLX's own fetch-content dependency pins from the vendored
#      MLX CMakeLists.txt and vendors them too:
#        vendor/json      nlohmann/json  (URL tarball, always fetched)
#        vendor/fmt       fmtlib/fmt     (git, always fetched)
#        vendor/metal-cpp Apple metal-cpp (URL zip, macOS Metal builds)
#      Without these, even a "vendored" MLX build reaches the network at
#      CMake configure time. build.rs overrides every one of them with
#      FETCHCONTENT_SOURCE_DIR_* defines — see PINS.md for the mapping.
#   4. Computes a deterministic content digest of each tree, records
#      commit / git tree hash / digest in PINS.md, and mirrors digests in
#      vendor/<tree>.sha256 (named test pins_hash_matches' input).
#
# Re-vendoring the same pins on the same UTC day reproduces every written
# file byte-for-byte (dates are UTC; digests are content-only).
#
# After a pin bump you must also:
#   - regenerate crates/mlx-ffi/src/bindings.rs   (scripts/bindgen_check.sh)
#   - regenerate crates/mlx-ffi/coverage/inventory.txt (scripts/coverage_inventory.sh)
#   - run the test suite and record the outcome in docs/decisions/DR-0001
#
# The .sha256 files live at vendor/ root, OUTSIDE the hashed trees — do
# not move them into vendor/<tree>/ or the digests become circular.

set -euo pipefail

MLX_C_REPO="https://github.com/ml-explore/mlx-c.git"
MLX_REPO="https://github.com/ml-explore/mlx.git"
FMT_REPO="https://github.com/fmtlib/fmt.git"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- requirements ---------------------------------------------------------

die() { echo "vendor.sh: error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }
need git; need awk; need tar; need mktemp; need curl; need unzip

if command -v sha256sum >/dev/null 2>&1; then
  SHA256() { sha256sum; }
elif command -v shasum >/dev/null 2>&1; then
  SHA256() { shasum -a 256; }
else
  die "no sha256 tool found (need sha256sum or shasum)"
fi

hash_file() { SHA256 < "$1" | awk '{print $1}'; }

# Deterministic content digest of a directory: every regular file (sorted
# by path under LC_ALL=C, .DS_Store excluded) contributes
# "relpath\0sha256(content)\0" to a final sha256. Content-only: mode bits
# are ignored so exports are identical across umasks and filesystems.
# This is the exact algorithm that scripts/pins_check.sh (named test
# `pins_hash_matches`) recomputes — keep the two in sync.
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

# current_mlx_c_ref — read the pin back out of PINS.md, if present.
# Contract: the mlx-c section is written FIRST in PINS.md and uses the
# only `- Ref:` line ahead of MLX's; if PINS.md's format changes, this
# parser changes with it.
current_mlx_c_ref() {
  [ -f PINS.md ] || return 1
  awk '/^- Ref: / {print $3; exit}' PINS.md
}

# --- pin derivation (single source of truth: upstream CMakeLists) ---------

# MLX core pin: from mlx-c's FetchContent GIT_TAG.
# Single-process selection: no awk|tr|head pipeline (a pipefail/SIGPIPE
# hazard if upstream ever grows a second GIT_TAG line).
derive_mlx_ref() {
  local cmake="vendor/mlx-c/CMakeLists.txt"
  [ -f "$cmake" ] || die "cannot derive MLX pin: $cmake missing"
  local ref
  ref="$(awk '/GIT_TAG/ { line = $NF; gsub(/\)/, "", line); print line; exit }' "$cmake")"
  [ -n "$ref" ] || die "could not parse GIT_TAG from $cmake"
  echo "$ref"
}

# nlohmann/json tarball URL, from MLX's CMakeLists (FetchContent URL).
derive_json_url() {
  local cmake="vendor/mlx/CMakeLists.txt"
  local urls url
  urls="$(grep -oE 'https://github.com/nlohmann/json/releases/download/[^") ]*' "$cmake")" || true
  url="${urls%%$'\n'*}"
  [ -n "$url" ] || die "could not parse nlohmann/json URL from $cmake"
  echo "$url"
}

# fmt git tag, from MLX's CMakeLists (the GIT_TAG following fmt's
# GIT_REPOSITORY line). Note: fmt tags are NOT v-prefixed (e.g. 12.1.0).
derive_fmt_ref() {
  local cmake="vendor/mlx/CMakeLists.txt"
  local tag
  tag="$(awk '/GIT_REPOSITORY .*fmtlib\/fmt\.git/ { in_fmt = 1 }
              in_fmt && /GIT_TAG/ { print $NF; exit }' "$cmake")"
  [ -n "$tag" ] || die "could not parse fmt GIT_TAG from $cmake"
  echo "$tag"
}

# Apple metal-cpp zip URL, from MLX's CMakeLists (METAL_CPP_URL).
derive_metal_cpp_url() {
  local cmake="vendor/mlx/CMakeLists.txt"
  local urls url
  urls="$(grep -oE 'https://developer\.apple\.com/metal/cpp/files/[^") ]*\.zip' "$cmake")" || true
  url="${urls%%$'\n'*}"
  [ -n "$url" ] || die "could not parse metal-cpp URL from $cmake"
  echo "$url"
}

# --- tree acquisition ------------------------------------------------------

# export_tree <repo_url> <ref> <dest>
# Side effects: sets EXPORT_COMMIT and EXPORT_TREE for the caller.
# Partial clone (--filter=blob:none) keeps the fetch light; the checkout
# downloads the one tree's blobs; git archive exports a clean tree
# (no .git, exec bits preserved). Bash globbing (not `find -quit`, which
# is not reliably available on macOS's BSD find) checks the export.
export_tree() {
  local repo="$1" ref="$2" dest="$3"
  local tmp="$TMP_ROOT/$(basename "$dest")"

  git clone --quiet --filter=blob:none "$repo" "$tmp/clone"
  EXPORT_COMMIT="$(git -C "$tmp/clone" rev-parse "${ref}^{commit}")" \
    || die "cannot resolve '$ref' in $repo"
  git -C "$tmp/clone" -c advice.detachedHead=false checkout --quiet \
    --detach "$EXPORT_COMMIT"
  EXPORT_TREE="$(git -C "$tmp/clone" rev-parse "$EXPORT_COMMIT^{tree}")"

  mkdir -p "$tmp/export"
  git -C "$tmp/clone" archive --format=tar "$EXPORT_COMMIT" \
    | tar -x -C "$tmp/export"

  local -a contents=()
  shopt -s nullglob dotglob
  contents=("$tmp/export"/*)
  shopt -u nullglob dotglob
  [ "${#contents[@]}" -gt 0 ] || die "export of $ref produced an empty tree"

  rm -rf "$dest"
  mv "$tmp/export" "$dest"
  echo "  vendored $ref -> $dest"
}

# extract_content_root <archive> <scratch> — extracts an archive and
# echoes the path of its content root, transparently stripping a single
# top-level wrapper directory (release tarballs and the metal-cpp zip
# disagree on this). Bash globbing for portability; dotglob so hidden
# files count as content.
extract_content_root() {
  local archive="$1" scratch="$2"
  rm -rf "$scratch"
  mkdir -p "$scratch"
  case "$archive" in
    *.zip) unzip -q "$archive" -d "$scratch" ;;
    *)     tar -xf "$archive" -C "$scratch" ;;
  esac
  local -a entries=()
  shopt -s nullglob dotglob
  entries=("$scratch"/*)
  shopt -u nullglob dotglob
  if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ]; then
    echo "${entries[0]}"
  else
    echo "$scratch"
  fi
}

# fetch_archive <url> <dest> — download an archive and install its
# normalized content root at <dest>.
fetch_archive() {
  local url="$1" dest="$2"
  local name tmp archive content_root
  name="$(basename "$dest")"
  tmp="$TMP_ROOT/$name"
  mkdir -p "$tmp"
  archive="$tmp/archive"

  curl --fail --silent --show-error --location --retry 3 \
    -o "$archive" "$url"

  content_root="$(extract_content_root "$archive" "$tmp/extract")"
  rm -rf "$dest"
  mv "$content_root" "$dest"
  echo "  vendored ${url} -> $dest"
}

# --- main ------------------------------------------------------------------

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

MLX_C_REF="${1:-$(current_mlx_c_ref || true)}"
[ -n "$MLX_C_REF" ] || die "no mlx-c ref given and no pin found in PINS.md; usage: scripts/vendor.sh <mlx-c-ref> [<mlx-ref>]"

echo "Vendoring mlx-c at ${MLX_C_REF}..."
export_tree "$MLX_C_REPO" "$MLX_C_REF" vendor/mlx-c
MLX_C_COMMIT="$EXPORT_COMMIT"; MLX_C_TREE="$EXPORT_TREE"

if [ -n "${2:-}" ]; then
  MLX_REF="$2"
  MLX_REF_PROVENANCE="explicit override on the command line"
else
  MLX_REF="$(derive_mlx_ref)"
  MLX_REF_PROVENANCE="derived from mlx-c's CMakeLists.txt FetchContent pin"
  echo "Derived MLX pin from mlx-c CMakeLists: ${MLX_REF}"
fi
echo "Vendoring MLX core at ${MLX_REF}..."
export_tree "$MLX_REPO" "$MLX_REF" vendor/mlx
MLX_COMMIT="$EXPORT_COMMIT"; MLX_TREE="$EXPORT_TREE"

# MLX's own fetch-content deps — derived from the FRESH MLX CMakeLists,
# so a pin bump pulls its dependencies forward automatically.
JSON_URL="$(derive_json_url)"
FMT_REF="$(derive_fmt_ref)"
METAL_CPP_URL="$(derive_metal_cpp_url)"
echo "Derived MLX dependency pins:"
echo "  json:     ${JSON_URL}"
echo "  fmt:      ${FMT_REF}"
echo "  metal-cpp: ${METAL_CPP_URL}"

echo "Vendoring MLX dependencies..."
fetch_archive "$JSON_URL" vendor/json
export_tree "$FMT_REPO" "$FMT_REF" vendor/fmt
FMT_COMMIT="$EXPORT_COMMIT"; FMT_TREE="$EXPORT_TREE"
fetch_archive "$METAL_CPP_URL" vendor/metal-cpp

echo "Hashing vendored trees..."
MLX_C_DIGEST="$(dir_hash vendor/mlx-c)"
MLX_DIGEST="$(dir_hash vendor/mlx)"
JSON_DIGEST="$(dir_hash vendor/json)"
FMT_DIGEST="$(dir_hash vendor/fmt)"
METAL_CPP_DIGEST="$(dir_hash vendor/metal-cpp)"
printf '%s\n' "$MLX_C_DIGEST"    > vendor/mlx-c.sha256
printf '%s\n' "$MLX_DIGEST"       > vendor/mlx.sha256
printf '%s\n' "$JSON_DIGEST"      > vendor/json.sha256
printf '%s\n' "$FMT_DIGEST"      > vendor/fmt.sha256
printf '%s\n' "$METAL_CPP_DIGEST" > vendor/metal-cpp.sha256

RUST_CHANNEL="$(awk -F'"' '/^channel/ {print $2}' rust-toolchain.toml)"
TODAY="$(date -u +%Y-%m-%d)"

cat > PINS.md <<EOF
# PINS.md — vendored pins

Updated (UTC): ${TODAY} by \`scripts/vendor.sh\`. Do not hand-edit the
recorded values; re-run the script instead.

The git tree hashes below are provenance metadata from the upstream
repositories — they cannot be recomputed from the export. The content
digests are the enforceable gate: \`scripts/pins_check.sh\` (named test
\`pins_hash_matches\`) recomputes them using the \`dir_hash\` algorithm
defined in \`scripts/vendor.sh\`, and compares against this file and the
\`vendor/*.sha256\` companions.

This file is scoped to vendored sources and the Rust channel. Toolchain
pins beyond Rust (minijinja, Node LTS) are recorded in
\`docs/decisions/DR-0001\`, not here.

## mlx-c (vendor/mlx-c)

- Repository: https://github.com/ml-explore/mlx-c
- Ref: ${MLX_C_REF}
- Commit: ${MLX_C_COMMIT}
- Git tree hash: ${MLX_C_TREE} (provenance only)
- Content digest (sha256): ${MLX_C_DIGEST} (mirrored in vendor/mlx-c.sha256)

## MLX core (vendor/mlx)

- Repository: https://github.com/ml-explore/mlx
- Ref: ${MLX_REF} (${MLX_REF_PROVENANCE})
- Commit: ${MLX_COMMIT}
- Git tree hash: ${MLX_TREE} (provenance only)
- Content digest (sha256): ${MLX_DIGEST} (mirrored in vendor/mlx.sha256)

## MLX fetch-content dependencies

Pins derived from MLX's own CMakeLists.txt at the MLX commit above.
Without these vendored trees, MLX's FetchContent declarations reach the
network at CMake configure time.

### nlohmann/json (vendor/json)

- URL: ${JSON_URL}
- Content digest (sha256): ${JSON_DIGEST} (mirrored in vendor/json.sha256)

### fmt (vendor/fmt)

- Repository: https://github.com/fmtlib/fmt
- Tag: ${FMT_REF}
- Commit: ${FMT_COMMIT}
- Git tree hash: ${FMT_TREE} (provenance only)
- Content digest (sha256): ${FMT_DIGEST} (mirrored in vendor/fmt.sha256)

### metal-cpp (vendor/metal-cpp)

- URL: ${METAL_CPP_URL}
- Content digest (sha256): ${METAL_CPP_DIGEST} (mirrored in vendor/metal-cpp.sha256)

## Hermeticity mapping

crates/mlx-ffi/build.rs passes these CMake defines, which neutralize
every network FetchContent in the mlx-c + MLX build:

- \`-DFETCHCONTENT_SOURCE_DIR_MLX=\` vendor/mlx
- \`-DFETCHCONTENT_SOURCE_DIR_JSON=\` vendor/json
- \`-DFETCHCONTENT_SOURCE_DIR_FMT=\` vendor/fmt
- \`-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=\` vendor/metal-cpp (macOS only)

## Toolchain

- Rust channel: ${RUST_CHANNEL} (rust-toolchain.toml)
EOF

echo
echo "Pins updated:"
grep -E '^- (Ref|Commit|Content digest|Rust)' PINS.md
echo
echo "Next steps after a pin bump:"
echo "  1. scripts/bindgen_check.sh        # regenerate + verify bindings.rs"
echo "  2. scripts/coverage_inventory.sh   # regenerate inventory (CI gate)"
echo "  3. cargo test -p mlx-ffi           # named tests 1-7 for this platform"
echo "  4. record the outcome in docs/decisions/DR-0001"