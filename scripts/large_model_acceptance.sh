#!/usr/bin/env bash
# Reproducible REAL two-checkpoint M9 acceptance. No model downloads occur.
#
# Required host: native Apple Silicon macOS with enough measured free unified
# memory for the production 14B startup plan. A low-memory preflight is a
# failure, never a tiny-model or CPU substitute.
#
# Optional environment:
#   ALLOY_M9_MODELS_ROOT       default: $HOME/alloy-omlx-models
#   ALLOY_M9_MODEL_A_DIR       default: $root/alloy-qwen3-4b-4bit
#   ALLOY_M9_MODEL_B_DIR       default: $root/alloy-qwen3-14b-4bit
#   ALLOY_M9_EXPECTED_COMMIT   exact clean checkout HEAD by default
#   ALLOY_M9_OUTPUT            new JSONL path (default under ${TMPDIR:-/tmp})
#   ALLOY_M9_TIMEOUT_SECONDS   60..7200; default 1800, then 120s cleanup grace
#
# CI should upload ALLOY_M9_OUTPUT even on failure. Exit 0 requires the final
# acceptance_passed record and actual Metal prefill/decode for BOTH distinct
# frozen checkpoints. Files are pinned by bench/corpus/models.json, not by the
# local directory names. Native allocation limits are process-local only.
set -euo pipefail

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    printf '%s\n' 'M9 real large-model acceptance requires native Apple Silicon macOS; CPU/Linux is not an acceptance result.' >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(cd -- "$script_dir/.." && pwd -P)"
source_commit="$(git -C "$source_root" rev-parse HEAD)"
expected_commit="${ALLOY_M9_EXPECTED_COMMIT:-$source_commit}"
if [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ || "$source_commit" != "$expected_commit" ]]; then
    printf '%s\n' 'ALLOY_M9_EXPECTED_COMMIT must exactly match the checkout full HEAD SHA.' >&2
    exit 2
fi
if [[ -n "$(git -C "$source_root" status --porcelain=v1 --untracked-files=normal --ignore-submodules=none)" ]]; then
    printf '%s\n' 'M9 acceptance requires a clean committed checkout, including submodules and untracked source files.' >&2
    exit 2
fi

models_root="${ALLOY_M9_MODELS_ROOT:-$HOME/alloy-omlx-models}"
export ALLOY_M9_MODEL_A_DIR="${ALLOY_M9_MODEL_A_DIR:-$models_root/alloy-qwen3-4b-4bit}"
export ALLOY_M9_MODEL_B_DIR="${ALLOY_M9_MODEL_B_DIR:-$models_root/alloy-qwen3-14b-4bit}"
for model_dir in "$ALLOY_M9_MODEL_A_DIR" "$ALLOY_M9_MODEL_B_DIR"; do
    if [[ ! -d "$model_dir" ]]; then
        printf 'Missing frozen model directory: %s\n' "$model_dir" >&2
        exit 2
    fi
done

export ALLOY_M9_SOURCE_ROOT="$source_root"
export ALLOY_M9_EXPECTED_COMMIT="$expected_commit"
# option_env! records this dependency, so Cargo rebuilds the example if the
# commit changes. The binary also verifies its compiled source directory.
export ALLOY_M9_BUILD_COMMIT="$expected_commit"
export ALLOY_M9_TIMEOUT_SECONDS="${ALLOY_M9_TIMEOUT_SECONDS:-1800}"
export ALLOY_M9_OUTPUT="${ALLOY_M9_OUTPUT:-${TMPDIR:-/tmp}/alloy-m9-${expected_commit:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$.jsonl}"
if [[ -e "$ALLOY_M9_OUTPUT" ]]; then
    printf 'Refusing to overwrite existing evidence: %s\n' "$ALLOY_M9_OUTPUT" >&2
    exit 2
fi
if [[ ! -d "$(dirname -- "$ALLOY_M9_OUTPUT")" ]]; then
    printf 'Evidence parent directory must exist: %s\n' "$(dirname -- "$ALLOY_M9_OUTPUT")" >&2
    exit 2
fi
printf 'M9 acceptance commit: %s\nEvidence JSONL: %s\n' "$expected_commit" "$ALLOY_M9_OUTPUT" >&2
cd -- "$source_root"
cargo run --locked --release -p engine --example large_model_acceptance
