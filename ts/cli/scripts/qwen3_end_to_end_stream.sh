#!/usr/bin/env bash
# Named test 13. Builds and exercises only this candidate checkout.
# ALLOY_QWEN3_ARTIFACT: preinstalled, flat corpus-pinned Qwen3-4B artifact.
# ALLOY_ACCEPTANCE_OUTPUT_DIR: retained logs, CLI/RPC stream and result.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec node "$SCRIPT_DIR/real-model-acceptance.mjs" qwen3_end_to_end_stream
