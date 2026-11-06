#!/usr/bin/env bash
# Named test 14. Restarts the actual candidate daemon against the same SQLite DB.
# Verifies unchanged prior event contents and a streamed, history-dependent turn.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec node "$SCRIPT_DIR/real-model-acceptance.mjs" session_restart_real_model
