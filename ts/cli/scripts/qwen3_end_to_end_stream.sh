#!/usr/bin/env bash
# Named test: qwen3_end_to_end_stream
#
# Quantized Qwen3-4B cold load needs oMLX-method Affine4-resident weights
# (memory-omlx-parity / M8). Override the engine tree with ALLOY_ENGINE_ROOT
# when the CLI lives on 7-ts-cli but alloyd must come from the memory worktree.
set -euo pipefail

CLI_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CLI_ROOT/../.." && pwd)"
REAL_HOME="${REAL_HOME:-$HOME}"
# Prefer the memory-parity worktree when present so admit uses Affine4 residency.
ENGINE_ROOT="${ALLOY_ENGINE_ROOT:-}"
if [[ -z "$ENGINE_ROOT" && -d "$REAL_HOME/alloy-omlx-memory/crates/alloyd" ]]; then
  ENGINE_ROOT="$REAL_HOME/alloy-omlx-memory"
fi
if [[ -z "$ENGINE_ROOT" ]]; then
  ENGINE_ROOT="$ROOT"
fi
ARTIFACT="${ALLOY_QWEN3_ARTIFACT:-$REAL_HOME/alloy-omlx-models/alloy-qwen3-4b-4bit}"
if [[ ! -d "$ARTIFACT" || ! -f "$ARTIFACT/config.json" ]]; then
  echo "set ALLOY_QWEN3_ARTIFACT to a flat Qwen3-4B-4bit directory (HF hub snapshots with blob symlinks are rejected)"
  exit 1
fi
NODE26_BIN="${REAL_HOME}/.asdf/installs/nodejs/26.13.0/bin"
if [[ -d "$NODE26_BIN" ]]; then
  export PATH="$NODE26_BIN:$PATH"
fi

echo "building alloyd (release) from $ENGINE_ROOT …"
(
  cd "$ENGINE_ROOT"
  cargo build --locked -p alloyd --release
)
ALLOYD="$ENGINE_ROOT/target/release/alloyd"

WORKDIR="$(mktemp -d /tmp/alloy-e2e-XXXXXX)"
ALLOYD_PID=""
cleanup() {
  if [[ -n "${ALLOYD_PID}" ]] && kill -0 "$ALLOYD_PID" 2>/dev/null; then
    kill "$ALLOYD_PID" 2>/dev/null || true
    wait "$ALLOYD_PID" 2>/dev/null || true
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

# Isolate CLI settings under a temp HOME after cargo has finished.
export HOME="$WORKDIR/home"
mkdir -p "$HOME/.alloy"
SOCK="$HOME/.alloy/alloyd.sock"
DB="$HOME/.alloy/sessions.sqlite"

cat >"$HOME/.alloy/settings.json" <<EOF
{
  "version": 1,
  "profile": "qwen3-4b-4bit",
  "socket": "$SOCK",
  "db": "$DB",
  "artifact": "$ARTIFACT",
  "alloyd_bin": "$ALLOYD",
  "corpus_models": "$ROOT/bench/corpus/models.json"
}
EOF

"$ALLOYD" \
  --artifact "$ARTIFACT" \
  --profile qwen3-4b-4bit \
  --socket "$SOCK" \
  --db "$DB" \
  --corpus-models "$ROOT/bench/corpus/models.json" \
  >"$WORKDIR/alloyd.log" 2>&1 &
ALLOYD_PID=$!

for _ in $(seq 1 180); do
  if [[ -S "$SOCK" ]]; then
    break
  fi
  if ! kill -0 "$ALLOYD_PID" 2>/dev/null; then
    echo "alloyd exited before binding socket"
    cat "$WORKDIR/alloyd.log"
    exit 1
  fi
  sleep 1
done
[[ -S "$SOCK" ]] || {
  echo "alloyd failed to bind $SOCK"
  cat "$WORKDIR/alloyd.log"
  exit 1
}

cd "$CLI_ROOT"
npm run build
OUT="$WORKDIR/out.txt"
set +e
# settings.json already points at SOCK; pass --socket for clarity when HOME is isolated.
node bin/alloy.js --socket "$SOCK" -m "Reply with exactly the word: pong" >"$OUT" 2>"$WORKDIR/cli.err"
CLI_STATUS=$?
set -e

if [[ "$CLI_STATUS" -ne 0 ]]; then
  echo "alloy exited with status $CLI_STATUS"
  echo "--- cli.err ---"
  cat "$WORKDIR/cli.err"
  echo "--- out ---"
  cat "$OUT" || true
  echo "--- alloyd.log (tail) ---"
  tail -n 80 "$WORKDIR/alloyd.log" || true
  exit "$CLI_STATUS"
fi

if [[ ! -s "$OUT" ]]; then
  echo "expected streamed assistant text on stdout"
  echo "--- cli.err ---"
  cat "$WORKDIR/cli.err"
  echo "--- alloyd.log (tail) ---"
  tail -n 80 "$WORKDIR/alloyd.log" || true
  exit 1
fi

if ! grep -qi 'pong' "$OUT"; then
  echo "expected reply to contain 'pong'"
  echo "--- out ---"
  cat "$OUT"
  echo "--- cli.err ---"
  cat "$WORKDIR/cli.err"
  exit 1
fi

echo "qwen3_end_to_end_stream PASS"
