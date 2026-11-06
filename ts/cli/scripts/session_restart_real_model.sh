#!/usr/bin/env bash
# Named test: session_restart_real_model
#
# Same engine-root rule as qwen3_end_to_end_stream.sh: prefer
# alloy-omlx-memory (Affine4-resident) unless ALLOY_ENGINE_ROOT is set.
set -euo pipefail

CLI_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CLI_ROOT/../.." && pwd)"
REAL_HOME="${REAL_HOME:-$HOME}"
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

WORKDIR="$(mktemp -d /tmp/alloy-e2e-restart-XXXXXX)"
ALLOYD_PID=""
cleanup() {
  if [[ -n "${ALLOYD_PID}" ]] && kill -0 "$ALLOYD_PID" 2>/dev/null; then
    kill "$ALLOYD_PID" 2>/dev/null || true
    wait "$ALLOYD_PID" 2>/dev/null || true
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

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

start_daemon() {
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
      return 0
    fi
    if ! kill -0 "$ALLOYD_PID" 2>/dev/null; then
      echo "alloyd exited before binding socket"
      cat "$WORKDIR/alloyd.log"
      exit 1
    fi
    sleep 1
  done
  echo "alloyd failed to bind $SOCK"
  cat "$WORKDIR/alloyd.log"
  exit 1
}

stop_daemon() {
  if [[ -n "${ALLOYD_PID}" ]] && kill -0 "$ALLOYD_PID" 2>/dev/null; then
    kill "$ALLOYD_PID" 2>/dev/null || true
    wait "$ALLOYD_PID" 2>/dev/null || true
  fi
  ALLOYD_PID=""
  rm -f "$SOCK"
}

start_daemon

cd "$CLI_ROOT"
npm run build

OUT1="$WORKDIR/out1.txt"
ERR1="$WORKDIR/cli1.err"
node bin/alloy.js --socket "$SOCK" -m "Reply with exactly the word: alpha" >"$OUT1" 2>"$ERR1"
test -s "$OUT1"
grep -qi 'alpha' "$OUT1"

SESSION_ID="$(sed -n 's/^session \(sess-[0-9][0-9]*\) .*/\1/p' "$ERR1" | head -1)"
if [[ -z "$SESSION_ID" ]]; then
  echo "failed to parse session id from cli stderr"
  cat "$ERR1"
  exit 1
fi

stop_daemon
start_daemon

OUT2="$WORKDIR/out2.txt"
ERR2="$WORKDIR/cli2.err"
node bin/alloy.js --socket "$SOCK" --resume "$SESSION_ID" -m "Reply with exactly the word: beta" >"$OUT2" 2>"$ERR2"
test -s "$OUT2"
grep -qi 'beta' "$OUT2"
grep -q "resumed $SESSION_ID" "$ERR2"

echo "session_restart_real_model PASS"
