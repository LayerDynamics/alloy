#!/usr/bin/env bash
# W4-scoped evidence collection, not a final acceptance decision.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TERM_COLOR=never
case "$(uname -s)" in Linux) lane=linux ;; Darwin) lane=macos ;; *) echo "Linux/macOS required" >&2; exit 2 ;; esac
out="${ALLOY_W4_OUTPUT:-$PWD/target/w4-acceptance/$lane}"
mkdir -p "$out"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  test -z "$(git status --porcelain)" || { echo "Clean committed checkout required" >&2; exit 2; }
  candidate="$(git rev-parse HEAD)"; source_mode=git-checkout
  if test -n "${ALLOY_W4_CANDIDATE:-}"; then test "$candidate" = "$ALLOY_W4_CANDIDATE" || { echo "Candidate mismatch" >&2; exit 2; }; fi
else
  candidate="${ALLOY_W4_CANDIDATE:?CI copied checkout must supply candidate SHA}"; source_mode=ci-copied-checkout
fi
[[ "$candidate" =~ ^[0-9a-f]{40}$ ]] || { echo "Full SHA required" >&2; exit 2; }
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$out/status.txt"' EXIT
{ printf 'candidate_sha=%s\nplatform=%s\nsource_mode=%s\n' "$candidate" "$lane" "$source_mode"; uname -a; rustc -Vv; cargo -V; } | tee "$out/context.txt"
cat > "$out/scope.txt" <<'SCOPE'
P2-07: real W4 builtin scope/output/deadline checks with controlled local fixtures.
P2-09: W4 builtin cancellation, two sessions, partial-output/ownership cleanup and
SQLite progress. Generation/approval-wait/plugin pause integration remains W5/W6.
A passing script is not final Linux+macOS/native-regression review or sign-off.
SCOPE
hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
{ printf '%s\n' Cargo.toml Cargo.lock scripts/w4_acceptance.sh .github/workflows/ci.yaml;
  find crates/tool-registry crates/harness-core -type f \( -name '*.rs' -o -name 'Cargo.toml' \);
} | LC_ALL=C sort -u | while IFS= read -r path; do printf '%s  %s\n' "$(hash_file "$path")" "$path"; done > "$out/w4-source-sha256.txt"
run_gate() {
  local label="$1" package="$2" target="$3" name="$4"
  cargo test --locked -p "$package" --test "$target" -- --list 2>&1 | tee "$out/$label-discovery.log"
  test "$(grep -Fxc "$name: test" "$out/$label-discovery.log")" -eq 1
  cargo test --locked -p "$package" --test "$target" "$name" -- --exact --nocapture 2>&1 | tee "$out/$label-run.log"
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored;' "$out/$label-run.log"
}
run_gate p2-07 tool-registry builtins builtin_scope_and_output_limits
run_gate p2-09 harness-core tool_cancellation tool_cancel_releases_owned_resources
printf 'W4 scoped gates passed for %s on %s. Acceptance review remains pending.\n' "$candidate" "$lane"
