#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
OUT="${ALLOY_W5_OUTPUT:?Set ALLOY_W5_OUTPUT outside the source tree or under target}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CANDIDATE="${ALLOY_W5_CANDIDATE:-$(git rev-parse HEAD)}"
export ALLOY_W5_CANDIDATE="$CANDIDATE"
case "$CANDIDATE" in *[!0-9a-f]*|'') echo 'Invalid candidate SHA' >&2; exit 1;; esac
[[ ${#CANDIDATE} -eq 40 ]] || { echo 'Full SHA required' >&2; exit 1; }
if [[ -d .git ]]; then
  [[ "$(git rev-parse HEAD)" == "$CANDIDATE" ]] || { echo 'Candidate mismatch' >&2; exit 1; }
  [[ -z "$(git status --porcelain)" ]] || { echo 'Acceptance requires a clean candidate; use validation for development.' >&2; exit 1; }
fi
printf '%s\n' "$CANDIDATE" > "$OUT/candidate.txt"
{ date -u; uname -a; rustc --version; cargo --version; } > "$OUT/platform.txt"
snapshot() {
  python3 - "$ROOT" <<'PY'
import hashlib, os, pathlib, sys
root = pathlib.Path(sys.argv[1]); h = hashlib.sha256(); paths = []
for base, dirs, names in os.walk(root):
    dirs[:] = sorted(d for d in dirs if d not in {'.git', 'target'})
    paths.extend(pathlib.Path(base)/n for n in names)
for p in sorted(paths):
    rel = p.relative_to(root)
    if '.git' in rel.parts or 'target' in rel.parts: continue
    if p.is_symlink():
        h.update(str(rel).encode() + b'\0link\0' + str(p.readlink()).encode() + b'\0')
    elif p.is_file():
        h.update(str(rel).encode() + b'\0file\0')
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1048576), b''): h.update(b)
        h.update(b'\0')
print(h.hexdigest())
PY
}
snapshot > "$OUT/source-before.sha256"
gate() {
  local name="$1"; shift
  printf '%s\n' "$*" > "$OUT/$name.command.txt"
  "$@" 2>&1 | tee "$OUT/$name.log"
}
gate format cargo fmt --all -- --check
gate compile cargo check --locked -p harness-core --all-targets
gate clippy cargo clippy --locked -p harness-core --all-targets -- -D warnings
cargo test --locked -p harness-core --test plugin_hooks -- --list > "$OUT/protocol-discovery.txt"
for name in plugin_tool_hooks_order_veto_annotation plugin_protocol_timeout_and_sandbox \
    w5_failures_reap_real_children_and_stop_later_hooks \
    w5_registration_is_atomic_and_drop_reaps \
    w5_cancellation_isolated_and_after_failure_is_warning_only \
    w5_capability_subsets_fragmentation_and_real_zero_effect_marker \
    w5_inherited_descriptor_is_closed_on_exec \
    w5_invalid_limits_and_duplicate_registration_spawn_nothing \
    w5_blocked_stdin_deadline_reaps_owned_process; do
  grep -Fx "$name: test" "$OUT/protocol-discovery.txt"
  gate "$name" cargo test --locked -p harness-core --test plugin_hooks "$name" -- --exact --nocapture
done
gate strict-json cargo test --locked -p harness-core plugin_host::strict_json::duplicate_escaped_nested_and_trailing_json_rejects -- --exact --nocapture
# Explicit discovery prevents a misspelled unit filter from passing with zero tests.
cargo test --locked -p harness-core --lib -- --list > "$OUT/unit-discovery.txt"
grep -Fx 'plugin_host::strict_json::duplicate_escaped_nested_and_trailing_json_rejects: test' "$OUT/unit-discovery.txt"
if [[ "$(uname -s)" == Darwin ]]; then
  cargo test --locked -p harness-core --test plugin_sandbox -- --list > "$OUT/sandbox-discovery.txt"
  grep -Fx 'plugin_protocol_timeout_and_sandbox: test' "$OUT/sandbox-discovery.txt"
  grep -Fx 'w5_alias_read_works_but_symlink_cannot_escape: test' "$OUT/sandbox-discovery.txt"
  gate sandbox-alias cargo test --locked -p harness-core --test plugin_sandbox w5_alias_read_works_but_symlink_cannot_escape -- --exact --nocapture
  gate sandbox cargo test --locked -p harness-core --test plugin_sandbox plugin_protocol_timeout_and_sandbox -- --exact --nocapture
  grep -Fx 'w5_actual_sandboxed_host_denies_undeclared_io_network_and_fork: test' "$OUT/protocol-discovery.txt"
  gate sandboxed-host cargo test --locked -p harness-core --test plugin_hooks w5_actual_sandboxed_host_denies_undeclared_io_network_and_fork -- --exact --nocapture
else
  printf '%s\n' 'NOT APPLICABLE: macOS Seatbelt lane required separately.' > "$OUT/sandbox-platform.txt"
fi
gate all-targets cargo test --locked -p harness-core --all-targets
gate doctests cargo test --locked -p harness-core --doc
gate rustdoc env RUSTDOCFLAGS='-D warnings' cargo doc --locked -p harness-core --no-deps
snapshot > "$OUT/source-after.sha256"
cmp "$OUT/source-before.sha256" "$OUT/source-after.sha256"
python3 - "$OUT" <<'PY'
import hashlib, json, os, pathlib, re, sys
p=pathlib.Path(sys.argv[1]); logs={}
for f in sorted(p.glob('*.log')):
    logs[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
for f in p.glob('*.log'):
    text=f.read_text(errors='replace')
    if f.name not in ['format.log','compile.log','clippy.log','rustdoc.log']:
        summaries=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored', text)
        if not summaries or not any(int(a)>0 for a,_,_ in summaries): raise SystemExit(f'No executed tests: {f}')
        if any(int(b) or int(c) for _,b,c in summaries): raise SystemExit(f'Failed/ignored gate: {f}')
(p/'report.json').write_text(json.dumps({'candidate':os.environ['ALLOY_W5_CANDIDATE'],
    'platform':sys.platform,'source_sha256':(p/'source-after.sha256').read_text().strip(),
    'gate_status':'passed','security_review':'requires explicit reviewer sign-off',
    'logs_sha256':logs}, indent=2)+'\n')
PY
printf '%s\n' 'W5 named gate collection passed. Other-platform results and explicit security review remain separate.'
