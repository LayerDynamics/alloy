#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${W5_EVIDENCE_DIR:?Set W5_EVIDENCE_DIR to an external evidence directory}"
exec > >(tee "$W5_EVIDENCE_DIR/validation.log") 2>&1
printf 'candidate: '; git rev-parse HEAD
git status --short
rustc --version
cargo --version
uname -a
cargo metadata --locked --no-deps --format-version 1 > "$W5_EVIDENCE_DIR/metadata.json"
python3 - "$W5_EVIDENCE_DIR/metadata.json" <<'PY'
import json, pathlib, sys
m = json.load(open(sys.argv[1]))
members = set(m['workspace_members'])
for p in m['packages']:
    if p['id'] not in members: continue
    for t in p['targets']:
        source = pathlib.Path(t['src_path'])
        if not source.is_file(): raise SystemExit(f'Missing target entry: {source}')
        print(p['name'], ','.join(t['kind']), source)
PY
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
for test in plugin_tool_hooks_order_veto_annotation plugin_protocol_timeout_and_sandbox; do
    list=$(cargo test --locked -p harness-core --test plugin_hooks -- --list)
    grep -Fx "$test: test" <<< "$list"
    cargo test --locked -p harness-core --test plugin_hooks "$test" -- --exact --nocapture
done
cargo test --locked -p harness-core --test plugin_hooks
if [[ $(uname -s) == Darwin ]]; then
    list=$(cargo test --locked -p harness-core --test plugin_sandbox -- --list)
    grep -Fx 'plugin_protocol_timeout_and_sandbox: test' <<< "$list"
    cargo test --locked -p harness-core --test plugin_sandbox plugin_protocol_timeout_and_sandbox -- --exact --nocapture
else
    printf '%s\n' 'Mac sandbox acceptance: NOT RUN on this host; remains pending.'
fi
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --doc
RUSTDOCFLAGS='-D warnings' cargo doc --locked -p engine -p harness-core -p alloyd --no-deps
printf '%s\n' 'Command suite finished; this is NOT automatic W5 acceptance or sandbox review.'
