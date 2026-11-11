#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd -P)"
OUT="${ALLOY_W5_OUTPUT:?Set ALLOY_W5_OUTPUT outside the source tree or under target}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
# Cleanup must never remove evidence-shaped source files or an ancestor's files.
if [[ "$OUT" == / ]]; then
  echo 'Set ALLOY_W5_OUTPUT outside the source tree or under target' >&2
  exit 1
fi
case "$ROOT/" in "$OUT/"*)
  echo 'Set ALLOY_W5_OUTPUT outside the source tree or under target' >&2; exit 1;;
esac
case "$OUT/" in "$ROOT/"*)
  case "$OUT/" in "$ROOT/target/"*) ;; *)
    echo 'Set ALLOY_W5_OUTPUT outside the source tree or under target' >&2; exit 1;;
  esac;;
esac
PLUGIN_TESTS=(
  plugin_tool_hooks_order_veto_annotation plugin_protocol_timeout_and_sandbox
  unsolicited_output_between_calls_fails_closed
  raced_preconstructed_response_fails_closed
  plugin_receives_canonical_payload_and_cannot_replace_result
  hardening::plugin_failures_reap_real_children_and_stop_later_hooks
  hardening::plugin_registration_is_atomic_and_drop_reaps
  hardening::plugin_cancellation_isolated_and_after_failure_is_warning_only
  hardening::plugin_capability_subsets_fragmentation_and_real_zero_effect_marker
  hardening::plugin_inherited_descriptor_is_closed_on_exec
  hardening::plugin_invalid_limits_and_duplicate_registration_spawn_nothing
  hardening::plugin_blocked_stdin_deadline_reaps_owned_process
  hardening::plugin_high_fd_above_lowered_limit_is_not_inherited
)
if ! mkdir "$OUT/.plugin-acceptance.lock"; then
  echo 'Acceptance output is already in use; choose a separate output directory.' >&2
  exit 1
fi
finish_collection() {
  local status=$?
  trap - EXIT
  rm -f -- "$OUT/report.json.tmp" || status=$?
  if [[ "$status" -ne 0 ]]; then
    rm -f -- "$OUT/report.json" || status=$?
  fi
  # Release ownership only after failure cleanup, so a new run cannot have its
  # report removed by this run's EXIT trap.
  if ! rmdir "$OUT/.plugin-acceptance.lock"; then
    rm -f -- "$OUT/report.json"
    exit 1
  fi
  exit "$status"
}
trap finish_collection EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Invalidate proof first, but never glob caller logs (a wrapper may already have
# OUT/driver.log open). Only remove recorded or exact legacy collector outputs.
rm -f -- "$OUT/report.json" "$OUT/report.json.tmp"
if [[ -f "$OUT/collector-files.txt" && ! -L "$OUT/collector-files.txt" ]]; then
  while IFS= read -r artifact || [[ -n "$artifact" ]]; do
    case "$artifact" in */*|''|.|..)
      echo 'Invalid collector output manifest entry' >&2; exit 1;;
    esac
    case "$artifact" in *.log|*.command.txt) rm -f -- "$OUT/$artifact";; *)
      echo 'Invalid collector output manifest entry' >&2; exit 1;;
    esac
  done < "$OUT/collector-files.txt"
fi
# Older collectors have no manifest. Include their pre-module and interim
# colon-separated names as well as the current normalized names.
for name in external-reaper linux-reaper owned-process format compile clippy strict-json sandbox-alias sandbox sandboxed-host artifact-sandbox fileport-failures fileport-concurrency fileport-stdio fileport-exhaustion fileport-malformed fileport-adversarial hardlink-integrity \
    all-targets doctests rustdoc "${PLUGIN_TESTS[@]}"; do
  for stem in "$name" "${name//::/_}" "${name#hardening::}"; do
    rm -f -- "$OUT/$stem.log" "$OUT/$stem.command.txt"
  done
done
rm -f -- "$OUT/collector-files.txt" "$OUT/candidate.txt" "$OUT/platform.txt" \
  "$OUT/source-before.sha256" "$OUT/source-after.sha256" \
  "$OUT/protocol-discovery.txt" "$OUT/unit-discovery.txt" \
  "$OUT/sandbox-discovery.txt" "$OUT/sandbox-platform.txt" "$OUT/artifact-discovery.txt" "$OUT/owned-discovery.txt" "$OUT/linux-reaper-discovery.txt"
: > "$OUT/collector-files.txt"
# Clear old proof before prerequisite/candidate validation can fail.
command -v python3 >/dev/null 2>&1 || {
  echo "W5 acceptance requires python3 in the execution environment; provision the runner/container." >&2
  exit 127
}
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
GATE_LOGS=()
gate() {
  local name="$1"; shift
  # Rust module separators are valid selectors but invalid artifact filenames.
  local artifact_name="${name//::/_}"
  GATE_LOGS+=("$artifact_name.log")
  printf '%s\n' "$artifact_name.command.txt" "$artifact_name.log" >> "$OUT/collector-files.txt"
  printf '%s\n' "$*" > "$OUT/$artifact_name.command.txt"
  "$@" 2>&1 | tee "$OUT/$artifact_name.log"
}
gate format cargo fmt --all -- --check
gate compile cargo check --locked -p harness-core --all-targets
gate clippy cargo clippy --locked -p harness-core --all-targets -- -D warnings
cargo test --locked -p harness-core --test plugin_hooks -- --list > "$OUT/protocol-discovery.txt"
for name in "${PLUGIN_TESTS[@]}"; do
  grep -Fx "$name: test" "$OUT/protocol-discovery.txt"
  gate "$name" cargo test --locked -p harness-core --test plugin_hooks "$name" -- --exact --nocapture
done
gate strict-json cargo test --locked -p harness-core plugin_host::host::strict_json::duplicate_escaped_nested_and_trailing_json_rejects -- --exact --nocapture
# Explicit discovery prevents a misspelled unit filter from passing with zero tests.
cargo test --locked -p harness-core --lib -- --list > "$OUT/unit-discovery.txt"
grep -Fx 'plugin_host::host::strict_json::duplicate_escaped_nested_and_trailing_json_rejects: test' "$OUT/unit-discovery.txt"
if [[ "$(uname -s)" == Darwin ]]; then
  cargo test --locked -p harness-core --test owned_process -- --list > "$OUT/owned-discovery.txt"
  grep -Fx 'macos_identity_signals_survive_exec_reaping_and_pid_mismatch: test' "$OUT/owned-discovery.txt"
  gate owned-process cargo test --locked -p harness-core --test owned_process macos_identity_signals_survive_exec_reaping_and_pid_mismatch -- --exact --nocapture
  grep -Fx 'hardening::plugin_external_reaper_owns_supervisor_statuses: test' "$OUT/protocol-discovery.txt"
  gate external-reaper cargo test --locked -p harness-core --test plugin_hooks hardening::plugin_external_reaper_owns_supervisor_statuses -- --exact --nocapture
  cargo test --locked -p harness-core --test plugin_sandbox -- --list > "$OUT/sandbox-discovery.txt"
  grep -Fx 'plugin_protocol_timeout_and_sandbox: test' "$OUT/sandbox-discovery.txt"
  grep -Fx 'hardening::plugin_alias_read_works_but_symlink_cannot_escape: test' "$OUT/sandbox-discovery.txt"
  gate sandbox-alias cargo test --locked -p harness-core --test plugin_sandbox hardening::plugin_alias_read_works_but_symlink_cannot_escape -- --exact --nocapture
  gate sandbox cargo test --locked -p harness-core --test plugin_sandbox plugin_protocol_timeout_and_sandbox -- --exact --nocapture
  grep -Fx 'hardening::plugin_actual_sandboxed_host_denies_undeclared_io_network_and_fork: test' "$OUT/protocol-discovery.txt"
  gate sandboxed-host cargo test --locked -p harness-core --test plugin_hooks hardening::plugin_actual_sandboxed_host_denies_undeclared_io_network_and_fork -- --exact --nocapture
  grep -Fx 'hardening::fileport_helper_failure_and_cancellation_reap_exact_child: test' "$OUT/protocol-discovery.txt"
  grep -Fx 'hardening::fileport_concurrent_unrelated_exec_inherits_no_endpoints: test' "$OUT/protocol-discovery.txt"
  grep -Fx 'hardening::plugin_rejects_writable_hardlinks_and_helper_overlap_before_spawn: test' "$OUT/protocol-discovery.txt"
  grep -Fx 'hardening::fileport_works_when_standard_descriptors_are_closed: test' "$OUT/protocol-discovery.txt"
  grep -Fx 'hardening::fileport_descriptor_exhaustion_fails_closed_without_fd_leaks: test' "$OUT/protocol-discovery.txt"
  grep -Fx 'hardening::fileport_rejects_malformed_helper_message: test' "$OUT/protocol-discovery.txt"
  grep -Fx 'hardening::fileport_rejects_wrong_sender_and_complex_messages: test' "$OUT/protocol-discovery.txt"
  gate fileport-failures cargo test --locked -p harness-core --test plugin_hooks hardening::fileport_helper_failure_and_cancellation_reap_exact_child -- --exact --nocapture
  gate fileport-concurrency cargo test --locked -p harness-core --test plugin_hooks hardening::fileport_concurrent_unrelated_exec_inherits_no_endpoints -- --exact --nocapture
  gate hardlink-integrity cargo test --locked -p harness-core --test plugin_hooks hardening::plugin_rejects_writable_hardlinks_and_helper_overlap_before_spawn -- --exact --nocapture
  gate fileport-stdio cargo test --locked -p harness-core --test plugin_hooks hardening::fileport_works_when_standard_descriptors_are_closed -- --exact --nocapture
  gate fileport-exhaustion cargo test --locked -p harness-core --test plugin_hooks hardening::fileport_descriptor_exhaustion_fails_closed_without_fd_leaks -- --exact --nocapture
  gate fileport-malformed cargo test --locked -p harness-core --test plugin_hooks hardening::fileport_rejects_malformed_helper_message -- --exact --nocapture
  gate fileport-adversarial cargo test --locked -p harness-core --test plugin_hooks hardening::fileport_rejects_wrong_sender_and_complex_messages -- --exact --nocapture
  cargo test --locked -p alloy-process-owner --test macos_broker_recovery -- --list > "$OUT/broker-recovery-discovery.txt"
  BROKER_TESTS=(
    broker_loss_before_launch_recovers_without_replay
    broker_loss_during_launch_releases_other_launches
    broker_loss_during_execution_preserves_control
    broker_loss_during_cancellation_preserves_cleanup
    restart_budget_refuses_excess_reconnects_then_recovers
    stale_generation_never_launches_or_replays_a_command
    peer_audit_authentication_rejects_unrelated_code
    live_supervisor_capacity_survives_broker_restart
    unconfirmed_stopped_supervisor_stays_charged_until_proven_exit
    repeated_confirmed_presend_cancellation_does_not_exhaust_capacity
    inconclusive_host_identity_preserves_other_live_service
    concurrent_host_retirement_does_not_break_registration
    completion_published_before_supervisor_exit_keeps_real_status
    malformed_message_does_not_retire_authenticated_service
    native_framework_command_retains_bootstrap_access
    registration_and_workloads_create_no_embedding_host_children
  )
  for name in "${BROKER_TESTS[@]}"; do
    grep -Fx "$name: test" "$OUT/broker-recovery-discovery.txt"
  done
  gate broker-recovery cargo test --locked -p alloy-process-owner --test macos_broker_recovery -- --nocapture --test-threads=1
  cargo test --locked -p harness-core --test tool_artifacts -- --list > "$OUT/artifact-discovery.txt"
  grep -Fx 'plugin_artifact::sandboxed_after_hook_cannot_replace_artifact_or_its_bytes: test' "$OUT/artifact-discovery.txt"
  gate artifact-sandbox cargo test --locked -p harness-core --test tool_artifacts plugin_artifact::sandboxed_after_hook_cannot_replace_artifact_or_its_bytes -- --exact --nocapture
else
  printf '%s\n' 'NOT APPLICABLE: macOS Seatbelt lane required separately.' > "$OUT/sandbox-platform.txt"
  cargo test --locked -p harness-core --test linux_reaper -- --list > "$OUT/linux-reaper-discovery.txt"
  grep -Fx 'linux_plugin_reaping_survives_competing_reapers: test' "$OUT/linux-reaper-discovery.txt"
  gate linux-reaper cargo test --locked -p harness-core --test linux_reaper linux_plugin_reaping_survives_competing_reapers -- --exact --nocapture
  cargo test --locked -p alloy-process-owner --lib -- --list > "$OUT/owner-unit-discovery.txt"
  grep -Fx 'spawn::linux_supervisor_tests::deferred_wait_reaps_exact_supervisor_after_timeout: test' "$OUT/owner-unit-discovery.txt"
  gate owner-deferred-reaper cargo test --locked -p alloy-process-owner --lib spawn::linux_supervisor_tests::deferred_wait_reaps_exact_supervisor_after_timeout -- --exact --nocapture
fi
gate all-targets cargo test --locked -p harness-core --all-targets
gate doctests cargo test --locked -p harness-core --doc
gate rustdoc env RUSTDOCFLAGS='-D warnings' cargo doc --locked -p harness-core --no-deps
snapshot > "$OUT/source-after.sha256"
cmp "$OUT/source-before.sha256" "$OUT/source-after.sha256"
python3 - "$OUT" "${GATE_LOGS[@]}" <<'PY'
import hashlib, json, os, pathlib, re, sys
p=pathlib.Path(sys.argv[1]); logs={}
for name in sorted(sys.argv[2:]):
    f=p/name
    logs[name]=hashlib.sha256(f.read_bytes()).hexdigest()
    text=f.read_text(errors='replace')
    if f.name not in ['format.log','compile.log','clippy.log','rustdoc.log']:
        summaries=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored', text)
        if not summaries or not any(int(a)>0 for a,_,_ in summaries): raise SystemExit(f'No executed tests: {f}')
        if any(int(b) or int(c) for _,b,c in summaries): raise SystemExit(f'Failed/ignored gate: {f}')
(p/'report.json.tmp').write_text(json.dumps({'candidate':os.environ['ALLOY_W5_CANDIDATE'],
    'platform':sys.platform,'source_sha256':(p/'source-after.sha256').read_text().strip(),
    'gate_status':'passed','security_review':'requires explicit reviewer sign-off',
    'logs_sha256':logs}, indent=2)+'\n')
os.replace(p/'report.json.tmp', p/'report.json')
PY
printf '%s\n' 'W5 named gate collection passed. Other-platform results and explicit security review remain separate.'
