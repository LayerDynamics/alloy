#!/usr/bin/env python3
"""Collect candidate-bound W2 evidence; passing tests are not a W2 sign-off."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import subprocess
import time

SCHEMA_VERSION = 2
IMPLEMENTATION_COMMITS = {
    "slab_decoder_and_ownership": "2009b0eddce81150deabbf52be697e7d7ff997ed",
    "strict_rustdoc_correction": "50d262b75e9b7637984299086ca01cd9a7c06bd2",
    "borrowed_rpc_transport": "362aeef902c921f9f47a8c8153df777e708d7814",
}
SOURCE_PATHS = (
    "scripts/w2_evidence.py",
    "scripts/tests/test_w2_evidence.py",
    "docs/decisions/w2-tool-output-memory-contract.md",
    "docs/decisions/w2-parser-placement-and-syntax.md",
    "docs/acceptance/w2-deployed-path-boundary.md",
    "crates/engine/src/slab_parser/parity.rs",
    "crates/alloyd/src/main.rs",
    "crates/alloyd/src/rpc_transport_config.rs",
    "crates/alloyd/tests/rpc_transport_args.rs",
    "crates/engine/src/parser_allocation.rs",
    "crates/engine/src/slab_parser/mod.rs",
    "crates/engine/src/runtime.rs",
    "crates/engine/src/generation/mod.rs",
    "crates/engine/src/types/mod.rs",
    "crates/engine/src/backends/qwen/mod.rs",
    "crates/engine/src/backends/qwen/output.rs",
    "crates/engine/src/backends/qwen/memory.rs",
    "crates/engine/tests/tool_rendering.rs",
    "crates/families/qwen3/src/tool_calls.rs",
    "crates/families/qwen3/fixture/tool_calls.json",
    "crates/alloyd/src/rpc/slab_transport.rs",
    "crates/alloyd/src/service.rs",
    "crates/alloyd/src/server.rs",
    "bench/corpus/models.json",
)


@dataclass(frozen=True)
class TestGate:
    name: str
    package: str
    target: str
    selector: str
    required: tuple[str, ...] = ()
    minimum: int = 1
    exact: bool = False
    evidence_kind: str = "fixture_or_scripted_test"

    def command(self, discovery: bool) -> list[str]:
        command = ["cargo", "test", "--locked", "--color", "never", "-p", self.package]
        if self.target == "lib":
            command.append("--lib")
        elif self.target.startswith("bin:"):
            command += ["--bin", self.target.split(":", 1)[1]]
        else:
            command += ["--test", self.target]
        if self.selector:
            command.append(self.selector)
        command.append("--")
        if self.exact:
            command.append("--exact")
        command += (["--list", "--format", "terse", "--color", "never"] if discovery else
                    ["--format", "pretty", "--color", "never", "--test-threads=1"])
        return command


def gates() -> tuple[TestGate, ...]:
    return (
        TestGate("parser-shared-fixture-parity", "engine", "lib", "slab_parser::parity::", (
            "accepted_shared_fixtures_match_at_every_utf8_split",
            "rejected_shared_fixtures_fail_in_both_streaming_parsers",
            "alternate_coder_function_parameter_syntax_is_explicitly_unsupported",
        ), minimum=6, evidence_kind="shared_fixture_semantic_projection_not_identical_publication_schedule"),
        TestGate("daemon-transport-options", "alloyd", "bin:alloyd", "rpc_transport_config::tests::", (
            "paired_options_preserve_exact_values",
            "actual_service_configuration_helper_applies_only_the_supplied_policy",
        ), minimum=8, evidence_kind="binary_configuration_and_actual_service_fixture"),
        TestGate("daemon-transport-cli", "alloyd", "rpc_transport_args", "", (
            "rpc_transport_valid_pair_reaches_the_existing_artifact_guard",
            "rpc_transport_incomplete_pair_is_rejected_before_artifact_setup",
        ), minimum=5, evidence_kind="candidate_built_binary_options_no_model_start"),
        TestGate("P2-03-compat", "qwen3", "tool_calls", "qwen3_tool_call_chunk_boundaries",
                 ("qwen3_tool_call_chunk_boundaries",), exact=True,
                 evidence_kind="legacy_family_byte_fragment_parser"),
        TestGate("P2-03-slab", "engine", "lib",
                 "slab_parser::tests::every_utf8_chunk_boundary_preserves_raw_arguments_and_text",
                 ("every_utf8_chunk_boundary_preserves_raw_arguments_and_text",), exact=True,
                 evidence_kind="admitted_slab_utf8_fragment_parser"),
        TestGate("P2-04-slab", "engine", "lib",
                 "generation::w2_slab_delivery_tests::malformed_later_slab_call_publishes_zero_effects",
                 ("malformed_later_slab_call_publishes_zero_effects",), exact=True,
                 evidence_kind="production_worker_scripted_inference_effect_counter"),
        TestGate("P2-04-compat", "engine", "lib",
                 "generation::w2_completion_gate_tests::qwen3_malformed_tool_calls_never_execute",
                 ("qwen3_malformed_tool_calls_never_execute",), exact=True,
                 evidence_kind="compatibility_session_effect_counter"),
        TestGate("P2-05", "engine", "tool_rendering", "tool_context_exact_count_and_identity",
                 ("tool_context_exact_count_and_identity",), exact=True,
                 evidence_kind="shared_renderer_and_actual_fixture_tokenizer"),
        TestGate("slab-allocation-layout", "engine", "lib", "parser_allocation::", (
            "exact_layout_is_reserved_and_released_once",
            "allocator_failure_releases_the_preexisting_reservation",
            "odd_regions_account_for_every_alignment_byte",
            "atomic_startup_partition_does_not_double_reserve_parser_bytes",
            "bounded_handoff_keeps_the_slab_charged_until_consumer_drop",
        ), minimum=20, evidence_kind="allocation_layout_and_actual_admission_ledger"),
        TestGate("output-memory-plan", "engine", "lib", "w2_output_memory_plan_tests", (
            "parser_slab_uses_the_actual_layout_including_padding_once",
            "startup_plan_partitions_parser_and_non_parser_without_double_charge",
        ), minimum=9),
        TestGate("runtime-slab-handoff", "engine", "lib", "w2_runtime_parser_allocation_tests", (
            "pending_finish_releases_slab_and_remainder_but_not_stream_output",
            "decoder_receives_exclusive_slab_ownership",
            "wrong_pending_parser_configuration_is_rejected_without_losing_owner",
        ), minimum=11),
        TestGate("slab-parser", "engine", "lib", "slab_parser::tests", (
            "escaped_equivalent_duplicate_keys_are_rejected",
            "unicode_name_and_surrogate_pair_strings_validate_without_value_trees",
            "call_body_exact_fit_and_one_byte_over_fail_without_leaking_storage",
            "a_second_call_over_descriptor_capacity_invalidates_the_batch",
            "cumulative_output_limit_counts_previously_emitted_text",
            "failed_text_delivery_poisons_the_parser",
        ), minimum=12),
        TestGate("slab-delivery", "engine", "lib", "w2_slab_delivery_tests", (
            "successful_batch_outlives_decoder_worker_and_stream",
            "malformed_later_slab_call_publishes_zero_effects",
            "slab_final_usage_failure_publishes_zero_effects",
            "slab_cleanup_failure_publishes_zero_effects",
            "text_is_available_while_slab_finish_is_blocked",
            "stream_drop_unblocks_slab_publication_and_releases_reservation",
            "cancellation_after_a_valid_slab_call_publishes_zero_effects",
            "truncated_call_after_a_valid_slab_call_publishes_zero_effects",
        ), minimum=8, evidence_kind="production_worker_scripted_inference_owned_batch"),
        TestGate("qwen-decoder-hooks", "engine", "lib", "qwen_parser_hooks_", (
            "qwen_parser_hooks_supplied_slab_decodes_and_outlives_the_decoder",
            "qwen_parser_hooks_snapshot_all_explicit_capacities_and_plan_slab_once",
        ), minimum=8, evidence_kind="actual_qwen_decoder_fixture_tokens_not_model_inference"),
        TestGate("compatibility-completion-gates", "engine", "lib", "w2_completion_gate_tests", (
            "cancellation_after_a_proposal_publishes_zero_proposals",
            "cleanup_failure_publishes_zero_proposals",
        ), minimum=8),
        TestGate("tool-rendering", "engine", "tool_rendering", "", (
            "tool_context_exact_count_and_identity",
            "schema_identity_changes_even_when_template_and_token_ids_collide",
        ), minimum=7),
        TestGate("schema-cache-key-gates", "engine", "lib", "w2_tool_schema_cache_tests", (
            "changed_or_removed_schemas_fail_the_production_compatibility_gate",
            "disk_keys_bind_schema_identity_and_preserve_the_text_only_key",
        ), minimum=2, evidence_kind="compatibility_predicate_and_disk_key_tests_not_ssd_restore"),
        TestGate("host-cache-schema-change", "engine", "lib",
                 "cache::tests::tool_schema_change_invalidates_actual_retained_prefix",
                 ("tool_schema_change_invalidates_actual_retained_prefix",), exact=True,
                 evidence_kind="host_reference_qwen_cache_not_mlx_or_ssd"),
        TestGate("host-cache-schema-toggle", "engine", "lib",
                 "cache::tests::adding_and_removing_tool_schemas_invalidates_actual_retained_prefix",
                 ("adding_and_removing_tool_schemas_invalidates_actual_retained_prefix",), exact=True,
                 evidence_kind="host_reference_qwen_cache_not_mlx_or_ssd"),
        TestGate("qwen-tool-renderer", "engine", "lib",
                 "backends::qwen::shared_rendering_wiring_tests::tool_context_reaches_the_production_qwen_renderer",
                 ("tool_context_reaches_the_production_qwen_renderer",), exact=True,
                 evidence_kind="production_adapter_fixture_rendering_not_inference"),
        TestGate("rpc-borrowed-transport", "alloyd", "lib", "rpc::slab_transport::tests", (
            "one_byte_short_is_rejected_before_any_destination_write",
            "source_reservation_remains_live_through_write_and_flush",
            "unix_socket_roundtrip_uses_the_borrowed_notification_writer",
        ), minimum=6, evidence_kind="borrowed_codec_and_actual_unix_socket_fixture"),
        TestGate("rpc-notification-dispatch", "alloyd", "lib", "w2_notification_dispatch_tests", (
            "real_notification_dispatch_requires_explicit_slab_transport_limits",
            "real_notification_dispatch_routes_slab_batches_to_borrowed_writer",
        ), minimum=3),
        TestGate("rpc-write-failure-ownership", "alloyd", "lib", "w2_connection_writer_tests", (
            "partial_frame_failure_prevents_appending_a_fatal_response",
            "failed_slab_transport_retains_source_until_owner_drop_and_cannot_append",
        ), minimum=5),
    )


def quality_checks() -> tuple[tuple[str, list[str], dict[str, str]], ...]:
    return (
        ("collector-self-tests", [sys.executable, "-B", "-m", "unittest", "discover",
                                  "-s", "scripts/tests", "-p", "test_w2_evidence.py", "-v"], {}),
        ("format", ["cargo", "fmt", "--all", "--", "--check"], {}),
        ("workspace-check", ["cargo", "check", "--locked", "--workspace", "--all-targets"], {}),
        ("workspace-clippy", ["cargo", "clippy", "--locked", "--workspace", "--all-targets", "--", "-D", "warnings"], {}),
        ("workspace-regressions", ["cargo", "test", "--locked", "--workspace"], {}),
        ("engine-rustdoc", ["cargo", "doc", "--locked", "-p", "engine", "--no-deps"], {"RUSTDOCFLAGS": "-D warnings"}),
        ("alloyd-rustdoc", ["cargo", "doc", "--locked", "-p", "alloyd", "--no-deps"], {"RUSTDOCFLAGS": "-D warnings"}),
    )


LISTING = re.compile(r"^(\S+): test$")
OUTCOME = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored(?:,.*)?)\s*$")
SUMMARY = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;")


def parse_listing(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := LISTING.fullmatch(line.strip()))]


def discovery_errors(gate: TestGate, names: list[str]) -> list[str]:
    errors = []
    if not names:
        errors.append("zero tests discovered")
    if len(names) != len(set(names)):
        errors.append("duplicate discovered test names")
    if len(names) < gate.minimum:
        errors.append(f"discovered {len(names)} tests; expected at least {gate.minimum}")
    for required in gate.required:
        matching = [name for name in names if name == required or name.endswith("::" + required)]
        if len(matching) != 1:
            errors.append(f"required test must be discovered exactly once: {required}")
    return errors


def execution_evidence(text: str, discovered: list[str]) -> dict:
    outcomes = [(m.group(1), m.group(2)) for line in text.splitlines() if (m := OUTCOME.fullmatch(line.strip()))]
    summaries = [m.groups() for line in text.splitlines() if (m := SUMMARY.match(line.strip()))]
    passed = [name for name, status in outcomes if status == "ok"]
    errors = []
    if not passed:
        errors.append("zero tests observed passing")
    if len(outcomes) != len({name for name, _ in outcomes}):
        errors.append("duplicate execution results")
    if any(status != "ok" for _, status in outcomes):
        errors.append("failed or ignored test observed")
    if Counter(passed) != Counter(discovered):
        errors.append("passing results do not exactly match discovery")
    if len(summaries) != 1:
        errors.append("expected exactly one libtest result summary")
    else:
        status, ok, failed, ignored, measured, _ = summaries[0]
        if status != "ok" or int(failed) or int(ignored) or int(measured):
            errors.append("summary reports failure, ignored tests, or measured benchmarks")
        if int(ok) != len(passed):
            errors.append("summary passed count differs from named passing results")
    return {"observed_passed_tests": passed, "validation_errors": errors}


def python_selftest_errors(text: str) -> list[str]:
    summaries = re.findall(r"^Ran (\d+) tests? in ", text, re.MULTILINE)
    errors = []
    if len(summaries) != 1 or int(summaries[0]) < 32:
        errors.append("collector self-tests did not run at least 32 cases")
    if re.search(r"^OK$", text, re.MULTILINE) is None:
        errors.append("collector self-tests failed or skipped required coverage")
    return errors


class CandidateChanged(RuntimeError):
    pass


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def ensure_candidate(root: Path, candidate: str) -> None:
    if git(root, "rev-parse", "HEAD") != candidate:
        raise CandidateChanged("HEAD changed from the pinned candidate")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise CandidateChanged("candidate checkout is no longer clean")


def validate_output_directory(root: Path, requested: Path) -> Path:
    root = root.expanduser().resolve()
    output = requested.expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("evidence must be outside the candidate checkout")
    if output.exists():
        raise ValueError("evidence output directory already exists; choose a new directory")
    return output


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CommandRun:
    command: list[str]
    returncode: int | None
    log: Path
    log_sha256: str
    elapsed_seconds: float
    timed_out: bool = False
    spawn_error: str | None = None
    env_overrides: dict[str, str] | None = None

    def record(self) -> dict:
        value = asdict(self)
        value["log"] = self.log.name
        return value

    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.spawn_error is None


def run_logged(command: list[str], root: Path, log: Path, timeout: float,
               overrides: dict[str, str] | None = None) -> CommandRun:
    env_overrides = {"CARGO_TERM_COLOR": "never", **(overrides or {})}
    env = os.environ.copy()
    env.update(env_overrides)
    started = time.monotonic()
    returncode = None
    timed_out = False
    spawn_error = None
    with log.open("wb") as output:
        try:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            spawn_error = str(error)
            output.write(("collector spawn error: " + spawn_error + "\n").encode())
        else:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait()
                output.write(b"\ncollector: owned command process group timed out\n")
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
    return CommandRun(command, returncode, log, hash_file(log), time.monotonic() - started,
                      timed_out, spawn_error, env_overrides)


def collect_test_gate(gate: TestGate, execute) -> dict:
    result = {"gate": gate.name, "evidence_kind": gate.evidence_kind,
              "required_tests": list(gate.required), "minimum_tests": gate.minimum,
              "passed": False, "status": "failed"}
    listing = execute("list", gate.command(True), {})
    result["discovery"] = listing.record()
    discovered = parse_listing(listing.log.read_text(errors="replace"))
    result["discovered_tests"] = discovered
    errors = discovery_errors(gate, discovered)
    if not listing.ok():
        errors.append("test discovery command failed")
    if errors:
        result["validation_errors"] = errors
        return result
    execution = execute("run", gate.command(False), {})
    result["execution"] = execution.record()
    evidence = execution_evidence(execution.log.read_text(errors="replace"), discovered)
    result.update(evidence)
    if not execution.ok():
        result["validation_errors"].append("test execution command failed")
    result["passed"] = not result["validation_errors"]
    result["status"] = "passed" if result["passed"] else "failed"
    return result


def acceptance_gaps() -> list[dict]:
    return [
        {"id": "syntax-scope", "status": "candidate_evidence_pending",
         "detail": "Shared-fixture parity and parser placement are documented and tested. Artifact-wide Qwen3/Coder scope is not inferred; alternate function/parameter syntax remains explicitly unsupported."},
        {"id": "mlx-schema-cache-restore", "status": "not_collected",
         "detail": "Host-reference cache tests and SSD-key unit tests do not establish native MLX retained-cache/SSD restore acceptance after schema changes."},
        {"id": "deployed-tool-path", "status": "not_collected",
         "detail": "Scripted inference, direct decoder tests, and Unix-socket fixtures are not one candidate-built trained-model daemon tool-loop acceptance run."},
        {"id": "daemon-transport-policy", "status": "candidate_configuration_evidence_pending",
         "detail": "Paired RPC transport flags are wired to the real daemon constructor. Actual deployment must supply both flags; tests do not attest a running model daemon or remove the fail-closed supervisor."},
        {"id": "final-review", "status": "pending",
         "detail": "A reviewer must reconcile source, gates, provenance, and remaining W2 requirements. This collector cannot self-sign W2 complete."},
    ]


def acceptance_boundary(results: list[dict]) -> dict:
    passed = {value["gate"] for value in results if value.get("passed")}
    layers = {
        "qwen_decoder_fixture": ["qwen-decoder-hooks"],
        "production_worker_scripted_inference": ["slab-delivery", "runtime-slab-handoff"],
        "notification_dispatch_and_unix_socket_fixtures": ["rpc-borrowed-transport", "rpc-notification-dispatch", "rpc-write-failure-ownership"],
        "daemon_configuration_no_model_boot": ["daemon-transport-options", "daemon-transport-cli"],
        "supported_shared_parser_fixtures": ["parser-shared-fixture-parity"],
    }
    return {
        "layers": {name: {"gates": gates, "passed": set(gates) <= passed,
                           "trained_model_tool_loop_evidence": False}
                   for name, gates in layers.items()},
        "candidate_built_trained_model_daemon_tool_loop": {
            "status": "not_collected", "artifact": None,
            "reason": "No required tool-aware supervisor/authorization execution evidence is produced by these test gates.",
        },
        "acceptance_sign_off": "pending_independent_review",
    }


def write_report(output: Path, report: dict) -> None:
    temporary = output / "w2-evidence.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "w2-evidence.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", help="optional full SHA; must equal clean HEAD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("timeout must be positive and finite")
    root = Path(git(args.repo.expanduser().resolve(), "rev-parse", "--show-toplevel")).resolve()
    candidate = git(root, "rev-parse", "HEAD")
    if args.candidate is not None and args.candidate != candidate:
        parser.error("--candidate must equal HEAD exactly")
    try:
        ensure_candidate(root, candidate)
        output = validate_output_directory(root, args.out)
    except (ValueError, CandidateChanged) as error:
        parser.error(str(error))
    for label, commit in IMPLEMENTATION_COMMITS.items():
        if subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", commit, candidate],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            parser.error("candidate does not contain implementation checkpoint: " + label)
    blobs = {}
    for path in SOURCE_PATHS:
        try:
            blobs[path] = git(root, "rev-parse", "--verify", candidate + ":" + path)
        except subprocess.CalledProcessError:
            blobs[path] = None
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": SCHEMA_VERSION, "candidate_sha": candidate,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "implementation_checkpoints_in_history": IMPLEMENTATION_COMMITS,
        "source_blobs": blobs, "results": [], "quality_checks": [],
        "candidate_unchanged_and_clean": True, "recorded_tests_passed": False,
        "quality_checks_passed": False, "collection_passed": False,
        "w2_complete": False, "acceptance_status": "pending_review",
        "remaining_required_review": acceptance_gaps(),
        "requirements_to_gates": {"P2-03": ["P2-03-compat", "P2-03-slab"],
                                  "P2-04": ["P2-04-slab", "P2-04-compat"], "P2-05": ["P2-05"]},
        "scope": "Requested slab layouts and selected borrowed RPC transport; not RSS/OS OOM prevention, all RPC paths, or tool authorization.",
        "candidate_check_scope": "Clean HEAD is checked before and after every command; this is not an immutable-worktree or adversarial concurrent-mutation proof.",
    }
    report["missing_source_blobs"] = [path for path, blob in blobs.items() if blob is None]
    write_report(output, report)

    def execute(label, phase, command, overrides):
        ensure_candidate(root, candidate)
        result = run_logged(command, root, output / f"{label}.{phase}.log",
                            args.timeout_seconds, overrides)
        try:
            ensure_candidate(root, candidate)
        except CandidateChanged:
            report["integrity_failure_command"] = {
                "gate": label, "phase": phase, "execution": result.record(),
            }
            raise
        return result

    try:
        for tool in ("rustc", "cargo"):
            run = execute("toolchain-" + tool, "version", [tool, "--version"], {})
            report[tool] = run.record()
            if not run.ok():
                raise RuntimeError("toolchain command failed: " + tool)
        for gate in gates():
            result = collect_test_gate(gate, lambda phase, cmd, env, gate=gate:
                                       execute(gate.name, phase, cmd, env))
            report["results"].append(result)
            print(gate.name, "PASS" if result["passed"] else "FAIL", flush=True)
            write_report(output, report)
        for label, command, overrides in quality_checks():
            run = execute(label, "check", command, overrides)
            errors = (python_selftest_errors(run.log.read_text(errors="replace"))
                      if label == "collector-self-tests" else [])
            result = {"gate": label, "passed": run.ok() and not errors, "execution": run.record(),
                      "validation_errors": errors,
                      "evidence_kind": ("collector_self_tests_with_nonzero_count" if label == "collector-self-tests"
                                        else "command_exit_status_not_named_test_evidence")}
            report["quality_checks"].append(result)
            print(label, "PASS" if result["passed"] else "FAIL", flush=True)
            write_report(output, report)
        ensure_candidate(root, candidate)
    except CandidateChanged as error:
        report["candidate_unchanged_and_clean"] = False
        report["collection_error"] = str(error)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        report["collection_error"] = str(error)
    report["recorded_tests_passed"] = (len(report["results"]) == len(gates())
                                       and all(value["passed"] for value in report["results"]))
    report["quality_checks_passed"] = (len(report["quality_checks"]) == len(quality_checks())
                                       and all(value["passed"] for value in report["quality_checks"]))
    report["collection_passed"] = (report["candidate_unchanged_and_clean"]
                                   and report["recorded_tests_passed"] and report["quality_checks_passed"]
                                   and "collection_error" not in report
                                   and not report["missing_source_blobs"])
    report["acceptance_boundary"] = acceptance_boundary(report["results"])
    passed_ids = {value["gate"] for value in report["results"] if value.get("passed")}
    for gap in report["remaining_required_review"]:
        if gap["id"] == "syntax-scope" and "parser-shared-fixture-parity" in passed_ids:
            gap["status"] = "fixture_parity_passed_artifact_scope_review_pending"
        if gap["id"] == "daemon-transport-policy" and {"daemon-transport-options", "daemon-transport-cli"} <= passed_ids:
            gap["status"] = "configuration_tests_passed_deployment_not_attested"
    report["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_report(output, report)
    print("Evidence:", output)
    print("W2 acceptance remains pending review; collection success is not sign-off.")
    return 0 if report["collection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
