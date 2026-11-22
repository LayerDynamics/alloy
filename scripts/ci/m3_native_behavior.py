#!/usr/bin/env python3
"""Run the pinned native behavioral suite under mac_locked_step.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from m3_native_numerical import SUITES, build, execute, pinned_run, require_single_pass, sha256


# Each ID denotes one executable assertion set. The release specification
# independently determines which assertion sets satisfy its requirements.
SHARED = (
    ("alloyd", "llama_real_model", (
        "llama3_tool_stop_context_cache", "llama3_long_context_stream_cancel_memory",
        "native_validated_context_boundary",
    )),
    ("alloyd", "compaction_real_model", (
        "real_model_compaction_preserves_history_and_continues",
        "real_model_ssd_prefix_survives_process_restart",
        "real_model_generated_history_has_verified_canonical_bytes",
        "real_skill_change_invalidates_stale_cache_and_matches_cold",
        "real_rpc_skill_load_changes_next_generation",
    )),
)
GEMMA4 = (
    ("alloyd", "gemma4_chat", (
        "gemma4_daemon_text_session", "gemma4_publisher_tool_prompt_parity",
        "gemma4_small_budget_generation",
    )),
    ("inference-infra", "gemma4_memory", ("gemma4_native_memory_tiers",)),
)


def inventory(profile):
    if profile not in SUITES:
        raise ValueError(f"unknown profile: {profile}")
    if profile == "gemma-4-e4b-it-4bit":
        return SHARED + GEMMA4
    return SHARED + (
        ("inference-infra", "qwen35_memory", ("qwen35_native_memory_tiers",)),
        ("engine", "qwen35_engine", ("qwen35_real_engine_tool_proposals",)),
    )


# Each category corresponds to assertions in the named native test. Every
# named test remains required even if another test covers the same category.
COVERAGE = {
    "llama3_tool_stop_context_cache": ("policy_gated_task", "stops"),
    "llama3_long_context_stream_cancel_memory": ("streaming", "cancellation"),
    "native_validated_context_boundary": ("long_context", "exact_counting"),
    "real_model_compaction_preserves_history_and_continues": ("compaction",),
    "real_model_ssd_prefix_survives_process_restart": ("restart",),
    "real_model_generated_history_has_verified_canonical_bytes": ("cached_uncached",),
    "real_skill_change_invalidates_stale_cache_and_matches_cold": ("cached_uncached",),
    "real_rpc_skill_load_changes_next_generation": ("cached_uncached",),
    "gemma4_daemon_text_session": ("streaming", "exact_counting"),
    "gemma4_publisher_tool_prompt_parity": ("tool_schema",),
    "gemma4_small_budget_generation": ("memory8_gi_b",),
    "gemma4_native_memory_tiers": ("memory8_gi_b", "memory36_gi_b"),
    "qwen35_native_memory_tiers": ("memory8_gi_b", "memory36_gi_b"),
    "qwen35_real_engine_tool_proposals": ("tool_schema",),
}


def fixture_env(selected):
    # Restart children set ALLOY_TEST_REAL_SSD_PHASE themselves. Inheriting
    # it here would bypass the parent assertion that crosses two processes.
    return {**{k: v for k, v in os.environ.items() if not k.startswith("ALLOY_")}, **selected}


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def run_suite(out, record, selected, groups):
    observations = []
    env = fixture_env(selected)
    for package, target, names in groups:
        try:
            binary = build("test", package, target, out, record)
            build_error = None
        except (RuntimeError, ValueError, OSError) as error:
            build_error = str(error)
        build_command = record["commands"][-1]
        for name in names:
            evidence = {k: record[k] for k in (
                "profile", "candidate", "manifest_sha256", "artifact_revision",
                "reference_sha256", "cargo_lock_sha256", "pins_sha256", "host", "architecture",
            )}
            evidence.update(fixture_id=name, environment=selected, build=build_command,
                            executable=record["binaries"].get(target))
            passed = False
            detail = build_error
            if build_error is None:
                try:
                    log = execute([binary, name, "--exact", "--ignored", "--test-threads=1", "--show-output"],
                                  out, name, record, env)
                    require_single_pass(log.read_text(), name)
                    passed = True
                    detail = f"{target}::{name} passed all native assertions"
                except (OSError, RuntimeError, ValueError) as error:
                    detail = str(error)
                evidence["execution"] = record["commands"][-1]
            evidence.update(passed=passed, detail=detail)
            path = out / (name + ".json")
            write_json(path, evidence)
            observations.append(dict(fixture_id=name, passed=passed, detail=detail, evidence_sha256=sha256(path)))
            print(f"{record['profile']} {name}: {'passed' if passed else detail}", flush=True)
    return observations


def inspect_profile(out, record):
    environment = {k: record[k] for k in ("cargo_lock_sha256", "pins_sha256", "host", "architecture")}
    environment["rustc_sha256"] = sha256(out / "rustc-version.stdout")
    encoded = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    dependencies = hashlib.sha256(encoded).hexdigest()
    binary = build("example", "alloyd", "m3_profile_identity", out, record)
    path = execute([binary, record["profile"], record["artifact_directory"], "40960",
                    record["candidate"], dependencies], out, "profile-identity", record)
    inspected = json.loads(path.read_bytes())
    identity = inspected["identity"]
    if inspected["profile"] != record["profile"] or inspected["context"] != 40960 or (
        identity["candidate_revision"], identity["dependencies_sha256"]
    ) != (record["candidate"], dependencies):
        raise ValueError("native profile inspection did not match the requested configuration")
    record.update(dependency_inputs=environment, dependencies_sha256=dependencies,
                  profile_identity=path.name, profile_identity_sha256=sha256(path))


def run(args):
    with pinned_run(args, "behavior") as (out, record, selected):
        selected.update(ALLOY_REAL_MODEL=record["artifact_directory"], ALLOY_REAL_PROFILE=args.profile)
        if args.profile in ("ornith-1.5-9b-4bit", "qwen3.5-0.8b-4bit"):
            selected.update(ALLOY_HYBRID_ARTIFACT=record["artifact_directory"],
                            ALLOY_HYBRID_PROFILE="ornith" if args.profile.startswith("ornith-") else "upstream")
        record["fixture_environment"] = selected
        inspect_profile(out, record)
        observations = run_suite(out, record, selected, inventory(args.profile))
        path = out / "fixtures.json"
        write_json(path, observations)
        record["fixtures_sha256"] = sha256(path)
        failures = [item["fixture_id"] for item in observations if not item["passed"]]
        if failures:
            raise ValueError("failed native fixtures: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(SUITES), required=True)
    parser.add_argument("--candidate", required=True, help="full expected Git commit ID")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new external output directory")
    try:
        run(parser.parse_args())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
