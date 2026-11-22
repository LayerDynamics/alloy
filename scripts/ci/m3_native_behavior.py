#!/usr/bin/env python3
"""Run the pinned native behavioral suite under mac_locked_step.py."""
import argparse
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
    return SHARED + (("inference-infra", "qwen35_memory", ("qwen35_native_memory_tiers",)),)


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


def run(args):
    with pinned_run(args, "behavior") as (out, record, selected):
        selected.update(ALLOY_REAL_MODEL=record["artifact_directory"], ALLOY_REAL_PROFILE=args.profile)
        record["fixture_environment"] = selected
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
