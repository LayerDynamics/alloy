#!/usr/bin/env python3
"""Plan the advisory M3 pilot offline. Print a plan without executing fixtures."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import NamedTuple

from m3_native_behavior import inventory
from m3_native_numerical import ROOT, SUITES


RULE_VERSION = 1
TARGETS = {
    "crates/alloyd/tests/llama_real_model.rs": ("alloyd", "llama_real_model"),
    "crates/alloyd/tests/compaction_real_model.rs": ("alloyd", "compaction_real_model"),
}


class Case(NamedTuple):
    profile: str
    package: str
    target: str
    fixture: str


FIXED = (
    Case("smollm3-3b-4bit", "alloyd", "llama_real_model", "llama3_tool_stop_context_cache"),
    Case("qwen3.5-0.8b-4bit", "alloyd", "llama_real_model", "llama3_tool_stop_context_cache"),
    Case("qwen3-14b-4bit", "alloyd", "compaction_real_model", "real_skill_change_invalidates_stale_cache_and_matches_cold"),
    Case("llama-3.1-8b-8bit", "alloyd", "compaction_real_model", "real_skill_change_invalidates_stale_cache_and_matches_cold"),
)


def select(paths, profiles):
    changed = sorted(set(paths))
    selected, omitted = [], []
    mapped = {path for path in changed if path in TARGETS}
    reasons = {TARGETS[path]: path for path in mapped}
    cases = {Case(profile, package, target, fixture)
             for profile in profiles for package, target, names in inventory(profile)
             for fixture in names}
    if not cases.issuperset(FIXED):
        raise ValueError("pinned inventory lacks a frozen comparator case")
    for case in sorted(cases):
        source = reasons.get((case.package, case.target))
        row = {**case._asdict(), "reason": source or "no changed path selects this target"}
        (selected if source else omitted).append(row)
    unknown = [{"path": p, "reason": "no v1 mapping"} for p in changed if p not in mapped]
    return {
        "rule_version": RULE_VERSION,
        "complete": bool(selected) and not unknown,
        "mapped_paths": sorted(mapped),
        "unknown_paths": unknown,
        "path_counts": {"mapped": len(mapped), "total": len(changed)},
        "selected": selected,
        "omitted": omitted,
        "fixed_comparator": [case._asdict() for case in FIXED],
    }


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def revisions(root, base, candidate):
    for revision in (base, candidate):
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            raise ValueError("provide full lowercase Git commit IDs")
        if git(root, "cat-file", "-t", revision).strip() != b"commit":
            raise ValueError(f"expected a commit: {revision}")
    bases = git(root, "merge-base", "--all", base, candidate).decode().splitlines()
    if len(bases) != 1:
        raise ValueError("expected a unique merge base")
    # Disabling rename detection emits the old and new paths as deletion/addition.
    names = git(root, "diff", "--name-only", "--no-renames", "-z", bases[0], candidate, "--")
    return bases[0], [os.fsdecode(path) for path in names.split(b"\0") if path]


def plan(base, candidate):
    ancestor, paths = revisions(ROOT, base, candidate)
    # The loaded Python tables supply the pinned inventory. Refuse to label a
    # different revision's inventory as the candidate's.
    for filename in ("m3_native_behavior.py", "m3_native_numerical.py"):
        source = Path("scripts/ci") / filename
        if git(ROOT, "show", f"{candidate}:{source.as_posix()}") != (ROOT / source).read_bytes():
            raise ValueError(f"candidate inventory source differs from loaded source: {source}")
    return {"type": "m3-native-pilot-plan", "base": base, "candidate": candidate,
            "merge_base": ancestor, **select(paths, SUITES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="full PR base commit")
    parser.add_argument("--candidate", required=True, help="full PR head commit")
    args = parser.parse_args()
    try:
        result = plan(args.base, args.candidate)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
