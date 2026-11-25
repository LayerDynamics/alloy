#!/usr/bin/env python3
"""Compare archived native executions with a versioned, candidate-specific claim.

This is an offline comparison. Neither a manifest nor a matching archive
establishes a trusted producer or grants approval. Each case in the claim
specifies profile/package/target/fixture, manifest_sha256, reference_sha256,
fixture_environment, expected_defect, and markers keyed by defect ID. Each
marker has a literal panic location and exact, consecutive message lines.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from m3_native_behavior import inventory
from m3_native_numerical import SUITES, sha256
from m3_native_qualification import check_execution, checked_json
from m3_native_pilot import Case, plan


def classify(name, code, stdout, stderr, markers):
    # libtest writes its terminal result to stdout. Match complete lines so a
    # generic panic, filtered test or truncated transcript cannot earn credit.
    lines = stdout.rstrip("\r\n").splitlines()
    summaries = [line for line in lines if line.startswith("test result: ")]
    named = [line for line in lines if line.startswith(f"test {name} ... ")]
    if (len(summaries) != 1 or not lines or lines[-1] != summaries[0]
            or [line for line in lines if re.fullmatch(r"running \d+ tests?", line)] != ["running 1 test"]):
        return "inconclusive", None
    for rc, status, summary, passed, failed in ((0, "ok", "ok", 1, 0), (101, "FAILED", "FAILED", 0, 1)):
        prefix = f"test result: {summary}. {passed} passed; {failed} failed; 0 ignored; 0 measured; "
        suffix = r"\d+ filtered out(?:; finished in \d+(?:\.\d+)?s)?"
        if code == rc and named == [f"test {name} ... {status}"] and re.fullmatch(re.escape(prefix) + suffix, summaries[0]):
            break
    else:
        return "inconclusive", None
    if code == 0:
        return "passed", None
    # Rust may include the thread ID; only that numeric field is variable.
    matches = []
    for defect, marker in markers.items():
        panic = re.compile(r"thread '" + re.escape(name) + r"'(?: \(\d+\))? panicked at "
                           + re.escape(marker["location"]) + ":")
        message = marker["lines"]
        for stream in (lines, stderr.splitlines()):
            if any(panic.fullmatch(line) and stream[i+1:i+1+len(message)] == message
                   for i, line in enumerate(stream)):
                matches.append(defect)
    if len(matches) == 1:
        return "product_assertion", matches[0]
    return "inconclusive", None


def observation(root, candidate, case):
    name, profile = case["fixture"], case["profile"]
    directory = root / (profile + "-behavior")
    receipt = json.loads((directory / "receipt.json").read_bytes())
    if (receipt["version"], receipt["candidate"], receipt["profile"], receipt["scope"]) != (1, candidate, profile, "behavior"):
        raise ValueError("wrong receipt identity")
    if receipt["status"] not in ("passed", "failed"):
        raise ValueError("scope has no terminal receipt")
    for key in ("manifest_sha256", "reference_sha256", "fixture_environment"):
        if receipt[key] != case[key]:
            raise ValueError(f"claim differs in {key}")
    rows = checked_json(directory / "fixtures.json", receipt["fixtures_sha256"])
    row, = [row for row in rows if row["fixture_id"] == name]
    evidence = checked_json(directory / (name + ".json"), row["evidence_sha256"])
    if (evidence["candidate"], evidence["profile"], evidence["fixture_id"]) != (candidate, profile, name):
        raise ValueError("wrong fixture identity")
    if (evidence["environment"] != case["fixture_environment"] or
            any(evidence[key] != receipt[key] for key in ("manifest_sha256", "reference_sha256",
                "artifact_revision", "pins_sha256", "cargo_lock_sha256", "host", "architecture"))):
        raise ValueError("fixture and scope provenance differ")
    build = evidence["build"]
    command = evidence["execution"]
    binary = receipt["binaries"][case["target"]]
    if binary != evidence["executable"] or sha256(Path(binary["path"])) != binary["sha256"]:
        raise ValueError("fixture executable changed")
    if (build not in receipt["commands"] or build["exit_code"] != 0 or build["argv"] !=
            ["cargo", "test", "--no-run", "--locked", "--release", "-p", case["package"],
             "--test", case["target"], "--message-format=json"]):
        raise ValueError("fixture build differs")
    check_execution(directory, build)
    if (command not in receipt["commands"] or command["argv"] !=
            [binary["path"], name, "--exact", "--ignored", "--test-threads=1", "--show-output"]):
        raise ValueError("fixture command differs")
    stdout = check_execution(directory, command).read_text()
    outcome, defect = classify(name, command["exit_code"], stdout,
                               (directory / command["stderr"]).read_text(), case["markers"])
    if evidence["passed"] is not row["passed"] or row["passed"] is not (outcome == "passed"):
        raise ValueError("recorded pass flag differs from harness evidence")
    return {"outcome": outcome, "observed_defect": defect,
            "matches_expected_defect": defect is not None and defect == case["expected_defect"],
            "evidence_sha256": row["evidence_sha256"],
            **({"reason": "no unique verified pass or recognized product assertion"}
               if outcome == "inconclusive" else {})}


def replay(root, manifest_bytes):
    claim = json.loads(manifest_bytes)
    if (claim["version"] != 1 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", claim["candidate"])
            or not isinstance(claim["cases"], list) or not claim["cases"]):
        raise ValueError("expected a version 1 claim with candidate and cases")
    seen = set()
    observations = []
    for case in claim["cases"]:
        fields = {k: case[k] for k in ("profile", "package", "target", "fixture")}
        key = tuple(fields.values())
        profile = case["profile"]
        if (key in seen or profile not in SUITES or
                not any((p, t) == (case["package"], case["target"]) and case["fixture"] in names
                        for p, t, names in inventory(profile))):
            raise ValueError(f"duplicate or unknown case: {key}")
        seen.add(key)
        markers = case["markers"]
        if (not isinstance(markers, dict) or case["expected_defect"] not in markers
                or not isinstance(case["fixture_environment"], dict)
                or any(not re.fullmatch(r"[a-f0-9]{64}", case[k]) for k in ("manifest_sha256", "reference_sha256"))):
            raise ValueError("incomplete case claim")
        for defect, marker in markers.items():
            if (not defect or not marker["location"] or not isinstance(marker["lines"], list) or
                    not marker["lines"] or not all(isinstance(l, str) and l for l in marker["lines"])):
                raise ValueError("expected a literal panic location and nonempty message lines")
        try:
            row = observation(root, claim["candidate"], case)
        except (ValueError, OSError, KeyError, TypeError) as error:
            row = {"outcome": "inconclusive", "reason": str(error), "matches_expected_defect": False}
        observations.append({**fields, **row})
    return {"type": "m3-pilot-archive-comparison", "candidate": claim["candidate"],
            "claim_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "approval_credit": False, "observations": observations}


def compare(selection, replayed):
    """Measure both selectors against the very same archived observations."""
    if selection["candidate"] != replayed["candidate"]:
        raise ValueError("planner and archive candidates differ")

    def key(row):
        return Case(*(row[field] for field in Case._fields))

    observations = {key(row): row for row in replayed["observations"]}
    failures = {case for case, row in observations.items() if row["outcome"] == "product_assertion"}
    strategies = {}
    for strategy, field in (("fixed", "fixed_comparator"), ("change_based", "selected")):
        selected = {key(row) for row in selection[field]}
        detected = selected & failures
        unknown = [case for case in sorted(selected) if case not in observations
                   or observations[case]["outcome"] == "inconclusive"]
        strategies[strategy] = {
            "selected_native_cases": len(selected),
            "unknown_native_cases": [case._asdict() for case in unknown],
            "detected_failing_fixtures": [case._asdict() for case in sorted(detected)],
            "missed_known_failing_fixtures": [case._asdict() for case in sorted(failures - selected)],
            "expected_defect_matches": sum(observations[case]["matches_expected_defect"] for case in detected),
            "known_failing_fixture_denominator": len(failures),
            "archive_advice": "reject" if detected else "inconclusive",
        }
    return {"base": selection["base"], "merge_base": selection["merge_base"],
            "rule_version": selection["rule_version"], "fully_mapped_request": selection["complete"],
            "path_counts": selection["path_counts"], "synthetic_controls": "unobserved",
            "strategies": strategies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--claim-manifest", required=True, type=Path)
    parser.add_argument("--compare-base", help="compare both strategies using the full PR diff from this base")
    args = parser.parse_args()
    try:
        result = replay(args.matrix.resolve(strict=True), args.claim_manifest.read_bytes())
        if args.compare_base:
            result["comparison"] = compare(plan(args.compare_base, result["candidate"]), result)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"invalid archive or claim: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
