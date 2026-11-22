#!/usr/bin/env python3
"""Run all implemented M3 numerical and behavioral suites under one Mac CI lease."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import m3_native_behavior
import m3_native_numerical as native
import m3_native_qualification
import m3_native_package


RUNNERS = {"numerical": native.run, "behavior": m3_native_behavior.run}
RUN_ERRORS = (OSError, ValueError, RuntimeError, subprocess.SubprocessError)


def verified_receipt(directory, profile, candidate, scope):
    path = directory / "receipt.json"
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in {
        "version": 1, "profile": profile, "candidate": candidate, "scope": scope,
    }.items()):
        raise ValueError("terminal receipt does not identify the requested execution")
    if receipt.get("status") not in ("failed", "passed"):
        raise ValueError("missing terminal status in runner receipt")
    return {"receipt": str(Path(directory.name) / path.name), "receipt_sha256": native.sha256(path),
            "status": receipt["status"]}


def run_one(profile, scope, candidate, inputs, output):
    result = {"profile": profile, "scope": scope, "status": "failed"}
    directory = output / (profile + "-" + scope)
    try:
        paths = inputs.get(profile)
        if not isinstance(paths, dict) or set(paths) != {"artifact", "reference"}:
            raise ValueError("profile input must specify artifact and reference paths")
        if any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths.values()):
            raise ValueError("profile paths must be absolute")
        args = argparse.Namespace(profile=profile, candidate=candidate, output=directory,
                                  artifact=Path(paths["artifact"]), reference=Path(paths["reference"]))
        RUNNERS[scope](args)
    except RUN_ERRORS as error:
        result["error"] = str(error)
    if directory.exists() or "error" not in result:
        try:
            result.update(verified_receipt(directory, profile, candidate, scope))
        except RUN_ERRORS as error:
            result["receipt_error"] = str(error)
    if "error" in result or "receipt_error" in result:
        result["status"] = "failed"
    return result


def qualify_profile(profile, candidate, inputs, output, runs):
    result = {"profile": profile, "status": "failed"}
    directory = output / (profile + "-qualification")
    try:
        by_scope = {r["scope"]: r for r in runs}
        if any("receipt_sha256" not in r for r in runs):
            raise ValueError("missing native evidence receipt")
        m3_native_qualification.issue(profile, candidate, inputs[profile], output, by_scope)
    except (KeyError, TypeError, *RUN_ERRORS) as error:
        result["error"] = str(error)
    if directory.exists() or "error" not in result:
        try:
            result.update(verified_receipt(directory, profile, candidate, "qualification"))
            verdict = directory / "report/verdict.json"
            if verdict.is_file():
                result["verdict_sha256"] = native.sha256(verdict)
        except RUN_ERRORS as error:
            result["receipt_error"] = str(error)
    if "error" in result or "receipt_error" in result:
        result["status"] = "failed"
    return result


def run(candidate, inputs_path, output, package=False):
    output = output.resolve()
    if output.is_relative_to(native.ROOT):
        raise ValueError("matrix output must be a new external directory")
    output.mkdir()
    index = {"version": 1, "candidate": candidate, "runs": [], "qualifications": [], "status": "failed"}
    try:
        native.verify_candidate(candidate)
        inputs_path = inputs_path.resolve(strict=True)
        if inputs_path.is_relative_to(native.ROOT):
            raise ValueError("use an external profile path inventory")
        inputs = json.loads(inputs_path.read_bytes())
        if not isinstance(inputs, dict) or inputs.keys() - native.SUITES.keys():
            raise ValueError("invalid or unexpected profiles in input inventory")
        for profile in native.SUITES:
            runs = []
            for scope in RUNNERS:
                result = run_one(profile, scope, candidate, inputs, output)
                runs.append(result)
                index["runs"].append(result)
                print(f"{profile} {scope}: {result['status']}", flush=True)
            qualification = qualify_profile(profile, candidate, inputs, output, runs)
            index["qualifications"].append(qualification)
            print(f"{profile} qualification: {qualification['status']}", flush=True)
        native.verify_candidate(candidate)
        if package:
            index["package"] = m3_native_package.publish(index, output)
        if all(r["status"] == "passed" for r in index["runs"] + index["qualifications"]):
            index["status"] = "passed"
    except (KeyError, TypeError, *RUN_ERRORS) as error:
        index["error"] = str(error)
    pending = output / "index.pending"
    m3_native_behavior.write_json(pending, index)
    pending.rename(output / "index.json")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="full expected Git commit ID")
    parser.add_argument("--inputs", type=Path, required=True,
                        help='external JSON object keyed by profile; each value has absolute "artifact" and "reference" paths')
    parser.add_argument("--output", type=Path, required=True, help="new external directory in an existing parent")
    parser.add_argument("--package", action="store_true", help="build a local production daemon for passing profiles under OUTPUT/package")
    args = parser.parse_args()
    try:
        index = run(args.candidate, args.inputs, args.output, args.package)
    except RUN_ERRORS as error:
        print(error, file=sys.stderr)
        return 1
    if "error" in index:
        print(index["error"], file=sys.stderr)
    return 0 if index["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
