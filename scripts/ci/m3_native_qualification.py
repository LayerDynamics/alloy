"""Evaluate freshly captured native evidence. Release provenance is established by the executing job."""
import argparse
import json
from pathlib import Path

from m3_native_behavior import COVERAGE, inventory, inspect_profile, write_json
from m3_native_numerical import SUITES, build, execute, pinned_run, require_single_pass, sha256


def checked_json(path, expected_digest):
    if not path.is_file() or sha256(path) != expected_digest:
        raise ValueError(f"unexpected or missing bytes: {path}")
    return json.loads(path.read_bytes())


def check_execution(directory, entry):
    for stream in ("stdout", "stderr"):
        path = directory / entry[stream]
        if path.parent != directory or sha256(path) != entry[stream + "_sha256"]:
            raise ValueError(f"changed execution log: {path}")
    return directory / entry["stdout"]


def native_pass(directory, command, binary, name):
    expected = [binary["path"], name, "--exact", "--ignored", "--test-threads=1", "--show-output"]
    if command["argv"] != expected or sha256(Path(binary["path"])) != binary["sha256"]:
        raise ValueError(f"incorrect executable or command: {name}")
    log = check_execution(directory, command)
    if command["exit_code"] == 0:
        try:
            require_single_pass(log.read_text(), name)
            return True
        except ValueError:
            pass  # A successful process can have zero matching tests.
    return False


def fixture_rows(directory, receipt, profile):
    required, observed = {}, []
    rows = checked_json(directory / "fixtures.json", receipt["fixtures_sha256"])
    by_name = {row["fixture_id"]: row for row in rows}
    expected = {name: (package, target) for package, target, names in inventory(profile) for name in names}
    if len(by_name) != len(rows) or expected.keys() != by_name.keys():
        raise ValueError("behavioral inventory differs from the checked-in suite")
    for name, (package, target) in expected.items():
        row = by_name[name]
        evidence = checked_json(directory / (name + ".json"), row["evidence_sha256"])
        if any(evidence[k] != receipt[k] for k in ("profile", "candidate", "manifest_sha256",
                "artifact_revision", "reference_sha256", "cargo_lock_sha256", "pins_sha256", "host", "architecture")):
            raise ValueError(f"stale fixture identity: {name}")
        if evidence["fixture_id"] != name or evidence["environment"] != receipt["fixture_environment"]:
            raise ValueError(f"incorrect fixture selection: {name}")
        build_log = evidence["build"]
        check_execution(directory, build_log)
        if build_log not in receipt["commands"] or build_log["argv"] != [
            "cargo", "test", "--no-run", "--locked", "--release", "-p", package,
            "--test", target, "--message-format=json",
        ]:
            raise ValueError(f"unrecorded build: {target}")
        passed = False
        if "execution" in evidence:
            binary = receipt["binaries"][target]
            command = evidence["execution"]
            if binary != evidence["executable"] or command not in receipt["commands"] or build_log["exit_code"] != 0:
                raise ValueError(f"unrecorded native execution: {name}")
            passed = native_pass(directory, command, binary, name)
        if evidence["passed"] is not passed or row["passed"] is not passed or row["detail"] != evidence["detail"]:
            raise ValueError(f"assertion result differs from retained execution: {name}")
        for kind in COVERAGE[name]:
            fixture_id = f"{name}/{kind}"
            required[fixture_id] = kind
            observed.append({**row, "fixture_id": fixture_id})
    return required, observed


def check_inputs(record, other, scope):
    if other.get("version") != 1 or other.get("scope") != scope or other.get("status") not in ("passed", "failed"):
        raise ValueError(f"invalid {scope} receipt")
    for key in ("profile", "candidate", "manifest_sha256", "artifact_directory", "artifact_revision",
                "reference", "reference_sha256", "cargo_lock_sha256", "pins_sha256", "host", "architecture"):
        if other[key] != record[key]:
            raise ValueError(f"{scope} receipt differs in {key}")


def issue(profile, candidate, paths, root, runs):
    out = root / (profile + "-qualification")
    args = argparse.Namespace(profile=profile, candidate=candidate, artifact=Path(paths["artifact"]),
                              reference=Path(paths["reference"]), output=out)
    with pinned_run(args, "qualification") as (out, record, _):
        numerical_dir = root / (profile + "-numerical")
        behavior_dir = root / (profile + "-behavior")
        numerical = checked_json(numerical_dir / "receipt.json", runs["numerical"]["receipt_sha256"])
        behavior = checked_json(behavior_dir / "receipt.json", runs["behavior"]["receipt_sha256"])
        check_inputs(record, numerical, "numerical")
        check_inputs(record, behavior, "behavior")
        for name, digest in numerical.get("files", {}).items():
            if sha256(numerical_dir / name) != digest:
                raise ValueError(f"changed numerical file: {name}")
        # Independently derive the runtime identity from the verified inputs.
        inspect_profile(out, record)
        if any(r.get("metal_library") != record["metal_library"] for r in (numerical, behavior)):
            raise ValueError("numerical and behavioral kernels differ from the issuing build")
        actual = checked_json(out / "profile-identity.stdout", record["profile_identity_sha256"])
        inspected = checked_json(behavior_dir / "profile-identity.stdout", behavior["profile_identity_sha256"])
        if actual != inspected or actual["profile"] != profile or actual["artifact"] != profile:
            raise ValueError("native runtime profile identity changed")
        required, observed = fixture_rows(behavior_dir, behavior, profile)
        if (behavior["status"] == "passed") != all(row["passed"] for row in observed):
            raise ValueError("behavioral receipt status differs from terminal fixtures")
        _, target, name, _ = SUITES[profile]
        native_commands = [c for c in numerical["commands"] if c.get("stdout") == "native-test.stdout"]
        if len(native_commands) != 1:
            raise ValueError("no unique native reference execution")
        passed = native_pass(numerical_dir, native_commands[0], numerical["binaries"][target], name)
        fixture_id = f"{name}/exact_counting"
        required[fixture_id] = "exact_counting"
        passed = passed and numerical["status"] == "passed"
        observed.append({"fixture_id": fixture_id, "passed": passed,
                         "detail": f"{name}: {numerical['status']}",
                         "evidence_sha256": runs["numerical"]["receipt_sha256"]})
        fixture_file = out / "fixture-run.json"
        write_json(fixture_file, {"profile_id": profile, "artifact_id": profile, "identity": actual["identity"],
                                  "required": required, "observed": observed})
        record["fixture_run_sha256"] = sha256(fixture_file)
        record["input_receipts"] = {scope: item["receipt_sha256"] for scope, item in runs.items()}
        evaluator = build("example", "engine", "m3_numerical_report", out, record)
        report_dir = out / "report"
        try:
            execute([evaluator, "qualify", profile, args.reference, numerical_dir / "raw", fixture_file, report_dir],
                    out, "qualify", record)
        finally:
            record["files"] = {str(p.relative_to(out)): sha256(p) for p in sorted(report_dir.glob("*.json"))}
