#!/usr/bin/env python3
"""Run from the Mac CI host lease, e.g. mac_locked_step.py -- python3 THIS_SCRIPT --help."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "gemma-4-e4b-it-4bit": ("gemma4_e4b", "gemma4_quantized", "gemma4_native_trained_reference", "ALLOY_GEMMA4"),
    "ornith-1.5-9b-4bit": ("ornith_1_5_9b", "qwen35_quantized", "ornith_9b_quantized_native_reference", "ALLOY_ORNITH"),
    "qwen3.5-0.8b-4bit": ("qwen35_0_8b", "qwen35_quantized", "qwen35_0_8b_quantized_native_reference", "ALLOY_QWEN35_SMALL"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(manifest, artifact, reference):
    for entry in manifest["artifact"]["files"]:
        path = artifact / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"artifact does not match the pinned manifest: {path}")
    if sha256(reference) != manifest["reference"]["sha256"]:
        raise ValueError("reference SHA-256 differs from the pinned capture")


def verify_candidate(expected):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"], cwd=ROOT)
    if head != expected or dirty:
        raise ValueError(f"requires a clean checkout of {expected}; found {head}, dirty={bool(dirty)}")


def require_single_pass(output, name):
    if not re.search(r"^running 1 test$", output, re.M) or not re.search(
        r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured;", output, re.M
    ) or not re.search(r"^test " + re.escape(name) + r" \.\.\. ok$", output, re.M):
        raise ValueError(f"native test did not report exactly one named pass: {name}")


def execute(argv, directory, label, record, env=None):
    argv = [str(arg) for arg in argv]
    stdout = directory / (label + ".stdout")
    stderr = directory / (label + ".stderr")
    entry = {"argv": argv, "stdout": stdout.name, "stderr": stderr.name}
    record["commands"].append(entry)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err).returncode
    entry.update(exit_code=rc, stdout_sha256=sha256(stdout), stderr_sha256=sha256(stderr))
    if rc:
        raise RuntimeError(f"{label} exited {rc}; see {stdout} and {stderr}")
    return stdout


def build(kind, package, target, out, record):
    argv = ["cargo", "test", "--no-run"] if kind == "test" else ["cargo", "build"]
    argv += ["--locked", "--release", "-p", package, "--" + kind, target, "--message-format=json"]
    log = execute(argv, out, "build-" + target, record)
    with log.open() as src:
        artifacts = [json.loads(line) for line in src if line.strip()]
    matches = [Path(a["executable"]) for a in artifacts
               if a.get("reason") == "compiler-artifact" and a.get("executable")
               and a["target"]["name"] == target and kind in a["target"]["kind"]]
    if len(matches) != 1:
        raise ValueError(f"Cargo produced {len(matches)} executables for {target}")
    record["binaries"][target] = {"path": str(matches[0]), "sha256": sha256(matches[0])}
    return matches[0]


def run(args):
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("choose a new output directory outside the checkout")
    out.mkdir()  # Never mix artifacts from different invocations.
    record = {"version": 1, "scope": "numerical", "profile": args.profile,
              "candidate": args.candidate, "commands": [], "binaries": {}, "status": "failed"}
    try:
        verify_candidate(args.candidate)
        manifest_stem, target, name, prefix = SUITES[args.profile]
        pin = ROOT / "crates/inference-infra/tests/fixtures" / (manifest_stem + "_profile.json")
        manifest = json.loads(pin.read_bytes())
        if manifest["id"] != args.profile:
            raise ValueError("wrong pinned profile")
        artifact = args.artifact.resolve(strict=True)
        reference = args.reference.resolve(strict=True)
        verify_inputs(manifest, artifact, reference)
        record.update(artifact_directory=str(artifact), artifact_revision=manifest["artifact"]["revision"],
                      manifest_sha256=sha256(pin), reference=str(reference),
                      reference_sha256=manifest["reference"]["sha256"], host=platform.platform(),
                      architecture=platform.machine(), cargo_lock_sha256=sha256(ROOT / "Cargo.lock"),
                      pins_sha256=sha256(ROOT / "PINS.md"))
        execute(["rustc", "-Vv"], out, "rustc-version", record)
        binary = build("test", "inference-infra", target, out, record)
        evaluator = build("example", "engine", "m3_numerical_report", out, record)
        rows = out / "raw"
        rows.mkdir()
        env = dict(os.environ)
        selected = {prefix + "_ARTIFACT": str(artifact), prefix + "_REFERENCE": str(reference),
                    "ALLOY_M3_NUMERICAL_OUTPUT": str(rows)}
        env.update(selected)
        record["fixture_environment"] = selected
        # Keep fixture stdout separate from libtest's named result line.
        log = execute([binary, name, "--exact", "--ignored", "--test-threads=1", "--show-output"],
                      out, "native-test", record, env)
        require_single_pass(log.read_text(), name)
        report = out / "report"
        execute([evaluator, args.profile, reference, rows, report], out, "evaluate", record)
        verify_candidate(args.candidate)
        record["files"] = {str(p.relative_to(out)): sha256(p) for folder in (rows, report)
                           for p in sorted(folder.iterdir())}
        record["status"] = "passed"
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        record["error"] = str(error)
        raise
    finally:
        # Publish the terminal receipt last; a killed run cannot leave a successful receipt.
        pending = out / "receipt.pending"
        with pending.open("x") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        pending.rename(out / "receipt.json")
    print(f"{args.profile}: numerical qualification passed ({out / 'receipt.json'})")


def main():
    p = argparse.ArgumentParser(description="Issue native numerical evidence for one pinned M3 profile")
    p.add_argument("--profile", choices=sorted(SUITES), required=True)
    p.add_argument("--candidate", required=True, help="full expected Git commit ID")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new directory inside an existing external run directory")
    try:
        run(p.parse_args())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
