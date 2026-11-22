#!/usr/bin/env python3
"""Run from the Mac CI host lease, e.g. mac_locked_step.py -- python3 THIS_SCRIPT --help."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
class Suite(NamedTuple):
    manifest: Path
    package: str
    target: str
    test: str
    prefix: str


def infra(stem, target, test, prefix):
    return Suite(Path("crates/inference-infra/tests/fixtures") / (stem + "_profile.json"),
                 "inference-infra", target, test, prefix)


SUITES = {
    "gemma-4-e4b-it-4bit": infra("gemma4_e4b", "gemma4_quantized", "gemma4_native_trained_reference", "ALLOY_GEMMA4"),
    "ornith-1.5-9b-4bit": infra("ornith_1_5_9b", "qwen35_quantized", "ornith_9b_quantized_native_reference", "ALLOY_ORNITH"),
    "qwen3.5-0.8b-4bit": infra("qwen35_0_8b", "qwen35_quantized", "qwen35_0_8b_quantized_native_reference", "ALLOY_QWEN35_SMALL"),
    "gemma-3-4b-4bit": infra("gemma3_4b", "gemma3_quantized", "gemma3_4b_real_quantized_native_reference", "ALLOY_GEMMA3_SMALL"),
    "gemma-3-12b-4bit": infra("gemma3_12b", "gemma3_quantized", "gemma3_12b_real_quantized_native_reference", "ALLOY_GEMMA3"),
    "phi4-mini-instruct-4bit": infra("phi4_mini", "phi_quantized", "phi4mini_real_quantized_native_reference", "ALLOY_PHI"),
    "ministral-3-3b-instruct-2512-4bit": infra("ministral3", "ministral3_quantized", "ministral3_real_quantized_native_reference", "ALLOY_MINISTRAL3"),
    "ministral-3-3b-instruct-2512-8bit": infra("ministral3_8bit", "ministral3_quantized", "ministral3_real_affine8_native_reference", "ALLOY_MINISTRAL3"),
    "llama-3.1-8b-4bit": Suite(Path("crates/families/llama3/fixture/affine4_profile.json"),
                               "llama3", "quantized_loading", "llama3_4bit_8bit_load", "ALLOY_LLAMA"),
    "llama-3.1-8b-8bit": Suite(Path("crates/families/llama3/fixture/affine8_profile.json"),
                               "llama3", "quantized_loading", "llama3_4bit_8bit_load", "ALLOY_LLAMA"),
    "smollm3-3b-4bit": Suite(Path("crates/families/smollm/fixture/profile.json"),
                             "smollm", "quantized_loading", "smollm3_quantized_native_reference", "ALLOY_SMOLLM"),
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


def cargo_target(candidate, output_parent, checkout=None):
    checkout = (checkout or ROOT).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", candidate):
        raise ValueError("invalid clean candidate for compilation")
    base = Path(os.environ.get("CARGO_TARGET_DIR", output_parent / "native-builds")).expanduser().resolve()
    if base.is_relative_to(checkout):
        raise ValueError("native build root must be external to the checkout")
    checkout_id = hashlib.sha256(os.fsencode(checkout)).hexdigest()[:16]
    return base / "m3-native" / checkout_id / candidate


def metal_library(artifacts, executable):
    roots = [Path(a["out_dir"]) for a in artifacts if a.get("reason") == "build-script-executed"
             and a["package_id"].split("#")[0].endswith("/crates/mlx-ffi")]
    if len(roots) != 1:
        raise ValueError("expected exactly one linked mlx-ffi build")
    lib = roots[0] / "lib/mlx.metallib"
    digest = sha256(lib)
    # The static MLX loader checks these paths before its build-time default.
    for path in (executable.parent / "mlx.metallib", executable.parent / "Resources/mlx.metallib",
                 roots[0] / "build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"):
        if path.exists() and sha256(path) != digest:
            raise ValueError(f"conflicting Metal kernels at {path}")
    return {"path": str(lib), "sha256": digest}


def build(kind, package, target, out, record):
    argv = ["cargo", "test", "--no-run"] if kind == "test" else ["cargo", "build"]
    argv += ["--locked", "--release", "-p", package, "--" + kind, target, "--message-format=json"]
    build_dir = Path(record["cargo_target_directory"])
    log = execute(argv, out, "build-" + target, record, {**os.environ, "CARGO_TARGET_DIR": str(build_dir)})
    with log.open() as src:
        artifacts = [json.loads(line) for line in src if line.strip()]
    matches = [Path(a["executable"]) for a in artifacts
               if a.get("reason") == "compiler-artifact" and a.get("executable")
               and a["target"]["name"] == target and kind in a["target"]["kind"]
               and Path(a["manifest_path"]).resolve().is_relative_to(ROOT)
               and Path(a["executable"]).resolve().is_relative_to(build_dir)]
    if len(matches) != 1:
        raise ValueError(f"Cargo produced {len(matches)} executables for {target}")
    kernels = metal_library(artifacts, matches[0])
    if record.setdefault("metal_library", kernels) != kernels:
        raise ValueError("Metal library changed between fixture builds")
    record["binaries"][target] = {"path": str(matches[0]), "sha256": sha256(matches[0])}
    return matches[0]


@contextmanager
def pinned_run(args, scope):
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("choose a new output directory outside the checkout")
    out.mkdir()  # Never mix artifacts from different invocations.
    record = {"version": 1, "scope": scope, "profile": args.profile,
              "candidate": args.candidate, "commands": [], "binaries": {}, "status": "failed"}
    try:
        verify_candidate(args.candidate)
        record["cargo_target_directory"] = str(cargo_target(args.candidate, out.parent))
        suite = SUITES[args.profile]
        pin = ROOT / suite.manifest
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
        selected = {suite.prefix + "_ARTIFACT": str(artifact), suite.prefix + "_REFERENCE": str(reference)}
        yield out, record, selected
        kernel = record["metal_library"]
        if sha256(Path(kernel["path"])) != kernel["sha256"]:
            raise ValueError("Metal library changed during execution")
        verify_candidate(args.candidate)
        verify_inputs(manifest, artifact, reference)
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


def run(args):
    with pinned_run(args, "numerical") as (out, record, selected):
        suite = SUITES[args.profile]
        name = suite.test
        binary = build("test", suite.package, suite.target, out, record)
        evaluator = build("example", "engine", "m3_numerical_report", out, record)
        rows = out / "raw"
        rows.mkdir()
        env = dict(os.environ)
        selected["ALLOY_M3_NUMERICAL_OUTPUT"] = str(rows)
        env.update(selected)
        record["fixture_environment"] = selected
        # Keep fixture stdout separate from libtest's named result line.
        log = execute([binary, name, "--exact", "--ignored", "--test-threads=1", "--show-output"],
                      out, "native-test", record, env)
        require_single_pass(log.read_text(), name)
        report = out / "report"
        execute([evaluator, args.profile, record["reference"], rows, report], out, "evaluate", record)
        record["files"] = {str(p.relative_to(out)): sha256(p) for folder in (rows, report)
                           for p in sorted(folder.iterdir())}
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
