#!/usr/bin/env python3
"""Collect W3 evidence and complete source slices; never signs acceptance."""
import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
from pathlib import Path

AUDIT_FILES = [
    "crates/harness-core/src/artifact.rs",
    "crates/harness-core/src/store/mod.rs",
    "crates/harness-core/src/store/durable/mod.rs",
    "crates/harness-core/src/store/durable/failure_recovery.rs",
    "crates/harness-core/tests/tool_artifacts.rs",
]
GATES = [
    ("harness-core", "tool_policy", "destructive_effect_requires_bound_approval"),
    ("harness-core", "tool_artifacts", "oversized_tool_result_artifact_roundtrip"),
    ("qwen3", "tool_calls", "w3_boundary_duplicate_keys_fail_in_atomic_and_streaming_modes"),
    ("harness-core", "tool_policy", "w3_boundary_five_frozen_builtin_defaults_preserve_prompt_resolution"),
    ("harness-core", "w3_contract", "freeze_manifest_matches_normative_record"),
    ("harness-core", "w3_provenance", "frozen_vectors_are_reproduced_by_rust"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == root or root in output.parents:
        raise SystemExit("Evidence must be outside checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"accepted": False, "source_review_complete": False,
              "passed": False, "capture_only": args.capture_only,
              "platform": platform.platform(), "machine": platform.machine(),
              "commands": [], "gates": [], "source_files": []}
    def run(argv, env=None):
        number = len(report["commands"]) + 1
        log = output / f"command-{number:03d}.log"
        record = {"argv": argv, "log": log.name, "returncode": None}
        report["commands"].append(record)
        with log.open("wb") as stream:
            process = subprocess.Popen(argv, cwd=root, stdout=stream,
                                       stderr=subprocess.STDOUT, env=env,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record["timed_out"] = True
                raise RuntimeError("Command timeout: " + str(log))
        record["returncode"] = code
        if code:
            raise RuntimeError("Command failed: " + str(log))
        return log.read_text(errors="replace")
    def git(*items):
        return run(["git", *items]).strip()
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
            raise RuntimeError("Expected commit must be a full SHA")
        head = git("rev-parse", "HEAD")
        report["candidate"] = head
        if head != args.expected_commit:
            raise RuntimeError("Candidate mismatch")
        status = git("status", "--porcelain=v1", "--untracked-files=all",
                     "--ignore-submodules=none")
        report["initial_status"] = status
        if status:
            raise RuntimeError("Checkout is dirty")
        report["submodules"] = git("submodule", "status", "--recursive")
        if any(line[:1] in ("-", "+", "U") for line in report["submodules"].splitlines()):
            raise RuntimeError("Submodule not at recorded revision")
        snapshots = {}
        for name in AUDIT_FILES:
            data = (root / name).read_bytes()
            snapshots[name] = data
            lines = data.decode().splitlines(keepends=True)
            parts = []
            for first in range(0, len(lines), 120):
                end = min(first + 120, len(lines))
                part = name.replace("/", "__") + f".L{first+1}-L{end}.txt"
                text = "".join(f"{n+1:05d}: {lines[n]}" for n in range(first, end))
                (output / part).write_text(text)
                parts.append(part)
            report["source_files"].append({"path": name, "lines": len(lines),
                "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})
        if not args.capture_only:
            report["rustc"] = run(["rustc", "-Vv"])
            report["cargo"] = run(["cargo", "-V"])
            env = os.environ.copy()
            env["CARGO_TERM_COLOR"] = "never"
            env["ALLOY_W3_PROVENANCE_OUTPUT"] = str(output / "rust-generated-vectors.json")
            for package, target, name in GATES:
                prefix = ["cargo", "test", "--locked", "-p", package, "--test", target, name]
                listing = run(prefix + ["--", "--exact", "--list", "--format", "terse"], env)
                names = re.findall(r"^(.+): test$", listing, re.MULTILINE)
                if names != [name]:
                    raise RuntimeError("Exact discovery mismatch: " + name)
                result = run(prefix + ["--", "--exact", "--nocapture"], env)
                if not re.search(r"^test " + re.escape(name) + r" \.\.\. ok$", result, re.MULTILINE):
                    raise RuntimeError("Missing named passing result: " + name)
                if "test result: ok. 1 passed; 0 failed;" not in result:
                    raise RuntimeError("Exact count mismatch: " + name)
                report["gates"].append({"package": package, "target": target,
                                        "name": name, "passed": True})
            if len(json.loads((output / "rust-generated-vectors.json").read_text())["cases"]) != 12:
                raise RuntimeError("Missing Rust reproduction evidence")
            env.pop("ALLOY_W3_PROVENANCE_OUTPUT")
            run(["cargo", "fmt", "--all", "--", "--check"], env)
            run(["cargo", "check", "--locked", "--workspace", "--all-targets"], env)
            run(["cargo", "clippy", "--locked", "--workspace", "--all-targets", "--", "-D", "warnings"], env)
            # Serial library tests preserve existing process-global FFI isolation.
            run(["cargo", "test", "--locked", "--workspace", "--all-targets", "--", "--test-threads=1"], env)
            run(["cargo", "test", "--locked", "--workspace", "--doc"], env)
            # Also exercise W3 under ordinary parallel test scheduling.
            run(["cargo", "test", "--locked", "-p", "harness-core", "--all-targets"], env)
            env["RUSTDOCFLAGS"] = "-D warnings"
            run(["cargo", "doc", "--locked", "--workspace", "--no-deps"], env)
        for name, original in snapshots.items():
            if (root / name).read_bytes() != original:
                raise RuntimeError("Source changed during run: " + name)
        if git("rev-parse", "HEAD") != head or git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
            raise RuntimeError("Checkout changed during run")
        report["passed"] = not args.capture_only
        report["remaining"] = ["manual full-source audit", "historical dependency audit",
            "exact Step C inventory review", "other-platform evidence",
            "required CLI/native/real-model CI jobs", "scoped acceptance decision"]
    except Exception as error:
        report["error"] = str(error)
    finally:
        retained = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir() if p.is_file()}
        report["retained_sha256"] = retained
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Evidence:", output)
        print("Captured only" if args.capture_only and "error" not in report
              else "Passed collected gates" if report["passed"] else "Failed: " + report.get("error", "unknown"))
    return 0 if report["passed"] or (args.capture_only and "error" not in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
