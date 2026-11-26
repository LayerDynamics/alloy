#!/usr/bin/env python3
"""Run a scheduling trial on a reserved Mac using a locally verified native build.

The operator supplies a frozen claim with candidate, profile, package, target,
ordered fixtures and the path and SHA-256 of a trusted local behavior receipt.
Run against the original clean checkout. Create STOP in the new output directory
to end admissions. The receipt describes scheduling and grants no model approval.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

from m3_native_behavior import fixture_env, inventory, write_json
from m3_native_numerical import ROOT, SUITES, require_single_pass, sha256, verify_candidate, verify_inputs
from m3_native_qualification import checked_json, check_execution


def stamp():
    return {"utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def event(out, name, **fields):
    row = {"event": name, **stamp(), **fields}
    with (out / "events.jsonl").open("a") as log:
        log.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def prepare(claim):
    if claim["version"] != 1:
        raise ValueError("unsupported window claim")
    profile, target = claim["profile"], claim["target"]
    names = claim["fixtures"]
    allowed = {n for p, t, group in inventory(profile)
               if (p, t) == (claim["package"], target) for n in group}
    if not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError("expected distinct fixtures from one pinned native target")
    verify_candidate(claim["candidate"])
    source = Path(claim["behavior_receipt"])
    r = checked_json(source, claim["behavior_receipt_sha256"])
    if (r["version"], r["candidate"], r["profile"], r["scope"]) != (1, claim["candidate"], profile, "behavior"):
        raise ValueError("build receipt belongs to a different candidate or profile")
    if (r["host"], r["architecture"]) != (platform.platform(), platform.machine()):
        raise ValueError("native build belongs to a different host configuration")
    pin = ROOT / SUITES[profile].manifest
    manifest = json.loads(pin.read_bytes())
    if (manifest["id"] != profile or sha256(pin) != r["manifest_sha256"]
            or manifest["artifact"]["revision"] != r["artifact_revision"]
            or manifest["reference"]["sha256"] != r["reference_sha256"]
            or sha256(ROOT / "Cargo.lock") != r["cargo_lock_sha256"] or sha256(ROOT / "PINS.md") != r["pins_sha256"]):
        raise ValueError("pinned inputs differ from the build receipt")
    verify_inputs(manifest, Path(r["artifact_directory"]), Path(r["reference"]))
    identity = checked_json(source.parent / r["profile_identity"], r["profile_identity_sha256"])
    if identity["profile"] != profile or identity["identity"]["candidate_revision"] != claim["candidate"]:
        raise ValueError("native identity differs")
    binary = r["binaries"][target]
    for artifact in (binary, r["metal_library"]):
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("native build bytes changed")
    argv = ["cargo", "test", "--no-run", "--locked", "--release", "-p", claim["package"],
            "--test", target, "--message-format=json"]
    build, = [c for c in r["commands"] if c["argv"] == argv and c.get("exit_code") == 0]
    with check_execution(source.parent, build).open() as log:
        products = (json.loads(line) for line in log if line.strip())
        if not any(p.get("reason") == "compiler-artifact" and p.get("executable") == binary["path"]
                   and p["target"]["name"] == target and "test" in p["target"]["kind"] for p in products):
            raise ValueError("recorded Cargo build did not produce the selected test binary")
    prefix = SUITES[profile].prefix
    env = {prefix + "_ARTIFACT": r["artifact_directory"], prefix + "_REFERENCE": r["reference"],
           "ALLOY_REAL_PROFILE": profile, "ALLOY_REAL_MODEL": r["artifact_directory"]}
    if profile in ("ornith-1.5-9b-4bit", "qwen3.5-0.8b-4bit"):
        env.update(ALLOY_HYBRID_ARTIFACT=r["artifact_directory"],
                   ALLOY_HYBRID_PROFILE="ornith" if profile.startswith("ornith-") else "upstream")
    if env != r["fixture_environment"]:
        raise ValueError("fixture environment differs")
    return binary, env


def run_cases(out, names, binary, env):
    rows = []
    for name in names:
        if (out / "STOP").exists():
            rows.append({"fixture": name, "status": "unstarted"})
            continue
        argv = [binary["path"], name, "--exact", "--ignored", "--test-threads=1", "--show-output"]
        stdout, stderr = (out / (name + suffix) for suffix in (".stdout", ".stderr"))
        with stdout.open("xb") as log, stderr.open("xb") as err:
            with subprocess.Popen(argv, cwd=ROOT, env=fixture_env(env), stdout=log, stderr=err) as proc:
                event(out, "fixture_started", fixture=name, pid=proc.pid, log=str(log.name))
                status = proc.wait()
                event(out, "fixture_exited", fixture=name, exit_code=status)
        passed = False
        if status == 0:
            try:
                require_single_pass(stdout.read_text(), name)
                passed = True
            except ValueError:
                pass
        rows.append({"fixture": name, "status": "passed" if passed else "inconclusive",
                     "argv": argv, "exit_code": status, "stdout_sha256": sha256(stdout),
                     "stderr_sha256": sha256(stderr)})
    return rows


def check_owned_group():
    group = os.getpgrp()
    if group != os.getpid():
        raise ValueError("worker needs its own process group under mac_locked_step")
    with subprocess.Popen(["ps", "-axo", "pid=,pgid="], stdout=subprocess.PIPE, text=True) as ps:
        listing, _ = ps.communicate()
        if ps.returncode:
            raise ValueError("unable to verify worker group cleanup")
        for line in listing.splitlines():
            pid, pgid = map(int, line.split())
            if pgid == group and pid not in (group, ps.pid):
                raise ValueError(f"owned subprocess still present: {pid}")


def worker(out):
    lock = Path("/tmp/alloy-ci-mac-host.lock")
    # mac_locked_step explicitly passes its locked descriptor to its child.
    inherited = False
    for name in os.listdir("/dev/fd"):
        try:
            fd = int(name)
            if os.path.samestat(os.fstat(fd), lock.stat()) and os.pread(fd, 32, 0) == b"dirty\n":
                inherited = True
                break
        except OSError:
            continue
    if not inherited:
        raise ValueError("missing inherited host lease")
    index = json.loads((out / "attempt.json").read_bytes())
    claim = checked_json(out / "claim.json", index["claim_sha256"])
    event(out, "worker_started", pid=os.getpid(), host=os.uname().nodename)
    binary, env = prepare(claim)
    rows = run_cases(out, claim["fixtures"], binary, env)
    verify_candidate(claim["candidate"])
    check_owned_group()
    write_json(out / "terminal.json", {**index, "type": "m3-reserved-window-timing",
               "approval_credit": False, "binary": binary, "fixture_environment": env,
               "fixtures": rows, "cleanup_verified": stamp()})
    event(out, "receipt_finalized")


def launch(out, claim_path):
    if out.is_relative_to(ROOT) or not claim_path:
        raise ValueError("supply a claim and a new external output directory")
    data = claim_path.read_bytes()
    claim = json.loads(data)
    verify_candidate(claim["candidate"])
    out.mkdir(mode=0o700)
    (out / "claim.json").write_bytes(data)
    index = {"attempt_id": uuid.uuid4().hex, "candidate": claim["candidate"],
             "claim_sha256": sha256(out / "claim.json"), "started": stamp()}
    write_json(out / "attempt.json", index)
    event(out, "launcher_started")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALLOY_CI_LOCK_TEST_")}
    proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/ci/mac_locked_step.py"), "--",
                           sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                           "--output", str(out)], env=env)
    event(out, "wrapper_exited", exit_code=proc.returncode)
    proc.check_returncode()
    terminal = out / "terminal.json"
    receipt = json.loads(terminal.read_bytes())
    if any(receipt[k] != v for k, v in index.items()):
        raise ValueError("terminal receipt does not match this attempt")
    write_json(out / "finished.json", {**index, "terminal_sha256": sha256(terminal),
                                        "wrapper_release_observed_by": stamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.worker:
        worker(directory)
    else:
        launch(directory, args.claim)
