"""Build the local production package from the matrix that just ran in this process."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import m3_native_numerical as native
from m3_native_qualification import checked_json, check_inputs


def checked_copy(source, destination, digest):
    shutil.copyfile(source, destination)
    if native.sha256(destination) != digest:
        raise ValueError(f"package input changed: {source}")


def selected(index):
    for q in index["qualifications"]:
        if q["status"] != "passed":
            continue
        profile = q["profile"]
        scopes = [r for r in index["runs"] if r["profile"] == profile]
        if (len(scopes) != 2 or {r["scope"] for r in scopes} != {"numerical", "behavior"}
                or any(r["status"] != "passed" for r in scopes)):
            raise ValueError(f"qualification contradicts terminal executions: {profile}")
        yield profile, q, scopes


def package_profile(root, destination, candidate, profile, qualification, runs):
    if profile not in native.SUITES:
        raise ValueError("unknown profile in package selection")
    directory = root / (profile + "-qualification")
    receipt = checked_json(directory / "receipt.json", qualification["receipt_sha256"])
    if (receipt["profile"], receipt["candidate"], receipt["scope"], receipt["status"]) != (
            profile, candidate, "qualification", "passed"):
        raise ValueError("stale qualification receipt")
    expected_build = str(native.cargo_target(candidate, root))
    if receipt["cargo_target_directory"] != expected_build:
        raise ValueError("qualification came from a different compiler target")
    kernel = receipt["metal_library"]
    for item in runs:
        scope = item["scope"]
        if receipt["input_receipts"][scope] != item["receipt_sha256"]:
            raise ValueError("qualification refers to another fixture run")
        other = checked_json(root / (profile + "-" + scope) / "receipt.json", item["receipt_sha256"])
        check_inputs(receipt, other, scope)
        if (other["status"] != "passed" or other["metal_library"] != kernel
                or other["cargo_target_directory"] != expected_build):
            raise ValueError("failed fixture or changed Metal library")
    dependency_inputs = {
        "cargo_lock_sha256": native.sha256(native.ROOT / "Cargo.lock"),
        "pins_sha256": native.sha256(native.ROOT / "PINS.md"),
        "host": platform.platform(), "architecture": platform.machine(),
        "rustc_sha256": hashlib.sha256(subprocess.check_output(["rustc", "-Vv"])).hexdigest(),
        "metal_library_sha256": kernel["sha256"],
    }
    digest = hashlib.sha256(json.dumps(dependency_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if dependency_inputs != receipt["dependency_inputs"] or digest != receipt["dependencies_sha256"]:
        raise ValueError("build inputs changed since native qualification")
    ident = checked_json(directory / "profile-identity.stdout", receipt["profile_identity_sha256"])
    if (ident["profile"], ident["artifact"], ident["identity"]["candidate_revision"],
            ident["identity"]["dependencies_sha256"]) != (profile, profile, candidate, digest):
        raise ValueError("incorrect native identity")
    verdict = checked_json(directory / "report/verdict.json", qualification["verdict_sha256"])
    if not verdict["usable"] or verdict["failures"]:
        raise ValueError("profile did not pass native qualification")
    target = destination / profile
    target.mkdir()
    entry = {"context": ident["context"], "dependencies_sha256": digest}
    for name in ("spec", "report"):
        filename = name + ".json"
        expected = receipt["files"]["report/" + filename]
        checked_copy(directory / "report" / filename, target / filename, expected)
        entry[name + "_sha256"] = expected
    return entry, kernel


def host():
    def sysctl(name):
        return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()
    return {"model": sysctl("hw.model"), "os_build": sysctl("kern.osversion"), "memory_bytes": int(sysctl("hw.memsize"))}


def publish(index, root):
    candidate = index["candidate"]
    native.verify_candidate(candidate)
    destination = root / "package"
    if destination.exists():
        raise ValueError("package destination already exists")
    build_receipt = {"commands": []}
    with tempfile.TemporaryDirectory(dir=root, prefix=".package-") as temporary:
        package = Path(temporary) / "package"
        qualification = package / "qualification"
        qualification.mkdir(parents=True)
        profiles = {}
        kernel = None
        for profile, q, runs in selected(index):
            entry, found = package_profile(root, qualification, candidate, profile, q, runs)
            if kernel is not None and found != kernel:
                raise ValueError("package profiles used different Metal libraries")
            profiles[profile] = entry
            kernel = found
        if not profiles:
            raise ValueError("no passing profiles to package")
        catalog = {"version": 1, "candidate": candidate, "host": host(), "profiles": profiles,
                   "metal_library_sha256": kernel["sha256"]}
        path = qualification / "catalog.json"
        path.write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n")
        digest = native.sha256(path)
        build_dir = native.cargo_target(candidate, root)
        env = {**os.environ, "CARGO_TARGET_DIR": str(build_dir), "ALLOY_M3_RELEASE_CATALOG_SHA256": digest, "ALLOY_M3_RELEASE_REVISION": candidate}
        log = native.execute(["cargo", "build", "--release", "--locked", "-p", "alloyd", "--bin", "alloyd",
                              "--message-format=json"], root, "build-packaged-alloyd", build_receipt, env)
        artifacts = [json.loads(line) for line in log.read_text().splitlines() if line]
        binaries = [Path(a["executable"]) for a in artifacts if a.get("reason") == "compiler-artifact"
                    and a.get("executable") and a["target"]["name"] == "alloyd" and "bin" in a["target"]["kind"]
                    and Path(a["manifest_path"]).resolve() == native.ROOT / "crates/alloyd/Cargo.toml"
                    and Path(a["executable"]).resolve().is_relative_to(build_dir)]
        if len(binaries) != 1 or native.metal_library(artifacts, binaries[0]) != kernel:
            raise ValueError("daemon compiler output differs from the qualified build")
        checked_copy(Path(kernel["path"]), package / "mlx.metallib", kernel["sha256"])
        binary = package / "alloyd"
        checked_copy(binaries[0], binary, native.sha256(binaries[0]))
        binary.chmod(0o755)
        native.verify_candidate(candidate)
        result = {"directory": "package", "profiles": sorted(profiles), "catalog_sha256": digest,
                  "binary_sha256": native.sha256(binary), "build": build_receipt["commands"]}
        package.rename(destination)
        return result
