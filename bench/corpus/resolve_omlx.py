"""One-time local verification of the oMLX comparison row.

Prerequisite: oMLX installed (brew tap jundot/omlx https://github.com/jundot/omlx
&& brew install omlx) and on PATH. Requires macOS 15+ on Apple Silicon.

For each profile: materializes the pinned mlx-community artifact at its pinned
revision into a local models directory (reusing the Hugging Face cache from
the 2026-11-03 resolution run, so no re-download), re-verifies every per-file
SHA-256 and the aggregate digest against models.json, loads it through
`omlx serve`, and sends one chat completion. Sets verified: true only for
profiles whose bytes matched and whose completion returned a finish_reason.

Thinking models (Qwen3) may spend the whole token budget reasoning, in which
case the response carries reasoning_content and no content. That is accepted
as proof of load; the row records which field was produced.
"""

import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS_PATH = ROOT / "models.json"
MODELS_ROOT = Path.home() / "alloy-omlx-models"
PORT = 8017
BASE = f"http://127.0.0.1:{PORT}"
READY_TIMEOUT = 300
REQUEST_TIMEOUT = 300


def aggregate_sha256(files):
    files = sorted(files, key=lambda f: f["path"].encode("utf-8"))
    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(profile):
    artifact = profile["artifact"]
    model_dir = MODELS_ROOT / f"alloy-{profile['id']}"
    print(f"materializing {artifact['repository']} @ {artifact['revision']}")
    snapshot_download(
        repo_id=artifact["repository"],
        revision=artifact["revision"],
        local_dir=model_dir,
    )
    recorded = {f["path"]: f["sha256"] for f in artifact["files"]}
    for rel, expected in sorted(recorded.items()):
        actual = sha256_file(model_dir / rel)
        if actual != expected:
            print(f"FAIL {profile['id']}: {rel} hash mismatch")
            return None
    computed = aggregate_sha256(artifact["files"])
    if computed != artifact["sha256"]:
        print(f"FAIL {profile['id']}: aggregate digest mismatch")
        return None
    print(f"  ok  {len(recorded)} files hash-match, aggregate verified")
    return model_dir


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.status, json.loads(response.read())


def main():
    try:
        version = subprocess.run(
            ["omlx", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        sys.exit(f"omlx not usable ({exc}); install it first, see the header")
    print(f"omlx version: {version}")

    models = json.loads(MODELS_PATH.read_text())
    for profile in models["profiles"]:
        if verify_artifact(profile) is None:
            sys.exit(f"byte verification failed for {profile['id']}")

    print(f"starting omlx serve on port {PORT}")
    server = subprocess.Popen(
        [
            "omlx",
            "serve",
            "--model-dir",
            str(MODELS_ROOT),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + READY_TIMEOUT
        served = None
        while time.time() < deadline:
            try:
                status, served = http("GET", "/v1/models")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(2)
        if served is None:
            sys.exit("omlx serve did not become ready")
        ids = [m["id"] for m in served.get("data", [])]
        print(f"  {len(ids)} model ids served")

        report = []
        for profile in models["profiles"]:
            entry = profile["comparison_artifacts"]["omlx"]
            model_id = f"alloy-{profile['id']}"
            ok = False
            finish_reason = None
            text_kind = None
            snippet = ""
            if model_id not in ids:
                print(f"FAIL {profile['id']}: {model_id} not in /v1/models")
            else:
                try:
                    status, body = http(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model_id,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Reply with the single word: ready",
                                }
                            ],
                            "max_tokens": 64,
                            "temperature": 0,
                        },
                    )
                    choice = body["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    message = choice.get("message", {})
                    content = message.get("content")
                    reasoning = message.get("reasoning_content")
                    if content:
                        text_kind, snippet = "content", content
                        ok = True
                    elif reasoning:
                        text_kind, snippet = "reasoning_content", reasoning
                        ok = finish_reason in ("stop", "length")
                except (urllib.error.URLError, KeyError, IndexError, OSError) as exc:
                    print(f"FAIL {profile['id']}: {exc}")
            entry["omlx_version"] = version
            entry["model_id"] = model_id
            entry["bytes_verified"] = (
                "files re-hashed from disk and aggregate digest re-verified "
                "before the load; oMLX served that exact directory"
            )
            entry["completion"] = {
                "finish_reason": finish_reason,
                "text_field": text_kind,
                "text": snippet[:80],
            }
            entry["verified"] = bool(ok)
            print(
                f"  {'ok  ' if ok else 'FAIL'} {profile['id']} "
                f"finish={finish_reason} field={text_kind}"
            )
            report.append((profile["id"], bool(ok)))

        all_green = all(row for _, row in report)
        if all_green:
            for profile in models["profiles"]:
                profile["comparison_artifacts"]["omlx"]["verified_at"] = "2026-11-04"
            if all(
                all(
                    profile["comparison_artifacts"][b].get("verified")
                    for b in ("omlx", "llama.cpp", "ollama")
                )
                for profile in models["profiles"]
            ):
                models["status"] = "frozen"
                print("all comparison rows verified: status set to frozen")

        text = json.dumps(models, indent=4) + "\n"
        MODELS_PATH.write_text(text)
        manifest = json.loads((ROOT / "manifest.json").read_text())
        manifest["models"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=4) + "\n")

        print()
        for pid, ok in report:
            print(f"{'ok' if ok else 'FAILED'}: omlx load {pid}")
        sys.exit(0 if all_green else 1)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
