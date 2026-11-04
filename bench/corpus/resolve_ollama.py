"""One-time local resolution of the Ollama comparison artifacts.

Fixes the two recorded tags that do not exist in the registry, fetches each
tag's manifest from registry.ollama.ai, pins it by the SHA-256 of the
manifest bytes, and records the layer digests the manifest names. The
manifest digest transitively pins every layer, so the model is content-
addressed without downloading ~33 GB of weights. Recorded in
bench/corpus/README.md.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_PATH = ROOT / "models.json"

REGISTRY = "https://registry.ollama.ai/v2/library"

# tag fixes: models.json recorded two tags that do not exist
TAG_FIXES = {
    "qwen3-4b-4bit": "qwen3:4b-q4_K_M",
    "qwen3-14b-4bit": "qwen3:14b-q4_K_M",
}

# short digests published on the ollama.com tags pages, 2026-11-04
EXPECTED_SHORT = {
    "qwen3:4b-q4_K_M": "2bfd38a7daaf",
    "qwen3:14b-q4_K_M": "bdbd181c33f2",
    "llama3.1:8b-instruct-q4_K_M": "46e0c10c039e",
    "llama3.1:8b-instruct-q8_0": "b158ded76fa0",
    "gemma3:12b-it-q4_K_M": "b92b218582c5",
}

CHECKPOINT_NOTES = {
    "qwen3:4b-q4_K_M": (
        "Ollama does not publish the HF source commit. The non-suffixed "
        "40K-context tags are the original Qwen3 hybrid checkpoints; the "
        "2507-suffixed tags are the July-2025 Instruct/Thinking refreshes "
        "and are different checkpoints, deliberately avoided."
    ),
    "qwen3:14b-q4_K_M": (
        "Ollama does not publish the HF source commit. The non-suffixed "
        "40K-context tags are the original Qwen3 hybrid checkpoints; the "
        "2507-suffixed tags are the July-2025 refreshes, deliberately avoided."
    ),
    "llama3.1:8b-instruct-q4_K_M": (
        "Ollama library page describes Llama 3.1 from Meta, 8B size; "
        "repository-level match, no source commit published."
    ),
    "llama3.1:8b-instruct-q8_0": (
        "Ollama library page describes Llama 3.1 from Meta, 8B size; "
        "repository-level match, no source commit published."
    ),
    "gemma3:12b-it-q4_K_M": (
        "Ollama library page describes Google's Gemma 3 12B; repository-level "
        "match, no source commit published. Multimodal build (Text+Image)."
    ),
}


def fetch_manifest(model, tag):
    url = f"{REGISTRY}/{model}/manifests/{tag}"
    accept = ", ".join(
        [
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        ]
    )
    request = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(request) as response:
        return response.read()


def main():
    models = json.loads(MODELS_PATH.read_text())
    report = []
    for profile in models["profiles"]:
        entry = profile["comparison_artifacts"]["ollama"]
        tag = TAG_FIXES.get(profile["id"], entry["model_tag"])
        entry["model_tag"] = tag
        model, _, rest = tag.partition(":")
        print(f"resolving {tag}", flush=True)
        body = fetch_manifest(model, rest)
        digest = hashlib.sha256(body).hexdigest()
        manifest = json.loads(body)
        short = EXPECTED_SHORT[tag]
        if not digest.startswith(short):
            print(
                f"FAIL {tag}: manifest sha256 {digest} does not start with "
                f"published short digest {short}"
            )
            report.append((tag, digest, False))
            continue
        layers = [
            {
                "digest": layer["digest"],
                "size_bytes": layer["size"],
            }
            for layer in manifest.get("layers", [])
        ]
        total = sum(layer["size_bytes"] for layer in layers)
        print(f"  manifest sha256 {digest}")
        print(f"  layers {len(layers)}, {total} bytes total")
        entry["manifest_sha256"] = digest
        entry["config_digest"] = manifest["config"]["digest"]
        entry["layer_digests"] = [layer["digest"] for layer in layers]
        entry["size_bytes"] = total
        entry["checkpoint_match"] = CHECKPOINT_NOTES[tag]
        entry["verification_method"] = (
            "registry manifest pinned by SHA-256; layer digests recorded "
            "from the manifest; weights not downloaded"
        )
        entry["verified"] = True
        report.append((tag, digest, True))

    text = json.dumps(models, indent=4) + "\n"
    MODELS_PATH.write_text(text)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")

    print()
    ok = True
    for tag, digest, match in report:
        state = "ok" if match else "FAILED"
        print(f"{state}: {tag} {digest}")
        ok = ok and match
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
