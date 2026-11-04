"""One-time local verification of the llama.cpp comparison artifacts.

Downloads each pinned GGUF at its pinned revision, hashes the downloaded
bytes, parses the GGUF header for source metadata, and checks it against
the pinned source checkpoint. Sets verified: true only when the metadata
names the same source repository. Recorded in bench/corpus/README.md.

GGUF header parsing reads only the metadata block, not the weights.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODELS_PATH = ROOT / "models.json"


def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected end of GGUF file")
    return data


def read_str(f):
    n = struct.unpack("<Q", read_exact(f, 8))[0]
    return read_exact(f, n).decode("utf-8", "replace")


def read_value(f, gguf_type):
    if gguf_type == 0:
        return read_exact(f, 1)[0]
    if gguf_type == 1:
        return struct.unpack("<b", read_exact(f, 1))[0]
    if gguf_type == 2:
        return struct.unpack("<H", read_exact(f, 2))[0]
    if gguf_type == 3:
        return struct.unpack("<h", read_exact(f, 2))[0]
    if gguf_type == 4:
        return struct.unpack("<I", read_exact(f, 4))[0]
    if gguf_type == 5:
        return struct.unpack("<i", read_exact(f, 4))[0]
    if gguf_type == 6:
        return struct.unpack("<f", read_exact(f, 4))[0]
    if gguf_type == 7:
        return read_exact(f, 1)[0]
    if gguf_type == 8:
        return read_str(f)
    if gguf_type == 9:
        element_type = struct.unpack("<I", read_exact(f, 4))[0]
        count = struct.unpack("<Q", read_exact(f, 8))[0]
        return [read_value(f, element_type) for _ in range(count)]
    if gguf_type == 10:
        return struct.unpack("<Q", read_exact(f, 8))[0]
    if gguf_type == 11:
        return struct.unpack("<q", read_exact(f, 8))[0]
    if gguf_type == 12:
        return struct.unpack("<d", read_exact(f, 8))[0]
    raise ValueError(f"unknown GGUF metadata type {gguf_type}")


def gguf_source_metadata(path):
    with open(path, "rb") as f:
        if read_exact(f, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = struct.unpack("<I", read_exact(f, 4))[0]
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        struct.unpack("<Q", read_exact(f, 8))  # tensor_count
        kv_count = struct.unpack("<Q", read_exact(f, 8))[0]
        source = {}
        for _ in range(kv_count):
            key = read_str(f)
            gguf_type = struct.unpack("<I", read_exact(f, 4))[0]
            value = read_value(f, gguf_type)
            if key.startswith("general.source") and isinstance(value, str):
                source[key] = value
        return source


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    models = json.loads(MODELS_PATH.read_text())
    report = []
    for profile in models["profiles"]:
        entry = profile["comparison_artifacts"]["llama.cpp"]
        label = f"{profile['id']} <- {entry['repository']}@{entry['revision']}"
        print(f"downloading {label} ({entry['quantization_file']})", flush=True)
        path = hf_hub_download(
            repo_id=entry["repository"],
            revision=entry["revision"],
            filename=entry["quantization_file"],
        )
        digest = sha256_file(path)
        size = Path(path).stat().st_size
        source_meta = gguf_source_metadata(path)
        print(f"  sha256 {digest}")
        print(f"  bytes  {size}")
        for key, value in source_meta.items():
            print(f"  {key} = {value}")

        source_repo = profile["source"]["repository"]
        blob = json.dumps(source_meta)
        checkpoint_match = source_repo in blob

        entry["sha256"] = digest
        entry["file_size_bytes"] = size
        entry["source_metadata"] = source_meta
        if checkpoint_match:
            entry["checkpoint_match"] = "metadata names the pinned source repository"
            entry["verified"] = True
        else:
            entry["checkpoint_match"] = None
            entry["verified"] = False
        report.append((label, digest, checkpoint_match))

    text = json.dumps(models, indent=4) + "\n"
    MODELS_PATH.write_text(text)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")

    print()
    for label, digest, match in report:
        state = "ok" if match else "CHECKPOINT UNCONFIRMED"
        print(f"{state}: {label} {digest}")
    if not all(match for _, _, match in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
