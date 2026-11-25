"""Offline, byte-preserving extraction of the pinned single-shard text model."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_header(path):
    with path.open("rb") as file:
        size = int.from_bytes(file.read(8), "little")
        assert 0 < size <= 16 * 1024 * 1024
        header = json.loads(file.read(size))
    cursor = 0
    for name, tensor in sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda pair: pair[1]["data_offsets"][0],
    ):
        assert name.startswith(("language_model.", "vision_tower.")), name
        start, end = tensor["data_offsets"]
        assert start == cursor and end > start
        cursor = end
    assert cursor + size + 8 == path.stat().st_size
    return header, size + 8


def extract(source, destination, profile):
    metadata = json.loads(profile.read_text())
    artifact = metadata["artifact"]
    source_files = artifact.get("derivation", {}).get("original_files", artifact["files"])
    for file in source_files:
        assert Path(file["path"]).name == file["path"]
        path = source / file["path"]
        assert path.stat().st_size == file["bytes"]
        assert digest(path) == file["sha256"], path
    original, base = read_header(source / "model.safetensors")
    selected = {k: v for k, v in original.items() if k.startswith("language_model.")}
    assert selected and any(k.startswith("vision_tower.") for k in original)
    header = {"__metadata__": {"format": "mlx"}}
    cursor = 0
    for name, tensor in sorted(selected.items()):
        start, end = tensor["data_offsets"]
        header[name] = {**tensor, "data_offsets": [cursor, cursor + end - start]}
        cursor += end - start
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    destination.mkdir()  # Never overwrite an existing artifact.
    with (source / "model.safetensors").open("rb") as src, (destination / "model.safetensors").open("xb") as dst:
        dst.write(len(encoded).to_bytes(8, "little"))
        dst.write(encoded)
        for _, tensor in sorted(selected.items()):
            start, end = tensor["data_offsets"]
            src.seek(base + start)
            remaining = end - start
            while remaining:
                chunk = src.read(min(remaining, 1024 * 1024))
                assert chunk
                dst.write(chunk)
                remaining -= len(chunk)
    index = {"metadata": {"total_size": cursor}, "weight_map": {k: "model.safetensors" for k in selected}}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for file in source_files:
        if file["path"] not in ["model.safetensors", "model.safetensors.index.json"]:
            shutil.copyfile(source / file["path"], destination / file["path"])
    derived, derived_base = read_header(destination / "model.safetensors")
    assert set(derived) - {"__metadata__"} == set(selected)
    with (source / "model.safetensors").open("rb") as src, (destination / "model.safetensors").open("rb") as dst:
        for name, tensor in selected.items():
            assert derived[name]["dtype"] == tensor["dtype"]
            assert derived[name]["shape"] == tensor["shape"]
            start, end = tensor["data_offsets"]
            src.seek(base + start)
            dst.seek(derived_base + derived[name]["data_offsets"][0])
            remaining = end - start
            while remaining:
                count = min(remaining, 1024 * 1024)
                assert src.read(count) == dst.read(count)
                remaining -= count
    manifest = {"recipe_sha256": digest(Path(__file__)), "retained_tensors": len(selected),
                "omitted_vision_tensors": len(original) - len(selected) - 1,
                "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                          for p in sorted(destination.iterdir())]}
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.source, args.destination, args.profile)
    with args.manifest.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
