#!/usr/bin/env python3
"""Losslessly archive large JSON payloads from a completed M3 matrix.

Indexes, receipts, logs, and the production package remain in place. Restore
the payloads before rechecking or packaging a historical matrix.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile


ARCHIVE = "numerical-payloads.tar.gz"
MANIFEST = "numerical-payloads.manifest.json"
MIN_BYTES = 8 << 20


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates(root):
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not directory.name.endswith(("-numerical", "-qualification")):
            continue
        for folder in ("raw", "report"):
            path = directory / folder
            if not path.is_dir():
                continue
            for file in sorted(path.glob("*.json")):
                if file.is_symlink() or not file.is_file():
                    raise ValueError(f"unsafe payload: {file}")
                if file.stat().st_size >= MIN_BYTES:
                    yield file


def checked_members(archive, files):
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if {member.name for member in members} != set(files) or len(members) != len(files):
            raise ValueError("archive member inventory differs from manifest")
        for member in members:
            if not member.isfile() or member.name not in files or member.size != files[member.name]["bytes"]:
                raise ValueError(f"invalid archive member: {member.name}")
            digest = hashlib.sha256()
            with source.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != files[member.name]["sha256"]:
                raise ValueError(f"archive bytes differ: {member.name}")


@contextmanager
def matrix_lock(root):
    # Lock the directory inode so aliases of the same matrix share a lock.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("an archive or restore already holds the matrix lock") from error
        yield
    finally:
        os.close(fd)


def archive(root):
    with matrix_lock(root):
        return archive_locked(root)


def archive_locked(root):
    index = root / "index.json"
    if not index.is_file() or json.loads(index.read_bytes()).get("status") not in ("passed", "failed"):
        raise ValueError("matrix has no terminal index")
    output = root / ARCHIVE
    manifest = root / MANIFEST
    pending = root / (MANIFEST + ".pending")
    leftovers = set(root.glob("*.pending"))
    resume = (output.is_file() and not output.is_symlink() and not os.path.lexists(manifest)
              and leftovers == {pending} and pending.is_file() and not pending.is_symlink())
    if not resume:
        if leftovers:
            raise ValueError("matrix has pending output")
        if os.path.lexists(output) or os.path.lexists(manifest):
            raise ValueError("matrix already has an archive or manifest")
    files = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in candidates(root)}
    if not files:
        raise ValueError("no large numerical JSON payloads")
    if resume:
        record = json.loads(pending.read_bytes())
        if record != {"version": 1, "archive": ARCHIVE, "archive_sha256": sha256(output), "files": files}:
            raise ValueError("pending manifest differs from archive or source payloads")
        checked_members(output, files)
    else:
        write_archive(root, output, pending, files)
    pending.rename(manifest)
    for name, expected in files.items():
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"source changed during archiving: {path}")
        path.unlink()
    return len(files)


def write_archive(root, output, pending, files):
    temporary = root / (ARCHIVE + ".pending")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as target:
            for name in files:
                target.add(root / name, arcname=name, recursive=False)
        checked_members(temporary, files)
        record = {"version": 1, "archive": ARCHIVE, "archive_sha256": sha256(temporary), "files": files}
        with pending.open("x") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)


def restore(root):
    with matrix_lock(root):
        return restore_locked(root)


def restore_locked(root):
    record = json.loads((root / MANIFEST).read_bytes())
    if record.get("version") != 1 or record.get("archive") != ARCHIVE:
        raise ValueError("unknown archive manifest")
    files = record["files"]
    source = root / ARCHIVE
    if sha256(source) != record["archive_sha256"]:
        raise ValueError("archive digest differs from manifest")
    checked_members(source, files)
    with tarfile.open(source, "r:gz") as archive_file:
        for member in archive_file:
            name = member.name
            path = root / name
            if path.resolve().is_relative_to(root.resolve()) is False or not path.parent.is_dir():
                raise ValueError(f"unsafe restore path: {name}")
            if path.exists():
                if sha256(path) != files[name]["sha256"]:
                    raise ValueError(f"existing file differs: {path}")
                continue
            pending = path.with_name(path.name + ".pending")
            with archive_file.extractfile(member) as input_stream, pending.open("xb") as output_stream:
                for block in iter(lambda: input_stream.read(1 << 20), b""):
                    output_stream.write(block)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            if pending.stat().st_size != files[name]["bytes"] or sha256(pending) != files[name]["sha256"]:
                pending.unlink(missing_ok=True)
                raise ValueError(f"restored bytes differ: {name}")
            pending.rename(path)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "restore"))
    parser.add_argument("matrix", type=Path, help="completed external M3 matrix directory")
    args = parser.parse_args()
    root = args.matrix.resolve(strict=True)
    count = archive(root) if args.action == "archive" else restore(root)
    print(f"{args.action}: {count} losslessly verified payloads in {root}")


if __name__ == "__main__":
    main()
