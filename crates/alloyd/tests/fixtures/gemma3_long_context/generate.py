"""Reproduce the bounded Gemma fixture without changing the frozen corpus.

Run with the pinned reference environment and --artifact pointing to either
frozen Gemma 3 model. Default mode verifies; --write updates intended fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[4] / "bench/corpus/fixtures/documents/document-32k.txt"
SOURCE_SHA = "393bc22d01056d7e6f5fbba865071725867679aa00521f84780802c660cb0c75"
TOKENIZER_SHA = "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"


def prompt(document):
    data = document.encode()
    middle = len(data) // 2
    while data[middle] & 0xC0 == 0x80:
        middle += 1
    data = data[:middle] + b"\nARCHIVE RECORD: The vault access code is COBALT-7392.\n" + data[middle:]
    return (
        "<bos><start_of_turn>user\n"
        "Read this archive and retrieve the vault access code.\n"
        + data.decode()
        + "\nWhat is the vault access code in the ARCHIVE RECORD? Reply with only the code."
        "<end_of_turn>\n<start_of_turn>model\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    tokenizer_bytes = (args.artifact / "tokenizer.json").read_bytes()
    assert hashlib.sha256(tokenizer_bytes).hexdigest() == TOKENIZER_SHA
    tokenizer = Tokenizer.from_buffer(tokenizer_bytes)
    lines = source.decode().splitlines(keepends=True)
    # Remove complete trailing lines only at fixture creation, never at runtime.
    while True:
        document = "".join(lines)
        count = len(tokenizer.encode(prompt(document), add_special_tokens=False).ids)
        if count <= 32100:
            break
        lines.pop()
    assert 32000 <= count <= 32100
    fixture = document.encode()
    manifest = {
        "source": "bench/corpus/fixtures/documents/document-32k.txt",
        "source_sha256": SOURCE_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "fixture_bytes": len(fixture),
        "prompt_tokens": count,
        "context_limit": 32768,
        "output_reserve": 512,
        "safety_margin": 64,
    }
    assert count + manifest["output_reserve"] + manifest["safety_margin"] <= manifest["context_limit"]
    encoded_manifest = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, data in [("document.txt", fixture), ("manifest.json", encoded_manifest)]:
        path = ROOT / name
        if args.write:
            path.write_bytes(data)
        else:
            assert path.read_bytes() == data, f"fixture drift: {path}"
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
