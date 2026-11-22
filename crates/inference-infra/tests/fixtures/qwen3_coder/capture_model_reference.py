#!/usr/bin/env python3
"""Offline eager Qwen3 and Qwen3-Coder oracle. Never imported or invoked by production inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model
from tokenizers import Tokenizer


def capture(artifact, output, profile_path):
    profile = json.loads(profile_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "numpy"]}
    assert versions == profile["reference"]["versions"], versions
    for file in profile["artifact"]["files"]:
        path = artifact / file["path"]
        assert path.stat().st_size == file["bytes"], path
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        assert digest == file["sha256"], path
    mx.disable_compile()
    mx.set_default_device(mx.gpu)
    model, config = load_model(artifact, lazy=True)
    architecture = {
        "qwen3-coder-30b-a3b-instruct-4bit": "qwen3_moe",
        "qwen3-4b-4bit": "qwen3",
        "qwen3-14b-4bit": "qwen3",
    }[profile["id"]]
    assert config["model_type"] == architecture
    assert type(model).__module__ == "mlx_lm.models." + architecture
    # Native payload retains packed U32 matrices and promotes floating metadata
    # and norms to F32. Match that declared representation in the oracle.
    model.apply(lambda a: a.astype(mx.float32) if mx.issubdtype(a.dtype, mx.floating) else a)
    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(artifact / "tokenizer.json"))
    cases = []
    for index, prompt in enumerate(profile["prompts"]):
        tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
        assert tokens == profile["prompt_tokens"][index]
        cache = make_prompt_cache(model)
        logits = model(mx.array([tokens]), cache=cache)[:, -1, :]
        mx.eval(logits)
        rows = [logits[0].tolist()]
        for token in profile["decode_tokens"]:
            logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
            mx.eval(logits)
            rows.append(logits[0].tolist())
        cases.append(dict(tokens=tokens, decode_tokens=profile["decode_tokens"], logits=rows))
        print("captured prompt", tokens, flush=True)
    result = dict(profile=profile["id"], artifact_revision=profile["artifact"]["revision"],
                  versions=versions, compile_enabled=False, device="gpu",
                  cases=cases)
    with output.open("x") as f:
        json.dump(result, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    print("reference_sha256", hashlib.sha256(output.read_bytes()).hexdigest(), flush=True)
    print("memory", mx.get_active_memory(), mx.get_peak_memory(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path,
                        default=Path(__file__).parent.parent / "qwen3_coder_profile.json")
    args = parser.parse_args()
    capture(args.artifact, args.output, args.profile)
