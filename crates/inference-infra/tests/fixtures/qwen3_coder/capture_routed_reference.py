"""Offline CPU oracle for the first trained Qwen3-Coder sparse MLP."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_moe import ModelArgs, Qwen3MoeSparseMoeBlock

SHARD = "model-00001-of-00004.safetensors"
SHARD_SHA = "25c6d0a62e02224605774ce2e9dc9a8c95910cde33557b938e55a368da1b5212"


def capture(artifact, destination):
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")}
    if versions != {"mlx": "0.32.4", "mlx-lm": "0.31.2", "numpy": "2.4.2"}:
        raise ValueError(f"unexpected oracle versions {versions}")
    config = (artifact / "config.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != "8200545374776769bfd5ae4a30533990c96ca1cad625a9500194f46a8549fa13":
        raise ValueError("unexpected pinned model configuration")
    with (artifact / SHARD).open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SHARD_SHA:
            raise ValueError("unexpected trained layer shard")
    config = json.loads(config)
    args = ModelArgs.from_dict(config)
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    block = Qwen3MoeSparseMoeBlock(args)
    quant = config["quantization"]

    def quantizer(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        return quant.get("model.layers.0.mlp." + path, {"bits": quant["bits"], "group_size": quant["group_size"]})

    nn.quantize(block, class_predicate=quantizer)
    prefix = "model.layers.0.mlp."
    trained = [(name[len(prefix):], tensor) for name, tensor in mx.load(str(artifact / SHARD)).items()
               if name.startswith(prefix)]
    block.load_weights(trained, strict=True)
    block.apply(lambda a: a.astype(mx.float32) if mx.issubdtype(a.dtype, mx.floating) else a)
    block.eval()
    cases = []
    for count in (1, 9):
        # Exactly representable values allow independent generation on the Rust side.
        x = (((mx.arange(count * args.hidden_size) * 37) % 251 - 125).astype(mx.float32) / 128)
        x = x.reshape(1, count, args.hidden_size)
        y = block(x)
        mx.eval(y)
        cases.append({"tokens": count, "output": y.reshape(-1).tolist()})
    with destination.open("x") as f:
        json.dump({"shard_sha256": SHARD_SHA, "versions": versions, "cases": cases}, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture(args.artifact, args.output)
