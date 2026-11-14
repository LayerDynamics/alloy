#!/usr/bin/env python3
"""Offline reference for the complete trained recurrent block, never inference.

Pinned MLX 0.32.4 / mlx-lm 0.31.2, eager float32 on Metal. Deterministic binary
fraction inputs/weights, no Rust implementation imported. Preserve all output
and state values, not just checksums or greedy predictions.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

assert importlib.metadata.version("mlx") == "0.32.4"
assert importlib.metadata.version("mlx-lm") == "0.31.2"
mx.disable_compile()
mx.set_default_device(mx.gpu)


def values(n, salt, denominator=256):
    return [(((i * 17 + salt * 13) % 61) - 30) / denominator for i in range(n)]


args = TextModelArgs(hidden_size=64, linear_num_key_heads=1,
                     linear_num_value_heads=2, linear_key_head_dim=128,
                     linear_value_head_dim=128, linear_conv_kernel_dim=4,
                     rms_norm_eps=1e-6)
model = GatedDeltaNet(args)
weights = [
    ("in_proj_qkv.weight", (512, 64), 1),
    ("in_proj_z.weight", (256, 64), 2),
    ("in_proj_a.weight", (2, 64), 3),
    ("in_proj_b.weight", (2, 64), 4),
    ("conv1d.weight", (512, 4, 1), 5),
    ("A_log", (2,), 6),
    ("dt_bias", (2,), 7),
    ("norm.weight", (128,), 8),
    ("out_proj.weight", (64, 256), 9),
]
import math
arrays = []
for name, shape, salt in weights:
    data = values(math.prod(shape), salt)
    if name == "norm.weight":
        data = [1 + v for v in data]
    arrays.append((name, mx.array(data, dtype=mx.float32).reshape(shape)))
model.load_weights(arrays)
model.eval()
inputs = mx.array(values(5 * 64, 10), dtype=mx.float32).reshape(1, 5, 64)
cache = ArraysCache(2)
out = model(inputs, cache=cache)
mx.eval(out, cache[0], cache[1])
payload = {
    "mlx": "0.32.4", "mlx_lm": "0.31.2", "compile": False,
    "device": "gpu", "sequence": 5, "hidden": 64,
    "output": out.flatten().tolist(),
    "convolution": cache[0].flatten().tolist(),
    "recurrent": cache[1].flatten().tolist(),
}
path = Path(__file__).with_name("reference.json")
path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
print(path, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
