#!/usr/bin/env python3
"""Offline fixed mixed-layer Qwen3.5 reference; never invoked in inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

for package, version in [("mlx", "0.32.4"), ("mlx-lm", "0.31.2"), ("numpy", "2.4.2")]:
    assert importlib.metadata.version(package) == version
mx.disable_compile()
mx.set_default_device(mx.gpu)
root = Path(__file__).parent
text = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, vocab_size=256,
            max_position_embeddings=128, linear_num_value_heads=2,
            linear_num_key_heads=1, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4,
            tie_word_embeddings=False, attention_bias=False,
            attn_output_gate=True, hidden_act="silu", full_attention_interval=2,
            layer_types=["linear_attention", "full_attention"] * 2,
            partial_rotary_factor=0.25,
            rope_parameters=dict(type="default", rope_theta=10000000,
                                 partial_rotary_factor=0.25))
config = dict(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
              tie_word_embeddings=False, text_config=text)
model = TextModel(TextModelArgs.from_dict(text))
rng = np.random.default_rng(20261114)
weights = []
for name, array in sorted(tree_flatten(model.parameters())):
    data = rng.integers(-64, 65, size=array.shape).astype(np.float32) / np.float32(256)
    if "norm.weight" in name or "layernorm.weight" in name:
        data = np.float32(1) + data
    weights.append((name, mx.array(data, dtype=mx.float32)))
model.load_weights(weights)
model.eval()
mx.eval(model.parameters())
(root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
mx.save_safetensors(str(root / "model.safetensors"), {"language_model." + n: w for n, w in weights})
tokens = [3, 41, 7, 200, 128, 95, 12, 250]
full = model(mx.array([tokens]))
mx.eval(full)
cache = model.make_cache()
prefill = model(mx.array([tokens[:4]]), cache=cache)
mx.eval(prefill)
continuation = []
for token in tokens[4:]:
    output = model(mx.array([[token]]), cache=cache)
    mx.eval(output)
    continuation.append(output.flatten().tolist())
reference = dict(mlx="0.32.4", mlx_lm="0.31.2", numpy="2.4.2", compile=False,
                 device="gpu", tokens=tokens, full=full.flatten().tolist(),
                 prefill=prefill.flatten().tolist(), continuation=continuation)
# Preserve the GPU capture and capture the identical equations/weights on CPU.
# Device-specific reduction kernels can differ near zero; both remain subject
# to the same strict thresholds against their corresponding upstream device.
mx.set_default_device(mx.cpu)
cpu_full = model(mx.array([tokens]))
mx.eval(cpu_full)
cpu_cache = model.make_cache()
cpu_prefill = model(mx.array([tokens[:4]]), cache=cpu_cache)
mx.eval(cpu_prefill)
cpu_continuation = []
for token in tokens[4:]:
    output = model(mx.array([[token]]), cache=cpu_cache)
    mx.eval(output)
    cpu_continuation.append(output.flatten().tolist())
reference["cpu"] = dict(full=cpu_full.flatten().tolist(),
                        prefill=cpu_prefill.flatten().tolist(),
                        continuation=cpu_continuation)
# Record each chunk geometry independently: matrix and attention reductions
# change with chunk shape even in upstream MLX. FR8 compares the same workload;
# cached-versus-full equivalence is a separate cache correctness assertion.
for device, target in [(mx.gpu, reference), (mx.cpu, reference["cpu"])]:
    mx.set_default_device(device)
    target["splits"] = []
    for split in range(1, len(tokens)):
        split_cache = model.make_cache()
        prefix = model(mx.array([tokens[:split]]), cache=split_cache)
        mx.eval(prefix)
        suffix = model(mx.array([tokens[split:]]), cache=split_cache)
        mx.eval(suffix)
        target["splits"].append(suffix.flatten().tolist())
if (root / "reference.json").exists():
    previous = json.loads((root / "reference.json").read_text())
    for key in ["full", "prefill", "continuation"]:
        assert previous[key] == reference[key], "fixed GPU reference changed"
(root / "reference.json").write_text(json.dumps(reference, separators=(",", ":"), allow_nan=False) + "\n")
manifest = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ["config.json", "model.safetensors", "reference.json"]}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
