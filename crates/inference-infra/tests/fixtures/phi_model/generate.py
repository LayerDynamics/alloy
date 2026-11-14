#!/usr/bin/env python3
"""Offline Phi fused-model reference using pinned upstream MLX, never production."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.phi3 import Model, ModelArgs
from mlx_lm.models.cache import make_prompt_cache

versions = {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "numpy"]}
assert versions == {"mlx": "0.32.4", "mlx-lm": "0.31.2", "numpy": "2.4.2"}
mx.disable_compile()
mx.set_default_device(mx.gpu)
root = Path(__file__).parent
config = dict(model_type="phi3", architectures=["Phi3ForCausalLM"], hidden_size=64,
              intermediate_size=128, vocab_size=128, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
              rope_theta=10000.0, partial_rotary_factor=0.75, hidden_act="silu",
              max_position_embeddings=64, original_max_position_embeddings=16,
              tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
              lm_head_bias=False, full_attn_mod=1, sliding_window=64,
              rope_scaling=dict(type="longrope", short_factor=[1.0]*6,
                                long_factor=[1.0,1.25,1.5,2.0,2.5,3.0]),
              quantization=dict(bits=4,group_size=64))
model = Model(ModelArgs.from_dict(config))
rng = np.random.default_rng(20261114)
weights = []
for name, array in sorted(tree_flatten(model.parameters())):
    data = rng.integers(-64,65,size=array.shape).astype(np.float32)/np.float32(256)
    if "norm.weight" in name:
        data = np.float32(1) + data
    weights.append((name,mx.array(data,dtype=mx.float32)))
model.load_weights(weights)
nn.quantize(model,group_size=64,bits=4)
model.eval()
mx.eval(model.parameters())
mx.save_safetensors(str(root/"model.safetensors"),dict(tree_flatten(model.parameters())))
(root/"config.json").write_text(json.dumps(config,indent=2)+"\n")
tokens = [3,41,7,120,63,95,12,110]
reference = dict(versions=versions,compile_enabled=False,regime="long",tokens=tokens)
for name, device in [("gpu",mx.gpu),("cpu",mx.cpu)]:
    mx.set_default_device(device)
    full=model(mx.array([tokens]));mx.eval(full)
    results=dict(full=full.flatten().tolist(),splits=[])
    for split in range(1,len(tokens)):
        cache=make_prompt_cache(model)
        prefix=model(mx.array([tokens[:split]]),cache=cache);mx.eval(prefix)
        suffix=model(mx.array([tokens[split:]]),cache=cache);mx.eval(suffix)
        results["splits"].append(dict(prefix=prefix.flatten().tolist(),suffix=suffix.flatten().tolist()))
    reference[name]=results
    print("captured",name,flush=True)
(root/"reference.json").write_text(json.dumps(reference,separators=(",",":"),allow_nan=False)+"\n")
manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ["config.json","model.safetensors","reference.json"]}
(root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest),flush=True)
