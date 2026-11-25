#!/usr/bin/env python3
"""Offline native-position reference; pinned upstream, never production."""
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import ministral3, rope_utils

root = Path(__file__).parent
host = (platform.system(), platform.machine())
if host == ("Linux", "aarch64"):
    devices = [("cpu", mx.cpu)]
    suffix = "-linux-aarch64"
elif host == ("Darwin", "arm64"):
    devices = [("cpu", mx.cpu), ("gpu", mx.gpu)]
    suffix = ""
else:
    raise SystemExit(f"No pinned position capture for {host}")
reference_name = f"reference{suffix}.json"
versions = {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm"]}
assert versions == {"mlx": "0.32.4", "mlx-lm": "0.31.2"}
assert hashlib.sha256((root / "config.json").read_bytes()).hexdigest() == "cdd4bfa0a702d0241faaa3b80d15d481d7f8abf2659fba0958b1e7f24a28055f"
text = json.loads((root / "config.json").read_text())["text_config"]
mx.disable_compile()
data = {"versions": versions, "compile_enabled": False, "devices": {}}
# Full real head width; bounded grouped-head fixture. Geometry checked separately.
q_data = [((i * 17 % 97) - 48) / 64 for i in range(4 * 3 * 128)]
k_data = [((i * 11 % 89) - 44) / 64 for i in range(2 * 3 * 128)]
for device_name, device in devices:
    mx.set_default_device(device)
    q = mx.array(q_data, dtype=mx.float32).reshape(1, 4, 3, 128)
    k = mx.array(k_data, dtype=mx.float32).reshape(1, 2, 3, 128)
    cases = []
    for all_dim in [1.0, 0.0]:
        parameters = dict(text["rope_parameters"], mscale_all_dim=all_dim)
        rope = rope_utils.initialize_rope(128, parameters["rope_theta"], False, parameters, text["max_position_embeddings"])
        for offset in [0, 16383, 16384, 32767, 262141]:
            scale = ministral3._get_llama_4_attn_scale(3, offset, parameters["llama_4_scaling_beta"], parameters["original_max_position_embeddings"])
            qr = rope(q, offset=offset) * scale
            kr = rope(k, offset=offset)
            mx.eval(qr, kr, scale, rope._freqs)
            cases.append(dict(offset=offset, mscale_all_dim=all_dim, amplitude=rope.mscale,
                              frequencies=rope._freqs.tolist(), scales=scale.flatten().tolist(),
                              query=qr.flatten().tolist(), key=kr.flatten().tolist()))
    data["devices"][device_name] = cases
    print("captured", device_name, len(cases), flush=True)
(root / reference_name).write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
manifest = {"files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ["config.json", "generate.py", reference_name]},
            "upstream": {Path(inspect.getfile(mod)).name: hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest()
                         for mod in [ministral3, rope_utils]},
            "artifact_revision": "a962dcb09eee4169c890e544c9eb938f1113fdee",
            "scope": "Position operations only, not complete model numerical qualification"}
(root / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest), flush=True)
