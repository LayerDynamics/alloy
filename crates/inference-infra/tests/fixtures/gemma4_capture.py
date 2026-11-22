"""Capture fixed Gemma 4 E4B reference logits offline with the pinned MLX-LM implementation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4, gemma4_text, rope_utils
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model
from tokenizers import Tokenizer


class CheckpointModel(gemma4.Model):
    def __init__(self, args):
        super().__init__(args)
        # The published checkpoint omits the unexecuted projections and key
        # norm of 18 consumer layers. Keep strict loading for all active weights.
        text = self.language_model.model
        for i, owner in enumerate(text.previous_kvs):
            if owner != i:
                assert i >= 24 and owner in (22, 23)
                attn = text.layers[i].self_attn
                for name in ("k_proj", "v_proj", "k_norm"):
                    delattr(attn, name)


def capture(root, output):
    profile = json.loads(Path(__file__).with_name("gemma4_e4b_profile.json").read_text())
    reference = profile["reference"]
    for package, version in reference["versions"].items():
        assert importlib.metadata.version(package) == version
    for module in (gemma4, gemma4_text, rope_utils):
        path = Path(module.__file__)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["source_sha256"][path.name]
    for entry in profile["artifact"]["files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        with path.open("rb") as src:
            assert hashlib.file_digest(src, "sha256").hexdigest() == entry["sha256"]
    mx.disable_compile()
    mx.set_default_device(mx.gpu)
    model, _ = load_model(root, strict=True, get_model_classes=lambda config: (CheckpointModel, gemma4.ModelArgs))
    expected = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    assert {k for k, _ in tree_flatten(model.parameters())} == {k for k in expected if k.startswith("language_model.")}
    model.apply(lambda a: a.astype(mx.float32) if mx.issubdtype(a.dtype, mx.floating) else a)
    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    cases = []
    for prompt, ids in zip(profile["prompts"], profile["prompt_tokens"], strict=True):
        assert tokenizer.encode(prompt, add_special_tokens=False).ids == ids
        cache = make_prompt_cache(model)
        rows = []
        for tokens in [ids, *([d] for d in profile["decode_tokens"])]:
            row = model(mx.array([tokens]), cache=cache)[0, -1]
            mx.eval(row)
            assert row.shape == (262144,)
            rows.append(row.tolist())
        cases.append(dict(tokens=ids, decode_tokens=profile["decode_tokens"], logits=rows))
    result = dict(profile=profile["id"], revision=profile["artifact"]["revision"],
                  versions=reference["versions"], device="gpu", compile_enabled=False, cases=cases)
    data = (json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    if "sha256" in reference:
        assert digest == reference["sha256"], digest
    output.write_bytes(data)
    print(f"reference_sha256={digest} rows={sum(len(c['logits']) for c in cases)} bytes={len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(args.artifact, args.output)
