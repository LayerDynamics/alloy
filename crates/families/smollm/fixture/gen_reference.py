#!/usr/bin/env python3
"""Independent NumPy float64 SmolLM3 tiny-fixture oracle.

Uses numpy==2.4.2 and the documented SplitMix64 fixture stream, with no
Q/K normalization, epsilon 1e-6 and alternating RoPE/NoPE layers. Outputs
all vocabulary logits at each of eight fixed positions. This is an offline
test asset generator, never part of inference. See the Phase 3 execution plan.
"""
from pathlib import Path

import numpy as np


MASK64 = (1 << 64) - 1
SEED = 0x51454E3346495853  # Must equal FIXTURE_SEED in fixture.rs.
TOKENS = [3, 41, 7, 200, 128, 95, 12, 250]  # Must match the Rust test.

CONFIG = dict(
    hidden=32,
    layers=4,
    q_heads=4,
    kv_heads=2,
    head_dim=8,
    inter=64,
    vocab=256,
    rope_theta=5000000.0,
    eps=1e-6,
)


class SplitMix64:
    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def f32_stream(rng, n):
    """Match fixture.rs::unit over the low 32 bits of each draw."""
    bits = np.empty(n, dtype=np.uint32)
    for i in range(n):
        bits[i] = rng.next() & 0xFFFFFFFF

    values = bits.astype(np.float32)
    return (values / np.float32(np.iinfo(np.uint32).max)) * np.float32(
        2.0
    ) - np.float32(1.0)


def generate():
    """Generate weights in the same order as Rust's generate_fixture."""
    c = CONFIG
    rng = SplitMix64(SEED)
    q_dim = c["q_heads"] * c["head_dim"]
    kv_dim = c["kv_heads"] * c["head_dim"]

    def mat(rows, cols):
        return f32_stream(rng, rows * cols).reshape(rows, cols)

    weights = {
        "embed": mat(c["vocab"], c["hidden"]).astype(np.float64),
        "layers": [],
    }

    for _ in range(c["layers"]):
        weights["layers"].append(
            {
                "input_layernorm": f32_stream(rng, c["hidden"]).astype(np.float64),
                "q_proj": mat(q_dim, c["hidden"]).astype(np.float64),
                "k_proj": mat(kv_dim, c["hidden"]).astype(np.float64),
                "v_proj": mat(kv_dim, c["hidden"]).astype(np.float64),
                "o_proj": mat(c["hidden"], q_dim).astype(np.float64),
                "post_attention_layernorm": f32_stream(rng, c["hidden"]).astype(
                    np.float64
                ),
                "gate_proj": mat(c["inter"], c["hidden"]).astype(np.float64),
                "up_proj": mat(c["inter"], c["hidden"]).astype(np.float64),
                "down_proj": mat(c["hidden"], c["inter"]).astype(np.float64),
            }
        )

    weights["final_layernorm"] = f32_stream(rng, c["hidden"]).astype(np.float64)

    return weights


def rmsnorm(x, weight, eps):
    """Normalize over the last dimension, then apply the learned scale."""
    return x * weight / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rope(x, positions, theta):
    """Apply split-half RoPE to [tokens, heads, head_dim] input."""
    if x.ndim != 3:
        raise ValueError(f"RoPE expects [tokens, heads, head_dim], got {x.shape}")

    tokens, _, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (tokens,):
        raise ValueError(f"Expected {tokens} positions, got shape {positions.shape}")

    half = head_dim // 2
    j = np.arange(half, dtype=np.float64)
    inv_freq = theta ** (-(2.0 * j) / head_dim)
    angles = positions[:, None] * inv_freq[None, :]

    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]

    lo = x[..., :half]
    hi = x[..., half:]

    out = np.empty_like(x)
    out[..., :half] = lo * cos - hi * sin
    out[..., half:] = lo * sin + hi * cos
    return out


def attention(q, k, v, scale):
    """Grouped-query causal attention over [tokens, heads, head_dim]."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Attention expects three-dimensional Q, K, and V tensors")

    tokens, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]

    if kv_heads == 0 or q_heads % kv_heads != 0:
        raise ValueError("Query-head count must be a multiple of the KV-head count")

    if k.shape != (tokens, kv_heads, head_dim):
        raise ValueError(f"Unexpected key shape: {k.shape}")

    if v.shape != k.shape:
        raise ValueError(f"Value shape {v.shape} must match key shape {k.shape}")

    group = q_heads // kv_heads
    expanded_k = np.repeat(k, group, axis=1)
    expanded_v = np.repeat(v, group, axis=1)

    scores = np.einsum("thd,shd->hts", q, expanded_k) * scale

    key_positions = np.arange(tokens)[None, None, :]
    query_positions = np.arange(tokens)[None, :, None]
    scores = np.where(
        key_positions <= query_positions,
        scores,
        -np.inf,
    )

    maximum = scores.max(axis=-1, keepdims=True)
    exponentials = np.exp(scores - maximum)
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)

    return np.einsum(
        "hts,shd->thd",
        probabilities,
        expanded_v,
    )


def silu(x):
    """Numerically stable SiLU without overflowing exp for negative x."""
    sigmoid = np.empty_like(x)
    nonnegative = x >= 0

    sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))

    exp_x = np.exp(x[~nonnegative])
    sigmoid[~nonnegative] = exp_x / (1.0 + exp_x)

    return x * sigmoid


def forward(weights, tokens):
    c = CONFIG
    tokens = np.asarray(tokens, dtype=np.int64)

    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("Expected a nonempty one-dimensional token sequence")

    if np.any(tokens < 0) or np.any(tokens >= c["vocab"]):
        raise ValueError("Token IDs must be within the fixture vocabulary")

    token_count = len(tokens)
    positions = np.arange(token_count, dtype=np.float64)
    scale = 1.0 / np.sqrt(c["head_dim"])
    q_dim = c["q_heads"] * c["head_dim"]

    h = weights["embed"][tokens]

    for index, layer in enumerate(weights["layers"]):
        x = rmsnorm(h, layer["input_layernorm"], c["eps"])

        q = (x @ layer["q_proj"].T).reshape(
            token_count,
            c["q_heads"],
            c["head_dim"],
        )
        k = (x @ layer["k_proj"].T).reshape(
            token_count,
            c["kv_heads"],
            c["head_dim"],
        )
        v = (x @ layer["v_proj"].T).reshape(
            token_count,
            c["kv_heads"],
            c["head_dim"],
        )


        if index % 2 == 0:
            q = rope(q, positions, c["rope_theta"])
            k = rope(k, positions, c["rope_theta"])

        attended = attention(q, k, v, scale).reshape(
            token_count,
            q_dim,
        )
        h = h + attended @ layer["o_proj"].T

        x = rmsnorm(h, layer["post_attention_layernorm"], c["eps"])
        gate = x @ layer["gate_proj"].T
        up = x @ layer["up_proj"].T

        h = h + (silu(gate) * up) @ layer["down_proj"].T

    h = rmsnorm(h, weights["final_layernorm"], c["eps"])
    return h @ weights["embed"].T


def main():
    weights = generate()
    logits = forward(weights, TOKENS)

    expected_shape = (len(TOKENS), CONFIG["vocab"])
    if logits.shape != expected_shape:
        raise ValueError(f"Expected logits shape {expected_shape}, got {logits.shape}")

    if not np.isfinite(logits).all():
        raise ValueError("Reference logits contain NaN or infinity")

    out = Path(__file__).with_name("reference_logits.json")
    import json
    # Capture the specified eager float32 reference as well as the independent
    # float64 mathematical oracle. Near-zero logits cannot be compared across
    # precisions using a pure relative bound alone.
    import mlx.core as mx
    from mlx_lm.models.smollm3 import Model, ModelArgs
    mx.disable_compile()
    c = CONFIG
    model = Model(ModelArgs(model_type="smollm3", hidden_size=c["hidden"],
        num_hidden_layers=c["layers"], intermediate_size=c["inter"],
        num_attention_heads=c["q_heads"], num_key_value_heads=c["kv_heads"],
        head_dim=c["head_dim"], rms_norm_eps=c["eps"], vocab_size=c["vocab"],
        rope_theta=c["rope_theta"], tie_word_embeddings=True,
        no_rope_layers=[1, 0, 1, 0]))
    tensors = [("model.embed_tokens.weight", weights["embed"]),
               ("model.norm.weight", weights["final_layernorm"])]
    for i, layer in enumerate(weights["layers"]):
        for key, value in layer.items():
            group = "self_attn." if key in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp." if key in ["gate_proj", "up_proj", "down_proj"] else ""
            tensors.append((f"model.layers.{i}.{group}{key}.weight", value))
    model.load_weights([(name, mx.array(value.astype(np.float32))) for name, value in tensors])
    mx.eval(model.parameters())
    # CPU placement matches the portable native tiny-fixture check.
    with mx.stream(mx.cpu):
        eager = model(mx.array([TOKENS]))
        mx.eval(eager)
    eager = np.array(eager[0])
    print("independent_f64_vs_eager_f32_max_abs", np.max(np.abs(eager - logits)), flush=True)
    out.write_text(json.dumps({"numpy_float64": logits.tolist(), "mlx_eager_float32": eager.tolist()}, allow_nan=False) + "\n")

    print(f"wrote {out}: {logits.shape[0]} tokens x {logits.shape[1]} vocab")
    print(f"first row head: {logits[0, :4]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.artifact is None:
        main()
    else:
        # Offline capture only. This never participates in native inference.
        import hashlib
        import importlib.metadata
        import json
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from mlx_lm.models.cache import make_prompt_cache
        from tokenizers import Tokenizer
        if args.output is None:
            parser.error("--artifact requires --output")
        profile = json.loads(Path(__file__).with_name("profile.json").read_text())
        for file in profile["artifact"]["files"]:
            with (args.artifact / file["path"]).open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            assert digest == file["sha256"], file["path"]
        versions = {name: importlib.metadata.version(name) for name in ["mlx", "mlx-lm", "numpy"]}
        assert versions == {"mlx": "0.32.4", "mlx-lm": "0.31.2", "numpy": "2.4.2"}, versions
        mx.disable_compile()
        model, config = load_model(args.artifact, lazy=False)
        model.apply(lambda x: x.astype(mx.float32) if mx.issubdtype(x.dtype, mx.floating) else x)
        mx.eval(model.parameters())
        tokenizer = Tokenizer.from_file(str(args.artifact / "tokenizer.json"))
        cases = []
        for question in ["What is 2 + 2?", "Explain why the sky is blue."]:
            text = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face. Reasoning Mode: /no_think<|im_end|>\n"
            text += "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
            tokens = tokenizer.encode(text, add_special_tokens=False).ids
            cache = make_prompt_cache(model)
            logits = model(mx.array([tokens]), cache=cache)[:, -1, :]
            mx.eval(logits)
            rows = [np.array(logits[0]).tolist()]
            for token in [19, 13]:
                logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
                mx.eval(logits)
                rows.append(np.array(logits[0]).tolist())
            cases.append({"tokens": tokens, "decode_tokens": [19, 13], "logits": rows})
        result = {"profile": profile["id"], "artifact_revision": profile["artifact"]["revision"],
                  "versions": versions, "compile_enabled": False, "cases": cases}
        args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
        print("reference_sha256", hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)
        print("prompt_tokens", [case["tokens"] for case in cases], flush=True)
