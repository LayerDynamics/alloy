#!/usr/bin/env python3
"""Independent numerical reference for the Qwen3 tiny fixture.

Generates the same deterministic weights as
crates/families/qwen3/src/fixture.rs using one SplitMix64 stream,
canonical tensor order, and matching u32-to-f32 conversion.

Computes the forward pass in NumPy and writes reference_logits.txt:
one line per input token, one value per vocabulary entry.

The Rust implementation in model.rs::forward is compared against these
logits with absolute tolerance 2e-2 and relative tolerance 1e-2.

The reference includes learned per-head Q/K RMS normalization before
RoPE. Each normalization scale has shape [head_dim] and is shared
across the corresponding attention heads.

Canonical per-layer weight-generation order:
input_layernorm, q_proj, k_proj, v_proj, q_norm, k_norm, o_proj,
post_attention_layernorm, gate_proj, up_proj, down_proj.

This is a float64 NumPy reference over deterministically generated
float32 fixture weights. It is not a bit-exact backend reference or
proof of full model-loading and generation conformance.
"""

from pathlib import Path

import numpy as np


MASK64 = (1 << 64) - 1
SEED = 0x51454E3346495853  # Must equal FIXTURE_SEED in fixture.rs.
TOKENS = [3, 41, 7, 200, 128, 95, 12, 250]  # Must match the Rust test.

CONFIG = dict(
    hidden=32,
    layers=2,
    q_heads=4,
    kv_heads=2,
    head_dim=8,
    inter=64,
    vocab=256,
    rope_theta=10000.0,
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
                "q_norm": f32_stream(rng, c["head_dim"]).astype(np.float64),
                "k_norm": f32_stream(rng, c["head_dim"]).astype(np.float64),
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

    for layer in weights["layers"]:
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

        q = rmsnorm(q, layer["q_norm"], c["eps"])
        k = rmsnorm(k, layer["k_norm"], c["eps"])

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

    out = Path(__file__).with_name("reference_logits.txt")
    np.savetxt(out, logits, fmt="%.9e")

    print(f"wrote {out}: {logits.shape[0]} tokens x {logits.shape[1]} vocab")
    print(f"first row head: {logits[0, :4]}")


if __name__ == "__main__":
    main()
