#!/usr/bin/env python3
"""Independent reference for the Qwen3 tiny fixture.

Generates the same deterministic weights as
crates/families/qwen3/src/fixture.rs (one SplitMix64 stream, canonical
tensor order, op-for-op identical u32->f32 conversion) and computes the
forward pass in numpy. Writes reference_logits.txt: one line per input
token, one value per vocabulary entry.

The Rust implementation (model.rs::forward) must match within FR-8
tolerances (2e-2 abs / 1e-2 rel). This file defines the semantics.
"""

import numpy as np
from pathlib import Path

MASK64 = (1 << 64) - 1
SEED = 0x51454E3346495853  # must equal FIXTURE_SEED in fixture.rs
TOKENS = [3, 41, 7, 200, 128, 95, 12, 250]  # must equal TOKENS in the test

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
    """n values in [-1, 1), op-for-op identical to fixture.rs::unit:
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0 over the low 32 bits."""
    bits = np.empty(n, dtype=np.uint32)
    for i in range(n):
        bits[i] = rng.next() & 0xFFFFFFFF
    b = bits.astype(np.float32)
    return (b / np.float32(np.iinfo(np.uint32).max)) * np.float32(2.0) - np.float32(1.0)


def generate():
    c = CONFIG
    rng = SplitMix64(SEED)
    q_dim = c["q_heads"] * c["head_dim"]
    kv_dim = c["kv_heads"] * c["head_dim"]

    def mat(rows, cols):
        return f32_stream(rng, rows * cols).reshape(rows, cols)

    w = {"embed": mat(c["vocab"], c["hidden"]).astype(np.float64)}
    w["layers"] = []
    for _ in range(c["layers"]):
        w["layers"].append(
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
    w["final_layernorm"] = f32_stream(rng, c["hidden"]).astype(np.float64)
    return w


def rmsnorm(x, weight, eps):
    return x * weight / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rope(x, positions, theta):
    """Split-half RoPE, matching inference-infra apply_rope: dim j pairs
    with dim j + d/2; angle = position * theta^(-2j/d)."""
    tokens, heads, d = x.shape
    half = d // 2
    j = np.arange(half, dtype=np.float64)
    inv_freq = theta ** (-(2.0 * j) / d)
    ang = positions[:, None] * inv_freq[None, :]  # [tokens, half]
    cos = np.cos(ang)[:, None, :]  # [tokens, 1, half]
    sin = np.sin(ang)[:, None, :]
    lo, hi = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = lo * cos - hi * sin
    out[..., half:] = lo * sin + hi * cos
    return out


def attention(q, k, v, scale):
    """GQA causal attention. Query head h uses KV head h // group,
    matching inference-infra attention; causal mask by allowed prefix."""
    tokens, q_heads, d = q.shape
    kv_heads = k.shape[1]
    group = q_heads // kv_heads
    kx = np.repeat(k, group, axis=1)
    vx = np.repeat(v, group, axis=1)
    scores = np.einsum("thd,shd->hts", q, kx) * scale
    s_idx = np.arange(tokens)[None, None, :]
    t_idx = np.arange(tokens)[:, None, None]
    scores = np.where(s_idx <= t_idx, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    w = e / e.sum(axis=-1, keepdims=True)
    return np.einsum("hts,shd->thd", w, vx)


def forward(w, tokens):
    c = CONFIG
    T = len(tokens)
    pos = np.arange(T, dtype=np.float64)
    scale = 1.0 / np.sqrt(c["head_dim"])
    q_dim = c["q_heads"] * c["head_dim"]
    kv_dim = c["kv_heads"] * c["head_dim"]

    h = w["embed"][tokens]  # tied embeddings
    for lw in w["layers"]:
        x = rmsnorm(h, lw["input_layernorm"], c["eps"])
        q = rope(x @ lw["q_proj"].T, pos, c["rope_theta"]).reshape(
            T, c["q_heads"], c["head_dim"]
        )
        k = rope(x @ lw["k_proj"].T, pos, c["rope_theta"]).reshape(
            T, c["kv_heads"], c["head_dim"]
        )
        v = (x @ lw["v_proj"].T).reshape(T, c["kv_heads"], c["head_dim"])
        a = attention(q, k, v, scale).reshape(T, q_dim)
        h = h + a @ lw["o_proj"].T

        x = rmsnorm(h, lw["post_attention_layernorm"], c["eps"])
        gate = x @ lw["gate_proj"].T
        up = x @ lw["up_proj"].T
        h = h + (gate / (1.0 + np.exp(-gate)) * up) @ lw["down_proj"].T

    h = rmsnorm(h, w["final_layernorm"], c["eps"])
    return h @ w["embed"].T  # tied lm_head


def main():
    w = generate()
    logits = forward(w, TOKENS)
    out = Path(__file__).with_name("reference_logits.txt")
    np.savetxt(out, logits, fmt="%.9e")
    print(f"wrote {out}: {logits.shape[0]} tokens x {logits.shape[1]} vocab")
    print(f"first row head: {logits[0, :4]}")


if __name__ == "__main__":
    main()
