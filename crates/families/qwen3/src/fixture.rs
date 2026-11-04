use crate::config::Qwen3Config;

/// Seed for the tiny synthetic fixture. Bit-identical to `SEED` in
/// fixture/gen_reference.py — changing either without the other
/// invalidates the committed reference logits.
pub const FIXTURE_SEED: u64 = 0x5145_4E33_4649_5853;

/// SplitMix64 — matches gen_reference.py exactly.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// One uniform value in [-1, 1) from the low 32 bits of a draw. The
/// conversion (u32 -> f32 -> /u32::MAX -> *2 - 1) is reproduced
/// op-for-op in gen_reference.py so both sides see identical weights.
fn unit(rng: &mut SplitMix64) -> f32 {
    let u = (rng.next() & 0xFFFF_FFFF) as u32;
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn fill(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n).map(|_| unit(rng)).collect()
}

/// One transformer layer's weights. All matrices are row-major
/// `[out, in]`, matching HF Qwen3 layout for future safetensors loading.
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    /// Learned RMSNorm scale over each query head's last dimension.
    /// Shape: [head_dim], shared across query heads.
    pub q_norm: Vec<f32>,
    /// Learned RMSNorm scale over each key head's last dimension.
    /// Shape: [head_dim], shared across key heads.
    pub k_norm: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

pub struct Qwen3Weights {
    /// `[vocab, hidden]`
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    /// `[hidden]`
    pub final_layernorm: Vec<f32>,
    /// Separate [vocab, hidden] output weights for untied configurations.
    pub lm_head: Option<Vec<f32>>,
}

/// Deterministic fixture weights. Canonical generation order — MUST match
/// gen_reference.py: `embed_tokens`, then per layer (`input_layernorm`,
/// `q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`, `o_proj`, `post_attention_layernorm`,
/// `gate_proj`, `up_proj`, `down_proj`), then `final_layernorm`.
pub fn generate_fixture(config: &Qwen3Config) -> Qwen3Weights {
    let c = config;
    let mut rng = SplitMix64(FIXTURE_SEED);

    let embed_tokens = fill(&mut rng, c.vocab_size * c.hidden_size);
    let mut layers = Vec::with_capacity(c.num_layers);
    for _ in 0..c.num_layers {
        layers.push(LayerWeights {
            input_layernorm: fill(&mut rng, c.hidden_size),
            q_proj: fill(&mut rng, c.q_dim() * c.hidden_size),
            k_proj: fill(&mut rng, c.kv_dim() * c.hidden_size),
            v_proj: fill(&mut rng, c.kv_dim() * c.hidden_size),
            q_norm: fill(&mut rng, c.head_dim),
            k_norm: fill(&mut rng, c.head_dim),
            o_proj: fill(&mut rng, c.hidden_size * c.q_dim()),
            post_attention_layernorm: fill(&mut rng, c.hidden_size),
            gate_proj: fill(&mut rng, c.intermediate_size * c.hidden_size),
            up_proj: fill(&mut rng, c.intermediate_size * c.hidden_size),
            down_proj: fill(&mut rng, c.hidden_size * c.intermediate_size),
        });
    }
    let final_layernorm = fill(&mut rng, c.hidden_size);
    // Append only for untied fixtures: tied reference RNG ordering is unchanged.
    let lm_head = if c.tie_embeddings {
        None
    } else {
        Some(fill(&mut rng, c.vocab_size * c.hidden_size))
    };

    Qwen3Weights {
        embed_tokens,
        layers,
        final_layernorm,
        lm_head,
    }
}
