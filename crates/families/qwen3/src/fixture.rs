use crate::config::Qwen3Config;

/// Seed for the tiny synthetic fixture. Bit-identical to `SEED` in
/// fixture/gen_reference.py — changing either without the other
/// invalidates the committed reference logits.
pub const FIXTURE_SEED: u64 = 0x5145_4E33_4649_5853;

/// Two-dimensional weight matrix: float32 or resident affine 4-bit.
///
/// Affine4 keeps MLX-packed uint32 codes plus per-group scales/biases.
/// Production MLX execution must not expand these to a host f32 copy.
#[derive(Clone, Debug)]
pub enum MatrixWeight {
    /// Dense float32 row-major `[rows, cols]`.
    F32(Vec<f32>),
    /// Resident affine 4-bit: packed `[rows, cols/8]` uint32 + scales/biases.
    Affine4 {
        /// Packed nibbles (`bits=4`) in uint32 words, row-major.
        packed: Vec<u32>,
        /// Per-group scales, length `rows * groups`.
        scales: Vec<f32>,
        /// Per-group biases, length `rows * groups`.
        biases: Vec<f32>,
        /// Logical output rows.
        rows: usize,
        /// Logical input columns (must be divisible by 8 and `group_size`).
        cols: usize,
        /// Elements sharing one scale/bias.
        group_size: usize,
    },
}

impl MatrixWeight {
    /// Float32 constructor.
    pub fn f32(values: Vec<f32>) -> Self {
        Self::F32(values)
    }

    /// Logical float element count (`rows * cols`).
    pub fn logical_elements(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Affine4 { rows, cols, .. } => rows.saturating_mul(*cols),
        }
    }

    /// Resident payload bytes (packed codes + scales/biases, or f32).
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.len().saturating_mul(std::mem::size_of::<f32>()),
            Self::Affine4 {
                packed,
                scales,
                biases,
                ..
            } => packed
                .len()
                .saturating_mul(std::mem::size_of::<u32>())
                .saturating_add(scales.len().saturating_mul(std::mem::size_of::<f32>()))
                .saturating_add(biases.len().saturating_mul(std::mem::size_of::<f32>())),
        }
    }

    /// Borrow dense f32 storage when present.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values.as_slice()),
            Self::Affine4 { .. } => None,
        }
    }

    /// Require dense f32 (CPU reference path / fixtures).
    pub fn require_f32(&self, tensor: &'static str) -> Result<&[f32], crate::model::ModelError> {
        self.as_f32().ok_or_else(|| {
            crate::model::ModelError::CheckpointValidation(format!(
                "{tensor}: CPU reference path requires float32 weights; Affine4 is MLX-only"
            ))
        })
    }

    /// Mutable dense f32 storage when present (test helpers).
    pub fn as_f32_mut(&mut self) -> Option<&mut Vec<f32>> {
        match self {
            Self::F32(values) => Some(values),
            Self::Affine4 { .. } => None,
        }
    }

    /// Whether this matrix is resident Affine4.
    pub fn is_affine4(&self) -> bool {
        matches!(self, Self::Affine4 { .. })
    }

    /// Affine4 group size when quantized.
    pub fn affine4_group_size(&self) -> Option<usize> {
        match self {
            Self::Affine4 { group_size, .. } => Some(*group_size),
            Self::F32(_) => None,
        }
    }
}

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
    pub q_proj: MatrixWeight,
    pub k_proj: MatrixWeight,
    pub v_proj: MatrixWeight,
    /// Learned RMSNorm scale over each query head's last dimension.
    /// Shape: [head_dim], shared across query heads.
    pub q_norm: Vec<f32>,
    /// Learned RMSNorm scale over each key head's last dimension.
    /// Shape: [head_dim], shared across key heads.
    pub k_norm: Vec<f32>,
    pub o_proj: MatrixWeight,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj: MatrixWeight,
    pub up_proj: MatrixWeight,
    pub down_proj: MatrixWeight,
}

pub struct Qwen3Weights {
    /// `[vocab, hidden]`
    pub embed_tokens: MatrixWeight,
    pub layers: Vec<LayerWeights>,
    /// `[hidden]`
    pub final_layernorm: Vec<f32>,
    /// Separate [vocab, hidden] output weights for untied configurations.
    pub lm_head: Option<MatrixWeight>,
}

/// Deterministic fixture weights. Canonical generation order — MUST match
/// gen_reference.py: `embed_tokens`, then per layer (`input_layernorm`,
/// `q_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`, `o_proj`, `post_attention_layernorm`,
/// `gate_proj`, `up_proj`, `down_proj`), then `final_layernorm`.
pub fn generate_fixture(config: &Qwen3Config) -> Qwen3Weights {
    let c = config;
    let mut rng = SplitMix64(FIXTURE_SEED);

    let embed_tokens = MatrixWeight::f32(fill(&mut rng, c.vocab_size * c.hidden_size));
    let mut layers = Vec::with_capacity(c.num_layers);
    for _ in 0..c.num_layers {
        layers.push(LayerWeights {
            input_layernorm: fill(&mut rng, c.hidden_size),
            q_proj: MatrixWeight::f32(fill(&mut rng, c.q_dim() * c.hidden_size)),
            k_proj: MatrixWeight::f32(fill(&mut rng, c.kv_dim() * c.hidden_size)),
            v_proj: MatrixWeight::f32(fill(&mut rng, c.kv_dim() * c.hidden_size)),
            q_norm: fill(&mut rng, c.head_dim),
            k_norm: fill(&mut rng, c.head_dim),
            o_proj: MatrixWeight::f32(fill(&mut rng, c.hidden_size * c.q_dim())),
            post_attention_layernorm: fill(&mut rng, c.hidden_size),
            gate_proj: MatrixWeight::f32(fill(&mut rng, c.intermediate_size * c.hidden_size)),
            up_proj: MatrixWeight::f32(fill(&mut rng, c.intermediate_size * c.hidden_size)),
            down_proj: MatrixWeight::f32(fill(&mut rng, c.hidden_size * c.intermediate_size)),
        });
    }
    let final_layernorm = fill(&mut rng, c.hidden_size);
    // Append only for untied fixtures: tied reference RNG ordering is unchanged.
    let lm_head = if c.tie_embeddings {
        None
    } else {
        Some(MatrixWeight::f32(fill(
            &mut rng,
            c.vocab_size * c.hidden_size,
        )))
    };

    Qwen3Weights {
        embed_tokens,
        layers,
        final_layernorm,
        lm_head,
    }
}
