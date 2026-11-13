/// Architecture hyperparameters for a Qwen3 causal LM.
#[derive(Clone, Debug, PartialEq)]
pub struct DecoderConfig {
    pub family: DecoderFamily,
    pub rms_eps: f32,
    pub rope_scaling: Option<Llama31Scaling>,
    pub quantization_bits: u8,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub max_context: usize,
    pub tie_embeddings: bool,
}

impl DecoderConfig {
    /// The tiny synthetic model for `qwen3_forward_cpu`. Small enough that
    /// an independent numpy implementation (fixture/gen_reference.py)
    /// is the source of truth for its reference logits.
    pub fn tiny_fixture() -> Self {
        DecoderConfig {
            family: DecoderFamily::Qwen3,
            rms_eps: 1e-6,
            rope_scaling: None,
            quantization_bits: 4,
            hidden_size: 32,
            num_layers: 2,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            vocab_size: 256,
            rope_theta: 10_000.0,
            max_context: 512,
            tie_embeddings: true,
        }
    }

    /// Qwen3-4B architecture with a conservative 32K runtime context limit.
    /// The published source config declares tied embeddings. Check the exact
    /// artifact config before loading; untied variants use a separate lm_head.
    pub fn qwen3_4b() -> Self {
        DecoderConfig {
            family: DecoderFamily::Qwen3,
            rms_eps: 1e-6,
            rope_scaling: None,
            quantization_bits: 4,
            hidden_size: 2560,
            num_layers: 36,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            vocab_size: 151_936,
            rope_theta: 1_000_000.0,
            max_context: 32_768,
            tie_embeddings: true,
        }
    }

    /// Total query projection width. Qwen3 decouples this from
    /// `hidden_size` (4B: 32 * 128 = 4096 vs hidden 2560).
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.head_dim
    }

    /// Total K/V projection width.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Attention softmax scale, 1/sqrt(head_dim).
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// RMSNorm epsilon used by Qwen3.
    pub const RMS_EPS: f32 = 1e-6;
}

/// Dense architectures with different attention semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderFamily {
    Qwen3,
    Llama3,
}

/// Llama 3.1 wavelength scaling, applied to inverse angular frequencies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Llama31Scaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}
impl Llama31Scaling {
    pub fn validate(&self) -> bool {
        self.factor.is_finite()
            && self.factor >= 1.0
            && self.low_freq_factor.is_finite()
            && self.low_freq_factor > 0.0
            && self.high_freq_factor.is_finite()
            && self.high_freq_factor > self.low_freq_factor
            && self.original_max_position_embeddings > 0
    }
    pub fn scale_inverse_frequency(&self, frequency: f64) -> f64 {
        let wavelength = std::f64::consts::TAU / frequency;
        let context = self.original_max_position_embeddings as f64;
        let low = f64::from(self.low_freq_factor);
        let high = f64::from(self.high_freq_factor);
        let factor = f64::from(self.factor);
        if wavelength < context / high {
            frequency
        } else if wavelength > context / low {
            frequency / factor
        } else {
            let smooth = (context / wavelength - low) / (high - low);
            (1.0 - smooth) * frequency / factor + smooth * frequency
        }
    }
}
impl DecoderConfig {
    pub fn has_qk_norm(&self) -> bool {
        self.family == DecoderFamily::Qwen3
    }
    pub fn norm_width(&self) -> usize {
        if self.has_qk_norm() { self.head_dim } else { 0 }
    }
    /// MLX custom RoPE takes divisors, the reciprocal of angular frequencies.
    pub fn rope_divisors(&self) -> Vec<f32> {
        (0..self.head_dim / 2)
            .map(|i| {
                let frequency =
                    f64::from(self.rope_theta).powf(-2.0 * i as f64 / self.head_dim as f64);
                let scaled = self
                    .rope_scaling
                    .map_or(frequency, |s| s.scale_inverse_frequency(frequency));
                (1.0 / scaled) as f32
            })
            .collect()
    }
}
