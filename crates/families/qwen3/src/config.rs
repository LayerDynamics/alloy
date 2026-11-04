/// Architecture hyperparameters for a Qwen3 causal LM.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3Config {
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

impl Qwen3Config {
    /// The tiny synthetic model for `qwen3_forward_cpu`. Small enough that
    /// an independent numpy implementation (fixture/gen_reference.py)
    /// is the source of truth for its reference logits.
    pub fn tiny_fixture() -> Self {
        Qwen3Config {
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

    /// The qwen3-4b-4bit profile. Verify every field against the corpus
    /// artifact's config.json before relying on these for the real model.
    pub fn qwen3_4b() -> Self {
        Qwen3Config {
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