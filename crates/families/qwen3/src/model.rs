use crate::config::Qwen3Config;
use crate::fixture::Qwen3Weights;
use std::fmt;

/// Failures surfaced by model construction and execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// A tensor shape does not match the configuration.
    ShapeMismatch { tensor: &'static str, expected: usize, got: usize },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::ShapeMismatch { tensor, expected, got } => {
                write!(f, "{tensor}: expected {expected} elements, got {got}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

pub struct Qwen3Model {
    config: Qwen3Config,
    weights: Qwen3Weights,
}

impl Qwen3Model {
    /// Construct from in-memory weights, validating every tensor shape
    /// against the config. The safetensors loader feeds this later.
    pub fn load(config: Qwen3Config, weights: Qwen3Weights) -> Result<Self, ModelError> {
        let c = &config;
        let check = |tensor: &'static str, expected: usize, got: usize| -> Result<(), ModelError> {
            if expected == got {
                Ok(())
            } else {
                Err(ModelError::ShapeMismatch { tensor, expected, got })
            }
        };

        check("embed_tokens", c.vocab_size * c.hidden_size, weights.embed_tokens.len())?;
        check("layers", c.num_layers, weights.layers.len())?;
        for (i, l) in weights.layers.iter().enumerate() {
            let _ = i;
            check("input_layernorm", c.hidden_size, l.input_layernorm.len())?;
            check("q_proj", c.q_dim() * c.hidden_size, l.q_proj.len())?;
            check("k_proj", c.kv_dim() * c.hidden_size, l.k_proj.len())?;
            check("v_proj", c.kv_dim() * c.hidden_size, l.v_proj.len())?;
            check("o_proj", c.hidden_size * c.q_dim(), l.o_proj.len())?;
            check("post_attention_layernorm", c.hidden_size, l.post_attention_layernorm.len())?;
            check("gate_proj", c.intermediate_size * c.hidden_size, l.gate_proj.len())?;
            check("up_proj", c.intermediate_size * c.hidden_size, l.up_proj.len())?;
            check("down_proj", c.hidden_size * c.intermediate_size, l.down_proj.len())?;
        }
        check("final_layernorm", c.hidden_size, weights.final_layernorm.len())?;

        Ok(Qwen3Model { config, weights })
    }

    /// Prefill: logits for every position. Returns one row per token,
    /// each of length `vocab_size`. Semantics are defined by
    /// fixture/gen_reference.py (the independent reference): RMSNorm
    /// (eps 1e-6), QKV projections, split-half RoPE, GQA causal
    /// attention at 1/sqrt(head_dim), SwiGLU MLP, tied lm_head.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        let _ = (&self.config, &self.weights, tokens);
        todo!("forward pass (issue #3): implement to match gen_reference.py within FR-8 tolerances")
    }
}