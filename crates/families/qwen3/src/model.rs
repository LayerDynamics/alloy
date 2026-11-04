use crate::config::Qwen3Config;
use crate::fixture::Qwen3Weights;
use std::fmt;

/// Failures surfaced by model construction and execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// A tensor shape does not match the configuration.
    ShapeMismatch {
        tensor: &'static str,
        expected: usize,
        got: usize,
    },
    /// The configuration is invalid or unsupported.
    InvalidConfig(&'static str),
    /// A dimension calculation overflowed.
    DimensionOverflow(&'static str),
    /// The input token sequence is empty.
    EmptyInput,
    /// The input exceeds the configured context limit.
    ContextTooLong { maximum: usize, got: usize },
    /// An input token is outside the vocabulary.
    InvalidToken {
        position: usize,
        token: u32,
        vocab_size: usize,
    },
    /// A weight tensor contains NaN or infinity.
    NonFiniteWeight { tensor: &'static str, index: usize },
    /// A computation produced NaN or infinity.
    NonFiniteActivation(&'static str),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                tensor,
                expected,
                got,
            } => {
                write!(f, "{tensor}: expected {expected} elements, got {got}")
            }
            Self::InvalidConfig(message) => {
                write!(f, "invalid configuration: {message}")
            }
            Self::DimensionOverflow(name) => {
                write!(f, "dimension overflow: {name}")
            }
            Self::EmptyInput => write!(f, "input token sequence is empty"),
            Self::ContextTooLong { maximum, got } => {
                write!(f, "context length {got} exceeds maximum {maximum}")
            }
            Self::InvalidToken {
                position,
                token,
                vocab_size,
            } => {
                write!(
                    f,
                    "token {token} at position {position} is outside \
                     vocabulary size {vocab_size}"
                )
            }
            Self::NonFiniteWeight { tensor, index } => {
                write!(f, "{tensor}: non-finite weight at index {index}")
            }
            Self::NonFiniteActivation(stage) => {
                write!(f, "non-finite activation during {stage}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

/// A Qwen3 model constructed from in-memory weights.
///
/// Execution currently uses host-side f64 arithmetic and returns f32
/// logits. This is a CPU reference path, not an MLX-accelerated path.
pub struct Qwen3Model {
    config: Qwen3Config,
    weights: Qwen3Weights,
}

impl Qwen3Model {
    /// Construct from in-memory weights and validate tensor shapes.
    ///
    /// Q/K normalization scales have shape `[head_dim]` and are shared
    /// across their corresponding attention heads.
    ///
    /// Only tied input/output embeddings are currently supported.
    pub fn load(config: Qwen3Config, weights: Qwen3Weights) -> Result<Self, ModelError> {
        let c = &config;

        if c.hidden_size == 0
            || c.num_layers == 0
            || c.q_heads == 0
            || c.kv_heads == 0
            || c.head_dim == 0
            || c.intermediate_size == 0
            || c.vocab_size == 0
            || c.max_context == 0
        {
            return Err(ModelError::InvalidConfig(
                "model dimensions and context limit must be positive",
            ));
        }

        if c.q_heads % c.kv_heads != 0 {
            return Err(ModelError::InvalidConfig(
                "query-head count must be divisible by KV-head count",
            ));
        }

        if c.head_dim % 2 != 0 {
            return Err(ModelError::InvalidConfig(
                "split-half RoPE requires an even head dimension",
            ));
        }

        if !c.rope_theta.is_finite() || c.rope_theta <= 0.0 {
            return Err(ModelError::InvalidConfig(
                "rope_theta must be finite and positive",
            ));
        }

        if !c.tie_embeddings {
            return Err(ModelError::InvalidConfig(
                "untied output embeddings are not implemented",
            ));
        }

        let q_dim = checked_product(c.q_heads, c.head_dim, "q_dim")?;
        let kv_dim = checked_product(c.kv_heads, c.head_dim, "kv_dim")?;

        check_tensor(
            "embed_tokens",
            checked_product(c.vocab_size, c.hidden_size, "embed_tokens")?,
            &weights.embed_tokens,
        )?;

        if weights.layers.len() != c.num_layers {
            return Err(ModelError::ShapeMismatch {
                tensor: "layers",
                expected: c.num_layers,
                got: weights.layers.len(),
            });
        }

        for layer in &weights.layers {
            check_tensor("input_layernorm", c.hidden_size, &layer.input_layernorm)?;
            check_tensor(
                "q_proj",
                checked_product(q_dim, c.hidden_size, "q_proj")?,
                &layer.q_proj,
            )?;
            check_tensor(
                "k_proj",
                checked_product(kv_dim, c.hidden_size, "k_proj")?,
                &layer.k_proj,
            )?;
            check_tensor(
                "v_proj",
                checked_product(kv_dim, c.hidden_size, "v_proj")?,
                &layer.v_proj,
            )?;
            check_tensor("q_norm", c.head_dim, &layer.q_norm)?;
            check_tensor("k_norm", c.head_dim, &layer.k_norm)?;
            check_tensor(
                "o_proj",
                checked_product(c.hidden_size, q_dim, "o_proj")?,
                &layer.o_proj,
            )?;
            check_tensor(
                "post_attention_layernorm",
                c.hidden_size,
                &layer.post_attention_layernorm,
            )?;
            check_tensor(
                "gate_proj",
                checked_product(c.intermediate_size, c.hidden_size, "gate_proj")?,
                &layer.gate_proj,
            )?;
            check_tensor(
                "up_proj",
                checked_product(c.intermediate_size, c.hidden_size, "up_proj")?,
                &layer.up_proj,
            )?;
            check_tensor(
                "down_proj",
                checked_product(c.hidden_size, c.intermediate_size, "down_proj")?,
                &layer.down_proj,
            )?;
        }

        check_tensor("final_layernorm", c.hidden_size, &weights.final_layernorm)?;

        Ok(Self { config, weights })
    }

    /// Compute prefill logits for every input position.
    ///
    /// Returns `[token_count][vocab_size]`.
    ///
    /// Implements input RMSNorm, Q/K/V projections, per-head Q/K
    /// RMSNorm, split-half RoPE, grouped-query causal attention,
    /// attention and MLP residuals, final RMSNorm, and tied output
    /// projection.
    ///
    /// Positions start at zero. This method does not use a KV cache.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        let c = &self.config;

        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        if tokens.len() > c.max_context {
            return Err(ModelError::ContextTooLong {
                maximum: c.max_context,
                got: tokens.len(),
            });
        }

        let token_count = tokens.len();
        let q_dim = c.q_heads * c.head_dim;
        let kv_dim = c.kv_heads * c.head_dim;

        checked_product(token_count, c.hidden_size, "hidden activations")?;
        checked_product(token_count, q_dim, "query activations")?;
        checked_product(token_count, kv_dim, "key/value activations")?;
        checked_product(token_count, c.intermediate_size, "MLP activations")?;
        checked_product(token_count, c.vocab_size, "output logits")?;

        let mut hidden = Vec::with_capacity(token_count * c.hidden_size);

        for (position, &token) in tokens.iter().enumerate() {
            let id = token as usize;

            if id >= c.vocab_size {
                return Err(ModelError::InvalidToken {
                    position,
                    token,
                    vocab_size: c.vocab_size,
                });
            }

            let start = id * c.hidden_size;
            hidden.extend(
                self.weights.embed_tokens[start..start + c.hidden_size]
                    .iter()
                    .map(|&value| f64::from(value)),
            );
        }

        for layer in &self.weights.layers {
            let normalized = rmsnorm_rows(&hidden, c.hidden_size, &layer.input_layernorm);

            let mut query = linear(&normalized, c.hidden_size, &layer.q_proj, q_dim);
            let mut key = linear(&normalized, c.hidden_size, &layer.k_proj, kv_dim);
            let value = linear(&normalized, c.hidden_size, &layer.v_proj, kv_dim);

            query = rmsnorm_rows(&query, c.head_dim, &layer.q_norm);
            key = rmsnorm_rows(&key, c.head_dim, &layer.k_norm);

            apply_rope(
                &mut query,
                token_count,
                c.q_heads,
                c.head_dim,
                f64::from(c.rope_theta),
            );
            apply_rope(
                &mut key,
                token_count,
                c.kv_heads,
                c.head_dim,
                f64::from(c.rope_theta),
            );

            ensure_finite(&query, "query normalization and RoPE")?;
            ensure_finite(&key, "key normalization and RoPE")?;
            ensure_finite(&value, "value projection")?;

            let attended = causal_attention(
                &query,
                &key,
                &value,
                token_count,
                c.q_heads,
                c.kv_heads,
                c.head_dim,
            )?;

            let attention_output = linear(&attended, q_dim, &layer.o_proj, c.hidden_size);

            add_residual(&mut hidden, &attention_output);
            ensure_finite(&hidden, "attention residual")?;

            let normalized = rmsnorm_rows(&hidden, c.hidden_size, &layer.post_attention_layernorm);

            let mut gate = linear(
                &normalized,
                c.hidden_size,
                &layer.gate_proj,
                c.intermediate_size,
            );
            let up = linear(
                &normalized,
                c.hidden_size,
                &layer.up_proj,
                c.intermediate_size,
            );

            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value = silu(*gate_value) * up_value;
            }

            let mlp_output = linear(&gate, c.intermediate_size, &layer.down_proj, c.hidden_size);

            add_residual(&mut hidden, &mlp_output);
            ensure_finite(&hidden, "MLP residual")?;
        }

        let normalized = rmsnorm_rows(&hidden, c.hidden_size, &self.weights.final_layernorm);

        let logits = linear(
            &normalized,
            c.hidden_size,
            &self.weights.embed_tokens,
            c.vocab_size,
        );

        ensure_finite(&logits, "output projection")?;

        logits
            .chunks_exact(c.vocab_size)
            .map(|row| {
                row.iter()
                    .map(|&value| {
                        let value = value as f32;
                        if value.is_finite() {
                            Ok(value)
                        } else {
                            Err(ModelError::NonFiniteActivation("f32 logit conversion"))
                        }
                    })
                    .collect::<Result<Vec<f32>, ModelError>>()
            })
            .collect()
    }
}

fn checked_product(left: usize, right: usize, name: &'static str) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::DimensionOverflow(name))
}

fn check_tensor(tensor: &'static str, expected: usize, values: &[f32]) -> Result<(), ModelError> {
    if values.len() != expected {
        return Err(ModelError::ShapeMismatch {
            tensor,
            expected,
            got: values.len(),
        });
    }

    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteWeight { tensor, index });
    }

    Ok(())
}

fn ensure_finite(values: &[f64], stage: &'static str) -> Result<(), ModelError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ModelError::NonFiniteActivation(stage))
    }
}

/// Apply RMSNorm to contiguous rows, sharing the scale across rows.
fn rmsnorm_rows(input: &[f64], width: usize, weight: &[f32]) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());

    for row in input.chunks_exact(width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f64>() / width as f64;
        let denominator = (mean_square + f64::from(Qwen3Config::RMS_EPS)).sqrt();

        output.extend(
            row.iter()
                .zip(weight)
                .map(|(&value, &scale)| value * f64::from(scale) / denominator),
        );
    }

    output
}

/// Multiply input `[rows, input_width]` by weight `[output_width, input_width]`.
fn linear(input: &[f64], input_width: usize, weight: &[f32], output_width: usize) -> Vec<f64> {
    let rows = input.len() / input_width;
    let mut output = vec![0.0; rows * output_width];

    for (row_index, row) in input.chunks_exact(input_width).enumerate() {
        for (output_index, weight_row) in weight.chunks_exact(input_width).enumerate() {
            output[row_index * output_width + output_index] = row
                .iter()
                .zip(weight_row)
                .map(|(&value, &coefficient)| value * f64::from(coefficient))
                .sum();
        }
    }

    output
}

/// Rotate the two halves of each head using its absolute token position.
fn apply_rope(values: &mut [f64], token_count: usize, heads: usize, head_dim: usize, theta: f64) {
    let half = head_dim / 2;

    for position in 0..token_count {
        for head in 0..heads {
            let base = (position * heads + head) * head_dim;

            for dimension in 0..half {
                let inverse_frequency = theta.powf(-(2.0 * dimension as f64) / head_dim as f64);
                let angle = position as f64 * inverse_frequency;
                let (sin, cos) = angle.sin_cos();

                let low = values[base + dimension];
                let high = values[base + half + dimension];

                values[base + dimension] = low * cos - high * sin;
                values[base + half + dimension] = low * sin + high * cos;
            }
        }
    }
}

/// Causal grouped-query attention with a stable softmax.
///
/// Each query head uses KV head `query_head / group_size`.
fn causal_attention(
    query: &[f64],
    key: &[f64],
    value: &[f64],
    token_count: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f64>, ModelError> {
    let group_size = query_heads / kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0; query.len()];
    let mut scores = Vec::with_capacity(token_count);

    for position in 0..token_count {
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_base = (position * query_heads + query_head) * head_dim;

            scores.clear();

            for key_position in 0..=position {
                let key_base = (key_position * kv_heads + kv_head) * head_dim;

                let score = (0..head_dim)
                    .map(|dimension| query[query_base + dimension] * key[key_base + dimension])
                    .sum::<f64>()
                    * scale;

                scores.push(score);
            }

            ensure_finite(&scores, "attention scores")?;

            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            for score in &mut scores {
                *score = (*score - maximum).exp();
            }

            let denominator: f64 = scores.iter().sum();

            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(ModelError::NonFiniteActivation("attention softmax"));
            }

            for (key_position, &score) in scores.iter().enumerate() {
                let probability = score / denominator;
                let value_base = (key_position * kv_heads + kv_head) * head_dim;

                for dimension in 0..head_dim {
                    output[query_base + dimension] += probability * value[value_base + dimension];
                }
            }
        }
    }

    ensure_finite(&output, "attention output")?;
    Ok(output)
}

fn add_residual(hidden: &mut [f64], residual: &[f64]) {
    for (value, &update) in hidden.iter_mut().zip(residual) {
        *value += update;
    }
}

fn silu(value: f64) -> f64 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        value * exponential / (1.0 + exponential)
    }
}
