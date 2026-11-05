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

    /// Cooperative cancellation was observed during loading or execution.
    ///
    /// This reports an observed cancellation boundary. It does not imply
    /// that an in-flight native operation was interrupted immediately.
    Cancelled,

    /// The input exceeds the configured context limit.
    ContextTooLong {
        maximum: usize,
        got: usize,
    },

    /// An input token is outside the vocabulary.
    InvalidToken {
        position: usize,
        token: u32,
        vocab_size: usize,
    },

    /// A weight tensor contains NaN or infinity.
    NonFiniteWeight {
        tensor: &'static str,
        index: usize,
    },

    /// A computation produced NaN or infinity.
    NonFiniteActivation(&'static str),

    /// A configured resource budget would be exceeded.
    MemoryLimit {
        required: usize,
        limit: usize,
    },

    /// A fallible allocation failed.
    AllocationFailed,

    /// A checkpoint could not be read or validated.
    Checkpoint(String),

    /// I/O failure, classified without parsing an error message.
    CheckpointIo {
        operation: &'static str,
        path: std::path::PathBuf,
        kind: std::io::ErrorKind,
        message: String,
    },

    /// Malformed JSON or safetensors representation.
    CheckpointFormat {
        format: &'static str,
        message: String,
    },

    /// Well-formed checkpoint data violates the supported model contract.
    CheckpointValidation(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                tensor,
                expected,
                got,
            } => {
                write!(
                    f,
                    "{tensor}: expected {expected} elements, got {got}"
                )
            }

            Self::InvalidConfig(message) => {
                write!(f, "invalid configuration: {message}")
            }

            Self::DimensionOverflow(name) => {
                write!(f, "dimension overflow: {name}")
            }

            Self::EmptyInput => {
                write!(f, "input token sequence is empty")
            }

            Self::Cancelled => {
                write!(f, "model loading or execution cancelled")
            }

            Self::ContextTooLong { maximum, got } => {
                write!(
                    f,
                    "context length {got} exceeds maximum {maximum}"
                )
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
                write!(
                    f,
                    "{tensor}: non-finite weight at index {index}"
                )
            }

            Self::NonFiniteActivation(stage) => {
                write!(f, "non-finite activation during {stage}")
            }

            Self::MemoryLimit { required, limit } => {
                write!(
                    f,
                    "memory budget exceeded: {required} bytes required, \
                     limit {limit}"
                )
            }

            Self::AllocationFailed => {
                write!(f, "allocation failed")
            }

            Self::Checkpoint(message) => {
                write!(f, "checkpoint error: {message}")
            }

            Self::CheckpointIo {
                operation,
                path,
                kind,
                message,
            } => {
                write!(
                    f,
                    "checkpoint I/O ({operation}, {}, {kind:?}): {message}",
                    path.display()
                )
            }

            Self::CheckpointFormat { format, message } => {
                write!(
                    f,
                    "checkpoint {format} parse error: {message}"
                )
            }

            Self::CheckpointValidation(message) => {
                write!(f, "checkpoint validation error: {message}")
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
    cache_owner: std::sync::Arc<()>,
}

impl Qwen3Model {
    /// Construct from in-memory weights and validate tensor shapes.
    ///
    /// Q/K normalization scales have shape `[head_dim]` and are shared
    /// across their corresponding attention heads.
    ///
    /// Paths load bounded safetensors; in-memory weights remain accepted.
    pub fn load<S: crate::loader::ModelWeightSource>(
        config: Qwen3Config,
        source: S,
    ) -> Result<Self, ModelError> {
        let weights = source.into_weights(&config)?;
        Self::from_weights(config, weights)
    }

    /// Load safetensors with explicitly chosen raw-file and decoded-weight
    /// limits.
    ///
    /// This entry point uses the ordinary loader. Engine paths requiring
    /// trusted checkpoint digests must use the verified loader path.
    pub fn load_safetensors(
        config: Qwen3Config,
        path: &std::path::Path,
        options: crate::loader::LoadOptions,
    ) -> Result<Self, ModelError> {
        let weights =
            crate::loader::load_weights(&config, path, options)?;

        Self::from_weights(config, weights)
    }

    /// Validate in-memory weights.
    ///
    /// Supports tied and separate output projections.
    pub fn from_weights(
        config: Qwen3Config,
        weights: Qwen3Weights,
    ) -> Result<Self, ModelError> {
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

        if c.tie_embeddings && weights.lm_head.is_some() {
            return Err(ModelError::InvalidConfig(
                "tied embeddings must not supply lm_head",
            ));
        }

        if !c.tie_embeddings && weights.lm_head.is_none() {
            return Err(ModelError::InvalidConfig(
                "untied embeddings require lm_head",
            ));
        }

        if let Some(head) = &weights.lm_head {
            check_tensor(
                "lm_head",
                checked_product(
                    c.vocab_size,
                    c.hidden_size,
                    "lm_head",
                )?,
                head,
            )?;
        }

        let q_dim =
            checked_product(c.q_heads, c.head_dim, "q_dim")?;
        let kv_dim =
            checked_product(c.kv_heads, c.head_dim, "kv_dim")?;

        check_tensor(
            "embed_tokens",
            checked_product(
                c.vocab_size,
                c.hidden_size,
                "embed_tokens",
            )?,
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
            check_tensor(
                "input_layernorm",
                c.hidden_size,
                &layer.input_layernorm,
            )?;

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

            check_tensor(
                "q_norm",
                c.head_dim,
                &layer.q_norm,
            )?;

            check_tensor(
                "k_norm",
                c.head_dim,
                &layer.k_norm,
            )?;

            check_tensor(
                "o_proj",
                checked_product(
                    c.hidden_size,
                    q_dim,
                    "o_proj",
                )?,
                &layer.o_proj,
            )?;

            check_tensor(
                "post_attention_layernorm",
                c.hidden_size,
                &layer.post_attention_layernorm,
            )?;

            check_tensor(
                "gate_proj",
                checked_product(
                    c.intermediate_size,
                    c.hidden_size,
                    "gate_proj",
                )?,
                &layer.gate_proj,
            )?;

            check_tensor(
                "up_proj",
                checked_product(
                    c.intermediate_size,
                    c.hidden_size,
                    "up_proj",
                )?,
                &layer.up_proj,
            )?;

            check_tensor(
                "down_proj",
                checked_product(
                    c.hidden_size,
                    c.intermediate_size,
                    "down_proj",
                )?,
                &layer.down_proj,
            )?;
        }

        check_tensor(
            "final_layernorm",
            c.hidden_size,
            &weights.final_layernorm,
        )?;

        Ok(Self {
            config,
            weights,
            cache_owner: std::sync::Arc::new(()),
        })
    }

    /// Compute prefill logits for every input position.
    ///
    /// Returns `[token_count][vocab_size]`.
    ///
    /// Implements input RMSNorm, Q/K/V projections, per-head Q/K
    /// RMSNorm, split-half RoPE, grouped-query causal attention,
    /// attention and MLP residuals, final RMSNorm, and configured output
    /// projection.
    ///
    /// Positions start at zero. This method does not use a KV cache.
    pub fn forward(
        &self,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.forward_with_output_limit(
            tokens,
            DEFAULT_MAX_LOGITS_BYTES,
        )
    }

    /// Full-logits prefill with an explicit peak output-memory budget.
    ///
    /// The budget covers result storage, row headers, and one f64
    /// projection row. It is not a process-wide budget for weights
    /// and hidden activations.
    pub fn forward_with_output_limit(
        &self,
        tokens: &[u32],
        max_output_bytes: usize,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
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

        check_output_budget(
            token_count,
            c.vocab_size,
            max_output_bytes,
        )?;

        let q_dim = c.q_heads * c.head_dim;
        let kv_dim = c.kv_heads * c.head_dim;

        checked_product(
            token_count,
            c.hidden_size,
            "hidden activations",
        )?;

        checked_product(
            token_count,
            q_dim,
            "query activations",
        )?;

        checked_product(
            token_count,
            kv_dim,
            "key/value activations",
        )?;

        checked_product(
            token_count,
            c.intermediate_size,
            "MLP activations",
        )?;

        checked_product(
            token_count,
            c.vocab_size,
            "output logits",
        )?;

        let mut hidden =
            Vec::with_capacity(token_count * c.hidden_size);

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
                self.weights.embed_tokens
                    [start..start + c.hidden_size]
                    .iter()
                    .map(|&value| f64::from(value)),
            );
        }

        for layer in &self.weights.layers {
            let normalized = rmsnorm_rows(
                &hidden,
                c.hidden_size,
                &layer.input_layernorm,
            );

            let mut query = linear(
                &normalized,
                c.hidden_size,
                &layer.q_proj,
                q_dim,
            );

            let mut key = linear(
                &normalized,
                c.hidden_size,
                &layer.k_proj,
                kv_dim,
            );

            let value = linear(
                &normalized,
                c.hidden_size,
                &layer.v_proj,
                kv_dim,
            );

            query = rmsnorm_rows(
                &query,
                c.head_dim,
                &layer.q_norm,
            );

            key = rmsnorm_rows(
                &key,
                c.head_dim,
                &layer.k_norm,
            );

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

            ensure_finite(
                &query,
                "query normalization and RoPE",
            )?;

            ensure_finite(
                &key,
                "key normalization and RoPE",
            )?;

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

            let attention_output = linear(
                &attended,
                q_dim,
                &layer.o_proj,
                c.hidden_size,
            );

            add_residual(&mut hidden, &attention_output);
            ensure_finite(&hidden, "attention residual")?;

            let normalized = rmsnorm_rows(
                &hidden,
                c.hidden_size,
                &layer.post_attention_layernorm,
            );

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

            let mlp_output = linear(
                &gate,
                c.intermediate_size,
                &layer.down_proj,
                c.hidden_size,
            );

            add_residual(&mut hidden, &mlp_output);
            ensure_finite(&hidden, "MLP residual")?;
        }

        let normalized = rmsnorm_rows(
            &hidden,
            c.hidden_size,
            &self.weights.final_layernorm,
        );

        let head = if c.tie_embeddings {
            &self.weights.embed_tokens
        } else {
            self.weights
                .lm_head
                .as_ref()
                .ok_or(ModelError::InvalidConfig(
                    "missing lm_head",
                ))?
        };

        let mut result = Vec::new();

        result
            .try_reserve_exact(token_count)
            .map_err(|_| ModelError::AllocationFailed)?;

        // Project one row at a time: never retain the full f64
        // logits matrix.
        for row in normalized.chunks_exact(c.hidden_size) {
            let logits = linear(
                row,
                c.hidden_size,
                head,
                c.vocab_size,
            );

            ensure_finite(&logits, "output projection")?;

            let mut output = Vec::new();

            output
                .try_reserve_exact(c.vocab_size)
                .map_err(|_| ModelError::AllocationFailed)?;

            for value in logits {
                let value = value as f32;

                if !value.is_finite() {
                    return Err(ModelError::NonFiniteActivation(
                        "f32 logit conversion",
                    ));
                }

                output.push(value);
            }

            result.push(output);
        }

        Ok(result)
    }
}

fn checked_product(
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::DimensionOverflow(name))
}

fn check_tensor(
    tensor: &'static str,
    expected: usize,
    values: &[f32],
) -> Result<(), ModelError> {
    if values.len() != expected {
        return Err(ModelError::ShapeMismatch {
            tensor,
            expected,
            got: values.len(),
        });
    }

    if let Some(index) =
        values.iter().position(|value| !value.is_finite())
    {
        return Err(ModelError::NonFiniteWeight {
            tensor,
            index,
        });
    }

    Ok(())
}

fn ensure_finite(
    values: &[f64],
    stage: &'static str,
) -> Result<(), ModelError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ModelError::NonFiniteActivation(stage))
    }
}

/// Apply RMSNorm to contiguous rows, sharing the scale across rows.
fn rmsnorm_rows(
    input: &[f64],
    width: usize,
    weight: &[f32],
) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());

    for row in input.chunks_exact(width) {
        let mean_square = row
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / width as f64;

        let denominator =
            (mean_square + f64::from(Qwen3Config::RMS_EPS)).sqrt();

        output.extend(
            row.iter()
                .zip(weight)
                .map(|(&value, &scale)| {
                    value * f64::from(scale) / denominator
                }),
        );
    }

    output
}

/// Multiply input `[rows, input_width]` by weight
/// `[output_width, input_width]`.
fn linear(
    input: &[f64],
    input_width: usize,
    weight: &[f32],
    output_width: usize,
) -> Vec<f64> {
    let rows = input.len() / input_width;
    let mut output = vec![0.0; rows * output_width];

    for (row_index, row) in
        input.chunks_exact(input_width).enumerate()
    {
        for (output_index, weight_row) in
            weight.chunks_exact(input_width).enumerate()
        {
            output[row_index * output_width + output_index] = row
                .iter()
                .zip(weight_row)
                .map(|(&value, &coefficient)| {
                    value * f64::from(coefficient)
                })
                .sum();
        }
    }

    output
}

/// Rotate the two halves of each head using its absolute token position.
fn apply_rope(
    values: &mut [f64],
    token_count: usize,
    heads: usize,
    head_dim: usize,
    theta: f64,
) {
    let half = head_dim / 2;

    for position in 0..token_count {
        for head in 0..heads {
            let base = (position * heads + head) * head_dim;

            for dimension in 0..half {
                let inverse_frequency =
                    theta.powf(
                        -(2.0 * dimension as f64) / head_dim as f64,
                    );

                let angle = position as f64 * inverse_frequency;
                let (sin, cos) = angle.sin_cos();

                let low = values[base + dimension];
                let high = values[base + half + dimension];

                values[base + dimension] = low * cos - high * sin;
                values[base + half + dimension] =
                    low * sin + high * cos;
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

            let query_base =
                (position * query_heads + query_head) * head_dim;

            scores.clear();

            for key_position in 0..=position {
                let key_base =
                    (key_position * kv_heads + kv_head) * head_dim;

                let score = (0..head_dim)
                    .map(|dimension| {
                        query[query_base + dimension]
                            * key[key_base + dimension]
                    })
                    .sum::<f64>()
                    * scale;

                scores.push(score);
            }

            ensure_finite(&scores, "attention scores")?;

            let maximum = scores
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            for score in &mut scores {
                *score = (*score - maximum).exp();
            }

            let denominator: f64 = scores.iter().sum();

            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(ModelError::NonFiniteActivation(
                    "attention softmax",
                ));
            }

            for (key_position, &score) in scores.iter().enumerate() {
                let probability = score / denominator;

                let value_base =
                    (key_position * kv_heads + kv_head) * head_dim;

                for dimension in 0..head_dim {
                    output[query_base + dimension] +=
                        probability * value[value_base + dimension];
                }
            }
        }
    }

    ensure_finite(&output, "attention output")?;
    Ok(output)
}

fn add_residual(
    hidden: &mut [f64],
    residual: &[f64],
) {
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

/// Default full-logits output limit: 64 MiB, excluding model/hidden storage.
pub const DEFAULT_MAX_LOGITS_BYTES: usize = 64 * 1024 * 1024;

/// Result rows, row headers, and one temporary f64 projection row.
pub fn full_logits_bytes(
    tokens: usize,
    vocabulary: usize,
) -> Result<usize, ModelError> {
    let output = checked_product(
        checked_product(tokens, vocabulary, "logit elements")?,
        std::mem::size_of::<f32>(),
        "f32 output bytes",
    )?;

    let temporary = checked_product(
        vocabulary,
        std::mem::size_of::<f64>(),
        "f64 row bytes",
    )?;

    let headers = checked_product(
        tokens,
        std::mem::size_of::<Vec<f32>>(),
        "row headers",
    )?;

    output
        .checked_add(temporary)
        .and_then(|value| value.checked_add(headers))
        .ok_or(ModelError::DimensionOverflow("logit memory"))
}

fn check_output_budget(
    tokens: usize,
    vocabulary: usize,
    limit: usize,
) -> Result<(), ModelError> {
    let required = full_logits_bytes(tokens, vocabulary)?;

    if required > limit {
        Err(ModelError::MemoryLimit {
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

include!("decode_impl.rs");

#[cfg(test)]
mod model_error_tests {
    use super::*;

    #[test]
    fn cancellation_has_a_typed_error_variant() {
        let error = ModelError::Cancelled;

        assert_eq!(error.clone(), ModelError::Cancelled);
        assert_ne!(
            error,
            ModelError::CheckpointValidation(
                "checkpoint loading cancelled".to_owned(),
            ),
        );
    }

    #[test]
    fn cancellation_has_explicit_display_text() {
        assert_eq!(
            ModelError::Cancelled.to_string(),
            "model loading or execution cancelled",
        );
    }
}