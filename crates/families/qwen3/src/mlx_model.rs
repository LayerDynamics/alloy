//! Unified-memory MLX Qwen3 execution.
//!
//! This module owns one set of dense or packed Affine4 MLX weights and executes the full
//! transformer through explicit-stream MLX operations.
//!
//! Supported:
//!
//! - Batch size one.
//! - Tied or untied output projection.
//! - Qwen3 per-head Q/K RMSNorm.
//! - Split-half RoPE with absolute cached positions.
//! - Full-cache grouped-query attention.
//! - Complete forward logits.
//! - Cached prefill returning final-position logits.
//! - Single-token cached decode.
//! - Transactional all-layer KV publication.
//!
//! Weights are not copied into separate CPU and Metal model instances.
//! The caller selects a stream for each operation.
//!
//! The existing host-reference Qwen3Model remains a numerical oracle.
//! It is not invoked by production execution in this module.
//!
//! Limits here cover specified tensor payloads, not process RSS, MLX
//! allocator caches, attention workspace, or complete loading overlap.
//! Engine admission and allocator integration remain adapter/runtime work.
//!
//! Native arrays remain worker-local. No unsafe Send/Sync impls are added.

use crate::{
    config::Qwen3Config,
    fixture::{LayerWeights, MatrixWeight, Qwen3Weights},
    model::ModelError,
};
use inference_infra::kv_cache::{KvCacheError, KvCacheSpec, MlxKvCache, MlxModelId};
use mlx_ffi::{
    Array, Error as MlxError, Stream,
    attention::{self, AttentionMask},
    ops::{self, OperationError},
};
use std::{cell::Cell, fmt};

/// Errors from the actual MLX model path.
#[derive(Debug)]
pub enum MlxQwenError {
    /// Existing model configuration, shape, or token validation failure.
    Model(ModelError),

    /// An MLX tensor operation failed.
    Operation(OperationError),

    /// Cache ownership, shape, capacity, or commit failed.
    Cache(KvCacheError),

    /// A native MLX completion operation failed.
    Mlx(MlxError),

    /// Native backing-buffer measurement failed.
    Memory(mlx_ffi::memory::MemoryError),

    /// Cooperative cancellation was observed before cache publication.
    Cancelled,

    /// A previous native failure invalidated this loaded model.
    Invalidated,

    /// A model payload or output-payload limit was exceeded.
    PayloadLimit {
        /// Allocation or output being bounded.
        operation: &'static str,
        /// Required payload allowance.
        required_bytes: u64,
        /// Caller-supplied allowance.
        limit_bytes: u64,
    },

    /// A Rust metadata allocation failed.
    AllocationFailed(&'static str),
}

impl fmt::Display for MlxQwenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(formatter, "{error}"),
            Self::Operation(error) => write!(formatter, "{error}"),
            Self::Cache(error) => write!(formatter, "{error}"),
            Self::Mlx(error) => write!(formatter, "{error}"),
            Self::Memory(error) => write!(formatter, "{error}"),
            Self::Cancelled => write!(formatter, "MLX Qwen execution cancelled"),
            Self::Invalidated => write!(
                formatter,
                "MLX Qwen model invalidated by a previous native failure"
            ),
            Self::PayloadLimit {
                operation,
                required_bytes,
                limit_bytes,
            } => write!(
                formatter,
                "{operation} requires {required_bytes} bytes; \
                 supplied allowance is {limit_bytes}"
            ),
            Self::AllocationFailed(operation) => {
                write!(formatter, "allocation failed during {operation}")
            }
        }
    }
}

impl std::error::Error for MlxQwenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Cache(error) => Some(error),
            Self::Mlx(error) => Some(error),
            Self::Memory(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for MlxQwenError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<OperationError> for MlxQwenError {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

impl From<KvCacheError> for MlxQwenError {
    fn from(error: KvCacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<MlxError> for MlxQwenError {
    fn from(error: MlxError) -> Self {
        Self::Mlx(error)
    }
}

/// Result of MLX Qwen model execution.
pub type MlxQwenResult<T> = Result<T, MlxQwenError>;

/// Evaluated, finite float32 model logits.
///
/// Host storage is intentionally limited to the selected final output.
/// Intermediate transformer tensors remain in MLX.
#[derive(Debug)]
pub struct MlxLogits {
    values: Vec<f32>,
    rows: usize,
    vocabulary: usize,
}

impl MlxLogits {
    /// Number of returned positions.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of vocabulary entries per position.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }

    /// Borrow the complete row-major result.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Borrow one position's vocabulary logits.
    pub fn row(&self, position: usize) -> Option<&[f32]> {
        if position >= self.rows {
            return None;
        }
        let start = position * self.vocabulary;
        Some(&self.values[start..start + self.vocabulary])
    }

    /// Borrow the final returned position.
    pub fn last_row(&self) -> &[f32] {
        let start = (self.rows - 1) * self.vocabulary;
        &self.values[start..]
    }

    /// Transfer the evaluated row-major output buffer.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Dense float32 or resident Affine4 projection.
enum MlxLinear {
    Dense(Array),
    Affine4 {
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
    },
}

impl MlxLinear {
    fn upload(matrix: MatrixWeight, rows: usize, cols: usize) -> MlxQwenResult<Self> {
        match matrix {
            MatrixWeight::F32(values) => Ok(Self::Dense(upload_f32(values, &[rows, cols])?)),
            MatrixWeight::Affine4 {
                packed,
                scales,
                biases,
                rows: packed_rows,
                cols: packed_cols,
                group_size,
            } => {
                if packed_rows != rows || packed_cols != cols {
                    return Err(ModelError::ShapeMismatch {
                        tensor: "affine4 matrix",
                        expected: rows.saturating_mul(cols),
                        got: packed_rows.saturating_mul(packed_cols),
                    }
                    .into());
                }
                if !cols.is_multiple_of(8) || group_size == 0 || !cols.is_multiple_of(group_size) {
                    return Err(ModelError::InvalidConfig(
                        "affine4 cols must be divisible by 8 and group_size",
                    )
                    .into());
                }
                let groups = cols / group_size;
                let weight = upload_u32(packed, &[rows, cols / 8])?;
                let scales = upload_f32(scales, &[rows, groups])?;
                let biases = upload_f32(biases, &[rows, groups])?;
                let group_size = i32::try_from(group_size)
                    .map_err(|_| ModelError::InvalidConfig("affine4 group_size exceeds C int"))?;
                Ok(Self::Affine4 {
                    weight,
                    scales,
                    biases,
                    group_size,
                })
            }
        }
    }

    fn embedding_lookup(&self, tokens: &[u32], stream: &Stream) -> MlxQwenResult<Array> {
        match self {
            Self::Dense(weight) => Ok(ops::embedding_lookup(weight, tokens, stream)?),
            Self::Affine4 {
                weight,
                scales,
                biases,
                group_size,
            } => Ok(ops::quantized_embedding_lookup(
                weight,
                scales,
                biases,
                tokens,
                *group_size,
                stream,
            )?),
        }
    }

    fn apply(&self, input: &Array, stream: &Stream) -> MlxQwenResult<Array> {
        match self {
            Self::Dense(weight) => Ok(ops::linear(input, weight, stream)?),
            Self::Affine4 {
                weight,
                scales,
                biases,
                group_size,
            } => Ok(ops::quantized_matmul(
                input,
                weight,
                scales,
                biases,
                true,
                *group_size,
                4,
                stream,
            )?),
        }
    }

    fn visit<'a>(&'a self, visitor: &mut impl FnMut(&'a Array)) {
        match self {
            Self::Dense(weight) => visitor(weight),
            Self::Affine4 {
                weight,
                scales,
                biases,
                ..
            } => {
                visitor(weight);
                visitor(scales);
                visitor(biases);
            }
        }
    }
}

struct MlxLayer {
    input_layernorm: Array,
    q_proj: MlxLinear,
    k_proj: MlxLinear,
    v_proj: MlxLinear,
    q_norm: Array,
    k_norm: Array,
    o_proj: MlxLinear,
    post_attention_layernorm: Array,
    gate_proj: MlxLinear,
    up_proj: MlxLinear,
    down_proj: MlxLinear,
}

impl MlxLayer {
    fn upload(config: &Qwen3Config, weights: LayerWeights) -> MlxQwenResult<Self> {
        Ok(Self {
            input_layernorm: upload_f32(weights.input_layernorm, &[config.hidden_size])?,
            q_proj: MlxLinear::upload(weights.q_proj, config.q_dim(), config.hidden_size)?,
            k_proj: MlxLinear::upload(weights.k_proj, config.kv_dim(), config.hidden_size)?,
            v_proj: MlxLinear::upload(weights.v_proj, config.kv_dim(), config.hidden_size)?,
            q_norm: upload_f32(weights.q_norm, &[config.head_dim])?,
            k_norm: upload_f32(weights.k_norm, &[config.head_dim])?,
            o_proj: MlxLinear::upload(weights.o_proj, config.hidden_size, config.q_dim())?,
            post_attention_layernorm: upload_f32(
                weights.post_attention_layernorm,
                &[config.hidden_size],
            )?,
            gate_proj: MlxLinear::upload(
                weights.gate_proj,
                config.intermediate_size,
                config.hidden_size,
            )?,
            up_proj: MlxLinear::upload(
                weights.up_proj,
                config.intermediate_size,
                config.hidden_size,
            )?,
            down_proj: MlxLinear::upload(
                weights.down_proj,
                config.hidden_size,
                config.intermediate_size,
            )?,
        })
    }

    fn visit<'a>(&'a self, visitor: &mut impl FnMut(&'a Array)) {
        visitor(&self.input_layernorm);
        self.q_proj.visit(visitor);
        self.k_proj.visit(visitor);
        self.v_proj.visit(visitor);
        visitor(&self.q_norm);
        visitor(&self.k_norm);
        self.o_proj.visit(visitor);
        visitor(&self.post_attention_layernorm);
        self.gate_proj.visit(visitor);
        self.up_proj.visit(visitor);
        self.down_proj.visit(visitor);
    }
}

struct MlxWeights {
    embedding: MlxLinear,
    layers: Vec<MlxLayer>,
    final_layernorm: Array,
    lm_head: Option<MlxLinear>,
}

impl MlxWeights {
    fn visit<'a>(&'a self, mut visitor: impl FnMut(&'a Array)) {
        self.embedding.visit(&mut visitor);
        for layer in &self.layers {
            layer.visit(&mut visitor);
        }
        visitor(&self.final_layernorm);
        if let Some(head) = &self.lm_head {
            head.visit(&mut visitor);
        }
    }

    fn output_head(&self) -> &MlxLinear {
        self.lm_head.as_ref().unwrap_or(&self.embedding)
    }
}

/// One loaded MLX Qwen model with shared CPU/Metal-accessible weights.
///
/// The caller must construct and use it on its native worker. Explicit
/// streams select execution placement without reloading the weights.
pub struct MlxQwen3Model {
    config: Qwen3Config,
    weights: MlxWeights,
    owner: MlxModelId,
    weight_payload_bytes: u64,
    healthy: Cell<bool>,
}

#[derive(Clone, Copy)]
enum OutputSelection {
    All,
    Last,
}

impl MlxQwen3Model {
    /// Validate and consume decoded dense/Affine4 weights into MLX arrays.
    ///
    /// max_weight_payload_bytes must cover logical payload AND measured
    /// native backing buffers. Both are checked.
    /// The caller must separately admit source/decoded/native overlap
    /// during loading and MLX allocator overhead.
    ///
    /// Each source vector is released after its MLX array is constructed.
    /// A complete host-reference model is not retained.
    pub fn from_weights(
        config: Qwen3Config,
        weights: Qwen3Weights,
        max_weight_payload_bytes: u64,
        stream: &Stream,
    ) -> MlxQwenResult<Self> {
        let result = (|| {
            let payload = validate_weights(&config, &weights)?;
            if payload > max_weight_payload_bytes {
                return Err(MlxQwenError::PayloadLimit {
                    operation: "MLX resident weight payload",
                    required_bytes: payload,
                    limit_bytes: max_weight_payload_bytes,
                });
            }

            let Qwen3Weights {
                embed_tokens,
                layers,
                final_layernorm,
                lm_head,
            } = weights;

            let mut native_layers = Vec::new();
            native_layers
                .try_reserve_exact(config.num_layers)
                .map_err(|_| MlxQwenError::AllocationFailed("MLX layer metadata"))?;

            let embedding = MlxLinear::upload(embed_tokens, config.vocab_size, config.hidden_size)?;

            for layer in layers {
                native_layers.push(MlxLayer::upload(&config, layer)?);
            }

            let final_layernorm = upload_f32(final_layernorm, &[config.hidden_size])?;

            let lm_head = match lm_head {
                Some(head) => Some(MlxLinear::upload(
                    head,
                    config.vocab_size,
                    config.hidden_size,
                )?),
                None => None,
            };

            let native = MlxWeights {
                embedding,
                layers: native_layers,
                final_layernorm,
                lm_head,
            };

            let mut failure = None;
            native.visit(|array| {
                if let Err(error) = array.eval()
                    && failure.is_none()
                {
                    failure = Some(error);
                }
            });

            if let Some(error) = failure {
                return Err(MlxQwenError::Mlx(error));
            }

            let model = Self {
                config,
                weights: native,
                owner: MlxModelId::new(),
                weight_payload_bytes: payload,
                healthy: Cell::new(true),
            };
            let measured = model.weight_buffer_bytes()?;
            if measured > max_weight_payload_bytes {
                return Err(MlxQwenError::PayloadLimit {
                    operation: "MLX resident weight backing buffers",
                    required_bytes: measured,
                    limit_bytes: max_weight_payload_bytes,
                });
            }
            Ok(model)
        })();

        complete_stream(result, stream)
    }

    /// Exact architecture configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Identity to which this model's MLX caches are bound.
    pub fn instance_id(&self) -> &MlxModelId {
        &self.owner
    }

    /// Resident dense/Affine4 tensor payload, excluding allocator overhead.
    ///
    /// Tied embeddings are counted once.
    pub fn weight_payload_bytes(&self) -> u64 {
        self.weight_payload_bytes
    }

    /// Whether no native execution/completion failure invalidated this load.
    pub fn is_healthy(&self) -> bool {
        self.healthy.get()
    }

    /// Measure unique evaluated native backing buffers of all owned weights.
    ///
    /// Excludes allocator cache, activations, and other loaded models.
    /// Tied output storage is counted once. No implicit evaluation or
    /// silent fallback to logical payload estimates occurs.
    pub fn weight_buffer_bytes(&self) -> MlxQwenResult<u64> {
        self.check_healthy()?;
        let count = self
            .weights
            .layers
            .len()
            .checked_mul(25) // Affine4 projections: up to 3 arrays each + norms
            .and_then(|n| n.checked_add(4)) // three embedding arrays + final norm
            .and_then(|n| n.checked_add(3 * usize::from(self.weights.lm_head.is_some())))
            .ok_or(ModelError::DimensionOverflow("weight query handle count"))?;
        let mut arrays = Vec::new();
        arrays
            .try_reserve_exact(count)
            .map_err(|_| MlxQwenError::AllocationFailed("weight buffer query metadata"))?;
        self.weights.visit(|array| arrays.push(array));
        let bytes = mlx_ffi::memory::array_buffer_bytes(&arrays)?;
        Ok(u64::try_from(bytes)
            .map_err(|_| ModelError::DimensionOverflow("measured weight bytes"))?)
    }

    fn check_healthy(&self) -> MlxQwenResult<()> {
        if !self.healthy.get() {
            return Err(MlxQwenError::Invalidated);
        }
        Ok(())
    }

    /// Allocate a batch-one full cache belonging to this model.
    ///
    /// The supplied allowance must cover the cache owner's update-overlap
    /// payload plan, not just one committed cache version.
    pub fn new_cache(
        &self,
        capacity_tokens: usize,
        max_cache_payload_bytes: u64,
        stream: &Stream,
    ) -> MlxQwenResult<MlxKvCache> {
        self.check_healthy()?;

        let result = (|| {
            if capacity_tokens == 0 || capacity_tokens > self.config.max_context {
                return Err(ModelError::InvalidConfig(
                    "MLX cache capacity is outside model context bounds",
                )
                .into());
            }

            Ok(MlxKvCache::new(
                &self.owner,
                KvCacheSpec {
                    layers: self.config.num_layers,
                    batch: 1,
                    kv_heads: self.config.kv_heads,
                    capacity_tokens,
                    head_dim: self.config.head_dim,
                },
                max_cache_payload_bytes,
                stream,
            )?)
        })();

        self.finish_operation(result, stream)
    }

    /// Compute logits for every input position using a fresh cache.
    ///
    /// This is the complete cold-forward path used for model verification.
    pub fn forward(
        &self,
        tokens: &[u32],
        max_cache_payload_bytes: u64,
        max_logits_payload_bytes: u64,
        stream: &Stream,
    ) -> MlxQwenResult<MlxLogits> {
        self.check_healthy()?;
        validate_tokens(&self.config, tokens)?;

        let mut cache = self.new_cache(tokens.len(), max_cache_payload_bytes, stream)?;

        self.run(
            tokens,
            &mut cache,
            OutputSelection::All,
            max_logits_payload_bytes,
            stream,
            &mut || false,
        )
    }

    /// Append an actual input chunk and return its final-position logits.
    pub fn prefill(
        &self,
        tokens: &[u32],
        cache: &mut MlxKvCache,
        max_logits_payload_bytes: u64,
        stream: &Stream,
    ) -> MlxQwenResult<MlxLogits> {
        self.prefill_cancellable(tokens, cache, max_logits_payload_bytes, stream, || false)
    }

    /// Append an input chunk with cooperative cancellation.
    ///
    /// The callback returns true when cancellation is requested.
    /// It is checked before graph construction, between layers, before
    /// evaluation, and after evaluated output validation before commit.
    ///
    /// Every return attempts completion of the supplied execution stream.
    pub fn prefill_cancellable(
        &self,
        tokens: &[u32],
        cache: &mut MlxKvCache,
        max_logits_payload_bytes: u64,
        stream: &Stream,
        mut cancelled: impl FnMut() -> bool,
    ) -> MlxQwenResult<MlxLogits> {
        self.check_healthy()?;
        self.run(
            tokens,
            cache,
            OutputSelection::Last,
            max_logits_payload_bytes,
            stream,
            &mut cancelled,
        )
    }

    /// Consume one selected token and return its actual next-position logits.
    ///
    /// Token selection and incremental text decoding remain engine duties.
    pub fn decode(
        &self,
        token: u32,
        cache: &mut MlxKvCache,
        max_logits_payload_bytes: u64,
        stream: &Stream,
    ) -> MlxQwenResult<MlxLogits> {
        self.prefill(&[token], cache, max_logits_payload_bytes, stream)
    }

    fn run(
        &self,
        tokens: &[u32],
        cache: &mut MlxKvCache,
        selection: OutputSelection,
        max_logits_payload_bytes: u64,
        stream: &Stream,
        cancelled: &mut impl FnMut() -> bool,
    ) -> MlxQwenResult<MlxLogits> {
        let result = self.run_inner(
            tokens,
            cache,
            selection,
            max_logits_payload_bytes,
            stream,
            cancelled,
        );

        self.finish_operation(result, stream)
    }

    fn finish_operation<T>(&self, result: MlxQwenResult<T>, stream: &Stream) -> MlxQwenResult<T> {
        let result = complete_stream(result, stream);

        if result.as_ref().err().is_some_and(native_failure) {
            self.healthy.set(false);
        }

        result
    }

    fn run_inner(
        &self,
        tokens: &[u32],
        cache: &mut MlxKvCache,
        selection: OutputSelection,
        max_logits_payload_bytes: u64,
        stream: &Stream,
        cancelled: &mut impl FnMut() -> bool,
    ) -> MlxQwenResult<MlxLogits> {
        check_cancelled(cancelled)?;
        validate_tokens(&self.config, tokens)?;

        let c = &self.config;
        let cache_spec = cache.spec();

        if cache_spec.layers != c.num_layers
            || cache_spec.batch != 1
            || cache_spec.kv_heads != c.kv_heads
            || cache_spec.head_dim != c.head_dim
            || cache_spec.capacity_tokens > c.max_context
        {
            return Err(
                ModelError::InvalidConfig("MLX cache dimensions disagree with model").into(),
            );
        }

        let output_rows = match selection {
            OutputSelection::All => tokens.len(),
            OutputSelection::Last => 1,
        };
        let output_elements = checked_product(output_rows, c.vocab_size, "MLX output elements")?;
        let output_bytes = byte_count(output_elements)?;

        // Account conservatively for projected output, a possible
        // contiguous output, and the final host output payload.
        let required_output = output_bytes
            .checked_mul(3)
            .ok_or(ModelError::DimensionOverflow("MLX output payload overlap"))?;
        if required_output > max_logits_payload_bytes {
            return Err(MlxQwenError::PayloadLimit {
                operation: "MLX output payload",
                required_bytes: required_output,
                limit_bytes: max_logits_payload_bytes,
            });
        }

        let mut append = cache.begin_append(&self.owner, tokens.len(), stream)?;
        let position = append.start_position();
        let token_count = tokens.len();

        let mut hidden = self.weights.embedding.embedding_lookup(tokens, stream)?;

        for (index, layer) in self.weights.layers.iter().enumerate() {
            check_cancelled(cancelled)?;

            let normalized = ops::rms_norm(
                &hidden,
                &layer.input_layernorm,
                Qwen3Config::RMS_EPS,
                stream,
            )?;

            let query = layer.q_proj.apply(&normalized, stream)?;
            let key = layer.k_proj.apply(&normalized, stream)?;
            let value = layer.v_proj.apply(&normalized, stream)?;

            let query = ops::reshape(&query, &[1, token_count, c.q_heads, c.head_dim], stream)?;
            let key = ops::reshape(&key, &[1, token_count, c.kv_heads, c.head_dim], stream)?;
            let value = ops::reshape(&value, &[1, token_count, c.kv_heads, c.head_dim], stream)?;

            // Learned Q/K scales are shared across heads and normalize
            // only the final head dimension.
            let query = ops::rms_norm(&query, &layer.q_norm, Qwen3Config::RMS_EPS, stream)?;
            let key = ops::rms_norm(&key, &layer.k_norm, Qwen3Config::RMS_EPS, stream)?;

            let query = ops::transpose(&query, &[0, 2, 1, 3], stream)?;
            let key = ops::transpose(&key, &[0, 2, 1, 3], stream)?;
            let value = ops::transpose(&value, &[0, 2, 1, 3], stream)?;

            let query = attention::rope(&query, c.rope_theta, position, stream)?;
            let key = attention::rope(&key, c.rope_theta, position, stream)?;

            let prefix = append.stage_layer(index, &key, &value)?;

            let mask = if token_count == 1 {
                AttentionMask::Full
            } else {
                AttentionMask::CausalSuffix {
                    prefix_tokens: position,
                }
            };

            let attended = attention::grouped_query_attention(
                &query,
                prefix.keys(),
                prefix.values(),
                mask,
                stream,
            )?;

            let attended = ops::transpose(&attended, &[0, 2, 1, 3], stream)?;
            let attended = ops::reshape(&attended, &[token_count, c.q_dim()], stream)?;
            let attention_output = layer.o_proj.apply(&attended, stream)?;
            hidden = ops::residual_add(&hidden, &attention_output, stream)?;

            let normalized = ops::rms_norm(
                &hidden,
                &layer.post_attention_layernorm,
                Qwen3Config::RMS_EPS,
                stream,
            )?;
            let gate = layer.gate_proj.apply(&normalized, stream)?;
            let up = layer.up_proj.apply(&normalized, stream)?;
            let activated = ops::swiglu(&gate, &up, stream)?;
            let mlp_output = layer.down_proj.apply(&activated, stream)?;
            hidden = ops::residual_add(&hidden, &mlp_output, stream)?;
        }

        let normalized = ops::rms_norm(
            &hidden,
            &self.weights.final_layernorm,
            Qwen3Config::RMS_EPS,
            stream,
        )?;

        let selected = match selection {
            OutputSelection::All => normalized,
            OutputSelection::Last => ops::slice(
                &normalized,
                &[token_count - 1, 0],
                &[token_count, c.hidden_size],
                stream,
            )?,
        };

        let logits = self.weights.output_head().apply(&selected, stream)?;

        check_cancelled(cancelled)?;

        // This is the sole model-output readback. Intermediate transformer
        // tensors remain native MLX arrays.
        let values = ops::to_vec_f32(&logits, stream)?;

        if values.len() != output_elements {
            return Err(ModelError::ShapeMismatch {
                tensor: "MLX output logits",
                expected: output_elements,
                got: values.len(),
            }
            .into());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteActivation("MLX output logits").into());
        }

        // Check actual host-vector capacity as well as requested length.
        let actual_overlap = output_bytes
            .checked_mul(2)
            .and_then(|bytes| {
                byte_count(values.capacity())
                    .ok()
                    .and_then(|host| bytes.checked_add(host))
            })
            .ok_or(ModelError::DimensionOverflow("MLX actual output payload"))?;

        if actual_overlap > max_logits_payload_bytes {
            return Err(MlxQwenError::PayloadLimit {
                operation: "MLX actual output payload",
                required_bytes: actual_overlap,
                limit_bytes: max_logits_payload_bytes,
            });
        }

        check_cancelled(cancelled)?;

        // All layers and final logits succeeded. Publish K/V handles and
        // logical positions only after evaluation and cancellation checks.
        append.commit()?;

        Ok(MlxLogits {
            values,
            rows: output_rows,
            vocabulary: c.vocab_size,
        })
    }
}

fn check_cancelled(cancelled: &mut impl FnMut() -> bool) -> MlxQwenResult<()> {
    if cancelled() {
        Err(MlxQwenError::Cancelled)
    } else {
        Ok(())
    }
}

fn complete_stream<T>(result: MlxQwenResult<T>, stream: &Stream) -> MlxQwenResult<T> {
    // Completion errors take precedence because outstanding native work
    // must never be reported as successfully completed.
    stream.synchronize().map_err(MlxQwenError::Mlx)?;
    result
}

fn native_failure(error: &MlxQwenError) -> bool {
    match error {
        MlxQwenError::Memory(error) | MlxQwenError::Cache(KvCacheError::Memory(error)) => matches!(
            error,
            mlx_ffi::memory::MemoryError::Native { .. }
                | mlx_ffi::memory::MemoryError::Mlx(_)
                | mlx_ffi::memory::MemoryError::StreamSynchronization { .. }
        ),
        MlxQwenError::Mlx(_) => true,
        MlxQwenError::Operation(OperationError::Mlx(_)) => true,
        MlxQwenError::Cache(KvCacheError::Mlx(_)) => true,
        MlxQwenError::Cache(KvCacheError::Operation(OperationError::Mlx(_))) => true,
        _ => false,
    }
}

fn checked_product(left: usize, right: usize, name: &'static str) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::DimensionOverflow(name))
}

fn byte_count(elements: usize) -> Result<u64, ModelError> {
    let elements =
        u64::try_from(elements).map_err(|_| ModelError::DimensionOverflow("MLX payload length"))?;
    elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or(ModelError::DimensionOverflow("MLX payload bytes"))
}

fn shape_i32(shape: &[usize]) -> MlxQwenResult<(usize, Vec<i32>)> {
    let mut expected = 1_usize;
    let mut dimensions = Vec::new();
    dimensions
        .try_reserve_exact(shape.len())
        .map_err(|_| MlxQwenError::AllocationFailed("weight shape metadata"))?;
    for &dimension in shape {
        expected = checked_product(expected, dimension, "MLX upload shape")?;
        dimensions.push(
            i32::try_from(dimension)
                .map_err(|_| ModelError::InvalidConfig("MLX weight dimension exceeds C int"))?,
        );
    }
    Ok((expected, dimensions))
}

fn upload_f32(values: Vec<f32>, shape: &[usize]) -> MlxQwenResult<Array> {
    let (expected, dimensions) = shape_i32(shape)?;
    if expected != values.len() {
        return Err(ModelError::ShapeMismatch {
            tensor: "MLX upload",
            expected,
            got: values.len(),
        }
        .into());
    }
    let array = Array::from_f32_slice(&values, &dimensions)?;
    drop(values);
    Ok(array)
}

fn upload_u32(values: Vec<u32>, shape: &[usize]) -> MlxQwenResult<Array> {
    let (expected, dimensions) = shape_i32(shape)?;
    if expected != values.len() {
        return Err(ModelError::ShapeMismatch {
            tensor: "MLX u32 upload",
            expected,
            got: values.len(),
        }
        .into());
    }
    let array = Array::from_u32_slice(&values, &dimensions)?;
    drop(values);
    Ok(array)
}

fn check_weight(
    name: &'static str,
    expected: usize,
    values: &[f32],
    total: &mut u64,
) -> Result<(), ModelError> {
    if values.len() != expected {
        return Err(ModelError::ShapeMismatch {
            tensor: name,
            expected,
            got: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteWeight {
            tensor: name,
            index,
        });
    }

    *total = total
        .checked_add(byte_count(values.len())?)
        .ok_or(ModelError::DimensionOverflow("MLX total weight bytes"))?;
    Ok(())
}

fn check_matrix(
    name: &'static str,
    expected_elements: usize,
    matrix: &MatrixWeight,
    total: &mut u64,
) -> Result<(), ModelError> {
    if matrix.logical_elements() != expected_elements {
        return Err(ModelError::ShapeMismatch {
            tensor: name,
            expected: expected_elements,
            got: matrix.logical_elements(),
        });
    }
    match matrix {
        MatrixWeight::F32(values) => {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(ModelError::NonFiniteWeight {
                    tensor: name,
                    index,
                });
            }
        }
        MatrixWeight::Affine4 {
            scales,
            biases,
            cols,
            group_size,
            ..
        } => {
            if *group_size == 0 || cols % group_size != 0 || cols % 8 != 0 {
                return Err(ModelError::InvalidConfig(
                    "affine4 matrix geometry is invalid",
                ));
            }
            if let Some(index) = scales
                .iter()
                .chain(biases.iter())
                .position(|value| !value.is_finite())
            {
                return Err(ModelError::NonFiniteWeight {
                    tensor: name,
                    index,
                });
            }
        }
    }
    let bytes = u64::try_from(matrix.resident_bytes())
        .map_err(|_| ModelError::DimensionOverflow("MLX matrix bytes"))?;
    *total = total
        .checked_add(bytes)
        .ok_or(ModelError::DimensionOverflow("MLX total weight bytes"))?;
    Ok(())
}

fn validate_weights(c: &Qwen3Config, weights: &Qwen3Weights) -> Result<u64, ModelError> {
    for dimension in [
        c.hidden_size,
        c.num_layers,
        c.q_heads,
        c.kv_heads,
        c.head_dim,
        c.intermediate_size,
        c.vocab_size,
        c.max_context,
    ] {
        if dimension == 0 || i32::try_from(dimension).is_err() {
            return Err(ModelError::InvalidConfig(
                "MLX model dimensions must be positive and fit C int",
            ));
        }
    }

    if !c.q_heads.is_multiple_of(c.kv_heads) || !c.head_dim.is_multiple_of(2) {
        return Err(ModelError::InvalidConfig(
            "MLX Qwen requires integral GQA groups and even head dimensions",
        ));
    }
    if !c.rope_theta.is_finite() || c.rope_theta <= 0.0 {
        return Err(ModelError::InvalidConfig(
            "rope_theta must be finite and positive",
        ));
    }
    if c.tie_embeddings == weights.lm_head.is_some() {
        return Err(ModelError::InvalidConfig(
            "output-head presence disagrees with tied-embedding configuration",
        ));
    }

    let q_dim = checked_product(c.q_heads, c.head_dim, "MLX query width")?;
    let kv_dim = checked_product(c.kv_heads, c.head_dim, "MLX KV width")?;
    if i32::try_from(q_dim).is_err() || i32::try_from(kv_dim).is_err() {
        return Err(ModelError::InvalidConfig(
            "MLX projection width exceeds C int",
        ));
    }

    if weights.layers.len() != c.num_layers {
        return Err(ModelError::ShapeMismatch {
            tensor: "layers",
            expected: c.num_layers,
            got: weights.layers.len(),
        });
    }

    let mut total = 0_u64;
    check_matrix(
        "embed_tokens",
        checked_product(c.vocab_size, c.hidden_size, "embedding")?,
        &weights.embed_tokens,
        &mut total,
    )?;

    for layer in &weights.layers {
        check_weight(
            "input_layernorm",
            c.hidden_size,
            &layer.input_layernorm,
            &mut total,
        )?;
        check_matrix(
            "q_proj",
            checked_product(q_dim, c.hidden_size, "q_proj")?,
            &layer.q_proj,
            &mut total,
        )?;
        check_matrix(
            "k_proj",
            checked_product(kv_dim, c.hidden_size, "k_proj")?,
            &layer.k_proj,
            &mut total,
        )?;
        check_matrix(
            "v_proj",
            checked_product(kv_dim, c.hidden_size, "v_proj")?,
            &layer.v_proj,
            &mut total,
        )?;
        check_weight("q_norm", c.head_dim, &layer.q_norm, &mut total)?;
        check_weight("k_norm", c.head_dim, &layer.k_norm, &mut total)?;
        check_matrix(
            "o_proj",
            checked_product(c.hidden_size, q_dim, "o_proj")?,
            &layer.o_proj,
            &mut total,
        )?;
        check_weight(
            "post_attention_layernorm",
            c.hidden_size,
            &layer.post_attention_layernorm,
            &mut total,
        )?;
        check_matrix(
            "gate_proj",
            checked_product(c.intermediate_size, c.hidden_size, "gate_proj")?,
            &layer.gate_proj,
            &mut total,
        )?;
        check_matrix(
            "up_proj",
            checked_product(c.intermediate_size, c.hidden_size, "up_proj")?,
            &layer.up_proj,
            &mut total,
        )?;
        check_matrix(
            "down_proj",
            checked_product(c.hidden_size, c.intermediate_size, "down_proj")?,
            &layer.down_proj,
            &mut total,
        )?;
    }

    check_weight(
        "final_layernorm",
        c.hidden_size,
        &weights.final_layernorm,
        &mut total,
    )?;

    if let Some(head) = &weights.lm_head {
        check_matrix(
            "lm_head",
            checked_product(c.vocab_size, c.hidden_size, "lm_head")?,
            head,
            &mut total,
        )?;
    }

    Ok(total)
}

fn validate_tokens(c: &Qwen3Config, tokens: &[u32]) -> Result<(), ModelError> {
    if tokens.is_empty() {
        return Err(ModelError::EmptyInput);
    }
    if tokens.len() > c.max_context {
        return Err(ModelError::ContextTooLong {
            maximum: c.max_context,
            got: tokens.len(),
        });
    }

    for (position, &token) in tokens.iter().enumerate() {
        if u64::from(token) >= c.vocab_size as u64 {
            return Err(ModelError::InvalidToken {
                position,
                token,
                vocab_size: c.vocab_size,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fixture::generate_fixture, model::Qwen3Model};
    use mlx_ffi::{Device, DeviceType};

    const WEIGHTS: u64 = 16 * 1024 * 1024;
    const CACHE: u64 = 16 * 1024 * 1024;
    const LOGITS: u64 = 16 * 1024 * 1024;

    fn stream(kind: DeviceType) -> Stream {
        let device = Device::new(kind, 0).expect("device");
        assert!(
            device.is_available().expect("availability"),
            "requested device unavailable: {kind:?}",
        );
        let stream = Stream::new(&device).expect("stream");
        assert_eq!(stream.device().expect("device").kind().expect("kind"), kind,);
        stream
    }

    fn models(config: Qwen3Config, stream: &Stream) -> (MlxQwen3Model, Qwen3Model) {
        let native =
            MlxQwen3Model::from_weights(config.clone(), generate_fixture(&config), WEIGHTS, stream)
                .expect("actual MLX model");

        let reference = Qwen3Model::from_weights(config.clone(), generate_fixture(&config))
            .expect("reference model");

        (native, reference)
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(actual.is_finite());
            assert!(expected.is_finite());
            let tolerance = 2e-2 + 1e-2 * expected.abs();

            assert!(
                (actual - expected).abs() <= tolerance,
                "logit {index}: actual={actual}, expected={expected}",
            );
        }
    }

    fn reference_last(model: &Qwen3Model, tokens: &[u32]) -> Vec<f32> {
        model
            .forward(tokens)
            .expect("reference forward")
            .pop()
            .expect("reference final row")
    }

    fn forward_and_cached_suite(kind: DeviceType) {
        let stream = stream(kind);

        for tied in [true, false] {
            let mut config = Qwen3Config::tiny_fixture();
            config.tie_embeddings = tied;

            let (model, reference) = models(config, &stream);
            let tokens = [3, 41, 7, 12, 8, 45];

            let actual = model
                .forward(&tokens, CACHE, LOGITS, &stream)
                .expect("complete native forward");
            let expected = reference
                .forward(&tokens)
                .expect("complete reference forward");

            assert_eq!(actual.rows(), tokens.len());
            for (position, expected) in expected.iter().enumerate() {
                assert_close(actual.row(position).expect("row"), expected);
            }

            let mut cache = model.new_cache(16, CACHE, &stream).expect("cache");

            let first = model
                .prefill(&tokens[..2], &mut cache, LOGITS, &stream)
                .expect("initial native prefill");
            assert_eq!(cache.len(), 2);
            assert_eq!(first.rows(), 1);
            assert_close(first.last_row(), &reference_last(&reference, &tokens[..2]));

            let next = model
                .prefill(&tokens[2..5], &mut cache, LOGITS, &stream)
                .expect("cached multi-token prefill");
            assert_eq!(cache.len(), 5);
            assert_close(next.last_row(), &reference_last(&reference, &tokens[..5]));

            let decoded = model
                .decode(tokens[5], &mut cache, LOGITS, &stream)
                .expect("actual single-token decode");
            assert_eq!(cache.len(), 6);
            assert_close(decoded.last_row(), &reference_last(&reference, &tokens));
            assert_close(decoded.last_row(), actual.last_row());
        }
    }

    fn vocabulary_pair(rows: usize, cols: usize, seed: usize) -> (MatrixWeight, MatrixWeight) {
        const GROUP: usize = 64;
        assert_eq!(cols % GROUP, 0);
        let groups = cols / GROUP;
        let mut packed = vec![0_u32; rows * cols / 8];
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        let mut dense = Vec::new();
        for row in 0..rows {
            for group in 0..groups {
                let scale = 0.01 * (1 + (row + group + seed) % 3) as f32;
                let bias = -0.06 + 0.01 * (row % 5) as f32;
                scales.push(scale);
                biases.push(bias);
                for column in group * GROUP..(group + 1) * GROUP {
                    let code = ((row * 5 + column * 3 + seed) % 16) as u32;
                    packed[row * (cols / 8) + column / 8] |= code << ((column % 8) * 4);
                    dense.push(code as f32 * scale + bias);
                }
            }
        }
        (
            MatrixWeight::Affine4 {
                packed,
                scales,
                biases,
                rows,
                cols,
                group_size: GROUP,
            },
            MatrixWeight::F32(dense),
        )
    }

    fn packed_vocabulary_suite(streams: &[&Stream]) {
        for tied in [true, false] {
            let mut config = Qwen3Config::tiny_fixture();
            config.hidden_size = 64;
            config.intermediate_size = 128;
            config.tie_embeddings = tied;
            let mut packed = generate_fixture(&config);
            let mut dense = generate_fixture(&config);
            (packed.embed_tokens, dense.embed_tokens) =
                vocabulary_pair(config.vocab_size, config.hidden_size, 1);
            if !tied {
                let (quantized, reference) =
                    vocabulary_pair(config.vocab_size, config.hidden_size, 7);
                packed.lm_head = Some(quantized);
                dense.lm_head = Some(reference);
            }
            let expected_payload = validate_weights(&config, &packed).expect("packed payload");
            let dense_payload = validate_weights(&config, &dense).expect("dense payload");
            assert!(expected_payload < dense_payload);
            let reference =
                Qwen3Model::from_weights(config.clone(), dense).expect("independent dense oracle");
            let model = MlxQwen3Model::from_weights(config, packed, WEIGHTS, streams[0])
                .expect("packed vocabulary model");
            assert!(matches!(
                &model.weights.embedding,
                MlxLinear::Affine4 { .. }
            ));
            assert!(matches!(
                model.weights.output_head(),
                MlxLinear::Affine4 { .. }
            ));
            assert_eq!(model.weight_payload_bytes(), expected_payload);
            let measured = model.weight_buffer_bytes().expect("native packed storage");
            assert!(
                measured < dense_payload,
                "the vocabulary must not become a dense native table"
            );
            let identity = model.instance_id().clone();
            let tokens = [3, 41, 7, 12, 8];
            let mut cache = model.new_cache(16, CACHE, streams[0]).expect("one cache");
            let first = model
                .prefill(&tokens[..2], &mut cache, LOGITS, streams[0])
                .expect("CPU/selected prefix");
            assert_close(first.last_row(), &reference_last(&reference, &tokens[..2]));
            let continuation_stream = streams[streams.len() - 1];
            let next = model
                .prefill(&tokens[2..4], &mut cache, LOGITS, continuation_stream)
                .expect("same cache continuation");
            assert_close(next.last_row(), &reference_last(&reference, &tokens[..4]));
            let last = model
                .decode(tokens[4], &mut cache, LOGITS, continuation_stream)
                .expect("quantized output head decode");
            assert_close(last.last_row(), &reference_last(&reference, &tokens));
            assert_eq!(cache.len(), tokens.len());
            assert!(cache.owner().same_instance(&identity));
            assert!(model.instance_id().same_instance(&identity));
            assert_eq!(
                model
                    .weight_buffer_bytes()
                    .expect("unchanged packed storage"),
                measured
            );
        }
    }

    #[test]
    fn packed_vocabulary_cpu_prefill_decode_match_dense_oracle() {
        let cpu = stream(DeviceType::Cpu);
        packed_vocabulary_suite(&[&cpu]);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn one_packed_vocabulary_and_cache_continue_cpu_to_metal() {
        let cpu = stream(DeviceType::Cpu);
        let metal = stream(DeviceType::Gpu);
        packed_vocabulary_suite(&[&cpu, &metal]);
    }

    #[test]
    fn mlx_cpu_complete_forward_prefill_and_decode_match_reference() {
        forward_and_cached_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_complete_forward_prefill_and_decode_match_reference() {
        forward_and_cached_suite(DeviceType::Gpu);
    }

    #[test]
    fn query_projection_width_may_differ_from_hidden_size() {
        let stream = stream(DeviceType::Cpu);
        let mut config = Qwen3Config::tiny_fixture();
        config.q_heads = 6;

        assert_ne!(config.q_dim(), config.hidden_size);
        let (model, reference) = models(config, &stream);
        let tokens = [3, 41, 7];

        let actual = model
            .forward(&tokens, CACHE, LOGITS, &stream)
            .expect("decoupled projection forward");
        let expected = reference.forward(&tokens).expect("reference");

        for (index, expected) in expected.iter().enumerate() {
            assert_close(actual.row(index).expect("row"), expected);
        }
    }

    #[test]
    fn cancellation_abandons_staged_layers_without_advancing_cache() {
        let stream = stream(DeviceType::Cpu);
        let (model, _) = models(Qwen3Config::tiny_fixture(), &stream);
        let mut cache = model.new_cache(16, CACHE, &stream).expect("cache");

        model
            .prefill(&[3, 41], &mut cache, LOGITS, &stream)
            .expect("seed");
        let before = cache.len();

        let mut checks = 0;
        let result = model.prefill_cancellable(&[7, 12], &mut cache, LOGITS, &stream, || {
            checks += 1;
            checks >= 3
        });

        assert!(matches!(result, Err(MlxQwenError::Cancelled)));
        assert_eq!(cache.len(), before);
        assert!(cache.is_healthy());
        assert!(model.is_healthy());

        model
            .decode(7, &mut cache, LOGITS, &stream)
            .expect("recovery");
        assert_eq!(cache.len(), before + 1);
    }

    #[test]
    fn foreign_cache_and_invalid_tokens_do_not_advance_state() {
        let stream = stream(DeviceType::Cpu);
        let (owner, _) = models(Qwen3Config::tiny_fixture(), &stream);
        let (foreign, _) = models(Qwen3Config::tiny_fixture(), &stream);
        let mut cache = owner.new_cache(16, CACHE, &stream).expect("cache");

        owner
            .prefill(&[3], &mut cache, LOGITS, &stream)
            .expect("seed");
        let before = cache.len();

        assert!(matches!(
            foreign.decode(7, &mut cache, LOGITS, &stream),
            Err(MlxQwenError::Cache(KvCacheError::ForeignModel)),
        ));
        assert!(matches!(
            owner.decode(u32::MAX, &mut cache, LOGITS, &stream),
            Err(MlxQwenError::Model(ModelError::InvalidToken { .. })),
        ));
        assert_eq!(cache.len(), before);
    }

    #[test]
    fn output_limit_failure_preserves_committed_cache() {
        let stream = stream(DeviceType::Cpu);
        let (model, _) = models(Qwen3Config::tiny_fixture(), &stream);
        let mut cache = model.new_cache(16, CACHE, &stream).expect("cache");

        model
            .prefill(&[3], &mut cache, LOGITS, &stream)
            .expect("seed");
        let before = cache.len();

        assert!(matches!(
            model.decode(7, &mut cache, 1, &stream),
            Err(MlxQwenError::PayloadLimit { .. }),
        ));
        assert_eq!(cache.len(), before);
    }

    #[test]
    fn invalid_weights_and_small_weight_limit_are_rejected() {
        let stream = stream(DeviceType::Cpu);
        let config = Qwen3Config::tiny_fixture();

        let mut weights = generate_fixture(&config);
        weights.layers[0].q_norm.pop();

        assert!(matches!(
            MlxQwen3Model::from_weights(config.clone(), weights, WEIGHTS, &stream,),
            Err(MlxQwenError::Model(ModelError::ShapeMismatch { .. })),
        ));

        assert!(matches!(
            MlxQwen3Model::from_weights(config.clone(), generate_fixture(&config), 1, &stream,),
            Err(MlxQwenError::PayloadLimit { .. }),
        ));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn one_model_and_cache_continue_from_cpu_to_metal() {
        let cpu = stream(DeviceType::Cpu);
        let metal = stream(DeviceType::Gpu);
        let (model, reference) = models(Qwen3Config::tiny_fixture(), &cpu);
        let identity = model.instance_id().clone();
        let payload = model.weight_payload_bytes();

        let mut cache = model.new_cache(16, CACHE, &cpu).expect("cache");
        model
            .prefill(&[3, 41], &mut cache, LOGITS, &cpu)
            .expect("CPU prefix");

        let actual = model
            .prefill(&[7, 12], &mut cache, LOGITS, &metal)
            .expect("Metal continuation");

        assert_eq!(cache.len(), 4);
        assert!(cache.owner().same_instance(&identity));
        assert!(model.instance_id().same_instance(&identity));
        assert_eq!(model.weight_payload_bytes(), payload);
        assert_close(
            actual.last_row(),
            &reference_last(&reference, &[3, 41, 7, 12]),
        );

        metal.synchronize().expect("Metal completion");
        cpu.synchronize().expect("CPU completion");
    }
}

impl From<mlx_ffi::memory::MemoryError> for MlxQwenError {
    fn from(error: mlx_ffi::memory::MemoryError) -> Self {
        Self::Memory(error)
    }
}

#[cfg(test)]
mod measured_weight_buffer_tests {
    use super::*;
    use crate::fixture::generate_fixture;
    use mlx_ffi::{Device, DeviceType};
    fn check(kind: DeviceType) {
        let device = Device::new(kind, 0).expect("device");
        assert!(device.is_available().expect("availability"));
        let stream = Stream::new(&device).expect("stream");
        for tied in [true, false] {
            let mut config = Qwen3Config::tiny_fixture();
            config.tie_embeddings = tied;
            let model = MlxQwen3Model::from_weights(
                config.clone(),
                generate_fixture(&config),
                16 * 1024 * 1024,
                &stream,
            )
            .expect("native model");
            let owner = model.instance_id().clone();
            let measured = model.weight_buffer_bytes().expect("measurement");
            assert!(measured >= model.weight_payload_bytes());
            let mut arrays = Vec::new();
            model.weights.visit(|array| arrays.push(array));
            assert_eq!(
                measured as usize,
                mlx_ffi::memory::array_buffer_bytes(&arrays).expect("buffer set")
            );
            if !tied {
                let without_head = mlx_ffi::memory::array_buffer_bytes(&arrays[..arrays.len() - 1])
                    .expect("without output head");
                assert!(measured as usize > without_head);
            }
            model
                .weights
                .output_head()
                .visit(&mut |array| arrays.push(array));
            assert_eq!(
                measured as usize,
                mlx_ffi::memory::array_buffer_bytes(&arrays).expect("duplicate alias")
            );
            drop(arrays);
            model
                .forward(&[3, 41], 16 * 1024 * 1024, 16 * 1024 * 1024, &stream)
                .expect("execution");
            assert!(model.instance_id().same_instance(&owner));
            assert_eq!(model.weight_buffer_bytes().expect("same weights"), measured);
        }
    }
    #[test]
    fn measured_cpu_weights_deduplicate_and_survive_execution() {
        check(DeviceType::Cpu);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn measured_metal_weights_deduplicate_and_survive_execution() {
        check(DeviceType::Gpu);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn one_weight_set_keeps_its_measurement_across_cpu_and_metal() {
        let cpu_device = Device::new(DeviceType::Cpu, 0).expect("CPU");
        let metal_device = Device::new(DeviceType::Gpu, 0).expect("Metal");
        assert!(metal_device.is_available().expect("Metal availability"));
        let cpu = Stream::new(&cpu_device).expect("CPU stream");
        let metal = Stream::new(&metal_device).expect("Metal stream");
        let config = Qwen3Config::tiny_fixture();
        let model = MlxQwen3Model::from_weights(
            config.clone(),
            generate_fixture(&config),
            16 * 1024 * 1024,
            &cpu,
        )
        .expect("one loaded model");
        let identity = model.instance_id().clone();
        let before = model.weight_buffer_bytes().expect("initial weights");
        for stream in [&cpu, &metal] {
            model
                .forward(&[3, 41], 16 * 1024 * 1024, 16 * 1024 * 1024, stream)
                .expect("actual execution");
            assert!(model.instance_id().same_instance(&identity));
            assert_eq!(
                model.weight_buffer_bytes().expect("same weight set"),
                before
            );
        }
    }
}

// ALLOY_MEASURED_BUFFERS_V1
