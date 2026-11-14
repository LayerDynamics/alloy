//! Native Qwen3.5 linear attention with explicit, immutable recurrent state.
//!
//! This block is shared by the hybrid model. It does not implement full
//! attention, tokenize, advertise a qualified profile, or persist a cache.
//! Arrays stay on the caller's worker and selected stream. A successful step
//! returns evaluated output and both new state components; the caller publishes
//! them only when the entire model step succeeds.
use super::{
    fixture::MatrixWeight,
    mlx_model::{
        MlxDecoderError, MlxDecoderResult, MlxLinear, check_matrix, check_weight, upload_f32,
    },
    model::ModelError,
};
use crate::kv_cache::MlxModelId;
use mlx_ffi::{Array, DType, Stream, ops};

/// Architecture of one Qwen3.5 GatedDeltaNet block.
#[derive(Clone, Copy, Debug)]
pub struct LinearAttentionConfig {
    /// Input and output residual width.
    pub hidden_size: usize,
    /// Query/key heads; value heads may be an integral multiple.
    pub key_heads: usize,
    /// Value/recurrent-state heads.
    pub value_heads: usize,
    /// Query/key head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// Causal convolution taps, including the current sample.
    pub convolution_kernel: usize,
    /// Learned output RMS normalization epsilon.
    pub rms_epsilon: f32,
}

#[derive(Clone, Copy)]
struct Geometry {
    key_width: usize,
    value_width: usize,
    convolution_width: usize,
    history_elements: usize,
    recurrent_elements: usize,
}

fn invalid(message: &'static str) -> MlxDecoderError {
    ModelError::InvalidConfig(message).into()
}
fn product(a: usize, b: usize) -> MlxDecoderResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("hybrid tensor dimension overflow"))
}
fn sum(a: usize, b: usize) -> MlxDecoderResult<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("hybrid tensor dimension overflow"))
}
fn bytes(elements: usize) -> MlxDecoderResult<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("hybrid payload overflow"))
}
fn limit(operation: &'static str, required_bytes: u64, limit_bytes: u64) -> MlxDecoderResult<()> {
    if required_bytes > limit_bytes {
        return Err(MlxDecoderError::PayloadLimit {
            operation,
            required_bytes,
            limit_bytes,
        });
    }
    Ok(())
}
impl LinearAttentionConfig {
    /// Logical convolution and recurrent state payload for one layer.
    pub fn state_payload_bytes(self) -> MlxDecoderResult<u64> {
        let g = self.geometry()?;
        bytes(sum(g.history_elements, g.recurrent_elements)?)
    }

    /// Named step buffers, including previous/new state. Native allocator
    /// caches and kernel workspace require separate caller headroom.
    pub fn step_payload_bytes(self, tokens: usize) -> MlxDecoderResult<u64> {
        let g = self.geometry()?;
        if tokens == 0 || i32::try_from(sum(tokens, self.convolution_kernel - 1)?).is_err() {
            return Err(invalid("invalid hybrid step token count"));
        }
        let per_token = sum(
            sum(product(5, g.convolution_width)?, product(6, g.value_width)?)?,
            sum(product(8, self.value_heads)?, self.hidden_size)?,
        )?;
        bytes(sum(
            product(tokens, per_token)?,
            product(3, sum(g.history_elements, g.recurrent_elements)?)?,
        )?)
    }

    fn geometry(self) -> MlxDecoderResult<Geometry> {
        for n in [
            self.hidden_size,
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
            self.convolution_kernel,
        ] {
            if n == 0 || i32::try_from(n).is_err() {
                return Err(invalid("hybrid dimensions must be positive C integers"));
            }
        }
        if self.convolution_kernel < 2
            || !self.value_heads.is_multiple_of(self.key_heads)
            || !self.rms_epsilon.is_finite()
            || self.rms_epsilon <= 0.0
        {
            return Err(invalid(
                "invalid hybrid convolution, grouped heads or RMS epsilon",
            ));
        }
        let key_width = product(self.key_heads, self.key_dim)?;
        let value_width = product(self.value_heads, self.value_dim)?;
        let convolution_width = sum(product(2, key_width)?, value_width)?;
        if i32::try_from(convolution_width).is_err() {
            return Err(invalid("hybrid projection width exceeds C integer"));
        }
        Ok(Geometry {
            key_width,
            value_width,
            convolution_width,
            history_elements: product(self.convolution_kernel - 1, convolution_width)?,
            recurrent_elements: product(value_width, self.key_dim)?,
        })
    }
}

/// Host weights before one native upload. Affine matrices remain packed.
#[derive(Clone, Debug)]
pub struct LinearAttentionWeights {
    /// Concatenated Q, K, V projections in that exact channel order.
    pub qkv: MatrixWeight,
    /// Per-value-channel output gate projection.
    pub z: MatrixWeight,
    /// Per-value-head decay projection.
    pub a: MatrixWeight,
    /// Per-value-head update projection.
    pub b: MatrixWeight,
    /// Depthwise convolution [channels, kernel, 1], already sanitized.
    pub convolution: Vec<f32>,
    /// Log decay rates [value heads].
    pub a_log: Vec<f32>,
    /// Softplus time bias [value heads].
    pub dt_bias: Vec<f32>,
    /// Gated output normalization [value head dimension].
    pub norm: Vec<f32>,
    /// Residual output projection.
    pub output: MatrixWeight,
}

impl LinearAttentionWeights {
    pub(crate) fn validated_bytes(&self, config: LinearAttentionConfig) -> MlxDecoderResult<u64> {
        let c = config;
        let g = config.geometry()?;
        let weights = self;
        let mut total = 0;
        for (name, matrix, rows, cols) in [
            (
                "hybrid qkv",
                &weights.qkv,
                g.convolution_width,
                c.hidden_size,
            ),
            ("hybrid z", &weights.z, g.value_width, c.hidden_size),
            ("hybrid a", &weights.a, c.value_heads, c.hidden_size),
            ("hybrid b", &weights.b, c.value_heads, c.hidden_size),
            (
                "hybrid output",
                &weights.output,
                c.hidden_size,
                g.value_width,
            ),
        ] {
            check_matrix(name, product(rows, cols)?, matrix, &mut total)?;
        }
        for (name, values, count) in [
            (
                "hybrid convolution",
                &weights.convolution,
                product(g.convolution_width, c.convolution_kernel)?,
            ),
            ("hybrid A_log", &weights.a_log, c.value_heads),
            ("hybrid dt_bias", &weights.dt_bias, c.value_heads),
            ("hybrid norm", &weights.norm, c.value_dim),
        ] {
            check_weight(name, count, values, &mut total)?;
        }
        Ok(total)
    }
}

/// Evaluated immutable state, bound to the exact block that produced it.
/// Recurrent history cannot be truncated by slicing a KV tensor. Rebuild an
/// edited prefix from `None`, or restore an independently validated checkpoint.
pub struct LinearAttentionState {
    owner: MlxModelId,
    tokens: usize,
    convolution: Array,
    recurrent: Array,
}
impl LinearAttentionState {
    /// Number of tokens represented by both state components.
    pub fn len(&self) -> usize {
        self.tokens
    }
    /// Whether no tokens have been consumed.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }
    /// Borrow convolution history [1,K-1,C].
    pub fn convolution(&self) -> &Array {
        &self.convolution
    }
    /// Borrow recurrent state [1,Hv,Dv,Dk].
    pub fn recurrent(&self) -> &Array {
        &self.recurrent
    }
}

/// One resident native block. No unsafe Send/Sync implementation is used.
pub struct MlxLinearAttention {
    config: LinearAttentionConfig,
    geometry: Geometry,
    owner: MlxModelId,
    qkv: MlxLinear,
    z: MlxLinear,
    a: MlxLinear,
    b: MlxLinear,
    convolution: Array,
    a_log: Array,
    dt_bias: Array,
    norm: Array,
    output: MlxLinear,
    weight_bytes: u64,
}
impl MlxLinearAttention {
    /// Validate and upload once. The allowance covers resident tensor payload,
    /// not allocator overhead, input host buffers or temporary upload overlap.
    pub fn from_weights(
        config: LinearAttentionConfig,
        weights: LinearAttentionWeights,
        weight_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<Self> {
        let g = config.geometry()?;
        let c = config;
        let total = weights.validated_bytes(config)?;
        limit("hybrid weights", total, weight_limit)?;
        let block = Self {
            config,
            geometry: g,
            owner: MlxModelId::new(),
            weight_bytes: total,
            qkv: MlxLinear::upload(weights.qkv, g.convolution_width, c.hidden_size)?,
            z: MlxLinear::upload(weights.z, g.value_width, c.hidden_size)?,
            a: MlxLinear::upload(weights.a, c.value_heads, c.hidden_size)?,
            b: MlxLinear::upload(weights.b, c.value_heads, c.hidden_size)?,
            output: MlxLinear::upload(weights.output, c.hidden_size, g.value_width)?,
            convolution: upload_f32(
                weights.convolution,
                &[g.convolution_width, c.convolution_kernel, 1],
            )?,
            a_log: upload_f32(weights.a_log, &[c.value_heads])?,
            dt_bias: upload_f32(weights.dt_bias, &[c.value_heads])?,
            norm: upload_f32(weights.norm, &[c.value_dim])?,
        };
        // Evaluate resident projections now; failures cannot publish a block.
        let mut evaluation = Ok(());
        let mut evaluate = |a: &Array| {
            if evaluation.is_ok() {
                evaluation = a.eval();
            }
        };
        for linear in [&block.qkv, &block.z, &block.a, &block.b, &block.output] {
            linear.visit(&mut evaluate);
        }
        for array in [
            &block.convolution,
            &block.a_log,
            &block.dt_bias,
            &block.norm,
        ] {
            evaluate(array);
        }
        evaluation?;
        stream.synchronize()?;
        Ok(block)
    }

    pub(crate) fn visit<'a>(&'a self, visitor: &mut impl FnMut(&'a Array)) {
        for linear in [&self.qkv, &self.z, &self.a, &self.b, &self.output] {
            linear.visit(visitor);
        }
        for array in [&self.convolution, &self.a_log, &self.dt_bias, &self.norm] {
            visitor(array);
        }
    }

    /// Resident weight tensor bytes, excluding allocator overhead.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Fixed two-component state payload for batch one.
    pub fn state_bytes(&self) -> MlxDecoderResult<u64> {
        self.config.state_payload_bytes()
    }

    /// Step with [1,T,hidden] float32 input, returning evaluated output and new
    /// state. `capacity` bounds total sequence length. `payload_limit` covers
    /// named intermediate/output/state tensors, not native kernel workspaces or
    /// allocator caches. Engine admission must account for those separately.
    /// Cancellation or failure leaves the borrowed prior state untouched.
    pub fn forward(
        &self,
        input: &Array,
        previous: Option<&LinearAttentionState>,
        capacity: usize,
        payload_limit: u64,
        stream: &Stream,
        cancelled: &dyn Fn() -> bool,
    ) -> MlxDecoderResult<(Array, LinearAttentionState)> {
        let c = self.config;
        let g = self.geometry;
        let shape = input.shape();
        if input.dtype() != DType::Float32
            || shape.len() != 3
            || shape[0] != 1
            || shape[1] == 0
            || shape[2] != c.hidden_size
        {
            return Err(invalid(
                "hybrid input must be nonempty float32 [1,T,hidden]",
            ));
        }
        let t = shape[1];
        if previous.is_some_and(|s| s.owner != self.owner) {
            return Err(invalid("hybrid state belongs to a different loaded block"));
        }
        let tokens = sum(previous.map_or(0, |s| s.tokens), t)?;
        if tokens > capacity || i32::try_from(sum(t, c.convolution_kernel - 1)?).is_err() {
            return Err(invalid("hybrid sequence exceeds capacity"));
        }
        // Conservative named-tensor payload, including prior and next state,
        // QKV/conv, normalized Q/K, gates, gated values and projected output.
        let required = c.step_payload_bytes(t)?;
        limit("hybrid step tensor payload", required, payload_limit)?;
        if cancelled() {
            return Err(MlxDecoderError::Cancelled);
        }
        if !ops::all_finite(input, stream)? {
            return Err(invalid("nonfinite hybrid input"));
        }
        let flat_input = ops::reshape(input, &[t, c.hidden_size], stream)?;
        let project = |linear: &MlxLinear, width| -> MlxDecoderResult<Array> {
            Ok(ops::reshape(
                &linear.apply(&flat_input, stream)?,
                &[1, t, width],
                stream,
            )?)
        };
        let qkv = project(&self.qkv, g.convolution_width)?;
        let z = ops::reshape(
            &project(&self.z, g.value_width)?,
            &[1, t, c.value_heads, c.value_dim],
            stream,
        )?;
        let a = project(&self.a, c.value_heads)?;
        let b = project(&self.b, c.value_heads)?;
        let zero;
        let history = if let Some(state) = previous {
            &state.convolution
        } else {
            zero = ops::broadcast_to(
                &Array::from_f32_slice(&[0.0], &[])?,
                &[1, c.convolution_kernel - 1, g.convolution_width],
                stream,
            )?;
            &zero
        };
        let conv_input = ops::concatenate_axis(history, &qkv, 1, stream)?;
        // A contiguous slice can still retain the entire concatenation's
        // backing buffer. Gather exactly K-1 rows to materialize bounded state.
        let history_rows: Vec<u32> = (t..t + c.convolution_kernel - 1)
            .map(|row| row as u32)
            .collect();
        let convolution = ops::reshape(
            &ops::embedding_lookup(
                &ops::reshape(
                    &conv_input,
                    &[t + c.convolution_kernel - 1, g.convolution_width],
                    stream,
                )?,
                &history_rows,
                stream,
            )?,
            &[1, c.convolution_kernel - 1, g.convolution_width],
            stream,
        )?;
        let conv = ops::silu(
            &ops::depthwise_conv1d(&conv_input, &self.convolution, stream)?,
            stream,
        )?;
        let split = |start, end, heads, dim| -> MlxDecoderResult<Array> {
            Ok(ops::reshape(
                &ops::slice(&conv, &[0, 0, start], &[1, t, end], stream)?,
                &[1, t, heads, dim],
                stream,
            )?)
        };
        let q = split(0, g.key_width, c.key_heads, c.key_dim)?;
        let k = split(g.key_width, 2 * g.key_width, c.key_heads, c.key_dim)?;
        let v = split(
            2 * g.key_width,
            g.convolution_width,
            c.value_heads,
            c.value_dim,
        )?;
        let inv_scale = (c.key_dim as f64).powf(-0.5);
        let q = ops::scale(
            &ops::rms_norm_unweighted(&q, 1e-6, stream)?,
            (inv_scale * inv_scale) as f32,
            stream,
        )?;
        let k = ops::scale(
            &ops::rms_norm_unweighted(&k, 1e-6, stream)?,
            inv_scale as f32,
            stream,
        )?;
        let dt = ops::broadcast_to(&self.dt_bias, &[1, t, c.value_heads], stream)?;
        let softplus = ops::softplus(&ops::residual_add(&a, &dt, stream)?, stream)?;
        let rate = ops::broadcast_to(
            &ops::scale(&ops::exp(&self.a_log, stream)?, -1.0, stream)?,
            &[1, t, c.value_heads],
            stream,
        )?;
        let decay = ops::exp(&ops::multiply(&rate, &softplus, stream)?, stream)?;
        let beta = ops::sigmoid(&b, stream)?;
        let (out, recurrent) = ops::gated_delta_update(
            &q,
            &k,
            &v,
            &decay,
            &beta,
            previous.map(|s| &s.recurrent),
            stream,
        )?;
        let out = ops::multiply(
            &ops::rms_norm(&out, &self.norm, c.rms_epsilon, stream)?,
            &ops::silu(&z, stream)?,
            stream,
        )?;
        let out = self
            .output
            .apply(&ops::reshape(&out, &[t, g.value_width], stream)?, stream)?;
        let out = ops::reshape(&out, &[1, t, c.hidden_size], stream)?;
        for array in [&out, &convolution, &recurrent] {
            if !ops::all_finite(array, stream)? {
                return Err(invalid("nonfinite hybrid output/state"));
            }
        }
        if cancelled() {
            return Err(MlxDecoderError::Cancelled);
        }
        Ok((
            out,
            LinearAttentionState {
                owner: self.owner.clone(),
                tokens,
                convolution,
                recurrent,
            },
        ))
    }
}
