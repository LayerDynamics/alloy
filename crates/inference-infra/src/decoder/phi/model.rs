//! Native Phi fused projections, packed weights and transactional full KV.
use super::{PhiConfig, PhiRopeRegime, config::invalid, partial_longrope, weights::PhiWeights};
use crate::{
    decoder::{
        fixture::MatrixWeight,
        mlx_model::{
            MlxDecoderError, MlxDecoderResult, MlxLinear, MlxLogits, check_matrix, check_weight,
            upload_f32,
        },
    },
    kv_cache::{KvCacheError, MlxKvCache, MlxModelId},
};
use mlx_ffi::{
    Array, Stream,
    attention::{self, AttentionMask},
    memory, ops,
};
use std::cell::Cell;

fn limit(operation: &'static str, required_bytes: u64, limit_bytes: u64) -> MlxDecoderResult<()> {
    if required_bytes > limit_bytes {
        Err(MlxDecoderError::PayloadLimit {
            operation,
            required_bytes,
            limit_bytes,
        })
    } else {
        Ok(())
    }
}
fn cancelled(check: &mut impl FnMut() -> bool) -> MlxDecoderResult<()> {
    if check() {
        Err(MlxDecoderError::Cancelled)
    } else {
        Ok(())
    }
}
struct Layer {
    input_norm: Array,
    qkv: MlxLinear,
    output: MlxLinear,
    post_norm: Array,
    gate_up: MlxLinear,
    down: MlxLinear,
}
impl Layer {
    fn visit<'a>(&'a self, f: &mut impl FnMut(&'a Array)) {
        f(&self.input_norm);
        f(&self.post_norm);
        for m in [&self.qkv, &self.output, &self.gate_up, &self.down] {
            m.visit(f);
        }
    }
}
pub struct MlxPhiModel {
    config: PhiConfig,
    regime: PhiRopeRegime,
    owner: MlxModelId,
    embedding: MlxLinear,
    layers: Vec<Layer>,
    norm: Array,
    head: Option<MlxLinear>,
    payload: u64,
    healthy: Cell<bool>,
}
fn validate_weights(c: &PhiConfig, w: &PhiWeights) -> MlxDecoderResult<u64> {
    c.validate()?;
    if w.layers.len() != c.layers || c.tied_embeddings == w.head.is_some() {
        return Err(invalid("Phi weights/config mismatch").into());
    }
    let mut total = 0;
    let matrix = |name: &'static str,
                  m: &MatrixWeight,
                  rows: usize,
                  cols: usize,
                  total: &mut u64|
     -> MlxDecoderResult<()> {
        if m.bits() != c.group_size.map(|_| c.quantization_bits)
            || m.affine4_group_size() != c.group_size
        {
            return Err(invalid("Phi matrix representation differs from declared plan").into());
        }
        let n = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("Phi matrix size overflow"))?;
        Ok(check_matrix(name, n, m, total)?)
    };
    matrix(
        "Phi embedding",
        &w.embedding,
        c.vocabulary,
        c.hidden,
        &mut total,
    )?;
    if let Some(h) = &w.head {
        matrix("Phi head", h, c.vocabulary, c.hidden, &mut total)?;
    }
    check_weight("Phi norm", c.hidden, &w.norm, &mut total)?;
    for l in &w.layers {
        check_weight("Phi input norm", c.hidden, &l.input_norm, &mut total)?;
        check_weight("Phi post norm", c.hidden, &l.post_norm, &mut total)?;
        for (name, m, rows, cols) in [
            ("Phi fused QKV", &l.qkv, c.qkv_width(), c.hidden),
            ("Phi attention output", &l.output, c.hidden, c.hidden),
            ("Phi fused gate/up", &l.gate_up, c.gate_up_width(), c.hidden),
            ("Phi down", &l.down, c.hidden, c.intermediate),
        ] {
            matrix(name, m, rows, cols, &mut total)?;
        }
    }
    if total != c.weight_payload_plan()?.payload_bytes {
        return Err(invalid("Phi weight payload differs from declared plan").into());
    }
    Ok(total)
}
impl MlxPhiModel {
    pub fn from_weights(
        config: PhiConfig,
        weights: PhiWeights,
        regime: PhiRopeRegime,
        weight_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<Self> {
        let result = (|| {
            let payload = validate_weights(&config, &weights)?;
            limit("Phi weight payload", payload, weight_limit)?;
            let c = &config;
            let embedding = MlxLinear::upload(weights.embedding, c.vocabulary, c.hidden)?;
            let norm = upload_f32(weights.norm, &[c.hidden])?;
            let head = weights
                .head
                .map(|h| MlxLinear::upload(h, c.vocabulary, c.hidden))
                .transpose()?;
            let mut layers = Vec::new();
            layers
                .try_reserve_exact(c.layers)
                .map_err(|_| MlxDecoderError::AllocationFailed("Phi layer metadata"))?;
            for l in weights.layers {
                layers.push(Layer {
                    input_norm: upload_f32(l.input_norm, &[c.hidden])?,
                    post_norm: upload_f32(l.post_norm, &[c.hidden])?,
                    qkv: MlxLinear::upload(l.qkv, c.qkv_width(), c.hidden)?,
                    output: MlxLinear::upload(l.output, c.hidden, c.hidden)?,
                    gate_up: MlxLinear::upload(l.gate_up, c.gate_up_width(), c.hidden)?,
                    down: MlxLinear::upload(l.down, c.hidden, c.intermediate)?,
                });
            }
            let model = Self {
                config,
                regime,
                owner: MlxModelId::new(),
                embedding,
                layers,
                norm,
                head,
                payload,
                healthy: Cell::new(true),
            };
            let mut failure = None;
            model.visit(&mut |a| {
                if let Err(e) = a.eval() {
                    failure.get_or_insert(e);
                }
            });
            if let Some(e) = failure {
                return Err(e.into());
            }
            limit(
                "Phi native weight buffers",
                model.weight_buffer_bytes()?,
                weight_limit,
            )?;
            Ok(model)
        })();
        stream.synchronize()?;
        result
    }
    fn visit<'a>(&'a self, f: &mut impl FnMut(&'a Array)) {
        self.embedding.visit(f);
        f(&self.norm);
        if let Some(h) = &self.head {
            h.visit(f);
        }
        for l in &self.layers {
            l.visit(f);
        }
    }
    pub fn config(&self) -> &PhiConfig {
        &self.config
    }
    pub fn regime(&self) -> PhiRopeRegime {
        self.regime
    }
    pub fn instance_id(&self) -> &MlxModelId {
        &self.owner
    }
    pub fn is_healthy(&self) -> bool {
        self.healthy.get()
    }
    pub fn weight_payload_bytes(&self) -> u64 {
        self.payload
    }
    pub fn weight_buffer_bytes(&self) -> MlxDecoderResult<u64> {
        self.check_healthy()?;
        let mut arrays = Vec::new();
        arrays
            .try_reserve_exact(14 * self.layers.len() + 7)
            .map_err(|_| MlxDecoderError::AllocationFailed("Phi weight buffer metadata"))?;
        self.visit(&mut |a| arrays.push(a));
        Ok(memory::array_buffer_bytes(&arrays)? as u64)
    }
    fn check_healthy(&self) -> MlxDecoderResult<()> {
        if self.healthy.get() {
            Ok(())
        } else {
            Err(MlxDecoderError::Invalidated)
        }
    }
    fn finish<T>(&self, result: MlxDecoderResult<T>, stream: &Stream) -> MlxDecoderResult<T> {
        if let Err(e) = stream.synchronize() {
            self.healthy.set(false);
            return Err(e.into());
        }
        if let Err(e) = &result {
            let native = match e {
                MlxDecoderError::Mlx(_)
                | MlxDecoderError::Operation(ops::OperationError::Mlx(_))
                | MlxDecoderError::Cache(KvCacheError::Mlx(_))
                | MlxDecoderError::Cache(KvCacheError::Operation(ops::OperationError::Mlx(_))) => {
                    true
                }
                MlxDecoderError::Memory(e) | MlxDecoderError::Cache(KvCacheError::Memory(e)) => {
                    matches!(
                        e,
                        memory::MemoryError::Native { .. }
                            | memory::MemoryError::Mlx(_)
                            | memory::MemoryError::StreamSynchronization { .. }
                    )
                }
                _ => false,
            };
            if native {
                self.healthy.set(false);
            }
        }
        result
    }
    pub fn new_cache(
        &self,
        capacity: usize,
        cache_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<MlxKvCache> {
        self.check_healthy()?;
        let result = (|| {
            Ok(MlxKvCache::new(
                &self.owner,
                self.config.cache_spec(capacity)?,
                cache_limit,
                stream,
            )?)
        })();
        self.finish(result, stream)
    }
    /// Full eager projection geometry used by the fixed numerical oracle.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache_limit: u64,
        output_limit: u64,
        work_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<MlxLogits> {
        self.check_healthy()?;
        if tokens.is_empty()
            || tokens.len() > self.config.context
            || tokens.iter().any(|&t| t as usize >= self.config.vocabulary)
        {
            return Err(invalid("invalid Phi input tokens").into());
        }
        let mut cache = self.new_cache(tokens.len(), cache_limit, stream)?;
        self.forward_cached(
            tokens,
            &mut cache,
            false,
            output_limit,
            work_limit,
            stream,
            &mut || false,
        )
    }
    /// Append transactionally. A last-only return still projects all input rows
    /// before slicing, preserving the qualified eager quantized-matmul geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        tokens: &[u32],
        cache: &mut MlxKvCache,
        last_only: bool,
        output_limit: u64,
        work_limit: u64,
        stream: &Stream,
        check: &mut impl FnMut() -> bool,
    ) -> MlxDecoderResult<MlxLogits> {
        self.check_healthy()?;
        let result = self.forward_inner(
            tokens,
            cache,
            last_only,
            output_limit,
            work_limit,
            stream,
            check,
        );
        self.finish(result, stream)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        tokens: &[u32],
        cache: &mut MlxKvCache,
        last_only: bool,
        output_limit: u64,
        work_limit: u64,
        stream: &Stream,
        check: &mut impl FnMut() -> bool,
    ) -> MlxDecoderResult<MlxLogits> {
        cancelled(check)?;
        let c = &self.config;
        if tokens.is_empty() || tokens.iter().any(|&t| t as usize >= c.vocabulary) {
            return Err(invalid("invalid Phi input tokens").into());
        }
        let end = cache
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("Phi position overflow"))?;
        let plan = c.step_payload_plan(tokens.len(), end)?;
        limit("Phi output payload", plan.output_bytes, output_limit)?;
        limit("Phi step workspace", plan.work_bytes, work_limit)?;
        if cache.spec() != c.cache_spec(cache.spec().capacity_tokens)? {
            return Err(invalid("Phi cache geometry mismatch").into());
        }
        let mut append = cache.begin_append(&self.owner, tokens.len(), stream)?;
        let offset = append.start_position();
        let t = tokens.len();
        let kv = c.kv_heads * c.head_dim;
        let mut hidden = self.embedding.embedding_lookup(tokens, stream)?;
        for (index, l) in self.layers.iter().enumerate() {
            cancelled(check)?;
            let normalized = ops::rms_norm(&hidden, &l.input_norm, c.epsilon, stream)?;
            let qkv = l.qkv.apply(&normalized, stream)?;
            let split = |start, width, heads| -> MlxDecoderResult<Array> {
                let x = ops::slice(&qkv, &[0, start], &[t, start + width], stream)?;
                let x = ops::reshape(&x, &[1, t, heads, c.head_dim], stream)?;
                Ok(ops::transpose(&x, &[0, 2, 1, 3], stream)?)
            };
            let q = partial_longrope(
                &split(0, c.hidden, c.query_heads)?,
                c,
                self.regime,
                offset,
                stream,
            )?;
            let k = partial_longrope(
                &split(c.hidden, kv, c.kv_heads)?,
                c,
                self.regime,
                offset,
                stream,
            )?;
            let v = split(c.hidden + kv, kv, c.kv_heads)?;
            let prefix = append.stage_layer(index, &k, &v)?;
            let mask = if t == 1 {
                AttentionMask::Full
            } else {
                AttentionMask::CausalSuffix {
                    prefix_tokens: offset,
                }
            };
            let attention = attention::grouped_query_attention(
                &q,
                prefix.keys(),
                prefix.values(),
                mask,
                stream,
            )?;
            let attention = ops::transpose(&attention, &[0, 2, 1, 3], stream)?;
            let attention = ops::reshape(&attention, &[t, c.hidden], stream)?;
            hidden = ops::residual_add(&hidden, &l.output.apply(&attention, stream)?, stream)?;
            let normalized = ops::rms_norm(&hidden, &l.post_norm, c.epsilon, stream)?;
            let gate_up = l.gate_up.apply(&normalized, stream)?;
            let gate = ops::slice(&gate_up, &[0, 0], &[t, c.intermediate], stream)?;
            let up = ops::slice(
                &gate_up,
                &[0, c.intermediate],
                &[t, c.gate_up_width()],
                stream,
            )?;
            let activated = ops::swiglu(&gate, &up, stream)?;
            hidden = ops::residual_add(&hidden, &l.down.apply(&activated, stream)?, stream)?;
            hidden.eval()?;
            cancelled(check)?;
        }
        let normalized = ops::rms_norm(&hidden, &self.norm, c.epsilon, stream)?;
        let logits = self
            .head
            .as_ref()
            .unwrap_or(&self.embedding)
            .apply(&normalized, stream)?;
        let rows = if last_only { 1 } else { t };
        let logits = if last_only {
            ops::slice(&logits, &[t - 1, 0], &[t, c.vocabulary], stream)?
        } else {
            logits
        };
        cancelled(check)?;
        let values = ops::to_vec_f32(&logits, stream)?;
        if values.len() != rows * c.vocabulary || values.iter().any(|n| !n.is_finite()) {
            return Err(invalid("invalid Phi output logits").into());
        }
        let actual = (values.capacity() as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(2 * t as u64 * c.vocabulary as u64 * 4))
            .ok_or_else(|| invalid("Phi output capacity overflow"))?;
        limit("Phi output backing overlap", actual, output_limit)?;
        cancelled(check)?;
        append.commit()?;
        Ok(MlxLogits {
            values,
            rows,
            vocabulary: c.vocabulary,
        })
    }
}
