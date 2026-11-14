//! Native mixed full/recurrent decoder with transactional state publication.
use super::{
    config::{Qwen35Config, invalid},
    weights::{AttentionWeights, Qwen35Weights},
};
use crate::{
    decoder::{
        hybrid::{LinearAttentionState, MlxLinearAttention},
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

fn add(a: u64, b: u64) -> MlxDecoderResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("hybrid payload overflow").into())
}
fn mul(a: usize, b: usize) -> MlxDecoderResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("hybrid dimension overflow").into())
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
struct FullAttention {
    q: MlxLinear,
    k: MlxLinear,
    v: MlxLinear,
    output: MlxLinear,
    q_norm: Array,
    k_norm: Array,
}
enum NativeAttention {
    Linear(MlxLinearAttention),
    Full(FullAttention),
}
struct Layer {
    input_norm: Array,
    attention: NativeAttention,
    post_norm: Array,
    gate: MlxLinear,
    up: MlxLinear,
    down: MlxLinear,
}

/// Explicit output selection and per-step payload allowances. Workspaces and
/// output are separately admitted; neither includes resident model/cache bytes.
#[derive(Clone, Copy, Debug)]
pub struct ForwardOptions {
    pub all_logits: bool,
    pub output_limit: u64,
    pub work_limit: u64,
}

/// One loaded, worker-local dense Qwen3.5 text model.
pub struct MlxQwen35Model {
    config: Qwen35Config,
    owner: MlxModelId,
    embedding: MlxLinear,
    layers: Vec<Layer>,
    norm: Array,
    head: Option<MlxLinear>,
    weight_bytes: u64,
    healthy: Cell<bool>,
}
/// Both kinds of state belong to one model load. There is deliberately no
/// arbitrary truncate operation: changed prefixes must reset and rebuild.
pub struct Qwen35Cache {
    owner: MlxModelId,
    full: MlxKvCache,
    linear: Vec<Option<LinearAttentionState>>,
    recurrent_overlap_limit: u64,
}
impl Qwen35Cache {
    pub fn len(&self) -> usize {
        self.full.len()
    }
    pub fn is_empty(&self) -> bool {
        self.full.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.full.spec().capacity_tokens
    }
    /// Clear both kinds of history together after checking model ownership.
    pub fn reset(&mut self, model: &MlxQwen35Model) -> MlxDecoderResult<()> {
        if self.owner != model.owner {
            return Err(invalid("foreign Qwen3.5 cache").into());
        }
        self.full.reset(&self.owner)?;
        self.linear.iter_mut().for_each(|state| *state = None);
        Ok(())
    }
    /// Evaluated unique backing bytes of current full and recurrent buffers.
    pub fn buffer_bytes(&self) -> MlxDecoderResult<u64> {
        let arrays: Vec<_> = self
            .linear
            .iter()
            .flatten()
            .flat_map(|s| [s.convolution(), s.recurrent()])
            .collect();
        add(
            self.full.buffer_bytes()?,
            memory::array_buffer_bytes(&arrays)? as u64,
        )
    }
}
fn validate_weights(c: &Qwen35Config, w: &Qwen35Weights) -> MlxDecoderResult<u64> {
    c.validate()?;
    if w.layers.len() != c.linear_layers.len() || c.tied_embeddings == w.head.is_some() {
        return Err(invalid("Qwen3.5 weight/config mismatch").into());
    }
    let mut total = 0;
    check_matrix(
        "Qwen3.5 embedding",
        mul(c.vocabulary, c.hidden)?,
        &w.embedding,
        &mut total,
    )?;
    if let Some(h) = &w.head {
        check_matrix("Qwen3.5 head", mul(c.vocabulary, c.hidden)?, h, &mut total)?;
    }
    check_weight("Qwen3.5 norm", c.hidden, &w.norm, &mut total)?;
    for (i, l) in w.layers.iter().enumerate() {
        check_weight("Qwen3.5 input norm", c.hidden, &l.input_norm, &mut total)?;
        check_weight("Qwen3.5 post norm", c.hidden, &l.post_norm, &mut total)?;
        for (name, m, rows, cols) in [
            ("Qwen3.5 gate", &l.gate, c.intermediate, c.hidden),
            ("Qwen3.5 up", &l.up, c.intermediate, c.hidden),
            ("Qwen3.5 down", &l.down, c.hidden, c.intermediate),
        ] {
            check_matrix(name, mul(rows, cols)?, m, &mut total)?;
        }
        match (&l.attention, c.linear_layers[i]) {
            (AttentionWeights::Linear(w), true) => {
                total = add(total, w.validated_bytes(c.linear)?)?
            }
            (AttentionWeights::Full(w), false) => {
                for (name, m, rows, cols) in [
                    (
                        "Qwen3.5 query gate",
                        &w.q,
                        2 * c.query_heads * c.head_dim,
                        c.hidden,
                    ),
                    ("Qwen3.5 key", &w.k, c.kv_heads * c.head_dim, c.hidden),
                    ("Qwen3.5 value", &w.v, c.kv_heads * c.head_dim, c.hidden),
                    (
                        "Qwen3.5 attention out",
                        &w.output,
                        c.hidden,
                        c.query_heads * c.head_dim,
                    ),
                ] {
                    check_matrix(name, mul(rows, cols)?, m, &mut total)?;
                }
                check_weight("Qwen3.5 query norm", c.head_dim, &w.q_norm, &mut total)?;
                check_weight("Qwen3.5 key norm", c.head_dim, &w.k_norm, &mut total)?;
            }
            _ => return Err(invalid("Qwen3.5 layer type disagrees with weights").into()),
        }
    }
    Ok(total)
}
impl MlxQwen35Model {
    pub fn from_weights(
        config: Qwen35Config,
        weights: Qwen35Weights,
        weight_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<Self> {
        let payload = validate_weights(&config, &weights)?;
        limit("Qwen3.5 weight payload", payload, weight_limit)?;
        let c = &config;
        let embedding = MlxLinear::upload(weights.embedding, c.vocabulary, c.hidden)?;
        let norm = upload_f32(weights.norm, &[c.hidden])?;
        let head = weights
            .head
            .map(|w| MlxLinear::upload(w, c.vocabulary, c.hidden))
            .transpose()?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(weights.layers.len())
            .map_err(|_| MlxDecoderError::AllocationFailed("Qwen3.5 layers"))?;
        for l in weights.layers {
            let attention = match l.attention {
                AttentionWeights::Linear(w) => NativeAttention::Linear(
                    MlxLinearAttention::from_weights(c.linear, w, weight_limit, stream)?,
                ),
                AttentionWeights::Full(w) => NativeAttention::Full(FullAttention {
                    q: MlxLinear::upload(w.q, 2 * c.query_heads * c.head_dim, c.hidden)?,
                    k: MlxLinear::upload(w.k, c.kv_heads * c.head_dim, c.hidden)?,
                    v: MlxLinear::upload(w.v, c.kv_heads * c.head_dim, c.hidden)?,
                    output: MlxLinear::upload(w.output, c.hidden, c.query_heads * c.head_dim)?,
                    q_norm: upload_f32(w.q_norm, &[c.head_dim])?,
                    k_norm: upload_f32(w.k_norm, &[c.head_dim])?,
                }),
            };
            layers.push(Layer {
                input_norm: upload_f32(l.input_norm, &[c.hidden])?,
                attention,
                post_norm: upload_f32(l.post_norm, &[c.hidden])?,
                gate: MlxLinear::upload(l.gate, c.intermediate, c.hidden)?,
                up: MlxLinear::upload(l.up, c.intermediate, c.hidden)?,
                down: MlxLinear::upload(l.down, c.hidden, c.intermediate)?,
            });
        }
        let model = Self {
            config,
            owner: MlxModelId::new(),
            embedding,
            layers,
            norm,
            head,
            weight_bytes: payload,
            healthy: Cell::new(true),
        };
        let mut result = Ok(());
        model.visit(&mut |a| {
            if result.is_ok() {
                result = a.eval();
            }
        });
        result?;
        stream.synchronize()?;
        limit(
            "Qwen3.5 weight backing buffers",
            model.weight_buffer_bytes()?,
            weight_limit,
        )?;
        Ok(model)
    }
    fn visit<'a>(&'a self, v: &mut impl FnMut(&'a Array)) {
        self.embedding.visit(v);
        v(&self.norm);
        if let Some(h) = &self.head {
            h.visit(v);
        }
        for l in &self.layers {
            v(&l.input_norm);
            v(&l.post_norm);
            l.gate.visit(v);
            l.up.visit(v);
            l.down.visit(v);
            match &l.attention {
                NativeAttention::Linear(a) => a.visit(v),
                NativeAttention::Full(a) => {
                    a.q.visit(v);
                    a.k.visit(v);
                    a.v.visit(v);
                    a.output.visit(v);
                    v(&a.q_norm);
                    v(&a.k_norm);
                }
            }
        }
    }
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }
    pub fn weight_payload_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn is_healthy(&self) -> bool {
        self.healthy.get()
    }
    pub fn weight_buffer_bytes(&self) -> MlxDecoderResult<u64> {
        let mut arrays = Vec::new();
        arrays
            .try_reserve_exact(self.layers.len() * 32 + 7)
            .map_err(|_| MlxDecoderError::AllocationFailed("Qwen3.5 weight measurement"))?;
        self.visit(&mut |a| arrays.push(a));
        Ok(memory::array_buffer_bytes(&arrays)? as u64)
    }
    /// Reserve full KV update overlap and both old/new recurrent payloads.
    /// Allocator caches, model weights and inference workspaces are separate.
    pub fn new_cache(
        &self,
        capacity: usize,
        payload_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<Qwen35Cache> {
        if !self.healthy.get() {
            return Err(MlxDecoderError::Invalidated);
        }
        let plan = self.config.cache_payload_plan(capacity)?;
        let spec = plan.full;
        let full_minimum = plan.full_overlap_bytes;
        let minimum = plan.total_overlap_bytes;
        limit("Qwen3.5 cache overlap", minimum, payload_limit)?;
        // Logical payload excludes allocator padding and reusable backing
        // buffers. Split the caller's remaining allowance between components;
        // each independently enforces its measured old/new buffer budget.
        let extra = payload_limit - minimum;
        let full_budget = add(full_minimum, extra / 2)?;
        let recurrent_budget = payload_limit - full_budget;
        let full = MlxKvCache::new(&self.owner, spec, full_budget, stream)?;
        let mut linear = Vec::new();
        linear
            .try_reserve_exact(self.layers.len())
            .map_err(|_| MlxDecoderError::AllocationFailed("Qwen3.5 recurrent slots"))?;
        linear.resize_with(self.layers.len(), || None);
        Ok(Qwen35Cache {
            owner: self.owner.clone(),
            full,
            linear,
            recurrent_overlap_limit: recurrent_budget,
        })
    }
    /// Execute a nonempty token chunk. `all_logits=false` returns only its last
    /// position. State commits after finite evaluated logits and cancellation.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35Cache,
        options: ForwardOptions,
        stream: &Stream,
        cancelled: &dyn Fn() -> bool,
    ) -> MlxDecoderResult<MlxLogits> {
        if !self.healthy.get() {
            return Err(MlxDecoderError::Invalidated);
        }
        let result = self.forward_impl(
            tokens,
            cache,
            options.all_logits,
            options.output_limit,
            options.work_limit,
            stream,
            cancelled,
        );
        // Complete outstanding work even on cancellation before releasing staged
        // arrays. Native failures invalidate this load; ordinary refusal does not.
        let completion = stream.synchronize();
        if completion.is_err()
            || matches!(
                &result,
                Err(MlxDecoderError::Mlx(_)
                    | MlxDecoderError::Operation(_)
                    | MlxDecoderError::Memory(_)
                    | MlxDecoderError::Cache(
                        KvCacheError::Mlx(_)
                            | KvCacheError::Operation(_)
                            | KvCacheError::Memory(_)
                            | KvCacheError::Invalidated
                    ))
            )
        {
            self.healthy.set(false);
        }
        completion?;
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35Cache,
        all_logits: bool,
        output_limit: u64,
        work_limit: u64,
        stream: &Stream,
        cancelled: &dyn Fn() -> bool,
    ) -> MlxDecoderResult<MlxLogits> {
        if cache.owner != self.owner {
            return Err(invalid("foreign Qwen3.5 cache").into());
        }
        let c = &self.config;
        let t = tokens.len();
        if t == 0 || tokens.iter().any(|&id| id as usize >= c.vocabulary) {
            return Err(invalid("invalid Qwen3.5 input tokens").into());
        }
        let end = cache
            .len()
            .checked_add(t)
            .ok_or_else(|| invalid("Qwen3.5 position overflow"))?;
        if end > cache.capacity() {
            return Err(invalid("Qwen3.5 context capacity exceeded").into());
        }
        if cache
            .linear
            .iter()
            .flatten()
            .any(|state| state.len() != cache.len())
            || (!cache.is_empty()
                && cache.linear.iter().flatten().count()
                    != c.linear_layers.iter().filter(|&&v| v).count())
        {
            return Err(invalid("Qwen3.5 cache components disagree").into());
        }
        let plan = c.step_payload_plan(t, end, all_logits)?;
        let rows = if all_logits { t } else { 1 };
        limit("Qwen3.5 logits", plan.output_bytes, output_limit)?;
        limit(
            "Qwen3.5 layer workspace payload",
            plan.work_bytes,
            work_limit,
        )?;
        if cancelled() {
            return Err(MlxDecoderError::Cancelled);
        }
        let mut x = self.embedding.embedding_lookup(tokens, stream)?;
        let capacity = cache.capacity();
        let mut append = cache.full.begin_append(&self.owner, t, stream)?;
        let offset = append.start_position();
        let mut next = Vec::new();
        next.try_reserve_exact(self.layers.len())
            .map_err(|_| MlxDecoderError::AllocationFailed("Qwen3.5 staged recurrent slots"))?;
        let mut full_index = 0;
        for (index, l) in self.layers.iter().enumerate() {
            if cancelled() {
                return Err(MlxDecoderError::Cancelled);
            }
            let normalized = ops::rms_norm(&x, &l.input_norm, c.epsilon, stream)?;
            let delta = match &l.attention {
                NativeAttention::Linear(a) => {
                    let input = ops::reshape(&normalized, &[1, t, c.hidden], stream)?;
                    let (out, state) = a.forward(
                        &input,
                        cache.linear[index].as_ref(),
                        capacity,
                        work_limit,
                        stream,
                        cancelled,
                    )?;
                    next.push(Some(state));
                    ops::reshape(&out, &[t, c.hidden], stream)?
                }
                NativeAttention::Full(a) => {
                    next.push(None);
                    let qgate = ops::reshape(
                        &a.q.apply(&normalized, stream)?,
                        &[1, t, c.query_heads, 2 * c.head_dim],
                        stream,
                    )?;
                    let q = ops::slice(
                        &qgate,
                        &[0, 0, 0, 0],
                        &[1, t, c.query_heads, c.head_dim],
                        stream,
                    )?;
                    let gate = ops::slice(
                        &qgate,
                        &[0, 0, 0, c.head_dim],
                        &[1, t, c.query_heads, 2 * c.head_dim],
                        stream,
                    )?;
                    let q = ops::transpose(
                        &ops::rms_norm(&q, &a.q_norm, c.epsilon, stream)?,
                        &[0, 2, 1, 3],
                        stream,
                    )?;
                    let k = ops::reshape(
                        &a.k.apply(&normalized, stream)?,
                        &[1, t, c.kv_heads, c.head_dim],
                        stream,
                    )?;
                    let k = ops::transpose(
                        &ops::rms_norm(&k, &a.k_norm, c.epsilon, stream)?,
                        &[0, 2, 1, 3],
                        stream,
                    )?;
                    let v = ops::transpose(
                        &ops::reshape(
                            &a.v.apply(&normalized, stream)?,
                            &[1, t, c.kv_heads, c.head_dim],
                            stream,
                        )?,
                        &[0, 2, 1, 3],
                        stream,
                    )?;
                    let q = partial_rope(&q, c.rotary_dim, c.rope_theta, offset, stream)?;
                    let k = partial_rope(&k, c.rotary_dim, c.rope_theta, offset, stream)?;
                    let prefix = append.stage_layer(full_index, &k, &v)?;
                    full_index += 1;
                    let attended = attention::grouped_query_attention(
                        &q,
                        prefix.keys(),
                        prefix.values(),
                        AttentionMask::CausalSuffix {
                            prefix_tokens: offset,
                        },
                        stream,
                    )?;
                    let attended = ops::reshape(
                        &ops::transpose(&attended, &[0, 2, 1, 3], stream)?,
                        &[t, c.query_heads * c.head_dim],
                        stream,
                    )?;
                    let gate = ops::sigmoid(
                        &ops::reshape(&gate, &[t, c.query_heads * c.head_dim], stream)?,
                        stream,
                    )?;
                    a.output
                        .apply(&ops::multiply(&attended, &gate, stream)?, stream)?
                }
            };
            x = ops::residual_add(&x, &delta, stream)?;
            let post = ops::rms_norm(&x, &l.post_norm, c.epsilon, stream)?;
            let mlp = ops::swiglu(
                &l.gate.apply(&post, stream)?,
                &l.up.apply(&post, stream)?,
                stream,
            )?;
            x = ops::residual_add(&x, &l.down.apply(&mlp, stream)?, stream)?;
            x.eval()?;
        }
        let normalized = ops::rms_norm(&x, &self.norm, c.epsilon, stream)?;
        let selected = if all_logits {
            normalized
        } else {
            ops::slice(&normalized, &[t - 1, 0], &[t, c.hidden], stream)?
        };
        let logits = self
            .head
            .as_ref()
            .unwrap_or(&self.embedding)
            .apply(&selected, stream)?;
        let values = ops::to_vec_f32(&logits, stream)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite Qwen3.5 logits").into());
        }
        let arrays: Vec<_> = next
            .iter()
            .flatten()
            .flat_map(|s| [s.convolution(), s.recurrent()])
            .collect();
        let measured = memory::array_buffer_bytes(&arrays)? as u64;
        limit(
            "Qwen3.5 recurrent backing overlap",
            add(measured, measured)?,
            cache.recurrent_overlap_limit,
        )?;
        if cancelled() {
            return Err(MlxDecoderError::Cancelled);
        }
        append.commit()?;
        cache.linear = next;
        Ok(MlxLogits {
            values,
            rows,
            vocabulary: c.vocabulary,
        })
    }
}
fn partial_rope(
    input: &Array,
    rotary: usize,
    theta: f32,
    offset: usize,
    stream: &Stream,
) -> MlxDecoderResult<Array> {
    let shape = input.shape();
    if rotary == shape[3] {
        return Ok(attention::rope(input, theta, offset, stream)?);
    }
    let mut leading = shape.clone();
    leading[3] = rotary;
    let rotated = attention::rope(
        &ops::slice(input, &[0, 0, 0, 0], &leading, stream)?,
        theta,
        offset,
        stream,
    )?;
    let tail = ops::slice(input, &[0, 0, 0, rotary], &shape, stream)?;
    Ok(ops::concatenate_axis(&rotated, &tail, 3, stream)?)
}
