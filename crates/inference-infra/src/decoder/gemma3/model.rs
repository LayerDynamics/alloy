//! Native Gemma 3 transformer with transactional bounded sliding/full KV.
use super::{Gemma3Config, config::invalid, weights::Gemma3Weights};
use crate::{
    decoder::{
        fixture::MatrixWeight,
        mlx_model::{
            MlxDecoderError, MlxDecoderResult, MlxLinear, MlxLogits, check_matrix, check_weight,
            upload_f32,
        },
    },
    kv_cache::{KvCacheError, MlxModelId},
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
fn upload_norm(mut values: Vec<f32>, width: usize) -> MlxDecoderResult<Array> {
    // Gemma stores the delta from unity. This is a one-time weight transform,
    // not a host implementation of runtime normalization.
    for v in &mut values {
        *v += 1.0;
    }
    upload_f32(values, &[width])
}
struct Layer {
    input_norm: Array,
    post_attention_norm: Array,
    pre_ffn_norm: Array,
    post_ffn_norm: Array,
    q_norm: Array,
    k_norm: Array,
    q: MlxLinear,
    k: MlxLinear,
    v: MlxLinear,
    output: MlxLinear,
    gate: MlxLinear,
    up: MlxLinear,
    down: MlxLinear,
}
impl Layer {
    fn visit<'a>(&'a self, f: &mut impl FnMut(&'a Array)) {
        for n in [
            &self.input_norm,
            &self.post_attention_norm,
            &self.pre_ffn_norm,
            &self.post_ffn_norm,
            &self.q_norm,
            &self.k_norm,
        ] {
            f(n);
        }
        for m in [
            &self.q,
            &self.k,
            &self.v,
            &self.output,
            &self.gate,
            &self.up,
            &self.down,
        ] {
            m.visit(f);
        }
    }
}
pub struct MlxGemma3Model {
    config: Gemma3Config,
    owner: MlxModelId,
    embedding: MlxLinear,
    layers: Vec<Layer>,
    norm: Array,
    head: Option<MlxLinear>,
    payload: u64,
    healthy: Cell<bool>,
}
/// Worker-local KV. Local layers retain a window; global layers retain history.
/// Offset is absolute and is never derived from an evicted layer's length.
pub struct Gemma3Cache {
    owner: MlxModelId,
    capacity: usize,
    offset: usize,
    layers: Vec<Vec<KvPage>>,
}

#[derive(Clone)]
struct KvPage {
    start: usize,
    end: usize,
    keys: Array,
    values: Array,
}

// Align pages to absolute positions so advancing the sliding window shares
// every unchanged interior page. The old version remains valid until commit.
fn stage_pages(
    old: &[KvPage],
    keys: &Array,
    values: &Array,
    source_start: usize,
    mut start: usize,
    end: usize,
    stream: &Stream,
) -> MlxDecoderResult<Vec<KvPage>> {
    const PAGE_TOKENS: usize = 128;
    let mut pages = Vec::new();
    while start < end {
        let stop = start
            .saturating_add(PAGE_TOKENS - start % PAGE_TOKENS)
            .min(end);
        if let Ok(i) = old.binary_search_by_key(&start, |p| p.start)
            && old[i].end == stop
        {
            pages.push(old[i].clone());
        } else {
            let keep = |array: &Array| -> MlxDecoderResult<Array> {
                let view =
                    ops::cache_slice(array, start - source_start, stop - source_start, stream)?;
                let page = ops::contiguous(&view, stream)?;
                page.eval()?;
                Ok(page)
            };
            pages.push(KvPage {
                start,
                end: stop,
                keys: keep(keys)?,
                values: keep(values)?,
            });
        }
        start = stop;
    }
    Ok(pages)
}
impl Gemma3Cache {
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.offset
    }
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }
    pub fn retained_tokens(&self, layer: usize) -> Option<usize> {
        self.layers
            .get(layer)
            .map(|pages| pages.first().map_or(0, |p| self.offset - p.start))
    }
    pub fn buffer_bytes(&self) -> MlxDecoderResult<u64> {
        let arrays: Vec<&Array> = self
            .layers
            .iter()
            .flatten()
            .flat_map(|page| [&page.keys, &page.values])
            .collect();
        Ok(memory::array_buffer_bytes(&arrays)? as u64)
    }
    pub fn reset(&mut self, owner: &MlxModelId) -> MlxDecoderResult<()> {
        if !self.owner.same_instance(owner) {
            return Err(invalid("foreign Gemma 3 cache owner").into());
        }
        for layer in &mut self.layers {
            layer.clear();
        }
        self.offset = 0;
        Ok(())
    }
}
fn validate_weights(c: &Gemma3Config, w: &Gemma3Weights) -> MlxDecoderResult<u64> {
    c.validate()?;
    if w.layers.len() != c.layers || c.tied_embeddings == w.head.is_some() {
        return Err(invalid("Gemma 3 weights/config mismatch").into());
    }
    let mut total = 0;
    let matrix =
        |m: &MatrixWeight, rows: usize, cols: usize, total: &mut u64| -> MlxDecoderResult<()> {
            if m.bits() != c.group_size.map(|_| c.quantization_bits)
                || m.affine4_group_size() != c.group_size
            {
                return Err(invalid("Gemma 3 matrix representation mismatch").into());
            }
            Ok(check_matrix(
                "Gemma 3 matrix",
                rows.checked_mul(cols)
                    .ok_or_else(|| invalid("Gemma 3 matrix overflow"))?,
                m,
                total,
            )?)
        };
    matrix(&w.embedding, c.vocabulary, c.hidden, &mut total)?;
    if let Some(h) = &w.head {
        matrix(h, c.vocabulary, c.hidden, &mut total)?;
    }
    check_weight("Gemma 3 final norm", c.hidden, &w.norm, &mut total)?;
    for l in &w.layers {
        for n in [
            &l.input_norm,
            &l.post_attention_norm,
            &l.pre_ffn_norm,
            &l.post_ffn_norm,
        ] {
            check_weight("Gemma 3 block norm", c.hidden, n, &mut total)?;
        }
        for n in [&l.q_norm, &l.k_norm] {
            check_weight("Gemma 3 head norm", c.head_dim, n, &mut total)?;
        }
        for (m, rows, cols) in [
            (&l.q, c.q_width(), c.hidden),
            (&l.k, c.kv_width(), c.hidden),
            (&l.v, c.kv_width(), c.hidden),
            (&l.output, c.hidden, c.q_width()),
            (&l.gate, c.intermediate, c.hidden),
            (&l.up, c.intermediate, c.hidden),
            (&l.down, c.hidden, c.intermediate),
        ] {
            matrix(m, rows, cols, &mut total)?;
        }
    }
    if total != c.weight_payload_plan()?.payload_bytes {
        return Err(invalid("Gemma 3 payload plan mismatch").into());
    }
    Ok(total)
}
impl MlxGemma3Model {
    pub fn from_weights(
        config: Gemma3Config,
        weights: Gemma3Weights,
        weight_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<Self> {
        let result = (|| {
            let payload = validate_weights(&config, &weights)?;
            limit("Gemma 3 weight payload", payload, weight_limit)?;
            let c = &config;
            let embedding = MlxLinear::upload(weights.embedding, c.vocabulary, c.hidden)?;
            let norm = upload_norm(weights.norm, c.hidden)?;
            let head = weights
                .head
                .map(|h| MlxLinear::upload(h, c.vocabulary, c.hidden))
                .transpose()?;
            let mut layers = Vec::new();
            layers
                .try_reserve_exact(c.layers)
                .map_err(|_| MlxDecoderError::AllocationFailed("Gemma 3 layer metadata"))?;
            for l in weights.layers {
                layers.push(Layer {
                    input_norm: upload_norm(l.input_norm, c.hidden)?,
                    post_attention_norm: upload_norm(l.post_attention_norm, c.hidden)?,
                    pre_ffn_norm: upload_norm(l.pre_ffn_norm, c.hidden)?,
                    post_ffn_norm: upload_norm(l.post_ffn_norm, c.hidden)?,
                    q_norm: upload_norm(l.q_norm, c.head_dim)?,
                    k_norm: upload_norm(l.k_norm, c.head_dim)?,
                    q: MlxLinear::upload(l.q, c.q_width(), c.hidden)?,
                    k: MlxLinear::upload(l.k, c.kv_width(), c.hidden)?,
                    v: MlxLinear::upload(l.v, c.kv_width(), c.hidden)?,
                    output: MlxLinear::upload(l.output, c.hidden, c.q_width())?,
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
                "Gemma 3 native weight buffers",
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
    pub fn config(&self) -> &Gemma3Config {
        &self.config
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
            .try_reserve_exact(27 * self.layers.len() + 7)
            .map_err(|_| MlxDecoderError::AllocationFailed("Gemma 3 weight buffer metadata"))?;
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
    pub fn new_cache(&self, capacity: usize, cache_limit: u64) -> MlxDecoderResult<Gemma3Cache> {
        self.check_healthy()?;
        limit(
            "Gemma 3 retained KV",
            self.config.cache_payload(capacity)?,
            cache_limit,
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.config.layers)
            .map_err(|_| MlxDecoderError::AllocationFailed("Gemma 3 cache metadata"))?;
        layers.resize_with(self.config.layers, Vec::new);
        Ok(Gemma3Cache {
            owner: self.owner.clone(),
            capacity,
            offset: 0,
            layers,
        })
    }
    pub fn forward(
        &self,
        tokens: &[u32],
        cache_limit: u64,
        output_limit: u64,
        work_limit: u64,
        stream: &Stream,
    ) -> MlxDecoderResult<MlxLogits> {
        let mut cache = self.new_cache(tokens.len(), cache_limit)?;
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
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        tokens: &[u32],
        cache: &mut Gemma3Cache,
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
        cache: &mut Gemma3Cache,
        last_only: bool,
        output_limit: u64,
        work_limit: u64,
        stream: &Stream,
        check: &mut impl FnMut() -> bool,
    ) -> MlxDecoderResult<MlxLogits> {
        cancelled(check)?;
        let c = &self.config;
        let t = tokens.len();
        if !cache.owner.same_instance(&self.owner) || cache.layers.len() != c.layers {
            return Err(invalid("foreign Gemma 3 cache owner or geometry").into());
        }
        if t == 0 || tokens.iter().any(|&v| v as usize >= c.vocabulary) {
            return Err(invalid("invalid Gemma 3 input tokens").into());
        }
        let end = cache
            .offset
            .checked_add(t)
            .filter(|&v| v <= cache.capacity)
            .ok_or_else(|| invalid("Gemma 3 cache capacity exceeded"))?;
        let plan = c.step_payload_plan(t, end)?;
        limit("Gemma 3 output payload", plan.output_bytes, output_limit)?;
        limit("Gemma 3 step workspace", plan.work_bytes, work_limit)?;
        let mut staged = Vec::new();
        staged
            .try_reserve_exact(c.layers)
            .map_err(|_| MlxDecoderError::AllocationFailed("Gemma 3 staged KV metadata"))?;
        let mut hidden = ops::scale(
            &self.embedding.embedding_lookup(tokens, stream)?,
            c.embedding_multiplier(),
            stream,
        )?;
        for (i, l) in self.layers.iter().enumerate() {
            cancelled(check)?;
            let normalized = ops::rms_norm(&hidden, &l.input_norm, c.epsilon, stream)?;
            let heads = |m: &MlxLinear, count| -> MlxDecoderResult<Array> {
                let a = m.apply(&normalized, stream)?;
                let a = ops::reshape(&a, &[1, t, count, c.head_dim], stream)?;
                Ok(ops::transpose(&a, &[0, 2, 1, 3], stream)?)
            };
            let q = ops::rms_norm(&heads(&l.q, c.query_heads)?, &l.q_norm, c.epsilon, stream)?;
            let k = ops::rms_norm(&heads(&l.k, c.kv_heads)?, &l.k_norm, c.epsilon, stream)?;
            let v = heads(&l.v, c.kv_heads)?;
            let rope = |x: &Array| -> MlxDecoderResult<Array> {
                if c.sliding[i] {
                    return Ok(attention::rope(x, c.local_theta, cache.offset, stream)?);
                }
                Ok(attention::rope_scaled(
                    x,
                    c.theta,
                    1.0 / c.rope_factor,
                    cache.offset,
                    stream,
                )?)
            };
            let q = rope(&q)?;
            let k = rope(&k)?;
            let pages = &cache.layers[i];
            let source_start = pages.first().map_or(cache.offset, |p| p.start);
            let prefix = cache.offset - source_start;
            let join = |select: fn(&KvPage) -> &Array, suffix: Array| {
                let inputs: Vec<_> = pages.iter().map(select).cloned().chain([suffix]).collect();
                ops::concatenate_arrays(&inputs, 2, stream)
            };
            let k = join(|page| &page.keys, k)?;
            let v = join(|page| &page.values, v)?;
            let mask = if c.sliding[i] {
                AttentionMask::SlidingCausalSuffix {
                    prefix_tokens: prefix,
                    window_tokens: c.window,
                }
            } else {
                AttentionMask::CausalSuffix {
                    prefix_tokens: prefix,
                }
            };
            let attended = attention::grouped_query_attention_scaled(
                &q,
                &k,
                &v,
                f64::from(c.query_scalar).powf(-0.5) as f32,
                mask,
                stream,
            )?;
            let attended = ops::transpose(&attended, &[0, 2, 1, 3], stream)?;
            let attended = ops::reshape(&attended, &[t, c.q_width()], stream)?;
            let residual = l.output.apply(&attended, stream)?;
            let residual = ops::rms_norm(&residual, &l.post_attention_norm, c.epsilon, stream)?;
            hidden = ops::residual_add(&hidden, &residual, stream)?;
            let normalized = ops::rms_norm(&hidden, &l.pre_ffn_norm, c.epsilon, stream)?;
            let gate = ops::gelu_approx(&l.gate.apply(&normalized, stream)?, stream)?;
            let activated = ops::multiply(&gate, &l.up.apply(&normalized, stream)?, stream)?;
            let residual = l.down.apply(&activated, stream)?;
            let residual = ops::rms_norm(&residual, &l.post_ffn_norm, c.epsilon, stream)?;
            hidden = ops::residual_add(&hidden, &residual, stream)?;
            hidden.eval()?;
            // Materialize trimmed contiguous buffers before dropping the full
            // concatenation, so eviction releases its backing allocation.
            let retain = if c.sliding[i] {
                (prefix + t).min(c.window)
            } else {
                prefix + t
            };
            staged.push(stage_pages(
                pages,
                &k,
                &v,
                source_start,
                end - retain,
                end,
                stream,
            )?);
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
        let values = ops::to_vec_f32(&logits, stream)?;
        if values.len() != rows * c.vocabulary || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("invalid Gemma 3 output logits").into());
        }
        let actual = (values.capacity() as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(2 * t as u64 * c.vocabulary as u64 * 4))
            .ok_or_else(|| invalid("Gemma 3 output capacity overflow"))?;
        limit("Gemma 3 output backing overlap", actual, output_limit)?;
        // Synchronization and cancellation precede publication of every layer.
        stream.synchronize()?;
        cancelled(check)?;
        cache.layers = staged;
        cache.offset = end;
        Ok(MlxLogits {
            values,
            rows,
            vocabulary: c.vocabulary,
        })
    }
}
