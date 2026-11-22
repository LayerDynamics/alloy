//! Worker-local packed Gemma 4 E4B decoder. A step publishes cache state only
//! after the final logits and all retained owner buffers have been evaluated.
use super::{Gemma4Attention, Gemma4TextConfig, invalid, weights};
use crate::{
    decoder::{
        loader::{CheckpointTensor, CheckpointVerification, LoadOptions, LoadedTensorSet},
        mlx_model::{MlxDecoderError, MlxDecoderResult, MlxLinear, MlxLogits, upload_f32},
    },
    kv_cache::MlxModelId,
};
use mlx_ffi::{Array, Stream, memory, ops};
use std::path::Path;

fn limit(operation: &'static str, required: u128, allowed: u64) -> MlxDecoderResult<()> {
    if required > allowed.into() {
        return Err(MlxDecoderError::PayloadLimit {
            operation,
            required_bytes: u64::try_from(required).unwrap_or(u64::MAX),
            limit_bytes: allowed,
        });
    }
    Ok(())
}

struct Upload {
    tensors: LoadedTensorSet,
    buffers: Vec<Array>,
}
impl Upload {
    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> MlxDecoderResult<MlxLinear> {
        let Some(CheckpointTensor::Matrix(m)) = self.tensors.remove(name) else {
            return Err(invalid(name).into());
        };
        let m = MlxLinear::upload(m, rows, cols)?;
        m.visit(&mut |a| self.buffers.push(a.clone()));
        Ok(m)
    }
    fn vector(&mut self, name: &str, width: usize) -> MlxDecoderResult<Array> {
        let Some(CheckpointTensor::Float { values, shape }) = self.tensors.remove(name) else {
            return Err(invalid(name).into());
        };
        if shape != [width] {
            return Err(invalid(name).into());
        }
        let a = upload_f32(values, &shape)?;
        self.buffers.push(a.clone());
        Ok(a)
    }
}
struct OwnerWeights {
    key: MlxLinear,
    value: MlxLinear,
    key_norm: Array,
}
struct Layer {
    input_norm: Array,
    post_attention_norm: Array,
    pre_ffn_norm: Array,
    post_ffn_norm: Array,
    query: MlxLinear,
    query_norm: Array,
    owner: Option<OwnerWeights>,
    out: MlxLinear,
    gate: MlxLinear,
    up: MlxLinear,
    down: MlxLinear,
    input_gate: MlxLinear,
    input_projection: MlxLinear,
    post_input_norm: Array,
    scalar: Array,
}
impl Layer {
    fn upload(u: &mut Upload, c: &Gemma4TextConfig, i: usize) -> MlxDecoderResult<Self> {
        let p = format!("language_model.model.layers.{i}.");
        let name = |n: &str| format!("{p}{n}.weight");
        let geometry = c.layers[i];
        let d = geometry.head_dim;
        let q = c.query_heads * d;
        let kv = c.kv_heads * d;
        let h = c.hidden;
        Ok(Self {
            input_norm: u.vector(&name("input_layernorm"), h)?,
            post_attention_norm: u.vector(&name("post_attention_layernorm"), h)?,
            pre_ffn_norm: u.vector(&name("pre_feedforward_layernorm"), h)?,
            post_ffn_norm: u.vector(&name("post_feedforward_layernorm"), h)?,
            query: u.matrix(&name("self_attn.q_proj"), q, h)?,
            query_norm: u.vector(&name("self_attn.q_norm"), d)?,
            owner: if geometry.kv_owner == i {
                Some(OwnerWeights {
                    key: u.matrix(&name("self_attn.k_proj"), kv, h)?,
                    value: u.matrix(&name("self_attn.v_proj"), kv, h)?,
                    key_norm: u.vector(&name("self_attn.k_norm"), d)?,
                })
            } else {
                None
            },
            out: u.matrix(&name("self_attn.o_proj"), h, q)?,
            gate: u.matrix(&name("mlp.gate_proj"), c.intermediate, h)?,
            up: u.matrix(&name("mlp.up_proj"), c.intermediate, h)?,
            down: u.matrix(&name("mlp.down_proj"), h, c.intermediate)?,
            input_gate: u.matrix(&name("per_layer_input_gate"), c.input_width, h)?,
            input_projection: u.matrix(&name("per_layer_projection"), h, c.input_width)?,
            post_input_norm: u.vector(&name("post_per_layer_input_norm"), h)?,
            scalar: u.vector(&format!("{p}layer_scalar"), 1)?,
        })
    }
}

type KeyValue = (Array, Array);
pub struct Gemma4Cache {
    owner: MlxModelId,
    capacity: usize,
    offset: usize,
    byte_limit: u64,
    slots: Vec<Option<KeyValue>>,
}
impl Gemma4Cache {
    pub fn len(&self) -> usize {
        self.offset
    }
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }
    pub fn buffer_bytes(&self) -> MlxDecoderResult<u64> {
        let arrays: Vec<_> = self
            .slots
            .iter()
            .flatten()
            .flat_map(|(k, v)| [k, v])
            .collect();
        Ok(if arrays.is_empty() {
            0
        } else {
            memory::array_buffer_bytes(&arrays)? as u64
        })
    }
}

pub struct Gemma4Model {
    attention: Gemma4Attention,
    owner: MlxModelId,
    embedding: MlxLinear,
    layer_embedding: MlxLinear,
    layer_projection: MlxLinear,
    projection_norm: Array,
    norm: Array,
    layers: Vec<Layer>,
    weight_bytes: u64,
}
impl Gemma4Model {
    pub fn load_verified(
        path: &Path,
        options: LoadOptions,
        verification: &CheckpointVerification<'_>,
        stream: &Stream,
    ) -> MlxDecoderResult<Self> {
        let max_weights = options.max_weight_bytes as u64;
        let (c, tensors) = weights::load_verified(path, options, verification)?;
        let mut u = Upload {
            tensors,
            buffers: Vec::new(),
        };
        let h = c.hidden;
        let all_inputs = c.layers.len() * c.input_width;
        let embedding = u.matrix("language_model.model.embed_tokens.weight", c.vocabulary, h)?;
        let layer_embedding = u.matrix(
            "language_model.model.embed_tokens_per_layer.weight",
            c.input_vocab,
            all_inputs,
        )?;
        let layer_projection = u.matrix(
            "language_model.model.per_layer_model_projection.weight",
            all_inputs,
            h,
        )?;
        let projection_norm = u.vector(
            "language_model.model.per_layer_projection_norm.weight",
            c.input_width,
        )?;
        let norm = u.vector("language_model.model.norm.weight", h)?;
        let layers = (0..c.layers.len())
            .map(|i| Layer::upload(&mut u, &c, i))
            .collect::<MlxDecoderResult<_>>()?;
        if !u.tensors.is_empty() {
            return Err(invalid("unconsumed trained parameters").into());
        }
        for array in &u.buffers {
            array.eval()?;
        }
        stream.synchronize()?;
        let arrays = u.buffers.iter().collect::<Vec<_>>();
        let weight_bytes = memory::array_buffer_bytes(&arrays)? as u64;
        limit(
            "Gemma 4 native weight buffers",
            weight_bytes.into(),
            max_weights,
        )?;
        Ok(Self {
            attention: Gemma4Attention::new(c, stream)?,
            owner: MlxModelId::new(),
            embedding,
            layer_embedding,
            layer_projection,
            projection_norm,
            norm,
            layers,
            weight_bytes,
        })
    }
    pub fn weight_buffer_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn new_cache(&self, capacity: usize, byte_limit: u64) -> MlxDecoderResult<Gemma4Cache> {
        let c = &self.attention.config;
        if capacity == 0 || capacity > c.context || byte_limit == 0 {
            return Err(invalid("invalid cache capacity or byte limit").into());
        }
        Ok(Gemma4Cache {
            owner: self.owner.clone(),
            capacity,
            offset: 0,
            byte_limit,
            slots: vec![
                None;
                c.layers
                    .iter()
                    .enumerate()
                    .filter(|(i, l)| *i == l.kv_owner)
                    .count()
            ],
        })
    }

    /// Append a prompt chunk or one decode token and return final-position logits.
    pub fn step(
        &self,
        tokens: &[u32],
        cache: &mut Gemma4Cache,
        work_limit: u64,
        stream: &Stream,
        check: &mut dyn FnMut() -> bool,
    ) -> MlxDecoderResult<MlxLogits> {
        let c = &self.attention.config;
        let t = tokens.len();
        let end = cache
            .offset
            .checked_add(t)
            .ok_or_else(|| invalid("position overflow"))?;
        if !cache.owner.same_instance(&self.owner)
            || t == 0
            || end > cache.capacity
            || tokens
                .iter()
                .any(|&id| id as usize >= c.vocabulary.min(c.input_vocab))
        {
            return Err(invalid("foreign cache, invalid token or context capacity").into());
        }
        if check() {
            return Err(MlxDecoderError::Cancelled);
        }
        // During a chunk, all consumer layers need the owner's untrimmed view.
        // Include the old cache, appended owner views and new retained storage.
        let mut staged_bytes = 0u128;
        for layer in c.layers.iter().take(cache.slots.len()) {
            let old = if layer.sliding {
                cache.offset.min(c.window)
            } else {
                cache.offset
            };
            let keep = if layer.sliding {
                end.min(c.window)
            } else {
                end
            };
            staged_bytes += (old as u128 + t as u128 + keep as u128)
                * 2
                * c.kv_heads as u128
                * layer.head_dim as u128
                * 4;
        }
        limit(
            "Gemma 4 KV step overlap",
            cache.buffer_bytes()? as u128 + staged_bytes,
            cache.byte_limit,
        )?;
        let head_dim = c.layers.iter().map(|l| l.head_dim).max().unwrap();
        let per_token = 6 * c.layers.len() * c.input_width
            + 8 * c.hidden
            + 4 * c.intermediate
            + 6 * c.query_heads * head_dim;
        let work = 4
            * t as u128
            * (per_token as u128 + c.query_heads as u128 * end as u128 + 4 * c.vocabulary as u128);
        limit("Gemma 4 step workspace", work, work_limit)?;
        let mut hidden = ops::scale(
            &self.embedding.embedding_lookup(tokens, stream)?,
            (c.hidden as f64).sqrt() as f32,
            stream,
        )?;
        let shape = [t, c.layers.len(), c.input_width];
        let embedded = ops::scale(
            &self.layer_embedding.embedding_lookup(tokens, stream)?,
            (c.input_width as f64).sqrt() as f32,
            stream,
        )?;
        let projected = ops::scale(
            &self.layer_projection.apply(&hidden, stream)?,
            (c.hidden as f64).powf(-0.5) as f32,
            stream,
        )?;
        let projected = ops::rms_norm(
            &ops::reshape(&projected, &shape, stream)?,
            &self.projection_norm,
            c.epsilon,
            stream,
        )?;
        let inputs = ops::scale(
            &ops::residual_add(
                &projected,
                &ops::reshape(&embedded, &shape, stream)?,
                stream,
            )?,
            std::f32::consts::FRAC_1_SQRT_2,
            stream,
        )?;
        let mut views: Vec<KeyValue> = Vec::with_capacity(cache.slots.len());
        for (i, l) in self.layers.iter().enumerate() {
            if check() {
                return Err(MlxDecoderError::Cancelled);
            }
            let a = c.layers[i];
            let d = a.head_dim;
            let x = ops::rms_norm(&hidden, &l.input_norm, c.epsilon, stream)?;
            let heads = |m: &MlxLinear, count: usize| -> MlxDecoderResult<Array> {
                let z = m.apply(&x, stream)?;
                Ok(ops::transpose(
                    &ops::reshape(&z, &[1, t, count, d], stream)?,
                    &[0, 2, 1, 3],
                    stream,
                )?)
            };
            let q = ops::rms_norm(
                &heads(&l.query, c.query_heads)?,
                &l.query_norm,
                c.epsilon,
                stream,
            )?;
            let q = self.attention.rotate_queries(i, &q, cache.offset, stream)?;
            if let Some(owner) = &l.owner {
                let k = ops::rms_norm(
                    &heads(&owner.key, c.kv_heads)?,
                    &owner.key_norm,
                    c.epsilon,
                    stream,
                )?;
                let k = self
                    .attention
                    .rotate_owner_keys(i, &k, cache.offset, stream)?;
                let v =
                    ops::rms_norm_unweighted(&heads(&owner.value, c.kv_heads)?, c.epsilon, stream)?;
                views.push(match &cache.slots[i] {
                    Some((pk, pv)) => (
                        ops::concatenate_axis(pk, &k, 2, stream)?,
                        ops::concatenate_axis(pv, &v, 2, stream)?,
                    ),
                    None => (k, v),
                });
            }
            let (k, v) = &views[a.kv_owner];
            let attended = self.attention.attend(i, &q, k, v, cache.offset, stream)?;
            let z = ops::reshape(
                &ops::transpose(&attended, &[0, 2, 1, 3], stream)?,
                &[t, c.query_heads * d],
                stream,
            )?;
            hidden = ops::residual_add(
                &hidden,
                &ops::rms_norm(
                    &l.out.apply(&z, stream)?,
                    &l.post_attention_norm,
                    c.epsilon,
                    stream,
                )?,
                stream,
            )?;
            let x = ops::rms_norm(&hidden, &l.pre_ffn_norm, c.epsilon, stream)?;
            let up = l.up.apply(&x, stream)?;
            let ff = ops::multiply(
                &ops::gelu_approx(&l.gate.apply(&x, stream)?, stream)?,
                &up,
                stream,
            )?;
            hidden = ops::residual_add(
                &hidden,
                &ops::rms_norm(
                    &l.down.apply(&ff, stream)?,
                    &l.post_ffn_norm,
                    c.epsilon,
                    stream,
                )?,
                stream,
            )?;
            let input = ops::reshape(
                &ops::slice(&inputs, &[0, i, 0], &[t, i + 1, c.input_width], stream)?,
                &[t, c.input_width],
                stream,
            )?;
            let gated = ops::multiply(
                &ops::gelu_approx(&l.input_gate.apply(&hidden, stream)?, stream)?,
                &input,
                stream,
            )?;
            let extra = ops::rms_norm(
                &l.input_projection.apply(&gated, stream)?,
                &l.post_input_norm,
                c.epsilon,
                stream,
            )?;
            hidden = ops::residual_add(&hidden, &extra, stream)?;
            hidden = ops::multiply(
                &hidden,
                &ops::broadcast_to(&l.scalar, &[t, c.hidden], stream)?,
                stream,
            )?;
            hidden.eval()?;
        }
        let normalized = ops::rms_norm(&hidden, &self.norm, c.epsilon, stream)?;
        let logits = ops::logit_softcap(
            &self.embedding.apply(&normalized, stream)?,
            c.softcap,
            stream,
        )?;
        let last = ops::slice(&logits, &[t - 1, 0], &[t, c.vocabulary], stream)?;
        let values = ops::to_vec_f32(&last, stream)?;
        if values.len() != c.vocabulary || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("invalid Gemma 4 trained logits").into());
        }
        let mut next = Vec::with_capacity(views.len());
        for (i, (k, v)) in views.into_iter().enumerate() {
            let layer = c.layers[i];
            let trim = |a: Array| -> MlxDecoderResult<Array> {
                let count = a.shape()[2];
                if !layer.sliding || count <= c.window {
                    a.eval()?;
                    return Ok(a);
                }
                let tail = ops::slice(
                    &a,
                    &[0, 0, count - c.window, 0],
                    &[1, c.kv_heads, count, layer.head_dim],
                    stream,
                )?;
                let tail = ops::contiguous(&tail, stream)?;
                tail.eval()?;
                Ok(tail)
            };
            next.push(Some((trim(k)?, trim(v)?)));
        }
        stream.synchronize()?;
        let arrays: Vec<_> = next.iter().flatten().flat_map(|(k, v)| [k, v]).collect();
        limit(
            "Gemma 4 committed KV buffers",
            memory::array_buffer_bytes(&arrays)? as u128,
            cache.byte_limit,
        )?;
        if check() {
            return Err(MlxDecoderError::Cancelled);
        }
        cache.slots = next;
        cache.offset = end;
        Ok(MlxLogits {
            values,
            rows: 1,
            vocabulary: c.vocabulary,
        })
    }
}
