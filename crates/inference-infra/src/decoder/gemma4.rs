//! Gemma 4 E4B text attention. Shared layers consume the rotated keys and
//! normalized values of the most recent independent layer of the same type.
use super::{mlx_model::MlxDecoderResult, model::ModelError};
use mlx_ffi::{
    Array, Stream,
    attention::{self, AttentionMask},
    ops,
};
use serde_json::Value;

pub mod weights;

fn invalid(message: &str) -> ModelError {
    ModelError::CheckpointValidation(format!("Gemma 4 text attention: {message}"))
}

fn dimension(config: &Value, name: &str, maximum: usize) -> Result<usize, ModelError> {
    config[name]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && n <= maximum)
        .ok_or_else(|| invalid(name))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAttention {
    pub kv_owner: usize,
    pub head_dim: usize,
    pub sliding: bool,
}

/// Dense text geometry parsed before any weight allocation.
#[derive(Debug, Clone)]
pub struct Gemma4TextConfig {
    layers: Vec<LayerAttention>,
    query_heads: usize,
    kv_heads: usize,
    context: usize,
    window: usize,
    local_theta: f32,
    global_theta: f32,
    global_dim: usize,
    rotated_pairs: usize,
    hidden: usize,
    intermediate: usize,
    vocabulary: usize,
    input_width: usize,
    input_vocab: usize,
}

impl Gemma4TextConfig {
    pub fn from_json(config: &Value) -> Result<Self, ModelError> {
        let c = &config["text_config"];
        if config["model_type"] != "gemma4"
            || c["model_type"] != "gemma4_text"
            || config["architectures"] != serde_json::json!(["Gemma4ForConditionalGeneration"])
            || c["attention_k_eq_v"] != false
            || c["attention_bias"] != false
            || !c["num_global_key_value_heads"].is_null()
            || !c["use_bidirectional_attention"].is_null()
            || c.get("rope_traditional").is_some_and(|v| v != false)
        {
            return Err(invalid("unsupported attention architecture"));
        }
        let count = dimension(c, "num_hidden_layers", 512)?;
        let shared = c["num_kv_shared_layers"]
            .as_u64()
            .filter(|&n| n < count as u64)
            .ok_or_else(|| invalid("num_kv_shared_layers"))? as usize;
        let types = c["layer_types"]
            .as_array()
            .filter(|v| v.len() == count)
            .ok_or_else(|| invalid("layer types must describe every layer"))?;
        let local_dim = dimension(c, "head_dim", 4096)?;
        let global_dim = dimension(c, "global_head_dim", 4096)?;
        let query_heads = dimension(c, "num_attention_heads", 1024)?;
        let kv_heads = dimension(c, "num_key_value_heads", query_heads)?;
        let context = dimension(c, "max_position_embeddings", i32::MAX as usize)?;
        let window = dimension(c, "sliding_window", context)?;
        if !query_heads.is_multiple_of(kv_heads)
            || !local_dim.is_multiple_of(2)
            || !global_dim.is_multiple_of(2)
        {
            return Err(invalid("invalid head geometry"));
        }
        let mut layers = Vec::with_capacity(count);
        let mut owners = [None; 2];
        for (i, kind) in types.iter().enumerate() {
            let sliding = match kind.as_str() {
                Some("sliding_attention") => true,
                Some("full_attention") => false,
                _ => return Err(invalid("unrecognized layer type")),
            };
            let owner = &mut owners[usize::from(sliding)];
            if i < count - shared {
                *owner = Some(i);
            }
            layers.push(LayerAttention {
                sliding,
                head_dim: if sliding { local_dim } else { global_dim },
                kv_owner: owner.ok_or_else(|| invalid("shared layer has no matching KV owner"))?,
            });
        }
        let local = &c["rope_parameters"]["sliding_attention"];
        let global = &c["rope_parameters"]["full_attention"];
        let theta = |v: &Value| {
            v["rope_theta"]
                .as_f64()
                .map(|n| n as f32)
                .filter(|n| n.is_finite() && *n > 0.0)
                .ok_or_else(|| invalid("rope_theta"))
        };
        if local["rope_type"] != "default"
            || global["rope_type"] != "proportional"
            || local.as_object().is_none_or(|o| {
                o.keys()
                    .any(|k| !matches!(k.as_str(), "rope_type" | "rope_theta"))
            })
            || global.as_object().is_none_or(|o| {
                o.keys().any(|k| {
                    !matches!(
                        k.as_str(),
                        "rope_type" | "rope_theta" | "partial_rotary_factor"
                    )
                })
            })
        {
            return Err(invalid("unsupported rotary parameters"));
        }
        let local_theta = theta(local)?;
        let global_theta = theta(global)?;
        let width = global["partial_rotary_factor"].as_f64().unwrap_or(0.0) * global_dim as f64;
        if !(2.0..=global_dim as f64).contains(&width)
            || width.fract() != 0.0
            || !(width as usize).is_multiple_of(2)
        {
            return Err(invalid("invalid rotated width"));
        }
        if c["hidden_activation"] != "gelu_pytorch_tanh"
            || c["enable_moe_block"] != false
            || c["use_double_wide_mlp"] != false
            || c["tie_word_embeddings"] != true
            || config.get("tie_word_embeddings").is_some_and(|v| v != true)
            || c["attention_dropout"].as_f64() != Some(0.0)
        {
            return Err(invalid("unsupported dense text configuration"));
        }
        Ok(Self {
            layers,
            query_heads,
            kv_heads,
            context,
            window,
            local_theta,
            global_theta,
            global_dim,
            rotated_pairs: width as usize / 2,
            hidden: dimension(c, "hidden_size", 65536)?,
            intermediate: dimension(c, "intermediate_size", 1 << 20)?,
            vocabulary: dimension(c, "vocab_size", 1 << 24)?,
            input_width: dimension(c, "hidden_size_per_layer_input", 65536)?,
            input_vocab: dimension(c, "vocab_size_per_layer_input", 1 << 24)?,
        })
    }
}

pub struct Gemma4Attention {
    config: Gemma4TextConfig,
    global_frequencies: Array,
}

impl Gemma4Attention {
    pub fn from_config(config: &Value, stream: &Stream) -> MlxDecoderResult<Self> {
        Self::new(Gemma4TextConfig::from_json(config)?, stream)
    }
    pub fn new(config: Gemma4TextConfig, stream: &Stream) -> MlxDecoderResult<Self> {
        let global_dim = config.global_dim;
        let global_theta = config.global_theta;
        // The exponents use the whole head width. Infinite divisors leave
        // the remaining split-half pairs at zero angle, including at decode.
        let divisors = attention::longrope_frequencies(
            global_dim,
            global_theta,
            &vec![1.0; global_dim / 2],
            stream,
        )?;
        let pairs = config.rotated_pairs;
        let global_frequencies = if pairs == global_dim / 2 {
            divisors
        } else {
            let active = ops::slice(&divisors, &[0], &[pairs], stream)?;
            let inactive = global_dim / 2 - pairs;
            let infinity =
                Array::from_f32_slice(&vec![f32::INFINITY; inactive], &[inactive as i32])?;
            ops::concatenate_axis(&active, &infinity, 0, stream)?
        };
        Ok(Self {
            config,
            global_frequencies,
        })
    }

    pub fn layers(&self) -> &[LayerAttention] {
        &self.config.layers
    }

    fn layer(&self, index: usize) -> Result<LayerAttention, ModelError> {
        self.config
            .layers
            .get(index)
            .copied()
            .ok_or_else(|| invalid("layer index out of bounds"))
    }

    fn rotate(
        &self,
        layer: LayerAttention,
        input: &Array,
        heads: usize,
        offset: usize,
        stream: &Stream,
    ) -> MlxDecoderResult<Array> {
        let shape = input.shape();
        if shape.len() != 4
            || shape[0] != 1
            || shape[1] != heads
            || shape[3] != layer.head_dim
            || shape[2] == 0
            || offset
                .checked_add(shape[2])
                .is_none_or(|end| end > self.config.context)
        {
            return Err(invalid("invalid head layout or absolute position").into());
        }
        Ok(if layer.sliding {
            attention::rope(input, self.config.local_theta, offset, stream)?
        } else {
            attention::rope_with_frequencies(input, &self.global_frequencies, offset, stream)?
        })
    }

    pub fn rotate_queries(
        &self,
        index: usize,
        q: &Array,
        offset: usize,
        stream: &Stream,
    ) -> MlxDecoderResult<Array> {
        self.rotate(
            self.layer(index)?,
            q,
            self.config.query_heads,
            offset,
            stream,
        )
    }

    pub fn rotate_owner_keys(
        &self,
        index: usize,
        k: &Array,
        offset: usize,
        stream: &Stream,
    ) -> MlxDecoderResult<Array> {
        let layer = self.layer(index)?;
        if layer.kv_owner != index {
            return Err(invalid("shared layer cannot append keys").into());
        }
        self.rotate(layer, k, self.config.kv_heads, offset, stream)
    }

    /// Q and K have already been normalized and rotated. V has parameter-free RMS normalization.
    pub fn attend(
        &self,
        index: usize,
        q: &Array,
        k: &Array,
        v: &Array,
        offset: usize,
        stream: &Stream,
    ) -> MlxDecoderResult<Array> {
        let layer = self.layer(index)?;
        let query = q.shape();
        let key = k.shape();
        if query.len() != 4
            || key.len() != 4
            || query[0] != 1
            || query[1] != self.config.query_heads
            || key[1] != self.config.kv_heads
            || query[3] != layer.head_dim
            || query[2] == 0
            || offset
                .checked_add(query[2])
                .is_none_or(|end| end > self.config.context)
        {
            return Err(invalid("invalid attention input or offset").into());
        }
        let prefix = key[2]
            .checked_sub(query[2])
            .ok_or_else(|| invalid("truncated current keys"))?;
        let mask = if layer.sliding {
            if prefix < offset.min(self.config.window - 1) || prefix > offset {
                return Err(invalid("invalid sliding prefix").into());
            }
            AttentionMask::SlidingCausalSuffix {
                prefix_tokens: prefix,
                window_tokens: self.config.window,
            }
        } else {
            if prefix != offset {
                return Err(invalid("full layer requires complete prefix").into());
            }
            AttentionMask::CausalSuffix {
                prefix_tokens: prefix,
            }
        };
        Ok(attention::grouped_query_attention_scaled(
            q, k, v, 1.0, mask, stream,
        )?)
    }
}
