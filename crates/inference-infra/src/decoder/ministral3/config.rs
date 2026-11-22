//! Validated nested Mistral3 text configuration. Multimodal execution is absent.
use crate::decoder::model::ModelError;
use mlx_ffi::attention::YarnParameters;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct MinistralConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocabulary: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub epsilon: f32,
    pub tied_embeddings: bool,
    pub yarn: YarnParameters,
    pub query_scaling_beta: f32,
    pub quantization_bits: u8,
    pub group_size: Option<usize>,
}

pub(super) fn invalid(message: impl ToString) -> ModelError {
    ModelError::CheckpointValidation(message.to_string())
}
fn dimension(v: &Value, key: &str) -> Result<usize, ModelError> {
    v[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && n <= 1 << 24)
        .ok_or_else(|| invalid(format!("invalid Ministral dimension {key}")))
}
fn real(v: &Value, key: &str) -> Result<f32, ModelError> {
    v[key]
        .as_f64()
        .map(|n| n as f32)
        .filter(|n| n.is_finite())
        .ok_or_else(|| invalid(format!("invalid Ministral number {key}")))
}
impl MinistralConfig {
    pub fn from_json(root: &Value) -> Result<Self, ModelError> {
        let text = &root["text_config"];
        if root["model_type"] != "mistral3"
            || root["architectures"] != serde_json::json!(["Mistral3ForConditionalGeneration"])
            || text["model_type"] != "ministral3"
            || text["hidden_act"] != "silu"
        {
            return Err(invalid(
                "expected nested Mistral3/Ministral3 text architecture",
            ));
        }
        for key in [
            "attention_bias",
            "mlp_bias",
            "lm_head_bias",
            "rope_traditional",
            "use_sliding_window",
        ] {
            if text.get(key).is_some_and(|v| v.as_bool() != Some(false)) {
                return Err(invalid(format!("unsupported Ministral {key}")));
            }
        }
        for key in [
            "sliding_window",
            "rope_scaling",
            "partial_rotary_factor",
            "num_local_experts",
            "num_experts_per_tok",
        ] {
            if text.get(key).is_some_and(|v| !v.is_null()) {
                return Err(invalid(format!("unsupported Ministral {key}")));
            }
        }
        let layers = dimension(text, "num_hidden_layers")?;
        if text.get("layer_types").is_some_and(|v| {
            v.as_array()
                .is_none_or(|a| a.len() != layers || a.iter().any(|t| t != "full_attention"))
        }) {
            return Err(invalid(
                "only full-attention Ministral layers are implemented",
            ));
        }
        let rope = &text["rope_parameters"];
        let object = rope
            .as_object()
            .ok_or_else(|| invalid("missing Ministral rope_parameters"))?;
        if object.get("type").is_some_and(|v| v != "yarn")
            || object.get("rope_type").is_some_and(|v| v != "yarn")
            || (!object.contains_key("type") && !object.contains_key("rope_type"))
            || object.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "type"
                        | "rope_type"
                        | "rope_theta"
                        | "factor"
                        | "original_max_position_embeddings"
                        | "beta_fast"
                        | "beta_slow"
                        | "mscale"
                        | "mscale_all_dim"
                        | "llama_4_scaling_beta"
                )
            })
        {
            return Err(invalid("unsupported or conflicting Ministral YaRN variant"));
        }
        let theta = real(rope, "rope_theta")?;
        if text
            .get("rope_theta")
            .is_some_and(|v| v.as_f64() != Some(f64::from(theta)))
        {
            return Err(invalid("conflicting Ministral rotary base"));
        }
        let quant = root.get("quantization").filter(|v| !v.is_null());
        let other = root.get("quantization_config").filter(|v| !v.is_null());
        if quant.zip(other).is_some_and(|(a, b)| a != b)
            || ["quantization", "quantization_config"]
                .iter()
                .any(|key| text.get(*key).is_some_and(|v| !v.is_null()))
        {
            return Err(invalid("conflicting Ministral quantization metadata"));
        }
        let (bits, group_size) = match quant.or(other) {
            None => (4, None),
            Some(q) => {
                let bits = dimension(q, "bits")?;
                if !matches!(bits, 4 | 8)
                    || dimension(q, "group_size")? != 64
                    || q.get("mode").is_some_and(|v| v != "affine")
                    || q.as_object().is_none_or(|o| {
                        o.keys()
                            .any(|k| !matches!(k.as_str(), "bits" | "group_size" | "mode"))
                    })
                {
                    return Err(invalid("unsupported Ministral affine quantization"));
                }
                (bits as u8, Some(64))
            }
        };
        let head_dim = dimension(text, "head_dim")?;
        let config = Self {
            hidden: dimension(text, "hidden_size")?,
            intermediate: dimension(text, "intermediate_size")?,
            vocabulary: dimension(text, "vocab_size")?,
            layers,
            query_heads: dimension(text, "num_attention_heads")?,
            kv_heads: dimension(text, "num_key_value_heads")?,
            head_dim,
            context: dimension(text, "max_position_embeddings")?,
            epsilon: real(text, "rms_norm_eps")?,
            tied_embeddings: text["tie_word_embeddings"]
                .as_bool()
                .ok_or_else(|| invalid("missing Ministral embedding contract"))?,
            yarn: YarnParameters {
                dimensions: head_dim,
                theta,
                factor: real(rope, "factor")?,
                original_context: dimension(rope, "original_max_position_embeddings")?,
                beta_fast: real(rope, "beta_fast")?,
                beta_slow: real(rope, "beta_slow")?,
                mscale: real(rope, "mscale")?,
                mscale_all_dim: real(rope, "mscale_all_dim")?,
            },
            query_scaling_beta: real(rope, "llama_4_scaling_beta")?,
            quantization_bits: bits,
            group_size,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.yarn.validate().map_err(invalid)?;
        if self.hidden == 0
            || self.hidden > 65536
            || self.intermediate == 0
            || self.intermediate > 1 << 20
            || self.vocabulary == 0
            || self.vocabulary > 1 << 21
            || self.layers == 0
            || self.layers > 4096
            || self.query_heads == 0
            || self.query_heads > 4096
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.head_dim != self.yarn.dimensions
            || self.context < self.yarn.original_context
            || self.context > 1 << 24
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.query_scaling_beta.is_finite()
            || self.query_scaling_beta < 0.0
            || f64::from(self.query_scaling_beta)
                * (1.0 + self.context as f64 / self.yarn.original_context as f64).ln()
                + 1.0
                > f64::from(f32::MAX)
            || !matches!(self.quantization_bits, 4 | 8)
            || self.group_size.is_some_and(|g| {
                g != 64
                    || !self.hidden.is_multiple_of(g)
                    || !self.intermediate.is_multiple_of(g)
                    || !self.q_width().is_multiple_of(g)
            })
        {
            return Err(invalid("invalid Ministral geometry or position parameters"));
        }
        Ok(())
    }
    /// Attention projection width is independent of the residual hidden width.
    pub fn q_width(&self) -> usize {
        self.query_heads * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

impl MinistralConfig {
    /// Geometry consumed by the full-attention native decoder after text-only loading.
    pub fn decoder_config(&self) -> Result<crate::decoder::config::DecoderConfig, ModelError> {
        use crate::decoder::config::{DecoderConfig, DecoderFamily};
        self.validate()?;
        Ok(DecoderConfig {
            family: DecoderFamily::Ministral3 {
                yarn: self.yarn,
                query_beta: self.query_scaling_beta,
            },
            hidden_size: self.hidden,
            intermediate_size: self.intermediate,
            vocab_size: self.vocabulary,
            num_layers: self.layers,
            q_heads: self.query_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            rms_eps: self.epsilon,
            rope_theta: self.yarn.theta,
            rope_scaling: None,
            rope_layers: None,
            max_context: self.context,
            quantization_bits: self.quantization_bits,
            tie_embeddings: self.tied_embeddings,
        })
    }
}
