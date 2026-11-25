use crate::decoder::{hybrid::LinearAttentionConfig, model::ModelError};
use serde_json::Value;

/// Validated dense Qwen3.5 text architecture, including its actual layer schedule.
#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocabulary: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
    pub context: usize,
    pub tied_embeddings: bool,
    /// True selects recurrent attention; false selects gated full attention.
    pub linear_layers: Vec<bool>,
    pub linear: LinearAttentionConfig,
    pub quantization_bits: u8,
    pub group_size: Option<usize>,
}
pub(super) fn invalid(message: impl ToString) -> ModelError {
    ModelError::CheckpointValidation(message.to_string())
}
fn number(value: &Value, key: &str) -> Result<usize, ModelError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && i32::try_from(n).is_ok())
        .ok_or_else(|| invalid(format!("invalid Qwen3.5 dimension {key}")))
}
fn real(value: &Value, key: &str, default: f64) -> Result<f64, ModelError> {
    let n = match value.get(key) {
        None => default,
        Some(v) => v
            .as_f64()
            .ok_or_else(|| invalid(format!("invalid Qwen3.5 number {key}")))?,
    };
    if !n.is_finite() || n <= 0.0 {
        return Err(invalid(format!("invalid Qwen3.5 number {key}")));
    }
    Ok(n)
}
fn boolean(value: &Value, key: &str, default: bool) -> Result<bool, ModelError> {
    match value.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| invalid(format!("invalid Qwen3.5 flag {key}"))),
    }
}
impl Qwen35Config {
    /// Parse the hash-verified nested text configuration. Unsupported trained
    /// equations fail before weight loading; unrelated generation metadata is
    /// not interpreted as architecture or tool authorization.
    pub fn from_json(root: &Value) -> Result<Self, ModelError> {
        if root["model_type"] != "qwen3_5" || !root["text_config"].is_object() {
            return Err(invalid("expected nested dense Qwen3.5 text config"));
        }
        if let Some(architectures) = root.get("architectures")
            && architectures != &serde_json::json!(["Qwen3_5ForConditionalGeneration"])
        {
            return Err(invalid("unsupported Qwen3.5 architecture"));
        }
        let text = &root["text_config"];
        if text["model_type"] != "qwen3_5_text" {
            return Err(invalid("unsupported Qwen3.5 text model type"));
        }
        if boolean(text, "attention_bias", false)?
            || !boolean(text, "attn_output_gate", true)?
            || text.get("hidden_act").is_some_and(|v| v != "silu")
            || text
                .get("num_experts")
                .is_some_and(|v| v.as_u64() != Some(0))
            || text
                .get("mlp_only_layers")
                .is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty()))
            || text.get("rope_scaling").is_some_and(|v| !v.is_null())
        {
            return Err(invalid("unsupported Qwen3.5 attention/MLP/RoPE variant"));
        }
        let hidden = number(text, "hidden_size")?;
        let epsilon = real(text, "rms_norm_eps", 1e-6)? as f32;
        let count = number(text, "num_hidden_layers")?;
        if count > 4096 {
            return Err(invalid("Qwen3.5 layer count exceeds limit"));
        }
        let interval = number(text, "full_attention_interval")?;
        let linear_layers = match text.get("layer_types") {
            Some(v) => {
                let a = v
                    .as_array()
                    .filter(|a| a.len() == count)
                    .ok_or_else(|| invalid("invalid Qwen3.5 layer schedule"))?;
                a.iter()
                    .map(|v| match v.as_str() {
                        Some("linear_attention") => Ok(true),
                        Some("full_attention") => Ok(false),
                        _ => Err(invalid("unsupported Qwen3.5 layer type")),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => (0..count).map(|i| (i + 1) % interval != 0).collect(),
        };
        let rope = text
            .get("rope_parameters")
            .ok_or_else(|| invalid("missing Qwen3.5 rope_parameters"))?;
        if !rope.is_object()
            || rope
                .get("type")
                .or_else(|| rope.get("rope_type"))
                .and_then(Value::as_str)
                != Some("default")
        {
            return Err(invalid("unsupported Qwen3.5 RoPE scaling"));
        }
        let fraction = real(rope, "partial_rotary_factor", 0.25)?;
        if fraction > 1.0
            || text
                .get("partial_rotary_factor")
                .is_some_and(|v| v.as_f64() != Some(fraction))
        {
            return Err(invalid("inconsistent Qwen3.5 rotary fraction"));
        }
        let head_dim = number(text, "head_dim")?;
        let rotary = head_dim as f64 * fraction;
        if rotary.fract() != 0.0 {
            return Err(invalid("Qwen3.5 rotary width must be integral"));
        }
        let tied = boolean(text, "tie_word_embeddings", false)?;
        if root
            .get("tie_word_embeddings")
            .is_some_and(|v| v.as_bool() != Some(tied))
        {
            return Err(invalid("inconsistent tied embeddings"));
        }
        let quant = root
            .get("quantization")
            .or_else(|| root.get("quantization_config"));
        if let (Some(a), Some(b)) = (root.get("quantization"), root.get("quantization_config"))
            && a != b
        {
            return Err(invalid("inconsistent Qwen3.5 quantization declarations"));
        }
        let (bits, group_size) = if let Some(q) = quant {
            if q.get("mode").is_some_and(|v| v != "affine") {
                return Err(invalid("only affine Qwen3.5 quantization is supported"));
            }
            let bits = number(q, "bits")?;
            if !matches!(bits, 4 | 8) {
                return Err(invalid("unsupported Qwen3.5 quantization bits"));
            }
            (bits as u8, Some(number(q, "group_size")?))
        } else {
            (4, None)
        };
        let config = Self {
            hidden,
            intermediate: number(text, "intermediate_size")?,
            vocabulary: number(text, "vocab_size")?,
            query_heads: number(text, "num_attention_heads")?,
            kv_heads: number(text, "num_key_value_heads")?,
            head_dim,
            rotary_dim: rotary as usize,
            rope_theta: real(rope, "rope_theta", 100000.0)? as f32,
            epsilon,
            context: number(text, "max_position_embeddings")?,
            tied_embeddings: tied,
            linear_layers,
            linear: LinearAttentionConfig {
                hidden_size: hidden,
                key_heads: number(text, "linear_num_key_heads")?,
                value_heads: number(text, "linear_num_value_heads")?,
                key_dim: number(text, "linear_key_head_dim")?,
                value_dim: number(text, "linear_value_head_dim")?,
                convolution_kernel: number(text, "linear_conv_kernel_dim")?,
                rms_epsilon: epsilon,
            },
            quantization_bits: bits,
            group_size,
        };
        config.validate()?;
        Ok(config)
    }
    /// Recheck public fields before any native allocation.
    pub fn validate(&self) -> Result<(), ModelError> {
        for n in [
            self.hidden,
            self.intermediate,
            self.vocabulary,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.context,
            self.linear.hidden_size,
            self.linear.key_heads,
            self.linear.value_heads,
            self.linear.key_dim,
            self.linear.value_dim,
            self.linear.convolution_kernel,
        ] {
            if n == 0 || i32::try_from(n).is_err() {
                return Err(invalid("invalid Qwen3.5 dimension"));
            }
        }
        if self.linear_layers.is_empty()
            || self.linear_layers.len() > 4096
            || !self.linear_layers.contains(&true)
            || !self.linear_layers.contains(&false)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || !self.rotary_dim.is_multiple_of(2)
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.linear.hidden_size != self.hidden
            || self.linear.rms_epsilon != self.epsilon
            || self.linear.convolution_kernel < 2
            || !self
                .linear
                .value_heads
                .is_multiple_of(self.linear.key_heads)
            || !matches!(self.quantization_bits, 4 | 8)
            || self.group_size == Some(0)
        {
            return Err(invalid("invalid Qwen3.5 hybrid configuration"));
        }
        for (a, b) in [
            (self.query_heads, self.head_dim),
            (self.kv_heads, self.head_dim),
            (self.linear.key_heads, self.linear.key_dim),
            (self.linear.value_heads, self.linear.value_dim),
        ] {
            if a.checked_mul(b)
                .and_then(|n| n.checked_mul(4))
                .is_none_or(|n| i32::try_from(n).is_err())
            {
                return Err(invalid("Qwen3.5 projection dimension overflow"));
            }
        }
        Ok(())
    }
}
