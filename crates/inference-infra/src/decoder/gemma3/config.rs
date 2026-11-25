//! Strict Gemma 3 text metadata, including publisher defaults omitted by MLX.
use crate::decoder::model::ModelError;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma3Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocabulary: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub window: usize,
    pub sliding: Vec<bool>,
    pub theta: f32,
    pub local_theta: f32,
    pub rope_factor: f32,
    pub query_scalar: f32,
    pub epsilon: f32,
    pub tied_embeddings: bool,
    pub quantization_bits: u8,
    pub group_size: Option<usize>,
    /// MLX's wrapped text artifact stores tensors under language_model.
    pub wrapped: bool,
}
pub(super) fn invalid(message: impl ToString) -> ModelError {
    ModelError::CheckpointValidation(message.to_string())
}
fn integer(v: &Value, key: &str, default: usize) -> Result<usize, ModelError> {
    match v.get(key) {
        None => Ok(default),
        Some(n) => n
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0 && i32::try_from(n).is_ok())
            .ok_or_else(|| invalid(format!("invalid Gemma 3 dimension {key}"))),
    }
}
fn number(v: &Value, key: &str, default: f32) -> Result<f32, ModelError> {
    match v.get(key) {
        None => Ok(default),
        Some(n) => n
            .as_f64()
            .map(|n| n as f32)
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| invalid(format!("invalid Gemma 3 number {key}"))),
    }
}
impl Gemma3Config {
    pub fn from_json(outer: &Value) -> Result<Self, ModelError> {
        let v = match outer["model_type"].as_str() {
            Some("gemma3_text") => outer,
            Some("gemma3") => outer
                .get("text_config")
                .filter(|v| v.is_object())
                .ok_or_else(|| invalid("missing Gemma 3 text configuration"))?,
            _ => return Err(invalid("not a Gemma 3 text architecture")),
        };
        if v["model_type"] != "gemma3_text"
            || v.get("rope_parameters").is_some()
            || matches!((outer.get("tie_word_embeddings"), v.get("tie_word_embeddings")), (Some(a),Some(b)) if a != b)
            || outer.get("architectures").is_some_and(|a| {
                a != &serde_json::json!(["Gemma3ForConditionalGeneration"])
                    && a != &serde_json::json!(["Gemma3ForCausalLM"])
            })
            || ["attention_bias", "mlp_bias", "rope_traditional"]
                .iter()
                .any(|k| v.get(*k).is_some_and(|x| x.as_bool() != Some(false)))
            || ["hidden_activation", "hidden_act"].iter().any(|k| {
                v.get(*k)
                    .is_some_and(|x| x != "gelu_pytorch_tanh" && x != "gelu_approx")
            })
            || ["attn_logit_softcapping", "final_logit_softcapping"]
                .iter()
                .any(|k| v.get(*k).is_some_and(|x| !x.is_null()))
        {
            return Err(invalid("unsupported Gemma 3 equations"));
        }
        let layers = integer(v, "num_hidden_layers", 26)?;
        if layers > 4096 {
            return Err(invalid("Gemma 3 layer count exceeds bound"));
        }
        let period = integer(v, "sliding_window_pattern", 6)?;
        let sliding = if let Some(explicit) = v.get("layer_types").filter(|v| !v.is_null()) {
            let a = explicit
                .as_array()
                .filter(|a| a.len() == layers)
                .ok_or_else(|| invalid("Gemma 3 layer schedule length mismatch"))?;
            a.iter()
                .map(|x| match x.as_str() {
                    Some("sliding_attention") => Ok(true),
                    Some("full_attention") => Ok(false),
                    _ => Err(invalid("unknown Gemma 3 attention layer")),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..layers)
                .map(|i| !(i + 1).is_multiple_of(period))
                .collect()
        };
        let rope_factor = match v.get("rope_scaling").filter(|r| !r.is_null()) {
            None => 1.0,
            Some(r) => {
                if r.get("rope_type")
                    .or_else(|| r.get("type"))
                    .and_then(Value::as_str)
                    != Some("linear")
                    || r.as_object().is_none_or(|o| {
                        o.keys()
                            .any(|k| !matches!(k.as_str(), "rope_type" | "type" | "factor"))
                    })
                    || matches!((r.get("type"), r.get("rope_type")), (Some(a), Some(b)) if a != b)
                {
                    return Err(invalid("unsupported Gemma 3 global RoPE"));
                }
                if r.get("factor").is_none() {
                    return Err(invalid("missing Gemma 3 linear RoPE factor"));
                }
                number(r, "factor", 1.0)?
            }
        };
        let mut quant = None;
        for owner in [outer, v] {
            for key in ["quantization", "quantization_config"] {
                if let Some(q) = owner.get(key) {
                    if quant.is_some_and(|previous| previous != q) {
                        return Err(invalid("conflicting Gemma 3 quantization"));
                    }
                    quant = Some(q);
                }
            }
        }
        let (bits, group) = if let Some(q) = quant {
            let bits = integer(q, "bits", 0)?;
            let group = integer(q, "group_size", 0)?;
            if !matches!(bits, 4 | 8)
                || group != 64
                || q.get("mode").is_some_and(|m| m != "affine")
                || q.as_object().is_none_or(|o| {
                    o.keys()
                        .any(|k| !matches!(k.as_str(), "bits" | "group_size" | "mode"))
                })
            {
                return Err(invalid("unsupported Gemma 3 quantization recipe"));
            }
            (bits as u8, Some(group))
        } else {
            (4, None)
        };
        let tied = v
            .get("tie_word_embeddings")
            .or_else(|| outer.get("tie_word_embeddings"));
        let c = Self {
            hidden: integer(v, "hidden_size", 1152)?,
            intermediate: integer(v, "intermediate_size", 6912)?,
            vocabulary: integer(v, "vocab_size", 262144)?,
            layers,
            sliding,
            query_heads: integer(v, "num_attention_heads", 4)?,
            kv_heads: integer(v, "num_key_value_heads", 1)?,
            head_dim: integer(v, "head_dim", 256)?,
            context: integer(v, "max_position_embeddings", 32768)?,
            window: integer(v, "sliding_window", 512)?,
            theta: number(v, "rope_theta", 1_000_000.0)?,
            local_theta: number(v, "rope_local_base_freq", 10_000.0)?,
            query_scalar: number(v, "query_pre_attn_scalar", 256.0)?,
            epsilon: number(v, "rms_norm_eps", 1e-6)?,
            rope_factor,
            // The frozen MLX wrapped artifact carries an explicit output head.
            // Missing/extra heads are rejected by exact tensor inventory loading.
            tied_embeddings: match tied {
                None => outer["model_type"] != "gemma3",
                Some(b) => b
                    .as_bool()
                    .ok_or_else(|| invalid("invalid Gemma 3 tied embedding declaration"))?,
            },
            quantization_bits: bits,
            group_size: group,
            wrapped: outer["model_type"] == "gemma3",
        };
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.hidden == 0
            || self.hidden > 65536
            || self.intermediate == 0
            || self.intermediate > 1 << 20
            || self.vocabulary == 0
            || self.vocabulary > 1 << 21
            || self.layers == 0
            || self.layers > 4096
            || self.query_heads == 0
            || self.query_heads > 1024
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.head_dim == 0
            || self.head_dim > 4096
            || !self.head_dim.is_multiple_of(2)
            || self.sliding.len() != self.layers
            || self.context == 0
            || self.context > 1 << 24
            || self.window == 0
            || self.window > 1 << 24
            || [
                self.theta,
                self.local_theta,
                self.query_scalar,
                self.epsilon,
                self.rope_factor,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || self.rope_factor < 1.0
            || !matches!(self.quantization_bits, 4 | 8)
            || self.group_size.is_some_and(|g| {
                g != 64
                    || [self.hidden, self.intermediate, self.q_width()]
                        .iter()
                        .any(|d| !d.is_multiple_of(g))
            })
        {
            return Err(invalid("invalid Gemma 3 geometry or position parameters"));
        }
        Ok(())
    }
    pub fn q_width(&self) -> usize {
        self.query_heads * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    pub fn embedding_multiplier(&self) -> f32 {
        half::bf16::from_f64((self.hidden as f64).sqrt()).to_f32()
    }
    pub fn tensor_prefix(&self) -> &'static str {
        if self.wrapped { "language_model." } else { "" }
    }
}
