//! Validated Phi-4-mini dense geometry and explicit LongRoPE parameters.
use crate::decoder::model::ModelError;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhiRopeRegime {
    Short,
    Long,
}

#[derive(Clone, Debug)]
pub struct PhiConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocabulary: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub context: usize,
    pub original_context: usize,
    pub theta: f32,
    pub epsilon: f32,
    pub tied_embeddings: bool,
    pub short_factors: Vec<f32>,
    pub long_factors: Vec<f32>,
    pub quantization_bits: u8,
    pub group_size: Option<usize>,
}

pub(super) fn invalid(message: impl ToString) -> ModelError {
    ModelError::CheckpointValidation(message.to_string())
}
fn positive(v: &Value, key: &str) -> Result<usize, ModelError> {
    v[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && i32::try_from(n).is_ok())
        .ok_or_else(|| invalid(format!("invalid Phi dimension {key}")))
}
fn real(v: &Value, key: &str) -> Result<f32, ModelError> {
    v[key]
        .as_f64()
        .map(|n| n as f32)
        .filter(|n| n.is_finite() && *n > 0.0)
        .ok_or_else(|| invalid(format!("invalid Phi number {key}")))
}
impl PhiConfig {
    pub fn from_json(v: &Value) -> Result<Self, ModelError> {
        if v["model_type"] != "phi3"
            || v.get("architectures")
                .is_some_and(|a| a != &serde_json::json!(["Phi3ForCausalLM"]))
            || v["hidden_act"] != "silu"
            || [
                "attention_bias",
                "mlp_bias",
                "lm_head_bias",
                "rope_traditional",
            ]
            .iter()
            .any(|key| v.get(*key).is_some_and(|b| b.as_bool() != Some(false)))
            || v.get("full_attn_mod")
                .is_some_and(|n| n.as_u64() != Some(1))
            || v.get("interpolate_factor")
                .is_some_and(|n| n.as_f64() != Some(1.0))
        {
            return Err(invalid("unsupported Phi architecture equations"));
        }
        let hidden = positive(v, "hidden_size")?;
        let query_heads = positive(v, "num_attention_heads")?;
        if !hidden.is_multiple_of(query_heads) {
            return Err(invalid("Phi head width is not integral"));
        }
        let head_dim = hidden / query_heads;
        if v.get("head_dim")
            .is_some_and(|n| n.as_u64() != Some(head_dim as u64))
        {
            return Err(invalid("inconsistent Phi head width"));
        }
        let fraction = v["partial_rotary_factor"]
            .as_f64()
            .filter(|n| n.is_finite() && *n > 0.0 && *n <= 1.0)
            .ok_or_else(|| invalid("invalid Phi rotary fraction"))?;
        let rotary = head_dim as f64 * fraction;
        if rotary.fract() != 0.0 {
            return Err(invalid("nonintegral Phi rotary width"));
        }
        let rope = v["rope_scaling"]
            .as_object()
            .ok_or_else(|| invalid("missing Phi LongRoPE"))?;
        if rope.get("type").and_then(Value::as_str) != Some("longrope")
            || rope
                .keys()
                .any(|k| !matches!(k.as_str(), "type" | "short_factor" | "long_factor"))
        {
            return Err(invalid("unsupported Phi LongRoPE variant"));
        }
        let factors = |key: &str| -> Result<Vec<f32>, ModelError> {
            let a = rope
                .get(key)
                .and_then(Value::as_array)
                .filter(|a| a.len() == rotary as usize / 2 && a.len() <= 2048)
                .ok_or_else(|| invalid("Phi factor width mismatch"))?;
            a.iter()
                .map(|n| {
                    n.as_f64()
                        .map(|n| n as f32)
                        .filter(|n| n.is_finite() && *n > 0.0)
                        .ok_or_else(|| invalid("invalid Phi frequency factor"))
                })
                .collect()
        };
        let context = positive(v, "max_position_embeddings")?;
        if v.get("sliding_window")
            .is_some_and(|n| !n.is_null() && n.as_u64().is_none_or(|n| n < context as u64))
        {
            return Err(invalid(
                "sliding Phi attention is not supported by this profile",
            ));
        }
        if let (Some(a), Some(b)) = (v.get("quantization"), v.get("quantization_config"))
            && a != b
        {
            return Err(invalid("inconsistent Phi quantization declarations"));
        }
        let (bits, group_size) = match v
            .get("quantization")
            .or_else(|| v.get("quantization_config"))
        {
            None => (4, None),
            Some(q) => {
                let bits = positive(q, "bits")?;
                if !matches!(bits, 4 | 8)
                    || positive(q, "group_size")? != 64
                    || q.get("mode").is_some_and(|m| m != "affine")
                {
                    return Err(invalid("unsupported Phi affine representation"));
                }
                (bits as u8, Some(64))
            }
        };
        let config = Self {
            hidden,
            query_heads,
            head_dim,
            rotary_dim: rotary as usize,
            context,
            intermediate: positive(v, "intermediate_size")?,
            vocabulary: positive(v, "vocab_size")?,
            layers: positive(v, "num_hidden_layers")?,
            kv_heads: positive(v, "num_key_value_heads")?,
            original_context: positive(v, "original_max_position_embeddings")?,
            theta: real(v, "rope_theta")?,
            epsilon: real(v, "rms_norm_eps")?,
            tied_embeddings: v["tie_word_embeddings"]
                .as_bool()
                .ok_or_else(|| invalid("missing Phi embedding contract"))?,
            short_factors: factors("short_factor")?,
            long_factors: factors("long_factor")?,
            quantization_bits: bits,
            group_size,
        };
        config.validate()?;
        Ok(config)
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
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.head_dim == 0
            || self.head_dim > 4096
            || self.query_heads.checked_mul(self.head_dim) != Some(self.hidden)
            || self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || !self.rotary_dim.is_multiple_of(2)
            || self.original_context < 2
            || self.original_context > self.context
            || self.context > 1 << 24
            || !self.theta.is_finite()
            || self.theta <= 0.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !matches!(self.quantization_bits, 4 | 8)
            || self.group_size.is_some_and(|g| {
                g != 64 || !self.hidden.is_multiple_of(g) || !self.intermediate.is_multiple_of(g)
            })
            || [&self.short_factors, &self.long_factors].iter().any(|a| {
                a.len() != self.rotary_dim / 2 || a.iter().any(|n| !n.is_finite() || *n <= 0.0)
            })
        {
            return Err(invalid("invalid Phi geometry or position parameters"));
        }
        for regime in [PhiRopeRegime::Short, PhiRopeRegime::Long] {
            if self
                .frequency_denominators(regime)
                .iter()
                .any(|n| !n.is_finite() || *n <= 0.0)
            {
                return Err(invalid("unrepresentable Phi frequencies"));
            }
        }
        Ok(())
    }
    pub fn qkv_width(&self) -> usize {
        self.hidden + 2 * self.kv_heads * self.head_dim
    }
    pub fn gate_up_width(&self) -> usize {
        2 * self.intermediate
    }
    /// Explicit selection: a caller must bind this regime to cache identity.
    pub fn frequency_denominators(&self, regime: PhiRopeRegime) -> Vec<f32> {
        let factors = match regime {
            PhiRopeRegime::Short => &self.short_factors,
            PhiRopeRegime::Long => &self.long_factors,
        };
        factors
            .iter()
            .enumerate()
            .map(|(i, &factor)| factor * self.theta.powf((2 * i) as f32 / self.rotary_dim as f32))
            .collect()
    }
    pub fn rotary_amplitude(&self) -> f32 {
        let factor = self.context as f64 / self.original_context as f64;
        if factor <= 1.0 {
            1.0
        } else {
            (1.0 + factor.ln() / (self.original_context as f64).ln()).sqrt() as f32
        }
    }
}
