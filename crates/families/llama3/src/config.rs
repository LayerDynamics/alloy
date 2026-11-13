use inference_infra::decoder::{
    config::{DecoderConfig, DecoderFamily, Llama31Scaling},
    model::ModelError,
};
use serde_json::Value;
fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::CheckpointValidation(message.into())
}
/// Validate the supported dense Llama layout before loading any weights.
pub fn parse_config(value: &Value) -> Result<DecoderConfig, ModelError> {
    if value["model_type"].as_str() != Some("llama")
        || value["architectures"] != serde_json::json!(["LlamaForCausalLM"])
    {
        return Err(invalid("expected model_type=llama and LlamaForCausalLM"));
    }
    for key in ["attention_bias", "mlp_bias", "use_sliding_window"] {
        if value.get(key).is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(invalid(format!("unsupported {key}")));
        }
    }
    for key in [
        "sliding_window",
        "partial_rotary_factor",
        "layer_types",
        "rope_traditional",
        "num_local_experts",
        "num_experts_per_tok",
    ] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return Err(invalid(format!("unsupported {key}")));
        }
    }
    if value["hidden_act"].as_str() != Some("silu") {
        return Err(invalid("only SiLU is supported"));
    }
    if value
        .get("pretraining_tp")
        .is_some_and(|v| v.as_u64() != Some(1))
    {
        return Err(invalid("pretraining_tp must be one"));
    }
    let dim = |key: &str| {
        value[key]
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .filter(|&x| x > 0 && x <= i32::MAX as usize)
            .ok_or_else(|| invalid(format!("invalid {key}")))
    };
    let hidden_size = dim("hidden_size")?;
    let q_heads = dim("num_attention_heads")?;
    let kv_heads = dim("num_key_value_heads")?;
    if !hidden_size.is_multiple_of(q_heads) || !q_heads.is_multiple_of(kv_heads) {
        return Err(invalid("invalid attention head divisibility"));
    }
    let head_dim = hidden_size / q_heads;
    if !head_dim.is_multiple_of(2)
        || value
            .get("head_dim")
            .is_some_and(|v| v.as_u64() != Some(head_dim as u64))
    {
        return Err(invalid(
            "Llama requires full even heads derived from hidden_size",
        ));
    }
    let positive = |key: &str| {
        value[key]
            .as_f64()
            .map(|x| x as f32)
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or_else(|| invalid(format!("invalid {key}")))
    };
    let rope_scaling = match value.get("rope_scaling").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => {
            if v["rope_type"].as_str() != Some("llama3") {
                return Err(invalid(
                    "only Llama 3.1 rope_type=llama3 scaling is implemented",
                ));
            }
            let number = |key: &str| {
                v[key]
                    .as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| invalid(format!("invalid rope {key}")))
            };
            let scaling = Llama31Scaling {
                factor: number("factor")?,
                low_freq_factor: number("low_freq_factor")?,
                high_freq_factor: number("high_freq_factor")?,
                original_max_position_embeddings: v["original_max_position_embeddings"]
                    .as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .ok_or_else(|| invalid("missing original context"))?,
            };
            if !scaling.validate() {
                return Err(invalid("invalid Llama 3.1 frequency scaling"));
            }
            Some(scaling)
        }
    };
    let quant = value.get("quantization").filter(|v| !v.is_null());
    let other = value.get("quantization_config").filter(|v| !v.is_null());
    if quant.zip(other).is_some_and(|(a, b)| a != b) {
        return Err(invalid("conflicting quantization metadata"));
    }
    let bits = if let Some(q) = quant.or(other) {
        if q.get("mode").is_some_and(|v| v.as_str() != Some("affine"))
            || q["group_size"].as_u64() != Some(64)
        {
            return Err(invalid(
                "only affine group_size=64 is qualified by this adapter",
            ));
        }
        match q["bits"].as_u64() {
            Some(4) => 4,
            Some(8) => 8,
            _ => return Err(invalid("only affine 4-bit and 8-bit are supported")),
        }
    } else {
        4
    };
    Ok(DecoderConfig {
        family: DecoderFamily::Llama3,
        rms_eps: positive("rms_norm_eps")?,
        rope_scaling,
        quantization_bits: bits,
        hidden_size,
        q_heads,
        kv_heads,
        head_dim,
        num_layers: dim("num_hidden_layers")?,
        intermediate_size: dim("intermediate_size")?,
        vocab_size: dim("vocab_size")?,
        max_context: dim("max_position_embeddings")?,
        rope_theta: positive("rope_theta")?,
        tie_embeddings: value["tie_word_embeddings"]
            .as_bool()
            .ok_or_else(|| invalid("tie_word_embeddings must be explicit"))?,
    })
}
