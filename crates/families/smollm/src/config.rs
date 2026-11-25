use inference_infra::decoder::{
    config::{DecoderConfig, DecoderFamily},
    model::ModelError,
};
use serde_json::Value;
fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::CheckpointValidation(message.into())
}
/// Validate the supported SmolLM3 dense/NoPE layout before loading any weights.
pub fn parse_config(value: &Value) -> Result<DecoderConfig, ModelError> {
    if value["model_type"].as_str() != Some("smollm3")
        || value["architectures"] != serde_json::json!(["SmolLM3ForCausalLM"])
    {
        return Err(invalid(
            "expected model_type=smollm3 and SmolLM3ForCausalLM",
        ));
    }
    for key in ["attention_bias", "mlp_bias", "use_sliding_window"] {
        if value.get(key).is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(invalid(format!("unsupported {key}")));
        }
    }
    for key in [
        "sliding_window",
        "partial_rotary_factor",
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
            "SmolLM3 requires full even heads derived from hidden_size",
        ));
    }
    let positive = |key: &str| {
        value[key]
            .as_f64()
            .map(|x| x as f32)
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or_else(|| invalid(format!("invalid {key}")))
    };
    if value.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        return Err(invalid("SmolLM3 scaled RoPE is not implemented"));
    }
    let num_layers = dim("num_hidden_layers")?;
    let rope_layers =
        inference_infra::decoder::config::smollm_rope_layers(value, num_layers).map_err(invalid)?;
    if value.get("layer_types").is_some_and(|v| {
        v.as_array().is_none_or(|layers| {
            layers.len() != num_layers
                || layers.iter().any(|l| l.as_str() != Some("full_attention"))
        })
    }) {
        return Err(invalid("SmolLM3 requires full attention in every layer"));
    }
    // Exported weights are complete dense projections, even when pretraining
    // used tensor parallel partitions. Validate their divisibility, not a
    // different runtime graph or partitioned checkpoint format.
    if let Some(tp) = value.get("pretraining_tp") {
        let tp = tp
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .filter(|x| *x > 0)
            .ok_or_else(|| invalid("invalid pretraining_tp"))?;
        if !hidden_size.is_multiple_of(tp) || !dim("intermediate_size")?.is_multiple_of(tp) {
            return Err(invalid("invalid tensor-parallel projection dimensions"));
        }
    }
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
        family: DecoderFamily::SmolLm3,
        rms_eps: positive("rms_norm_eps")?,
        rope_scaling: None,
        rope_layers: Some(rope_layers),
        quantization_bits: bits,
        hidden_size,
        q_heads,
        kv_heads,
        head_dim,
        num_layers,
        intermediate_size: dim("intermediate_size")?,
        vocab_size: dim("vocab_size")?,
        max_context: dim("max_position_embeddings")?,
        rope_theta: positive("rope_theta")?,
        tie_embeddings: match value.get("tie_word_embeddings") {
            None => true, // Publisher and pinned MLX SmolLM3 default.
            Some(v) => v
                .as_bool()
                .ok_or_else(|| invalid("invalid tie_word_embeddings"))?,
        },
    })
}
