//! Checkpoint layout for the Qwen3-Coder 30B-A3B routed decoder.
use super::{
    config::{DecoderConfig, DecoderFamily},
    loader::{PackedGeometry, PackedInventory, TensorLayout, TensorRequest},
    model::ModelError,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) fn invalid(reason: &'static str) -> ModelError {
    ModelError::InvalidConfig(reason)
}

#[derive(Clone, Debug)]
pub struct CoderConfig {
    pub decoder: DecoderConfig,
    pub experts: usize,
    pub top_k: usize,
}

impl CoderConfig {
    /// Only the released 30B-A3B architecture and its mixed affine conversion
    /// are supported. Generation metadata is independent of the forward graph.
    pub fn from_json(root: &Value) -> Result<Self, ModelError> {
        let supported = json!({
            "model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],
            "hidden_size":2048,"moe_intermediate_size":768,"head_dim":128,
            "num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,
            "num_experts":128,"num_experts_per_tok":8,"vocab_size":151936,
            "max_position_embeddings":262144,"rope_theta":10000000,"rms_norm_eps":0.000001,
            "decoder_sparse_step":1,"mlp_only_layers":[],"hidden_act":"silu",
            "norm_topk_prob":true,"tie_word_embeddings":false,"use_qk_norm":true,
            "qkv_bias":false,"use_sliding_window":false,"sliding_window":null,
            "rope_scaling":null,"shared_expert_intermediate_size":0
        });
        if supported
            .as_object()
            .unwrap()
            .iter()
            .any(|(k, v)| root.get(k) != Some(v))
            || root.get("attention_bias").is_some_and(|v| v != false)
        {
            return Err(invalid("unsupported Qwen3-Coder architecture"));
        }
        let mut quant = json!({"bits":4,"group_size":64});
        for i in 0..48 {
            quant[format!("model.layers.{i}.mlp.gate")] = json!({"bits":8,"group_size":64});
        }
        if root["quantization"] != quant || root["quantization_config"] != quant {
            return Err(invalid("unexpected Qwen3-Coder affine packing"));
        }
        Ok(Self {
            experts: 128,
            top_k: 8,
            decoder: DecoderConfig {
                family: DecoderFamily::Qwen3Moe,
                rms_eps: 1e-6,
                rope_scaling: None,
                rope_layers: None,
                quantization_bits: 4,
                hidden_size: 2048,
                num_layers: 48,
                q_heads: 32,
                kv_heads: 4,
                head_dim: 128,
                intermediate_size: 768,
                vocab_size: 151936,
                rope_theta: 10_000_000.,
                max_context: 262144,
                tie_embeddings: false,
            },
        })
    }

    /// Resident F32 norm/affine metadata and packed U32 payload for the full
    /// mixed-precision checkpoint. Derived from the same inventory as loading.
    pub fn weight_payload_bytes(root: &Value) -> Result<u64, ModelError> {
        let (_, layout, packing) = Self::inventory(root)?;
        let mut total = 0u64;
        for (name, tensor) in layout.tensors {
            // inventory accepts only the validated, fixed 30B-A3B geometry.
            total += match tensor {
                TensorRequest::Float(shape) => 4 * shape.iter().product::<usize>() as u64,
                TensorRequest::Matrix { rows, cols } => {
                    let elements = rows as u64 * cols as u64;
                    let bits = packing[&name].bits as u64;
                    elements * bits / 8 + elements / 64 * 8
                }
            };
        }
        Ok(total)
    }

    pub(super) fn inventory(root: &Value) -> Result<PackedInventory<Self>, ModelError> {
        let config = Self::from_json(root)?;
        let c = &config.decoder;
        let mut tensors = BTreeMap::new();
        let mut packed = BTreeMap::new();
        let mut matrix = |name: String, leading: &[usize], cols, bits| {
            tensors.insert(
                name.clone(),
                TensorRequest::Matrix {
                    rows: leading.iter().product(),
                    cols,
                },
            );
            packed.insert(
                name,
                PackedGeometry {
                    leading: leading.to_vec(),
                    bits,
                },
            );
        };
        matrix(
            "model.embed_tokens.weight".into(),
            &[c.vocab_size],
            c.hidden_size,
            4,
        );
        matrix("lm_head.weight".into(), &[c.vocab_size], c.hidden_size, 4);
        for i in 0..c.num_layers {
            let p = format!("model.layers.{i}");
            for (name, rows, cols) in [
                ("q_proj", c.q_dim(), c.hidden_size),
                ("k_proj", c.kv_dim(), c.hidden_size),
                ("v_proj", c.kv_dim(), c.hidden_size),
                ("o_proj", c.hidden_size, c.q_dim()),
            ] {
                matrix(format!("{p}.self_attn.{name}.weight"), &[rows], cols, 4);
            }
            matrix(
                format!("{p}.mlp.gate.weight"),
                &[config.experts],
                c.hidden_size,
                8,
            );
            for (name, rows, cols) in [
                ("gate_proj", c.intermediate_size, c.hidden_size),
                ("up_proj", c.intermediate_size, c.hidden_size),
                ("down_proj", c.hidden_size, c.intermediate_size),
            ] {
                matrix(
                    format!("{p}.mlp.switch_mlp.{name}.weight"),
                    &[config.experts, rows],
                    cols,
                    4,
                );
            }
        }
        tensors.insert(
            "model.norm.weight".into(),
            TensorRequest::Float(vec![c.hidden_size]),
        );
        for i in 0..c.num_layers {
            for (name, width) in [
                ("input_layernorm", c.hidden_size),
                ("post_attention_layernorm", c.hidden_size),
                ("self_attn.q_norm", c.head_dim),
                ("self_attn.k_norm", c.head_dim),
            ] {
                tensors.insert(
                    format!("model.layers.{i}.{name}.weight"),
                    TensorRequest::Float(vec![width]),
                );
            }
        }
        Ok((
            config,
            TensorLayout {
                quantization_bits: 4,
                group_size: Some(64),
                tensors,
            },
            packed,
        ))
    }
}
