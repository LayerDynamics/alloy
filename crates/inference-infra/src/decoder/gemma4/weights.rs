//! Packed E4B text inventory and authenticated checkpoint loading.
use super::{Gemma4TextConfig, invalid};
use crate::decoder::{
    loader::{
        CheckpointTensor, CheckpointVerification, LoadOptions, LoadedTensorSet, TensorLayout,
        TensorRequest, load_selected_tensor_set_verified,
    },
    model::ModelError,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

pub struct WeightPayloadPlan {
    pub payload_bytes: u64,
    pub largest_upload_bytes: u64,
}

impl Gemma4TextConfig {
    /// Validate the pinned packed recipe before planning memory or reading any tensor.
    pub fn from_affine4_json(value: &Value) -> Result<Self, ModelError> {
        let recipe = json!({"group_size":64,"bits":4,"mode":"affine"});
        if value["quantization"] != recipe
            || value
                .get("quantization_config")
                .is_some_and(|q| q != &recipe)
            || ["quantization", "quantization_config"]
                .iter()
                .any(|k| value["text_config"].get(k).is_some_and(|q| q != &recipe))
        {
            return Err(invalid("unsupported or conflicting affine recipe"));
        }
        Self::from_json(value)
    }

    /// Packed U32 values and two F32 group parameters are separate uploads.
    pub fn weight_payload_plan(&self) -> Result<WeightPayloadPlan, ModelError> {
        let mut total = 0u128;
        let mut largest = 0u128;
        for request in self.affine4_layout()?.tensors.into_values() {
            let (payload, upload) = match request {
                TensorRequest::Matrix { rows, cols } => {
                    if !cols.is_multiple_of(64) {
                        return Err(invalid("unaligned affine4 matrix"));
                    }
                    let elements = rows as u128 * cols as u128;
                    let packed = elements / 2;
                    (packed + 2 * (elements / 64) * 4, packed)
                }
                TensorRequest::Float(shape) => {
                    let bytes = shape
                        .into_iter()
                        .try_fold(4u128, |n, d| n.checked_mul(d as u128))
                        .ok_or_else(|| invalid("float tensor overflow"))?;
                    (bytes, bytes)
                }
            };
            total += payload;
            largest = largest.max(upload);
        }
        Ok(WeightPayloadPlan {
            payload_bytes: u64::try_from(total).map_err(|_| invalid("weight payload overflow"))?,
            largest_upload_bytes: u64::try_from(largest)
                .map_err(|_| invalid("weight upload overflow"))?,
        })
    }

    /// Exact logical names and shapes. Affine scales and biases are collected
    /// by the shared loader. Only the cache owners have K/V parameters.
    pub fn affine4_layout(&self) -> Result<TensorLayout, ModelError> {
        let per_layer = self
            .input_width
            .checked_mul(self.layers.len())
            .ok_or_else(|| invalid("per-layer embedding width overflow"))?;
        let matrix = |rows, cols| TensorRequest::Matrix { rows, cols };
        let vector = |n| TensorRequest::Float(vec![n]);
        let prefix = "language_model.model";
        let mut tensors = BTreeMap::from([
            (
                format!("{prefix}.embed_tokens.weight"),
                matrix(self.vocabulary, self.hidden),
            ),
            (
                format!("{prefix}.embed_tokens_per_layer.weight"),
                matrix(self.input_vocab, per_layer),
            ),
            (
                format!("{prefix}.per_layer_model_projection.weight"),
                matrix(per_layer, self.hidden),
            ),
            (
                format!("{prefix}.per_layer_projection_norm.weight"),
                vector(self.input_width),
            ),
            (format!("{prefix}.norm.weight"), vector(self.hidden)),
        ]);
        for (i, layer) in self.layers.iter().enumerate() {
            let p = format!("{prefix}.layers.{i}");
            let q_width = self.query_heads * layer.head_dim;
            let kv_width = self.kv_heads * layer.head_dim;
            for (name, shape) in [
                ("self_attn.q_proj.weight", matrix(q_width, self.hidden)),
                ("self_attn.o_proj.weight", matrix(self.hidden, q_width)),
                ("self_attn.q_norm.weight", vector(layer.head_dim)),
                (
                    "mlp.gate_proj.weight",
                    matrix(self.intermediate, self.hidden),
                ),
                ("mlp.up_proj.weight", matrix(self.intermediate, self.hidden)),
                (
                    "mlp.down_proj.weight",
                    matrix(self.hidden, self.intermediate),
                ),
                (
                    "per_layer_input_gate.weight",
                    matrix(self.input_width, self.hidden),
                ),
                (
                    "per_layer_projection.weight",
                    matrix(self.hidden, self.input_width),
                ),
                ("input_layernorm.weight", vector(self.hidden)),
                ("post_attention_layernorm.weight", vector(self.hidden)),
                ("pre_feedforward_layernorm.weight", vector(self.hidden)),
                ("post_feedforward_layernorm.weight", vector(self.hidden)),
                ("post_per_layer_input_norm.weight", vector(self.hidden)),
                ("layer_scalar", vector(1)),
            ] {
                tensors.insert(format!("{p}.{name}"), shape);
            }
            if layer.kv_owner == i {
                for name in ["k_proj", "v_proj"] {
                    tensors.insert(
                        format!("{p}.self_attn.{name}.weight"),
                        matrix(kv_width, self.hidden),
                    );
                }
                tensors.insert(
                    format!("{p}.self_attn.k_norm.weight"),
                    vector(layer.head_dim),
                );
            }
        }
        Ok(TensorLayout {
            quantization_bits: 4,
            group_size: Some(64),
            tensors,
        })
    }
}

fn configure(value: &Value) -> Result<(Gemma4TextConfig, TensorLayout), ModelError> {
    let config = Gemma4TextConfig::from_affine4_json(value)?;
    let layout = config.affine4_layout()?;
    Ok((config, layout))
}

/// Authenticates the complete shard, then decodes only the text decoder weights.
pub fn load_verified(
    path: &Path,
    options: LoadOptions,
    verification: &CheckpointVerification<'_>,
) -> Result<(Gemma4TextConfig, LoadedTensorSet), ModelError> {
    let (config, tensors) =
        load_selected_tensor_set_verified(path, options, verification, configure, |name| {
            [
                "audio_tower.",
                "vision_tower.",
                "embed_audio.",
                "embed_vision.",
            ]
            .iter()
            .any(|prefix| name.starts_with(prefix))
        })?;
    for (name, t) in &tensors {
        if let CheckpointTensor::Matrix(matrix) = t
            && (matrix.bits() != Some(4) || matrix.affine4_group_size() != Some(64))
        {
            return Err(invalid(&format!(
                "expected a packed affine4/group64 matrix at {name}"
            )));
        }
    }
    Ok((config, tensors))
}
