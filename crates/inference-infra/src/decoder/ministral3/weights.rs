//! Verified text weights from the nested Ministral conditional-generation checkpoint.
use super::{MinistralConfig, config::invalid};
use crate::decoder::{
    fixture::{DecoderWeights, LayerWeights, MatrixWeight},
    loader::{
        CheckpointTensor, CheckpointVerification, LoadOptions, LoadedTensorSet, TensorLayout,
        TensorRequest, load_selected_tensor_set_verified,
    },
    model::ModelError,
};
use std::{collections::BTreeMap, path::Path};

impl MinistralConfig {
    pub fn tensor_layout(&self) -> Result<TensorLayout, ModelError> {
        self.validate()?;
        let mut tensors = BTreeMap::new();
        let matrix = |rows, cols| TensorRequest::Matrix { rows, cols };
        let norm = || TensorRequest::Float(vec![self.hidden]);
        tensors.insert(
            "language_model.model.embed_tokens.weight".into(),
            matrix(self.vocabulary, self.hidden),
        );
        tensors.insert("language_model.model.norm.weight".into(), norm());
        if !self.tied_embeddings {
            tensors.insert(
                "language_model.lm_head.weight".into(),
                matrix(self.vocabulary, self.hidden),
            );
        }
        for i in 0..self.layers {
            let layer = format!("language_model.model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                tensors.insert(format!("{layer}.{name}.weight"), norm());
            }
            for (name, rows, cols) in [
                ("self_attn.q_proj", self.q_width(), self.hidden),
                ("self_attn.k_proj", self.kv_width(), self.hidden),
                ("self_attn.v_proj", self.kv_width(), self.hidden),
                ("self_attn.o_proj", self.hidden, self.q_width()),
                ("mlp.gate_proj", self.intermediate, self.hidden),
                ("mlp.up_proj", self.intermediate, self.hidden),
                ("mlp.down_proj", self.hidden, self.intermediate),
            ] {
                tensors.insert(format!("{layer}.{name}.weight"), matrix(rows, cols));
            }
        }
        Ok(TensorLayout {
            quantization_bits: self.quantization_bits,
            group_size: self.group_size,
            tensors,
        })
    }
}

fn norm(tensors: &mut LoadedTensorSet, name: &str) -> Result<Vec<f32>, ModelError> {
    match tensors.remove(name) {
        Some(CheckpointTensor::Float { values, .. }) => Ok(values),
        _ => Err(invalid(format!("missing Ministral norm {name}"))),
    }
}
fn matrix(
    tensors: &mut LoadedTensorSet,
    name: &str,
    config: &MinistralConfig,
) -> Result<MatrixWeight, ModelError> {
    match tensors.remove(name) {
        Some(CheckpointTensor::Matrix(m))
            if m.bits() == config.group_size.map(|_| config.quantization_bits)
                && m.affine4_group_size() == config.group_size =>
        {
            Ok(m)
        }
        _ => Err(invalid(format!(
            "missing or inconsistent Ministral matrix {name}"
        ))),
    }
}
/// Keep packed text matrices, discarding only vision/projector payloads while
/// still authenticating their complete shard. No image capability is registered.
pub fn load_verified(
    path: &Path,
    options: LoadOptions,
    verification: &CheckpointVerification<'_>,
) -> Result<(MinistralConfig, DecoderWeights), ModelError> {
    let (c, mut tensors) = load_selected_tensor_set_verified(
        path,
        options,
        verification,
        |value| {
            let config = MinistralConfig::from_json(value)?;
            let layout = config.tensor_layout()?;
            Ok((config, layout))
        },
        |name| name.starts_with("vision_tower.") || name.starts_with("multi_modal_projector."),
    )?;
    let embed_tokens = matrix(&mut tensors, "language_model.model.embed_tokens.weight", &c)?;
    let final_layernorm = norm(&mut tensors, "language_model.model.norm.weight")?;
    let lm_head = if c.tied_embeddings {
        None
    } else {
        Some(matrix(&mut tensors, "language_model.lm_head.weight", &c)?)
    };
    let mut layers = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let layer = format!("language_model.model.layers.{i}");
        let mut projection =
            |name: &str| matrix(&mut tensors, &format!("{layer}.{name}.weight"), &c);
        layers.push(LayerWeights {
            q_proj: projection("self_attn.q_proj")?,
            k_proj: projection("self_attn.k_proj")?,
            v_proj: projection("self_attn.v_proj")?,
            o_proj: projection("self_attn.o_proj")?,
            gate_proj: projection("mlp.gate_proj")?,
            up_proj: projection("mlp.up_proj")?,
            down_proj: projection("mlp.down_proj")?,
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            input_layernorm: norm(&mut tensors, &format!("{layer}.input_layernorm.weight"))?,
            post_attention_layernorm: norm(
                &mut tensors,
                &format!("{layer}.post_attention_layernorm.weight"),
            )?,
        });
    }
    if !tensors.is_empty() {
        return Err(invalid("unconsumed Ministral text tensors"));
    }
    Ok((
        c,
        DecoderWeights {
            embed_tokens,
            final_layernorm,
            lm_head,
            layers,
        },
    ))
}
