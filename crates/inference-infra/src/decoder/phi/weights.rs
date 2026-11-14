//! Exact Phi fused-projection inventory through the shared verified loader.
use super::config::{PhiConfig, invalid};
use crate::decoder::{
    fixture::MatrixWeight,
    loader::{
        CheckpointTensor, CheckpointVerification, LoadOptions, LoadedTensorSet, TensorLayout,
        TensorRequest, load_tensor_set_verified,
    },
    model::ModelError,
};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug)]
pub struct PhiLayerWeights {
    pub input_norm: Vec<f32>,
    pub qkv: MatrixWeight,
    pub output: MatrixWeight,
    pub post_norm: Vec<f32>,
    pub gate_up: MatrixWeight,
    pub down: MatrixWeight,
}
#[derive(Clone, Debug)]
pub struct PhiWeights {
    pub embedding: MatrixWeight,
    pub layers: Vec<PhiLayerWeights>,
    pub norm: Vec<f32>,
    pub head: Option<MatrixWeight>,
}

impl PhiConfig {
    pub fn tensor_layout(&self) -> Result<TensorLayout, ModelError> {
        self.validate()?;
        let mut tensors = BTreeMap::new();
        let matrix = |rows, cols| TensorRequest::Matrix { rows, cols };
        let norm = || TensorRequest::Float(vec![self.hidden]);
        tensors.insert(
            "model.embed_tokens.weight".into(),
            matrix(self.vocabulary, self.hidden),
        );
        tensors.insert("model.norm.weight".into(), norm());
        if !self.tied_embeddings {
            tensors.insert(
                "lm_head.weight".into(),
                matrix(self.vocabulary, self.hidden),
            );
        }
        for layer in 0..self.layers {
            let prefix = format!("model.layers.{layer}");
            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                tensors.insert(format!("{prefix}.{suffix}"), norm());
            }
            for (name, rows, cols) in [
                ("self_attn.qkv_proj.weight", self.qkv_width(), self.hidden),
                ("self_attn.o_proj.weight", self.hidden, self.hidden),
                ("mlp.gate_up_proj.weight", self.gate_up_width(), self.hidden),
                ("mlp.down_proj.weight", self.hidden, self.intermediate),
            ] {
                tensors.insert(format!("{prefix}.{name}"), matrix(rows, cols));
            }
        }
        Ok(TensorLayout {
            quantization_bits: self.quantization_bits,
            group_size: self.group_size,
            tensors,
        })
    }
}
fn matrix(tensors: &mut LoadedTensorSet, name: &str) -> Result<MatrixWeight, ModelError> {
    match tensors.remove(name) {
        Some(CheckpointTensor::Matrix(m)) => Ok(m),
        _ => Err(invalid(format!("missing Phi matrix {name}"))),
    }
}
fn floats(tensors: &mut LoadedTensorSet, name: &str) -> Result<Vec<f32>, ModelError> {
    match tensors.remove(name) {
        Some(CheckpointTensor::Float { values, .. }) => Ok(values),
        _ => Err(invalid(format!("missing Phi norm {name}"))),
    }
}
pub fn load_verified(
    path: &Path,
    options: LoadOptions,
    verification: &CheckpointVerification<'_>,
) -> Result<(PhiConfig, PhiWeights), ModelError> {
    let (c, mut tensors) = load_tensor_set_verified(path, options, verification, |v| {
        let c = PhiConfig::from_json(v)?;
        let layout = c.tensor_layout()?;
        Ok((c, layout))
    })?;
    let embedding = matrix(&mut tensors, "model.embed_tokens.weight")?;
    let norm = floats(&mut tensors, "model.norm.weight")?;
    let head = if c.tied_embeddings {
        None
    } else {
        Some(matrix(&mut tensors, "lm_head.weight")?)
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(c.layers)
        .map_err(|_| invalid("Phi layer metadata allocation failed"))?;
    for layer in 0..c.layers {
        let p = format!("model.layers.{layer}");
        layers.push(PhiLayerWeights {
            input_norm: floats(&mut tensors, &format!("{p}.input_layernorm.weight"))?,
            post_norm: floats(
                &mut tensors,
                &format!("{p}.post_attention_layernorm.weight"),
            )?,
            qkv: matrix(&mut tensors, &format!("{p}.self_attn.qkv_proj.weight"))?,
            output: matrix(&mut tensors, &format!("{p}.self_attn.o_proj.weight"))?,
            gate_up: matrix(&mut tensors, &format!("{p}.mlp.gate_up_proj.weight"))?,
            down: matrix(&mut tensors, &format!("{p}.mlp.down_proj.weight"))?,
        });
    }
    if !tensors.is_empty() {
        return Err(invalid("unconsumed Phi tensors"));
    }
    Ok((
        c,
        PhiWeights {
            embedding,
            layers,
            norm,
            head,
        },
    ))
}
