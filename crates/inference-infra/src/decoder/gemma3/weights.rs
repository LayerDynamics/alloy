//! Exact Gemma 3 text inventory using the shared bounded, hash-verified loader.
use super::{Gemma3Config, config::invalid};
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
pub struct Gemma3LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub pre_ffn_norm: Vec<f32>,
    pub post_ffn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q: MatrixWeight,
    pub k: MatrixWeight,
    pub v: MatrixWeight,
    pub output: MatrixWeight,
    pub gate: MatrixWeight,
    pub up: MatrixWeight,
    pub down: MatrixWeight,
}
#[derive(Clone, Debug)]
pub struct Gemma3Weights {
    pub embedding: MatrixWeight,
    pub norm: Vec<f32>,
    pub head: Option<MatrixWeight>,
    pub layers: Vec<Gemma3LayerWeights>,
}
impl Gemma3Config {
    pub fn tensor_layout(&self) -> Result<TensorLayout, ModelError> {
        self.validate()?;
        let p = self.tensor_prefix();
        let mut tensors = BTreeMap::new();
        let matrix = |rows, cols| TensorRequest::Matrix { rows, cols };
        tensors.insert(
            format!("{p}model.embed_tokens.weight"),
            matrix(self.vocabulary, self.hidden),
        );
        tensors.insert(
            format!("{p}model.norm.weight"),
            TensorRequest::Float(vec![self.hidden]),
        );
        if !self.tied_embeddings {
            tensors.insert(
                format!("{p}lm_head.weight"),
                matrix(self.vocabulary, self.hidden),
            );
        }
        for i in 0..self.layers {
            let p = format!("{p}model.layers.{i}");
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                tensors.insert(
                    format!("{p}.{name}.weight"),
                    TensorRequest::Float(vec![self.hidden]),
                );
            }
            for name in ["q_norm", "k_norm"] {
                tensors.insert(
                    format!("{p}.self_attn.{name}.weight"),
                    TensorRequest::Float(vec![self.head_dim]),
                );
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
                tensors.insert(format!("{p}.{name}.weight"), matrix(rows, cols));
            }
        }
        Ok(TensorLayout {
            quantization_bits: self.quantization_bits,
            group_size: self.group_size,
            tensors,
        })
    }
}
fn matrix(t: &mut LoadedTensorSet, name: &str) -> Result<MatrixWeight, ModelError> {
    match t.remove(name) {
        Some(CheckpointTensor::Matrix(m)) => Ok(m),
        _ => Err(invalid(format!("missing Gemma 3 matrix {name}"))),
    }
}
fn floats(t: &mut LoadedTensorSet, name: &str) -> Result<Vec<f32>, ModelError> {
    match t.remove(name) {
        Some(CheckpointTensor::Float { values, .. }) => Ok(values),
        _ => Err(invalid(format!("missing Gemma 3 norm {name}"))),
    }
}
pub fn load_verified(
    path: &Path,
    options: LoadOptions,
    verification: &CheckpointVerification<'_>,
) -> Result<(Gemma3Config, Gemma3Weights), ModelError> {
    let (c, mut tensors) = load_tensor_set_verified(path, options, verification, |v| {
        let c = Gemma3Config::from_json(v)?;
        let layout = c.tensor_layout()?;
        Ok((c, layout))
    })?;
    let p = c.tensor_prefix();
    let embedding = matrix(&mut tensors, &format!("{p}model.embed_tokens.weight"))?;
    let norm = floats(&mut tensors, &format!("{p}model.norm.weight"))?;
    let head = if c.tied_embeddings {
        None
    } else {
        Some(matrix(&mut tensors, &format!("{p}lm_head.weight"))?)
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(c.layers)
        .map_err(|_| invalid("Gemma 3 layer metadata allocation failed"))?;
    for i in 0..c.layers {
        let p = format!("{p}model.layers.{i}");
        layers.push(Gemma3LayerWeights {
            input_norm: floats(&mut tensors, &format!("{p}.input_layernorm.weight"))?,
            post_attention_norm: floats(
                &mut tensors,
                &format!("{p}.post_attention_layernorm.weight"),
            )?,
            pre_ffn_norm: floats(
                &mut tensors,
                &format!("{p}.pre_feedforward_layernorm.weight"),
            )?,
            post_ffn_norm: floats(
                &mut tensors,
                &format!("{p}.post_feedforward_layernorm.weight"),
            )?,
            q_norm: floats(&mut tensors, &format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: floats(&mut tensors, &format!("{p}.self_attn.k_norm.weight"))?,
            q: matrix(&mut tensors, &format!("{p}.self_attn.q_proj.weight"))?,
            k: matrix(&mut tensors, &format!("{p}.self_attn.k_proj.weight"))?,
            v: matrix(&mut tensors, &format!("{p}.self_attn.v_proj.weight"))?,
            output: matrix(&mut tensors, &format!("{p}.self_attn.o_proj.weight"))?,
            gate: matrix(&mut tensors, &format!("{p}.mlp.gate_proj.weight"))?,
            up: matrix(&mut tensors, &format!("{p}.mlp.up_proj.weight"))?,
            down: matrix(&mut tensors, &format!("{p}.mlp.down_proj.weight"))?,
        });
    }
    if !tensors.is_empty() {
        return Err(invalid("unconsumed Gemma 3 tensors"));
    }
    Ok((
        c,
        Gemma3Weights {
            embedding,
            norm,
            head,
            layers,
        },
    ))
}
