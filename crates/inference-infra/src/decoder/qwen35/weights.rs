//! Exact converted text-checkpoint inventory. No tensor-name guessing, vision
//! fallback, dense dequantization, or second normalization-weight sanitization.
use super::config::{Qwen35Config, invalid};
use crate::decoder::{
    fixture::MatrixWeight,
    hybrid::LinearAttentionWeights,
    loader::{
        CheckpointTensor, CheckpointVerification, LoadOptions, LoadedTensorSet, TensorLayout,
        TensorRequest, load_tensor_set_verified,
    },
    model::ModelError,
};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug)]
pub struct FullAttentionWeights {
    pub q: MatrixWeight,
    pub k: MatrixWeight,
    pub v: MatrixWeight,
    pub output: MatrixWeight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}
#[derive(Clone, Debug)]
pub enum AttentionWeights {
    Linear(LinearAttentionWeights),
    Full(FullAttentionWeights),
}
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub attention: AttentionWeights,
    pub post_norm: Vec<f32>,
    pub gate: MatrixWeight,
    pub up: MatrixWeight,
    pub down: MatrixWeight,
}
#[derive(Clone, Debug)]
pub struct Qwen35Weights {
    pub embedding: MatrixWeight,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
    pub head: Option<MatrixWeight>,
}
impl Qwen35Config {
    /// Exact sanitized MLX text inventory, with no multimodal/MTP extras.
    pub fn tensor_layout(&self) -> Result<TensorLayout, ModelError> {
        self.validate()?;
        let mut tensors = BTreeMap::new();
        let matrix = |rows, cols| TensorRequest::Matrix { rows, cols };
        let vector = |n| TensorRequest::Float(vec![n]);
        tensors.insert(
            "language_model.model.embed_tokens.weight".into(),
            matrix(self.vocabulary, self.hidden),
        );
        tensors.insert(
            "language_model.model.norm.weight".into(),
            vector(self.hidden),
        );
        if !self.tied_embeddings {
            tensors.insert(
                "language_model.lm_head.weight".into(),
                matrix(self.vocabulary, self.hidden),
            );
        }
        let q = self.query_heads * self.head_dim;
        let kv = self.kv_heads * self.head_dim;
        let key = self.linear.key_heads * self.linear.key_dim;
        let val = self.linear.value_heads * self.linear.value_dim;
        for (index, &linear) in self.linear_layers.iter().enumerate() {
            let prefix = format!("language_model.model.layers.{index}");
            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                tensors.insert(format!("{prefix}.{suffix}"), vector(self.hidden));
            }
            for (suffix, rows, cols) in [
                ("mlp.gate_proj.weight", self.intermediate, self.hidden),
                ("mlp.up_proj.weight", self.intermediate, self.hidden),
                ("mlp.down_proj.weight", self.hidden, self.intermediate),
            ] {
                tensors.insert(format!("{prefix}.{suffix}"), matrix(rows, cols));
            }
            if linear {
                for (suffix, rows, cols) in [
                    ("in_proj_qkv.weight", 2 * key + val, self.hidden),
                    ("in_proj_z.weight", val, self.hidden),
                    ("in_proj_a.weight", self.linear.value_heads, self.hidden),
                    ("in_proj_b.weight", self.linear.value_heads, self.hidden),
                    ("out_proj.weight", self.hidden, val),
                ] {
                    tensors.insert(format!("{prefix}.linear_attn.{suffix}"), matrix(rows, cols));
                }
                tensors.insert(
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    TensorRequest::Float(vec![2 * key + val, self.linear.convolution_kernel, 1]),
                );
                for (suffix, n) in [
                    ("A_log", self.linear.value_heads),
                    ("dt_bias", self.linear.value_heads),
                    ("norm.weight", self.linear.value_dim),
                ] {
                    tensors.insert(format!("{prefix}.linear_attn.{suffix}"), vector(n));
                }
            } else {
                for (suffix, rows, cols) in [
                    ("q_proj.weight", 2 * q, self.hidden),
                    ("k_proj.weight", kv, self.hidden),
                    ("v_proj.weight", kv, self.hidden),
                    ("o_proj.weight", self.hidden, q),
                ] {
                    tensors.insert(format!("{prefix}.self_attn.{suffix}"), matrix(rows, cols));
                }
                for suffix in ["q_norm.weight", "k_norm.weight"] {
                    tensors.insert(
                        format!("{prefix}.self_attn.{suffix}"),
                        vector(self.head_dim),
                    );
                }
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
        _ => Err(invalid(format!("missing Qwen3.5 matrix {name}"))),
    }
}
fn floats(tensors: &mut LoadedTensorSet, name: &str) -> Result<Vec<f32>, ModelError> {
    match tensors.remove(name) {
        Some(CheckpointTensor::Float { values, .. }) => Ok(values),
        _ => Err(invalid(format!("missing Qwen3.5 floating tensor {name}"))),
    }
}
/// Load exact pinned bytes and move their decoded payload into architecture
/// weights. Native upload is a separate admitted step; no host model is run.
pub fn load_verified<'a>(
    path: &Path,
    options: LoadOptions,
    verification: &'a CheckpointVerification<'a>,
) -> Result<(Qwen35Config, Qwen35Weights), ModelError> {
    let (c, mut tensors) = load_tensor_set_verified(path, options, verification, |data| {
        let c = Qwen35Config::from_json(data)?;
        let layout = c.tensor_layout()?;
        Ok((c, layout))
    })?;
    let embedding = matrix(&mut tensors, "language_model.model.embed_tokens.weight")?;
    let norm = floats(&mut tensors, "language_model.model.norm.weight")?;
    let head = if c.tied_embeddings {
        None
    } else {
        Some(matrix(&mut tensors, "language_model.lm_head.weight")?)
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(c.linear_layers.len())
        .map_err(|_| ModelError::AllocationFailed)?;
    for (index, &linear) in c.linear_layers.iter().enumerate() {
        if (verification.cancelled)() {
            return Err(ModelError::Cancelled);
        }
        let p = format!("language_model.model.layers.{index}");
        let attention = if linear {
            AttentionWeights::Linear(LinearAttentionWeights {
                qkv: matrix(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"))?,
                z: matrix(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"))?,
                a: matrix(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"))?,
                b: matrix(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"))?,
                output: matrix(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"))?,
                convolution: floats(&mut tensors, &format!("{p}.linear_attn.conv1d.weight"))?,
                a_log: floats(&mut tensors, &format!("{p}.linear_attn.A_log"))?,
                dt_bias: floats(&mut tensors, &format!("{p}.linear_attn.dt_bias"))?,
                norm: floats(&mut tensors, &format!("{p}.linear_attn.norm.weight"))?,
            })
        } else {
            AttentionWeights::Full(FullAttentionWeights {
                q: matrix(&mut tensors, &format!("{p}.self_attn.q_proj.weight"))?,
                k: matrix(&mut tensors, &format!("{p}.self_attn.k_proj.weight"))?,
                v: matrix(&mut tensors, &format!("{p}.self_attn.v_proj.weight"))?,
                output: matrix(&mut tensors, &format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: floats(&mut tensors, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: floats(&mut tensors, &format!("{p}.self_attn.k_norm.weight"))?,
            })
        };
        layers.push(LayerWeights {
            input_norm: floats(&mut tensors, &format!("{p}.input_layernorm.weight"))?,
            attention,
            post_norm: floats(
                &mut tensors,
                &format!("{p}.post_attention_layernorm.weight"),
            )?,
            gate: matrix(&mut tensors, &format!("{p}.mlp.gate_proj.weight"))?,
            up: matrix(&mut tensors, &format!("{p}.mlp.up_proj.weight"))?,
            down: matrix(&mut tensors, &format!("{p}.mlp.down_proj.weight"))?,
        });
    }
    if !tensors.is_empty() {
        return Err(invalid("unconsumed Qwen3.5 weights"));
    }
    Ok((
        c,
        Qwen35Weights {
            embedding,
            layers,
            norm,
            head,
        },
    ))
}
