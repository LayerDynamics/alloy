//! Bounded safetensors loading with explicit dtype and shape validation.
use crate::{
    config::Qwen3Config,
    fixture::{LayerWeights, Qwen3Weights},
    model::ModelError,
};
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::{Path, PathBuf},
};

/// Resource limits for raw checkpoint bytes and decoded f32 weights separately.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub max_file_bytes: usize,
    pub max_weight_bytes: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_file_bytes: 512 * 1024 * 1024,
            max_weight_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Input accepted by Qwen3Model::load. Paths use bounded safetensors loading.
pub trait ModelWeightSource {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError>;
}
impl ModelWeightSource for Qwen3Weights {
    fn into_weights(self, _: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        Ok(self)
    }
}
impl ModelWeightSource for &Path {
    fn into_weights(self, c: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        load_weights(c, self, LoadOptions::default())
    }
}
impl ModelWeightSource for PathBuf {
    fn into_weights(self, c: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        self.as_path().into_weights(c)
    }
}
impl ModelWeightSource for &str {
    fn into_weights(self, c: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        Path::new(self).into_weights(c)
    }
}
impl ModelWeightSource for String {
    fn into_weights(self, c: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        Path::new(&self).into_weights(c)
    }
}
fn error(message: impl ToString) -> ModelError {
    ModelError::Checkpoint(message.to_string())
}
fn product(a: usize, b: usize) -> Result<usize, ModelError> {
    a.checked_mul(b)
        .ok_or(ModelError::DimensionOverflow("checkpoint tensor"))
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, ModelError> {
    let file = std::fs::File::open(path).map_err(error)?;
    let size = usize::try_from(file.metadata().map_err(error)?.len()).map_err(error)?;
    if size > limit {
        return Err(ModelError::MemoryLimit {
            required: size,
            limit,
        });
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| ModelError::AllocationFailed)?;
    file.take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(error)?;
    if bytes.len() > limit {
        return Err(ModelError::MemoryLimit {
            required: bytes.len(),
            limit,
        });
    }
    Ok(bytes)
}

/// Load a single file or a directory containing a single file/sharded index.
/// Supports F32/F16/BF16 and MLX affine 4-bit U32 matrices with scales/biases.
/// Large checkpoints require explicit resource-limit opt-in.
pub fn load_weights(
    c: &Qwen3Config,
    path: &Path,
    options: LoadOptions,
) -> Result<Qwen3Weights, ModelError> {
    let q_dim = product(c.q_heads, c.head_dim)?;
    let kv_dim = product(c.kv_heads, c.head_dim)?;
    if [
        c.hidden_size,
        c.num_layers,
        c.q_heads,
        c.kv_heads,
        c.head_dim,
        c.intermediate_size,
        c.vocab_size,
    ]
    .contains(&0)
    {
        return Err(error("zero checkpoint dimension"));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let root = root.canonicalize().map_err(error)?;
    let mut group_size = 64;
    let config_path = root.join("config.json");
    if config_path.exists() {
        let data: serde_json::Value =
            serde_json::from_slice(&read_bounded(&config_path, 1024 * 1024)?).map_err(error)?;
        for (name, expected) in [
            ("hidden_size", c.hidden_size),
            ("num_hidden_layers", c.num_layers),
            ("num_attention_heads", c.q_heads),
            ("num_key_value_heads", c.kv_heads),
            ("head_dim", c.head_dim),
            ("intermediate_size", c.intermediate_size),
            ("vocab_size", c.vocab_size),
        ] {
            if data.get(name).and_then(|v| v.as_u64()) != Some(expected as u64) {
                return Err(error(format!("config mismatch: {name}")));
            }
        }
        if data.get("tie_word_embeddings").and_then(|v| v.as_bool()) != Some(c.tie_embeddings) {
            return Err(error("config mismatch: tie_word_embeddings"));
        }
        if data
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .is_some_and(|v| (v - 1e-6).abs() > 1e-12)
        {
            return Err(error("unsupported rms_norm_eps"));
        }
        if data
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v as f32 != c.rope_theta)
        {
            return Err(error("config mismatch: rope_theta"));
        }
        if data.get("rope_scaling").is_some_and(|v| !v.is_null())
            || data.get("attention_bias").and_then(|v| v.as_bool()) == Some(true)
        {
            return Err(error("rope scaling and attention bias are unsupported"));
        }
        if let Some(quant) = data
            .get("quantization")
            .filter(|v| !v.is_null())
            .or_else(|| data.get("quantization_config").filter(|v| !v.is_null()))
        {
            if quant.get("bits").and_then(|v| v.as_u64()) != Some(4)
                || quant
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .is_some_and(|v| v != "affine")
            {
                return Err(error("only MLX affine 4-bit quantization is supported"));
            }
            group_size = quant
                .get("group_size")
                .and_then(|v| v.as_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| error("missing quantization group_size"))?;
            if group_size == 0 {
                return Err(error("zero quantization group_size"));
            }
        }
    }
    let mut paths = BTreeSet::new();
    let mut index_map = BTreeMap::new();
    if path.is_file() {
        paths.insert(path.canonicalize().map_err(error)?);
    } else if root.join("model.safetensors.index.json").exists() {
        let index: serde_json::Value = serde_json::from_slice(&read_bounded(
            &root.join("model.safetensors.index.json"),
            16 * 1024 * 1024,
        )?)
        .map_err(error)?;
        let map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| error("missing weight_map"))?;
        for (name, file) in map {
            let file = file
                .as_str()
                .ok_or_else(|| error("invalid shard filename"))?;
            let child = Path::new(file);
            if child.components().count() != 1
                || child.extension().and_then(|v| v.to_str()) != Some("safetensors")
            {
                return Err(error("unsafe shard filename"));
            }
            let resolved = root.join(child).canonicalize().map_err(error)?;
            if resolved.parent() != Some(root.as_path()) {
                return Err(error("shard escapes checkpoint directory"));
            }
            index_map.insert(name.clone(), resolved.clone());
            paths.insert(resolved);
        }
    } else {
        paths.insert(root.join("model.safetensors"));
    }
    let mut buffers = BTreeMap::new();
    let mut raw_bytes = 0usize;
    for path in paths {
        let bytes = read_bounded(&path, options.max_file_bytes.saturating_sub(raw_bytes))?;
        raw_bytes = raw_bytes
            .checked_add(bytes.len())
            .ok_or(ModelError::DimensionOverflow("checkpoint bytes"))?;
        buffers.insert(path, bytes);
    }
    let mut shards = BTreeMap::new();
    let mut names = BTreeMap::new();
    for (path, bytes) in &buffers {
        let tensors = SafeTensors::deserialize(bytes).map_err(error)?;
        for name in tensors.names() {
            if names.insert(name.to_string(), path.clone()).is_some() {
                return Err(error(format!("duplicate tensor: {name}")));
            }
        }
        shards.insert(path.clone(), tensors);
    }
    for (name, path) in index_map {
        if names.get(&name) != Some(&path) {
            return Err(error(format!("index mismatch for {name}")));
        }
    }
    let mut reader = Reader {
        shards,
        names,
        used: 0,
        limit: options.max_weight_bytes,
        group_size,
    };
    let embed_tokens =
        reader.tensor("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(c.num_layers)
        .map_err(|_| ModelError::AllocationFailed)?;
    for index in 0..c.num_layers {
        let p = format!("model.layers.{index}");
        layers.push(LayerWeights {
            input_layernorm: reader
                .tensor(&format!("{p}.input_layernorm.weight"), &[c.hidden_size])?,
            q_proj: reader.tensor(
                &format!("{p}.self_attn.q_proj.weight"),
                &[q_dim, c.hidden_size],
            )?,
            k_proj: reader.tensor(
                &format!("{p}.self_attn.k_proj.weight"),
                &[kv_dim, c.hidden_size],
            )?,
            v_proj: reader.tensor(
                &format!("{p}.self_attn.v_proj.weight"),
                &[kv_dim, c.hidden_size],
            )?,
            q_norm: reader.tensor(&format!("{p}.self_attn.q_norm.weight"), &[c.head_dim])?,
            k_norm: reader.tensor(&format!("{p}.self_attn.k_norm.weight"), &[c.head_dim])?,
            o_proj: reader.tensor(
                &format!("{p}.self_attn.o_proj.weight"),
                &[c.hidden_size, q_dim],
            )?,
            post_attention_layernorm: reader.tensor(
                &format!("{p}.post_attention_layernorm.weight"),
                &[c.hidden_size],
            )?,
            gate_proj: reader.tensor(
                &format!("{p}.mlp.gate_proj.weight"),
                &[c.intermediate_size, c.hidden_size],
            )?,
            up_proj: reader.tensor(
                &format!("{p}.mlp.up_proj.weight"),
                &[c.intermediate_size, c.hidden_size],
            )?,
            down_proj: reader.tensor(
                &format!("{p}.mlp.down_proj.weight"),
                &[c.hidden_size, c.intermediate_size],
            )?,
        });
    }
    let final_layernorm = reader.tensor("model.norm.weight", &[c.hidden_size])?;
    let lm_head = if c.tie_embeddings {
        None
    } else {
        Some(reader.tensor("lm_head.weight", &[c.vocab_size, c.hidden_size])?)
    };
    Ok(Qwen3Weights {
        embed_tokens,
        layers,
        final_layernorm,
        lm_head,
    })
}
struct Reader<'a> {
    shards: BTreeMap<PathBuf, SafeTensors<'a>>,
    names: BTreeMap<String, PathBuf>,
    used: usize,
    limit: usize,
    group_size: usize,
}
impl Reader<'_> {
    fn view(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>, ModelError> {
        let path = self
            .names
            .get(name)
            .ok_or_else(|| error(format!("missing tensor: {name}")))?;
        self.shards
            .get(path)
            .ok_or_else(|| error("missing shard"))?
            .tensor(name)
            .map_err(error)
    }
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, ModelError> {
        let count = shape.iter().try_fold(1usize, |a, &b| product(a, b))?;
        let bytes = product(count, 4)?;
        let required = self
            .used
            .checked_add(bytes)
            .ok_or(ModelError::DimensionOverflow("decoded weights"))?;
        if required > self.limit {
            return Err(ModelError::MemoryLimit {
                required,
                limit: self.limit,
            });
        }
        let view = self.view(name)?;
        let output = if view.dtype() == Dtype::U32 && shape.len() == 2 {
            let rows = shape[0];
            let width = shape[1];
            let group = self.group_size;
            if width % 8 != 0 || width % group != 0 || view.shape() != [rows, width / 8] {
                return Err(error(format!("invalid packed shape: {name}")));
            }
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| error("invalid matrix name"))?;
            let scales = self.view(&format!("{base}.scales"))?;
            let biases = self.view(&format!("{base}.biases"))?;
            if scales.shape() != [rows, width / group] || biases.shape() != scales.shape() {
                return Err(error("invalid quantization parameter shape"));
            }
            let mut output = Vec::new();
            output
                .try_reserve_exact(count)
                .map_err(|_| ModelError::AllocationFailed)?;
            for row in 0..rows {
                for col in 0..width {
                    let word_index = row * (width / 8) + col / 8;
                    let data = &view.data()[word_index * 4..word_index * 4 + 4];
                    let word = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let code = ((word >> ((col % 8) * 4)) & 15) as f32;
                    let param = row * (width / group) + col / group;
                    let value = code * float_at(&scales, param)? + float_at(&biases, param)?;
                    if !value.is_finite() {
                        return Err(error("non-finite dequantized weight"));
                    }
                    output.push(value);
                }
            }
            output
        } else {
            if view.shape() != shape {
                return Err(error(format!(
                    "wrong tensor shape: {name}: {:?} vs {shape:?}",
                    view.shape()
                )));
            }
            let mut output = Vec::new();
            output
                .try_reserve_exact(count)
                .map_err(|_| ModelError::AllocationFailed)?;
            for index in 0..count {
                output.push(float_at(&view, index)?);
            }
            output
        };
        self.used = required;
        Ok(output)
    }
}
fn float_at(view: &safetensors::tensor::TensorView<'_>, index: usize) -> Result<f32, ModelError> {
    let bytes = view.data();
    let value = match view.dtype() {
        Dtype::F32 => {
            let start = product(index, 4)?;
            let data = bytes
                .get(start..start + 4)
                .ok_or_else(|| error("truncated float"))?;
            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        }
        Dtype::F16 | Dtype::BF16 => {
            let start = product(index, 2)?;
            let data = bytes
                .get(start..start + 2)
                .ok_or_else(|| error("truncated half"))?;
            let bits = u16::from_le_bytes([data[0], data[1]]);
            if view.dtype() == Dtype::F16 {
                half::f16::from_bits(bits).to_f32()
            } else {
                half::bf16::from_bits(bits).to_f32()
            }
        }
        dtype => return Err(error(format!("unsupported float dtype: {dtype:?}"))),
    };
    if !value.is_finite() {
        return Err(error("non-finite checkpoint value"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_four_bit_unpacking_and_half_types() {
        let packed = 0x7654_3210u32.to_le_bytes();
        let scale = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let bias = half::bf16::from_f32(-1.0).to_bits().to_le_bytes();
        let views = vec![
            (
                "test.weight",
                safetensors::tensor::TensorView::new(Dtype::U32, vec![1, 1], &packed)
                    .expect("packed"),
            ),
            (
                "test.scales",
                safetensors::tensor::TensorView::new(Dtype::F16, vec![1, 1], &scale)
                    .expect("scale"),
            ),
            (
                "test.biases",
                safetensors::tensor::TensorView::new(Dtype::BF16, vec![1, 1], &bias).expect("bias"),
            ),
        ];
        let bytes = safetensors::tensor::serialize(views, &None).expect("serialize");
        let tensors = SafeTensors::deserialize(&bytes).expect("deserialize");
        let path = PathBuf::from("test");
        let names = tensors
            .names()
            .into_iter()
            .map(|name| (name.to_string(), path.clone()))
            .collect();
        let mut reader = Reader {
            shards: BTreeMap::from([(path, tensors)]),
            names,
            used: 0,
            limit: 32,
            group_size: 8,
        };
        let values = reader.tensor("test.weight", &[1, 8]).expect("dequantize");
        assert_eq!(
            values,
            (0..8).map(|v| v as f32 * 0.5 - 1.0).collect::<Vec<_>>()
        );
        assert!(reader.tensor("test.weight", &[1, 8]).is_err());
    }
}
