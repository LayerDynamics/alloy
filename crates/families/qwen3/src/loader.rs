//! Safetensors loading with typed failures and bounded cumulative byte accounting.
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

/// Limits for one loading operation, not a guarantee about process RSS.
/// max_file_bytes counts all config, index, and safetensors bytes read cumulatively.
/// max_weight_bytes limits decoded f32 tensor storage retained in the model.
/// Defaults allow at most 16 MiB of each representation. JSON objects, names,
/// allocator overhead, and other application memory are additional costs.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub max_file_bytes: usize,
    pub max_weight_bytes: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_file_bytes: 16 * 1024 * 1024,
            max_weight_bytes: 16 * 1024 * 1024,
        }
    }
}
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
impl ModelWeightSource for &PathBuf {
    fn into_weights(self, c: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        self.as_path().into_weights(c)
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
fn invalid(message: impl ToString) -> ModelError {
    ModelError::CheckpointValidation(message.to_string())
}
fn io_error(operation: &'static str, path: &Path, error: std::io::Error) -> ModelError {
    ModelError::CheckpointIo {
        operation,
        path: path.to_path_buf(),
        kind: error.kind(),
        message: error.to_string(),
    }
}
fn json_error(error: serde_json::Error) -> ModelError {
    ModelError::CheckpointFormat {
        format: "json",
        message: error.to_string(),
    }
}
fn tensor_error(error: safetensors::SafeTensorError) -> ModelError {
    ModelError::CheckpointFormat {
        format: "safetensors",
        message: error.to_string(),
    }
}
fn product(a: usize, b: usize) -> Result<usize, ModelError> {
    a.checked_mul(b)
        .ok_or(ModelError::DimensionOverflow("checkpoint tensor"))
}

/// Canonicalize the actual supplied path first: never substitute a sibling artifact.
/// This also makes bare relative filenames independent of empty parent semantics.
fn resolve_source(path: &Path) -> Result<(PathBuf, PathBuf, bool), ModelError> {
    let supplied = path
        .canonicalize()
        .map_err(|e| io_error("canonicalize checkpoint", path, e))?;
    let metadata =
        std::fs::metadata(&supplied).map_err(|e| io_error("stat checkpoint", &supplied, e))?;
    if metadata.is_dir() {
        return Ok((supplied.clone(), supplied, true));
    }
    if !metadata.is_file() {
        return Err(invalid("checkpoint must be a regular file or directory"));
    }
    let root = supplied
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    Ok((supplied, root, false))
}
fn optional_file(path: &Path) -> Result<bool, ModelError> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(true),
        Ok(_) => Err(invalid(format!(
            "not a regular metadata file: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error("stat metadata", path, error)),
    }
}
struct RawBudget {
    used: usize,
    limit: usize,
}
impl RawBudget {
    /// Metadata and shards share this accounting; fixed metadata caps are additional.
    fn read(&mut self, path: &Path, per_file_limit: usize) -> Result<Vec<u8>, ModelError> {
        let mut file =
            std::fs::File::open(path).map_err(|e| io_error("open checkpoint", path, e))?;
        let metadata = file
            .metadata()
            .map_err(|e| io_error("stat open checkpoint", path, e))?;
        if !metadata.is_file() {
            return Err(invalid("checkpoint component is not a regular file"));
        }
        let size = usize::try_from(metadata.len())
            .map_err(|_| ModelError::DimensionOverflow("file length"))?;
        let required = self
            .used
            .checked_add(size)
            .ok_or(ModelError::DimensionOverflow("raw checkpoint bytes"))?;
        if required > self.limit {
            return Err(ModelError::MemoryLimit {
                required,
                limit: self.limit,
            });
        }
        if size > per_file_limit {
            return Err(ModelError::MemoryLimit {
                required: size,
                limit: per_file_limit,
            });
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| ModelError::AllocationFailed)?;
        bytes.resize(size, 0);
        file.read_exact(&mut bytes)
            .map_err(|e| io_error("read checkpoint", path, e))?;
        let mut probe = [0u8; 1];
        if file
            .read(&mut probe)
            .map_err(|e| io_error("check checkpoint length", path, e))?
            != 0
        {
            return Err(invalid("checkpoint changed length while loading"));
        }
        self.used = required;
        Ok(bytes)
    }
}
fn validate_config(c: &Qwen3Config, data: &serde_json::Value) -> Result<Option<usize>, ModelError> {
    if !data.is_object() {
        return Err(invalid("config must be a JSON object"));
    }
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
            return Err(invalid(format!("config mismatch: {name}")));
        }
    }
    if data.get("tie_word_embeddings").and_then(|v| v.as_bool()) != Some(c.tie_embeddings) {
        return Err(invalid("config mismatch: tie_word_embeddings"));
    }
    if let Some(value) = data.get("hidden_act") {
        if value.as_str() != Some("silu") {
            return Err(invalid("only hidden_act=silu is supported"));
        }
    }
    if let Some(value) = data.get("rms_norm_eps") {
        let value = value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| invalid("rms_norm_eps must be a finite number"))?;
        if (value - 1e-6).abs() > 1e-12 {
            return Err(invalid("unsupported rms_norm_eps"));
        }
    }
    if let Some(value) = data.get("rope_theta") {
        let value = value
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| invalid("rope_theta must be a positive finite number"))?;
        if value as f32 != c.rope_theta {
            return Err(invalid("config mismatch: rope_theta"));
        }
    }
    if data.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        return Err(invalid("rope scaling is unsupported"));
    }
    if let Some(value) = data.get("attention_bias") {
        if value.as_bool() != Some(false) {
            return Err(invalid("attention_bias must be false"));
        }
    }
    let first = data.get("quantization").filter(|v| !v.is_null());
    let second = data.get("quantization_config").filter(|v| !v.is_null());
    if let (Some(a), Some(b)) = (first, second) {
        if a != b {
            return Err(invalid("conflicting quantization metadata"));
        }
    }
    if let Some(quant) = first.or(second) {
        if !quant.is_object() || quant.get("bits").and_then(|v| v.as_u64()) != Some(4) {
            return Err(invalid("only affine four-bit quantization is supported"));
        }
        if let Some(mode) = quant.get("mode") {
            if mode.as_str() != Some("affine") {
                return Err(invalid("unsupported quantization mode"));
            }
        }
        let group = quant
            .get("group_size")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| invalid("group_size must be a positive integer"))?;
        return Ok(Some(group));
    }
    Ok(None)
}

/// Load a single file or a directory, with exact index-to-shard inventory matching.
/// Supports F32/F16/BF16 and MLX affine U32-packed four-bit matrices.
/// Absent quantization metadata, infer each matrix's group size from validated
/// scale/bias shapes; declared metadata must agree with those shapes.
pub fn load_weights(
    c: &Qwen3Config,
    path: &Path,
    options: LoadOptions,
) -> Result<Qwen3Weights, ModelError> {
    let (supplied, root, is_directory) = resolve_source(path)?;
    if [
        c.hidden_size,
        c.num_layers,
        c.q_heads,
        c.kv_heads,
        c.head_dim,
        c.intermediate_size,
        c.vocab_size,
        c.max_context,
    ]
    .contains(&0)
    {
        return Err(invalid("zero checkpoint dimension"));
    }
    if c.q_heads % c.kv_heads != 0
        || c.head_dim % 2 != 0
        || !c.rope_theta.is_finite()
        || c.rope_theta <= 0.0
    {
        return Err(invalid("invalid attention configuration"));
    }
    let q_dim = product(c.q_heads, c.head_dim)?;
    let kv_dim = product(c.kv_heads, c.head_dim)?;
    let mut budget = RawBudget {
        used: 0,
        limit: options.max_file_bytes,
    };
    let config_path = root.join("config.json");
    let group_size = if optional_file(&config_path)? {
        let bytes = budget.read(&config_path, 1024 * 1024)?;
        let data = serde_json::from_slice(&bytes).map_err(json_error)?;
        validate_config(c, &data)?
    } else {
        None
    };
    let mut paths = BTreeSet::new();
    let mut index_map = BTreeMap::new();
    let index_path = root.join("model.safetensors.index.json");
    let indexed = is_directory && optional_file(&index_path)?;
    if !is_directory {
        paths.insert(supplied);
    } else if indexed {
        let bytes = budget.read(&index_path, 16 * 1024 * 1024)?;
        let index: serde_json::Value = serde_json::from_slice(&bytes).map_err(json_error)?;
        let map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("missing or empty weight_map"))?;
        for (name, file) in map {
            let filename = file
                .as_str()
                .ok_or_else(|| invalid("invalid shard filename"))?;
            let child = Path::new(filename);
            if child.components().count() != 1
                || child.extension().and_then(|v| v.to_str()) != Some("safetensors")
            {
                return Err(invalid("unsafe shard filename"));
            }
            let candidate = root.join(child);
            let resolved = candidate
                .canonicalize()
                .map_err(|e| io_error("canonicalize shard", &candidate, e))?;
            if resolved.parent() != Some(root.as_path()) {
                return Err(invalid("shard escapes checkpoint directory"));
            }
            index_map.insert(name.clone(), resolved.clone());
            paths.insert(resolved);
        }
    } else {
        paths.insert(root.join("model.safetensors"));
    }
    let mut buffers = BTreeMap::new();
    for path in paths {
        let bytes = budget.read(&path, options.max_file_bytes)?;
        buffers.insert(path, bytes);
    }
    let mut shards = BTreeMap::new();
    let mut names = BTreeMap::new();
    for (path, bytes) in &buffers {
        let tensors = SafeTensors::deserialize(bytes).map_err(tensor_error)?;
        for name in tensors.names() {
            if names.insert(name.to_string(), path.clone()).is_some() {
                return Err(invalid(format!("duplicate tensor: {name}")));
            }
        }
        shards.insert(path.clone(), tensors);
    }
    if indexed && names != index_map {
        return Err(invalid(
            "shard inventory must exactly match weight_map (names and files)",
        ));
    }
    // A declared head in a tied checkpoint is never silently retained or ignored.
    if c.tie_embeddings && names.contains_key("lm_head.weight") {
        return Err(invalid(
            "tied checkpoint must not supply a separate lm_head",
        ));
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
    group_size: Option<usize>,
}
impl Reader<'_> {
    fn view(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>, ModelError> {
        let path = self
            .names
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor: {name}")))?;
        self.shards
            .get(path)
            .ok_or_else(|| invalid("missing shard"))?
            .tensor(name)
            .map_err(tensor_error)
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
            if rows == 0 || width == 0 || width % 8 != 0 || view.shape() != [rows, width / 8] {
                return Err(invalid(format!("invalid packed shape: {name}")));
            }
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("invalid matrix name"))?;
            let scales = self.view(&format!("{base}.scales"))?;
            let biases = self.view(&format!("{base}.biases"))?;
            if scales.shape().len() != 2
                || scales.shape()[0] != rows
                || biases.shape() != scales.shape()
            {
                return Err(invalid("invalid quantization parameter shape"));
            }
            let groups = scales.shape()[1];
            if groups == 0 || width % groups != 0 {
                return Err(invalid("quantization groups do not divide matrix width"));
            }
            let group = width / groups;
            if self.group_size.is_some_and(|declared| declared != group) {
                return Err(invalid(
                    "quantization group_size disagrees with tensor shapes",
                ));
            }
            validate_float_dtype(scales.dtype())?;
            validate_float_dtype(biases.dtype())?;
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
                    let param = row * groups + col / group;
                    let value = code * float_at(&scales, param)? + float_at(&biases, param)?;
                    if !value.is_finite() {
                        return Err(invalid("non-finite dequantized weight"));
                    }
                    output.push(value);
                }
            }
            output
        } else {
            if view.shape() != shape {
                return Err(invalid(format!("wrong tensor shape: {name}")));
            }
            validate_float_dtype(view.dtype())?;
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
fn validate_float_dtype(dtype: Dtype) -> Result<(), ModelError> {
    if matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
        Ok(())
    } else {
        Err(invalid(format!("unsupported float dtype: {dtype:?}")))
    }
}
fn float_at(view: &safetensors::tensor::TensorView<'_>, index: usize) -> Result<f32, ModelError> {
    let bytes = view.data();
    let value = match view.dtype() {
        Dtype::F32 => {
            let start = product(index, 4)?;
            let end = start
                .checked_add(4)
                .ok_or(ModelError::DimensionOverflow("float offset"))?;
            let d = bytes
                .get(start..end)
                .ok_or_else(|| invalid("truncated float"))?;
            f32::from_le_bytes([d[0], d[1], d[2], d[3]])
        }
        Dtype::F16 | Dtype::BF16 => {
            let start = product(index, 2)?;
            let end = start
                .checked_add(2)
                .ok_or(ModelError::DimensionOverflow("half offset"))?;
            let d = bytes
                .get(start..end)
                .ok_or_else(|| invalid("truncated half"))?;
            let bits = u16::from_le_bytes([d[0], d[1]]);
            if view.dtype() == Dtype::F16 {
                half::f16::from_bits(bits).to_f32()
            } else {
                half::bf16::from_bits(bits).to_f32()
            }
        }
        dtype => return Err(invalid(format!("unsupported float dtype: {dtype:?}"))),
    };
    if !value.is_finite() {
        return Err(invalid("non-finite checkpoint value"));
    }
    Ok(value)
}

#[cfg(test)]
mod hardening_unit_tests {
    use super::*;
    #[test]
    fn inferred_groups_and_half_parameters_preserve_values() {
        let packed = 0x7654_3210u32.to_le_bytes();
        let scale: Vec<u8> = [0.5f32, 2.0]
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let bias: Vec<u8> = [-1f32, 3.0]
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let views = vec![
            (
                "test.weight",
                safetensors::tensor::TensorView::new(Dtype::U32, vec![1, 1], &packed)
                    .expect("packed"),
            ),
            (
                "test.scales",
                safetensors::tensor::TensorView::new(Dtype::F16, vec![1, 2], &scale)
                    .expect("scales"),
            ),
            (
                "test.biases",
                safetensors::tensor::TensorView::new(Dtype::BF16, vec![1, 2], &bias)
                    .expect("biases"),
            ),
        ];
        let bytes = safetensors::tensor::serialize(views, &None).expect("serialize");
        let tensors = SafeTensors::deserialize(&bytes).expect("tensors");
        let path = PathBuf::from("shard");
        let names = tensors
            .names()
            .into_iter()
            .map(|n| (n.to_string(), path.clone()))
            .collect();
        let mut reader = Reader {
            shards: BTreeMap::from([(path, tensors)]),
            names,
            used: 0,
            limit: 64,
            group_size: None,
        };
        assert_eq!(
            reader.tensor("test.weight", &[1, 8]).expect("inferred"),
            vec![-1.0, -0.5, 0.0, 0.5, 11.0, 13.0, 15.0, 17.0]
        );
        reader.group_size = Some(64);
        assert!(matches!(
            reader.tensor("test.weight", &[1, 8]),
            Err(ModelError::CheckpointValidation(_))
        ));
        reader.group_size = None;
        reader.limit = 1;
        assert!(matches!(
            reader.tensor("test.weight", &[1, 8]),
            Err(ModelError::MemoryLimit { .. })
        ));
    }
}
