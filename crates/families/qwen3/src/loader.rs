//! Safetensors loading with typed failures, bounded cumulative byte
//! accounting, and optional verification of the exact bytes decoded.
//!
//! Verified loading checks trusted SHA-256 digests against the retained
//! source buffers before deserialization. It does not verify one read
//! and then reopen the component for an unverified second read.
//!
//! Cancellation is observed before filesystem resolution, between source
//! read chunks, between loading stages, and during tensor conversion.

use crate::{
    config::Qwen3Config,
    fixture::{LayerWeights, MatrixWeight, Qwen3Weights},
    model::ModelError,
};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::{Path, PathBuf},
};

/// Limits for one loading operation, not a guarantee about process RSS.
///
/// `max_file_bytes` counts all consumed config, index, and safetensors
/// bytes cumulatively.
///
/// `max_weight_bytes` limits resident weight payload retained in the
/// returned weights (f32 elements, or Affine4 packed+scales+biases).
///
/// Defaults allow at most 16 MiB of each representation. JSON objects,
/// collection headers, allocator overhead, and other application memory
/// are additional costs.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Maximum cumulative source bytes consumed.
    pub max_file_bytes: usize,

    /// Maximum resident weight payload bytes (f32 or Affine4).
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

/// Trusted content identities and cancellation observation for one load.
///
/// The checksum map must come from the caller's trusted artifact policy.
/// Computing these pins from untrusted mutable files immediately before
/// loading does not establish artifact authenticity.
///
/// Every component actually consumed by verified loading must have an
/// entry keyed by the path used by the loader.
///
/// Digests are checked against the retained source buffers subsequently
/// deserialized.
pub struct CheckpointVerification<'a> {
    /// Trusted SHA-256 digest for every consumed checkpoint component.
    pub sha256: &'a BTreeMap<PathBuf, [u8; 32]>,

    /// Observe the caller's existing cancellation state.
    ///
    /// Returning true aborts loading with ModelError::Cancelled.
    pub cancelled: &'a dyn Fn() -> bool,
}

/// Source accepted by ordinary model construction.
///
/// Paths use the ordinary bounded loader. Engine execution requiring a
/// trusted integrity manifest must use load_weights_verified instead.
pub trait ModelWeightSource {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError>;
}

impl ModelWeightSource for Qwen3Weights {
    fn into_weights(self, _: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        Ok(self)
    }
}

impl ModelWeightSource for &Path {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        load_weights(config, self, LoadOptions::default())
    }
}

impl ModelWeightSource for &PathBuf {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        self.as_path().into_weights(config)
    }
}

impl ModelWeightSource for PathBuf {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        self.as_path().into_weights(config)
    }
}

impl ModelWeightSource for &str {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        Path::new(self).into_weights(config)
    }
}

impl ModelWeightSource for String {
    fn into_weights(self, config: &Qwen3Config) -> Result<Qwen3Weights, ModelError> {
        Path::new(&self).into_weights(config)
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

fn product(left: usize, right: usize) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::DimensionOverflow("checkpoint tensor"))
}

fn check_loading_cancelled(cancelled: Option<&dyn Fn() -> bool>) -> Result<(), ModelError> {
    if cancelled.is_some_and(|check| check()) {
        return Err(ModelError::Cancelled);
    }

    Ok(())
}

/// Canonicalize the actual supplied source rather than substituting a
/// sibling artifact.
///
/// This also makes bare relative filenames independent of empty-parent
/// semantics.
fn resolve_source(path: &Path) -> Result<(PathBuf, PathBuf, bool), ModelError> {
    let supplied = path
        .canonicalize()
        .map_err(|error| io_error("canonicalize checkpoint", path, error))?;

    let metadata = std::fs::metadata(&supplied)
        .map_err(|error| io_error("stat checkpoint", &supplied, error))?;

    if metadata.is_dir() {
        return Ok((supplied.clone(), supplied, true));
    }

    if !metadata.is_file() {
        return Err(invalid("checkpoint must be a regular file or directory"));
    }

    let root = supplied
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
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

struct RawBudget<'a> {
    used: usize,
    limit: usize,
    verification: Option<&'a CheckpointVerification<'a>>,
}

impl RawBudget<'_> {
    /// Read a bounded component and optionally verify its retained bytes.
    ///
    /// Metadata and shards share cumulative accounting. Per-file metadata
    /// limits remain additional bounds.
    fn read(&mut self, path: &Path, per_file_limit: usize) -> Result<Vec<u8>, ModelError> {
        let cancelled = self.verification.map(|verification| verification.cancelled);

        check_loading_cancelled(cancelled)?;

        // Reject unpinned components before opening or allocating them.
        let expected_digest = match self.verification {
            Some(verification) => Some(verification.sha256.get(path).ok_or_else(|| {
                invalid(format!(
                    "checkpoint component has no trusted digest: {}",
                    path.display()
                ))
            })?),

            None => None,
        };

        let mut file =
            std::fs::File::open(path).map_err(|error| io_error("open checkpoint", path, error))?;

        let metadata = file
            .metadata()
            .map_err(|error| io_error("stat open checkpoint", path, error))?;

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

        check_loading_cancelled(cancelled)?;

        let mut bytes = Vec::new();

        bytes
            .try_reserve_exact(size)
            .map_err(|_| ModelError::AllocationFailed)?;

        bytes.resize(size, 0);

        let mut hash = expected_digest.map(|_| Sha256::new());

        // Cancellation-check granularity, not an additional allocation
        // or checkpoint-size policy.
        const READ_CHUNK_BYTES: usize = 64 * 1024;

        for chunk in bytes.chunks_mut(READ_CHUNK_BYTES) {
            check_loading_cancelled(cancelled)?;

            file.read_exact(chunk)
                .map_err(|error| io_error("read checkpoint", path, error))?;

            if let Some(hash) = &mut hash {
                hash.update(&*chunk);
            }
        }

        check_loading_cancelled(cancelled)?;

        let mut probe = [0_u8; 1];

        if file
            .read(&mut probe)
            .map_err(|error| io_error("check checkpoint length", path, error))?
            != 0
        {
            return Err(invalid("checkpoint changed length while loading"));
        }

        if let (Some(expected), Some(hash)) = (expected_digest, hash) {
            let observed: [u8; 32] = hash.finalize().into();

            if observed != *expected {
                return Err(invalid(format!(
                    "checkpoint component SHA-256 mismatch: {}",
                    path.display()
                )));
            }
        }

        check_loading_cancelled(cancelled)?;

        self.used = required;

        Ok(bytes)
    }
}

fn validate_config(
    config: &Qwen3Config,
    data: &serde_json::Value,
) -> Result<Option<usize>, ModelError> {
    if !data.is_object() {
        return Err(invalid("config must be a JSON object"));
    }

    for (name, expected) in [
        ("hidden_size", config.hidden_size),
        ("num_hidden_layers", config.num_layers),
        ("num_attention_heads", config.q_heads),
        ("num_key_value_heads", config.kv_heads),
        ("head_dim", config.head_dim),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
    ] {
        if data.get(name).and_then(|value| value.as_u64()) != Some(expected as u64) {
            return Err(invalid(format!("config mismatch: {name}")));
        }
    }

    if data
        .get("tie_word_embeddings")
        .and_then(|value| value.as_bool())
        != Some(config.tie_embeddings)
    {
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
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid("rms_norm_eps must be a finite number"))?;

        if (value - 1e-6).abs() > 1e-12 {
            return Err(invalid("unsupported rms_norm_eps"));
        }
    }

    if let Some(value) = data.get("rope_theta") {
        let value = value
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| invalid("rope_theta must be a positive finite number"))?;

        if value as f32 != config.rope_theta {
            return Err(invalid("config mismatch: rope_theta"));
        }
    }

    if data
        .get("rope_scaling")
        .is_some_and(|value| !value.is_null())
    {
        return Err(invalid("rope scaling is unsupported"));
    }

    if let Some(value) = data.get("attention_bias") {
        if value.as_bool() != Some(false) {
            return Err(invalid("attention_bias must be false"));
        }
    }

    let first = data.get("quantization").filter(|value| !value.is_null());

    let second = data
        .get("quantization_config")
        .filter(|value| !value.is_null());

    if let (Some(first), Some(second)) = (first, second) {
        if first != second {
            return Err(invalid("conflicting quantization metadata"));
        }
    }

    if let Some(quantization) = first.or(second) {
        if !quantization.is_object()
            || quantization.get("bits").and_then(|value| value.as_u64()) != Some(4)
        {
            return Err(invalid("only affine four-bit quantization is supported"));
        }

        if let Some(mode) = quantization.get("mode") {
            if mode.as_str() != Some("affine") {
                return Err(invalid("unsupported quantization mode"));
            }
        }

        let group = quantization
            .get("group_size")
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| invalid("group_size must be a positive integer"))?;

        return Ok(Some(group));
    }

    Ok(None)
}

/// Load a single file or directory with exact index-to-shard inventory
/// matching.
///
/// Supports F32/F16/BF16 and MLX affine U32-packed four-bit matrices.
///
/// Absent quantization metadata, each matrix's group size is inferred
/// from validated scale/bias shapes. Declared metadata must agree.
///
/// This entry point does not require a trusted integrity manifest.
/// Engine paths requiring verified artifacts must use
/// load_weights_verified.
pub fn load_weights(
    config: &Qwen3Config,
    path: &Path,
    options: LoadOptions,
) -> Result<Qwen3Weights, ModelError> {
    load_weights_impl(config, path, options, None)
}

/// Load checkpoint weights using trusted component SHA-256 digests.
///
/// Every consumed configuration, index, and shard must have a trusted
/// digest entry. Hashes cover the exact source buffers subsequently
/// deserialized.
///
/// Cancellation is observed before filesystem resolution, between source
/// read chunks, and during tensor conversion.
///
/// A cancellation observation returns ModelError::Cancelled.
pub fn load_weights_verified<'a>(
    config: &Qwen3Config,
    path: &Path,
    options: LoadOptions,
    verification: &'a CheckpointVerification<'a>,
) -> Result<Qwen3Weights, ModelError> {
    load_weights_impl(config, path, options, Some(verification))
}

fn load_weights_impl<'a>(
    config: &Qwen3Config,
    path: &Path,
    options: LoadOptions,
    verification: Option<&'a CheckpointVerification<'a>>,
) -> Result<Qwen3Weights, ModelError> {
    let cancelled = verification.map(|verification| verification.cancelled);

    check_loading_cancelled(cancelled)?;

    let (supplied, root, is_directory) = resolve_source(path)?;

    if [
        config.hidden_size,
        config.num_layers,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.vocab_size,
        config.max_context,
    ]
    .contains(&0)
    {
        return Err(invalid("zero checkpoint dimension"));
    }

    if config.q_heads % config.kv_heads != 0
        || config.head_dim % 2 != 0
        || !config.rope_theta.is_finite()
        || config.rope_theta <= 0.0
    {
        return Err(invalid("invalid attention configuration"));
    }

    let q_dim = product(config.q_heads, config.head_dim)?;

    let kv_dim = product(config.kv_heads, config.head_dim)?;

    let mut budget = RawBudget {
        used: 0,
        limit: options.max_file_bytes,
        verification,
    };

    let config_path = root.join("config.json");

    let group_size = if optional_file(&config_path)? {
        let bytes = budget.read(&config_path, 1024 * 1024)?;

        let data = serde_json::from_slice(&bytes).map_err(json_error)?;

        validate_config(config, &data)?
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
            .and_then(|value| value.as_object())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("missing or empty weight_map"))?;

        for (name, file) in map {
            check_loading_cancelled(cancelled)?;

            let filename = file
                .as_str()
                .ok_or_else(|| invalid("invalid shard filename"))?;

            let child = Path::new(filename);

            if child.components().count() != 1
                || child.extension().and_then(|value| value.to_str()) != Some("safetensors")
            {
                return Err(invalid("unsafe shard filename"));
            }

            let candidate = root.join(child);

            let resolved = candidate
                .canonicalize()
                .map_err(|error| io_error("canonicalize shard", &candidate, error))?;

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
        check_loading_cancelled(cancelled)?;

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
            "shard inventory must exactly match weight_map \
             (names and files)",
        ));
    }

    // A declared head in a tied checkpoint is never silently retained
    // or ignored.
    if config.tie_embeddings && names.contains_key("lm_head.weight") {
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
        cancelled,
    };

    let embed_tokens = reader.matrix(
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;

    let mut layers = Vec::new();

    layers
        .try_reserve_exact(config.num_layers)
        .map_err(|_| ModelError::AllocationFailed)?;

    for index in 0..config.num_layers {
        check_loading_cancelled(cancelled)?;

        let prefix = format!("model.layers.{index}");

        layers.push(LayerWeights {
            input_layernorm: reader.vector(
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
            )?,

            q_proj: reader.matrix(
                &format!("{prefix}.self_attn.q_proj.weight"),
                &[q_dim, config.hidden_size],
            )?,

            k_proj: reader.matrix(
                &format!("{prefix}.self_attn.k_proj.weight"),
                &[kv_dim, config.hidden_size],
            )?,

            v_proj: reader.matrix(
                &format!("{prefix}.self_attn.v_proj.weight"),
                &[kv_dim, config.hidden_size],
            )?,

            q_norm: reader.vector(
                &format!("{prefix}.self_attn.q_norm.weight"),
                config.head_dim,
            )?,

            k_norm: reader.vector(
                &format!("{prefix}.self_attn.k_norm.weight"),
                config.head_dim,
            )?,

            o_proj: reader.matrix(
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[config.hidden_size, q_dim],
            )?,

            post_attention_layernorm: reader.vector(
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?,

            gate_proj: reader.matrix(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
            )?,

            up_proj: reader.matrix(
                &format!("{prefix}.mlp.up_proj.weight"),
                &[config.intermediate_size, config.hidden_size],
            )?,

            down_proj: reader.matrix(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[config.hidden_size, config.intermediate_size],
            )?,
        });
    }

    let final_layernorm = reader.vector("model.norm.weight", config.hidden_size)?;

    let lm_head = if config.tie_embeddings {
        None
    } else {
        Some(reader.matrix(
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?)
    };

    check_loading_cancelled(cancelled)?;

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
    cancelled: Option<&'a dyn Fn() -> bool>,
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

    fn charge(&mut self, bytes: usize) -> Result<(), ModelError> {
        let required = self
            .used
            .checked_add(bytes)
            .ok_or(ModelError::DimensionOverflow("resident weights"))?;
        if required > self.limit {
            return Err(ModelError::MemoryLimit {
                required,
                limit: self.limit,
            });
        }
        self.used = required;
        Ok(())
    }

    fn vector(&mut self, name: &str, length: usize) -> Result<Vec<f32>, ModelError> {
        check_loading_cancelled(self.cancelled)?;
        let bytes = product(length, std::mem::size_of::<f32>())?;
        self.charge(bytes)?;
        let view = self.view(name)?;
        if view.shape() != [length] {
            return Err(invalid(format!("wrong tensor shape: {name}")));
        }
        validate_float_dtype(view.dtype())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| ModelError::AllocationFailed)?;
        for index in 0..length {
            if index % 4096 == 0 {
                check_loading_cancelled(self.cancelled)?;
            }
            output.push(float_at(&view, index)?);
        }
        check_loading_cancelled(self.cancelled)?;
        Ok(output)
    }

    fn matrix(&mut self, name: &str, shape: &[usize]) -> Result<MatrixWeight, ModelError> {
        check_loading_cancelled(self.cancelled)?;
        if shape.len() != 2 {
            return Err(invalid(format!("matrix requires rank-2 shape: {name}")));
        }
        let rows = shape[0];
        let cols = shape[1];

        let dtype = self.view(name)?.dtype();
        if dtype == Dtype::U32 {
            if rows == 0 || cols == 0 || cols % 8 != 0 {
                return Err(invalid(format!("invalid packed shape: {name}")));
            }
            {
                let view = self.view(name)?;
                if view.shape() != [rows, cols / 8] {
                    return Err(invalid(format!("invalid packed shape: {name}")));
                }
            }
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("invalid matrix name"))?;
            let scale_name = format!("{base}.scales");
            let bias_name = format!("{base}.biases");
            let (groups, group_size) = {
                let scales = self.view(&scale_name)?;
                let biases = self.view(&bias_name)?;
                if scales.shape().len() != 2
                    || scales.shape()[0] != rows
                    || biases.shape() != scales.shape()
                {
                    return Err(invalid("invalid quantization parameter shape"));
                }
                let groups = scales.shape()[1];
                if groups == 0 || cols % groups != 0 {
                    return Err(invalid("quantization groups do not divide matrix width"));
                }
                let group_size = cols / groups;
                if self.group_size.is_some_and(|declared| declared != group_size) {
                    return Err(invalid(
                        "quantization group_size disagrees with tensor shapes",
                    ));
                }
                validate_float_dtype(scales.dtype())?;
                validate_float_dtype(biases.dtype())?;
                (groups, group_size)
            };

            let packed_words = product(rows, cols / 8)?;
            let scale_count = product(rows, groups)?;
            let packed_bytes = product(packed_words, std::mem::size_of::<u32>())?;
            let scale_bytes = product(scale_count, std::mem::size_of::<f32>())?;
            let bias_bytes = product(scale_count, std::mem::size_of::<f32>())?;
            let bytes = packed_bytes
                .checked_add(scale_bytes)
                .and_then(|total| total.checked_add(bias_bytes))
                .ok_or(ModelError::DimensionOverflow("affine4 weights"))?;
            self.charge(bytes)?;

            let mut packed = Vec::new();
            packed
                .try_reserve_exact(packed_words)
                .map_err(|_| ModelError::AllocationFailed)?;
            {
                let view = self.view(name)?;
                for index in 0..packed_words {
                    if index % 4096 == 0 {
                        check_loading_cancelled(self.cancelled)?;
                    }
                    let data = &view.data()[index * 4..index * 4 + 4];
                    packed.push(u32::from_le_bytes([data[0], data[1], data[2], data[3]]));
                }
            }

            let mut scale_values = Vec::new();
            let mut bias_values = Vec::new();
            scale_values
                .try_reserve_exact(scale_count)
                .map_err(|_| ModelError::AllocationFailed)?;
            bias_values
                .try_reserve_exact(scale_count)
                .map_err(|_| ModelError::AllocationFailed)?;
            {
                let scales = self.view(&scale_name)?;
                let biases = self.view(&bias_name)?;
                for index in 0..scale_count {
                    if index % 4096 == 0 {
                        check_loading_cancelled(self.cancelled)?;
                    }
                    let scale = float_at(&scales, index)?;
                    let bias = float_at(&biases, index)?;
                    if !scale.is_finite() || !bias.is_finite() {
                        return Err(invalid("non-finite quantization parameter"));
                    }
                    scale_values.push(scale);
                    bias_values.push(bias);
                }
            }
            check_loading_cancelled(self.cancelled)?;
            return Ok(MatrixWeight::Affine4 {
                packed,
                scales: scale_values,
                biases: bias_values,
                rows,
                cols,
                group_size,
            });
        }

        let count = product(rows, cols)?;
        let bytes = product(count, std::mem::size_of::<f32>())?;
        self.charge(bytes)?;
        let view = self.view(name)?;
        if view.shape() != shape {
            return Err(invalid(format!("wrong tensor shape: {name}")));
        }
        validate_float_dtype(view.dtype())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(count)
            .map_err(|_| ModelError::AllocationFailed)?;
        for index in 0..count {
            if index % 4096 == 0 {
                check_loading_cancelled(self.cancelled)?;
            }
            output.push(float_at(&view, index)?);
        }
        check_loading_cancelled(self.cancelled)?;
        Ok(MatrixWeight::F32(output))
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

            let data = bytes
                .get(start..end)
                .ok_or_else(|| invalid("truncated float"))?;

            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        }

        Dtype::F16 | Dtype::BF16 => {
            let start = product(index, 2)?;

            let end = start
                .checked_add(2)
                .ok_or(ModelError::DimensionOverflow("half offset"))?;

            let data = bytes
                .get(start..end)
                .ok_or_else(|| invalid("truncated half"))?;

            let bits = u16::from_le_bytes([data[0], data[1]]);

            if view.dtype() == Dtype::F16 {
                half::f16::from_bits(bits).to_f32()
            } else {
                half::bf16::from_bits(bits).to_f32()
            }
        }

        dtype => {
            return Err(invalid(format!("unsupported float dtype: {dtype:?}")));
        }
    };

    if !value.is_finite() {
        return Err(invalid("non-finite checkpoint value"));
    }

    Ok(value)
}

#[cfg(test)]
mod hardening_unit_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);

            loop {
                let path = std::env::temp_dir().join(format!(
                    "alloy-loader-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed,),
                ));

                match std::fs::create_dir(&path) {
                    Ok(()) => {
                        return Self(path.canonicalize().expect("canonical test directory"));
                    }

                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        continue;
                    }

                    Err(error) => {
                        panic!("create test directory: {error}");
                    }
                }
            }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn inferred_groups_and_half_parameters_preserve_values() {
        let packed = 0x7654_3210u32.to_le_bytes();

        let scale: Vec<u8> = [0.5f32, 2.0]
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_le_bytes())
            .collect();

        let bias: Vec<u8> = [-1f32, 3.0]
            .iter()
            .flat_map(|&value| half::bf16::from_f32(value).to_bits().to_le_bytes())
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
            .map(|name| (name.to_string(), path.clone()))
            .collect();

        let mut reader = Reader {
            shards: BTreeMap::from([(path, tensors)]),
            names,
            used: 0,
            limit: 64,
            group_size: None,
            cancelled: None,
        };

        let loaded = reader.matrix("test.weight", &[1, 8]).expect("inferred");
        match loaded {
            MatrixWeight::Affine4 {
                packed,
                scales,
                biases,
                rows,
                cols,
                group_size,
            } => {
                assert_eq!(packed, vec![0x7654_3210u32]);
                assert_eq!(scales, vec![0.5, 2.0]);
                assert_eq!(biases, vec![-1.0, 3.0]);
                assert_eq!((rows, cols, group_size), (1, 8, 4));
            }
            MatrixWeight::F32(_) => panic!("expected Affine4"),
        }

        reader.group_size = Some(64);

        assert!(matches!(
            reader.matrix("test.weight", &[1, 8],),
            Err(ModelError::CheckpointValidation(_)),
        ));

        reader.group_size = None;
        reader.used = 0;
        reader.limit = 1;

        assert!(matches!(
            reader.matrix("test.weight", &[1, 8],),
            Err(ModelError::MemoryLimit { .. }),
        ));
    }

    #[test]
    fn verified_read_hashes_the_returned_buffer() {
        let directory = TestDirectory::new();
        let path = directory.0.join("component");

        let data = b"actual bytes retained for parsing";

        std::fs::write(&path, data).expect("write component");

        let digest: [u8; 32] = Sha256::digest(data).into();

        let pins = BTreeMap::from([(path.clone(), digest)]);

        let cancelled = || false;

        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };

        let mut budget = RawBudget {
            used: 0,
            limit: data.len(),
            verification: Some(&verification),
        };

        let returned = budget.read(&path, data.len()).expect("verified read");

        assert_eq!(returned.as_slice(), &data[..],);

        assert_eq!(budget.used, data.len(),);
    }

    #[test]
    fn digest_mismatch_does_not_commit_source_accounting() {
        let directory = TestDirectory::new();
        let path = directory.0.join("component");

        std::fs::write(&path, b"changed bytes").expect("write component");

        let pins = BTreeMap::from([(path.clone(), [0; 32])]);

        let cancelled = || false;

        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };

        let mut budget = RawBudget {
            used: 0,
            limit: 100,
            verification: Some(&verification),
        };

        assert!(matches!(
            budget.read(&path, 100),
            Err(ModelError::CheckpointValidation(_)),
        ));

        assert_eq!(budget.used, 0);
    }

    #[test]
    fn unpinned_component_is_rejected_before_opening() {
        let directory = TestDirectory::new();
        let missing = directory.0.join("not-created");

        let pins = BTreeMap::new();
        let cancelled = || false;

        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };

        let mut budget = RawBudget {
            used: 0,
            limit: 100,
            verification: Some(&verification),
        };

        // The file does not exist. A validation error rather than an
        // I/O error establishes that the missing pin was checked first.
        assert!(matches!(
            budget.read(&missing, 100),
            Err(ModelError::CheckpointValidation(_)),
        ));

        assert_eq!(budget.used, 0);
    }

    #[test]
    fn cancellation_precedes_filesystem_resolution() {
        let directory = TestDirectory::new();
        let missing = directory.0.join("not-created");

        let pins = BTreeMap::new();
        let cancelled = || true;

        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };

        let result = load_weights_verified(
            &Qwen3Config::tiny_fixture(),
            &missing,
            LoadOptions::default(),
            &verification,
        );

        assert!(matches!(result, Err(ModelError::Cancelled),));
    }

    #[test]
    fn verified_read_enforces_source_budget() {
        let directory = TestDirectory::new();
        let path = directory.0.join("component");

        let data = b"source bytes";

        std::fs::write(&path, data).expect("write component");

        let digest: [u8; 32] = Sha256::digest(data).into();

        let pins = BTreeMap::from([(path.clone(), digest)]);

        let cancelled = || false;

        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };

        let mut budget = RawBudget {
            used: 0,
            limit: data.len() - 1,
            verification: Some(&verification),
        };

        assert!(matches!(
            budget.read(&path, data.len()),
            Err(ModelError::MemoryLimit {
                required,
                limit,
            }) if required == data.len()
                && limit == data.len() - 1,
        ));

        assert_eq!(budget.used, 0);
    }
}
