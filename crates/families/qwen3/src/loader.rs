//! Safetensors loading with typed failures, bounded cumulative byte
//! accounting, and optional verification of the exact bytes decoded.
//!
//! Shard headers are validated before allocating weight payloads. Bodies
//! are decoded through one bounded buffer while hashing the exact bytes
//! consumed from the same open descriptors. No weights are published
//! until every complete component digest has passed verification.
//!
//! Cancellation is observed before filesystem resolution, between source
//! read chunks, between loading stages, and during tensor conversion.

use crate::{
    config::Qwen3Config,
    fixture::{LayerWeights, MatrixWeight, Qwen3Weights},
    model::ModelError,
};
use safetensors::{Dtype, tensor::TensorInfo};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

const CONFIG_BYTES_LIMIT: usize = 1024 * 1024;
const INDEX_BYTES_LIMIT: usize = 16 * 1024 * 1024;
const HEADER_BYTES_LIMIT: usize = 8 * 1024 * 1024;
const SHARD_COUNT_LIMIT: usize = 64;
const TENSOR_COUNT_LIMIT: usize = 16_384;
const TENSOR_RANK_LIMIT: usize = 8;
const PATH_BYTES_LIMIT: usize = 4096;
const STREAM_CHUNK_BYTES: usize = 64 * 1024;

/// Conservative loader-owned workspace allowance, excluding the decoded
/// weight payload counted by [`LoadOptions::max_weight_bytes`].
///
/// At most 1 MiB of config JSON, 16 MiB of index JSON, and 8 MiB of
/// aggregate shard headers can be consumed. The 128x allowance covers
/// retained JSON, parser/container capacity, copied names, shape vectors,
/// lookup/selection maps, and construction of the final weight structs.
/// It includes the source buffer alongside JSON arrays whose initial
/// capacity can greatly exceed their number of elements.
/// Tensor count/rank and shard count are additionally capped below;
/// shard paths are interned instead of copied for every index entry.
/// A separate allowance covers 64 paths/descriptors and the single body
/// read buffer. The metadata allowance scales with the cumulative source
/// limit, so a tiny checkpoint does not reserve a production-sized arena.
///
/// The read-phase payload peak is therefore `decoded_weight_bytes +
/// streaming_workspace_bound(max_file_bytes)`: no whole shard, tensor
/// byte buffer, or dense dequantized matrix is retained. This is an
/// admission bound for loader-owned allocations, not a process RSS bound.
pub fn streaming_workspace_bound(max_file_bytes: usize) -> Result<usize, ModelError> {
    let metadata_bytes =
        max_file_bytes.min(CONFIG_BYTES_LIMIT + INDEX_BYTES_LIMIT + HEADER_BYTES_LIMIT);
    metadata_bytes
        .checked_mul(128)
        .and_then(|bytes| bytes.checked_add(SHARD_COUNT_LIMIT * (PATH_BYTES_LIMIT + 1024)))
        .and_then(|bytes| bytes.checked_add(max_file_bytes.min(STREAM_CHUNK_BYTES)))
        .ok_or(ModelError::DimensionOverflow("streaming loader workspace"))
}

/// Limits for one loading operation, not a guarantee about process RSS.
///
/// `max_file_bytes` counts all consumed config, index, and safetensors
/// bytes cumulatively.
///
/// `max_weight_bytes` limits resident weight payload retained in the
/// returned weights (f32 elements, or Affine4 packed+scales+biases).
///
/// Defaults allow at most 16 MiB of each representation. The additional
/// loader workspace is bounded by [`streaming_workspace_bound`]. Other
/// application memory and allocator/process overhead remain separate.
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
/// Metadata digests are checked before parsing. Shard digests cover the
/// exact header and body bytes decoded from one open descriptor, and are
/// all checked before the decoded weights can be returned to a caller.
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

fn product(left: usize, right: usize) -> Result<usize, ModelError> {
    left.checked_mul(right)
        .ok_or(ModelError::DimensionOverflow("checkpoint tensor"))
}

fn dtype_byte_width(dtype: Dtype) -> Result<usize, ModelError> {
    let bits = dtype.bitsize();
    if bits == 0 || !bits.is_multiple_of(8) {
        return Err(invalid(format!(
            "unsupported non-byte-aligned tensor dtype: {dtype:?}"
        )));
    }
    Ok(bits / 8)
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

fn safetensors_error(message: impl ToString) -> ModelError {
    ModelError::CheckpointFormat {
        format: "safetensors",
        message: message.to_string(),
    }
}

/// Only checked header metadata and one descriptor are retained per
/// shard. In particular, this type cannot own the raw tensor body.
struct OpenShard {
    path: Arc<PathBuf>,
    file: File,
    hash: Option<Sha256>,
    expected_digest: Option<[u8; 32]>,
    tensors: Vec<(String, TensorInfo)>,
}

impl OpenShard {
    fn open(
        path: Arc<PathBuf>,
        budget: &mut RawBudget<'_>,
        header_bytes: &mut usize,
        tensor_count: &mut usize,
    ) -> Result<Self, ModelError> {
        let cancelled = budget
            .verification
            .map(|verification| verification.cancelled);
        check_loading_cancelled(cancelled)?;
        // The pin check precedes even opening a component, as it does
        // for config and index reads.
        let expected_digest = match budget.verification {
            Some(verification) => {
                Some(*verification.sha256.get(path.as_path()).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint component has no trusted digest: {}",
                        path.display()
                    ))
                })?)
            }
            None => None,
        };
        let mut file = File::open(path.as_path())
            .map_err(|error| io_error("open checkpoint", &path, error))?;
        let metadata = file
            .metadata()
            .map_err(|error| io_error("stat open checkpoint", &path, error))?;
        if !metadata.is_file() {
            return Err(invalid("checkpoint component is not a regular file"));
        }
        let size = usize::try_from(metadata.len())
            .map_err(|_| ModelError::DimensionOverflow("file length"))?;
        let required = budget
            .used
            .checked_add(size)
            .ok_or(ModelError::DimensionOverflow("raw checkpoint bytes"))?;
        if required > budget.limit {
            return Err(ModelError::MemoryLimit {
                required,
                limit: budget.limit,
            });
        }
        if size < 8 {
            return Err(safetensors_error("safetensors header is too small"));
        }
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|error| io_error("read checkpoint header length", &path, error))?;
        let header_length = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| safetensors_error("safetensors header length overflow"))?;
        let next_header_bytes = header_bytes
            .checked_add(header_length)
            .ok_or(ModelError::DimensionOverflow("checkpoint headers"))?;
        if next_header_bytes > HEADER_BYTES_LIMIT {
            return Err(ModelError::MemoryLimit {
                required: next_header_bytes,
                limit: HEADER_BYTES_LIMIT,
            });
        }
        let data_start = header_length
            .checked_add(8)
            .filter(|&offset| offset <= size)
            .ok_or_else(|| safetensors_error("safetensors header exceeds file length"))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_length)
            .map_err(|_| ModelError::AllocationFailed)?;
        header.resize(header_length, 0);
        let mut hash = expected_digest.map(|_| Sha256::new());
        if let Some(hash) = &mut hash {
            hash.update(length_bytes);
        }
        for chunk in header.chunks_mut(STREAM_CHUNK_BYTES) {
            check_loading_cancelled(cancelled)?;
            file.read_exact(chunk)
                .map_err(|error| io_error("read checkpoint header", &path, error))?;
            if let Some(hash) = &mut hash {
                hash.update(&*chunk);
            }
        }
        if header.first() != Some(&b'{') {
            return Err(safetensors_error(
                "safetensors header must begin with an object",
            ));
        }
        check_loading_cancelled(cancelled)?;
        let metadata: safetensors::tensor::Metadata =
            serde_json::from_slice(&header).map_err(safetensors_error)?;
        let metadata_tensors = metadata.tensors();
        let next_tensor_count = tensor_count
            .checked_add(metadata_tensors.len())
            .ok_or(ModelError::DimensionOverflow("checkpoint tensor count"))?;
        if next_tensor_count > TENSOR_COUNT_LIMIT {
            return Err(invalid("checkpoint exceeds tensor count limit"));
        }
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(metadata_tensors.len())
            .map_err(|_| ModelError::AllocationFailed)?;
        for (name, info) in metadata_tensors {
            check_loading_cancelled(cancelled)?;
            if info.shape.len() > TENSOR_RANK_LIMIT {
                return Err(invalid("checkpoint exceeds tensor rank limit"));
            }
            tensors.push((name, info.clone()));
        }
        tensors.sort_unstable_by_key(|tensor| tensor.1.data_offsets);
        let mut offset = 0usize;
        for (name, info) in &tensors {
            let (start, end) = info.data_offsets;
            if start != offset || end < start {
                return Err(safetensors_error(format!("invalid tensor offsets: {name}")));
            }
            let elements = info
                .shape
                .iter()
                .try_fold(1usize, |count, &dimension| product(count, dimension))?;
            if end - start != product(elements, dtype_byte_width(info.dtype)?)? {
                return Err(safetensors_error(format!(
                    "tensor size does not match shape: {name}"
                )));
            }
            offset = end;
        }
        if offset.checked_add(data_start) != Some(size) {
            return Err(safetensors_error(
                "tensor offsets do not cover the complete file body",
            ));
        }
        check_loading_cancelled(cancelled)?;
        // Charge the complete source once before allocating any decoded
        // payload. An unsuccessful load never publishes this budget.
        budget.used = required;
        *header_bytes = next_header_bytes;
        *tensor_count = next_tensor_count;
        Ok(Self {
            path,
            file,
            hash,
            expected_digest,
            tensors,
        })
    }

    fn read_body(
        &mut self,
        bytes: &mut [u8],
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<(), ModelError> {
        check_loading_cancelled(cancelled)?;
        self.file
            .read_exact(bytes)
            .map_err(|error| io_error("read checkpoint body", &self.path, error))?;
        if let Some(hash) = &mut self.hash {
            hash.update(&*bytes);
        }
        Ok(())
    }

    fn finish(mut self, cancelled: Option<&dyn Fn() -> bool>) -> Result<(), ModelError> {
        check_loading_cancelled(cancelled)?;
        let mut probe = [0u8; 1];
        if self
            .file
            .read(&mut probe)
            .map_err(|error| io_error("check checkpoint length", &self.path, error))?
            != 0
        {
            return Err(invalid("checkpoint changed length while loading"));
        }
        if let (Some(expected), Some(hash)) = (self.expected_digest, self.hash.take()) {
            let observed: [u8; 32] = hash.finalize().into();
            if observed != expected {
                return Err(invalid(format!(
                    "checkpoint component SHA-256 mismatch: {}",
                    self.path.display()
                )));
            }
        }
        check_loading_cancelled(cancelled)
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

    if let Some(value) = data.get("hidden_act")
        && value.as_str() != Some("silu")
    {
        return Err(invalid("only hidden_act=silu is supported"));
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

    if let Some(value) = data.get("attention_bias")
        && value.as_bool() != Some(false)
    {
        return Err(invalid("attention_bias must be false"));
    }

    let first = data.get("quantization").filter(|value| !value.is_null());

    let second = data
        .get("quantization_config")
        .filter(|value| !value.is_null());

    if let (Some(first), Some(second)) = (first, second)
        && first != second
    {
        return Err(invalid("conflicting quantization metadata"));
    }

    if let Some(quantization) = first.or(second) {
        if !quantization.is_object()
            || quantization.get("bits").and_then(|value| value.as_u64()) != Some(4)
        {
            return Err(invalid("only affine four-bit quantization is supported"));
        }

        if let Some(mode) = quantization.get("mode")
            && mode.as_str() != Some("affine")
        {
            return Err(invalid("unsupported quantization mode"));
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

    if supplied.as_os_str().len() > PATH_BYTES_LIMIT || root.as_os_str().len() > PATH_BYTES_LIMIT {
        return Err(invalid("checkpoint path exceeds loader path limit"));
    }

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

    if !config.q_heads.is_multiple_of(config.kv_heads)
        || !config.head_dim.is_multiple_of(2)
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
        let bytes = budget.read(&config_path, CONFIG_BYTES_LIMIT)?;

        let data = serde_json::from_slice(&bytes).map_err(json_error)?;

        validate_config(config, &data)?
    } else {
        None
    };

    let mut paths: BTreeSet<Arc<PathBuf>> = BTreeSet::new();
    let mut index_map: BTreeMap<String, Arc<PathBuf>> = BTreeMap::new();

    let index_path = root.join("model.safetensors.index.json");

    let indexed = is_directory && optional_file(&index_path)?;

    if !is_directory {
        paths.insert(Arc::new(supplied));
    } else if indexed {
        let bytes = budget.read(&index_path, INDEX_BYTES_LIMIT)?;

        let index: serde_json::Value = serde_json::from_slice(&bytes).map_err(json_error)?;

        let map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("missing or empty weight_map"))?;

        if map.len() > TENSOR_COUNT_LIMIT {
            return Err(invalid("checkpoint index exceeds tensor count limit"));
        }

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

            if resolved.as_os_str().len() > PATH_BYTES_LIMIT {
                return Err(invalid("checkpoint path exceeds loader path limit"));
            }
            let interned = if let Some(existing) = paths.get(&resolved) {
                Arc::clone(existing)
            } else {
                if paths.len() == SHARD_COUNT_LIMIT {
                    return Err(invalid("checkpoint exceeds shard count limit"));
                }
                let interned = Arc::new(resolved);
                paths.insert(Arc::clone(&interned));
                interned
            };
            index_map.insert(name.clone(), interned);
        }
    } else {
        paths.insert(Arc::new(root.join("model.safetensors")));
    }

    let mut shards = Vec::new();
    let mut names = BTreeMap::new();
    let mut header_bytes = 0usize;
    let mut tensor_count = 0usize;

    for path in paths {
        check_loading_cancelled(cancelled)?;
        let shard = OpenShard::open(path, &mut budget, &mut header_bytes, &mut tensor_count)?;
        let shard_index = shards.len();
        for (tensor_index, (name, _)) in shard.tensors.iter().enumerate() {
            if names
                .insert(name.clone(), (shard_index, tensor_index))
                .is_some()
            {
                return Err(invalid(format!("duplicate tensor: {name}")));
            }
        }
        shards.push(shard);
    }

    if indexed
        && (names.len() != index_map.len()
            || names
                .iter()
                .any(|(name, &(shard, _))| index_map.get(name) != Some(&shards[shard].path)))
    {
        return Err(invalid(
            "shard inventory must exactly match weight_map \
             (names and files)",
        ));
    }
    drop(index_map);

    // A declared head in a tied checkpoint is never silently retained
    // or ignored.
    if config.tie_embeddings && names.contains_key("lm_head.weight") {
        return Err(invalid(
            "tied checkpoint must not supply a separate lm_head",
        ));
    }

    let mut inventory = Inventory {
        shards: &shards,
        names: &names,
        needed: BTreeSet::new(),
        matrices: BTreeMap::new(),
        used: 0,
        limit: options.max_weight_bytes,
        group_size,
        cancelled,
    };
    inventory.prepare(config, q_dim, kv_dim)?;
    let Inventory {
        needed,
        matrices,
        used,
        ..
    } = inventory;
    drop(names);
    let (tensors, stats) = decode_shards(shards, &needed, options.max_file_bytes, cancelled)?;
    debug_assert_eq!(stats.decoded_payload_bytes, used);
    drop(needed);
    let mut reader = Reader {
        tensors,
        matrices,
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
        Some(reader.matrix("lm_head.weight", &[config.vocab_size, config.hidden_size])?)
    };

    check_loading_cancelled(cancelled)?;

    Ok(Qwen3Weights {
        embed_tokens,
        layers,
        final_layernorm,
        lm_head,
    })
}

#[derive(Clone, Copy)]
struct MatrixSpec {
    rows: usize,
    cols: usize,
    group_size: Option<usize>,
}

/// Header-only validation and one-time payload accounting. This runs to
/// completion before any potentially large tensor allocation is made.
struct Inventory<'a> {
    shards: &'a [OpenShard],
    names: &'a BTreeMap<String, (usize, usize)>,
    needed: BTreeSet<String>,
    matrices: BTreeMap<String, MatrixSpec>,
    used: usize,
    limit: usize,
    group_size: Option<usize>,
    cancelled: Option<&'a dyn Fn() -> bool>,
}

impl Inventory<'_> {
    fn info(&self, name: &str) -> Result<&TensorInfo, ModelError> {
        let &(shard, tensor) = self
            .names
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor: {name}")))?;
        Ok(&self.shards[shard].tensors[tensor].1)
    }

    fn require(&mut self, name: &str, count: usize) -> Result<(), ModelError> {
        if !self.needed.insert(name.to_owned()) {
            return Err(invalid(format!("tensor selected more than once: {name}")));
        }
        let required = self
            .used
            .checked_add(product(count, 4)?)
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

    fn vector(&mut self, name: &str, length: usize) -> Result<(), ModelError> {
        check_loading_cancelled(self.cancelled)?;
        let info = self.info(name)?;
        if info.shape != [length] {
            return Err(invalid(format!("wrong tensor shape: {name}")));
        }
        validate_float_dtype(info.dtype)?;
        self.require(name, length)
    }

    fn matrix(&mut self, name: &str, shape: &[usize]) -> Result<(), ModelError> {
        check_loading_cancelled(self.cancelled)?;
        if shape.len() != 2 {
            return Err(invalid(format!("matrix requires rank-2 shape: {name}")));
        }
        let (rows, cols) = (shape[0], shape[1]);
        let group_size = if self.info(name)?.dtype == Dtype::U32 {
            if rows == 0 || cols == 0 || cols % 8 != 0 || self.info(name)?.shape != [rows, cols / 8]
            {
                return Err(invalid(format!("invalid packed shape: {name}")));
            }
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("invalid matrix name"))?;
            let scale_name = format!("{base}.scales");
            let bias_name = format!("{base}.biases");
            let scales = self.info(&scale_name)?;
            let biases = self.info(&bias_name)?;
            if scales.shape.len() != 2 || scales.shape[0] != rows || biases.shape != scales.shape {
                return Err(invalid("invalid quantization parameter shape"));
            }
            let groups = scales.shape[1];
            if groups == 0 || cols % groups != 0 {
                return Err(invalid("quantization groups do not divide matrix width"));
            }
            let group_size = cols / groups;
            if self
                .group_size
                .is_some_and(|declared| declared != group_size)
            {
                return Err(invalid(
                    "quantization group_size disagrees with tensor shapes",
                ));
            }
            validate_float_dtype(scales.dtype)?;
            validate_float_dtype(biases.dtype)?;
            self.require(name, product(rows, cols / 8)?)?;
            self.require(&scale_name, product(rows, groups)?)?;
            self.require(&bias_name, product(rows, groups)?)?;
            Some(group_size)
        } else {
            let info = self.info(name)?;
            if info.shape != shape {
                return Err(invalid(format!("wrong tensor shape: {name}")));
            }
            validate_float_dtype(info.dtype)?;
            self.require(name, product(rows, cols)?)?;
            None
        };
        self.matrices.insert(
            name.to_owned(),
            MatrixSpec {
                rows,
                cols,
                group_size,
            },
        );
        Ok(())
    }

    fn prepare(
        &mut self,
        config: &Qwen3Config,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<(), ModelError> {
        let minimum_tensors = product(config.num_layers, 11)?
            .checked_add(2 + usize::from(!config.tie_embeddings))
            .ok_or(ModelError::DimensionOverflow("required checkpoint tensors"))?;
        if minimum_tensors > self.names.len() {
            return Err(invalid(
                "checkpoint does not contain the required tensor inventory",
            ));
        }
        self.matrix(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        for layer in 0..config.num_layers {
            check_loading_cancelled(self.cancelled)?;
            let prefix = format!("model.layers.{layer}");
            for (suffix, length) in [
                ("input_layernorm.weight", config.hidden_size),
                ("self_attn.q_norm.weight", config.head_dim),
                ("self_attn.k_norm.weight", config.head_dim),
                ("post_attention_layernorm.weight", config.hidden_size),
            ] {
                self.vector(&format!("{prefix}.{suffix}"), length)?;
            }
            for (suffix, rows, cols) in [
                ("self_attn.q_proj.weight", q_dim, config.hidden_size),
                ("self_attn.k_proj.weight", kv_dim, config.hidden_size),
                ("self_attn.v_proj.weight", kv_dim, config.hidden_size),
                ("self_attn.o_proj.weight", config.hidden_size, q_dim),
                (
                    "mlp.gate_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                (
                    "mlp.up_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                (
                    "mlp.down_proj.weight",
                    config.hidden_size,
                    config.intermediate_size,
                ),
            ] {
                self.matrix(&format!("{prefix}.{suffix}"), &[rows, cols])?;
            }
        }
        self.vector("model.norm.weight", config.hidden_size)?;
        if !config.tie_embeddings {
            self.matrix("lm_head.weight", &[config.vocab_size, config.hidden_size])?;
        }
        check_loading_cancelled(self.cancelled)
    }
}

enum DecodedTensor {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl DecodedTensor {
    fn allocate(dtype: Dtype, count: usize) -> Result<Self, ModelError> {
        if dtype == Dtype::U32 {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| ModelError::AllocationFailed)?;
            Ok(Self::U32(values))
        } else {
            validate_float_dtype(dtype)?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| ModelError::AllocationFailed)?;
            Ok(Self::F32(values))
        }
    }

    fn append(
        &mut self,
        dtype: Dtype,
        bytes: &[u8],
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<(), ModelError> {
        let element_width = dtype_byte_width(dtype)?;
        for (index, element) in bytes.chunks_exact(element_width).enumerate() {
            if index % 4096 == 0 {
                check_loading_cancelled(cancelled)?;
            }
            match self {
                Self::U32(values) => values.push(u32::from_le_bytes(
                    element
                        .try_into()
                        .map_err(|_| invalid("truncated packed element"))?,
                )),
                Self::F32(values) => values.push(float_from_bytes(dtype, element)?),
            }
        }
        Ok(())
    }

    fn payload_bytes(&self) -> usize {
        match self {
            Self::F32(values) => values.len() * 4,
            Self::U32(values) => values.len() * 4,
        }
    }
}

#[derive(Default)]
struct DecodeStats {
    decoded_payload_bytes: usize,
    #[cfg(test)]
    raw_body_buffer_bytes: usize,
    #[cfg(test)]
    body_bytes_read: usize,
    #[cfg(test)]
    body_read_chunks: usize,
}

fn decode_shards(
    shards: Vec<OpenShard>,
    needed: &BTreeSet<String>,
    source_limit: usize,
    cancelled: Option<&dyn Fn() -> bool>,
) -> Result<(BTreeMap<String, DecodedTensor>, DecodeStats), ModelError> {
    let mut scratch = Vec::new();
    let scratch_bytes = source_limit.min(STREAM_CHUNK_BYTES);
    scratch
        .try_reserve_exact(scratch_bytes)
        .map_err(|_| ModelError::AllocationFailed)?;
    scratch.resize(scratch_bytes, 0u8);
    let mut output = BTreeMap::new();
    let mut stats = DecodeStats::default();
    #[cfg(test)]
    {
        stats.raw_body_buffer_bytes = scratch.capacity();
    }
    for mut shard in shards {
        // Move the checked metadata out; each consumed tensor releases
        // its name/shape while the already decoded typed payload stays.
        for (name, info) in std::mem::take(&mut shard.tensors) {
            check_loading_cancelled(cancelled)?;
            let mut remaining = info.data_offsets.1 - info.data_offsets.0;
            let mut decoded = if needed.contains(&name) {
                Some(DecodedTensor::allocate(
                    info.dtype,
                    remaining / dtype_byte_width(info.dtype)?,
                )?)
            } else {
                None
            };
            while remaining > 0 {
                let mut chunk_bytes = remaining.min(scratch.len());
                if decoded.is_some() {
                    chunk_bytes -= chunk_bytes % dtype_byte_width(info.dtype)?;
                }
                if chunk_bytes == 0 {
                    return Err(safetensors_error("read buffer cannot fit a tensor element"));
                }
                let chunk = &mut scratch[..chunk_bytes];
                shard.read_body(chunk, cancelled)?;
                if let Some(decoded) = &mut decoded {
                    decoded.append(info.dtype, chunk, cancelled)?;
                }
                #[cfg(test)]
                {
                    stats.body_bytes_read += chunk_bytes;
                    stats.body_read_chunks += 1;
                }
                remaining -= chunk_bytes;
            }
            if let Some(decoded) = decoded {
                stats.decoded_payload_bytes = stats
                    .decoded_payload_bytes
                    .checked_add(decoded.payload_bytes())
                    .ok_or(ModelError::DimensionOverflow("decoded checkpoint payload"))?;
                output.insert(name, decoded);
            }
        }
        // This checks every body byte, including tensors intentionally
        // skipped by the model. Failure drops all previously decoded
        // vectors instead of returning partially verified weights.
        shard.finish(cancelled)?;
    }
    check_loading_cancelled(cancelled)?;
    Ok((output, stats))
}

/// Assembly only moves owned vectors. The payload was charged exactly
/// once by Inventory and is never cloned or dequantized on this path.
struct Reader<'a> {
    tensors: BTreeMap<String, DecodedTensor>,
    matrices: BTreeMap<String, MatrixSpec>,
    cancelled: Option<&'a dyn Fn() -> bool>,
}

impl Reader<'_> {
    fn floats(&mut self, name: &str) -> Result<Vec<f32>, ModelError> {
        match self.tensors.remove(name) {
            Some(DecodedTensor::F32(values)) => Ok(values),
            _ => Err(invalid(format!("missing decoded float tensor: {name}"))),
        }
    }

    fn vector(&mut self, name: &str, length: usize) -> Result<Vec<f32>, ModelError> {
        check_loading_cancelled(self.cancelled)?;
        let values = self.floats(name)?;
        if values.len() != length {
            return Err(invalid(format!("wrong decoded vector length: {name}")));
        }
        Ok(values)
    }

    fn matrix(&mut self, name: &str, shape: &[usize]) -> Result<MatrixWeight, ModelError> {
        check_loading_cancelled(self.cancelled)?;
        let spec = self
            .matrices
            .remove(name)
            .ok_or_else(|| invalid(format!("missing validated matrix: {name}")))?;
        if shape != [spec.rows, spec.cols] {
            return Err(invalid(format!("wrong decoded matrix shape: {name}")));
        }
        if let Some(group_size) = spec.group_size {
            let packed = match self.tensors.remove(name) {
                Some(DecodedTensor::U32(values)) => values,
                _ => return Err(invalid(format!("missing decoded packed tensor: {name}"))),
            };
            let base = name
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("invalid matrix name"))?;
            Ok(MatrixWeight::Affine4 {
                packed,
                scales: self.floats(&format!("{base}.scales"))?,
                biases: self.floats(&format!("{base}.biases"))?,
                rows: spec.rows,
                cols: spec.cols,
                group_size,
            })
        } else {
            Ok(MatrixWeight::F32(self.floats(name)?))
        }
    }
}

fn validate_float_dtype(dtype: Dtype) -> Result<(), ModelError> {
    if matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
        Ok(())
    } else {
        Err(invalid(format!("unsupported float dtype: {dtype:?}")))
    }
}

fn float_from_bytes(dtype: Dtype, data: &[u8]) -> Result<f32, ModelError> {
    let value = match dtype {
        Dtype::F32 => f32::from_le_bytes(data.try_into().map_err(|_| invalid("truncated float"))?),
        Dtype::F16 | Dtype::BF16 => {
            let bits = u16::from_le_bytes(data.try_into().map_err(|_| invalid("truncated half"))?);
            if dtype == Dtype::F16 {
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

        let bytes = safetensors::tensor::serialize(views, None).expect("serialize");

        let directory = TestDirectory::new();
        let path = directory.0.join("model.safetensors");
        std::fs::write(&path, bytes).expect("write shard");
        let (shards, names) = open_test_shards(&[path], None);
        let mut inventory = Inventory {
            shards: &shards,
            names: &names,
            needed: BTreeSet::new(),
            matrices: BTreeMap::new(),
            used: 0,
            limit: 20,
            group_size: Some(64),
            cancelled: None,
        };
        assert!(matches!(
            inventory.matrix("test.weight", &[1, 8]),
            Err(ModelError::CheckpointValidation(_))
        ));
        inventory.group_size = None;
        inventory.limit = 1;
        assert!(matches!(
            inventory.matrix("test.weight", &[1, 8]),
            Err(ModelError::MemoryLimit { .. })
        ));
        inventory.needed.clear();
        inventory.limit = 20;
        inventory
            .matrix("test.weight", &[1, 8])
            .expect("validate inferred group");
        assert_eq!(
            inventory.used, 20,
            "packed and half parameters are charged once as typed payload"
        );
        let Inventory {
            needed, matrices, ..
        } = inventory;
        let (tensors, stats) = decode_shards(shards, &needed, 1024, None).expect("decode");
        assert_eq!(stats.decoded_payload_bytes, 20);
        let mut reader = Reader {
            tensors,
            matrices,
            cancelled: None,
        };
        match reader.matrix("test.weight", &[1, 8]).expect("inferred") {
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
        assert!(
            reader.tensors.is_empty(),
            "assembly moves every typed buffer"
        );
        assert!(reader.matrices.is_empty());
    }

    fn open_test_shards(
        paths: &[PathBuf],
        verification: Option<&CheckpointVerification<'_>>,
    ) -> (Vec<OpenShard>, BTreeMap<String, (usize, usize)>) {
        let mut budget = RawBudget {
            used: 0,
            limit: usize::MAX,
            verification,
        };
        let mut headers = 0;
        let mut tensors = 0;
        let mut shards = Vec::new();
        let mut names = BTreeMap::new();
        for path in paths {
            let shard = OpenShard::open(
                Arc::new(path.clone()),
                &mut budget,
                &mut headers,
                &mut tensors,
            )
            .expect("open test shard");
            for (tensor, (name, _)) in shard.tensors.iter().enumerate() {
                assert!(names.insert(name.clone(), (shards.len(), tensor)).is_none());
            }
            shards.push(shard);
        }
        (shards, names)
    }

    fn write_tensor_shard(path: &Path, name: &str, dtype: Dtype, count: usize, body: &[u8]) {
        let view = safetensors::tensor::TensorView::new(dtype, vec![count], body).expect("view");
        let bytes = safetensors::tensor::serialize([(name, view)], None).expect("serialize");
        std::fs::write(path, bytes).expect("write shard");
    }

    #[test]
    fn streaming_workspace_scales_for_small_sources_and_caps_metadata() {
        let small = streaming_workspace_bound(1024).expect("small bound");
        let large = streaming_workspace_bound(40 * 1024 * 1024 * 1024).expect("large bound");
        assert_eq!(
            small,
            1024 * 128 + SHARD_COUNT_LIMIT * (PATH_BYTES_LIMIT + 1024) + 1024
        );
        assert!(small < 1024 * 1024);
        assert_eq!(
            large,
            streaming_workspace_bound(usize::MAX).expect("capped bound")
        );
        assert_eq!(large, 3_355_836_416);
    }

    #[test]
    fn streamed_shards_keep_one_small_body_buffer_and_skip_unused_payload() {
        let directory = TestDirectory::new();
        let paths = [
            directory.0.join("one.safetensors"),
            directory.0.join("two.safetensors"),
        ];
        let bytes_per_shard = 5 * STREAM_CHUNK_BYTES + 17;
        for (index, path) in paths.iter().enumerate() {
            write_tensor_shard(
                path,
                &format!("unused.{index}"),
                Dtype::U8,
                bytes_per_shard,
                &vec![0x5a; bytes_per_shard],
            );
        }
        let pins = paths
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    Sha256::digest(std::fs::read(path).expect("read pin input")).into(),
                )
            })
            .collect();
        let cancelled = || false;
        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };
        let (shards, _) = open_test_shards(&paths, Some(&verification));
        let (decoded, stats) = decode_shards(shards, &BTreeSet::new(), usize::MAX, None)
            .expect("stream and verify both complete shards");
        assert!(decoded.is_empty());
        assert_eq!(stats.decoded_payload_bytes, 0);
        assert_eq!(stats.body_bytes_read, 2 * bytes_per_shard);
        assert_eq!(stats.body_read_chunks, 12);
        assert_eq!(stats.raw_body_buffer_bytes, STREAM_CHUNK_BYTES);
        assert!(stats.raw_body_buffer_bytes < bytes_per_shard);
    }

    #[test]
    fn streamed_digest_checks_unused_bytes_before_returning_any_weights() {
        use std::io::{Seek, SeekFrom, Write};
        let directory = TestDirectory::new();
        let kept = directory.0.join("one.safetensors");
        let unused = directory.0.join("two.safetensors");
        write_tensor_shard(&kept, "kept", Dtype::F32, 1, &1.25f32.to_le_bytes());
        write_tensor_shard(&unused, "unused", Dtype::U8, 4, &[1, 2, 3, 4]);
        let paths = [kept, unused.clone()];
        let pins = paths
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    Sha256::digest(std::fs::read(path).expect("read pin input")).into(),
                )
            })
            .collect();
        let cancelled = || false;
        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };
        let (shards, _) = open_test_shards(&paths, Some(&verification));
        let mut changed = std::fs::OpenOptions::new()
            .write(true)
            .open(&unused)
            .expect("open mutation");
        changed.seek(SeekFrom::End(-1)).expect("seek");
        changed.write_all(&[9]).expect("change unused byte");
        let result = decode_shards(shards, &BTreeSet::from(["kept".to_owned()]), 4096, None);
        assert!(
            matches!(result, Err(ModelError::CheckpointValidation(message))
            if message.contains("SHA-256 mismatch"))
        );
    }

    #[test]
    fn streamed_decode_keeps_the_verified_descriptor_after_path_replacement() {
        let directory = TestDirectory::new();
        let path = directory.0.join("model.safetensors");
        write_tensor_shard(&path, "value", Dtype::F32, 1, &1.25f32.to_le_bytes());
        let pins = BTreeMap::from([(
            path.clone(),
            Sha256::digest(std::fs::read(&path).expect("pin bytes")).into(),
        )]);
        let cancelled = || false;
        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };
        let (shards, _) = open_test_shards(std::slice::from_ref(&path), Some(&verification));
        std::fs::rename(&path, directory.0.join("original.safetensors")).expect("replace path");
        write_tensor_shard(&path, "value", Dtype::F32, 1, &9.0f32.to_le_bytes());
        let (mut tensors, _) =
            decode_shards(shards, &BTreeSet::from(["value".to_owned()]), 4096, None)
                .expect("consume only the pinned open file");
        match tensors.remove("value") {
            Some(DecodedTensor::F32(values)) => assert_eq!(values, vec![1.25]),
            _ => panic!("expected float tensor"),
        }
    }

    #[test]
    fn streamed_conversion_cancellation_stops_during_a_tensor_body() {
        use std::{cell::Cell, io::Seek};
        let directory = TestDirectory::new();
        let path = directory.0.join("model.safetensors");
        let body = vec![0u8; 8 * STREAM_CHUNK_BYTES];
        write_tensor_shard(&path, "value", Dtype::F32, body.len() / 4, &body);
        let pins = BTreeMap::from([(
            path.clone(),
            Sha256::digest(std::fs::read(&path).expect("pin bytes")).into(),
        )]);
        let armed = Cell::new(false);
        let checks = Cell::new(0usize);
        let cancelled = || {
            if !armed.get() {
                return false;
            }
            checks.set(checks.get() + 1);
            checks.get() > 8
        };
        let verification = CheckpointVerification {
            sha256: &pins,
            cancelled: &cancelled,
        };
        let (shards, _) = open_test_shards(&[path], Some(&verification));
        let mut position = shards[0]
            .file
            .try_clone()
            .expect("observe shared file position");
        let body_start = position.stream_position().expect("initial position");
        let file_end = position.metadata().expect("metadata").len();
        armed.set(true);
        let result = decode_shards(
            shards,
            &BTreeSet::from(["value".to_owned()]),
            usize::MAX,
            Some(&cancelled),
        );
        assert!(matches!(result, Err(ModelError::Cancelled)));
        let stopped = position.stream_position().expect("cancelled position");
        assert!(
            stopped > body_start && stopped < file_end,
            "must cancel during payload consumption"
        );
    }

    #[test]
    fn safetensors_headers_reject_gaps_size_mismatch_and_trailing_body() {
        let directory = TestDirectory::new();
        let path = directory.0.join("model.safetensors");
        for (shape, offsets, body_length) in [
            (vec![1usize], [4usize, 8], 8usize),
            (vec![2], [0, 4], 4),
            (vec![1], [0, 4], 8),
        ] {
            let header = serde_json::to_vec(&serde_json::json!({"value": {
                "dtype": "F32", "shape": shape, "data_offsets": offsets
            }}))
            .expect("header");
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.resize(bytes.len() + body_length, 0);
            std::fs::write(&path, bytes).expect("write malformed shard");
            let mut budget = RawBudget {
                used: 0,
                limit: 4096,
                verification: None,
            };
            assert!(matches!(
                OpenShard::open(Arc::new(path.clone()), &mut budget, &mut 0, &mut 0),
                Err(ModelError::CheckpointFormat {
                    format: "safetensors",
                    ..
                })
            ));
        }
    }

    #[test]
    fn aggregate_header_limit_applies_across_shards() {
        let directory = TestDirectory::new();
        let header_length = HEADER_BYTES_LIMIT / 2 + 8;
        let mut bytes = (header_length as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        bytes.resize(header_length + 8, b' ');
        let paths = [
            directory.0.join("one.safetensors"),
            directory.0.join("two.safetensors"),
        ];
        for path in &paths {
            std::fs::write(path, &bytes).expect("write padded header");
        }
        let mut budget = RawBudget {
            used: 0,
            limit: usize::MAX,
            verification: None,
        };
        let mut headers = 0;
        let mut tensors = 0;
        let first = OpenShard::open(
            Arc::new(paths[0].clone()),
            &mut budget,
            &mut headers,
            &mut tensors,
        )
        .expect("first header is individually permitted");
        assert!(
            matches!(OpenShard::open(Arc::new(paths[1].clone()), &mut budget, &mut headers, &mut tensors),
            Err(ModelError::MemoryLimit { required, limit }) if required > HEADER_BYTES_LIMIT && limit == HEADER_BYTES_LIMIT)
        );
        first.finish(None).expect("empty body is valid");
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
