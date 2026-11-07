//! Worker-owned MLX KV state with transactional multi-layer updates.
//!
//! Layout per layer:
//! [batch, kv_heads, capacity_tokens, head_dim].
//!
//! K/V updates are functional MLX operations. A transaction stages new
//! arrays without replacing the committed cache. Commit requires every
//! layer, evaluates the staged arrays, synchronizes the update stream,
//! and only then publishes all layer handles and the new logical length.
//!
//! A transaction may provide staged prefix views for constructing the
//! model's attention graph. The model must validate its final execution
//! result before committing the cache transaction.
//!
//! Dropping a transaction does not publish its staged state. It does not
//! cancel native work that a caller may already have evaluated through
//! returned views. Runtime execution ownership must outlive outstanding
//! native work.
//!
//! Arrays remain usable from supported CPU and Metal streams through
//! MLX's unified-memory/dependency model. No unsafe Send/Sync impls are
//! introduced; cache arrays remain confined by mlx-ffi's ownership types.
//!
//! Payload planning excludes allocator overhead, library metadata,
//! attention temporaries, external views, and unrelated live graphs.
//! This module does not acquire engine admission reservations.

use mlx_ffi::{
    Array, DType, Error as MlxError, Stream, bindings as sys,
    ops::{self, OperationError},
};
use std::{fmt, os::raw::c_int, ptr, sync::Arc};

/// Representation identity for this initial full float32 KV layout.
///
/// This is distinct from the existing host-reference f64 representation.
/// A model-family adapter must use an appropriate versioned identity when
/// exposing this layout through the engine.
pub const MLX_KV_REPRESENTATION: &str = "mlx.f32.full.bhtd.v1";

/// Identity of one actually loaded MLX model instance.
///
/// Construct once per load and retain a clone with each cache belonging
/// to that model. Reusing an artifact identifier is not a substitute for
/// matching this instance identity.
#[derive(Clone, Debug)]
pub struct MlxModelId(Arc<()>);

impl MlxModelId {
    /// Create an identity for one new model load.
    pub fn new() -> Self {
        Self(Arc::new(()))
    }

    /// Whether both identities denote the same model load.
    pub fn same_instance(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Default for MlxModelId {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for MlxModelId {
    fn eq(&self, other: &Self) -> bool {
        self.same_instance(other)
    }
}

impl Eq for MlxModelId {}

/// Configuration of a full, bounded float32 multi-layer KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheSpec {
    /// Number of transformer layers.
    pub layers: usize,
    /// Batch dimension.
    pub batch: usize,
    /// Number of KV heads per layer.
    pub kv_heads: usize,
    /// Maximum logical token positions.
    pub capacity_tokens: usize,
    /// Scalar values in each head.
    pub head_dim: usize,
}

/// Checked tensor-payload planning information.
///
/// These values are not allocator measurements or acquired reservations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPayloadPlan {
    /// Payload of all committed K/V arrays.
    pub resident_bytes: u64,

    /// Conservative payload for old plus new full-cache versions during
    /// one all-layer functional update.
    ///
    /// Additional graph temporaries and externally retained versions
    /// require separate accounting.
    pub update_overlap_bytes: u64,
}

/// Host and native tensor-payload bounds for transferring one KV block.
///
/// A caller must reserve these bytes before exporting or restoring the
/// block. They are additional to the cache's resident/update allowance.
/// Allocator padding, metadata, serialized bytes, and externally retained
/// native graphs require separate accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBlockPayloadPlan {
    /// Logical float32 K/V payload, excluding spare vector capacity.
    pub host_bytes: u64,
    /// Conservative native payload for contiguous readback or input arrays.
    pub native_transfer_bytes: u64,
    /// Host plus native transfer payload that may be live together.
    pub transfer_overlap_bytes: u64,
}

/// One layer of portable, row-major float32 KV data.
///
/// Each vector has shape [batch, kv_heads, tokens, head_dim] according
/// to its containing [`KvBlock`]. Keys already include the model's
/// normalization and positional encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct KvBlockLayer {
    /// Normalized and position-encoded key scalars.
    pub keys: Vec<f32>,
    /// Value scalars for the same token positions.
    pub values: Vec<f32>,
}

/// A bounded host snapshot of committed K/V positions across every layer.
///
/// This value carries tensor data, not authority to access an existing
/// cache. Restore checks the destination's actual loaded-model identity.
/// The caller must separately validate the persisted model artifact and
/// representation identity before using this data with a newly loaded
/// model instance.
#[derive(Clone, Debug, PartialEq)]
pub struct KvBlock {
    /// Inclusive absolute token position of the block.
    pub start: usize,
    /// Number of token positions in this block; must be positive.
    pub tokens: usize,
    /// Source cache shape and capacity, excluding loaded-model identity.
    pub spec: KvCacheSpec,
    /// Exactly one K/V pair per source cache layer, in layer order.
    pub layers: Vec<KvBlockLayer>,
}

impl KvBlock {
    /// Validate this host block and return its checked payload bounds.
    ///
    /// Checks all metadata, vector lengths, positions, and finite values
    /// without allocating or starting native work. Native array creation
    /// must never precede these exact vector-length checks.
    ///
    /// Reports logical tensor payload, not retained vector capacity. After
    /// [`Self::truncate_tokens`], keep the original host-buffer allowance
    /// until this block is dropped; truncation does not shrink allocations.
    pub fn payload_plan(&self) -> KvCacheResult<KvBlockPayloadPlan> {
        let plan = self.spec.block_payload_plan(self.tokens)?;
        let end = self
            .start
            .checked_add(self.tokens)
            .ok_or(KvCacheError::InvalidInput(
                "KV block position arithmetic overflow",
            ))?;
        if end > self.spec.capacity_tokens {
            return Err(KvCacheError::InvalidInput(
                "KV block exceeds source cache capacity",
            ));
        }
        if self.layers.len() != self.spec.layers {
            return Err(KvCacheError::InvalidInput(
                "KV block must contain every configured layer exactly once",
            ));
        }
        let expected = self.spec.block_tensor_values(self.tokens)?;
        for layer in &self.layers {
            if layer.keys.len() != expected || layer.values.len() != expected {
                return Err(KvCacheError::InvalidInput(
                    "KV block vectors must match the configured float32 chunk shape",
                ));
            }
            if layer
                .keys
                .iter()
                .chain(&layer.values)
                .any(|value| !value.is_finite())
            {
                return Err(KvCacheError::InvalidInput(
                    "KV block contains non-finite values",
                ));
            }
        }
        Ok(plan)
    }

    /// Retain the first `token_count` positions of every batch/head row.
    ///
    /// The count must be positive and no greater than the current count.
    /// Validates the complete block and all arithmetic before modifying
    /// either vector. Invalid requests leave data and metadata unchanged.
    ///
    /// Uses overlapping in-place copies and truncation, without allocating
    /// a second host block or shrinking existing vector capacities. Keep
    /// the original host-buffer reservation until the block is dropped.
    /// Absolute start and source cache shape/capacity remain unchanged.
    pub fn truncate_tokens(&mut self, token_count: usize) -> KvCacheResult<()> {
        self.payload_plan()?;
        if token_count == 0 || token_count > self.tokens {
            return Err(KvCacheError::InvalidInput(
                "KV block truncation requires a positive non-increasing token count",
            ));
        }
        if token_count == self.tokens {
            return Ok(());
        }
        let rows =
            self.spec
                .batch
                .checked_mul(self.spec.kv_heads)
                .ok_or(KvCacheError::InvalidInput(
                    "KV block row-count arithmetic overflow",
                ))?;
        let source_stride =
            self.tokens
                .checked_mul(self.spec.head_dim)
                .ok_or(KvCacheError::InvalidInput(
                    "KV block source-stride arithmetic overflow",
                ))?;
        let target_stride =
            token_count
                .checked_mul(self.spec.head_dim)
                .ok_or(KvCacheError::InvalidInput(
                    "KV block target-stride arithmetic overflow",
                ))?;
        let target_values = self.spec.block_tensor_values(token_count)?;

        // Full validation proves every source range is within the old
        // vector and every target range precedes it. Ascending row order
        // cannot overwrite a later row before that row has been copied.
        // No fallible operations or allocations remain after this point.
        for layer in &mut self.layers {
            for values in [&mut layer.keys, &mut layer.values] {
                for row in 1..rows {
                    let source = row * source_stride;
                    let target = row * target_stride;
                    values.copy_within(source..source + target_stride, target);
                }
                values.truncate(target_values);
            }
        }
        self.tokens = token_count;
        Ok(())
    }
}

/// Failure managing actual MLX KV state.
#[derive(Debug)]
pub enum KvCacheError {
    /// Invalid shape, position, or transaction usage.
    InvalidInput(&'static str),

    /// A cache operation used another model's identity.
    ForeignModel,

    /// A prior native execution/completion failure invalidated this cache.
    Invalidated,

    /// An all-layer transaction was committed before all layers were staged.
    Incomplete {
        /// Required layer count.
        expected_layers: usize,
        /// Successfully staged layer count.
        staged_layers: usize,
    },

    /// Payload planning or measured backing buffers exceed the allowance.
    PayloadLimit {
        /// Required old/new overlap allowance.
        required_bytes: u64,
        /// Supplied payload allowance.
        limit_bytes: u64,
    },

    /// A Rust metadata allocation failed.
    AllocationFailed(&'static str),

    /// A tensor operation failed.
    Operation(OperationError),

    /// A native MLX call or completion operation failed.
    Mlx(MlxError),

    /// Native backing-buffer measurement failed.
    Memory(mlx_ffi::memory::MemoryError),
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => {
                write!(formatter, "invalid MLX KV operation: {message}")
            }
            Self::ForeignModel => {
                write!(formatter, "MLX KV cache belongs to another model load")
            }
            Self::Invalidated => {
                write!(formatter, "MLX KV cache was invalidated by native failure")
            }
            Self::Incomplete {
                expected_layers,
                staged_layers,
            } => write!(
                formatter,
                "incomplete MLX KV transaction: staged {staged_layers} of \
                 {expected_layers} layers"
            ),
            Self::PayloadLimit {
                required_bytes,
                limit_bytes,
            } => write!(
                formatter,
                "MLX KV payload plan requires {required_bytes} bytes; \
                 supplied allowance is {limit_bytes}"
            ),
            Self::AllocationFailed(operation) => {
                write!(formatter, "allocation failed during {operation}")
            }
            Self::Operation(error) => write!(formatter, "{error}"),
            Self::Mlx(error) => write!(formatter, "{error}"),
            Self::Memory(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for KvCacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::Mlx(error) => Some(error),
            Self::Memory(error) => Some(error),
            _ => None,
        }
    }
}

impl From<OperationError> for KvCacheError {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

impl From<MlxError> for KvCacheError {
    fn from(error: MlxError) -> Self {
        Self::Mlx(error)
    }
}

/// Result of an MLX cache operation.
pub type KvCacheResult<T> = Result<T, KvCacheError>;

impl KvCacheSpec {
    /// Validate dimensions and compute payload/overlap bounds.
    pub fn payload_plan(self) -> KvCacheResult<KvPayloadPlan> {
        if [
            self.layers,
            self.batch,
            self.kv_heads,
            self.capacity_tokens,
            self.head_dim,
        ]
        .contains(&0)
        {
            return Err(KvCacheError::InvalidInput(
                "all cache dimensions must be positive",
            ));
        }

        for dimension in [
            self.batch,
            self.kv_heads,
            self.capacity_tokens,
            self.head_dim,
        ] {
            c_int::try_from(dimension)
                .map_err(|_| KvCacheError::InvalidInput("cache dimension exceeds C int"))?;
        }

        let mut resident = 1_u64;
        for factor in [
            self.layers,
            self.batch,
            self.kv_heads,
            self.capacity_tokens,
            self.head_dim,
            2,
            std::mem::size_of::<f32>(),
        ] {
            let factor = u64::try_from(factor)
                .map_err(|_| KvCacheError::InvalidInput("cache size cannot fit u64"))?;
            resident = resident
                .checked_mul(factor)
                .ok_or(KvCacheError::InvalidInput(
                    "cache payload arithmetic overflow",
                ))?;
        }

        let overlap = resident.checked_mul(2).ok_or(KvCacheError::InvalidInput(
            "cache overlap arithmetic overflow",
        ))?;

        Ok(KvPayloadPlan {
            resident_bytes: resident,
            update_overlap_bytes: overlap,
        })
    }

    /// Plan bounded host/native payload for one nonempty token block.
    ///
    /// No vectors or native arrays are allocated. The caller chooses the
    /// block size and acquires admission for this transfer allowance before
    /// invoking export or restore. Restore also needs the existing full
    /// cache update-overlap allowance.
    pub fn block_payload_plan(self, tokens: usize) -> KvCacheResult<KvBlockPayloadPlan> {
        self.payload_plan()?;
        if tokens == 0 || tokens > self.capacity_tokens {
            return Err(KvCacheError::InvalidInput(
                "KV block token count must fit the nonempty cache capacity",
            ));
        }
        self.block_tensor_values(tokens)?;
        let block = Self {
            capacity_tokens: tokens,
            ..self
        }
        .payload_plan()?;
        Ok(KvBlockPayloadPlan {
            host_bytes: block.resident_bytes,
            native_transfer_bytes: block.resident_bytes,
            transfer_overlap_bytes: block.update_overlap_bytes,
        })
    }

    fn block_tensor_values(self, tokens: usize) -> KvCacheResult<usize> {
        let values =
            self.chunk_shape(tokens)
                .into_iter()
                .try_fold(1_usize, |count, dimension| {
                    count
                        .checked_mul(dimension)
                        .ok_or(KvCacheError::InvalidInput(
                            "KV block element-count arithmetic overflow",
                        ))
                })?;
        let bytes =
            values
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(KvCacheError::InvalidInput(
                    "KV block vector byte-count arithmetic overflow",
                ))?;
        if bytes > isize::MAX as usize {
            return Err(KvCacheError::InvalidInput(
                "KV block tensor exceeds the host slice size limit",
            ));
        }
        Ok(values)
    }

    fn native_shape(self) -> KvCacheResult<[c_int; 4]> {
        self.payload_plan()?;

        Ok([
            c_int::try_from(self.batch)
                .map_err(|_| KvCacheError::InvalidInput("batch exceeds C int"))?,
            c_int::try_from(self.kv_heads)
                .map_err(|_| KvCacheError::InvalidInput("head count exceeds C int"))?,
            c_int::try_from(self.capacity_tokens)
                .map_err(|_| KvCacheError::InvalidInput("capacity exceeds C int"))?,
            c_int::try_from(self.head_dim)
                .map_err(|_| KvCacheError::InvalidInput("head width exceeds C int"))?,
        ])
    }

    fn chunk_shape(self, tokens: usize) -> [usize; 4] {
        [self.batch, self.kv_heads, tokens, self.head_dim]
    }
}

struct LayerKv {
    keys: Array,
    values: Array,
}

/// A coherent immutable view of one layer's valid K/V prefix.
///
/// Both arrays describe the same logical positions. Holding a view may
/// keep an older native allocation version alive after later updates.
pub struct KvPrefix {
    keys: Array,
    values: Array,
    start: usize,
    end: usize,
}

impl KvPrefix {
    /// Borrow normalized/rotated keys for the selected positions.
    pub fn keys(&self) -> &Array {
        &self.keys
    }

    /// Borrow values for the same selected positions.
    pub fn values(&self) -> &Array {
        &self.values
    }

    /// Inclusive starting logical position.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Exclusive ending logical position.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Number of selected token positions.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether no positions are selected.
    ///
    /// Current constructors reject empty slices.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Full multi-layer MLX KV state owned by one loaded model instance.
///
/// This type is intentionally not Clone. Native arrays retain mlx-ffi's
/// thread-affine ownership restrictions.
pub struct MlxKvCache {
    owner: MlxModelId,
    spec: KvCacheSpec,
    plan: KvPayloadPlan,
    max_buffer_overlap_bytes: u64,
    layers: Vec<LayerKv>,
    length: usize,
    healthy: bool,
}

impl MlxKvCache {
    /// Allocate and materialize full float32 K/V arrays.
    ///
    /// max_payload_bytes must cover planned AND measured buffer overlap,
    /// including provisioned allocator padding, not merely the
    /// committed payload. It is a planning limit, not an allocator-level
    /// hard memory limit or an engine reservation.
    ///
    /// Initial arrays are created through native MLX full operations;
    /// no full-size Rust zero buffer is constructed.
    pub fn new(
        owner: &MlxModelId,
        spec: KvCacheSpec,
        max_payload_bytes: u64,
        stream: &Stream,
    ) -> KvCacheResult<Self> {
        let plan = spec.payload_plan()?;
        if plan.update_overlap_bytes > max_payload_bytes {
            return Err(KvCacheError::PayloadLimit {
                required_bytes: plan.update_overlap_bytes,
                limit_bytes: max_payload_bytes,
            });
        }

        let shape = spec.native_shape()?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(spec.layers)
            .map_err(|_| KvCacheError::AllocationFailed("cache layer metadata"))?;

        for _ in 0..spec.layers {
            layers.push(LayerKv {
                keys: native_zeros(&shape, stream)?,
                values: native_zeros(&shape, stream)?,
            });
        }

        // Constructor success means actual initial storage was evaluated,
        // not merely that lazy allocation graphs were constructed.
        let mut failure = None;
        for layer in &layers {
            record_native_failure(&mut failure, layer.keys.eval());
            record_native_failure(&mut failure, layer.values.eval());
        }
        record_native_failure(&mut failure, stream.synchronize());

        if let Some(error) = failure {
            return Err(KvCacheError::Mlx(error));
        }

        let cache = Self {
            owner: owner.clone(),
            spec,
            plan,
            max_buffer_overlap_bytes: max_payload_bytes,
            layers,
            length: 0,
            healthy: true,
        };
        let required = cache
            .buffer_bytes()?
            .checked_mul(2)
            .ok_or(KvCacheError::InvalidInput("measured KV overlap overflow"))?;
        if required > max_payload_bytes {
            return Err(KvCacheError::PayloadLimit {
                required_bytes: required,
                limit_bytes: max_payload_bytes,
            });
        }
        Ok(cache)
    }

    /// Loaded-model identity owning this cache.
    pub fn owner(&self) -> &MlxModelId {
        &self.owner
    }

    /// Exact fixed cache configuration.
    pub fn spec(&self) -> KvCacheSpec {
        self.spec
    }

    /// Tensor-payload plan, not allocator measurements.
    pub fn payload_plan(&self) -> KvPayloadPlan {
        self.plan
    }

    /// Number of committed logical positions.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether no logical positions are committed.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Whether no native execution/completion failure has invalidated it.
    pub fn is_healthy(&self) -> bool {
        self.healthy
    }

    /// Measure unique backing buffers of the currently committed K/V arrays.
    ///
    /// Excludes allocator cache, externally held older views, staged
    /// transactions, and attention workspace. No payload fallback occurs.
    pub fn buffer_bytes(&self) -> KvCacheResult<u64> {
        if !self.healthy {
            return Err(KvCacheError::Invalidated);
        }
        measure_layer_buffers(self.layers.iter(), self.layers.len())
    }

    fn check_owner(&self, owner: &MlxModelId) -> KvCacheResult<()> {
        if !self.owner.same_instance(owner) {
            return Err(KvCacheError::ForeignModel);
        }
        if !self.healthy {
            return Err(KvCacheError::Invalidated);
        }
        Ok(())
    }

    /// Reset logical visibility without reallocating or zeroing K/V.
    ///
    /// Old bytes remain physically present but cannot be obtained through
    /// this cache's checked logical-view API.
    ///
    /// Reset does not recover a cache invalidated by native failure.
    pub fn reset(&mut self, owner: &MlxModelId) -> KvCacheResult<()> {
        self.check_owner(owner)?;
        self.length = 0;
        Ok(())
    }

    /// Shorten the currently valid full prefix.
    ///
    /// Positions beyond the new length become logically inaccessible.
    /// Previously discarded positions cannot be restored by increasing
    /// the length through this operation.
    pub fn truncate(&mut self, owner: &MlxModelId, tokens: usize) -> KvCacheResult<()> {
        self.check_owner(owner)?;
        if tokens > self.length {
            return Err(KvCacheError::InvalidInput(
                "truncate cannot increase logical cache length",
            ));
        }
        self.length = tokens;
        Ok(())
    }

    /// Obtain a coherent committed K/V range for one layer.
    ///
    /// Bounds are checked against logical length, not physical capacity.
    pub fn view(
        &self,
        owner: &MlxModelId,
        layer: usize,
        start: usize,
        end: usize,
        stream: &Stream,
    ) -> KvCacheResult<KvPrefix> {
        self.check_owner(owner)?;

        if start >= end || end > self.length {
            return Err(KvCacheError::InvalidInput(
                "cache view must select a nonempty committed logical range",
            ));
        }

        let layer = self.layers.get(layer).ok_or(KvCacheError::InvalidInput(
            "cache layer index is out of range",
        ))?;

        prefix_view(layer, start, end, stream)
    }

    /// Copy only a nonempty committed token range into portable host data.
    ///
    /// Call [`KvCacheSpec::block_payload_plan`] and acquire the transfer
    /// allowance before this method. Each layer is sliced first; the safe
    /// MLX readback helper makes that view row-major contiguous before
    /// copying. No full-cache host buffer is constructed.
    ///
    /// A native construction, evaluation, or completion failure leaves the
    /// committed handles/length unchanged and invalidates this cache.
    pub fn export_block(
        &mut self,
        owner: &MlxModelId,
        start: usize,
        end: usize,
        stream: &Stream,
    ) -> KvCacheResult<KvBlock> {
        self.check_owner(owner)?;
        if start >= end || end > self.length {
            return Err(KvCacheError::InvalidInput(
                "KV block export must select a nonempty committed logical range",
            ));
        }
        let tokens = end - start;
        self.spec.block_payload_plan(tokens)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.spec.layers)
            .map_err(|_| KvCacheError::AllocationFailed("KV block layer metadata"))?;

        let result = (|| {
            for layer in &self.layers {
                let view = prefix_view(layer, start, end, stream)?;
                let keys = ops::to_vec_f32(view.keys(), stream)?;
                let values = ops::to_vec_f32(view.values(), stream)?;
                if keys.iter().chain(&values).any(|value| !value.is_finite()) {
                    return Err(KvCacheError::InvalidInput(
                        "KV block contains non-finite values",
                    ));
                }
                layers.push(KvBlockLayer { keys, values });
            }
            Ok(KvBlock {
                start,
                tokens,
                spec: self.spec,
                layers,
            })
        })();
        result.map_err(|error| self.failed_block_transfer(error, stream))
    }

    /// Transactionally append a validated host block to this cache.
    ///
    /// The block must begin at the current committed length. Tensor shape
    /// facts must match; source and destination capacities may differ if
    /// both contain the block's ending position. The destination owner is
    /// checked independently of the instance from which data was exported.
    ///
    /// All layers and values are validated before native array creation.
    /// The existing all-layer append transaction publishes handles and
    /// length only after successful native completion and overlap checks.
    /// Returns the new committed length only after that publication.
    ///
    /// Call [`KvCacheSpec::block_payload_plan`] and reserve transfer payload
    /// plus the full-cache update allowance before invoking this method.
    /// Native failures preserve old handles/length and invalidate the cache.
    pub fn restore_block(
        &mut self,
        owner: &MlxModelId,
        block: &KvBlock,
        stream: &Stream,
    ) -> KvCacheResult<usize> {
        self.check_owner(owner)?;
        block.payload_plan()?;
        if block.start != self.length {
            return Err(KvCacheError::InvalidInput(
                "KV block restore must append at the committed logical length",
            ));
        }
        if block.spec.layers != self.spec.layers
            || block.spec.batch != self.spec.batch
            || block.spec.kv_heads != self.spec.kv_heads
            || block.spec.head_dim != self.spec.head_dim
        {
            return Err(KvCacheError::InvalidInput(
                "KV block tensor shape does not match the destination cache",
            ));
        }
        let end = block
            .start
            .checked_add(block.tokens)
            .ok_or(KvCacheError::InvalidInput(
                "KV block position arithmetic overflow",
            ))?;
        if end > self.spec.capacity_tokens {
            return Err(KvCacheError::InvalidInput(
                "KV block restore exceeds destination capacity",
            ));
        }
        let mut shape = self.spec.native_shape()?;
        shape[2] = c_int::try_from(block.tokens)
            .map_err(|_| KvCacheError::InvalidInput("KV block tokens exceed C int"))?;
        let result = (|| {
            let mut append = self.begin_append(owner, block.tokens, stream)?;
            for (index, layer) in block.layers.iter().enumerate() {
                // payload_plan validated both exact lengths before this
                // safe wrapper enters its copying native constructor.
                let keys = Array::from_f32_slice(&layer.keys, &shape)?;
                let values = Array::from_f32_slice(&layer.values, &shape)?;
                append.stage_layer(index, &keys, &values)?;
            }
            append.commit()?;
            Ok(end)
        })();
        result.map_err(|error| self.failed_block_transfer(error, stream))
    }

    fn failed_block_transfer(&mut self, error: KvCacheError, stream: &Stream) -> KvCacheError {
        let native_failure = matches!(
            error,
            KvCacheError::Mlx(_)
                | KvCacheError::Operation(OperationError::Mlx(_))
                | KvCacheError::Memory(_)
        );
        if native_failure {
            self.healthy = false;
        }
        // Readback may fail before its own completion call. Finish any
        // scheduled transfer work before returning, even after failure.
        if let Err(completion) = stream.synchronize() {
            self.healthy = false;
            if !native_failure {
                return KvCacheError::Mlx(completion);
            }
        }
        error
    }

    /// Begin one nonempty all-layer append.
    ///
    /// The exclusive borrow prevents another cache mutation while this
    /// transaction is active. New logical positions become visible only
    /// when commit succeeds.
    pub fn begin_append<'a>(
        &'a mut self,
        owner: &MlxModelId,
        tokens: usize,
        stream: &'a Stream,
    ) -> KvCacheResult<KvAppend<'a>> {
        self.check_owner(owner)?;

        if tokens == 0 {
            return Err(KvCacheError::InvalidInput(
                "cache append requires a nonempty chunk",
            ));
        }

        let end = self
            .length
            .checked_add(tokens)
            .ok_or(KvCacheError::InvalidInput(
                "cache position arithmetic overflow",
            ))?;
        if end > self.spec.capacity_tokens {
            return Err(KvCacheError::InvalidInput(
                "cache append exceeds allocated capacity",
            ));
        }

        let mut staged = Vec::new();
        staged
            .try_reserve_exact(self.spec.layers)
            .map_err(|_| KvCacheError::AllocationFailed("cache transaction metadata"))?;
        staged.resize_with(self.spec.layers, || None);

        let start = self.length;

        Ok(KvAppend {
            cache: self,
            stream,
            staged,
            start,
            end,
            tokens,
        })
    }
}

/// An exclusive staged update spanning every cache layer.
///
/// Dropping this value without commit leaves committed handles and
/// logical length unchanged.
///
/// It does not interrupt or cancel native work the caller may already
/// have scheduled using staged views.
#[must_use = "stage every layer and commit, or drop to abandon the append"]
pub struct KvAppend<'a> {
    cache: &'a mut MlxKvCache,
    stream: &'a Stream,
    staged: Vec<Option<LayerKv>>,
    start: usize,
    end: usize,
    tokens: usize,
}

impl KvAppend<'_> {
    /// First absolute position of this query chunk.
    pub fn start_position(&self) -> usize {
        self.start
    }

    /// Exclusive ending absolute position after successful commit.
    pub fn end_position(&self) -> usize {
        self.end
    }

    /// Number of positions appended by this transaction.
    pub fn token_count(&self) -> usize {
        self.tokens
    }

    /// Stage one layer's K/V pair and obtain its prospective full prefix.
    ///
    /// Inputs must both have:
    /// [batch, kv_heads, token_count(), head_dim].
    ///
    /// Neither committed cache handle is changed if constructing either
    /// update or either returned prefix view fails.
    ///
    /// Keys should already have the normalization and positional encoding
    /// required by the model. This owner does not apply them itself.
    pub fn stage_layer(
        &mut self,
        layer: usize,
        new_keys: &Array,
        new_values: &Array,
    ) -> KvCacheResult<KvPrefix> {
        if layer >= self.staged.len() {
            return Err(KvCacheError::InvalidInput(
                "cache layer index is out of range",
            ));
        }
        if self.staged[layer].is_some() {
            return Err(KvCacheError::InvalidInput(
                "cache layer was already staged in this transaction",
            ));
        }

        let expected = self.cache.spec.chunk_shape(self.tokens);
        if new_keys.dtype() != DType::Float32
            || new_values.dtype() != DType::Float32
            || new_keys.shape().as_slice() != expected.as_slice()
            || new_values.shape().as_slice() != expected.as_slice()
        {
            return Err(KvCacheError::InvalidInput(
                "new K/V must match the configured float32 chunk shape",
            ));
        }

        let committed = &self.cache.layers[layer];

        // Construct BOTH next handles before publishing either.
        let next_keys = ops::cache_update(&committed.keys, new_keys, self.start, self.stream)?;
        let next_values =
            ops::cache_update(&committed.values, new_values, self.start, self.stream)?;

        let next = LayerKv {
            keys: next_keys,
            values: next_values,
        };

        // Prefix construction can also fail. Do it before recording this
        // layer as successfully staged.
        let prefix = prefix_view(&next, 0, self.end, self.stream)?;
        self.staged[layer] = Some(next);

        Ok(prefix)
    }

    /// Evaluate and publish a complete all-layer update.
    ///
    /// The caller should invoke this only after the model's final logits
    /// and cancellation checks have succeeded.
    ///
    /// Missing layers reject the transaction without changing the cache.
    /// Native evaluation/completion errors preserve the old handles and
    /// logical length but invalidate the cache: reconstruct it before reuse.
    ///
    /// This method synchronizes the update stream. Runtime code remains
    /// responsible for other outstanding streams and request-local work.
    pub fn commit(mut self) -> KvCacheResult<()> {
        let staged_layers = self.staged.iter().filter(|layer| layer.is_some()).count();

        if staged_layers != self.cache.spec.layers {
            return Err(KvCacheError::Incomplete {
                expected_layers: self.cache.spec.layers,
                staged_layers,
            });
        }

        let mut failure = None;

        for layer in self.staged.iter().flatten() {
            record_native_failure(&mut failure, layer.keys.eval());
            record_native_failure(&mut failure, layer.values.eval());
        }

        // Attempt completion even if an evaluation returned an error.
        record_native_failure(&mut failure, self.stream.synchronize());

        if let Some(error) = failure {
            self.cache.healthy = false;
            return Err(KvCacheError::Mlx(error));
        }

        // Evaluation completed; publication has not. Measure old/new
        // buffers together so native aliases are counted once.
        let entries = self
            .cache
            .layers
            .len()
            .checked_add(self.staged.len())
            .ok_or(KvCacheError::InvalidInput(
                "KV overlap handle count overflow",
            ))?;
        let measured = match measure_layer_buffers(
            self.cache.layers.iter().chain(self.staged.iter().flatten()),
            entries,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.cache.healthy = false;
                return Err(error);
            }
        };
        if measured > self.cache.max_buffer_overlap_bytes {
            return Err(KvCacheError::PayloadLimit {
                required_bytes: measured,
                limit_bytes: self.cache.max_buffer_overlap_bytes,
            });
        }

        // No fallible work remains. The exclusive cache borrow prevents
        // external observation of a partially replaced layer vector.
        for (committed, staged) in self.cache.layers.iter_mut().zip(self.staged.iter_mut()) {
            if let Some(next) = staged.take() {
                *committed = next;
            }
        }

        self.cache.length = self.end;
        Ok(())
    }
}

fn measure_layer_buffers<'a>(
    layers: impl IntoIterator<Item = &'a LayerKv>,
    layer_entries: usize,
) -> KvCacheResult<u64> {
    let count = layer_entries
        .checked_mul(2)
        .ok_or(KvCacheError::InvalidInput("KV query handle count overflow"))?;
    let mut arrays = Vec::new();
    arrays
        .try_reserve_exact(count)
        .map_err(|_| KvCacheError::AllocationFailed("KV buffer query metadata"))?;
    for layer in layers {
        arrays.push(&layer.keys);
        arrays.push(&layer.values);
    }
    let bytes = mlx_ffi::memory::array_buffer_bytes(&arrays)?;
    u64::try_from(bytes).map_err(|_| KvCacheError::InvalidInput("measured KV bytes exceed u64"))
}

fn prefix_view(
    layer: &LayerKv,
    start: usize,
    end: usize,
    stream: &Stream,
) -> KvCacheResult<KvPrefix> {
    let keys = ops::cache_slice(&layer.keys, start, end, stream)?;
    let values = ops::cache_slice(&layer.values, start, end, stream)?;

    Ok(KvPrefix {
        keys,
        values,
        start,
        end,
    })
}

fn record_native_failure(first: &mut Option<MlxError>, result: Result<(), MlxError>) {
    if let Err(error) = result
        && first.is_none()
    {
        *first = Some(error);
    }
}

/// The only raw allocation helper in this owner.
///
/// Uses the existing pinned mlx_full signature to create zero-filled
/// native arrays without a full-size Rust staging buffer.
fn native_zeros(shape: &[c_int; 4], stream: &Stream) -> KvCacheResult<Array> {
    let zero = Array::from_f32(0.0)?;
    let mut output = NativeOutput {
        raw: sys::mlx_array {
            ctx: ptr::null_mut(),
        },
    };

    // SAFETY: shape is positive and checked to fit C int; zero and stream
    // are valid borrowed handles; output is writable.
    let status = unsafe {
        sys::mlx_full(
            &mut output.raw,
            shape.as_ptr(),
            shape.len(),
            zero.as_raw(),
            DType::Float32 as u32,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(KvCacheError::Mlx(MlxError::Status(status)));
    }

    let raw = std::mem::replace(
        &mut output.raw,
        sys::mlx_array {
            ctx: ptr::null_mut(),
        },
    );

    Ok(Array::from_raw(raw)?)
}

struct NativeOutput {
    raw: sys::mlx_array,
}

impl Drop for NativeOutput {
    fn drop(&mut self) {
        if !self.raw.ctx.is_null() {
            // SAFETY: exclusively owned, untransferred output handle.
            unsafe {
                sys::mlx_array_free(self.raw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_ffi::{Device, DeviceType};

    fn stream(kind: DeviceType) -> Stream {
        let device = Device::new(kind, 0).expect("device");
        assert!(
            device.is_available().expect("availability"),
            "requested device is unavailable: {kind:?}",
        );
        let stream = Stream::new(&device).expect("stream");
        assert_eq!(stream.device().expect("device").kind().expect("kind"), kind,);
        stream
    }

    fn spec() -> KvCacheSpec {
        KvCacheSpec {
            layers: 2,
            batch: 1,
            kv_heads: 2,
            capacity_tokens: 4,
            head_dim: 2,
        }
    }

    fn cache(owner: &MlxModelId, stream: &Stream) -> MlxKvCache {
        let spec = spec();
        MlxKvCache::new(
            owner,
            spec,
            spec.payload_plan()
                .expect("plan")
                .update_overlap_bytes
                .checked_add(1024 * 1024)
                .expect("test native buffer allowance"),
            stream,
        )
        .expect("native cache")
    }

    fn chunk(values: &[f32], tokens: i32) -> Array {
        Array::from_f32_slice(values, &[1, 2, tokens, 2]).expect("chunk")
    }

    fn read(view: &KvPrefix, stream: &Stream) -> (Vec<f32>, Vec<f32>) {
        (
            ops::to_vec_f32(view.keys(), stream).expect("keys"),
            ops::to_vec_f32(view.values(), stream).expect("values"),
        )
    }

    fn seed(cache: &mut MlxKvCache, owner: &MlxModelId, stream: &Stream) {
        let keys = chunk(&[1.0, 2.0, 3.0, 4.0], 1);
        let values = chunk(&[11.0, 12.0, 13.0, 14.0], 1);

        let mut append = cache.begin_append(owner, 1, stream).expect("begin");
        for layer in 0..2 {
            append.stage_layer(layer, &keys, &values).expect("stage");
        }
        append.commit().expect("commit");
    }

    fn run_append_suite(kind: DeviceType) {
        let stream = stream(kind);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, &stream);

        seed(&mut cache, &owner, &stream);
        assert_eq!(cache.len(), 1);

        let old = cache.view(&owner, 0, 0, 1, &stream).expect("old view");
        let old_values = read(&old, &stream);

        let keys = chunk(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2);
        let values = chunk(&[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0], 2);

        let mut append = cache
            .begin_append(&owner, 2, &stream)
            .expect("begin second");
        assert_eq!(append.start_position(), 1);
        assert_eq!(append.end_position(), 3);

        for layer in 0..2 {
            let staged = append.stage_layer(layer, &keys, &values).expect("stage");
            assert_eq!(staged.len(), 3);
            assert_eq!(staged.keys().shape(), vec![1, 2, 3, 2]);
        }

        append.commit().expect("all-layer commit");
        assert_eq!(cache.len(), 3);

        for layer in 0..2 {
            let view = cache.view(&owner, layer, 0, 3, &stream).expect("view");
            let (keys, values) = read(&view, &stream);

            assert_eq!(
                keys,
                [
                    1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0
                ],
            );
            assert_eq!(
                values,
                [
                    11.0, 12.0, 15.0, 16.0, 17.0, 18.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0
                ],
            );
        }

        // Previously obtained immutable views remain valid snapshots.
        assert_eq!(read(&old, &stream), old_values);
    }

    #[test]
    fn mlx_cpu_cache_commits_matching_keys_values_and_length() {
        run_append_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_cache_commits_matching_keys_values_and_length() {
        run_append_suite(DeviceType::Gpu);
    }

    #[test]
    fn incomplete_or_abandoned_transaction_preserves_committed_state() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, &stream);
        seed(&mut cache, &owner, &stream);

        let before = read(
            &cache.view(&owner, 0, 0, 1, &stream).expect("before"),
            &stream,
        );

        let keys = chunk(&[21.0, 22.0, 23.0, 24.0], 1);
        let values = chunk(&[31.0, 32.0, 33.0, 34.0], 1);

        {
            let mut append = cache.begin_append(&owner, 1, &stream).expect("begin");
            append.stage_layer(0, &keys, &values).expect("stage");
            // No commit.
        }

        assert_eq!(cache.len(), 1);
        assert_eq!(
            read(
                &cache.view(&owner, 0, 0, 1, &stream).expect("after abort"),
                &stream,
            ),
            before,
        );

        let mut append = cache.begin_append(&owner, 1, &stream).expect("begin");
        append.stage_layer(0, &keys, &values).expect("stage");

        assert!(matches!(
            append.commit(),
            Err(KvCacheError::Incomplete {
                expected_layers: 2,
                staged_layers: 1,
            }),
        ));

        assert_eq!(cache.len(), 1);
        assert_eq!(
            read(
                &cache.view(&owner, 0, 0, 1, &stream).expect("unchanged"),
                &stream,
            ),
            before,
        );
    }

    #[test]
    fn invalid_value_shape_does_not_stage_or_publish_keys() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, &stream);

        let keys = chunk(&[1.0, 2.0, 3.0, 4.0], 1);
        let wrong_values = Array::from_f32_slice(&[5.0, 6.0], &[1, 1, 1, 2]).expect("wrong values");
        let values = chunk(&[5.0, 6.0, 7.0, 8.0], 1);

        let mut append = cache.begin_append(&owner, 1, &stream).expect("begin");

        assert!(append.stage_layer(0, &keys, &wrong_values).is_err());

        // The failed pair did not mark layer zero as staged.
        append.stage_layer(0, &keys, &values).expect("retry layer");
        assert!(append.stage_layer(0, &keys, &values).is_err());
        append.stage_layer(1, &keys, &values).expect("second layer");
        append.commit().expect("valid commit");

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn foreign_model_cannot_read_reset_truncate_or_append() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let foreign = MlxModelId::new();
        let mut cache = cache(&owner, &stream);
        seed(&mut cache, &owner, &stream);

        assert!(matches!(
            cache.view(&foreign, 0, 0, 1, &stream),
            Err(KvCacheError::ForeignModel),
        ));
        assert!(matches!(
            cache.reset(&foreign),
            Err(KvCacheError::ForeignModel),
        ));
        assert!(matches!(
            cache.truncate(&foreign, 0),
            Err(KvCacheError::ForeignModel),
        ));
        assert!(matches!(
            cache.begin_append(&foreign, 1, &stream),
            Err(KvCacheError::ForeignModel),
        ));

        assert_eq!(cache.len(), 1);
        assert!(cache.is_healthy());
    }

    #[test]
    fn reset_hides_old_positions_and_allows_fresh_append() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, &stream);
        seed(&mut cache, &owner, &stream);

        let plan = cache.payload_plan();
        cache.reset(&owner).expect("reset");

        assert!(cache.is_empty());
        assert_eq!(cache.payload_plan(), plan);
        assert!(cache.view(&owner, 0, 0, 1, &stream).is_err());

        let keys = chunk(&[41.0, 42.0, 43.0, 44.0], 1);
        let values = chunk(&[51.0, 52.0, 53.0, 54.0], 1);
        let mut append = cache.begin_append(&owner, 1, &stream).expect("fresh begin");

        for layer in 0..2 {
            append.stage_layer(layer, &keys, &values).expect("stage");
        }
        append.commit().expect("fresh commit");

        let actual = read(
            &cache.view(&owner, 0, 0, 1, &stream).expect("fresh view"),
            &stream,
        );
        assert_eq!(actual.0, [41.0, 42.0, 43.0, 44.0]);
        assert_eq!(actual.1, [51.0, 52.0, 53.0, 54.0]);
        assert!(cache.view(&owner, 0, 0, 2, &stream).is_err());
    }

    #[test]
    fn capacity_layer_and_logical_bounds_are_enforced() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, &stream);

        assert!(cache.begin_append(&owner, 0, &stream).is_err());
        assert!(cache.begin_append(&owner, 5, &stream).is_err());
        assert!(cache.begin_append(&owner, usize::MAX, &stream).is_err());

        seed(&mut cache, &owner, &stream);

        assert!(cache.view(&owner, 2, 0, 1, &stream).is_err());
        assert!(cache.view(&owner, 0, 0, 2, &stream).is_err());
        assert!(cache.view(&owner, 0, 1, 1, &stream).is_err());
        assert!(cache.truncate(&owner, 2).is_err());

        cache.truncate(&owner, 0).expect("truncate");
        assert!(cache.is_empty());
    }

    #[test]
    fn payload_limit_rejects_missing_update_overlap_allowance() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let spec = spec();
        let plan = spec.payload_plan().expect("plan");

        assert!(matches!(
            MlxKvCache::new(&owner, spec, plan.resident_bytes, &stream),
            Err(KvCacheError::PayloadLimit { .. }),
        ));
    }

    #[test]
    fn invalid_or_overflowing_specs_are_rejected() {
        let mut invalid = spec();
        invalid.layers = 0;
        assert!(invalid.payload_plan().is_err());

        let mut overflow = spec();
        overflow.layers = usize::MAX;
        assert!(overflow.payload_plan().is_err());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn one_cache_is_updated_on_metal_and_consumed_on_cpu() {
        let cpu = stream(DeviceType::Cpu);
        let metal = stream(DeviceType::Gpu);
        let owner = MlxModelId::new();

        // One cache allocation, initialized on CPU.
        let mut cache = cache(&owner, &cpu);

        // The same cache arrays are updated on Metal.
        seed(&mut cache, &owner, &metal);

        // CPU views consume the committed Metal-produced arrays.
        let actual = read(&cache.view(&owner, 0, 0, 1, &cpu).expect("CPU view"), &cpu);

        assert_eq!(actual.0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(actual.1, [11.0, 12.0, 13.0, 14.0]);

        metal.synchronize().expect("Metal completion");
        cpu.synchronize().expect("CPU completion");
    }
}

#[cfg(test)]
mod block_transfer_tests {
    use super::*;
    use mlx_ffi::{Device, DeviceType};

    fn stream(kind: DeviceType) -> Stream {
        let device = Device::new(kind, 0).expect("device");
        assert!(device.is_available().expect("device availability"));
        Stream::new(&device).expect("stream")
    }

    fn spec() -> KvCacheSpec {
        KvCacheSpec {
            layers: 2,
            batch: 2,
            kv_heads: 2,
            capacity_tokens: 6,
            head_dim: 3,
        }
    }

    fn cache(owner: &MlxModelId, spec: KvCacheSpec, stream: &Stream) -> MlxKvCache {
        MlxKvCache::new(owner, spec, 1024 * 1024, stream).expect("cache")
    }

    fn block(spec: KvCacheSpec, start: usize, tokens: usize) -> KvBlock {
        let mut layers = Vec::new();
        for layer in 0..spec.layers {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for batch in 0..spec.batch {
                for head in 0..spec.kv_heads {
                    for token in start..start + tokens {
                        for dimension in 0..spec.head_dim {
                            let value = (layer * 1000
                                + batch * 100
                                + head * 20
                                + token * spec.head_dim
                                + dimension) as f32;
                            keys.push(value + 0.5);
                            values.push(-value - 0.25);
                        }
                    }
                }
            }
            layers.push(KvBlockLayer { keys, values });
        }
        KvBlock {
            start,
            tokens,
            spec,
            layers,
        }
    }

    fn append_directly(cache: &mut MlxKvCache, owner: &MlxModelId, tokens: usize, stream: &Stream) {
        let block = block(cache.spec(), cache.len(), tokens);
        let mut shape = cache.spec().native_shape().expect("shape");
        shape[2] = tokens as c_int;
        let mut append = cache.begin_append(owner, tokens, stream).expect("append");
        for (index, layer) in block.layers.iter().enumerate() {
            let keys = Array::from_f32_slice(&layer.keys, &shape).expect("keys");
            let values = Array::from_f32_slice(&layer.values, &shape).expect("values");
            append.stage_layer(index, &keys, &values).expect("stage");
        }
        append.commit().expect("commit");
    }

    fn roundtrip(kind: DeviceType) {
        let stream = stream(kind);
        let old_owner = MlxModelId::new();
        let new_owner = MlxModelId::new();
        let spec = spec();
        let mut source = cache(&old_owner, spec, &stream);
        append_directly(&mut source, &old_owner, 4, &stream);

        // Slicing the token axis of a capacity-six array with multiple
        // batches/heads produces a strided view. Its raw pointer cannot be
        // read as a flat token block without first making it contiguous.
        let middle = source
            .export_block(&old_owner, 1, 3, &stream)
            .expect("noncontiguous middle block");
        assert_eq!(middle, block(spec, 1, 2));
        assert_eq!(middle.payload_plan().expect("plan").host_bytes, 384);

        let first = source
            .export_block(&old_owner, 0, 2, &stream)
            .expect("first block");
        let second = source
            .export_block(&old_owner, 2, 4, &stream)
            .expect("second block");
        assert_eq!(first, block(spec, 0, 2));
        assert_eq!(second, block(spec, 2, 2));

        let destination_spec = KvCacheSpec {
            capacity_tokens: 5,
            ..spec
        };
        let mut restored = cache(&new_owner, destination_spec, &stream);
        assert!(!old_owner.same_instance(restored.owner()));
        assert!(new_owner.same_instance(restored.owner()));
        assert_eq!(
            restored
                .restore_block(&new_owner, &first, &stream)
                .expect("first restore"),
            2,
        );
        assert_eq!(
            restored
                .restore_block(&new_owner, &second, &stream)
                .expect("second restore"),
            4,
        );

        // Both caches accept ordinary new model output after restore.
        append_directly(&mut source, &old_owner, 1, &stream);
        append_directly(&mut restored, &new_owner, 1, &stream);
        let source = source
            .export_block(&old_owner, 0, 5, &stream)
            .expect("source continuation");
        let restored = restored
            .export_block(&new_owner, 0, 5, &stream)
            .expect("restored continuation");
        assert_eq!(restored.tokens, 5);
        assert_eq!(restored.layers, source.layers);
        assert_eq!(restored.layers, block(destination_spec, 0, 5).layers);
    }

    #[test]
    fn mlx_cpu_block_roundtrip_preserves_strided_values_and_continuation() {
        roundtrip(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_block_roundtrip_preserves_strided_values_and_continuation() {
        roundtrip(DeviceType::Gpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_blocks_transfer_from_metal_to_new_cpu_owned_cache() {
        let cpu = stream(DeviceType::Cpu);
        let metal = stream(DeviceType::Gpu);
        let old_owner = MlxModelId::new();
        let new_owner = MlxModelId::new();
        let mut source = cache(&old_owner, spec(), &metal);
        append_directly(&mut source, &old_owner, 3, &metal);
        let data = source
            .export_block(&old_owner, 0, 3, &metal)
            .expect("Metal export");
        let mut restored = cache(&new_owner, spec(), &cpu);
        assert_eq!(restored.restore_block(&new_owner, &data, &cpu).unwrap(), 3);
        append_directly(&mut restored, &new_owner, 1, &cpu);
        assert_eq!(
            restored
                .export_block(&new_owner, 0, 4, &cpu)
                .expect("CPU continuation"),
            block(spec(), 0, 4),
        );
    }

    #[test]
    fn block_payload_plan_checks_all_dimensions_and_transfer_overlap() {
        let plan = spec().block_payload_plan(2).expect("plan");
        assert_eq!(plan.host_bytes, 2 * 2 * 2 * 2 * 3 * 2 * 4);
        assert_eq!(plan.native_transfer_bytes, plan.host_bytes);
        assert_eq!(plan.transfer_overlap_bytes, plan.host_bytes * 2);
        assert!(spec().block_payload_plan(0).is_err());
        assert!(spec().block_payload_plan(7).is_err());
        assert!(spec().block_payload_plan(usize::MAX).is_err());
        let mut invalid = spec();
        invalid.batch = 0;
        assert!(invalid.block_payload_plan(1).is_err());
        invalid = spec();
        invalid.layers = usize::MAX;
        assert!(invalid.block_payload_plan(1).is_err());
        invalid = spec();
        invalid.kv_heads = usize::MAX;
        assert!(invalid.block_payload_plan(1).is_err());
    }

    #[test]
    fn block_truncation_crops_every_batch_head_row_in_place() {
        let mut data = block(spec(), 1, 4);
        let allocations: Vec<_> = data
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.keys.as_ptr(),
                    layer.keys.capacity(),
                    layer.values.as_ptr(),
                    layer.values.capacity(),
                )
            })
            .collect();
        data.truncate_tokens(2).expect("crop four tokens to two");
        assert_eq!(data, block(spec(), 1, 2));
        assert_eq!(data.payload_plan().unwrap().host_bytes, 384);
        // Repeated cropping must use the updated row stride, while the
        // source position and each original allocation remain intact.
        data.truncate_tokens(1).expect("crop two tokens to one");
        assert_eq!(data, block(spec(), 1, 1));
        for (layer, (key_ptr, key_capacity, value_ptr, value_capacity)) in
            data.layers.iter().zip(allocations)
        {
            assert_eq!(layer.keys.as_ptr(), key_ptr);
            assert_eq!(layer.keys.capacity(), key_capacity);
            assert_eq!(layer.values.as_ptr(), value_ptr);
            assert_eq!(layer.values.capacity(), value_capacity);
        }
        let before = data.clone();
        data.truncate_tokens(1).expect("unchanged token count");
        assert_eq!(data, before);
    }

    #[test]
    fn invalid_block_truncation_leaves_all_data_and_metadata_unchanged() {
        let mut data = block(spec(), 1, 4);
        for token_count in [0, 5, usize::MAX] {
            let before = data.clone();
            assert!(matches!(
                data.truncate_tokens(token_count),
                Err(KvCacheError::InvalidInput(_)),
            ));
            assert_eq!(data, before);
        }
        // A malformed later layer must be found before earlier valid
        // layers are compacted. This also checks validation on no-op crops.
        data.layers[1].values.pop();
        let before = data.clone();
        assert!(data.truncate_tokens(2).is_err());
        assert_eq!(data, before);
        assert!(data.truncate_tokens(4).is_err());
        assert_eq!(data, before);

        let mut data = block(spec(), 1, 4);
        data.layers[1].keys[2] = f32::INFINITY;
        let before = data.clone();
        assert!(data.truncate_tokens(2).is_err());
        assert_eq!(data, before);
    }

    #[test]
    fn malformed_blocks_never_change_committed_native_state() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, spec(), &stream);
        append_directly(&mut cache, &owner, 1, &stream);
        let before = cache.export_block(&owner, 0, 1, &stream).unwrap();
        let valid = block(spec(), 1, 1);
        let mut malformed = Vec::new();
        let mut value = valid.clone();
        value.layers.pop();
        malformed.push(value);
        let mut value = valid.clone();
        value.layers.push(value.layers[0].clone());
        malformed.push(value);
        let mut value = valid.clone();
        value.layers[1].keys.pop();
        malformed.push(value);
        let mut value = valid.clone();
        value.layers[1].values.push(0.0);
        malformed.push(value);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut value = valid.clone();
            value.layers[1].values[0] = invalid;
            malformed.push(value);
        }
        let mut value = valid.clone();
        value.start = 0;
        malformed.push(value);
        let mut value = valid.clone();
        value.start = 2;
        malformed.push(value);
        let mut value = valid.clone();
        value.start = usize::MAX;
        malformed.push(value);
        let mut value = valid.clone();
        value.tokens = 0;
        malformed.push(value);
        let mut value = valid.clone();
        value.tokens = usize::MAX;
        malformed.push(value);
        let mut value = valid.clone();
        value.spec.head_dim += 1;
        malformed.push(value);
        let mut value = valid.clone();
        value.spec.layers = usize::MAX;
        malformed.push(value);
        // Well-formed host data with a different tensor shape is also
        // rejected before creating any native input arrays.
        malformed.push(block(KvCacheSpec { batch: 1, ..spec() }, 1, 1));
        malformed.push(block(
            KvCacheSpec {
                kv_heads: 1,
                ..spec()
            },
            1,
            1,
        ));

        for invalid in &malformed {
            assert!(matches!(
                cache.restore_block(&owner, invalid, &stream),
                Err(KvCacheError::InvalidInput(_)),
            ));
            assert_eq!(cache.len(), 1);
            assert!(cache.is_healthy());
        }
        assert_eq!(cache.export_block(&owner, 0, 1, &stream).unwrap(), before);
        assert_eq!(cache.restore_block(&owner, &valid, &stream).unwrap(), 2);
    }

    #[test]
    fn block_transfer_checks_owner_logical_bounds_and_destination_capacity() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let foreign = MlxModelId::new();
        let mut cache = cache(&owner, spec(), &stream);
        append_directly(&mut cache, &owner, 1, &stream);
        for (start, end) in [(0, 0), (1, 1), (0, 2), (2, 1), (0, usize::MAX)] {
            assert!(cache.export_block(&owner, start, end, &stream).is_err());
        }
        assert!(matches!(
            cache.export_block(&foreign, 0, 1, &stream),
            Err(KvCacheError::ForeignModel),
        ));
        let valid = block(spec(), 1, 1);
        assert!(matches!(
            cache.restore_block(&foreign, &valid, &stream),
            Err(KvCacheError::ForeignModel),
        ));
        let too_large = block(
            KvCacheSpec {
                capacity_tokens: 8,
                ..spec()
            },
            1,
            6,
        );
        assert!(matches!(
            cache.restore_block(&owner, &too_large, &stream),
            Err(KvCacheError::InvalidInput(_)),
        ));
        assert_eq!(cache.len(), 1);
        assert!(cache.is_healthy());
    }

    #[test]
    fn restore_commit_rejection_preserves_all_old_layer_handles_and_length() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, spec(), &stream);
        append_directly(&mut cache, &owner, 1, &stream);
        let before = cache.export_block(&owner, 0, 1, &stream).unwrap();
        let valid = block(spec(), 1, 1);
        let allowance = cache.max_buffer_overlap_bytes;
        // Force the existing post-evaluation, pre-publication gate to
        // reject the otherwise complete native transaction.
        cache.max_buffer_overlap_bytes = 0;
        assert!(matches!(
            cache.restore_block(&owner, &valid, &stream),
            Err(KvCacheError::PayloadLimit { .. }),
        ));
        assert_eq!(cache.len(), 1);
        assert!(cache.is_healthy());
        assert_eq!(cache.export_block(&owner, 0, 1, &stream).unwrap(), before);
        cache.max_buffer_overlap_bytes = allowance;
        assert_eq!(cache.restore_block(&owner, &valid, &stream).unwrap(), 2);
        assert_eq!(
            cache.export_block(&owner, 0, 2, &stream).unwrap(),
            block(spec(), 0, 2),
        );
    }

    #[test]
    fn export_rejects_nonfinite_native_values_without_publishing_host_data() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, spec(), &stream);
        let mut data = block(spec(), 0, 1);
        data.layers[1].keys[2] = f32::NAN;
        let mut shape = spec().native_shape().unwrap();
        shape[2] = 1;
        let mut append = cache.begin_append(&owner, 1, &stream).unwrap();
        for (index, layer) in data.layers.iter().enumerate() {
            let keys = Array::from_f32_slice(&layer.keys, &shape).unwrap();
            let values = Array::from_f32_slice(&layer.values, &shape).unwrap();
            append.stage_layer(index, &keys, &values).unwrap();
        }
        append.commit().unwrap();
        assert!(matches!(
            cache.export_block(&owner, 0, 1, &stream),
            Err(KvCacheError::InvalidInput(
                "KV block contains non-finite values"
            )),
        ));
        assert_eq!(cache.len(), 1);
        assert!(cache.is_healthy());
    }

    #[test]
    fn reported_native_transfer_failure_invalidates_future_access() {
        let stream = stream(DeviceType::Cpu);
        let owner = MlxModelId::new();
        let mut cache = cache(&owner, spec(), &stream);
        append_directly(&mut cache, &owner, 1, &stream);
        let result = cache.failed_block_transfer(
            KvCacheError::Operation(OperationError::Mlx(MlxError::Status(-1))),
            &stream,
        );
        assert!(matches!(result, KvCacheError::Operation(_)));
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_healthy());
        assert!(matches!(
            cache.export_block(&owner, 0, 1, &stream),
            Err(KvCacheError::Invalidated),
        ));
        assert!(matches!(
            cache.restore_block(&owner, &block(spec(), 1, 1), &stream),
            Err(KvCacheError::Invalidated),
        ));
    }
}

impl From<mlx_ffi::memory::MemoryError> for KvCacheError {
    fn from(error: mlx_ffi::memory::MemoryError) -> Self {
        Self::Memory(error)
    }
}

#[cfg(test)]
mod measured_kv_buffer_tests {
    use super::*;
    use mlx_ffi::{Device, DeviceType};
    fn check(kind: DeviceType) {
        let device = Device::new(kind, 0).expect("device");
        assert!(device.is_available().expect("availability"));
        let stream = Stream::new(&device).expect("stream");
        let owner = MlxModelId::new();
        let spec = KvCacheSpec {
            layers: 2,
            batch: 1,
            kv_heads: 2,
            capacity_tokens: 4,
            head_dim: 2,
        };
        let mut cache = MlxKvCache::new(&owner, spec, 1024 * 1024, &stream).expect("cache");
        assert!(
            cache.buffer_bytes().expect("initial buffers") >= cache.payload_plan().resident_bytes
        );
        let keys = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2]).expect("keys");
        let values = Array::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 1, 2]).expect("values");
        let mut append = cache.begin_append(&owner, 1, &stream).expect("append");
        for layer in 0..2 {
            append.stage_layer(layer, &keys, &values).expect("stage");
        }
        append.commit().expect("measured commit");
        let measured = cache.buffer_bytes().expect("committed buffers");
        assert!(measured >= cache.payload_plan().resident_bytes);
        assert!(measured <= cache.max_buffer_overlap_bytes);
        cache.reset(&owner).expect("logical reset");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.buffer_bytes().expect("retained buffers"), measured);
        cache.healthy = false;
        assert!(matches!(
            cache.buffer_bytes(),
            Err(KvCacheError::Invalidated)
        ));
    }
    #[test]
    fn measured_cpu_kv_reports_buffers_not_logical_length() {
        check(DeviceType::Cpu);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn measured_metal_kv_reports_buffers_not_logical_length() {
        check(DeviceType::Gpu);
    }
}

// ALLOY_MEASURED_BUFFERS_V1
