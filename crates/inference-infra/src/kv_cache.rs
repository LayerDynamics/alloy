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
    if let Err(error) = result {
        if first.is_none() {
            *first = Some(error);
        }
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
