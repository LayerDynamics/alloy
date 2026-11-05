//! MLX allocator observations and process-wide memory controls.
//!
//! This module wraps the memory API from the vendored mlx-c pin.
//!
//! Important distinctions:
//!
//! - Active memory is not process RSS.
//! - Cached memory is reusable allocator storage, not live model tensors.
//! - Engine reservations are separate accounting ownership.
//! - MemorySnapshot reads multiple native counters sequentially. It is
//!   not an atomic allocator snapshot.
//! - MLX's memory limit is a graph-evaluation guideline, not a guaranteed
//!   hard allocation or operating-system memory ceiling.
//! - Array-buffer accounting counts unique backing buffers and may
//!   exceed logical tensor payload because allocator buffer sizes are used.
//!
//! Limit setters, cache clearing, and peak reset affect process-wide MLX
//! state. The application must coordinate those operations centrally.
//! Individual model instances should not independently change them.
//!
//! Functions do not silently report zero when native calls fail.
//!
//! This module does not discover physical RAM, select admission budgets,
//! inspect OS pressure, automatically evict engine state, or modify
//! engine reservation accounting.

use crate::{Array, Error as MlxError, Stream, VectorArray, bindings as sys};
use std::{fmt, os::raw::c_int};

/// Failure observing or configuring MLX memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// A native memory operation returned a nonzero status.
    Native {
        /// Operation that failed.
        operation: &'static str,
        /// Native status code.
        status: c_int,
    },

    /// An existing MLX wrapper operation failed.
    Mlx(MlxError),

    /// Array-buffer measurement requires evaluated arrays.
    ArrayNotEvaluated {
        /// Index of the unevaluated array in the supplied list.
        index: usize,
    },

    /// Allocation of temporary Rust handle metadata failed.
    AllocationFailed,

    /// A checked byte-count calculation overflowed.
    ArithmeticOverflow,

    /// An input does not meet the operation's contract.
    InvalidInput(&'static str),

    /// A requested completion boundary failed.
    StreamSynchronization {
        /// Index in the supplied stream list.
        index: usize,
        /// Error returned by the stream wrapper.
        source: MlxError,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native { operation, status } => {
                write!(
                    formatter,
                    "MLX memory operation {operation} failed with status {status}"
                )
            }
            Self::Mlx(error) => write!(formatter, "{error}"),
            Self::ArrayNotEvaluated { index } => {
                write!(
                    formatter,
                    "array {index} must be evaluated before buffer measurement"
                )
            }
            Self::AllocationFailed => {
                write!(formatter, "MLX memory-query metadata allocation failed")
            }
            Self::ArithmeticOverflow => {
                write!(formatter, "MLX observed-memory arithmetic overflow")
            }
            Self::InvalidInput(message) => {
                write!(formatter, "invalid MLX memory operation: {message}")
            }
            Self::StreamSynchronization { index, source } => {
                write!(
                    formatter,
                    "MLX memory operation could not complete stream {index}: \
                     {source}"
                )
            }
        }
    }
}

impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mlx(error) => Some(error),
            Self::StreamSynchronization { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<MlxError> for MemoryError {
    fn from(error: MlxError) -> Self {
        Self::Mlx(error)
    }
}

/// Result of an MLX memory observation or control operation.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Sequential observations of process-wide MLX allocator state.
///
/// Native allocation, deallocation, or another process-wide control call
/// may occur between fields. Do not interpret this as an atomic snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Actively used allocator memory.
    ///
    /// Excludes cached free allocator buffers and is not process RSS.
    pub active_bytes: usize,

    /// Free allocator buffers retained for reuse.
    pub cache_bytes: usize,

    /// Peak used memory since process startup or the last peak reset.
    pub peak_bytes: usize,

    /// Currently configured graph-evaluation memory guideline.
    pub memory_limit_bytes: usize,
}

impl MemorySnapshot {
    /// Checked sum of the observed active and cached byte counts.
    ///
    /// Because the fields were read sequentially, this is an observed
    /// sum, not an atomic total or an operating-system memory measurement.
    pub fn observed_allocator_bytes(&self) -> MemoryResult<usize> {
        self.active_bytes
            .checked_add(self.cache_bytes)
            .ok_or(MemoryError::ArithmeticOverflow)
    }
}

/// Observations around an explicit allocator-cache clear.
///
/// Differences may also reflect other concurrent allocator activity.
/// This is not evidence of an equal reduction in process RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheReclaimReport {
    /// Allocator observations before clearing cached buffers.
    pub before: MemorySnapshot,

    /// Allocator observations after clearing cached buffers.
    pub after: MemorySnapshot,
}

impl CacheReclaimReport {
    /// Nonnegative reduction between the observed cached-byte counts.
    ///
    /// This is deliberately not named "freed bytes": concurrent activity
    /// and operating-system allocator behavior can affect the result.
    pub fn observed_cache_reduction_bytes(&self) -> usize {
        self.before
            .cache_bytes
            .saturating_sub(self.after.cache_bytes)
    }
}

fn check(operation: &'static str, status: c_int) -> MemoryResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(MemoryError::Native { operation, status })
    }
}

fn read_size(
    operation: &'static str,
    call: impl FnOnce(&mut usize) -> c_int,
) -> MemoryResult<usize> {
    let mut value = 0_usize;
    check(operation, call(&mut value))?;
    Ok(value)
}

fn replace_limit(
    operation: &'static str,
    bytes: usize,
    call: impl FnOnce(&mut usize, usize) -> c_int,
) -> MemoryResult<usize> {
    let mut previous = 0_usize;
    check(operation, call(&mut previous, bytes))?;
    Ok(previous)
}

/// Observe actively used MLX allocator memory in bytes.
///
/// Does not include retained free cache buffers or all process memory.
pub fn active_memory_bytes() -> MemoryResult<usize> {
    read_size("get_active_memory", |out| {
        // SAFETY: out is a valid native size_t output target.
        unsafe { sys::mlx_get_active_memory(out) }
    })
}

/// Observe free allocator buffers retained by MLX in bytes.
pub fn cache_memory_bytes() -> MemoryResult<usize> {
    read_size("get_cache_memory", |out| {
        // SAFETY: out is a valid native size_t output target.
        unsafe { sys::mlx_get_cache_memory(out) }
    })
}

/// Observe peak used allocator memory in bytes.
///
/// The measurement window is process startup or the last peak reset.
/// This does not create a request-local measurement window.
pub fn peak_memory_bytes() -> MemoryResult<usize> {
    read_size("get_peak_memory", |out| {
        // SAFETY: out is a valid native size_t output target.
        unsafe { sys::mlx_get_peak_memory(out) }
    })
}

/// Read the configured MLX graph-evaluation memory guideline.
pub fn memory_limit_bytes() -> MemoryResult<usize> {
    read_size("get_memory_limit", |out| {
        // SAFETY: out is a valid native size_t output target.
        unsafe { sys::mlx_get_memory_limit(out) }
    })
}

/// Read active, cached, peak, and configured-limit observations.
///
/// The reads are sequential, do not synchronize streams, and are not
/// atomic with respect to allocation or process-wide configuration changes.
pub fn snapshot() -> MemoryResult<MemorySnapshot> {
    Ok(MemorySnapshot {
        active_bytes: active_memory_bytes()?,
        cache_bytes: cache_memory_bytes()?,
        peak_bytes: peak_memory_bytes()?,
        memory_limit_bytes: memory_limit_bytes()?,
    })
}

/// Reset the process-wide peak-memory measurement.
///
/// Coordinate with other measurements. This function does not
/// synchronize streams, clear allocations, or establish a request-local
/// exclusive measurement interval.
pub fn reset_peak_memory() -> MemoryResult<()> {
    // SAFETY: native function takes no arguments.
    check("reset_peak_memory", unsafe { sys::mlx_reset_peak_memory() })
}

/// Set the process-wide MLX memory guideline and return its previous value.
///
/// This is not an engine admission reservation or a hard OS-memory limit.
/// The wrapper does not reinterpret zero or automatically select a value.
///
/// Coordinate this operation centrally; independently scoped model-level
/// setters/restorers can overwrite each other's process-wide policy.
pub fn set_memory_limit_bytes(bytes: usize) -> MemoryResult<usize> {
    replace_limit("set_memory_limit", bytes, |out, limit| {
        // SAFETY: out is writable and limit is a native size_t value.
        unsafe { sys::mlx_set_memory_limit(out, limit) }
    })
}

/// Set the process-wide free-cache limit and return its previous value.
///
/// A limit of zero disables retention of free cache buffers according to
/// the native API. Lowering the limit does not itself promise immediate
/// reclamation; use clear_cache at a coordinated completion boundary.
///
/// This function does not synchronize streams.
pub fn set_cache_limit_bytes(bytes: usize) -> MemoryResult<usize> {
    replace_limit("set_cache_limit", bytes, |out, limit| {
        // SAFETY: out is writable and limit is a native size_t value.
        unsafe { sys::mlx_set_cache_limit(out, limit) }
    })
}

/// Set the process-wide wired-memory limit and return its previous value.
///
/// Useful for the supported Metal backend on macOS 15 or newer. A request
/// exceeding the system wired limit is rejected by MLX.
///
/// This wrapper does not discover the system limit or assume that a
/// successful call implies a particular resident-memory amount.
///
/// Unsupported native environments return the native error rather than
/// a fabricated success or zero previous value.
pub fn set_wired_limit_bytes(bytes: usize) -> MemoryResult<usize> {
    replace_limit("set_wired_limit", bytes, |out, limit| {
        // SAFETY: out is writable and limit is a native size_t value.
        unsafe { sys::mlx_set_wired_limit(out, limit) }
    })
}

/// Clear unused MLX allocator cache buffers.
///
/// This does not destroy live model/cache arrays, modify engine
/// reservations, synchronize streams, or promise a particular RSS drop.
///
/// Prefer clear_cache_after_synchronizing when the runtime is performing
/// a coordinated reclamation attempt.
pub fn clear_cache() -> MemoryResult<()> {
    // SAFETY: native function takes no arguments.
    check("clear_cache", unsafe { sys::mlx_clear_cache() })
}

/// Complete all supplied streams, then observe and clear allocator cache.
///
/// Every supplied stream is attempted even if an earlier synchronization
/// fails. If any synchronization fails, no cache-clear call is made.
///
/// The caller must provide all streams relevant to its reclamation
/// boundary and prevent concurrent work if it needs a quiescent interval.
/// The wrapper cannot discover arbitrary streams elsewhere in the process.
///
/// This operation never changes the engine's reservation ledger.
pub fn clear_cache_after_synchronizing(streams: &[&Stream]) -> MemoryResult<CacheReclaimReport> {
    if streams.is_empty() {
        return Err(MemoryError::InvalidInput(
            "provide at least one relevant execution stream",
        ));
    }

    let mut failure = None;

    for (index, stream) in streams.iter().enumerate() {
        if let Err(source) = stream.synchronize() {
            if failure.is_none() {
                failure = Some(MemoryError::StreamSynchronization { index, source });
            }
        }
    }

    if let Some(error) = failure {
        return Err(error);
    }

    let before = snapshot()?;
    clear_cache()?;
    let after = snapshot()?;

    Ok(CacheReclaimReport { before, after })
}

/// Measure unique native backing-buffer bytes for evaluated arrays.
///
/// Each unique buffer is counted once by MLX, even when multiple supplied
/// arrays share it. Full allocator buffer sizes are counted, which may
/// exceed logical tensor payload or a sliced view's element count.
///
/// This function does not evaluate arrays or synchronize streams.
/// It rejects arrays without available evaluated storage before calling
/// the native query.
///
/// An empty input has no backing buffers and returns zero.
pub fn array_buffer_bytes(arrays: &[&Array]) -> MemoryResult<usize> {
    if arrays.is_empty() {
        return Ok(0);
    }

    let mut handles = Vec::new();
    handles
        .try_reserve_exact(arrays.len())
        .map_err(|_| MemoryError::AllocationFailed)?;

    for (index, array) in arrays.iter().enumerate() {
        if !array.is_available()? {
            return Err(MemoryError::ArrayNotEvaluated { index });
        }

        handles.push(array.as_raw());
    }

    // SAFETY: handles is a valid length-matched array of borrowed native
    // handles. mlx_vector_array_new_data retains its own array references.
    let raw = unsafe { sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };
    let vector = VectorArray::from_raw(raw)?;

    read_size("get_array_buffer_size", |out| {
        // SAFETY: out is writable; vector owns valid references to arrays
        // whose evaluated availability was checked above.
        unsafe { sys::mlx_get_array_buffer_size(out, vector.as_raw()) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, DeviceType, ops};

    fn cpu_stream() -> Stream {
        let device = Device::new(DeviceType::Cpu, 0).expect("CPU device");
        assert!(device.is_available().expect("CPU availability"));
        Stream::new(&device).expect("CPU stream")
    }

    #[test]
    fn snapshot_sum_is_checked_without_assuming_atomic_native_reads() {
        let sample = MemorySnapshot {
            active_bytes: 100,
            cache_bytes: 40,
            peak_bytes: 300,
            memory_limit_bytes: 1000,
        };

        assert_eq!(sample.observed_allocator_bytes().expect("sum"), 140,);

        let overflowing = MemorySnapshot {
            active_bytes: usize::MAX,
            cache_bytes: 1,
            peak_bytes: 0,
            memory_limit_bytes: 0,
        };

        assert_eq!(
            overflowing.observed_allocator_bytes(),
            Err(MemoryError::ArithmeticOverflow),
        );
    }

    #[test]
    fn cache_reduction_is_an_observation_not_a_negative_freed_count() {
        let before = MemorySnapshot {
            active_bytes: 100,
            cache_bytes: 80,
            peak_bytes: 300,
            memory_limit_bytes: 1000,
        };

        let after = MemorySnapshot {
            cache_bytes: 20,
            ..before
        };

        assert_eq!(
            CacheReclaimReport { before, after }.observed_cache_reduction_bytes(),
            60,
        );

        let growing = MemorySnapshot {
            cache_bytes: 120,
            ..before
        };

        assert_eq!(
            CacheReclaimReport {
                before,
                after: growing,
            }
            .observed_cache_reduction_bytes(),
            0,
        );
    }

    #[test]
    fn native_counter_queries_return_status_checked_observations() {
        let observed = snapshot().expect("native allocator observations");

        // Do not assert peak >= active here. These are sequential reads,
        // and another caller can reset peak independently.
        let _ = observed
            .observed_allocator_bytes()
            .expect("observed active/cache sum");

        active_memory_bytes().expect("active bytes");
        cache_memory_bytes().expect("cached bytes");
        peak_memory_bytes().expect("peak bytes");
        memory_limit_bytes().expect("memory guideline");
    }

    #[test]
    fn empty_array_list_has_no_backing_buffers() {
        assert_eq!(array_buffer_bytes(&[]).expect("empty buffer set"), 0,);
    }

    #[test]
    fn native_buffer_measurement_deduplicates_shared_array_storage() {
        let stream = cpu_stream();
        let values: Vec<f32> = (0..4096).map(|value| value as f32).collect();

        let array = Array::from_f32_slice(&values, &[4096]).expect("array");
        array.eval().expect("materialize array");
        stream.synchronize().expect("complete array");

        let alias = array.clone();

        let single = array_buffer_bytes(&[&array]).expect("single buffer");
        let repeated = array_buffer_bytes(&[&array, &array]).expect("repeated handle");
        let aliased = array_buffer_bytes(&[&array, &alias]).expect("shared storage");

        assert!(single >= values.len() * std::mem::size_of::<f32>());
        assert_eq!(single, repeated);
        assert_eq!(single, aliased);
    }

    #[test]
    fn lazy_array_measurement_is_rejected_without_implicit_evaluation() {
        let stream = cpu_stream();
        let input = Array::from_f32_slice(&[1.0, -2.0, 3.0, 4.0], &[4]).expect("input");
        input.eval().expect("materialize input");

        let lazy = ops::silu(&input, &stream).expect("lazy MLX graph");
        assert!(!lazy.is_available().expect("availability"));

        assert_eq!(
            array_buffer_bytes(&[&lazy]),
            Err(MemoryError::ArrayNotEvaluated { index: 0 }),
        );

        lazy.eval().expect("explicit evaluation");
        stream.synchronize().expect("explicit completion");

        assert!(
            array_buffer_bytes(&[&lazy]).expect("evaluated buffer")
                >= 4 * std::mem::size_of::<f32>(),
        );
    }

    #[test]
    fn coordinated_reclamation_requires_an_explicit_stream_set() {
        assert_eq!(
            clear_cache_after_synchronizing(&[]),
            Err(MemoryError::InvalidInput(
                "provide at least one relevant execution stream",
            )),
        );
    }

    // Process-wide mutation tests are opt-in. They must run in a process
    // whose other threads are not independently changing MLX policy.

    struct RestoreCacheLimit {
        previous: Option<usize>,
    }

    impl RestoreCacheLimit {
        fn disable() -> MemoryResult<Self> {
            Ok(Self {
                previous: Some(set_cache_limit_bytes(0)?),
            })
        }

        fn restore(mut self) -> MemoryResult<()> {
            if let Some(previous) = self.previous {
                set_cache_limit_bytes(previous)?;
                self.previous = None;
            }
            Ok(())
        }
    }

    impl Drop for RestoreCacheLimit {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                // Best effort only during test unwinding.
                let _ = set_cache_limit_bytes(previous);
            }
        }
    }

    #[test]
    #[ignore = "mutates process-wide MLX cache policy; run explicitly and serially"]
    fn process_global_cache_controls_and_reclamation_preserve_live_arrays() {
        let stream = cpu_stream();
        let restore = RestoreCacheLimit::disable().expect("disable cache");

        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let live = Array::from_f32_slice(&values, &[4]).expect("live array");
        live.eval().expect("materialize live array");
        stream.synchronize().expect("complete live allocation");

        let before_buffers = array_buffer_bytes(&[&live]).expect("live buffer measurement");

        let report = clear_cache_after_synchronizing(&[&stream]).expect("coordinated cache clear");
        let _ = report.observed_cache_reduction_bytes();

        let after_buffers = array_buffer_bytes(&[&live]).expect("retained live buffer");

        assert_eq!(before_buffers, after_buffers);
        assert_eq!(live.data_f32().expect("live data"), &values);

        drop(live);
        restore.restore().expect("restore cache policy");
    }

    #[test]
    #[ignore = "resets process-wide peak measurements; run explicitly and serially"]
    fn process_global_peak_reset_and_memory_limit_roundtrip() {
        let stream = cpu_stream();
        stream.synchronize().expect("completion boundary");

        let current = memory_limit_bytes().expect("current guideline");

        // Set the same value: validate the previous-value contract
        // without intentionally tightening the process memory policy.
        let previous = set_memory_limit_bytes(current).expect("set same guideline");

        assert_eq!(previous, current);
        assert_eq!(memory_limit_bytes().expect("read guideline"), current,);

        reset_peak_memory().expect("reset peak");
        peak_memory_bytes().expect("read reset measurement");
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "mutates process-wide wired policy; requires supported macOS/Metal"]
    fn process_global_wired_limit_roundtrip() {
        struct RestoreWiredLimit(Option<usize>);

        impl Drop for RestoreWiredLimit {
            fn drop(&mut self) {
                if let Some(previous) = self.0.take() {
                    let _ = set_wired_limit_bytes(previous);
                }
            }
        }

        let device = Device::new(DeviceType::Gpu, 0).expect("Metal device");
        assert!(device.is_available().expect("Metal availability"));
        let stream = Stream::new(&device).expect("Metal stream");
        stream.synchronize().expect("Metal completion boundary");

        let previous = set_wired_limit_bytes(0).expect("temporarily disable wiring");
        let mut restore = RestoreWiredLimit(Some(previous));

        set_wired_limit_bytes(previous).expect("restore wired limit");
        restore.0 = None;
    }
}
