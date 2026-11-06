//! # mlx-ffi — RAII layer over vendored mlx-c (Phase 0 spike)
//!
//! Safe, ownership-correct wrappers over the mlx-c C API surface this
//! phase exercises: arrays, devices, streams, closures, and `mlx_compile`
//! (SPEC-1 §3.2; the FFI layer is deliberately policy-free and low-level).
//!
//! ## Ownership conventions (verified against the vendored headers)
//!
//! - Handles (`mlx_array`, `mlx_device`, `mlx_stream`, `mlx_closure`,
//!   `mlx_vector_array`, ...) are single-field structs (`{ void* ctx }`)
//!   passed **by value**. A null `ctx` is the empty/failure handle.
//! - Functions returning `int` are status codes: `0` = success. A richer
//!   error surface (mlx/c/error.h) is a post-Phase-0 follow-up; this
//!   layer maps non-zero statuses to [`Error::Status`].
//! - Out-parameters (`mlx_array* res`, ...) yield **owned** handles that
//!   the caller must free; every wrapper here frees in its `Drop`.
//! - `const` by-value arguments are borrowed, never freed by us.
//! - Streams are thread-affine per mlx/c/stream.h: a stream must be used
//!   from its creating thread. The raw-pointer handle field makes all
//!   wrapper types `!Send + !Sync` automatically, which is correct.
//!
//! ## Verified deviations from the mlx-c headers
//!
//! - The data accessors (`mlx_array_data_*`) are documented as returning
//!   NULL for unevaluated arrays. At the vendored pin they instead
//!   dereference a null buffer and segfault. [`Array::data_f32`] therefore
//!   checks [`Array::is_available`] first and never reaches the accessor
//!   for a lazy array.
//! - `DType` and `DeviceType` values are hardcoded from mlx/c/array.h and
//!   mlx/c/device.h (`MLX_BOOL = 0 … MLX_COMPLEX64 = 13`; `MLX_CPU = 0`,
//!   `MLX_GPU = 1`). This file never references bindgen's enum codegen,
//!   which varies by bindgen version.
//!
//! ## Named-test mapping (Phase 0 execution plan)
//!
//! - tests 1, 5: [`Array`] create/clone/eval + [`Stream::synchronize`]
//! - test 2: [`Device`], [`Stream`], default selection
//! - test 3: [`Closure::wrap_fn`] + [`Closure::apply`]
//! - test 4: [`Closure::compile`] (mlx_compile)
//! - tests 6, 7: raw [`bindings`] (unified memory views, quantized ops)

#![warn(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Raw bindgen output over every public mlx-c header. Regenerated only
/// by `scripts/bindgen_check.sh` (pinned bindgen + pinned libclang);
/// checked in and treated as unstable — do not rely on item names.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    missing_docs,
    clippy::all
)]
pub mod bindings;

use std::fmt;
use std::mem;
use std::os::raw::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

use bindings as sys;

/// Explicit-stream rotary position encoding and grouped-query attention.
pub mod attention;
/// MLX allocator observations, limits, and explicit cache reclamation.
pub mod memory;
/// Explicit-stream MLX operations used by native model implementations.
pub mod ops;

/// Errors surfaced by the mlx-c C API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A C call returned a non-zero status code.
    Status(c_int),
    /// A constructor or out-parameter returned the empty (null) handle.
    NullHandle,
    /// The array has not been evaluated, so it has no data buffer yet.
    NotEvaluated,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Status(code) => write!(f, "mlx-c call failed with status {code}"),
            Error::NullHandle => write!(f, "mlx-c returned an empty (null) handle"),
            Error::NotEvaluated => write!(f, "array has not been evaluated"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type for this crate.
pub type Result<T> = std::result::Result<T, Error>;

fn check(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Status(status))
    }
}

/// Element type of an [`Array`]. Mirrors `mlx_dtype` in mlx/c/array.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    /// `bool`
    Bool = 0,
    /// `uint8`
    Uint8 = 1,
    /// `uint16`
    Uint16 = 2,
    /// `uint32`
    Uint32 = 3,
    /// `uint64`
    Uint64 = 4,
    /// `int8`
    Int8 = 5,
    /// `int16`
    Int16 = 6,
    /// `int32`
    Int32 = 7,
    /// `int64`
    Int64 = 8,
    /// `float16`
    Float16 = 9,
    /// `float32`
    Float32 = 10,
    /// `float64`
    Float64 = 11,
    /// `bfloat16`
    BFloat16 = 12,
    /// `complex64`
    Complex64 = 13,
}

// Compile-time check that the hardcoded discriminants match the generated
// constants. If an mlx-c pin ever renumbers mlx_dtype, the build fails here
// instead of silently misreading dtypes.
const _: () = {
    assert!(sys::mlx_dtype__MLX_BOOL == DType::Bool as u32);
    assert!(sys::mlx_dtype__MLX_UINT8 == DType::Uint8 as u32);
    assert!(sys::mlx_dtype__MLX_FLOAT32 == DType::Float32 as u32);
    assert!(sys::mlx_dtype__MLX_COMPLEX64 == DType::Complex64 as u32);
};

impl TryFrom<u32> for DType {
    type Error = Error;

    fn try_from(raw: u32) -> Result<Self> {
        Ok(match raw {
            0 => DType::Bool,
            1 => DType::Uint8,
            2 => DType::Uint16,
            3 => DType::Uint32,
            4 => DType::Uint64,
            5 => DType::Int8,
            6 => DType::Int16,
            7 => DType::Int32,
            8 => DType::Int64,
            9 => DType::Float16,
            10 => DType::Float32,
            11 => DType::Float64,
            12 => DType::BFloat16,
            13 => DType::Complex64,
            _ => return Err(Error::NullHandle),
        })
    }
}

/// Device type. Mirrors `mlx_device_type` in mlx/c/device.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DeviceType {
    /// Host CPU.
    Cpu = 0,
    /// GPU (Metal on Apple Silicon).
    Gpu = 1,
}

/// An owned MLX array.
///
/// Wraps `mlx_array`; freed with `mlx_array_free` on drop. Arrays are
/// lazy: nothing is computed until [`Array::eval`] (or a synchronizing
/// op) runs.
pub struct Array {
    raw: sys::mlx_array,
}

impl Array {
    /// New array from a buffer of `f32` elements (copied) with `shape`
    /// (row-major element counts per dimension).
    pub fn from_f32_slice(data: &[f32], shape: &[i32]) -> Result<Self> {
        // SAFETY: `data`/`shape` are valid for the call and mlx-c copies
        // the buffer synchronously.
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len() as c_int,
                DType::Float32 as u32,
            )
        };
        Array::from_raw(raw)
    }

    /// New array from a buffer of `u32` elements (copied) with `shape`.
    pub fn from_u32_slice(data: &[u32], shape: &[i32]) -> Result<Self> {
        // SAFETY: `data`/`shape` are valid for the call and mlx-c copies
        // the buffer synchronously.
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len() as c_int,
                DType::Uint32 as u32,
            )
        };
        Array::from_raw(raw)
    }

    /// New scalar `f32` array.
    pub fn from_f32(value: f32) -> Result<Self> {
        // SAFETY: plain constructor call.
        let raw = unsafe { sys::mlx_array_new_float32(value) };
        Array::from_raw(raw)
    }

    /// Adopt an owned raw handle (e.g. an op out-parameter result).
    /// The handle must not be owned by anyone else.
    pub fn from_raw(raw: sys::mlx_array) -> Result<Self> {
        if raw.ctx.is_null() {
            Err(Error::NullHandle)
        } else {
            Ok(Array { raw })
        }
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        // SAFETY: valid handle; `mlx_array_size` has no failure mode.
        unsafe { sys::mlx_array_size(self.raw) }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        // SAFETY: valid handle.
        unsafe { sys::mlx_array_ndim(self.raw) }
    }

    /// Shape (element counts per dimension).
    pub fn shape(&self) -> Vec<usize> {
        let ndim = self.ndim();
        (0..ndim)
            .map(|d| {
                // SAFETY: valid handle, in-range dimension index.
                unsafe { sys::mlx_array_dim(self.raw, d as c_int) as usize }
            })
            .collect()
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        // SAFETY: valid handle. An unknown dtype is impossible while the
        // vendored pin and `DType` stay in sync.
        DType::try_from(unsafe { sys::mlx_array_dtype(self.raw) } as u32)
            .expect("mlx_dtype out of sync with DType")
    }

    /// Materialize this (possibly lazy) array.
    pub fn eval(&self) -> Result<()> {
        // SAFETY: valid handle; `mlx_array_eval` takes the handle by value.
        check(unsafe { sys::mlx_array_eval(self.raw) })
    }

    /// Whether the array has been evaluated and its data is available.
    /// Safe to call on lazy arrays.
    pub fn is_available(&self) -> Result<bool> {
        let mut avail = false;
        // SAFETY: valid out-target and handle; this documented internal
        // helper is safe on unevaluated arrays.
        check(unsafe { sys::_mlx_array_is_available(&mut avail, self.raw) })?;
        Ok(avail)
    }

    /// Value of a scalar `float32` array.
    pub fn item_f32(&self) -> Result<f32> {
        let mut out = 0.0f32;
        // SAFETY: `out` is a valid write target for the call.
        check(unsafe { sys::mlx_array_item_float32(&mut out, self.raw) })?;
        Ok(out)
    }

    /// Contiguous `f32` data of an **evaluated** array. Returns
    /// [`Error::NotEvaluated`] for lazy arrays instead of touching their
    /// (null) buffer: at the vendored pin, mlx-c's data accessors do not
    /// return NULL for unevaluated arrays, they dereference a null buffer.
    ///
    /// The borrow is tied to `&self`: the underlying buffer lives as long
    /// as the array handle, and MLX arrays are immutable once evaluated.
    pub fn data_f32(&self) -> Result<&[f32]> {
        if !self.is_available()? {
            return Err(Error::NotEvaluated);
        }
        // SAFETY: valid, available handle.
        let ptr = unsafe { sys::mlx_array_data_float32(self.raw) };
        if ptr.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: mlx-c guarantees `size()` elements, readable for the
        // lifetime of the handle (and the borrow).
        Ok(unsafe { slice::from_raw_parts(ptr, self.size()) })
    }

    /// Raw handle, for tests and future op wrappers.
    pub fn as_raw(&self) -> sys::mlx_array {
        self.raw
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut raw = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: `raw` is a valid out-target; source handle is valid.
        let status = unsafe { sys::mlx_array_set(&mut raw, self.raw) };
        if status != 0 || raw.ctx.is_null() {
            // SAFETY: best-effort free of an owned handle.
            unsafe { sys::mlx_array_free(raw) };
            panic!("mlx_array_set failed with status {status}");
        }
        Array { raw }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: we own exactly one reference to this handle.
        // Status is ignored: failing a destructor is worse than leaking.
        unsafe { sys::mlx_array_free(self.raw) };
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

/// An owned MLX device.
pub struct Device {
    raw: sys::mlx_device,
}

impl Device {
    /// New device of `kind` at `index`.
    pub fn new(kind: DeviceType, index: c_int) -> Result<Self> {
        // SAFETY: plain constructor call.
        let raw = unsafe { sys::mlx_device_new_type(kind as u32, index) };
        if raw.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Device { raw })
    }

    /// The process-wide default device (owned handle).
    pub fn get_default() -> Result<Self> {
        let mut raw = sys::mlx_device {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid out-target; status-checked.
        check(unsafe { sys::mlx_get_default_device(&mut raw) })?;
        if raw.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Device { raw })
    }

    /// Make this the process-wide default device.
    pub fn set_as_default(&self) -> Result<()> {
        // SAFETY: valid handle.
        check(unsafe { sys::mlx_set_default_device(self.raw) })
    }

    /// Device kind.
    pub fn kind(&self) -> Result<DeviceType> {
        let mut ty = 0u32;
        // SAFETY: valid out-target and handle.
        check(unsafe { sys::mlx_device_get_type(&mut ty, self.raw) })?;
        Ok(match ty {
            1 => DeviceType::Gpu,
            _ => DeviceType::Cpu,
        })
    }

    /// Device index within its kind.
    pub fn index(&self) -> Result<c_int> {
        let mut idx = 0;
        // SAFETY: valid out-target and handle.
        check(unsafe { sys::mlx_device_get_index(&mut idx, self.raw) })?;
        Ok(idx)
    }

    /// Number of available devices of `kind`.
    pub fn count(kind: DeviceType) -> Result<c_int> {
        let mut n = 0;
        // SAFETY: valid out-target.
        check(unsafe { sys::mlx_device_count(&mut n, kind as u32) })?;
        Ok(n)
    }

    /// Whether this device is usable on this host.
    pub fn is_available(&self) -> Result<bool> {
        let mut avail = false;
        // SAFETY: valid out-target and handle.
        check(unsafe { sys::mlx_device_is_available(&mut avail, self.raw) })?;
        Ok(avail)
    }

    /// Handle equality (same kind and index).
    pub fn equals(&self, other: &Device) -> bool {
        // SAFETY: both handles valid; `mlx_device_equal` has no failure mode.
        unsafe { sys::mlx_device_equal(self.raw, other.raw) }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: owned handle; status ignored in destructor.
        unsafe { sys::mlx_device_free(self.raw) };
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("kind", &self.kind())
            .field("index", &self.index())
            .finish()
    }
}

/// An owned MLX stream. **Thread-affine**: must be used from the thread
/// that created it (per mlx/c/stream.h); this type is `!Send + !Sync`.
pub struct Stream {
    raw: sys::mlx_stream,
}

impl Stream {
    /// New stream on `device`. Affine to the calling thread.
    pub fn new(device: &Device) -> Result<Self> {
        // SAFETY: valid device handle.
        let raw = unsafe { sys::mlx_stream_new_device(device.raw) };
        if raw.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Stream { raw })
    }

    /// The default stream on `device` (owned handle).
    pub fn get_default(device: &Device) -> Result<Self> {
        let mut raw = sys::mlx_stream {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid out-target and device handle.
        check(unsafe { sys::mlx_get_default_stream(&mut raw, device.raw) })?;
        if raw.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Stream { raw })
    }

    /// Block until all work enqueued on this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: valid handle.
        check(unsafe { sys::mlx_synchronize(self.raw) })
    }

    /// The device this stream runs on.
    pub fn device(&self) -> Result<Device> {
        let mut raw = sys::mlx_device {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid out-target and stream handle.
        check(unsafe { sys::mlx_stream_get_device(&mut raw, self.raw) })?;
        if raw.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Device { raw })
    }

    /// Raw handle.
    pub fn as_raw(&self) -> sys::mlx_stream {
        self.raw
    }
}

/// Synchronize the default stream.
pub fn synchronize_default_stream() -> Result<()> {
    // SAFETY: takes no arguments.
    check(unsafe { sys::mlx_synchronize_default() })
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: owned handle; status ignored in destructor.
        unsafe { sys::mlx_stream_free(self.raw) };
    }
}

/// An owned `mlx_vector_array` (MLX's vector-of-arrays argument type,
/// used by closures and by multi-result ops such as `mlx_quantize`).
pub struct VectorArray {
    raw: sys::mlx_vector_array,
}

impl VectorArray {
    /// Build from [`Array`]s (handles are copied; inputs remain valid).
    pub fn from_arrays(arrays: &[Array]) -> Result<Self> {
        let raws: Vec<sys::mlx_array> = arrays.iter().map(|a| a.as_raw()).collect();
        // SAFETY: `raws` is a valid, length-matched buffer of handles.
        let raw = unsafe { sys::mlx_vector_array_new_data(raws.as_ptr(), raws.len()) };
        VectorArray::from_raw(raw)
    }

    /// Adopt an owned raw handle (e.g. an op out-parameter result).
    /// The handle must not be owned by anyone else.
    pub fn from_raw(raw: sys::mlx_vector_array) -> Result<Self> {
        if raw.ctx.is_null() {
            Err(Error::NullHandle)
        } else {
            Ok(VectorArray { raw })
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        // SAFETY: valid handle; no failure mode.
        unsafe { sys::mlx_vector_array_size(self.raw) }
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Owned copy of the array at `idx`.
    pub fn get(&self, idx: usize) -> Result<Array> {
        let mut raw = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid out-target and handle; an out-of-range index is
        // reported through the status return.
        check(unsafe { sys::mlx_vector_array_get(&mut raw, self.raw, idx) })?;
        Array::from_raw(raw)
    }

    /// Owned copies of all elements.
    pub fn to_arrays(&self) -> Result<Vec<Array>> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    /// Consume the wrapper, yielding the raw handle (transfers ownership
    /// to the caller). Used by the closure trampoline to hand its
    /// result to mlx-c without a double free.
    fn into_raw(mut self) -> sys::mlx_vector_array {
        let raw = self.raw;
        self.raw.ctx = std::ptr::null_mut();
        mem::forget(self);
        raw
    }

    /// Raw handle.
    pub fn as_raw(&self) -> sys::mlx_vector_array {
        self.raw
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        if self.raw.ctx.is_null() {
            return;
        }
        // SAFETY: owned handle; status ignored in destructor.
        unsafe { sys::mlx_vector_array_free(self.raw) };
    }
}

/// The erased payload type behind [`Closure::wrap_fn`].
type ClosurePayload = Box<dyn FnMut(&[Array]) -> Vec<Array>>;

/// An owned MLX closure (`mlx_closure`), a callable over
/// vector-of-arrays.
pub struct Closure {
    raw: sys::mlx_closure,
}

impl Closure {
    /// Wrap a Rust closure as an MLX closure.
    ///
    /// The boxed closure is owned by the MLX closure and freed with it
    /// (via the payload destructor). Panics inside `f` are caught at the
    /// FFI boundary and reported as a status error, never unwound into C.
    pub fn wrap_fn<F>(f: F) -> Result<Self>
    where
        F: FnMut(&[Array]) -> Vec<Array> + 'static,
    {
        // Double-box: `Box<dyn FnMut>` is a fat pointer, so box it again to
        // get a thin pointer that survives a round trip through `void*`.
        let payload: Box<ClosurePayload> = Box::new(Box::new(f));

        // SAFETY: the payload pointer comes from `Box::into_raw` and is
        // released by `payload_dtor`, which mlx-c calls when the closure
        // is freed.
        let raw = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(trampoline),
                Box::into_raw(payload) as *mut c_void,
                Some(payload_dtor),
            )
        };
        if raw.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Closure { raw })
    }

    /// Call the closure. Returns an owned result vector.
    pub fn apply(&self, input: &VectorArray) -> Result<VectorArray> {
        let mut res = sys::mlx_vector_array {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid out-target and handles; `input` is borrowed
        // (const by value) and remains owned by the caller.
        check(unsafe { sys::mlx_closure_apply(&mut res, self.raw, input.as_raw()) })?;
        VectorArray::from_raw(res)
    }

    /// Compile this closure with `mlx_compile` into an optimized
    /// equivalent (SPEC-1: the compiled-decode path's entry point).
    /// `shapeless = true` compiles without shape specialization.
    pub fn compile(&self, shapeless: bool) -> Result<Closure> {
        let mut res = sys::mlx_closure {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid out-target; `self` is borrowed and remains owned
        // by the caller.
        check(unsafe { sys::mlx_compile(&mut res, self.raw, shapeless) })?;
        if res.ctx.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Closure { raw: res })
    }

    /// Raw handle.
    pub fn as_raw(&self) -> sys::mlx_closure {
        self.raw
    }
}

impl Drop for Closure {
    fn drop(&mut self) {
        // SAFETY: owned handle; frees the Rust payload via its dtor.
        unsafe { sys::mlx_closure_free(self.raw) };
    }
}

/// Payload destructor: drops the double-boxed Rust closure.
unsafe extern "C" fn payload_dtor(payload: *mut c_void) {
    // SAFETY: `payload` was produced by `Box::into_raw` in `wrap_fn`.
    drop(unsafe { Box::from_raw(payload as *mut ClosurePayload) });
}

/// Trampoline between mlx-c and a Rust closure.
///
/// Contract: `input` is borrowed (the caller retains ownership); `res` is
/// an out-target that receives an owned `mlx_vector_array` on success.
unsafe extern "C" fn trampoline(
    res: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> c_int {
    // Unwinding into C is undefined behavior: catch everything.
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<sys::mlx_vector_array> {
        // SAFETY: `payload` came from `wrap_fn`'s `Box::into_raw`.
        let f = unsafe { &mut *(payload as *mut ClosurePayload) };

        // Borrowed input: wrap it, read it, then `forget` the wrapper so
        // `Drop` never frees a vector the caller still owns.
        let borrowed = VectorArray { raw: input };
        let inputs = borrowed.to_arrays();
        mem::forget(borrowed);
        let inputs = inputs?;

        let outputs = f(&inputs);
        let owned = VectorArray::from_arrays(&outputs)?;
        Ok(owned.into_raw())
    }));
    match outcome {
        Ok(Ok(raw)) => {
            // SAFETY: `res` is mlx-c's out-target for this call.
            unsafe { *res = raw };
            0
        }
        Ok(Err(e)) => {
            eprintln!("mlx-ffi: closure failed: {e}");
            1
        }
        Err(panic) => {
            eprintln!("mlx-ffi: closure panicked: {panic:?}");
            1
        }
    }
}
