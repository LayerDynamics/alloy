//! Named test 6 — `cpu_gpu_views_alias_memory` (Mac lane only, per the
//! Phase 0 execution plan; this file compiles to nothing on Linux).
//!
//! Proves, through the FFI, that a GPU-resident array and its CPU data
//! view are the same memory — the FR-17 zero-copy property on Apple
//! unified memory. mlx-c exposes NO transfer/copy API (verified against
//! metal.h / memory.h): there is no `to_device`, no staging, nothing.
//! So the only way a CPU pointer can read data the Metal allocator is
//! holding for the GPU is shared memory.
//!
//! Evidence chain:
//!   1. An array created while the GPU is the default device is resident
//!      on the Metal allocator.
//!   2. `mlx_array_data_float32` (the CPU view) returns non-null and the
//!      correct values after eval + stream sync.
//!   3. The CPU-view pointer is stable across synchronize calls — it is
//!      the array's own buffer, not a per-call staging copy.
//!   4. `mlx_get_active_memory` does not grow when the CPU view is
//!      taken — reading GPU memory from the CPU allocates nothing.
//!
//! This test mutates the process-wide default device (GPU). Integration
//! tests each run in their own process, so this cannot disturb the
//! other named-test binaries.

#![cfg(target_os = "macos")]

use mlx_ffi::bindings as sys;
use mlx_ffi::{Array, Device, DeviceType};

/// Named test 6: CPU/GPU views of one array alias the same memory.
#[test]
fn cpu_gpu_views_alias_memory() {
    // Metal required: skip (not fail) on hosts without it.
    let mut metal = false;
    // SAFETY: `metal` is a valid out-target for this call.
    let status = unsafe { sys::mlx_metal_is_available(&mut metal) };
    assert_eq!(status, 0, "mlx_metal_is_available failed");
    if !metal {
        eprintln!("skipping cpu_gpu_views_alias_memory: no Metal on this host");
        return;
    }

    // GPU becomes the default device: arrays created from here on are
    // resident on the Metal allocator.
    let gpu = Device::new(DeviceType::Gpu, 0).expect("GPU device 0");
    assert!(gpu.is_available().expect("availability probe"));
    gpu.set_as_default().expect("set GPU as default device");
    let default = Device::get_default().expect("read default device");
    assert!(default.equals(&gpu), "default device did not stick");

    let data = [1.0f32, 2.0, 3.0, 4.0];
    let a = Array::from_f32_slice(&data, &[4]).expect("array on GPU device");
    assert_eq!(a.dtype(), mlx_ffi::DType::Float32);
    assert_eq!(a.size(), 4);

    // Materialize on the GPU stream and wait for the device.
    a.eval().expect("eval on GPU");
    mlx_ffi::synchronize_default_stream().expect("synchronize GPU stream");

    // (2) CPU view exists and is correct for GPU-resident data. With no
    // transfer API in mlx-c, this read can only work if the CPU pointer
    // addresses the same memory the GPU uses.
    let view = a.data_f32().expect("CPU view of GPU-resident array");
    assert_eq!(view, data.as_slice(), "CPU view of GPU memory is wrong");

    // (3) Pointer stability: re-synchronizing must not move the view.
    let ptr = view.as_ptr();
    mlx_ffi::synchronize_default_stream().expect("synchronize again");
    let view2 = a.data_f32().expect("CPU view after re-sync");
    assert_eq!(
        view2.as_ptr(),
        ptr,
        "CPU view moved across synchronize — staging copy, not aliasing"
    );

    // (4) Zero-copy: taking the CPU view allocates nothing on the
    // device allocator holding the array.
    let before = active_memory();
    let _ = a.data_f32().expect("CPU view (memory probe)");
    let after = active_memory();
    assert_eq!(
        before, after,
        "taking a CPU view of GPU memory allocated memory — not zero-copy"
    );

    // Sanity: the array's bytes are accounted on the active allocator.
    // SAFETY: valid array handle.
    let nbytes = unsafe { sys::mlx_array_nbytes(a.as_raw()) };
    assert!(
        after >= nbytes,
        "active memory ({after} B) does not account for the array ({nbytes} B)"
    );
}

/// Bytes currently held by the active device allocator (memory.h).
fn active_memory() -> usize {
    let mut bytes = 0usize;
    // SAFETY: `bytes` is a valid out-target for this call.
    let status = unsafe { sys::mlx_get_active_memory(&mut bytes) };
    assert_eq!(status, 0, "mlx_get_active_memory failed");
    bytes
}
