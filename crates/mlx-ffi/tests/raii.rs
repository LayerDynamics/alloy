//! Named tests 1, 2, 3, 5 (Phase 0 execution plan). Lane: Linux + Mac —
//! every test here must pass on both CI lanes.
//!
//!   1. array_raii_roundtrip      — create/dispose through RAII guards,
//!                                  no leaks (run under ASan or `leaks`)
//!   2. device_and_stream_lifecycle — device + stream creation, sync,
//!                                  default selection
//!   3. closure_smoke             — mlx_closure create/call through FFI
//!   5. lazy_eval_and_synchronize — lazy evaluation + stream sync hold
//!
//! Leak checking (test 1's "no leaks" clause) is external to the binary:
//!   - Linux: MLX_FFI_ASAN=1 RUSTFLAGS="-Zsanitizer=address" \
//!            cargo test -p mlx-ffi --test raii
//!   - macOS: cargo test -p mlx-ffi --test raii && \
//!            leaks -atExit -- cargo test -p mlx-ffi --test raii
//! The create/dispose churn loop below exists to give those tools a
//! statistically meaningful number of allocation cycles.
//!
//! This file also settles two ownership caveats flagged in lib.rs:
//! `mlx_vector_array_get` returning owned handles (test 3 will leak or
//! double-free if the assumption is wrong) and the closure trampoline's
//! borrow/transfer split (test 3 exercises both directions).

use mlx_ffi::bindings as sys;
use mlx_ffi::{Array, Closure, Device, DeviceType, Stream, VectorArray};

/// Named test 1: mlx_array create/dispose through RAII guards, no leaks.
#[test]
fn array_raii_roundtrip() {
    // Scalar creation and read-back.
    let a = Array::from_f32(3.5).expect("scalar array");
    assert_eq!(a.size(), 1);
    assert!(a.shape().is_empty(), "scalar shape should be empty");
    assert_eq!(a.dtype(), mlx_ffi::DType::Float32);
    assert_eq!(a.item_f32().expect("item"), 3.5);

    // Buffer creation, eval, and full read-back.
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = Array::from_f32_slice(&data, &[2, 3]).expect("buffer array");
    assert_eq!(b.shape(), vec![2, 3]);
    assert_eq!(b.size(), 6);
    b.eval().expect("eval");
    assert_eq!(b.data_f32().expect("data view"), data.as_slice());

    // Clone is a distinct handle over shared storage: same values,
    // independent lifetimes (drops below prove nothing double-frees).
    let c = b.clone();
    assert_eq!(c.data_f32().expect("clone data"), data.as_slice());
    drop(b);
    assert_eq!(
        c.data_f32().expect("clone survives source drop"),
        data.as_slice()
    );
    drop(c);

    // Create/dispose churn: 1000 scalar arrays, each cloned and read,
    // then dropped. Under ASan/leaks this turns ownership bugs into
    // visible leak reports rather than one-shot accidents.
    for i in 0..1000u32 {
        let t = Array::from_f32(i as f32).expect("churn array");
        assert_eq!(t.item_f32().expect("churn item"), i as f32);
        let _copy = t.clone();
        let _view = t.item_f32();
    }
}

/// Named test 2: device + stream creation, synchronize, default selection.
#[test]
fn device_and_stream_lifecycle() {
    // CPU device always exists on both lanes.
    let cpu = Device::new(DeviceType::Cpu, 0).expect("cpu device");
    assert!(cpu.is_available().expect("cpu availability"));
    assert_eq!(cpu.kind().expect("cpu kind"), DeviceType::Cpu);
    assert!(Device::count(DeviceType::Cpu).expect("cpu count") >= 1);

    // Default-device round trip, restoring the original afterwards —
    // tests share this process, so leave the world as found.
    let original = Device::get_default().expect("original default");
    cpu.set_as_default().expect("set cpu default");
    assert!(
        Device::get_default().expect("read default").equals(&cpu),
        "default device selection did not take effect"
    );
    original.set_as_default().expect("restore default");

    // Stream lifecycle: creation, device affinity, synchronize.
    let stream = Stream::new(&cpu).expect("stream on cpu");
    assert!(stream.device().expect("stream device").equals(&cpu));
    stream.synchronize().expect("stream synchronize");

    let default_stream = Stream::get_default(&cpu).expect("default stream");
    default_stream
        .synchronize()
        .expect("default stream synchronize");
    mlx_ffi::synchronize_default_stream().expect("global default sync");

    // GPU: exercised on the Mac lane, skipped on the Linux CPU lane.
    // Both outcomes pass — presence of Metal is an environment property,
    // not an FFI property.
    let gpu_count = Device::count(DeviceType::Gpu).expect("gpu count");
    if gpu_count >= 1 {
        let gpu = Device::new(DeviceType::Gpu, 0).expect("gpu device");
        assert!(gpu.is_available().expect("gpu availability"));
        assert_eq!(gpu.kind().expect("gpu kind"), DeviceType::Gpu);
        let gpu_stream = Stream::new(&gpu).expect("stream on gpu");
        gpu_stream.synchronize().expect("gpu stream synchronize");
    } else {
        eprintln!("skipping GPU stream checks: no GPU devices on this host");
    }
}

/// Named test 3: mlx_closure create/call through the FFI.
///
/// The closure duplicates its input vector and appends a fixed scalar —
/// exercising payload construction, the trampoline's borrowed-input /
/// transferred-output split, and `mlx_vector_array_get` ownership in
/// both directions.
#[test]
fn closure_smoke() {
    let closure = Closure::wrap_fn(|inputs: &[Array]| {
        let mut out: Vec<Array> = inputs.iter().map(|a| a.clone()).collect();
        out.push(Array::from_f32(42.0).expect("closure scalar"));
        out
    })
    .expect("wrap closure");

    let input = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[3]).expect("input array");
    input.eval().expect("eval input");
    let vector = VectorArray::from_arrays(&[input]).expect("input vector");

    let out = closure.apply(&vector).expect("apply closure");

    // Output vector: one duplicate of the input, plus the scalar.
    assert_eq!(out.len(), 2);
    let duplicated = out.get(0).expect("output[0]");
    duplicated.eval().expect("eval output[0]");
    assert_eq!(
        duplicated.data_f32().expect("output[0] data"),
        [1.0, 2.0, 3.0].as_slice()
    );
    let scalar = out.get(1).expect("output[1]");
    scalar.eval().expect("eval output[1]");
    assert_eq!(scalar.item_f32().expect("output[1] item"), 42.0);

    // Apply twice: closure state and ownership survive reuse.
    let out2 = closure.apply(&vector).expect("re-apply closure");
    assert_eq!(out2.len(), 2);
}

/// Named test 5: lazy evaluation + stream sync semantics through the FFI.
///
/// `mlx_add` is the verified out-param op shape (res, a, b, stream). MLX
/// ops build a lazy graph: the result's data view must be unavailable
/// before `eval`, correct after `eval` + `synchronize`.
#[test]
fn lazy_eval_and_synchronize() {
    let cpu = Device::new(DeviceType::Cpu, 0).expect("cpu device");
    let stream = Stream::get_default(&cpu).expect("cpu default stream");

    let a = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).expect("a");
    let b = Array::from_f32_slice(&[10.0, 20.0, 30.0, 40.0], &[4]).expect("b");

    let mut res = sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    // SAFETY: `res` is a valid out-target; inputs and stream are valid.
    let status = unsafe { sys::mlx_add(&mut res, a.as_raw(), b.as_raw(), stream.as_raw()) };
    assert_eq!(status, 0, "mlx_add failed");
    let c = Array::from_raw(res).expect("adopt lazy result");

    // Lazy: the op result is not available, and the safe accessor
    // refuses to read it.
    assert!(
        !c.is_available().expect("availability probe"),
        "result available before eval, so the graph was not lazy"
    );
    assert!(c.data_f32().is_err(), "data view of a lazy array must fail");

    // Materialize, synchronize, then read.
    c.eval().expect("eval lazy array");
    stream.synchronize().expect("synchronize");
    assert!(c.is_available().expect("availability after eval"));
    assert_eq!(
        c.data_f32().expect("data after eval+sync"),
        [11.0, 22.0, 33.0, 44.0].as_slice()
    );
}
