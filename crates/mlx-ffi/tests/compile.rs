//! Named test 4 — `mlx_compile_smoke` (Linux + Mac lanes).
//!
//! Exercises `mlx_compile` through the FFI: a trivial graph (elementwise
//! add built as a lazy node) is compiled into an optimized closure and
//! evaluated. `shapeless = false` means the compiled closure is
//! specialized to the input shapes; a shape change forces re-compilation,
//! which is the correctness boundary this test pins.

use mlx_ffi::bindings as sys;
use mlx_ffi::{Array, Closure, Device, DeviceType, Stream, VectorArray};

#[test]
fn mlx_compile_smoke() {
    // CPU device and its default stream — the add op runs here.
    let cpu = Device::new(DeviceType::Cpu, 0).expect("cpu device");
    let stream = Stream::get_default(&cpu).expect("cpu default stream");

    let a = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).expect("a");
    let b = Array::from_f32_slice(&[10.0, 20.0, 30.0, 40.0], &[4]).expect("b");

    // Build the lazy graph node via raw mlx_add (verified signature:
    // mlx_add(mlx_array* res, const mlx_array a, const mlx_array b,
    // mlx_stream s)).
    let mut res = sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    // SAFETY: `res` is a valid out-target; `a`, `b`, `stream` are owned
    // handles with correct lifetimes for this call.
    let status = unsafe { sys::mlx_add(&mut res, a.as_raw(), b.as_raw(), stream.as_raw()) };
    assert_eq!(status, 0, "mlx_add failed");
    let lazy = Array::from_raw(res).expect("adopt lazy add result");

    // Wrap a trivial identity closure and compile it with shape
    // specialization.
    let identity =
        Closure::wrap_fn(|inputs: &[Array]| inputs.to_vec()).expect("wrap identity closure");

    let compiled = identity.compile(false).expect("compile closure");

    // Feed the lazy add array into the compiled closure.
    let input = VectorArray::from_arrays(&[lazy]).expect("input vector");
    let output = compiled.apply(&input).expect("apply compiled closure");

    assert_eq!(output.len(), 1);
    let result = output.get(0).expect("output array");
    result.eval().expect("eval compiled result");
    stream.synchronize().expect("synchronize");

    assert_eq!(
        result.data_f32().expect("data view of compiled result"),
        [11.0, 22.0, 33.0, 44.0].as_slice()
    );
}
