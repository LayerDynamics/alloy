//! Named test 7 — `quantized_matmul_smoke` (Linux + Mac lanes).
//!
//! Proves the quantized-matmul op surface is reachable through the FFI
//! and numerically sane: quantize a weight matrix with `mlx_quantize`,
//! multiply with `mlx_quantized_matmul`, and compare against a plain
//! Rust f32 reference.
//!
//! Signatures verified against the vendored mlx/c/ops.h (v0.8.0):
//!   mlx_quantize(mlx_vector_array* res, w, mlx_optional_int group_size,
//!                mlx_optional_int bits, const char* mode,
//!                mlx_array global_scale /* may be null */, mlx_stream s)
//!   mlx_quantized_matmul(mlx_array* res, x, w, scales,
//!                mlx_array biases /* may be null */, bool transpose,
//!                mlx_optional_int group_size, mlx_optional_int bits,
//!                const char* mode, mlx_stream s)
//!
//! The weights are chosen so the check is tight: every row holds values
//! k/16 for k in 0..=15, so with group_size = 64, bits = 4 the affine
//! quantizer's scale is exactly 1/16 and bias exactly 0. Quantization
//! is lossless here, which means any mismatch is an FFI or op-wiring bug
//! rather than quantization noise.

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

use mlx_ffi::bindings as sys;
use mlx_ffi::{Array, DType, Device, DeviceType, Stream, VectorArray};

const OUT: usize = 64;
const IN: usize = 64;
const GROUP_SIZE: c_int = 64;
const BITS: c_int = 4;

fn null_array() -> sys::mlx_array {
    sys::mlx_array {
        ctx: ptr::null_mut(),
    }
}

fn some_int(value: c_int) -> sys::mlx_optional_int {
    sys::mlx_optional_int {
        value,
        has_value: true,
    }
}

#[test]
fn quantized_matmul_smoke() {
    let cpu = Device::new(DeviceType::Cpu, 0).expect("cpu device");
    let stream = Stream::get_default(&cpu).expect("cpu default stream");
    let mode = CString::new("affine").expect("mode string");

    // Weights [OUT, IN]: w[i][j] = ((7i + 3j) mod 16) / 16. Since gcd(3,16)=1,
    // each row covers all 16 residues, so each group spans exactly [0, 15/16].
    let w_data: Vec<f32> = (0..OUT)
        .flat_map(|i| (0..IN).map(move |j| (((7 * i + 3 * j) % 16) as f32) / 16.0))
        .collect();
    let w = Array::from_f32_slice(&w_data, &[OUT as i32, IN as i32]).expect("weights");

    // Activations [1, IN].
    let x_data: Vec<f32> = (0..IN).map(|j| ((j % 5) as f32 - 2.0) * 0.25).collect();
    let x = Array::from_f32_slice(&x_data, &[1, IN as i32]).expect("activations");

    // --- quantize: returns [w_q, scales, biases] in affine mode ---------
    let mut q_raw = sys::mlx_vector_array {
        ctx: ptr::null_mut(),
    };
    // SAFETY: valid out-target; handles and the NUL-terminated mode string
    // outlive the call; a null global_scale is documented as allowed.
    let status = unsafe {
        sys::mlx_quantize(
            &mut q_raw,
            w.as_raw(),
            some_int(GROUP_SIZE),
            some_int(BITS),
            mode.as_ptr(),
            null_array(),
            stream.as_raw(),
        )
    };
    assert_eq!(status, 0, "mlx_quantize failed");
    let quantized = VectorArray::from_raw(q_raw).expect("adopt quantize result");
    assert_eq!(
        quantized.len(),
        3,
        "affine quantize returns (w_q, scales, biases)"
    );

    let w_q = quantized.get(0).expect("w_q");
    let scales = quantized.get(1).expect("scales");
    let biases = quantized.get(2).expect("biases");

    // Packed layout: 4-bit values packed into uint32 => IN * BITS / 32 columns.
    assert_eq!(w_q.dtype(), DType::Uint32);
    assert_eq!(w_q.shape(), vec![OUT, IN * BITS as usize / 32]);

    // --- quantized matmul: y = x @ w^T, shape [1, OUT] ------------------
    let mut y_raw = null_array();
    // SAFETY: valid out-target and handles; transpose = true because w is
    // stored [out, in].
    let status = unsafe {
        sys::mlx_quantized_matmul(
            &mut y_raw,
            x.as_raw(),
            w_q.as_raw(),
            scales.as_raw(),
            biases.as_raw(),
            true,
            some_int(GROUP_SIZE),
            some_int(BITS),
            mode.as_ptr(),
            stream.as_raw(),
        )
    };
    assert_eq!(status, 0, "mlx_quantized_matmul failed");
    let y = Array::from_raw(y_raw).expect("adopt matmul result");

    y.eval().expect("eval quantized matmul");
    stream.synchronize().expect("synchronize");

    assert_eq!(y.shape(), vec![1, OUT]);
    assert_eq!(y.dtype(), DType::Float32);

    // --- independent f32 reference --------------------------------------
    let got = y.data_f32().expect("result data");
    for i in 0..OUT {
        let expected: f32 = (0..IN).map(|j| x_data[j] * w_data[i * IN + j]).sum();
        assert!(
            (got[i] - expected).abs() < 1e-3,
            "row {i}: quantized matmul gave {}, reference {expected}",
            got[i]
        );
    }
}

#[test]
fn routed_swiglu_matches_selected_experts() {
    for tokens in [2, 31, 32, 55] {
        check_selected_experts(DeviceType::Cpu, tokens);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn routed_swiglu_matches_selected_experts_metal() {
    for tokens in [2, 31, 32, 55] {
        check_selected_experts(DeviceType::Gpu, tokens);
    }
}

fn check_selected_experts(kind: DeviceType, tokens: usize) {
    use mlx_ffi::ops::{AffineExperts, SwitchGluExperts, routed_swiglu, to_vec_f32};
    let device = Device::new(kind, 0).unwrap();
    let stream = Stream::get_default(&device).unwrap();
    const EXPERTS: usize = 3;
    const WIDTH: usize = 64;
    let diagonal = |offset: usize| {
        let mut packed = vec![0u32; EXPERTS * WIDTH * (WIDTH / 8)];
        for e in 0..EXPERTS {
            for i in 0..WIDTH {
                packed[(e * WIDTH + i) * (WIDTH / 8) + i / 8] =
                    ((e + offset) as u32) << (4 * (i % 8));
            }
        }
        Array::from_u32_slice(&packed, &[3, 64, 8]).unwrap()
    };
    let gate = diagonal(1);
    let up = diagonal(2);
    let down = diagonal(3);
    let scales = Array::from_f32_slice(&[1.; EXPERTS * WIDTH], &[3, 64, 1]).unwrap();
    let bias = Array::from_f32_slice(&[0.; EXPERTS * WIDTH], &[3, 64, 1]).unwrap();
    let bank = |packed| AffineExperts {
        packed,
        scales: &scales,
        biases: &bias,
        group_size: 64,
        bits: 4,
    };
    let mut banks = SwitchGluExperts {
        gate: bank(&gate),
        up: bank(&up),
        down: bank(&down),
    };
    let inputs: Vec<f32> = (0..tokens * WIDTH)
        .map(|i| ((i % 127) as f32 - 62.) / 64.)
        .collect();
    let x = Array::from_f32_slice(&inputs, &[tokens as i32, 64]).unwrap();
    let patterns = [[2.0f32, -1., 1.], [-2., 3., 1.], [1., 2., -1.]];
    let choices = [[0, 2], [1, 2], [1, 0]];
    let logits = (0..tokens)
        .map(|i| patterns[i % patterns.len()])
        .collect::<Vec<_>>();
    let router = Array::from_f32_slice(logits.as_flattened(), &[tokens as i32, 3]).unwrap();
    for normalize in [true, false] {
        let out = routed_swiglu(&x, &router, &banks, 2, normalize, &stream).unwrap();
        assert_eq!(out.shape(), [tokens, WIDTH]);
        let values = to_vec_f32(&out, &stream).unwrap();
        for (row, logit) in logits.iter().enumerate() {
            let selected = choices[row % choices.len()];
            let probs = logit.map(f32::exp);
            let sum: f32 = if normalize {
                selected.iter().map(|&e| probs[e]).sum()
            } else {
                probs.iter().sum()
            };
            for column in 0..WIDTH {
                let x = inputs[row * WIDTH + column];
                let expected: f32 = selected
                    .iter()
                    .map(|&e| {
                        let g = x * (e + 1) as f32;
                        probs[e] / sum
                            * (g / (1. + (-g).exp()))
                            * (x * (e + 2) as f32)
                            * (e + 3) as f32
                    })
                    .sum();
                assert!(
                    (values[row * WIDTH + column] - expected).abs() < 2e-5,
                    "row={row}, column={column}"
                );
            }
        }
    }
    for invalid in [0, 4] {
        assert!(routed_swiglu(&x, &router, &banks, invalid, true, &stream).is_err());
    }
    banks.gate.packed = &scales;
    assert!(routed_swiglu(&x, &router, &banks, 1, true, &stream).is_err());
}
