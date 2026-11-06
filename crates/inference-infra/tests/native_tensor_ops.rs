use inference_infra::{
    error::KernelError,
    ops::{AttentionMask, apply_rope, apply_rope_on_stream, attention, attention_on_stream},
    tensor::Tensor,
};
use mlx_ffi::{Array, Device, DeviceType, Stream, ops as native_ops};

fn stream(kind: DeviceType) -> Stream {
    let device = Device::new(kind, 0).expect("requested native device");
    assert!(
        device.is_available().expect("device availability"),
        "the requested {kind:?} device is required; no substitution or skip",
    );
    let stream = Stream::new(&device).expect("explicit native stream");
    assert_eq!(stream.device().unwrap().kind().unwrap(), kind);
    stream
}

fn input(length: usize, salt: usize) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 11 + salt * 3) % 29) as f32 / 10.0 - 1.4)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2e-4 * (1.0 + expected.abs());
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "element {index}: actual {actual}, expected {expected}, tolerance {tolerance}",
        );
    }
}

// These small host references are test oracles, independent of the native
// layout transforms, native masking, and MLX attention implementation.
fn rope_reference(
    input: &[f32],
    [tokens, heads, width]: [usize; 3],
    position: usize,
    theta: f32,
) -> Vec<f32> {
    let mut output = input.to_vec();
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * width;
            for pair in 0..width / 2 {
                let frequency = f64::from(theta).powf(-2.0 * pair as f64 / width as f64);
                let (sin, cos) = ((position + token) as f64 * frequency).sin_cos();
                let low = f64::from(input[base + pair]);
                let high = f64::from(input[base + width / 2 + pair]);
                output[base + pair] = (low * cos - high * sin) as f32;
                output[base + width / 2 + pair] = (low * sin + high * cos) as f32;
            }
        }
    }
    output
}

fn attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    [tokens, query_heads, width]: [usize; 3],
    [key_tokens, key_heads, _]: [usize; 3],
    scale: f32,
    mask: AttentionMask,
) -> Vec<f32> {
    let mut output = vec![0.0; query.len()];
    for token in 0..tokens {
        let visible = match mask {
            AttentionMask::Full => key_tokens,
            AttentionMask::Causal { offset } => offset
                .saturating_add(token)
                .saturating_add(1)
                .min(key_tokens),
        };
        for query_head in 0..query_heads {
            let key_head = query_head / (query_heads / key_heads);
            let query_base = (token * query_heads + query_head) * width;
            let mut scores: Vec<f64> = (0..visible)
                .map(|key_token| {
                    let key_base = (key_token * key_heads + key_head) * width;
                    (0..width)
                        .map(|dimension| {
                            f64::from(query[query_base + dimension])
                                * f64::from(key[key_base + dimension])
                        })
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - maximum).exp();
            }
            let denominator: f64 = scores.iter().sum();
            for dimension in 0..width {
                output[query_base + dimension] = scores
                    .iter()
                    .enumerate()
                    .map(|(key_token, score)| {
                        let value_index = (key_token * key_heads + key_head) * width + dimension;
                        score / denominator * f64::from(value[value_index])
                    })
                    .sum::<f64>() as f32;
            }
        }
    }
    output
}

fn check_attention_case(
    stream: &Stream,
    query_shape: [usize; 3],
    key_shape: [usize; 3],
    cases: &[(f32, AttentionMask)],
) {
    let query = input(query_shape.iter().product(), 1);
    let key = input(key_shape.iter().product(), 4);
    let value = input(key_shape.iter().product(), 7);
    let q = Tensor::new(&query_shape, query.clone()).unwrap();
    let k = Tensor::new(&key_shape, key.clone()).unwrap();
    let v = Tensor::new(&key_shape, value.clone()).unwrap();
    for &(scale, mask) in cases {
        let output = attention_on_stream(&q, &k, &v, scale, mask, stream).unwrap();
        assert_eq!(output.shape(), query_shape);
        assert!(
            !output.as_array().is_available().unwrap(),
            "attention must return an unevaluated native graph",
        );
        let expected =
            attention_reference(&query, &key, &value, query_shape, key_shape, scale, mask);
        assert_close(output.data_on_stream(stream).unwrap(), &expected);
    }
}

fn native_operation_suite(kind: DeviceType) {
    let stream = stream(kind);
    let shape = [3, 4, 8];
    let data = input(shape.iter().product(), 2);
    let tensor = Tensor::new(&shape, data.clone()).unwrap();
    for (position, theta) in [(0, 10_000.0), (7, 10_000.0), (13, 500_000.0)] {
        let output = apply_rope_on_stream(&tensor, position, theta, &stream).unwrap();
        assert_eq!(output.shape(), shape);
        assert!(
            !output.as_array().is_available().unwrap(),
            "RoPE must return an unevaluated native graph",
        );
        assert_close(
            output.data_on_stream(&stream).unwrap(),
            &rope_reference(&data, shape, position, theta),
        );
    }
    assert_eq!(tensor.data_on_stream(&stream).unwrap(), data);

    check_attention_case(
        &stream,
        [3, 4, 8],
        [5, 2, 8],
        &[
            (1.25, AttentionMask::Full),
            (0.0, AttentionMask::Full),
            (-0.75, AttentionMask::Full),
            (1.25, AttentionMask::Causal { offset: 0 }),
            (-0.75, AttentionMask::Causal { offset: 1 }),
            (0.5, AttentionMask::Causal { offset: 2 }),
            (0.8, AttentionMask::Causal { offset: 4 }),
            (0.8, AttentionMask::Causal { offset: 9 }),
            (0.8, AttentionMask::Causal { offset: usize::MAX }),
        ],
    );
    // Query chunks may extend beyond the available keys. Each later row
    // must keep all existing keys, without offset or mask arithmetic overflow.
    check_attention_case(
        &stream,
        [5, 4, 8],
        [2, 2, 8],
        &[(0.7, AttentionMask::Causal { offset: 0 })],
    );
}

#[test]
fn native_cpu_rope_and_attention_match_independent_references() {
    native_operation_suite(DeviceType::Cpu);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_metal_rope_and_attention_match_independent_references() {
    native_operation_suite(DeviceType::Gpu);
}

fn strided_tensor_suite(kind: DeviceType) {
    let stream = stream(kind);
    let data: Vec<f32> = (0..32).map(|value| value as f32).collect();
    let source = Array::from_f32_slice(&data, &[2, 4, 4]).unwrap();
    let doubled = native_ops::residual_add(&source, &source, &stream).unwrap();
    let transposed = native_ops::transpose(&doubled, &[1, 0, 2], &stream).unwrap();
    let cropped = native_ops::slice(&transposed, &[1, 0, 0], &[3, 2, 4], &stream).unwrap();
    assert!(!cropped.is_available().unwrap());
    let tensor = Tensor::from_array(cropped).unwrap();
    assert!(!tensor.as_array().is_available().unwrap());
    assert_eq!(tensor.shape(), [2, 2, 4]);
    let mut expected = Vec::new();
    for token in 1..3 {
        for head in 0..2 {
            for dimension in 0..4 {
                expected.push(data[(head * 4 + token) * 4 + dimension] * 2.0);
            }
        }
    }
    let readback = tensor.data_on_stream(&stream).unwrap();
    assert_eq!(readback, expected);
    assert_eq!(
        readback.as_ptr(),
        tensor.data_on_stream(&stream).unwrap().as_ptr(),
        "repeated borrowed readback must retain its immutable native buffer",
    );

    let good = Tensor::new(&[2, 2, 4], input(16, 3)).unwrap();
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = input(24, 5);
        values[23] = nonfinite;
        let raw = Array::from_f32_slice(&values, &[2, 3, 4]).unwrap();
        let transposed = native_ops::transpose(&raw, &[1, 0, 2], &stream).unwrap();
        let cropped = native_ops::slice(&transposed, &[1, 0, 0], &[3, 2, 4], &stream).unwrap();
        let bad = Tensor::from_array(cropped).unwrap();
        assert_eq!(
            apply_rope_on_stream(&bad, 1, 10_000.0, &stream).unwrap_err(),
            KernelError::NonFinite,
        );
        for (q, k, v) in [
            (&bad, &good, &good),
            (&good, &bad, &good),
            (&good, &good, &bad),
        ] {
            assert_eq!(
                attention_on_stream(q, k, v, 1.0, AttentionMask::Full, &stream).unwrap_err(),
                KernelError::NonFinite,
            );
        }
        assert_eq!(
            attention_on_stream(&good, &good, &good, nonfinite, AttentionMask::Full, &stream)
                .unwrap_err(),
            KernelError::NonFinite,
        );
    }
}

#[test]
fn native_cpu_strided_readback_and_finite_validation_are_safe() {
    strided_tensor_suite(DeviceType::Cpu);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_metal_strided_readback_and_finite_validation_are_safe() {
    strided_tensor_suite(DeviceType::Gpu);
}

#[test]
fn tensor_rejects_unsafe_shapes_and_native_dtypes_before_use() {
    for shape in [
        vec![usize::MAX, 2],
        vec![i32::MAX as usize + 1, 0],
        vec![i32::MAX as usize, i32::MAX as usize, 8],
        vec![i32::MAX as usize, i32::MAX as usize],
    ] {
        assert!(
            Tensor::new(&shape, Vec::new()).is_err(),
            "unsafe dimensions must not reach the native buffer constructor: {shape:?}",
        );
    }
    assert_eq!(
        Tensor::new(&[2, 3], vec![0.0; 5]).unwrap_err(),
        KernelError::ShapeMismatch {
            expected: 6,
            got: 5
        },
    );
    let integers = Array::from_u32_slice(&[1, 2], &[2]).unwrap();
    assert!(Tensor::from_array(integers).is_err());

    let stream = stream(DeviceType::Cpu);
    let empty = Tensor::new(&[0, 2], Vec::new()).unwrap();
    assert!(empty.data_on_stream(&stream).unwrap().is_empty());
    let scalar = Tensor::new(&[], vec![3.5]).unwrap();
    assert_eq!(scalar.data_on_stream(&stream).unwrap(), &[3.5]);
}

#[test]
fn native_kernels_reject_incompatible_shapes_and_positions_without_panicking() {
    let stream = stream(DeviceType::Cpu);
    let q = Tensor::new(&[2, 2, 4], vec![0.0; 16]).unwrap();
    let wrong_width = Tensor::new(&[3, 1, 2], vec![0.0; 6]).unwrap();
    let wrong_heads = Tensor::new(&[3, 3, 4], vec![0.0; 36]).unwrap();
    let rank_two = Tensor::new(&[2, 4], vec![0.0; 8]).unwrap();
    for key in [&wrong_width, &wrong_heads, &rank_two] {
        assert!(attention_on_stream(&q, key, key, 1.0, AttentionMask::Full, &stream).is_err());
    }
    let no_heads = Tensor::new(&[1, 0, 4], Vec::new()).unwrap();
    assert!(
        attention_on_stream(&q, &no_heads, &no_heads, 1.0, AttentionMask::Full, &stream).is_err()
    );
    let empty = Tensor::new(&[0, 1, 4], Vec::new()).unwrap();
    assert_eq!(
        attention_on_stream(&q, &empty, &empty, 1.0, AttentionMask::Full, &stream).unwrap_err(),
        KernelError::Empty,
    );
    let odd = Tensor::new(&[1, 1, 3], vec![0.0; 3]).unwrap();
    assert!(apply_rope_on_stream(&odd, 0, 10_000.0, &stream).is_err());
    assert!(apply_rope_on_stream(&q, i32::MAX as usize, 10_000.0, &stream).is_err());
    assert!(apply_rope_on_stream(&q, usize::MAX, 10_000.0, &stream).is_err());
    assert!(apply_rope_on_stream(&q, 0, 0.0, &stream).is_err());
}

#[test]
fn convenience_kernels_use_the_native_default_device() {
    let device = Device::get_default().unwrap();
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    assert_eq!(device.kind().unwrap(), DeviceType::Gpu);
    let stream = Stream::get_default(&device).unwrap();
    let tensor = Tensor::new(&[3, 2, 4], input(24, 1)).unwrap();
    let rope = apply_rope(&tensor, 3, 10_000.0).unwrap();
    assert!(!rope.as_array().is_available().unwrap());
    let explicit = apply_rope_on_stream(&tensor, 3, 10_000.0, &stream).unwrap();
    assert_close(
        rope.data().unwrap(),
        explicit.data_on_stream(&stream).unwrap(),
    );
    let output = attention(
        &tensor,
        &tensor,
        &tensor,
        -0.4,
        AttentionMask::Causal { offset: 0 },
    )
    .unwrap();
    assert!(!output.as_array().is_available().unwrap());
    let explicit = attention_on_stream(
        &tensor,
        &tensor,
        &tensor,
        -0.4,
        AttentionMask::Causal { offset: 0 },
        &stream,
    )
    .unwrap();
    assert_close(
        output.data().unwrap(),
        explicit.data_on_stream(&stream).unwrap(),
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_cpu_and_metal_share_the_same_input_arrays() {
    let cpu = stream(DeviceType::Cpu);
    let metal = stream(DeviceType::Gpu);
    // Construct each input once. Both devices consume these same MLX
    // handles, including a lazy result produced on the other device.
    let q = Tensor::new(&[3, 4, 8], input(96, 1)).unwrap();
    let k = Tensor::new(&[5, 2, 8], input(80, 2)).unwrap();
    let v = Tensor::new(&[5, 2, 8], input(80, 3)).unwrap();
    let cpu_rope = apply_rope_on_stream(&q, 7, 10_000.0, &cpu).unwrap();
    let metal_rope = apply_rope_on_stream(&q, 7, 10_000.0, &metal).unwrap();
    assert_close(
        cpu_rope.data_on_stream(&cpu).unwrap(),
        metal_rope.data_on_stream(&metal).unwrap(),
    );
    for mask in [AttentionMask::Full, AttentionMask::Causal { offset: 1 }] {
        let cpu_output = attention_on_stream(&q, &k, &v, 0.6, mask, &cpu).unwrap();
        let metal_output = attention_on_stream(&q, &k, &v, 0.6, mask, &metal).unwrap();
        // Read each result through the other device's stream as well.
        assert_close(
            cpu_output.data_on_stream(&metal).unwrap(),
            metal_output.data_on_stream(&cpu).unwrap(),
        );
    }
    let from_metal = apply_rope_on_stream(&q, 2, 10_000.0, &metal).unwrap();
    assert!(!from_metal.as_array().is_available().unwrap());
    let consumed_on_cpu =
        attention_on_stream(&from_metal, &k, &v, 0.5, AttentionMask::Full, &cpu).unwrap();
    let consumed_on_metal =
        attention_on_stream(&from_metal, &k, &v, 0.5, AttentionMask::Full, &metal).unwrap();
    assert_close(
        consumed_on_cpu.data_on_stream(&cpu).unwrap(),
        consumed_on_metal.data_on_stream(&metal).unwrap(),
    );
}
