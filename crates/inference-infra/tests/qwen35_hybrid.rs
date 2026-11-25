//! Complete recurrent-block checks; these do not qualify an Ornith profile.
use inference_infra::decoder::{
    fixture::MatrixWeight,
    hybrid::{LinearAttentionConfig, LinearAttentionWeights, MlxLinearAttention},
    mlx_model::MlxDecoderError,
};
use mlx_ffi::{Array, Device, DeviceType, Stream, ops};
use serde_json::Value;
use std::cell::Cell;

fn streams() -> Vec<Stream> {
    let mut kinds = vec![DeviceType::Cpu];
    if cfg!(target_os = "macos") {
        kinds.push(DeviceType::Gpu);
    }
    kinds
        .into_iter()
        .map(|kind| {
            let device = Device::new(kind, 0).unwrap();
            assert!(device.is_available().unwrap());
            Stream::new(&device).unwrap()
        })
        .collect()
}
fn config() -> LinearAttentionConfig {
    LinearAttentionConfig {
        hidden_size: 64,
        key_heads: 1,
        value_heads: 2,
        key_dim: 128,
        value_dim: 128,
        convolution_kernel: 4,
        rms_epsilon: 1e-6,
    }
}
fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + salt * 13) % 61) as f32 - 30.0) / 256.0)
        .collect()
}
fn weights() -> LinearAttentionWeights {
    LinearAttentionWeights {
        qkv: MatrixWeight::f32(values(512 * 64, 1)),
        z: MatrixWeight::f32(values(256 * 64, 2)),
        a: MatrixWeight::f32(values(2 * 64, 3)),
        b: MatrixWeight::f32(values(2 * 64, 4)),
        convolution: values(512 * 4, 5),
        a_log: values(2, 6),
        dt_bias: values(2, 7),
        norm: values(128, 8).into_iter().map(|x| 1.0 + x).collect(),
        output: MatrixWeight::f32(values(64 * 256, 9)),
    }
}
fn input(start: usize, end: usize) -> Array {
    Array::from_f32_slice(
        &values(5 * 64, 10)[start * 64..end * 64],
        &[1, (end - start) as i32, 64],
    )
    .unwrap()
}
fn read(array: &Array, stream: &Stream) -> Vec<f32> {
    ops::to_vec_f32(array, stream).unwrap()
}
fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        maximum = maximum.max((a - e).abs());
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() < 2e-5,
            "{label}[{i}]: {a} vs {e}"
        );
    }
    eprintln!("{label} max_absolute={maximum}");
}
fn expected(reference: &Value, key: &str) -> Vec<f32> {
    reference[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
#[test]
fn qwen35_linear_projection_layout() {
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/qwen35_hybrid/reference.json")).unwrap();
    assert_eq!(reference["mlx"], "0.32.4");
    assert_eq!(reference["mlx_lm"], "0.31.2");
    assert_eq!(reference["compile"], false);
    for stream in streams() {
        let block =
            MlxLinearAttention::from_weights(config(), weights(), 1 << 20, &stream).unwrap();
        let (output, state) = block
            .forward(&input(0, 5), None, 5, 1 << 24, &stream, &|| false)
            .unwrap();
        assert_eq!(state.len(), 5);
        assert_eq!(block.state_bytes().unwrap(), (1536 + 32768) * 4);
        close(
            &read(&output, &stream),
            &expected(&reference, "output"),
            "upstream output",
        );
        close(
            &read(state.convolution(), &stream),
            &expected(&reference, "convolution"),
            "upstream convolution",
        );
        close(
            &read(state.recurrent(), &stream),
            &expected(&reference, "recurrent"),
            "upstream recurrent",
        );
    }
}
#[test]
fn qwen35_recurrent_chunk_equivalence() {
    for stream in streams() {
        let block =
            MlxLinearAttention::from_weights(config(), weights(), 1 << 20, &stream).unwrap();
        let (full, final_state) = block
            .forward(&input(0, 5), None, 5, 1 << 24, &stream, &|| false)
            .unwrap();
        let expected = read(&full, &stream);
        for split in 1..5 {
            let (prefix, state) = block
                .forward(&input(0, split), None, 5, 1 << 24, &stream, &|| false)
                .unwrap();
            let old_conv = read(state.convolution(), &stream);
            let old_recurrent = read(state.recurrent(), &stream);
            let (suffix, next) = block
                .forward(&input(split, 5), Some(&state), 5, 1 << 24, &stream, &|| {
                    false
                })
                .unwrap();
            close(&read(&prefix, &stream), &expected[..split * 64], "prefix");
            close(&read(&suffix, &stream), &expected[split * 64..], "suffix");
            close(
                &read(next.convolution(), &stream),
                &read(final_state.convolution(), &stream),
                "chunk convolution",
            );
            close(
                &read(next.recurrent(), &stream),
                &read(final_state.recurrent(), &stream),
                "chunk recurrent",
            );
            assert_eq!(old_conv, read(state.convolution(), &stream));
            assert_eq!(old_recurrent, read(state.recurrent(), &stream));
        }
        let mut state = None;
        for i in 0..5 {
            let (out, next) = block
                .forward(
                    &input(i, i + 1),
                    state.as_ref(),
                    5,
                    1 << 24,
                    &stream,
                    &|| false,
                )
                .unwrap();
            close(
                &read(&out, &stream),
                &expected[i * 64..(i + 1) * 64],
                "token continuation",
            );
            state = Some(next);
        }
        close(
            &read(state.unwrap().recurrent(), &stream),
            &read(final_state.recurrent(), &stream),
            "token final state",
        );
    }
}
#[test]
fn qwen35_recurrent_owner_limits_and_reset() {
    for stream in streams() {
        let block =
            MlxLinearAttention::from_weights(config(), weights(), 1 << 20, &stream).unwrap();
        let other =
            MlxLinearAttention::from_weights(config(), weights(), 1 << 20, &stream).unwrap();
        let (initial, prior) = block
            .forward(&input(0, 2), None, 5, 1 << 24, &stream, &|| false)
            .unwrap();
        let before = read(prior.recurrent(), &stream);
        assert!(
            other
                .forward(&input(2, 3), Some(&prior), 5, 1 << 24, &stream, &|| false)
                .is_err()
        );
        assert!(
            block
                .forward(&input(2, 3), Some(&prior), 2, 1 << 24, &stream, &|| false)
                .is_err()
        );
        assert!(matches!(
            block.forward(&input(2, 3), Some(&prior), 5, 0, &stream, &|| false),
            Err(MlxDecoderError::PayloadLimit { .. })
        ));
        for cancel_at in [0, 1] {
            let calls = Cell::new(0);
            let cancelled = || {
                let n = calls.get();
                calls.set(n + 1);
                n >= cancel_at
            };
            assert!(matches!(
                block.forward(&input(2, 3), Some(&prior), 5, 1 << 24, &stream, &cancelled),
                Err(MlxDecoderError::Cancelled)
            ));
        }
        let bad = Array::from_f32_slice(&[f32::NAN; 64], &[1, 1, 64]).unwrap();
        assert!(
            block
                .forward(&bad, Some(&prior), 5, 1 << 24, &stream, &|| false)
                .is_err()
        );
        assert_eq!(before, read(prior.recurrent(), &stream));
        assert_eq!(prior.len(), 2);
        let (reset, _) = block
            .forward(&input(0, 2), None, 5, 1 << 24, &stream, &|| false)
            .unwrap();
        assert_eq!(read(&initial, &stream), read(&reset, &stream));
        assert!(MlxLinearAttention::from_weights(config(), weights(), 0, &stream).is_err());
        let mut bad = weights();
        bad.convolution[0] = f32::INFINITY;
        assert!(MlxLinearAttention::from_weights(config(), bad, 1 << 20, &stream).is_err());
        let mut bad = weights();
        bad.qkv = MatrixWeight::f32(vec![0.0]);
        assert!(MlxLinearAttention::from_weights(config(), bad, 1 << 20, &stream).is_err());
        let mut bad = config();
        bad.key_heads = usize::MAX;
        assert!(MlxLinearAttention::from_weights(bad, weights(), 1 << 20, &stream).is_err());
    }
}

// Packed projections exercise the real native quantized path against the exact
// same represented dense values. This is separate from artifact quantization
// error and never replaces the upstream reference comparison above.
#[test]
fn qwen35_packed_projections_preserve_recurrent_state() {
    fn pair(rows: usize, cols: usize, salt: usize) -> (MatrixWeight, MatrixWeight) {
        let codes: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 7 + salt) % 16) as u32)
            .collect();
        let dense = codes.iter().map(|&n| n as f32 / 128.0 - 0.0625).collect();
        let packed = codes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| c.iter().enumerate().fold(0, |v, (i, n)| v | (n << (i * 4))))
            .collect();
        (
            MatrixWeight::Affine4 {
                packed,
                scales: vec![1.0 / 128.0; rows * cols / 64],
                biases: vec![-0.0625; rows * cols / 64],
                rows,
                cols,
                group_size: 64,
            },
            MatrixWeight::f32(dense),
        )
    }
    for stream in streams() {
        let mut packed = weights();
        let mut dense = weights();
        (packed.qkv, dense.qkv) = pair(512, 64, 1);
        (packed.z, dense.z) = pair(256, 64, 2);
        (packed.a, dense.a) = pair(2, 64, 3);
        (packed.b, dense.b) = pair(2, 64, 4);
        (packed.output, dense.output) = pair(64, 256, 5);
        let packed = MlxLinearAttention::from_weights(config(), packed, 1 << 20, &stream).unwrap();
        let dense = MlxLinearAttention::from_weights(config(), dense, 1 << 20, &stream).unwrap();
        assert!(packed.weight_bytes() < dense.weight_bytes() / 3);
        let (p, ps) = packed
            .forward(&input(0, 5), None, 5, 1 << 24, &stream, &|| false)
            .unwrap();
        let (d, ds) = dense
            .forward(&input(0, 5), None, 5, 1 << 24, &stream, &|| false)
            .unwrap();
        close(&read(&p, &stream), &read(&d, &stream), "packed output");
        close(
            &read(ps.convolution(), &stream),
            &read(ds.convolution(), &stream),
            "packed convolution",
        );
        close(
            &read(ps.recurrent(), &stream),
            &read(ds.recurrent(), &stream),
            "packed recurrent",
        );
    }
}

#[test]
fn qwen35_block_operations_reject_invalid_metadata() {
    for stream in streams() {
        let input = Array::from_f32_slice(&[1.0, 2.0], &[1, 2]).unwrap();
        assert!(ops::broadcast_to(&input, &[3, 3], &stream).is_err());
        assert!(ops::broadcast_to(&input, &[2], &stream).is_err());
        assert!(ops::broadcast_to(&input, &[0, 2], &stream).is_err());
        assert!(ops::broadcast_to(&input, &[usize::MAX, usize::MAX, 2], &stream).is_err());
        assert!(ops::concatenate_axis(&input, &input, 2, &stream).is_err());
        let other = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        assert!(ops::concatenate_axis(&input, &other, 0, &stream).is_err());
        assert!(ops::scale(&input, f32::INFINITY, &stream).is_err());
        assert!(ops::rms_norm_unweighted(&input, 0.0, &stream).is_err());
        assert!(ops::rms_norm_unweighted(&input, f32::NAN, &stream).is_err());
    }
}
