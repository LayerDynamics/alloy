//! Independent native Gemma3 model checks; tiny fixtures do not qualify a profile.
use inference_infra::decoder::{
    gemma3::{
        Gemma3Config,
        model::MlxGemma3Model,
        weights::{Gemma3Weights, load_verified},
    },
    loader::{CheckpointVerification, LoadOptions},
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
const LIMIT: u64 = 32 * 1024 * 1024;
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma3_model")
}
fn pins(root: &Path) -> BTreeMap<PathBuf, [u8; 32]> {
    let manifest: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/gemma3_model/manifest.json")).unwrap();
    manifest
        .into_iter()
        .map(|(name, hex)| {
            let bytes: Vec<u8> = (0..64)
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            (root.join(name), bytes.try_into().unwrap())
        })
        .collect()
}
fn weights() -> (Gemma3Config, Gemma3Weights) {
    let root = root();
    let pins = pins(&root);
    load_verified(
        &root,
        LoadOptions {
            max_file_bytes: LIMIT as usize,
            max_weight_bytes: LIMIT as usize,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap()
}
fn streams() -> Vec<(&'static str, Stream)> {
    let mut kinds = vec![("cpu", DeviceType::Cpu)];
    if cfg!(target_os = "macos") {
        kinds.push(("gpu", DeviceType::Gpu));
    }
    kinds
        .into_iter()
        .map(|(name, kind)| {
            let d = Device::new(kind, 0).unwrap();
            assert!(d.is_available().unwrap());
            (name, Stream::new(&d).unwrap())
        })
        .collect()
}
fn reference() -> Value {
    let bytes = include_bytes!("fixtures/gemma3_model/reference.json");
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/gemma3_model/manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        manifest["reference.json"].as_str().unwrap()
    );
    serde_json::from_slice(bytes).unwrap()
}
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn tokens(v: &Value) -> Vec<u32> {
    v["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}
fn strict(actual: &[f32], expected: &Value) {
    let expected = numbers(expected);
    assert_eq!(actual.len(), expected.len());
    let mut absolute = 0.0_f64;
    let mut relative = 0.0_f64;
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite());
        let error = (f64::from(a) - b).abs();
        absolute = absolute.max(error);
        if b == 0.0 {
            assert_eq!(error, 0.0);
        } else {
            relative = relative.max(error / b.abs());
        }
    }
    assert!(
        absolute < 0.02 && relative < 0.01,
        "FR8 max_abs={absolute} max_pointwise_relative={relative}"
    );
}
fn model(stream: &Stream) -> MlxGemma3Model {
    let (c, w) = weights();
    MlxGemma3Model::from_weights(c, w, LIMIT, stream).unwrap()
}
fn eight_bit_weights(mut w: Gemma3Weights) -> Gemma3Weights {
    use inference_infra::decoder::fixture::MatrixWeight;
    fn expand(m: &mut MatrixWeight) {
        let MatrixWeight::Affine4 {
            packed,
            scales,
            biases,
            rows,
            cols,
            group_size,
        } = m
        else {
            panic!("expected packed 4-bit fixture")
        };
        let bytes = packed
            .iter()
            .flat_map(|word| {
                [0, 4].map(|start| {
                    (0..4).fold(0u32, |out, i| {
                        out | (((word >> (4 * (start + i))) & 15) << (8 * i))
                    })
                })
            })
            .collect();
        *m = MatrixWeight::Affine8 {
            packed: bytes,
            scales: scales.clone(),
            biases: biases.clone(),
            rows: *rows,
            cols: *cols,
            group_size: *group_size,
        };
    }
    expand(&mut w.embedding);
    if let Some(h) = &mut w.head {
        expand(h);
    }
    for l in &mut w.layers {
        for m in [
            &mut l.q,
            &mut l.k,
            &mut l.v,
            &mut l.output,
            &mut l.gate,
            &mut l.up,
            &mut l.down,
        ] {
            expand(m);
        }
    }
    w
}
#[test]
fn gemma3_reference_logits() {
    let r = reference();
    let tokens = tokens(&r);
    for (name, stream) in streams() {
        let m = model(&stream);
        let out = m.forward(&tokens, LIMIT, LIMIT, LIMIT, &stream).unwrap();
        strict(out.values(), &r[name]["full"]);
        assert_eq!(out.rows(), tokens.len());
        assert_eq!(
            m.weight_payload_bytes(),
            m.config().weight_payload_plan().unwrap().payload_bytes
        );
        assert!(m.weight_buffer_bytes().unwrap() >= m.weight_payload_bytes());
        let (mut c, w) = weights();
        c.quantization_bits = 8;
        let eight = MlxGemma3Model::from_weights(c, eight_bit_weights(w), LIMIT, &stream).unwrap();
        strict(
            eight
                .forward(&tokens, LIMIT, LIMIT, LIMIT, &stream)
                .unwrap()
                .values(),
            &r[name]["full"],
        );
        assert_eq!(
            eight.weight_payload_bytes(),
            eight.config().weight_payload_plan().unwrap().payload_bytes
        );
    }
}
#[test]
fn gemma3_sliding_full_attention() {
    let r = reference();
    let tokens = tokens(&r);
    for (name, stream) in streams() {
        let m = model(&stream);
        for split in 1..tokens.len() {
            let mut cache = m.new_cache(64, LIMIT).unwrap();
            let a = m
                .forward_cached(
                    &tokens[..split],
                    &mut cache,
                    false,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false,
                )
                .unwrap();
            strict(a.values(), &r[name]["splits"][split - 1]["prefix"]);
            let b = m
                .forward_cached(
                    &tokens[split..],
                    &mut cache,
                    false,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false,
                )
                .unwrap();
            strict(b.values(), &r[name]["splits"][split - 1]["suffix"]);
            assert_eq!(cache.len(), tokens.len());
            assert_eq!(cache.retained_tokens(0), Some(4));
            assert_eq!(cache.retained_tokens(1), Some(4));
            assert_eq!(cache.retained_tokens(2), Some(tokens.len()));
        }
        let mut cache = m.new_cache(64, LIMIT).unwrap();
        let mut single = Vec::new();
        for &token in &tokens {
            single.extend(
                m.forward_cached(
                    &[token],
                    &mut cache,
                    true,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false,
                )
                .unwrap()
                .into_values(),
            );
        }
        strict(&single, &r[name]["single"]);
        // Bounded local backing allocation must not retain the entire prefix.
        let initial_bytes = cache.buffer_bytes().unwrap();
        for _ in 0..12 {
            m.forward_cached(&[7], &mut cache, true, LIMIT, LIMIT, &stream, &mut || false)
                .unwrap();
        }
        assert_eq!(cache.retained_tokens(0), Some(4));
        assert!(
            cache.buffer_bytes().unwrap()
                <= initial_bytes + 12 * m.config().kv_width() as u64 * 8 + 8192
        );
    }
}
#[test]
fn gemma3_paged_cache_matches_cold() {
    let tokens: Vec<u32> = (0..321).map(|i| (i * 37 + 11) % 128).collect();
    for (_, stream) in streams() {
        let (mut config, weights) = weights();
        config.context = tokens.len();
        config.window = 150;
        let model = MlxGemma3Model::from_weights(config, weights, LIMIT, &stream).unwrap();
        let cold = model
            .forward(&tokens, LIMIT, LIMIT, LIMIT, &stream)
            .unwrap();
        let mut cache = model.new_cache(tokens.len(), LIMIT).unwrap();
        for chunk in tokens.chunks(63) {
            let logits = model
                .forward_cached(chunk, &mut cache, false, LIMIT, LIMIT, &stream, &mut || {
                    false
                })
                .unwrap();
            let vocab = model.config().vocabulary;
            let end = cache.len();
            let expected = &cold.values()[(end - chunk.len()) * vocab..end * vocab];
            strict(logits.values(), &serde_json::json!(expected));
            assert_eq!(
                cache.retained_tokens(0),
                Some(end.min(model.config().window))
            );
            assert_eq!(cache.retained_tokens(2), Some(end));
        }
    }
}

#[test]
fn gemma3_cancel_and_foreign_cache_are_atomic() {
    for (_, stream) in streams() {
        let m = model(&stream);
        let other = model(&stream);
        let mut cache = m.new_cache(64, LIMIT).unwrap();
        m.forward_cached(
            &[3, 41, 7, 120, 63],
            &mut cache,
            false,
            LIMIT,
            LIMIT,
            &stream,
            &mut || false,
        )
        .unwrap();
        let bytes = cache.buffer_bytes().unwrap();
        for cancel_after in 0..10 {
            let mut checks = 0;
            let result =
                m.forward_cached(&[95], &mut cache, true, LIMIT, LIMIT, &stream, &mut || {
                    checks += 1;
                    checks > cancel_after
                });
            if result.is_err() {
                assert_eq!(cache.len(), 5);
                assert_eq!(cache.buffer_bytes().unwrap(), bytes);
            } else {
                break;
            }
        }
        assert!(
            other
                .forward_cached(&[1], &mut cache, true, LIMIT, LIMIT, &stream, &mut || false)
                .is_err()
        );
        assert!(cache.reset(other.instance_id()).is_err());
        cache.reset(m.instance_id()).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.buffer_bytes().unwrap(), 0);
        assert!(
            m.forward_cached(&[128], &mut cache, true, LIMIT, LIMIT, &stream, &mut || {
                false
            })
            .is_err()
        );
        assert!(
            m.forward_cached(&[1], &mut cache, true, 0, LIMIT, &stream, &mut || false)
                .is_err()
        );
        assert!(
            m.forward_cached(&[1], &mut cache, true, LIMIT, 0, &stream, &mut || false)
                .is_err()
        );
        assert!(cache.is_empty());
        assert!(m.is_healthy());
        assert!(m.new_cache(64, 0).is_err());
    }
}
#[test]
fn gemma3_verified_loading_and_payload_bounds() {
    let root = root();
    let mut hashes = pins(&root);
    let options = LoadOptions {
        max_file_bytes: LIMIT as usize,
        max_weight_bytes: LIMIT as usize,
    };
    hashes.get_mut(&root.join("model.safetensors")).unwrap()[0] ^= 1;
    assert!(
        load_verified(
            &root,
            options,
            &CheckpointVerification {
                sha256: &hashes,
                cancelled: &|| false
            }
        )
        .is_err()
    );
    for (_, stream) in streams() {
        let (c, mut w) = weights();
        assert!(MlxGemma3Model::from_weights(c.clone(), w.clone(), 0, &stream).is_err());
        w.layers[0].q_norm.pop();
        assert!(MlxGemma3Model::from_weights(c, w, LIMIT, &stream).is_err());
    }
}
#[test]
fn gemma3_native_gelu_and_sliding_attention() {
    use mlx_ffi::{
        Array,
        attention::{self, AttentionMask},
        ops,
    };
    for (_, stream) in streams() {
        let x = [-5.0_f32, -1.0, 0.0, 0.25, 2.0, 5.0];
        let a = Array::from_f32_slice(&x, &[6]).unwrap();
        let actual = ops::to_vec_f32(&ops::gelu_approx(&a, &stream).unwrap(), &stream).unwrap();
        for (&a, &v) in actual.iter().zip(&x) {
            let v = f64::from(v);
            let expected = 0.5
                * v
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh());
            assert!((f64::from(a) - expected).abs() < 1e-6);
        }
        // Zero logits make each output the mean of exactly the permitted keys.
        let q = Array::from_f32_slice(&[0.; 8], &[1, 2, 2, 2]).unwrap();
        let k = Array::from_f32_slice(&[0.; 12], &[1, 1, 6, 2]).unwrap();
        let v = Array::from_f32_slice(
            &[0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5.],
            &[1, 1, 6, 2],
        )
        .unwrap();
        for window in [1, 2, 4, 6, 20] {
            let a = attention::grouped_query_attention_scaled(
                &q,
                &k,
                &v,
                0.3,
                AttentionMask::SlidingCausalSuffix {
                    prefix_tokens: 4,
                    window_tokens: window,
                },
                &stream,
            )
            .unwrap();
            let actual = ops::to_vec_f32(&a, &stream).unwrap();
            for h in 0..2 {
                for t in 0..2 {
                    let end: usize = 5 + t;
                    let start = end.saturating_sub(window);
                    let expected = ((start..end).sum::<usize>() as f32) / (end - start) as f32;
                    for d in 0..2 {
                        assert!((actual[h * 4 + t * 2 + d] - expected).abs() < 1e-6);
                    }
                }
            }
        }
        assert!(
            attention::grouped_query_attention(
                &q,
                &k,
                &v,
                AttentionMask::SlidingCausalSuffix {
                    prefix_tokens: 4,
                    window_tokens: 0
                },
                &stream
            )
            .is_err()
        );
    }
}
