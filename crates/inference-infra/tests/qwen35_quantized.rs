//! Explicit real-artifact numerical gate, not an agent/profile qualification.
use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    qwen35::{
        model::{ForwardOptions, MlxQwen35Model},
        weights::{AttentionWeights, load_verified},
    },
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

#[test]
#[ignore = "requires ALLOY_ORNITH_ARTIFACT and ALLOY_ORNITH_REFERENCE with pinned offline capture"]
fn ornith_9b_quantized_native_reference() {
    check_quantized_reference(
        "ALLOY_ORNITH_ARTIFACT",
        "ALLOY_ORNITH_REFERENCE",
        include_str!("fixtures/ornith_1_5_9b_profile.json"),
        (4096, 32, 32, false),
    );
}

#[test]
#[ignore = "requires ALLOY_QWEN35_SMALL_ARTIFACT and ALLOY_QWEN35_SMALL_REFERENCE with pinned offline capture"]
fn qwen35_0_8b_quantized_native_reference() {
    check_quantized_reference(
        "ALLOY_QWEN35_SMALL_ARTIFACT",
        "ALLOY_QWEN35_SMALL_REFERENCE",
        include_str!("fixtures/qwen35_0_8b_profile.json"),
        (1024, 24, 16, true),
    );
}

fn check_quantized_reference(
    artifact_env: &str,
    reference_env: &str,
    profile_json: &str,
    (hidden, layers, value_heads, tied): (usize, usize, usize, bool),
) {
    let root = PathBuf::from(std::env::var_os(artifact_env).expect("artifact"))
        .canonicalize()
        .unwrap();
    let reference_path = PathBuf::from(std::env::var_os(reference_env).expect("reference"));
    assert!(std::fs::metadata(&reference_path).unwrap().len() < 64 * 1024 * 1024);
    let bytes = std::fs::read(reference_path).unwrap();
    let profile: Value = serde_json::from_str(profile_json).unwrap();
    let label = profile["id"].as_str().unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        profile["reference"]["sha256"]
            .as_str()
            .expect("reference must be frozen before native execution")
    );
    let reference: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(reference["profile"], profile["id"]);
    assert_eq!(
        reference["artifact_revision"],
        profile["artifact"]["revision"]
    );
    assert_eq!(reference["compile_enabled"], false);
    assert_eq!(reference["device"], "gpu");
    assert_eq!(reference["versions"], profile["reference"]["versions"]);
    let pins: BTreeMap<_, _> = profile["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| {
            let hex = file["sha256"].as_str().unwrap();
            let mut hash = [0; 32];
            for (i, b) in hash.iter_mut().enumerate() {
                *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
            }
            (root.join(file["path"].as_str().unwrap()), hash)
        })
        .collect();
    let (c, w) = load_verified(
        &root,
        LoadOptions {
            max_file_bytes: 8 * 1024 * 1024 * 1024,
            max_weight_bytes: 8 * 1024 * 1024 * 1024,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap();
    assert_eq!(
        (c.hidden, c.vocabulary, c.linear_layers.len()),
        (hidden, 248320, layers)
    );
    assert_eq!(
        (
            c.head_dim,
            c.rotary_dim,
            c.linear.key_heads,
            c.linear.value_heads
        ),
        (256, 64, 16, value_heads)
    );
    assert_eq!((c.quantization_bits, c.group_size), (4, Some(64)));
    assert_eq!(w.embedding.bits(), Some(4));
    assert_eq!(c.tied_embeddings, tied);
    assert_eq!(w.head.is_none(), tied);
    if let Some(head) = &w.head {
        assert_eq!(head.bits(), Some(4));
    }
    for (i, layer) in w.layers.iter().enumerate() {
        for matrix in [&layer.gate, &layer.up, &layer.down] {
            assert_eq!(matrix.bits(), Some(4));
        }
        match &layer.attention {
            AttentionWeights::Linear(a) => {
                assert!((i + 1) % 4 != 0);
                for matrix in [&a.qkv, &a.z, &a.a, &a.b, &a.output] {
                    assert_eq!(matrix.bits(), Some(4));
                }
            }
            AttentionWeights::Full(a) => {
                assert_eq!((i + 1) % 4, 0);
                for matrix in [&a.q, &a.k, &a.v, &a.output] {
                    assert_eq!(matrix.bits(), Some(4));
                }
            }
        }
    }
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let model = MlxQwen35Model::from_weights(c, w, 8 * 1024 * 1024 * 1024, &stream).unwrap();
    eprintln!(
        "{label} resident payload={} measured_buffers={}",
        model.weight_payload_bytes(),
        model.weight_buffer_bytes().unwrap()
    );
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let options = ForwardOptions {
        // The frozen eager reference projects every prompt position, then
        // selects the final row. Match that matrix geometry for strict FR8.
        all_logits: true,
        output_limit: 16 * 1024 * 1024,
        work_limit: 1024 * 1024 * 1024,
    };
    let mut maximum_absolute = 0f64;
    let mut maximum_relative = 0f64;
    for (case_index, case) in cases.iter().enumerate() {
        assert_eq!(case["tokens"], profile["prompt_tokens"][case_index]);
        assert_eq!(case["decode_tokens"], serde_json::json!([19, 13]));
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let mut cache = model
            .new_cache(tokens.len() + 2, 1024 * 1024 * 1024, &stream)
            .unwrap();
        let mut actual = vec![
            model
                .forward(&tokens, &mut cache, options, &stream, &|| false)
                .unwrap()
                .into_values(),
        ];
        for token in [19, 13] {
            actual.push(
                model
                    .forward(&[token], &mut cache, options, &stream, &|| false)
                    .unwrap()
                    .into_values(),
            );
        }
        assert_eq!(cache.len(), tokens.len() + 2);
        let mut last_cache = model
            .new_cache(tokens.len() + 2, 1024 * 1024 * 1024, &stream)
            .unwrap();
        for (row, chunk) in [tokens.as_slice(), &[19], &[13]].into_iter().enumerate() {
            let last = model
                .forward(
                    chunk,
                    &mut last_cache,
                    ForwardOptions {
                        all_logits: false,
                        ..options
                    },
                    &stream,
                    &|| false,
                )
                .unwrap()
                .into_values();
            let full = &actual[row][actual[row].len() - 248320..];
            assert_eq!(last.len(), full.len());
            for (&a, &b) in last.iter().zip(full) {
                assert!(a.is_finite() && (a - b).abs() <= 0.02 + 0.01 * b.abs());
            }
            // Selecting logits must not change recurrent or KV continuation.
            if row > 0 {
                assert_eq!(last, full);
            }
        }
        assert_eq!(last_cache.len(), cache.len());
        let expected = case["logits"].as_array().unwrap();
        assert_eq!(expected.len(), 3);
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual = &actual[actual.len() - 248320..];
            let expected: Vec<f64> = serde_json::from_value(expected.clone()).unwrap();
            assert_eq!(actual.len(), 248320);
            assert_eq!(actual.len(), expected.len());
            let mut abs = 0f64;
            let mut rel = 0f64;
            let mut worst = (0, 0f64, 0f64);
            for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(a.is_finite() && e.is_finite());
                let difference = (f64::from(a) - e).abs();
                abs = abs.max(difference);
                let ratio = if difference == 0.0 {
                    0.0
                } else {
                    difference / e.abs()
                };
                if ratio > rel {
                    rel = ratio;
                    worst = (index, f64::from(a), e);
                }
            }
            eprintln!(
                "{label} case={case_index} row={row} max_abs={abs} max_pointwise_relative={rel} worst={worst:?}"
            );
            maximum_absolute = maximum_absolute.max(abs);
            maximum_relative = maximum_relative.max(rel);
        }
    }
    assert!(
        maximum_absolute < 0.02 && maximum_relative < 0.01,
        "strict {label} FR8 failed: max_abs={maximum_absolute}, max_relative={maximum_relative}"
    );
}
