//! Explicit real-artifact numerical gate, not an agent/profile qualification.
use inference_infra::decoder::{
    gemma3::{model::MlxGemma3Model, weights::load_verified},
    loader::{CheckpointVerification, LoadOptions},
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

#[test]
#[ignore = "requires ALLOY_GEMMA3_ARTIFACT and ALLOY_GEMMA3_REFERENCE with pinned offline capture"]
fn gemma3_12b_real_quantized_native_reference() {
    let artifact_env = "ALLOY_GEMMA3_ARTIFACT";
    let reference_env = "ALLOY_GEMMA3_REFERENCE";
    let profile_json = include_str!("fixtures/gemma3_12b_profile.json");
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
            max_file_bytes: 12 * 1024 * 1024 * 1024,
            max_weight_bytes: 12 * 1024 * 1024 * 1024,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap();
    assert_eq!((c.hidden, c.vocabulary, c.layers), (3840, 262208, 48));
    assert_eq!((c.head_dim, c.query_heads, c.kv_heads), (256, 16, 8));
    assert_eq!((c.quantization_bits, c.group_size), (4, Some(64)));
    assert_eq!(w.embedding.bits(), Some(4));
    assert!(!c.tied_embeddings && w.head.is_some());
    assert!(c.wrapped);
    assert_eq!(c.sliding.iter().filter(|&&v| !v).count(), 8);
    assert_eq!(profile["position_regime"], "gemma3-local-global-linear-v1");
    assert_eq!(reference["position_regime"], profile["position_regime"]);
    for layer in &w.layers {
        for matrix in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.output,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            assert_eq!(matrix.bits(), Some(4));
        }
    }
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let weight_plan = c.weight_payload_plan().unwrap();
    let model = MlxGemma3Model::from_weights(c, w, 12 * 1024 * 1024 * 1024, &stream).unwrap();
    assert_eq!(weight_plan.payload_bytes, model.weight_payload_bytes());
    assert!(weight_plan.largest_upload_bytes < weight_plan.payload_bytes);
    eprintln!(
        "{label} resident payload={} measured_buffers={}",
        model.weight_payload_bytes(),
        model.weight_buffer_bytes().unwrap()
    );
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let output_limit = 32 * 1024 * 1024;
    let work_limit = 1024 * 1024 * 1024;
    let mut maximum_absolute = 0f64;
    let mut maximum_relative = 0f64;
    for (case_index, case) in cases.iter().enumerate() {
        assert_eq!(case["tokens"], profile["prompt_tokens"][case_index]);
        assert_eq!(case["decode_tokens"], profile["decode_tokens"]);
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let mut cache = model
            .new_cache(tokens.len() + 2, 1024 * 1024 * 1024)
            .unwrap();
        let mut actual = vec![
            model
                .forward_cached(
                    &tokens,
                    &mut cache,
                    false,
                    output_limit,
                    work_limit,
                    &stream,
                    &mut || false,
                )
                .unwrap()
                .into_values(),
        ];
        for token in [10, 12] {
            actual.push(
                model
                    .forward_cached(
                        &[token],
                        &mut cache,
                        false,
                        output_limit,
                        work_limit,
                        &stream,
                        &mut || false,
                    )
                    .unwrap()
                    .into_values(),
            );
        }
        assert_eq!(cache.len(), tokens.len() + 2);
        let mut last_cache = model
            .new_cache(tokens.len() + 2, 1024 * 1024 * 1024)
            .unwrap();
        for (row, chunk) in [tokens.as_slice(), &[10], &[12]].into_iter().enumerate() {
            let last = model
                .forward_cached(
                    chunk,
                    &mut last_cache,
                    true,
                    output_limit,
                    work_limit,
                    &stream,
                    &mut || false,
                )
                .unwrap()
                .into_values();
            let full = &actual[row][actual[row].len() - 262208..];
            assert_eq!(last.len(), full.len());
            // Last-only uses the identical full projection before slicing.
            // Every row, including prefill, must be bit-identical.
            assert_eq!(last, full);
        }
        assert_eq!(last_cache.len(), cache.len());
        let expected = case["logits"].as_array().unwrap();
        assert_eq!(expected.len(), 3);
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual = &actual[actual.len() - 262208..];
            let expected: Vec<f64> = serde_json::from_value(expected.clone()).unwrap();
            assert_eq!(actual.len(), 262208);
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
