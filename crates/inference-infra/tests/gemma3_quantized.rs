//! Explicit real-artifact numerical gate, not an agent/profile qualification.
#[path = "support/cache_memory.rs"]
mod cache_memory;
#[path = "support/numerical_capture.rs"]
mod numerical_capture;
#[path = "support/pinned_checkpoint.rs"]
mod pinned_checkpoint;

use inference_infra::decoder::gemma3::{model::MlxGemma3Model, weights::load_verified};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires ALLOY_GEMMA3_ARTIFACT and ALLOY_GEMMA3_REFERENCE with pinned offline capture"]
fn gemma3_12b_real_quantized_native_reference() {
    real_model(
        include_str!("fixtures/gemma3_12b_profile.json"),
        "ALLOY_GEMMA3_ARTIFACT",
        "ALLOY_GEMMA3_REFERENCE",
        (3840, 48, 16, 8),
    );
}

#[test]
#[ignore = "requires ALLOY_GEMMA3_SMALL_ARTIFACT and ALLOY_GEMMA3_SMALL_REFERENCE with pinned offline capture"]
fn gemma3_4b_real_quantized_native_reference() {
    real_model(
        include_str!("fixtures/gemma3_4b_profile.json"),
        "ALLOY_GEMMA3_SMALL_ARTIFACT",
        "ALLOY_GEMMA3_SMALL_REFERENCE",
        (2560, 34, 8, 4),
    );
}

fn real_model(
    profile_json: &str,
    artifact_env: &str,
    reference_env: &str,
    geometry: (usize, usize, usize, usize),
) {
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
    let (c, w) = pinned_checkpoint::load(&profile, artifact_env, 12 << 30, load_verified);
    assert_eq!((c.hidden, c.layers, c.query_heads, c.kv_heads), geometry);
    assert_eq!(c.vocabulary, 262208);
    assert_eq!(c.head_dim, 256);
    assert_eq!((c.quantization_bits, c.group_size), (4, Some(64)));
    assert_eq!(w.embedding.bits(), Some(4));
    assert!(!c.tied_embeddings && w.head.is_some());
    assert!(c.wrapped);
    assert_eq!(c.sliding.iter().filter(|&&v| !v).count(), geometry.1 / 6);
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
        let decode_tokens = [10, 12];
        for token in decode_tokens {
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
            numerical_capture::record(
                &profile,
                case_index,
                row,
                &tokens,
                &decode_tokens[..row],
                actual,
            );
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

#[test]
#[ignore = "requires pinned profile weights, Metal and a physical 36 GiB Mac"]
fn gemma3_native_memory_tiers() {
    let name = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let (json, env) = match name.as_str() {
        "gemma-3-4b-4bit" => (
            include_str!("fixtures/gemma3_4b_profile.json"),
            "ALLOY_GEMMA3_SMALL_ARTIFACT",
        ),
        "gemma-3-12b-4bit" => (
            include_str!("fixtures/gemma3_12b_profile.json"),
            "ALLOY_GEMMA3_ARTIFACT",
        ),
        other => panic!("unknown Gemma 3 profile {other}"),
    };
    let profile: Value = serde_json::from_str(json).unwrap();
    assert_eq!(profile["id"], name);
    cache_memory::run(&profile, |tier| {
        let (config, weights) = pinned_checkpoint::load(&profile, env, 12 << 30, load_verified);
        let model = MlxGemma3Model::from_weights(config, weights, tier.budget as u64, &tier.stream)
            .unwrap();
        let load_peak = tier.load_peak();
        let weights = model.weight_buffer_bytes().unwrap();
        assert!(weights >= model.weight_payload_bytes());
        let allowance = (tier.budget as u64).checked_sub(weights).unwrap();
        let cache = model.new_cache(tier.tokens.len() + 1, allowance).unwrap();
        let (initial_kv, kv) = tier.measure(
            load_peak,
            weights,
            model.config().vocabulary,
            cache,
            |tokens, cache| {
                let plan = model
                    .config()
                    .step_payload_plan(tokens.len(), cache.len() + tokens.len())
                    .unwrap();
                model
                    .forward_cached(
                        tokens,
                        cache,
                        true,
                        plan.output_bytes,
                        plan.work_bytes,
                        &tier.stream,
                        &mut || false,
                    )
                    .unwrap()
            },
            |cache| (cache.len(), cache.buffer_bytes().unwrap()),
        );
        assert_eq!(initial_kv, 0);
        assert!(kv >= model.config().cache_payload(tier.tokens.len() + 1).unwrap());
    });
}
