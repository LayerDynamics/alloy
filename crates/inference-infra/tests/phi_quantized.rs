//! Explicit real-artifact numerical gate, not an agent/profile qualification.
#[path = "support/cache_memory.rs"]
mod cache_memory;
#[path = "support/numerical_capture.rs"]
mod numerical_capture;
#[path = "support/pinned_checkpoint.rs"]
mod pinned_checkpoint;

use inference_infra::decoder::phi::{PhiRopeRegime, model::MlxPhiModel, weights::load_verified};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires ALLOY_PHI_ARTIFACT and ALLOY_PHI_REFERENCE with pinned offline capture"]
fn phi4mini_real_quantized_native_reference() {
    let artifact_env = "ALLOY_PHI_ARTIFACT";
    let reference_env = "ALLOY_PHI_REFERENCE";
    let profile_json = include_str!("fixtures/phi4_mini_profile.json");
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
    let (c, w) = pinned_checkpoint::load(&profile, artifact_env, 8 << 30, load_verified);
    assert_eq!((c.hidden, c.vocabulary, c.layers), (3072, 200064, 32));
    assert_eq!(
        (c.head_dim, c.rotary_dim, c.query_heads, c.kv_heads),
        (128, 96, 24, 8)
    );
    assert_eq!((c.quantization_bits, c.group_size), (4, Some(64)));
    assert_eq!(w.embedding.bits(), Some(4));
    assert!(c.tied_embeddings && w.head.is_none());
    assert_eq!(profile["position_regime"], "mlx-su-long-fixed-v1");
    assert_eq!(reference["position_regime"], profile["position_regime"]);
    for layer in &w.layers {
        for matrix in [&layer.qkv, &layer.output, &layer.gate_up, &layer.down] {
            assert_eq!(matrix.bits(), Some(4));
        }
    }
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let weight_plan = c.weight_payload_plan().unwrap();
    let model =
        MlxPhiModel::from_weights(c, w, PhiRopeRegime::Long, 4 * 1024 * 1024 * 1024, &stream)
            .unwrap();
    assert_eq!(weight_plan.payload_bytes, 2_398_187_520);
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
            .new_cache(tokens.len() + 2, 1024 * 1024 * 1024, &stream)
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
        let decode_tokens = [19, 13];
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
            .new_cache(tokens.len() + 2, 1024 * 1024 * 1024, &stream)
            .unwrap();
        for (row, chunk) in [tokens.as_slice(), &[19], &[13]].into_iter().enumerate() {
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
            let full = &actual[row][actual[row].len() - 200064..];
            assert_eq!(last.len(), full.len());
            // Last-only uses the identical full projection before slicing.
            // Every row, including prefill, must be bit-identical.
            assert_eq!(last, full);
        }
        assert_eq!(last_cache.len(), cache.len());
        let expected = case["logits"].as_array().unwrap();
        assert_eq!(expected.len(), 3);
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let actual = &actual[actual.len() - 200064..];
            numerical_capture::record(
                &profile,
                case_index,
                row,
                &tokens,
                &decode_tokens[..row],
                actual,
            );
            let expected: Vec<f64> = serde_json::from_value(expected.clone()).unwrap();
            assert_eq!(actual.len(), 200064);
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
fn phi_native_memory_tiers() {
    let name = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let json = include_str!("fixtures/phi4_mini_profile.json");
    let env = "ALLOY_PHI_ARTIFACT";
    let profile: Value = serde_json::from_str(json).unwrap();
    assert_eq!(profile["id"], name);
    cache_memory::run(&profile, |tier| {
        let (config, weights) = pinned_checkpoint::load(&profile, env, 8 << 30, load_verified);
        let model = MlxPhiModel::from_weights(
            config,
            weights,
            PhiRopeRegime::Long,
            tier.budget as u64,
            &tier.stream,
        )
        .unwrap();
        let load_peak = tier.load_peak();
        let weights = model.weight_buffer_bytes().unwrap();
        assert!(weights >= model.weight_payload_bytes());
        let allowance = (tier.budget as u64).checked_sub(weights).unwrap();
        let cache = model
            .new_cache(tier.tokens.len() + 1, allowance, &tier.stream)
            .unwrap();
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
        assert_eq!(initial_kv, kv);
    });
}
