//! Full trained Qwen3 numerical and memory qualification on the native decoder.
#[path = "../../inference-infra/tests/support/dense_memory.rs"]
mod dense_memory;
#[path = "../../inference-infra/tests/support/numerical_capture.rs"]
mod numerical_capture;
#[path = "../../inference-infra/tests/support/pinned_checkpoint.rs"]
mod pinned_checkpoint;

use engine::backends::qwen::parse_config;
use inference_infra::decoder::{loader::load_weights_verified, mlx_model::MlxDecoderModel};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;

fn profile() -> Value {
    let id = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let manifest = match id.as_str() {
        "qwen3-4b-4bit" => include_str!("../../families/qwen3/fixture/affine4_4b_profile.json"),
        "qwen3-14b-4bit" => include_str!("../../families/qwen3/fixture/affine4_14b_profile.json"),
        other => panic!("unsupported dense Qwen3 profile {other}"),
    };
    let profile: Value = serde_json::from_str(manifest).unwrap();
    assert_eq!(profile["id"], id);
    profile
}

#[test]
#[ignore = "requires a pinned trained Qwen3 model and its eager Metal reference"]
fn qwen3_dense_trained_reference() {
    let profile = profile();
    let bytes = fs::read(std::env::var_os("ALLOY_QWEN3_REFERENCE").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        profile["reference"]["sha256"]
    );
    let reference: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(reference["profile"], profile["id"]);
    assert_eq!(
        reference["artifact_revision"],
        profile["artifact"]["revision"]
    );
    assert_eq!(reference["versions"], profile["reference"]["versions"]);
    assert_eq!(reference["compile_enabled"], false);
    assert_eq!(reference["device"], "gpu");
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let (config, weights) = pinned_checkpoint::load(
        &profile,
        "ALLOY_QWEN3_ARTIFACT",
        16 << 30,
        |root, limits, pins| {
            let value =
                serde_json::from_slice(&fs::read(root.join("config.json")).unwrap()).unwrap();
            let config = parse_config(&value).unwrap();
            load_weights_verified(&config, root, limits, pins).map(|weights| (config, weights))
        },
    );
    assert_eq!(weights.embed_tokens.bits(), Some(4));
    assert_eq!(config.vocab_size, 151936);
    let model = MlxDecoderModel::from_weights(config, weights, 16 << 30, &stream).unwrap();
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(case["tokens"], profile["prompt_tokens"][index]);
        assert_eq!(case["decode_tokens"], profile["decode_tokens"]);
        let prompt: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let decode: Vec<u32> = serde_json::from_value(case["decode_tokens"].clone()).unwrap();
        let mut cache = model
            .new_cache(prompt.len() + decode.len(), 128 << 20, &stream)
            .unwrap();
        let expected = case["logits"].as_array().unwrap();
        assert_eq!(expected.len(), 1 + decode.len());
        for (step, want) in expected.iter().enumerate() {
            let row = if step == 0 {
                model.prefill(&prompt, &mut cache, 256 << 20, &stream)
            } else {
                model.decode(decode[step - 1], &mut cache, 256 << 20, &stream)
            }
            .unwrap();
            assert_eq!(row.rows(), 1);
            let actual = row.last_row();
            let expected = want.as_array().unwrap();
            assert_eq!(actual.len(), 151936);
            assert_eq!(expected.len(), actual.len());
            assert_eq!(cache.len(), prompt.len() + step);
            numerical_capture::record(&profile, index, step, &prompt, &decode[..step], actual);
            let mut absolute = 0f64;
            let mut relative = 0f64;
            for (&value, want) in actual.iter().zip(expected) {
                let want = want.as_f64().unwrap();
                assert!(value.is_finite() && want.is_finite());
                let delta = (f64::from(value) - want).abs();
                absolute = absolute.max(delta);
                if delta > 0.0 {
                    relative = relative.max(delta / want.abs());
                }
            }
            println!("case={index} step={step} max_absolute={absolute} max_relative={relative}");
            assert!(absolute < 0.02 && relative < 0.01);
        }
    }
}

#[test]
#[ignore = "requires pinned dense Qwen3 weights, Metal and a physical 36 GiB Mac"]
fn qwen3_dense_native_memory_tiers() {
    dense_memory::run(&profile(), parse_config);
}
