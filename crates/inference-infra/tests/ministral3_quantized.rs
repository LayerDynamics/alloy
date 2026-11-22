//! Pinned trained-weight test of native prefill and cached decode on Metal.
use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    ministral3::weights::load_verified,
    mlx_model::MlxDecoderModel,
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};

#[test]
#[ignore = "requires pinned ALLOY_MINISTRAL3_ARTIFACT and ALLOY_MINISTRAL3_REFERENCE"]
fn ministral3_real_quantized_native_reference() {
    let profile: Value =
        serde_json::from_str(include_str!("fixtures/ministral3_profile.json")).unwrap();
    let root = PathBuf::from(std::env::var_os("ALLOY_MINISTRAL3_ARTIFACT").unwrap())
        .canonicalize()
        .unwrap();
    let path = PathBuf::from(std::env::var_os("ALLOY_MINISTRAL3_REFERENCE").unwrap());
    assert!(std::fs::metadata(&path).unwrap().len() < 32 << 20);
    let reference = std::fs::read(path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&reference)),
        profile["reference"]["sha256"].as_str().unwrap()
    );
    let reference: Value = serde_json::from_slice(&reference).unwrap();
    assert_eq!(reference["profile"], profile["id"]);
    assert_eq!(
        reference["artifact_revision"],
        profile["artifact"]["revision"]
    );
    assert_eq!(reference["versions"], profile["reference"]["versions"]);
    assert_eq!(reference["position_regime"], profile["position_regime"]);
    assert_eq!(reference["compile_enabled"], false);
    assert_eq!(reference["device"], "gpu");
    let pins: BTreeMap<_, _> = profile["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| {
            let hex = file["sha256"].as_str().unwrap();
            (
                root.join(file["path"].as_str().unwrap()),
                std::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()),
            )
        })
        .collect();
    let (config, weights) = load_verified(
        &root,
        LoadOptions {
            max_file_bytes: 3 << 30,
            max_weight_bytes: 2_143_678_464,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap();
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let model = MlxDecoderModel::from_weights(
        config.decoder_config().unwrap(),
        weights,
        2_143_678_464,
        &stream,
    )
    .unwrap();
    assert_eq!(model.weight_buffer_bytes().unwrap(), 2_143_678_464);
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for (case_id, case) in cases.iter().enumerate() {
        assert_eq!(case["tokens"], profile["prompt_tokens"][case_id]);
        assert_eq!(case["decode_tokens"], profile["decode_tokens"]);
        let prompt: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let decode: [u32; 2] = serde_json::from_value(case["decode_tokens"].clone()).unwrap();
        let mut cache = model.new_cache(prompt.len() + 2, 1 << 27, &stream).unwrap();
        let rows = case["logits"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        for (step, expected) in rows.iter().enumerate() {
            let logits = if step == 0 {
                model.prefill(&prompt, &mut cache, 1 << 27, &stream)
            } else {
                model.decode(decode[step - 1], &mut cache, 1 << 27, &stream)
            }
            .unwrap();
            let expected = expected.as_array().unwrap();
            assert_eq!(expected.len(), 131072);
            assert_eq!(logits.values().len(), expected.len());
            let (mut abs, mut rel) = (0.0_f64, 0.0_f64);
            for (&a, b) in logits.values().iter().zip(expected) {
                let b = b.as_f64().unwrap();
                assert!(a.is_finite() && b.is_finite());
                let diff = (f64::from(a) - b).abs();
                abs = abs.max(diff);
                if b == 0.0 {
                    assert_eq!(diff, 0.0);
                } else {
                    rel = rel.max(diff / b.abs());
                }
            }
            eprintln!("case {case_id} step {step}, maximum absolute {abs}, maximum relative {rel}");
            assert!(abs < 0.02 && rel < 0.01);
        }
        assert_eq!(cache.len(), prompt.len() + 2);
    }
}
