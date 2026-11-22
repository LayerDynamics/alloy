//! Pinned trained-weight test of native prefill and cached decode on Metal.
#[path = "support/ministral3.rs"]
mod ministral3;

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires pinned ALLOY_MINISTRAL3_ARTIFACT and ALLOY_MINISTRAL3_REFERENCE"]
fn ministral3_real_quantized_native_reference() {
    check_reference(
        include_str!("fixtures/ministral3_profile.json"),
        2_143_678_464,
    );
}

#[test]
#[ignore = "requires pinned 8-bit ALLOY_MINISTRAL3_ARTIFACT and ALLOY_MINISTRAL3_REFERENCE"]
fn ministral3_real_affine8_native_reference() {
    check_reference(
        include_str!("fixtures/ministral3_8bit_profile.json"),
        3_858_100_224,
    );
}

fn check_reference(manifest: &str, resident_bytes: usize) {
    let profile: Value = serde_json::from_str(manifest).unwrap();
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
    let (model, stream) = ministral3::load_pinned_model(&profile, resident_bytes);
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
