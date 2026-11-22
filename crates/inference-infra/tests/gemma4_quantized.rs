//! Strict comparisons with pre-recorded independent trained Gemma 4 logits.
#[path = "support/gemma4.rs"]
mod gemma4;
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires verified ALLOY_GEMMA4_ARTIFACT, ALLOY_GEMMA4_REFERENCE and Metal"]
fn gemma4_native_trained_reference() {
    let profile = gemma4::profile();
    let reference = PathBuf::from(std::env::var_os("ALLOY_GEMMA4_REFERENCE").unwrap());
    assert!(reference.metadata().unwrap().len() < 32 << 20);
    let data = std::fs::read(reference).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        profile["reference"]["sha256"].as_str().unwrap()
    );
    let r: Value = serde_json::from_slice(&data).unwrap();
    assert_eq!(r["profile"], profile["id"]);
    assert_eq!(r["revision"], profile["artifact"]["revision"]);
    assert_eq!(r["versions"], profile["reference"]["versions"]);
    assert_eq!(r["device"], "gpu");
    assert_eq!(r["compile_enabled"], false);
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let load = || gemma4::load(&profile, &stream);
    let mut foreign = load().new_cache(16, 64 << 20).unwrap();
    let model = load();
    let error = model
        .step(&[2], &mut foreign, 512 << 20, &stream, &mut || false)
        .unwrap_err();
    assert!(error.to_string().contains("foreign cache"), "{error}");
    assert!(foreign.is_empty());
    println!("evaluated_weight_buffers={}", model.weight_buffer_bytes());
    let cases = r["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let mut all_pass = true;
    for (i, case) in cases.iter().enumerate() {
        assert_eq!(case["tokens"], profile["prompt_tokens"][i]);
        assert_eq!(case["decode_tokens"], profile["decode_tokens"]);
        let prompt: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let decode: [u32; 2] = serde_json::from_value(case["decode_tokens"].clone()).unwrap();
        let mut cache = model.new_cache(prompt.len() + 2, 64 << 20).unwrap();
        assert!(
            model
                .step(&prompt, &mut cache, 1, &stream, &mut || false)
                .is_err()
        );
        assert!(cache.is_empty());
        let rows = case["logits"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        for (step, (ids, row)) in [prompt.as_slice(), &decode[..1], &decode[1..]]
            .into_iter()
            .zip(rows)
            .enumerate()
        {
            let before = cache.len();
            let mut checks = 0;
            assert!(
                model
                    .step(ids, &mut cache, 512 << 20, &stream, &mut || {
                        checks += 1;
                        checks == 4
                    })
                    .is_err()
            );
            assert_eq!(cache.len(), before);
            let logits = model
                .step(ids, &mut cache, 512 << 20, &stream, &mut || false)
                .unwrap();
            assert_eq!(cache.len(), before + ids.len());
            let exp = row.as_array().unwrap();
            assert_eq!(exp.len(), 262144);
            assert_eq!(logits.values().len(), exp.len());
            let (mut abs, mut rel) = (0.0_f64, 0.0_f64);
            for (a, b) in logits.values().iter().zip(exp) {
                let b = b.as_f64().unwrap();
                let diff = (f64::from(*a) - b).abs();
                assert!(diff.is_finite());
                abs = abs.max(diff);
                rel = rel.max(if b == 0.0 {
                    if diff == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    diff / b.abs()
                });
            }
            println!("gemma4 case={i} step={step} abs={abs:e} relative={rel:e}");
            all_pass &= abs < 0.02 && rel < 0.01;
        }
        assert!(
            model
                .step(&[1], &mut cache, 512 << 20, &stream, &mut || false)
                .is_err()
        );
        assert_eq!(cache.len(), prompt.len() + 2);
        assert!(cache.buffer_bytes().unwrap() > 0);
    }
    assert!(all_pass, "Gemma 4 FR-8 trained reference mismatch");
}
