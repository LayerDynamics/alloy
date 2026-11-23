use inference_infra::decoder::{
    config::{DecoderConfig, DecoderFamily},
    fixture::generate_fixture,
    mlx_model::MlxDecoderModel,
};
use mlx_ffi::{Device, DeviceType, Stream};

fn config() -> DecoderConfig {
    DecoderConfig {
        family: DecoderFamily::SmolLm3,
        num_layers: 4,
        rope_layers: Some(vec![true, false, true, false]),
        rope_theta: 5000000.0,
        ..DecoderConfig::tiny_fixture()
    }
}
const TOKENS: &[u32] = &[3, 41, 7, 200, 128, 95, 12, 250];
#[test]
fn smollm3_reference_logits() {
    let c = config();
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    let model =
        MlxDecoderModel::from_weights(c.clone(), generate_fixture(&c), 8 * 1024 * 1024, &stream)
            .unwrap();
    let actual = model
        .forward(TOKENS, 1024 * 1024, 8 * 256 * 16, &stream)
        .unwrap();
    let references: serde_json::Value =
        serde_json::from_str(include_str!("../fixture/reference_logits.json")).unwrap();
    let independent: Vec<Vec<f64>> =
        serde_json::from_value(references["numpy_float64"].clone()).unwrap();
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    let references: serde_json::Value = {
        use sha2::{Digest, Sha256};
        let bytes = include_bytes!("../fixture/reference_logits-linux-aarch64.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "7c6b593fff803b9bbf898577a1c3507a08c7180d72a5c79b01f5dae6bd7c278a"
        );
        serde_json::from_slice(bytes).unwrap()
    };
    let expected: Vec<Vec<f64>> =
        serde_json::from_value(references["mlx_eager_float32"].clone()).unwrap();
    for rows in [&expected, &independent] {
        assert_eq!(rows.len(), TOKENS.len());
        assert!(rows.iter().all(|row| row.len() == actual.vocabulary_size()));
    }
    // The independent float64 oracle checks the architecture; FR8 compares
    // native float32 against the pinned upstream eager float32 implementation.
    for (&a, b) in actual
        .values()
        .iter()
        .zip(independent.into_iter().flatten())
    {
        assert!((f64::from(a) - b).abs() < 0.00002);
    }
    assert_eq!(
        actual.values().len(),
        expected.iter().map(Vec::len).sum::<usize>()
    );
    let mut absolute = 0f64;
    let mut relative = 0f64;
    for (&a, b) in actual.values().iter().zip(expected.into_iter().flatten()) {
        let error = (f64::from(a) - b).abs();
        assert!(error.is_finite());
        absolute = absolute.max(error);
        relative = relative.max(if error == 0.0 { 0.0 } else { error / b.abs() });
        if error > 0.0 && error / b.abs() >= 0.01 {
            eprintln!("relative mismatch native={a:.12e} eager={b:.12e} abs={error:.12e}");
        }
    }
    eprintln!("SmolLM3 tiny max_abs={absolute} max_relative={relative}");
    assert!(absolute < 0.02 && relative < 0.01);
}
#[test]
fn smollm3_cached_matches_full_across_nope_layers() {
    let c = config();
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    let model =
        MlxDecoderModel::from_weights(c.clone(), generate_fixture(&c), 8 * 1024 * 1024, &stream)
            .unwrap();
    let full = model.forward(TOKENS, 1024 * 1024, 32768, &stream).unwrap();
    for split in 1..TOKENS.len() {
        let mut cache = model.new_cache(32, 1024 * 1024, &stream).unwrap();
        model
            .prefill(&TOKENS[..split], &mut cache, 32768, &stream)
            .unwrap();
        for (i, &token) in TOKENS[split..].iter().enumerate() {
            let actual = model.decode(token, &mut cache, 32768, &stream).unwrap();
            for (&a, &b) in actual.values().iter().zip(full.row(split + i).unwrap()) {
                assert!((a - b).abs() < 0.002, "split {split}, row {i}: {a} vs {b}");
            }
        }
        assert_eq!(cache.len(), TOKENS.len());
    }
    let mut invalid = c.clone();
    invalid.rope_layers = Some(vec![true]);
    assert!(
        MlxDecoderModel::from_weights(invalid, generate_fixture(&c), 8 * 1024 * 1024, &stream)
            .is_err()
    );
}
