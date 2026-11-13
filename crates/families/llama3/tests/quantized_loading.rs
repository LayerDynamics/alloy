use llama3::{
    config::parse_config,
    loader::{CheckpointVerification, LoadOptions, load_weights_verified},
    mlx_model::MlxDecoderModel,
};
use mlx_ffi::{Device, DeviceType, Stream};
#[test]
#[ignore = "requires ALLOY_LLAMA_ARTIFACT and ALLOY_LLAMA_REFERENCE with verified offline capture"]
fn llama3_4bit_8bit_load() {
    let root =
        std::path::PathBuf::from(std::env::var_os("ALLOY_LLAMA_ARTIFACT").expect("artifact"));
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("ALLOY_LLAMA_REFERENCE").expect("reference")).unwrap(),
    )
    .unwrap();
    let config = parse_config(
        &serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap(),
    )
    .unwrap();
    let corpus: serde_json::Value =
        serde_json::from_str(include_str!("../../../../bench/corpus/models.json")).unwrap();
    let profile = corpus["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == reference["profile"])
        .expect("frozen profile");
    assert!(matches!(
        profile["id"].as_str().unwrap(),
        "llama-3.1-8b-4bit" | "llama-3.1-8b-8bit"
    ));
    assert_eq!(reference["artifact_sha256"], profile["artifact"]["sha256"]);
    assert_eq!(reference["compile_enabled"], false);
    for (package, version) in [("mlx", "0.32.4"), ("mlx-lm", "0.31.2"), ("numpy", "2.4.2")] {
        assert_eq!(reference["versions"][package], version);
    }
    assert_eq!(reference["cases"].as_array().unwrap().len(), 2);
    let checksums = profile["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let hex = f["sha256"].as_str().unwrap();
            let mut digest = [0; 32];
            for (i, byte) in digest.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
            }
            (root.join(f["path"].as_str().unwrap()), digest)
        })
        .collect();
    let verification = CheckpointVerification {
        sha256: &checksums,
        cancelled: &|| false,
    };
    let weights = load_weights_verified(
        &config,
        &root,
        LoadOptions {
            max_file_bytes: 16 * 1024 * 1024 * 1024,
            max_weight_bytes: 16 * 1024 * 1024 * 1024,
        },
        &verification,
    )
    .unwrap();
    assert_eq!(weights.embed_tokens.bits(), Some(config.quantization_bits));
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let model =
        MlxDecoderModel::from_weights(config.clone(), weights, 16 * 1024 * 1024 * 1024, &stream)
            .unwrap();
    let mut absolute = 0f64;
    let mut relative = 0f64;
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    const PROMPTS: [&[u32]; 2] = [
        &[
            128000, 128006, 882, 128007, 271, 3923, 374, 220, 17, 489, 220, 17, 30, 128009, 128006,
            78191, 128007, 271,
        ],
        &[
            128000, 128006, 9125, 128007, 271, 16533, 3613, 285, 989, 13, 128009, 128006, 882,
            128007, 271, 849, 21435, 3249, 279, 13180, 374, 6437, 13, 128009, 128006, 78191,
            128007, 271,
        ],
    ];
    for (case_index, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let decode: Vec<u32> = serde_json::from_value(case["decode_tokens"].clone()).unwrap();
        assert_eq!(tokens, PROMPTS[case_index]);
        assert_eq!(decode, [19, 13]);
        assert_eq!(case["logits"].as_array().unwrap().len(), 3);
        let mut cache = model
            .new_cache(tokens.len() + decode.len(), 1024 * 1024 * 1024, &stream)
            .unwrap();
        let first = model
            .prefill(
                &tokens,
                &mut cache,
                config.vocab_size as u64 * 16 * tokens.len() as u64,
                &stream,
            )
            .unwrap();
        let mut rows = vec![first.into_values()];
        for token in decode {
            rows.push(
                model
                    .decode(
                        token,
                        &mut cache,
                        config.vocab_size as u64 * 16 * tokens.len() as u64,
                        &stream,
                    )
                    .unwrap()
                    .into_values(),
            );
        }
        for (row, (actual, expected)) in rows
            .iter()
            .zip(case["logits"].as_array().unwrap())
            .enumerate()
        {
            let mut row_abs = 0f64;
            let mut row_rel = 0f64;
            let mut worst = (0usize, 0f64, 0f64);
            let expected: Vec<f64> = serde_json::from_value(expected.clone()).unwrap();
            assert_eq!(actual.len(), config.vocab_size);
            assert_eq!(actual.len(), expected.len());
            for (index, (&a, b)) in actual.iter().zip(expected).enumerate() {
                let a = f64::from(a);
                assert!(a.is_finite() && b.is_finite());
                let d = (a - b).abs();
                row_abs = row_abs.max(d);
                let r = if d == 0.0 { 0.0 } else { d / b.abs() };
                if r > row_rel {
                    row_rel = r;
                    worst = (index, a, b);
                }
                absolute = absolute.max(d);
                relative = relative.max(if d == 0.0 { 0.0 } else { d / b.abs() });
                squared_error += d * d;
                squared_reference += b * b;
            }
            eprintln!("row={row} abs={row_abs} relative={row_rel} worst={worst:?}");
        }
    }
    eprintln!(
        "profile={} bits={} native_weight_bytes={} max_abs={absolute} max_relative={relative} relative_l2={}",
        reference["profile"],
        config.quantization_bits,
        model.weight_buffer_bytes().unwrap(),
        (squared_error / squared_reference).sqrt()
    );
    assert!(absolute < 0.02, "absolute logit error {absolute}");
    assert!(relative < 0.01, "relative logit error {relative}");
}
