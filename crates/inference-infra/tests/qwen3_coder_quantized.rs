#[path = "support/numerical_capture.rs"]
mod numerical_capture;
#[path = "support/pinned_checkpoint.rs"]
mod pinned_checkpoint;

use inference_infra::decoder::{
    config::DecoderFamily,
    mlx_model::{MlxDecoderError, MlxDecoderModel},
    model::ModelError,
    qwen3_coder::CoderConfig,
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

const ARTIFACT: &str = "ALLOY_QWEN3_CODER_ARTIFACT";
const PAYLOAD: usize = 19_089_580_032; // All packed codes and F32 metadata across the four original shards.
fn profile() -> Value {
    serde_json::from_str(include_str!("fixtures/qwen3_coder_profile.json")).unwrap()
}
fn pinned_config() -> Value {
    let root = PathBuf::from(std::env::var_os(ARTIFACT).unwrap());
    let bytes = fs::read(root.join("config.json")).unwrap();
    let profile = profile();
    let entry = profile["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"] == "config.json")
        .unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), entry["sha256"]);
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
#[ignore = "requires pinned 30B-A3B four-shard checkpoint"]
fn qwen3_coder_verified_geometry() {
    let c = CoderConfig::from_json(&pinned_config()).unwrap();
    assert_eq!(c.decoder.family, DecoderFamily::Qwen3Moe);
    assert_eq!((c.decoder.num_layers, c.experts, c.top_k), (48, 128, 8));
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    // Preflight must visit every packed component and reject a one-byte deficit
    // before materializing any resident weights.
    pinned_checkpoint::load(&profile(), ARTIFACT, 24 << 30, |path, mut options, v| {
        options.max_weight_bytes = PAYLOAD - 1;
        assert!(
            matches!(MlxDecoderModel::load_qwen3_coder_verified(path,options,v,&stream),
            Err(MlxDecoderError::Model(ModelError::MemoryLimit { required,limit })) if required == PAYLOAD && limit == PAYLOAD-1)
        );
        Ok::<_, String>(())
    });
}

#[test]
#[ignore = "requires pinned 30B-A3B configuration"]
fn qwen3_coder_mixed_packing_rejected() {
    let valid = pinned_config();
    for key in ["quantization", "quantization_config"] {
        let mut changed = valid.clone();
        changed[key]["model.layers.47.mlp.gate"]["bits"] = 4.into();
        assert!(CoderConfig::from_json(&changed).is_err());
    }
    let mut changed = valid;
    changed["mlp_only_layers"] = serde_json::json!([1]);
    assert!(CoderConfig::from_json(&changed).is_err());
}

#[test]
#[ignore = "requires the pinned checkpoint and eager Metal reference"]
fn qwen3_coder_trained_prefill_decode() {
    let profile = profile();
    let bytes = fs::read(std::env::var_os("ALLOY_QWEN3_CODER_REFERENCE").unwrap()).unwrap();
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
    assert_eq!(reference["device"], "gpu");
    assert_eq!(reference["compile_enabled"], false);
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let model = pinned_checkpoint::load(&profile, ARTIFACT, 24 << 30, |dir, opts, v| {
        MlxDecoderModel::load_qwen3_coder_verified(dir, opts, v, &stream)
    });
    assert_eq!(model.weight_payload_bytes(), PAYLOAD as u64);
    assert!(model.weight_buffer_bytes().unwrap() >= PAYLOAD as u64);
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for (index, case) in cases.iter().enumerate() {
        let prompt: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let decode: Vec<u32> = serde_json::from_value(case["decode_tokens"].clone()).unwrap();
        assert_eq!(case["tokens"], profile["prompt_tokens"][index]);
        assert_eq!(case["decode_tokens"], profile["decode_tokens"]);
        let mut cache = model
            .new_cache(prompt.len() + decode.len(), 128 << 20, &stream)
            .unwrap();
        let expected = case["logits"].as_array().unwrap();
        assert_eq!(expected.len(), decode.len() + 1);
        for (step, want) in expected.iter().enumerate() {
            let rows = if step == 0 {
                model.prefill(&prompt, &mut cache, 256 << 20, &stream)
            } else {
                model.decode(decode[step - 1], &mut cache, 256 << 20, &stream)
            }
            .unwrap();
            assert_eq!(rows.rows(), 1);
            let actual = rows.last_row();
            assert_eq!(actual.len(), 151936);
            numerical_capture::record(&profile, index, step, &prompt, &decode[..step], actual);
            let expected = want.as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            let mut abs = 0.0_f64;
            let mut rel = 0.0_f64;
            for (&obtained, expected) in actual.iter().zip(expected) {
                let expected = expected.as_f64().unwrap();
                assert!(expected.is_finite() && obtained.is_finite());
                let d = (f64::from(obtained) - expected).abs();
                abs = abs.max(d);
                if d != 0.0 {
                    rel = rel.max(d / expected.abs());
                }
            }
            eprintln!("case={index} step={step} max_abs={abs} max_relative={rel}");
            assert!(abs < 0.02 && rel < 0.01);
        }
    }
}

#[path = "support/cache_memory.rs"]
mod cache_memory;

#[test]
#[ignore = "requires pinned 30B-A3B weights, Metal and a physical 36 GiB Mac"]
fn qwen3_coder_native_memory_tiers() {
    let profile = profile();
    cache_memory::run(&profile, |tier| {
        let before = mlx_ffi::memory::active_memory_bytes().unwrap();
        // The four input shards exceed 8 GiB even when the resident limit is smaller.
        // Keep the file-read and resident-payload limits independent.
        pinned_checkpoint::load(
            &profile,
            ARTIFACT,
            24 << 30,
            |path, mut options, verification| {
                options.max_weight_bytes = tier.budget;
                let result = MlxDecoderModel::load_qwen3_coder_verified(
                    path,
                    options,
                    verification,
                    &tier.stream,
                );
                if tier.budget < PAYLOAD {
                    let Err(MlxDecoderError::Model(ModelError::MemoryLimit { required, limit })) =
                        result
                    else {
                        panic!("expected pre-upload weight refusal, got {:?}", result.err());
                    };
                    // Inventory stops at the first tensor that crosses the limit.
                    // verified_geometry separately checks the exact full payload.
                    assert_eq!(limit, tier.budget);
                    assert!(required > limit && required <= PAYLOAD);
                    let after = mlx_ffi::memory::active_memory_bytes().unwrap();
                    assert_eq!(before, after);
                    println!(
                        "native_cache_memory={}",
                        serde_json::json!({
                    "profile":profile["id"], "budget_bytes":tier.budget, "refused_weight_bytes":required,
                    "active_before":before, "active_after":after})
                    );
                    return Ok::<_, MlxDecoderError>(());
                }
                let model = result?;
                let peak = tier.load_peak();
                assert_eq!(model.weight_payload_bytes(), PAYLOAD as u64);
                let weights = model.weight_buffer_bytes()?;
                assert!(weights >= PAYLOAD as u64);
                let cache = model.new_cache(
                    tier.tokens.len() + 1,
                    (tier.budget as u64).checked_sub(weights).unwrap(),
                    &tier.stream,
                )?;
                let vocabulary = model.config().vocab_size;
                let (initial, measured) = tier.measure(
                    peak,
                    weights,
                    vocabulary,
                    cache,
                    |tokens, cache| {
                        model
                            .prefill(
                                tokens,
                                cache,
                                // Projection, contiguous output and F32 readback.
                                (tokens.len() * vocabulary * 3 * size_of::<f32>()) as u64,
                                &tier.stream,
                            )
                            .unwrap()
                    },
                    |cache| (cache.len(), cache.buffer_bytes().unwrap()),
                );
                let c = model.config();
                let kv = (tier.tokens.len() + 1) * c.num_layers * c.kv_heads * c.head_dim * 2 * 4;
                assert!(measured >= kv as u64 && initial <= measured);
                Ok(())
            },
        );
    });
}
