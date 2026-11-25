//! Shared by the family-owned Llama and SmolLM integration test executables.
#[path = "cache_memory.rs"]
mod cache_memory;

use inference_infra::decoder::{
    config::DecoderConfig,
    loader::{CheckpointVerification, LoadOptions, load_weights_verified},
    mlx_model::MlxDecoderModel,
    model::ModelError,
};
use mlx_ffi::memory;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

pub fn run<E: std::fmt::Debug>(profile: &Value, parse: fn(&Value) -> Result<DecoderConfig, E>) {
    let root = PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap())
        .canonicalize()
        .unwrap();
    let mut pins = BTreeMap::new();
    for file in profile["artifact"]["files"].as_array().unwrap() {
        let hash = file["sha256"].as_str().unwrap();
        let bytes =
            std::array::from_fn(|i| u8::from_str_radix(&hash[2 * i..2 * i + 2], 16).unwrap());
        assert!(
            pins.insert(root.join(file["path"].as_str().unwrap()), bytes)
                .is_none()
        );
    }
    let config =
        parse(&serde_json::from_slice(&fs::read(root.join("config.json")).unwrap()).unwrap())
            .unwrap();
    let verification = CheckpointVerification {
        sha256: &pins,
        cancelled: &|| false,
    };
    cache_memory::run(profile, |tier| {
        let before = memory::active_memory_bytes().unwrap();
        let weights = load_weights_verified(
            &config,
            &root,
            LoadOptions {
                max_file_bytes: 16 << 30,
                max_weight_bytes: tier.budget,
            },
            &verification,
        );
        if tier.budget == 8 << 30
            && matches!(
                profile["id"].as_str(),
                Some("llama-3.1-8b-8bit" | "qwen3-14b-4bit")
            )
        {
            let Err(ModelError::MemoryLimit { required, limit }) = weights else {
                panic!("expected native 8 GiB weight budget refusal");
            };
            assert!(required > limit && limit == tier.budget);
            assert_eq!(memory::active_memory_bytes().unwrap(), before);
            println!(
                "native_cache_memory={}",
                json!({"profile":profile["id"], "budget_bytes":limit,
                "refused_weight_bytes":required, "active_before":before, "active_after":memory::active_memory_bytes().unwrap()})
            );
            return;
        }
        let model = MlxDecoderModel::from_weights(
            config.clone(),
            weights.unwrap(),
            tier.budget as u64,
            &tier.stream,
        )
        .unwrap();
        let peak = tier.load_peak();
        let weights = model.weight_buffer_bytes().unwrap();
        assert!(weights >= model.weight_payload_bytes());
        let cache = model
            .new_cache(
                tier.tokens.len() + 1,
                (tier.budget as u64).checked_sub(weights).unwrap(),
                &tier.stream,
            )
            .unwrap();
        let (initial, measured) = tier.measure(
            peak,
            weights,
            config.vocab_size,
            cache,
            |tokens, cache| {
                model
                    .prefill(
                        tokens,
                        cache,
                        // Projection, contiguous output and F32 readback.
                        (config.vocab_size * tokens.len() * 3 * size_of::<f32>()) as u64,
                        &tier.stream,
                    )
                    .unwrap()
            },
            |cache| (cache.len(), cache.buffer_bytes().unwrap()),
        );
        let kv =
            (tier.tokens.len() + 1) * config.num_layers * config.kv_heads * config.head_dim * 2 * 4;
        assert!(measured >= kv as u64 && initial <= measured);
    });
}
