//! Real text-only loading of the pinned MLX conditional generation checkpoint.
use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    ministral3::weights::load_verified,
    model::ModelError,
};
use std::{collections::BTreeMap, path::PathBuf};

#[test]
#[ignore = "requires pinned ALLOY_MINISTRAL3_ARTIFACT"]
fn ministral3_text_checkpoint_loading() {
    check_loading(
        include_str!("fixtures/ministral3_profile.json"),
        4,
        2_143_678_464,
    );
}

#[test]
#[ignore = "requires pinned 8-bit ALLOY_MINISTRAL3_ARTIFACT"]
fn ministral3_affine8_checkpoint_loading() {
    check_loading(
        include_str!("fixtures/ministral3_8bit_profile.json"),
        8,
        3_858_100_224,
    );
}

fn check_loading(manifest: &str, bits: u8, resident_bytes: usize) {
    let profile: serde_json::Value = serde_json::from_str(manifest).unwrap();
    let root =
        PathBuf::from(std::env::var_os("ALLOY_MINISTRAL3_ARTIFACT").expect("pinned artifact"))
            .canonicalize()
            .unwrap();
    let files = profile["artifact"]["files"].as_array().unwrap();
    let pins: BTreeMap<_, _> = files
        .iter()
        .map(|file| {
            let hex = file["sha256"].as_str().unwrap();
            let digest =
                std::array::from_fn(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap());
            (root.join(file["path"].as_str().unwrap()), digest)
        })
        .collect();
    let verification = CheckpointVerification {
        sha256: &pins,
        cancelled: &|| false,
    };
    let options = LoadOptions {
        max_file_bytes: files
            .iter()
            .filter(|f| {
                matches!(
                    f["path"].as_str(),
                    Some("config.json" | "model.safetensors" | "model.safetensors.index.json")
                )
            })
            .map(|f| f["bytes"].as_u64().unwrap() as usize)
            .sum(),
        max_weight_bytes: resident_bytes,
    };
    assert!(
        matches!(load_verified(&root,LoadOptions {max_weight_bytes:options.max_weight_bytes-1,..options},&verification),Err(ModelError::MemoryLimit{required,limit}) if required == options.max_weight_bytes && limit == required-1)
    );
    let (config, weights) = load_verified(&root, options, &verification).unwrap();
    assert_eq!(weights.layers.len(), 26);
    assert!(config.tied_embeddings && weights.lm_head.is_none());
    let mut matrices = vec![&weights.embed_tokens];
    let mut norms = weights.final_layernorm.len();
    for l in &weights.layers {
        assert!(l.q_norm.is_empty() && l.k_norm.is_empty());
        norms += l.input_layernorm.len() + l.post_attention_layernorm.len();
        matrices.extend([
            &l.q_proj,
            &l.k_proj,
            &l.v_proj,
            &l.o_proj,
            &l.gate_proj,
            &l.up_proj,
            &l.down_proj,
        ]);
    }
    assert_eq!(matrices.len(), 183);
    assert!(
        matrices
            .iter()
            .all(|m| m.bits() == Some(bits) && m.affine4_group_size() == Some(64))
    );
    let bytes = norms * 4 + matrices.iter().map(|m| m.resident_bytes()).sum::<usize>();
    assert_eq!(bytes, options.max_weight_bytes);
    println!("verified 183 packed text matrices and 53 norms, {bytes} resident bytes");
}
