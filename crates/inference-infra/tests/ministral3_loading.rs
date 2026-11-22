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
    let root =
        PathBuf::from(std::env::var_os("ALLOY_MINISTRAL3_ARTIFACT").expect("pinned artifact"))
            .canonicalize()
            .unwrap();
    let pins: BTreeMap<_, _> = [
        (
            "config.json",
            "cdd4bfa0a702d0241faaa3b80d15d481d7f8abf2659fba0958b1e7f24a28055f",
        ),
        (
            "model.safetensors.index.json",
            "22e571d413d8aa4e5f81f5ae678aa8b4f5b1d67598a67c08b31182075592ec93",
        ),
        (
            "model.safetensors",
            "69a68c0a961a64bb1145a697a83cf6cec45220d425c77a6c0a3568924078c3b4",
        ),
    ]
    .into_iter()
    .map(|(name, hex)| {
        let digest =
            std::array::from_fn(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap());
        (root.join(name), digest)
    })
    .collect();
    let verification = CheckpointVerification {
        sha256: &pins,
        cancelled: &|| false,
    };
    let options = LoadOptions {
        max_file_bytes: 2_745_210_934 + 74_906 + 1_988,
        max_weight_bytes: 2_143_678_464,
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
            .all(|m| m.bits() == Some(4) && m.affine4_group_size() == Some(64))
    );
    let bytes = norms * 4 + matrices.iter().map(|m| m.resident_bytes()).sum::<usize>();
    assert_eq!(bytes, options.max_weight_bytes);
    println!("verified 183 packed text matrices and 53 norms, {bytes} resident bytes");
}
