//! Frozen E4B trained text checkpoint inventory and bounded ingestion.
use inference_infra::decoder::{
    gemma4::{Gemma4TextConfig, weights::load_verified},
    loader::{CheckpointTensor, CheckpointVerification, LoadOptions, TensorRequest},
    model::ModelError,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
const WEIGHT_BYTES: usize = 4_666_273_960;

#[test]
fn gemma4_e4b_tensor_inventory() {
    let c: Value = serde_json::from_str(include_str!("fixtures/gemma4_e4b_config.json")).unwrap();
    let layout = Gemma4TextConfig::from_json(&c)
        .unwrap()
        .affine4_layout()
        .unwrap();
    let matrices = layout
        .tensors
        .values()
        .filter(|v| matches!(v, TensorRequest::Matrix { .. }))
        .count();
    assert_eq!((matrices, layout.tensors.len() - matrices), (345, 320));
    let matrix = |name: &str| match layout.tensors.get(&format!("language_model.model.{name}")) {
        Some(TensorRequest::Matrix { rows, cols }) => (*rows, *cols),
        _ => panic!("missing matrix {name}"),
    };
    assert_eq!(matrix("embed_tokens_per_layer.weight"), (262144, 10752));
    assert_eq!(matrix("layers.22.self_attn.k_proj.weight"), (512, 2560));
    assert_eq!(matrix("layers.23.self_attn.v_proj.weight"), (1024, 2560));
    assert_eq!(matrix("layers.29.self_attn.q_proj.weight"), (4096, 2560));
    for i in 24..42 {
        for name in ["k_proj", "v_proj", "k_norm"] {
            assert!(!layout.tensors.contains_key(&format!(
                "language_model.model.layers.{i}.self_attn.{name}.weight"
            )));
        }
    }
    for (key, value) in [
        ("use_double_wide_mlp", json!(true)),
        ("enable_moe_block", json!(true)),
        ("tie_word_embeddings", json!(false)),
        ("hidden_size_per_layer_input", json!(0)),
    ] {
        let mut bad = c.clone();
        bad["text_config"][key] = value;
        assert!(Gemma4TextConfig::from_json(&bad).is_err());
    }
}

#[test]
#[ignore = "requires the pinned trained ALLOY_GEMMA4_ARTIFACT"]
fn gemma4_e4b_verified_quantized_loading() {
    let root = PathBuf::from(std::env::var_os("ALLOY_GEMMA4_ARTIFACT").unwrap())
        .canonicalize()
        .unwrap();
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/gemma4_e4b_profile.json")).unwrap();
    let files = manifest["artifact"]["files"].as_array().unwrap();
    let pins: BTreeMap<_, _> = files
        .iter()
        .map(|f| {
            let hex = f["sha256"].as_str().unwrap();
            let digest =
                std::array::from_fn(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap());
            (root.join(f["path"].as_str().unwrap()), digest)
        })
        .collect();
    let verification = CheckpointVerification {
        sha256: &pins,
        cancelled: &|| false,
    };
    let options = LoadOptions {
        max_file_bytes: files
            .iter()
            .map(|f| f["bytes"].as_u64().unwrap() as usize)
            .sum(),
        max_weight_bytes: WEIGHT_BYTES,
    };
    assert!(
        matches!(load_verified(&root, LoadOptions{max_weight_bytes:WEIGHT_BYTES-1,..options}, &verification),
        Err(ModelError::MemoryLimit{required,limit}) if required==WEIGHT_BYTES && limit+1==required)
    );
    let (_, tensors) = load_verified(&root, options, &verification).unwrap();
    let (mut matrices, mut vectors, mut bytes) = (0, 0, 0);
    for (name, t) in &tensors {
        assert!(name.starts_with("language_model.model."));
        match t {
            CheckpointTensor::Matrix(m) => {
                matrices += 1;
                assert_eq!((m.bits(), m.affine4_group_size()), (Some(4), Some(64)));
                bytes += m.resident_bytes();
            }
            CheckpointTensor::Float { values, .. } => {
                vectors += 1;
                assert!(values.iter().all(|x| x.is_finite()));
                bytes += values.len() * 4;
            }
        }
    }
    assert_eq!((matrices, vectors, bytes), (345, 320, WEIGHT_BYTES));
    println!(
        "gemma4_text_loading: packed_matrices={matrices} float_tensors={vectors} resident_payload_bytes={bytes}"
    );
}
