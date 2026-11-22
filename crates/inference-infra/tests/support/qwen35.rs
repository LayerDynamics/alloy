use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    qwen35::{
        config::Qwen35Config,
        weights::{Qwen35Weights, load_verified},
    },
};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

pub fn load(profile: &Value, artifact_env: &str) -> (Qwen35Config, Qwen35Weights) {
    let root = PathBuf::from(std::env::var_os(artifact_env).expect("artifact"))
        .canonicalize()
        .unwrap();
    let pins: BTreeMap<_, _> = profile["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| {
            let hex = file["sha256"].as_str().unwrap();
            let mut hash = [0; 32];
            for (i, b) in hash.iter_mut().enumerate() {
                *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
            }
            (root.join(file["path"].as_str().unwrap()), hash)
        })
        .collect();
    load_verified(
        &root,
        LoadOptions {
            max_file_bytes: 8 * 1024 * 1024 * 1024,
            max_weight_bytes: 8 * 1024 * 1024 * 1024,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap()
}
