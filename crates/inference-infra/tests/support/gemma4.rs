use inference_infra::decoder::{
    gemma4::model::Gemma4Model,
    loader::{CheckpointVerification, LoadOptions},
};
use mlx_ffi::Stream;
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

pub fn profile() -> Value {
    serde_json::from_str(include_str!("../fixtures/gemma4_e4b_profile.json")).unwrap()
}

pub fn load(profile: &Value, stream: &Stream) -> Gemma4Model {
    let root = PathBuf::from(std::env::var_os("ALLOY_GEMMA4_ARTIFACT").unwrap())
        .canonicalize()
        .unwrap();
    let files = profile["artifact"]["files"].as_array().unwrap();
    let pins: BTreeMap<_, _> = files
        .iter()
        .map(|f| {
            let hex = f["sha256"].as_str().unwrap();
            (
                root.join(f["path"].as_str().unwrap()),
                std::array::from_fn(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap()),
            )
        })
        .collect();
    let options = LoadOptions {
        max_file_bytes: files
            .iter()
            .map(|f| f["bytes"].as_u64().unwrap() as usize)
            .sum(),
        max_weight_bytes: 4_666_273_960 + 64 * 1024 * 1024,
    };
    let verification = CheckpointVerification {
        sha256: &pins,
        cancelled: &|| false,
    };
    Gemma4Model::load_verified(&root, options, &verification, stream).unwrap()
}
