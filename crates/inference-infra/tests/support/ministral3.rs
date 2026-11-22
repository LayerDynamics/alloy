use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    ministral3::weights::load_verified,
    mlx_model::MlxDecoderModel,
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

pub fn load_pinned_model(profile: &Value, resident_bytes: usize) -> (MlxDecoderModel, Stream) {
    let root = PathBuf::from(std::env::var_os("ALLOY_MINISTRAL3_ARTIFACT").unwrap())
        .canonicalize()
        .unwrap();
    let pins: BTreeMap<_, _> = profile["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| {
            let hex = file["sha256"].as_str().unwrap();
            (
                root.join(file["path"].as_str().unwrap()),
                std::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()),
            )
        })
        .collect();
    let (config, weights) = load_verified(
        &root,
        LoadOptions {
            max_file_bytes: profile["artifact"]["files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["bytes"].as_u64().unwrap() as usize)
                .sum(),
            max_weight_bytes: resident_bytes,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap();
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let model = MlxDecoderModel::from_weights(
        config.decoder_config().unwrap(),
        weights,
        resident_bytes as u64,
        &stream,
    )
    .unwrap();
    assert_eq!(model.weight_buffer_bytes().unwrap(), resident_bytes as u64);
    (model, stream)
}
