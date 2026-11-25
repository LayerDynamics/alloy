use inference_infra::decoder::loader::{CheckpointVerification, LoadOptions};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

pub fn load<T, E: Debug>(
    profile: &Value,
    env: &str,
    limit: usize,
    loader: impl FnOnce(&Path, LoadOptions, &CheckpointVerification<'_>) -> Result<T, E>,
) -> T {
    let root = PathBuf::from(std::env::var_os(env).expect(env))
        .canonicalize()
        .unwrap();
    let mut pins = BTreeMap::new();
    for file in profile["artifact"]["files"].as_array().unwrap() {
        let hex = file["sha256"].as_str().unwrap();
        assert_eq!(hex.len(), 64);
        let mut digest = [0; 32];
        for (i, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        assert!(
            pins.insert(root.join(file["path"].as_str().unwrap()), digest)
                .is_none()
        );
    }
    loader(
        &root,
        LoadOptions {
            max_file_bytes: limit,
            max_weight_bytes: limit,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap()
}
