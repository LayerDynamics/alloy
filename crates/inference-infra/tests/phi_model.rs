//! Independent native Phi model checks; tiny fixtures do not qualify a profile.
use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    phi::{
        PhiConfig, PhiRopeRegime,
        model::MlxPhiModel,
        weights::{PhiWeights, load_verified},
    },
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
const LIMIT: u64 = 16 * 1024 * 1024;
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/phi_model")
}
fn pins(root: &Path) -> BTreeMap<PathBuf, [u8; 32]> {
    let manifest: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/phi_model/manifest.json")).unwrap();
    manifest
        .into_iter()
        .map(|(name, hex)| {
            let bytes: Vec<u8> = (0..64)
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            (root.join(name), bytes.try_into().unwrap())
        })
        .collect()
}
fn weights() -> (PhiConfig, PhiWeights) {
    let root = root();
    let pins = pins(&root);
    load_verified(
        &root,
        LoadOptions {
            max_file_bytes: LIMIT as usize,
            max_weight_bytes: LIMIT as usize,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap()
}
fn streams() -> Vec<(&'static str, Stream)> {
    let mut kinds = vec![("cpu", DeviceType::Cpu)];
    if cfg!(target_os = "macos") {
        kinds.push(("gpu", DeviceType::Gpu));
    }
    kinds
        .into_iter()
        .map(|(name, kind)| {
            let d = Device::new(kind, 0).unwrap();
            assert!(d.is_available().unwrap());
            (name, Stream::new(&d).unwrap())
        })
        .collect()
}
fn reference() -> Value {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    let (bytes, digest) = (
        include_bytes!("fixtures/phi_model/reference-linux-aarch64.json").as_slice(),
        "fbd68a15501e88a8129d8260765759871e4518f4f82c809e15e84b2f881443cb",
    );
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/phi_model/manifest.json")).unwrap();
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    let (bytes, digest) = (
        include_bytes!("fixtures/phi_model/reference.json").as_slice(),
        manifest["reference.json"].as_str().unwrap(),
    );
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), digest);
    serde_json::from_slice(bytes).unwrap()
}
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn tokens(v: &Value) -> Vec<u32> {
    v["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}
fn strict(actual: &[f32], expected: &Value) {
    let expected = numbers(expected);
    assert_eq!(actual.len(), expected.len());
    let mut absolute = 0.0_f64;
    let mut relative = 0.0_f64;
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite());
        let error = (f64::from(a) - b).abs();
        absolute = absolute.max(error);
        if b == 0.0 {
            assert_eq!(error, 0.0);
        } else {
            relative = relative.max(error / b.abs());
        }
    }
    assert!(
        absolute < 0.02 && relative < 0.01,
        "FR8 max_abs={absolute} max_pointwise_relative={relative}"
    );
}
fn model(stream: &Stream, regime: PhiRopeRegime) -> MlxPhiModel {
    let (c, w) = weights();
    MlxPhiModel::from_weights(c, w, regime, LIMIT, stream).unwrap()
}
#[test]
fn phi4mini_native_reference() {
    let reference = reference();
    let tokens = tokens(&reference);
    for (name, stream) in streams() {
        let model = model(&stream, PhiRopeRegime::Long);
        let result = model
            .forward(&tokens, LIMIT, LIMIT, LIMIT, &stream)
            .unwrap();
        strict(result.values(), &reference[name]["full"]);
        assert_eq!(result.rows(), tokens.len());
        assert_eq!(result.vocabulary_size(), 128);
        assert_eq!(
            model.weight_payload_bytes(),
            model.config().weight_payload_plan().unwrap().payload_bytes
        );
        assert!(model.weight_buffer_bytes().unwrap() >= model.weight_payload_bytes());
    }
}
#[test]
fn phi4mini_native_cached_matches_cold_and_regime_is_owner_bound() {
    let reference = reference();
    let tokens = tokens(&reference);
    for (name, stream) in streams() {
        let model = model(&stream, PhiRopeRegime::Long);
        let full = model
            .forward(&tokens, LIMIT, LIMIT, LIMIT, &stream)
            .unwrap();
        for split in 1..tokens.len() {
            let mut cache = model.new_cache(64, LIMIT, &stream).unwrap();
            let prefix = model
                .forward_cached(
                    &tokens[..split],
                    &mut cache,
                    false,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false,
                )
                .unwrap();
            strict(
                prefix.values(),
                &reference[name]["splits"][split - 1]["prefix"],
            );
            let suffix = model
                .forward_cached(
                    &tokens[split..],
                    &mut cache,
                    false,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false,
                )
                .unwrap();
            strict(
                suffix.values(),
                &reference[name]["splits"][split - 1]["suffix"],
            );
            for (&a, &b) in suffix.values().iter().zip(&full.values()[split * 128..]) {
                assert!((a - b).abs() < 0.0002, "cached/full diverged: {a} vs {b}");
            }
            assert_eq!(cache.len(), tokens.len());
            cache.truncate(model.instance_id(), split).unwrap();
            let last = model
                .forward_cached(
                    &tokens[split..],
                    &mut cache,
                    true,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false,
                )
                .unwrap();
            assert_eq!(last.values(), suffix.last_row());
            cache.reset(model.instance_id()).unwrap();
            assert_eq!(cache.len(), 0);
        }
        let other = crate::model(&stream, PhiRopeRegime::Short);
        let mut foreign = other.new_cache(64, LIMIT, &stream).unwrap();
        assert!(
            model
                .forward_cached(
                    &tokens,
                    &mut foreign,
                    false,
                    LIMIT,
                    LIMIT,
                    &stream,
                    &mut || false
                )
                .is_err()
        );
        assert_eq!(foreign.len(), 0);
        assert!(model.is_healthy());
    }
}
#[test]
fn phi4mini_native_cancel_and_invalid_inputs_preserve_cache() {
    let reference = reference();
    let tokens = tokens(&reference);
    for (_, stream) in streams() {
        let model = model(&stream, PhiRopeRegime::Long);
        let mut cache = model.new_cache(64, LIMIT, &stream).unwrap();
        model
            .forward_cached(
                &tokens[..2],
                &mut cache,
                false,
                LIMIT,
                LIMIT,
                &stream,
                &mut || false,
            )
            .unwrap();
        let mut calls = 0;
        let result = model.forward_cached(
            &tokens[2..],
            &mut cache,
            false,
            LIMIT,
            LIMIT,
            &stream,
            &mut || {
                calls += 1;
                calls >= 3
            },
        );
        assert!(matches!(
            result,
            Err(inference_infra::decoder::mlx_model::MlxDecoderError::Cancelled)
        ));
        assert_eq!(cache.len(), 2);
        assert!(cache.is_healthy());
        assert!(model.is_healthy());
        for input in [vec![], vec![128]] {
            assert!(
                model
                    .forward_cached(
                        &input,
                        &mut cache,
                        false,
                        LIMIT,
                        LIMIT,
                        &stream,
                        &mut || false
                    )
                    .is_err()
            );
            assert_eq!(cache.len(), 2);
        }
        assert!(
            model
                .forward_cached(
                    &tokens[2..],
                    &mut cache,
                    false,
                    0,
                    LIMIT,
                    &stream,
                    &mut || false
                )
                .is_err()
        );
        assert!(
            model
                .forward_cached(
                    &tokens[2..],
                    &mut cache,
                    false,
                    LIMIT,
                    0,
                    &stream,
                    &mut || false
                )
                .is_err()
        );
        assert_eq!(cache.len(), 2);
        let recovered = model
            .forward_cached(
                &tokens[2..],
                &mut cache,
                false,
                LIMIT,
                LIMIT,
                &stream,
                &mut || false,
            )
            .unwrap();
        let mut fresh = model.new_cache(64, LIMIT, &stream).unwrap();
        model
            .forward_cached(
                &tokens[..2],
                &mut fresh,
                false,
                LIMIT,
                LIMIT,
                &stream,
                &mut || false,
            )
            .unwrap();
        let expected = model
            .forward_cached(
                &tokens[2..],
                &mut fresh,
                false,
                LIMIT,
                LIMIT,
                &stream,
                &mut || false,
            )
            .unwrap();
        assert_eq!(recovered.values(), expected.values());
    }
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "alloy-phi-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn phi4mini_verified_loader_rejects_wrong_inventory_and_hashes() {
    let dir = Temp::new();
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(root().join(name), dir.0.join(name)).unwrap();
    }
    let mut hashes = pins(&dir.0);
    let path = dir.0.join("model.safetensors");
    // Keep the inventory valid so this rejection specifically exercises trust
    // in the supplied digest, independently of tensor-name validation.
    hashes.insert(path.clone(), [0; 32]);
    let options = LoadOptions {
        max_file_bytes: LIMIT as usize,
        max_weight_bytes: LIMIT as usize,
    };
    assert!(
        load_verified(
            &dir.0,
            options,
            &CheckpointVerification {
                sha256: &hashes,
                cancelled: &|| false
            }
        )
        .is_err(),
        "correct inventory cannot authorize a wrong trusted hash"
    );
    let mut bytes = std::fs::read(&path).unwrap();
    let at = bytes
        .windows(b"model.norm.weight".len())
        .position(|w| w == b"model.norm.weight")
        .unwrap();
    bytes[at + 7] = b'e';
    std::fs::write(&path, &bytes).unwrap();
    hashes.insert(path, Sha256::digest(&bytes).into());
    assert!(
        load_verified(
            &dir.0,
            options,
            &CheckpointVerification {
                sha256: &hashes,
                cancelled: &|| false
            }
        )
        .is_err(),
        "valid hash cannot authorize wrong inventory"
    );
    assert!(
        load_verified(
            &root(),
            options,
            &CheckpointVerification {
                sha256: &pins(&root()),
                cancelled: &|| true
            }
        )
        .is_err()
    );
}
#[test]
fn phi4mini_payload_plans_match_declared_representation() {
    let (c, w) = weights();
    let plan = c.weight_payload_plan().unwrap();
    assert!(plan.payload_bytes < LIMIT);
    assert!(plan.largest_upload_bytes <= plan.payload_bytes);
    let real = PhiConfig::from_json(
        &serde_json::from_str(include_str!("fixtures/phi4_mini_config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        real.weight_payload_plan().unwrap().payload_bytes,
        2_398_187_520
    );
    assert!(c.step_payload_plan(0, 0).is_err());
    assert!(c.step_payload_plan(2, 1).is_err());
    assert!(c.step_payload_plan(1, 65).is_err());
    assert!(c.cache_spec(65).is_err());
    for (_, stream) in streams() {
        assert!(
            MlxPhiModel::from_weights(
                c.clone(),
                w.clone(),
                PhiRopeRegime::Long,
                plan.payload_bytes - 1,
                &stream
            )
            .is_err()
        );
        let mut wrong = w.clone();
        wrong.embedding = inference_infra::decoder::fixture::MatrixWeight::f32(vec![
                0.0;
                c.hidden * c.vocabulary
            ]);
        assert!(
            MlxPhiModel::from_weights(c.clone(), wrong, PhiRopeRegime::Long, LIMIT, &stream)
                .is_err()
        );
        let mut wrong = w.clone();
        wrong.norm[0] = f32::NAN;
        assert!(
            MlxPhiModel::from_weights(c.clone(), wrong, PhiRopeRegime::Long, LIMIT, &stream)
                .is_err()
        );
    }
}
