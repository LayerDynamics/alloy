//! Metadata registration does not qualify a profile or execute its weights.
use engine::{
    registry::{FamilyRegistration, FamilyVersion, Registry},
    types::{ArtifactId, Capabilities, DecodingCapability, ExecutionBackend, WeightRepresentation},
};
use serde_json::{Value, json};
use std::path::PathBuf;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT_DIRECTORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = NEXT_DIRECTORY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "alloy-hybrid-registry-{}-{stamp}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, value: &Value) {
        std::fs::write(
            self.0.join("config.json"),
            serde_json::to_vec(value).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn registry() -> Registry {
    let mut registry = Registry::new();
    registry
        .register_family(
            FamilyRegistration::new(
                FamilyVersion::new(1, 0, 0),
                vec!["Qwen3_5ForConditionalGeneration".into()],
                Capabilities {
                    task: "text-generation".into(),
                    family: "qwen35-registry-fixture".into(),
                    backend: ExecutionBackend::HostCpu,
                    weight_representations: vec![WeightRepresentation::F32],
                    cache_modes: vec![],
                    context_limit_tokens: 256,
                    supports_streaming: false,
                    supports_incremental_kv: false,
                    supports_prefix_reuse: false,
                    decoding_modes: vec![DecodingCapability::Greedy],
                },
            )
            .unwrap(),
        )
        .unwrap();
    registry
}
fn config() -> Value {
    serde_json::from_str(include_str!(
        "../../inference-infra/tests/fixtures/qwen35_model/config.json"
    ))
    .unwrap()
}
#[test]
fn qwen35_registry_reads_validated_nested_context() {
    let directory = Directory::new();
    let mut config = config();
    let expected = config["text_config"]["max_position_embeddings"]
        .as_u64()
        .unwrap() as usize;
    for redundant in [false, true] {
        if redundant {
            config["max_position_embeddings"] = json!(expected);
        }
        directory.write(&config);
        let mut registry = registry();
        let id = ArtifactId::new("hybrid").unwrap();
        registry
            .register_artifact(id.clone(), &directory.0)
            .unwrap();
        assert_eq!(
            registry
                .artifact(&id)
                .unwrap()
                .declared_context_limit_tokens(),
            expected
        );
    }
}
#[test]
fn qwen35_registry_rejects_malformed_hybrid_without_publication() {
    let directory = Directory::new();
    for index in 0..6 {
        let mut malformed = config();
        match index {
            0 => malformed["text_config"] = Value::Null,
            1 => malformed["text_config"] = json!([]),
            2 => {
                malformed["text_config"]
                    .as_object_mut()
                    .unwrap()
                    .remove("max_position_embeddings");
            }
            3 => malformed["text_config"]["max_position_embeddings"] = json!(0),
            4 => malformed["text_config"]["max_position_embeddings"] = json!("256"),
            _ => malformed["max_position_embeddings"] = json!(1),
        }
        directory.write(&malformed);
        let mut registry = registry();
        let id = ArtifactId::new("hybrid").unwrap();
        assert!(
            registry
                .register_artifact(id.clone(), &directory.0)
                .is_err()
        );
        assert!(registry.artifact(&id).is_err());
        directory.write(&config());
        registry.register_artifact(id, &directory.0).unwrap();
    }
}
