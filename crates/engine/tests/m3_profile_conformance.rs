//! These exercise qualification policy; synthetic observations do not qualify a model.
use engine::{
    conformance::*,
    registry::{FamilyRegistration, FamilyVersion, ModelProfile, Registry},
    types::{
        ArtifactId, Capabilities, DecodingCapability, ExecutionBackend, GenerationOptions,
        ModelProfileId, WeightRepresentation,
    },
};
use std::collections::BTreeMap;

fn digest(n: u8) -> String {
    format!("{n:02x}").repeat(32)
}

fn spec() -> QualificationSpec {
    QualificationSpec {
        profile_id: "fixture-profile".into(),
        artifact_id: "fixture-artifact".into(),
        identity: QualificationIdentity {
            candidate_revision: "a".repeat(40),
            dependencies_sha256: digest(1),
            artifact_manifest_sha256: digest(2),
            family: "fixture-family".into(),
            family_version: "1.0.0".into(),
            engine_api_version: engine::api::ENGINE_API_VERSION,
            tokenizer_sha256: digest(3),
            template_sha256: digest(4),
            tool_protocol_sha256: digest(5),
            profile_sha256: digest(6),
        },
        fixtures: FixtureKind::REQUIRED
            .iter()
            .enumerate()
            .map(|(i, &kind)| (format!("must-pass-{i}"), kind))
            .collect(),
        references: BTreeMap::from([(
            "fixed-prompt".into(),
            NumericalReference {
                input_sha256: digest(7),
                logits: vec![0.0, 1.0, -2.0, 3.0],
            },
        )]),
    }
}

fn passing(spec: &QualificationSpec) -> QualificationReport {
    QualificationReport {
        version: REPORT_VERSION,
        identity: spec.identity.clone(),
        suite_sha256: spec.sha256().unwrap(),
        numerical: spec
            .references
            .iter()
            .map(|(id, r)| NumericalObservation {
                fixture_id: id.clone(),
                input_sha256: r.input_sha256.clone(),
                logits: r.logits.clone(),
            })
            .collect(),
        fixtures: spec
            .fixtures
            .keys()
            .map(|id| FixtureObservation {
                fixture_id: id.clone(),
                passed: true,
                evidence_sha256: digest(8),
                detail: "Synthetic observation for policy testing only".into(),
            })
            .collect(),
    }
}

#[test]
fn m3_profile_numerical_reference() {
    // A large reference logit must not hide a failing near-zero component.
    let masked = NumericalMetrics::compare(&[100.0, 0.001], &[100.0, 0.002]).unwrap();
    assert!(masked.relative_l2 < 0.01);
    assert!(!masked.passes());
    assert!(
        !NumericalMetrics::compare(&[100.0, 0.0], &[100.0, 0.001])
            .unwrap()
            .passes()
    );
    assert!(
        NumericalMetrics::compare(&[0.0, -1.0], &[0.0, -1.001])
            .unwrap()
            .passes()
    );
    // The old additive tolerance would accept this; strict absolute must reject.
    assert!(
        !NumericalMetrics::compare(&[100.0], &[100.025])
            .unwrap()
            .passes()
    );
    // Absolute passes, relative fails.
    assert!(
        !NumericalMetrics::compare(&[0.1], &[0.102])
            .unwrap()
            .passes()
    );
    assert!(!NumericalMetrics::compare(&[1.0], &[1.01]).unwrap().passes());
    assert!(
        !NumericalMetrics::compare(&[0.0, 100.0], &[0.02, 100.0])
            .unwrap()
            .passes()
    );
    assert!(NumericalMetrics::compare(&[0.0], &[0.0]).unwrap().passes());
    assert!(NumericalMetrics::compare(&[0.0], &[0.00001]).is_err());
    for (reference, observed) in [
        (vec![], vec![]),
        (vec![1.0], vec![1.0, 2.0]),
        (vec![f64::NAN], vec![1.0]),
        (vec![1.0], vec![f64::INFINITY]),
        (vec![-f64::MAX], vec![f64::MAX]),
    ] {
        assert!(NumericalMetrics::compare(&reference, &observed).is_err());
    }
    // Norm accumulation must not overflow for merely large finite values.
    assert!(
        NumericalMetrics::compare(&[1e200, -1e200], &[1e200, -1e200])
            .unwrap()
            .passes()
    );
    let spec = spec();
    let mut report = passing(&spec);
    assert!(spec.evaluate(&report).usable);
    report.numerical[0].logits[0] = 0.03;
    assert!(!spec.evaluate(&report).usable);
}

#[test]
fn m3_profile_requires_every_declared_fixture() {
    let spec = spec();
    let original = passing(&spec);
    for i in 0..original.fixtures.len() {
        let mut report = original.clone();
        report.fixtures.remove(i);
        assert!(!spec.evaluate(&report).usable);
        let mut report = original.clone();
        report.fixtures[i].passed = false;
        assert!(!spec.evaluate(&report).usable);
    }
    let mut report = original.clone();
    report.fixtures.push(report.fixtures[0].clone());
    assert!(!spec.evaluate(&report).usable);
    let mut report = original.clone();
    report.fixtures[0].fixture_id = "undeclared".into();
    assert!(!spec.evaluate(&report).usable);
    let mut report = original.clone();
    report.numerical.clear();
    assert!(!spec.evaluate(&report).usable);
    let mut report = original.clone();
    report.numerical.push(report.numerical[0].clone());
    assert!(!spec.evaluate(&report).usable);
    let mut report = original;
    report.fixtures[0].evidence_sha256.clear();
    assert!(!spec.evaluate(&report).usable);
    let mut incomplete = spec.clone();
    incomplete.fixtures.remove("must-pass-0");
    assert!(incomplete.sha256().is_err());
    let mut incomplete = spec;
    incomplete.references.clear();
    assert!(incomplete.sha256().is_err());
}

#[test]
fn m3_profile_identity_changes_invalidate_qualification() {
    let spec = spec();
    let original = passing(&spec);
    let encoded = serde_json::to_value(&original.identity).unwrap();
    for key in encoded.as_object().unwrap().keys() {
        let mut changed = encoded.clone();
        changed[key] = if key == "engine_api_version" {
            serde_json::json!(999)
        } else {
            serde_json::json!("different")
        };
        let mut report = original.clone();
        report.identity = serde_json::from_value(changed).unwrap();
        assert!(
            !spec.evaluate(&report).usable,
            "identity field {key} was ignored"
        );
    }
    let mut changed = spec.clone();
    changed.references.get_mut("fixed-prompt").unwrap().logits[0] = 0.001;
    assert!(!changed.evaluate(&original).usable);
    let mut report = original.clone();
    report.numerical[0].input_sha256 = digest(42);
    assert!(!spec.evaluate(&report).usable);
    let mut report = original;
    report.version += 1;
    assert!(!spec.evaluate(&report).usable);
}

#[test]
fn m3_profile_report_roundtrip_preserves_failures() {
    let spec = spec();
    let mut report = passing(&spec);
    report.fixtures[0].passed = false;
    report.fixtures[0].detail = "tool effect did not occur".into();
    let bytes = serde_json::to_vec(&report).unwrap();
    let decoded: QualificationReport = serde_json::from_slice(&bytes).unwrap();
    let verdict = spec.evaluate(&decoded);
    assert!(!verdict.usable);
    assert!(
        verdict
            .failures
            .iter()
            .any(|s| s.contains("tool effect did not occur"))
    );
    let mut json = serde_json::to_value(report).unwrap();
    json["usable"] = serde_json::json!(true);
    assert!(serde_json::from_value::<QualificationReport>(json).is_err());
}

struct Directory(std::path::PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn registry() -> (Registry, ModelProfileId, QualificationSpec, Directory) {
    static NEXT_DIRECTORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NEXT_DIRECTORY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("alloy-m3-{}-{stamp}-{nonce}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let dir = Directory(path);
    std::fs::write(
        dir.0.join("config.json"),
        br#"{"architectures":["FixtureForCausalLM"],"max_position_embeddings":128}"#,
    )
    .unwrap();
    let mut registry = Registry::new();
    registry
        .register_family(
            FamilyRegistration::new(
                FamilyVersion::new(1, 0, 0),
                vec!["FixtureForCausalLM".into()],
                Capabilities {
                    task: "text-generation".into(),
                    family: "fixture-family".into(),
                    backend: ExecutionBackend::HostCpu,
                    weight_representations: vec![WeightRepresentation::F32],
                    cache_modes: vec![],
                    context_limit_tokens: 128,
                    supports_streaming: true,
                    supports_incremental_kv: false,
                    supports_prefix_reuse: false,
                    decoding_modes: vec![DecodingCapability::Greedy],
                },
            )
            .unwrap(),
        )
        .unwrap();
    let mut spec = spec();
    let artifact = ArtifactId::new(&spec.artifact_id).unwrap();
    let id = ModelProfileId::new(&spec.profile_id).unwrap();
    registry
        .register_artifact(artifact.clone(), &dir.0)
        .unwrap();
    let profile = ModelProfile::new(
        id.clone(),
        artifact,
        128,
        8,
        8,
        GenerationOptions::greedy(4).unwrap(),
    )
    .unwrap();
    spec.identity.profile_sha256 = profile.qualification_sha256();
    registry.register_profile(profile).unwrap();
    (registry, id, spec, dir)
}

#[test]
fn m3_profile_registry_rejects_missing_or_stale_qualification() {
    let (mut registry, id, spec, _dir) = registry();
    let report = passing(&spec);
    registry.require_qualification(spec.clone()).unwrap();
    assert!(registry.resolve_profile(&id).is_err());
    assert!(
        registry
            .validate_generation_options(&id, &GenerationOptions::greedy(4).unwrap())
            .is_err()
    );
    assert!(registry.record_qualification(&id, &report).unwrap().usable);
    assert!(registry.resolve_profile(&id).is_ok());
    let mut updated = spec;
    updated.identity.dependencies_sha256 = digest(55);
    registry.require_qualification(updated).unwrap();
    assert!(registry.resolve_profile(&id).is_err());
    assert!(!registry.record_qualification(&id, &report).unwrap().usable);
    assert!(registry.resolve_profile(&id).is_err());
}

#[test]
fn m3_profile_failure_revokes_prior_qualification() {
    let (mut registry, id, spec, _dir) = registry();
    let mut report = passing(&spec);
    registry.require_qualification(spec).unwrap();
    assert!(registry.record_qualification(&id, &report).unwrap().usable);
    report.fixtures[0].passed = false;
    assert!(!registry.record_qualification(&id, &report).unwrap().usable);
    assert!(!registry.qualification_verdict(&id).unwrap().usable);
    assert!(registry.resolve_profile(&id).is_err());
}

#[test]
fn m3_profile_registry_checks_actual_profile_settings() {
    let (mut registry, id, mut spec, _dir) = registry();
    spec.identity.profile_sha256 = digest(99);
    assert!(registry.require_qualification(spec.clone()).is_err());
    spec.identity.profile_sha256 = registry.profile(&id).unwrap().qualification_sha256();
    spec.identity.family_version = "2.0.0".into();
    assert!(registry.require_qualification(spec).is_err());
}
