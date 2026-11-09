use super::*;
use crate::{memory_guard::CeilingBinding, residency_pin::ModelPinOwner};

#[test]
fn retained_protection_diagnostics_do_not_retain_pin_or_address_replacement() {
    let original = ModelInstanceId::new();
    let mut owner = ModelPinOwner::new(original.clone());
    let pin = owner.try_pin(&original, false).expect("pin original");
    let error = EngineError::ProtectedResources {
        operation: "generation_startup",
        resources: vec![ProtectedResource::ModelPin {
            artifact_id: ArtifactId::new("model-a").expect("artifact"),
            instance_id: original.clone(),
            handles: NonZeroUsize::new(owner.protection_observer().strong_count())
                .expect("observed positive count"),
        }],
    };
    drop(pin);
    assert!(!owner.is_pinned(), "diagnostics must not extend protection");
    let replacement = ModelInstanceId::new();
    assert!(!original.same_instance(&replacement));
    assert_ne!(original.diagnostic_id(), replacement.diagnostic_id());
    assert_eq!(original.diagnostic_id(), original.clone().diagnostic_id());
    let report = error.diagnostic();
    assert_eq!(error.kind(), ErrorKind::ProtectedResources);
    assert_eq!(report["kind"], "protected_resources");
    assert_eq!(report["resources"][0]["protection"], "model_pin");
    assert_eq!(report["resources"][0]["handles"], 1);
    assert_eq!(
        report["resources"][0]["instance_id"],
        format!("model-{}", original.diagnostic_id())
    );
    assert!(report.get("active_request_id").is_none());
}

#[test]
fn busy_and_leased_publication_have_distinct_machine_readable_owners() {
    let busy = EngineError::Busy {
        active_request_id: Some("request-a".into()),
        message: "native startup owns the slot".into(),
    }
    .diagnostic();
    assert_eq!(busy["kind"], "busy");
    assert_eq!(busy["active_request_id"], "request-a");
    assert!(busy.get("resources").is_none());

    let state_id = SessionStateId::from_digest([9; 32]);
    let leased = EngineError::ProtectedResources {
        operation: "release_state",
        resources: vec![ProtectedResource::SessionStateLease {
            artifact_id: ArtifactId::new("model-a").expect("artifact"),
            instance_id: ModelInstanceId::new(),
            state_id,
            handles: NonZeroUsize::new(2).expect("leases"),
        }],
    }
    .diagnostic();
    assert_eq!(leased["kind"], "protected_resources");
    assert_eq!(leased["resources"][0]["state_id"], state_id.to_string());
    assert_eq!(leased["resources"][0]["protection"], "session_state_lease");
    assert_eq!(leased["resources"][0]["handles"], 2);
}

#[test]
fn host_preflight_records_plan_and_full_observed_ceiling() {
    let error = EngineError::MemoryLimitExceeded {
        reason: MemoryLimitReason::HostPreflight {
            artifact_id: ArtifactId::new("large-model").expect("artifact"),
            ceiling: CeilingBreakdown {
                reserve_bytes: 10,
                static_bytes: 90,
                dynamic_bytes: 40,
                metal_cap_bytes: 70,
                hard_limit_bytes: 40,
                binding: Some(CeilingBinding::Dynamic),
            },
        },
        operation: "startup preflight",
        required_bytes: 60,
        available_bytes: 40,
    };
    let report = error.diagnostic();
    assert_eq!(report["kind"], "memory_limit_exceeded");
    assert_eq!(report["reason"], "host_preflight");
    assert_eq!(report["required_bytes"], 60);
    assert_eq!(report["available_bytes"], 40);
    assert_eq!(report["ceiling"]["binding"], "dynamic");
    assert_eq!(report["ceiling"]["metal_cap_bytes"], 70);
    assert!(error.to_string().contains("planned 60 bytes"));
    assert!(error.to_string().contains("native loading did not start"));
}

#[test]
fn paused_admission_is_not_misreported_as_insufficient_available_bytes() {
    let error = EngineError::MemoryLimitExceeded {
        reason: MemoryLimitReason::AdmissionsPaused,
        operation: "request startup",
        required_bytes: 10,
        available_bytes: 100,
    };
    let report = error.diagnostic();
    assert_eq!(report["reason"], "admissions_paused");
    assert_eq!(report["available_bytes"], 100);
    assert!(
        error
            .to_string()
            .starts_with("admissions paused by memory pressure")
    );
    assert!(report.get("budget_bytes").is_none());
}
