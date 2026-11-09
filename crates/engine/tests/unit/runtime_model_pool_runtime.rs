use super::*;
use crate::family::BufferMemoryUsage;
use crate::types::{ExecutionBackend, WeightRepresentation};

// Ownership-only fixture: not an executable model or checkpoint test.
struct OwnedFixture {
    instance: ModelInstanceId,
    artifact: ArtifactId,
    payload: Vec<u8>,
    admission: AdmissionController,
    drops: Arc<Mutex<Vec<(String, u64)>>>,
}
impl Drop for OwnedFixture {
    fn drop(&mut self) {
        self.drops.lock().expect("drops").push((
            self.artifact.as_str().to_owned(),
            self.admission.snapshot().expect("ledger").reserved_bytes(),
        ));
    }
}
impl LoadedModel for OwnedFixture {
    fn instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }
    fn artifact_id(&self) -> &ArtifactId {
        &self.artifact
    }
    fn registration(&self) -> &crate::registry::FamilyRegistration {
        panic!("ownership fixture does not implement execution registration")
    }
    fn capabilities(&self) -> &Capabilities {
        panic!("ownership fixture does not implement execution")
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        BufferMemoryUsage::new(self.payload.len() as u64, self.payload.len() as u64)
    }
    fn autoregressive(&mut self) -> Option<&mut dyn crate::family::AutoregressiveModel> {
        None
    }
}
fn capabilities() -> Capabilities {
    Capabilities {
        task: "ownership-test".to_owned(),
        family: "fixture".to_owned(),
        backend: ExecutionBackend::HostCpu,
        weight_representations: vec![WeightRepresentation::F32],
        cache_modes: vec![],
        context_limit_tokens: 1,
        supports_streaming: false,
        supports_incremental_kv: false,
        supports_prefix_reuse: false,
        decoding_modes: vec![],
    }
}
fn state() -> NativeState {
    NativeState {
        active: None,
        pending: None,
        resident: None,
        states: HashMap::new(),
        pool: Vec::new(),
        pool_enabled: true,
        pool_ttl: Some(Duration::from_secs(10)),
        tiered: None,
        admission: AdmissionController::new(256).expect("ledger"),
        retention: RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: false,
        },
        summary: Arc::new(Mutex::new(Summary::empty(true))),
    }
}
fn resident(
    state: &NativeState,
    name: &str,
    now: Instant,
    drops: &Arc<Mutex<Vec<(String, u64)>>>,
) -> Resident {
    let instance = ModelInstanceId::new();
    let weights = state
        .admission
        .reserve("fixture weights", 64)
        .expect("weights");
    Resident {
        pins: ModelPinOwner::new(instance.clone()),
        idle_since: now,
        idle_ttl: Some(Duration::from_secs(10)),
        cache: None,
        model: Box::new(OwnedFixture {
            instance,
            artifact: ArtifactId::new(name).expect("artifact"),
            payload: vec![0; 64],
            admission: state.admission.clone(),
            drops: drops.clone(),
        }),
        weights,
        capabilities: capabilities(),
        candidate: None,
    }
}
#[test]
fn two_owned_models_keep_distinct_instances_and_committed_guards() {
    let mut state = state();
    let drops = Arc::new(Mutex::new(Vec::new()));
    let now = Instant::now();
    let a = resident(&state, "a", now, &drops);
    let b = resident(&state, "b", now, &drops);
    assert!(!a.model.instance_id().same_instance(b.model.instance_id()));
    state.pool.extend([a, b]);
    assert_eq!(state.residency_info().len(), 2);
    assert_eq!(state.admission.snapshot().unwrap().reserved_bytes(), 128);
    assert!(drops.lock().unwrap().is_empty());
}
#[test]
fn lru_destroys_native_model_before_releasing_its_guard() {
    let mut state = state();
    let drops = Arc::new(Mutex::new(Vec::new()));
    let now = Instant::now();
    let a = resident(&state, "a", now, &drops);
    let b = resident(&state, "b", now + Duration::from_secs(1), &drops);
    state.pool.extend([b, a]);
    assert!(state.evict_one_pooled_model());
    assert_eq!(drops.lock().unwrap().as_slice(), &[("a".to_owned(), 128)]);
    assert_eq!(state.admission.snapshot().unwrap().reserved_bytes(), 64);
    assert_eq!(state.pool[0].model.artifact_id().as_str(), "b");
}
#[test]
fn expired_pinned_model_survives_until_last_handle_drops() {
    let mut state = state();
    let drops = Arc::new(Mutex::new(Vec::new()));
    let now = Instant::now();
    let a = resident(&state, "a", now, &drops);
    let instance = a.model.instance_id().clone();
    state.pool.push(a);
    let pin = state.pin_instance(&instance).unwrap().unwrap();
    let clone = pin.clone();
    state.expire_models_at(now + Duration::from_secs(10));
    assert_eq!(state.pool.len(), 1);
    drop(pin);
    assert!(!state.evict_one_pooled_model());
    drop(clone);
    state.expire_models_at(now + Duration::from_secs(10));
    assert!(state.pool.is_empty());
    assert_eq!(state.admission.snapshot().unwrap().reserved_bytes(), 0);
    assert!(state.pin_instance(&instance).unwrap().is_none());
}
#[test]
fn ttl_boundary_uses_injected_time_without_sleeping() {
    let mut state = state();
    let drops = Arc::new(Mutex::new(Vec::new()));
    let now = Instant::now();
    let a = resident(&state, "a", now, &drops);
    state.pool.push(a);
    state.expire_models_at(now + Duration::from_secs(9));
    assert_eq!(state.pool.len(), 1);
    state.expire_models_at(now + Duration::from_secs(10));
    assert!(state.pool.is_empty());
}
#[test]
fn all_pinned_models_prevent_reclamation_and_do_not_credit_bytes() {
    let mut state = state();
    let drops = Arc::new(Mutex::new(Vec::new()));
    let a = resident(&state, "a", Instant::now(), &drops);
    let instance = a.model.instance_id().clone();
    state.pool.push(a);
    let _pin = state.pin_instance(&instance).unwrap().unwrap();
    let before = state.admission.snapshot().unwrap();
    assert!(!state.evict_one_pooled_model());
    assert_eq!(state.admission.snapshot().unwrap(), before);
    assert!(drops.lock().unwrap().is_empty());
}
