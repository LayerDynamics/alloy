use super::*;
fn empty_state(admission: AdmissionController) -> NativeState {
    NativeState {
        active: None,
        pending: None,
        resident: None,
        states: HashMap::new(),
        pool: Vec::new(),
        pool_enabled: false,
        pool_ttl: None,
        tiered: None,
        admission,
        retention: RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: true,
        },
        summary: Arc::new(Mutex::new(Summary::empty(true))),
    }
}
#[test]
fn pressure_actions_do_not_credit_unrelated_reservation_guards() {
    let admission = AdmissionController::new(4096).expect("admission");
    let _unrelated = admission.reserve("unrelated owner", 512).expect("guard");
    let before = admission.snapshot().expect("before");
    let mut state = empty_state(admission.clone());
    state
        .pressure_release_idle_state()
        .expect("empty state release");
    state
        .pressure_unload_idle_model()
        .expect("empty model unload");
    assert_eq!(admission.snapshot().expect("after"), before);
}
#[test]
fn pending_native_ownership_blocks_destructive_pressure_actions() {
    let admission = AdmissionController::new(4096).expect("admission");
    let mut state = empty_state(admission.clone());
    state.pending = Some(PendingStartup {
        request_id: RequestId::new("pending-pressure").expect("ID"),
        parser: None,
        parser_limits: None,
        grant: admission
            .reserve_grant("test pending startup", &[1])
            .expect("grant"),
    });
    assert_eq!(admission.snapshot().expect("before").reserved_bytes(), 1);
    assert!(matches!(
        state.pressure_release_idle_state(),
        Err(EngineError::Busy { .. })
    ));
    assert!(matches!(
        state.pressure_unload_idle_model(),
        Err(EngineError::Busy { .. })
    ));
    assert_eq!(admission.snapshot().expect("after").reserved_bytes(), 1);
}
