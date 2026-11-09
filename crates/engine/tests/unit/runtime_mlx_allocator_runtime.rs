use super::*;

#[test]
fn allocator_observation_types_can_cross_dispatch_boundaries() {
    fn assert_send<T: Send>() {}
    assert_send::<mlx_ffi::memory::MemorySnapshot>();
    assert_send::<mlx_ffi::memory::CacheReclaimReport>();
}

#[test]
fn pending_startup_rejects_reclamation_before_allocator_access() {
    let admission = AdmissionController::new(4096).expect("admission");
    let _unrelated = admission
        .reserve("unrelated owner", 512)
        .expect("reservation");
    let pending = PendingStartup {
        request_id: RequestId::new("pending-allocator-reclaim").expect("request"),
        parser: None,
        parser_limits: None,
        grant: admission
            .reserve_grant("test pending startup", &[1])
            .expect("grant"),
    };
    let before = admission.snapshot().expect("before");

    let state = NativeState {
        active: None,
        pending: Some(pending),
        resident: None,
        states: HashMap::new(),
        pool: Vec::new(),
        pool_enabled: false,
        pool_ttl: None,
        tiered: None,
        admission: admission.clone(),
        retention: RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: true,
        },
        summary: Arc::new(Mutex::new(Summary::empty(true))),
    };

    assert!(matches!(
        state.reclaim_mlx_allocator_cache(),
        Err(EngineError::Busy { .. }),
    ));
    assert_eq!(admission.snapshot().expect("after"), before);
}
