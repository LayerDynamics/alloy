use super::*;

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}

fn empty_native_state(retention: RetentionPolicy) -> NativeState {
    NativeState {
        active: None,
        pending: None,
        resident: None,
        states: HashMap::new(),
        pool: Vec::new(),
        pool_enabled: false,
        pool_ttl: None,
        tiered: None,
        admission: AdmissionController::new(4096).expect("controller"),
        retention,
        summary: Arc::new(Mutex::new(Summary::empty(true))),
    }
}

#[test]
fn control_and_generation_proxy_can_cross_task_boundaries() {
    assert_send_sync::<FamilyRuntime>();
    assert_send::<NativeSession>();
    assert_send::<Command>();
    assert_send::<AdmissionFuture>();
    assert_send::<AdmittedStartup>();
    assert_send::<OperationFuture<Option<SessionStateLease>>>();
    assert_send_sync::<SessionStateLease>();
    assert_send_sync::<RetainedSessionState>();
}

#[test]
fn exact_extension_reuses_complete_retained_prefix() {
    assert_eq!(exact_reusable_prefix(&[3, 41], &[3, 41, 7, 12], true), 2,);
}

#[test]
fn exact_extension_does_not_require_old_logits() {
    assert_eq!(exact_reusable_prefix(&[3, 41], &[3, 41, 7], false), 2,);
}

#[test]
fn identical_input_requires_actual_retained_logits() {
    assert_eq!(exact_reusable_prefix(&[3, 41], &[3, 41], true), 2,);
    assert_eq!(exact_reusable_prefix(&[3, 41], &[3, 41], false), 0,);
}

#[test]
fn shortened_or_divergent_context_rebuilds() {
    assert_eq!(exact_reusable_prefix(&[3, 41, 7], &[3, 41], true), 0,);
    assert_eq!(exact_reusable_prefix(&[3, 41, 7], &[3, 12, 7, 9], true), 0,);
}

#[test]
fn matching_length_is_not_prefix_evidence() {
    assert_eq!(exact_reusable_prefix(&[3, 41], &[3, 12], true), 0,);
}

#[test]
fn empty_history_never_reports_a_hit() {
    assert_eq!(exact_reusable_prefix(&[], &[3, 41], true), 0);
    assert_eq!(exact_reusable_prefix(&[], &[], true), 0);
}

#[test]
fn retained_logits_have_independent_lifetime_accounting() {
    let admission = AdmissionController::new(4096).expect("controller");
    let logits = TokenLogits::new(vec![0.0, 1.0, -1.0]).expect("actual logits");
    let retained = RetainedLogits::new(logits, &admission).expect("retained");

    assert!(
        admission.snapshot().expect("snapshot").reserved_bytes()
            >= 3 * std::mem::size_of::<f32>() as u64
    );

    drop(retained);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}

#[test]
fn retained_logits_rejection_does_not_leak_a_reservation() {
    let admission = AdmissionController::new(4).expect("controller");
    let logits = TokenLogits::new(vec![0.0, 1.0, -1.0]).expect("logits");

    assert!(RetainedLogits::new(logits, &admission).is_err());
    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 0);
}

#[test]
fn idle_reservations_are_reclaimed_before_output_admission() {
    const MIB: u64 = 1024 * 1024;
    let admission = AdmissionController::new(100 * MIB).expect("controller");
    let idle = admission.reserve("idle resources", 90 * MIB).expect("idle");

    assert!(
        admission
            .reserve("output before reclamation", 20 * MIB)
            .is_err()
    );

    let (output, grant) = admit_startup_after_reclaim(&admission, 20 * MIB, || {
        drop(idle);
        Ok(vec![20 * MIB])
    })
    .expect("output after reclamation");

    assert!(grant.is_empty());
    assert_eq!(
        admission.snapshot().expect("snapshot").reserved_bytes(),
        20 * MIB,
    );
    drop(output);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}

#[test]
fn cache_reclamation_preserves_model_reservation() {
    let admission = AdmissionController::new(100).expect("controller");
    let model = admission.reserve("model", 60).expect("model");
    let cache = admission.reserve("cache", 30).expect("cache");

    let (output, grant) = admit_startup_after_reclaim(&admission, 20, || {
        drop(cache);
        Ok(vec![20])
    })
    .expect("output");

    assert!(grant.is_empty());
    assert_eq!(model.bytes(), 60);
    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 80);
    drop(output);
    assert_eq!(admission.snapshot().expect("model").reserved_bytes(), 60);
    drop(model);
}

#[test]
fn unrelated_reservations_are_not_reclaimable() {
    let admission = AdmissionController::new(100).expect("controller");
    let unrelated = admission.reserve("unrelated", 85).expect("unrelated");
    let idle = admission.reserve("idle", 10).expect("idle");

    let result = admit_startup_after_reclaim(&admission, 20, || {
        drop(idle);
        Ok(vec![20])
    });

    assert!(matches!(
        result,
        Err(EngineError::MemoryLimitExceeded {
            required_bytes: 20,
            available_bytes: 15,
            ..
        })
    ));

    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 85);
    drop(unrelated);
}

#[test]
fn reclamation_failure_does_not_acquire_output_budget() {
    let admission = AdmissionController::new(100).expect("controller");
    let _idle = admission.reserve("idle", 40).expect("idle");
    let before = admission.snapshot().expect("before");

    assert!(
        admit_startup_after_reclaim(&admission, 20, || { Err(internal("reclamation failed")) })
            .is_err()
    );
    assert_eq!(admission.snapshot().expect("after"), before);
}

#[test]
fn startup_grant_partitions_output_without_double_counting() {
    let admission = AdmissionController::new(200).expect("controller");
    // output 20 + workspace 30 + peak 80 + cache 40 + history 10
    let components = vec![20, 30, 80, 40, 10];

    let (output, mut grant) =
        admit_startup_after_reclaim(&admission, 20, || Ok(components.clone())).expect("grant");

    assert_eq!(output.bytes(), 20);
    assert_eq!(grant.remaining_bytes(), 160);
    assert_eq!(
        admission
            .snapshot()
            .expect("after output take")
            .reserved_bytes(),
        180
    );

    let workspace = grant
        .take("generation workspaces", 30)
        .expect("workspace")
        .expect("nonzero");
    let peak = grant
        .take("model loading peak", 80)
        .expect("peak")
        .expect("nonzero");
    assert_eq!(grant.remaining_bytes(), 50);
    assert_eq!(
        admission.snapshot().expect("after takes").reserved_bytes(),
        180
    );

    drop(workspace);
    assert_eq!(
        admission
            .snapshot()
            .expect("workspace dropped")
            .reserved_bytes(),
        150
    );

    grant.release().expect("release unused cache+history");
    assert_eq!(
        admission
            .snapshot()
            .expect("remainder released")
            .reserved_bytes(),
        100
    );

    drop(peak);
    drop(output);
    assert_eq!(admission.snapshot().expect("empty").reserved_bytes(), 0);
}

#[test]
fn startup_grant_rejects_when_incremental_total_does_not_fit() {
    let admission = AdmissionController::new(100).expect("controller");
    let _blocking = admission.reserve("blocking", 50).expect("blocking");

    let error = admit_startup_after_reclaim(&admission, 20, || Ok(vec![20, 20, 40, 30, 0]))
        .expect_err("110 does not fit in 50 available");

    assert!(matches!(
        error,
        EngineError::MemoryLimitExceeded {
            required_bytes: 110,
            available_bytes: 50,
            ..
        }
    ));
    assert_eq!(
        admission.snapshot().expect("unchanged").reserved_bytes(),
        50
    );
}

fn test_pending(admission: &AdmissionController, id: &str) -> PendingStartup {
    PendingStartup {
        request_id: RequestId::new(id).expect("request"),
        parser: None,
        parser_limits: None,
        grant: admission
            .reserve_grant("test pending startup", &[1])
            .expect("grant"),
    }
}

#[test]
fn pending_startup_can_finish_without_prefill() {
    let mut state = empty_native_state(RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: true,
    });
    let pending = test_pending(&state.admission, "pending-startup");
    let request = pending.request_id.clone();
    assert_eq!(
        state
            .admission
            .snapshot()
            .expect("granted")
            .reserved_bytes(),
        1
    );
    state.pending = Some(pending);
    state.publish();

    assert!(lock_metadata(&state.summary).active);
    state.finish(&request).expect("finish");
    state.publish();

    assert!(state.pending.is_none());
    assert!(!lock_metadata(&state.summary).active);
    assert_eq!(
        state
            .admission
            .snapshot()
            .expect("released")
            .reserved_bytes(),
        0
    );
}

#[test]
fn absent_state_operations_are_honest() {
    let mut state = empty_native_state(RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: true,
    });
    let id = SessionStateId::from_digest([7; 32]);

    assert!(state.lookup_state(&id).expect("lookup").is_none());
    assert!(state.lease_state(&id).expect("lease").is_none());
    assert_eq!(
        state.release_state(&id).expect("release"),
        ReleaseOutcome::AlreadyAbsent,
    );
}

#[test]
fn publication_requires_actual_completed_state() {
    let mut state = empty_native_state(RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: true,
    });
    let request = RequestId::new("missing-state").expect("request");

    assert!(matches!(
        state.retain_state(&request, b"tokenizer-prefix"),
        Err(EngineError::NotFound { .. })
    ));
}

#[test]
fn publication_cannot_bypass_pending_ownership() {
    let mut state = empty_native_state(RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: true,
    });
    let pending = test_pending(&state.admission, "pending-publication");
    let request = pending.request_id.clone();
    state.pending = Some(pending);

    assert!(matches!(
        state.retain_state(&request, b"prefix"),
        Err(EngineError::Busy { .. })
    ));
}

#[test]
fn publication_requires_both_retention_flags() {
    let mut state = empty_native_state(RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: false,
    });
    let request = RequestId::new("no-cache-retention").expect("request");

    assert!(state.retain_state(&request, b"prefix").is_err());
}

#[test]
fn token_history_is_accounted_and_bounded() {
    let admission = AdmissionController::new(1024).expect("controller");
    let mut history = TokenHistory::new(&admission, 4).expect("history");

    history.append(&[1, 2, 3, 4]).expect("append");
    assert_eq!(history.tokens, [1, 2, 3, 4]);

    let remaining = history.tokens.capacity() - history.tokens.len();
    history.append(&vec![0; remaining]).expect("fill capacity");
    assert!(history.append(&[5]).is_err());

    drop(history);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}

#[test]
fn weak_pin_observes_lease_lifetime_without_worker_commands() {
    let pin = Arc::new(());
    let weak = Arc::downgrade(&pin);
    let second = Arc::clone(&pin);

    drop(pin);
    assert_eq!(weak.strong_count(), 1);
    drop(second);
    assert_eq!(weak.strong_count(), 0);
    assert!(weak.upgrade().is_none());
}

struct SignalWake {
    sender: mpsc::Sender<()>,
}

impl std::task::Wake for SignalWake {
    fn wake(self: Arc<Self>) {
        let _ = self.sender.send(());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.sender.send(());
    }
}

#[test]
fn admission_completion_wakes_consumer() {
    let (mut future, shared) = admission_channel();
    let (sender, receiver) = mpsc::channel();
    let waker = Waker::from(Arc::new(SignalWake { sender }));
    let mut context = Context::from_waker(&waker);

    assert!(Pin::new(&mut future).poll(&mut context).is_pending());

    let worker = thread::spawn(move || {
        complete_admission(shared, Err(internal("startup rejected")));
    });

    receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("completion wake");

    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Err(_))
    ));
    drop(future);
    worker.join().expect("completion acknowledged");
}

#[test]
fn abandoned_admission_does_not_wait_for_consumer() {
    let (future, shared) = admission_channel();
    drop(future);

    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        complete_admission(shared, Err(internal("abandoned startup")));
        sender.send(()).expect("signal");
    });

    receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("completion");
    worker.join().expect("cleanup");
}

#[test]
fn state_operation_wakes_consumer() {
    let (start_sender, start_receiver) = mpsc::channel();
    let mut future = dispatch("test-state-operation", move || {
        start_receiver.recv().expect("start");
        Ok(42_u32)
    })
    .expect("dispatcher");

    let (sender, receiver) = mpsc::channel();
    let waker = Waker::from(Arc::new(SignalWake { sender }));
    let mut context = Context::from_waker(&waker);

    assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    start_sender.send(()).expect("release");

    receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("wake");

    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Ok(42))
    ));
}

#[test]
fn abandoned_operation_drops_owned_reply() {
    struct DropSignal(mpsc::Sender<()>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    let (start_sender, start_receiver) = mpsc::channel();
    let (drop_sender, drop_receiver) = mpsc::channel();

    let future = dispatch("test-abandoned-operation", move || {
        start_receiver.recv().expect("start");
        Ok(DropSignal(drop_sender))
    })
    .expect("dispatcher");

    drop(future);
    start_sender.send(()).expect("release");

    drop_receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("reply destruction");
}
