//! Emergency interruption through the production pressure pump and scheduler.
//!
//! The native gates pause the real Qwen backend. Observations are controlled;
//! model execution, reservations, terminal arbitration, and cleanup are real.

use super::*;
use engine::{
    memory_enforcer::{EnforcerError, MemoryEnforcer, MemoryObservation, MemoryObservationSource},
    memory_guard::{MemoryGuardConfig, PressureLevel},
    runtime::MemoryEnforcementPump,
    scheduler::RequestState,
};

const GIB: u64 = 1024 * 1024 * 1024;

fn set_usage(source: &PoolPressureSource, usage: u64) {
    let mut observation = source.0.lock().unwrap();
    observation.host.free_bytes = 5 * GIB;
    observation.host.phys_footprint_bytes = usage;
    observation.mlx_active_bytes = usage;
    observation.mlx_pool_bytes = 3 * GIB;
}

fn assert_reservations(environment: &PolicyEnvironment, expected: &AdmissionSnapshot) {
    let actual = environment.snapshot();
    // The live ceiling may resize the budget; ownership charges must not change.
    assert_eq!(actual.reserved_bytes(), expected.reserved_bytes());
    assert_eq!(actual.active_reservations(), expected.active_reservations());
}

fn assert_no_terminal_available(stream: &mut GenerationStream, cx: &mut Context<'_>) {
    loop {
        match stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(GenerationEvent::RequestFailed { .. }))
            | Poll::Ready(Some(GenerationEvent::GenerationCompleted { .. })) => {
                panic!("cancellation cannot finish while native execution is gated")
            }
            Poll::Ready(None) => panic!("stream closed before native cleanup"),
            Poll::Ready(Some(_)) => {}
            Poll::Pending => break,
        }
    }
}

fn assert_one_cancelled_terminal(mut stream: GenerationStream, request_id: &RequestId) {
    let mut failures = 0;
    let limit = deadline();
    while let Some(event) = next_event(&mut stream, limit) {
        match event {
            GenerationEvent::RequestFailed { error, .. } => {
                match error {
                    EngineError::Cancelled { request_id: actual } => {
                        assert_eq!(actual, request_id.as_str());
                    }
                    other => panic!("expected cancellation, got {other:?}"),
                }
                failures += 1;
            }
            GenerationEvent::GenerationCompleted { .. } => {
                panic!("emergency-cancelled request completed successfully")
            }
            _ => {}
        }
    }
    assert_eq!(
        failures, 1,
        "one request must have exactly one terminal outcome"
    );
}

fn finish_pending_action(pump: &mut MemoryEnforcementPump, cx: &mut Context<'_>) {
    let limit = deadline();
    while pump.has_pending() {
        if let Some(result) = pump.poll_tick(cx).unwrap().completed {
            result.expect("native pressure action completes after execution returns");
        }
        assert!(
            Instant::now() < limit,
            "native pressure action never completed"
        );
        thread::yield_now();
    }
}

#[derive(Debug)]
struct FalliblePressureSource {
    source: Arc<PoolPressureSource>,
    fail: AtomicBool,
}

impl MemoryObservationSource for FalliblePressureSource {
    fn observe(&self) -> Result<MemoryObservation, EnforcerError> {
        if self.fail.load(Ordering::Acquire) {
            return Err(EnforcerError::HostProbe(
                "injected unavailable observation".to_owned(),
            ));
        }
        self.source.observe()
    }
}

#[test]
fn active_cancellation_bypasses_blocked_native_queue_and_preserves_leased_publication() {
    let environment = PolicyEnvironment::new_pool(false);
    let (a_id, a) = instance_lease_publish(&environment, "emergency-publish-a", profile_a());
    let lease = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let _ = environment
        .runtime
        .clone()
        .with_model_pool(Some(Duration::ZERO))
        .unwrap();
    let baseline = environment.snapshot();
    let checkpoint = environment.control.events().len();

    let name = "emergency-active-b";
    let request_id = RequestId::new(name).unwrap();
    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        name,
        profile_b(),
        "hello",
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let native_control = environment.control.execution_control(&request_id);
    assert!(!native_control.is_cancelled());
    let blocked = environment.snapshot();
    let mut stream = generation.into_events();

    let (_, source) = pool_pressure_enforcer(environment.admission.clone(), true);
    source.set_normal();
    let observations = Arc::new(FalliblePressureSource {
        source: Arc::clone(&source),
        fail: AtomicBool::new(false),
    });
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::balanced(),
        environment.admission.clone(),
        observations.clone(),
    );
    let mut pump = MemoryEnforcementPump::new(environment.engine.clone(), enforcer);
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);

    // Occupy the one native enforcement slot before the emergency exists.
    let initial = pump.poll_tick(&mut cx).unwrap();
    assert_eq!(initial.tick.status.pressure, PressureLevel::Ok);
    assert!(initial.tick.request_execution_cancellation.is_none());
    assert!(initial.cancellation.is_none());
    assert!(initial.completed.is_none());
    assert!(pump.has_pending());

    // The 33 GiB abort ceiling plus its 2 GiB margin triggers immediately,
    // despite the large allocator pool and the already-queued native action.
    set_usage(&source, 35 * GIB);
    for _ in 0..3 {
        let poll = pump.poll_tick(&mut cx).unwrap();
        assert!(poll.tick.status.emergency);
        assert!(!poll.tick.deferred_for_pool_drain);
        assert_eq!(
            poll.tick
                .request_execution_cancellation
                .as_ref()
                .map(|target| target.request_id()),
            Some(&request_id),
        );
        assert_eq!(
            poll.cancellation.unwrap().unwrap(),
            CancelOutcome::Requested
        );
        assert!(poll.completed.is_none());
        assert!(pump.has_pending());
        assert!(
            native_control.is_cancelled(),
            "native control must receive the out-of-band signal"
        );
        assert_eq!(
            environment.scheduler.request_state(&request_id).unwrap(),
            RequestState::CancellationRequested
        );
        assert!(environment.scheduler.is_busy().unwrap());
        assert_eq!(
            environment.engine.active_request_id().unwrap(),
            Some(request_id.clone())
        );
        assert!(environment.admission.admissions_paused());
        assert_reservations(&environment, &blocked);
        assert_no_terminal_available(&mut stream, &mut cx);
        assert!(
            environment.control.events()[checkpoint..]
                .iter()
                .all(|event| { !matches!(event.kind, "cache-destroyed" | "model-destroyed") })
        );
    }

    release.release();
    assert_one_cancelled_terminal(stream, &request_id);
    environment.wait_idle();
    finish_pending_action(&mut pump, &mut cx);
    assert_eq!(
        environment.scheduler.request_state(&request_id).unwrap(),
        RequestState::Finished
    );
    let residents = block_on(environment.runtime.model_residency()).unwrap();
    assert!(
        !residents
            .iter()
            .any(|resident| resident.artifact == artifact_b())
    );
    instance_lease_assert_unchanged(&environment, &a);
    assert_reservations(&environment, &baseline);
    let events = environment.control.events();
    let events = &events[checkpoint..];
    assert_b_destruction_after_prefill_return(events);
    event_index(events, "model-destroyed", &artifact_b());
    assert!(!events.iter().any(|event| event.artifact == artifact_a()
        && matches!(event.kind, "cache-destroyed" | "model-destroyed")));

    // Cleanup and a failed probe cannot reopen admission. Merely changing
    // the provider to normal also does nothing until a fresh probe succeeds.
    assert!(environment.admission.admissions_paused());
    observations.fail.store(true, Ordering::Release);
    assert!(pump.poll_tick(&mut cx).is_err());
    assert!(!pump.has_pending());
    assert!(environment.admission.admissions_paused());
    source.set_normal();
    assert!(environment.admission.admissions_paused());
    assert!(pump.poll_tick(&mut cx).is_err());
    assert!(environment.admission.admissions_paused());
    assert!(
        environment
            .admission
            .reserve("admission remains paused after emergency cleanup", 1)
            .is_err()
    );
    observations.fail.store(false, Ordering::Release);
    let recovery = pump.poll_tick(&mut cx).unwrap();
    assert_eq!(recovery.tick.status.pressure, PressureLevel::Ok);
    assert!(!recovery.tick.status.emergency);
    assert!(recovery.tick.request_execution_cancellation.is_none());
    assert!(recovery.cancellation.is_none());
    assert!(!environment.admission.admissions_paused());
    drop(
        environment
            .admission
            .reserve("fresh normal pressure permits admission", 1)
            .unwrap(),
    );
    finish_pending_action(&mut pump, &mut cx);

    drop(lease);
    block_on(environment.engine.release_session_state(a_id)).unwrap();
    environment.assert_empty();
}

fn check_pending_startup(gate_stage: usize) {
    let environment = PolicyEnvironment::new_pool(false);
    let (a_id, a) =
        instance_lease_publish(&environment, "emergency-pending-publish-a", profile_a());
    let lease = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let _ = environment
        .runtime
        .clone()
        .with_model_pool(Some(Duration::ZERO))
        .unwrap();
    let baseline = environment.snapshot();
    let checkpoint = environment.control.events().len();
    environment
        .control
        .startup_gate_stage
        .store(gate_stage, Ordering::Release);
    let release = environment.control.startup_gate.arm();
    let name = if gate_stage == 1 {
        "emergency-pending-before-load"
    } else {
        "emergency-pending-after-load"
    };
    let request_id = RequestId::new(name).unwrap();
    let generation = block_on(environment.engine.generate(environment.request(
        name,
        profile_b(),
        "hello",
    )))
    .unwrap();
    environment.control.startup_gate.wait_entered();
    let native_control = environment.control.execution_control(&request_id);
    let blocked = environment.snapshot();
    assert_eq!(
        blocked.reserved_bytes(),
        baseline.reserved_bytes() + pending_startup_reservation(&environment.plan_b)
    );
    let mut stream = generation.into_events();

    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), true);
    source.set_normal();
    let mut pump = MemoryEnforcementPump::new(environment.engine.clone(), enforcer);
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    assert!(pump.poll_tick(&mut cx).unwrap().completed.is_none());
    assert!(pump.has_pending());

    // At the abort ceiling without the margin, the first observation is
    // ordinary hard pressure; only the consecutive observation interrupts.
    set_usage(&source, 33 * GIB);
    let first = pump.poll_tick(&mut cx).unwrap();
    assert_eq!(first.tick.status.pressure, PressureLevel::Hard);
    assert!(!first.tick.status.emergency);
    assert!(first.tick.request_execution_cancellation.is_none());
    assert!(first.cancellation.is_none());
    assert!(!native_control.is_cancelled());
    assert!(environment.admission.admissions_paused());
    assert_reservations(&environment, &blocked);

    let emergency = pump.poll_tick(&mut cx).unwrap();
    assert!(emergency.tick.status.emergency);
    assert_eq!(
        emergency
            .tick
            .request_execution_cancellation
            .as_ref()
            .map(|target| target.request_id()),
        Some(&request_id)
    );
    assert_eq!(
        emergency.cancellation.unwrap().unwrap(),
        CancelOutcome::Requested
    );
    assert!(emergency.completed.is_none());
    assert!(native_control.is_cancelled());
    assert_eq!(
        environment.scheduler.request_state(&request_id).unwrap(),
        RequestState::CancellationRequested
    );
    assert!(environment.scheduler.is_busy().unwrap());
    assert_reservations(&environment, &blocked);
    assert_no_terminal_available(&mut stream, &mut cx);
    let events = environment.control.events();
    assert_eq!(
        events[checkpoint..]
            .iter()
            .any(|event| event.kind == "load-constructed" && event.artifact == artifact_b()),
        gate_stage == 2
    );
    assert!(
        events[checkpoint..]
            .iter()
            .all(|event| !matches!(event.kind, "cache-destroyed" | "model-destroyed"))
    );

    release.release();
    assert_one_cancelled_terminal(stream, &request_id);
    environment.wait_idle();
    finish_pending_action(&mut pump, &mut cx);
    let residents = block_on(environment.runtime.model_residency()).unwrap();
    assert!(
        !residents
            .iter()
            .any(|resident| resident.artifact == artifact_b())
    );
    assert_reservations(&environment, &baseline);
    instance_lease_assert_unchanged(&environment, &a);
    let events = environment.control.events();
    let events = &events[checkpoint..];
    let returned = event_index(events, "load-gate-returned", &artifact_b());
    if gate_stage == 2 {
        assert!(event_index(events, "model-destroyed", &artifact_b()) > returned);
    }
    assert!(!events.iter().any(|event| event.artifact == artifact_a()
        && matches!(event.kind, "cache-destroyed" | "model-destroyed")));
    assert!(environment.admission.admissions_paused());

    source.set_normal();
    assert_eq!(
        pump.poll_tick(&mut cx).unwrap().tick.status.pressure,
        PressureLevel::Ok
    );
    assert!(!environment.admission.admissions_paused());
    finish_pending_action(&mut pump, &mut cx);
    environment
        .control
        .startup_gate_stage
        .store(0, Ordering::Release);
    successful(&environment.run("emergency-pending-recovery-b", profile_b(), "hello"));
    instance_lease_assert_unchanged(&environment, &a);
    assert_reservations(&environment, &baseline);
    drop(lease);
    block_on(environment.engine.release_session_state(a_id)).unwrap();
    environment.assert_empty();
}

#[test]
fn pending_startup_cancellation_preserves_grant_until_loader_returns() {
    check_pending_startup(1);
}

#[test]
fn pending_startup_cancellation_after_native_construction_releases_grant() {
    check_pending_startup(2);
}

struct GatedPressureSource {
    source: Arc<PoolPressureSource>,
    gate: Arc<StartupGate>,
}

impl std::fmt::Debug for GatedPressureSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GatedPressureSource")
            .finish_non_exhaustive()
    }
}

impl MemoryObservationSource for GatedPressureSource {
    fn observe(&self) -> Result<MemoryObservation, EnforcerError> {
        let observed = self.source.observe()?;
        self.gate.enter_if_armed();
        Ok(observed)
    }
}

#[test]
fn delayed_emergency_observation_cannot_cancel_replacement_request() {
    let environment = PolicyEnvironment::new_pool(false);
    let a_id = RequestId::new("emergency-stale-a").unwrap();
    let a_release = environment.control.execution_gate.arm();
    let a_generation = block_on(environment.engine.generate(environment.request(
        a_id.as_str(),
        profile_a(),
        "hello",
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let (_, source) = pool_pressure_enforcer(environment.admission.clone(), true);
    set_usage(&source, 35 * GIB);
    let probe_gate = Arc::new(StartupGate::default());
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::balanced(),
        environment.admission.clone(),
        Arc::new(GatedPressureSource {
            source: Arc::clone(&source),
            gate: Arc::clone(&probe_gate),
        }),
    );

    thread::scope(|scope| {
        let probe_release = probe_gate.arm();
        let engine = environment.engine.clone();
        let polling_enforcer = enforcer.clone();
        let polling = scope.spawn(move || {
            let mut pump = MemoryEnforcementPump::new(engine, polling_enforcer);
            let waker = Waker::from(Arc::new(ThreadWake {
                thread: thread::current(),
            }));
            let mut cx = Context::from_waker(&waker);
            let poll = pump.poll_tick(&mut cx).unwrap();
            (pump, poll)
        });
        probe_gate.wait_entered();

        // A completes and B acquires while the emergency observation about A
        // is outstanding. No pressure decision has paused admission yet.
        a_release.release();
        drop(a_release);
        let mut stream = a_generation.into_events();
        successful(&drain(&mut stream));
        drop(stream);
        environment.wait_idle();
        let b_id = RequestId::new("emergency-replacement-b").unwrap();
        let b_release = environment.control.execution_gate.arm();
        let b_generation = block_on(environment.engine.generate(environment.request(
            b_id.as_str(),
            profile_b(),
            "hello",
        )))
        .unwrap();
        environment.control.execution_gate.wait_entered();
        let native_control = environment.control.execution_control(&b_id);
        let blocked = environment.snapshot();

        probe_release.release();
        let (mut pump, stale) = polling.join().unwrap();
        assert!(stale.tick.status.emergency);
        let target = stale.tick.request_execution_cancellation.as_ref().unwrap();
        assert_eq!(target.request_id(), &a_id);
        assert_eq!(
            stale.cancellation.unwrap().unwrap(),
            CancelOutcome::AlreadyFinished
        );
        assert_eq!(
            target.request_cancel().unwrap(),
            CancelOutcome::AlreadyFinished
        );
        assert_eq!(
            environment.scheduler.request_state(&b_id).unwrap(),
            RequestState::Running
        );
        assert!(!native_control.is_cancelled());
        assert_eq!(environment.engine.active_request_id().unwrap(), Some(b_id));
        assert_reservations(&environment, &blocked);
        assert!(pump.has_pending());
        assert!(stale.completed.is_none());

        source.set_normal();
        enforcer.tick().unwrap();
        b_release.release();
        let mut stream = b_generation.into_events();
        successful(&drain(&mut stream));
        drop(stream);
        environment.wait_idle();
        let waker = Waker::from(Arc::new(ThreadWake {
            thread: thread::current(),
        }));
        let mut cx = Context::from_waker(&waker);
        finish_pending_action(&mut pump, &mut cx);
    });
}
