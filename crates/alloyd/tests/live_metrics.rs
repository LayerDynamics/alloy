//! Metrics RPC reads remain responsive and cannot advance enforcement or ownership.

use alloyd::{DaemonService, LiveEngineMetrics, MetricsSource, RpcRequest};
use engine::admission::AdmissionController;
use engine::api::{EngineCore, EngineFuture};
use engine::error::EngineError;
use engine::family::{ExecutionControl, LoadedModel, ModelFamily, ModelLoadRequest};
use engine::generation::GenerationLimits;
use engine::memory_enforcer::{
    EnforcerError, MemoryEnforcer, MemoryObservation, MemoryObservationSource,
};
use engine::memory_guard::{HostMemorySnapshot, MemoryGuardConfig};
use engine::registry::{ArtifactRegistration, FamilyRegistration, FamilyVersion, Registry};
use engine::runtime::{FamilyRuntime, RetentionPolicy, RuntimeBackend, RuntimePlan, TextDecoder};
use engine::scheduler::Scheduler;
use engine::types::{
    Capabilities, ContextSnapshot, DecodingCapability, ExecutionBackend, ModelProfileId,
    RenderedContext, RequestId, WeightRepresentation,
};
use harness_core::{HardwareTier, TelemetryClient};
use serde_json::{Value, json};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::task::{Context, Poll, Wake, Waker};
use std::thread;
use std::time::Duration;

const GIB: u64 = 1024 * 1024 * 1024;

// No model operation should be reachable from an observational RPC. The test
// uses a real FamilyRuntime worker and blocks its existing enforcement command.
struct UnusedBackend(FamilyRegistration);

impl ModelFamily for UnusedBackend {
    fn registration(&self) -> &FamilyRegistration {
        &self.0
    }
    fn load(&self, _: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        panic!("metrics must not load models")
    }
}

impl RuntimeBackend for UnusedBackend {
    fn render_and_count<'a>(
        &'a self,
        _: &'a Registry,
        _: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        panic!("metrics must not render")
    }
    fn plan<'a>(
        &'a self,
        _: &'a Registry,
        _: &'a ModelProfileId,
        _: usize,
    ) -> EngineFuture<'a, RuntimePlan> {
        panic!("metrics must not plan execution")
    }
    fn verify_artifact(
        &self,
        _: &ArtifactRegistration,
        _: &ExecutionControl,
    ) -> Result<(), EngineError> {
        panic!("metrics must not verify artifacts")
    }
    fn decoder(
        &self,
        _: &Registry,
        _: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        panic!("metrics must not create a decoder")
    }
    fn eos_tokens(&self, _: &Registry, _: &ModelProfileId) -> Result<Vec<u32>, EngineError> {
        panic!("metrics must not inspect model tokens")
    }
}

fn engine() -> (EngineCore<FamilyRuntime>, AdmissionController, Scheduler) {
    let admission = AdmissionController::new(16 * GIB).unwrap();
    let scheduler = Scheduler::new(8).unwrap();
    let registration = FamilyRegistration::new(
        FamilyVersion::new(1, 0, 0),
        vec!["MetricsFixture".to_owned()],
        Capabilities {
            task: "text-generation".to_owned(),
            family: "metrics-fixture".to_owned(),
            backend: ExecutionBackend::HostCpu,
            weight_representations: vec![WeightRepresentation::F32],
            cache_modes: Vec::new(),
            context_limit_tokens: 16,
            supports_streaming: false,
            supports_incremental_kv: false,
            supports_prefix_reuse: false,
            decoding_modes: vec![DecodingCapability::Greedy],
        },
    )
    .unwrap();
    let runtime = FamilyRuntime::new(
        vec![Arc::new(UnusedBackend(registration))],
        admission.clone(),
        GenerationLimits::new(4, 1024, 1024).unwrap(),
        RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: true,
        },
    )
    .unwrap()
    .with_model_pool(None)
    .unwrap();
    (
        EngineCore::new(Registry::new(), scheduler.clone(), runtime),
        admission,
        scheduler,
    )
}

#[derive(Debug, Default)]
struct GateState {
    armed: bool,
    entered: bool,
    released: bool,
}

#[derive(Debug, Default)]
struct ObservationGate {
    state: Mutex<GateState>,
    changed: Condvar,
    calls: AtomicUsize,
    fail: AtomicBool,
}

impl ObservationGate {
    fn arm(self: &Arc<Self>) -> ReleaseGate {
        self.state.lock().unwrap().armed = true;
        ReleaseGate(Arc::clone(self))
    }
    fn wait_entered(&self) {
        let (state, _) = self
            .changed
            .wait_timeout_while(
                self.state.lock().unwrap(),
                Duration::from_secs(5),
                |state| !state.entered,
            )
            .unwrap();
        assert!(
            state.entered,
            "native worker did not enter its gated command"
        );
    }
    fn release(&self) {
        self.state.lock().unwrap().released = true;
        self.changed.notify_all();
    }
}

struct ReleaseGate(Arc<ObservationGate>);
impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.release();
    }
}

impl MemoryObservationSource for ObservationGate {
    fn observe(&self) -> Result<MemoryObservation, EnforcerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.lock().unwrap();
        if state.armed && !state.released {
            state.entered = true;
            self.changed.notify_all();
            while !state.released {
                state = self.changed.wait(state).unwrap();
            }
        }
        drop(state);
        if self.fail.load(Ordering::SeqCst) {
            return Err(EnforcerError::HostProbe(
                "injected probe failure".to_owned(),
            ));
        }
        Ok(MemoryObservation {
            host: HostMemorySnapshot {
                physical_memory_bytes: 36 * GIB,
                free_bytes: 30 * GIB,
                inactive_bytes: 0,
                active_file_bytes: 0,
                other_apps_active_bytes: 0,
                phys_footprint_bytes: GIB,
                graphics_footprint_bytes: 0,
                iogpu_wired_limit_bytes: 0,
                recommended_working_set_bytes: 32 * GIB,
                release_credit_bytes: 0,
            },
            mlx_active_bytes: GIB,
            mlx_pool_bytes: 0,
        })
    }
}

struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

fn call_metrics(service: &DaemonService<EngineCore<FamilyRuntime>>) -> Value {
    let mut response = Vec::new();
    service
        .handle(
            RpcRequest {
                jsonrpc: "2.0".to_owned(),
                id: Some(json!(1)),
                method: "daemon.metrics".to_owned(),
                params: json!({}),
                api_version: Some(1),
            },
            &mut response,
        )
        .unwrap();
    let response: Value = serde_json::from_slice(&response).unwrap();
    assert!(response.get("error").is_none(), "{response}");
    response["result"].clone()
}

#[test]
fn metrics_rpc_responds_while_native_queue_is_blocked_without_mutating_ownership() {
    let (engine, admission, scheduler) = engine();
    let source = Arc::new(ObservationGate::default());
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::custom(16 * GIB).unwrap(),
        admission.clone(),
        source.clone(),
    );
    let metrics = Arc::new(LiveEngineMetrics::new(engine.clone(), enforcer.clone()));
    let dir = tempfile::tempdir().unwrap();
    let service = Arc::new(
        DaemonService::open(
            dir.path().join("metrics.sqlite"),
            engine.clone(),
            Arc::new(TelemetryClient::new(
                "test",
                HardwareTier::Unknown,
                "https://127.0.0.1:1/drop",
            )),
            4,
            99,
            777,
        )
        .unwrap()
        .with_metrics_source(metrics),
    );

    let initial = call_metrics(&service);
    assert_eq!(initial["residency"]["models_resident"], 0);
    assert_eq!(initial["admission"]["reserved_bytes"], 0);
    assert!(initial["memory_guard"]["usage_bytes"].is_null());
    assert_eq!(initial["memory_guard"]["pressure"], "unknown");
    assert_eq!(source.calls.load(Ordering::SeqCst), 0);

    let tick = enforcer.tick().unwrap();
    let reservation = admission
        .reserve("owned through observation", 4096)
        .unwrap();
    let request = RequestId::new("metrics-active-request").unwrap();
    let permit = scheduler.try_acquire(request.clone()).unwrap();
    scheduler.cancel(&request).unwrap();
    admission.set_admissions_paused(true);
    let before = admission.snapshot().unwrap();
    let execution_before = scheduler.snapshot().unwrap();

    // The actual native command queue is occupied, not merely a slow RPC mock.
    let release = source.arm();
    let runtime = engine.runtime().clone();
    let enforcement = thread::spawn(move || block_on(runtime.apply_memory_tick(enforcer, tick)));
    source.wait_entered();
    let observed_calls = source.calls.load(Ordering::SeqCst);
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        let responses: Vec<_> = (0..8).map(|_| call_metrics(&service)).collect();
        sender.send(responses).unwrap();
    });
    let responses = receiver.recv_timeout(Duration::from_secs(2));
    let after = admission.snapshot().unwrap();
    let execution_after = scheduler.snapshot().unwrap();
    let paused_after = admission.admissions_paused();
    let calls_after = source.calls.load(Ordering::SeqCst);
    drop(release);
    enforcement.join().unwrap().unwrap();
    reader.join().unwrap();

    let responses = responses.expect("metrics RPC queued behind occupied native worker");
    for response in responses {
        assert_eq!(response["schema_version"], 1);
        assert_eq!(response["process_id"], std::process::id());
        assert_eq!(response["admission"]["reserved_bytes"], 4096);
        assert_eq!(response["admission"]["active_reservations"], 1);
        assert_eq!(response["admission"]["admissions_paused"], true);
        assert_eq!(response["execution"]["active_request_id"], request.as_str());
        assert_eq!(response["execution"]["cancellation_requested"], true);
        assert_eq!(response["residency"]["models_resident"], 0);
        assert_eq!(response["memory_guard"]["hard_limit_bytes"], 16 * GIB);
        assert!(response["residency"]["sample_age_ms"].is_number());
    }
    assert_eq!(after, before);
    assert_eq!(execution_after, execution_before);
    assert!(paused_after);
    assert_eq!(
        calls_after, observed_calls,
        "metrics reads must not observe or enforce pressure"
    );
    drop(permit);
    drop(reservation);
}

#[test]
fn metrics_preserve_last_success_and_report_probe_errors_without_retrying() {
    let (engine, admission, _) = engine();
    let source = Arc::new(ObservationGate::default());
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::custom(16 * GIB).unwrap(),
        admission,
        source.clone(),
    );
    let metrics = LiveEngineMetrics::new(engine, enforcer.clone());
    source.fail.store(true, Ordering::SeqCst);
    assert!(enforcer.tick().is_err());
    let first = metrics.snapshot().unwrap();
    assert!(first["memory_guard"]["sample_age_ms"].is_null());
    assert!(first["memory_guard"]["usage_bytes"].is_null());
    assert!(
        first["memory_guard"]["last_error"]
            .as_str()
            .unwrap()
            .contains("injected probe failure")
    );

    source.fail.store(false, Ordering::SeqCst);
    enforcer.tick().unwrap();
    source.fail.store(true, Ordering::SeqCst);
    assert!(enforcer.tick().is_err());
    let calls = source.calls.load(Ordering::SeqCst);
    let failed = metrics.snapshot().unwrap();
    assert_eq!(failed["memory_guard"]["usage_bytes"], GIB);
    assert_eq!(failed["memory_guard"]["dynamic_ceiling_bytes"], 16 * GIB);
    assert!(failed["memory_guard"]["sample_age_ms"].is_number());
    assert!(failed["memory_guard"]["last_error"].is_string());
    assert_eq!(source.calls.load(Ordering::SeqCst), calls);

    source.fail.store(false, Ordering::SeqCst);
    enforcer.tick().unwrap();
    assert!(metrics.snapshot().unwrap()["memory_guard"]["last_error"].is_null());
}
