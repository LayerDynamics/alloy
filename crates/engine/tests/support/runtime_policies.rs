//! Runtime retention, reclamation, and failure-policy integration tests.
//!
//! Loaded models, caches, verification, prefill, and decode delegate to
//! the actual Qwen backend.
//!
//! Apple Silicon uses the production CPU host work and Metal transformer
//! placement. CPU and GPU share each model's MLX weights and KV buffers,
//! and all residents use one admission ledger. Other hosts run the same
//! ownership checks through MLX CPU. Metal unavailability fails loading.
//!
//! Instrumentation records native destruction and reservation snapshots.
//! Deliberate verification/loading errors exercise cleanup boundaries;
//! they do not provide synthetic model outputs.
//!
//! These tests are included as a child of tests/generation.rs and reuse
//! that file's actual safetensors/tokenizer fixtures.

use super::*;
use engine::{
    admission::AdmissionSnapshot, backends::Qwen3Execution, types::CacheRepresentationId,
};
use std::{
    any::Any,
    sync::{Condvar, atomic::AtomicBool, mpsc},
};

const POLICY_OUTPUT: usize = 4;
const POLICY_MARGIN: usize = 8;

fn artifact_a() -> ArtifactId {
    ArtifactId::new("policy-artifact-a").expect("artifact A")
}

fn artifact_b() -> ArtifactId {
    ArtifactId::new("policy-artifact-b").expect("artifact B")
}

fn profile_a() -> ModelProfileId {
    ModelProfileId::new("policy-profile-a").expect("profile A")
}

fn profile_b() -> ModelProfileId {
    ModelProfileId::new("policy-profile-b").expect("profile B")
}

fn artifact_c() -> ArtifactId {
    ArtifactId::new("policy-artifact-c").expect("artifact C")
}

fn profile_c() -> ModelProfileId {
    ModelProfileId::new("policy-profile-c").expect("profile C")
}

fn profile_large_cache() -> ModelProfileId {
    ModelProfileId::new("policy-profile-large-cache").expect("large-cache profile")
}

#[derive(Clone, Debug)]
struct PolicyEvent {
    kind: &'static str,
    artifact: ArtifactId,
    snapshot: AdmissionSnapshot,
    logical_before: Option<usize>,
}

#[derive(Default)]
struct GateState {
    armed: bool,
    entered: bool,
    released: bool,
}

/// One-shot gate for a native startup operation.
#[derive(Default)]
struct StartupGate {
    state: Mutex<GateState>,
    changed: Condvar,
}

impl StartupGate {
    fn arm(self: &Arc<Self>) -> GateRelease {
        let mut state = self.state.lock().expect("gate state");
        assert!(!state.armed, "startup gate already armed");

        *state = GateState {
            armed: true,
            entered: false,
            released: false,
        };

        GateRelease {
            gate: Arc::clone(self),
        }
    }

    fn enter_if_armed(&self) {
        let mut state = self.state.lock().expect("gate state");
        if !state.armed {
            return;
        }

        state.entered = true;
        self.changed.notify_all();

        while !state.released {
            let (next, timeout) = self
                .changed
                .wait_timeout(state, TEST_TIMEOUT)
                .expect("gate wait");
            state = next;

            assert!(
                !timeout.timed_out() || state.released,
                "startup gate was never released",
            );
        }

        state.armed = false;
    }

    fn wait_entered(&self) {
        let deadline = deadline();
        let mut state = self.state.lock().expect("gate state");

        while !state.entered {
            let now = Instant::now();
            assert!(now < deadline, "native worker did not enter startup gate");

            let (next, timeout) = self
                .changed
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .expect("wait for startup gate");
            state = next;

            assert!(
                !timeout.timed_out() || state.entered,
                "native worker did not enter startup gate",
            );
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().expect("gate state");
        state.released = true;
        self.changed.notify_all();
    }
}

/// Releases the gate during unwinding as well as ordinary completion.
struct GateRelease {
    gate: Arc<StartupGate>,
}

impl GateRelease {
    fn release(&self) {
        self.gate.release();
    }
}

impl Drop for GateRelease {
    fn drop(&mut self) {
        self.gate.release();
    }
}

struct PolicyControl {
    admission: AdmissionController,
    events: Mutex<Vec<PolicyEvent>>,
    instance_lease_tokens: Mutex<BTreeMap<RequestId, Vec<u32>>>,
    execution_controls: Mutex<BTreeMap<RequestId, ExecutionControl>>,
    fail_verification_once: AtomicBool,
    fail_loading_once: AtomicBool,
    startup_gate: Arc<StartupGate>,
    // 0: existing verification gate; 1: before load; 2: after load.
    startup_gate_stage: AtomicUsize,
    execution_gate: Arc<StartupGate>,
}

impl PolicyControl {
    fn new(admission: AdmissionController) -> Self {
        Self {
            admission,
            events: Mutex::new(Vec::new()),
            instance_lease_tokens: Mutex::new(BTreeMap::new()),
            execution_controls: Mutex::new(BTreeMap::new()),
            fail_verification_once: AtomicBool::new(false),
            fail_loading_once: AtomicBool::new(false),
            startup_gate: Arc::new(StartupGate::default()),
            startup_gate_stage: AtomicUsize::new(0),
            execution_gate: Arc::new(StartupGate::default()),
        }
    }

    fn record(&self, kind: &'static str, artifact: &ArtifactId, logical_before: Option<usize>) {
        let event = PolicyEvent {
            kind,
            artifact: artifact.clone(),
            snapshot: self.admission.snapshot().expect("event accounting"),
            logical_before,
        };

        self.events.lock().expect("policy events").push(event);
    }

    fn events(&self) -> Vec<PolicyEvent> {
        self.events.lock().expect("policy events").clone()
    }

    fn execution_control(&self, request_id: &RequestId) -> ExecutionControl {
        self.execution_controls
            .lock()
            .expect("execution controls")
            .get(request_id)
            .expect("native execution received this request's control")
            .clone()
    }
}

fn injected_failure(stage: &'static str) -> EngineError {
    EngineError::ExecutionFailed {
        stage,
        message: "deliberately injected runtime-policy test failure".to_owned(),
    }
}

/// Keeps the actual cache visible to family-private downcasts.
///
/// Its Any methods delegate to the real cache. Returning this wrapper
/// instead would break the family's concrete-cache inspection.
struct PolicyCache {
    inner: Option<Box<dyn FamilyCache>>,
    artifact: ArtifactId,
    control: Arc<PolicyControl>,
}

impl PolicyCache {
    fn inner(&self) -> &dyn FamilyCache {
        self.inner.as_deref().expect("live native cache")
    }

    fn inner_mut(&mut self) -> &mut dyn FamilyCache {
        self.inner.as_deref_mut().expect("live native cache")
    }
}

impl FamilyCache for PolicyCache {
    fn owner(&self) -> &ModelInstanceId {
        self.inner().owner()
    }

    fn representation(&self) -> &CacheRepresentationId {
        self.inner().representation()
    }

    fn mode(&self) -> &engine::types::CacheMode {
        self.inner().mode()
    }

    fn len(&self) -> usize {
        self.inner().len()
    }

    fn retained_tokens(&self) -> usize {
        self.inner().retained_tokens()
    }

    fn capacity_tokens(&self) -> usize {
        self.inner().capacity_tokens()
    }

    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        self.inner().memory_usage()
    }

    fn supports_truncation(&self) -> bool {
        self.inner().supports_truncation()
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner_mut().reset()
    }

    fn truncate(&mut self, token_count: usize) -> Result<(), EngineError> {
        self.inner_mut().truncate(token_count)
    }

    fn as_any(&self) -> &dyn Any {
        self.inner().as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self.inner_mut().as_any_mut()
    }
}

impl Drop for PolicyCache {
    fn drop(&mut self) {
        // Record after actual native destruction, not merely before Drop.
        drop(self.inner.take());
        self.control.record("cache-destroyed", &self.artifact, None);
    }
}

struct PolicyModel {
    inner: Option<ObservedModel>,
    artifact: ArtifactId,
    control: Arc<PolicyControl>,
}

impl PolicyModel {
    fn inner(&self) -> &ObservedModel {
        self.inner.as_ref().expect("live observed model")
    }

    fn inner_mut(&mut self) -> &mut ObservedModel {
        self.inner.as_mut().expect("live observed model")
    }
}

impl LoadedModel for PolicyModel {
    fn instance_id(&self) -> &ModelInstanceId {
        self.inner().instance_id()
    }

    fn artifact_id(&self) -> &ArtifactId {
        self.inner().artifact_id()
    }

    fn registration(&self) -> &FamilyRegistration {
        self.inner().registration()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner().capabilities()
    }

    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        self.inner().memory_usage()
    }

    fn autoregressive(&mut self) -> Option<&mut dyn AutoregressiveModel> {
        Some(self)
    }
}

impl AutoregressiveModel for PolicyModel {
    fn vocabulary_size(&self) -> usize {
        self.inner().vocabulary_size()
    }

    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        execution: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        self.control
            .record("cache-allocation-entered", &self.artifact, None);

        let cache = self.inner_mut().allocate_cache(request, execution)?;

        self.control.record("cache-allocated", &self.artifact, None);

        Ok(Box::new(PolicyCache {
            inner: Some(cache),
            artifact: self.artifact.clone(),
            control: Arc::clone(&self.control),
        }))
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        execution: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        self.control
            .record("prefill-entered", &self.artifact, Some(cache.len()));
        self.control.execution_gate.enter_if_armed();
        let result = self.inner_mut().prefill(tokens, cache, execution);
        self.control
            .record("prefill-returned", &self.artifact, None);
        result
    }

    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        execution: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        let logits = self.inner_mut().decode(token, cache, execution)?;
        self.control
            .instance_lease_tokens
            .lock()
            .expect("observed decoded tokens")
            .entry(execution.request_id().clone())
            .or_default()
            .push(token);
        Ok(logits)
    }
}

impl Drop for PolicyModel {
    fn drop(&mut self) {
        drop(self.inner.take());
        self.control.record("model-destroyed", &self.artifact, None);
    }
}

struct PolicyBackend {
    inner: Qwen3Backend,
    control: Arc<PolicyControl>,
    observations: Arc<Observations>,
}

impl ModelFamily for PolicyBackend {
    fn registration(&self) -> &FamilyRegistration {
        self.inner.registration()
    }

    fn load(&self, request: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        let artifact = request.artifact().id().clone();

        self.control.record("load-entered", &artifact, None);
        let gate_stage = self.control.startup_gate_stage.load(Ordering::Acquire);
        if gate_stage == 1 {
            self.control.startup_gate.enter_if_armed();
            self.control.record("load-gate-returned", &artifact, None);
        }

        let model = self.inner.load(request)?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(
            model.capabilities().backend,
            engine::types::ExecutionBackend::MlxMetal,
            "policy tests must load the unified Apple Silicon production path",
        );
        self.control.record("load-constructed", &artifact, None);

        if gate_stage == 2 {
            self.control.startup_gate.enter_if_armed();
            self.control.record("load-gate-returned", &artifact, None);
        }

        if self.control.fail_loading_once.swap(false, Ordering::AcqRel) {
            // Exercise failure after actual native model construction.
            drop(model);
            self.control.record("load-failed", &artifact, None);
            return Err(injected_failure("policy injected loading failure"));
        }

        self.observations.loads.fetch_add(1, Ordering::Relaxed);
        self.control.record("model-loaded", &artifact, None);

        Ok(Box::new(PolicyModel {
            inner: Some(ObservedModel {
                inner: model,
                vocabulary: 258,
                observations: Arc::clone(&self.observations),
            }),
            artifact,
            control: Arc::clone(&self.control),
        }))
    }
}

impl RuntimeBackend for PolicyBackend {
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        self.inner.render_and_count(registry, context)
    }

    fn plan<'a>(
        &'a self,
        registry: &'a Registry,
        profile: &'a ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, RuntimePlan> {
        self.inner.plan(registry, profile, input_tokens)
    }

    fn verify_artifact(
        &self,
        artifact: &ArtifactRegistration,
        execution: &ExecutionControl,
    ) -> Result<(), EngineError> {
        self.control
            .execution_controls
            .lock()
            .expect("execution controls")
            .insert(execution.request_id().clone(), execution.clone());
        if self.control.startup_gate_stage.load(Ordering::Acquire) == 0 {
            self.control.startup_gate.enter_if_armed();
        }
        self.inner.verify_artifact(artifact, execution)?;

        if self
            .control
            .fail_verification_once
            .swap(false, Ordering::AcqRel)
        {
            self.control
                .record("verification-failed", artifact.id(), None);
            return Err(injected_failure("policy injected verification failure"));
        }

        self.control.record("verified", artifact.id(), None);
        Ok(())
    }

    fn decoder(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        self.inner.decoder(registry, profile)
    }

    fn eos_tokens(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Vec<u32>, EngineError> {
        self.inner.eos_tokens(registry, profile)
    }
}

struct PolicyEnvironment {
    runtime: FamilyRuntime,
    engine: EngineCore<FamilyRuntime>,
    admission: AdmissionController,
    scheduler: Scheduler,
    control: Arc<PolicyControl>,
    observations: Arc<Observations>,
    plan_a: RuntimePlan,
    plan_b: RuntimePlan,
    plan_c: RuntimePlan,
    plan_large: RuntimePlan,

    // The parent fixtures create real pinned checkpoint directories.
    // Keep them alive while this environment uses their files.
    _fixtures: Vec<TestEnvironment>,
}

fn fixture_specification(fixture: &TestEnvironment, id: ArtifactId) -> Qwen3Artifact {
    let directory = fixture._directory.path().to_path_buf();
    let mut files = BTreeMap::new();

    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        let bytes = std::fs::read(directory.join(name)).expect("fixture file");
        files.insert(
            name.to_owned(),
            FilePin {
                bytes: bytes.len() as u64,
                sha256: digest(&bytes),
            },
        );
    }

    Qwen3Artifact {
        id,
        directory,
        files,
        template_sha256: digest(CHAT_TEMPLATE.as_bytes()),
    }
}

fn request_context(profile: ModelProfileId, body: &str) -> ContextSnapshot {
    context("policy-context", profile, body)
}

fn sum(values: &[u64]) -> u64 {
    values
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value))
        .expect("fixture memory arithmetic")
}

/// Conservative startup bound accepted by the runtime estimate.
///
/// The additional logits allowance also permits independent completed
/// logits retention while execution workspace is still alive.
fn startup_budget(plan: &RuntimePlan) -> u64 {
    sum(&[
        plan.output_bytes,
        plan.temporary_bytes,
        plan.headroom_bytes,
        plan.load_limits.max_load_peak_bytes(),
        plan.cache.max_cache_bytes(),
        (plan.cache.capacity_tokens() * std::mem::size_of::<u32>()) as u64,
        (258 * std::mem::size_of::<f32>()) as u64,
    ])
}

/// Full cold-start grant after AdmitOutput for retained-contents policy.
///
/// Output has been taken; workspace, loading peak, KV, and history remain
/// on the grant until prepare partitions them. Reserved bytes equal this
/// total for the whole pending interval.
fn pending_startup_reservation(plan: &RuntimePlan) -> u64 {
    sum(&[
        plan.output_bytes,
        plan.temporary_bytes,
        plan.headroom_bytes,
        plan.load_limits.max_load_peak_bytes(),
        plan.cache.max_cache_bytes(),
        (plan.cache.capacity_tokens() * std::mem::size_of::<u32>()) as u64,
    ])
}

fn load_entry_reservation(plan: &RuntimePlan) -> u64 {
    // Load runs during prepare after workspace/peak takes. Cache and
    // history bytes stay on the grant, so the ledger still holds the
    // full pending startup total.
    pending_startup_reservation(plan)
}

fn cache_entry_reservation(plan: &RuntimePlan) -> u64 {
    // Peak has been resized to resident; history remains on the grant
    // until candidate construction.
    sum(&[
        plan.output_bytes,
        plan.temporary_bytes,
        plan.headroom_bytes,
        plan.load_limits.max_resident_bytes(),
        plan.cache.max_cache_bytes(),
        (plan.cache.capacity_tokens() * std::mem::size_of::<u32>()) as u64,
    ])
}

impl PolicyEnvironment {
    fn new(retention: RetentionPolicy, tight_budget: bool) -> Self {
        Self::build(retention, tight_budget, false)
    }

    fn new_pool(tight_budget: bool) -> Self {
        Self::build(retained_policy(), tight_budget, true)
    }

    fn new_pool_with_event_capacity(event_capacity: usize) -> Self {
        Self::build_with_event_capacity(retained_policy(), false, true, event_capacity)
    }

    fn build(retention: RetentionPolicy, tight_budget: bool, pooling: bool) -> Self {
        Self::build_with_event_capacity(retention, tight_budget, pooling, 8)
    }

    fn build_with_event_capacity(
        retention: RetentionPolicy,
        tight_budget: bool,
        pooling: bool,
        event_capacity: usize,
    ) -> Self {
        let fixture_a = TestEnvironment::new(POLICY_OUTPUT, 8, false);
        let fixture_b = TestEnvironment::new(POLICY_OUTPUT, 8, false);
        let fixture_c = TestEnvironment::new(POLICY_OUTPUT, 8, false);
        let context_limit = Qwen3Config::tiny_fixture().max_context;
        let generation_limits =
            GenerationLimits::new(event_capacity, 4096, 1024).expect("generation limits");

        // Match alloyd's platform default; load checks Metal availability.
        let execution = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Qwen3Execution::MlxMetal
        } else {
            Qwen3Execution::MlxCpu
        };
        let inner = Qwen3Backend::new_with_execution(
            vec![
                fixture_specification(&fixture_a, artifact_a()),
                fixture_specification(&fixture_b, artifact_b()),
                fixture_specification(&fixture_c, artifact_c()),
            ],
            Qwen3BackendLimits {
                max_files: 16,
                max_config_bytes: 1024 * 1024,
                max_tokenizer_bytes: 1024 * 1024,
                max_messages: 16,
                max_input_bytes: 4096,
                max_rendered_bytes: 16 * 1024,
                max_token_bytes: 128,
                verification_chunk_bytes: 4096,
                context_limit_tokens: context_limit,
                prefill_chunk_tokens: NonZeroUsize::new(2).expect("chunk"),
                loader_headroom_bytes: 64 * 1024,
                execution_headroom_bytes: 64 * 1024,
                generation: generation_limits,
            },
            execution,
        )
        .expect("policy Qwen backend");

        let make_profile = |profile, artifact, output| {
            ModelProfile::new(
                profile,
                artifact,
                context_limit,
                output,
                POLICY_MARGIN,
                GenerationOptions::greedy(output).expect("default options"),
            )
            .expect("policy profile")
        };

        let registry = inner
            .registry(vec![
                make_profile(profile_a(), artifact_a(), POLICY_OUTPUT),
                make_profile(profile_b(), artifact_b(), POLICY_OUTPUT),
                make_profile(profile_c(), artifact_c(), POLICY_OUTPUT),
                make_profile(profile_large_cache(), artifact_a(), POLICY_OUTPUT + 4),
            ])
            .expect("policy registry");

        let plan_for = |profile: ModelProfileId| {
            let rendered = block_on(
                inner.render_and_count(&registry, request_context(profile.clone(), "hello")),
            )
            .expect("policy rendering");

            block_on(inner.plan(&registry, &profile, rendered.token_count())).expect("policy plan")
        };

        let plan_a = plan_for(profile_a());
        let plan_b = plan_for(profile_b());
        let plan_c = plan_for(profile_c());
        let plan_large = plan_for(profile_large_cache());

        let budget = if tight_budget {
            startup_budget(&plan_a)
                .max(startup_budget(&plan_b))
                .max(startup_budget(&plan_c))
                .max(startup_budget(&plan_large))
        } else {
            BUDGET_BYTES
        };

        let admission = AdmissionController::new(budget).expect("admission");
        let scheduler = Scheduler::new(64).expect("scheduler");
        let control = Arc::new(PolicyControl::new(admission.clone()));
        let observations = Arc::new(Observations::default());

        let backend: Arc<dyn RuntimeBackend> = Arc::new(PolicyBackend {
            inner,
            control: Arc::clone(&control),
            observations: Arc::clone(&observations),
        });

        let runtime = FamilyRuntime::new(
            vec![backend],
            admission.clone(),
            generation_limits,
            retention,
        )
        .expect("policy runtime");

        let runtime = if pooling {
            // No wall-clock expiration during switching tests.
            runtime.with_model_pool(None).expect("enable model pool")
        } else {
            runtime
        };

        Self {
            runtime: runtime.clone(),
            engine: EngineCore::new(registry, scheduler.clone(), runtime),
            admission,
            scheduler,
            control,
            observations,
            plan_a,
            plan_b,
            plan_c,
            plan_large,
            _fixtures: vec![fixture_a, fixture_b, fixture_c],
        }
    }

    fn request(&self, name: &str, profile: ModelProfileId, body: &str) -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(name).expect("request ID"),
            profile.clone(),
            request_context(profile, body),
            GenerationOptions::greedy(POLICY_OUTPUT).expect("options"),
        )
        .expect("policy request")
    }

    fn run(&self, name: &str, profile: ModelProfileId, body: &str) -> Transcript {
        let generation = block_on(self.engine.generate(self.request(name, profile, body)))
            .expect("policy generation accepted");

        let mut stream = generation.into_events();
        let transcript = drain(&mut stream);
        drop(stream);
        self.wait_idle();
        transcript
    }

    fn wait_idle(&self) {
        let deadline = deadline();
        while self.scheduler.is_busy().expect("scheduler status") {
            assert!(
                Instant::now() < deadline,
                "policy worker did not finish cleanup",
            );
            thread::yield_now();
        }
        assert!(self.engine.active_request_id().expect("owner").is_none());
    }

    fn snapshot(&self) -> AdmissionSnapshot {
        self.admission.snapshot().expect("policy snapshot")
    }

    fn assert_empty(&self) {
        self.wait_idle();
        let snapshot = self.snapshot();
        assert_eq!(snapshot.reserved_bytes(), 0);
        assert_eq!(snapshot.active_reservations(), 0);
    }
}

fn event_index(events: &[PolicyEvent], kind: &'static str, artifact: &ArtifactId) -> usize {
    events
        .iter()
        .position(|event| event.kind == kind && &event.artifact == artifact)
        .unwrap_or_else(|| panic!("missing {kind} for {artifact}: {events:#?}"))
}

fn successful(transcript: &Transcript) {
    let usage = completed_usage(transcript, FinishReason::OutputLimit);
    assert_eq!(usage.output_tokens, POLICY_OUTPUT);
    assert_eq!(transcript.text, "!".repeat(POLICY_OUTPUT));
}

#[test]
fn policy_model_only_retention_releases_request_and_cache_reservations() {
    let environment = PolicyEnvironment::new(
        RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: false,
        },
        false,
    );

    successful(&environment.run("model-only-a", profile_a(), "hello"));

    let snapshot = environment.snapshot();
    assert_eq!(
        snapshot.reserved_bytes(),
        environment.plan_a.load_limits.max_resident_bytes(),
    );
    assert_eq!(snapshot.active_reservations(), 1);

    let events = environment.control.events();
    event_index(&events, "cache-destroyed", &artifact_a());
    assert!(
        !events.iter().any(|event| event.kind == "model-destroyed"),
        "retained model must remain alive",
    );

    successful(&environment.run("model-only-again", profile_a(), "hello"));

    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1,);
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        2,
    );
    assert_eq!(environment.snapshot(), snapshot);
}

#[test]
fn policy_full_retention_accounts_for_native_state_and_resets_on_mismatch() {
    let environment = PolicyEnvironment::new(retained_policy(), false);

    successful(&environment.run("retain-both-first", profile_a(), "hello"));

    let baseline = environment.snapshot();
    let minimum_retained = sum(&[
        environment.plan_a.load_limits.max_resident_bytes(),
        environment.plan_a.cache.max_cache_bytes(),
        (environment.plan_a.cache.capacity_tokens() * std::mem::size_of::<u32>()) as u64,
        (258 * std::mem::size_of::<f32>()) as u64,
    ]);

    assert!(baseline.reserved_bytes() >= minimum_retained);

    // Weights, KV, exact token history, and retained logits.
    // Output and request workspace have been released.
    assert_eq!(baseline.active_reservations(), 4);

    let checkpoint = environment.control.events().len();

    // Same byte/token length, different prompt. The physical allocation
    // can stay, but its logical contents must not be reused.
    successful(&environment.run("retain-both-changed", profile_a(), "jello"));

    let events = environment.control.events();
    let new_events = &events[checkpoint..];
    let first_prefill = new_events
        .iter()
        .find(|event| event.kind == "prefill-entered")
        .expect("new actual prefill");

    assert_eq!(
        first_prefill.logical_before,
        Some(0),
        "incompatible context must begin with empty logical KV",
    );
    assert!(
        !new_events
            .iter()
            .any(|event| event.kind == "cache-destroyed"),
        "a same-sized context change need not replace native buffers",
    );
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        1,
    );
    assert_eq!(environment.snapshot(), baseline);
}

#[test]
fn policy_artifact_switch_reclaims_actual_idle_resources_before_loading() {
    let environment = PolicyEnvironment::new(retained_policy(), true);
    successful(&environment.run("artifact-switch-a", profile_a(), "hello"));

    let idle = environment.snapshot();
    let checkpoint = environment.control.events().len();

    assert!(
        idle.reserved_bytes()
            .checked_add(load_entry_reservation(&environment.plan_b))
            .expect("combined peak")
            > idle.budget_bytes(),
        "fixture must require reclamation, not merely permit overlap",
    );

    successful(&environment.run("artifact-switch-b", profile_b(), "hello"));

    let events = environment.control.events();
    let events = &events[checkpoint..];

    let cache_drop = event_index(events, "cache-destroyed", &artifact_a());
    let model_drop = event_index(events, "model-destroyed", &artifact_a());
    let load_b = event_index(events, "load-entered", &artifact_b());

    assert!(cache_drop < model_drop);
    assert!(model_drop < load_b);

    // At actual B loading entry, only B's output, workspace, and loader
    // peak should remain admitted. A's guards must already be released.
    assert_eq!(
        events[load_b].snapshot.reserved_bytes(),
        load_entry_reservation(&environment.plan_b),
    );

    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2,);
}

#[test]
fn policy_incompatible_cache_is_destroyed_and_unreserved_before_replacement() {
    let environment = PolicyEnvironment::new(retained_policy(), false);
    successful(&environment.run("cache-small", profile_a(), "hello"));

    assert!(
        environment.plan_large.cache.capacity_tokens() > environment.plan_a.cache.capacity_tokens(),
    );
    let checkpoint = environment.control.events().len();

    successful(&environment.run("cache-large", profile_large_cache(), "hello"));

    let events = environment.control.events();
    let events = &events[checkpoint..];

    let old_drop = event_index(events, "cache-destroyed", &artifact_a());
    let replacement = event_index(events, "cache-allocation-entered", &artifact_a());

    assert!(old_drop < replacement);
    assert!(
        !events.iter().any(|event| event.kind == "model-destroyed"),
        "cache replacement must preserve its compatible model",
    );

    // Runtime has already reserved the new cache when the family
    // allocator is called. The old KV/history/logits guards must be gone.
    assert_eq!(
        events[replacement].snapshot.reserved_bytes(),
        cache_entry_reservation(&environment.plan_large),
    );

    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1,);
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        2,
    );
}

struct AdmissionWake {
    sender: mpsc::Sender<()>,
}

impl Wake for AdmissionWake {
    fn wake(self: Arc<Self>) {
        let _ = self.sender.send(());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.sender.send(());
    }
}

#[test]
fn policy_abandoned_pending_startup_cleans_admission_before_slot_release() {
    let environment = PolicyEnvironment::new(retained_policy(), false);
    let release = environment.control.startup_gate.arm();

    let request_id = RequestId::new("abandon-pending").expect("request ID");
    let mut startup =
        environment
            .engine
            .generate(environment.request("abandon-pending", profile_a(), "hello"));

    let (sender, receiver) = mpsc::channel();
    let waker = Waker::from(Arc::new(AdmissionWake { sender }));
    let mut context = Context::from_waker(&waker);

    assert!(startup.as_mut().poll(&mut context).is_pending());

    environment.control.startup_gate.wait_entered();
    assert!(environment.scheduler.is_busy().expect("busy"));

    release.release();

    // The admission dispatcher publishes its result and wakes this
    // consumer only after native output admission and pending ownership.
    receiver
        .recv_timeout(TEST_TIMEOUT)
        .expect("startup admission ready");

    // Do not poll the result: that would accept the handoff and start
    // generation. Abandon the ready-but-unconsumed native startup instead.
    assert_eq!(
        environment.engine.active_request_id().expect("owner"),
        Some(request_id),
    );
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        pending_startup_reservation(&environment.plan_a),
    );
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 0,);

    drop(startup);
    environment.assert_empty();

    // No stale pending native owner may block recovery.
    successful(&environment.run("after-abandon", profile_a(), "hello"));
}

#[test]
fn policy_verification_failure_preserves_existing_idle_retention() {
    let environment = PolicyEnvironment::new(retained_policy(), false);
    successful(&environment.run("verify-baseline", profile_a(), "hello"));
    let baseline = environment.snapshot();
    let checkpoint = environment.control.events().len();

    environment
        .control
        .fail_verification_once
        .store(true, Ordering::Release);

    let error = expect_failure(
        block_on(environment.engine.generate(environment.request(
            "verify-fails",
            profile_b(),
            "hello",
        ))),
        "verification failure was ignored",
    );

    match error {
        EngineError::ExecutionFailed { stage, .. } => {
            assert_eq!(stage, "policy injected verification failure");
        }
        other => panic!("unexpected verification error: {other:?}"),
    }

    environment.wait_idle();
    assert_eq!(environment.snapshot(), baseline);

    let events = environment.control.events();
    assert!(
        !events[checkpoint..]
            .iter()
            .any(|event| event.kind == "model-destroyed" || event.kind == "cache-destroyed"),
        "startup verification failure must not reclaim existing idle state",
    );

    successful(&environment.run("verify-recovery", profile_a(), "hello"));
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1,);
}

#[test]
fn policy_loading_failure_releases_partial_work_and_allows_recovery() {
    let environment = PolicyEnvironment::new(retained_policy(), false);
    successful(&environment.run("load-baseline", profile_a(), "hello"));

    environment
        .control
        .fail_loading_once
        .store(true, Ordering::Release);

    let failed = environment.run("load-fails", profile_b(), "hello");

    match failed.terminal {
        GenerationEvent::RequestFailed { error, .. } => match error {
            EngineError::ExecutionFailed { stage, .. } => {
                assert_eq!(stage, "policy injected loading failure");
            }
            other => panic!("unexpected loading error: {other:?}"),
        },
        other => panic!("expected loading failure, got {other:?}"),
    }

    // A was evicted before B loading. B's actual temporary model was
    // destroyed by the injected failure. There is no model left to retain.
    environment.assert_empty();

    let events = environment.control.events();
    let old_model_drop = event_index(&events, "model-destroyed", &artifact_a());
    let new_load = event_index(&events, "load-entered", &artifact_b());
    let new_failure = event_index(&events, "load-failed", &artifact_b());

    assert!(old_model_drop < new_load);
    assert!(new_load < new_failure);

    successful(&environment.run("load-recovery", profile_b(), "hello"));
}

// These tests use actual safetensors checkpoints and Qwen execution.
// The two artifacts have deterministic fixture contents, separate IDs,
// and separate checkpoint directories.

fn pool_instance(environment: &PolicyEnvironment, artifact: &ArtifactId) -> ModelInstanceId {
    let entries =
        block_on(environment.runtime.model_residency()).expect("inspect actual residents");
    let matches: Vec<_> = entries
        .iter()
        .filter(|entry| &entry.artifact == artifact)
        .collect();
    assert_eq!(matches.len(), 1, "expected exactly one load of {artifact}");
    matches[0].instance.clone()
}

#[test]
fn pool_two_artifacts_reuse_original_instances() {
    let environment = PolicyEnvironment::new_pool(false);

    successful(&environment.run("pool-a-first", profile_a(), "hello"));
    let original_a = pool_instance(&environment, &artifact_a());

    successful(&environment.run("pool-b-first", profile_b(), "hello"));
    let original_b = pool_instance(&environment, &artifact_b());

    assert!(!original_a.same_instance(&original_b));
    assert_eq!(
        block_on(environment.runtime.model_residency())
            .unwrap()
            .len(),
        2,
    );
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);

    successful(&environment.run("pool-a-again", profile_a(), "hello"));

    assert!(
        pool_instance(&environment, &artifact_a()).same_instance(&original_a),
        "returning to A must reuse its original loaded instance",
    );
    assert!(
        pool_instance(&environment, &artifact_b()).same_instance(&original_b),
        "switching back to A must not destroy B",
    );
    assert_eq!(
        environment.observations.loads.load(Ordering::Relaxed),
        2,
        "A -> B -> A must perform only two actual loads",
    );

    assert!(
        !environment
            .control
            .events()
            .iter()
            .any(|event| event.kind == "model-destroyed"),
        "both models fit, so neither should be destroyed",
    );
}

#[test]
fn pool_tight_budget_reclaims_a_before_loading_b() {
    let environment = PolicyEnvironment::new_pool(true);

    successful(&environment.run("pool-tight-a", profile_a(), "hello"));
    let checkpoint = environment.control.events().len();
    let idle = environment.snapshot();

    assert!(
        idle.reserved_bytes()
            .checked_add(load_entry_reservation(&environment.plan_b))
            .expect("combined demand")
            > idle.budget_bytes(),
        "test must require reclamation",
    );

    successful(&environment.run("pool-tight-b", profile_b(), "hello"));

    let all_events = environment.control.events();
    let events = &all_events[checkpoint..];
    let destroyed_a = event_index(events, "model-destroyed", &artifact_a());
    let load_b = event_index(events, "load-entered", &artifact_b());

    assert!(destroyed_a < load_b);
    assert_eq!(
        events[load_b].snapshot.reserved_bytes(),
        load_entry_reservation(&environment.plan_b),
        "A's reservations must be gone before B's actual loading",
    );

    let entries = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].artifact, artifact_b());
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
}

#[test]
fn pool_pinned_a_rejects_tight_switch_without_destroying_a() {
    let environment = PolicyEnvironment::new_pool(true);

    successful(&environment.run("pool-pin-a", profile_a(), "hello"));
    let original_a = pool_instance(&environment, &artifact_a());
    let pin = block_on(environment.runtime.pin_model(original_a.clone()))
        .expect("pin command")
        .expect("A exists");
    let before = environment.snapshot();
    let checkpoint = environment.control.events().len();

    let attempt = block_on(environment.engine.generate(environment.request(
        "pool-pin-b",
        profile_b(),
        "hello",
    )));

    match attempt {
        Err(error) => assert!(
            matches!(
                error,
                EngineError::MemoryLimitExceeded { .. } | EngineError::Busy { .. }
            ),
            "unexpected startup error: {error:?}",
        ),
        Ok(generation) => {
            let mut stream = generation.into_events();
            let limit = deadline();
            let mut failed = false;
            let mut completed = false;

            while let Some(event) = next_event(&mut stream, limit) {
                match event {
                    GenerationEvent::RequestFailed { error, .. } => {
                        assert!(
                            matches!(
                                error,
                                EngineError::MemoryLimitExceeded { .. } | EngineError::Busy { .. }
                            ),
                            "unexpected worker error: {error:?}",
                        );
                        failed = true;
                    }
                    GenerationEvent::GenerationCompleted { .. } => completed = true,
                    _ => {}
                }
            }
            drop(stream);
            assert!(failed && !completed, "pinned tight switch must fail");
        }
    }

    environment.wait_idle();

    assert_eq!(environment.snapshot(), before);
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&original_a),);
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);

    let events = environment.control.events();
    assert!(
        !events[checkpoint..]
            .iter()
            .any(|event| event.kind == "model-destroyed"),
        "failed admission must preserve pinned A",
    );

    drop(pin);
    successful(&environment.run("pool-pin-b-retry", profile_b(), "hello"));
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
}

#[test]
fn pool_per_model_ttl_respects_pins_and_releases_committed_weights() {
    let environment = PolicyEnvironment::new_pool(false);

    successful(&environment.run("pool-ttl-a", profile_a(), "hello"));
    let retained_a = environment.snapshot();
    let a = pool_instance(&environment, &artifact_a());
    let pin = block_on(environment.runtime.pin_model(a.clone()))
        .unwrap()
        .expect("pin A");

    successful(&environment.run("pool-ttl-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());

    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(a.clone(), Some(Duration::ZERO),)
        )
        .unwrap(),
    );
    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(b.clone(), Some(Duration::ZERO),)
        )
        .unwrap(),
    );

    // Command barriers permit the worker's between-command TTL maintenance.
    // Zero TTL avoids sleeps and scheduler-sensitive duration thresholds.
    let _ = block_on(environment.runtime.model_residency()).unwrap();
    let entries = block_on(environment.runtime.model_residency()).unwrap();

    assert_eq!(entries.len(), 1);
    assert!(entries[0].instance.same_instance(&a));
    assert!(entries[0].pinned);
    assert_eq!(
        environment.snapshot(),
        retained_a,
        "B must release its reservations while pinned A retains its original state",
    );

    drop(pin);

    let _ = block_on(environment.runtime.model_residency()).unwrap();
    assert!(
        block_on(environment.runtime.model_residency())
            .unwrap()
            .is_empty(),
    );
    environment.assert_empty();
}

#[test]
fn pool_three_artifacts_evict_lru_not_fifo() {
    let environment = PolicyEnvironment::new_pool(false);

    successful(&environment.run("pool-lru-a-first", profile_a(), "hello"));
    let original_a = pool_instance(&environment, &artifact_a());
    let retained_a = environment.snapshot().reserved_bytes();

    successful(&environment.run("pool-lru-b-first", profile_b(), "hello"));
    let original_b = pool_instance(&environment, &artifact_b());
    let after_b = environment.snapshot().reserved_bytes();
    let retained_b = after_b
        .checked_sub(retained_a)
        .expect("B adds committed resident resources");
    assert!(retained_b > 0);

    // A was inserted first, but is now more recently used than B.
    successful(&environment.run("pool-lru-a-touch", profile_a(), "hello"));
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&original_a),);
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&original_b),);
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        after_b,
        "touching A must not introduce an additional resident allocation",
    );

    // Reserve unrelated bytes to leave C short by B's retained footprint.
    // This guard remains alive throughout C's loading. The runtime must
    // release a real model's resources, not credit unrelated reservations.
    let cold_c = startup_budget(&environment.plan_c);
    let desired_available = cold_c
        .checked_sub(retained_b)
        .expect("C cold-start demand exceeds one idle resident");
    assert!(desired_available < pending_startup_reservation(&environment.plan_c));

    let available = environment.snapshot().available_bytes();
    let blocking_bytes = available
        .checked_sub(desired_available)
        .expect("fixture has enough budget before forcing pressure");
    assert!(blocking_bytes > 0);

    let blocker = environment
        .admission
        .reserve("three-artifact unrelated allocation", blocking_bytes)
        .expect("reserve pressure fixture");

    assert_eq!(environment.snapshot().available_bytes(), desired_available,);

    let checkpoint = environment.control.events().len();
    successful(&environment.run("pool-lru-c-first", profile_c(), "hello"));

    let all_events = environment.control.events();
    let events = &all_events[checkpoint..];

    let b_cache_drop = event_index(events, "cache-destroyed", &artifact_b());
    let b_model_drop = event_index(events, "model-destroyed", &artifact_b());
    let c_load = event_index(events, "load-entered", &artifact_c());

    assert!(b_cache_drop < b_model_drop);
    assert!(b_model_drop < c_load);

    let destroyed: Vec<_> = events
        .iter()
        .filter(|event| event.kind == "model-destroyed")
        .map(|event| event.artifact.clone())
        .collect();
    assert_eq!(
        destroyed,
        vec![artifact_b()],
        "evict only B: FIFO would incorrectly evict A",
    );

    assert_eq!(
        events[c_load].snapshot.reserved_bytes(),
        sum(&[
            blocking_bytes,
            retained_a,
            load_entry_reservation(&environment.plan_c),
        ]),
        "B must be unreserved before loading C; A and the unrelated guard remain",
    );

    let entries = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().any(|entry| entry.artifact == artifact_a()));
    assert!(entries.iter().any(|entry| entry.artifact == artifact_c()));
    assert!(!entries.iter().any(|entry| entry.artifact == artifact_b()));
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 3);

    assert!(
        pool_instance(&environment, &artifact_a()).same_instance(&original_a),
        "LRU eviction must preserve A's original native instance",
    );

    drop(blocker);

    successful(&environment.run("pool-lru-a-after-c", profile_a(), "hello"));
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&original_a),);
    assert_eq!(
        environment.observations.loads.load(Ordering::Relaxed),
        3,
        "using surviving A again must not reload it",
    );
}

fn pool_advisory(
    environment: &PolicyEnvironment,
    profile: ModelProfileId,
) -> engine::types::MemoryEstimate {
    let rendered = block_on(
        environment
            .engine
            .render_and_count(request_context(profile.clone(), "hello")),
    )
    .expect("render estimate input");
    block_on(
        environment
            .engine
            .estimate_memory(profile, rendered.token_count()),
    )
    .expect("pool-aware advisory estimate")
}

#[test]
fn pool_advisory_parked_reuse_fits_when_cold_load_does_not() {
    use engine::api::EngineRuntime;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("advisory-reuse-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    successful(&environment.run("advisory-reuse-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let _pin_b = block_on(environment.runtime.pin_model(b)).unwrap().unwrap();

    assert!(environment.runtime.supports_memory_reclamation());
    let initial = pool_advisory(&environment, profile_a());
    assert_eq!(initial.breakdown().weights_bytes, 0);
    assert_eq!(initial.breakdown().kv_bytes, 0);
    let free_for_reuse = initial
        .reserved_bytes()
        .checked_add(258 * 4)
        .expect("retained logits allowance");
    assert!(pending_startup_reservation(&environment.plan_a) > free_for_reuse);
    let blocking = environment
        .snapshot()
        .available_bytes()
        .checked_sub(free_for_reuse)
        .expect("room to constrain fixture");
    let _blocker = environment
        .admission
        .reserve("advisory reuse blocker", blocking)
        .expect("blocker");
    let before = environment.snapshot();
    let checkpoint = environment.control.events().len();
    let estimate = pool_advisory(&environment, profile_a());
    assert!(estimate.fits());
    assert_eq!(estimate.breakdown().weights_bytes, 0);
    assert_eq!(estimate.breakdown().kv_bytes, 0);
    assert_eq!(
        environment.snapshot(),
        before,
        "estimation must not reserve or reclaim"
    );
    assert_eq!(environment.control.events().len(), checkpoint);

    successful(&environment.run("advisory-reuse-a-again", profile_a(), "hello"));
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
    let events = environment.control.events();
    assert!(
        !events[checkpoint..]
            .iter()
            .any(|event| event.kind == "model-destroyed")
    );
}

#[test]
fn pool_advisory_credits_only_eligible_eviction_before_cold_c_load() {
    use engine::api::EngineRuntime;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("advisory-evict-a", profile_a(), "hello"));
    let retained_a = environment.snapshot().reserved_bytes();
    successful(&environment.run("advisory-evict-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let _pin_b = block_on(environment.runtime.pin_model(b.clone()))
        .unwrap()
        .unwrap();
    let free = startup_budget(&environment.plan_c)
        .checked_sub(retained_a)
        .unwrap();
    assert!(free < pending_startup_reservation(&environment.plan_c));
    let blocking = environment
        .snapshot()
        .available_bytes()
        .checked_sub(free)
        .unwrap();
    let _blocker = environment
        .admission
        .reserve("advisory eviction blocker", blocking)
        .unwrap();
    let before = environment.snapshot();
    let checkpoint = environment.control.events().len();

    let estimate = pool_advisory(&environment, profile_c());
    assert!(environment.runtime.supports_memory_reclamation());
    assert!(
        estimate.reserved_bytes() > before.available_bytes(),
        "reclamation is necessary"
    );
    assert_eq!(
        estimate.available_bytes(),
        before.available_bytes() + retained_a
    );
    assert!(
        estimate.fits(),
        "eligible A makes C fit; pinned B contributes no credit"
    );
    assert_eq!(environment.snapshot(), before);
    assert_eq!(environment.control.events().len(), checkpoint);

    successful(&environment.run("advisory-evict-c", profile_c(), "hello"));
    let all_events = environment.control.events();
    let events = &all_events[checkpoint..];
    assert!(
        event_index(events, "model-destroyed", &artifact_a())
            < event_index(events, "load-entered", &artifact_c())
    );
    assert!(
        !events
            .iter()
            .any(|event| event.kind == "model-destroyed" && event.artifact == artifact_b())
    );
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 3);
}

#[test]
fn pool_advisory_does_not_credit_any_pinned_model() {
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("advisory-protect-a", profile_a(), "hello"));
    let retained_a = environment.snapshot().reserved_bytes();
    let a = pool_instance(&environment, &artifact_a());
    let _pin_a = block_on(environment.runtime.pin_model(a)).unwrap().unwrap();
    successful(&environment.run("advisory-protect-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let _pin_b = block_on(environment.runtime.pin_model(b)).unwrap().unwrap();
    let free = startup_budget(&environment.plan_c)
        .checked_sub(retained_a)
        .unwrap();
    assert!(free < pending_startup_reservation(&environment.plan_c));
    let blocking = environment
        .snapshot()
        .available_bytes()
        .checked_sub(free)
        .unwrap();
    let _blocker = environment
        .admission
        .reserve("all pinned blocker", blocking)
        .unwrap();
    let before = environment.snapshot();
    let checkpoint = environment.control.events().len();
    let estimate = pool_advisory(&environment, profile_c());
    assert!(!estimate.fits());
    assert_eq!(estimate.available_bytes(), before.available_bytes());
    assert_eq!(environment.snapshot(), before);
    assert_eq!(environment.control.events().len(), checkpoint);
}

// Returns with B blocked inside prefill, zero idle TTL, and no explicit
// model pin remaining. Only active execution ownership protects B.
fn pool_block_active_b(
    name: &str,
) -> (
    PolicyEnvironment,
    Generation,
    GateRelease,
    usize,
    AdmissionSnapshot,
) {
    let environment = PolicyEnvironment::new_pool(false);

    successful(&environment.run(&format!("{name}-warm-a"), profile_a(), "hello"));
    successful(&environment.run(&format!("{name}-warm-b"), profile_b(), "hello"));

    let b = pool_instance(&environment, &artifact_b());
    let setup_pin = block_on(environment.runtime.pin_model(b.clone()))
        .unwrap()
        .expect("B is loaded");

    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(b, Some(Duration::ZERO),)
        )
        .unwrap(),
    );

    // Leave A selected and idle, with protected B in the idle pool.
    successful(&environment.run(&format!("{name}-touch-a"), profile_a(), "hello"));

    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        name,
        profile_b(),
        "jello",
    )))
    .expect("B generation accepted");

    environment.control.execution_gate.wait_entered();

    // No explicit residency protection remains after this point.
    drop(setup_pin);

    assert!(environment.scheduler.is_busy().unwrap());
    assert_eq!(
        environment.engine.active_request_id().unwrap(),
        Some(RequestId::new(name).unwrap()),
    );

    let checkpoint = environment.control.events().len();
    let accounting = environment.snapshot();

    assert!(accounting.reserved_bytes() > 0);
    assert!(
        accounting.reserved_bytes()
            >= environment.plan_b.load_limits.max_resident_bytes()
                + environment.plan_b.cache.max_cache_bytes(),
        "B's native model and cache must remain reserved",
    );

    (environment, generation, release, checkpoint, accounting)
}

fn assert_no_b_destruction(events: &[PolicyEvent]) {
    assert!(
        !events.iter().any(|event| event.artifact == artifact_b()
            && matches!(event.kind, "cache-destroyed" | "model-destroyed")),
        "active B must retain its native model and cache",
    );
}

fn assert_b_destruction_after_prefill_return(events: &[PolicyEvent]) {
    let returned = event_index(events, "prefill-returned", &artifact_b());
    for (index, event) in events.iter().enumerate() {
        if event.artifact == artifact_b()
            && matches!(event.kind, "cache-destroyed" | "model-destroyed")
        {
            assert!(
                index > returned,
                "B's resources were destroyed before its native call returned",
            );
        }
    }
}

#[test]
fn pool_active_generation_survives_expiry_until_cleanup() {
    let (environment, generation, release, checkpoint, accounting) =
        pool_block_active_b("pool-active-normal");

    // Submit a real allocator-reclamation request while the worker is
    // blocked. It cannot run concurrently with the native operation.
    let mut reclaim = environment.runtime.reclaim_mlx_allocator_cache();
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    assert!(matches!(reclaim.as_mut().poll(&mut cx), Poll::Pending));

    assert_eq!(environment.snapshot(), accounting);
    assert!(environment.scheduler.is_busy().unwrap());
    let events = environment.control.events();
    assert_no_b_destruction(&events[checkpoint..]);

    release.release();

    let mut stream = generation.into_events();
    let transcript = drain(&mut stream);
    drop(stream);
    environment.wait_idle();
    successful(&transcript);

    // Depending on command ordering, the request either sees active
    // ownership and returns Busy, or runs after native cleanup.
    match block_on(reclaim) {
        Ok(_) => {}
        Err(error) => assert_eq!(error.kind(), ErrorKind::Busy),
    }

    // Synchronize with subsequent worker maintenance.
    let _ = block_on(environment.runtime.model_residency()).unwrap();
    let events = environment.control.events();
    assert_b_destruction_after_prefill_return(&events[checkpoint..]);
    assert!(!environment.scheduler.is_busy().unwrap());
}

#[test]
fn pool_active_cancellation_keeps_resources_until_native_return() {
    let name = "pool-active-cancel";
    let (environment, generation, release, checkpoint, accounting) = pool_block_active_b(name);

    assert_eq!(
        block_on(environment.engine.cancel(RequestId::new(name).unwrap())).unwrap(),
        CancelOutcome::Requested,
    );

    // Cancellation is a signal, not completion or permission to reclaim.
    assert!(environment.scheduler.is_busy().unwrap());
    assert_eq!(
        environment.engine.active_request_id().unwrap(),
        Some(RequestId::new(name).unwrap()),
    );
    assert_eq!(environment.snapshot(), accounting);

    let events = environment.control.events();
    assert_no_b_destruction(&events[checkpoint..]);

    release.release();

    let mut stream = generation.into_events();
    let limit = deadline();
    let mut failures = 0;
    let mut completions = 0;

    while let Some(event) = next_event(&mut stream, limit) {
        match event {
            GenerationEvent::RequestFailed { error, .. } => {
                assert!(
                    matches!(error, EngineError::Cancelled { .. }),
                    "expected cancellation, got {error:?}",
                );
                failures += 1;
            }
            GenerationEvent::GenerationCompleted { .. } => completions += 1,
            _ => {}
        }
    }

    drop(stream);
    environment.wait_idle();

    assert_eq!(failures, 1, "exactly one cancellation terminal");
    assert_eq!(completions, 0);
    let _ = block_on(environment.runtime.model_residency()).unwrap();
    let events = environment.control.events();
    assert_b_destruction_after_prefill_return(&events[checkpoint..]);
    assert!(!environment.scheduler.is_busy().unwrap());
}

fn check_pool_pending_startup(gate_stage: usize, abandon_stream: bool) {
    assert!(matches!(gate_stage, 1 | 2));
    let environment = PolicyEnvironment::new_pool(false);

    successful(&environment.run("pending-protect-warm-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());

    // Set zero default TTL for incoming B without accidentally expiring A.
    let setup_pin = block_on(environment.runtime.pin_model(a.clone()))
        .unwrap()
        .expect("A is loaded");
    let _ = environment
        .runtime
        .clone()
        .with_model_pool(Some(Duration::ZERO))
        .expect("zero default idle TTL");
    assert!(block_on(environment.runtime.set_model_idle_ttl(a.clone(), None)).unwrap(),);
    drop(setup_pin);

    let baseline = environment.snapshot();
    let checkpoint = environment.control.events().len();
    environment
        .control
        .startup_gate_stage
        .store(gate_stage, Ordering::Release);
    let release = environment.control.startup_gate.arm();

    let name = match (gate_stage, abandon_stream) {
        (1, false) => "pending-cancel-before-load",
        (1, true) => "pending-abandon-before-load",
        (2, false) => "pending-cancel-after-load",
        _ => unreachable!(),
    };
    let id = RequestId::new(name).unwrap();
    let generation = block_on(environment.engine.generate(environment.request(
        name,
        profile_b(),
        "hello",
    )))
    .expect("B admitted");

    environment.control.startup_gate.wait_entered();

    let blocked = environment.snapshot();
    assert_eq!(
        blocked.reserved_bytes(),
        sum(&[
            baseline.reserved_bytes(),
            pending_startup_reservation(&environment.plan_b),
        ]),
        "A plus B's full startup grant must remain charged",
    );
    assert!(environment.scheduler.is_busy().unwrap());
    assert_eq!(
        environment.engine.active_request_id().unwrap(),
        Some(id.clone()),
    );

    let events = environment.control.events();
    let startup_events = &events[checkpoint..];
    event_index(startup_events, "load-entered", &artifact_b());
    assert_eq!(
        startup_events
            .iter()
            .any(|event| event.artifact == artifact_b() && event.kind == "load-constructed"),
        gate_stage == 2,
    );
    assert!(
        !startup_events
            .iter()
            .any(|event| matches!(event.kind, "cache-destroyed" | "model-destroyed")),
        "neither retained A nor startup-owned B may be destroyed",
    );

    let mut stream = Some(generation.into_events());
    if abandon_stream {
        drop(stream.take());
    } else {
        assert_eq!(
            block_on(environment.engine.cancel(id.clone())).unwrap(),
            CancelOutcome::Requested,
        );
    }

    // Dropping the consumer or requesting cancellation is not cleanup.
    assert!(environment.scheduler.is_busy().unwrap());
    assert_eq!(environment.engine.active_request_id().unwrap(), Some(id),);
    assert_eq!(environment.snapshot(), blocked);
    let events = environment.control.events();
    assert!(
        !events[checkpoint..]
            .iter()
            .any(|event| matches!(event.kind, "cache-destroyed" | "model-destroyed")),
    );

    // Reclamation is submitted while native startup is still blocked.
    let mut reclaim = environment.runtime.reclaim_mlx_allocator_cache();
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    assert!(matches!(reclaim.as_mut().poll(&mut cx), Poll::Pending));
    assert_eq!(environment.snapshot(), blocked);
    assert!(environment.scheduler.is_busy().unwrap());

    release.release();

    if let Some(mut stream) = stream {
        let limit = deadline();
        let mut failures = 0;
        let mut completions = 0;
        while let Some(event) = next_event(&mut stream, limit) {
            match event {
                GenerationEvent::RequestFailed { error, .. } => {
                    assert!(
                        matches!(error, EngineError::Cancelled { .. }),
                        "expected cancellation, got {error:?}",
                    );
                    failures += 1;
                }
                GenerationEvent::GenerationCompleted { .. } => completions += 1,
                _ => {}
            }
        }
        drop(stream);
        assert_eq!(failures, 1);
        assert_eq!(completions, 0);
    }

    environment.wait_idle();
    match block_on(reclaim) {
        Ok(_) => {}
        Err(error) => assert_eq!(error.kind(), ErrorKind::Busy),
    }

    // Serialize behind cleanup and the worker's zero-TTL maintenance.
    let residents = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(pool_instance(&environment, &artifact_a()), a);
    assert!(
        !residents
            .iter()
            .any(|resident| resident.artifact == artifact_b()),
        "cancelled B must not leave a zero-TTL resident behind",
    );
    assert_eq!(
        environment.snapshot(),
        baseline,
        "B's output, workspace, loading peak and unused grant must be released",
    );

    let events = environment.control.events();
    let startup_events = &events[checkpoint..];
    assert!(
        !startup_events
            .iter()
            .any(|event| event.artifact == artifact_a()
                && matches!(event.kind, "cache-destroyed" | "model-destroyed")),
        "unrelated retained A must survive startup cleanup",
    );

    let returned = event_index(startup_events, "load-gate-returned", &artifact_b());
    if gate_stage == 2 {
        let destroyed = event_index(startup_events, "model-destroyed", &artifact_b());
        assert!(
            destroyed > returned,
            "constructed B may be destroyed only after the loader gate returns",
        );
    }

    environment
        .control
        .startup_gate_stage
        .store(0, Ordering::Release);
    successful(&environment.run("pending-recovery-b", profile_b(), "hello"));
    assert_eq!(pool_instance(&environment, &artifact_a()), a);
    let _ = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(environment.snapshot(), baseline);
}

#[test]
fn pool_pending_startup_cancel_before_load() {
    check_pool_pending_startup(1, false);
}

#[test]
fn pool_pending_startup_abandoned_stream_before_load() {
    check_pool_pending_startup(1, true);
}

#[test]
fn pool_pending_startup_cancel_after_native_construction() {
    check_pool_pending_startup(2, false);
}

// M9_ADMISSION_ENFORCER_BOUNDARIES_V1
struct PoolStartupWake {
    sender: mpsc::Sender<usize>,
    epoch: usize,
}

impl Wake for PoolStartupWake {
    fn wake(self: Arc<Self>) {
        let _ = self.sender.send(self.epoch);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.sender.send(self.epoch);
    }
}

struct PoolAdmissionReady {
    receiver: mpsc::Receiver<usize>,
    epoch: usize,
}

impl PoolAdmissionReady {
    fn recv_timeout(&self, timeout: Duration) -> Result<(), mpsc::RecvTimeoutError> {
        let limit = Instant::now() + timeout;
        loop {
            let remaining = limit.saturating_duration_since(Instant::now());
            let epoch = self.receiver.recv_timeout(remaining)?;
            if epoch == self.epoch {
                return Ok(());
            }
        }
    }
}

fn pool_unconsumed_startup<'a>(
    environment: &'a PolicyEnvironment,
    name: &str,
) -> (
    engine::api::EngineFuture<'a, Generation>,
    GateRelease,
    PoolAdmissionReady,
) {
    environment
        .control
        .startup_gate_stage
        .store(0, Ordering::Release);
    let release = environment.control.startup_gate.arm();
    let mut startup = environment
        .engine
        .generate(environment.request(name, profile_b(), "hello"));
    let (sender, receiver) = mpsc::channel();
    let mut epoch = 0;
    // Every poll gets a distinct waker epoch. A delayed advisory wake
    // cannot be mistaken for publication of the admitted startup reply.
    thread::scope(|scope| {
        let gate = Arc::clone(&environment.control.startup_gate);
        let gate_sender = sender.clone();
        scope.spawn(move || {
            gate.wait_entered();
            let _ = gate_sender.send(usize::MAX);
        });
        loop {
            if environment
                .control
                .startup_gate
                .state
                .lock()
                .unwrap()
                .entered
            {
                break;
            }
            epoch += 1;
            let waker = Waker::from(Arc::new(PoolStartupWake {
                sender: sender.clone(),
                epoch,
            }));
            let mut cx = Context::from_waker(&waker);
            assert!(startup.as_mut().poll(&mut cx).is_pending());
            if environment
                .control
                .startup_gate
                .state
                .lock()
                .unwrap()
                .entered
            {
                break;
            }
            receiver
                .recv_timeout(TEST_TIMEOUT)
                .expect("startup or gate wake");
        }
    });
    while receiver.try_recv().is_ok() {}
    (startup, release, PoolAdmissionReady { receiver, epoch })
}

fn assert_pool_only_a(environment: &PolicyEnvironment, a: &ModelInstanceId) {
    let residents = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(residents.len(), 1);
    assert!(residents[0].instance.same_instance(a));
}

#[test]
fn pool_pending_generate_drop_before_grant_preserves_idle_a() {
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("drop-before-warm-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let baseline = environment.snapshot();
    let checkpoint = environment.control.events().len();
    let (startup, release, _receiver) = pool_unconsumed_startup(&environment, "drop-before-grant");
    assert_eq!(environment.snapshot(), baseline);
    drop(startup);
    assert!(environment.scheduler.is_busy().unwrap());
    assert_eq!(environment.snapshot(), baseline);
    release.release();
    environment.wait_idle();
    assert_eq!(environment.snapshot(), baseline);
    assert_pool_only_a(&environment, &a);
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| !matches!(
                event.kind,
                "load-entered" | "model-destroyed" | "cache-destroyed"
            ))
    );
    successful(&environment.run("drop-before-recover", profile_b(), "hello"));
}

#[test]
fn pool_pending_generate_drop_after_grant_releases_unconsumed_reply() {
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("drop-reply-warm-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let baseline = environment.snapshot();
    let checkpoint = environment.control.events().len();
    let (startup, release, receiver) =
        pool_unconsumed_startup(&environment, "drop-unconsumed-reply");
    release.release();
    let expected = baseline.reserved_bytes() + pending_startup_reservation(&environment.plan_b);
    loop {
        receiver
            .recv_timeout(TEST_TIMEOUT)
            .expect("admission reply wake");
        if environment.snapshot().reserved_bytes() == expected {
            break;
        }
    }
    // Never repoll generate: that would consume the reply and start prefill.
    assert!(environment.scheduler.is_busy().unwrap());
    let error = expect_failure(
        block_on(environment.runtime.reclaim_mlx_allocator_cache()),
        "pending startup must reject allocator reclamation",
    );
    assert_eq!(error.kind(), ErrorKind::Busy);
    assert_eq!(environment.snapshot().reserved_bytes(), expected);
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
    drop(startup);
    environment.wait_idle();
    assert_eq!(environment.snapshot(), baseline);
    assert_pool_only_a(&environment, &a);
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| !matches!(
                event.kind,
                "load-entered" | "model-destroyed" | "cache-destroyed"
            ))
    );
    successful(&environment.run("drop-reply-recover", profile_b(), "hello"));
}

#[derive(Debug)]
struct PoolPressureSource(Mutex<engine::memory_enforcer::MemoryObservation>);

impl engine::memory_enforcer::MemoryObservationSource for PoolPressureSource {
    fn observe(
        &self,
    ) -> Result<engine::memory_enforcer::MemoryObservation, engine::memory_enforcer::EnforcerError>
    {
        Ok(*self.0.lock().unwrap())
    }
}

impl PoolPressureSource {
    fn set_normal(&self) {
        let mut observation = self.0.lock().unwrap();
        observation.host.free_bytes = 20 * 1024 * 1024 * 1024;
        observation.host.phys_footprint_bytes = 1024 * 1024 * 1024;
        observation.mlx_active_bytes = 1024 * 1024 * 1024;
        observation.mlx_pool_bytes = 0;
    }
}

fn pool_pressure_enforcer(
    admission: AdmissionController,
    hard: bool,
) -> (
    engine::memory_enforcer::MemoryEnforcer,
    Arc<PoolPressureSource>,
) {
    use engine::memory_enforcer::{MemoryEnforcer, MemoryObservation};
    use engine::memory_guard::{HostMemorySnapshot, MemoryGuardConfig};
    const GIB: u64 = 1024 * 1024 * 1024;
    let usage = if hard { 32 * GIB } else { 20 * GIB };
    let source = Arc::new(PoolPressureSource(Mutex::new(MemoryObservation {
        host: HostMemorySnapshot {
            physical_memory_bytes: 36 * GIB,
            free_bytes: 5 * GIB,
            inactive_bytes: 0,
            active_file_bytes: 0,
            other_apps_active_bytes: 0,
            phys_footprint_bytes: usage,
            graphics_footprint_bytes: 0,
            iogpu_wired_limit_bytes: 0,
            recommended_working_set_bytes: 100 * GIB,
            release_credit_bytes: 0,
        },
        mlx_active_bytes: usage,
        mlx_pool_bytes: 3 * GIB,
    })));
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::balanced(),
        admission,
        source.clone(),
    );
    (enforcer, source)
}

fn pool_pressure_after_grace(
    environment: &PolicyEnvironment,
    enforcer: &engine::memory_enforcer::MemoryEnforcer,
) -> engine::runtime::MemoryEnforcementReport {
    use engine::memory_enforcer::PRESSURE_RECLAIM_GRACE_POLLS_MAX;
    use engine::runtime::IdleModelReclaim;
    for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        let report = block_on(environment.runtime.enforce_memory_tick(enforcer.clone())).unwrap();
        assert!(report.tick.deferred_for_pool_drain);
        assert!(matches!(report.model, IdleModelReclaim::NotRequested));
    }
    let report = block_on(environment.runtime.enforce_memory_tick(enforcer.clone())).unwrap();
    assert!(!report.tick.deferred_for_pool_drain);
    report
}

#[test]
fn pool_enforcer_grace_then_actual_lru_eviction_and_recovery() {
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("enforcer-warm-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let a_bytes = environment.snapshot().reserved_bytes();
    successful(&environment.run("enforcer-warm-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let before = environment.snapshot().reserved_bytes();
    let checkpoint = environment.control.events().len();
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    let report = pool_pressure_after_grace(&environment, &enforcer);
    match report.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected actual A eviction: {other:?}"),
    }
    assert_eq!(environment.snapshot().reserved_bytes(), before - a_bytes);
    let events = environment.control.events();
    let events = &events[checkpoint..];
    let cache = event_index(events, "cache-destroyed", &artifact_a());
    let model = event_index(events, "model-destroyed", &artifact_a());
    assert!(cache < model);
    assert!(
        events[model].snapshot.reserved_bytes() > environment.snapshot().reserved_bytes(),
        "model guard must still be charged when native destruction is recorded"
    );
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    source.set_normal();
    let report = block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    assert!(matches!(report.model, IdleModelReclaim::NotRequested));
    assert!(!environment.admission.admissions_paused());
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    successful(&environment.run("enforcer-recover-b", profile_b(), "hello"));
}

#[test]
fn pool_enforcer_all_pins_report_no_eligible_victim() {
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("enforcer-pinned-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let pin_a = block_on(environment.runtime.pin_model(a.clone()))
        .unwrap()
        .unwrap();
    successful(&environment.run("enforcer-pinned-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let _pin_b = block_on(environment.runtime.pin_model(b.clone()))
        .unwrap()
        .unwrap();
    let before = environment.snapshot();
    let (enforcer, _) = pool_pressure_enforcer(environment.admission.clone(), false);
    let report = pool_pressure_after_grace(&environment, &enforcer);
    assert!(matches!(report.model, IdleModelReclaim::NoEligibleVictim));
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        before.reserved_bytes()
    );
    assert_eq!(
        environment.snapshot().active_reservations(),
        before.active_reservations()
    );
    drop(pin_a);
    let report = block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    match report.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected unpinned A eviction: {other:?}"),
    }
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
}

#[test]
fn pool_enforcer_evicts_a_without_reclaiming_pending_b_grant() {
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("enforcer-pending-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let baseline = environment.snapshot().reserved_bytes();
    let (startup, release, receiver) = pool_unconsumed_startup(&environment, "enforcer-pending-b");
    release.release();
    let pending = pending_startup_reservation(&environment.plan_b);
    loop {
        receiver
            .recv_timeout(TEST_TIMEOUT)
            .expect("admission reply wake");
        if environment.snapshot().reserved_bytes() == baseline + pending {
            break;
        }
    }
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    let report = pool_pressure_after_grace(&environment, &enforcer);
    match report.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected idle A eviction: {other:?}"),
    }
    assert!(environment.scheduler.is_busy().unwrap());
    assert_eq!(environment.snapshot().reserved_bytes(), pending);
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
    drop(startup);
    environment.wait_idle();
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert_eq!(environment.snapshot().active_reservations(), 0);
    source.set_normal();
    block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    successful(&environment.run("enforcer-pending-recover", profile_b(), "hello"));
}

#[test]
fn pool_enforcer_evicts_idle_a_but_preserves_active_b() {
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool_with_event_capacity(1);
    successful(&environment.run("enforcer-active-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    successful(&environment.run("enforcer-active-warm-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let checkpoint = environment.control.events().len();
    let generation = block_on(environment.engine.generate(environment.request(
        "enforcer-active-b",
        profile_b(),
        "jello",
    )))
    .unwrap();
    // A capacity-one event channel is not drained. Its producer cannot
    // finish the four-token generation while this test inspects ownership.
    let limit = deadline();
    loop {
        if environment.control.events()[checkpoint..]
            .iter()
            .any(|event| event.artifact == artifact_b() && event.kind == "prefill-returned")
        {
            break;
        }
        assert!(Instant::now() < limit, "B did not reach prefill");
        thread::yield_now();
    }
    assert!(environment.scheduler.is_busy().unwrap());
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    let report = pool_pressure_after_grace(&environment, &enforcer);
    match report.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected idle A eviction: {other:?}"),
    }
    assert!(environment.scheduler.is_busy().unwrap());
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    let report = block_on(environment.runtime.enforce_memory_tick(enforcer.clone())).unwrap();
    assert!(matches!(report.model, IdleModelReclaim::NoEligibleVictim));
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| event.artifact != artifact_b()
                || !matches!(event.kind, "cache-destroyed" | "model-destroyed"))
    );
    source.set_normal();
    block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    let mut stream = generation.into_events();
    let transcript = drain(&mut stream);
    drop(stream);
    environment.wait_idle();
    successful(&transcript);
}

#[test]
fn pool_enforcer_hard_pressure_destroys_an_actual_idle_model() {
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("enforcer-hard-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let (enforcer, _) = pool_pressure_enforcer(environment.admission.clone(), true);
    let report = pool_pressure_after_grace(&environment, &enforcer);
    assert_eq!(
        report.tick.status.pressure,
        engine::memory_guard::PressureLevel::Hard
    );
    match report.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected hard-pressure model eviction: {other:?}"),
    }
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert_eq!(environment.snapshot().active_reservations(), 0);
}

#[test]
fn pool_enforcer_live_session_lease_blocks_model_destruction() {
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    let name = "enforcer-leased-a";
    let rendered = block_on(
        environment
            .engine
            .render_and_count(request_context(profile_a(), "hello")),
    )
    .unwrap();
    successful(&environment.run(name, profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    // The fixture executes a real zero-head checkpoint and emits token 0
    // for each output position (the successful transcript is "!!!!").
    let tokenizer = tokenizers::Tokenizer::from_file(
        environment._fixtures[0]
            ._directory
            .path()
            .join("tokenizer.json"),
    )
    .unwrap();
    let mut tokens = rendered.token_ids().to_vec();
    tokens.extend_from_slice(&[0; POLICY_OUTPUT]);
    let prefix = tokenizer.decode(&tokens, false).unwrap();
    assert_eq!(
        tokenizer.encode(prefix.as_str(), false).unwrap().get_ids(),
        &tokens
    );
    let state = block_on(
        environment
            .runtime
            .retain_session_state(RequestId::new(name).unwrap(), prefix.into_bytes()),
    )
    .unwrap();
    let lease = block_on(environment.runtime.lease_session_state(state))
        .unwrap()
        .unwrap();
    let before = environment.snapshot();
    let checkpoint = environment.control.events().len();
    let (enforcer, _) = pool_pressure_enforcer(environment.admission.clone(), false);
    let report = pool_pressure_after_grace(&environment, &enforcer);
    assert!(matches!(report.model, IdleModelReclaim::NoEligibleVictim));
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        before.reserved_bytes()
    );
    assert_eq!(
        environment.snapshot().active_reservations(),
        before.active_reservations()
    );
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| !matches!(event.kind, "cache-destroyed" | "model-destroyed"))
    );
    drop(lease);
    let report = block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    match report.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected unleased A eviction: {other:?}"),
    }
    assert!(
        block_on(environment.runtime.lookup_session_state(state))
            .unwrap()
            .is_none()
    );
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert_eq!(environment.snapshot().active_reservations(), 0);
}

#[test]
fn pool_enforcer_queued_eviction_is_revoked_after_pressure_recovers() {
    use engine::memory_enforcer::PRESSURE_RECLAIM_GRACE_POLLS_MAX;
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("enforcer-stale-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    successful(&environment.run("enforcer-stale-warm-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let checkpoint = environment.control.events().len();
    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        "enforcer-stale-b",
        profile_b(),
        "jello",
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        assert!(enforcer.tick().unwrap().deferred_for_pool_drain);
    }
    let mut enforcement = environment.runtime.enforce_memory_tick(enforcer);
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    assert!(enforcement.as_mut().poll(&mut cx).is_pending());
    assert!(
        environment.admission.admissions_paused(),
        "admission pause must not wait for the blocked native worker"
    );
    source.set_normal();
    release.release();
    let report = block_on(enforcement).unwrap();
    assert!(matches!(report.model, IdleModelReclaim::NotRequested));
    assert!(!environment.admission.admissions_paused());
    let mut stream = generation.into_events();
    let transcript = drain(&mut stream);
    drop(stream);
    environment.wait_idle();
    successful(&transcript);
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| !matches!(event.kind, "cache-destroyed" | "model-destroyed"))
    );
}

// M9_INSTANCE_SCOPED_LEASE_EXECUTION_V1
fn instance_lease_publish(
    environment: &PolicyEnvironment,
    name: &str,
    profile: ModelProfileId,
) -> (
    engine::types::SessionStateId,
    engine::runtime::RetainedSessionState,
) {
    let rendered = block_on(
        environment
            .engine
            .render_and_count(request_context(profile.clone(), "hello")),
    )
    .unwrap();
    successful(&environment.run(name, profile.clone(), "hello"));
    let generated = environment
        .control
        .instance_lease_tokens
        .lock()
        .unwrap()
        .get(&RequestId::new(name).unwrap())
        .cloned()
        .expect("actual decoded token IDs");
    assert_eq!(generated.len(), POLICY_OUTPUT);
    let mut tokens = rendered.token_ids().to_vec();
    tokens.extend_from_slice(&generated);
    let artifact = environment
        .engine
        .registry()
        .resolve_profile(&profile)
        .unwrap()
        .artifact()
        .id()
        .clone();
    let index = if artifact == artifact_a() {
        0
    } else if artifact == artifact_b() {
        1
    } else {
        2
    };
    let tokenizer = tokenizers::Tokenizer::from_file(
        environment._fixtures[index]
            ._directory
            .path()
            .join("tokenizer.json"),
    )
    .unwrap();
    let prefix = tokenizer.decode(&tokens, false).unwrap();
    assert_eq!(
        tokenizer.encode(prefix.as_str(), false).unwrap().get_ids(),
        tokens.as_slice()
    );
    let id = block_on(
        environment
            .runtime
            .retain_session_state(RequestId::new(name).unwrap(), prefix.into_bytes()),
    )
    .unwrap();
    let state = block_on(environment.runtime.lookup_session_state(id))
        .unwrap()
        .unwrap();
    (id, state)
}

fn instance_lease_assert_unchanged(
    environment: &PolicyEnvironment,
    expected: &engine::runtime::RetainedSessionState,
) {
    let actual = block_on(environment.runtime.lookup_session_state(expected.state_id))
        .unwrap()
        .expect("leased canonical publication remains present");
    assert_eq!(
        format!("{actual:?}"),
        format!("{expected:?}"),
        "every canonical metadata field must remain unchanged"
    );
    assert!(
        pool_instance(environment, &expected.artifact_id)
            .same_instance(&expected.model_instance_id)
    );
    let minimum = expected.weights_reserved_bytes
        + expected.cache_reserved_bytes
        + expected.token_history_reserved_bytes
        + expected.logits_reserved_bytes;
    assert!(environment.snapshot().reserved_bytes() >= minimum);
}

#[test]
fn pool_instance_leases_b_executes_and_publishes_while_a_is_leased() {
    let environment = PolicyEnvironment::new_pool(false);
    let (a_id, a) = instance_lease_publish(&environment, "scoped-publish-a", profile_a());
    let first = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let second = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let a_accounting = environment.snapshot();
    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(a.model_instance_id.clone(), Some(Duration::ZERO),)
        )
        .unwrap()
    );
    let (b_id, b) = instance_lease_publish(&environment, "scoped-publish-b", profile_b());
    instance_lease_assert_unchanged(&environment, &a);
    assert_ne!(a_id, b_id);
    assert!(block_on(environment.engine.capabilities(artifact_a())).is_ok());
    assert!(block_on(environment.engine.capabilities(artifact_b())).is_ok());
    let b_bytes = b.weights_reserved_bytes
        + b.cache_reserved_bytes
        + b.token_history_reserved_bytes
        + b.logits_reserved_bytes;
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        a_accounting.reserved_bytes() + b_bytes
    );
    let error = expect_failure(
        block_on(environment.engine.generate(environment.request(
            "scoped-deny-a",
            profile_a(),
            "jello",
        ))),
        "generation must not mutate leased A",
    );
    assert_eq!(error.kind(), ErrorKind::Busy);
    instance_lease_assert_unchanged(&environment, &a);
    assert_eq!(
        block_on(environment.engine.release_session_state(b_id)).unwrap(),
        engine::types::ReleaseOutcome::Released
    );
    assert_eq!(environment.snapshot(), a_accounting);
    drop(first);
    instance_lease_assert_unchanged(&environment, &a);
    drop(second);
    let limit = deadline();
    while block_on(environment.runtime.lookup_session_state(a_id))
        .unwrap()
        .is_some()
    {
        assert!(Instant::now() < limit);
        thread::yield_now();
    }
    environment.assert_empty();
}

#[test]
fn pool_instance_leases_cancel_b_preserves_a_and_its_accounting() {
    let environment = PolicyEnvironment::new_pool(false);
    let (a_id, a) = instance_lease_publish(&environment, "scoped-cancel-a", profile_a());
    let lease = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let a_accounting = environment.snapshot();
    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        "scoped-cancel-b",
        profile_b(),
        "hello",
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let blocked = environment.snapshot();
    assert!(
        blocked.reserved_bytes()
            >= a_accounting.reserved_bytes()
                + environment.plan_b.load_limits.max_resident_bytes()
                + environment.plan_b.cache.max_cache_bytes()
    );
    assert_eq!(
        block_on(
            environment
                .engine
                .cancel(RequestId::new("scoped-cancel-b").unwrap())
        )
        .unwrap(),
        CancelOutcome::Requested
    );
    assert_eq!(environment.snapshot(), blocked);
    assert!(environment.scheduler.is_busy().unwrap());
    release.release();
    let mut stream = generation.into_events();
    let mut failures = 0;
    while let Some(event) = next_event(&mut stream, deadline()) {
        match event {
            GenerationEvent::RequestFailed { error, .. } => {
                assert!(matches!(error, EngineError::Cancelled { .. }));
                failures += 1;
            }
            GenerationEvent::GenerationCompleted { .. } => panic!("cancelled B completed"),
            _ => {}
        }
    }
    drop(stream);
    environment.wait_idle();
    assert_eq!(failures, 1);
    instance_lease_assert_unchanged(&environment, &a);
    let b = pool_instance(&environment, &artifact_b());
    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(b, Some(Duration::ZERO))
        )
        .unwrap()
    );
    let limit = deadline();
    while block_on(environment.runtime.model_residency())
        .unwrap()
        .iter()
        .any(|resident| resident.artifact == artifact_b())
    {
        assert!(Instant::now() < limit);
        thread::yield_now();
    }
    assert_eq!(environment.snapshot(), a_accounting);
    drop(lease);
    assert_eq!(
        block_on(environment.engine.release_session_state(a_id)).unwrap(),
        engine::types::ReleaseOutcome::Released
    );
    environment.assert_empty();
}

#[test]
fn pool_instance_leases_failed_b_startup_does_not_invalidate_a() {
    let environment = PolicyEnvironment::new_pool(false);
    let (a_id, a) = instance_lease_publish(&environment, "scoped-failure-a", profile_a());
    let lease = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let accounting = environment.snapshot();
    environment
        .control
        .fail_loading_once
        .store(true, Ordering::Release);
    let generation = block_on(environment.engine.generate(environment.request(
        "scoped-failure-b",
        profile_b(),
        "hello",
    )))
    .unwrap();
    let mut stream = generation.into_events();
    let mut failures = 0;
    while let Some(event) = next_event(&mut stream, deadline()) {
        match event {
            GenerationEvent::RequestFailed { error, .. } => {
                assert!(matches!(error, EngineError::ExecutionFailed { .. }));
                failures += 1;
            }
            GenerationEvent::GenerationCompleted { .. } => panic!("injected failed load completed"),
            _ => {}
        }
    }
    drop(stream);
    environment.wait_idle();
    assert_eq!(failures, 1);
    assert_eq!(environment.snapshot(), accounting);
    instance_lease_assert_unchanged(&environment, &a);
    successful(&environment.run("scoped-failure-recovery-b", profile_b(), "hello"));
    instance_lease_assert_unchanged(&environment, &a);
    drop(lease);
}

#[test]
fn pool_instance_leases_admission_does_not_credit_leased_a() {
    let environment = PolicyEnvironment::new_pool(false);
    let (a_id, a) = instance_lease_publish(&environment, "scoped-budget-a", profile_a());
    let lease = block_on(environment.runtime.lease_session_state(a_id))
        .unwrap()
        .unwrap();
    let required = pending_startup_reservation(&environment.plan_b);
    let available = environment.snapshot().available_bytes();
    assert!(available > required);
    let blocker = environment
        .admission
        .reserve("scoped lease admission blocker", available - (required - 1))
        .unwrap();
    let accounting = environment.snapshot();
    let loads = environment.observations.loads.load(Ordering::Relaxed);
    let error = expect_failure(
        block_on(environment.engine.generate(environment.request(
            "scoped-budget-b",
            profile_b(),
            "hello",
        ))),
        "leased A cannot be credited as reclaimable memory",
    );
    assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded);
    environment.wait_idle();
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(
        environment.observations.loads.load(Ordering::Relaxed),
        loads
    );
    instance_lease_assert_unchanged(&environment, &a);
    drop(blocker);
    successful(&environment.run("scoped-budget-recovery-b", profile_b(), "hello"));
    instance_lease_assert_unchanged(&environment, &a);
    drop(lease);
}

// M9_LIVE_PRESSURE_PUMP_V2
#[test]
fn pool_pressure_pump_keeps_polling_while_native_worker_is_blocked() {
    use engine::memory_enforcer::PRESSURE_RECLAIM_GRACE_POLLS_MAX;
    use engine::memory_guard::PressureLevel;
    use engine::runtime::{IdleModelReclaim, MemoryEnforcementPump};
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("pump-observe-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    successful(&environment.run("pump-observe-warm-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let checkpoint = environment.control.events().len();
    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        "pump-observe-active-b",
        profile_b(),
        "jello",
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    let pressured = *source.0.lock().unwrap();
    source.set_normal();
    let mut pump = MemoryEnforcementPump::new(environment.engine.clone(), enforcer);
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    let first = pump.poll_tick(&mut cx).unwrap();
    assert_eq!(first.tick.status.pressure, PressureLevel::Ok);
    assert!(!environment.admission.admissions_paused());
    assert!(first.completed.is_none());
    assert!(first.tick.request_execution_cancellation.is_none());
    assert!(first.cancellation.is_none());
    assert!(pump.has_pending());
    *source.0.lock().unwrap() = pressured;
    for index in 0..=PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        let poll = pump.poll_tick(&mut cx).unwrap();
        assert_eq!(poll.tick.status.pressure, PressureLevel::Soft);
        assert!(
            environment.admission.admissions_paused(),
            "pressure pausing must continue while the native action is blocked"
        );
        assert_eq!(
            poll.tick.deferred_for_pool_drain,
            index < PRESSURE_RECLAIM_GRACE_POLLS_MAX
        );
        assert!(poll.completed.is_none());
        assert!(poll.tick.request_execution_cancellation.is_none());
        assert!(poll.cancellation.is_none());
        assert!(
            pump.has_pending(),
            "one pending action must be retained, not replaced"
        );
    }
    // Repeated ordinary hard pressure also keeps the active request running,
    // even after allocator grace and the emergency consecutive-poll count.
    {
        let mut observation = source.0.lock().unwrap();
        observation.host.phys_footprint_bytes = 32 * 1024 * 1024 * 1024;
        observation.mlx_active_bytes = observation.host.phys_footprint_bytes;
    }
    for _ in 0..=PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        let poll = pump.poll_tick(&mut cx).unwrap();
        assert_eq!(poll.tick.status.pressure, PressureLevel::Hard);
        assert!(!poll.tick.status.emergency);
        assert!(environment.admission.admissions_paused());
        assert!(poll.tick.request_execution_cancellation.is_none());
        assert!(poll.cancellation.is_none());
        assert!(poll.completed.is_none());
        assert!(pump.has_pending());
    }
    let request_id = RequestId::new("pump-observe-active-b").unwrap();
    assert_eq!(
        environment.scheduler.request_state(&request_id).unwrap(),
        engine::scheduler::RequestState::Running,
    );
    assert!(
        !environment
            .control
            .execution_control(&request_id)
            .is_cancelled()
    );
    source.set_normal();
    let recovery = pump.poll_tick(&mut cx).unwrap();
    assert_eq!(recovery.tick.status.pressure, PressureLevel::Ok);
    assert!(!environment.admission.admissions_paused());
    assert!(recovery.completed.is_none());
    release.release();
    let limit = deadline();
    loop {
        let poll = pump.poll_tick(&mut cx).unwrap();
        if let Some(result) = poll.completed {
            assert!(matches!(
                result.unwrap().model,
                IdleModelReclaim::NotRequested
            ));
            break;
        }
        assert!(Instant::now() < limit);
        thread::yield_now();
    }
    let mut stream = generation.into_events();
    let transcript = drain(&mut stream);
    drop(stream);
    environment.wait_idle();
    successful(&transcript);
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| !matches!(event.kind, "cache-destroyed" | "model-destroyed"))
    );
}

#[path = "emergency_pressure.rs"]
mod emergency_pressure;

#[derive(Debug)]
struct PoolDrainRecoverySource {
    pressured: engine::memory_enforcer::MemoryObservation,
    phase: AtomicUsize,
}

impl engine::memory_enforcer::MemoryObservationSource for PoolDrainRecoverySource {
    fn observe(
        &self,
    ) -> Result<engine::memory_enforcer::MemoryObservation, engine::memory_enforcer::EnforcerError>
    {
        let phase = self.phase.load(Ordering::Acquire);
        let mut observation = self.pressured;
        if phase != 0 && self.phase.fetch_add(1, Ordering::AcqRel) >= 3 {
            observation.host.free_bytes = 20 * 1024 * 1024 * 1024;
            observation.host.phys_footprint_bytes = 1024 * 1024 * 1024;
            observation.mlx_active_bytes = 1024 * 1024 * 1024;
            observation.mlx_pool_bytes = 0;
        }
        Ok(observation)
    }
}

#[test]
fn pool_pressure_reobserves_after_allocator_step_before_destroying_model() {
    use engine::memory_enforcer::{MemoryEnforcer, PRESSURE_RECLAIM_GRACE_POLLS_MAX};
    use engine::memory_guard::{MemoryGuardConfig, PressureLevel};
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("pump-drain-recovery-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    let before = environment.snapshot();
    let checkpoint = environment.control.events().len();
    let (_, original) = pool_pressure_enforcer(environment.admission.clone(), false);
    let source = Arc::new(PoolDrainRecoverySource {
        pressured: *original.0.lock().unwrap(),
        phase: AtomicUsize::new(0),
    });
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::balanced(),
        environment.admission.clone(),
        source.clone(),
    );
    for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        assert!(enforcer.tick().unwrap().deferred_for_pool_drain);
    }
    // Poll observation: soft. Worker entry: soft. After allocator step:
    // normal. The production executor must not destroy the now-unneeded model.
    source.phase.store(1, Ordering::Release);
    let report = block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    assert_eq!(report.tick.status.pressure, PressureLevel::Ok);
    assert!(matches!(report.model, IdleModelReclaim::NotRequested));
    assert!(!environment.admission.admissions_paused());
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        before.reserved_bytes()
    );
    assert_eq!(
        environment.snapshot().active_reservations(),
        before.active_reservations()
    );
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert_eq!(environment.control.events().len(), checkpoint);
}

#[test]
fn pool_pressure_new_grace_revokes_an_older_queued_eviction() {
    use engine::memory_enforcer::PRESSURE_RECLAIM_GRACE_POLLS_MAX;
    use engine::runtime::IdleModelReclaim;
    let environment = PolicyEnvironment::new_pool(false);
    successful(&environment.run("pump-new-grace-a", profile_a(), "hello"));
    let a = pool_instance(&environment, &artifact_a());
    successful(&environment.run("pump-new-grace-warm-b", profile_b(), "hello"));
    let b = pool_instance(&environment, &artifact_b());
    let checkpoint = environment.control.events().len();
    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        "pump-new-grace-b",
        profile_b(),
        "jello",
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    let pressured = *source.0.lock().unwrap();
    for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        assert!(enforcer.tick().unwrap().deferred_for_pool_drain);
    }
    let mut old = environment.runtime.enforce_memory_tick(enforcer.clone());
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut cx = Context::from_waker(&waker);
    assert!(old.as_mut().poll(&mut cx).is_pending());
    source.set_normal();
    assert!(!enforcer.tick().unwrap().request_model_eviction);
    *source.0.lock().unwrap() = pressured;
    let new_grace = enforcer.tick().unwrap();
    assert!(new_grace.deferred_for_pool_drain);
    assert!(!new_grace.request_model_eviction);
    release.release();
    let report = block_on(old).unwrap();
    assert!(
        matches!(report.model, IdleModelReclaim::NotRequested),
        "old post-grace authority must not bypass a newer grace window"
    );
    source.set_normal();
    enforcer.tick().unwrap();
    let mut stream = generation.into_events();
    let transcript = drain(&mut stream);
    drop(stream);
    environment.wait_idle();
    successful(&transcript);
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| !matches!(event.kind, "cache-destroyed" | "model-destroyed"))
    );
}
