//! Runtime retention, reclamation, and failure-policy integration tests.
//!
//! Loaded models, caches, verification, prefill, and decode delegate to
//! the actual Qwen backend.
//!
//! Instrumentation records native destruction and reservation snapshots.
//! Deliberate verification/loading errors exercise cleanup boundaries;
//! they do not provide synthetic model outputs.
//!
//! These tests are included as a child of tests/generation.rs and reuse
//! that file's actual safetensors/tokenizer fixtures.

use super::*;
use engine::{admission::AdmissionSnapshot, types::CacheRepresentationId};
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
    fail_verification_once: AtomicBool,
    fail_loading_once: AtomicBool,
    startup_gate: Arc<StartupGate>,
}

impl PolicyControl {
    fn new(admission: AdmissionController) -> Self {
        Self {
            admission,
            events: Mutex::new(Vec::new()),
            fail_verification_once: AtomicBool::new(false),
            fail_loading_once: AtomicBool::new(false),
            startup_gate: Arc::new(StartupGate::default()),
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
        self.inner_mut().prefill(tokens, cache, execution)
    }

    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        execution: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        self.inner_mut().decode(token, cache, execution)
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
        let model = self.inner.load(request)?;

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
        self.control.startup_gate.enter_if_armed();
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
    engine: EngineCore<FamilyRuntime>,
    admission: AdmissionController,
    scheduler: Scheduler,
    control: Arc<PolicyControl>,
    observations: Arc<Observations>,
    plan_a: RuntimePlan,
    plan_b: RuntimePlan,
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
        let fixture_a = TestEnvironment::new(POLICY_OUTPUT, 8, false);
        let fixture_b = TestEnvironment::new(POLICY_OUTPUT, 8, false);
        let context_limit = Qwen3Config::tiny_fixture().max_context;
        let generation_limits = GenerationLimits::new(8, 4096, 1024).expect("generation limits");

        let inner = Qwen3Backend::new(
            vec![
                fixture_specification(&fixture_a, artifact_a()),
                fixture_specification(&fixture_b, artifact_b()),
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
        let plan_large = plan_for(profile_large_cache());

        let budget = if tight_budget {
            startup_budget(&plan_a)
                .max(startup_budget(&plan_b))
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

        Self {
            engine: EngineCore::new(registry, scheduler.clone(), runtime),
            admission,
            scheduler,
            control,
            observations,
            plan_a,
            plan_b,
            plan_large,
            _fixtures: vec![fixture_a, fixture_b],
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
