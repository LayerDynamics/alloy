//! Canonical runtime-state publication, lease, and release integration tests.
//!
//! Included as a child of tests/generation.rs.
//!
//! Uses the parent's actual safetensors/tokenizer fixture, the real Qwen
//! adapter, FamilyRuntime, EngineCore::generate, and shared admission.
//!
//! Instrumentation observes actual model/cache destruction. It does not
//! provide model outputs or simulate retained state.
//!
//! Canonical prefix bytes come from the fixture's actual tokenizer decode
//! with special tokens preserved, followed by an exact token-ID roundtrip
//! check against the input plus actual generated tokens.
//!
//! That byte convention is specific to this lossless ByteLevel fixture.
//! These tests do not establish a universal production-tokenizer prefix
//! byte convention or add provenance validation to the runtime API.

use super::*;
use engine::{
    admission::AdmissionSnapshot,
    cache_identity::{CacheIdentityInputs, compute_session_state_id},
    error::ProtectedResource,
    runtime::RetainedSessionState,
    types::{ReleaseOutcome, SessionStateId},
};
use std::{
    any::Any,
    sync::{Condvar, mpsc},
};

#[derive(Default)]
struct GateState {
    armed: bool,
    entered: bool,
    released: bool,
}

/// Blocks one cache-memory inspection on the actual native worker.
#[derive(Default)]
struct MemoryGate {
    state: Mutex<GateState>,
    changed: Condvar,
}

impl MemoryGate {
    fn arm(self: &Arc<Self>) -> GateRelease {
        let mut state = self.state.lock().expect("memory gate");
        assert!(!state.armed, "memory gate already armed");

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
        let mut state = self.state.lock().expect("memory gate");
        if !state.armed {
            return;
        }

        state.entered = true;
        self.changed.notify_all();

        let deadline = deadline();

        while !state.released {
            let now = Instant::now();
            assert!(now < deadline, "memory inspection gate was not released",);

            let (next, timeout) = self
                .changed
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .expect("memory gate wait");
            state = next;

            assert!(
                !timeout.timed_out() || state.released,
                "memory inspection gate was not released",
            );
        }

        state.armed = false;
    }

    fn wait_entered(&self) {
        let deadline = deadline();
        let mut state = self.state.lock().expect("memory gate");

        while !state.entered {
            let now = Instant::now();
            assert!(
                now < deadline,
                "native worker did not enter memory inspection",
            );

            let (next, timeout) = self
                .changed
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .expect("wait for native memory inspection");
            state = next;

            assert!(
                !timeout.timed_out() || state.entered,
                "native worker did not enter memory inspection",
            );
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().expect("memory gate");
        state.released = true;
        self.changed.notify_all();
    }
}

/// Ensures test unwinding releases a blocked native worker.
struct GateRelease {
    gate: Arc<MemoryGate>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResourceCounts {
    models_created: usize,
    models_destroyed: usize,
    caches_created: usize,
    caches_destroyed: usize,
}

#[derive(Default)]
struct LifecycleTracker {
    models_created: AtomicUsize,
    models_destroyed: AtomicUsize,
    caches_created: AtomicUsize,
    caches_destroyed: AtomicUsize,
    model_instance: Mutex<Option<ModelInstanceId>>,
    decoded_tokens: Mutex<BTreeMap<RequestId, Vec<u32>>>,
    destruction_order: Mutex<Vec<&'static str>>,
    memory_gate: Arc<MemoryGate>,
}

impl LifecycleTracker {
    fn counts(&self) -> ResourceCounts {
        ResourceCounts {
            models_created: self.models_created.load(Ordering::SeqCst),
            models_destroyed: self.models_destroyed.load(Ordering::SeqCst),
            caches_created: self.caches_created.load(Ordering::SeqCst),
            caches_destroyed: self.caches_destroyed.load(Ordering::SeqCst),
        }
    }

    fn model_instance(&self) -> ModelInstanceId {
        self.model_instance
            .lock()
            .expect("model identity")
            .clone()
            .expect("actual loaded model identity")
    }

    fn generated_tokens(&self, request: &RequestId) -> Vec<u32> {
        self.decoded_tokens
            .lock()
            .expect("decoded tokens")
            .get(request)
            .cloned()
            .unwrap_or_default()
    }
}

struct TrackedCache {
    inner: Option<Box<dyn FamilyCache>>,
    tracker: Arc<LifecycleTracker>,
}

impl TrackedCache {
    fn inner(&self) -> &dyn FamilyCache {
        self.inner.as_deref().expect("live native cache")
    }

    fn inner_mut(&mut self) -> &mut dyn FamilyCache {
        self.inner.as_deref_mut().expect("live native cache")
    }
}

impl FamilyCache for TrackedCache {
    fn owner(&self) -> &ModelInstanceId {
        self.inner().owner()
    }

    fn representation(&self) -> &engine::types::CacheRepresentationId {
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
        self.tracker.memory_gate.enter_if_armed();
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
        // Preserve the actual adapter's concrete-cache downcast.
        self.inner().as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self.inner_mut().as_any_mut()
    }
}

impl Drop for TrackedCache {
    fn drop(&mut self) {
        // Record only after the actual native cache was destroyed.
        drop(self.inner.take());

        self.tracker.caches_destroyed.fetch_add(1, Ordering::SeqCst);
        self.tracker
            .destruction_order
            .lock()
            .expect("destruction order")
            .push("cache");
    }
}

struct TrackedModel {
    inner: Option<Box<dyn LoadedModel>>,
    vocabulary: usize,
    tracker: Arc<LifecycleTracker>,
}

impl TrackedModel {
    fn inner(&self) -> &dyn LoadedModel {
        self.inner.as_deref().expect("live native model")
    }

    fn executable(&mut self) -> Result<&mut (dyn AutoregressiveModel + '_), EngineError> {
        match self
            .inner
            .as_deref_mut()
            .expect("live native model")
            .autoregressive()
        {
            Some(model) => Ok(model),
            None => Err(EngineError::unsupported(
                "state lifecycle fixture",
                "actual model is not autoregressive",
            )),
        }
    }
}

impl LoadedModel for TrackedModel {
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

impl AutoregressiveModel for TrackedModel {
    fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }

    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        let cache = self.executable()?.allocate_cache(request, control)?;

        self.tracker.caches_created.fetch_add(1, Ordering::SeqCst);

        Ok(Box::new(TrackedCache {
            inner: Some(cache),
            tracker: Arc::clone(&self.tracker),
        }))
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        self.executable()?.prefill(tokens, cache, control)
    }

    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        let logits = self.executable()?.decode(token, cache, control)?;

        self.tracker
            .decoded_tokens
            .lock()
            .expect("decoded tokens")
            .entry(control.request_id().clone())
            .or_default()
            .push(token);

        Ok(logits)
    }
}

impl Drop for TrackedModel {
    fn drop(&mut self) {
        drop(self.inner.take());

        self.tracker.models_destroyed.fetch_add(1, Ordering::SeqCst);
        self.tracker
            .destruction_order
            .lock()
            .expect("destruction order")
            .push("model");
    }
}

struct LifecycleBackend {
    inner: Qwen3Backend,
    tracker: Arc<LifecycleTracker>,
}

impl ModelFamily for LifecycleBackend {
    fn registration(&self) -> &FamilyRegistration {
        self.inner.registration()
    }

    fn load(&self, request: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        let mut model = self.inner.load(request)?;

        let vocabulary = model
            .autoregressive()
            .ok_or_else(|| {
                EngineError::unsupported(
                    "state lifecycle fixture",
                    "loaded model is not autoregressive",
                )
            })?
            .vocabulary_size();

        *self.tracker.model_instance.lock().expect("model identity") =
            Some(model.instance_id().clone());

        self.tracker.models_created.fetch_add(1, Ordering::SeqCst);

        Ok(Box::new(TrackedModel {
            inner: Some(model),
            vocabulary,
            tracker: Arc::clone(&self.tracker),
        }))
    }
}

impl RuntimeBackend for LifecycleBackend {
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
        control: &ExecutionControl,
    ) -> Result<(), EngineError> {
        self.inner.verify_artifact(artifact, control)
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

struct CompletedRequest {
    request_id: RequestId,
    rendered: RenderedContext,
    prefix_bytes: Vec<u8>,
    expected_state_id: SessionStateId,
    usage: Usage,
}

struct StateEnvironment {
    engine: EngineCore<FamilyRuntime>,
    runtime: FamilyRuntime,
    admission: AdmissionController,
    scheduler: Scheduler,
    tracker: Arc<LifecycleTracker>,
    tokenizer: Tokenizer,

    // Keep actual pinned artifact files alive until the engine is dropped.
    _fixture: TestEnvironment,
}

impl StateEnvironment {
    fn new() -> Self {
        Self::with_event_capacity(8)
    }

    fn with_event_capacity(event_capacity: usize) -> Self {
        // Reuse a real zero-output-head checkpoint. It generates token 0
        // ("!") after actual model execution, never a synthetic response.
        let fixture = TestEnvironment::new(4, 8, false);
        let directory = fixture._directory.path().to_path_buf();
        let mut files = BTreeMap::new();

        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            let bytes = std::fs::read(directory.join(name)).expect("actual fixture component");

            files.insert(
                name.to_owned(),
                FilePin {
                    bytes: bytes.len() as u64,
                    sha256: digest(&bytes),
                },
            );
        }

        let tokenizer = Tokenizer::from_bytes(
            &std::fs::read(directory.join("tokenizer.json")).expect("tokenizer source"),
        )
        .expect("actual fixture tokenizer");

        let config = Qwen3Config::tiny_fixture();
        let limits = GenerationLimits::new(event_capacity, 4096, 1024).expect("limits");

        let inner = Qwen3Backend::new(
            vec![Qwen3Artifact {
                id: artifact_id(),
                directory,
                files,
                template_sha256: digest(CHAT_TEMPLATE.as_bytes()),
            }],
            Qwen3BackendLimits {
                max_files: 16,
                max_config_bytes: 1024 * 1024,
                max_tokenizer_bytes: 1024 * 1024,
                max_messages: 16,
                max_input_bytes: 4096,
                max_rendered_bytes: 16 * 1024,
                max_token_bytes: 128,
                verification_chunk_bytes: 4096,
                context_limit_tokens: config.max_context,
                prefill_chunk_tokens: NonZeroUsize::new(2).expect("chunk"),
                loader_headroom_bytes: 64 * 1024,
                execution_headroom_bytes: 64 * 1024,
                generation: limits,
            },
        )
        .expect("actual lifecycle backend");

        let profile = ModelProfile::new(
            profile_id(),
            artifact_id(),
            config.max_context,
            4,
            8,
            GenerationOptions::greedy(4).expect("default options"),
        )
        .expect("profile");

        let registry = inner.registry(vec![profile]).expect("registry");
        let admission = AdmissionController::new(BUDGET_BYTES).expect("admission");
        let scheduler = Scheduler::new(64).expect("scheduler");
        let tracker = Arc::new(LifecycleTracker::default());

        let backend: Arc<dyn RuntimeBackend> = Arc::new(LifecycleBackend {
            inner,
            tracker: Arc::clone(&tracker),
        });

        let runtime =
            FamilyRuntime::new(vec![backend], admission.clone(), limits, retained_policy())
                .expect("lifecycle runtime");

        let engine = EngineCore::new(registry, scheduler.clone(), runtime.clone());

        Self {
            engine,
            runtime,
            admission,
            scheduler,
            tracker,
            tokenizer,
            _fixture: fixture,
        }
    }

    fn request(&self, name: &str) -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(name).expect("request ID"),
            profile_id(),
            context(name, profile_id(), "hello"),
            GenerationOptions::greedy(4).expect("options"),
        )
        .expect("request")
    }

    fn wait_idle(&self) {
        let deadline = deadline();

        while self.scheduler.is_busy().expect("scheduler status") {
            assert!(
                Instant::now() < deadline,
                "lifecycle request did not relinquish native ownership",
            );
            thread::yield_now();
        }

        assert!(
            self.engine
                .active_request_id()
                .expect("active request")
                .is_none(),
        );
    }

    fn snapshot(&self) -> AdmissionSnapshot {
        self.admission.snapshot().expect("admission snapshot")
    }

    fn complete(&self, name: &str) -> CompletedRequest {
        let request = self.request(name);

        let rendered = block_on(self.engine.render_and_count(request.context().clone()))
            .expect("actual rendering");

        let generation = block_on(self.engine.generate(request)).expect("generation");

        let request_id = generation.request_id().clone();
        let mut events = generation.into_events();
        let transcript = drain(&mut events);
        drop(events);
        self.wait_idle();

        let usage = completed_usage(&transcript, FinishReason::OutputLimit);
        assert_eq!(usage.input_tokens, rendered.token_count());
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(transcript.text, "!!!!");

        let generated = self.tracker.generated_tokens(&request_id);
        assert_eq!(generated.len(), usage.output_tokens);
        assert_eq!(generated, [0, 0, 0, 0]);

        let mut executed = rendered.token_ids().to_vec();
        executed.extend_from_slice(&generated);

        // Preserve structural special tokens and obtain the actual
        // fixture tokenizer's byte stream. Verify exact token roundtrip.
        let decoded = self
            .tokenizer
            .decode(&executed, false)
            .expect("lossless fixture tokenizer decode");

        let roundtrip = self
            .tokenizer
            .encode(decoded.as_str(), false)
            .expect("fixture tokenizer roundtrip");

        assert_eq!(
            roundtrip.get_ids(),
            executed.as_slice(),
            "publication bytes must reconstruct the exact executed history",
        );

        let prefix_bytes = decoded.into_bytes();

        let capabilities =
            block_on(self.engine.capabilities(artifact_id())).expect("actual loaded capabilities");
        assert_eq!(capabilities.cache_modes.len(), 1);

        let resolved = self
            .engine
            .registry()
            .resolve_profile(rendered.profile_id())
            .expect("resolved profile");

        let expected_state_id = compute_session_state_id(&CacheIdentityInputs::for_current_engine(
            resolved.artifact().id(),
            resolved.family().implementation_version(),
            None,
            &prefix_bytes,
            &capabilities.cache_modes[0].representation,
        ))
        .expect("canonical fixture identity");

        CompletedRequest {
            request_id,
            rendered,
            prefix_bytes,
            expected_state_id,
            usage,
        }
    }

    fn publish(&self, completed: &CompletedRequest) -> SessionStateId {
        let id = block_on(
            self.runtime
                .retain_session_state(completed.request_id.clone(), completed.prefix_bytes.clone()),
        )
        .expect("actual state publication");

        assert_eq!(id, completed.expected_state_id);
        id
    }

    fn lookup(&self, id: SessionStateId) -> RetainedSessionState {
        block_on(self.runtime.lookup_session_state(id))
            .expect("lookup")
            .expect("actual published state")
    }

    fn release(&self, id: SessionStateId) -> ReleaseOutcome {
        block_on(self.engine.release_session_state(id)).expect("state release")
    }
}

fn unknown_id(actual: SessionStateId) -> SessionStateId {
    let mut digest = *actual.as_bytes();
    digest[0] ^= 1;
    SessionStateId::from_digest(digest)
}

fn assert_same_allocation(before: &RetainedSessionState, after: &RetainedSessionState) {
    assert_eq!(before.state_id, after.state_id);
    assert_eq!(before.request_id, after.request_id);
    assert_eq!(before.artifact_id, after.artifact_id);
    assert_eq!(before.model_instance_id, after.model_instance_id);
    assert_eq!(before.family_impl_version, after.family_impl_version);
    assert_eq!(before.profile_id, after.profile_id);
    assert_eq!(before.tokenizer_id, after.tokenizer_id);
    assert_eq!(before.template_hash, after.template_hash);
    assert_eq!(before.render_hash, after.render_hash);
    assert_eq!(before.renderer, after.renderer);
    assert_eq!(before.cache_mode, after.cache_mode);
    assert_eq!(before.logical_tokens, after.logical_tokens);
    assert_eq!(before.retained_tokens, after.retained_tokens);
    assert_eq!(before.capacity_tokens, after.capacity_tokens);
    assert_eq!(before.weights_reserved_bytes, after.weights_reserved_bytes,);
    assert_eq!(before.cache_reserved_bytes, after.cache_reserved_bytes,);
    assert_eq!(
        before.token_history_reserved_bytes,
        after.token_history_reserved_bytes,
    );
    assert_eq!(before.logits_reserved_bytes, after.logits_reserved_bytes);
}

fn assert_lease_protection(
    error: &EngineError,
    state: &RetainedSessionState,
    expected_handles: usize,
) {
    let EngineError::ProtectedResources { resources, .. } = error else {
        panic!("expected typed canonical lease protection, got {error:?}");
    };
    assert_eq!(
        resources.len(),
        1,
        "only the addressed allocation is blocked"
    );
    match &resources[0] {
        ProtectedResource::SessionStateLease {
            artifact_id,
            instance_id,
            state_id,
            handles,
        } => {
            assert_eq!(artifact_id, &state.artifact_id);
            assert!(instance_id.same_instance(&state.model_instance_id));
            assert_eq!(state_id, &state.state_id);
            assert_eq!(handles.get(), expected_handles);
        }
        other => panic!("expected a canonical lease, got {other:?}"),
    }
}

#[test]
fn completed_publication_binds_canonical_identity_to_actual_native_state() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("publish-completed");
    let before = environment.snapshot();
    let counts = environment.tracker.counts();

    let id = environment.publish(&completed);
    let state = environment.lookup(id);

    assert_eq!(state.request_id, completed.request_id);
    assert_eq!(state.artifact_id, artifact_id());
    assert_eq!(
        state.model_instance_id,
        environment.tracker.model_instance(),
    );
    assert_eq!(state.profile_id, *completed.rendered.profile_id());
    assert_eq!(state.tokenizer_id, *completed.rendered.tokenizer_id());
    assert_eq!(state.template_hash, *completed.rendered.template_hash());
    assert_eq!(state.render_hash, *completed.rendered.render_hash());
    assert_eq!(state.renderer, *completed.rendered.renderer());

    assert_eq!(
        state.logical_tokens,
        completed.usage.input_tokens + completed.usage.output_tokens,
    );
    assert_eq!(state.retained_tokens, state.logical_tokens);
    assert!(state.capacity_tokens >= state.logical_tokens);
    assert_eq!(state.reused_input_tokens, 0);
    assert_eq!(
        state.prefilled_input_tokens,
        completed.rendered.token_count(),
    );

    let expected_reserved = [
        state.weights_reserved_bytes,
        state.cache_reserved_bytes,
        state.token_history_reserved_bytes,
        state.logits_reserved_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))
    .expect("reservation sum");

    assert_eq!(before.reserved_bytes(), expected_reserved);
    assert_eq!(before.active_reservations(), 4);

    // Publication moves ownership; it must not allocate a second model
    // or leave its temporary prefix-byte reservation behind.
    assert_eq!(environment.snapshot(), before);
    assert_eq!(environment.tracker.counts(), counts);
}

#[test]
fn repeated_publication_is_idempotent_without_additional_reservations() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("idempotent-publication");
    let first = environment.publish(&completed);
    let before = environment.lookup(first);
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();

    let second = environment.publish(&completed);

    assert_eq!(first, second);
    assert_same_allocation(&before, &environment.lookup(second));
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);
}

#[test]
fn conflicting_publication_cannot_relabel_an_existing_allocation() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("conflicting-publication");
    let id = environment.publish(&completed);
    let state = environment.lookup(id);
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();

    // This is intentionally not the completed request's prefix. The
    // runtime must reject relabeling an already-published allocation.
    let mut conflicting = completed.prefix_bytes.clone();
    conflicting.push(b'!');

    let error = expect_failure(
        block_on(
            environment
                .runtime
                .retain_session_state(completed.request_id.clone(), conflicting),
        ),
        "conflicting publication relabeled existing state",
    );

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_same_allocation(&state, &environment.lookup(id));
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);
}

#[test]
fn absent_lookup_lease_and_release_do_not_change_existing_state() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("absent-operations");
    let id = environment.publish(&completed);
    let absent = unknown_id(id);
    let state = environment.lookup(id);
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();

    assert!(
        block_on(environment.runtime.lookup_session_state(absent))
            .expect("absent lookup")
            .is_none(),
    );
    assert!(
        block_on(environment.runtime.lease_session_state(absent))
            .expect("absent lease")
            .is_none(),
    );
    assert_eq!(environment.release(absent), ReleaseOutcome::AlreadyAbsent,);

    assert_same_allocation(&state, &environment.lookup(id));
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);
}

#[test]
fn every_live_lease_must_be_dropped_before_release() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("multiple-leases");
    let id = environment.publish(&completed);
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();

    let first = block_on(environment.runtime.lease_session_state(id))
        .expect("first lease")
        .expect("first state");
    let second = block_on(environment.runtime.lease_session_state(id))
        .expect("second lease")
        .expect("second state");

    assert_eq!(first.state_id(), id);
    assert_eq!(second.state_id(), id);
    assert_same_allocation(first.state(), second.state());

    let error = expect_failure(
        block_on(environment.engine.release_session_state(id)),
        "released state while both leases were alive",
    );
    assert_lease_protection(&error, second.state(), 2);
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);

    drop(first);

    let error = expect_failure(
        block_on(environment.engine.release_session_state(id)),
        "released state while its second lease was alive",
    );
    assert_lease_protection(&error, second.state(), 1);
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);

    drop(second);
    assert_eq!(environment.release(id), ReleaseOutcome::Released);
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert_eq!(environment.snapshot().active_reservations(), 0);
}

#[test]
fn generation_cannot_mutate_or_reclaim_a_leased_native_allocation() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("leased-generation-seed");
    let id = environment.publish(&completed);
    let before = environment.lookup(id);
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();

    let lease = block_on(environment.runtime.lease_session_state(id))
        .expect("lease")
        .expect("published state");

    let error = expect_failure(
        block_on(
            environment
                .engine
                .generate(environment.request("generation-blocked-by-lease")),
        ),
        "generation mutated a leased allocation",
    );

    assert_lease_protection(&error, &before, 1);
    environment.wait_idle();

    assert_same_allocation(&before, &environment.lookup(id));
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);

    drop(lease);
    assert_eq!(environment.release(id), ReleaseOutcome::Released);

    // A failed startup while leased must not leave pending ownership.
    let recovered = environment.complete("generation-after-lease-release");
    assert_eq!(recovered.usage.output_tokens, 4);
}

#[test]
fn release_acknowledges_actual_destruction_and_is_then_already_absent() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("destruction-before-ack");
    let id = environment.publish(&completed);

    let before = environment.tracker.counts();
    assert_eq!(before.models_created, 1);
    assert_eq!(before.caches_created, 1);
    assert_eq!(before.models_destroyed, 0);
    assert_eq!(before.caches_destroyed, 0);

    assert_eq!(environment.release(id), ReleaseOutcome::Released);

    // No polling or eventual-cleanup wait: destruction and accounting
    // must already be complete when Released is returned.
    let after = environment.tracker.counts();
    assert_eq!(after.models_destroyed, 1);
    assert_eq!(after.caches_destroyed, 1);
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert_eq!(environment.snapshot().active_reservations(), 0);

    let order = environment
        .tracker
        .destruction_order
        .lock()
        .expect("destruction order")
        .clone();
    assert_eq!(order, ["cache", "model"]);

    assert!(
        block_on(environment.runtime.lookup_session_state(id))
            .expect("released lookup")
            .is_none(),
    );
    assert_eq!(environment.release(id), ReleaseOutcome::AlreadyAbsent,);
    assert_eq!(environment.tracker.counts(), after);
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
}

struct LeaseWake {
    sender: mpsc::Sender<()>,
}

impl Wake for LeaseWake {
    fn wake(self: Arc<Self>) {
        let _ = self.sender.send(());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.sender.send(());
    }
}

#[test]
fn abandoning_a_ready_unconsumed_lease_reply_does_not_leave_a_pin() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("abandoned-lease");
    let id = environment.publish(&completed);
    let accounting = environment.snapshot();

    // Hold the actual native worker inside validation of an ordinary
    // metadata lookup. That operation acquires no lease.
    let release = environment.tracker.memory_gate.arm();

    let lookup_waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut lookup_context = Context::from_waker(&lookup_waker);

    let mut blocking_lookup = environment.runtime.lookup_session_state(id);
    assert!(
        blocking_lookup
            .as_mut()
            .poll(&mut lookup_context)
            .is_pending(),
    );
    environment.tracker.memory_gate.wait_entered();

    // The worker is blocked, so this first lease poll must be Pending.
    // Its waker is installed before the worker can acquire the lease.
    let (sender, receiver) = mpsc::channel();
    let lease_waker = Waker::from(Arc::new(LeaseWake { sender }));
    let mut lease_context = Context::from_waker(&lease_waker);

    let mut lease_future = environment.runtime.lease_session_state(id);
    assert!(lease_future.as_mut().poll(&mut lease_context).is_pending());

    release.release();

    // Runtime dispatch wakes only after storing the actual lease result.
    // Do not poll again: abandon that ready-but-unconsumed result.
    receiver
        .recv_timeout(TEST_TIMEOUT)
        .expect("actual lease result became ready");

    drop(lease_future);
    drop(blocking_lookup);

    assert_eq!(environment.snapshot(), accounting);

    // A leaked pin would make this return Busy.
    assert_eq!(environment.release(id), ReleaseOutcome::Released);
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert_eq!(environment.snapshot().active_reservations(), 0);
}

#[test]
fn unknown_request_cannot_publish_another_requests_completed_state() {
    let environment = StateEnvironment::new();
    let completed = environment.complete("actual-completed-request");
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();

    let error = expect_failure(
        block_on(environment.runtime.retain_session_state(
            RequestId::new("never-executed-request").expect("request ID"),
            completed.prefix_bytes.clone(),
        )),
        "unknown request published another request's allocation",
    );

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);

    // Correct request association still works after rejection.
    let id = environment.publish(&completed);
    assert_eq!(environment.lookup(id).request_id, completed.request_id,);
}

#[test]
fn publication_rejects_active_or_pending_generation() {
    let environment = StateEnvironment::with_event_capacity(1);
    let generation = block_on(
        environment
            .engine
            .generate(environment.request("active-publication")),
    )
    .expect("accepted bounded generation");

    let request_id = generation.request_id().clone();

    // Four nonempty text events cannot complete into a one-slot queue
    // before the consumer starts draining it.
    let error = expect_failure(
        block_on(
            environment
                .runtime
                .retain_session_state(request_id, Vec::new()),
        ),
        "published state while execution still owned native resources",
    );

    assert_eq!(error.kind(), ErrorKind::Busy);

    let mut events = generation.into_events();
    let transcript = drain(&mut events);
    completed_usage(&transcript, FinishReason::OutputLimit);
    assert_eq!(transcript.text, "!!!!");

    drop(events);
    environment.wait_idle();
}

// M9_POOLED_CANONICAL_LEASES_V1
struct PooledLeaseFixture {
    environment: StateEnvironment,
    id: SessionStateId,
    original: RetainedSessionState,
    accounting: AdmissionSnapshot,
    counts: ResourceCounts,
    leases: Vec<engine::runtime::SessionStateLease>,
}

impl PooledLeaseFixture {
    fn new(name: &str) -> Self {
        let environment = StateEnvironment::new();
        // Both runtime handles address the same native worker. Enable the
        // pool before loading, without changing the non-pool constructors.
        let _ = environment
            .runtime
            .clone()
            .with_model_pool(None)
            .expect("enable lifecycle model pool");
        let completed = environment.complete(name);
        let id = environment.publish(&completed);
        let original = environment.lookup(id);
        let first = block_on(environment.runtime.lease_session_state(id))
            .unwrap()
            .expect("first canonical lease");
        let second = block_on(environment.runtime.lease_session_state(id))
            .unwrap()
            .expect("second canonical lease");
        assert!(
            block_on(
                environment
                    .runtime
                    .set_model_idle_ttl(original.model_instance_id.clone(), Some(Duration::ZERO),)
            )
            .unwrap()
        );
        let accounting = environment.snapshot();
        let counts = environment.tracker.counts();
        let fixture = Self {
            environment,
            id,
            original,
            accounting,
            counts,
            leases: vec![first, second],
        };
        fixture.assert_protected();
        fixture
    }

    fn assert_protected(&self) {
        let actual = self.environment.lookup(self.id);
        assert_same_allocation(&self.original, &actual);
        assert_eq!(
            actual.reused_input_tokens,
            self.original.reused_input_tokens
        );
        assert_eq!(
            actual.prefilled_input_tokens,
            self.original.prefilled_input_tokens
        );
        assert!(
            actual
                .model_instance_id
                .same_instance(&self.original.model_instance_id)
        );
        let residents = block_on(self.environment.runtime.model_residency()).unwrap();
        assert_eq!(residents.len(), 1);
        assert!(
            residents[0]
                .instance
                .same_instance(&self.original.model_instance_id)
        );
        assert!(
            !residents[0].pinned,
            "explicit model pins must not mask lease protection"
        );
        let now = self.environment.snapshot();
        // Live enforcement can resize the budget, but must not change
        // ownership of any of these four native-state reservations.
        assert_eq!(now.reserved_bytes(), self.accounting.reserved_bytes());
        assert_eq!(
            now.active_reservations(),
            self.accounting.active_reservations()
        );
        assert_eq!(self.environment.tracker.counts(), self.counts);
        for lease in &self.leases {
            assert_eq!(lease.state_id(), self.id);
            assert_same_allocation(&self.original, lease.state());
        }
    }

    fn assert_release_and_generation_blocked(&self, name: &str) {
        let error = expect_failure(
            block_on(self.environment.engine.release_session_state(self.id)),
            "release must reject a live canonical lease",
        );
        assert_lease_protection(&error, &self.original, self.leases.len());
        let error = expect_failure(
            block_on(
                self.environment
                    .engine
                    .generate(self.environment.request(name)),
            ),
            "ordinary generation must preserve the existing conservative lease barrier",
        );
        assert_lease_protection(&error, &self.original, self.leases.len());
        self.environment.wait_idle();
        self.assert_protected();
        // This is allocator-only reclamation, not the destructive pressure
        // test below. It must not invalidate live canonical state.
        block_on(self.environment.runtime.reclaim_mlx_allocator_cache())
            .expect("idle allocator-only reclamation");
        self.assert_protected();
    }

    fn assert_destroyed(&self) {
        assert!(
            block_on(self.environment.runtime.lookup_session_state(self.id))
                .unwrap()
                .is_none()
        );
        assert!(
            block_on(self.environment.runtime.model_residency())
                .unwrap()
                .is_empty()
        );
        let now = self.environment.snapshot();
        assert_eq!(now.reserved_bytes(), 0);
        assert_eq!(now.active_reservations(), 0);
        let counts = self.environment.tracker.counts();
        assert_eq!(counts.models_created, self.counts.models_created);
        assert_eq!(counts.caches_created, self.counts.caches_created);
        assert_eq!(counts.models_destroyed, self.counts.models_destroyed + 1);
        assert_eq!(counts.caches_destroyed, self.counts.caches_destroyed + 1);
        assert_eq!(
            self.environment
                .tracker
                .destruction_order
                .lock()
                .unwrap()
                .as_slice(),
            &["cache", "model"],
        );
    }

    fn disable_ttl_before_final_drop(&self) {
        assert_eq!(self.leases.len(), 1);
        assert!(
            block_on(
                self.environment
                    .runtime
                    .set_model_idle_ttl(self.original.model_instance_id.clone(), None,)
            )
            .unwrap()
        );
        self.assert_protected();
    }
}

#[test]
fn pool_live_leases_expired_ttl_waits_for_every_lease() {
    let mut fixture = PooledLeaseFixture::new("pool-two-leases-expiry");
    fixture.assert_release_and_generation_blocked("pool-two-leases-deny");
    drop(fixture.leases.pop());
    fixture.assert_release_and_generation_blocked("pool-one-lease-deny");
    drop(fixture.leases.pop());
    // A worker may already be waiting for its next command after the last
    // protected sweep. Drive maintenance until the actual state disappears;
    // a single metadata query is not assumed to force a prior expiry sweep.
    let limit = deadline();
    while block_on(fixture.environment.runtime.lookup_session_state(fixture.id))
        .unwrap()
        .is_some()
    {
        assert!(
            Instant::now() < limit,
            "expired unleased state did not disappear"
        );
        thread::yield_now();
    }
    fixture.assert_destroyed();
    assert_eq!(
        fixture.environment.release(fixture.id),
        ReleaseOutcome::AlreadyAbsent
    );
    let _ = fixture.environment.complete("pool-expiry-recovery");
}

#[test]
fn pool_live_leases_explicit_release_after_final_drop() {
    let mut fixture = PooledLeaseFixture::new("pool-two-leases-release");
    fixture.assert_release_and_generation_blocked("pool-release-two-deny");
    drop(fixture.leases.pop());
    fixture.assert_release_and_generation_blocked("pool-release-one-deny");
    // Both lease counts were tested with expired TTL. Disable automatic
    // expiry while the last lease still protects A, so this test proves
    // explicit release rather than racing it against an expiry sweep.
    fixture.disable_ttl_before_final_drop();
    drop(fixture.leases.pop());
    assert_eq!(
        fixture.environment.release(fixture.id),
        ReleaseOutcome::Released
    );
    fixture.assert_destroyed();
    let _ = fixture.environment.complete("pool-release-recovery");
}

#[derive(Debug)]
struct PooledLeasePressure;

impl engine::memory_enforcer::MemoryObservationSource for PooledLeasePressure {
    fn observe(
        &self,
    ) -> Result<engine::memory_enforcer::MemoryObservation, engine::memory_enforcer::EnforcerError>
    {
        const GIB: u64 = 1024 * 1024 * 1024;
        Ok(engine::memory_enforcer::MemoryObservation {
            host: engine::memory_guard::HostMemorySnapshot {
                physical_memory_bytes: 36 * GIB,
                free_bytes: 5 * GIB,
                inactive_bytes: 0,
                active_file_bytes: 0,
                other_apps_active_bytes: 0,
                phys_footprint_bytes: 20 * GIB,
                graphics_footprint_bytes: 0,
                iogpu_wired_limit_bytes: 0,
                recommended_working_set_bytes: 100 * GIB,
                release_credit_bytes: 0,
            },
            mlx_active_bytes: 20 * GIB,
            mlx_pool_bytes: 3 * GIB,
        })
    }
}

#[test]
fn pool_live_leases_enforcer_rechecks_both_lease_counts() {
    use engine::memory_enforcer::{MemoryEnforcer, PRESSURE_RECLAIM_GRACE_POLLS_MAX};
    use engine::memory_guard::{MemoryGuardConfig, PressureLevel};
    use engine::runtime::IdleModelReclaim;
    let mut fixture = PooledLeaseFixture::new("pool-two-leases-enforcer");
    let enforcer = MemoryEnforcer::with_observation_source(
        MemoryGuardConfig::balanced(),
        fixture.environment.admission.clone(),
        Arc::new(PooledLeasePressure),
    );
    for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX {
        let report = block_on(
            fixture
                .environment
                .runtime
                .enforce_memory_tick(enforcer.clone()),
        )
        .unwrap();
        assert_eq!(report.tick.status.pressure, PressureLevel::Soft);
        assert!(report.tick.deferred_for_pool_drain);
        assert!(matches!(report.model, IdleModelReclaim::NotRequested));
        fixture.assert_protected();
    }
    let report = block_on(
        fixture
            .environment
            .runtime
            .enforce_memory_tick(enforcer.clone()),
    )
    .unwrap();
    assert!(!report.tick.deferred_for_pool_drain);
    assert!(matches!(report.model, IdleModelReclaim::NoEligibleVictim));
    fixture.assert_protected();
    drop(fixture.leases.pop());
    let report = block_on(
        fixture
            .environment
            .runtime
            .enforce_memory_tick(enforcer.clone()),
    )
    .unwrap();
    assert!(matches!(report.model, IdleModelReclaim::NoEligibleVictim));
    fixture.assert_protected();
    fixture.disable_ttl_before_final_drop();
    drop(fixture.leases.pop());
    let report = block_on(fixture.environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    match report.model {
        IdleModelReclaim::Evicted(instance) => {
            assert!(instance.same_instance(&fixture.original.model_instance_id))
        }
        other => panic!("expected actual unleased model eviction, got {other:?}"),
    }
    fixture.assert_destroyed();
}

// M9_MODEL_PIN_REPLY_AND_STATE_RELEASE_V1
struct ModelPinReplyWake {
    sender: mpsc::Sender<()>,
}

impl std::task::Wake for ModelPinReplyWake {
    fn wake(self: Arc<Self>) {
        let _ = self.sender.send(());
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.sender.send(());
    }
}

fn model_pin_lifecycle_environment(
    name: &str,
) -> (StateEnvironment, SessionStateId, RetainedSessionState) {
    let environment = StateEnvironment::new();
    let _ = environment
        .runtime
        .clone()
        .with_model_pool(None)
        .expect("enable model-pool lifecycle fixture");
    let completed = environment.complete(name);
    let id = environment.publish(&completed);
    let state = environment.lookup(id);
    (environment, id, state)
}

fn unconsumed_model_pin_reply<'a>(
    environment: &'a StateEnvironment,
    state: &RetainedSessionState,
) -> engine::api::EngineFuture<'a, Option<engine::residency_pin::ModelResidencyPin>> {
    let release = environment.tracker.memory_gate.arm();
    let mut lookup = environment.runtime.lookup_session_state(state.state_id);
    let lookup_waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut lookup_cx = Context::from_waker(&lookup_waker);
    assert!(lookup.as_mut().poll(&mut lookup_cx).is_pending());
    environment.tracker.memory_gate.wait_entered();

    let mut acquisition = environment
        .runtime
        .pin_model(state.model_instance_id.clone());
    let (sender, receiver) = mpsc::channel();
    let waker = Waker::from(Arc::new(ModelPinReplyWake { sender }));
    let mut cx = Context::from_waker(&waker);
    // The native worker is held inside real cache inspection. Acquisition
    // has been dispatched but cannot complete during this first poll.
    assert!(acquisition.as_mut().poll(&mut cx).is_pending());
    release.release();
    let actual = block_on(lookup).unwrap().expect("original canonical state");
    assert_same_allocation(state, &actual);

    receiver
        .recv_timeout(TEST_TIMEOUT)
        .expect("pin reply published");
    // OperationFuture publishes its result before invoking this waker.
    // Do not poll acquisition again: the future must own the unconsumed pin.
    let residents = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(residents.len(), 1);
    assert!(
        residents[0]
            .instance
            .same_instance(&state.model_instance_id)
    );
    assert!(residents[0].pinned);
    acquisition
}

fn assert_model_pin_lifecycle_unchanged(
    environment: &StateEnvironment,
    state: &RetainedSessionState,
    accounting: AdmissionSnapshot,
    counts: ResourceCounts,
) {
    let actual = environment.lookup(state.state_id);
    assert_same_allocation(state, &actual);
    assert_eq!(actual.reused_input_tokens, state.reused_input_tokens);
    assert_eq!(actual.prefilled_input_tokens, state.prefilled_input_tokens);
    let residents = block_on(environment.runtime.model_residency()).unwrap();
    assert_eq!(residents.len(), 1);
    assert!(
        residents[0]
            .instance
            .same_instance(&state.model_instance_id)
    );
    assert!(residents[0].pinned);
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);
}

fn wait_model_pin_lifecycle_expired(
    environment: &StateEnvironment,
    state: &RetainedSessionState,
    counts: ResourceCounts,
) {
    let limit = deadline();
    loop {
        let residents = block_on(environment.runtime.model_residency()).unwrap();
        if residents.is_empty() {
            break;
        }
        assert_eq!(residents.len(), 1);
        assert!(
            residents[0]
                .instance
                .same_instance(&state.model_instance_id)
        );
        assert!(
            Instant::now() < limit,
            "model protection leaked after final pin drop"
        );
        thread::yield_now();
    }
    assert!(
        block_on(environment.runtime.lookup_session_state(state.state_id))
            .unwrap()
            .is_none()
    );
    let accounting = environment.snapshot();
    assert_eq!(accounting.reserved_bytes(), 0);
    assert_eq!(accounting.active_reservations(), 0);
    let after = environment.tracker.counts();
    assert_eq!(after.models_created, counts.models_created);
    assert_eq!(after.caches_created, counts.caches_created);
    assert_eq!(after.models_destroyed, counts.models_destroyed + 1);
    assert_eq!(after.caches_destroyed, counts.caches_destroyed + 1);
    assert_eq!(
        environment
            .tracker
            .destruction_order
            .lock()
            .unwrap()
            .as_slice(),
        &["cache", "model"]
    );
}

#[test]
fn pool_model_pin_abandoned_ready_reply_does_not_leak_protection() {
    let (environment, id, state) = model_pin_lifecycle_environment("model-pin-abandoned-reply");
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();
    let acquisition = unconsumed_model_pin_reply(&environment, &state);
    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(state.model_instance_id.clone(), Some(Duration::ZERO),)
        )
        .unwrap()
    );
    // Only the unread reply protects A; no explicit pin handle or session
    // lease has been returned to this caller.
    assert_model_pin_lifecycle_unchanged(&environment, &state, accounting, counts);
    drop(acquisition);
    wait_model_pin_lifecycle_expired(&environment, &state, counts);
    assert_eq!(environment.release(id), ReleaseOutcome::AlreadyAbsent);
    let _ = environment.complete("model-pin-abandon-recovery");
}

#[test]
fn pool_model_pin_abandoned_reply_preserves_separate_explicit_pin() {
    let (environment, _id, state) = model_pin_lifecycle_environment("model-pin-separate-owner");
    let held = block_on(
        environment
            .runtime
            .pin_model(state.model_instance_id.clone()),
    )
    .unwrap()
    .expect("separately held pin");
    assert!(held.instance_id().same_instance(&state.model_instance_id));
    let accounting = environment.snapshot();
    let counts = environment.tracker.counts();
    let acquisition = unconsumed_model_pin_reply(&environment, &state);
    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(state.model_instance_id.clone(), Some(Duration::ZERO),)
        )
        .unwrap()
    );
    drop(acquisition);
    assert_model_pin_lifecycle_unchanged(&environment, &state, accounting, counts);
    drop(held);
    wait_model_pin_lifecycle_expired(&environment, &state, counts);
    let _ = environment.complete("model-pin-separate-recovery");
}

#[test]
fn pool_model_pin_state_release_drops_session_but_retains_weights() {
    let (environment, id, state) = model_pin_lifecycle_environment("model-pin-release-session");
    let pin = block_on(
        environment
            .runtime
            .pin_model(state.model_instance_id.clone()),
    )
    .unwrap()
    .expect("model residency pin, not a session lease");
    let final_pin = pin.clone();
    let counts = environment.tracker.counts();
    let before = environment.snapshot();
    let expected_before = [
        state.weights_reserved_bytes,
        state.cache_reserved_bytes,
        state.token_history_reserved_bytes,
        state.logits_reserved_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))
    .expect("native-state reservation sum");
    assert_eq!(before.reserved_bytes(), expected_before);
    assert_eq!(before.active_reservations(), 4);
    assert!(
        block_on(
            environment
                .runtime
                .set_model_idle_ttl(state.model_instance_id.clone(), Some(Duration::ZERO),)
        )
        .unwrap()
    );

    assert_eq!(environment.release(id), ReleaseOutcome::Released);
    assert!(
        block_on(environment.runtime.lookup_session_state(id))
            .unwrap()
            .is_none()
    );
    assert!(
        block_on(environment.runtime.lease_session_state(id))
            .unwrap()
            .is_none()
    );
    assert_eq!(environment.release(id), ReleaseOutcome::AlreadyAbsent);

    let assert_weights_only = || {
        let residents = block_on(environment.runtime.model_residency()).unwrap();
        assert_eq!(residents.len(), 1);
        assert!(
            residents[0]
                .instance
                .same_instance(&state.model_instance_id)
        );
        assert!(residents[0].pinned);
        assert_eq!(
            residents[0].committed_weight_bytes,
            state.weights_reserved_bytes
        );
        let after = environment.snapshot();
        assert_eq!(after.reserved_bytes(), state.weights_reserved_bytes);
        assert_eq!(after.active_reservations(), 1);
        assert_eq!(
            before.reserved_bytes() - after.reserved_bytes(),
            state.cache_reserved_bytes
                + state.token_history_reserved_bytes
                + state.logits_reserved_bytes
        );
        let actual = environment.tracker.counts();
        assert_eq!(actual.models_created, counts.models_created);
        assert_eq!(actual.caches_created, counts.caches_created);
        assert_eq!(actual.models_destroyed, counts.models_destroyed);
        assert_eq!(actual.caches_destroyed, counts.caches_destroyed + 1);
        assert_eq!(
            environment
                .tracker
                .destruction_order
                .lock()
                .unwrap()
                .as_slice(),
            &["cache"]
        );
    };
    assert_weights_only();
    drop(pin);
    assert_weights_only();
    drop(final_pin);
    wait_model_pin_lifecycle_expired(&environment, &state, counts);
    let _ = environment.complete("model-pin-release-recovery");
}
