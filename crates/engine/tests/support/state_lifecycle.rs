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
    assert_eq!(error.kind(), ErrorKind::Busy);
    assert_eq!(environment.snapshot(), accounting);
    assert_eq!(environment.tracker.counts(), counts);

    drop(first);

    let error = expect_failure(
        block_on(environment.engine.release_session_state(id)),
        "released state while its second lease was alive",
    );
    assert_eq!(error.kind(), ErrorKind::Busy);
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

    assert_eq!(error.kind(), ErrorKind::Busy);
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
