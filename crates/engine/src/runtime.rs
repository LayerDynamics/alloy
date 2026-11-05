//! Admission-aware runtime with worker-local native model ownership.
//!
//! The asynchronous handle contains only shareable factories, accounting,
//! and command handles. LoadedModel, FamilyCache, tokenizer decoders, and
//! their reservations are constructed and destroyed on a dedicated worker.
//!
//! Startup reclaims incompatible idle resources and acquires output
//! admission on the native worker before the event queue is allocated.
//! A pending request claim prevents competing startup from evicting those
//! resources before prefill. Abandoned handoffs are cleaned up by a
//! dispatcher while scheduler ownership remains retained.
//!
//! Existing generation workers use a Send command proxy; no native model
//! is moved between threads or unsafely marked Send.
//!
//! This runtime implements greedy text generation. It reuses retained
//! allocations, but resets logical KV contents between requests. It does
//! not advertise cross-request prefix reuse or canonical session storage.
//!
//! Backend implementations must supply real verification, memory planning,
//! rendering, tokenization, and model loading. There is no fallback backend.

use crate::{
    admission::{AdmissionController, AdmissionSnapshot, MemoryReservation},
    api::{EngineFuture, EngineRuntime, ExecutionContext},
    cancellation::CancellationToken,
    error::EngineError,
    family::{
        CacheAllocationRequest, ExecutionControl, FamilyCache, LoadedModel, ModelFamily,
        ModelLoadLimits, ModelLoadRequest, TokenLogits,
    },
    generation::{self, GenerationLimits, GenerationSession, TokenStep},
    registry::{ArtifactRegistration, Registry},
    types::{
        ArtifactId, CacheMode, Capabilities, ContextSnapshot, DecodingCapability, DecodingMode,
        Generation, GenerationEvent, GenerationEventStream, GenerationOptions, GenerationRequest,
        GenerationStream, MemoryBreakdown, MemoryEstimate, ModelProfileId, ReleaseOutcome,
        RenderedContext, RequestId, SessionStateId, Usage,
    },
};
use std::{
    collections::BTreeMap,
    future::Future,
    num::NonZeroUsize,
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    task::{Context, Poll},
    thread,
};

/// Explicit lifetime policy for idle native resources.
#[derive(Clone, Copy, Debug)]
pub struct RetentionPolicy {
    /// Keep the last loaded model after a request completes.
    pub retain_model: bool,
    /// Keep its allocated KV buffers after resetting logical contents.
    /// Requires retain_model. This does not enable prefix reuse.
    pub retain_cache_buffers: bool,
}

/// Backend-derived upper bounds for the actual selected representation.
#[derive(Clone, Debug)]
pub struct RuntimePlan {
    /// Actual loader source/resident/overlap limits.
    pub load_limits: ModelLoadLimits,
    /// Actual family cache mode, capacity, and allocation limit.
    pub cache: CacheAllocationRequest,
    /// Prefill/decode/decoder workspace allowance, excluding loader overlap.
    pub temporary_bytes: u64,
    /// Event-queue, pending-text, and output allowance.
    pub output_bytes: u64,
    /// Additional request-local headroom.
    pub headroom_bytes: u64,
    /// Actual chunk bound used for cancellable prefill.
    pub prefill_chunk_tokens: NonZeroUsize,
}

/// Actual incremental tokenizer decoder, confined to the native worker.
pub trait TextDecoder {
    /// Decode one real generated token without corrupting incremental state.
    fn push(&mut self, token: u32) -> Result<String, EngineError>;
    /// Finish pending tokenizer bytes after normal generation.
    fn finish(&mut self) -> Result<String, EngineError>;
}

/// Required real backend services in addition to the actual family loader.
///
/// Implement this on a concrete family adapter. Methods have no default
/// success path; registering metadata alone cannot execute a model.
pub trait RuntimeBackend: ModelFamily + 'static {
    /// Render/tokenize with the verified profile inputs and pinned identities.
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext>;

    /// Compute representation-aware bounds without loading native objects
    /// onto the asynchronous control thread.
    fn plan<'a>(
        &'a self,
        registry: &'a Registry,
        profile: &'a ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, RuntimePlan>;

    /// Enforce the pinned artifact integrity policy before loading or reuse.
    /// Runs on the native worker and must observe the supplied control.
    fn verify_artifact(
        &self,
        artifact: &ArtifactRegistration,
        control: &ExecutionControl,
    ) -> Result<(), EngineError>;

    /// Construct the real profile-specific incremental tokenizer decoder.
    /// Runs on the native worker under request-local admission.
    fn decoder(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError>;

    /// Return the verified profile's actual EOS token IDs.
    /// An explicitly empty list means output-limit-only stopping.
    fn eos_tokens(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Vec<u32>, EngineError>;
}

/// Concrete EngineRuntime with shared admission and one native model worker.
#[derive(Clone)]
pub struct FamilyRuntime {
    backends: Arc<BTreeMap<String, Arc<dyn RuntimeBackend>>>,
    admission: AdmissionController,
    limits: GenerationLimits,
    worker: Arc<NativeWorker>,
}

impl FamilyRuntime {
    /// Start a dedicated worker without loading a native model on this thread.
    ///
    /// Supply actual adapters and explicit budget/buffering/retention policy.
    /// Registry metadata must match each executable adapter's registration.
    pub fn new(
        backends: Vec<Arc<dyn RuntimeBackend>>,
        admission: AdmissionController,
        limits: GenerationLimits,
        retention: RetentionPolicy,
    ) -> Result<Self, EngineError> {
        if backends.is_empty() {
            return Err(EngineError::invalid_request(
                "backends",
                "must not be empty",
            ));
        }
        if retention.retain_cache_buffers && !retention.retain_model {
            return Err(EngineError::invalid_request(
                "retention",
                "retaining cache buffers requires retaining their model",
            ));
        }
        let mut entries = BTreeMap::new();
        for backend in backends {
            let name = backend.registration().family().to_owned();
            if entries.contains_key(&name) {
                return Err(EngineError::invalid_request(
                    "backends",
                    "duplicate family adapter",
                ));
            }
            entries.insert(name, backend);
        }
        let summary = Arc::new(Mutex::new(Summary::empty(true)));
        let exit = Arc::new(ExitSignal {
            exited: Mutex::new(false),
            changed: Condvar::new(),
        });
        // Rendezvous transport: no hidden queue of generation requests.
        let (sender, receiver) = mpsc::sync_channel(0);
        let worker_summary = Arc::clone(&summary);
        let worker_exit = Arc::clone(&exit);
        let worker_admission = admission.clone();
        thread::Builder::new()
            .name("engine-native-model".to_owned())
            .spawn(move || {
                // Locals drop in reverse order: native state first, exit signal last.
                let _lifetime = WorkerLifetime {
                    summary: Arc::clone(&worker_summary),
                    exit: worker_exit,
                };
                let mut state = NativeState {
                    active: None,
                    pending: None,
                    resident: None,
                    admission: worker_admission,
                    retention,
                    summary: worker_summary,
                };
                state.run(receiver);
            })
            .map_err(|error| EngineError::ExecutionFailed {
                stage: "spawn native model worker",
                message: error.to_string(),
            })?;
        Ok(Self {
            backends: Arc::new(entries),
            admission,
            limits,
            worker: Arc::new(NativeWorker {
                sender,
                summary,
                exit,
            }),
        })
    }

    /// Inspect the shared memory ledger without acquiring a reservation.
    pub fn admission_snapshot(&self) -> Result<AdmissionSnapshot, EngineError> {
        self.admission.snapshot()
    }

    fn backend(
        &self,
        registry: &Registry,
        artifact: &ArtifactRegistration,
    ) -> Result<Arc<dyn RuntimeBackend>, EngineError> {
        let backend =
            self.backends
                .get(artifact.family())
                .ok_or_else(|| EngineError::NotFound {
                    resource: "executable family",
                    id: artifact.family().to_owned(),
                })?;
        let expected = registry.family(artifact.family())?;
        let actual = backend.registration();
        if actual.implementation_version() != expected.implementation_version()
            || actual.architectures() != expected.architectures()
            || actual.capabilities() != expected.capabilities()
        {
            return Err(internal(
                "executable family does not match registry metadata",
            ));
        }
        Ok(Arc::clone(backend))
    }
}

impl EngineRuntime for FamilyRuntime {
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        Box::pin(async move {
            let artifact = registry.resolve_profile(context.profile_id())?.artifact();
            self.backend(registry, artifact)?
                .render_and_count(registry, context)
                .await
        })
    }

    fn estimate_memory<'a>(
        &'a self,
        registry: &'a Registry,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, MemoryEstimate> {
        Box::pin(async move {
            let resolved = registry.resolve_profile(&profile)?;
            let backend = self.backend(registry, resolved.artifact())?;
            let plan = backend.plan(registry, &profile, input_tokens).await?;
            validate_plan(registry, &profile, input_tokens, &plan, self.limits)?;
            let summary = lock_metadata(&self.worker.summary).clone();
            if !summary.alive {
                return Err(internal("native worker has exited"));
            }
            let same_model = summary.artifact.as_ref() == Some(resolved.artifact().id());
            let same_cache = same_model && summary.cache_matches(&plan);
            let mut available = self.admission.snapshot()?.available_bytes();
            // Credit only allocations this worker will actually evict while idle.
            if !summary.active {
                let reclaim = if !same_model {
                    add(summary.weights_bytes, summary.cache_bytes)?
                } else if !same_cache {
                    summary.cache_bytes
                } else {
                    0
                };
                available = add(available, reclaim)?;
            }
            let loading_extra = if same_model {
                0
            } else {
                plan.load_limits
                    .max_load_peak_bytes()
                    .checked_sub(plan.load_limits.max_resident_bytes())
                    .ok_or_else(|| internal("invalid loader overlap bounds"))?
            };
            MemoryEstimate::new(
                MemoryBreakdown {
                    weights_bytes: if same_model {
                        0
                    } else {
                        plan.load_limits.max_resident_bytes()
                    },
                    kv_bytes: if same_cache {
                        0
                    } else {
                        plan.cache.max_cache_bytes()
                    },
                    temporary_bytes: add(plan.temporary_bytes, loading_extra)?,
                    output_bytes: plan.output_bytes,
                    draft_bytes: 0,
                    headroom_bytes: plan.headroom_bytes,
                },
                available,
                Vec::new(),
            )
        })
    }

    fn start_generation(
        &self,
        registry: Arc<Registry>,
        request: GenerationRequest,
        rendered: RenderedContext,
        execution: ExecutionContext,
    ) -> EngineFuture<'_, Generation> {
        Box::pin(async move {
            execution.check_cancelled()?;
            if execution.request_id() != request.request_id()
                || rendered.profile_id() != request.profile_id()
                || rendered.snapshot_id() != request.context().snapshot_id()
            {
                return Err(internal("generation startup identities disagree"));
            }
            if !matches!(request.options().decoding(), DecodingMode::Greedy) {
                return Err(EngineError::unsupported(
                    "decoding mode",
                    "this runtime implements greedy selection only",
                ));
            }
            registry.validate_generation_options(request.profile_id(), request.options())?;
            let resolved = registry.resolve_profile(request.profile_id())?;
            if rendered.artifact_id() != resolved.artifact().id() {
                return Err(internal(
                    "rendered artifact does not match selected profile",
                ));
            }
            let backend = self.backend(&registry, resolved.artifact())?;
            let plan = backend
                .plan(&registry, request.profile_id(), rendered.token_count())
                .await?;
            validate_plan(
                &registry,
                request.profile_id(),
                rendered.token_count(),
                &plan,
                self.limits,
            )?;
            execution.check_cancelled()?;
            let setup = Setup {
                registry,
                backend,
                plan,
                profile: request.profile_id().clone(),
                options: request.options().clone(),
            };
            // Reclaim incompatible idle resources and reserve output on the
            // native worker before generation.rs allocates its event queue.
            // The handoff owns execution until accepted or fully cleaned up.
            let admitted = self.worker.admit(setup, execution)?.await?;
            let output = Arc::clone(&admitted.session._output);
            let generation = generation::start_generation(
                admitted.execution,
                rendered,
                request.options().clone(),
                Box::new(admitted.session),
                self.limits,
            )?;
            let Generation { request_id, events } = generation;
            Ok(Generation {
                request_id,
                events: Box::pin(BudgetedStream {
                    inner: events,
                    _output: output,
                }),
            })
        })
    }

    fn release_session_state(&self, _state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        // No canonical SessionStateId registry is implemented by this runtime.
        // Its reusable empty KV allocations are not published session states.
        Box::pin(async { Ok(ReleaseOutcome::AlreadyAbsent) })
    }

    fn capabilities<'a>(
        &'a self,
        registry: &'a Registry,
        artifact_id: ArtifactId,
    ) -> EngineFuture<'a, Capabilities> {
        Box::pin(async move {
            let artifact = registry.artifact(&artifact_id)?;
            self.backend(registry, artifact)?;
            let summary = lock_metadata(&self.worker.summary);
            if !summary.alive {
                return Err(internal("native worker has exited"));
            }
            if summary.artifact.as_ref() != Some(&artifact_id) {
                return Err(EngineError::NotFound {
                    resource: "verified loaded artifact",
                    id: artifact_id.into_string(),
                });
            }
            summary
                .capabilities
                .clone()
                .ok_or_else(|| internal("loaded capabilities unavailable"))
        })
    }
}

struct BudgetedStream {
    inner: GenerationStream,
    // Declared after inner: destroy queue ownership before this Arc owner.
    _output: Arc<MemoryReservation>,
}

impl GenerationEventStream for BudgetedStream {
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

#[derive(Clone)]
struct Setup {
    registry: Arc<Registry>,
    backend: Arc<dyn RuntimeBackend>,
    plan: RuntimePlan,
    profile: ModelProfileId,
    options: GenerationOptions,
}

struct Reply<T> {
    result: Result<T, EngineError>,
    usage: Usage,
}

enum Command {
    AdmitOutput {
        setup: Setup,
        control: ExecutionControl,
        abandoned: Arc<AtomicBool>,
        reply: SyncSender<Reply<Arc<MemoryReservation>>>,
    },
    Prepare {
        setup: Setup,
        context: RenderedContext,
        control: ExecutionControl,
        reply: SyncSender<Reply<()>>,
    },
    Decode {
        request: RequestId,
        reply: SyncSender<Reply<TokenStep>>,
    },
    Flush {
        request: RequestId,
        reply: SyncSender<Reply<String>>,
    },
    Finish {
        request: RequestId,
        reply: SyncSender<Reply<()>>,
    },
}

struct NativeWorker {
    sender: SyncSender<Command>,
    summary: Arc<Mutex<Summary>>,
    exit: Arc<ExitSignal>,
}

impl NativeWorker {
    fn admit(
        self: &Arc<Self>,
        setup: Setup,
        execution: ExecutionContext,
    ) -> Result<AdmissionFuture, EngineError> {
        let (future, shared) = admission_channel();
        let worker = Arc::clone(self);
        let abandoned = Arc::clone(&shared.abandoned);
        thread::Builder::new()
            .name("engine-startup-admission".to_owned())
            .spawn(move || {
                // This dispatcher, not the async caller, retains execution
                // while the native worker reclaims resources or unwinds.
                let result = (|| {
                    let request_id = execution.request_id().clone();
                    let control =
                        ExecutionControl::new(request_id.clone(), execution.cancellation_token());
                    check_startup(&control, &abandoned)?;
                    let output = worker
                        .call(|reply| Command::AdmitOutput {
                            setup: setup.clone(),
                            control,
                            abandoned: Arc::clone(&abandoned),
                            reply,
                        })?
                        .result?;
                    let session = NativeSession {
                        worker: Arc::clone(&worker),
                        options: setup.options.clone(),
                        setup: Some(setup),
                        request_id,
                        usage: Usage::default(),
                        // Admission registered pending native ownership.
                        // Cleanup is needed even if prefill never starts.
                        started: true,
                        _output: output,
                    };
                    let admitted = AdmittedStartup { session, execution };
                    admitted.execution.check_cancelled()?;
                    if abandoned.load(Ordering::Acquire) {
                        return Err(EngineError::Cancelled {
                            request_id: admitted.execution.request_id().as_str().to_owned(),
                        });
                    }
                    Ok(admitted)
                })();
                complete_admission(shared, result);
            })
            .map_err(|error| EngineError::ExecutionFailed {
                stage: "spawn startup admission dispatcher",
                message: error.to_string(),
            })?;
        Ok(future)
    }

    fn call<T>(
        &self,
        command: impl FnOnce(SyncSender<Reply<T>>) -> Command,
    ) -> Result<Reply<T>, EngineError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        if let Err(unsent) = self.sender.send(command(sender)) {
            // Retain command ownership until native destruction completes.
            self.exit.wait();
            drop(unsent);
            return Err(internal("native worker exited before accepting execution"));
        }
        match receiver.recv() {
            Ok(reply) => Ok(reply),
            Err(_) => {
                // Do not relinquish scheduler ownership while a failed
                // native worker is still destroying model resources.
                self.exit.wait();
                Err(EngineError::ExecutionFailed {
                    stage: "native model worker",
                    message: "worker exited before completing execution and cleanup".to_owned(),
                })
            }
        }
    }
}

// Field order is intentional: pending native ownership is cleaned up
// before execution ownership is relinquished on an abandoned handoff.
struct AdmittedStartup {
    session: NativeSession,
    execution: ExecutionContext,
}

struct AdmissionShared {
    state: Mutex<AdmissionReplyState>,
    changed: Condvar,
    abandoned: Arc<AtomicBool>,
}

struct AdmissionReplyState {
    result: Option<Result<AdmittedStartup, EngineError>>,
    consumed: bool,
    waker: Option<std::task::Waker>,
}

struct AdmissionFuture {
    shared: Arc<AdmissionShared>,
    finished: bool,
}

fn admission_channel() -> (AdmissionFuture, Arc<AdmissionShared>) {
    let shared = Arc::new(AdmissionShared {
        state: Mutex::new(AdmissionReplyState {
            result: None,
            consumed: false,
            waker: None,
        }),
        changed: Condvar::new(),
        abandoned: Arc::new(AtomicBool::new(false)),
    });
    (
        AdmissionFuture {
            shared: Arc::clone(&shared),
            finished: false,
        },
        shared,
    )
}

impl Future for AdmissionFuture {
    type Output = Result<AdmittedStartup, EngineError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(Err(internal("startup admission polled after completion")));
        }
        let mut replacement = Some(cx.waker().clone());
        let old_waker;
        let result;
        {
            let mut state = lock_metadata(&this.shared.state);
            if let Some(reply) = state.result.take() {
                state.consumed = true;
                this.finished = true;
                old_waker = state.waker.take();
                result = Poll::Ready(reply);
            } else {
                old_waker = std::mem::replace(&mut state.waker, replacement.take());
                result = Poll::Pending;
            }
        }
        drop(old_waker);
        drop(replacement);
        if this.finished {
            this.shared.changed.notify_all();
        }
        result
    }
}

impl Drop for AdmissionFuture {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let old_waker;
        {
            let mut state = lock_metadata(&self.shared.state);
            self.shared.abandoned.store(true, Ordering::Release);
            old_waker = state.waker.take();
        }
        drop(old_waker);
        self.shared.changed.notify_all();
        // The dispatcher owns cleanup. Never block an async executor here
        // waiting for native destruction or a Finish acknowledgment.
    }
}

fn complete_admission(shared: Arc<AdmissionShared>, result: Result<AdmittedStartup, EngineError>) {
    let mut unpublished = Some(result);
    let consumer_waker;
    {
        let mut state = lock_metadata(&shared.state);
        if !shared.abandoned.load(Ordering::Acquire) {
            state.result = unpublished.take();
        }
        consumer_waker = state.waker.take();
    }
    drop(unpublished);
    if let Some(waker) = consumer_waker {
        waker.wake();
    }
    let discarded;
    let old_waker;
    {
        let mut state = lock_metadata(&shared.state);
        while !state.consumed && !shared.abandoned.load(Ordering::Acquire) {
            state = match shared.changed.wait(state) {
                Ok(guard) => guard,
                Err(error) => error.into_inner(),
            };
        }
        discarded = state.result.take();
        old_waker = state.waker.take();
    }
    drop(old_waker);
    // If startup was abandoned, NativeSession's Finish acknowledgment is
    // awaited on this dispatcher before the execution permit is dropped.
    drop(discarded);
}

fn check_startup(control: &ExecutionControl, abandoned: &AtomicBool) -> Result<(), EngineError> {
    control.check_cancelled()?;
    if abandoned.load(Ordering::Acquire) {
        return Err(EngineError::Cancelled {
            request_id: control.request_id().as_str().to_owned(),
        });
    }
    Ok(())
}

struct ExitSignal {
    exited: Mutex<bool>,
    changed: Condvar,
}

impl ExitSignal {
    fn wait(&self) {
        let mut exited = lock_metadata(&self.exited);
        while !*exited {
            exited = match self.changed.wait(exited) {
                Ok(guard) => guard,
                Err(error) => error.into_inner(),
            };
        }
    }
}

struct WorkerLifetime {
    summary: Arc<Mutex<Summary>>,
    exit: Arc<ExitSignal>,
}

impl Drop for WorkerLifetime {
    fn drop(&mut self) {
        *lock_metadata(&self.summary) = Summary::empty(false);
        *lock_metadata(&self.exit.exited) = true;
        self.exit.changed.notify_all();
    }
}

// Only this proxy crosses into generation.rs's Send session. All native
// objects remain in NativeState on the dedicated model thread.
struct NativeSession {
    worker: Arc<NativeWorker>,
    setup: Option<Setup>,
    request_id: RequestId,
    options: GenerationOptions,
    usage: Usage,
    started: bool,
    _output: Arc<MemoryReservation>,
}

impl GenerationSession for NativeSession {
    fn prefill(
        &mut self,
        context: &RenderedContext,
        options: &GenerationOptions,
        cancellation: &CancellationToken,
    ) -> Result<(), EngineError> {
        cancellation.check_cancelled(&self.request_id)?;
        if options != &self.options {
            return Err(internal("session generation options changed"));
        }
        let setup = self
            .setup
            .take()
            .ok_or_else(|| internal("session prefill invoked twice"))?;
        let control = ExecutionControl::new(self.request_id.clone(), cancellation.clone());
        self.started = true;
        let reply = self.worker.call(|reply| Command::Prepare {
            setup,
            context: context.clone(),
            control,
            reply,
        })?;
        self.usage = reply.usage;
        reply.result
    }

    fn decode_next(
        &mut self,
        options: &GenerationOptions,
        cancellation: &CancellationToken,
    ) -> Result<TokenStep, EngineError> {
        cancellation.check_cancelled(&self.request_id)?;
        if options != &self.options {
            return Err(internal("session generation options changed"));
        }
        let reply = self.worker.call(|reply| Command::Decode {
            request: self.request_id.clone(),
            reply,
        })?;
        self.usage = reply.usage;
        reply.result
    }

    fn finish_text(&mut self) -> Result<String, EngineError> {
        let reply = self.worker.call(|reply| Command::Flush {
            request: self.request_id.clone(),
            reply,
        })?;
        self.usage = reply.usage;
        reply.result
    }

    fn usage(&self) -> Usage {
        self.usage
    }
}

impl Drop for NativeSession {
    fn drop(&mut self) {
        if self.started {
            // The acknowledgment is sent only AFTER request-local native
            // resources are dropped and idle retention policy is applied.
            let _ = self.worker.call(|reply| Command::Finish {
                request: self.request_id.clone(),
                reply,
            });
        }
    }
}

#[derive(Clone)]
struct Summary {
    alive: bool,
    active: bool,
    artifact: Option<ArtifactId>,
    weights_bytes: u64,
    cache_bytes: u64,
    cache_mode: Option<CacheMode>,
    cache_capacity: usize,
    capabilities: Option<Capabilities>,
}

impl Summary {
    fn empty(alive: bool) -> Self {
        Self {
            alive,
            active: false,
            artifact: None,
            weights_bytes: 0,
            cache_bytes: 0,
            cache_mode: None,
            cache_capacity: 0,
            capabilities: None,
        }
    }
    fn cache_matches(&self, plan: &RuntimePlan) -> bool {
        self.cache_mode.as_ref() == Some(plan.cache.mode())
            && self.cache_capacity >= plan.cache.capacity_tokens()
            && self.cache_bytes >= plan.cache.max_cache_bytes()
    }
}

// Field order is intentional: actual native allocations precede guards.
struct ResidentCache {
    cache: Box<dyn FamilyCache>,
    reservation: MemoryReservation,
}
struct Resident {
    cache: Option<ResidentCache>,
    model: Box<dyn LoadedModel>,
    weights: MemoryReservation,
    capabilities: Capabilities,
}
struct Active {
    decoder: Box<dyn TextDecoder>,
    logits: Option<TokenLogits>,
    eos: Vec<u32>,
    control: ExecutionControl,
    options: GenerationOptions,
    usage: Usage,
    _workspace: Option<MemoryReservation>,
}
struct NativeState {
    active: Option<Active>,
    pending: Option<RequestId>,
    resident: Option<Resident>,
    admission: AdmissionController,
    retention: RetentionPolicy,
    summary: Arc<Mutex<Summary>>,
}

impl NativeState {
    fn run(&mut self, receiver: Receiver<Command>) {
        while let Ok(command) = receiver.recv() {
            match command {
                Command::AdmitOutput {
                    setup,
                    control,
                    abandoned,
                    reply,
                } => {
                    let result = self.admit_output(&setup, &control, &abandoned);
                    self.publish();
                    // If the reply receiver disappeared, cancel the pending
                    // claim before dropping the admitted output reservation.
                    if let Err(undelivered) = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    }) {
                        if undelivered.0.result.is_ok() {
                            let _ = self.finish(control.request_id());
                            self.publish();
                        }
                        drop(undelivered);
                    }
                }
                Command::Prepare {
                    setup,
                    context,
                    control,
                    reply,
                } => {
                    let request = control.request_id().clone();
                    let result = self.prepare(setup, &context, control);
                    if result.is_err() {
                        self.invalidate_cache_for(&request);
                    }
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: self.usage(&request),
                    });
                }
                Command::Decode { request, reply } => {
                    let result = self.decode(&request);
                    if result.is_err() {
                        self.invalidate_cache_for(&request);
                    }
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: self.usage(&request),
                    });
                }
                Command::Flush { request, reply } => {
                    let result = self.flush(&request);
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: self.usage(&request),
                    });
                }
                Command::Finish { request, reply } => {
                    let usage = self.usage(&request);
                    let result = self.finish(&request);
                    self.publish();
                    let _ = reply.send(Reply { result, usage });
                }
            }
        }
    }

    fn admit_output(
        &mut self,
        setup: &Setup,
        control: &ExecutionControl,
        abandoned: &AtomicBool,
    ) -> Result<Arc<MemoryReservation>, EngineError> {
        if let Some(owner) = self
            .active
            .as_ref()
            .map(|active| active.control.request_id())
            .or(self.pending.as_ref())
        {
            return Err(EngineError::Busy {
                active_request_id: Some(owner.as_str().to_owned()),
                message: "native worker already owns an active or pending request".to_owned(),
            });
        }
        check_startup(control, abandoned)?;
        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        setup.backend.verify_artifact(artifact, control)?;
        check_startup(control, abandoned)?;
        let admission = self.admission.clone();
        let output = reserve_output_after_reclaim(&admission, setup.plan.output_bytes, || {
            let same_model = self
                .resident
                .as_ref()
                .is_some_and(|resident| resident.model.artifact_id() == artifact.id());
            if !same_model {
                drop(self.resident.take());
            } else if let Some(resident) = &mut self.resident {
                if resident
                    .cache
                    .as_ref()
                    .is_some_and(|cache| !cache_matches(cache, &setup.plan))
                {
                    drop(resident.cache.take());
                }
            }
            // Publish only after actual native resources and their
            // guards have been destroyed; estimates remain advisory.
            self.publish();
            check_startup(control, abandoned)
        })?;
        self.pending = Some(control.request_id().clone());
        Ok(output)
    }

    fn prepare(
        &mut self,
        setup: Setup,
        context: &RenderedContext,
        control: ExecutionControl,
    ) -> Result<(), EngineError> {
        if let Some(active) = &self.active {
            return Err(EngineError::Busy {
                active_request_id: Some(active.control.request_id().as_str().to_owned()),
                message: "native worker already owns an active request".to_owned(),
            });
        }
        if self.pending.as_ref() != Some(control.request_id()) {
            return Err(internal("prefill has no matching admitted startup"));
        }
        control.check_cancelled()?;
        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        setup.backend.verify_artifact(artifact, &control)?;
        control.check_cancelled()?;
        let same_model = self
            .resident
            .as_ref()
            .is_some_and(|r| r.model.artifact_id() == artifact.id());
        if !same_model {
            // Destroy the old native cache/model before releasing their guards.
            drop(self.resident.take());
        } else if self
            .resident
            .as_ref()
            .and_then(|r| r.cache.as_ref())
            .is_some_and(|c| !cache_matches(c, &setup.plan))
        {
            if let Some(resident) = &mut self.resident {
                drop(resident.cache.take());
            }
        }
        let workspace_bytes = add(setup.plan.temporary_bytes, setup.plan.headroom_bytes)?;
        let workspace = if workspace_bytes == 0 {
            None
        } else {
            Some(
                self.admission
                    .reserve("generation workspaces", workspace_bytes)?,
            )
        };
        if self.resident.is_none() {
            // Full loader overlap is admitted BEFORE ModelFamily::load.
            let mut weights = self.admission.reserve(
                "model loading peak",
                setup.plan.load_limits.max_load_peak_bytes(),
            )?;
            let model = setup.backend.load(ModelLoadRequest::new(
                artifact,
                setup.plan.load_limits,
                &control,
            ))?;
            if model.artifact_id() != artifact.id()
                || model.registration().implementation_version()
                    != setup.backend.registration().implementation_version()
                || model.registration().family() != artifact.family()
            {
                return Err(internal(
                    "loader returned a different artifact or implementation",
                ));
            }
            let observed = model.memory_usage()?.allocated_bytes();
            if observed > setup.plan.load_limits.max_resident_bytes() {
                return Err(EngineError::MemoryLimitExceeded {
                    operation: "loaded model buffers",
                    required_bytes: observed,
                    available_bytes: setup.plan.load_limits.max_resident_bytes(),
                });
            }
            let mut capabilities = model.capabilities().clone();
            if capabilities.family != artifact.family()
                || capabilities.task != "text-generation"
                || !capabilities.supports_incremental_kv
                || !capabilities
                    .decoding_modes
                    .contains(&DecodingCapability::Greedy)
            {
                return Err(EngineError::unsupported(
                    "loaded execution path",
                    "requires actual greedy text generation with incremental KV",
                ));
            }
            // These are the capabilities of this runtime, not just the model.
            capabilities.supports_prefix_reuse = false;
            capabilities.supports_streaming = true;
            capabilities.decoding_modes = vec![DecodingCapability::Greedy];
            // Loading sources/temporaries must have been released by load().
            weights.try_resize(setup.plan.load_limits.max_resident_bytes())?;
            self.resident = Some(Resident {
                cache: None,
                model,
                weights,
                capabilities,
            });
        }
        control.check_cancelled()?;
        let resident = self
            .resident
            .as_mut()
            .ok_or_else(|| internal("loaded model unavailable"))?;
        if setup
            .registry
            .profile(&setup.profile)?
            .context_limit_tokens()
            > resident.capabilities.context_limit_tokens
        {
            return Err(EngineError::unsupported(
                "context limit",
                "profile exceeds the actual loaded model limit",
            ));
        }
        if !resident
            .capabilities
            .cache_modes
            .contains(setup.plan.cache.mode())
        {
            return Err(EngineError::unsupported(
                "cache mode",
                "actual loaded model does not support the planned cache",
            ));
        }
        if resident
            .cache
            .as_ref()
            .is_some_and(|c| !cache_matches(c, &setup.plan))
        {
            drop(resident.cache.take());
        }
        if resident.cache.is_none() {
            let reservation = self
                .admission
                .reserve("model KV buffers", setup.plan.cache.max_cache_bytes())?;
            let model = resident.model.autoregressive().ok_or_else(|| {
                EngineError::unsupported("task", "loaded model has no autoregressive execution")
            })?;
            let cache = model.allocate_cache(&setup.plan.cache, &control)?;
            if cache.owner() != model.instance_id()
                || cache.mode() != setup.plan.cache.mode()
                || cache.representation() != &setup.plan.cache.mode().representation
                || cache.capacity_tokens() < setup.plan.cache.capacity_tokens()
                || cache.len() != 0
            {
                return Err(internal("family allocated an incompatible cache"));
            }
            check_cache_bytes(cache.as_ref(), reservation.bytes())?;
            resident.cache = Some(ResidentCache { cache, reservation });
        }
        let cache = resident
            .cache
            .as_mut()
            .ok_or_else(|| internal("allocated cache unavailable"))?;
        cache.cache.reset()?;
        if cache.cache.len() != 0 {
            return Err(internal("cache reset did not clear logical state"));
        }
        check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes())?;
        let decoder = setup.backend.decoder(&setup.registry, &setup.profile)?;
        let eos = setup.backend.eos_tokens(&setup.registry, &setup.profile)?;
        let model = resident
            .model
            .autoregressive()
            .ok_or_else(|| internal("autoregressive model disappeared"))?;
        let vocabulary = model.vocabulary_size();
        if vocabulary == 0
            || eos
                .iter()
                .any(|&token| usize::try_from(token).map_or(true, |t| t >= vocabulary))
        {
            return Err(internal("invalid vocabulary or EOS configuration"));
        }
        self.pending = None;
        self.active = Some(Active {
            decoder,
            logits: None,
            eos,
            control,
            options: setup.options,
            usage: Usage::default(),
            _workspace: workspace,
        });
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| internal("active execution unavailable"))?;
        for chunk in context
            .token_ids()
            .chunks(setup.plan.prefill_chunk_tokens.get())
        {
            active.control.check_cancelled()?;
            let before = cache.cache.len();
            // Usage is the number of real token IDs supplied to execution.
            active.usage.input_tokens = active
                .usage
                .input_tokens
                .checked_add(chunk.len())
                .ok_or_else(|| internal("input usage overflow"))?;
            let output = model.prefill(chunk, cache.cache.as_mut(), &active.control)?;
            let expected = before
                .checked_add(chunk.len())
                .ok_or_else(|| internal("cache position overflow"))?;
            if output.processed_tokens() != chunk.len()
                || cache.cache.len() != expected
                || output.logits().vocabulary_size() != vocabulary
            {
                return Err(internal(
                    "family prefill progress disagrees with supplied tokens",
                ));
            }
            check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes())?;
            active.logits = Some(output.into_logits());
        }
        active.control.check_cancelled()
    }

    fn decode(&mut self, request: &RequestId) -> Result<TokenStep, EngineError> {
        let active = self
            .active
            .as_mut()
            .filter(|a| a.control.request_id() == request)
            .ok_or_else(|| internal("decode has no matching active request"))?;
        active.control.check_cancelled()?;
        if active.usage.output_tokens >= active.options.max_output_tokens() {
            return Err(internal("decode exceeded admitted output allowance"));
        }
        let logits = active
            .logits
            .as_ref()
            .ok_or_else(|| internal("decode has no preceding logits"))?;
        let mut best = 0;
        for i in 1..logits.values().len() {
            if logits.values()[i] > logits.values()[best] {
                best = i;
            }
        }
        let token =
            u32::try_from(best).map_err(|_| internal("selected token exceeds u32 range"))?;
        let resident = self
            .resident
            .as_mut()
            .ok_or_else(|| internal("decode model unavailable"))?;
        let cache = resident
            .cache
            .as_mut()
            .ok_or_else(|| internal("decode cache unavailable"))?;
        let before = cache.cache.len();
        let model = resident
            .model
            .autoregressive()
            .ok_or_else(|| internal("decode model is not autoregressive"))?;
        // This token ID was actually selected from actual model logits.
        active.usage.output_tokens = active
            .usage
            .output_tokens
            .checked_add(1)
            .ok_or_else(|| internal("output usage overflow"))?;
        let next = model.decode(token, cache.cache.as_mut(), &active.control)?;
        if cache.cache.len()
            != before
                .checked_add(1)
                .ok_or_else(|| internal("cache position overflow"))?
            || next.vocabulary_size() != model.vocabulary_size()
        {
            return Err(internal(
                "family decode progress disagrees with the selected token",
            ));
        }
        check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes())?;
        active.logits = Some(next);
        let end_of_sequence = active.eos.contains(&token);
        let text_delta = if end_of_sequence {
            String::new()
        } else {
            active.decoder.push(token)?
        };
        Ok(TokenStep {
            token_id: token,
            text_delta,
            end_of_sequence,
        })
    }

    fn flush(&mut self, request: &RequestId) -> Result<String, EngineError> {
        let active = self
            .active
            .as_mut()
            .filter(|a| a.control.request_id() == request)
            .ok_or_else(|| internal("flush has no matching active request"))?;
        active.control.check_cancelled()?;
        active.decoder.finish()
    }

    fn finish(&mut self, request: &RequestId) -> Result<(), EngineError> {
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending != request)
        {
            return Err(internal("cleanup request does not own pending startup"));
        }
        if self
            .active
            .as_ref()
            .is_some_and(|a| a.control.request_id() != request)
        {
            return Err(internal(
                "cleanup request does not own the active execution",
            ));
        }
        self.pending = None;
        drop(self.active.take());
        if !self.retention.retain_model {
            drop(self.resident.take());
        } else if let Some(resident) = &mut self.resident {
            if !self.retention.retain_cache_buffers {
                drop(resident.cache.take());
            } else if let Some(cache) = &mut resident.cache {
                // A failed reset makes retained state unusable: discard it.
                if cache.cache.reset().is_err() || cache.cache.len() != 0 {
                    drop(resident.cache.take());
                }
            }
        }
        Ok(())
    }

    fn invalidate_cache_for(&mut self, request: &RequestId) {
        if self
            .active
            .as_ref()
            .is_some_and(|a| a.control.request_id() == request)
            || self.pending.as_ref() == Some(request)
        {
            if let Some(resident) = &mut self.resident {
                drop(resident.cache.take());
            }
        }
    }

    fn usage(&self, request: &RequestId) -> Usage {
        self.active
            .as_ref()
            .filter(|a| a.control.request_id() == request)
            .map_or(Usage::default(), |a| a.usage)
    }

    fn publish(&self) {
        let mut summary = Summary::empty(true);
        summary.active = self.active.is_some() || self.pending.is_some();
        if let Some(resident) = &self.resident {
            summary.artifact = Some(resident.model.artifact_id().clone());
            summary.weights_bytes = resident.weights.bytes();
            summary.capabilities = Some(resident.capabilities.clone());
            if let Some(cache) = &resident.cache {
                summary.cache_bytes = cache.reservation.bytes();
                summary.cache_mode = Some(cache.cache.mode().clone());
                summary.cache_capacity = cache.cache.capacity_tokens();
            }
        }
        *lock_metadata(&self.summary) = summary;
    }
}

/// Reclamation must finish before output admission is attempted. This is
/// also the exact ordering exercised by the reservation regression tests.
fn reserve_output_after_reclaim(
    admission: &AdmissionController,
    output_bytes: u64,
    reclaim: impl FnOnce() -> Result<(), EngineError>,
) -> Result<Arc<MemoryReservation>, EngineError> {
    reclaim()?;
    admission
        .reserve("generation output buffers", output_bytes)
        .map(Arc::new)
}

fn cache_matches(cache: &ResidentCache, plan: &RuntimePlan) -> bool {
    cache.cache.mode() == plan.cache.mode()
        && cache.cache.capacity_tokens() >= plan.cache.capacity_tokens()
        && cache.reservation.bytes() >= plan.cache.max_cache_bytes()
}

fn check_cache_bytes(cache: &dyn FamilyCache, limit: u64) -> Result<(), EngineError> {
    let allocated = cache.memory_usage()?.allocated_bytes();
    if allocated > limit {
        Err(EngineError::MemoryLimitExceeded {
            operation: "actual KV buffer allocation",
            required_bytes: allocated,
            available_bytes: limit,
        })
    } else {
        Ok(())
    }
}

fn validate_plan(
    registry: &Registry,
    profile: &ModelProfileId,
    input: usize,
    plan: &RuntimePlan,
    limits: GenerationLimits,
) -> Result<(), EngineError> {
    let resolved = registry.resolve_profile(profile)?;
    let profile = resolved.profile();
    profile.check_context(input)?;
    let capacity = input
        .checked_add(profile.max_output_tokens())
        .and_then(|v| v.checked_add(profile.safety_margin_tokens()))
        .ok_or_else(|| internal("planned context arithmetic overflow"))?;
    if plan.cache.capacity_tokens() < capacity
        || plan.cache.capacity_tokens() > profile.context_limit_tokens()
    {
        return Err(EngineError::invalid_request(
            "cache capacity",
            "must cover the full profile output reserve and safety margin within its context limit",
        ));
    }
    let headers = limits
        .event_capacity()
        .checked_mul(std::mem::size_of::<String>())
        .ok_or_else(|| internal("output header size overflow"))?;
    let minimum = limits
        .max_buffered_text_bytes()
        .checked_add(limits.max_text_delta_bytes())
        .and_then(|v| v.checked_add(headers))
        .ok_or_else(|| internal("output budget size overflow"))?;
    let minimum = u64::try_from(minimum).map_err(|_| internal("output budget exceeds u64"))?;
    if plan.output_bytes < minimum {
        return Err(EngineError::invalid_request(
            "output_bytes",
            "does not cover queued text, a pending delta, and queue headers",
        ));
    }
    add(plan.temporary_bytes, plan.headroom_bytes)?;
    Ok(())
}

fn add(left: u64, right: u64) -> Result<u64, EngineError> {
    left.checked_add(right)
        .ok_or_else(|| internal("runtime memory arithmetic overflow"))
}
fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        message: message.to_owned(),
    }
}

// Metadata/exit notification only. Never recovers the admission ledger.
fn lock_metadata<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(error) => error.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn control_and_generation_proxy_can_cross_task_boundaries() {
        assert_send_sync::<FamilyRuntime>();
        assert_send::<NativeSession>();
        assert_send::<Command>();
        assert_send::<AdmissionFuture>();
        assert_send::<AdmittedStartup>();
    }

    #[test]
    fn idle_reservations_are_reclaimed_before_output_admission() {
        const MIB: u64 = 1024 * 1024;
        let admission = AdmissionController::new(100 * MIB).expect("controller");
        let idle = admission
            .reserve("idle resources", 90 * MIB)
            .expect("idle guard");
        assert!(
            admission
                .reserve("output before reclamation", 20 * MIB)
                .is_err()
        );
        let output = reserve_output_after_reclaim(&admission, 20 * MIB, || {
            drop(idle);
            Ok(())
        })
        .expect("output fits after real reservation release");
        let snapshot = admission.snapshot().expect("snapshot");
        assert_eq!(snapshot.reserved_bytes(), 20 * MIB);
        assert_eq!(snapshot.active_reservations(), 1);
        drop(output);
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    }

    #[test]
    fn cache_reclamation_preserves_the_retained_model_reservation() {
        let admission = AdmissionController::new(100).expect("controller");
        let model = admission.reserve("retained model", 60).expect("model");
        let cache = admission.reserve("incompatible cache", 30).expect("cache");
        let output = reserve_output_after_reclaim(&admission, 20, || {
            drop(cache);
            Ok(())
        })
        .expect("output after cache release");
        assert_eq!(model.bytes(), 60);
        assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 80);
        drop(output);
        assert_eq!(
            admission
                .snapshot()
                .expect("model remains")
                .reserved_bytes(),
            60
        );
        drop(model);
    }

    #[test]
    fn unrelated_reservations_are_not_credited_as_reclaimable() {
        let admission = AdmissionController::new(100).expect("controller");
        let unrelated = admission.reserve("unrelated owner", 85).expect("unrelated");
        let idle = admission.reserve("idle resources", 10).expect("idle");
        let result = reserve_output_after_reclaim(&admission, 20, || {
            drop(idle);
            Ok(())
        });
        match result {
            Err(EngineError::MemoryLimitExceeded {
                required_bytes,
                available_bytes,
                ..
            }) => {
                assert_eq!(required_bytes, 20);
                assert_eq!(available_bytes, 15);
            }
            _ => panic!("expected output admission rejection"),
        }
        let snapshot = admission.snapshot().expect("snapshot");
        assert_eq!(snapshot.reserved_bytes(), 85);
        assert_eq!(snapshot.active_reservations(), 1);
        drop(unrelated);
    }

    #[test]
    fn reclamation_failure_does_not_acquire_output_budget() {
        let admission = AdmissionController::new(100).expect("controller");
        let _idle = admission.reserve("idle resources", 40).expect("idle");
        let before = admission.snapshot().expect("before");
        assert!(
            reserve_output_after_reclaim(&admission, 20, || {
                Err(internal("reclamation did not complete"))
            })
            .is_err()
        );
        assert_eq!(admission.snapshot().expect("after"), before);
    }

    #[test]
    fn pending_startup_can_be_finished_without_prefill() {
        let request = RequestId::new("pending-startup").expect("request");
        let summary = Arc::new(Mutex::new(Summary::empty(true)));
        let mut state = NativeState {
            active: None,
            pending: Some(request.clone()),
            resident: None,
            admission: AdmissionController::new(100).expect("controller"),
            retention: RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: true,
            },
            summary: Arc::clone(&summary),
        };
        state.publish();
        assert!(lock_metadata(&summary).active);
        state.finish(&request).expect("finish pending startup");
        state.publish();
        assert!(state.pending.is_none());
        assert!(!lock_metadata(&summary).active);
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
    fn admission_completion_wakes_the_registered_consumer() {
        let (mut future, shared) = admission_channel();
        let (sender, receiver) = mpsc::channel();
        let waker = std::task::Waker::from(Arc::new(SignalWake { sender }));
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
    fn abandoned_admission_does_not_wait_for_consumer_acknowledgment() {
        let (future, shared) = admission_channel();
        drop(future);
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            complete_admission(shared, Err(internal("abandoned startup")));
            sender.send(()).expect("completion signal");
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("abandoned completion must not wait for polling");
        worker.join().expect("completion cleanup");
    }
}
