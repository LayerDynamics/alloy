//! Admission-aware runtime with worker-local native model ownership.
//!
//! Native models, caches, decoders, and their allocation reservations stay
//! on a dedicated worker. Only command handles and metadata cross threads.
//!
//! Completed requests can retain exact executed-token history, actual KV,
//! and actual final-position logits. Ordinary generation may reuse that
//! state only after verifying native ownership, execution compatibility,
//! exact token-prefix compatibility, and retained-position availability.
//!
//! Supported reuse:
//!
//! - Identical input: reuse KV and retained final-position logits.
//! - Exact extension: reuse the complete retained prefix and prefill the
//!   missing suffix, provided the existing allocation satisfies the plan.
//!
//! Shortened, divergent, incompatible, or insufficiently retained context
//! rebuilds from an empty logical cache. This implementation does not
//! migrate KV into a replacement allocation or restore partial prefixes.
//!
//! Canonical publication remains explicit because RenderedContext does not
//! contain the tokenizer-produced prefix-byte payload required by the
//! canonical identity contract. Runtime prefix matching uses the actual
//! executed token IDs and actual native ownership; it does not fabricate
//! canonical tokenizer bytes.
//!
//! Leases pin published native state against release and mutation. Native
//! allocations and their reservations are destroyed before release
//! acknowledgements.
//!
//! Prefix-reuse capability reporting intentionally remains false until
//! production-path verification has passed.

use crate::{
    admission::{AdmissionController, AdmissionSnapshot, MemoryReservation},
    api::{ENGINE_API_VERSION, EngineFuture, EngineRuntime, ExecutionContext},
    cache_identity::{CacheIdentityInputs, compute_session_state_id},
    cancellation::CancellationToken,
    error::EngineError,
    family::{
        CacheAllocationRequest, ExecutionControl, FamilyCache, LoadedModel, ModelFamily,
        ModelInstanceId, ModelLoadLimits, ModelLoadRequest, TokenLogits,
    },
    generation::{self, GenerationLimits, GenerationSession, TokenStep},
    registry::{ArtifactRegistration, FamilyVersion, Registry},
    types::{
        ArtifactId, CacheKind, CacheMode, Capabilities, ContextSnapshot, DecodingCapability,
        DecodingMode, Generation, GenerationEvent, GenerationEventStream, GenerationOptions,
        GenerationRequest, GenerationStream, MemoryBreakdown, MemoryEstimate, ModelProfileId,
        ReleaseOutcome, RenderedContext, RendererIdentity, RequestId, SessionStateId, TokenizerId,
        Usage,
    },
};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    num::NonZeroUsize,
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    task::{Context, Poll, Waker},
    thread,
};

/// Stable identity of the Qwen3 host-reference KV representation.
///
/// Describes full-attention host-side f64 keys and values, not quantized
/// or MLX/Metal-resident KV storage.
///
/// Change this identifier when the retained cache layout or its
/// interpretation changes. Coordinate incompatible changes with the
/// family implementation version.
pub const QWEN3_CACHE_REPRESENTATION: &str = "qwen3.host-f64.full.v1";

/// Explicit lifetime policy for idle native resources.
#[derive(Clone, Copy, Debug)]
pub struct RetentionPolicy {
    /// Keep the last loaded model after a request completes.
    pub retain_model: bool,

    /// Keep allocated KV buffers after completion.
    ///
    /// Requires retain_model. Valid completed logical contents, token
    /// history, and final-position logits may remain available for exact
    /// prefix reuse and explicit canonical publication.
    pub retain_cache_buffers: bool,
}

/// Backend-derived bounds for the actual selected representation.
#[derive(Clone, Debug)]
pub struct RuntimePlan {
    /// Actual loader source/resident/overlap limits.
    pub load_limits: ModelLoadLimits,
    /// Actual family cache mode, capacity, and allocation limit.
    pub cache: CacheAllocationRequest,
    /// Execution workspace allowance, excluding loader overlap.
    pub temporary_bytes: u64,
    /// Event-queue, pending-text, and output allowance.
    pub output_bytes: u64,
    /// Additional request-local headroom.
    pub headroom_bytes: u64,
    /// Actual cancellable prefill chunk bound.
    pub prefill_chunk_tokens: NonZeroUsize,
}

/// Actual incremental tokenizer decoder, confined to the native worker.
pub trait TextDecoder {
    /// Decode one generated token without corrupting incremental state.
    fn push(&mut self, token: u32) -> Result<String, EngineError>;

    /// Finish pending tokenizer bytes after normal generation.
    fn finish(&mut self) -> Result<String, EngineError>;
}

/// Required real backend services in addition to the actual family loader.
pub trait RuntimeBackend: ModelFamily + 'static {
    /// Render and tokenize with verified profile inputs.
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext>;

    /// Compute bounds without loading native objects onto this thread.
    fn plan<'a>(
        &'a self,
        registry: &'a Registry,
        profile: &'a ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, RuntimePlan>;

    /// Verify the pinned artifact before loading or reuse.
    fn verify_artifact(
        &self,
        artifact: &ArtifactRegistration,
        control: &ExecutionControl,
    ) -> Result<(), EngineError>;

    /// Construct the actual profile-specific incremental decoder.
    fn decoder(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError>;

    /// Return the actual profile EOS token IDs.
    fn eos_tokens(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Vec<u32>, EngineError>;
}

/// Descriptive metadata for an actual retained native state.
///
/// A lookup is a snapshot. Acquire a lease to prevent subsequent release
/// or reclamation.
#[derive(Clone, Debug)]
pub struct RetainedSessionState {
    /// Canonical identity associated with the allocation.
    pub state_id: SessionStateId,
    /// Request that produced this state.
    pub request_id: RequestId,
    /// Verified artifact actually loaded.
    pub artifact_id: ArtifactId,
    /// Actual owning native model instance.
    pub model_instance_id: ModelInstanceId,
    /// Actual family implementation version.
    pub family_impl_version: FamilyVersion,
    /// Profile used for execution.
    pub profile_id: ModelProfileId,
    /// Pinned tokenizer identity.
    pub tokenizer_id: TokenizerId,
    /// Exact initial prompt template digest.
    pub template_hash: [u8; 32],
    /// Exact initial prompt rendering digest.
    pub render_hash: [u8; 32],
    /// Exact renderer identity.
    pub renderer: RendererIdentity,
    /// Actual cache mode.
    pub cache_mode: CacheMode,
    /// Logical positions processed.
    pub logical_tokens: usize,
    /// Physically retained positions.
    pub retained_tokens: usize,
    /// Allocated logical capacity.
    pub capacity_tokens: usize,
    /// Retained model-weight reservation.
    pub weights_reserved_bytes: u64,
    /// Retained KV-buffer reservation.
    pub cache_reserved_bytes: u64,
    /// Retained exact-token-history reservation.
    pub token_history_reserved_bytes: u64,
    /// Retained final-position-logits reservation.
    pub logits_reserved_bytes: u64,
    /// Input positions reused by the producing request.
    pub reused_input_tokens: usize,
    /// Input positions newly executed by the producing request.
    pub prefilled_input_tokens: usize,
}

/// RAII pin protecting actual native state from release and mutation.
///
/// Dropping a lease releases only its pin. It does not issue a blocking
/// worker command or move native ownership off the worker.
#[derive(Debug)]
pub struct SessionStateLease {
    state: RetainedSessionState,
    _pin: Arc<()>,
}

impl SessionStateLease {
    /// Metadata describing the pinned native allocation.
    pub fn state(&self) -> &RetainedSessionState {
        &self.state
    }

    /// Canonical identity of the pinned allocation.
    pub fn state_id(&self) -> SessionStateId {
        self.state.state_id
    }
}

/// Concrete engine runtime with one native-model worker.
#[derive(Clone)]
pub struct FamilyRuntime {
    backends: Arc<BTreeMap<String, Arc<dyn RuntimeBackend>>>,
    admission: AdmissionController,
    limits: GenerationLimits,
    retention: RetentionPolicy,
    worker: Arc<NativeWorker>,
}

impl FamilyRuntime {
    /// Start the worker without loading native model objects here.
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
        let (sender, receiver) = mpsc::sync_channel(0);
        let worker_summary = Arc::clone(&summary);
        let worker_exit = Arc::clone(&exit);
        let worker_admission = admission.clone();

        thread::Builder::new()
            .name("engine-native-model".to_owned())
            .spawn(move || {
                // Native state drops before the exit notification.
                let _lifetime = WorkerLifetime {
                    summary: Arc::clone(&worker_summary),
                    exit: worker_exit,
                };
                let mut state = NativeState {
                    active: None,
                    pending: None,
                    resident: None,
                    states: HashMap::new(),
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
            retention,
            worker: Arc::new(NativeWorker {
                sender,
                summary,
                exit,
            }),
        })
    }

    /// Inspect the ledger without acquiring a reservation.
    pub fn admission_snapshot(&self) -> Result<AdmissionSnapshot, EngineError> {
        self.admission.snapshot()
    }

    /// Publish the actual completed request under a canonical identity.
    ///
    /// Call after native cleanup, normally after the terminal event.
    /// Both retention flags must be enabled and valid completed state
    /// must still exist.
    ///
    /// token_prefix_bytes must be the exact tokenizer-produced bytes for
    /// the full executed sequence, including generated EOS when processed.
    /// This API trusts provenance; it does not synthesize these bytes.
    ///
    /// Publication is idempotent for the same request and identity.
    /// Dropping the future does not undo an already-completed publication.
    pub fn retain_session_state(
        &self,
        request_id: RequestId,
        token_prefix_bytes: Vec<u8>,
    ) -> EngineFuture<'_, SessionStateId> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            let bytes = bytes_as_u64(token_prefix_bytes.capacity())?;
            let reservation = if bytes == 0 {
                None
            } else {
                Some(
                    self.admission
                        .reserve("canonical state publication bytes", bytes)?,
                )
            };

            dispatch("engine-state-publication", move || {
                worker
                    .call(|reply| Command::RetainState {
                        request_id,
                        token_prefix_bytes,
                        _bytes_reservation: reservation,
                        reply,
                    })?
                    .result
            })?
            .await
        })
    }

    /// Look up actual canonical state without pinning it.
    pub fn lookup_session_state(
        &self,
        state_id: SessionStateId,
    ) -> EngineFuture<'_, Option<RetainedSessionState>> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-state-lookup", move || {
                worker
                    .call(|reply| Command::LookupState { state_id, reply })?
                    .result
            })?
            .await
        })
    }

    /// Pin actual canonical state against release and mutation.
    ///
    /// A live lease prevents ordinary generation from consuming or
    /// reclaiming this worker's published allocation.
    pub fn lease_session_state(
        &self,
        state_id: SessionStateId,
    ) -> EngineFuture<'_, Option<SessionStateLease>> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-state-lease", move || {
                worker
                    .call(|reply| Command::LeaseState { state_id, reply })?
                    .result
            })?
            .await
        })
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

            if !summary.active && summary.pin.strong_count() == 0 {
                let reclaim = if !same_model {
                    add(
                        add(summary.weights_bytes, summary.cache_bytes)?,
                        add(summary.history_bytes, summary.logits_bytes)?,
                    )?
                } else if !same_cache {
                    add(
                        summary.cache_bytes,
                        add(summary.history_bytes, summary.logits_bytes)?,
                    )?
                } else {
                    // Compatible candidate storage is retained through
                    // startup. Do not credit it before actual release.
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

            // Token-history storage is an explicit runtime allocation.
            // Retained logits are transferred from backend-accounted
            // execution output; their separate idle reservation is
            // acquired and reconciled before request cleanup.
            let history = if retains_contents(self.retention) {
                token_storage_bytes(plan.cache.capacity_tokens())?
            } else {
                0
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
                    temporary_bytes: add(add(plan.temporary_bytes, loading_extra)?, history)?,
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

    fn release_session_state(&self, state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-state-release", move || {
                worker
                    .call(|reply| Command::ReleaseState { state_id, reply })?
                    .result
            })?
            .await
        })
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
    RetainState {
        request_id: RequestId,
        token_prefix_bytes: Vec<u8>,
        _bytes_reservation: Option<MemoryReservation>,
        reply: SyncSender<Reply<SessionStateId>>,
    },
    LookupState {
        state_id: SessionStateId,
        reply: SyncSender<Reply<Option<RetainedSessionState>>>,
    },
    LeaseState {
        state_id: SessionStateId,
        reply: SyncSender<Reply<Option<SessionStateLease>>>,
    },
    ReleaseState {
        state_id: SessionStateId,
        reply: SyncSender<Reply<ReleaseOutcome>>,
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
            self.exit.wait();
            drop(unsent);
            return Err(internal("native worker exited before accepting execution"));
        }

        match receiver.recv() {
            Ok(reply) => Ok(reply),
            Err(_) => {
                self.exit.wait();
                Err(EngineError::ExecutionFailed {
                    stage: "native model worker",
                    message: "worker exited before completing execution and cleanup".to_owned(),
                })
            }
        }
    }
}

// Session cleanup precedes scheduler-permit destruction.
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
    waker: Option<Waker>,
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
    drop(discarded);
}

struct OperationShared<T> {
    state: Mutex<OperationReply<T>>,
}

struct OperationReply<T> {
    result: Option<Result<T, EngineError>>,
    abandoned: bool,
    waker: Option<Waker>,
}

struct OperationFuture<T> {
    shared: Arc<OperationShared<T>>,
    finished: bool,
}

fn dispatch<T: Send + 'static>(
    name: &'static str,
    operation: impl FnOnce() -> Result<T, EngineError> + Send + 'static,
) -> Result<OperationFuture<T>, EngineError> {
    let shared = Arc::new(OperationShared {
        state: Mutex::new(OperationReply {
            result: None,
            abandoned: false,
            waker: None,
        }),
    });
    let producer = Arc::clone(&shared);

    thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            let result = operation();
            let mut unpublished = Some(result);
            let waker;
            {
                let mut state = lock_metadata(&producer.state);
                if !state.abandoned {
                    state.result = unpublished.take();
                }
                waker = state.waker.take();
            }

            drop(unpublished);
            if let Some(waker) = waker {
                waker.wake();
            }
        })
        .map_err(|error| EngineError::ExecutionFailed {
            stage: "spawn state operation dispatcher",
            message: error.to_string(),
        })?;

    Ok(OperationFuture {
        shared,
        finished: false,
    })
}

impl<T> Future for OperationFuture<T> {
    type Output = Result<T, EngineError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(Err(internal("state operation polled after completion")));
        }

        let mut replacement = Some(cx.waker().clone());
        let old_waker;
        let result;
        {
            let mut state = lock_metadata(&this.shared.state);
            if let Some(reply) = state.result.take() {
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
        result
    }
}

impl<T> Drop for OperationFuture<T> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }

        let discarded;
        let old_waker;
        {
            let mut state = lock_metadata(&self.shared.state);
            state.abandoned = true;
            discarded = state.result.take();
            old_waker = state.waker.take();
        }
        drop(old_waker);
        drop(discarded);
    }
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
    pin: Weak<()>,
    artifact: Option<ArtifactId>,
    weights_bytes: u64,
    cache_bytes: u64,
    history_bytes: u64,
    logits_bytes: u64,
    cache_mode: Option<CacheMode>,
    cache_capacity: usize,
    capabilities: Option<Capabilities>,
}

impl Summary {
    fn empty(alive: bool) -> Self {
        Self {
            alive,
            active: false,
            pin: Weak::new(),
            artifact: None,
            weights_bytes: 0,
            cache_bytes: 0,
            history_bytes: 0,
            logits_bytes: 0,
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

// Actual native allocations precede their accounting guards.
struct ResidentCache {
    cache: Box<dyn FamilyCache>,
    reservation: MemoryReservation,
}

struct Resident {
    cache: Option<ResidentCache>,
    model: Box<dyn LoadedModel>,
    weights: MemoryReservation,
    capabilities: Capabilities,
    candidate: Option<StateCandidate>,
}

struct TokenHistory {
    tokens: Vec<u32>,
    reservation: MemoryReservation,
}

impl TokenHistory {
    fn new(admission: &AdmissionController, capacity: usize) -> Result<Self, EngineError> {
        let requested = token_storage_bytes(capacity)?;
        let mut reservation = admission.reserve("executed token history", requested)?;
        let mut tokens = Vec::new();

        tokens
            .try_reserve_exact(capacity)
            .map_err(|_| EngineError::AllocationFailed {
                operation: "allocate executed token history",
                requested_bytes: Some(requested),
            })?;
        reservation.try_resize(token_storage_bytes(tokens.capacity())?)?;

        Ok(Self {
            tokens,
            reservation,
        })
    }

    fn append(&mut self, tokens: &[u32]) -> Result<(), EngineError> {
        let required = self
            .tokens
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| internal("token history length overflow"))?;
        if required > self.tokens.capacity() {
            return Err(internal("executed tokens exceed admitted history capacity"));
        }
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }
}

/// Actual retained logits and their independent idle reservation.
struct RetainedLogits {
    logits: TokenLogits,
    reservation: MemoryReservation,
}

impl RetainedLogits {
    fn new(logits: TokenLogits, admission: &AdmissionController) -> Result<Self, EngineError> {
        let values = logits.into_values();
        let bytes = float_storage_bytes(values.capacity())?;
        let reservation = admission.reserve("retained final-position logits", bytes)?;
        let logits = TokenLogits::new(values)?;

        Ok(Self {
            logits,
            reservation,
        })
    }
}

#[derive(Clone)]
struct Compatibility {
    artifact_id: ArtifactId,
    model_instance_id: ModelInstanceId,
    family_impl_version: FamilyVersion,
    engine_api_version: u32,
    profile_id: ModelProfileId,
    tokenizer_id: TokenizerId,
    template_hash: [u8; 32],
    renderer: RendererIdentity,
    cache_mode: CacheMode,
}

impl Compatibility {
    fn new(model: &dyn LoadedModel, cache: &dyn FamilyCache, context: &RenderedContext) -> Self {
        Self {
            artifact_id: model.artifact_id().clone(),
            model_instance_id: model.instance_id().clone(),
            family_impl_version: model.registration().implementation_version(),
            engine_api_version: ENGINE_API_VERSION,
            profile_id: context.profile_id().clone(),
            tokenizer_id: context.tokenizer_id().clone(),
            template_hash: *context.template_hash(),
            renderer: context.renderer().clone(),
            cache_mode: cache.mode().clone(),
        }
    }

    fn matches(
        &self,
        model: &dyn LoadedModel,
        cache: &dyn FamilyCache,
        context: &RenderedContext,
    ) -> bool {
        self.artifact_id == *model.artifact_id()
            && self.artifact_id == *context.artifact_id()
            && self.model_instance_id == *model.instance_id()
            && cache.owner() == model.instance_id()
            && self.family_impl_version == model.registration().implementation_version()
            && self.engine_api_version == ENGINE_API_VERSION
            && self.profile_id == *context.profile_id()
            && self.tokenizer_id == *context.tokenizer_id()
            && self.template_hash == *context.template_hash()
            && self.renderer == *context.renderer()
            && self.cache_mode == *cache.mode()
            && cache.representation() == &self.cache_mode.representation
    }
}

struct StateCandidate {
    request_id: RequestId,
    compatibility: Compatibility,
    render_hash: [u8; 32],
    history: TokenHistory,
    last_logits: Option<RetainedLogits>,
    reused_input_tokens: usize,
    prefilled_input_tokens: usize,
}

struct StateEntry {
    resident: Resident,
    metadata: RetainedSessionState,
    pin: Weak<()>,
}

impl StateEntry {
    fn is_leased(&self) -> bool {
        self.pin.strong_count() != 0
    }

    fn lease(&mut self) -> SessionStateLease {
        let pin = match self.pin.upgrade() {
            Some(pin) => pin,
            None => {
                let pin = Arc::new(());
                self.pin = Arc::downgrade(&pin);
                pin
            }
        };

        SessionStateLease {
            state: self.metadata.clone(),
            _pin: pin,
        }
    }
}

struct Active {
    decoder: Box<dyn TextDecoder>,
    logits: Option<TokenLogits>,
    eos: Vec<u32>,
    control: ExecutionControl,
    options: GenerationOptions,
    usage: Usage,
    candidate: Option<StateCandidate>,
    text_finished: bool,
    _workspace: Option<MemoryReservation>,
}

struct NativeState {
    active: Option<Active>,
    pending: Option<RequestId>,
    resident: Option<Resident>,
    states: HashMap<SessionStateId, StateEntry>,
    admission: AdmissionController,
    retention: RetentionPolicy,
    summary: Arc<Mutex<Summary>>,
}

impl NativeState {
    fn run(&mut self, receiver: Receiver<Command>) {
        self.publish();

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
                    if result.is_err() {
                        self.invalidate_cache_for(&request);
                    }
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

                Command::RetainState {
                    request_id,
                    token_prefix_bytes,
                    _bytes_reservation,
                    reply,
                } => {
                    let result = self.retain_state(&request_id, &token_prefix_bytes);
                    drop(token_prefix_bytes);
                    drop(_bytes_reservation);
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }

                Command::LookupState { state_id, reply } => {
                    let result = self.lookup_state(&state_id);
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }

                Command::LeaseState { state_id, reply } => {
                    let result = self.lease_state(&state_id);
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }

                Command::ReleaseState { state_id, reply } => {
                    let result = self.release_state(&state_id);
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
            }
        }
    }

    fn request_owner(&self) -> Option<&RequestId> {
        self.active
            .as_ref()
            .map(|active| active.control.request_id())
            .or(self.pending.as_ref())
    }

    fn require_idle(&self) -> Result<(), EngineError> {
        if let Some(owner) = self.request_owner() {
            return Err(EngineError::Busy {
                active_request_id: Some(owner.as_str().to_owned()),
                message: "native worker owns an active or pending request".to_owned(),
            });
        }
        Ok(())
    }

    fn require_unleased(&self) -> Result<(), EngineError> {
        if self.states.values().any(StateEntry::is_leased) {
            return Err(EngineError::Busy {
                active_request_id: None,
                message: "retained native state has a live lease".to_owned(),
            });
        }
        Ok(())
    }

    fn unpublish_idle_state(&mut self) -> Result<(), EngineError> {
        self.require_unleased()?;
        if self.states.is_empty() {
            return Ok(());
        }
        if self.resident.is_some() || self.states.len() != 1 {
            return Err(internal("multiple resident ownership paths"));
        }

        let state_id = *self
            .states
            .keys()
            .next()
            .ok_or_else(|| internal("retained state disappeared"))?;
        let entry = self
            .states
            .remove(&state_id)
            .ok_or_else(|| internal("retained state disappeared"))?;

        // The old canonical identity is unpublished before this allocation
        // can become mutable for a new request.
        self.resident = Some(entry.resident);
        Ok(())
    }

    fn admit_output(
        &mut self,
        setup: &Setup,
        control: &ExecutionControl,
        abandoned: &AtomicBool,
    ) -> Result<Arc<MemoryReservation>, EngineError> {
        self.require_idle()?;
        self.require_unleased()?;
        check_startup(control, abandoned)?;

        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        setup.backend.verify_artifact(artifact, control)?;
        check_startup(control, abandoned)?;

        self.unpublish_idle_state()?;

        let admission = self.admission.clone();
        let output = reserve_output_after_reclaim(&admission, setup.plan.output_bytes, || {
            let same_model = self
                .resident
                .as_ref()
                .is_some_and(|resident| model_matches(resident, artifact, setup.backend.as_ref()));

            if !same_model {
                drop(self.resident.take());
            } else if let Some(resident) = &mut self.resident {
                if resident
                    .cache
                    .as_ref()
                    .is_some_and(|cache| !cache_matches(cache, &setup.plan))
                {
                    drop(resident.candidate.take());
                    drop(resident.cache.take());
                }
                // Keep compatible logical contents until prepare can
                // inspect the exact newly rendered token sequence.
            }

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
        if !self.states.is_empty() {
            return Err(internal("prefill cannot mutate a published entry"));
        }
        if context.profile_id() != &setup.profile || context.token_ids().is_empty() {
            return Err(internal("prefill context does not match admitted setup"));
        }

        control.check_cancelled()?;
        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        if context.artifact_id() != artifact.id() {
            return Err(internal("prefill artifact does not match admitted setup"));
        }

        setup.backend.verify_artifact(artifact, &control)?;
        control.check_cancelled()?;

        let same_model = self
            .resident
            .as_ref()
            .is_some_and(|resident| model_matches(resident, artifact, setup.backend.as_ref()));
        if !same_model {
            drop(self.resident.take());
        } else if let Some(resident) = &mut self.resident {
            if resident
                .cache
                .as_ref()
                .is_some_and(|cache| !cache_matches(cache, &setup.plan))
            {
                drop(resident.candidate.take());
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

            // Keep this conservative until production-path verification.
            capabilities.supports_prefix_reuse = false;
            capabilities.supports_streaming = true;
            capabilities.decoding_modes = vec![DecodingCapability::Greedy];

            weights.try_resize(setup.plan.load_limits.max_resident_bytes())?;
            self.resident = Some(Resident {
                cache: None,
                model,
                weights,
                capabilities,
                candidate: None,
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
                "loaded model does not support the planned cache",
            ));
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

        if let Some(candidate) = &resident.candidate {
            validate_candidate(resident, candidate)?;
        }

        let mut previous = resident.candidate.take();
        let cache = resident
            .cache
            .as_mut()
            .ok_or_else(|| internal("allocated cache unavailable"))?;

        let reused = previous.as_ref().map_or(0, |candidate| {
            if candidate.compatibility.matches(
                resident.model.as_ref(),
                cache.cache.as_ref(),
                context,
            ) && matches!(cache.cache.mode().kind, CacheKind::Full)
                && cache.cache.retained_tokens() == cache.cache.len()
            {
                exact_reusable_prefix(
                    &candidate.history.tokens,
                    context.token_ids(),
                    candidate.last_logits.is_some(),
                )
            } else {
                0
            }
        });

        let mut initial_logits = None;

        if reused == 0 {
            drop(previous.take());
            cache.cache.reset()?;
            if cache.cache.len() != 0 {
                return Err(internal("cache reset did not clear logical state"));
            }
        } else if reused == context.token_count() {
            // Identical input requires real retained final-position logits.
            let retained = previous
                .as_mut()
                .and_then(|candidate| candidate.last_logits.take())
                .ok_or_else(|| internal("identical prefix has no retained logits"))?;
            let RetainedLogits {
                logits,
                reservation,
            } = retained;
            initial_logits = Some(logits);

            // Active execution workspace now accounts for the logits.
            // The independent idle guard is no longer needed.
            drop(reservation);
        } else if let Some(candidate) = &mut previous {
            // Extension prefill will produce new final-position logits.
            drop(candidate.last_logits.take());
        }

        if cache.cache.len() != reused {
            return Err(internal("reused prefix length disagrees with actual KV"));
        }
        check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes())?;

        let decoder = setup.backend.decoder(&setup.registry, &setup.profile)?;
        let eos = setup.backend.eos_tokens(&setup.registry, &setup.profile)?;

        let compatibility =
            Compatibility::new(resident.model.as_ref(), cache.cache.as_ref(), context);

        let model = resident
            .model
            .autoregressive()
            .ok_or_else(|| internal("autoregressive model disappeared"))?;
        let vocabulary = model.vocabulary_size();

        if vocabulary == 0
            || eos
                .iter()
                .any(|&token| usize::try_from(token).map_or(true, |token| token >= vocabulary))
            || initial_logits
                .as_ref()
                .is_some_and(|logits| logits.vocabulary_size() != vocabulary)
        {
            return Err(internal("invalid vocabulary, EOS, or retained logits"));
        }

        let candidate = if retains_contents(self.retention) {
            let history = match previous.take() {
                Some(mut previous)
                    if previous.history.tokens.capacity() >= setup.plan.cache.capacity_tokens() =>
                {
                    previous.history.tokens.truncate(reused);
                    previous.history
                }
                Some(previous) => {
                    drop(previous);
                    let mut history =
                        TokenHistory::new(&self.admission, setup.plan.cache.capacity_tokens())?;
                    history.append(&context.token_ids()[..reused])?;
                    history
                }
                None => {
                    let mut history =
                        TokenHistory::new(&self.admission, setup.plan.cache.capacity_tokens())?;
                    history.append(&context.token_ids()[..reused])?;
                    history
                }
            };

            Some(StateCandidate {
                request_id: control.request_id().clone(),
                compatibility,
                render_hash: *context.render_hash(),
                history,
                last_logits: None,
                reused_input_tokens: reused,
                prefilled_input_tokens: 0,
            })
        } else {
            drop(previous);
            None
        };

        self.pending = None;
        self.active = Some(Active {
            decoder,
            logits: initial_logits,
            eos,
            control,
            options: setup.options,
            // Usage counts logical request input, including validated
            // cached positions. Candidate counters record actual work.
            usage: Usage {
                input_tokens: reused,
                output_tokens: 0,
            },
            candidate,
            text_finished: false,
            _workspace: workspace,
        });

        let active = self
            .active
            .as_mut()
            .ok_or_else(|| internal("active execution unavailable"))?;

        for chunk in context.token_ids()[reused..].chunks(setup.plan.prefill_chunk_tokens.get()) {
            active.control.check_cancelled()?;
            let before = cache.cache.len();

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
            if let Some(candidate) = &mut active.candidate {
                candidate.history.append(chunk)?;
                candidate.prefilled_input_tokens = candidate
                    .prefilled_input_tokens
                    .checked_add(chunk.len())
                    .ok_or_else(|| internal("prefill work counter overflow"))?;
            }
            active.logits = Some(output.into_logits());
        }

        if active.usage.input_tokens != context.token_count()
            || cache.cache.len() != context.token_count()
            || active.logits.is_none()
        {
            return Err(internal(
                "prefill completed with inconsistent request state",
            ));
        }

        active.control.check_cancelled()
    }

    fn decode(&mut self, request: &RequestId) -> Result<TokenStep, EngineError> {
        let active = self
            .active
            .as_mut()
            .filter(|active| active.control.request_id() == request)
            .ok_or_else(|| internal("decode has no matching active request"))?;

        active.control.check_cancelled()?;
        if active.text_finished {
            return Err(internal("decode requested after text completion"));
        }
        if active.usage.output_tokens >= active.options.max_output_tokens() {
            return Err(internal("decode exceeded admitted output allowance"));
        }

        let logits = active
            .logits
            .as_ref()
            .ok_or_else(|| internal("decode has no preceding logits"))?;
        let mut best = 0;
        for index in 1..logits.values().len() {
            if logits.values()[index] > logits.values()[best] {
                best = index;
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
                "family decode progress disagrees with selected token",
            ));
        }

        check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes())?;
        if let Some(candidate) = &mut active.candidate {
            candidate.history.append(&[token])?;
        }
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
            .filter(|active| active.control.request_id() == request)
            .ok_or_else(|| internal("flush has no matching active request"))?;

        active.control.check_cancelled()?;
        if active.text_finished {
            return Err(internal("text completion requested twice"));
        }

        let text = active.decoder.finish()?;
        active.control.check_cancelled()?;
        active.text_finished = true;
        Ok(text)
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
            .is_some_and(|active| active.control.request_id() != request)
        {
            return Err(internal("cleanup request does not own active execution"));
        }
        if self.pending.is_none() && self.active.is_none() {
            return Ok(());
        }

        self.pending = None;
        let mut active = self.active.take();
        let mut candidate = None;

        if let Some(active) = &mut active {
            if active.text_finished
                && !active.control.is_cancelled()
                && retains_contents(self.retention)
            {
                if let (Some(mut completed), Some(logits)) =
                    (active.candidate.take(), active.logits.take())
                {
                    // Retention is optional. If its independent idle
                    // reservation cannot fit, discard it rather than
                    // misaccounting allocations or failing completed work.
                    if let Ok(retained) = RetainedLogits::new(logits, &self.admission) {
                        completed.last_logits = Some(retained);
                        candidate = Some(completed);
                    }
                }
            }
        }

        // Drop decoder, transient logits, and workspace before replying.
        drop(active);

        if !self.retention.retain_model {
            drop(candidate);
            drop(self.resident.take());
            return Ok(());
        }

        let Some(resident) = &mut self.resident else {
            drop(candidate);
            return Ok(());
        };

        // Abandoned startup did not mutate the old completed candidate.
        if candidate.is_none() && resident.candidate.is_some() {
            return Ok(());
        }

        drop(resident.candidate.take());

        if !self.retention.retain_cache_buffers {
            drop(candidate);
            drop(resident.cache.take());
            return Ok(());
        }

        if let Some(completed) = candidate {
            if validate_candidate(resident, &completed).is_ok() {
                resident.candidate = Some(completed);
                return Ok(());
            }
            drop(completed);
        }

        if let Some(cache) = &mut resident.cache {
            if cache.cache.reset().is_err()
                || cache.cache.len() != 0
                || check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes()).is_err()
            {
                drop(resident.cache.take());
            }
        }

        Ok(())
    }

    fn invalidate_cache_for(&mut self, request: &RequestId) {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.control.request_id() == request)
            || self.pending.as_ref() == Some(request)
        {
            if let Some(active) = &mut self.active {
                active.text_finished = false;
                drop(active.candidate.take());
            }
            if let Some(resident) = &mut self.resident {
                drop(resident.candidate.take());
                drop(resident.cache.take());
            }
        }
    }

    fn retain_state(
        &mut self,
        request_id: &RequestId,
        token_prefix_bytes: &[u8],
    ) -> Result<SessionStateId, EngineError> {
        self.require_idle()?;
        if !retains_contents(self.retention) {
            return Err(EngineError::unsupported(
                "session state retention",
                "requires retained model and cache buffers",
            ));
        }

        if let Some(entry) = self
            .states
            .values()
            .find(|entry| &entry.metadata.request_id == request_id)
        {
            validate_entry(entry)?;
            let state_id = canonical_identity(&entry.resident, token_prefix_bytes)?;
            if state_id != entry.metadata.state_id {
                return Err(EngineError::invalid_request(
                    "token_prefix_bytes",
                    "request is already published under a different identity",
                ));
            }
            return Ok(state_id);
        }

        let resident = self
            .resident
            .as_ref()
            .ok_or_else(|| EngineError::NotFound {
                resource: "completed native request state",
                id: request_id.as_str().to_owned(),
            })?;
        let candidate = resident
            .candidate
            .as_ref()
            .filter(|candidate| &candidate.request_id == request_id)
            .ok_or_else(|| EngineError::NotFound {
                resource: "completed native request state",
                id: request_id.as_str().to_owned(),
            })?;

        validate_candidate(resident, candidate)?;
        let state_id = canonical_identity(resident, token_prefix_bytes)?;

        if !self.states.is_empty() {
            return Err(internal("multiple resident ownership paths"));
        }

        self.states
            .try_reserve(1)
            .map_err(|_| EngineError::AllocationFailed {
                operation: "allocate canonical state registry entry",
                requested_bytes: None,
            })?;

        let cache = resident
            .cache
            .as_ref()
            .ok_or_else(|| internal("completed cache unavailable"))?;
        let logits = candidate
            .last_logits
            .as_ref()
            .ok_or_else(|| internal("completed logits unavailable"))?;
        let compatibility = &candidate.compatibility;

        let metadata = RetainedSessionState {
            state_id,
            request_id: candidate.request_id.clone(),
            artifact_id: compatibility.artifact_id.clone(),
            model_instance_id: compatibility.model_instance_id.clone(),
            family_impl_version: compatibility.family_impl_version,
            profile_id: compatibility.profile_id.clone(),
            tokenizer_id: compatibility.tokenizer_id.clone(),
            template_hash: compatibility.template_hash,
            render_hash: candidate.render_hash,
            renderer: compatibility.renderer.clone(),
            cache_mode: cache.cache.mode().clone(),
            logical_tokens: cache.cache.len(),
            retained_tokens: cache.cache.retained_tokens(),
            capacity_tokens: cache.cache.capacity_tokens(),
            weights_reserved_bytes: resident.weights.bytes(),
            cache_reserved_bytes: cache.reservation.bytes(),
            token_history_reserved_bytes: candidate.history.reservation.bytes(),
            logits_reserved_bytes: logits.reservation.bytes(),
            reused_input_tokens: candidate.reused_input_tokens,
            prefilled_input_tokens: candidate.prefilled_input_tokens,
        };

        let resident = self
            .resident
            .take()
            .ok_or_else(|| internal("completed resident disappeared"))?;
        self.states.insert(
            state_id,
            StateEntry {
                resident,
                metadata,
                pin: Weak::new(),
            },
        );

        Ok(state_id)
    }

    fn lookup_state(
        &self,
        state_id: &SessionStateId,
    ) -> Result<Option<RetainedSessionState>, EngineError> {
        match self.states.get(state_id) {
            Some(entry) => {
                validate_entry(entry)?;
                Ok(Some(entry.metadata.clone()))
            }
            None => Ok(None),
        }
    }

    fn lease_state(
        &mut self,
        state_id: &SessionStateId,
    ) -> Result<Option<SessionStateLease>, EngineError> {
        match self.states.get_mut(state_id) {
            Some(entry) => {
                validate_entry(entry)?;
                Ok(Some(entry.lease()))
            }
            None => Ok(None),
        }
    }

    fn release_state(&mut self, state_id: &SessionStateId) -> Result<ReleaseOutcome, EngineError> {
        let Some(entry) = self.states.get(state_id) else {
            return Ok(ReleaseOutcome::AlreadyAbsent);
        };

        if entry.is_leased() {
            return Err(EngineError::Busy {
                active_request_id: None,
                message: "cannot release retained state while a lease is alive".to_owned(),
            });
        }

        self.require_idle()?;
        let entry = self
            .states
            .remove(state_id)
            .ok_or_else(|| internal("retained state disappeared during release"))?;

        drop(entry);
        Ok(ReleaseOutcome::Released)
    }

    fn usage(&self, request: &RequestId) -> Usage {
        self.active
            .as_ref()
            .filter(|active| active.control.request_id() == request)
            .map_or(Usage::default(), |active| active.usage)
    }

    fn publish(&self) {
        let mut summary = Summary::empty(true);
        summary.active = self.active.is_some() || self.pending.is_some();

        let entry = self.states.values().next();
        if let Some(entry) = entry {
            summary.pin = entry.pin.clone();
        }

        let resident = self
            .resident
            .as_ref()
            .or_else(|| entry.map(|entry| &entry.resident));

        if let Some(resident) = resident {
            summary.artifact = Some(resident.model.artifact_id().clone());
            summary.weights_bytes = resident.weights.bytes();
            summary.capabilities = Some(resident.capabilities.clone());

            if let Some(cache) = &resident.cache {
                summary.cache_bytes = cache.reservation.bytes();
                summary.cache_mode = Some(cache.cache.mode().clone());
                summary.cache_capacity = cache.cache.capacity_tokens();
            }
            if let Some(candidate) = &resident.candidate {
                summary.history_bytes = candidate.history.reservation.bytes();
                summary.logits_bytes = candidate
                    .last_logits
                    .as_ref()
                    .map_or(0, |logits| logits.reservation.bytes());
            }
        }

        if let Some(candidate) = self
            .active
            .as_ref()
            .and_then(|active| active.candidate.as_ref())
        {
            summary.history_bytes = candidate.history.reservation.bytes();
        }

        *lock_metadata(&self.summary) = summary;
    }
}

fn retains_contents(retention: RetentionPolicy) -> bool {
    retention.retain_model && retention.retain_cache_buffers
}

fn model_matches(
    resident: &Resident,
    artifact: &ArtifactRegistration,
    backend: &dyn RuntimeBackend,
) -> bool {
    resident.model.artifact_id() == artifact.id()
        && resident.model.registration().implementation_version()
            == backend.registration().implementation_version()
        && resident.model.registration().family() == artifact.family()
}

/// Conservative complete-prefix matching.
///
/// Identical inputs require actual retained logits. Extensions will
/// produce new logits by executing their nonempty suffix. Shortened or
/// divergent input always rebuilds.
fn exact_reusable_prefix(retained: &[u32], input: &[u32], has_logits: bool) -> usize {
    if retained.is_empty() || !input.starts_with(retained) {
        return 0;
    }
    if input.len() == retained.len() && !has_logits {
        return 0;
    }
    retained.len()
}

fn canonical_identity(
    resident: &Resident,
    token_prefix_bytes: &[u8],
) -> Result<SessionStateId, EngineError> {
    let cache = resident
        .cache
        .as_ref()
        .ok_or_else(|| internal("canonical state has no native cache"))?;
    let inputs = CacheIdentityInputs::for_retained_cache(
        resident.model.as_ref(),
        cache.cache.as_ref(),
        None,
        token_prefix_bytes,
    )?;
    compute_session_state_id(&inputs)
}

fn validate_candidate(resident: &Resident, candidate: &StateCandidate) -> Result<(), EngineError> {
    let cache = resident
        .cache
        .as_ref()
        .ok_or_else(|| internal("retained candidate has no native cache"))?;
    let compatibility = &candidate.compatibility;
    let logits = candidate
        .last_logits
        .as_ref()
        .ok_or_else(|| internal("retained candidate has no final-position logits"))?;

    if compatibility.artifact_id != *resident.model.artifact_id()
        || compatibility.model_instance_id != *resident.model.instance_id()
        || compatibility.family_impl_version
            != resident.model.registration().implementation_version()
        || compatibility.engine_api_version != ENGINE_API_VERSION
        || compatibility.cache_mode != *cache.cache.mode()
        || cache.cache.owner() != resident.model.instance_id()
        || cache.cache.representation() != &cache.cache.mode().representation
        || cache.cache.len() != candidate.history.tokens.len()
        || cache.cache.len() > cache.cache.capacity_tokens()
        || cache.cache.retained_tokens() > cache.cache.len()
        || candidate.history.tokens.is_empty()
    {
        return Err(internal(
            "retained candidate ownership or positions disagree",
        ));
    }

    let input_tokens = candidate
        .reused_input_tokens
        .checked_add(candidate.prefilled_input_tokens)
        .ok_or_else(|| internal("retained input accounting overflow"))?;
    if input_tokens > candidate.history.tokens.len() {
        return Err(internal(
            "retained input accounting exceeds executed history",
        ));
    }

    if token_storage_bytes(candidate.history.tokens.capacity())?
        > candidate.history.reservation.bytes()
        || float_storage_bytes(logits.logits.values().len())? > logits.reservation.bytes()
    {
        return Err(internal("retained bookkeeping exceeds its reservation"));
    }

    check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes())?;
    let weights = resident.model.memory_usage()?.allocated_bytes();
    if weights > resident.weights.bytes() {
        return Err(EngineError::MemoryLimitExceeded {
            operation: "retained model buffers",
            required_bytes: weights,
            available_bytes: resident.weights.bytes(),
        });
    }

    Ok(())
}

fn validate_entry(entry: &StateEntry) -> Result<(), EngineError> {
    let candidate = entry
        .resident
        .candidate
        .as_ref()
        .ok_or_else(|| internal("canonical entry has no executed history"))?;
    validate_candidate(&entry.resident, candidate)?;

    let cache = entry
        .resident
        .cache
        .as_ref()
        .ok_or_else(|| internal("canonical entry has no cache"))?;
    let logits = candidate
        .last_logits
        .as_ref()
        .ok_or_else(|| internal("canonical entry has no logits"))?;
    let compatibility = &candidate.compatibility;
    let metadata = &entry.metadata;

    if metadata.request_id != candidate.request_id
        || metadata.artifact_id != compatibility.artifact_id
        || metadata.model_instance_id != compatibility.model_instance_id
        || metadata.family_impl_version != compatibility.family_impl_version
        || metadata.profile_id != compatibility.profile_id
        || metadata.tokenizer_id != compatibility.tokenizer_id
        || metadata.template_hash != compatibility.template_hash
        || metadata.render_hash != candidate.render_hash
        || metadata.renderer != compatibility.renderer
        || metadata.cache_mode != *cache.cache.mode()
        || metadata.logical_tokens != cache.cache.len()
        || metadata.retained_tokens != cache.cache.retained_tokens()
        || metadata.capacity_tokens != cache.cache.capacity_tokens()
        || metadata.weights_reserved_bytes != entry.resident.weights.bytes()
        || metadata.cache_reserved_bytes != cache.reservation.bytes()
        || metadata.token_history_reserved_bytes != candidate.history.reservation.bytes()
        || metadata.logits_reserved_bytes != logits.reservation.bytes()
        || metadata.reused_input_tokens != candidate.reused_input_tokens
        || metadata.prefilled_input_tokens != candidate.prefilled_input_tokens
    {
        return Err(internal(
            "canonical metadata disagrees with native ownership",
        ));
    }

    Ok(())
}

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

fn token_storage_bytes(capacity: usize) -> Result<u64, EngineError> {
    storage_bytes(capacity, std::mem::size_of::<u32>())
}

fn float_storage_bytes(capacity: usize) -> Result<u64, EngineError> {
    storage_bytes(capacity, std::mem::size_of::<f32>())
}

fn storage_bytes(capacity: usize, element_bytes: usize) -> Result<u64, EngineError> {
    let bytes = capacity
        .checked_mul(element_bytes)
        .ok_or_else(|| internal("runtime allocation size overflow"))?;
    bytes_as_u64(bytes)
}

fn bytes_as_u64(bytes: usize) -> Result<u64, EngineError> {
    u64::try_from(bytes).map_err(|_| internal("runtime allocation size exceeds u64"))
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
        .and_then(|value| value.checked_add(profile.safety_margin_tokens()))
        .ok_or_else(|| internal("planned context arithmetic overflow"))?;

    if plan.cache.capacity_tokens() < capacity
        || plan.cache.capacity_tokens() > profile.context_limit_tokens()
    {
        return Err(EngineError::invalid_request(
            "cache capacity",
            "must cover profile output reserve and safety margin within the context limit",
        ));
    }

    let headers = limits
        .event_capacity()
        .checked_mul(std::mem::size_of::<String>())
        .ok_or_else(|| internal("output header size overflow"))?;
    let minimum = limits
        .max_buffered_text_bytes()
        .checked_add(limits.max_text_delta_bytes())
        .and_then(|value| value.checked_add(headers))
        .ok_or_else(|| internal("output budget size overflow"))?;

    if plan.output_bytes < bytes_as_u64(minimum)? {
        return Err(EngineError::invalid_request(
            "output_bytes",
            "does not cover queued text, a pending delta, and queue headers",
        ));
    }

    add(plan.temporary_bytes, plan.headroom_bytes)?;
    token_storage_bytes(plan.cache.capacity_tokens())?;
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

// Metadata and notifications only. Never recovers the admission ledger.
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

    fn empty_native_state(retention: RetentionPolicy) -> NativeState {
        NativeState {
            active: None,
            pending: None,
            resident: None,
            states: HashMap::new(),
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

        let output = reserve_output_after_reclaim(&admission, 20 * MIB, || {
            drop(idle);
            Ok(())
        })
        .expect("output after reclamation");

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

        let output = reserve_output_after_reclaim(&admission, 20, || {
            drop(cache);
            Ok(())
        })
        .expect("output");

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

        let result = reserve_output_after_reclaim(&admission, 20, || {
            drop(idle);
            Ok(())
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
            reserve_output_after_reclaim(&admission, 20, || {
                Err(internal("reclamation failed"))
            })
            .is_err()
        );
        assert_eq!(admission.snapshot().expect("after"), before);
    }

    #[test]
    fn pending_startup_can_finish_without_prefill() {
        let request = RequestId::new("pending-startup").expect("request");
        let mut state = empty_native_state(RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: true,
        });
        state.pending = Some(request.clone());
        state.publish();

        assert!(lock_metadata(&state.summary).active);
        state.finish(&request).expect("finish");
        state.publish();

        assert!(state.pending.is_none());
        assert!(!lock_metadata(&state.summary).active);
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
        let request = RequestId::new("pending-publication").expect("request");
        state.pending = Some(request.clone());

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
}
