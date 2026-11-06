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

use crate::residency_pin::{ModelPinOwner, ModelResidencyPin};
use crate::{
    admission::{AdmissionController, AdmissionSnapshot, MemoryReservation, ReservationGrant},
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
    time::{Duration, Instant},
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

/// Reusable native allocations for incremental MLX peak forecasting.
///
/// These are native-buffer observations, not reservation totals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeResidency {
    /// The selected artifact/implementation is already loaded and reusable.
    pub reusable_model: bool,
    /// Measured buffers of a compatible, reusable native cache.
    pub reusable_cache_buffer_bytes: u64,
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

    /// Conservative incremental MLX allocation forecast for this plan.
    ///
    /// Runs on the native worker. Exclude reused active buffers and host
    /// checkpoint/decode storage. None means unknown, not zero; allocator
    /// pressure checking must reject an unknown forecast. Non-MLX adapters
    /// can use an explicitly admission-only pressure policy.
    fn incremental_mlx_bytes(
        &self,
        _registry: &Registry,
        _profile: &ModelProfileId,
        _plan: &RuntimePlan,
        _residency: RuntimeResidency,
    ) -> Result<Option<u64>, EngineError> {
        Ok(None)
    }

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
    pressure_policy: Option<crate::pressure::PressurePolicy>,
    memory_enforcer: Option<crate::memory_enforcer::MemoryEnforcer>,
    prefill_min_chunk_tokens: NonZeroUsize,
    worker: Arc<NativeWorker>,
}

/// Metadata for one actually owned native model instance.
#[derive(Clone, Debug)]
pub struct ModelResidencyInfo {
    /// Identity of this particular load.
    pub instance: ModelInstanceId,
    /// Loaded artifact identity.
    pub artifact: ArtifactId,
    /// Weight bytes still committed in the admission ledger.
    pub committed_weight_bytes: u64,
    /// Whether explicit residency handles protect the model.
    pub pinned: bool,
}

impl FamilyRuntime {
    /// Enable worker-owned idle model pooling and optional per-model TTL.
    ///
    /// Existing instances receive this default. Ordinary session leases
    /// retain their current conservative worker-wide mutation protection.
    /// Use before serving requests. None disables expiry, not pooling.
    pub fn with_model_pool(self, ttl: Option<Duration>) -> Result<Self, EngineError> {
        self.worker
            .call(|reply| Command::ConfigureModelPool { ttl, reply })?
            .result?;
        Ok(self)
    }

    /// Change idle expiry for one exact loaded instance; false means absent.
    pub fn set_model_idle_ttl(
        &self,
        instance: ModelInstanceId,
        ttl: Option<Duration>,
    ) -> EngineFuture<'_, bool> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-model-ttl", move || {
                worker
                    .call(|reply| Command::SetModelTtl {
                        instance,
                        ttl,
                        reply,
                    })?
                    .result
            })?
            .await
        })
    }

    /// List actual loaded instances, including idle pool residents.
    pub fn model_residency(&self) -> EngineFuture<'_, Vec<ModelResidencyInfo>> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-model-residency", move || {
                worker
                    .call(|reply| Command::ModelResidency { reply })?
                    .result
            })?
            .await
        })
    }

    /// Acquire an explicit pin for an exact loaded instance, or return None.
    ///
    /// Acquisition and destruction are serialized on the native worker.
    /// Shutdown still destroys the worker's models, regardless of handles.
    pub fn pin_model(
        &self,
        instance: ModelInstanceId,
    ) -> EngineFuture<'_, Option<ModelResidencyPin>> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-model-pin", move || {
                worker
                    .call(|reply| Command::PinModel { instance, reply })?
                    .result
            })?
            .await
        })
    }

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
                    pool: Vec::new(),
                    pool_enabled: false,
                    pool_ttl: None,
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
            pressure_policy: None,
            memory_enforcer: None,
            prefill_min_chunk_tokens: NonZeroUsize::new(
                crate::prefill_admit::DEFAULT_PREFILL_MIN_CHUNK_TOKENS,
            )
            .expect("default prefill min chunk"),
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

    /// Clone of the shared admission controller (pause / resize from the enforcer).
    pub fn admission_controller(&self) -> AdmissionController {
        self.admission.clone()
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

    // ALLOY_RUNTIME_MLX_MEMORY_V1

    /// Observe process-wide MLX allocator counters on the native worker.
    ///
    /// This does not synchronize execution, acquire admission, or change
    /// reservation ownership. Counter fields are sequential observations,
    /// not an atomic snapshot or process RSS.
    pub fn mlx_memory_snapshot(&self) -> EngineFuture<'_, mlx_ffi::memory::MemorySnapshot> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-mlx-memory-snapshot", move || {
                worker
                    .call(|reply| Command::MlxMemorySnapshot { reply })?
                    .result
            })?
            .await
        })
    }

    /// Clear unused process-wide MLX allocator cache while this runtime
    /// has no active execution or pending startup.
    ///
    /// Does not destroy live tensors, invalidate published state, release
    /// leases, alter retention policy, or credit the admission ledger.
    ///
    /// This runtime's backends must satisfy their native-completion
    /// contract before becoming idle. This operation does not synchronize
    /// arbitrary streams belonging to other runtimes or libraries.
    ///
    /// The application must coordinate process-wide allocator policy.
    /// Reported before/after differences are observations, not proof of
    /// an equal reduction in process RSS.
    pub fn reclaim_mlx_allocator_cache(
        &self,
    ) -> EngineFuture<'_, mlx_ffi::memory::CacheReclaimReport> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-mlx-cache-reclaim", move || {
                worker
                    .call(|reply| Command::ReclaimMlxAllocatorCache { reply })?
                    .result
            })?
            .await
        })
    }

    /// Enable bounded startup-pressure handling for requests from this handle.
    ///
    /// Configure centrally before cloning/publishing the handle. This does
    /// not change allocator limits or mutate an existing request. Execution
    /// remains on the shared worker; policy travels with each startup setup.
    #[must_use]
    pub fn with_pressure_policy(mut self, policy: crate::pressure::PressurePolicy) -> Self {
        self.pressure_policy = Some(policy);
        self
    }

    /// Attach the live memory enforcer so prefill can shrink against the admit line.
    #[must_use]
    pub fn with_memory_enforcer(
        mut self,
        enforcer: crate::memory_enforcer::MemoryEnforcer,
    ) -> Self {
        self.memory_enforcer = Some(enforcer);
        self
    }

    /// Override the adaptive prefill floor (default
    /// [`crate::prefill_admit::DEFAULT_PREFILL_MIN_CHUNK_TOKENS`]).
    #[must_use]
    pub fn with_prefill_min_chunk_tokens(mut self, tokens: NonZeroUsize) -> Self {
        self.prefill_min_chunk_tokens = tokens;
        self
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
    fn supports_memory_reclamation(&self) -> bool {
        self.pressure_policy.is_some()
    }

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

            if !summary.active
                && summary.pin.strong_count() == 0
                && (!summary.model_pinned || same_model)
            {
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
            let mut plan = backend
                .plan(&registry, request.profile_id(), rendered.token_count())
                .await?;
            if let Some(enforcer) = &self.memory_enforcer {
                let decision = enforcer.select_prefill_chunk(
                    plan.prefill_chunk_tokens,
                    self.prefill_min_chunk_tokens,
                    plan.temporary_bytes,
                )?;
                plan.prefill_chunk_tokens = decision.chunk_tokens;
                plan.temporary_bytes = decision.workspace_bytes;
            }
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
                pressure_policy: self.pressure_policy,
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
    pressure_policy: Option<crate::pressure::PressurePolicy>,
}

struct Reply<T> {
    result: Result<T, EngineError>,
    usage: Usage,
}

enum Command {
    ConfigureModelPool {
        ttl: Option<Duration>,
        reply: SyncSender<Reply<()>>,
    },
    SetModelTtl {
        instance: ModelInstanceId,
        ttl: Option<Duration>,
        reply: SyncSender<Reply<bool>>,
    },
    ModelResidency {
        reply: SyncSender<Reply<Vec<ModelResidencyInfo>>>,
    },
    PinModel {
        instance: ModelInstanceId,
        reply: SyncSender<Reply<Option<ModelResidencyPin>>>,
    },
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
    MlxMemorySnapshot {
        reply: SyncSender<Reply<mlx_ffi::memory::MemorySnapshot>>,
    },
    ReclaimMlxAllocatorCache {
        reply: SyncSender<Reply<mlx_ffi::memory::CacheReclaimReport>>,
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
    model_pinned: bool,
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
            model_pinned: false,
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
    pins: ModelPinOwner,
    idle_since: Instant,
    idle_ttl: Option<Duration>,
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
    #[cfg(test)]
    fn new(admission: &AdmissionController, capacity: usize) -> Result<Self, EngineError> {
        let requested = token_storage_bytes(capacity)?;
        let reservation = admission.reserve("executed token history", requested)?;
        Self::from_reservation(reservation, capacity)
    }

    fn from_reservation(
        mut reservation: MemoryReservation,
        capacity: usize,
    ) -> Result<Self, EngineError> {
        let requested = token_storage_bytes(capacity)?;
        if reservation.bytes() < requested {
            return Err(internal(
                "history reservation is smaller than the requested capacity",
            ));
        }

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

/// Admitted startup that still owns unpartitioned grant bytes.
///
/// Output has already been taken. Remaining components are transferred in
/// `prepare`, or released when pending startup is abandoned.
struct PendingStartup {
    request_id: RequestId,
    grant: ReservationGrant,
}

struct NativeState {
    active: Option<Active>,
    pending: Option<PendingStartup>,
    resident: Option<Resident>,
    states: HashMap<SessionStateId, StateEntry>,
    pool: Vec<Resident>,
    pool_enabled: bool,
    pool_ttl: Option<Duration>,
    admission: AdmissionController,
    retention: RetentionPolicy,
    summary: Arc<Mutex<Summary>>,
}

impl NativeState {
    fn residency_info(&self) -> Vec<ModelResidencyInfo> {
        self.pool
            .iter()
            .chain(self.resident.iter())
            .chain(self.states.values().map(|entry| &entry.resident))
            .map(|resident| ModelResidencyInfo {
                instance: resident.model.instance_id().clone(),
                artifact: resident.model.artifact_id().clone(),
                committed_weight_bytes: resident.weights.bytes(),
                pinned: resident.pins.is_pinned(),
            })
            .collect()
    }

    fn pin_instance(
        &mut self,
        instance: &ModelInstanceId,
    ) -> Result<Option<ModelResidencyPin>, EngineError> {
        let resident = self
            .pool
            .iter_mut()
            .chain(self.resident.iter_mut())
            .chain(self.states.values_mut().map(|entry| &mut entry.resident))
            .find(|resident| resident.model.instance_id().same_instance(instance));
        match resident {
            None => Ok(None),
            Some(resident) => resident
                .pins
                .try_pin(instance, false)
                .map(Some)
                .map_err(|_| internal("model pin identity changed on native worker")),
        }
    }

    fn selected_model_pinned(&self) -> bool {
        self.resident
            .as_ref()
            .is_some_and(|resident| resident.pins.is_pinned())
            || self
                .states
                .values()
                .any(|entry| entry.resident.pins.is_pinned())
    }

    fn expire_models_at(&mut self, now: Instant) {
        if !self.pool_enabled {
            return;
        }
        self.pool
            .retain(|resident| resident.pins.is_pinned() || !model_ttl_expired(resident, now));
        if self.active.is_none()
            && self.pending.is_none()
            && !self.states.values().any(StateEntry::is_leased)
            && !self.selected_model_pinned()
        {
            let expired = self
                .resident
                .as_ref()
                .or_else(|| self.states.values().next().map(|entry| &entry.resident))
                .is_some_and(|resident| model_ttl_expired(resident, now));
            if expired {
                if self.unpublish_idle_state().is_ok() {
                    drop(self.resident.take());
                }
            }
        }
        self.publish();
    }

    fn evict_one_pooled_model(&mut self) -> bool {
        // Vec order supplies a stable insertion-order tie break.
        let victim = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, resident)| !resident.pins.is_pinned())
            .min_by_key(|(index, resident)| (resident.idle_since, *index))
            .map(|(index, _)| index);
        if let Some(index) = victim {
            // The Resident owns native resources and their real guards.
            // No estimated bytes are subtracted from the ledger.
            drop(self.pool.remove(index));
            true
        } else {
            false
        }
    }

    fn switch_model(
        &mut self,
        artifact: &ArtifactRegistration,
        backend: &dyn RuntimeBackend,
    ) -> Result<(), EngineError> {
        if self
            .resident
            .as_ref()
            .is_some_and(|resident| model_matches(resident, artifact, backend))
        {
            return Ok(());
        }
        if !self.pool_enabled {
            if self.selected_model_pinned() {
                return Err(EngineError::Busy {
                    active_request_id: None,
                    message: "model replacement blocked by explicit residency pin".to_owned(),
                });
            }
            drop(self.resident.take());
            return Ok(());
        }
        if self.resident.is_some() {
            self.pool
                .try_reserve(1)
                .map_err(|_| EngineError::AllocationFailed {
                    operation: "reserve idle model pool entry",
                    requested_bytes: None,
                })?;
        }
        if let Some(resident) = self.resident.take() {
            self.pool.push(resident);
        }
        if let Some(index) = self
            .pool
            .iter()
            .position(|resident| model_matches(resident, artifact, backend))
        {
            self.resident = Some(self.pool.remove(index));
        }
        Ok(())
    }

    fn run(&mut self, receiver: Receiver<Command>) {
        self.publish();

        loop {
            self.expire_models_at(Instant::now());
            let command = match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(command) => command,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
            match command {
                Command::ConfigureModelPool { ttl, reply } => {
                    let result = self.require_idle().map(|()| {
                        self.pool_enabled = true;
                        self.pool_ttl = ttl;
                        for resident in self
                            .pool
                            .iter_mut()
                            .chain(self.resident.iter_mut())
                            .chain(self.states.values_mut().map(|entry| &mut entry.resident))
                        {
                            resident.idle_ttl = ttl;
                        }
                    });
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::SetModelTtl {
                    instance,
                    ttl,
                    reply,
                } => {
                    let resident = self
                        .pool
                        .iter_mut()
                        .chain(self.resident.iter_mut())
                        .chain(self.states.values_mut().map(|entry| &mut entry.resident))
                        .find(|resident| resident.model.instance_id().same_instance(&instance));
                    let result = Ok(match resident {
                        Some(resident) => {
                            resident.idle_ttl = ttl;
                            true
                        }
                        None => false,
                    });
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::ModelResidency { reply } => {
                    let result = Ok(self.residency_info());
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::PinModel { instance, reply } => {
                    let result = self.pin_instance(&instance);
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
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

                Command::MlxMemorySnapshot { reply } => {
                    let result = self.mlx_memory_snapshot();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::ReclaimMlxAllocatorCache { reply } => {
                    let result = self.reclaim_mlx_allocator_cache();
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
            .or_else(|| self.pending.as_ref().map(|pending| &pending.request_id))
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

    fn mlx_memory_snapshot(&self) -> Result<mlx_ffi::memory::MemorySnapshot, EngineError> {
        mlx_ffi::memory::snapshot().map_err(|error| EngineError::ExecutionFailed {
            stage: "observe MLX allocator memory",
            message: error.to_string(),
        })
    }

    fn reclaim_mlx_allocator_cache(
        &self,
    ) -> Result<mlx_ffi::memory::CacheReclaimReport, EngineError> {
        // Check ownership before invoking any process-wide allocator call.
        self.require_idle()?;

        let before = self.mlx_memory_snapshot()?;
        mlx_ffi::memory::clear_cache().map_err(|error| EngineError::ExecutionFailed {
            stage: "clear unused MLX allocator cache",
            message: error.to_string(),
        })?;
        let after = self.mlx_memory_snapshot()?;

        // In particular, do not release or resize any MemoryReservation
        // based on observed allocator counter differences.
        Ok(mlx_ffi::memory::CacheReclaimReport { before, after })
    }

    fn reclaim_mlx_cache_before_idle_replacement(
        &self,
        setup: &Setup,
        artifact: &ArtifactRegistration,
    ) -> Result<(), EngineError> {
        let resident = self
            .resident
            .as_ref()
            .or_else(|| self.states.values().next().map(|entry| &entry.resident));

        let Some(resident) = resident else {
            return Ok(());
        };

        let incoming_is_mlx = matches!(
            &setup.backend.registration().capabilities().backend,
            crate::types::ExecutionBackend::MlxCpu | crate::types::ExecutionBackend::MlxMetal
        );
        let resident_is_mlx = matches!(
            &resident.capabilities.backend,
            crate::types::ExecutionBackend::MlxCpu | crate::types::ExecutionBackend::MlxMetal
        );

        if !incoming_is_mlx && !resident_is_mlx {
            return Ok(());
        }

        let incompatible_model = resident.model.artifact_id() != artifact.id()
            || resident.model.registration().implementation_version()
                != setup.backend.registration().implementation_version()
            || resident.model.registration().family() != artifact.family();

        let incompatible_cache = resident
            .cache
            .as_ref()
            .is_some_and(|cache| !cache_matches(cache, &setup.plan));

        if incompatible_model || incompatible_cache {
            // First clear unused allocator buffers. Actual incompatible
            // live resources are still destroyed by the existing startup
            // reclamation path, with their guards kept until destruction.
            let _report = self.reclaim_mlx_allocator_cache()?;
        }

        Ok(())
    }

    fn pressure_residency(&self, setup: &Setup) -> Result<RuntimeResidency, EngineError> {
        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        let resident = self
            .resident
            .as_ref()
            .or_else(|| self.states.values().next().map(|entry| &entry.resident));
        let Some(resident) = resident else {
            return Ok(RuntimeResidency {
                reusable_model: false,
                reusable_cache_buffer_bytes: 0,
            });
        };
        let same_model = resident.model.artifact_id() == artifact.id()
            && resident.model.registration().implementation_version()
                == setup.backend.registration().implementation_version()
            && resident.model.registration().family() == artifact.family();
        let cache_bytes = if same_model {
            match resident
                .cache
                .as_ref()
                .filter(|cache| cache_matches(cache, &setup.plan))
            {
                Some(cache) => cache.cache.memory_usage()?.allocated_bytes(),
                None => 0,
            }
        } else {
            0
        };
        Ok(RuntimeResidency {
            reusable_model: same_model,
            reusable_cache_buffer_bytes: cache_bytes,
        })
    }

    fn pressure_demand(
        &self,
        setup: &Setup,
    ) -> Result<crate::pressure::RequestDemand, EngineError> {
        let residency = self.pressure_residency(setup)?;
        let resident = self
            .resident
            .as_ref()
            .or_else(|| self.states.values().next().map(|entry| &entry.resident));
        let same_cache = residency.reusable_model
            && resident
                .and_then(|resident| resident.cache.as_ref())
                .is_some_and(|cache| cache_matches(cache, &setup.plan));
        let weights = if residency.reusable_model {
            0
        } else {
            setup.plan.load_limits.max_load_peak_bytes()
        };
        let cache = if same_cache {
            0
        } else {
            setup.plan.cache.max_cache_bytes()
        };
        let history = if retains_contents(self.retention) {
            token_storage_bytes(setup.plan.cache.capacity_tokens())?
        } else {
            0
        };
        // Same conservative full-startup ledger forecast as the existing
        // estimate, without crediting reservations before their guards die.
        let ledger = add(
            add(add(weights, cache)?, history)?,
            add(
                add(setup.plan.temporary_bytes, setup.plan.headroom_bytes)?,
                setup.plan.output_bytes,
            )?,
        )?;
        let policy = setup
            .pressure_policy
            .ok_or_else(|| internal("pressure policy missing"))?;
        let native = if policy.options().allocator_ceiling_bytes.is_some() {
            setup.backend.incremental_mlx_bytes(
                setup.registry.as_ref(), &setup.profile, &setup.plan, residency,
            )?.ok_or_else(|| EngineError::unsupported(
                "MLX pressure forecast",
                "adapter has no incremental native allocation forecast; use admission-only policy or implement the hook",
            ))?
        } else {
            0
        };
        Ok(crate::pressure::RequestDemand {
            additional_reservation_bytes: ledger,
            additional_mlx_bytes: native,
        })
    }

    fn pressure_observation(
        &self,
        setup: &Setup,
    ) -> Result<crate::pressure::PressureObservation, EngineError> {
        use crate::pressure::{AllocatorObservation, ExecutionState, ReclaimableResources};
        let policy = setup
            .pressure_policy
            .ok_or_else(|| internal("pressure policy missing"))?;
        let allocator = if policy.options().allocator_ceiling_bytes.is_some() {
            Some(
                AllocatorObservation::from_mlx(self.mlx_memory_snapshot()?).map_err(|error| {
                    EngineError::ExecutionFailed {
                        stage: "convert allocator pressure observation",
                        message: error.to_string(),
                    }
                })?,
            )
        } else {
            None
        };
        let resident = self
            .resident
            .as_ref()
            .or_else(|| self.states.values().next().map(|entry| &entry.resident));
        let leased = self.states.values().any(StateEntry::is_leased);
        let has_state = !self.states.is_empty()
            || resident
                .is_some_and(|resident| resident.cache.is_some() || resident.candidate.is_some());
        let execution = if self.active.is_some() {
            ExecutionState::Active
        } else if self.pending.is_some() {
            ExecutionState::PendingStartup
        } else {
            ExecutionState::Idle
        };
        Ok(crate::pressure::PressureObservation {
            admission: self.admission.snapshot()?,
            allocator,
            demand: self.pressure_demand(setup)?,
            execution,
            reclaimable: ReclaimableResources {
                idle_session_states: usize::from(has_state),
                leased_session_states: usize::from(leased),
                can_release_state_without_model: resident.is_some() && !leased,
                idle_model_loaded: resident.is_some() || !self.pool.is_empty(),
                idle_model_has_live_leases: if self.pool.iter().any(|r| !r.pins.is_pinned()) {
                    false
                } else {
                    leased || self.selected_model_pinned()
                },
            },
        })
    }

    fn pressure_release_idle_state(&mut self) -> Result<(), EngineError> {
        self.require_idle()?;
        self.require_unleased()?;
        // Remove canonical publication before mutation but preserve the
        // Resident containing the model and its weight reservation.
        self.unpublish_idle_state()?;
        if let Some(resident) = &mut self.resident {
            drop(resident.candidate.take());
            drop(resident.cache.take());
        }
        self.publish();
        Ok(())
    }

    fn pressure_unload_idle_model(&mut self) -> Result<(), EngineError> {
        self.require_idle()?;
        if self.evict_one_pooled_model() {
            self.publish();
            return Ok(());
        }
        self.require_unleased()?;
        if self.selected_model_pinned() {
            return Err(EngineError::Busy {
                active_request_id: None,
                message: "idle model has an explicit residency pin".to_owned(),
            });
        }
        self.unpublish_idle_state()?;
        // Resident drop order destroys native resources before their guards.
        drop(self.resident.take());
        self.publish();
        Ok(())
    }

    fn apply_startup_pressure(
        &mut self,
        setup: &Setup,
        control: &ExecutionControl,
        abandoned: &AtomicBool,
    ) -> Result<(), EngineError> {
        use crate::pressure::{ActionOutcome, PressureDecision, ReclaimAction, RejectReason};
        let Some(policy) = setup.pressure_policy else {
            return Ok(());
        };
        self.require_idle()?;
        let mut attempt = policy.begin();
        // Five actions plus one terminal evaluation. Never start a fresh
        // attempt to evade an exhausted budget.
        for _ in 0..=crate::pressure::MAX_RECLAMATION_ACTIONS {
            check_startup(control, abandoned)?;
            let observation = self.pressure_observation(setup)?;
            let decision =
                attempt
                    .evaluate(observation)
                    .map_err(|error| EngineError::ExecutionFailed {
                        stage: "evaluate startup pressure policy",
                        message: error.to_string(),
                    })?;
            let action = match decision {
                PressureDecision::Proceed => return Ok(()),
                PressureDecision::ClearAllocatorCache => ReclaimAction::ClearAllocatorCache,
                PressureDecision::ReleaseIdleSessionState => ReclaimAction::ReleaseIdleSessionState,
                PressureDecision::UnloadIdleModel => ReclaimAction::UnloadIdleModel,
                PressureDecision::Reject(reason) => {
                    if matches!(
                        reason,
                        RejectReason::NativeOwnerBusy | RejectReason::ProtectedResources
                    ) {
                        return Err(EngineError::Busy {
                            active_request_id: self
                                .request_owner()
                                .map(|id| id.as_str().to_owned()),
                            message: format!(
                                "startup pressure policy rejected protected ownership: {reason:?}"
                            ),
                        });
                    }
                    let operation = match reason {
                        RejectReason::ActionLimitReached => "startup pressure action limit reached",
                        RejectReason::NoProgress => "startup pressure reclamation made no progress",
                        RejectReason::NoReclaimableResources => {
                            "startup pressure has no reclaimable resources"
                        }
                        RejectReason::ReclamationExhausted => {
                            "startup pressure reclamation exhausted"
                        }
                        RejectReason::RequestExceedsAdmissionBudget => {
                            "startup demand exceeds admission budget"
                        }
                        RejectReason::RequestExceedsAllocatorCeiling => {
                            "startup native demand exceeds pressure ceiling"
                        }
                        _ => "startup pressure policy rejected admission",
                    };
                    let assessment = attempt.last_assessment();
                    if let Some(ceiling) = policy.options().allocator_ceiling_bytes {
                        if assessment.is_some_and(|a| a.reservation_deficit_bytes == 0) {
                            return Err(EngineError::MemoryLimitExceeded {
                                operation,
                                required_bytes: assessment
                                    .and_then(|a| a.projected_mlx_bytes)
                                    .unwrap_or(observation.demand.additional_mlx_bytes),
                                available_bytes: ceiling,
                            });
                        }
                    }
                    return Err(EngineError::MemoryLimitExceeded {
                        operation,
                        required_bytes: observation.demand.additional_reservation_bytes,
                        available_bytes: observation.admission.available_bytes(),
                    });
                }
            };
            check_startup(control, abandoned)?;
            let result = match action {
                ReclaimAction::ClearAllocatorCache => {
                    self.reclaim_mlx_allocator_cache().map(|_| ())
                }
                ReclaimAction::ReleaseIdleSessionState => self.pressure_release_idle_state(),
                ReclaimAction::UnloadIdleModel => self.pressure_unload_idle_model(),
            };
            let outcome = if result.is_ok() {
                ActionOutcome::Completed
            } else {
                ActionOutcome::Failed
            };
            attempt.complete_action(action, outcome).map_err(|error| {
                EngineError::ExecutionFailed {
                    stage: "acknowledge startup pressure action",
                    message: error.to_string(),
                }
            })?;
            // Return the actual operation error, not a synthetic retry error.
            result?;
            check_startup(control, abandoned)?;
            // Fresh native counters, ledger, eligibility, and incremental
            // demand are recomputed at the next iteration.
        }
        Err(internal(
            "pressure policy did not terminate within its action bound",
        ))
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

        self.reclaim_mlx_cache_before_idle_replacement(setup, artifact)?;
        self.unpublish_idle_state()?;

        let admission = self.admission.clone();
        let output_bytes = setup.plan.output_bytes;
        let (output, grant) = admit_startup_after_reclaim(&admission, output_bytes, || {
            let same_model = self
                .resident
                .as_ref()
                .is_some_and(|resident| model_matches(resident, artifact, setup.backend.as_ref()));

            if !same_model {
                self.switch_model(artifact, setup.backend.as_ref())?;
            }
            if let Some(resident) = &mut self.resident {
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
            check_startup(control, abandoned)?;
            // Reclaim only actual idle pool owners before pressure evaluation.
            loop {
                let residency = self.pressure_residency(setup)?;
                let compatible_cache = residency.reusable_model
                    && self
                        .resident
                        .as_ref()
                        .and_then(|r| r.cache.as_ref())
                        .is_some_and(|c| cache_matches(c, &setup.plan));
                let components = incremental_startup_components(
                    &setup.plan,
                    self.retention,
                    &residency,
                    compatible_cache,
                )?;
                let required = components.iter().try_fold(0u64, |n, b| add(n, *b))?;
                if self.admission.snapshot()?.available_bytes() >= required
                    || !self.evict_one_pooled_model()
                {
                    break;
                }
            }
            self.apply_startup_pressure(setup, control, abandoned)?;

            // Size the grant from post-reclaim residency so pressure and
            // admission cannot disagree about incremental demand.
            let residency = self.pressure_residency(setup)?;
            let same_cache = residency.reusable_model
                && self
                    .resident
                    .as_ref()
                    .or_else(|| self.states.values().next().map(|entry| &entry.resident))
                    .and_then(|resident| resident.cache.as_ref())
                    .is_some_and(|cache| cache_matches(cache, &setup.plan));
            let components = incremental_startup_components(
                &setup.plan,
                self.retention,
                &residency,
                same_cache,
            )?;
            let required = components
                .iter()
                .try_fold(0u64, |total, bytes| add(total, *bytes))?;
            while self.admission.snapshot()?.available_bytes() < required {
                if !self.evict_one_pooled_model() {
                    break;
                }
            }
            Ok(components)
        })?;

        self.pending = Some(PendingStartup {
            request_id: control.request_id().clone(),
            grant,
        });
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
        if self.pending.as_ref().map(|pending| &pending.request_id) != Some(control.request_id()) {
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
            self.switch_model(artifact, setup.backend.as_ref())?;
        }
        if let Some(resident) = &mut self.resident {
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
        let workspace = {
            let grant = &mut self
                .pending
                .as_mut()
                .ok_or_else(|| internal("admitted startup grant disappeared"))?
                .grant;
            grant.take("generation workspaces", workspace_bytes)?
        };

        if self.resident.is_none() {
            let peak = setup.plan.load_limits.max_load_peak_bytes();
            let mut weights = {
                let grant = &mut self
                    .pending
                    .as_mut()
                    .ok_or_else(|| internal("admitted startup grant disappeared"))?
                    .grant;
                grant
                    .take("model loading peak", peak)?
                    .ok_or_else(|| internal("startup grant missing model loading peak"))?
            };
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
                pins: ModelPinOwner::new(model.instance_id().clone()),
                idle_since: Instant::now(),
                idle_ttl: self.pool_ttl,
                cache: None,
                model,
                weights,
                capabilities,
                candidate: None,
            });
        }

        control.check_cancelled()?;
        let cache_bytes = setup.plan.cache.max_cache_bytes();
        let needs_cache = self
            .resident
            .as_ref()
            .ok_or_else(|| internal("loaded model unavailable"))?
            .cache
            .is_none();
        if needs_cache {
            let reservation = {
                let grant = &mut self
                    .pending
                    .as_mut()
                    .ok_or_else(|| internal("admitted startup grant disappeared"))?
                    .grant;
                grant
                    .take("model KV buffers", cache_bytes)?
                    .ok_or_else(|| internal("startup grant missing model KV buffers"))?
            };
            let resident = self
                .resident
                .as_mut()
                .ok_or_else(|| internal("loaded model unavailable"))?;
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

        let history_capacity = setup.plan.cache.capacity_tokens();
        let candidate = if retains_contents(self.retention) {
            let history = match previous.take() {
                Some(mut previous) if previous.history.tokens.capacity() >= history_capacity => {
                    previous.history.tokens.truncate(reused);
                    previous.history
                }
                previous => {
                    drop(previous);
                    let history_bytes = token_storage_bytes(history_capacity)?;
                    let reservation = {
                        let grant = &mut self
                            .pending
                            .as_mut()
                            .ok_or_else(|| internal("admitted startup grant disappeared"))?
                            .grant;
                        grant
                            .take("executed token history", history_bytes)?
                            .ok_or_else(|| {
                                internal("startup grant missing executed token history")
                            })?
                    };
                    let mut history =
                        TokenHistory::from_reservation(reservation, history_capacity)?;
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

        // Release any unused forecast (for example history when reused).
        if let Some(pending) = self.pending.take() {
            pending.grant.release()?;
        }
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
            .is_some_and(|pending| &pending.request_id != request)
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

        // Dropping pending releases any unpartitioned grant remainder.
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

        if let Some(resident) = &mut self.resident {
            resident.idle_since = Instant::now();
        }
        if !self.retention.retain_model && !self.selected_model_pinned() {
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
            || self
                .pending
                .as_ref()
                .is_some_and(|pending| &pending.request_id == request)
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
        if entry.resident.pins.is_pinned() && self.resident.is_some() {
            return Err(internal(
                "pinned state release conflicts with selected resident",
            ));
        }
        let entry = self
            .states
            .remove(state_id)
            .ok_or_else(|| internal("retained state disappeared during release"))?;

        if entry.resident.pins.is_pinned() {
            let mut resident = entry.resident;
            drop(resident.candidate.take());
            drop(resident.cache.take());
            self.resident = Some(resident);
        } else {
            drop(entry);
        }
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
        summary.model_pinned = self.selected_model_pinned();

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

fn model_ttl_expired(resident: &Resident, now: Instant) -> bool {
    match (
        resident.idle_ttl,
        now.checked_duration_since(resident.idle_since),
    ) {
        (Some(ttl), Some(elapsed)) => elapsed >= ttl,
        _ => false,
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

/// Run reclaim, then atomically admit the incremental startup components
/// and transfer the output allowance into an independent reservation.
///
/// `reclaim` must return the post-reclaim component vector. The first
/// component should be `output_bytes`; remaining entries are workspace,
/// optional loading peak, optional KV, and optional history.
fn admit_startup_after_reclaim(
    admission: &AdmissionController,
    output_bytes: u64,
    reclaim: impl FnOnce() -> Result<Vec<u64>, EngineError>,
) -> Result<(Arc<MemoryReservation>, ReservationGrant), EngineError> {
    let components = reclaim()?;
    let mut grant = admission.reserve_grant("generation startup", &components)?;
    let output = grant
        .take("generation output buffers", output_bytes)?
        .ok_or_else(|| {
            EngineError::invalid_request(
                "generation output buffers",
                "startup output allowance must be positive",
            )
        })?;
    Ok((Arc::new(output), grant))
}

/// Incremental startup ledger matching `pressure_demand` reservation bytes.
fn incremental_startup_components(
    plan: &RuntimePlan,
    retention: RetentionPolicy,
    residency: &RuntimeResidency,
    same_cache: bool,
) -> Result<Vec<u64>, EngineError> {
    let workspace = add(plan.temporary_bytes, plan.headroom_bytes)?;
    let weights = if residency.reusable_model {
        0
    } else {
        plan.load_limits.max_load_peak_bytes()
    };
    let cache = if same_cache {
        0
    } else {
        plan.cache.max_cache_bytes()
    };
    let history = if retains_contents(retention) {
        token_storage_bytes(plan.cache.capacity_tokens())?
    } else {
        0
    };
    Ok(vec![plan.output_bytes, workspace, weights, cache, history])
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
            pool: Vec::new(),
            pool_enabled: false,
            pool_ttl: None,
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
}

#[cfg(test)]
mod mlx_allocator_runtime_tests {
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
}

#[cfg(test)]
mod pressure_resource_reclamation_tests {
    use super::*;
    fn empty_state(admission: AdmissionController) -> NativeState {
        NativeState {
            active: None,
            pending: None,
            resident: None,
            states: HashMap::new(),
            pool: Vec::new(),
            pool_enabled: false,
            pool_ttl: None,
            admission,
            retention: RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: true,
            },
            summary: Arc::new(Mutex::new(Summary::empty(true))),
        }
    }
    #[test]
    fn pressure_actions_do_not_credit_unrelated_reservation_guards() {
        let admission = AdmissionController::new(4096).expect("admission");
        let _unrelated = admission.reserve("unrelated owner", 512).expect("guard");
        let before = admission.snapshot().expect("before");
        let mut state = empty_state(admission.clone());
        state
            .pressure_release_idle_state()
            .expect("empty state release");
        state
            .pressure_unload_idle_model()
            .expect("empty model unload");
        assert_eq!(admission.snapshot().expect("after"), before);
    }
    #[test]
    fn pending_native_ownership_blocks_destructive_pressure_actions() {
        let admission = AdmissionController::new(4096).expect("admission");
        let mut state = empty_state(admission.clone());
        state.pending = Some(PendingStartup {
            request_id: RequestId::new("pending-pressure").expect("ID"),
            grant: admission
                .reserve_grant("test pending startup", &[1])
                .expect("grant"),
        });
        assert_eq!(admission.snapshot().expect("before").reserved_bytes(), 1);
        assert!(matches!(
            state.pressure_release_idle_state(),
            Err(EngineError::Busy { .. })
        ));
        assert!(matches!(
            state.pressure_unload_idle_model(),
            Err(EngineError::Busy { .. })
        ));
        assert_eq!(admission.snapshot().expect("after").reserved_bytes(), 1);
    }
}

// ALLOY_PRESSURE_STARTUP_INTEGRATION_V1

#[cfg(test)]
mod model_pool_runtime_tests {
    use super::*;
    use crate::family::BufferMemoryUsage;
    use crate::types::{ExecutionBackend, WeightRepresentation};

    // Ownership-only fixture: not an executable model or checkpoint test.
    struct OwnedFixture {
        instance: ModelInstanceId,
        artifact: ArtifactId,
        payload: Vec<u8>,
        admission: AdmissionController,
        drops: Arc<Mutex<Vec<(String, u64)>>>,
    }
    impl Drop for OwnedFixture {
        fn drop(&mut self) {
            self.drops.lock().expect("drops").push((
                self.artifact.as_str().to_owned(),
                self.admission.snapshot().expect("ledger").reserved_bytes(),
            ));
        }
    }
    impl LoadedModel for OwnedFixture {
        fn instance_id(&self) -> &ModelInstanceId {
            &self.instance
        }
        fn artifact_id(&self) -> &ArtifactId {
            &self.artifact
        }
        fn registration(&self) -> &crate::registry::FamilyRegistration {
            panic!("ownership fixture does not implement execution registration")
        }
        fn capabilities(&self) -> &Capabilities {
            panic!("ownership fixture does not implement execution")
        }
        fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
            BufferMemoryUsage::new(self.payload.len() as u64, self.payload.len() as u64)
        }
        fn autoregressive(&mut self) -> Option<&mut dyn crate::family::AutoregressiveModel> {
            None
        }
    }
    fn capabilities() -> Capabilities {
        Capabilities {
            task: "ownership-test".to_owned(),
            family: "fixture".to_owned(),
            backend: ExecutionBackend::HostCpu,
            weight_representations: vec![WeightRepresentation::F32],
            cache_modes: vec![],
            context_limit_tokens: 1,
            supports_streaming: false,
            supports_incremental_kv: false,
            supports_prefix_reuse: false,
            decoding_modes: vec![],
        }
    }
    fn state() -> NativeState {
        NativeState {
            active: None,
            pending: None,
            resident: None,
            states: HashMap::new(),
            pool: Vec::new(),
            pool_enabled: true,
            pool_ttl: Some(Duration::from_secs(10)),
            admission: AdmissionController::new(256).expect("ledger"),
            retention: RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: false,
            },
            summary: Arc::new(Mutex::new(Summary::empty(true))),
        }
    }
    fn resident(
        state: &NativeState,
        name: &str,
        now: Instant,
        drops: &Arc<Mutex<Vec<(String, u64)>>>,
    ) -> Resident {
        let instance = ModelInstanceId::new();
        let weights = state
            .admission
            .reserve("fixture weights", 64)
            .expect("weights");
        Resident {
            pins: ModelPinOwner::new(instance.clone()),
            idle_since: now,
            idle_ttl: Some(Duration::from_secs(10)),
            cache: None,
            model: Box::new(OwnedFixture {
                instance,
                artifact: ArtifactId::new(name).expect("artifact"),
                payload: vec![0; 64],
                admission: state.admission.clone(),
                drops: drops.clone(),
            }),
            weights,
            capabilities: capabilities(),
            candidate: None,
        }
    }
    #[test]
    fn two_owned_models_keep_distinct_instances_and_committed_guards() {
        let mut state = state();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let now = Instant::now();
        let a = resident(&state, "a", now, &drops);
        let b = resident(&state, "b", now, &drops);
        assert!(!a.model.instance_id().same_instance(b.model.instance_id()));
        state.pool.extend([a, b]);
        assert_eq!(state.residency_info().len(), 2);
        assert_eq!(state.admission.snapshot().unwrap().reserved_bytes(), 128);
        assert!(drops.lock().unwrap().is_empty());
    }
    #[test]
    fn lru_destroys_native_model_before_releasing_its_guard() {
        let mut state = state();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let now = Instant::now();
        let a = resident(&state, "a", now, &drops);
        let b = resident(&state, "b", now + Duration::from_secs(1), &drops);
        state.pool.extend([b, a]);
        assert!(state.evict_one_pooled_model());
        assert_eq!(drops.lock().unwrap().as_slice(), &[("a".to_owned(), 128)]);
        assert_eq!(state.admission.snapshot().unwrap().reserved_bytes(), 64);
        assert_eq!(state.pool[0].model.artifact_id().as_str(), "b");
    }
    #[test]
    fn expired_pinned_model_survives_until_last_handle_drops() {
        let mut state = state();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let now = Instant::now();
        let a = resident(&state, "a", now, &drops);
        let instance = a.model.instance_id().clone();
        state.pool.push(a);
        let pin = state.pin_instance(&instance).unwrap().unwrap();
        let clone = pin.clone();
        state.expire_models_at(now + Duration::from_secs(10));
        assert_eq!(state.pool.len(), 1);
        drop(pin);
        assert!(!state.evict_one_pooled_model());
        drop(clone);
        state.expire_models_at(now + Duration::from_secs(10));
        assert!(state.pool.is_empty());
        assert_eq!(state.admission.snapshot().unwrap().reserved_bytes(), 0);
        assert!(state.pin_instance(&instance).unwrap().is_none());
    }
    #[test]
    fn ttl_boundary_uses_injected_time_without_sleeping() {
        let mut state = state();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let now = Instant::now();
        let a = resident(&state, "a", now, &drops);
        state.pool.push(a);
        state.expire_models_at(now + Duration::from_secs(9));
        assert_eq!(state.pool.len(), 1);
        state.expire_models_at(now + Duration::from_secs(10));
        assert!(state.pool.is_empty());
    }
    #[test]
    fn all_pinned_models_prevent_reclamation_and_do_not_credit_bytes() {
        let mut state = state();
        let drops = Arc::new(Mutex::new(Vec::new()));
        let a = resident(&state, "a", Instant::now(), &drops);
        let instance = a.model.instance_id().clone();
        state.pool.push(a);
        let _pin = state.pin_instance(&instance).unwrap().unwrap();
        let before = state.admission.snapshot().unwrap();
        assert!(!state.evict_one_pooled_model());
        assert_eq!(state.admission.snapshot().unwrap(), before);
        assert!(drops.lock().unwrap().is_empty());
    }
}
