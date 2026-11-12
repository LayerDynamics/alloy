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
//! Shortened, divergent, incompatible, or insufficiently retained hot state
//! starts from an empty logical cache. When tiered storage is configured,
//! a verified exact disk prefix may then be imported into that cache, even
//! for a newly loaded model instance. At least one input token is genuinely
//! prefilled to produce fresh final-position logits after disk restore.
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
//! A configured hot limit may spill and reclaim only unleased idle KV.
//!
//! Prefix-reuse capability reporting intentionally remains false until
//! production-path verification has passed.

use crate::parser_allocation::{
    FrozenParserAllocation, ParserAllocation, ParserAllocationError, ParserAllocationLimits,
};
use crate::residency_pin::{ModelPinOwner, ModelResidencyPin};
use crate::tiered_kv::{TieredKvConfig, TieredKvSnapshot, TieredKvStore};
use crate::{
    admission::{AdmissionController, AdmissionSnapshot, MemoryReservation, ReservationGrant},
    api::{ENGINE_API_VERSION, EngineCore, EngineFuture, EngineRuntime, ExecutionContext},
    cache_identity::{CacheIdentityInputs, compute_session_state_id},
    cancellation::CancellationToken,
    error::{EngineError, MemoryLimitReason, ProtectedResource},
    family::{
        CacheAllocationRequest, ExecutionControl, FamilyCache, LoadedModel, ModelFamily,
        ModelInstanceId, ModelLoadLimits, ModelLoadRequest, TokenLogits,
    },
    generation::{self, BlockStep, GenerationLimits, GenerationSession, TokenStep},
    registry::{ArtifactRegistration, FamilyVersion, Registry},
    types::{
        ArtifactId, CacheKind, CacheMode, CancelOutcome, Capabilities, ContextSnapshot,
        DecodingCapability, DecodingMode, Generation, GenerationEvent, GenerationEventStream,
        GenerationOptions, GenerationRequest, GenerationStream, MemoryBreakdown, MemoryEstimate,
        ModelProfileId, ReleaseOutcome, RenderedContext, RendererIdentity, RequestId,
        SessionStateId, TokenizerId, Usage,
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

/// Actual incremental output decoder, confined to the native worker.
///
/// Existing text-only implementations may use the default structured methods.
/// Protocol-aware implementations must override both structured methods.
/// Output is untrusted: decoding cannot assign invocation identities or
/// authorize effects. The runtime separately enforces delivery budgets.
pub trait TextDecoder {
    /// Decode one generated token without corrupting incremental state.
    fn push(&mut self, token: u32) -> Result<String, EngineError>;

    /// Finish pending tokenizer bytes after normal generation.
    fn finish(&mut self) -> Result<String, EngineError>;

    /// Process one token into zero or more parsed output blocks.
    ///
    /// Protocol-aware implementations must not publish incomplete calls or
    /// expose call markup as ordinary text. Pending output must be bounded.
    /// The caller separately enforces admitted buffering and delivery limits.
    fn push_blocks(
        &mut self,
        token: u32,
    ) -> Result<Vec<crate::types::GenerationOutputBlock>, EngineError> {
        let text = self.push(token)?;
        if text.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![crate::types::GenerationOutputBlock::Text(text)])
        }
    }

    /// Complete normal generation, including an output-token-limit stop.
    ///
    /// Protocol-aware implementations must reject truncated calls. They
    /// must not run another model step or change token usage here.
    /// On cancellation or execution failure, discard pending parser state
    /// instead of invoking this method to publish partial proposals.
    fn finish_blocks(&mut self) -> Result<Vec<crate::types::GenerationOutputBlock>, EngineError> {
        let text = self.finish()?;
        if text.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![crate::types::GenerationOutputBlock::Text(text)])
        }
    }

    /// Transfer a successfully finished parser batch, at most once.
    /// Ownership must follow returned proposal views until consumer disposal.
    /// The generation driver, not this method, applies the completion gate.
    fn take_parser_proposals(&mut self) -> Option<FrozenParserAllocation> {
        None
    }
}

/// Required real backend services in addition to the actual family loader.
pub trait RuntimeBackend: ModelFamily + 'static {
    /// Bound tokens for a complete projected rendered byte sequence.
    fn projected_token_upper_bound(
        &self,
        _registry: &Registry,
        _projection: crate::api::ProjectedToolRender<'_>,
    ) -> Result<usize, EngineError> {
        Err(EngineError::unsupported(
            "projected tokenizer bound",
            "backend has no verified bound",
        ))
    }
    /// Trusted content-manifest digest for cross-process KV compatibility.
    ///
    /// Return None when this backend cannot bind persisted KV to verified
    /// model bytes. The runtime calls this only after artifact verification.
    fn cache_manifest_digest(
        &self,
        _artifact: &ArtifactRegistration,
    ) -> Result<Option<[u8; 32]>, EngineError> {
        Ok(None)
    }
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

    /// Snapshot optional parser capacities before startup admission.
    ///
    /// None preserves text-only allocation behavior. Some requires `plan` to
    /// include the exact slab Layout in output_bytes and requires an override
    /// of decoder_with_parser_allocation. Metadata must remain immutable for
    /// the request. This method must not allocate native/model/parser storage.
    fn parser_allocation_limits(
        &self,
        _registry: &Registry,
        _profile: &ModelProfileId,
    ) -> Result<Option<ParserAllocationLimits>, EngineError> {
        Ok(None)
    }

    /// Construct a decoder with exclusive ownership of its admitted slab.
    ///
    /// A storage-backed decoder must retain/transfer this owner for every
    /// live view. The default rejects Some instead of allocating a side slab
    /// and silently continuing with the legacy Map/Value parser.
    fn decoder_with_parser_allocation(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        allocation: Option<ParserAllocation>,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        legacy_decoder_with_parser(allocation, || self.decoder(registry, profile))
    }

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
    /// Model-visible tool schemas associated with this retained state.
    pub tool_schema_hash: Option<[u8; 32]>,
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

/// Non-owning metadata copied from one resident at a native-worker boundary.
#[derive(Clone, Debug)]
pub struct ResidentModelSnapshot {
    /// Loaded artifact identity; this string does not retain its native model.
    pub artifact: ArtifactId,
    /// Weight bytes still owned by the resident's admission reservation.
    pub weights_reserved_bytes: u64,
    /// KV reservation bytes, not measured process or GPU memory.
    pub kv_reserved_bytes: u64,
    /// Whether an explicit model pin was alive when this metadata was copied.
    pub pinned: bool,
    /// Whether a canonical session lease protected this resident at publication.
    pub leased: bool,
}

/// Last published native residency metadata, available while native work is busy.
#[derive(Clone, Debug)]
pub struct ResidencySnapshot {
    /// Whether the native worker was alive when its metadata was published.
    pub worker_alive: bool,
    /// Monotonic time since the worker last published this metadata.
    pub sample_age: Duration,
    /// Independently owned residents, with no native objects or ownership handles.
    pub models: Vec<ResidentModelSnapshot>,
}

/// Result of a bounded live-pressure model-reclamation attempt.
#[derive(Clone, Debug)]
pub enum IdleModelReclaim {
    /// Pressure was normal or allocator-drain grace is still active.
    NotRequested,
    /// Idle KV was reclaimed; no weights were unloaded on this tick.
    DeferredForKv,
    /// This exact native model and its accounting guards were destroyed.
    Evicted(ModelInstanceId),
    /// All remaining candidates were protected, or no model was loaded.
    NoEligibleVictim,
}

/// One enforcement decision and the actions completed on the native worker.
pub struct MemoryEnforcementReport {
    /// Fresh enforcer decision, observed after the worker accepted the call.
    pub tick: crate::memory_enforcer::EnforcementTick,
    /// Allocator-only reclamation, when the native worker was idle.
    pub allocator_cache: Option<mlx_ffi::memory::CacheReclaimReport>,
    /// Allocator drain failure does not suppress eligible model eviction.
    pub allocator_error: Option<String>,
    /// At most one eligible idle model is destroyed per observation.
    pub model: IdleModelReclaim,
}

// M9_LIVE_PRESSURE_PUMP_V2

/// A bounded polling bridge between the live enforcer and native ownership.
///
/// Poll observations continue while the worker is busy. At most one native
/// enforcement future is retained; no dispatcher is spawned for extra polls.
/// Emergency cancellation is signalled directly through the engine scheduler
/// before advancing that future, including when native work is blocked.
pub struct MemoryEnforcementPump {
    engine: EngineCore<FamilyRuntime>,
    enforcer: crate::memory_enforcer::MemoryEnforcer,
    pending: Option<EngineFuture<'static, MemoryEnforcementReport>>,
}

/// One polling observation and an optional completed native action.
pub struct MemoryEnforcementPoll {
    /// Current poll-thread pressure; admission pausing has already occurred.
    pub tick: crate::memory_enforcer::EnforcementTick,
    /// Result of signalling the tick's exact execution target, if requested.
    /// Requested means the signal was accepted, not that cleanup completed.
    /// AlreadyFinished means the captured execution no longer owns the slot.
    pub cancellation: Option<Result<CancelOutcome, EngineError>>,
    /// Completion of the one outstanding worker call, when available.
    pub completed: Option<Result<MemoryEnforcementReport, EngineError>>,
}

impl MemoryEnforcementPump {
    /// Connect the engine's scheduler and runtime to their shared enforcer.
    pub fn new(
        engine: EngineCore<FamilyRuntime>,
        enforcer: crate::memory_enforcer::MemoryEnforcer,
    ) -> Self {
        Self {
            engine,
            enforcer,
            pending: None,
        }
    }

    /// Whether one bounded native enforcement operation is outstanding.
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Capture the execution owner, observe pressure, update admission, signal
    /// emergency cancellation, and advance at most one native reclamation call.
    pub fn poll_tick(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Result<MemoryEnforcementPoll, EngineError> {
        // Capture before probing memory: A may finish and B may start while
        // an observation is in progress. A decision about A must stay on A.
        let execution = self.engine.active_cancellation_target()?;
        let tick = self
            .enforcer
            .tick_for_execution(execution)
            .map_err(|error| EngineError::ExecutionFailed {
                stage: "poll live memory pressure",
                message: error.to_string(),
            })?;
        let cancellation = tick
            .request_execution_cancellation
            .as_ref()
            .map(|target| target.request_cancel());
        if self.pending.is_none() {
            let runtime = self.engine.runtime().clone();
            let enforcer = self.enforcer.clone();
            let decision = tick.clone();
            self.pending = Some(Box::pin(async move {
                runtime.apply_memory_tick(enforcer, decision).await
            }));
        }
        let completed = match self.pending.as_mut() {
            Some(future) => match future.as_mut().poll(cx) {
                Poll::Ready(result) => Some(result),
                Poll::Pending => None,
            },
            None => None,
        };
        if completed.is_some() {
            self.pending = None;
        }
        Ok(MemoryEnforcementPoll {
            tick,
            cancellation,
            completed,
        })
    }
}

impl FamilyRuntime {
    /// Observe live pressure and apply one bounded native reclamation step.
    ///
    /// Call serially from the polling loop. Observations and eligibility are
    /// recomputed on the native worker, not captured before a queued call.
    /// Native objects never leave that worker. This does not signal execution
    /// cancellation; use MemoryEnforcementPump to include the scheduler path.
    pub fn enforce_memory_tick(
        &self,
        enforcer: crate::memory_enforcer::MemoryEnforcer,
    ) -> EngineFuture<'_, MemoryEnforcementReport> {
        Box::pin(async move {
            let tick = enforcer
                .tick()
                .map_err(|error| EngineError::ExecutionFailed {
                    stage: "live memory enforcement tick",
                    message: error.to_string(),
                })?;
            self.apply_memory_tick(enforcer, tick).await
        })
    }

    /// Dispatch an already-polled decision without advancing grace twice.
    ///
    /// Prefer MemoryEnforcementPump for a background loop and out-of-band
    /// emergency cancellation. This worker path only reclaims idle resources,
    /// refreshing pressure and authorization before any destructive action.
    pub fn apply_memory_tick(
        &self,
        enforcer: crate::memory_enforcer::MemoryEnforcer,
        tick: crate::memory_enforcer::EnforcementTick,
    ) -> EngineFuture<'_, MemoryEnforcementReport> {
        let worker = Arc::clone(&self.worker);
        Box::pin(async move {
            dispatch("engine-live-memory-enforcement", move || {
                worker
                    .call(|reply| Command::EnforceMemory {
                        enforcer,
                        tick,
                        reply,
                    })?
                    .result
            })?
            .await
        })
    }

    /// Enable worker-owned idle model pooling and optional per-model TTL.
    ///
    /// Existing instances receive this default. Session leases protect their
    /// exact published model; unrelated artifacts may execute independently.
    /// Use before serving requests. None disables expiry, not pooling.
    pub fn with_model_pool(self, ttl: Option<Duration>) -> Result<Self, EngineError> {
        self.worker
            .call(|reply| Command::ConfigureModelPool { ttl, reply })?
            .result?;
        Ok(self)
    }

    /// Configure bounded hot retention and optional SSD KV before serving work.
    ///
    /// The worker admits its transfer allowance before scanning or reading cache
    /// metadata. Existing active/pending execution rejects reconfiguration.
    pub fn configure_tiered_kv(&self, config: TieredKvConfig) -> Result<(), EngineError> {
        self.worker
            .call(|reply| Command::ConfigureTieredKv { config, reply })?
            .result
    }

    /// Read cached KV-tier metrics without waiting on the native worker.
    pub fn tiered_kv_snapshot(&self) -> TieredKvSnapshot {
        lock_metadata(&self.worker.summary).tiered_kv.clone()
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
                    tiered: None,
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

    /// Read the latest published residents without queuing a native command.
    ///
    /// Long native operations leave this metadata unchanged. Its age is
    /// explicit, and reading it acquires no pins, leases, or reservations.
    pub fn residency_snapshot(&self) -> ResidencySnapshot {
        let summary = lock_metadata(&self.worker.summary);
        ResidencySnapshot {
            worker_alive: summary.alive,
            sample_age: summary.published_at.elapsed(),
            models: summary.resident_models.clone(),
        }
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
    /// A live lease prevents generation from mutating its owning model's
    /// published allocation, without blocking unrelated pooled models.
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
    fn projected_token_upper_bound<'a>(
        &'a self,
        registry: &'a Registry,
        projection: crate::api::ProjectedToolRender<'a>,
    ) -> EngineFuture<'a, usize> {
        Box::pin(async move {
            let artifact = registry
                .resolve_profile(projection.rendered.profile_id())?
                .artifact();
            self.backend(registry, artifact)?
                .projected_token_upper_bound(registry, projection)
        })
    }
    fn supports_memory_reclamation(&self) -> bool {
        self.pressure_policy.is_some() || lock_metadata(&self.worker.summary).pool_enabled
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
            let parser_limits = backend.parser_allocation_limits(registry, &profile)?;
            validate_parser_output_budget(plan.output_bytes, self.limits, parser_limits)?;

            let pooling = lock_metadata(&self.worker.summary).pool_enabled;
            if pooling {
                let artifact = resolved.artifact().id().clone();
                let family = resolved.artifact().family().to_owned();
                let version = backend.registration().implementation_version();
                let worker = Arc::clone(&self.worker);
                return dispatch("engine-pool-advisory-estimate", move || {
                    worker
                        .call(|reply| Command::AdvisoryPoolEstimate {
                            artifact,
                            family,
                            version,
                            plan,
                            reply,
                        })?
                        .result
                })?
                .await;
            }

            let summary = lock_metadata(&self.worker.summary).clone();
            if !summary.alive {
                return Err(internal("native worker has exited"));
            }

            let same_model = summary.artifact.as_ref() == Some(resolved.artifact().id());
            let same_cache = same_model && summary.cache_matches(&plan);
            let mut available = self.admission.snapshot()?.available_bytes();

            if !summary.active
                && summary.pin.strong_count() == 0
                && (summary.model_pin.strong_count() == 0 || same_model)
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

            let parser_limits =
                backend.parser_allocation_limits(&registry, request.profile_id())?;
            validate_parser_output_budget(plan.output_bytes, self.limits, parser_limits)?;

            let setup = Setup {
                registry,
                backend,
                plan,
                parser_limits,
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
            let worker = Arc::clone(&self.worker);
            dispatch("engine-owned-model-capabilities", move || {
                worker
                    .call(|reply| Command::LoadedCapabilities {
                        artifact: artifact_id,
                        reply,
                    })?
                    .result
            })?
            .await
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
    // W2_RUNTIME_PARSER_ALLOCATION_V1
    // Immutable metadata snapshot; allocation remains native-worker owned.
    parser_limits: Option<ParserAllocationLimits>,
    profile: ModelProfileId,
    options: GenerationOptions,
    pressure_policy: Option<crate::pressure::PressurePolicy>,
}

struct DecodedFinish {
    blocks: Vec<crate::types::GenerationOutputBlock>,
    proposals: Option<FrozenParserAllocation>,
}

struct Reply<T> {
    result: Result<T, EngineError>,
    usage: Usage,
}

#[expect(
    clippy::large_enum_variant,
    reason = "Worker commands keep request setup and reservation ownership inline without extra allocation in the bounded handoff queue"
)]
enum Command {
    ConfigureTieredKv {
        config: TieredKvConfig,
        reply: SyncSender<Reply<()>>,
    },
    LoadedCapabilities {
        artifact: ArtifactId,
        reply: SyncSender<Reply<Capabilities>>,
    },
    EnforceMemory {
        enforcer: crate::memory_enforcer::MemoryEnforcer,
        tick: crate::memory_enforcer::EnforcementTick,
        reply: SyncSender<Reply<MemoryEnforcementReport>>,
    },
    AdvisoryPoolEstimate {
        artifact: ArtifactId,
        family: String,
        version: FamilyVersion,
        plan: RuntimePlan,
        reply: SyncSender<Reply<MemoryEstimate>>,
    },
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
        reply: SyncSender<Reply<BlockStep>>,
    },
    Flush {
        request: RequestId,
        reply: SyncSender<Reply<DecodedFinish>>,
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
                        parser_proposals: None,
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
    parser_proposals: Option<FrozenParserAllocation>,
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
        let step = self.decode_next_blocks(options, cancellation)?;
        Ok(TokenStep {
            token_id: step.token_id,
            text_delta: blocks_to_text(step.blocks)?,
            end_of_sequence: step.end_of_sequence,
        })
    }

    fn decode_next_blocks(
        &mut self,
        options: &GenerationOptions,
        cancellation: &CancellationToken,
    ) -> Result<BlockStep, EngineError> {
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
        let blocks = self.finish_blocks()?;
        if self.parser_proposals.is_some() {
            return Err(internal(
                "slab proposals require structured session delivery",
            ));
        }
        blocks_to_text(blocks)
    }

    fn finish_blocks(&mut self) -> Result<Vec<crate::types::GenerationOutputBlock>, EngineError> {
        let reply = self.worker.call(|reply| Command::Flush {
            request: self.request_id.clone(),
            reply,
        })?;
        self.usage = reply.usage;
        let finished = reply.result?;
        self.parser_proposals = finished.proposals;
        Ok(finished.blocks)
    }

    fn take_parser_proposals(&mut self) -> Option<FrozenParserAllocation> {
        self.parser_proposals.take()
    }

    fn usage(&self) -> Usage {
        self.usage
    }
}

/// Legacy text-only view of parsed blocks. Structured proposals must use
/// the block methods, so this fails closed instead of dropping a call.
fn blocks_to_text(blocks: Vec<crate::types::GenerationOutputBlock>) -> Result<String, EngineError> {
    let mut text = String::new();
    for block in blocks {
        match block {
            crate::types::GenerationOutputBlock::Text(part) => text.push_str(&part),
            crate::types::GenerationOutputBlock::ToolCall(_) => {
                return Err(internal("structured output requires block delivery"));
            }
        }
    }
    Ok(text)
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
    tiered_kv: TieredKvSnapshot,
    published_at: Instant,
    resident_models: Vec<ResidentModelSnapshot>,
    pool_enabled: bool,
    alive: bool,
    active: bool,
    pin: Weak<()>,
    model_pin: Weak<()>,
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
            tiered_kv: TieredKvSnapshot::default(),
            published_at: Instant::now(),
            resident_models: Vec::new(),
            pool_enabled: false,
            alive,
            active: false,
            pin: Weak::new(),
            model_pin: Weak::new(),
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

impl Resident {
    fn pin_protection(&self) -> Option<ProtectedResource> {
        self.pins
            .live_handles()
            .map(|handles| ProtectedResource::ModelPin {
                artifact_id: self.model.artifact_id().clone(),
                instance_id: self.model.instance_id().clone(),
                handles,
            })
    }
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
    tool_schema_hash: Option<[u8; 32]>,
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
            tool_schema_hash: context.tool_schema_hash().copied(),
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
            && tool_schema_compatible(self.tool_schema_hash.as_ref(), context.tool_schema_hash())
            && self.renderer == *context.renderer()
            && self.cache_mode == *cache.mode()
            && cache.representation() == &self.cache_mode.representation
    }
}

struct StateCandidate {
    disk_identity: Option<String>,
    persisted: bool,
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

    fn lease_protection(&self) -> Option<ProtectedResource> {
        NonZeroUsize::new(self.pin.strong_count()).map(|handles| {
            ProtectedResource::SessionStateLease {
                artifact_id: self.resident.model.artifact_id().clone(),
                instance_id: self.resident.model.instance_id().clone(),
                state_id: self.metadata.state_id,
                handles,
            }
        })
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
    // Drop backing storage before releasing the remaining startup grant.
    parser: Option<ParserAllocation>,
    parser_limits: Option<ParserAllocationLimits>,
    grant: ReservationGrant,
}

impl PendingStartup {
    fn take_parser(
        &mut self,
        expected: Option<ParserAllocationLimits>,
    ) -> Result<Option<ParserAllocation>, EngineError> {
        let required = expected
            .map(ParserAllocation::required_bytes)
            .transpose()
            .map_err(parser_storage_error)?;
        let actual = self.parser.as_ref().map(ParserAllocation::reserved_bytes);
        if required != actual || !same_parser_configuration(self.parser_limits, expected) {
            return Err(internal(
                "pending parser allocation differs from admitted setup",
            ));
        }
        Ok(self.parser.take())
    }
}

// Deterministic idle-victim order: timestamp, owner class, pool index, state ID.
type IdleVictimKey = (Instant, u8, usize, [u8; 32]);

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
    tiered: Option<TieredKvStore>,
    summary: Arc<Mutex<Summary>>,
}

impl NativeState {
    fn tiered_snapshot(&self) -> TieredKvSnapshot {
        let Some(store) = &self.tiered else {
            return TieredKvSnapshot::default();
        };
        let mut snapshot = store.metrics.clone();
        snapshot.hot_limit_bytes = self.admission.snapshot().map_or(0, |ledger| {
            store.config.hot_limit.bytes(ledger.budget_bytes())
        });
        let bytes = |resident: &Resident| {
            resident
                .cache
                .as_ref()
                .map_or(0, |cache| cache.reservation.bytes())
        };
        snapshot.hot_reserved_bytes = self
            .pool
            .iter()
            .chain(self.resident.iter())
            .chain(self.states.values().map(|entry| &entry.resident))
            .fold(0u64, |sum, resident| sum.saturating_add(bytes(resident)));
        snapshot.protected_reserved_bytes = self
            .states
            .values()
            .filter(|entry| entry.is_leased())
            .fold(0u64, |sum, entry| {
                sum.saturating_add(bytes(&entry.resident))
            });
        if self.request_owner().is_some() {
            snapshot.protected_reserved_bytes = snapshot
                .protected_reserved_bytes
                .saturating_add(self.resident.as_ref().map_or(0, bytes));
        }
        snapshot
    }

    fn enforce_hot_limit(&mut self) {
        if self.tiered.is_none() {
            return;
        }
        loop {
            let snapshot = self.tiered_snapshot();
            if snapshot.hot_reserved_bytes <= snapshot.hot_limit_bytes {
                break;
            }
            match self.reclaim_one_idle_kv(true) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    if let Some(store) = &mut self.tiered {
                        store.record_error(error);
                    }
                    break;
                }
            }
        }
    }

    // Reclaim only unleased idle KV. Keep the actual model and weight guard.
    // The dedicated transfer guard was admitted before serving requests, so
    // this path does not try to obtain a new grant while admission is paused.
    fn reclaim_one_idle_kv(&mut self, include_selected: bool) -> Result<bool, EngineError> {
        if self.tiered.is_none() {
            return Ok(false);
        }
        enum Place {
            Pool(usize),
            Selected,
            Published(SessionStateId),
        }
        let mut best: Option<(IdleVictimKey, Place)> = None;
        let mut offer = |key, place| {
            if best.as_ref().is_none_or(|(current, _)| key < *current) {
                best = Some((key, place));
            }
        };
        for (index, resident) in self.pool.iter().enumerate() {
            if resident.cache.is_some() {
                offer((resident.idle_since, 0, index, [0; 32]), Place::Pool(index));
            }
        }
        if include_selected
            && self.request_owner().is_none()
            && let Some(resident) = &self.resident
            && resident.cache.is_some()
        {
            offer((resident.idle_since, 1, 0, [0; 32]), Place::Selected);
        }
        for (id, entry) in &self.states {
            if !entry.is_leased() && entry.resident.cache.is_some() {
                offer(
                    (entry.resident.idle_since, 2, 0, *id.as_bytes()),
                    Place::Published(*id),
                );
            }
        }
        let Some((_, place)) = best else {
            return Ok(false);
        };
        let park_published =
            !include_selected || self.resident.is_some() || self.request_owner().is_some();
        if matches!(place, Place::Published(_)) && park_published {
            self.pool
                .try_reserve(1)
                .map_err(|_| EngineError::AllocationFailed {
                    operation: "preserve model after KV spill",
                    requested_bytes: None,
                })?;
        }
        let store = self
            .tiered
            .as_mut()
            .ok_or_else(|| internal("KV storage disappeared"))?;
        let release = |store: &mut TieredKvStore, resident: &mut Resident| {
            persist_idle_kv(store, resident);
            drop(resident.candidate.take());
            // Native arrays drop before their reservation guard.
            drop(resident.cache.take());
        };
        match place {
            Place::Pool(index) => release(store, &mut self.pool[index]),
            Place::Selected => release(
                store,
                self.resident
                    .as_mut()
                    .ok_or_else(|| internal("KV owner disappeared"))?,
            ),
            Place::Published(id) => {
                let mut resident = self
                    .states
                    .remove(&id)
                    .ok_or_else(|| internal("KV publication disappeared"))?
                    .resident;
                release(store, &mut resident);
                if !park_published {
                    self.resident = Some(resident);
                } else {
                    self.pool.push(resident);
                }
            }
        }
        if let Some(store) = &mut self.tiered {
            store.metrics.hot_evictions = store.metrics.hot_evictions.saturating_add(1);
        }
        Ok(true)
    }

    // M9_INSTANCE_SCOPED_LEASE_EXECUTION_V1
    fn select_for_startup(
        &mut self,
        artifact: &ArtifactRegistration,
        backend: &dyn RuntimeBackend,
    ) -> Result<(), EngineError> {
        if !self.pool_enabled {
            self.require_unleased()?;
            self.unpublish_idle_state()?;
            return self.switch_model(artifact, backend);
        }
        // Reject the target allocation only. Other leased publications
        // remain independently owned by their StateEntry throughout B.
        let resources: Vec<_> = self
            .states
            .values()
            .filter(|entry| model_matches(&entry.resident, artifact, backend))
            .filter_map(StateEntry::lease_protection)
            .collect();
        if !resources.is_empty() {
            return Err(EngineError::ProtectedResources {
                operation: "select model for startup",
                resources,
            });
        }
        let published = self
            .states
            .iter()
            .find(|(_, entry)| model_matches(&entry.resident, artifact, backend))
            .map(|(id, _)| *id);
        if let Some(id) = published {
            if self.resident.is_some() {
                self.pool
                    .try_reserve(1)
                    .map_err(|_| EngineError::AllocationFailed {
                        operation: "park selected model before canonical reuse",
                        requested_bytes: None,
                    })?;
            }
            let entry = self
                .states
                .remove(&id)
                .ok_or_else(|| internal("selected canonical state disappeared"))?;
            if let Some(resident) = self.resident.take() {
                self.pool.push(resident);
            }
            self.resident = Some(entry.resident);
            Ok(())
        } else {
            self.switch_model(artifact, backend)
        }
    }

    fn evict_one_idle_model(
        &mut self,
        include_selected: bool,
    ) -> Result<Option<ModelInstanceId>, EngineError> {
        enum Place {
            Pool(usize),
            Selected,
            Published(SessionStateId),
        }
        let mut best: Option<(IdleVictimKey, Place)> = None;
        {
            let mut offer = |key, place| {
                if best.as_ref().is_none_or(|(current, _)| key < *current) {
                    best = Some((key, place));
                }
            };
            for (index, resident) in self.pool.iter().enumerate() {
                if !resident.pins.is_pinned() {
                    offer((resident.idle_since, 0, index, [0; 32]), Place::Pool(index));
                }
            }
            if include_selected
                && self.request_owner().is_none()
                && let Some(resident) = &self.resident
                && !resident.pins.is_pinned()
            {
                offer((resident.idle_since, 1, 0, [0; 32]), Place::Selected);
            }
            for (id, entry) in &self.states {
                if !entry.is_leased() && !entry.resident.pins.is_pinned() {
                    offer(
                        (entry.resident.idle_since, 2, 0, *id.as_bytes()),
                        Place::Published(*id),
                    );
                }
            }
        }
        let instance = match best.map(|(_, place)| place) {
            Some(Place::Pool(index)) => {
                let resident = self.pool.remove(index);
                let instance = resident.model.instance_id().clone();
                drop(resident);
                Some(instance)
            }
            Some(Place::Selected) => {
                let resident = self
                    .resident
                    .take()
                    .ok_or_else(|| internal("selected eviction owner disappeared"))?;
                let instance = resident.model.instance_id().clone();
                drop(resident);
                Some(instance)
            }
            Some(Place::Published(id)) => {
                let entry = self
                    .states
                    .remove(&id)
                    .ok_or_else(|| internal("canonical eviction owner disappeared"))?;
                let instance = entry.resident.model.instance_id().clone();
                drop(entry);
                Some(instance)
            }
            None => None,
        };
        Ok(instance)
    }

    fn estimate_pool_memory(
        &self,
        artifact: &ArtifactId,
        family: &str,
        version: FamilyVersion,
        plan: &RuntimePlan,
    ) -> Result<MemoryEstimate, EngineError> {
        let resources: Vec<_> = self
            .states
            .values()
            .filter(|entry| {
                entry.resident.model.artifact_id() == artifact
                    && entry.resident.model.registration().family() == family
                    && entry.resident.model.registration().implementation_version() == version
            })
            .filter_map(StateEntry::lease_protection)
            .collect();
        if !resources.is_empty() {
            return Err(EngineError::ProtectedResources {
                operation: "estimate model startup",
                resources,
            });
        }
        let matches = |resident: &&Resident| {
            resident.model.artifact_id() == artifact
                && resident.model.registration().family() == family
                && resident.model.registration().implementation_version() == version
        };
        let reusable = self
            .resident
            .iter()
            .chain(
                self.states
                    .values()
                    .filter(|entry| !entry.is_leased())
                    .map(|entry| &entry.resident),
            )
            .chain(self.pool.iter())
            .find(matches);
        let same_cache = reusable
            .and_then(|resident| resident.cache.as_ref())
            .is_some_and(|cache| cache_matches(cache, plan));
        let mut available = self.admission.snapshot()?.available_bytes();
        let idle = self.active.is_none() && self.pending.is_none();
        let reusable_instance = reusable.map(|resident| resident.model.instance_id());

        // These are forecast credits only. No reservation is changed here.
        if idle {
            for resident in &self.pool {
                let is_target = reusable_instance
                    .is_some_and(|id| resident.model.instance_id().same_instance(id));
                if !is_target && !resident.pins.is_pinned() {
                    available = add(available, resident_reserved_bytes(resident)?)?;
                }
            }
            {
                for resident in self.resident.iter().chain(
                    self.states
                        .values()
                        .filter(|entry| !entry.is_leased())
                        .map(|entry| &entry.resident),
                ) {
                    let is_target = reusable_instance
                        .is_some_and(|id| resident.model.instance_id().same_instance(id));
                    if !is_target && !resident.pins.is_pinned() {
                        available = add(available, resident_reserved_bytes(resident)?)?;
                    }
                }
                if !same_cache && let Some(resident) = reusable {
                    // Compatible model weights remain owned. Startup may
                    // replace incompatible KV and its associated history.
                    let replaceable = resident_reserved_bytes(resident)?
                        .checked_sub(resident.weights.bytes())
                        .ok_or_else(|| internal("invalid resident accounting"))?;
                    available = add(available, replaceable)?;
                }
            }
        }

        let loading_extra = if reusable.is_some() {
            0
        } else {
            plan.load_limits
                .max_load_peak_bytes()
                .checked_sub(plan.load_limits.max_resident_bytes())
                .ok_or_else(|| internal("invalid loader overlap bounds"))?
        };
        let history = if retains_contents(self.retention) {
            token_storage_bytes(plan.cache.capacity_tokens())?
        } else {
            0
        };
        MemoryEstimate::new(
            MemoryBreakdown {
                weights_bytes: if reusable.is_some() {
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
    }
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
            || (!self.pool_enabled
                && self
                    .states
                    .values()
                    .any(|entry| entry.resident.pins.is_pinned()))
    }

    fn expire_models_at(&mut self, now: Instant) {
        if !self.pool_enabled {
            return;
        }
        self.pool
            .retain(|resident| resident.pins.is_pinned() || !model_ttl_expired(resident, now));
        // Published allocations are disjoint from selected execution.
        // Each publication has its own lease and model-pin protection.
        self.states.retain(|_, entry| {
            entry.is_leased()
                || entry.resident.pins.is_pinned()
                || !model_ttl_expired(&entry.resident, now)
        });
        if self.request_owner().is_none()
            && self.resident.as_ref().is_some_and(|resident| {
                !resident.pins.is_pinned() && model_ttl_expired(resident, now)
            })
        {
            drop(self.resident.take());
        }
        self.publish();
    }

    fn enforce_memory_tick(
        &mut self,
        enforcer: &crate::memory_enforcer::MemoryEnforcer,
        mut tick: crate::memory_enforcer::EnforcementTick,
    ) -> Result<MemoryEnforcementReport, EngineError> {
        // Cancellation belongs to the poll-thread scheduler path. Never
        // replay an interruption decision after waiting in the native queue.
        tick.request_execution_cancellation = None;
        let refresh = || {
            enforcer
                .refresh_pressure()
                .map_err(|error| EngineError::ExecutionFailed {
                    stage: "refresh worker memory pressure",
                    message: error.to_string(),
                })
        };
        // The queued poll decision is not authority to destroy a model.
        tick.status = refresh()?;
        if tick.status.pressure == crate::memory_guard::PressureLevel::Ok {
            tick.request_model_eviction = false;
            tick.request_cache_reclaim = false;
            tick.deferred_for_pool_drain = false;
        }
        let mut allocator_error = None;
        let allocator_cache = if tick.request_cache_reclaim && self.request_owner().is_none() {
            match self.reclaim_mlx_allocator_cache() {
                Ok(report) => {
                    enforcer.clear_cache_reclaim_request();
                    Some(report)
                }
                Err(error) => {
                    allocator_error = Some(error.to_string());
                    None
                }
            }
        } else {
            None
        };
        // Pool draining can itself recover pressure. Observe again before
        // claiming model destruction, without consuming another grace poll.
        tick.status = refresh()?;
        if tick.status.pressure == crate::memory_guard::PressureLevel::Ok {
            tick.request_model_eviction = false;
            tick.request_cache_reclaim = false;
            tick.deferred_for_pool_drain = false;
        }
        // Recovery followed by new pressure starts a new grace window.
        // An older post-grace command must not bypass that newer decision.
        tick.request_model_eviction &= enforcer.model_eviction_requested();
        let reclaimed_kv = tick.request_model_eviction && self.reclaim_one_idle_kv(true)?;
        if reclaimed_kv {
            // Releasing live KV is an earlier tier than unloading weights.
            // Observe the result before authorizing another destructive step.
            if self.request_owner().is_none()
                && let Err(error) = self.reclaim_mlx_allocator_cache()
            {
                allocator_error = Some(error.to_string());
            }
            tick.status = refresh()?;
            if tick.status.pressure == crate::memory_guard::PressureLevel::Ok {
                tick.request_model_eviction = false;
                tick.request_cache_reclaim = false;
                tick.deferred_for_pool_drain = false;
            }
            tick.request_model_eviction &= enforcer.model_eviction_requested();
        }
        // Bound native work to one KV victim per tick. The next fresh tick
        // tries any remaining idle KV before weights become eligible.
        let model = if reclaimed_kv {
            IdleModelReclaim::DeferredForKv
        } else if tick.request_model_eviction {
            match self.evict_one_enforcer_model()? {
                Some(instance) => IdleModelReclaim::Evicted(instance),
                None => IdleModelReclaim::NoEligibleVictim,
            }
        } else {
            IdleModelReclaim::NotRequested
        };
        Ok(MemoryEnforcementReport {
            tick,
            allocator_cache,
            allocator_error,
            model,
        })
    }

    fn evict_one_enforcer_model(&mut self) -> Result<Option<ModelInstanceId>, EngineError> {
        self.evict_one_idle_model(true)
    }

    fn evict_one_pooled_model(&mut self) -> bool {
        // Startup may reclaim other idle owners, never its selected target.
        if self.reclaim_one_idle_kv(false).unwrap_or(false) {
            return true;
        }
        self.evict_one_idle_model(false).ok().flatten().is_some()
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
            if let Some(resource) = self.resident.as_ref().and_then(Resident::pin_protection) {
                return Err(EngineError::ProtectedResources {
                    operation: "replace resident model",
                    resources: vec![resource],
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
            self.enforce_hot_limit();
            self.publish();
            let command = match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(command) => command,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
            match command {
                Command::LoadedCapabilities { artifact, reply } => {
                    let result = self
                        .resident
                        .iter()
                        .chain(self.pool.iter())
                        .chain(self.states.values().map(|entry| &entry.resident))
                        .find(|resident| resident.model.artifact_id() == &artifact)
                        .map(|resident| resident.capabilities.clone())
                        .ok_or_else(|| EngineError::NotFound {
                            resource: "verified loaded artifact",
                            id: artifact.into_string(),
                        });
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::EnforceMemory {
                    enforcer,
                    tick,
                    reply,
                } => {
                    let result = self.enforce_memory_tick(&enforcer, tick);
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::AdvisoryPoolEstimate {
                    artifact,
                    family,
                    version,
                    plan,
                    reply,
                } => {
                    let result = self.estimate_pool_memory(&artifact, &family, version, &plan);
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
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
                    self.publish();
                    let _ = reply.send(Reply {
                        result,
                        usage: Usage::default(),
                    });
                }
                Command::ConfigureTieredKv { config, reply } => {
                    let result = self.require_idle().and_then(|()| {
                        let updated = self
                            .tiered
                            .as_mut()
                            .map_or(Ok(false), |store| store.update_limits(&config))?;
                        if !updated {
                            let store = TieredKvStore::open(config, &self.admission)?;
                            self.tiered = Some(store);
                        }
                        self.enforce_hot_limit();
                        Ok(())
                    });
                    self.publish();
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
                    if result.is_ok() {
                        if let (Some(store), Some(resident)) =
                            (&mut self.tiered, &mut self.resident)
                        {
                            persist_idle_kv(store, resident);
                        }
                        self.enforce_hot_limit();
                    }
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
        let resident = self.resident.as_ref();

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
        let resident = self.resident.as_ref();
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
        let resident = self.resident.as_ref();
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

    /// Capture eligibility and its blocker provenance together. These weak
    /// count observations never acquire ownership. A concurrent handle drop
    /// may make the result conservative; acquisition is worker-serialized.
    fn idle_reclamation_observation(
        &self,
    ) -> (
        crate::pressure::ReclaimableResources,
        Vec<ProtectedResource>,
    ) {
        let mut resources = Vec::new();
        let mut eligible_idle_models = 0;
        let mut leased_session_states = 0;
        for resident in &self.pool {
            if let Some(resource) = resident.pin_protection() {
                resources.push(resource);
            } else {
                eligible_idle_models += 1;
            }
        }
        if let Some(resident) = &self.resident {
            if let Some(resource) = resident.pin_protection() {
                resources.push(resource);
            } else if self.request_owner().is_none() {
                eligible_idle_models += 1;
            }
        }
        for entry in self.states.values() {
            let pin = entry.resident.pin_protection();
            let lease = entry.lease_protection();
            if pin.is_none() && lease.is_none() {
                eligible_idle_models += 1;
            }
            if let Some(resource) = pin {
                resources.push(resource);
            }
            if let Some(resource) = lease {
                leased_session_states += 1;
                resources.push(resource);
            }
        }
        let selected_state = self
            .resident
            .as_ref()
            .is_some_and(|resident| resident.cache.is_some() || resident.candidate.is_some());
        let pooled_states = self
            .pool
            .iter()
            .filter(|resident| resident.cache.is_some() || resident.candidate.is_some())
            .count();
        let idle_session_states = usize::from(selected_state) + pooled_states + self.states.len();
        (
            crate::pressure::ReclaimableResources {
                idle_session_states,
                leased_session_states,
                can_release_state_without_model: idle_session_states != 0,
                idle_model_loaded: self.resident.is_some()
                    || !self.pool.is_empty()
                    || !self.states.is_empty(),
                eligible_idle_models,
                idle_model_has_live_leases: !resources.is_empty(),
            },
            resources,
        )
    }

    fn pressure_observation(
        &self,
        setup: &Setup,
    ) -> Result<(crate::pressure::PressureObservation, Vec<ProtectedResource>), EngineError> {
        use crate::pressure::{AllocatorObservation, ExecutionState};
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
        let (reclaimable, resources) = self.idle_reclamation_observation();
        let execution = if self.active.is_some() {
            ExecutionState::Active
        } else if self.pending.is_some() {
            ExecutionState::PendingStartup
        } else {
            ExecutionState::Idle
        };
        Ok((
            crate::pressure::PressureObservation {
                admission: self.admission.snapshot()?,
                allocator,
                demand: self.pressure_demand(setup)?,
                execution,
                reclaimable,
            },
            resources,
        ))
    }

    fn pressure_release_idle_state(&mut self) -> Result<(), EngineError> {
        self.require_idle()?;
        if self
            .resident
            .as_ref()
            .is_some_and(|r| r.cache.is_some() || r.candidate.is_some())
        {
            if let Some(resident) = &mut self.resident {
                if let Some(store) = &mut self.tiered {
                    persist_idle_kv(store, resident);
                }
                drop(resident.candidate.take());
                drop(resident.cache.take());
            }
        } else if let Some(index) = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, resident)| resident.cache.is_some() || resident.candidate.is_some())
            .min_by_key(|(index, resident)| (resident.idle_since, *index))
            .map(|(index, _)| index)
        {
            // A model pin protects weights, not its unleased idle KV.
            // Pooled residents cannot own canonical state leases.
            let resident = &mut self.pool[index];
            if let Some(store) = &mut self.tiered {
                persist_idle_kv(store, resident);
            }
            drop(resident.candidate.take());
            drop(resident.cache.take());
        } else {
            let resources: Vec<_> = self
                .states
                .values()
                .filter_map(StateEntry::lease_protection)
                .collect();
            let id = self
                .states
                .iter()
                .filter(|(_, entry)| !entry.is_leased())
                .min_by_key(|(id, entry)| (entry.resident.idle_since, *id.as_bytes()))
                .map(|(id, _)| *id);
            if let Some(id) = id {
                if self.resident.is_some() {
                    self.pool
                        .try_reserve(1)
                        .map_err(|_| EngineError::AllocationFailed {
                            operation: "preserve model after idle-state reclamation",
                            requested_bytes: None,
                        })?;
                }
                let entry = self
                    .states
                    .remove(&id)
                    .ok_or_else(|| internal("idle-state reclamation owner disappeared"))?;
                let mut resident = entry.resident;
                if let Some(store) = &mut self.tiered {
                    persist_idle_kv(store, &mut resident);
                }
                drop(resident.candidate.take());
                drop(resident.cache.take());
                if self.resident.is_none() {
                    self.resident = Some(resident);
                } else {
                    self.pool.push(resident);
                }
            } else if !resources.is_empty() {
                return Err(EngineError::ProtectedResources {
                    operation: "reclaim idle session state",
                    resources,
                });
            }
        }
        self.publish();
        Ok(())
    }

    fn pressure_unload_idle_model(&mut self) -> Result<(), EngineError> {
        self.require_idle()?;
        let (_, resources) = self.idle_reclamation_observation();
        if self.evict_one_idle_model(true)?.is_none() && !resources.is_empty() {
            return Err(EngineError::ProtectedResources {
                operation: "unload idle model",
                resources,
            });
        }
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
            let (observation, resources) = self.pressure_observation(setup)?;
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
                    if reason == RejectReason::NativeOwnerBusy {
                        return Err(EngineError::Busy {
                            active_request_id: self
                                .request_owner()
                                .map(|id| id.as_str().to_owned()),
                            message: "native execution prevents startup pressure reclamation"
                                .to_owned(),
                        });
                    }
                    if reason == RejectReason::ProtectedResources {
                        let options = policy.options();
                        let resources: Vec<_> = resources
                            .into_iter()
                            .filter(|resource| match resource {
                                ProtectedResource::ModelPin { .. } => {
                                    options.allow_idle_model_unload
                                }
                                ProtectedResource::SessionStateLease { .. } => {
                                    options.allow_idle_model_unload
                                        || (options.allow_idle_state_release
                                            && observation
                                                .reclaimable
                                                .can_release_state_without_model)
                                }
                            })
                            .collect();
                        // Eligibility and counts were captured together, so
                        // even a concurrent final drop cannot create an
                        // empty or invented blocker report.
                        debug_assert!(!resources.is_empty());
                        return Err(EngineError::ProtectedResources {
                            operation: "startup pressure reclamation",
                            resources,
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
                    if reason == RejectReason::RequestExceedsAdmissionBudget {
                        return Err(EngineError::MemoryLimitExceeded {
                            operation,
                            reason: MemoryLimitReason::ReservationBudget {
                                budget_bytes: observation.admission.budget_bytes(),
                            },
                            required_bytes: observation.demand.additional_reservation_bytes,
                            available_bytes: observation.admission.available_bytes(),
                        });
                    }
                    // This early policy rejection occurs before an
                    // assessment exists. Its domain is the native ceiling,
                    // not the admission ledger, regardless of ledger fit.
                    if reason == RejectReason::RequestExceedsAllocatorCeiling {
                        let ceiling =
                            policy.options().allocator_ceiling_bytes.ok_or_else(|| {
                                internal("allocator ceiling rejection without a ceiling")
                            })?;
                        return Err(EngineError::MemoryLimitExceeded {
                            operation,
                            reason: MemoryLimitReason::AllocatorCeiling {
                                ceiling_bytes: ceiling,
                            },
                            required_bytes: observation.demand.additional_mlx_bytes,
                            available_bytes: ceiling,
                        });
                    }
                    if let Some(ceiling) = policy.options().allocator_ceiling_bytes
                        && assessment.is_some_and(|a| a.reservation_deficit_bytes == 0)
                    {
                        return Err(EngineError::MemoryLimitExceeded {
                            operation,
                            reason: MemoryLimitReason::AllocatorCeiling {
                                ceiling_bytes: ceiling,
                            },
                            required_bytes: assessment
                                .and_then(|a| a.projected_mlx_bytes)
                                .unwrap_or(observation.demand.additional_mlx_bytes),
                            available_bytes: ceiling,
                        });
                    }
                    return Err(EngineError::MemoryLimitExceeded {
                        operation,
                        reason: MemoryLimitReason::ReservationBudget {
                            budget_bytes: observation.admission.budget_bytes(),
                        },
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
        let resources: Vec<_> = self
            .states
            .values()
            .filter_map(StateEntry::lease_protection)
            .collect();
        if !resources.is_empty() {
            return Err(EngineError::ProtectedResources {
                operation: "reuse retained session state",
                resources,
            });
        }
        Ok(())
    }

    fn unpublish_idle_state(&mut self) -> Result<(), EngineError> {
        if !self.pool_enabled {
            self.require_unleased()?;
            if self.states.is_empty() {
                return Ok(());
            }
            if self.resident.is_some() || self.states.len() != 1 {
                return Err(internal("multiple resident ownership paths"));
            }
        } else if self.resident.is_some() {
            return Ok(());
        }
        let id = self
            .states
            .iter()
            .filter(|(_, entry)| !entry.is_leased())
            .min_by_key(|(id, entry)| (entry.resident.idle_since, *id.as_bytes()))
            .map(|(id, _)| *id);
        if let Some(id) = id {
            let entry = self
                .states
                .remove(&id)
                .ok_or_else(|| internal("canonical state disappeared"))?;
            self.resident = Some(entry.resident);
        }
        Ok(())
    }

    fn admit_output(
        &mut self,
        setup: &Setup,
        control: &ExecutionControl,
        abandoned: &AtomicBool,
    ) -> Result<Arc<MemoryReservation>, EngineError> {
        self.require_idle()?;
        check_startup(control, abandoned)?;

        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        setup.backend.verify_artifact(artifact, control)?;
        check_startup(control, abandoned)?;

        self.select_for_startup(artifact, setup.backend.as_ref())?;
        self.reclaim_mlx_cache_before_idle_replacement(setup, artifact)?;

        let admission = self.admission.clone();
        let output_bytes = setup.plan.output_bytes;
        let (output, parser, grant) = admit_startup_with_parser_after_reclaim(
            &admission,
            output_bytes,
            setup.parser_limits,
            || {
                let same_model = self.resident.as_ref().is_some_and(|resident| {
                    model_matches(resident, artifact, setup.backend.as_ref())
                });

                if !same_model {
                    self.switch_model(artifact, setup.backend.as_ref())?;
                }
                if let Some(resident) = &mut self.resident
                    && resident
                        .cache
                        .as_ref()
                        .is_some_and(|cache| !cache_matches(cache, &setup.plan))
                {
                    drop(resident.candidate.take());
                    drop(resident.cache.take());
                }
                // Keep compatible logical contents until prepare can
                // inspect the exact newly rendered token sequence.

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
            },
        )?;

        self.pending = Some(PendingStartup {
            request_id: control.request_id().clone(),
            parser,
            parser_limits: setup.parser_limits,
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
        if context.profile_id() != &setup.profile || context.token_ids().is_empty() {
            return Err(internal("prefill context does not match admitted setup"));
        }

        control.check_cancelled()?;
        let artifact = setup.registry.resolve_profile(&setup.profile)?.artifact();
        if context.artifact_id() != artifact.id() {
            return Err(internal("prefill artifact does not match admitted setup"));
        }

        if self
            .states
            .values()
            .any(|entry| model_matches(&entry.resident, artifact, setup.backend.as_ref()))
        {
            return Err(internal("startup target still has published ownership"));
        }
        setup.backend.verify_artifact(artifact, &control)?;
        control.check_cancelled()?;
        let disk_identity = if self
            .tiered
            .as_ref()
            .is_some_and(|store| store.metrics.disk_enabled)
            && matches!(setup.plan.cache.mode().kind, CacheKind::Full)
        {
            setup.backend.cache_manifest_digest(artifact)?.map(|manifest| {
                let identity = serde_json::json!({
                    "format": "alloy-kv-identity-v1",
                    "manifest_sha256": crate::tiered_kv::digest(&manifest),
                    "artifact": artifact.id().as_str(),
                    "family": artifact.family(),
                    "family_version": setup.backend.registration().implementation_version().to_string(),
                    "engine_api": ENGINE_API_VERSION,
                    "profile": context.profile_id().as_str(),
                    "tokenizer": context.tokenizer_id().as_str(),
                    "template": crate::tiered_kv::digest(context.template_hash()),
                    "renderer": [&context.renderer().implementation, &context.renderer().version],
                    "cache_representation": setup.plan.cache.mode().representation.as_str(),
                    "cache_kind": "full",
                });
                schema_bound_disk_identity(identity, context.tool_schema_hash())
            })
        } else {
            None
        };

        let same_model = self
            .resident
            .as_ref()
            .is_some_and(|resident| model_matches(resident, artifact, setup.backend.as_ref()));
        if !same_model {
            self.switch_model(artifact, setup.backend.as_ref())?;
        }
        if let Some(resident) = &mut self.resident
            && resident
                .cache
                .as_ref()
                .is_some_and(|cache| !cache_matches(cache, &setup.plan))
        {
            drop(resident.candidate.take());
            drop(resident.cache.take());
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
                    reason: MemoryLimitReason::BufferLimit,
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

        let mut reused = previous.as_ref().map_or(0, |candidate| {
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
            if let (Some(store), Some(identity)) = (&mut self.tiered, &disk_identity) {
                reused = store.restore(
                    identity,
                    context.token_ids(),
                    cache.cache.as_mut(),
                    &control,
                )?;
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

        let parser = self
            .pending
            .as_mut()
            .ok_or_else(|| internal("decoder construction lost its admitted startup"))?
            .take_parser(setup.parser_limits)?;
        let decoder = setup.backend.decoder_with_parser_allocation(
            &setup.registry,
            &setup.profile,
            parser,
        )?;
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
                disk_identity,
                persisted: false,
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

    fn decode(&mut self, request: &RequestId) -> Result<BlockStep, EngineError> {
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
        let blocks = if end_of_sequence {
            Vec::new()
        } else {
            active.decoder.push_blocks(token)?
        };

        Ok(BlockStep {
            token_id: token,
            blocks,
            end_of_sequence,
        })
    }

    fn flush(&mut self, request: &RequestId) -> Result<DecodedFinish, EngineError> {
        let active = self
            .active
            .as_mut()
            .filter(|active| active.control.request_id() == request)
            .ok_or_else(|| internal("flush has no matching active request"))?;

        active.control.check_cancelled()?;
        if active.text_finished {
            return Err(internal("text completion requested twice"));
        }

        let blocks = active.decoder.finish_blocks()?;
        active.control.check_cancelled()?;
        active.text_finished = true;
        let proposals = active.decoder.take_parser_proposals();
        Ok(DecodedFinish { blocks, proposals })
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

        if let Some(active) = &mut active
            && active.text_finished
            && !active.control.is_cancelled()
            && retains_contents(self.retention)
            && let (Some(mut completed), Some(logits)) =
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

        if let Some(cache) = &mut resident.cache
            && (cache.cache.reset().is_err()
                || cache.cache.len() != 0
                || check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes()).is_err())
        {
            drop(resident.cache.take());
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

        if self.states.contains_key(&state_id) {
            return Err(EngineError::invalid_request(
                "token_prefix_bytes",
                "canonical identity already owns another allocation",
            ));
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
            tool_schema_hash: compatibility.tool_schema_hash,
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
        if let Some(resource) = entry.lease_protection() {
            return Err(EngineError::ProtectedResources {
                operation: "release retained session state",
                resources: vec![resource],
            });
        }
        let park = self.resident.is_some() || self.request_owner().is_some();
        if entry.resident.pins.is_pinned() && park {
            self.pool
                .try_reserve(1)
                .map_err(|_| EngineError::AllocationFailed {
                    operation: "preserve pinned model after canonical release",
                    requested_bytes: None,
                })?;
        }
        let entry = self
            .states
            .remove(state_id)
            .ok_or_else(|| internal("retained state disappeared during release"))?;
        if entry.resident.pins.is_pinned() {
            let mut resident = entry.resident;
            drop(resident.candidate.take());
            drop(resident.cache.take());
            if park {
                self.pool.push(resident);
            } else {
                self.resident = Some(resident);
            }
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
        summary.tiered_kv = self.tiered_snapshot();
        let metadata = |resident: &Resident, leased| ResidentModelSnapshot {
            artifact: resident.model.artifact_id().clone(),
            weights_reserved_bytes: resident.weights.bytes(),
            kv_reserved_bytes: resident
                .cache
                .as_ref()
                .map_or(0, |cache| cache.reservation.bytes()),
            pinned: resident.pins.is_pinned(),
            leased,
        };
        summary.resident_models = self
            .pool
            .iter()
            .chain(self.resident.iter())
            .map(|resident| metadata(resident, false))
            .chain(
                self.states
                    .values()
                    .map(|entry| metadata(&entry.resident, entry.is_leased())),
            )
            .collect();
        summary.pool_enabled = self.pool_enabled;
        summary.active = self.active.is_some() || self.pending.is_some();
        summary.model_pin = self
            .resident
            .as_ref()
            .or_else(|| {
                if self.pool_enabled {
                    None
                } else {
                    self.states.values().next().map(|entry| &entry.resident)
                }
            })
            .map(|resident| resident.pins.protection_observer())
            .unwrap_or_default();

        let entry = if self.pool_enabled {
            None
        } else {
            self.states.values().next()
        };
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

fn persist_idle_kv(store: &mut TieredKvStore, resident: &mut Resident) {
    let (Some(candidate), Some(cache)) = (&mut resident.candidate, &mut resident.cache) else {
        return;
    };
    if candidate.persisted {
        return;
    }
    let Some(identity) = &candidate.disk_identity else {
        return;
    };
    match store.save(identity, &candidate.history.tokens, cache.cache.as_mut()) {
        Ok(saved) => candidate.persisted = saved,
        Err(error) => {
            store.record_error(error);
            // I/O failures leave valid hot state reusable. A failed native
            // readback can invalidate its tensors; discard those arrays and
            // guards now so the next request starts with a healthy cache.
            if check_cache_bytes(cache.cache.as_ref(), cache.reservation.bytes()).is_err() {
                drop(resident.candidate.take());
                drop(resident.cache.take());
            }
        }
    }
}

// Sum only guards actually owned by this resident. Native usage observations
// and logical tensor lengths must never be treated as released ledger bytes.
fn resident_reserved_bytes(resident: &Resident) -> Result<u64, EngineError> {
    let mut total = resident.weights.bytes();
    if let Some(cache) = &resident.cache {
        total = add(total, cache.reservation.bytes())?;
    }
    if let Some(candidate) = &resident.candidate {
        total = add(total, candidate.history.reservation.bytes())?;
        if let Some(logits) = &candidate.last_logits {
            total = add(total, logits.reservation.bytes())?;
        }
    }
    Ok(total)
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
            reason: MemoryLimitReason::BufferLimit,
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
/// component covers total output storage; remaining entries cover workspace,
/// optional loading peak, optional KV, and optional history. This helper takes
/// output_bytes for delivery; a parser-aware caller may take less than the
/// total output component and partition the slab from the same grant.
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
            reason: MemoryLimitReason::BufferLimit,
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

    validate_parser_output_budget(plan.output_bytes, limits, None)?;

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
#[path = "../tests/unit/runtime_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/runtime_mlx_allocator_runtime.rs"]
mod mlx_allocator_runtime_tests;

#[cfg(test)]
#[path = "../tests/unit/runtime_pressure_resource_reclamation.rs"]
mod pressure_resource_reclamation_tests;

// ALLOY_PRESSURE_STARTUP_INTEGRATION_V1

#[cfg(test)]
#[path = "../tests/unit/runtime_model_pool_runtime.rs"]
mod model_pool_runtime_tests;

// W2_RUNTIME_DECODER_CONTRACT_V1
#[cfg(test)]
#[path = "../tests/unit/runtime_decoder_contract.rs"]
mod decoder_contract_tests;

#[cfg(test)]
#[path = "../tests/unit/runtime_block_bridge.rs"]
mod block_bridge_tests;

fn tool_schema_compatible(retained: Option<&[u8; 32]>, requested: Option<&[u8; 32]>) -> bool {
    retained == requested
}

fn schema_bound_disk_identity(
    mut identity: serde_json::Value,
    schema: Option<&[u8; 32]>,
) -> String {
    if let Some(schema) = schema {
        identity["tool_schema_sha256"] =
            serde_json::Value::String(crate::tiered_kv::digest(schema));
    }
    crate::tiered_kv::digest(identity.to_string().as_bytes())
}

#[cfg(test)]
#[path = "../tests/unit/runtime_tool_schema_cache.rs"]
mod tool_schema_cache_tests;

fn same_parser_configuration(
    left: Option<ParserAllocationLimits>,
    right: Option<ParserAllocationLimits>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            [
                a.carry_bytes,
                a.call_bytes,
                a.decoded_bytes,
                a.depth,
                a.keys,
                a.nodes,
                a.proposal_bytes,
                a.calls,
            ] == [
                b.carry_bytes,
                b.call_bytes,
                b.decoded_bytes,
                b.depth,
                b.keys,
                b.nodes,
                b.proposal_bytes,
                b.calls,
            ]
        }
        _ => false,
    }
}

fn parser_storage_error(error: ParserAllocationError) -> EngineError {
    match error {
        ParserAllocationError::Admission(error) => error,
        ParserAllocationError::AllocationFailed { requested_bytes } => {
            EngineError::AllocationFailed {
                operation: "allocate admitted parser/proposal slab",
                requested_bytes: u64::try_from(requested_bytes).ok(),
            }
        }
        other => EngineError::invalid_request("parser_allocation", other.to_string()),
    }
}

fn validate_parser_output_budget(
    output_bytes: u64,
    limits: GenerationLimits,
    parser: Option<ParserAllocationLimits>,
) -> Result<(), EngineError> {
    let headers = limits
        .event_capacity()
        .checked_mul(std::mem::size_of::<String>())
        .ok_or_else(|| internal("output header size overflow"))?;
    let text = limits
        .max_buffered_text_bytes()
        .checked_add(limits.max_text_delta_bytes())
        .and_then(|value| value.checked_add(headers))
        .ok_or_else(|| internal("output budget size overflow"))?;
    let slab = parser
        .map(ParserAllocation::required_bytes)
        .transpose()
        .map_err(parser_storage_error)?
        .unwrap_or(0);
    let minimum = add(bytes_as_u64(text)?, slab)?;
    if output_bytes < minimum {
        return Err(EngineError::invalid_request(
            "output_bytes",
            "does not cover queued text, a pending delta, queue headers, and the selected parser slab",
        ));
    }
    Ok(())
}

fn legacy_decoder_with_parser(
    allocation: Option<ParserAllocation>,
    construct: impl FnOnce() -> Result<Box<dyn TextDecoder>, EngineError>,
) -> Result<Box<dyn TextDecoder>, EngineError> {
    if allocation.is_some() {
        return Err(EngineError::unsupported(
            "admitted parser storage",
            "backend must implement decoder_with_parser_allocation for its storage-backed parser",
        ));
    }
    construct()
}

/// Admit the same incremental total as before, but transfer only non-parser
/// output bytes to the stream and partition the slab from the SAME grant.
/// Parser allocation occurs on the native worker, after pressure/reclaim and
/// admission, and before pending startup is published.
fn admit_startup_with_parser_after_reclaim(
    admission: &AdmissionController,
    output_bytes: u64,
    parser_limits: Option<ParserAllocationLimits>,
    reclaim: impl FnOnce() -> Result<Vec<u64>, EngineError>,
) -> Result<
    (
        Arc<MemoryReservation>,
        Option<ParserAllocation>,
        ReservationGrant,
    ),
    EngineError,
> {
    let slab_bytes = parser_limits
        .map(ParserAllocation::required_bytes)
        .transpose()
        .map_err(parser_storage_error)?
        .unwrap_or(0);
    let delivery_bytes = output_bytes
        .checked_sub(slab_bytes)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            EngineError::invalid_request(
                "output_bytes",
                "parser slab must leave a positive text/delivery allowance",
            )
        })?;
    let (output, mut grant) = admit_startup_after_reclaim(admission, delivery_bytes, || {
        let components = reclaim()?;
        if components.first().copied() != Some(output_bytes) {
            return Err(internal(
                "startup output component differs from the admitted output plan",
            ));
        }
        Ok(components)
    })?;
    let parser = parser_limits
        .map(|limits| ParserAllocation::from_grant(&mut grant, limits))
        .transpose()
        .map_err(parser_storage_error)?;
    Ok((output, parser, grant))
}

#[cfg(test)]
#[path = "../tests/unit/runtime_runtime_parser_allocation.rs"]
mod runtime_parser_allocation_tests;
