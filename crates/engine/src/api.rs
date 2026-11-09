//! Versioned engine API and component wiring.
//!
//! EngineCore connects:
//!
//! - Registry lookup and profile validation.
//! - Exclusive scheduling through Scheduler.
//! - Cooperative cancellation through CancellationToken.
//! - Actual rendering, admission, execution, and state management supplied
//!   by an EngineRuntime implementation.
//!
//! This module does not fabricate an execution backend. Constructing an
//! EngineCore requires a concrete runtime implementation.
//!
//! The actual execution worker must own its ExecutionContext until model
//! work stops. Client-stream ownership is separate from execution ownership.

use crate::{
    cancellation::CancellationToken,
    error::{EngineError, MemoryLimitReason},
    registry::Registry,
    scheduler::{ExecutionCancellationTarget, ExecutionOutcome, ExecutionPermit, Scheduler},
    types::{
        ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, Generation, GenerationEvent,
        GenerationEventStream, GenerationRequest, GenerationStream, MemoryEstimate, ModelProfileId,
        ReleaseOutcome, RenderedContext, RequestId, SessionStateId,
    },
};
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

/// Version of the public engine contract.
///
/// Breaking changes to request types, event semantics, or cache identity
/// interpretation require a version increment.
///
/// This is the initial contract version, not a declaration that every
/// execution capability has been implemented.
pub const ENGINE_API_VERSION: u32 = 1;

/// An asynchronous engine operation that can cross task boundaries.
pub type EngineFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, EngineError>> + Send + 'a>>;

/// Execution ownership and cancellation observation for one accepted request.
///
/// Move this value into the actual model worker. Do not leave it owned by
/// the client-facing event stream while native execution continues elsewhere.
///
/// This type is intentionally not Clone.
#[derive(Debug)]
#[must_use = "the actual execution worker must retain this context until work stops"]
pub struct ExecutionContext {
    permit: ExecutionPermit,
    cancellation: CancellationToken,
}

impl ExecutionContext {
    fn new(permit: ExecutionPermit) -> Self {
        let cancellation = permit.cancellation_token();

        Self {
            permit,
            cancellation,
        }
    }

    /// Return the request owning this execution context.
    pub fn request_id(&self) -> &RequestId {
        self.permit.request_id()
    }

    /// Obtain a cancellation observation token.
    ///
    /// Token clones do not own the execution slot. This context must remain
    /// alive while model work uses that slot.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Check whether cancellation was requested.
    ///
    /// Execution should check before prefill, between prefill chunks, and
    /// between decode steps.
    pub fn check_cancelled(&self) -> Result<(), EngineError> {
        self.cancellation.check_cancelled(self.permit.request_id())
    }

    /// Commit completion after actual model execution has stopped.
    ///
    /// The scheduler arbitrates cancellation against completion and returns
    /// the single terminal event for delivery.
    ///
    /// Before calling this method, the worker must finish native operations
    /// and release resources that must not overlap the next execution.
    /// Request-local memory reservations must follow the admission
    /// implementation's corresponding release rules.
    pub fn finish(self, outcome: ExecutionOutcome) -> Result<GenerationEvent, EngineError> {
        self.permit.finish(outcome)
    }
}

/// Implemented execution services used by EngineCore.
///
/// This is an interface to real components, not a mock backend. There are
/// no default implementations that report success without performing work.
///
/// A runtime may dispatch to multiple registered model families. It must
/// not hardcode a single family into the shared engine control path.
pub trait EngineRuntime: Send + Sync + 'static {
    /// Render and tokenize using the selected profile's real artifacts.
    ///
    /// The registry supplied here is the same registry used by EngineCore
    /// for profile and artifact resolution.
    ///
    /// Implementations must verify the relevant template/tokenizer inputs
    /// and return exact token IDs, not an approximate token count.
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext>;

    /// Compute an estimate from the actual selected execution representation.
    ///
    /// The estimate is advisory and does not acquire a reservation.
    /// The profile's maximum output allowance must be included.
    fn estimate_memory<'a>(
        &'a self,
        registry: &'a Registry,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, MemoryEstimate>;

    /// Whether startup implements bounded reclamation after an advisory
    /// estimate fails. False preserves EngineCore's original fail-fast path.
    /// Returning true never waives atomic admission or native cleanup.
    fn supports_memory_reclamation(&self) -> bool {
        false
    }

    /// Reserve resources and arrange actual model execution.
    ///
    /// EngineCore has already:
    ///
    /// - Resolved the profile.
    /// - Validated generation options.
    /// - Acquired the scheduler's execution permit.
    /// - Rendered and counted the input.
    /// - Checked the profile context limit.
    /// - Checked an advisory memory estimate.
    ///
    /// This method must still atomically acquire real memory reservations
    /// against the current budget before loading or execution. An earlier
    /// estimate is not authorization to allocate.
    ///
    /// The supplied rendered input must be used as-is. If its identities
    /// cannot be honored, reject it rather than silently rerender with
    /// different tokenizer or template settings.
    ///
    /// Move execution into the actual worker. That worker must retain it
    /// until model work stops and call ExecutionContext::finish to obtain
    /// the terminal event.
    ///
    /// Returning an error must not abandon running model work. If startup
    /// began a worker, stop and join or otherwise retain ownership of that
    /// worker until cleanup completes.
    ///
    /// The returned Generation must carry the accepted request ID.
    fn start_generation(
        &self,
        registry: Arc<Registry>,
        request: GenerationRequest,
        rendered: RenderedContext,
        execution: ExecutionContext,
    ) -> EngineFuture<'_, Generation>;

    /// Release actual engine-owned session state.
    ///
    /// Return ProtectedResources if a canonical lease protects that state,
    /// or Busy if native execution owns the operation. Never free tensors
    /// underneath execution or delete durable session history.
    fn release_session_state(&self, state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome>;

    /// Report capabilities established by the selected implementation.
    ///
    /// Registry metadata alone is not artifact verification. Unknown,
    /// unverified, or unsupported artifacts must not be presented as usable.
    fn capabilities<'a>(
        &'a self,
        registry: &'a Registry,
        artifact_id: ArtifactId,
    ) -> EngineFuture<'a, Capabilities>;
}

/// Public, versioned interface used by the harness.
pub trait Engine: Send + Sync {
    /// Return the engine API version implemented by this handle.
    fn api_version(&self) -> u32;

    /// Render and tokenize the snapshot using its selected profile.
    ///
    /// Rendering does not reserve execution memory.
    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext>;

    /// Estimate memory using the profile's maximum output allowance.
    ///
    /// The returned fit decision is advisory. Generation must reserve
    /// resources against the current budget before execution.
    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate>;

    /// Validate, schedule, admit, and start actual generation.
    ///
    /// Startup failures return directly. After a handle is returned,
    /// execution failures are delivered through the generation stream.
    ///
    /// Dropping the returned stream requests cancellation; it does not
    /// release the worker's execution permit.
    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation>;

    /// Request cancellation of an active request.
    ///
    /// Requested does not imply that an in-flight native operation has
    /// already stopped. AlreadyFinished means terminal completion won
    /// the scheduler's lifecycle arbitration.
    fn cancel(&self, request_id: RequestId) -> EngineFuture<'_, CancelOutcome>;

    /// Release engine-owned state without affecting durable history.
    fn release_session_state(&self, state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome>;

    /// Report actual implementation capabilities for an artifact.
    fn capabilities(&self, artifact_id: ArtifactId) -> EngineFuture<'_, Capabilities>;
}

/// Concrete engine control path using registered metadata and real services.
///
/// Build and populate the registry before construction. All clones share
/// the same registry snapshot, scheduler, and runtime.
///
/// The runtime is responsible for actual model-family dispatch. EngineCore
/// itself contains no Qwen3-specific imports or execution assumptions.
pub struct EngineCore<R: EngineRuntime> {
    registry: Arc<Registry>,
    scheduler: Scheduler,
    runtime: Arc<R>,
}

impl<R: EngineRuntime> EngineCore<R> {
    /// Connect a populated registry, scheduler, and actual runtime.
    ///
    /// The caller supplies scheduler policy explicitly by constructing
    /// Scheduler with the desired completed-request history capacity.
    pub fn new(registry: Registry, scheduler: Scheduler, runtime: R) -> Self {
        Self {
            registry: Arc::new(registry),
            scheduler,
            runtime: Arc::new(runtime),
        }
    }

    /// Borrow the registry snapshot used by this engine.
    ///
    /// Inspection does not load models or acquire reservations.
    pub fn registry(&self) -> &Registry {
        self.registry.as_ref()
    }

    /// Return the active execution owner, if any.
    pub fn active_request_id(&self) -> Result<Option<RequestId>, EngineError> {
        self.scheduler.active_request_id()
    }

    /// Observe the execution owner and its cancellation state atomically.
    ///
    /// This read never enters the native worker's command queue.
    pub fn execution_snapshot(&self) -> Result<crate::scheduler::SchedulerSnapshot, EngineError> {
        self.scheduler.snapshot()
    }

    /// Snapshot the current execution for a delayed cancellation decision.
    ///
    /// Capture before observing the condition that may require interruption.
    /// The target is bound to this scheduler and this execution lease, so it
    /// cannot cancel a later request. It does not own native resources or the
    /// scheduler permit and never waits on the native worker's command queue.
    pub fn active_cancellation_target(
        &self,
    ) -> Result<Option<ExecutionCancellationTarget>, EngineError> {
        self.scheduler.active_cancellation_target()
    }

    /// Borrow the concrete runtime (memory reclaim, admission controller).
    pub fn runtime(&self) -> &R {
        self.runtime.as_ref()
    }
}

impl<R: EngineRuntime> Clone for EngineCore<R> {
    fn clone(&self) -> Self {
        Self {
            registry: Arc::clone(&self.registry),
            scheduler: self.scheduler.clone(),
            runtime: Arc::clone(&self.runtime),
        }
    }
}

impl<R: EngineRuntime> Engine for EngineCore<R> {
    fn api_version(&self) -> u32 {
        ENGINE_API_VERSION
    }

    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext> {
        Box::pin(async move {
            let expected_snapshot = context.snapshot_id().clone();
            let expected_profile = context.profile_id().clone();

            let expected_artifact = self
                .registry
                .resolve_profile(&expected_profile)?
                .artifact()
                .id()
                .clone();

            let rendered = self
                .runtime
                .render_and_count(self.registry.as_ref(), context)
                .await?;

            if rendered.snapshot_id() != &expected_snapshot
                || rendered.profile_id() != &expected_profile
                || rendered.artifact_id() != &expected_artifact
            {
                return Err(EngineError::Internal {
                    message: "runtime rendering identities do not match \
                              the requested snapshot, profile, and artifact"
                        .to_owned(),
                });
            }

            Ok(rendered)
        })
    }

    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        Box::pin(async move {
            self.registry
                .profile(&profile)?
                .check_context(input_tokens)?;

            self.runtime
                .estimate_memory(self.registry.as_ref(), profile, input_tokens)
                .await
        })
    }

    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        Box::pin(async move {
            self.registry
                .validate_generation_options(request.profile_id(), request.options())?;

            if request.context().profile_id() != request.profile_id() {
                return Err(EngineError::invalid_request(
                    "profile_id",
                    "request profile does not match its context snapshot",
                ));
            }

            let request_id = request.request_id().clone();

            // The permit is held throughout startup and then transferred
            // into the actual execution worker by the runtime.
            let permit = self.scheduler.try_acquire(request_id.clone())?;

            permit.check_cancelled()?;

            let rendered = self.render_and_count(request.context().clone()).await?;

            permit.check_cancelled()?;

            if rendered.token_ids().is_empty() {
                return Err(EngineError::invalid_request(
                    "token_ids",
                    "generation input must contain at least one token",
                ));
            }

            self.registry
                .profile(request.profile_id())?
                .check_context(rendered.token_count())?;

            let estimate = self
                .estimate_memory(request.profile_id().clone(), rendered.token_count())
                .await?;

            permit.check_cancelled()?;

            if !estimate.fits() && !self.runtime.supports_memory_reclamation() {
                return Err(EngineError::MemoryLimitExceeded {
                    operation: "generation startup estimate",
                    required_bytes: estimate.reserved_bytes(),
                    available_bytes: estimate.available_bytes(),
                    reason: MemoryLimitReason::AdvisoryEstimate,
                });
            }

            let execution = ExecutionContext::new(permit);

            let started = self
                .runtime
                .start_generation(Arc::clone(&self.registry), request, rendered, execution)
                .await;

            let generation = match started {
                Ok(generation) => generation,
                Err(error) => {
                    // A correct runtime has already stopped startup work
                    // before returning an error. Also signal any worker
                    // that still retains the permit; never free its slot
                    // here while it may still be executing.
                    let _ = self.scheduler.cancel(&request_id);
                    return Err(error);
                }
            };

            if generation.request_id() != &request_id {
                let _ = self.scheduler.cancel(&request_id);

                // Dropping the incorrect returned stream invokes its own
                // cleanup semantics. It is never returned to this caller.
                drop(generation);

                return Err(EngineError::Internal {
                    message: "runtime returned a generation handle for \
                              a different request"
                        .to_owned(),
                });
            }

            let Generation { request_id, events } = generation;

            let events: GenerationStream = Box::pin(CancelOnDropStream {
                inner: events,
                scheduler: self.scheduler.clone(),
                request_id: request_id.clone(),
            });

            Ok(Generation { request_id, events })
        })
    }

    fn cancel(&self, request_id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        Box::pin(async move { self.scheduler.cancel(&request_id) })
    }

    fn release_session_state(&self, state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        Box::pin(async move { self.runtime.release_session_state(state_id).await })
    }

    fn capabilities(&self, artifact_id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        Box::pin(async move {
            let artifact = self.registry.artifact(&artifact_id)?;
            let family = self.registry.family(artifact.family())?;

            let expected_family = artifact.family().to_owned();

            let context_ceiling = artifact
                .declared_context_limit_tokens()
                .min(family.capabilities().context_limit_tokens);

            let capabilities = self
                .runtime
                .capabilities(self.registry.as_ref(), artifact_id)
                .await?;

            if capabilities.family != expected_family {
                return Err(EngineError::Internal {
                    message: "runtime reported capabilities for \
                              a different registered family"
                        .to_owned(),
                });
            }

            if capabilities.context_limit_tokens == 0
                || capabilities.context_limit_tokens > context_ceiling
            {
                return Err(EngineError::Internal {
                    message: "runtime capability context limit exceeds \
                              the registered artifact/family bounds"
                        .to_owned(),
                });
            }

            if capabilities.supports_prefix_reuse && !capabilities.supports_incremental_kv {
                return Err(EngineError::Internal {
                    message: "runtime reported prefix reuse without \
                              incremental KV support"
                        .to_owned(),
                });
            }

            Ok(capabilities)
        })
    }
}

/// Ensures client-stream disposal requests cancellation without owning
/// or releasing the execution permit.
///
/// Request IDs must be fresh for generation attempts, as required by the
/// scheduler contract. They must not be recycled while old handles exist.
struct CancelOnDropStream {
    inner: GenerationStream,
    scheduler: Scheduler,
    request_id: RequestId,
}

impl GenerationEventStream for CancelOnDropStream {
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

impl Drop for CancelOnDropStream {
    fn drop(&mut self) {
        // Completion may already have committed, or its history entry may
        // have expired. Neither case permits releasing execution here.
        //
        // Scheduler cancellation signals the worker without taking its
        // permit away.
        let _ = self.scheduler.cancel(&self.request_id);
    }
}

// ALLOY_PRESSURE_STARTUP_INTEGRATION_V1

// W2_COMPLETION_GATE_TEST_CONTEXT_V1
#[cfg(test)]
pub(crate) fn tool_output_test_execution_context(permit: ExecutionPermit) -> ExecutionContext {
    ExecutionContext::new(permit)
}
