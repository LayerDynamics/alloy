//! Single-active-generation scheduling and request lifecycle.
//!
//! This scheduler provides:
//!
//! - Exclusive execution ownership through a non-cloneable permit.
//! - Rejection of conflicting or duplicate active requests.
//! - Cooperative cancellation through the real cancellation primitive.
//! - Serialized cancellation-versus-completion arbitration.
//! - Bounded tracking of recently finished requests.
//! - Cleanup when an execution permit is abandoned.
//!
//! There is no hidden request queue.
//!
//! The actual generation worker must own its ExecutionPermit until model
//! execution has stopped. The client-facing stream must not own the permit
//! in place of that worker: dropping a client stream must request cancellation,
//! not prematurely release an execution slot still used by native work.
//!
//! Memory admission, model loading, and event transport are separate
//! components. Acquiring a permit does not reserve model memory.

use crate::{
    cancellation::{CancellationSource, CancellationToken},
    error::EngineError,
    types::{CancelOutcome, FinishReason, GenerationEvent, RequestId, Usage},
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

/// Observable lifecycle state of a known request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestState {
    /// The request owns the execution slot.
    Running,
    /// Cancellation was accepted, but execution still owns the slot.
    CancellationRequested,
    /// Execution relinquished the slot.
    ///
    /// This includes explicit completion and abandoned execution permits.
    /// It does not prove that a terminal event reached a client.
    Finished,
}

/// One read-only observation of the execution slot under its scheduler lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    /// Current slot owner, including startup or cancellation awaiting cleanup.
    pub active_request_id: Option<RequestId>,
    /// Whether that exact owner has accepted a cancellation request.
    pub cancellation_requested: bool,
}

/// Actual result supplied by the generation worker after execution stops.
///
/// The scheduler converts this into one terminal GenerationEvent and
/// arbitrates any cancellation accepted before terminal completion.
#[derive(Debug)]
pub enum ExecutionOutcome {
    /// Generation completed successfully.
    Completed {
        /// Why token generation stopped.
        finish_reason: FinishReason,
        /// Usage measured by the actual execution implementation.
        usage: Usage,
    },

    /// Generation failed.
    Failed {
        /// Structured execution failure.
        error: EngineError,
        /// Usage measured before failure.
        partial_usage: Usage,
    },
}

/// Family-independent scheduler for one active generation.
///
/// Clones refer to the same execution slot and request history.
///
/// Completed history is bounded by the caller-selected capacity. Once a
/// completed ID is evicted, status and cancellation lookups return NotFound.
/// Callers must still allocate fresh request IDs for generation attempts.
#[derive(Clone, Debug)]
pub struct Scheduler {
    shared: Arc<Shared>,
}

/// An immutable cancellation target for one observed execution attempt.
///
/// Pressure decisions can outlive their observation. This handle binds a
/// decision to both the request ID and its scheduler lease, so it cannot
/// interrupt a replacement request, even if a completed ID is later reused.
/// It owns neither the execution permit nor any native resources.
#[derive(Clone, Debug)]
pub struct ExecutionCancellationTarget {
    shared: Arc<Shared>,
    request_id: RequestId,
    lease_id: u64,
}

impl PartialEq for ExecutionCancellationTarget {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
            && self.request_id == other.request_id
            && self.lease_id == other.lease_id
    }
}

impl Eq for ExecutionCancellationTarget {}

impl ExecutionCancellationTarget {
    /// Request ID captured when this target was observed.
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// Cooperatively cancel this exact execution attempt without queueing.
    ///
    /// Accepted cancellation retains execution ownership until its worker
    /// finishes cleanup. Completion and cancellation use the same scheduler
    /// lock as ordinary cancellation. A stale target returns AlreadyFinished,
    /// including after its request ID has left the bounded finished history.
    /// Cancellation wakers run after releasing the scheduler lock.
    pub fn request_cancel(&self) -> Result<CancelOutcome, EngineError> {
        let source = {
            let mut state = self.shared.lock_state()?;
            let Some(active) = state.active.as_mut().filter(|active| {
                active.request_id == self.request_id && active.lease_id == self.lease_id
            }) else {
                return Ok(CancelOutcome::AlreadyFinished);
            };

            active.phase = ActivePhase::CancellationRequested;
            active.cancellation.clone()
        };

        source.request_cancel();
        Ok(CancelOutcome::Requested)
    }
}

#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    active: Option<ActiveRequest>,
    finished: VecDeque<RequestId>,
    finished_history_capacity: usize,
    next_lease_id: u64,
}

#[derive(Debug)]
struct ActiveRequest {
    request_id: RequestId,
    lease_id: u64,
    phase: ActivePhase,
    cancellation: CancellationSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActivePhase {
    Running,
    CancellationRequested,
}

/// Exclusive ownership of the scheduler's execution slot.
///
/// This type is intentionally not Clone.
///
/// Move it into the actual generation worker. Keep it alive while loading,
/// prefill, decoding, or an in-flight native operation uses the execution
/// slot. Complete it only after that work has stopped.
///
/// Dropping the permit releases the slot and requests cancellation as a
/// cleanup safeguard. That is not a substitute for stopping native work
/// before relinquishing ownership.
#[derive(Debug)]
#[must_use = "the execution worker must retain this permit until execution stops"]
pub struct ExecutionPermit {
    shared: Arc<Shared>,
    request_id: RequestId,
    lease_id: u64,
    cancellation: CancellationToken,
    released: bool,
}

impl Scheduler {
    /// Construct a scheduler with an explicitly chosen completed-history limit.
    ///
    /// The limit must be greater than zero. There is no implicit policy
    /// default for how many completed request IDs to retain.
    pub fn new(finished_history_capacity: usize) -> Result<Self, EngineError> {
        if finished_history_capacity == 0 {
            return Err(EngineError::invalid_request(
                "finished_history_capacity",
                "must be greater than zero",
            ));
        }

        let requested_bytes = finished_history_capacity
            .checked_mul(std::mem::size_of::<RequestId>())
            .ok_or_else(|| {
                EngineError::invalid_request(
                    "finished_history_capacity",
                    "history capacity arithmetic overflow",
                )
            })?;

        let mut finished = VecDeque::new();
        finished
            .try_reserve_exact(finished_history_capacity)
            .map_err(|_| EngineError::AllocationFailed {
                operation: "allocate scheduler request history",
                requested_bytes: u64::try_from(requested_bytes).ok(),
            })?;

        Ok(Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    active: None,
                    finished,
                    finished_history_capacity,
                    next_lease_id: 1,
                }),
            }),
        })
    }

    /// Attempt to acquire the execution slot without waiting or queueing.
    ///
    /// Returns:
    ///
    /// - InvalidRequest for a duplicate active or recently completed ID.
    /// - Busy when a different request owns the slot.
    /// - An exclusive permit when the slot is available.
    ///
    /// A request with accepted cancellation still owns the slot until its
    /// execution worker finishes or relinquishes its permit.
    pub fn try_acquire(&self, request_id: RequestId) -> Result<ExecutionPermit, EngineError> {
        let mut state = self.shared.lock_state()?;

        if let Some(active) = &state.active {
            if active.request_id == request_id {
                return Err(EngineError::invalid_request(
                    "request_id",
                    "request ID is already active",
                ));
            }

            return Err(EngineError::Busy {
                active_request_id: Some(active.request_id.as_str().to_owned()),
                message: "another request owns the execution slot".to_owned(),
            });
        }

        if state.finished.contains(&request_id) {
            return Err(EngineError::invalid_request(
                "request_id",
                "request ID was recently completed; use a fresh request ID",
            ));
        }

        let lease_id = state.next_lease_id;
        let next_lease_id = lease_id
            .checked_add(1)
            .ok_or_else(|| EngineError::Internal {
                message: "scheduler lease identifier space exhausted".to_owned(),
            })?;

        let cancellation = CancellationSource::new();
        let token = cancellation.token();

        state.next_lease_id = next_lease_id;
        state.active = Some(ActiveRequest {
            request_id: request_id.clone(),
            lease_id,
            phase: ActivePhase::Running,
            cancellation,
        });

        Ok(ExecutionPermit {
            shared: Arc::clone(&self.shared),
            request_id,
            lease_id,
            cancellation: token,
            released: false,
        })
    }

    /// Request cancellation of a known generation attempt.
    ///
    /// Accepted cancellation changes lifecycle state before waking execution.
    /// It does not release the execution slot.
    ///
    /// If terminal completion already committed, returns AlreadyFinished.
    /// Unknown IDs, including evicted completed IDs, return NotFound.
    ///
    /// Wakers are invoked after releasing the scheduler mutex.
    pub fn cancel(&self, request_id: &RequestId) -> Result<CancelOutcome, EngineError> {
        let source = {
            let mut state = self.shared.lock_state()?;

            if let Some(active) = state.active.as_mut() {
                if &active.request_id == request_id {
                    active.phase = ActivePhase::CancellationRequested;
                    Some(active.cancellation.clone())
                } else if state.finished.contains(request_id) {
                    return Ok(CancelOutcome::AlreadyFinished);
                } else {
                    return Err(request_not_found(request_id));
                }
            } else if state.finished.contains(request_id) {
                return Ok(CancelOutcome::AlreadyFinished);
            } else {
                return Err(request_not_found(request_id));
            }
        };

        if let Some(source) = source {
            source.request_cancel();
        }

        Ok(CancelOutcome::Requested)
    }

    /// Return the lifecycle state of an active or recently finished request.
    pub fn request_state(&self, request_id: &RequestId) -> Result<RequestState, EngineError> {
        let state = self.shared.lock_state()?;

        if let Some(active) = &state.active {
            if &active.request_id == request_id {
                return Ok(match active.phase {
                    ActivePhase::Running => RequestState::Running,
                    ActivePhase::CancellationRequested => RequestState::CancellationRequested,
                });
            }
        }

        if state.finished.contains(request_id) {
            return Ok(RequestState::Finished);
        }

        Err(request_not_found(request_id))
    }

    /// Return the current execution-slot owner, if any.
    pub fn active_request_id(&self) -> Result<Option<RequestId>, EngineError> {
        let state = self.shared.lock_state()?;

        Ok(state
            .active
            .as_ref()
            .map(|active| active.request_id.clone()))
    }

    /// Read owner and cancellation state together without entering the worker.
    ///
    /// A replacement request cannot be paired with its predecessor's state.
    /// This observation does not acquire or release execution ownership.
    pub fn snapshot(&self) -> Result<SchedulerSnapshot, EngineError> {
        let state = self.shared.lock_state()?;
        Ok(SchedulerSnapshot {
            active_request_id: state
                .active
                .as_ref()
                .map(|active| active.request_id.clone()),
            cancellation_requested: state
                .active
                .as_ref()
                .is_some_and(|active| active.phase == ActivePhase::CancellationRequested),
        })
    }

    /// Capture the exact current execution attempt for a delayed decision.
    ///
    /// Capture before observing pressure; applying the result must use this
    /// target rather than looking up whichever request happens to run later.
    /// Cancellation-requested work remains a valid, idempotent target until
    /// its worker relinquishes the execution permit.
    pub fn active_cancellation_target(
        &self,
    ) -> Result<Option<ExecutionCancellationTarget>, EngineError> {
        let state = self.shared.lock_state()?;
        Ok(state
            .active
            .as_ref()
            .map(|active| ExecutionCancellationTarget {
                shared: Arc::clone(&self.shared),
                request_id: active.request_id.clone(),
                lease_id: active.lease_id,
            }))
    }

    /// Whether a request currently owns the execution slot.
    ///
    /// A cancellation-requested execution still counts as busy.
    pub fn is_busy(&self) -> Result<bool, EngineError> {
        let state = self.shared.lock_state()?;
        Ok(state.active.is_some())
    }

    /// Return the number of completed IDs currently retained.
    pub fn retained_finished_count(&self) -> Result<usize, EngineError> {
        let state = self.shared.lock_state()?;
        Ok(state.finished.len())
    }
}

impl ExecutionPermit {
    /// Return the request owning this permit.
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// Obtain an observation token for actual execution and buffer waits.
    ///
    /// A token does not own the execution slot. The worker must retain
    /// this permit independently of any token clones.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Check cancellation at a safe interruption point.
    ///
    /// This does not release the slot or emit a terminal event.
    pub fn check_cancelled(&self) -> Result<(), EngineError> {
        self.cancellation.check_cancelled(&self.request_id)
    }

    /// Commit terminal completion after actual execution has stopped.
    ///
    /// Consumes the permit, releases the slot, and returns the single
    /// terminal event for the generation layer to deliver.
    ///
    /// Cancellation and completion are serialized by the scheduler mutex:
    ///
    /// - Cancellation accepted first produces a cancelled failure event.
    /// - Completion committed first causes later cancellation to return
    ///   AlreadyFinished while this request remains in history.
    ///
    /// Actual usage supplied by the worker is preserved in either case.
    /// Event delivery occurs outside this function and outside the
    /// scheduler mutex.
    pub fn finish(mut self, outcome: ExecutionOutcome) -> Result<GenerationEvent, EngineError> {
        let (cancelled, cancellation_source) = {
            let mut state = self.shared.lock_state()?;

            let active = state.active.as_ref().ok_or_else(|| EngineError::Internal {
                message: "execution permit has no active request".to_owned(),
            })?;

            if active.lease_id != self.lease_id || active.request_id != self.request_id {
                return Err(EngineError::Internal {
                    message: "execution permit does not own the active request".to_owned(),
                });
            }

            let active = state.active.take().ok_or_else(|| EngineError::Internal {
                message: "active request disappeared during completion".to_owned(),
            })?;

            let cancelled = active.phase == ActivePhase::CancellationRequested;

            state.record_finished(active.request_id);

            (cancelled, active.cancellation)
        };

        self.released = true;

        // Drop cancellation ownership outside the scheduler mutex.
        drop(cancellation_source);

        let event = match outcome {
            ExecutionOutcome::Completed {
                finish_reason,
                usage,
            } => {
                if cancelled {
                    GenerationEvent::RequestFailed {
                        error: EngineError::Cancelled {
                            request_id: self.request_id.as_str().to_owned(),
                        },
                        partial_usage: usage,
                    }
                } else {
                    GenerationEvent::GenerationCompleted {
                        finish_reason,
                        usage,
                    }
                }
            }

            ExecutionOutcome::Failed {
                error,
                partial_usage,
            } => {
                if cancelled {
                    GenerationEvent::RequestFailed {
                        error: EngineError::Cancelled {
                            request_id: self.request_id.as_str().to_owned(),
                        },
                        partial_usage,
                    }
                } else {
                    GenerationEvent::RequestFailed {
                        error,
                        partial_usage,
                    }
                }
            }
        };

        Ok(event)
    }
}

impl Drop for ExecutionPermit {
    fn drop(&mut self) {
        if self.released {
            return;
        }

        let cancellation_source = {
            // Fail closed on poison. A poisoned scheduler must not admit
            // another request based on potentially inconsistent state.
            let Ok(mut state) = self.shared.state.lock() else {
                return;
            };

            let owns_slot = state.active.as_ref().is_some_and(|active| {
                active.lease_id == self.lease_id && active.request_id == self.request_id
            });

            if !owns_slot {
                return;
            }

            let Some(active) = state.active.take() else {
                return;
            };

            state.record_finished(active.request_id);
            Some(active.cancellation)
        };

        // Cancellation may wake code that calls back into the scheduler.
        // Never invoke its wakers while holding the scheduler mutex.
        if let Some(source) = cancellation_source {
            source.request_cancel();
        }

        self.released = true;
    }
}

impl Shared {
    fn lock_state(&self) -> Result<MutexGuard<'_, State>, EngineError> {
        self.state.lock().map_err(|_| EngineError::Internal {
            message: "scheduler state mutex is poisoned".to_owned(),
        })
    }
}

impl State {
    fn record_finished(&mut self, request_id: RequestId) {
        if self.finished.len() >= self.finished_history_capacity {
            self.finished.pop_front();
        }

        self.finished.push_back(request_id);
    }
}

fn request_not_found(request_id: &RequestId) -> EngineError {
    EngineError::NotFound {
        resource: "request",
        id: request_id.as_str().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use std::{sync::Barrier, thread};

    fn id(value: &str) -> RequestId {
        RequestId::new(value).expect("valid request ID")
    }

    fn completed() -> ExecutionOutcome {
        ExecutionOutcome::Completed {
            finish_reason: FinishReason::OutputLimit,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 2,
            },
        }
    }

    #[test]
    fn only_one_request_owns_execution() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let first = scheduler.try_acquire(id("first")).expect("permit");

        assert!(scheduler.is_busy().expect("busy state"));

        let error = scheduler
            .try_acquire(id("second"))
            .expect_err("second request must not acquire the slot");

        assert_eq!(error.kind(), ErrorKind::Busy);
        assert_eq!(
            scheduler.active_request_id().expect("active request"),
            Some(id("first"))
        );

        first.finish(completed()).expect("finish");

        assert!(!scheduler.is_busy().expect("idle state"));
        let second = scheduler
            .try_acquire(id("second"))
            .expect("slot available after execution finishes");

        second.finish(completed()).expect("finish second");
    }

    #[test]
    fn duplicate_active_request_id_is_rejected() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let permit = scheduler.try_acquire(id("same")).expect("permit");

        let error = scheduler
            .try_acquire(id("same"))
            .expect_err("duplicate active ID");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);

        permit.finish(completed()).expect("finish");
    }

    #[test]
    fn cancellation_keeps_execution_slot_owned() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = id("cancel-me");
        let permit = scheduler.try_acquire(request.clone()).expect("permit");
        let token = permit.cancellation_token();

        assert_eq!(
            scheduler.cancel(&request).expect("cancel"),
            CancelOutcome::Requested
        );
        assert!(token.is_cancelled());

        assert_eq!(
            scheduler.request_state(&request).expect("state"),
            RequestState::CancellationRequested
        );
        assert!(scheduler.is_busy().expect("still busy"));

        let error = scheduler
            .try_acquire(id("other"))
            .expect_err("cancelled work still owns the slot");

        assert_eq!(error.kind(), ErrorKind::Busy);

        let terminal = permit.finish(completed()).expect("finish");

        match terminal {
            GenerationEvent::RequestFailed {
                error,
                partial_usage,
            } => {
                assert_eq!(error.kind(), ErrorKind::Cancelled);
                assert_eq!(partial_usage.output_tokens, 2);
            }
            other => panic!("expected cancelled terminal event, got {other:?}"),
        }

        assert!(!scheduler.is_busy().expect("released"));
    }

    #[test]
    fn cancellation_target_signals_without_releasing_execution() {
        let scheduler = Scheduler::new(2).expect("scheduler");
        assert!(
            scheduler
                .active_cancellation_target()
                .expect("idle")
                .is_none()
        );
        let permit = scheduler.try_acquire(id("target")).expect("permit");
        let token = permit.cancellation_token();
        let target = scheduler
            .active_cancellation_target()
            .expect("snapshot")
            .expect("active target");
        assert_eq!(target.request_id(), &id("target"));

        assert_eq!(
            target.request_cancel().expect("cancel"),
            CancelOutcome::Requested
        );
        assert_eq!(
            target.request_cancel().expect("repeat"),
            CancelOutcome::Requested
        );
        assert!(token.is_cancelled());
        assert_eq!(
            scheduler.request_state(target.request_id()).expect("state"),
            RequestState::CancellationRequested
        );
        assert_eq!(
            scheduler.active_cancellation_target().expect("same target"),
            Some(target.clone())
        );
        assert_eq!(
            scheduler
                .try_acquire(id("replacement"))
                .expect_err("still owned")
                .kind(),
            ErrorKind::Busy
        );

        let terminal = permit.finish(completed()).expect("finish after cleanup");
        assert!(matches!(
            terminal,
            GenerationEvent::RequestFailed {
                error: EngineError::Cancelled { .. },
                ..
            }
        ));
        assert_eq!(
            target.request_cancel().expect("stale"),
            CancelOutcome::AlreadyFinished
        );
        assert!(!scheduler.is_busy().expect("released"));
    }

    #[test]
    fn stale_cancellation_target_cannot_cancel_replacement_or_reused_id() {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let first = scheduler.try_acquire(id("original")).expect("first permit");
        let target = scheduler
            .active_cancellation_target()
            .expect("snapshot")
            .expect("target");
        first.finish(completed()).expect("finish first");

        let replacement = scheduler
            .try_acquire(id("replacement"))
            .expect("replacement");
        let replacement_token = replacement.cancellation_token();
        assert_eq!(
            target.request_cancel().expect("stale target"),
            CancelOutcome::AlreadyFinished
        );
        assert!(!replacement_token.is_cancelled());
        assert!(matches!(
            replacement.finish(completed()).expect("finish replacement"),
            GenerationEvent::GenerationCompleted { .. }
        ));

        // A bounded history eventually permits the same public ID again.
        let reused = scheduler.try_acquire(id("original")).expect("reused ID");
        let reused_token = reused.cancellation_token();
        let current = scheduler
            .active_cancellation_target()
            .expect("snapshot")
            .expect("target");
        assert_ne!(target, current);
        assert_eq!(target.request_id(), current.request_id());
        assert_eq!(
            target.request_cancel().expect("expired target"),
            CancelOutcome::AlreadyFinished
        );
        assert!(!reused_token.is_cancelled());
        assert_eq!(
            scheduler
                .request_state(current.request_id())
                .expect("running"),
            RequestState::Running
        );
        assert!(matches!(
            reused.finish(completed()).expect("finish reused"),
            GenerationEvent::GenerationCompleted { .. }
        ));
    }

    #[test]
    fn cancellation_targets_preserve_scheduler_identity() {
        let first = Scheduler::new(1).expect("first scheduler");
        let second = Scheduler::new(1).expect("second scheduler");
        let first_permit = first.try_acquire(id("same")).expect("first permit");
        let second_permit = second.try_acquire(id("same")).expect("second permit");
        let first_target = first
            .active_cancellation_target()
            .expect("snapshot")
            .expect("target");
        let second_target = second
            .active_cancellation_target()
            .expect("snapshot")
            .expect("target");

        assert_ne!(first_target, second_target);
        first_target.request_cancel().expect("cancel first");
        assert!(first_permit.cancellation_token().is_cancelled());
        assert!(!second_permit.cancellation_token().is_cancelled());
        drop(first_target);
        drop(second_target);
        assert!(first.is_busy().expect("target does not own permit"));
        assert!(second.is_busy().expect("target does not own permit"));
        first_permit.finish(completed()).expect("finish first");
        assert!(matches!(
            second_permit.finish(completed()).expect("finish second"),
            GenerationEvent::GenerationCompleted { .. }
        ));
    }

    #[test]
    fn cancellation_target_and_completion_have_one_serialized_winner() {
        for _ in 0..32 {
            let scheduler = Scheduler::new(1).expect("scheduler");
            let permit = scheduler.try_acquire(id("race-target")).expect("permit");
            let target = scheduler
                .active_cancellation_target()
                .expect("snapshot")
                .expect("target");
            let barrier = Arc::new(Barrier::new(2));
            let cancel_barrier = Arc::clone(&barrier);
            let cancel_worker = thread::spawn(move || {
                cancel_barrier.wait();
                target.request_cancel().expect("cancel")
            });
            let finish_worker = thread::spawn(move || {
                barrier.wait();
                permit.finish(completed()).expect("finish")
            });
            let cancellation = cancel_worker.join().expect("cancel worker");
            let terminal = finish_worker.join().expect("finish worker");
            match cancellation {
                CancelOutcome::Requested => assert!(matches!(
                    terminal,
                    GenerationEvent::RequestFailed {
                        error: EngineError::Cancelled { .. },
                        ..
                    }
                )),
                CancelOutcome::AlreadyFinished => assert!(matches!(
                    terminal,
                    GenerationEvent::GenerationCompleted { .. }
                )),
            }
            assert!(!scheduler.is_busy().expect("released"));
        }
    }

    #[test]
    fn completion_before_cancellation_is_already_finished() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = id("done");

        let terminal = scheduler
            .try_acquire(request.clone())
            .expect("permit")
            .finish(completed())
            .expect("finish");

        assert!(matches!(
            terminal,
            GenerationEvent::GenerationCompleted { .. }
        ));

        assert_eq!(
            scheduler.cancel(&request).expect("cancel completed"),
            CancelOutcome::AlreadyFinished
        );
    }

    #[test]
    fn repeated_active_cancellation_is_safe() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = id("repeat-cancel");
        let permit = scheduler.try_acquire(request.clone()).expect("permit");

        assert_eq!(
            scheduler.cancel(&request).expect("first"),
            CancelOutcome::Requested
        );
        assert_eq!(
            scheduler.cancel(&request).expect("second"),
            CancelOutcome::Requested
        );

        permit.finish(completed()).expect("finish");
    }

    #[test]
    fn unknown_request_is_not_found() {
        let scheduler = Scheduler::new(8).expect("scheduler");

        let error = scheduler
            .cancel(&id("unknown"))
            .expect_err("unknown request");

        assert_eq!(error.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn abandoned_permit_releases_slot_and_signals_cancellation() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = id("abandoned");
        let permit = scheduler.try_acquire(request.clone()).expect("permit");
        let token = permit.cancellation_token();

        drop(permit);

        assert!(token.is_cancelled());
        assert!(!scheduler.is_busy().expect("released"));
        assert_eq!(
            scheduler.request_state(&request).expect("history"),
            RequestState::Finished
        );
    }

    #[test]
    fn completed_history_is_bounded() {
        let scheduler = Scheduler::new(2).expect("scheduler");

        for name in ["one", "two", "three"] {
            scheduler
                .try_acquire(id(name))
                .expect("permit")
                .finish(completed())
                .expect("finish");
        }

        assert_eq!(
            scheduler.retained_finished_count().expect("history size"),
            2
        );

        let error = scheduler
            .cancel(&id("one"))
            .expect_err("oldest completed ID was evicted");

        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            scheduler.cancel(&id("three")).expect("recent history"),
            CancelOutcome::AlreadyFinished
        );
    }

    #[test]
    fn recently_finished_id_cannot_be_reused() {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = id("finished");

        scheduler
            .try_acquire(request.clone())
            .expect("permit")
            .finish(completed())
            .expect("finish");

        let error = scheduler
            .try_acquire(request)
            .expect_err("request attempt IDs must be fresh");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    }

    #[test]
    fn execution_failure_is_preserved_without_cancellation() {
        let scheduler = Scheduler::new(8).expect("scheduler");

        let terminal = scheduler
            .try_acquire(id("execution-failure"))
            .expect("permit")
            .finish(ExecutionOutcome::Failed {
                error: EngineError::ExecutionFailed {
                    stage: "decode",
                    message: "non-finite logits".to_owned(),
                },
                partial_usage: Usage {
                    input_tokens: 3,
                    output_tokens: 1,
                },
            })
            .expect("finish");

        match terminal {
            GenerationEvent::RequestFailed {
                error,
                partial_usage,
            } => {
                assert_eq!(error.kind(), ErrorKind::ExecutionFailed);
                assert_eq!(partial_usage.output_tokens, 1);
            }
            other => panic!("expected execution failure, got {other:?}"),
        }
    }

    #[test]
    fn cancellation_and_completion_have_one_serialized_winner() {
        for _ in 0..32 {
            let scheduler = Scheduler::new(2).expect("scheduler");
            let request = id("race");
            let permit = scheduler.try_acquire(request.clone()).expect("permit");

            let barrier = Arc::new(Barrier::new(2));

            let cancel_scheduler = scheduler.clone();
            let cancel_request = request.clone();
            let cancel_barrier = barrier.clone();

            let cancel_worker = thread::spawn(move || {
                cancel_barrier.wait();
                cancel_scheduler.cancel(&cancel_request).expect("cancel")
            });

            let finish_worker = thread::spawn(move || {
                barrier.wait();
                permit.finish(completed()).expect("finish")
            });

            let cancel_outcome = cancel_worker.join().expect("cancel worker");
            let terminal = finish_worker.join().expect("finish worker");

            match cancel_outcome {
                CancelOutcome::Requested => match terminal {
                    GenerationEvent::RequestFailed { error, .. } => {
                        assert_eq!(error.kind(), ErrorKind::Cancelled);
                    }
                    other => {
                        panic!("accepted cancellation must win: {other:?}");
                    }
                },

                CancelOutcome::AlreadyFinished => {
                    assert!(matches!(
                        terminal,
                        GenerationEvent::GenerationCompleted { .. }
                    ));
                }
            }

            assert!(!scheduler.is_busy().expect("released"));
        }
    }

    #[test]
    fn zero_history_capacity_is_rejected() {
        let error = Scheduler::new(0).expect_err("invalid history capacity");
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    }
}
