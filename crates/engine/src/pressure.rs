//! Bounded unified-memory pressure decisions.
//!
//! This module performs no native operations and owns no reservations,
//! models, caches, or leases.
//!
//! Admission accounting and allocator observations are checked separately:
//!
//! - Admission: additional reservation bytes must fit the ledger.
//! - Allocator: active + cached + incremental MLX bytes are compared with
//!   an explicitly selected pressure ceiling.
//!
//! Reservations are NOT added to active allocator memory. Those systems
//! can describe the same allocations.
//!
//! Allocator projection is a conservative pressure signal: new allocations
//! may reuse cached buffers, but the policy can recommend clearing cached
//! storage before allowing further work.
//!
//! A pressure ceiling is application policy, not an assertion that MLX's
//! configured memory guideline is a hard allocation or OS-memory limit.
//!
//! Reclamation phases:
//!
//! 0. Before releasing idle state.
//! 1. After releasing eligible idle state.
//! 2. After unloading an idle model.
//!
//! Allocator cache may be cleared at most once per phase. State release
//! and model unload may each be requested at most once. Thus there are at
//! most five reclamation actions, even if observations do not improve.
//!
//! Every action must be acknowledged before another decision is requested.
//! Failed actions reject the attempt instead of escalating blindly.
//!
//! After an eviction, the runtime must replan the SAME logical request.
//! For example, unloading weights previously expected to be reused changes
//! the request's incremental loading requirement.
//!
//! Proceed is advisory. The runtime must still acquire real admission
//! atomically and retain scheduler/native ownership correctly.

use crate::admission::AdmissionSnapshot;
use std::fmt;

/// Maximum possible actions in one pressure attempt.
pub const MAX_RECLAMATION_ACTIONS: u8 = 5;

/// Native ownership state relevant to destructive reclamation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionState {
    /// No active native execution or pending startup.
    Idle,

    /// Native execution still owns resources.
    Active,

    /// A startup handoff owns pending native resources.
    PendingStartup,
}

/// Conservative allocator observations.
///
/// These values may be sequential or stale observations. They are not
/// process RSS or engine reservation accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocatorObservation {
    /// Actively used MLX allocator bytes.
    pub active_bytes: u64,

    /// Free MLX allocator buffers retained for reuse.
    pub cached_bytes: u64,
}

impl AllocatorObservation {
    /// Convert an existing native snapshot without performing native calls.
    ///
    /// Peak memory and the native configured guideline are not used to
    /// silently select this policy's pressure ceiling.
    pub fn from_mlx(snapshot: mlx_ffi::memory::MemorySnapshot) -> PressureResult<Self> {
        Ok(Self {
            active_bytes: u64::try_from(snapshot.active_bytes)
                .map_err(|_| PressureError::ArithmeticOverflow)?,
            cached_bytes: u64::try_from(snapshot.cache_bytes)
                .map_err(|_| PressureError::ArithmeticOverflow)?,
        })
    }
}

/// Current incremental requirement for the same logical request.
///
/// Recompute this after releasing resources. Do not keep assuming resident
/// weights/cache are reusable after an action removed them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestDemand {
    /// Additional bytes to acquire from the admission ledger.
    ///
    /// Excludes reservations already retained and reusable by this request.
    pub additional_reservation_bytes: u64,

    /// Conservative additional MLX allocation requirement at the selected
    /// execution/loading peak, excluding already-active reused buffers.
    ///
    /// Do not substitute the entire reservation total: that total can
    /// contain host output, headroom, and already-active native storage.
    pub additional_mlx_bytes: u64,
}

/// Reclamation eligibility observed by the native resource owner.
///
/// The runtime must revalidate eligibility immediately before destruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReclaimableResources {
    /// Number of idle session states considered by this attempt.
    pub idle_session_states: usize,

    /// Number of those states protected by live leases.
    pub leased_session_states: usize,

    /// Whether eligible state/KV can be reclaimed while preserving the
    /// loaded model.
    ///
    /// If the runtime cannot provide that operation, set false rather
    /// than treating whole-model destruction as a state-only action.
    pub can_release_state_without_model: bool,

    /// Whether an idle model is loaded.
    pub idle_model_loaded: bool,

    /// Number of independently owned idle models currently eligible for
    /// whole-model eviction. A lease on another instance does not protect
    /// these candidates. The runtime revalidates each selected victim.
    pub eligible_idle_models: usize,

    /// Whether explicit model pins or canonical leases protect any idle
    /// model. Protected and eligible instances may coexist in a pool.
    pub idle_model_has_live_leases: bool,
}

impl ReclaimableResources {
    fn validate(self) -> PressureResult<()> {
        if self.leased_session_states > self.idle_session_states {
            return Err(PressureError::InvalidObservation(
                "leased state count exceeds idle state count",
            ));
        }
        if !self.idle_model_loaded && self.eligible_idle_models != 0 {
            return Err(PressureError::InvalidObservation(
                "eligible model count is nonzero without a loaded model",
            ));
        }
        Ok(())
    }

    fn can_release_state(self) -> bool {
        self.can_release_state_without_model
            && self.idle_session_states > self.leased_session_states
    }

    fn can_unload_model(self) -> bool {
        self.idle_model_loaded && self.eligible_idle_models != 0
    }

    fn protection_blocks_reclamation(self, options: PressureOptions) -> bool {
        // An eligible candidate remaining after its bounded stage is an
        // exhausted-action case. An unrelated pin must not be blamed for
        // that bound, or for an operation the policy does not permit.
        if (options.allow_idle_state_release && self.can_release_state())
            || (options.allow_idle_model_unload && self.can_unload_model())
        {
            return false;
        }
        (options.allow_idle_state_release
            && self.can_release_state_without_model
            && self.leased_session_states != 0)
            || (options.allow_idle_model_unload && self.idle_model_has_live_leases)
    }
}

/// Inputs for one decision.
///
/// Observations must be refreshed after each completed action. In
/// particular, reservation availability changes only when actual owning
/// guards are released—not because allocator cache bytes decreased.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PressureObservation {
    /// Snapshot of the real shared admission ledger.
    pub admission: AdmissionSnapshot,

    /// Allocator observations, if available.
    pub allocator: Option<AllocatorObservation>,

    /// Current replanned incremental requirement.
    pub demand: RequestDemand,

    /// Active/pending native ownership.
    pub execution: ExecutionState,

    /// Current resource eligibility.
    pub reclaimable: ReclaimableResources,
}

/// Explicit configuration for pressure decisions.
///
/// No Default silently chooses a memory ceiling or eviction permission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PressureOptions {
    /// Application-selected allocator pressure ceiling.
    ///
    /// None explicitly selects admission-only decisions. Some requires
    /// allocator observations; absence is rejected rather than treated
    /// as zero active memory.
    pub allocator_ceiling_bytes: Option<u64>,

    /// Permit clearing free allocator cache.
    pub allow_allocator_cache_clear: bool,

    /// Permit one bounded pass releasing eligible idle state/KV while
    /// preserving the loaded model.
    pub allow_idle_state_release: bool,

    /// Permit unloading an unleased idle model as the last destructive
    /// reclamation step.
    pub allow_idle_model_unload: bool,

    /// Maximum actions for one attempt, between zero and five inclusive.
    ///
    /// Zero permits Proceed when already fitting but no reclamation.
    pub max_reclamation_actions: u8,
}

/// Validated pressure policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PressurePolicy {
    options: PressureOptions,
}

impl PressurePolicy {
    /// Validate an explicit policy.
    pub fn new(options: PressureOptions) -> PressureResult<Self> {
        if options.allocator_ceiling_bytes == Some(0) {
            return Err(PressureError::InvalidPolicy(
                "allocator pressure ceiling must be positive",
            ));
        }
        if options.max_reclamation_actions > MAX_RECLAMATION_ACTIONS {
            return Err(PressureError::InvalidPolicy(
                "at most five reclamation actions are supported",
            ));
        }
        Ok(Self { options })
    }

    /// Return this policy's explicit configuration.
    pub fn options(&self) -> PressureOptions {
        self.options
    }

    /// Begin one bounded attempt for one logical request.
    ///
    /// Do not create a fresh attempt merely to bypass exhausted retries.
    pub fn begin(&self) -> PressureAttempt {
        PressureAttempt {
            policy: *self,
            admission_budget: None,
            phase: 0,
            cleared_phases: 0,
            state_release_attempted: false,
            model_unload_attempted: false,
            actions_issued: 0,
            in_flight: None,
            completed: None,
            failed_action: None,
            last_action_made_progress: None,
            last_assessment: None,
            finished: false,
        }
    }
}

/// A native reclamation operation requested by the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReclaimAction {
    /// Clear unused allocator buffers without modifying reservations.
    ClearAllocatorCache,

    /// Perform one bounded pass over eligible unleased idle state/KV.
    ///
    /// Preserve the loaded model. This is not necessarily equivalent to
    /// calling a public release method that destroys an entire entry
    /// containing both model and cache ownership.
    ReleaseIdleSessionState,

    /// Destroy an unleased idle model after cheaper options were tried
    /// or were unavailable.
    UnloadIdleModel,
}

/// Result reported after actually executing a recommended action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionOutcome {
    /// The operation completed. Supply fresh observations next.
    Completed,

    /// The operation failed. The next decision rejects the attempt.
    ///
    /// The runtime should preserve and return the actual underlying error.
    Failed,
}

/// Why an admission/reclamation attempt cannot proceed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Native execution or pending startup prevents reclamation/admission.
    NativeOwnerBusy,

    /// Allocator checking was required but no observation was available.
    AllocatorObservationMissing,

    /// The request alone exceeds the complete admission budget.
    RequestExceedsAdmissionBudget,

    /// The incremental MLX requirement alone exceeds the pressure ceiling.
    RequestExceedsAllocatorCeiling,

    /// The configured action budget has been consumed.
    ActionLimitReached,

    /// Explicit model pins or canonical leases protect resources needed
    /// for further reclamation.
    ProtectedResources,

    /// No permitted eligible action is available.
    NoReclaimableResources,

    /// All available reclamation stages have been exhausted.
    ReclamationExhausted,

    /// No shortfall improvement was observed after the last action and
    /// no further eligible stage remains.
    NoProgress,

    /// A native reclamation operation failed.
    ActionFailed(ReclaimAction),
}

/// Next runtime operation.
///
/// Proceed does not acquire admission. Action variants do not perform
/// destruction; the runtime must execute and acknowledge them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressureDecision {
    /// Both required checks fit the supplied observations.
    Proceed,

    /// Clear free allocator cache.
    ClearAllocatorCache,

    /// Reclaim eligible idle state without unloading its model.
    ReleaseIdleSessionState,

    /// Unload an eligible idle model.
    UnloadIdleModel,

    /// Stop the attempt with an explicit reason.
    Reject(RejectReason),
}

/// Independent admission and allocator assessment.
///
/// Reservation bytes are deliberately not included in projected_mlx_bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PressureAssessment {
    /// Additional ledger bytes requested.
    pub reservation_required_bytes: u64,

    /// Observed unreserved ledger bytes.
    pub reservation_available_bytes: u64,

    /// Additional ledger bytes that do not currently fit.
    pub reservation_deficit_bytes: u64,

    /// Whether allocator observations are required by policy.
    pub allocator_check_required: bool,

    /// Active + cached + additional MLX bytes, if allocator checking is
    /// enabled and observations were supplied.
    pub projected_mlx_bytes: Option<u64>,

    /// Excess over the application pressure ceiling, when observable.
    pub allocator_deficit_bytes: Option<u64>,
}

impl PressureAssessment {
    /// Whether all policy-required checks fit.
    ///
    /// A required but missing allocator assessment is not a fit.
    pub fn fits(&self) -> bool {
        self.reservation_deficit_bytes == 0
            && (!self.allocator_check_required || self.allocator_deficit_bytes == Some(0))
    }
}

/// Invalid policy inputs or state-machine use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressureError {
    /// Invalid explicit policy.
    InvalidPolicy(&'static str),

    /// Contradictory input metadata.
    InvalidObservation(&'static str),

    /// Checked byte arithmetic overflowed.
    ArithmeticOverflow,

    /// Admission controller/budget changed during one logical attempt.
    AdmissionBudgetChanged,

    /// An issued action has not been acknowledged.
    ActionAwaitingCompletion,

    /// No action is awaiting acknowledgement.
    NoActionAwaitingCompletion,

    /// Acknowledgement does not match the issued action.
    WrongActionAcknowledgement,

    /// A terminal decision already ended this attempt.
    AttemptFinished,
}

impl fmt::Display for PressureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy(message) => {
                write!(formatter, "invalid pressure policy: {message}")
            }
            Self::InvalidObservation(message) => {
                write!(formatter, "invalid pressure observation: {message}")
            }
            Self::ArithmeticOverflow => {
                write!(formatter, "pressure byte arithmetic overflow")
            }
            Self::AdmissionBudgetChanged => {
                write!(
                    formatter,
                    "admission budget changed during pressure attempt"
                )
            }
            Self::ActionAwaitingCompletion => {
                write!(formatter, "pressure action must be acknowledged first")
            }
            Self::NoActionAwaitingCompletion => {
                write!(formatter, "no pressure action awaits acknowledgement")
            }
            Self::WrongActionAcknowledgement => {
                write!(formatter, "pressure acknowledgement names the wrong action")
            }
            Self::AttemptFinished => {
                write!(
                    formatter,
                    "pressure attempt already reached a terminal decision"
                )
            }
        }
    }
}

impl std::error::Error for PressureError {}

/// Result of pressure assessment or state-machine use.
pub type PressureResult<T> = Result<T, PressureError>;

#[derive(Clone, Copy, Debug)]
struct IssuedAction {
    action: ReclaimAction,
    before: PressureAssessment,
}

/// One bounded decision sequence for one logical request.
///
/// Intentionally not Clone. Runtime ownership and atomic admission remain
/// outside this pure state machine.
#[derive(Debug)]
pub struct PressureAttempt {
    policy: PressurePolicy,
    admission_budget: Option<u64>,
    phase: u8,
    cleared_phases: u8,
    state_release_attempted: bool,
    model_unload_attempted: bool,
    actions_issued: u8,
    in_flight: Option<IssuedAction>,
    completed: Option<IssuedAction>,
    failed_action: Option<ReclaimAction>,
    last_action_made_progress: Option<bool>,
    last_assessment: Option<PressureAssessment>,
    finished: bool,
}

impl PressureAttempt {
    /// Number of native actions issued so far.
    pub fn actions_issued(&self) -> u8 {
        self.actions_issued
    }

    /// Most recent independent budget assessment.
    pub fn last_assessment(&self) -> Option<PressureAssessment> {
        self.last_assessment
    }

    /// Whether the last acknowledged action reduced a relevant shortfall.
    ///
    /// None means no completed action has yet been compared with a fresh
    /// observation. A false result does not skip cheaper remaining stages.
    pub fn last_action_made_progress(&self) -> Option<bool> {
        self.last_action_made_progress
    }

    /// Whether a Proceed or Reject decision ended the attempt.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Evaluate fresh observations and issue at most one action.
    ///
    /// The runtime must replan incremental requirements after eviction,
    /// revalidate leases before destruction, and acknowledge every action.
    pub fn evaluate(
        &mut self,
        observation: PressureObservation,
    ) -> PressureResult<PressureDecision> {
        if self.finished {
            return Err(PressureError::AttemptFinished);
        }
        if self.in_flight.is_some() {
            return Err(PressureError::ActionAwaitingCompletion);
        }

        observation.reclaimable.validate()?;

        let budget = observation.admission.budget_bytes();
        match self.admission_budget {
            Some(previous) if previous != budget => {
                return Err(PressureError::AdmissionBudgetChanged);
            }
            None => self.admission_budget = Some(budget),
            _ => {}
        }

        if observation.execution != ExecutionState::Idle {
            return Ok(self.reject(RejectReason::NativeOwnerBusy));
        }

        if let Some(action) = self.failed_action {
            return Ok(self.reject(RejectReason::ActionFailed(action)));
        }

        if observation.demand.additional_reservation_bytes > budget {
            return Ok(self.reject(RejectReason::RequestExceedsAdmissionBudget));
        }

        let options = self.policy.options;

        if let Some(ceiling) = options.allocator_ceiling_bytes
            && observation.demand.additional_mlx_bytes > ceiling
        {
            return Ok(self.reject(RejectReason::RequestExceedsAllocatorCeiling));
        }

        let assessment = assess(options, observation)?;
        self.last_assessment = Some(assessment);

        if let Some(completed) = self.completed.take() {
            self.last_action_made_progress = Some(
                assessment.reservation_deficit_bytes < completed.before.reservation_deficit_bytes
                    || match (
                        assessment.allocator_deficit_bytes,
                        completed.before.allocator_deficit_bytes,
                    ) {
                        (Some(after), Some(before)) => after < before,
                        _ => false,
                    },
            );
        }

        if assessment.allocator_check_required && assessment.allocator_deficit_bytes.is_none() {
            return Ok(self.reject(RejectReason::AllocatorObservationMissing));
        }

        if assessment.fits() {
            self.finished = true;
            return Ok(PressureDecision::Proceed);
        }

        if self.actions_issued >= options.max_reclamation_actions {
            return Ok(self.reject(RejectReason::ActionLimitReached));
        }

        let phase_bit = 1_u8 << self.phase;
        let allocator_pressure = assessment
            .allocator_deficit_bytes
            .is_some_and(|bytes| bytes > 0);
        let has_cached_buffers = observation
            .allocator
            .is_some_and(|allocator| allocator.cached_bytes > 0);

        // Clearing free cache cannot fix ledger-only pressure.
        if options.allow_allocator_cache_clear
            && allocator_pressure
            && has_cached_buffers
            && self.cleared_phases & phase_bit == 0
        {
            self.cleared_phases |= phase_bit;
            return Ok(self.issue(ReclaimAction::ClearAllocatorCache, assessment));
        }

        if options.allow_idle_state_release
            && !self.state_release_attempted
            && !self.model_unload_attempted
            && observation.reclaimable.can_release_state()
        {
            self.state_release_attempted = true;
            return Ok(self.issue(ReclaimAction::ReleaseIdleSessionState, assessment));
        }

        if options.allow_idle_model_unload
            && !self.model_unload_attempted
            && observation.reclaimable.can_unload_model()
        {
            self.model_unload_attempted = true;
            return Ok(self.issue(ReclaimAction::UnloadIdleModel, assessment));
        }

        let reason = if observation
            .reclaimable
            .protection_blocks_reclamation(options)
        {
            RejectReason::ProtectedResources
        } else if self.last_action_made_progress == Some(false) {
            RejectReason::NoProgress
        } else if self.actions_issued > 0 {
            RejectReason::ReclamationExhausted
        } else {
            RejectReason::NoReclaimableResources
        };

        Ok(self.reject(reason))
    }

    /// Acknowledge the actual outcome of the last issued action.
    ///
    /// Completed means execute the next evaluation with fresh observations,
    /// not that any particular number of bytes was freed.
    pub fn complete_action(
        &mut self,
        action: ReclaimAction,
        outcome: ActionOutcome,
    ) -> PressureResult<()> {
        if self.finished {
            return Err(PressureError::AttemptFinished);
        }

        let issued = self
            .in_flight
            .ok_or(PressureError::NoActionAwaitingCompletion)?;

        if issued.action != action {
            return Err(PressureError::WrongActionAcknowledgement);
        }

        self.in_flight = None;
        self.last_action_made_progress = None;

        match outcome {
            ActionOutcome::Completed => {
                self.completed = Some(issued);
                match action {
                    ReclaimAction::ClearAllocatorCache => {}
                    ReclaimAction::ReleaseIdleSessionState => {
                        self.phase = 1;
                    }
                    ReclaimAction::UnloadIdleModel => {
                        self.phase = 2;
                    }
                }
            }
            ActionOutcome::Failed => {
                self.completed = None;
                self.failed_action = Some(action);
            }
        }

        Ok(())
    }

    fn issue(&mut self, action: ReclaimAction, before: PressureAssessment) -> PressureDecision {
        self.actions_issued += 1;
        self.in_flight = Some(IssuedAction { action, before });

        match action {
            ReclaimAction::ClearAllocatorCache => PressureDecision::ClearAllocatorCache,
            ReclaimAction::ReleaseIdleSessionState => PressureDecision::ReleaseIdleSessionState,
            ReclaimAction::UnloadIdleModel => PressureDecision::UnloadIdleModel,
        }
    }

    fn reject(&mut self, reason: RejectReason) -> PressureDecision {
        self.finished = true;
        PressureDecision::Reject(reason)
    }
}

fn assess(
    options: PressureOptions,
    observation: PressureObservation,
) -> PressureResult<PressureAssessment> {
    let required = observation.demand.additional_reservation_bytes;
    let available = observation.admission.available_bytes();

    let mut assessment = PressureAssessment {
        reservation_required_bytes: required,
        reservation_available_bytes: available,
        reservation_deficit_bytes: required.saturating_sub(available),
        allocator_check_required: options.allocator_ceiling_bytes.is_some(),
        projected_mlx_bytes: None,
        allocator_deficit_bytes: None,
    };

    if let (Some(ceiling), Some(allocator)) =
        (options.allocator_ceiling_bytes, observation.allocator)
    {
        let projected = allocator
            .active_bytes
            .checked_add(allocator.cached_bytes)
            .and_then(|bytes| bytes.checked_add(observation.demand.additional_mlx_bytes))
            .ok_or(PressureError::ArithmeticOverflow)?;

        assessment.projected_mlx_bytes = Some(projected);
        assessment.allocator_deficit_bytes = Some(projected.saturating_sub(ceiling));
    }

    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::AdmissionController;

    fn policy() -> PressurePolicy {
        PressurePolicy::new(PressureOptions {
            allocator_ceiling_bytes: Some(100),
            allow_allocator_cache_clear: true,
            allow_idle_state_release: true,
            allow_idle_model_unload: true,
            max_reclamation_actions: 5,
        })
        .expect("policy")
    }

    fn ledger(budget: u64, reserved: u64) -> AdmissionSnapshot {
        let admission = AdmissionController::new(budget).expect("controller");
        let _reservation = if reserved == 0 {
            None
        } else {
            Some(
                admission
                    .reserve("snapshot fixture", reserved)
                    .expect("guard"),
            )
        };
        admission
            .snapshot()
            .expect("consistent historical snapshot")
    }

    fn resources() -> ReclaimableResources {
        ReclaimableResources {
            idle_session_states: 1,
            leased_session_states: 0,
            can_release_state_without_model: true,
            idle_model_loaded: true,
            eligible_idle_models: 1,
            idle_model_has_live_leases: false,
        }
    }

    fn observation(reserved: u64, active: u64, cached: u64) -> PressureObservation {
        PressureObservation {
            admission: ledger(100, reserved),
            allocator: Some(AllocatorObservation {
                active_bytes: active,
                cached_bytes: cached,
            }),
            demand: RequestDemand {
                additional_reservation_bytes: 20,
                additional_mlx_bytes: 20,
            },
            execution: ExecutionState::Idle,
            reclaimable: resources(),
        }
    }

    fn acknowledge(attempt: &mut PressureAttempt, action: ReclaimAction) {
        attempt
            .complete_action(action, ActionOutcome::Completed)
            .expect("acknowledgement");
    }

    #[test]
    fn fitting_checks_proceed_without_double_counting_reservations() {
        let mut attempt = policy().begin();
        let observed = observation(70, 70, 0);

        assert_eq!(
            attempt.evaluate(observed).expect("decision"),
            PressureDecision::Proceed,
        );
        assert_eq!(
            attempt
                .last_assessment()
                .expect("assessment")
                .projected_mlx_bytes,
            Some(90),
        );
        assert_eq!(attempt.actions_issued(), 0);
    }

    #[test]
    fn cache_only_pressure_clears_before_destroying_live_state() {
        let mut attempt = policy().begin();

        assert_eq!(
            attempt.evaluate(observation(20, 60, 30)).expect("decision"),
            PressureDecision::ClearAllocatorCache,
        );

        acknowledge(&mut attempt, ReclaimAction::ClearAllocatorCache);

        assert_eq!(
            attempt.evaluate(observation(20, 60, 0)).expect("decision"),
            PressureDecision::Proceed,
        );
        assert_eq!(attempt.last_action_made_progress(), Some(true));
        assert_eq!(attempt.actions_issued(), 1);
    }

    #[test]
    fn ledger_only_pressure_does_not_clear_irrelevant_allocator_cache() {
        let mut attempt = policy().begin();

        assert_eq!(
            attempt.evaluate(observation(95, 10, 10)).expect("decision"),
            PressureDecision::ReleaseIdleSessionState,
        );

        acknowledge(&mut attempt, ReclaimAction::ReleaseIdleSessionState);

        assert_eq!(
            attempt.evaluate(observation(60, 5, 15)).expect("decision"),
            PressureDecision::Proceed,
        );
    }

    #[test]
    fn newly_cached_buffers_can_be_cleared_after_idle_state_release() {
        let mut attempt = policy().begin();

        assert_eq!(
            attempt.evaluate(observation(95, 60, 30)).expect("initial"),
            PressureDecision::ClearAllocatorCache,
        );
        acknowledge(&mut attempt, ReclaimAction::ClearAllocatorCache);

        assert_eq!(
            attempt
                .evaluate(observation(95, 60, 0))
                .expect("ledger pressure"),
            PressureDecision::ReleaseIdleSessionState,
        );
        acknowledge(&mut attempt, ReclaimAction::ReleaseIdleSessionState);

        let mut after_release = observation(80, 50, 40);
        after_release.reclaimable.idle_session_states = 0;

        assert_eq!(
            attempt.evaluate(after_release).expect("new cached buffers"),
            PressureDecision::ClearAllocatorCache,
        );
        acknowledge(&mut attempt, ReclaimAction::ClearAllocatorCache);

        after_release
            .allocator
            .as_mut()
            .expect("allocator")
            .cached_bytes = 0;

        assert_eq!(
            attempt.evaluate(after_release).expect("fit"),
            PressureDecision::Proceed,
        );
        assert_eq!(attempt.actions_issued(), 3);
    }

    #[test]
    fn model_unload_follows_the_state_release_stage() {
        let mut attempt = policy().begin();

        assert_eq!(
            attempt
                .evaluate(observation(95, 10, 0))
                .expect("state action"),
            PressureDecision::ReleaseIdleSessionState,
        );
        acknowledge(&mut attempt, ReclaimAction::ReleaseIdleSessionState);

        let mut after_state = observation(90, 5, 0);
        after_state.reclaimable.idle_session_states = 0;

        assert_eq!(
            attempt.evaluate(after_state).expect("model action"),
            PressureDecision::UnloadIdleModel,
        );
    }

    #[test]
    fn model_eviction_requires_replanning_loading_requirements() {
        let mut options = policy().options();
        options.allow_idle_state_release = false;
        let mut attempt = PressurePolicy::new(options).expect("policy").begin();

        assert_eq!(
            attempt.evaluate(observation(95, 10, 0)).expect("unload"),
            PressureDecision::UnloadIdleModel,
        );
        acknowledge(&mut attempt, ReclaimAction::UnloadIdleModel);

        let mut replanned = observation(0, 0, 0);
        replanned.reclaimable.idle_model_loaded = false;
        replanned.reclaimable.eligible_idle_models = 0;
        replanned.reclaimable.idle_session_states = 0;
        replanned.demand.additional_reservation_bytes = 110;

        assert_eq!(
            attempt.evaluate(replanned).expect("replanned rejection"),
            PressureDecision::Reject(RejectReason::RequestExceedsAdmissionBudget),
        );
    }

    #[test]
    fn active_and_pending_owners_prevent_actions_even_when_memory_fits() {
        for execution in [ExecutionState::Active, ExecutionState::PendingStartup] {
            let mut attempt = policy().begin();
            let mut observed = observation(0, 0, 0);
            observed.execution = execution;

            assert_eq!(
                attempt.evaluate(observed).expect("busy"),
                PressureDecision::Reject(RejectReason::NativeOwnerBusy),
            );
            assert_eq!(attempt.actions_issued(), 0);
        }
    }

    #[test]
    fn leased_resources_are_never_selected_for_release_or_unload() {
        let mut attempt = policy().begin();
        let mut observed = observation(95, 90, 0);
        observed.reclaimable.leased_session_states = 1;
        observed.reclaimable.eligible_idle_models = 0;
        observed.reclaimable.idle_model_has_live_leases = true;

        assert_eq!(
            attempt.evaluate(observed).expect("protected"),
            PressureDecision::Reject(RejectReason::ProtectedResources),
        );
        assert_eq!(attempt.actions_issued(), 0);
    }

    #[test]
    fn a_leased_publication_does_not_block_an_unrelated_eligible_model() {
        let mut attempt = policy().begin();
        let mut observed = observation(95, 90, 0);
        observed.reclaimable.leased_session_states = 1;
        observed.reclaimable.idle_model_has_live_leases = true;
        observed.reclaimable.eligible_idle_models = 1;

        assert_eq!(
            attempt
                .evaluate(observed)
                .expect("independent eligible model"),
            PressureDecision::UnloadIdleModel,
        );
        acknowledge(&mut attempt, ReclaimAction::UnloadIdleModel);
        observed.reclaimable.eligible_idle_models = 0;
        assert_eq!(
            attempt
                .evaluate(observed)
                .expect("only protected owners remain"),
            PressureDecision::Reject(RejectReason::ProtectedResources),
        );
        assert_eq!(attempt.actions_issued(), 1);
    }

    #[test]
    fn disabled_reclamation_does_not_blame_existing_protection() {
        let mut options = policy().options();
        options.allow_idle_state_release = false;
        options.allow_idle_model_unload = false;
        let mut attempt = PressurePolicy::new(options).unwrap().begin();
        let mut observed = observation(95, 90, 0);
        observed.reclaimable.leased_session_states = 1;
        observed.reclaimable.eligible_idle_models = 0;
        observed.reclaimable.idle_model_has_live_leases = true;

        assert_eq!(
            attempt.evaluate(observed).unwrap(),
            PressureDecision::Reject(RejectReason::NoReclaimableResources)
        );
        assert_eq!(attempt.actions_issued(), 0);
    }

    #[test]
    fn exhausted_model_stage_does_not_blame_an_unrelated_protected_owner() {
        let mut options = policy().options();
        options.allow_idle_state_release = false;
        let mut attempt = PressurePolicy::new(options).unwrap().begin();
        let mut observed = observation(95, 90, 0);
        observed.reclaimable.leased_session_states = 1;
        observed.reclaimable.eligible_idle_models = 2;
        observed.reclaimable.idle_model_has_live_leases = true;

        assert_eq!(
            attempt.evaluate(observed).unwrap(),
            PressureDecision::UnloadIdleModel
        );
        acknowledge(&mut attempt, ReclaimAction::UnloadIdleModel);
        observed.admission = ledger(100, 94);
        observed.reclaimable.eligible_idle_models = 1;
        assert_eq!(
            attempt.evaluate(observed).unwrap(),
            PressureDecision::Reject(RejectReason::ReclamationExhausted)
        );
        assert_eq!(attempt.actions_issued(), 1);
    }

    #[test]
    fn missing_required_allocator_observation_is_not_zero_memory() {
        let mut attempt = policy().begin();
        let mut observed = observation(0, 0, 0);
        observed.allocator = None;

        assert_eq!(
            attempt.evaluate(observed).expect("missing"),
            PressureDecision::Reject(RejectReason::AllocatorObservationMissing),
        );
        assert!(!attempt.last_assessment().expect("assessment").fits());
    }

    #[test]
    fn admission_only_mode_is_explicit() {
        let mut options = policy().options();
        options.allocator_ceiling_bytes = None;
        let mut attempt = PressurePolicy::new(options).expect("policy").begin();
        let mut observed = observation(0, 0, 0);
        observed.allocator = None;

        assert_eq!(
            attempt.evaluate(observed).expect("admission-only fit"),
            PressureDecision::Proceed,
        );
    }

    #[test]
    fn impossible_request_rejects_before_destructive_work() {
        let mut attempt = policy().begin();
        let mut observed = observation(0, 0, 0);
        observed.demand.additional_reservation_bytes = 101;

        assert_eq!(
            attempt.evaluate(observed).expect("impossible"),
            PressureDecision::Reject(RejectReason::RequestExceedsAdmissionBudget),
        );
        assert_eq!(attempt.actions_issued(), 0);
    }

    #[test]
    fn impossible_incremental_mlx_requirement_is_rejected() {
        let mut attempt = policy().begin();
        let mut observed = observation(0, 0, 0);
        observed.demand.additional_mlx_bytes = 101;

        assert_eq!(
            attempt.evaluate(observed).expect("impossible MLX demand"),
            PressureDecision::Reject(RejectReason::RequestExceedsAllocatorCeiling),
        );
    }

    #[test]
    fn action_must_be_acknowledged_and_must_match() {
        let mut attempt = policy().begin();
        let observed = observation(20, 60, 30);

        assert_eq!(
            attempt.evaluate(observed).expect("clear"),
            PressureDecision::ClearAllocatorCache,
        );
        assert_eq!(
            attempt.evaluate(observed),
            Err(PressureError::ActionAwaitingCompletion),
        );
        assert_eq!(
            attempt.complete_action(ReclaimAction::UnloadIdleModel, ActionOutcome::Completed,),
            Err(PressureError::WrongActionAcknowledgement),
        );

        acknowledge(&mut attempt, ReclaimAction::ClearAllocatorCache);
    }

    #[test]
    fn failed_action_does_not_blindly_escalate() {
        let mut attempt = policy().begin();
        let observed = observation(95, 60, 30);

        assert_eq!(
            attempt.evaluate(observed).expect("clear"),
            PressureDecision::ClearAllocatorCache,
        );
        attempt
            .complete_action(ReclaimAction::ClearAllocatorCache, ActionOutcome::Failed)
            .expect("failure acknowledgement");

        assert_eq!(
            attempt.evaluate(observed).expect("reject"),
            PressureDecision::Reject(RejectReason::ActionFailed(
                ReclaimAction::ClearAllocatorCache
            )),
        );
        assert_eq!(attempt.actions_issued(), 1);
    }

    #[test]
    fn ineffective_clear_does_not_repeat_forever_in_the_same_phase() {
        let mut attempt = policy().begin();
        let mut observed = observation(20, 60, 30);
        observed.reclaimable.idle_session_states = 0;
        observed.reclaimable.idle_model_loaded = false;
        observed.reclaimable.eligible_idle_models = 0;

        assert_eq!(
            attempt.evaluate(observed).expect("clear"),
            PressureDecision::ClearAllocatorCache,
        );
        acknowledge(&mut attempt, ReclaimAction::ClearAllocatorCache);

        assert_eq!(
            attempt.evaluate(observed).expect("no progress"),
            PressureDecision::Reject(RejectReason::NoProgress),
        );
        assert_eq!(attempt.actions_issued(), 1);
    }

    #[test]
    fn all_reclamation_phases_are_bounded_by_five_actions() {
        let mut attempt = policy().begin();
        let mut observed = observation(95, 95, 10);

        for (decision, action) in [
            (
                PressureDecision::ClearAllocatorCache,
                ReclaimAction::ClearAllocatorCache,
            ),
            (
                PressureDecision::ReleaseIdleSessionState,
                ReclaimAction::ReleaseIdleSessionState,
            ),
            (
                PressureDecision::ClearAllocatorCache,
                ReclaimAction::ClearAllocatorCache,
            ),
            (
                PressureDecision::UnloadIdleModel,
                ReclaimAction::UnloadIdleModel,
            ),
            (
                PressureDecision::ClearAllocatorCache,
                ReclaimAction::ClearAllocatorCache,
            ),
        ] {
            assert_eq!(
                attempt.evaluate(observed).expect("bounded action"),
                decision,
            );
            acknowledge(&mut attempt, action);

            match action {
                ReclaimAction::ReleaseIdleSessionState => {
                    observed.reclaimable.idle_session_states = 0;
                }
                ReclaimAction::UnloadIdleModel => {
                    observed.reclaimable.idle_model_loaded = false;
                    observed.reclaimable.eligible_idle_models = 0;
                }
                ReclaimAction::ClearAllocatorCache => {}
            }
        }

        assert_eq!(attempt.actions_issued(), 5);
        assert_eq!(
            attempt.evaluate(observed).expect("bounded rejection"),
            PressureDecision::Reject(RejectReason::ActionLimitReached),
        );
    }

    #[test]
    fn zero_action_budget_does_not_prevent_an_already_fitting_request() {
        let mut options = policy().options();
        options.max_reclamation_actions = 0;
        let selected = PressurePolicy::new(options).expect("policy");

        assert_eq!(
            selected
                .begin()
                .evaluate(observation(0, 0, 0))
                .expect("fit"),
            PressureDecision::Proceed,
        );
        assert_eq!(
            selected
                .begin()
                .evaluate(observation(95, 0, 0))
                .expect("reject"),
            PressureDecision::Reject(RejectReason::ActionLimitReached),
        );
    }

    #[test]
    fn allocator_arithmetic_overflow_fails_closed() {
        let mut attempt = policy().begin();
        let observed = observation(0, u64::MAX, 1);

        assert_eq!(
            attempt.evaluate(observed),
            Err(PressureError::ArithmeticOverflow),
        );
        assert_eq!(attempt.actions_issued(), 0);
    }

    #[test]
    fn contradictory_resource_counts_are_rejected() {
        let mut attempt = policy().begin();
        let mut observed = observation(0, 0, 0);
        observed.reclaimable.leased_session_states = 2;

        assert!(matches!(
            attempt.evaluate(observed),
            Err(PressureError::InvalidObservation(_)),
        ));
        assert_eq!(attempt.actions_issued(), 0);
    }

    #[test]
    fn terminal_attempt_cannot_reissue_stale_proceed() {
        let mut attempt = policy().begin();

        assert_eq!(
            attempt.evaluate(observation(0, 0, 0)).expect("fit"),
            PressureDecision::Proceed,
        );
        assert_eq!(
            attempt.evaluate(observation(95, 95, 0)),
            Err(PressureError::AttemptFinished),
        );
    }

    #[test]
    fn changing_admission_budget_requires_a_new_logical_attempt() {
        let mut attempt = policy().begin();

        assert_eq!(
            attempt.evaluate(observation(20, 60, 30)).expect("clear"),
            PressureDecision::ClearAllocatorCache,
        );
        acknowledge(&mut attempt, ReclaimAction::ClearAllocatorCache);

        let mut changed = observation(20, 60, 0);
        changed.admission = ledger(200, 20);

        assert_eq!(
            attempt.evaluate(changed),
            Err(PressureError::AdmissionBudgetChanged),
        );
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let mut options = policy().options();
        options.allocator_ceiling_bytes = Some(0);
        assert!(PressurePolicy::new(options).is_err());

        let mut options = policy().options();
        options.max_reclamation_actions = 6;
        assert!(PressurePolicy::new(options).is_err());
    }
}
