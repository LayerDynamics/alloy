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
