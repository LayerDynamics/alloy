use super::*;
use crate::memory_guard::{HostMemorySnapshot, MemoryGuardTier};
use crate::scheduler::{RequestState, Scheduler};
use crate::types::RequestId;

const GIB: u64 = 1024 * 1024 * 1024;

fn host_busy() -> HostMemorySnapshot {
    HostMemorySnapshot {
        physical_memory_bytes: 36 * GIB,
        free_bytes: GIB,
        inactive_bytes: 0,
        active_file_bytes: 0,
        other_apps_active_bytes: 0,
        phys_footprint_bytes: 30 * GIB,
        graphics_footprint_bytes: 20 * GIB,
        iogpu_wired_limit_bytes: 0,
        recommended_working_set_bytes: 100 * GIB,
        release_credit_bytes: 0,
    }
}

fn enforcer_with(host: HostMemorySnapshot, usage_mlx: u64, pool: u64) -> MemoryEnforcer {
    let admission = AdmissionController::new(32 * GIB).expect("admission");
    let enforcer = MemoryEnforcer::new(MemoryGuardConfig::balanced(), admission);
    enforcer.set_observation_overrides(Some(host), Some(usage_mlx), Some(pool));
    enforcer
}

#[test]
fn ok_pressure_unpauses_admissions() {
    let mut host = host_busy();
    host.phys_footprint_bytes = GIB;
    host.graphics_footprint_bytes = 0;
    host.free_bytes = 20 * GIB;
    host.inactive_bytes = 5 * GIB;
    let enforcer = enforcer_with(host, GIB, 0);
    enforcer.admission().set_admissions_paused(true);
    let tick = enforcer.tick().expect("tick");
    assert_eq!(tick.status.pressure, PressureLevel::Ok);
    assert!(!enforcer.admission().admissions_paused());
    assert!(!tick.deferred_for_pool_drain);
}

#[test]
fn soft_pressure_pauses_and_defers_when_pool_large() {
    // H = usage + free - reserve = 20 + 5 - 3 = 22 GiB
    // soft = 0.90*22 = 19.8, hard = 0.95*22 = 20.9 → usage 20 is soft.
    let mut host = host_busy();
    host.free_bytes = 5 * GIB;
    host.inactive_bytes = 0;
    host.active_file_bytes = 0;
    host.other_apps_active_bytes = 0;
    host.phys_footprint_bytes = 20 * GIB;
    host.graphics_footprint_bytes = 0;
    host.recommended_working_set_bytes = 100 * GIB;
    let enforcer = enforcer_with(host, 20 * GIB, 3 * GIB);
    let tick = enforcer.tick().expect("tick");
    assert_eq!(tick.status.pressure, PressureLevel::Soft);
    assert!(enforcer.admission().admissions_paused());
    assert!(tick.deferred_for_pool_drain);
    assert!(tick.request_cache_reclaim);
    assert!(enforcer.cache_reclaim_requested());
}

#[test]
fn hard_pressure_requests_reclaim_after_grace_exhausted() {
    // static = 33 GiB binds; soft=29.7 hard=31.35; usage 32 → hard. pool=0 skips grace.
    let mut host = host_busy();
    host.free_bytes = 5 * GIB;
    host.inactive_bytes = 0;
    host.active_file_bytes = 0;
    host.other_apps_active_bytes = 0;
    host.phys_footprint_bytes = 32 * GIB;
    host.graphics_footprint_bytes = 0;
    host.recommended_working_set_bytes = 100 * GIB;
    let enforcer = enforcer_with(host, 32 * GIB, 0);
    let tick = enforcer.tick().expect("tick");
    assert_eq!(tick.status.pressure, PressureLevel::Hard);
    assert!(enforcer.admission().admissions_paused());
    assert!(tick.request_cache_reclaim);
    assert!(!tick.deferred_for_pool_drain);
}

fn fixed_ceiling_host(usage: u64) -> HostMemorySnapshot {
    HostMemorySnapshot {
        // Balanced static ceiling is 33 GiB; dynamic and Metal ceilings
        // exceed it, so these tests exercise the abort classifier itself.
        free_bytes: 10 * GIB,
        phys_footprint_bytes: usage,
        graphics_footprint_bytes: 0,
        ..host_busy()
    }
}

#[test]
fn ordinary_soft_and_hard_pressure_never_request_cancellation() {
    let scheduler = Scheduler::new(1).expect("scheduler");
    let request = RequestId::new("ordinary-pressure").expect("request");
    let _permit = scheduler.try_acquire(request.clone()).expect("permit");
    let target = scheduler.active_cancellation_target().expect("target");
    for usage in [30 * GIB, 32 * GIB] {
        let enforcer = enforcer_with(fixed_ceiling_host(usage), usage, 3 * GIB);
        // Cover both grace and post-grace policy, with the real target.
        for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX + 2 {
            let tick = enforcer.tick_for_execution(target.clone()).expect("tick");
            assert_ne!(tick.status.pressure, PressureLevel::Ok);
            assert!(tick.status.admissions_paused);
            assert!(!tick.status.emergency);
            assert!(tick.request_execution_cancellation.is_none());
        }
    }
    assert_eq!(
        scheduler.request_state(&request).expect("state"),
        RequestState::Running
    );
}

#[test]
fn consecutive_over_ceiling_polls_interrupt_the_captured_execution() {
    let scheduler = Scheduler::new(1).expect("scheduler");
    let _permit = scheduler
        .try_acquire(RequestId::new("consecutive-pressure").expect("request"))
        .expect("permit");
    let target = scheduler.active_cancellation_target().expect("target");
    let enforcer = enforcer_with(fixed_ceiling_host(33 * GIB), 33 * GIB, 3 * GIB);

    let first = enforcer
        .tick_for_execution(target.clone())
        .expect("first tick");
    assert!(first.deferred_for_pool_drain);
    assert!(!first.status.emergency);
    assert!(first.request_execution_cancellation.is_none());

    let emergency = enforcer
        .tick_for_execution(target.clone())
        .expect("second tick");
    assert!(emergency.status.emergency);
    assert!(emergency.status.admissions_paused);
    assert!(!emergency.deferred_for_pool_drain);
    assert!(emergency.request_cache_reclaim);
    assert!(emergency.request_model_eviction);
    assert_eq!(emergency.request_execution_cancellation, target);
}

#[test]
fn emergency_margin_requests_cancellation_on_the_first_poll() {
    let scheduler = Scheduler::new(1).expect("scheduler");
    let _permit = scheduler
        .try_acquire(RequestId::new("margin-pressure").expect("request"))
        .expect("permit");
    let target = scheduler.active_cancellation_target().expect("target");
    let enforcer = enforcer_with(fixed_ceiling_host(35 * GIB), 35 * GIB, 3 * GIB);
    let tick = enforcer.tick_for_execution(target.clone()).expect("tick");
    assert!(tick.status.emergency);
    assert!(!tick.deferred_for_pool_drain);
    assert_eq!(tick.request_execution_cancellation, target);

    // Emergency pressure while idle still pauses admission and reclaims,
    // but must not manufacture a target for a request that starts later.
    let idle = enforcer.tick().expect("idle tick");
    assert!(idle.status.emergency);
    assert!(idle.status.admissions_paused);
    assert!(idle.request_execution_cancellation.is_none());
}

#[test]
fn acceptable_observation_resets_emergency_history_and_action() {
    let scheduler = Scheduler::new(1).expect("scheduler");
    let _permit = scheduler
        .try_acquire(RequestId::new("recovered-pressure").expect("request"))
        .expect("permit");
    let target = scheduler.active_cancellation_target().expect("target");
    let enforcer = enforcer_with(fixed_ceiling_host(35 * GIB), 35 * GIB, 0);
    assert!(
        enforcer
            .tick_for_execution(target.clone())
            .expect("emergency")
            .status
            .emergency
    );

    enforcer.set_observation_overrides(Some(fixed_ceiling_host(GIB)), Some(GIB), Some(0));
    let recovery = enforcer
        .tick_for_execution(target.clone())
        .expect("recovery");
    assert_eq!(recovery.status.pressure, PressureLevel::Ok);
    assert!(!recovery.status.admissions_paused);
    assert!(!recovery.status.emergency);
    assert!(recovery.request_execution_cancellation.is_none());

    enforcer.set_observation_overrides(Some(fixed_ceiling_host(33 * GIB)), Some(33 * GIB), Some(0));
    let first = enforcer.tick_for_execution(target).expect("new first poll");
    assert!(!first.status.emergency);
    assert!(first.request_execution_cancellation.is_none());
}

#[test]
fn disabled_guard_never_interrupts_an_execution() {
    let scheduler = Scheduler::new(1).expect("scheduler");
    let _permit = scheduler
        .try_acquire(RequestId::new("disabled-pressure").expect("request"))
        .expect("permit");
    let mut config = MemoryGuardConfig::balanced();
    config.enabled = false;
    let enforcer = MemoryEnforcer::new(
        config,
        AdmissionController::new(32 * GIB).expect("admission"),
    );
    enforcer.set_observation_overrides(Some(fixed_ceiling_host(35 * GIB)), Some(35 * GIB), Some(0));
    let tick = enforcer
        .tick_for_execution(scheduler.active_cancellation_target().expect("target"))
        .expect("tick");
    assert!(!tick.status.emergency);
    assert!(!tick.status.admissions_paused);
    assert!(tick.request_execution_cancellation.is_none());
}

#[test]
fn paused_admission_rejects_reserve() {
    let admission = AdmissionController::new(1024).expect("admission");
    admission.set_admissions_paused(true);
    let error = admission.reserve("load", 16).expect_err("paused");
    match error {
        crate::error::EngineError::MemoryLimitExceeded { operation, .. } => {
            assert!(operation.contains("paused"));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn tier_name_exposed_via_config() {
    let admission = AdmissionController::new(1024).expect("admission");
    let enforcer = MemoryEnforcer::new(MemoryGuardConfig::balanced(), admission);
    assert_eq!(enforcer.config().tier, MemoryGuardTier::Balanced);
}
