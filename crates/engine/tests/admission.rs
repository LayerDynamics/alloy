//! Public-API integration tests for engine memory admission.
//!
//! These tests exercise:
//!
//! - Explicit budget validation.
//! - Reservation acquisition and structured rejection.
//! - Shared accounting across controller clones.
//! - RAII and explicit release.
//! - Reservation resizing and rollback on failure.
//! - Component and estimate accounting.
//! - Rechecking current availability instead of trusting stale estimates.
//! - Reservation lifetime across threads and shared owners.
//! - Concurrent reservation and resize arbitration.
//! - Checked arithmetic at large accounting values.
//!
//! These are accounting and ownership tests. They do not measure native
//! model allocations, process RSS, Metal memory, or estimate accuracy
//! against a real loaded model.
//!
//! Poisoning is covered by admission.rs's internal unit tests because
//! integration tests intentionally do not access private mutex state.

use engine::{
    admission::{AdmissionController, MemoryReservation},
    error::{EngineError, ErrorKind},
    types::{MemoryBreakdown, MemoryEstimate},
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Barrier},
    thread,
};

fn assert_accounting(
    admission: &AdmissionController,
    budget_bytes: u64,
    reserved_bytes: u64,
    active_reservations: usize,
) {
    let snapshot = admission.snapshot().expect("accounting snapshot");

    assert_eq!(snapshot.budget_bytes(), budget_bytes,);

    assert_eq!(snapshot.reserved_bytes(), reserved_bytes,);

    assert_eq!(
        snapshot.available_bytes(),
        budget_bytes
            .checked_sub(reserved_bytes)
            .expect("expected reservation fits budget"),
    );

    assert_eq!(snapshot.active_reservations(), active_reservations,);

    assert_eq!(
        snapshot
            .reserved_bytes()
            .checked_add(snapshot.available_bytes()),
        Some(snapshot.budget_bytes()),
    );
}

fn assert_memory_limit(
    error: EngineError,
    expected_operation: &'static str,
    expected_required: u64,
    expected_available: u64,
) {
    match error {
        EngineError::MemoryLimitExceeded {
            operation,
            required_bytes,
            available_bytes,
        } => {
            assert_eq!(operation, expected_operation,);

            assert_eq!(required_bytes, expected_required,);

            assert_eq!(available_bytes, expected_available,);
        }

        other => {
            panic!("expected MemoryLimitExceeded, got {other:?}");
        }
    }
}

#[test]
fn controller_and_reservation_support_thread_transfer() {
    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<AdmissionController>();
    assert_send::<MemoryReservation>();
}

#[test]
fn zero_budget_is_rejected_through_public_api() {
    let error = AdmissionController::new(0).expect_err("zero budget must be rejected");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);
}

#[test]
fn initial_snapshot_reports_the_configured_budget() {
    let admission = AdmissionController::new(100).expect("controller");

    assert_accounting(&admission, 100, 0, 0);

    let snapshot = admission.snapshot().expect("snapshot");

    assert!(snapshot.can_reserve(100));
    assert!(!snapshot.can_reserve(101));
    assert!(!snapshot.can_reserve(0));
}

#[test]
fn reservation_records_its_size_and_operation() {
    let admission = AdmissionController::new(100).expect("controller");

    let reservation = admission
        .reserve("integration allocation", 40)
        .expect("reservation");

    assert_eq!(reservation.bytes(), 40,);

    assert_eq!(reservation.operation(), "integration allocation",);

    assert_accounting(&admission, 100, 40, 1);

    drop(reservation);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn zero_reservation_is_rejected_without_mutation() {
    let admission = AdmissionController::new(100).expect("controller");

    let before = admission.snapshot().expect("before");

    let error = admission
        .reserve("zero allocation", 0)
        .expect_err("zero reservation");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);

    assert_eq!(admission.snapshot().expect("after"), before,);
}

#[test]
fn exact_fit_exhausts_budget_until_released() {
    let admission = AdmissionController::new(100).expect("controller");

    let reservation = admission
        .reserve("exact allocation", 100)
        .expect("exact fit");

    assert_accounting(&admission, 100, 100, 1);

    let error = admission
        .reserve("additional allocation", 1)
        .expect_err("budget exhausted");

    assert_memory_limit(error, "additional allocation", 1, 0);

    reservation.release().expect("explicit release");

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn failed_admission_preserves_existing_reservations() {
    let admission = AdmissionController::new(100).expect("controller");

    let first = admission.reserve("first allocation", 60).expect("first");

    let second = admission.reserve("second allocation", 30).expect("second");

    let before = admission.snapshot().expect("before");

    let error = admission
        .reserve("rejected allocation", 11)
        .expect_err("insufficient budget");

    assert_memory_limit(error, "rejected allocation", 11, 10);

    assert_eq!(admission.snapshot().expect("after"), before,);

    assert_eq!(first.bytes(), 60);
    assert_eq!(second.bytes(), 30);

    drop(first);

    assert_accounting(&admission, 100, 30, 1);

    drop(second);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn controller_clones_enforce_the_same_budget() {
    let admission = AdmissionController::new(100).expect("controller");

    let other = admission.clone();

    let reservation = admission
        .reserve("shared allocation", 80)
        .expect("reservation");

    assert_accounting(&other, 100, 80, 1);

    let error = other
        .reserve("competing allocation", 21)
        .expect_err("shared budget");

    assert_memory_limit(error, "competing allocation", 21, 20);

    drop(reservation);

    let replacement = other
        .reserve("replacement allocation", 100)
        .expect("released budget is available");

    assert_accounting(&admission, 100, 100, 1);

    drop(replacement);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn explicit_release_does_not_double_release_on_drop() {
    let admission = AdmissionController::new(100).expect("controller");

    let first = admission.reserve("first allocation", 40).expect("first");

    let second = admission.reserve("second allocation", 30).expect("second");

    first.release().expect("release first");

    assert_accounting(&admission, 100, 30, 1);

    second.release().expect("release second");

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn dropping_one_controller_does_not_invalidate_a_live_guard() {
    let admission = AdmissionController::new(100).expect("controller");

    let observer = admission.clone();

    let reservation = admission
        .reserve("surviving allocation", 70)
        .expect("reservation");

    drop(admission);

    assert_accounting(&observer, 100, 70, 1);

    reservation.release().expect("guard remains valid");

    assert_accounting(&observer, 100, 0, 0);
}

#[test]
fn shared_guard_remains_reserved_until_last_owner_is_dropped() {
    let admission = AdmissionController::new(100).expect("controller");

    let first = Arc::new(
        admission
            .reserve("shared output ownership", 60)
            .expect("reservation"),
    );

    let second = Arc::clone(&first);
    let third = Arc::clone(&first);

    // Three Arc owners still represent one accounting reservation.
    assert_accounting(&admission, 100, 60, 1);

    drop(first);
    drop(second);

    assert_accounting(&admission, 100, 60, 1);

    drop(third);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn resize_grows_and_shrinks_only_the_owned_reservation() {
    let admission = AdmissionController::new(100).expect("controller");

    let mut first = admission
        .reserve("resizable allocation", 20)
        .expect("first");

    let second = admission
        .reserve("unrelated allocation", 30)
        .expect("second");

    first.try_resize(60).expect("grow first");

    assert_eq!(first.bytes(), 60);
    assert_eq!(second.bytes(), 30);

    assert_accounting(&admission, 100, 90, 2);

    first.try_resize(10).expect("shrink first");

    assert_eq!(first.bytes(), 10);
    assert_eq!(second.bytes(), 30);

    assert_accounting(&admission, 100, 40, 2);

    drop(first);

    assert_accounting(&admission, 100, 30, 1);

    drop(second);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn resize_to_same_size_preserves_accounting() {
    let admission = AdmissionController::new(100).expect("controller");

    let mut reservation = admission
        .reserve("unchanged allocation", 40)
        .expect("reservation");

    let before = admission.snapshot().expect("before");

    reservation.try_resize(40).expect("same-size resize");

    assert_eq!(reservation.bytes(), 40);

    assert_eq!(admission.snapshot().expect("after"), before,);

    drop(reservation);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn failed_resize_preserves_guard_and_shared_accounting() {
    let admission = AdmissionController::new(100).expect("controller");

    let mut reservation = admission
        .reserve("resizable allocation", 40)
        .expect("reservation");

    let other = admission.reserve("other allocation", 50).expect("other");

    let before = admission.snapshot().expect("before");

    let error = reservation
        .try_resize(51)
        .expect_err("eleven extra bytes do not fit");

    // Resize errors describe the additional bytes being requested.
    assert_memory_limit(error, "resizable allocation", 11, 10);

    assert_eq!(reservation.bytes(), 40);

    assert_eq!(admission.snapshot().expect("after"), before,);

    drop(other);
    drop(reservation);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn zero_resize_is_rejected_without_releasing_ownership() {
    let admission = AdmissionController::new(100).expect("controller");

    let mut reservation = admission
        .reserve("resizable allocation", 40)
        .expect("reservation");

    let before = admission.snapshot().expect("before");

    let error = reservation.try_resize(0).expect_err("zero resize");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);

    assert_eq!(reservation.bytes(), 40);

    assert_eq!(admission.snapshot().expect("after"), before,);

    drop(reservation);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn component_requirements_create_one_reservation() {
    let admission = AdmissionController::new(100).expect("controller");

    let reservation = admission
        .reserve_components("combined allocation", &[20, 0, 30, 10])
        .expect("component reservation");

    assert_eq!(reservation.bytes(), 60);

    assert_accounting(&admission, 100, 60, 1);

    drop(reservation);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn empty_and_zero_component_totals_are_rejected() {
    let admission = AdmissionController::new(100).expect("controller");

    for components in [&[][..], &[0_u64, 0, 0][..]] {
        let before = admission.snapshot().expect("before");

        let error = admission
            .reserve_components("zero components", components)
            .expect_err("positive total required");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest,);

        assert_eq!(admission.snapshot().expect("after"), before,);
    }
}

#[test]
fn overflowing_component_sum_is_rejected_without_mutation() {
    let admission = AdmissionController::new(u64::MAX).expect("controller");

    let before = admission.snapshot().expect("before");

    let error = admission
        .reserve_components("overflowing components", &[u64::MAX, 1])
        .expect_err("checked component arithmetic");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);

    assert_eq!(admission.snapshot().expect("after"), before,);
}

#[test]
fn estimate_breakdown_is_reserved_in_full() {
    let admission = AdmissionController::new(200).expect("controller");

    let breakdown = MemoryBreakdown {
        weights_bytes: 40,
        kv_bytes: 20,
        temporary_bytes: 10,
        output_bytes: 5,
        draft_bytes: 3,
        headroom_bytes: 2,
    };

    let estimate = MemoryEstimate::new(breakdown, 200, Vec::new()).expect("estimate");

    assert_eq!(estimate.reserved_bytes(), 80);
    assert!(estimate.fits());

    let reservation = admission
        .reserve_estimate("estimated allocation", &estimate)
        .expect("reserve estimate");

    assert_eq!(reservation.bytes(), 80);

    assert_accounting(&admission, 200, 80, 1);

    drop(reservation);

    assert_accounting(&admission, 200, 0, 0);
}

#[test]
fn optimistic_estimate_cannot_authorize_stale_budget() {
    let admission = AdmissionController::new(100).expect("controller");

    let estimate = MemoryEstimate::new(
        MemoryBreakdown {
            weights_bytes: 40,
            kv_bytes: 20,
            ..MemoryBreakdown::default()
        },
        100,
        Vec::new(),
    )
    .expect("estimate");

    assert!(estimate.fits());

    let existing = admission
        .reserve("new competing allocation", 70)
        .expect("competing reservation");

    let before = admission.snapshot().expect("before");

    let error = admission
        .reserve_estimate("stale optimistic estimate", &estimate)
        .expect_err("current budget must be checked");

    assert_memory_limit(error, "stale optimistic estimate", 60, 30);

    assert_eq!(admission.snapshot().expect("after"), before,);

    drop(existing);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn pessimistic_estimate_does_not_override_newly_available_budget() {
    let admission = AdmissionController::new(100).expect("controller");

    let existing = admission
        .reserve("temporary owner", 80)
        .expect("existing reservation");

    let estimate = MemoryEstimate::new(
        MemoryBreakdown {
            weights_bytes: 30,
            kv_bytes: 20,
            ..MemoryBreakdown::default()
        },
        admission.snapshot().expect("snapshot").available_bytes(),
        Vec::new(),
    )
    .expect("estimate");

    assert!(!estimate.fits());

    drop(existing);

    let reservation = admission
        .reserve_estimate("fresh admission check", &estimate)
        .expect("current budget now fits");

    assert_eq!(reservation.bytes(), 50);

    assert_accounting(&admission, 100, 50, 1);

    drop(reservation);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn separate_controller_instances_have_independent_budgets() {
    let first = AdmissionController::new(100).expect("first controller");

    let second = AdmissionController::new(100).expect("second controller");

    let first_guard = first
        .reserve("first controller allocation", 100)
        .expect("first reservation");

    let second_guard = second
        .reserve("second controller allocation", 100)
        .expect("second reservation");

    assert_accounting(&first, 100, 100, 1);
    assert_accounting(&second, 100, 100, 1);

    drop(first_guard);

    assert_accounting(&first, 100, 0, 0);
    assert_accounting(&second, 100, 100, 1);

    drop(second_guard);

    assert_accounting(&second, 100, 0, 0);
}

#[test]
fn explicit_non_power_of_two_budget_is_not_rounded() {
    const GIB: u64 = 1024 * 1024 * 1024;

    let budget = 36 * GIB;

    let admission = AdmissionController::new(budget).expect("36 GiB accounting budget");

    let reservation = admission
        .reserve("explicit budget allocation", 35 * GIB)
        .expect("reservation");

    assert_accounting(&admission, 36 * GIB, 35 * GIB, 1);

    assert_eq!(
        admission.snapshot().expect("snapshot").available_bytes(),
        GIB,
    );

    drop(reservation);

    assert_accounting(&admission, budget, 0, 0);
}

#[test]
fn maximum_byte_budget_uses_checked_accounting() {
    // These are ledger values, not native allocations of this size.
    let admission = AdmissionController::new(u64::MAX).expect("maximum accounting budget");

    let mut first = admission
        .reserve("almost full accounting reservation", u64::MAX - 1)
        .expect("first");

    let second = admission
        .reserve("last accounting byte", 1)
        .expect("second");

    assert_accounting(&admission, u64::MAX, u64::MAX, 2);

    let before = admission.snapshot().expect("before failed growth");

    let error = first.try_resize(u64::MAX).expect_err("no remaining budget");

    assert_memory_limit(error, "almost full accounting reservation", 1, 0);

    assert_eq!(first.bytes(), u64::MAX - 1,);

    assert_eq!(admission.snapshot().expect("after"), before,);

    drop(second);

    first
        .try_resize(u64::MAX)
        .expect("released byte permits growth");

    assert_accounting(&admission, u64::MAX, u64::MAX, 1);

    drop(first);

    assert_accounting(&admission, u64::MAX, 0, 0);
}

#[test]
fn unwinding_owner_releases_reservation_without_poisoning_ledger() {
    let admission = AdmissionController::new(100).expect("controller");

    let owner = admission.clone();

    let result = catch_unwind(AssertUnwindSafe(move || {
        let _reservation = owner.reserve("unwinding owner", 70).expect("reservation");

        // Panic outside the controller's internal critical section.
        panic!("exercise public RAII cleanup");
    }));

    assert!(result.is_err());

    assert_accounting(&admission, 100, 0, 0);

    let next = admission
        .reserve("allocation after unwind", 100)
        .expect("ledger remains usable");

    drop(next);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn concurrent_reservations_cannot_over_admit() {
    const WORKERS: usize = 8;

    let admission = AdmissionController::new(100).expect("controller");

    let start = Arc::new(Barrier::new(WORKERS + 1));

    let attempted = Arc::new(Barrier::new(WORKERS + 1));

    let release = Arc::new(Barrier::new(WORKERS + 1));

    let mut workers = Vec::new();

    for _ in 0..WORKERS {
        let admission = admission.clone();
        let start = Arc::clone(&start);
        let attempted = Arc::clone(&attempted);
        let release = Arc::clone(&release);

        workers.push(thread::spawn(move || {
            start.wait();

            let result = admission.reserve("concurrent allocation", 25);

            // Successful guards remain owned while the main thread
            // observes the state after all admission attempts.
            attempted.wait();
            release.wait();

            result.map(|reservation| {
                let bytes = reservation.bytes();
                drop(reservation);
                bytes
            })
        }));
    }

    start.wait();
    attempted.wait();

    // Defer assertions until workers have been released, so a failed
    // accounting assertion cannot leave them waiting at the barrier.
    let held_snapshot = admission.snapshot();

    release.wait();

    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("worker"))
        .collect();

    let snapshot = held_snapshot.expect("held snapshot");

    assert_eq!(snapshot.reserved_bytes(), 100);
    assert_eq!(snapshot.available_bytes(), 0);
    assert_eq!(snapshot.active_reservations(), 4);

    let mut accepted = 0;
    let mut rejected = 0;

    for result in results {
        match result {
            Ok(bytes) => {
                accepted += 1;
                assert_eq!(bytes, 25);
            }

            Err(error) => {
                rejected += 1;

                assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded,);
            }
        }
    }

    assert_eq!(accepted, 4);
    assert_eq!(rejected, 4);

    assert_accounting(&admission, 100, 0, 0);
}

#[test]
fn concurrent_growth_cannot_spend_the_same_available_bytes_twice() {
    let admission = AdmissionController::new(100).expect("controller");

    let first = admission.reserve("first resize owner", 20).expect("first");

    let second = admission
        .reserve("second resize owner", 20)
        .expect("second");

    let start = Arc::new(Barrier::new(3));
    let attempted = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));

    let mut workers = Vec::new();

    for mut reservation in [first, second] {
        let start = Arc::clone(&start);
        let attempted = Arc::clone(&attempted);
        let release = Arc::clone(&release);

        workers.push(thread::spawn(move || {
            start.wait();

            // Each growth needs 50 additional bytes. Only one can fit
            // in the initially available 60 bytes.
            let result = reservation.try_resize(70);
            let owned_bytes = reservation.bytes();

            attempted.wait();
            release.wait();

            drop(reservation);

            (result, owned_bytes)
        }));
    }

    start.wait();
    attempted.wait();

    let held_snapshot = admission.snapshot();

    release.wait();

    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("resize worker"))
        .collect();

    let snapshot = held_snapshot.expect("held snapshot");

    assert_eq!(snapshot.reserved_bytes(), 90);
    assert_eq!(snapshot.available_bytes(), 10);
    assert_eq!(snapshot.active_reservations(), 2);

    let mut grown = 0;
    let mut rejected = 0;

    for (result, owned_bytes) in results {
        match result {
            Ok(()) => {
                grown += 1;
                assert_eq!(owned_bytes, 70);
            }

            Err(error) => {
                rejected += 1;
                assert_eq!(owned_bytes, 20);

                assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded,);
            }
        }
    }

    assert_eq!(grown, 1);
    assert_eq!(rejected, 1);

    assert_accounting(&admission, 100, 0, 0);
}
