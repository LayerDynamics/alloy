use super::*;
use crate::error::ErrorKind;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Barrier},
    thread,
};

#[test]
fn zero_budget_is_rejected() {
    let error = AdmissionController::new(0).expect_err("zero budget must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
}

#[test]
fn zero_reservation_is_rejected_without_changing_accounting() {
    let admission = AdmissionController::new(100).expect("controller");
    let error = admission
        .reserve("test allocation", 0)
        .expect_err("zero reservation");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);

    let snapshot = admission.snapshot().expect("snapshot");
    assert_eq!(snapshot.reserved_bytes(), 0);
    assert_eq!(snapshot.active_reservations(), 0);
}

#[test]
fn reservation_updates_shared_accounting() {
    let admission = AdmissionController::new(100).expect("controller");
    let reservation = admission
        .reserve("test allocation", 40)
        .expect("reservation");

    assert_eq!(reservation.bytes(), 40);
    assert_eq!(reservation.operation(), "test allocation");

    let snapshot = admission.snapshot().expect("snapshot");
    assert_eq!(snapshot.budget_bytes(), 100);
    assert_eq!(snapshot.reserved_bytes(), 40);
    assert_eq!(snapshot.available_bytes(), 60);
    assert_eq!(snapshot.active_reservations(), 1);
}

#[test]
fn exact_budget_can_be_reserved() {
    let admission = AdmissionController::new(100).expect("controller");
    let reservation = admission
        .reserve("exact allocation", 100)
        .expect("exact fit");

    assert_eq!(admission.snapshot().expect("snapshot").available_bytes(), 0);

    let error = admission
        .reserve("additional allocation", 1)
        .expect_err("budget exhausted");

    match error {
        EngineError::MemoryLimitExceeded {
            required_bytes,
            available_bytes,
            reason,
            ..
        } => {
            assert_eq!(required_bytes, 1);
            assert_eq!(available_bytes, 0);
            assert_eq!(
                reason,
                MemoryLimitReason::ReservationBudget { budget_bytes: 100 }
            );
        }
        other => panic!("expected memory limit error, got {other:?}"),
    }

    drop(reservation);
}

#[test]
fn failed_reservation_leaves_accounting_unchanged() {
    let admission = AdmissionController::new(100).expect("controller");
    let _reservation = admission
        .reserve("existing allocation", 90)
        .expect("reservation");
    let before = admission.snapshot().expect("before");

    let error = admission
        .reserve("oversized allocation", 11)
        .expect_err("insufficient remaining budget");

    assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded);
    assert_eq!(admission.snapshot().expect("after"), before);
}

#[test]
fn paused_admissions_report_guard_cause_while_bytes_remain_available() {
    let admission = AdmissionController::new(100).expect("controller");
    let _retained = admission
        .reserve("retained owner", 40)
        .expect("reservation");
    let before = admission.snapshot().expect("snapshot");
    admission.set_admissions_paused(true);

    let failures = [
        admission
            .reserve("small allocation", 1)
            .expect_err("paused allocation"),
        admission
            .reserve_grant("small startup", &[1, 1])
            .expect_err("paused grant"),
    ];
    for error in failures {
        assert!(matches!(
            error,
            EngineError::MemoryLimitExceeded {
                required_bytes: 1 | 2,
                available_bytes: 60,
                reason: MemoryLimitReason::AdmissionsPaused,
                ..
            }
        ));
    }
    assert_eq!(admission.snapshot().expect("snapshot"), before);
    admission.set_admissions_paused(false);
    let resumed = admission
        .reserve("resumed allocation", 1)
        .expect("unpaused");
    drop(resumed);
    assert_eq!(admission.snapshot().expect("snapshot"), before);
}

#[test]
fn dropping_a_guard_releases_its_reservation() {
    let admission = AdmissionController::new(100).expect("controller");

    {
        let _reservation = admission
            .reserve("scoped allocation", 70)
            .expect("reservation");
        assert_eq!(admission.snapshot().expect("reserved").reserved_bytes(), 70);
    }

    let snapshot = admission.snapshot().expect("released");
    assert_eq!(snapshot.reserved_bytes(), 0);
    assert_eq!(snapshot.available_bytes(), 100);
    assert_eq!(snapshot.active_reservations(), 0);
}

#[test]
fn explicit_release_does_not_release_twice() {
    let admission = AdmissionController::new(100).expect("controller");
    let first = admission.reserve("first allocation", 40).expect("first");
    let second = admission.reserve("second allocation", 30).expect("second");

    first.release().expect("explicit release");
    let snapshot = admission.snapshot().expect("snapshot");
    assert_eq!(snapshot.reserved_bytes(), 30);
    assert_eq!(snapshot.active_reservations(), 1);

    second.release().expect("release second");
    assert_eq!(admission.snapshot().expect("empty").reserved_bytes(), 0);
}

#[test]
fn cloned_controllers_share_one_budget() {
    let admission = AdmissionController::new(100).expect("controller");
    let other = admission.clone();
    let _reservation = admission
        .reserve("first allocation", 80)
        .expect("reservation");

    let error = other
        .reserve("competing allocation", 21)
        .expect_err("shared budget must be enforced");

    assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded);
    assert_eq!(
        other.snapshot().expect("shared snapshot").reserved_bytes(),
        80
    );
}

#[test]
fn grant_admission_is_atomic_and_partitions_without_reacquiring() {
    let admission = AdmissionController::new(100).expect("controller");

    let mut grant = admission
        .reserve_grant("generation startup", &[20, 30, 10])
        .expect("atomic startup grant");

    let admitted = admission.snapshot().expect("admitted");
    assert_eq!(admitted.reserved_bytes(), 60);
    assert_eq!(admitted.active_reservations(), 1);
    assert_eq!(grant.remaining_bytes(), 60);

    let output = grant
        .take("output buffers", 20)
        .expect("output grant")
        .expect("positive portion");
    let workspace = grant
        .take("workspace", 30)
        .expect("workspace grant")
        .expect("positive portion");
    let kv = grant
        .take("KV allocation", 10)
        .expect("KV grant")
        .expect("positive portion");

    // Partition transfers ownership without accounting churn.
    let partitioned = admission.snapshot().expect("partitioned");
    assert_eq!(partitioned.reserved_bytes(), 60);
    assert_eq!(partitioned.active_reservations(), 3);
    assert_eq!(grant.remaining_bytes(), 0);
    assert!(grant.is_empty());

    drop(output);
    drop(workspace);
    drop(kv);

    let released = admission.snapshot().expect("all portions released");
    assert_eq!(released.reserved_bytes(), 0);
    assert_eq!(released.active_reservations(), 0);
}

#[test]
fn failed_grant_does_not_reserve_a_partial_total() {
    let admission = AdmissionController::new(100).expect("controller");
    let existing = admission.reserve("existing", 80).expect("existing");
    let before = admission.snapshot().expect("before");

    let error = admission
        .reserve_grant("startup", &[10, 11])
        .expect_err("complete startup grant does not fit");

    assert!(matches!(error, EngineError::MemoryLimitExceeded { .. }));
    assert_eq!(admission.snapshot().expect("after"), before);
    drop(existing);
}

#[test]
fn checked_grant_sum_overflow_does_not_change_accounting() {
    let admission = AdmissionController::new(u64::MAX).expect("controller");

    let error = admission
        .reserve_grant("overflow", &[u64::MAX, 1])
        .expect_err("checked sum");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 0);
}

#[test]
fn zero_total_grant_is_rejected() {
    let admission = AdmissionController::new(100).expect("controller");

    let error = admission
        .reserve_grant("zero grant", &[0, 0])
        .expect_err("zero total");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 0);
}

#[test]
fn grant_rejects_overpartition_without_changing_ownership() {
    let admission = AdmissionController::new(100).expect("controller");
    let mut grant = admission
        .reserve_grant("startup", &[40, 60])
        .expect("grant");
    let before = admission.snapshot().expect("before");

    let error = grant.take("oversized part", 101).expect_err("overclaim");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    assert_eq!(grant.remaining_bytes(), 100);
    assert_eq!(admission.snapshot().expect("after"), before);
}

#[test]
fn zero_partition_creates_no_reservation() {
    let admission = AdmissionController::new(100).expect("controller");
    let mut grant = admission.reserve_grant("startup", &[100]).expect("grant");
    let before = admission.snapshot().expect("before");

    assert!(
        grant
            .take("optional empty resource", 0)
            .expect("zero part")
            .is_none()
    );
    assert_eq!(grant.remaining_bytes(), 100);
    assert_eq!(admission.snapshot().expect("after"), before);
}

#[test]
fn dropping_partially_used_grant_releases_only_its_remainder() {
    let admission = AdmissionController::new(100).expect("controller");
    let mut grant = admission
        .reserve_grant("startup", &[70, 30])
        .expect("grant");
    let output = grant
        .take("output", 30)
        .expect("partition")
        .expect("portion");

    assert_eq!(
        admission
            .snapshot()
            .expect("grant plus portion")
            .reserved_bytes(),
        100
    );
    assert_eq!(grant.remaining_bytes(), 70);

    drop(grant);

    let snapshot = admission.snapshot().expect("grant remainder dropped");
    assert_eq!(snapshot.reserved_bytes(), 30);
    assert_eq!(snapshot.active_reservations(), 1);
    assert_eq!(output.bytes(), 30);

    drop(output);
    let empty = admission.snapshot().expect("all released");
    assert_eq!(empty.reserved_bytes(), 0);
    assert_eq!(empty.active_reservations(), 0);
}

#[test]
fn explicit_grant_release_preserves_previously_partitioned_resources() {
    let admission = AdmissionController::new(100).expect("controller");
    let mut grant = admission
        .reserve_grant("startup", &[25, 75])
        .expect("grant");
    let output = grant.take("output", 25).expect("take").expect("output");

    grant.release().expect("release unused grant");

    let snapshot = admission.snapshot().expect("only output remains");
    assert_eq!(snapshot.reserved_bytes(), 25);
    assert_eq!(snapshot.active_reservations(), 1);
    drop(output);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}

#[test]
fn grant_can_be_split_into_loading_peak_then_resident_weights() {
    let admission = AdmissionController::new(120).expect("controller");
    let mut grant = admission
        .reserve_grant("load request", &[100, 20])
        .expect("loading peak plus workspace");

    // The peak reservation owns overlapping source/resident bytes.
    // After the loader drops source temporaries, resize that same guard
    // to the retained resident model amount.
    let mut loading = grant
        .take("model loading peak", 80)
        .expect("take loading")
        .expect("loading guard");
    let workspace = grant
        .take("request workspace", 20)
        .expect("take workspace")
        .expect("workspace guard");

    assert_eq!(grant.remaining_bytes(), 20);
    // Before resizing: 80 loading + 20 workspace + 20 grant remainder.
    assert_eq!(
        admission
            .snapshot()
            .expect("before resize")
            .reserved_bytes(),
        120,
    );

    loading
        .try_resize(60)
        .expect("release loader-overlap allowance");
    assert_eq!(loading.bytes(), 60);
    // After resizing: 60 loading + 20 workspace + 20 grant remainder.
    assert_eq!(
        admission.snapshot().expect("after resize").reserved_bytes(),
        100,
    );

    let resident = grant
        .take("resident model remainder", 20)
        .expect("take resident")
        .expect("resident guard");
    assert!(grant.is_empty());
    // Taking the remaining 20 transfers ownership; it releases no bytes.
    assert_eq!(
        admission.snapshot().expect("transferred").reserved_bytes(),
        100,
    );

    drop(workspace);
    drop(loading);
    drop(resident);
    assert_eq!(admission.snapshot().expect("empty").reserved_bytes(), 0);
}

#[test]
fn concurrent_grants_cannot_overadmit() {
    let admission = AdmissionController::new(100).expect("controller");
    let start = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();

    for _ in 0..2 {
        let admission = admission.clone();
        let start = Arc::clone(&start);
        let release = Arc::clone(&release);

        workers.push(thread::spawn(move || {
            start.wait();
            let grant = admission.reserve_grant("concurrent startup", &[75]);
            release.wait();

            match grant {
                Ok(grant) => {
                    assert_eq!(grant.remaining_bytes(), 75);
                    drop(grant);
                    true
                }
                Err(error) => {
                    assert!(matches!(error, EngineError::MemoryLimitExceeded { .. }));
                    false
                }
            }
        }));
    }

    start.wait();
    release.wait();

    let successes = workers
        .into_iter()
        .map(|worker| worker.join().expect("worker"))
        .filter(|admitted| *admitted)
        .count();

    assert_eq!(successes, 1);
    let snapshot = admission.snapshot().expect("final snapshot");
    assert_eq!(snapshot.reserved_bytes(), 0);
    assert_eq!(snapshot.active_reservations(), 0);
}

#[test]
fn regular_resize_and_component_reservation_still_work() {
    let admission = AdmissionController::new(100).expect("controller");
    let mut reservation = admission.reserve("resizable", 20).expect("reserve");
    reservation.try_resize(40).expect("grow");
    assert_eq!(reservation.bytes(), 40);

    let components = admission
        .reserve_components("components", &[10, 0, 5])
        .expect("components");
    assert_eq!(components.bytes(), 15);

    let snapshot = admission.snapshot().expect("snapshot");
    assert_eq!(snapshot.reserved_bytes(), 55);
    assert_eq!(snapshot.active_reservations(), 2);
}

#[test]
fn unwind_drops_grant_and_partitioned_guards() {
    let admission = AdmissionController::new(100).expect("controller");
    let cloned = admission.clone();

    let result = catch_unwind(AssertUnwindSafe(move || {
        let mut grant = cloned
            .reserve_grant("unwinding startup", &[60, 40])
            .expect("grant");
        let _output = grant
            .take("output", 40)
            .expect("partition")
            .expect("output");
        panic!("exercise grant unwind cleanup");
    }));

    assert!(result.is_err());
    let snapshot = admission.snapshot().expect("unwound");
    assert_eq!(snapshot.reserved_bytes(), 0);
    assert_eq!(snapshot.active_reservations(), 0);
}

#[test]
fn poisoned_accounting_fails_closed_for_grant_and_partition() {
    let admission = AdmissionController::new(100).expect("controller");
    let shared = Arc::clone(&admission.shared);

    let worker = thread::spawn(move || {
        let _guard = shared.state.lock().expect("accounting lock");
        panic!("poison admission accounting");
    });
    assert!(worker.join().is_err());

    assert_eq!(
        admission
            .reserve_grant("after poison", &[1])
            .expect_err("grant must fail closed")
            .kind(),
        ErrorKind::Internal,
    );
    assert_eq!(
        admission
            .snapshot()
            .expect_err("snapshot must fail closed")
            .kind(),
        ErrorKind::Internal,
    );
}
