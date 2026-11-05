//! Shared memory admission and reservation ownership.
//!
//! This module provides:
//!
//! - An explicitly configured engine-managed memory budget.
//! - Atomic admission against current reservation accounting.
//! - Checked byte and reservation-count arithmetic.
//! - Non-cloneable RAII reservation ownership.
//! - Atomic reservation resizing.
//! - A transferable startup grant that reserves a complete budget once
//!   and partitions that ownership without releasing and reacquiring bytes.
//! - Explicit release with structured errors.
//! - Fail-closed handling of poisoned accounting state.
//!
//! This module does not:
//!
//! - Estimate model memory from checkpoint file size.
//! - Assume nominal quantization equals resident representation.
//! - Allocate model tensors or KV buffers.
//! - Measure process RSS or operating-system memory pressure.
//! - Reserve the scheduler's execution slot.
//! - Automatically deduplicate resident model allocations.
//!
//! The configured budget must already exclude memory that the engine must
//! leave available for the operating system, unrelated allocations, and
//! any control-plane overhead not represented by reservations.
//!
//! Reservations are accounting ownership, not native allocations. Real
//! allocation must occur only after admission, and reservation ownership
//! must remain alive until the corresponding allocations are no longer
//! retained or used.
//!
//! A ReservationGrant is useful for startup operations with multiple
//! resource categories. The whole total is checked and reserved under one
//! lock. Its reservation pieces then transfer to actual resource owners
//! without changing shared accounting. Dropping the grant releases only
//! its unclaimed remainder.

use crate::{error::EngineError, types::MemoryEstimate};
use std::sync::{Arc, Mutex, MutexGuard};

/// A consistent observation of the controller's reservation accounting.
///
/// A snapshot does not reserve memory. Another caller may acquire or
/// release reservations immediately after this value is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionSnapshot {
    budget_bytes: u64,
    reserved_bytes: u64,
    available_bytes: u64,
    active_reservations: usize,
}

impl AdmissionSnapshot {
    /// Configured engine-managed reservation ceiling.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Total bytes currently owned by active reservations.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    /// Unreserved bytes within the configured budget.
    pub fn available_bytes(&self) -> u64 {
        self.available_bytes
    }

    /// Number of independently owned reservation guards.
    pub fn active_reservations(&self) -> usize {
        self.active_reservations
    }

    /// Whether a positive reservation would fit this snapshot.
    ///
    /// This is advisory. Use AdmissionController::reserve to acquire
    /// actual accounting ownership.
    pub fn can_reserve(&self, required_bytes: u64) -> bool {
        required_bytes > 0 && required_bytes <= self.available_bytes
    }
}

/// Shared controller for engine-managed memory admission.
///
/// Clones share the same budget and accounting state. Use the same
/// controller for allocations that compete for the same engine budget.
///
/// The budget is fixed for this controller's lifetime.
#[derive(Clone, Debug)]
pub struct AdmissionController {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    budget_bytes: u64,
    reserved_bytes: u64,
    active_reservations: usize,
}

/// Exclusive accounting ownership of a positive number of bytes.
///
/// This type is intentionally not Clone. Moving it transfers reservation
/// ownership without changing accounting.
///
/// Dropping it releases its accounting reservation. It does not free
/// native tensors, stop model execution, or release a scheduler permit.
///
/// Store the guard with the actual resources it accounts for:
///
/// - Resident-weight reservations belong with the retained loaded model.
/// - Request-local reservations belong with the execution session.
/// - Retained KV reservations belong with the retained cache.
///
/// Do not release a reservation while its corresponding allocations remain
/// resident or are still used by an in-flight native operation.
#[derive(Debug)]
#[must_use = "retain the reservation until its accounted resources are released"]
pub struct MemoryReservation {
    shared: Arc<Shared>,
    operation: &'static str,
    bytes: u64,
    active: bool,
}

/// One atomically admitted startup budget partitioned into reservation
/// pieces.
///
/// Each call to `take` transfers accounting ownership of a portion to the
/// caller without changing total reserved bytes. The returned
/// MemoryReservation must be retained by the resource owner. Any remaining
/// grant bytes are released when the grant is dropped.
///
/// This type is intentionally not Clone.
#[derive(Debug)]
#[must_use = "take and retain each resource reservation or drop the grant"]
pub struct ReservationGrant {
    shared: Arc<Shared>,
    operation: &'static str,
    remaining_bytes: u64,
    active: bool,
}

impl AdmissionController {
    /// Construct a controller with an explicit positive byte budget.
    ///
    /// This does not inspect system RAM or select a budget automatically.
    pub fn new(budget_bytes: u64) -> Result<Self, EngineError> {
        if budget_bytes == 0 {
            return Err(EngineError::invalid_request(
                "budget_bytes",
                "admission budget must be greater than zero",
            ));
        }

        Ok(Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    budget_bytes,
                    reserved_bytes: 0,
                    active_reservations: 0,
                }),
            }),
        })
    }

    /// Observe current accounting without acquiring a reservation.
    ///
    /// This result is advisory and may become stale immediately.
    pub fn snapshot(&self) -> Result<AdmissionSnapshot, EngineError> {
        let state = self.shared.lock_state()?;

        let available_bytes = state
            .budget_bytes
            .checked_sub(state.reserved_bytes)
            .ok_or_else(|| internal_error("admission reservations exceed the configured budget"))?;

        Ok(AdmissionSnapshot {
            budget_bytes: state.budget_bytes,
            reserved_bytes: state.reserved_bytes,
            available_bytes,
            active_reservations: state.active_reservations,
        })
    }

    /// Atomically reserve a positive number of bytes.
    ///
    /// The fit check and accounting update occur under the same mutex.
    /// Competing callers cannot both acquire the same remaining budget.
    ///
    /// This method does not queue or wait for another reservation to be
    /// released. An insufficient budget returns MemoryLimitExceeded.
    ///
    /// The operation label identifies the allocation path in structured
    /// failures. It is not interpreted as a request or artifact identifier.
    pub fn reserve(
        &self,
        operation: &'static str,
        required_bytes: u64,
    ) -> Result<MemoryReservation, EngineError> {
        if required_bytes == 0 {
            return Err(EngineError::invalid_request(
                "required_bytes",
                "memory reservation must be greater than zero",
            ));
        }

        {
            let mut state = self.shared.lock_state()?;

            let available_bytes = state
                .budget_bytes
                .checked_sub(state.reserved_bytes)
                .ok_or_else(|| {
                    internal_error("admission reservations exceed the configured budget")
                })?;

            if required_bytes > available_bytes {
                return Err(EngineError::MemoryLimitExceeded {
                    operation,
                    required_bytes,
                    available_bytes,
                });
            }

            let reserved_bytes = state
                .reserved_bytes
                .checked_add(required_bytes)
                .ok_or_else(|| internal_error("admission reserved-byte arithmetic overflow"))?;

            let active_reservations = state
                .active_reservations
                .checked_add(1)
                .ok_or_else(|| internal_error("admission reservation-count arithmetic overflow"))?;

            // Commit only after all validation and arithmetic succeed.
            state.reserved_bytes = reserved_bytes;
            state.active_reservations = active_reservations;
        }

        Ok(MemoryReservation {
            shared: Arc::clone(&self.shared),
            operation,
            bytes: required_bytes,
            active: true,
        })
    }

    /// Atomically admit the sum of non-overlapping startup requirements.
    ///
    /// Every component must describe a distinct allocation or allowance.
    /// Do not list the same native allocation in multiple components.
    ///
    /// The total is checked and reserved under one lock. A failed admission
    /// leaves all accounting unchanged. The returned grant is exclusive
    /// accounting ownership of the entire total.
    ///
    /// Components may be zero, but the total must be positive.
    pub fn reserve_grant(
        &self,
        operation: &'static str,
        components: &[u64],
    ) -> Result<ReservationGrant, EngineError> {
        let total = components.iter().try_fold(0_u64, |total, &component| {
            total.checked_add(component).ok_or_else(|| {
                EngineError::invalid_request(
                    "memory_components",
                    "memory requirement arithmetic overflow",
                )
            })
        })?;

        if total == 0 {
            return Err(EngineError::invalid_request(
                "memory_components",
                "startup grant total must be positive",
            ));
        }

        {
            let mut state = self.shared.lock_state()?;

            let available_bytes = state
                .budget_bytes
                .checked_sub(state.reserved_bytes)
                .ok_or_else(|| {
                    internal_error("admission reservations exceed the configured budget")
                })?;

            if total > available_bytes {
                return Err(EngineError::MemoryLimitExceeded {
                    operation,
                    required_bytes: total,
                    available_bytes,
                });
            }

            let reserved_bytes = state
                .reserved_bytes
                .checked_add(total)
                .ok_or_else(|| internal_error("admission reserved-byte arithmetic overflow"))?;

            let active_reservations = state
                .active_reservations
                .checked_add(1)
                .ok_or_else(|| internal_error("admission reservation-count arithmetic overflow"))?;

            // One accounting entry represents the whole grant until pieces
            // are transferred. This is the atomic admission point.
            state.reserved_bytes = reserved_bytes;
            state.active_reservations = active_reservations;
        }

        Ok(ReservationGrant {
            shared: Arc::clone(&self.shared),
            operation,
            remaining_bytes: total,
            active: true,
        })
    }

    /// Reserve the full byte requirement recorded by an estimate.
    ///
    /// The estimate's earlier fit decision and available-byte observation
    /// are not used as authorization. Admission is rechecked against this
    /// controller's current accounting.
    ///
    /// This method reserves estimate.reserved_bytes() in full. It does not
    /// subtract already resident weights or deduplicate other reservations.
    /// Callers must avoid counting the same allocation as a new incremental
    /// requirement unless conservative double-accounting is intentional.
    pub fn reserve_estimate(
        &self,
        operation: &'static str,
        estimate: &MemoryEstimate,
    ) -> Result<MemoryReservation, EngineError> {
        self.reserve(operation, estimate.reserved_bytes())
    }

    /// Sum non-overlapping byte requirements and reserve their total.
    ///
    /// Every component must describe an allocation or allowance that is
    /// not already represented by another component in this list.
    ///
    /// Zero-valued components are permitted, but the total must be positive.
    pub fn reserve_components(
        &self,
        operation: &'static str,
        components: &[u64],
    ) -> Result<MemoryReservation, EngineError> {
        let total = components.iter().try_fold(0_u64, |total, &component| {
            total.checked_add(component).ok_or_else(|| {
                EngineError::invalid_request(
                    "memory_components",
                    "memory requirement arithmetic overflow",
                )
            })
        })?;

        self.reserve(operation, total)
    }
}

impl ReservationGrant {
    /// Total bytes still controlled by this grant.
    ///
    /// Bytes already transferred to independent reservations are excluded.
    pub fn remaining_bytes(&self) -> u64 {
        self.remaining_bytes
    }

    /// Whether this grant still owns unpartitioned bytes.
    pub fn is_empty(&self) -> bool {
        self.remaining_bytes == 0
    }

    /// Transfer part of the grant to the resource that will own it.
    ///
    /// Total reserved bytes remain unchanged.
    ///
    /// A partial transfer creates one additional independently owned
    /// reservation. A complete transfer replaces the grant's accounting
    /// entry with the returned reservation.
    ///
    /// A zero-byte request returns None without changing ownership.
    ///
    /// On failure, the grant and shared accounting remain unchanged.
    pub fn take(
        &mut self,
        operation: &'static str,
        bytes: u64,
    ) -> Result<Option<MemoryReservation>, EngineError> {
        if bytes == 0 {
            return Ok(None);
        }

        if !self.active {
            return Err(internal_error(
                "cannot partition an inactive reservation grant",
            ));
        }

        let remaining_bytes = self.remaining_bytes.checked_sub(bytes).ok_or_else(|| {
            EngineError::invalid_request(
                "grant_partition",
                "requested bytes exceed the remaining startup grant",
            )
        })?;

        let shared = Arc::clone(&self.shared);

        {
            let mut state = self.shared.lock_state()?;

            if state.reserved_bytes > state.budget_bytes {
                return Err(internal_error(
                    "admission reservations exceed the configured budget",
                ));
            }

            if self.remaining_bytes > state.reserved_bytes || state.active_reservations == 0 {
                return Err(internal_error(
                    "startup grant exceeds shared reservation accounting",
                ));
            }

            let active_reservations = if remaining_bytes == 0 {
                // The returned reservation replaces the grant's entry.
                state.active_reservations
            } else {
                // Both the remainder and returned reservation own bytes.
                state.active_reservations.checked_add(1).ok_or_else(|| {
                    internal_error("admission reservation-count arithmetic overflow")
                })?
            };

            // All fallible checks completed. Reserved bytes do not change.
            state.active_reservations = active_reservations;
            self.remaining_bytes = remaining_bytes;
            self.active = remaining_bytes != 0;
        }

        Ok(Some(MemoryReservation {
            shared,
            operation,
            bytes,
            active: true,
        }))
    }

    /// Explicitly release the grant's unpartitioned remainder.
    ///
    /// Already-transferred reservations remain owned by their receivers.
    pub fn release(mut self) -> Result<(), EngineError> {
        if !self.active {
            return Ok(());
        }

        self.shared.release(self.remaining_bytes)?;
        self.remaining_bytes = 0;
        self.active = false;
        Ok(())
    }
}

impl Drop for ReservationGrant {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        if self.shared.release(self.remaining_bytes).is_ok() {
            self.remaining_bytes = 0;
            self.active = false;
        }
    }
}

impl MemoryReservation {
    /// Number of bytes owned by this reservation.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Allocation-path label associated with this reservation.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Atomically change the size of this reservation.
    ///
    /// Growing a reservation checks and acquires additional budget before
    /// the caller performs the additional allocation.
    ///
    /// Shrinking is permitted only after the caller has released the
    /// corresponding resources. This method does not free them itself.
    ///
    /// A replacement allocation may temporarily require both old and new
    /// buffers. Reserve that peak requirement, not merely the final size.
    ///
    /// On failure, this guard and shared accounting remain unchanged.
    pub fn try_resize(&mut self, new_bytes: u64) -> Result<(), EngineError> {
        if new_bytes == 0 {
            return Err(EngineError::invalid_request(
                "new_bytes",
                "reservation size must be greater than zero; release it instead",
            ));
        }

        if !self.active {
            return Err(internal_error(
                "cannot resize an inactive memory reservation",
            ));
        }

        let mut state = self.shared.lock_state()?;

        let available_bytes = state
            .budget_bytes
            .checked_sub(state.reserved_bytes)
            .ok_or_else(|| internal_error("admission reservations exceed the configured budget"))?;

        if new_bytes > self.bytes {
            let additional_bytes = new_bytes
                .checked_sub(self.bytes)
                .ok_or_else(|| internal_error("reservation growth arithmetic underflow"))?;

            if additional_bytes > available_bytes {
                return Err(EngineError::MemoryLimitExceeded {
                    operation: self.operation,
                    required_bytes: additional_bytes,
                    available_bytes,
                });
            }
        }

        let other_reserved_bytes =
            state
                .reserved_bytes
                .checked_sub(self.bytes)
                .ok_or_else(|| {
                    internal_error("reservation size exceeds shared reserved-byte accounting")
                })?;

        let reserved_bytes = other_reserved_bytes
            .checked_add(new_bytes)
            .ok_or_else(|| internal_error("reservation resize arithmetic overflow"))?;

        if reserved_bytes > state.budget_bytes {
            return Err(internal_error(
                "validated reservation resize exceeds the budget",
            ));
        }

        state.reserved_bytes = reserved_bytes;
        self.bytes = new_bytes;
        Ok(())
    }

    /// Explicitly release this reservation and report accounting errors.
    ///
    /// Native resources must already have stopped using this reservation's
    /// budget. This method does not stop execution or free native memory.
    pub fn release(mut self) -> Result<(), EngineError> {
        if !self.active {
            return Err(internal_error("memory reservation was already released"));
        }

        self.shared.release(self.bytes)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        // Destructors cannot return a structured error. If accounting is
        // poisoned, leave it inaccessible rather than recover it and risk
        // admitting allocations against inconsistent state.
        if self.shared.release(self.bytes).is_ok() {
            self.active = false;
        }
    }
}

impl Shared {
    fn lock_state(&self) -> Result<MutexGuard<'_, State>, EngineError> {
        self.state
            .lock()
            .map_err(|_| internal_error("memory admission accounting mutex is poisoned"))
    }

    fn release(&self, bytes: u64) -> Result<(), EngineError> {
        let mut state = self.lock_state()?;

        let reserved_bytes = state.reserved_bytes.checked_sub(bytes).ok_or_else(|| {
            internal_error("memory reservation release would underflow byte accounting")
        })?;

        let active_reservations = state.active_reservations.checked_sub(1).ok_or_else(|| {
            internal_error("memory reservation release would underflow reservation count")
        })?;

        // Commit together after both checked operations succeed.
        state.reserved_bytes = reserved_bytes;
        state.active_reservations = active_reservations;
        Ok(())
    }
}

fn internal_error(message: &'static str) -> EngineError {
    EngineError::Internal {
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
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
                ..
            } => {
                assert_eq!(required_bytes, 1);
                assert_eq!(available_bytes, 0);
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
}
