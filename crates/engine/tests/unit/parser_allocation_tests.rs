use super::*;
use crate::admission::AdmissionController;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn limits() -> ParserAllocationLimits {
    ParserAllocationLimits {
        carry_bytes: 32,
        call_bytes: 128,
        decoded_bytes: 128,
        depth: 8,
        keys: 16,
        nodes: 32,
        proposal_bytes: 128,
        calls: 4,
    }
}

fn allocation(limits: ParserAllocationLimits) -> (AdmissionController, ParserAllocation) {
    let bytes = ParserAllocation::required_bytes(limits).expect("layout");

    let admission = AdmissionController::new(bytes).expect("controller");

    let mut grant = admission
        .reserve_grant("parser startup", &[bytes])
        .expect("grant");

    let allocation = ParserAllocation::from_grant(&mut grant, limits).expect("allocate");

    assert!(grant.is_empty());

    (admission, allocation)
}

fn commit(
    storage: &mut ParserAllocation,
    name: &[u8],
    arguments: &[u8],
) -> Result<usize, ParserAllocationError> {
    {
        let work = storage.workspaces();

        work.decoded[..name.len()].copy_from_slice(name);
        work.call[..arguments.len()].copy_from_slice(arguments);
    }

    storage.commit_validated_call(
        name.len(),
        0..name.len(),
        arguments.len(),
        0..arguments.len(),
    )
}

#[test]
fn exact_layout_is_reserved_and_released_once() {
    let (admission, storage) = allocation(limits());

    assert_eq!(storage.allocated_bytes() as u64, storage.reserved_bytes(),);

    assert_eq!(
        admission.snapshot().expect("snapshot").reserved_bytes(),
        storage.reserved_bytes(),
    );

    assert_eq!(
        admission
            .snapshot()
            .expect("snapshot")
            .active_reservations(),
        1,
    );

    drop(storage);

    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0,);

    assert_eq!(
        admission
            .snapshot()
            .expect("released")
            .active_reservations(),
        0,
    );
}

#[test]
fn one_byte_short_refuses_before_allocation() {
    let bytes = ParserAllocation::required_bytes(limits()).expect("layout");

    let admission = AdmissionController::new(bytes - 1).expect("controller");

    assert!(admission.reserve_grant("parser startup", &[bytes]).is_err());

    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 0,);
}

#[test]
fn wrong_reservation_size_is_rejected_and_released() {
    let bytes = ParserAllocation::required_bytes(limits()).expect("layout");

    let admission = AdmissionController::new(bytes + 1).expect("controller");

    let guard = admission
        .reserve("wrong partition", bytes + 1)
        .expect("reserve");

    assert!(matches!(
        ParserAllocation::from_reservation(guard, limits()),
        Err(ParserAllocationError::ReservationMismatch { .. }),
    ));

    assert_eq!(admission.snapshot().expect("snapshot").reserved_bytes(), 0,);
}

#[test]
fn fixed_regions_have_the_requested_lengths_and_alignment() {
    let (_, mut storage) = allocation(limits());

    let work = storage.workspaces();

    assert_eq!(work.carry.len(), 32);
    assert_eq!(work.call.len(), 128);
    assert_eq!(work.decoded.len(), 128);
    assert_eq!(work.stack.len(), 8);
    assert_eq!(work.keys.len(), 16);
    assert_eq!(work.nodes.len(), 32);

    assert_eq!(
        (work.stack.as_ptr() as usize) % std::mem::align_of::<JsonFrame>(),
        0,
    );

    work.stack[0].object_id = 9;

    assert_eq!(storage.workspaces().stack[0].object_id, 9,);
}

#[test]
fn exact_payload_fit_and_one_byte_over_are_atomic() {
    let mut selected = limits();
    selected.proposal_bytes = 3;

    let (_, mut storage) = allocation(selected);

    commit(&mut storage, b"a", b"{}").expect("exact fit");

    assert_eq!(storage.payload_used(), 3);

    assert!(matches!(
        commit(&mut storage, b"b", b"{}"),
        Err(ParserAllocationError::Capacity {
            region: "proposal payload",
            ..
        }),
    ));

    assert_eq!(storage.payload_used(), 3);
    assert_eq!(storage.proposal_count(), 1);
}

#[test]
fn descriptor_limit_is_checked_without_overwriting_existing_calls() {
    let mut selected = limits();
    selected.calls = 1;

    let (_, mut storage) = allocation(selected);

    commit(&mut storage, b"first", b"{}").expect("first");

    assert!(matches!(
        commit(&mut storage, b"second", b"{}"),
        Err(ParserAllocationError::Capacity {
            region: "proposal descriptors",
            ..
        }),
    ));

    assert_eq!(
        storage.freeze().proposal(0).expect("retained").name,
        "first",
    );
}

#[test]
fn frozen_delivery_keeps_reservation_and_payload_alive() {
    let (admission, mut storage) = allocation(limits());

    commit(&mut storage, b"fs.read", br#"{"path":"file"}"#).expect("commit");

    // Reusable scratch must not back committed proposal views.
    storage.workspaces().call.fill(0);
    storage.workspaces().decoded.fill(0);

    let delivery = storage.freeze();
    let proposal = delivery.proposal(0).expect("proposal");

    assert_eq!(proposal.name, "fs.read");
    assert_eq!(proposal.arguments_json, r#"{"path":"file"}"#);

    assert!(admission.snapshot().expect("retained").reserved_bytes() > 0);

    drop(delivery);

    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0,);
}

#[test]
fn consumer_thread_owns_storage_until_it_drops_delivery() {
    let (admission, mut storage) = allocation(limits());

    commit(&mut storage, b"a", b"{}").expect("commit");

    let delivery = storage.freeze();

    std::thread::spawn(move || {
        assert_eq!(delivery.proposal(0).expect("proposal").name, "a",);
    })
    .join()
    .expect("consumer");

    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0,);
}

#[test]
fn unwind_deallocates_before_releasing_the_guard() {
    let (admission, storage) = allocation(limits());

    let result = catch_unwind(AssertUnwindSafe(move || {
        let _storage = storage;
        panic!("injected parser failure");
    }));

    assert!(result.is_err());

    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0,);
}

#[test]
fn invalid_utf8_and_ranges_leave_commit_counters_unchanged() {
    let (_, mut storage) = allocation(limits());

    assert!(matches!(
        commit(&mut storage, b"a", &[0xff]),
        Err(ParserAllocationError::InvalidUtf8),
    ));

    assert!(matches!(
        storage.commit_validated_call(0, 0..1, 0, 0..0,),
        Err(ParserAllocationError::InvalidRange),
    ));

    assert_eq!(storage.payload_used(), 0);
    assert_eq!(storage.proposal_count(), 0);
}

#[test]
fn allocator_failure_releases_the_preexisting_reservation() {
    unsafe fn fail(_: Layout) -> *mut u8 {
        std::ptr::null_mut()
    }

    let bytes = ParserAllocation::required_bytes(limits()).expect("layout");

    let admission = AdmissionController::new(bytes).expect("controller");

    let guard = admission.reserve("parser", bytes).expect("guard");

    assert!(matches!(
        ParserAllocation::allocate_with(guard, limits(), fail,),
        Err(ParserAllocationError::AllocationFailed { .. }),
    ));

    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0,);
}

#[test]
fn layout_overflow_is_rejected_without_allocating() {
    let mut selected = limits();
    selected.nodes = usize::MAX;

    assert!(matches!(
        ParserAllocation::required_bytes(selected),
        Err(ParserAllocationError::LayoutOverflow),
    ));
}
