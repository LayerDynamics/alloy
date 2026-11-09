use super::*;
use crate::admission::AdmissionController;
use std::{sync::mpsc, time::Duration};

fn odd_limits() -> ParserAllocationLimits {
    ParserAllocationLimits {
        carry_bytes: 3,
        call_bytes: 5,
        decoded_bytes: 7,
        depth: 2,
        keys: 3,
        nodes: 5,
        proposal_bytes: 11,
        calls: 3,
    }
}
fn storage(limits: ParserAllocationLimits) -> (AdmissionController, ParserAllocation) {
    let bytes = ParserAllocation::required_bytes(limits).expect("size");
    let admission = AdmissionController::new(bytes).expect("admission");
    let mut grant = admission.reserve_grant("startup", &[bytes]).expect("grant");
    let storage = ParserAllocation::from_grant(&mut grant, limits).expect("storage");
    (admission, storage)
}
#[test]
fn odd_regions_account_for_every_alignment_byte() {
    let report = ParserAllocation::layout_report(odd_limits()).expect("report");
    assert_eq!(
        report.region_bytes + report.inter_region_padding + report.trailing_padding,
        report.total_bytes
    );
    assert_eq!(report.total_bytes % report.alignment, 0);
    if std::mem::size_of::<usize>() == 8 && std::mem::align_of::<usize>() == 8 {
        let offsets: [usize; 8] = std::array::from_fn(|i| report.regions[i].offset);
        assert_eq!(offsets, [0, 3, 8, 16, 80, 152, 312, 328]);
        assert_eq!(report.region_bytes, 418);
        assert_eq!(report.inter_region_padding, 6);
        assert_eq!(report.trailing_padding, 0);
        assert_eq!(report.total_bytes, 424);
    }
}
#[test]
fn all_regions_are_disjoint_for_a_grid_of_odd_capacities() {
    for carry in 1..=9 {
        for call in 1..=9 {
            for payload in 1..=9 {
                let mut limits = odd_limits();
                limits.carry_bytes = carry;
                limits.call_bytes = call;
                limits.proposal_bytes = payload;
                let report = ParserAllocation::layout_report(limits).expect("report");
                let mut end = 0;
                for region in report.regions {
                    assert!(region.offset >= end);
                    assert_eq!(region.offset % region.alignment, 0);
                    assert_eq!(region.preceding_padding, region.offset - end);
                    end = region.offset + region.bytes;
                }
                assert_eq!(end + report.trailing_padding, report.total_bytes);
            }
        }
    }
}
#[test]
fn actual_workspace_addresses_match_the_report() {
    let limits = odd_limits();
    let report = ParserAllocation::layout_report(limits).expect("report");
    let (_, mut allocation) = storage(limits);
    let work = allocation.workspaces();
    let base = work.carry.as_ptr() as usize;
    let addresses = [
        base,
        work.call.as_ptr() as usize,
        work.decoded.as_ptr() as usize,
        work.stack.as_ptr() as usize,
        work.keys.as_ptr() as usize,
        work.nodes.as_ptr() as usize,
    ];
    for (index, address) in addresses.into_iter().enumerate() {
        assert_eq!(address - base, report.regions[index].offset);
        assert_eq!(address % report.regions[index].alignment, 0);
    }
    work.carry.fill(1);
    work.call.fill(2);
    work.decoded.fill(3);
    work.stack[0].object_id = 4;
    work.keys[0].object_id = 5;
    work.nodes[0].kind = 6;
    assert!(work.carry.iter().all(|byte| *byte == 1));
    assert!(work.call.iter().all(|byte| *byte == 2));
    assert!(work.decoded.iter().all(|byte| *byte == 3));
    assert_eq!(work.stack[0].object_id, 4);
    assert_eq!(work.keys[0].object_id, 5);
    assert_eq!(work.nodes[0].kind, 6);
}
#[test]
fn independent_audit_rejects_overlap_and_misalignment() {
    let original = Plan::new(odd_limits()).expect("plan");
    let mut overlap = original;
    overlap.call.offset = 0;
    assert!(matches!(
        overlap.validate_layout(),
        Err(ParserAllocationError::InvalidLayout)
    ));
    let mut misaligned = original;
    misaligned.stack.offset += 1;
    assert!(matches!(
        misaligned.validate_layout(),
        Err(ParserAllocationError::InvalidLayout)
    ));
    let mut wrong_count = original;
    wrong_count.nodes.count = usize::MAX;
    assert!(wrong_count.validate_layout().is_err());
}
#[test]
fn atomic_startup_partition_does_not_double_reserve_parser_bytes() {
    let parser = ParserAllocation::required_bytes(odd_limits()).expect("size");
    let model = 1024u64;
    let workspace = 512u64;
    let total = parser + model + workspace;
    let admission = AdmissionController::new(total).expect("controller");
    let mut grant = admission
        .reserve_grant("full startup", &[model, parser, workspace])
        .expect("grant");
    let model_owner = grant
        .take("model", model)
        .expect("model")
        .expect("positive");
    let allocation = ParserAllocation::from_grant(&mut grant, odd_limits()).expect("parser");
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        total
    );
    drop(grant);
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        model + parser
    );
    drop(model_owner);
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        parser
    );
    drop(allocation);
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(), 0);
    assert_eq!(
        admission.snapshot().expect("ledger").active_reservations(),
        0
    );
}
#[test]
fn closing_admissions_after_grant_does_not_reacquire_budget() {
    let parser = ParserAllocation::required_bytes(odd_limits()).expect("size");
    let admission = AdmissionController::new(parser).expect("controller");
    let mut grant = admission
        .reserve_grant("startup", &[parser])
        .expect("grant");
    admission.set_admissions_paused(true);
    let allocation =
        ParserAllocation::from_grant(&mut grant, odd_limits()).expect("partition without reserve");
    assert_eq!(allocation.reserved_bytes(), parser);
    drop(allocation);
    drop(grant);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn bounded_handoff_keeps_the_slab_charged_until_consumer_drop() {
    let (admission, mut allocation) = storage(odd_limits());
    {
        let work = allocation.workspaces();
        work.decoded[..1].copy_from_slice(b"a");
        work.call[..2].copy_from_slice(b"{}");
    }
    allocation
        .commit_validated_call(1, 0..1, 2, 0..2)
        .expect("commit");
    let bytes = allocation.reserved_bytes();
    let (sender, receiver) = mpsc::sync_channel(1);
    let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
    let (release_sender, release_receiver) = mpsc::sync_channel(1);
    let consumer = std::thread::spawn(move || {
        let frozen: FrozenParserAllocation = receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("delivery");
        assert_eq!(frozen.proposal(0).expect("proposal").arguments_json, "{}");
        ready_sender.send(()).expect("ready");
        release_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("release");
        drop(frozen);
    });
    sender.send(allocation.freeze()).expect("handoff");
    drop(sender);
    ready_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("consumer owns slab");
    assert_eq!(
        admission.snapshot().expect("retained").reserved_bytes(),
        bytes
    );
    release_sender.send(()).expect("release consumer");
    consumer.join().expect("consumer");
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn dropping_a_queued_owner_releases_storage_without_consumption() {
    let (admission, allocation) = storage(odd_limits());
    let (sender, receiver) = mpsc::sync_channel(1);
    sender.send(allocation.freeze()).expect("queue");
    assert!(admission.snapshot().expect("retained").reserved_bytes() > 0);
    drop(receiver);
    drop(sender);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    assert_eq!(
        admission
            .snapshot()
            .expect("released")
            .active_reservations(),
        0
    );
}
