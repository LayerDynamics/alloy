use super::*;
use std::cell::Cell;

fn parser_limits() -> ParserAllocationLimits {
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
fn limits() -> GenerationLimits {
    GenerationLimits::new(2, 2048, 512).expect("limits")
}
fn text_minimum() -> u64 {
    2048 + 512 + 2 * std::mem::size_of::<String>() as u64
}
fn empty_state(admission: AdmissionController) -> NativeState {
    NativeState {
        active: None,
        pending: None,
        resident: None,
        states: HashMap::new(),
        pool: Vec::new(),
        pool_enabled: false,
        pool_ttl: None,
        tiered: None,
        admission,
        retention: RetentionPolicy {
            retain_model: false,
            retain_cache_buffers: false,
        },
        summary: Arc::new(Mutex::new(Summary::empty(true))),
    }
}
fn admitted(
    admission: &AdmissionController,
    workspace: u64,
) -> (Arc<MemoryReservation>, PendingStartup) {
    let parser = parser_limits();
    let slab = ParserAllocation::required_bytes(parser).expect("slab");
    let output = text_minimum() + slab;
    let (delivery, parser, grant) =
        admit_startup_with_parser_after_reclaim(admission, output, Some(parser), || {
            Ok(vec![output, workspace])
        })
        .expect("startup");
    (
        delivery,
        PendingStartup {
            request_id: RequestId::new("w2-parser-startup").expect("request"),
            parser,
            parser_limits: Some(parser_limits()),
            grant,
        },
    )
}
struct OwningDecoder {
    parser: ParserAllocation,
}
impl TextDecoder for OwningDecoder {
    fn push(&mut self, _: u32) -> Result<String, EngineError> {
        self.parser.workspaces().carry[0] = 7;
        Ok(String::new())
    }
    fn finish(&mut self) -> Result<String, EngineError> {
        Ok(String::new())
    }
}

#[test]
fn selected_parser_must_be_covered_in_addition_to_text_storage() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    assert!(
        validate_parser_output_budget(text_minimum(), limits(), Some(parser_limits())).is_err()
    );
    validate_parser_output_budget(text_minimum() + slab, limits(), Some(parser_limits()))
        .expect("exact cover");
    assert!(
        validate_parser_output_budget(text_minimum() + slab - 1, limits(), Some(parser_limits()))
            .is_err()
    );
    validate_parser_output_budget(text_minimum(), limits(), None).expect("legacy");
}
#[test]
fn one_atomic_grant_covers_delivery_parser_and_remainder() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let admission = AdmissionController::new(text_minimum() + slab + 64).expect("controller");
    let (delivery, pending) = admitted(&admission, 64);
    assert_eq!(delivery.bytes(), text_minimum());
    assert_eq!(
        pending.parser.as_ref().expect("parser").reserved_bytes(),
        slab
    );
    assert_eq!(pending.grant.remaining_bytes(), 64);
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        text_minimum() + slab + 64
    );
    drop(pending);
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        text_minimum()
    );
    drop(delivery);
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(), 0);
    assert_eq!(
        admission.snapshot().expect("ledger").active_reservations(),
        0
    );
}
#[test]
fn pending_finish_releases_slab_and_remainder_but_not_stream_output() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let admission = AdmissionController::new(text_minimum() + slab + 64).expect("controller");
    let (delivery, pending) = admitted(&admission, 64);
    let request = pending.request_id.clone();
    let mut state = empty_state(admission.clone());
    state.pending = Some(pending);
    state.finish(&request).expect("finish abandoned startup");
    assert!(state.pending.is_none());
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        delivery.bytes()
    );
    drop(delivery);
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(), 0);
}
#[test]
fn decoder_receives_exclusive_slab_ownership() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let admission = AdmissionController::new(text_minimum() + slab + 64).expect("controller");
    let (delivery, mut pending) = admitted(&admission, 64);
    let parser = pending
        .take_parser(Some(parser_limits()))
        .expect("transfer")
        .expect("parser");
    assert!(pending.parser.is_none());
    let mut decoder: Box<dyn TextDecoder> = Box::new(OwningDecoder { parser });
    drop(pending);
    drop(delivery);
    assert_eq!(
        admission
            .snapshot()
            .expect("decoder owns slab")
            .reserved_bytes(),
        slab
    );
    decoder.push(0).expect("access storage");
    drop(decoder);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn default_decoder_hook_rejects_slab_without_constructing_legacy_decoder() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let admission = AdmissionController::new(text_minimum() + slab).expect("controller");
    let (delivery, mut pending) = admitted(&admission, 0);
    let parser = pending
        .take_parser(Some(parser_limits()))
        .expect("transfer");
    let called = Cell::new(false);
    let result = legacy_decoder_with_parser(parser, || {
        called.set(true);
        Err(internal("legacy constructor must not run"))
    });
    assert!(result.is_err());
    assert!(!called.get());
    assert_eq!(
        admission
            .snapshot()
            .expect("rejected slab released")
            .reserved_bytes(),
        delivery.bytes()
    );
    drop(pending);
    drop(delivery);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn default_decoder_hook_preserves_none_and_calls_legacy_constructor() {
    let called = Cell::new(false);
    let result = legacy_decoder_with_parser(None, || {
        called.set(true);
        Err(internal("injected legacy constructor failure"))
    });
    assert!(result.is_err());
    assert!(called.get());
}
#[test]
fn wrong_pending_parser_configuration_is_rejected_without_losing_owner() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let admission = AdmissionController::new(text_minimum() + slab).expect("controller");
    let (delivery, mut pending) = admitted(&admission, 0);
    let mut changed = parser_limits();
    changed.calls += 1;
    assert!(pending.take_parser(Some(changed)).is_err());
    assert!(pending.parser.is_some());
    assert!(pending.take_parser(None).is_err());
    assert!(pending.parser.is_some());
    // Equal byte size does not mean equal capacities or interpretation.
    let mut same_size = parser_limits();
    same_size.carry_bytes += 1;
    same_size.decoded_bytes -= 1;
    assert_eq!(
        ParserAllocation::required_bytes(same_size).expect("size"),
        slab
    );
    assert!(pending.take_parser(Some(same_size)).is_err());
    assert!(pending.parser.is_some());
    drop(pending);
    drop(delivery);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn invalid_layout_fails_before_reclaim_and_admission() {
    let admission = AdmissionController::new(4096).expect("controller");
    let called = Cell::new(false);
    let mut parser = parser_limits();
    parser.nodes = usize::MAX;
    let result = admit_startup_with_parser_after_reclaim(&admission, 4096, Some(parser), || {
        called.set(true);
        Ok(vec![4096])
    });
    assert!(result.is_err());
    assert!(!called.get());
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(), 0);
}
#[test]
fn inconsistent_output_component_is_rejected_before_grant_acquisition() {
    let admission = AdmissionController::new(4096).expect("controller");
    let result =
        admit_startup_with_parser_after_reclaim(&admission, 100, None, || Ok(vec![99, 20]));
    assert!(result.is_err());
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(), 0);
}
#[test]
fn insufficient_total_budget_leaves_no_parser_or_output_reservations() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let output = text_minimum() + slab;
    let admission = AdmissionController::new(output + 64 - 1).expect("controller");
    let result =
        admit_startup_with_parser_after_reclaim(&admission, output, Some(parser_limits()), || {
            Ok(vec![output, 64])
        });
    assert!(matches!(
        result,
        Err(EngineError::MemoryLimitExceeded { .. })
    ));
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(), 0);
    assert_eq!(
        admission.snapshot().expect("ledger").active_reservations(),
        0
    );
}
#[test]
fn failed_decoder_construction_drops_the_transferred_slab() {
    let slab = ParserAllocation::required_bytes(parser_limits()).expect("slab");
    let admission = AdmissionController::new(text_minimum() + slab + 64).expect("controller");
    let (delivery, mut pending) = admitted(&admission, 64);
    let parser = pending
        .take_parser(Some(parser_limits()))
        .expect("transfer");
    let construct = |owned: Option<ParserAllocation>| -> Result<Box<dyn TextDecoder>, EngineError> {
        let _owned = owned;
        Err(internal("injected storage-backed constructor failure"))
    };
    assert!(construct(parser).is_err());
    assert_eq!(
        admission.snapshot().expect("ledger").reserved_bytes(),
        text_minimum() + 64
    );
    drop(pending);
    drop(delivery);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
