use super::*;
use crate::{admission::AdmissionController,
    parser_allocation::{ParserAllocation, ParserAllocationLimits}};
use std::alloc::Layout;

fn text_limits() -> GenerationLimits {
    GenerationLimits::new(3, 4096, 1024).expect("text limits")
}
fn parser_limits() -> ParserAllocationLimits {
    ParserAllocationLimits { carry_bytes:3, call_bytes:5, decoded_bytes:7,
        depth:2, keys:3, nodes:5, proposal_bytes:11, calls:3 }
}
#[test]
fn text_only_total_matches_the_previous_output_contract() {
    let limits=text_limits();
    let expected=4096+1024+3*std::mem::size_of::<String>() as u64;
    let plan=output_memory_plan_for(limits,None,&[]).expect("plan");
    assert_eq!(plan.total_bytes(),expected);
    assert_eq!(output_bytes(limits).expect("legacy caller"),expected);
    assert_eq!(plan.non_parser_bytes(),expected);
    assert_eq!(plan.parser_slab_bytes(),0);
}
#[test]
fn parser_slab_uses_the_actual_layout_including_padding_once() {
    let parser=parser_limits();
    let slab=ParserAllocation::required_bytes(parser).expect("layout");
    let plan=output_memory_plan_for(text_limits(),Some(parser),&[]).expect("plan");
    assert_eq!(plan.parser_slab_bytes(),slab);
    assert_eq!(plan.total_bytes(),plan.non_parser_bytes()+slab);
    if std::mem::size_of::<usize>()==8 && std::mem::align_of::<usize>()==8 {
        assert_eq!(slab,424);
    }
}
#[test]
fn separate_live_layouts_are_added_without_arbitrary_alignment_roundup() {
    let layouts=[Layout::from_size_align(3,8).expect("layout"),
        Layout::array::<u64>(7).expect("array")];
    let plan=output_memory_plan_for(text_limits(),Some(parser_limits()),&layouts).expect("plan");
    assert_eq!(plan.additional_layout_bytes(),59);
    assert_eq!(plan.non_parser_bytes(),output_bytes(text_limits()).expect("legacy")+59);
}
#[test]
fn two_distinct_equal_layouts_are_both_counted() {
    let layout=Layout::array::<u64>(7).expect("array");
    let plan=output_memory_plan_for(text_limits(),Some(parser_limits()),&[layout,layout]).expect("plan");
    assert_eq!(plan.additional_layout_bytes(),112);
}
#[test]
fn zero_parser_limits_fail_instead_of_falling_back_to_text_only() {
    let mut limits=parser_limits(); limits.nodes=0;
    assert!(output_memory_plan_for(text_limits(),Some(limits),&[]).is_err());
}
#[test]
fn impossible_parser_layout_fails_before_admission() {
    let mut limits=parser_limits(); limits.nodes=usize::MAX;
    let admission=AdmissionController::new(4096).expect("admission");
    assert!(output_memory_plan_for(text_limits(),Some(limits),&[]).is_err());
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(),0);
}
#[test]
fn sum_and_product_fail_closed_on_overflow() {
    assert!(sum(&[u64::MAX,1]).is_err());
    assert!(product(u64::MAX,2).is_err());
}
#[test]
fn startup_plan_partitions_parser_and_non_parser_without_double_charge() {
    let limits=parser_limits();
    let plan=output_memory_plan_for(text_limits(),Some(limits),&[]).expect("plan");
    let admission=AdmissionController::new(plan.total_bytes()).expect("admission");
    let mut grant=admission.reserve_grant("output startup",&[plan.total_bytes()]).expect("grant");
    let text=grant.take("text and delivery",plan.non_parser_bytes()).expect("take").expect("positive");
    let parser=ParserAllocation::from_grant(&mut grant,limits).expect("parser allocation");
    assert!(grant.is_empty());
    assert_eq!(parser.reserved_bytes(),plan.parser_slab_bytes());
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(),plan.total_bytes());
    drop(grant); drop(text);
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(),plan.parser_slab_bytes());
    let delivered=parser.freeze();
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(),plan.parser_slab_bytes());
    drop(delivered);
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(),0);
    assert_eq!(admission.snapshot().expect("ledger").active_reservations(),0);
}
#[test]
fn one_byte_short_total_refuses_the_entire_output_grant() {
    let plan=output_memory_plan_for(text_limits(),Some(parser_limits()),&[]).expect("plan");
    let admission=AdmissionController::new(plan.total_bytes()-1).expect("admission");
    assert!(admission.reserve_grant("output startup",&[plan.total_bytes()]).is_err());
    assert_eq!(admission.snapshot().expect("ledger").reserved_bytes(),0);
    assert_eq!(admission.snapshot().expect("ledger").active_reservations(),0);
}
