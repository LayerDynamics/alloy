use super::*;
use crate::{admission::AdmissionController, parser_allocation::ParserAllocationLimits};
fn limits() -> ParserAllocationLimits {
    ParserAllocationLimits {
        carry_bytes: 32,
        call_bytes: 4096,
        decoded_bytes: 4096,
        depth: 32,
        keys: 256,
        nodes: 512,
        proposal_bytes: 8192,
        calls: 8,
    }
}
fn parser(limits: ParserAllocationLimits) -> (AdmissionController, SlabParser) {
    let bytes = ParserAllocation::required_bytes(limits).expect("size");
    let admission = AdmissionController::new(bytes).expect("admission");
    let mut grant = admission.reserve_grant("slab", &[bytes]).expect("grant");
    let storage = ParserAllocation::from_grant(&mut grant, limits).expect("storage");
    (admission, SlabParser::new(storage, 65536).expect("parser"))
}
const CALL: &str = r#"<tool_call>{"name":"fs.read","arguments":{"path":"résumé.txt","nested":[true,null,7]}}</tool_call>"#;
#[test]
fn streamed_text_and_complete_calls_use_the_admitted_slab() {
    let (admission, mut p) = parser(limits());
    let mut text = String::new();
    p.push("lead<too", |s| {
        text.push_str(s);
        Ok(())
    })
    .expect("prefix");
    assert_eq!(text, "lead");
    p.push(
        "l_call>{\"name\":\"fs.read\",\"arguments\":{}}</tool_call>tail",
        |s| {
            text.push_str(s);
            Ok(())
        },
    )
    .expect("rest");
    let owner = p.finish().expect("finish");
    assert_eq!(text, "leadtail");
    assert_eq!(owner.proposal_count(), 1);
    assert_eq!(owner.proposal(0).expect("proposal").name, "fs.read");
    assert!(admission.snapshot().expect("live").reserved_bytes() > 0);
    drop(owner);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn every_utf8_chunk_boundary_preserves_raw_arguments_and_text() {
    let input = format!("before{CALL}after");
    for split in (0..=input.len()).filter(|i| input.is_char_boundary(*i)) {
        let (_, mut p) = parser(limits());
        let mut text = String::new();
        p.push(&input[..split], |s| {
            text.push_str(s);
            Ok(())
        })
        .expect("first");
        p.push(&input[split..], |s| {
            text.push_str(s);
            Ok(())
        })
        .expect("second");
        let owner = p.finish().expect("finish");
        assert_eq!(text, "beforeafter");
        let call = owner.proposal(0).expect("call");
        assert_eq!(call.name, "fs.read");
        assert_eq!(
            call.arguments_json,
            r#"{"path":"résumé.txt","nested":[true,null,7]}"#
        );
    }
}
#[test]
fn escaped_equivalent_duplicate_keys_are_rejected() {
    for call in [
        r#"<tool_call>{"name":"a","arguments":{"x":1,"\u0078":2}}</tool_call>"#,
        r#"<tool_call>{"name":"a","na\u006de":"b","arguments":{}}</tool_call>"#,
    ] {
        let (admission, mut p) = parser(limits());
        assert!(p.push(call, |_| Ok(())).is_err());
        assert!(p.finish().is_err());
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    }
}
#[test]
fn unicode_name_and_surrogate_pair_strings_validate_without_value_trees() {
    let (_, mut p) = parser(limits());
    p.push(
        r#"<tool_call>{"name":"fs.\u0072ead","arguments":{"emoji":"\uD83D\uDE00"}}</tool_call>"#,
        |_| Ok(()),
    )
    .expect("call");
    let owner = p.finish().expect("finish");
    assert_eq!(owner.proposal(0).expect("call").name, "fs.read");
}
#[test]
fn malformed_later_call_poisoning_discards_all_proposals() {
    let (admission, mut p) = parser(limits());
    p.push(CALL, |_| Ok(())).expect("first");
    assert!(
        p.push("<tool_call>{broken}</tool_call>", |_| Ok(()))
            .is_err()
    );
    assert!(p.finish().is_err());
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn key_node_depth_and_payload_limits_fail_closed() {
    for which in 0..4 {
        let mut l = limits();
        match which {
            0 => l.keys = 1,
            1 => l.nodes = 1,
            2 => l.depth = 1,
            _ => l.proposal_bytes = 1,
        }
        let (admission, mut p) = parser(l);
        assert!(p.push(CALL, |_| Ok(())).is_err());
        assert!(p.finish().is_err());
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    }
}
#[test]
fn invalid_envelopes_and_json_number_grammar_are_rejected() {
    for call in [
        r#"{"name":"a","arguments":{"n":01}}"#,
        r#"{"name":"a","arguments":{"n":1e}}"#,
        r#"{"name":"a","arguments":[],"approved":true}"#,
        r#"{"name":"a","arguments":{"x":"\uD800"}}"#,
    ] {
        let (_, mut p) = parser(limits());
        assert!(
            p.push(&format!("<tool_call>{call}</tool_call>"), |_| Ok(()))
                .is_err()
        );
    }
}
#[test]
fn incomplete_delimiter_and_call_fail_at_finish() {
    for text in ["<tool_", "<tool_call>{"] {
        let (_, mut p) = parser(limits());
        p.push(text, |_| Ok(())).expect("partial");
        assert!(p.finish().is_err());
    }
}
#[test]
fn call_body_exact_fit_and_one_byte_over_fail_without_leaking_storage() {
    let body = CALL
        .strip_prefix("<tool_call>")
        .expect("open")
        .strip_suffix("</tool_call>")
        .expect("close");
    let mut exact = limits();
    exact.call_bytes = body.len();
    let (admission, mut p) = parser(exact);
    p.push(CALL, |_| Ok(())).expect("exact body fit");
    let owner = p.finish().expect("finish");
    assert_eq!(owner.proposal_count(), 1);
    drop(owner);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    let mut short = limits();
    short.call_bytes = body.len() - 1;
    let (admission, mut p) = parser(short);
    assert!(p.push(CALL, |_| Ok(())).is_err());
    assert!(p.finish().is_err());
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn a_second_call_over_descriptor_capacity_invalidates_the_batch() {
    let mut selected = limits();
    selected.calls = 1;
    let (admission, mut p) = parser(selected);
    p.push(CALL, |_| Ok(())).expect("first call");
    assert!(p.push(CALL, |_| Ok(())).is_err());
    assert!(p.finish().is_err());
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn cumulative_output_limit_counts_previously_emitted_text() {
    let capacities = limits();
    let bytes = ParserAllocation::required_bytes(capacities).expect("size");
    let admission = AdmissionController::new(bytes).expect("admission");
    let mut grant = admission.reserve_grant("slab", &[bytes]).expect("grant");
    let storage = ParserAllocation::from_grant(&mut grant, capacities).expect("storage");
    let mut p = SlabParser::new(storage, 4).expect("parser");
    p.push("1234", |_| Ok(())).expect("exact output");
    assert!(p.push("5", |_| Ok(())).is_err());
    assert!(p.finish().is_err());
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn failed_text_delivery_poisons_the_parser() {
    let (admission, mut p) = parser(limits());
    assert!(
        p.push("text", |_| Err(bad("injected consumer failure")))
            .is_err()
    );
    assert!(p.push(CALL, |_| Ok(())).is_err());
    assert!(p.finish().is_err());
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
