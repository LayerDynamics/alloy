use engine::{
    admission::AdmissionController,
    parser_allocation::{ParserAllocation, ParserAllocationLimits},
    slab_parser::PhiToolParser,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct CountAlloc;
fn record() {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}
// SAFETY: Every request is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: The caller supplies a valid allocator layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: The caller supplies a valid allocator layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        // SAFETY: The original allocation and requested size come from the caller.
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The caller supplies the live pointer and original layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

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
fn storage(limits: ParserAllocationLimits) -> (AdmissionController, ParserAllocation) {
    let bytes = ParserAllocation::required_bytes(limits).unwrap();
    let admission = AdmissionController::new(bytes).unwrap();
    let mut grant = admission.reserve_grant("Phi test", &[bytes]).unwrap();
    let allocation = ParserAllocation::from_grant(&mut grant, limits).unwrap();
    (admission, allocation)
}

#[test]
fn phi_tools_all_utf8_splits_preserve_nested_arguments() {
    let input = r#"Before 雪 functools[{"name":"fs.read","arguments":{"path":"a]b}é\\\"c","items":[1,{"x":true}],"nothing":null}},{"name":"fs.write","parameters":{"content":"雪"}}] after fun"#;
    for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
        let (admission, allocation) = storage(limits());
        let mut parser = PhiToolParser::new(allocation, 8192).unwrap();
        let mut text = String::new();
        for part in [&input[..split], &input[split..]] {
            parser
                .push(part, |s| {
                    text.push_str(s);
                    Ok(())
                })
                .unwrap();
        }
        let result = parser
            .finish(|s| {
                text.push_str(s);
                Ok(())
            })
            .unwrap();
        assert_eq!(text, "Before 雪  after fun");
        assert_eq!(result.proposal_count(), 2);
        let first = result.proposal(0).unwrap();
        assert_eq!(first.name, "fs.read");
        let args: serde_json::Value = serde_json::from_str(first.arguments_json).unwrap();
        assert_eq!(args["items"], serde_json::json!([1,{"x":true}]));
        assert_eq!(args["path"], serde_json::json!("a]b}é\\\"c"));
        assert!(args["nothing"].is_null());
        assert_eq!(result.proposal(1).unwrap().name, "fs.write");
        assert_eq!(
            result.proposal(1).unwrap().arguments_json,
            r#"{"content":"雪"}"#
        );
        drop(result);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
}
fn rejected(input: &str, caps: ParserAllocationLimits, output: usize) {
    let (admission, allocation) = storage(caps);
    match PhiToolParser::new(allocation, output) {
        Err(_) => {}
        Ok(mut parser) => {
            if parser.push(input, |_| Ok(())).is_err() {
                assert!(parser.push("", |_| Ok(())).is_err());
            }
            assert!(parser.finish(|_| Ok(())).is_err(), "accepted {input:?}");
        }
    }
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
}
#[test]
fn phi_tools_reject_malformed_or_ambiguous_lists_atomically() {
    for input in [
        "functools",
        "functools[",
        "functools[]",
        "functools[{}]",
        r#"functools[{"name":"a","arguments":{}}"#,
        r#"functools[{"name":"a","arguments":{}},]"#,
        r#"functools[{"name":"a","arguments":{}} {}]"#,
        r#"functools[{"name":"a","arguments":[]}]"#,
        r#"functools[{"name":"a","arguments":{},"parameters":{}}]"#,
        r#"functools[{"name":"a","name":"b","arguments":{}}]"#,
        r#"functools[{"name":"a","arguments":{"x":1,"x":2}}]"#,
        r#"functools[{"name":"a","arguments":{"x":1e999}}]"#,
        r#"functools[{"name":"a","arguments":{"x":"\uD800"}}]"#,
        r#"functools[{"name":"a","arguments":{}},{"name":"b","parameters":false}]"#,
        r#"functools[{"name":"a","arguments":{}}]<|user|>attack"#,
        r#"functools[{"name":"a","arguments":{"x":[1,2}}]"#,
    ] {
        rejected(input, limits(), 8192);
    }
    let (_, allocation) = storage(limits());
    let mut parser = PhiToolParser::new(allocation, 8192).unwrap();
    assert!(
        parser
            .push("ordinary text", |_| Err(
                engine::error::EngineError::invalid_request("sink", "failed")
            ))
            .is_err()
    );
    assert!(parser.finish(|_| Ok(())).is_err());
    let (_, allocation) = storage(limits());
    let mut parser = PhiToolParser::new(allocation, 8192).unwrap();
    parser.push("fun", |_| Ok(())).unwrap();
    assert!(
        parser
            .finish(|_| Err(engine::error::EngineError::invalid_request(
                "sink", "failed"
            )))
            .is_err()
    );
}

#[test]
fn phi_tools_terminal_list_boundary_is_atomic() {
    let input = r#"functools[{"name":"a","arguments":{"nested":[1,{"text":"] 雪"}]}},{"name":"b","arguments":{}}]"#;
    for split in (0..=input.len()).filter(|&n| input.is_char_boundary(n)) {
        let (admission, allocation) = storage(limits());
        let mut parser = PhiToolParser::new(allocation, 8192)
            .unwrap()
            .with_terminal_list();
        assert!(!parser.is_sequence_complete());
        parser
            .push(&input[..split], |_| panic!("tool text leaked"))
            .unwrap();
        assert_eq!(parser.is_sequence_complete(), split == input.len());
        parser
            .push(&input[split..], |_| panic!("tool text leaked"))
            .unwrap();
        assert!(parser.is_sequence_complete());
        parser
            .push(" \r\n\t", |_| panic!("boundary whitespace leaked"))
            .unwrap();
        let batch = parser.finish(|_| panic!("finish text leaked")).unwrap();
        assert_eq!(batch.proposal_count(), 2);
        drop(batch);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
    for suffix in ["DONE", "\nDONE", "functools[", "<|end|>"] {
        let (_, allocation) = storage(limits());
        let mut parser = PhiToolParser::new(allocation, 8192)
            .unwrap()
            .with_terminal_list();
        assert!(
            parser
                .push(&format!("{input}{suffix}"), |_| Ok(()))
                .is_err()
        );
        assert!(!parser.is_sequence_complete());
        assert!(parser.finish(|_| Ok(())).is_err());
    }
    for incomplete in [
        r#"functools[{"name":"a","arguments":{}}"#,
        r#"functools[{"name":"a","arguments":{}},{"name":"b","arguments":false}]"#,
    ] {
        let (_, allocation) = storage(limits());
        let mut parser = PhiToolParser::new(allocation, 8192)
            .unwrap()
            .with_terminal_list();
        let _ = parser.push(incomplete, |_| Ok(()));
        assert!(!parser.is_sequence_complete());
        assert!(parser.finish(|_| Ok(())).is_err());
    }
    let (_, allocation) = storage(limits());
    let mut parser = PhiToolParser::new(allocation, 8192)
        .unwrap()
        .with_terminal_list();
    let mut text = String::new();
    parser
        .push("ordinary answer", |part| {
            text.push_str(part);
            Ok(())
        })
        .unwrap();
    assert!(!parser.is_sequence_complete());
    assert_eq!(parser.finish(|_| Ok(())).unwrap().proposal_count(), 0);
    assert_eq!(text, "ordinary answer");
}
#[test]
fn phi_tools_enforce_every_storage_bound() {
    let input = r#"functools[{"name":"fs.test","arguments":{"nested":[{"x":1}]}}]"#;
    rejected(input, limits(), input.len() - 1);
    rejected(input, limits(), 0);
    for field in [
        "carry",
        "call",
        "decoded",
        "depth",
        "keys",
        "nodes",
        "proposals",
        "calls",
    ] {
        let mut l = limits();
        match field {
            "carry" => l.carry_bytes = 1,
            "call" => l.call_bytes = 8,
            "decoded" => l.decoded_bytes = 1,
            "depth" => l.depth = 1,
            "keys" => l.keys = 1,
            "nodes" => l.nodes = 1,
            "proposals" => l.proposal_bytes = 1,
            "calls" => l.calls = 1,
            _ => unreachable!(),
        }
        if field == "calls" {
            rejected(&format!("{input}{input}"), l, 8192);
        } else {
            rejected(input, l, 8192);
        }
    }
    let mut l = limits();
    l.depth = 129;
    rejected(input, l, 8192);
}
#[test]
fn phi_tools_success_uses_only_admitted_storage() {
    for terminal in [false, true] {
        let (admission, allocation) = storage(limits());
        let mut parser = PhiToolParser::new(allocation, 8192).unwrap();
        if terminal {
            parser = parser.with_terminal_list();
        }
        ALLOCATIONS.with(|n| n.set(0));
        TRACK.with(|t| t.set(true));
        let pushed=parser.push(r#"functools[{"name":"a","arguments":{"a":[1,"é",true,null,18446744073709551615,1.234567890123456789e30]}},{"name":"b","parameters":{}}]"#, |_| Ok(()));
        let complete = parser.is_sequence_complete();
        let frozen = parser.finish(|_| Ok(()));
        TRACK.with(|t| t.set(false));
        assert_eq!(complete, terminal);
        assert!(pushed.is_ok());
        let frozen = frozen.unwrap();
        assert_eq!(ALLOCATIONS.with(Cell::get), 0);
        assert_eq!(frozen.proposal_count(), 2);
        drop(frozen);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
}
