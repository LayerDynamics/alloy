use super::*;
use crate::{admission::AdmissionController, parser_allocation::ParserAllocationLimits};
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

struct Schema;
impl XmlToolSchema for Schema {
    fn validate_tool(&self, name: &str) -> Result<(), EngineError> {
        if name == "fs.test" {
            Ok(())
        } else {
            Err(bad("unknown tool"))
        }
    }
    fn parameter_type(
        &self,
        tool: &str,
        parameter: &str,
        _: &str,
    ) -> Result<XmlParameterType, EngineError> {
        self.validate_tool(tool)?;
        match parameter {
            "s" => Ok(XmlParameterType::String),
            "i" => Ok(XmlParameterType::Integer),
            "n" => Ok(XmlParameterType::Number),
            "b" => Ok(XmlParameterType::Boolean),
            "o" => Ok(XmlParameterType::Object),
            "a" => Ok(XmlParameterType::Array),
            "z" => Ok(XmlParameterType::Null),
            _ => Err(bad("unknown parameter")),
        }
    }
}
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
    let mut grant = admission.reserve_grant("XML test", &[bytes]).unwrap();
    let allocation = ParserAllocation::from_grant(&mut grant, limits).unwrap();
    (admission, allocation)
}
fn put(allocation: &mut ParserAllocation, text: &str) {
    allocation.workspaces().call[..text.len()].copy_from_slice(text.as_bytes());
}
fn body(parameter: &str, raw: &str) -> String {
    format!("<function=fs.test>\n<parameter={parameter}>\n{raw}\n</parameter>\n</function>")
}
#[test]
fn xml_schema_types_preserve_values() {
    for (parameter, raw, expected) in [
        ("s", "0012", serde_json::json!("0012")),
        ("s", "null", serde_json::json!("null")),
        (
            "s",
            "\n résumé \"quoted\" \\ path\t\u{0}\n",
            serde_json::json!("\n résumé \"quoted\" \\ path\t\u{0}\n"),
        ),
        ("s", "&amp;<ordinary>", serde_json::json!("&amp;<ordinary>")),
        ("i", "-123", serde_json::json!(-123)),
        ("i", "18446744073709551615", serde_json::json!(u64::MAX)),
        ("n", "-1.25e3", serde_json::json!(-1250.0)),
        ("b", "TRUE", serde_json::json!(true)),
        ("b", "False", serde_json::json!(false)),
        ("z", "NULL", serde_json::Value::Null),
        (
            "o",
            "{\"nested\":{\"a\":[1,true,null]}}",
            serde_json::json!({"nested":{"a":[1,true,null]}}),
        ),
        ("a", "[1,{\"a\":\"é\"}]", serde_json::json!([1,{"a":"é"}])),
    ] {
        let (admission, mut allocation) = storage(limits());
        let input = body(parameter, raw);
        put(&mut allocation, &input);
        commit_xml_tool_call(&mut allocation, input.len(), &Schema).unwrap();
        let frozen = allocation.freeze();
        assert_eq!(frozen.proposal_count(), 1);
        let proposal = frozen.proposal(0).unwrap();
        assert_eq!(proposal.name, "fs.test");
        let args: serde_json::Value = serde_json::from_str(proposal.arguments_json).unwrap();
        assert_eq!(args[parameter], expected);
        drop(frozen);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
    let (_, mut allocation) = storage(limits());
    let input = "<function=fs.test></function>";
    put(&mut allocation, input);
    commit_xml_tool_call(&mut allocation, input.len(), &Schema).unwrap();
    assert_eq!(
        allocation.freeze().proposal(0).unwrap().arguments_json,
        "{}"
    );
}
#[test]
fn xml_structure_and_duplicate_keys_fail_closed() {
    let mut inputs = vec![
        "<function=unknown></function>".to_owned(),
        "<function=fs.test></function> trailing".to_owned(),
        "<function=fs.test><parameter=s>unterminated</function>".to_owned(),
        "<function=fs.test><parameter=s>x</parameter><parameter=s>y</parameter></function>"
            .to_owned(),
        "<function=fs.test><function=fs.test></function></function>".to_owned(),
        "<function=fs.test><parameter=>x</parameter></function>".to_owned(),
        "<function=fs.test>unexpected</function>".to_owned(),
    ];
    for (parameter, raw) in [
        ("unknown", "value"),
        ("i", "01"),
        ("i", "+1"),
        ("i", "1.0"),
        ("i", "18446744073709551616"),
        ("n", "NaN"),
        ("n", "inf"),
        ("n", "1e999"),
        ("n", "01"),
        ("n", ".5"),
        ("b", "yes"),
        ("z", "false"),
        ("o", "[]"),
        ("a", "{}"),
        ("o", "{\"x\":1,\"\\u0078\":2}"),
        ("a", "[{\"x\":1,\"x\":2}]"),
        ("o", "{},\"injected\":42"),
        ("a", "[],\"injected\":42"),
        ("o", "{\"a\":'python'}"),
        ("s", "<tool_call>"),
        ("s", "<parameter=evil>"),
        ("s", "<think>"),
    ] {
        inputs.push(body(parameter, raw));
    }
    for input in inputs {
        let (admission, mut allocation) = storage(limits());
        put(&mut allocation, &input);
        assert!(
            commit_xml_tool_call(&mut allocation, input.len(), &Schema).is_err(),
            "{input}"
        );
        let frozen = allocation.freeze();
        assert_eq!(frozen.proposal_count(), 0, "{input}");
        drop(frozen);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
}
#[test]
fn xml_fixed_capacity_boundaries_release_reservations() {
    let input = body("s", &"\u{1}".repeat(32));
    let output = format!(
        "{{\"name\":\"fs.test\",\"arguments\":{{\"s\":\"{}\"}}}}",
        "\\u0001".repeat(32)
    );
    for region in ["call", "decoded", "proposal"] {
        for short in [false, true] {
            let mut l = limits();
            match region {
                "call" => l.call_bytes = output.len() - usize::from(short),
                "decoded" => l.decoded_bytes = output.len() - usize::from(short),
                _ => {
                    l.proposal_bytes =
                        "fs.test".len() + "{\"s\":\"\"}".len() + 32 * 6 - usize::from(short)
                }
            }
            let (admission, mut allocation) = storage(l);
            put(&mut allocation, &input);
            let result = commit_xml_tool_call(&mut allocation, input.len(), &Schema);
            assert_eq!(result.is_err(), short, "{region}: {result:?}");
            assert_eq!(allocation.freeze().proposal_count(), usize::from(!short));
            assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        }
    }
    for region in ["depth", "keys", "nodes"] {
        let mut l = limits();
        match region {
            "depth" => l.depth = 1,
            "keys" => l.keys = 1,
            _ => l.nodes = 1,
        }
        let (admission, mut allocation) = storage(l);
        let input = body("o", "{\"x\":{\"y\":1}}");
        put(&mut allocation, &input);
        assert!(commit_xml_tool_call(&mut allocation, input.len(), &Schema).is_err());
        drop(allocation);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
}
#[test]
fn xml_conversion_success_allocates_no_heap() {
    let input = "<function=fs.test><parameter=s>é\n\"\\</parameter><parameter=o>{\"a\":[1,null,true]}</parameter></function>";
    let (_, mut allocation) = storage(limits());
    put(&mut allocation, input);
    ALLOCATIONS.with(|n| n.set(0));
    TRACK.with(|t| t.set(true));
    let result = commit_xml_tool_call(&mut allocation, input.len(), &Schema);
    TRACK.with(|t| t.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    result.unwrap();
    assert_eq!(allocations, 0);
    assert_eq!(allocation.freeze().proposal_count(), 1);
}

const XML_CALL: &str = "<tool_call>\n<function=fs.test>\n<parameter=s>\nrésumé\n</parameter>\n</function>\n</tool_call>";

#[test]
fn xml_stream_all_chunk_boundaries_separate_reasoning() {
    use crate::slab_parser::XmlToolParser;
    let reasoning = format!("Maybe {XML_CALL} is useful, 2 < 3.");
    let input = format!("pré<think>{reasoning}</think>answer{XML_CALL}\n{XML_CALL}\n");
    for split in (0..=input.len()).filter(|i| input.is_char_boundary(*i)) {
        let (admission, allocation) = storage(limits());
        let mut parser = XmlToolParser::new(allocation, Schema, input.len()).unwrap();
        let mut text = String::new();
        let mut thoughts = String::new();
        for part in [&input[..split], &input[split..]] {
            parser
                .push(
                    part,
                    |s| {
                        text.push_str(s);
                        Ok(())
                    },
                    |s| {
                        thoughts.push_str(s);
                        Ok(())
                    },
                )
                .unwrap();
        }
        let frozen = parser.finish().unwrap();
        assert_eq!(text, "préanswer");
        assert_eq!(thoughts, reasoning);
        assert_eq!(frozen.proposal_count(), 2, "reasoning call must be inert");
        for i in 0..2 {
            assert_eq!(
                frozen.proposal(i).unwrap().arguments_json,
                "{\"s\":\"résumé\"}"
            );
        }
        drop(frozen);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
    let (_, allocation) = storage(limits());
    let mut parser = XmlToolParser::new(allocation, Schema, input.len()).unwrap();
    let mut text = String::new();
    let mut thoughts = String::new();
    for ch in input.chars() {
        let mut bytes = [0; 4];
        parser
            .push(
                ch.encode_utf8(&mut bytes),
                |s| {
                    text.push_str(s);
                    Ok(())
                },
                |s| {
                    thoughts.push_str(s);
                    Ok(())
                },
            )
            .unwrap();
    }
    assert_eq!(text, "préanswer");
    assert_eq!(thoughts, reasoning);
    assert_eq!(parser.finish().unwrap().proposal_count(), 2);
}

#[test]
fn xml_stream_later_failure_discards_every_call() {
    use crate::slab_parser::XmlToolParser;
    for suffix in [
        "suffix",
        "<think>late</think>",
        "</tool_call>",
        "<tool_callx>",
        "<tool_call><function=unknown></function></tool_call>",
        "<tool_call><function=fs.test><parameter=s>x</parameter><parameter=s>y</parameter></function></tool_call>",
        "<tool_call><function=fs.test>",
        "<",
        "<tool_",
        "<|im_end|>",
    ] {
        let input = format!("{XML_CALL}{suffix}");
        for split in (0..=input.len()).filter(|i| input.is_char_boundary(*i)) {
            let (admission, allocation) = storage(limits());
            let mut parser = XmlToolParser::new(allocation, Schema, 8192).unwrap();
            let first = parser.push(&input[..split], |_| Ok(()), |_| Ok(()));
            if first.is_ok() {
                let _ = parser.push(&input[split..], |_| Ok(()), |_| Ok(()));
            }
            assert!(parser.finish().is_err(), "{suffix} at {split}");
            assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        }
    }
    for input in [
        "<think>never closed",
        "</think>",
        "<think><think>nested</think></think>",
        "<function=x>",
    ] {
        let (admission, allocation) = storage(limits());
        let mut parser = XmlToolParser::new(allocation, Schema, 8192).unwrap();
        let _ = parser.push(input, |_| Ok(()), |_| Ok(()));
        assert!(parser.finish().is_err());
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
}

#[test]
fn xml_stream_limits_and_sink_failure_release_storage() {
    use crate::slab_parser::XmlToolParser;
    let (admission, allocation) = storage(limits());
    let mut parser = XmlToolParser::new(allocation, Schema, XML_CALL.len() - 1).unwrap();
    assert!(parser.push(XML_CALL, |_| Ok(()), |_| Ok(())).is_err());
    assert!(parser.push("", |_| Ok(()), |_| Ok(())).is_err());
    assert!(parser.finish().is_err());
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    for which in [
        "carry",
        "call",
        "calls",
        "text_sink",
        "reasoning_sink",
        "drop",
    ] {
        let mut l = limits();
        match which {
            "carry" => l.carry_bytes = 1,
            "call" => l.call_bytes = 1,
            "calls" => l.calls = 1,
            _ => {}
        }
        let (admission, allocation) = storage(l);
        let parser = XmlToolParser::new(allocation, Schema, 8192);
        if which == "carry" {
            assert!(parser.is_err());
        } else {
            let mut parser = parser.unwrap();
            match which {
                "call" => {
                    assert!(parser.push(XML_CALL, |_| Ok(()), |_| Ok(())).is_err());
                }
                "calls" => {
                    parser.push(XML_CALL, |_| Ok(()), |_| Ok(())).unwrap();
                    assert!(parser.push(XML_CALL, |_| Ok(()), |_| Ok(())).is_err());
                }
                "text_sink" => {
                    assert!(
                        parser
                            .push("answer", |_| Err(bad("sink")), |_| Ok(()))
                            .is_err()
                    );
                }
                "reasoning_sink" => {
                    assert!(
                        parser
                            .push("<think>reason", |_| Ok(()), |_| Err(bad("sink")))
                            .is_err()
                    );
                }
                "drop" => {
                    parser.push(XML_CALL, |_| Ok(()), |_| Ok(())).unwrap();
                }
                _ => unreachable!(),
            }
            if which == "drop" {
                drop(parser);
            } else {
                assert!(parser.finish().is_err());
            }
        }
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0, "{which}");
    }
}

#[test]
fn xml_stream_success_allocates_no_heap() {
    use crate::slab_parser::XmlToolParser;
    let (_, allocation) = storage(limits());
    let mut parser = XmlToolParser::new(allocation, Schema, 8192).unwrap();
    ALLOCATIONS.with(|n| n.set(0));
    TRACK.with(|t| t.set(true));
    let result = parser
        .push(
            "<think>inert <tool_call>fake</tool_call></think>answer",
            |_| Ok(()),
            |_| Ok(()),
        )
        .and_then(|()| parser.push(XML_CALL, |_| Ok(()), |_| Ok(())))
        .and_then(|()| parser.finish());
    TRACK.with(|t| t.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    assert_eq!(result.unwrap().proposal_count(), 1);
    assert_eq!(allocations, 0);
}
