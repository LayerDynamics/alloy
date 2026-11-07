//! Shared-fixture parity for supported JSON envelopes, not all possible grammar.
use super::*;
use crate::admission::AdmissionController;
use crate::parser_allocation::ParserAllocationLimits;
use qwen3::tool_calls::{ParsedBlock, ParserLimits, ToolCallParser};
use serde_json::Value;

#[derive(Debug, PartialEq)]
struct Projection {
    text: String,
    calls: Vec<(String, Value)>,
}
fn empty() -> Projection {
    Projection {
        text: String::new(),
        calls: Vec::new(),
    }
}
fn fixtures() -> Vec<Value> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../families/qwen3/fixture/tool_calls.json"
    )))
    .expect("shared fixtures")
}
fn capacities() -> ParserAllocationLimits {
    ParserAllocationLimits {
        carry_bytes: 32,
        call_bytes: 8192,
        decoded_bytes: 8192,
        depth: 64,
        keys: 1024,
        nodes: 2048,
        proposal_bytes: 32768,
        calls: 16,
    }
}
fn legacy(parts: &[&str]) -> Result<Projection, String> {
    let mut parser = ToolCallParser::new(ParserLimits {
        max_output_bytes: 65536,
        max_call_bytes: 8192,
        max_calls: 16,
    })
    .map_err(|error| error.to_string())?;
    let mut blocks = Vec::new();
    for part in parts {
        blocks.extend(
            parser
                .push_stream(part.as_bytes())
                .map_err(|e| e.to_string())?,
        );
    }
    blocks.extend(parser.finish().map_err(|e| e.to_string())?);
    let mut result = empty();
    for block in blocks {
        match block {
            ParsedBlock::Text(text) => result.text.push_str(&text),
            ParsedBlock::ToolCall(call) => result
                .calls
                .push((call.name, Value::Object(call.arguments))),
        }
    }
    Ok(result)
}
fn slab(parts: &[&str], limits: ParserAllocationLimits) -> Result<Projection, String> {
    let bytes = ParserAllocation::required_bytes(limits).map_err(|e| e.to_string())?;
    let admission = AdmissionController::new(bytes).expect("admission");
    let mut grant = admission
        .reserve_grant("parity fixture", &[bytes])
        .expect("grant");
    let storage = ParserAllocation::from_grant(&mut grant, limits).map_err(|e| e.to_string())?;
    let mut parser = SlabParser::new(storage, 65536).map_err(|e| e.to_string())?;
    let mut result = empty();
    for part in parts {
        if let Err(error) = parser.push(part, |text| {
            result.text.push_str(text);
            Ok(())
        }) {
            drop(parser);
            assert_eq!(
                admission
                    .snapshot()
                    .expect("rejected fixture released")
                    .reserved_bytes(),
                0
            );
            return Err(error.to_string());
        }
    }
    let owner = match parser.finish() {
        Ok(owner) => owner,
        Err(error) => {
            assert_eq!(
                admission
                    .snapshot()
                    .expect("completion failure released")
                    .reserved_bytes(),
                0
            );
            return Err(error.to_string());
        }
    };
    for index in 0..owner.proposal_count() {
        let proposal = owner.proposal(index).expect("descriptor");
        // Test-only materialization checks semantic equality. Production keeps JSON text.
        let arguments =
            serde_json::from_str(proposal.arguments_json).expect("validated fixture JSON");
        result.calls.push((proposal.name.to_owned(), arguments));
    }
    drop(owner);
    assert_eq!(
        admission
            .snapshot()
            .expect("owner released")
            .reserved_bytes(),
        0
    );
    Ok(result)
}
fn expected(fixture: &Value) -> Projection {
    let mut result = empty();
    for block in fixture["expected"].as_array().expect("expected blocks") {
        match block["kind"].as_str().expect("kind") {
            "text" => result.text.push_str(block["text"].as_str().expect("text")),
            "call" => result.calls.push((
                block["name"].as_str().expect("name").to_owned(),
                block["arguments"].clone(),
            )),
            other => panic!("unknown fixture block {other}"),
        }
    }
    result
}

#[test]
fn accepted_shared_fixtures_match_at_every_utf8_split() {
    for fixture in fixtures().iter().filter(|f| f.get("expected").is_some()) {
        let text = fixture["output"].as_str().expect("output");
        let want = expected(fixture);
        for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            let parts = [&text[..split], &text[split..]];
            assert_eq!(
                legacy(&parts).expect("legacy accepted"),
                want,
                "legacy {} split {split}",
                fixture["name"]
            );
            assert_eq!(
                slab(&parts, capacities()).expect("slab accepted"),
                want,
                "slab {} split {split}",
                fixture["name"]
            );
        }
    }
}
#[test]
fn rejected_shared_fixtures_fail_in_both_streaming_parsers() {
    for fixture in fixtures().iter().filter(|f| f.get("error").is_some()) {
        let text = fixture["output"].as_str().expect("output");
        for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            let parts = [&text[..split], &text[split..]];
            assert!(
                legacy(&parts).is_err(),
                "legacy accepted {} split {split}",
                fixture["name"]
            );
            assert!(
                slab(&parts, capacities()).is_err(),
                "slab accepted {} split {split}",
                fixture["name"]
            );
        }
    }
}
#[test]
fn accepted_shared_fixtures_match_one_unicode_scalar_at_a_time() {
    for fixture in fixtures().iter().filter(|f| f.get("expected").is_some()) {
        let text = fixture["output"].as_str().expect("output");
        let parts: Vec<&str> = text
            .char_indices()
            .map(|(i, ch)| &text[i..i + ch.len_utf8()])
            .collect();
        let want = expected(fixture);
        assert_eq!(legacy(&parts).expect("legacy"), want);
        assert_eq!(slab(&parts, capacities()).expect("slab"), want);
    }
}
#[test]
fn alternate_coder_function_parameter_syntax_is_explicitly_unsupported() {
    let all = fixtures();
    let fixture = all
        .iter()
        .find(|f| f["name"] == "unsupported_coder_syntax")
        .expect("unsupported syntax fixture");
    assert!(fixture.get("error").is_some());
    let text = fixture["output"].as_str().expect("output");
    assert!(text.contains("<function="));
    assert!(text.contains("<parameter="));
    assert!(legacy(&[text]).is_err());
    assert!(slab(&[text], capacities()).is_err());
}
#[test]
fn slab_capacity_refusal_does_not_fall_back_to_the_legacy_parser() {
    let all = fixtures();
    let fixture = all
        .iter()
        .find(|f| f["name"] == "mixed_unicode_and_calls")
        .expect("multi-call fixture");
    let text = fixture["output"].as_str().expect("output");
    assert!(legacy(&[text]).is_ok());
    let mut limits = capacities();
    limits.calls = 1;
    assert!(slab(&[text], limits).is_err());
}
#[test]
fn shared_corpus_contains_positive_negative_and_unique_named_cases() {
    let all = fixtures();
    let mut names = std::collections::BTreeSet::new();
    assert!(all.iter().filter(|f| f.get("expected").is_some()).count() >= 5);
    assert!(all.iter().filter(|f| f.get("error").is_some()).count() >= 6);
    for fixture in all {
        assert!(names.insert(fixture["name"].as_str().expect("name").to_owned()));
    }
}
