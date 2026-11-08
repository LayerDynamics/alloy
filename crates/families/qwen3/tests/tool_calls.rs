use qwen3::tool_calls::{ParsedBlock, ParserLimits, ToolCallParser, ToolParseError};
use serde_json::{Value, json};

fn fixtures() -> Vec<Value> {
    serde_json::from_str(include_str!("../fixture/tool_calls.json")).expect("valid fixtures")
}

fn parse_chunks(
    chunks: &[&[u8]],
    limits: ParserLimits,
) -> Result<Vec<ParsedBlock>, ToolParseError> {
    let mut parser = ToolCallParser::new(limits)?;
    for chunk in chunks {
        parser.push(chunk)?;
    }
    parser.finish()
}

fn normalized(blocks: &[ParsedBlock]) -> Value {
    Value::Array(
        blocks
            .iter()
            .map(|block| match block {
                ParsedBlock::Text(text) => json!({
                    "kind": "text",
                    "text": text
                }),
                ParsedBlock::ToolCall(call) => json!({
                    "kind": "call",
                    "name": call.name,
                    "arguments": call.arguments
                }),
            })
            .collect(),
    )
}

fn expected_error(name: &str) -> ToolParseError {
    match name {
        "invalid_envelope" => ToolParseError::InvalidEnvelope,
        "invalid_json" => ToolParseError::InvalidJson,
        "incomplete_call" => ToolParseError::IncompleteCall,
        other => panic!("unknown fixture error: {other}"),
    }
}

#[test]
fn qwen3_tool_call_chunk_boundaries() {
    for fixture in fixtures() {
        if fixture.get("error").is_some() {
            continue;
        }

        let name = fixture["name"].as_str().expect("name");
        let output = fixture["output"].as_str().expect("output").as_bytes();

        for split in 0..=output.len() {
            let blocks = parse_chunks(
                &[&output[..split], &output[split..]],
                ParserLimits::default(),
            )
            .unwrap_or_else(|error| panic!("{name} failed at split {split}: {error}"));

            assert_eq!(
                normalized(&blocks),
                fixture["expected"],
                "{name}, split {split}"
            );
        }

        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        for byte in output {
            parser
                .push(std::slice::from_ref(byte))
                .expect("single-byte fragment");
        }

        assert_eq!(
            normalized(&parser.finish().expect("complete parse")),
            fixture["expected"],
            "{name}, single-byte fragments"
        );
    }
}

#[test]
fn malformed_fixtures_return_no_parsed_output() {
    for fixture in fixtures() {
        let Some(error_name) = fixture.get("error") else {
            continue;
        };

        let name = fixture["name"].as_str().expect("name");
        let output = fixture["output"].as_str().expect("output").as_bytes();
        let expected = expected_error(error_name.as_str().expect("error"));

        for split in 0..=output.len() {
            let actual = parse_chunks(
                &[&output[..split], &output[split..]],
                ParserLimits::default(),
            );

            assert_eq!(actual, Err(expected.clone()), "{name}, split {split}");
        }
    }
}

#[test]
fn model_cannot_supply_daemon_identity_or_authority() {
    for field in [
        "call_id",
        "session_id",
        "turn_id",
        "generation_id",
        "origin",
        "version",
        "policy",
        "approved",
    ] {
        let mut envelope = json!({
            "name": "fs.read",
            "arguments": { "path": "fixture.txt" }
        });
        envelope[field] = json!("model-controlled");

        let output = format!("<tool_call>{envelope}</tool_call>");

        assert_eq!(
            parse_chunks(&[output.as_bytes()], ParserLimits::default()),
            Err(ToolParseError::InvalidEnvelope),
            "accepted model authority field: {field}"
        );
    }
}

#[test]
fn call_body_limit_accepts_exact_boundary_and_rejects_one_byte_over() {
    let body = r#"{"name":"fs.read","arguments":{}}"#;
    let output = format!("<tool_call>{body}</tool_call>");

    let exact = ParserLimits {
        max_output_bytes: output.len(),
        max_call_bytes: body.len(),
        max_calls: 1,
    };

    assert!(parse_chunks(&[output.as_bytes()], exact).is_ok());

    let too_small = ParserLimits {
        max_call_bytes: body.len() - 1,
        ..exact
    };

    assert_eq!(
        parse_chunks(&[output.as_bytes()], too_small),
        Err(ToolParseError::CallTooLarge)
    );
}

#[test]
fn call_count_limit_rejects_entire_generation() {
    let call = r#"<tool_call>{"name":"fs.read","arguments":{}}</tool_call>"#;
    let output = call.repeat(2);

    let limits = ParserLimits {
        max_calls: 1,
        ..ParserLimits::default()
    };

    assert_eq!(
        parse_chunks(&[output.as_bytes()], limits),
        Err(ToolParseError::CallLimit)
    );
}

#[test]
fn output_limit_is_cumulative_across_fragments() {
    let limits = ParserLimits {
        max_output_bytes: 4,
        max_call_bytes: 4,
        max_calls: 1,
    };

    assert_eq!(
        parse_chunks(&[b"ab", b"cd"], limits),
        Ok(vec![ParsedBlock::Text("abcd".into())])
    );

    let mut parser = ToolCallParser::new(limits).expect("parser");
    parser.push(b"ab").expect("first fragment");

    assert_eq!(parser.push(b"cde"), Err(ToolParseError::OutputLimit));
    assert_eq!(parser.finish(), Err(ToolParseError::FailedParser));
}

#[test]
fn invalid_limits_are_rejected() {
    for limits in [
        ParserLimits {
            max_output_bytes: 0,
            max_call_bytes: 1,
            max_calls: 1,
        },
        ParserLimits {
            max_output_bytes: 8,
            max_call_bytes: 0,
            max_calls: 1,
        },
        ParserLimits {
            max_output_bytes: 8,
            max_call_bytes: 4,
            max_calls: 0,
        },
        ParserLimits {
            max_output_bytes: 8,
            max_call_bytes: 9,
            max_calls: 1,
        },
    ] {
        assert!(matches!(
            ToolCallParser::new(limits),
            Err(ToolParseError::InvalidLimits)
        ));
    }
}

#[test]
fn invalid_utf8_is_rejected_after_arbitrary_byte_fragments() {
    assert_eq!(
        parse_chunks(&[&[0xe9], &[0xff]], ParserLimits::default()),
        Err(ToolParseError::InvalidUtf8)
    );
}

#[test]
fn unresolved_tool_name_remains_data_not_a_registration() {
    let output = br#"<tool_call>{"name":"/bin/sh","arguments":{}}</tool_call>"#;
    let blocks = parse_chunks(&[output], ParserLimits::default()).expect("syntax parse");

    match blocks.as_slice() {
        [ParsedBlock::ToolCall(call)] => {
            assert_eq!(call.name, "/bin/sh");
            assert!(call.arguments.is_empty());
        }
        _ => panic!("expected exactly one untrusted parsed call"),
    }
}

#[test]
fn w3_boundary_duplicate_keys_fail_in_atomic_and_streaming_modes() {
    for body in [
        r#"{"name":"a","name":"b","arguments":{}}"#,
        r#"{"name":"a","arguments":{"x":1,"x":2}}"#,
        r#"{"name":"a","arguments":{"nested":[{"x":1,"x":2}]}}"#,
        r#"{"name":"a","arguments":{"x":1,"\u0078":2}}"#,
    ] {
        let text = format!("<tool_call>{body}</tool_call>");
        for split in 0..=text.len() {
            let chunks = [&text.as_bytes()[..split], &text.as_bytes()[split..]];
            assert_eq!(
                parse_chunks(&chunks, ParserLimits::default()),
                Err(ToolParseError::InvalidJson)
            );
            let mut parser = ToolCallParser::new(ParserLimits::default()).unwrap();
            let mut failure = None;
            for chunk in chunks {
                match parser.push_stream(chunk) {
                    Ok(blocks) => assert!(blocks.is_empty(), "duplicate call escaped as proposal"),
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            assert_eq!(failure, Some(ToolParseError::InvalidJson));
            assert_eq!(parser.finish(), Err(ToolParseError::FailedParser));
        }
    }
}
