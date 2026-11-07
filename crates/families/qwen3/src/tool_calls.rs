//! Bounded parsing of untrusted Qwen3 JSON tool-call envelopes.
//!
//! REQUIRED TRUST BOUNDARY:
//! This module returns only names and argument objects. It never constructs
//! daemon call/context/invocation types, assigns identities, authorizes,
//! registers implementations, or executes tools.
//!
//! Parsing commits output only at end of generation. A malformed later block
//! therefore cannot leave an earlier call available for execution.
//!
//! Supported syntax:
//! <tool_call>{"name":"fs.read","arguments":{"path":"file"}}</tool_call>
//!
//! Alternative Coder syntax is not supported by this parser version.

use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Number, Value};
use std::fmt;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Explicit per-generation parser limits.
#[derive(Clone, Copy, Debug)]
pub struct ParserLimits {
    pub max_output_bytes: usize,
    pub max_call_bytes: usize,
    pub max_calls: usize,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: 1024 * 1024,
            max_call_bytes: 64 * 1024,
            max_calls: 16,
        }
    }
}

/// Untrusted family-parser output, not an executable invocation.
///
/// It intentionally has no call ID, session context, version, origin,
/// policy, executable implementation, or approval flag.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// Ordered text and structured-call blocks from a complete generation.
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedBlock {
    Text(String),
    ToolCall(ParsedToolCall),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolParseError {
    InvalidLimits,
    OutputLimit,
    CallLimit,
    CallTooLarge,
    InvalidUtf8,
    MalformedDelimiter,
    IncompleteCall,
    InvalidJson,
    InvalidEnvelope,
    ReservedControlSyntax,
    FailedParser,
}

impl fmt::Display for ToolParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidLimits => "parser limits must be positive and consistent",
            Self::OutputLimit => "generation exceeds parser output limit",
            Self::CallLimit => "generation exceeds tool-call count limit",
            Self::CallTooLarge => "tool-call body exceeds byte limit",
            Self::InvalidUtf8 => "generation is not valid UTF-8",
            Self::MalformedDelimiter => "malformed or unexpected tool delimiter",
            Self::IncompleteCall => "generation ends inside a tool call",
            Self::InvalidJson => "tool-call body is invalid or ambiguous JSON",
            Self::InvalidEnvelope => "tool-call envelope requires only name and object arguments",
            Self::ReservedControlSyntax => "unsupported reserved control syntax",
            Self::FailedParser => "parser previously failed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ToolParseError {}

/// Byte-fragment accumulator with atomic end-of-generation parsing.
///
/// No parsed call is exposed by `push`. `finish` consumes the parser, so
/// completed output cannot be emitted twice from the same instance.
pub struct ToolCallParser {
    limits: ParserLimits,
    bytes: Vec<u8>,
    failed: bool,
}

impl ToolCallParser {
    pub fn new(limits: ParserLimits) -> Result<Self, ToolParseError> {
        if limits.max_output_bytes == 0
            || limits.max_call_bytes == 0
            || limits.max_calls == 0
            || limits.max_call_bytes > limits.max_output_bytes
        {
            return Err(ToolParseError::InvalidLimits);
        }

        Ok(Self {
            limits,
            bytes: Vec::new(),
            failed: false,
        })
    }

    /// Accept any byte boundary, including a split inside a UTF-8 character.
    pub fn push(&mut self, fragment: &[u8]) -> Result<(), ToolParseError> {
        if self.failed {
            return Err(ToolParseError::FailedParser);
        }

        let fits = self
            .bytes
            .len()
            .checked_add(fragment.len())
            .is_some_and(|length| length <= self.limits.max_output_bytes);

        if !fits {
            self.failed = true;
            return Err(ToolParseError::OutputLimit);
        }

        self.bytes.extend_from_slice(fragment);
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<ParsedBlock>, ToolParseError> {
        if self.failed {
            return Err(ToolParseError::FailedParser);
        }

        let text = std::str::from_utf8(&self.bytes).map_err(|_| ToolParseError::InvalidUtf8)?;

        parse_complete(text, self.limits)
    }
}

fn parse_complete(text: &str, limits: ParserLimits) -> Result<Vec<ParsedBlock>, ToolParseError> {
    let mut blocks = Vec::new();
    let mut remaining = text;
    let mut calls = 0usize;

    while let Some(open) = remaining.find(OPEN) {
        let prefix = &remaining[..open];
        validate_text(prefix)?;
        if !prefix.is_empty() {
            blocks.push(ParsedBlock::Text(prefix.to_owned()));
        }

        let body_and_tail = &remaining[open + OPEN.len()..];
        let close = body_and_tail
            .find(CLOSE)
            .ok_or(ToolParseError::IncompleteCall)?;
        let body = &body_and_tail[..close];

        if body.len() > limits.max_call_bytes {
            return Err(ToolParseError::CallTooLarge);
        }

        // Literal protocol delimiters inside a JSON string are deliberately
        // rejected; escaped spellings can represent ordinary string data.
        if body.contains(OPEN) {
            return Err(ToolParseError::MalformedDelimiter);
        }

        calls += 1;
        if calls > limits.max_calls {
            return Err(ToolParseError::CallLimit);
        }

        let strict: StrictJson =
            serde_json::from_str(body).map_err(|_| ToolParseError::InvalidJson)?;

        let mut envelope = strict
            .0
            .as_object()
            .cloned()
            .ok_or(ToolParseError::InvalidEnvelope)?;

        if envelope.len() != 2 {
            return Err(ToolParseError::InvalidEnvelope);
        }

        let name = match envelope.remove("name") {
            Some(Value::String(name))
                if !name.is_empty()
                    && name.trim() == name
                    && !name.chars().any(char::is_control) =>
            {
                name
            }
            _ => return Err(ToolParseError::InvalidEnvelope),
        };

        let arguments = match envelope.remove("arguments") {
            Some(Value::Object(arguments)) => arguments,
            _ => return Err(ToolParseError::InvalidEnvelope),
        };

        blocks.push(ParsedBlock::ToolCall(ParsedToolCall { name, arguments }));

        remaining = &body_and_tail[close + CLOSE.len()..];
    }

    validate_text(remaining)?;
    if !remaining.is_empty() {
        blocks.push(ParsedBlock::Text(remaining.to_owned()));
    }

    Ok(blocks)
}

fn validate_text(text: &str) -> Result<(), ToolParseError> {
    if text.contains("<tool_call") || text.contains("</tool_call") {
        return Err(ToolParseError::MalformedDelimiter);
    }

    if [
        "<|",
        "<think>",
        "</think>",
        "<tool_response",
        "</tool_response",
        "<function=",
        "<parameter=",
    ]
    .iter()
    .any(|marker| text.contains(marker))
    {
        return Err(ToolParseError::ReservedControlSyntax);
    }

    Ok(())
}

/// JSON decoder that rejects duplicate object keys at every nesting level.
///
/// serde_json::Value alone would silently retain only one duplicate value.
struct StrictJson(Value);

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = StrictJson;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::Bool(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::Number(value.into())))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::Number(value.into())))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Number::from_f64(value)
                    .map(|number| StrictJson(Value::Number(number)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::String(value.to_owned())))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::String(value)))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictJson(Value::Null))
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                self.visit_unit()
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<StrictJson>()? {
                    values.push(value.0);
                }
                Ok(StrictJson(Value::Array(values)))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("duplicate JSON object key"));
                    }
                    let value = map.next_value::<StrictJson>()?;
                    values.insert(key, value.0);
                }
                Ok(StrictJson(Value::Object(values)))
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(bytes: &[u8]) -> Result<Vec<ParsedBlock>, ToolParseError> {
        let mut parser = ToolCallParser::new(ParserLimits::default())?;
        parser.push(bytes)?;
        parser.finish()
    }

    #[test]
    fn qwen3_tool_call_chunk_boundaries() {
        let output = concat!(
            "Reading 雪.\n",
            "<tool_call>",
            r#"{"name":"fs.read","arguments":{"path":"résumé.txt"}}"#,
            "</tool_call>",
            "\nWaiting."
        );
        let expected = parse(output.as_bytes()).expect("complete parse");

        for split in 0..=output.len() {
            let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
            parser.push(&output.as_bytes()[..split]).expect("first");
            parser.push(&output.as_bytes()[split..]).expect("second");
            assert_eq!(parser.finish().expect("parse"), expected);
        }

        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        for byte in output.as_bytes() {
            parser.push(std::slice::from_ref(byte)).expect("byte");
        }
        assert_eq!(parser.finish().expect("parse"), expected);
    }

    #[test]
    fn identity_and_authority_fields_are_rejected() {
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
            let mut body = json!({
                "name": "fs.read",
                "arguments": {"path": "fixture.txt"}
            });
            body[field] = json!("model-claimed");
            let output = format!("{OPEN}{body}{CLOSE}");
            assert_eq!(
                parse(output.as_bytes()),
                Err(ToolParseError::InvalidEnvelope)
            );
        }
    }

    #[test]
    fn duplicate_keys_are_rejected_including_nested_arguments() {
        for body in [
            r#"{"name":"a","name":"b","arguments":{}}"#,
            r#"{"name":"a","arguments":{"path":"a","path":"b"}}"#,
            r#"{"name":"a","arguments":{"nested":[{"x":1,"x":2}]}}"#,
        ] {
            let output = format!("{OPEN}{body}{CLOSE}");
            assert_eq!(parse(output.as_bytes()), Err(ToolParseError::InvalidJson));
        }
    }

    #[test]
    fn malformed_later_call_invalidates_the_entire_generation() {
        let output = concat!(
            "<tool_call>",
            r#"{"name":"fs.read","arguments":{}}"#,
            "</tool_call>",
            "<tool_call>{broken}</tool_call>"
        );
        assert_eq!(parse(output.as_bytes()), Err(ToolParseError::InvalidJson));
    }

    #[test]
    fn truncated_and_unmatched_delimiters_fail() {
        assert_eq!(parse(b"<tool_call>{"), Err(ToolParseError::IncompleteCall));
        assert_eq!(
            parse(b"text </tool_call>"),
            Err(ToolParseError::MalformedDelimiter)
        );
        assert_eq!(
            parse(b"<tool_call unexpected>"),
            Err(ToolParseError::MalformedDelimiter)
        );
    }

    #[test]
    fn invalid_utf8_fails_at_finish() {
        assert_eq!(parse(&[0xff]), Err(ToolParseError::InvalidUtf8));
    }

    #[test]
    fn exceeding_output_limit_poisons_parser() {
        let mut parser = ToolCallParser::new(ParserLimits {
            max_output_bytes: 4,
            max_call_bytes: 4,
            max_calls: 1,
        })
        .expect("parser");

        assert_eq!(parser.push(b"12345"), Err(ToolParseError::OutputLimit));
        assert_eq!(parser.push(b""), Err(ToolParseError::FailedParser));
        assert_eq!(parser.finish(), Err(ToolParseError::FailedParser));
    }

    #[test]
    fn multiple_calls_preserve_order() {
        let output = concat!(
            "<tool_call>",
            r#"{"name":"fs.read","arguments":{"path":"a"}}"#,
            "</tool_call>",
            "<tool_call>",
            r#"{"name":"fs.list","arguments":{"path":"b"}}"#,
            "</tool_call>"
        );
        let blocks = parse(output.as_bytes()).expect("parse");
        assert_eq!(blocks.len(), 2);

        match (&blocks[0], &blocks[1]) {
            (ParsedBlock::ToolCall(first), ParsedBlock::ToolCall(second)) => {
                assert_eq!(first.name, "fs.read");
                assert_eq!(second.name, "fs.list");
            }
            _ => panic!("expected calls"),
        }
    }
}
