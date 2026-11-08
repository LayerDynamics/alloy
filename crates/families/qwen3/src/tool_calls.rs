//! Bounded parsing of untrusted Qwen3 JSON tool-call envelopes.
//!
//! REQUIRED TRUST BOUNDARY:
//! This module returns only names and argument objects. It never constructs
//! daemon call/context/invocation types, assigns identities, authorizes,
//! registers implementations, or executes tools.
//!
//! Atomic `push` keeps output until completion. Incremental `push_stream`
//! emits text and complete untrusted proposals before completion. Streamed
//! proposals MUST remain non-executable until successful generation finish;
//! a later protocol error invalidates the whole generation's proposals.
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

/// Bounded parser with atomic and incremental consumption modes.
///
/// `push` retains the original atomic API: `finish` returns all blocks.
/// `push_stream` emits text and complete untrusted call proposals as soon
/// as they are recognizable; `finish` then returns only any remaining tail.
/// Modes cannot be mixed on the same parser.
///
/// Incremental proposals are tentative until successful generation finish.
/// Consumers must not execute them before that completion gate. A later
/// error invalidates every proposal from this generation. Previously
/// displayed text cannot be retracted and is not evidence of success.
pub struct ToolCallParser {
    limits: ParserLimits,
    bytes: Vec<u8>,
    failed: bool,
    streaming: Option<bool>,
    received: usize,
    in_call: bool,
    calls: usize,
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
            streaming: None,
            received: 0,
            in_call: false,
            calls: 0,
        })
    }

    /// Atomic compatibility API. Do not mix with `push_stream`.
    pub fn push(&mut self, fragment: &[u8]) -> Result<(), ToolParseError> {
        self.select_mode(false)?;
        self.accept(fragment)
    }

    /// Consume arbitrary bytes and emit ordered incremental blocks.
    ///
    /// UTF-8 and delimiter fragments are held until complete. Call JSON is
    /// validated at its closing delimiter, not at end of generation.
    /// Returned calls contain no identity or authority and remain tentative
    /// until `finish` succeeds. An error permanently poisons the parser.
    pub fn push_stream(&mut self, fragment: &[u8]) -> Result<Vec<ParsedBlock>, ToolParseError> {
        self.select_mode(true)?;
        self.accept(fragment)?;
        match self.process_stream(false) {
            Ok(blocks) => Ok(blocks),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    /// Validate completion. Streaming mode returns only un-emitted tail.
    ///
    /// Success commits protocol validity, not registry validity, approval,
    /// or permission to execute. Cancellation must drop the parser instead.
    pub fn finish(mut self) -> Result<Vec<ParsedBlock>, ToolParseError> {
        if self.failed {
            return Err(ToolParseError::FailedParser);
        }
        if self.streaming == Some(true) {
            self.process_stream(true)
        } else {
            let text = std::str::from_utf8(&self.bytes).map_err(|_| ToolParseError::InvalidUtf8)?;
            parse_complete(text, self.limits)
        }
    }

    fn select_mode(&mut self, streaming: bool) -> Result<(), ToolParseError> {
        if self.failed || self.streaming.is_some_and(|mode| mode != streaming) {
            self.failed = true;
            return Err(ToolParseError::FailedParser);
        }
        self.streaming = Some(streaming);
        Ok(())
    }

    fn accept(&mut self, fragment: &[u8]) -> Result<(), ToolParseError> {
        let Some(total) = self.received.checked_add(fragment.len()) else {
            self.failed = true;
            return Err(ToolParseError::OutputLimit);
        };
        if total > self.limits.max_output_bytes {
            self.failed = true;
            return Err(ToolParseError::OutputLimit);
        }
        self.received = total;
        self.bytes.extend_from_slice(fragment);
        Ok(())
    }

    fn process_stream(&mut self, eof: bool) -> Result<Vec<ParsedBlock>, ToolParseError> {
        const FORBIDDEN: &[&[u8]] = &[
            b"<|",
            b"<think>",
            b"</think>",
            b"<tool_response",
            b"</tool_response",
            b"<function=",
            b"<parameter=",
        ];
        let mut output = Vec::new();
        while !self.bytes.is_empty() {
            if self.in_call {
                let closing = self
                    .bytes
                    .windows(CLOSE.len())
                    .position(|window| window == CLOSE.as_bytes());
                if let Some(end) = closing {
                    if end > self.limits.max_call_bytes {
                        return Err(ToolParseError::CallTooLarge);
                    }
                    if self.calls >= self.limits.max_calls {
                        return Err(ToolParseError::CallLimit);
                    }
                    let body = std::str::from_utf8(&self.bytes[..end])
                        .map_err(|_| ToolParseError::InvalidUtf8)?;
                    // Reuse the strict envelope/duplicate-key validation.
                    let envelope = format!("{OPEN}{body}{CLOSE}");
                    let blocks = parse_complete(&envelope, self.limits)?;
                    output.extend(blocks);
                    self.calls += 1;
                    self.bytes.drain(..end + CLOSE.len());
                    self.in_call = false;
                    continue;
                }
                if self
                    .bytes
                    .windows(OPEN.len())
                    .any(|window| window == OPEN.as_bytes())
                {
                    return Err(ToolParseError::MalformedDelimiter);
                }
                // A partial closing delimiter is syntax, not body bytes.
                let held = (1..CLOSE.len())
                    .rev()
                    .find(|&length| self.bytes.ends_with(&CLOSE.as_bytes()[..length]))
                    .unwrap_or(0);
                if self.bytes.len() - held > self.limits.max_call_bytes {
                    return Err(ToolParseError::CallTooLarge);
                }
                break;
            }

            let marker = self.bytes.iter().position(|&byte| byte == b'<');
            match marker {
                Some(0) => {
                    if self.bytes.starts_with(OPEN.as_bytes()) {
                        self.bytes.drain(..OPEN.len());
                        self.in_call = true;
                        continue;
                    }
                    if self.bytes.starts_with(b"</tool_call") {
                        return Err(ToolParseError::MalformedDelimiter);
                    }
                    if self.bytes.starts_with(b"<tool_call") && self.bytes.len() >= OPEN.len() {
                        return Err(ToolParseError::MalformedDelimiter);
                    }
                    if FORBIDDEN
                        .iter()
                        .any(|marker| self.bytes.starts_with(marker))
                    {
                        return Err(ToolParseError::ReservedControlSyntax);
                    }
                    let partial = OPEN.as_bytes().starts_with(&self.bytes)
                        || CLOSE.as_bytes().starts_with(&self.bytes)
                        || FORBIDDEN
                            .iter()
                            .any(|marker| marker.starts_with(&self.bytes));
                    if partial {
                        if eof {
                            return Err(ToolParseError::MalformedDelimiter);
                        }
                        break;
                    }
                    // Batch ordinary '<' characters rather than repeatedly
                    // draining the front of a large literal-text fragment.
                    let next = (1..self.bytes.len())
                        .find(|&index| {
                            if self.bytes[index] != b'<' {
                                return false;
                            }
                            let tail = &self.bytes[index..];
                            OPEN.as_bytes().starts_with(tail)
                                || tail.starts_with(b"<tool_call")
                                || CLOSE.as_bytes().starts_with(tail)
                                || tail.starts_with(b"</tool_call")
                                || FORBIDDEN.iter().any(|marker| {
                                    marker.starts_with(tail) || tail.starts_with(marker)
                                })
                        })
                        .unwrap_or(self.bytes.len());
                    let valid = match std::str::from_utf8(&self.bytes[..next]) {
                        Ok(_) => next,
                        Err(error)
                            if error.error_len().is_none() && !eof && next == self.bytes.len() =>
                        {
                            error.valid_up_to()
                        }
                        Err(_) => return Err(ToolParseError::InvalidUtf8),
                    };
                    if valid == 0 {
                        break;
                    }
                    self.emit_text(valid, &mut output)?;
                }
                Some(length) => self.emit_text(length, &mut output)?,
                None => {
                    let valid = match std::str::from_utf8(&self.bytes) {
                        Ok(_) => self.bytes.len(),
                        Err(error) if error.error_len().is_none() && !eof => error.valid_up_to(),
                        Err(_) => return Err(ToolParseError::InvalidUtf8),
                    };
                    if valid == 0 {
                        break;
                    }
                    self.emit_text(valid, &mut output)?;
                }
            }
        }
        if eof && self.in_call {
            return Err(ToolParseError::IncompleteCall);
        }
        Ok(output)
    }

    fn emit_text(
        &mut self,
        length: usize,
        output: &mut Vec<ParsedBlock>,
    ) -> Result<(), ToolParseError> {
        let text =
            std::str::from_utf8(&self.bytes[..length]).map_err(|_| ToolParseError::InvalidUtf8)?;
        if let Some(ParsedBlock::Text(previous)) = output.last_mut() {
            previous.push_str(text);
        } else {
            output.push(ParsedBlock::Text(text.to_owned()));
        }
        self.bytes.drain(..length);
        Ok(())
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

// W2_INCREMENTAL_PARSER_TESTS_V1
#[cfg(test)]
mod incremental_stream_tests {
    use super::*;

    fn normalize(blocks: Vec<ParsedBlock>) -> Vec<ParsedBlock> {
        let mut result = Vec::new();
        for block in blocks {
            match block {
                ParsedBlock::Text(text) => {
                    if let Some(ParsedBlock::Text(previous)) = result.last_mut() {
                        previous.push_str(&text);
                    } else {
                        result.push(ParsedBlock::Text(text));
                    }
                }
                other => result.push(other),
            }
        }
        result
    }

    #[test]
    fn text_is_available_before_finish_and_not_retained() {
        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        assert_eq!(
            parser.push_stream(b"hello ").expect("text"),
            vec![ParsedBlock::Text("hello ".into())]
        );
        assert!(parser.bytes.is_empty());
        assert_eq!(
            parser.push_stream("雪".as_bytes()).expect("unicode"),
            vec![ParsedBlock::Text("雪".into())]
        );
        assert!(parser.bytes.is_empty());
        assert!(parser.finish().expect("completion").is_empty());
    }

    #[test]
    fn call_is_emitted_only_when_its_closing_delimiter_arrives() {
        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        assert!(
            parser
                .push_stream(br#"<tool_call>{"name":"fs.read","arguments":{}}"#)
                .expect("body")
                .is_empty()
        );
        let output = parser.push_stream(b"</tool_call>").expect("close");
        assert!(
            matches!(output.as_slice(), [ParsedBlock::ToolCall(call)] if call.name == "fs.read")
        );
        assert!(parser.bytes.is_empty());
        assert!(parser.finish().expect("completion").is_empty());
    }

    #[test]
    fn every_two_chunk_split_matches_atomic_output() {
        let text = concat!(
            "Reading 雪. ",
            "<tool_call>",
            r#"{"name":"fs.read","arguments":{"path":"résumé.txt"}}"#,
            "</tool_call>",
            " done < 7"
        );
        let mut atomic = ToolCallParser::new(ParserLimits::default()).expect("parser");
        atomic.push(text.as_bytes()).expect("input");
        let expected = atomic.finish().expect("atomic");
        for split in 0..=text.len() {
            let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
            let mut output = parser
                .push_stream(&text.as_bytes()[..split])
                .expect("first");
            output.extend(
                parser
                    .push_stream(&text.as_bytes()[split..])
                    .expect("second"),
            );
            output.extend(parser.finish().expect("completion"));
            assert_eq!(normalize(output), expected, "split {split}");
        }
    }

    #[test]
    fn one_byte_chunks_preserve_multiple_calls_and_text_order() {
        let text = concat!(
            "雪",
            "<tool_call>",
            r#"{"name":"a","arguments":{}}"#,
            "</tool_call>",
            "middle",
            "<tool_call>",
            r#"{"name":"b","arguments":{}}"#,
            "</tool_call>",
            "tail"
        );
        let mut atomic = ToolCallParser::new(ParserLimits::default()).expect("parser");
        atomic.push(text.as_bytes()).expect("atomic input");
        let expected = atomic.finish().expect("atomic");
        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        let mut output = Vec::new();
        for byte in text.as_bytes() {
            output.extend(
                parser
                    .push_stream(std::slice::from_ref(byte))
                    .expect("byte"),
            );
        }
        output.extend(parser.finish().expect("completion"));
        assert_eq!(normalize(output), expected);
    }

    #[test]
    fn a_later_error_prevents_the_generation_commit_gate() {
        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        let proposals = parser
            .push_stream(br#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#)
            .expect("tentative proposal");
        assert_eq!(proposals.len(), 1);
        assert!(
            parser
                .push_stream(b"<tool_call>{broken}</tool_call>")
                .is_err()
        );
        let committed = parser.finish().is_ok();
        // This models the required consumer gate, not a real executor test.
        let effects = if committed { proposals.len() } else { 0 };
        assert_eq!(effects, 0);
    }

    #[test]
    fn incomplete_call_and_control_prefix_fail_at_completion() {
        for text in [
            b"<tool_call>{".as_slice(),
            b"<tool_".as_slice(),
            b"<thi".as_slice(),
        ] {
            let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
            parser.push_stream(text).expect("pending");
            assert!(parser.finish().is_err());
        }
    }

    #[test]
    fn total_output_limit_counts_already_emitted_bytes() {
        let mut parser = ToolCallParser::new(ParserLimits {
            max_output_bytes: 4,
            max_call_bytes: 4,
            max_calls: 1,
        })
        .expect("limits");
        parser.push_stream(b"1234").expect("four bytes");
        assert!(parser.bytes.is_empty());
        assert_eq!(parser.push_stream(b"5"), Err(ToolParseError::OutputLimit));
        assert_eq!(parser.finish(), Err(ToolParseError::FailedParser));
    }

    #[test]
    fn streaming_and_atomic_consumption_cannot_be_mixed() {
        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        parser.push_stream(b"hello").expect("streaming");
        assert_eq!(parser.push(b"world"), Err(ToolParseError::FailedParser));
    }

    #[test]
    fn call_byte_limit_handles_a_split_closing_delimiter() {
        let body = r#"{"name":"a","arguments":{}}"#;
        let mut parser = ToolCallParser::new(ParserLimits {
            max_output_bytes: 1024,
            max_call_bytes: body.len(),
            max_calls: 1,
        })
        .expect("limits");
        parser
            .push_stream(format!("{OPEN}{body}</tool_").as_bytes())
            .expect("partial close");
        assert!(matches!(
            parser.push_stream(b"call>").expect("close").as_slice(),
            [ParsedBlock::ToolCall(_)]
        ));
        parser.finish().expect("completion");
    }

    #[test]
    fn malformed_json_fails_at_close_without_waiting_for_finish() {
        let mut parser = ToolCallParser::new(ParserLimits::default()).expect("parser");
        assert_eq!(
            parser.push_stream(b"<tool_call>{broken}</tool_call>"),
            Err(ToolParseError::InvalidJson)
        );
        assert_eq!(
            parser.push_stream(b"later"),
            Err(ToolParseError::FailedParser)
        );
    }
}
