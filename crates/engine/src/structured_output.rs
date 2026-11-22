//! Bounded JSON object constraints applied before native token selection.
//!
//! This deliberately supports only the checkpoint schema subset. It never
//! generates field contents, repairs output, or decides whether facts are true.
use crate::{error::EngineError, family::ExecutionControl};
use std::sync::Arc;

const MAX_VOCAB: usize = 1 << 20;
const MAX_PIECES: usize = 4 << 20;
const MAX_PIECE: usize = 1024;
const MAX_OUTPUT: usize = 64 << 10;

/// Fixed host workspace: one 8-byte range per vocabulary slot, one raw-byte
/// buffer, and 128 KiB for schema, parser state, EOS and construction scratch
/// (including at most one 64 KiB tokenizer spelling).
/// This is separate from model/MLX workspace and is reserved before allocation.
pub const JSON_WORKSPACE_BYTES: u64 = (MAX_VOCAB * 8 + MAX_PIECES + (128 << 10)) as u64;

/// Supported required object-field types. Nested objects are not supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonFieldType {
    /// A JSON string; when true, at least one character is required.
    String {
        /// Require at least one decoded character (whitespace is a character).
        nonempty: bool,
    },
    /// An array containing zero or more JSON strings.
    StringArray,
    /// Either true or false, selected by the model.
    Boolean,
    /// One schema-defined nonnegative integer (for example a schema version).
    UnsignedConstant(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Field {
    name: Box<str>,
    kind: JsonFieldType,
}

/// A closed object with all fields required, accepted in any order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonOutputSchema(Arc<[Field]>);

impl JsonOutputSchema {
    /// Validate 1–32 unique ASCII identifier keys, each at most 64 bytes.
    pub fn new(fields: Vec<(String, JsonFieldType)>) -> Result<Self, EngineError> {
        if fields.is_empty() || fields.len() > 32 {
            return Err(invalid("schema must contain 1..=32 fields"));
        }
        for (index, (name, _)) in fields.iter().enumerate() {
            if name.is_empty()
                || name.len() > 64
                || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                || fields[..index].iter().any(|(other, _)| other == name)
            {
                return Err(invalid(
                    "schema keys must be unique ASCII identifiers of 1..=64 bytes",
                ));
            }
        }
        Ok(Self(
            fields
                .into_iter()
                .map(|(name, kind)| Field {
                    name: name.into_boxed_str(),
                    kind,
                })
                .collect(),
        ))
    }

    fn all(&self) -> u32 {
        u32::MAX >> (32 - self.0.len())
    }
}

fn invalid(message: &str) -> EngineError {
    EngineError::invalid_request("structured output", message)
}

fn failed(message: &str) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "structured output",
        message: message.into(),
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    Start,
    KeyStart,
    Key {
        candidates: u32,
        position: u8,
    },
    Colon,
    Value,
    ArrayStart,
    ArrayItem,
    String(StringState),
    ArrayAfter,
    Literal {
        bytes: [u8; 10],
        len: u8,
        position: u8,
    },
    After,
    Done,
}

#[derive(Clone, Copy, Debug, Default)]
enum Escape {
    #[default]
    None,
    Slash,
    Unicode {
        value: u16,
        digits: u8,
        low: bool,
    },
    LowSlash,
    LowU,
}

#[derive(Clone, Copy, Debug)]
struct StringState {
    escape: Escape,
    remaining: u8,
    lower: u8,
    upper: u8,
    content: bool,
}

impl Default for StringState {
    fn default() -> Self {
        Self {
            escape: Escape::None,
            remaining: 0,
            lower: 0x80,
            upper: 0xbf,
            content: false,
        }
    }
}

impl StringState {
    // true means the closing quote was consumed. Invalid prefixes fail closed.
    fn byte(&mut self, b: u8) -> Option<bool> {
        if self.remaining != 0 {
            if !(self.lower..=self.upper).contains(&b) {
                return None;
            }
            self.remaining -= 1;
            self.lower = 0x80;
            self.upper = 0xbf;
            return Some(false);
        }
        match self.escape {
            Escape::Slash => match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.escape = Escape::None
                }
                b'u' => {
                    self.escape = Escape::Unicode {
                        value: 0,
                        digits: 0,
                        low: false,
                    }
                }
                _ => return None,
            },
            Escape::LowSlash if b == b'\\' => self.escape = Escape::LowU,
            Escape::LowU if b == b'u' => {
                self.escape = Escape::Unicode {
                    value: 0,
                    digits: 0,
                    low: true,
                }
            }
            Escape::LowSlash | Escape::LowU => return None,
            Escape::Unicode { value, digits, low } => {
                let digit = (b as char).to_digit(16)? as u16;
                let value = (value << 4) | digit;
                self.escape = if digits < 3 {
                    Escape::Unicode {
                        value,
                        digits: digits + 1,
                        low,
                    }
                } else if low {
                    if !(0xdc00..=0xdfff).contains(&value) {
                        return None;
                    }
                    Escape::None
                } else if (0xd800..=0xdbff).contains(&value) {
                    Escape::LowSlash
                } else {
                    if (0xdc00..=0xdfff).contains(&value) {
                        return None;
                    }
                    Escape::None
                };
            }
            Escape::None => match b {
                b'"' => return Some(true),
                b'\\' => self.escape = Escape::Slash,
                0x20..=0x7f => {}
                0xc2..=0xdf => self.remaining = 1,
                0xe0..=0xef => {
                    self.remaining = 2;
                    if b == 0xe0 {
                        self.lower = 0xa0;
                    }
                    if b == 0xed {
                        self.upper = 0x9f;
                    }
                }
                0xf0..=0xf4 => {
                    self.remaining = 3;
                    if b == 0xf0 {
                        self.lower = 0x90;
                    }
                    if b == 0xf4 {
                        self.upper = 0x8f;
                    }
                }
                _ => return None,
            },
        }
        self.content = true;
        Some(false)
    }
}

#[derive(Clone, Copy, Debug)]
struct State {
    stage: Stage,
    seen: u32,
    field: usize,
    bytes: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            stage: Stage::Start,
            seen: 0,
            field: 0,
            bytes: 0,
        }
    }
}

impl State {
    fn accepting(self) -> bool {
        matches!(self.stage, Stage::Done)
    }

    fn feed(&mut self, schema: &JsonOutputSchema, bytes: &[u8]) -> Option<()> {
        if bytes.len() > MAX_OUTPUT.saturating_sub(self.bytes) {
            return None;
        }
        for &b in bytes {
            self.byte(schema, b)?;
        }
        self.bytes += bytes.len();
        Some(())
    }

    fn byte(&mut self, schema: &JsonOutputSchema, b: u8) -> Option<()> {
        if matches!(b, b' ' | b'\n' | b'\r' | b'\t')
            && !matches!(
                self.stage,
                Stage::Key { .. } | Stage::String(_) | Stage::Literal { .. }
            )
        {
            return Some(());
        }
        self.stage = match self.stage {
            Stage::Start if b == b'{' => Stage::KeyStart,
            Stage::KeyStart if b == b'"' => Stage::Key {
                candidates: schema.all() & !self.seen,
                position: 0,
            },
            Stage::Key {
                mut candidates,
                position,
            } => {
                for (i, field) in schema.0.iter().enumerate() {
                    let expected = field
                        .name
                        .as_bytes()
                        .get(position as usize)
                        .copied()
                        .unwrap_or(b'"');
                    if b != expected {
                        candidates &= !(1 << i);
                    }
                }
                if candidates == 0 {
                    return None;
                }
                if b == b'"' {
                    self.field = candidates.trailing_zeros() as usize;
                    self.seen |= 1 << self.field;
                    Stage::Colon
                } else {
                    Stage::Key {
                        candidates,
                        position: position + 1,
                    }
                }
            }
            Stage::Colon if b == b':' => Stage::Value,
            Stage::Value => match schema.0[self.field].kind {
                JsonFieldType::String { .. } if b == b'"' => Stage::String(StringState::default()),
                JsonFieldType::StringArray if b == b'[' => Stage::ArrayStart,
                JsonFieldType::Boolean if b == b't' || b == b'f' => {
                    let literal: &[u8] = if b == b't' { b"true" } else { b"false" };
                    let mut bytes = [0; 10];
                    bytes[..literal.len()].copy_from_slice(literal);
                    Stage::Literal {
                        bytes,
                        len: literal.len() as u8,
                        position: 1,
                    }
                }
                JsonFieldType::UnsignedConstant(mut value) => {
                    let mut bytes = [0; 10];
                    let mut len = 0;
                    loop {
                        bytes[len] = b'0' + (value % 10) as u8;
                        len += 1;
                        value /= 10;
                        if value == 0 {
                            break;
                        }
                    }
                    bytes[..len].reverse();
                    if b != bytes[0] {
                        return None;
                    }
                    if len == 1 {
                        Stage::After
                    } else {
                        Stage::Literal {
                            bytes,
                            len: len as u8,
                            position: 1,
                        }
                    }
                }
                _ => return None,
            },
            Stage::ArrayStart if b == b']' => Stage::After,
            Stage::ArrayStart | Stage::ArrayItem if b == b'"' => {
                Stage::String(StringState::default())
            }
            Stage::String(mut string) => {
                if string.byte(b)? {
                    if schema.0[self.field].kind == (JsonFieldType::String { nonempty: true })
                        && !string.content
                    {
                        return None;
                    }
                    if schema.0[self.field].kind == JsonFieldType::StringArray {
                        Stage::ArrayAfter
                    } else {
                        Stage::After
                    }
                } else {
                    Stage::String(string)
                }
            }
            Stage::ArrayAfter if b == b',' => Stage::ArrayItem,
            Stage::ArrayAfter if b == b']' => Stage::After,
            Stage::Literal {
                bytes,
                len,
                position,
            } if b == bytes[position as usize] => {
                if position + 1 == len {
                    Stage::After
                } else {
                    Stage::Literal {
                        bytes,
                        len,
                        position: position + 1,
                    }
                }
            }
            Stage::After if b == b',' && self.seen != schema.all() => Stage::KeyStart,
            Stage::After if b == b'}' && self.seen == schema.all() => Stage::Done,
            _ => return None,
        };
        Some(())
    }
}

#[derive(Clone, Copy)]
struct Piece {
    start: u32,
    len: u32,
}

/// A request-owned bounded token catalogue and incremental JSON recognizer.
/// Construct only after reserving [`JSON_WORKSPACE_BYTES`].
pub struct JsonConstraint {
    schema: JsonOutputSchema,
    pieces: Vec<Piece>,
    bytes: Vec<u8>,
    eos: Vec<u32>,
    state: State,
}

impl JsonConstraint {
    /// Build raw bytes using the exact verified incremental decoder semantics.
    /// Missing, special and empty tokens must return None. EOS is handled
    /// separately and is never interpreted as JSON bytes.
    pub fn new(
        schema: JsonOutputSchema,
        vocabulary: usize,
        eos: &[u32],
        control: &ExecutionControl,
        mut piece: impl FnMut(u32) -> Result<Option<Vec<u8>>, EngineError>,
    ) -> Result<Self, EngineError> {
        if vocabulary == 0
            || vocabulary > MAX_VOCAB
            || eos.is_empty()
            || eos.len() > 16
            || eos.iter().any(|&id| id as usize >= vocabulary)
        {
            return Err(invalid("unsupported vocabulary or EOS bounds"));
        }
        let mut pieces = Vec::new();
        let mut bytes = Vec::new();
        control.check_cancelled()?;
        pieces
            .try_reserve_exact(vocabulary)
            .map_err(|_| EngineError::AllocationFailed {
                operation: "structured-output token ranges",
                requested_bytes: Some((vocabulary * 8) as u64),
            })?;
        bytes
            .try_reserve_exact(MAX_PIECES)
            .map_err(|_| EngineError::AllocationFailed {
                operation: "structured-output token bytes",
                requested_bytes: Some(MAX_PIECES as u64),
            })?;
        for id in 0..vocabulary {
            if id % 256 == 0 {
                control.check_cancelled()?;
            }
            let token = if eos.contains(&(id as u32)) {
                None
            } else {
                piece(id as u32)?
            };
            let token = token.as_deref().unwrap_or_default();
            if token.len() > MAX_PIECE || token.len() > MAX_PIECES - bytes.len() {
                return Err(invalid("raw tokenizer bytes exceed constraint bounds"));
            }
            pieces.push(Piece {
                start: bytes.len() as u32,
                len: token.len() as u32,
            });
            bytes.extend_from_slice(token);
        }
        control.check_cancelled()?;
        Ok(Self {
            schema,
            pieces,
            bytes,
            eos: eos.to_vec(),
            state: State::default(),
        })
    }

    fn advance(&self, id: usize) -> Option<State> {
        let piece = *self.pieces.get(id)?;
        let mut state = self.state;
        if self.eos.contains(&(id as u32)) {
            return state.accepting().then_some(state);
        }
        if piece.len == 0 {
            return None;
        }
        state.feed(
            &self.schema,
            &self.bytes[piece.start as usize..(piece.start + piece.len) as usize],
        )?;
        Some(state)
    }

    /// Select the highest finite allowed logit, keeping the first token on ties.
    /// No model state or recognizer state is changed by selection.
    pub fn select(&self, logits: &[f32], control: &ExecutionControl) -> Result<u32, EngineError> {
        if logits.len() != self.pieces.len() {
            return Err(failed("constraint vocabulary mismatch"));
        }
        let mut best: Option<usize> = None;
        for (id, &logit) in logits.iter().enumerate() {
            if id % 256 == 0 {
                control.check_cancelled()?;
            }
            if logit.is_finite()
                && best.is_none_or(|old| logit > logits[old])
                && self.advance(id).is_some()
            {
                best = Some(id);
            }
        }
        control.check_cancelled()?;
        best.map(|id| id as u32)
            .ok_or_else(|| failed("no finite token satisfies the JSON constraint"))
    }

    /// Commit exactly the token that successfully executed in the native model.
    pub fn consume(&mut self, token: u32) -> Result<(), EngineError> {
        self.state = self
            .advance(token as usize)
            .ok_or_else(|| failed("selected token violates JSON constraint"))?;
        Ok(())
    }

    /// Require a complete valid object before successful generation completion.
    pub fn finish(&self) -> Result<(), EngineError> {
        if self.state.accepting() {
            Ok(())
        } else {
            Err(failed("generation ended with incomplete structured output"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cancellation::CancellationSource, types::RequestId};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn json_constraints_match_serde_strings(
            text in ".{1,100}", values in prop::collection::vec(".{0,40}", 0..8), valid in any::<bool>(),
        ) {
            let object = serde_json::json!({"v":4294967295u32,"s":text,"a":values,"b":valid});
            let input = serde_json::to_vec(&object).unwrap();
            let mut state = State::default();
            let schema = schema();
            for byte in input { prop_assert!(state.feed(&schema, &[byte]).is_some()); }
            prop_assert!(state.accepting());
        }
    }

    fn schema() -> JsonOutputSchema {
        JsonOutputSchema::new(vec![
            ("v".into(), JsonFieldType::UnsignedConstant(4294967295)),
            ("s".into(), JsonFieldType::String { nonempty: true }),
            ("a".into(), JsonFieldType::StringArray),
            ("b".into(), JsonFieldType::Boolean),
        ])
        .unwrap()
    }
    fn control() -> (CancellationSource, ExecutionControl) {
        let source = CancellationSource::new();
        let control = ExecutionControl::new(RequestId::new("json-test").unwrap(), source.token());
        (source, control)
    }
    fn accepts(schema: &JsonOutputSchema, input: &[u8]) -> bool {
        let mut state = State::default();
        state.feed(schema, input).is_some() && state.accepting()
    }

    #[test]
    fn json_constraints_accept_every_split_and_field_order() {
        let schema = schema();
        for input in [
            r#"{"v":4294967295,"s":"雪😀","a":["","one","two"],"b":true}"#,
            " \n{ \"b\" : false, \"a\": [], \"s\":\"\\uD83D\\uDE00\\u96ea\\n\\t\\r\\b\\f\\\\\\/\\\"\", \"v\":4294967295 }\t",
        ] {
            assert!(serde_json::from_str::<serde_json::Value>(input).is_ok());
            for split in 0..=input.len() {
                let mut state = State::default();
                state.feed(&schema, &input.as_bytes()[..split]).unwrap();
                state.feed(&schema, &input.as_bytes()[split..]).unwrap();
                assert!(state.accepting(), "split {split}");
            }
            let mut state = State::default();
            for b in input.as_bytes() {
                state.feed(&schema, &[*b]).unwrap();
            }
            assert!(state.accepting());
        }
    }

    #[test]
    fn json_constraints_reject_malformed_and_wrong_schema() {
        let schema = schema();
        let valid = r#"{"v":4294967295,"s":"x","a":[],"b":false}"#;
        for invalid in [
            valid.replace("4294967295", "1"),
            valid.replace("4294967295", "42949672950"),
            valid.replace("4294967295", "4294967295.0"),
            valid.replace("\"x\"", "\"\""),
            valid.replace("\"x\"", "\"\\x00\""),
            valid.replace("\"x\"", "\"\\ud800\""),
            valid.replace("\"x\"", "\"\\udc00\""),
            valid.replace("\"x\"", "\"\\ud800\\u0041\""),
            valid.replace("\"x\"", "\"line\nbreak\""),
            valid.replace("[]", "[1]"),
            valid.replace("[]", "[\"x\",]"),
            valid.replace("false", "False"),
            valid.replace("false", "f alse"),
            valid.replace("false", "\"false\""),
            valid.replace("\"b\":false", "\"s\":\"again\""),
            valid.replace(",\"b\":false", ""),
            valid.replace('}', ",\"extra\":true}"),
            valid.replace('}', ",}"),
            format!("```json\n{valid}\n```"),
            format!("{valid}x"),
            format!("{valid}{valid}"),
        ] {
            assert!(!accepts(&schema, invalid.as_bytes()), "{invalid}");
        }
        for bytes in [
            &b"\xc0\x80"[..],
            &b"\xed\xa0\x80"[..],
            &b"\xf4\x90\x80\x80"[..],
            &b"\x80"[..],
            &b"\xe2"[..],
        ] {
            let mut input = b"{\"s\":\"".to_vec();
            input.extend_from_slice(bytes);
            input.extend_from_slice(b"\",\"v\":4294967295,\"a\":[],\"b\":true}");
            assert!(!accepts(&schema, &input));
        }
        for cut in 0..valid.len() {
            assert!(!accepts(&schema, &valid.as_bytes()[..cut]));
        }
    }

    #[test]
    fn json_constraints_schema_and_output_bounds() {
        assert!(JsonOutputSchema::new(vec![]).is_err());
        assert!(JsonOutputSchema::new(vec![("x".repeat(65), JsonFieldType::Boolean)]).is_err());
        assert!(JsonOutputSchema::new(vec![("雪".into(), JsonFieldType::Boolean)]).is_err());
        assert!(JsonOutputSchema::new(vec![("x".into(), JsonFieldType::Boolean); 2]).is_err());
        assert!(
            JsonOutputSchema::new(
                (0..33)
                    .map(|i| (format!("x{i}"), JsonFieldType::Boolean))
                    .collect()
            )
            .is_err()
        );
        let schema = JsonOutputSchema::new(
            (0..32)
                .map(|i| (format!("x{i}"), JsonFieldType::Boolean))
                .collect(),
        )
        .unwrap();
        let object = format!(
            "{{{}}}",
            (0..32)
                .rev()
                .map(|i| format!("\"x{i}\":false"))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(accepts(&schema, object.as_bytes()));
        let mut state = State::default();
        state.feed(&schema, object.as_bytes()).unwrap();
        state
            .feed(&schema, &vec![b' '; MAX_OUTPUT - object.len()])
            .unwrap();
        assert!(state.accepting());
        assert!(state.feed(&schema, b" ").is_none());
        assert_eq!(std::mem::size_of::<Piece>(), 8);
    }

    #[test]
    fn json_constraints_selection_eos_cancellation_and_catalogue_bounds() {
        let (source, control) = control();
        let schema = JsonOutputSchema::new(vec![("valid".into(), JsonFieldType::Boolean)]).unwrap();
        let catalogue: &[&[u8]] = &[
            b"",
            b"```",
            b"{\"valid\":",
            b"true",
            b"false",
            b"}",
            b"true",
            b"",
        ];
        let make = || {
            JsonConstraint::new(schema.clone(), catalogue.len(), &[0], &control, |id| {
                Ok(Some(catalogue[id as usize].to_vec()))
            })
            .unwrap()
        };
        for (boolean, token) in [(true, 3), (false, 4)] {
            let mut constraint = make();
            assert!(constraint.finish().is_err());
            assert!(constraint.consume(0).is_err());
            let mut logits = [0.; 8];
            logits[0] = 100.;
            logits[1] = 99.;
            assert_eq!(constraint.select(&logits, &control).unwrap(), 2);
            constraint.consume(2).unwrap();
            logits[token] = 1.;
            assert_eq!(constraint.select(&logits, &control).unwrap(), token as u32);
            constraint.consume(token as u32).unwrap();
            assert!(constraint.finish().is_err());
            assert_eq!(constraint.select(&logits, &control).unwrap(), 5);
            constraint.consume(5).unwrap();
            constraint.finish().unwrap();
            assert_eq!(constraint.select(&logits, &control).unwrap(), 0);
            constraint.consume(0).unwrap();
            assert!(accepts(
                &schema,
                format!("{{\"valid\":{boolean}}}").as_bytes()
            ));
        }
        let mut constraint = make();
        constraint.consume(2).unwrap();
        assert_eq!(constraint.select(&[0.; 8], &control).unwrap(), 3); // stable tie
        assert!(constraint.select(&[f32::NAN; 8], &control).is_err());
        assert!(constraint.select(&[0.; 1], &control).is_err());
        assert!(
            JsonConstraint::new(
                schema.clone(),
                MAX_VOCAB + 1,
                &[0],
                &control,
                |_| unreachable!()
            )
            .is_err()
        );
        assert!(
            JsonConstraint::new(schema.clone(), 2, &[0], &control, |_| Ok(Some(vec![
                b'a';
                MAX_PIECE
                    + 1
            ])))
            .is_err()
        );
        assert!(
            JsonConstraint::new(schema.clone(), 4098, &[0], &control, |_| Ok(Some(vec![
                b'a';
                1024
            ])))
            .is_err()
        );
        source.request_cancel();
        assert!(constraint.select(&[0.; 8], &control).is_err());
        assert!(JsonConstraint::new(schema, 2, &[0], &control, |_| unreachable!()).is_err());
    }
}
