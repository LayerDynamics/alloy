//! Bounded Qwen tool-envelope parsing in caller-admitted storage.
//! Success-path JSON validation uses no Map, Value, String, or Vec storage.
//! The bounded recursive control flow is capped at 128 levels; heap scratch,
//! keys, nodes, pending bytes, and retained proposals reside in the slab.
fn execution_error(stage: &'static str, message: impl Into<String>) -> EngineError {
    EngineError::ExecutionFailed {
        stage,
        message: message.into(),
    }
}
fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        message: message.into(),
    }
}
use crate::{
    error::EngineError,
    parser_allocation::{FrozenParserAllocation, JsonFrame, JsonKey, JsonNode, ParserAllocation},
};
use std::{ops::Range, str};
const OPEN: &[u8] = b"<tool_call>";
const CLOSE: &[u8] = b"</tool_call>";
const FORBIDDEN: &[&[u8]] = &[
    b"<|",
    b"<think>",
    b"</think>",
    b"<tool_response",
    b"</tool_response",
    b"<function=",
    b"<parameter=",
];
fn bad(message: &'static str) -> EngineError {
    execution_error("slab tool parsing", message)
}

pub(crate) struct SlabParser {
    allocation: ParserAllocation,
    carry: usize,
    call: usize,
    inside: bool,
    failed: bool,
    received: usize,
    max_output: usize,
}
impl SlabParser {
    pub(crate) fn new(
        mut allocation: ParserAllocation,
        max_output: usize,
    ) -> Result<Self, EngineError> {
        let work = allocation.workspaces();
        let marker = FORBIDDEN
            .iter()
            .map(|x| x.len())
            .chain([OPEN.len(), CLOSE.len()])
            .max()
            .unwrap_or(0);
        if max_output == 0 || work.carry.len() < marker || work.stack.len() > 128 {
            return Err(bad(
                "invalid carry/output/depth limits; supported nesting maximum is 128",
            ));
        }
        Ok(Self {
            allocation,
            carry: 0,
            call: 0,
            inside: false,
            failed: false,
            received: 0,
            max_output,
        })
    }
    pub(crate) fn push(
        &mut self,
        text: &str,
        mut emit: impl FnMut(&str) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        if self.failed {
            return Err(bad("parser is poisoned"));
        }
        let result = self.process(text, &mut emit);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn process(
        &mut self,
        text: &str,
        emit: &mut impl FnMut(&str) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        self.received = self
            .received
            .checked_add(text.len())
            .ok_or_else(|| bad("output length overflow"))?;
        if self.received > self.max_output {
            return Err(bad("cumulative output byte limit exceeded"));
        }
        let input = text.as_bytes();
        let mut i = 0;
        while i < input.len() {
            if self.inside {
                if self.carry == 0 {
                    if input[i] == b'<' {
                        self.allocation.workspaces().carry[0] = b'<';
                        self.carry = 1;
                        i += 1;
                    } else {
                        self.append_call(input[i])?;
                        i += 1;
                    }
                } else {
                    let matches = CLOSE[self.carry] == input[i];
                    if matches {
                        self.allocation.workspaces().carry[self.carry] = input[i];
                        self.carry += 1;
                        i += 1;
                        if self.carry == CLOSE.len() {
                            self.complete_call()?;
                            self.carry = 0;
                            self.call = 0;
                            self.inside = false;
                        }
                    } else {
                        for k in 0..self.carry {
                            let byte = self.allocation.workspaces().carry[k];
                            self.append_call(byte)?;
                        }
                        self.carry = 0;
                    }
                }
                continue;
            }
            if self.carry == 0 {
                if input[i] == b'<' {
                    self.allocation.workspaces().carry[0] = b'<';
                    self.carry = 1;
                    i += 1;
                } else {
                    let start = i;
                    while i < input.len() && input[i] != b'<' {
                        i += 1;
                    }
                    emit(&text[start..i])?;
                }
                continue;
            }
            // Only ASCII structural prefixes enter carry. On a mismatch,
            // emit the old ASCII prefix and reprocess the current input byte.
            let (opening, reserved, closing, malformed, prefix) = {
                let w = self.allocation.workspaces();
                w.carry[self.carry] = input[i];
                let candidate = &w.carry[..self.carry + 1];
                (
                    candidate == OPEN,
                    FORBIDDEN.iter().any(|m| candidate.starts_with(m)),
                    candidate.starts_with(b"</tool_call"),
                    candidate.starts_with(b"<tool_call")
                        && candidate.len() > b"<tool_call".len()
                        && candidate != OPEN,
                    OPEN.starts_with(candidate)
                        || FORBIDDEN.iter().any(|m| m.starts_with(candidate))
                        || CLOSE.starts_with(candidate),
                )
            };
            if reserved {
                return Err(bad("reserved control syntax in assistant text"));
            }
            if closing || malformed {
                return Err(bad("malformed or unmatched tool delimiter"));
            }
            if opening {
                self.carry = 0;
                self.inside = true;
                i += 1;
            } else if prefix {
                self.carry += 1;
                i += 1;
            } else {
                let w = self.allocation.workspaces();
                let text = str::from_utf8(&w.carry[..self.carry])
                    .map_err(|_| internal("non-ASCII delimiter carry"))?;
                emit(text)?;
                self.carry = 0;
            }
        }
        Ok(())
    }
    fn append_call(&mut self, byte: u8) -> Result<(), EngineError> {
        let w = self.allocation.workspaces();
        if self.call >= w.call.len() {
            return Err(bad("call byte capacity exceeded"));
        }
        w.call[self.call] = byte;
        self.call += 1;
        Ok(())
    }
    fn complete_call(&mut self) -> Result<(), EngineError> {
        let (decoded, name, args) = {
            let w = self.allocation.workspaces();
            let body = &w.call[..self.call];
            str::from_utf8(body).map_err(|_| bad("invalid UTF-8 in tool envelope"))?;
            if body.windows(OPEN.len()).any(|x| x == OPEN) {
                return Err(bad("nested tool delimiter"));
            }
            let mut json = JsonCheck {
                input: body,
                pos: 0,
                decoded: w.decoded,
                used: 0,
                stack: w.stack,
                keys: w.keys,
                key_count: 0,
                nodes: w.nodes,
                node_count: 0,
                name: None,
                args: None,
            };
            json.value(0, usize::MAX, true)?;
            json.ws();
            if json.pos != body.len() {
                return Err(bad("trailing tool-envelope data"));
            }
            (
                json.used,
                json.name.ok_or_else(|| bad("missing tool name"))?,
                json.args.ok_or_else(|| bad("missing object arguments"))?,
            )
        };
        self.allocation
            .commit_validated_call(decoded, name, self.call, args)
            .map_err(|e| execution_error("commit slab proposal", e.to_string()))?;
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<FrozenParserAllocation, EngineError> {
        if self.failed {
            return Err(bad("parser is poisoned"));
        }
        if self.inside {
            return Err(bad("incomplete tool envelope"));
        }
        if self.carry != 0 {
            return Err(bad("incomplete structural prefix"));
        }
        Ok(self.allocation.freeze())
    }
}

struct Val {
    kind: usize,
    start: usize,
    end: usize,
    decoded: Range<usize>,
}
struct JsonCheck<'a> {
    input: &'a [u8],
    pos: usize,
    decoded: &'a mut [u8],
    used: usize,
    stack: &'a mut [JsonFrame],
    keys: &'a mut [JsonKey],
    key_count: usize,
    nodes: &'a mut [JsonNode],
    node_count: usize,
    name: Option<Range<usize>>,
    args: Option<Range<usize>>,
}
impl JsonCheck<'_> {
    fn ws(&mut self) {
        while self.pos < self.input.len()
            && matches!(self.input[self.pos], b' ' | b'\n' | b'\r' | b'\t')
        {
            self.pos += 1;
        }
    }
    fn eat(&mut self, byte: u8) -> bool {
        self.ws();
        if self.input.get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn required(&mut self, byte: u8) -> Result<(), EngineError> {
        if self.eat(byte) {
            Ok(())
        } else {
            Err(bad("invalid JSON separator"))
        }
    }
    fn write(&mut self, bytes: &[u8]) -> Result<(), EngineError> {
        let end = self
            .used
            .checked_add(bytes.len())
            .ok_or_else(|| bad("decoded scratch overflow"))?;
        if end > self.decoded.len() {
            return Err(bad("decoded-string capacity exceeded"));
        }
        self.decoded[self.used..end].copy_from_slice(bytes);
        self.used = end;
        Ok(())
    }
    fn hex4(&mut self) -> Result<u32, EngineError> {
        let mut value = 0;
        for _ in 0..4 {
            let b = *self
                .input
                .get(self.pos)
                .ok_or_else(|| bad("truncated Unicode escape"))?;
            self.pos += 1;
            let digit = match b {
                b'0'..=b'9' => u32::from(b - b'0'),
                b'a'..=b'f' => u32::from(b - b'a' + 10),
                b'A'..=b'F' => u32::from(b - b'A' + 10),
                _ => return Err(bad("invalid Unicode escape")),
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }
    fn string(&mut self) -> Result<Range<usize>, EngineError> {
        self.required(b'"')?;
        let start = self.used;
        loop {
            let byte = *self
                .input
                .get(self.pos)
                .ok_or_else(|| bad("unterminated JSON string"))?;
            self.pos += 1;
            match byte {
                b'"' => return Ok(start..self.used),
                0..=0x1f => return Err(bad("unescaped JSON control character")),
                b'\\' => {
                    let escape = *self
                        .input
                        .get(self.pos)
                        .ok_or_else(|| bad("truncated JSON escape"))?;
                    self.pos += 1;
                    match escape {
                        b'"' | b'\\' | b'/' => self.write(&[escape])?,
                        b'b' => self.write(&[8])?,
                        b'f' => self.write(&[12])?,
                        b'n' => self.write(b"\n")?,
                        b'r' => self.write(b"\r")?,
                        b't' => self.write(b"\t")?,
                        b'u' => {
                            let first = self.hex4()?;
                            let scalar = if (0xd800..=0xdbff).contains(&first) {
                                if self.input.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(bad("missing low surrogate"));
                                }
                                self.pos += 2;
                                let low = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err(bad("invalid low surrogate"));
                                }
                                0x10000 + ((first - 0xd800) << 10) + (low - 0xdc00)
                            } else {
                                first
                            };
                            let character = char::from_u32(scalar)
                                .ok_or_else(|| bad("invalid Unicode scalar"))?;
                            let mut buffer = [0; 4];
                            self.write(character.encode_utf8(&mut buffer).as_bytes())?;
                        }
                        _ => return Err(bad("invalid JSON escape")),
                    }
                }
                other => self.write(&[other])?,
            }
        }
    }
    fn number(&mut self) -> Result<(), EngineError> {
        if self.input.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.input.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while self.input.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
            }
            _ => return Err(bad("invalid JSON number")),
        }
        if self.input.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            let start = self.pos;
            while self.input.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
            if start == self.pos {
                return Err(bad("missing fractional digits"));
            }
        }
        if self
            .input
            .get(self.pos)
            .is_some_and(|x| matches!(x, b'e' | b'E'))
        {
            self.pos += 1;
            if self
                .input
                .get(self.pos)
                .is_some_and(|x| matches!(x, b'+' | b'-'))
            {
                self.pos += 1;
            }
            let start = self.pos;
            while self.input.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
            if start == self.pos {
                return Err(bad("missing exponent digits"));
            }
        }
        Ok(())
    }
    fn value(&mut self, depth: usize, parent: usize, root: bool) -> Result<Val, EngineError> {
        self.ws();
        let start = self.pos;
        if depth >= self.stack.len() || depth >= 128 {
            return Err(bad("JSON nesting capacity exceeded"));
        }
        if self.node_count >= self.nodes.len() {
            return Err(bad("JSON node capacity exceeded"));
        }
        let node = self.node_count;
        self.node_count += 1;
        self.stack[depth] = JsonFrame {
            kind: 0,
            state: 0,
            object_id: node,
            first_key: self.key_count,
        };
        let mut decoded = 0..0;
        let kind = match self.input.get(self.pos).copied() {
            Some(b'{') => {
                self.pos += 1;
                let mut members = 0;
                if !self.eat(b'}') {
                    loop {
                        let key = self.string()?;
                        for prior in &self.keys[..self.key_count] {
                            if prior.object_id == node
                                && self.decoded[prior.start..prior.start + prior.len]
                                    == self.decoded[key.clone()]
                            {
                                return Err(bad("duplicate JSON object key"));
                            }
                        }
                        if self.key_count >= self.keys.len() {
                            return Err(bad("JSON key capacity exceeded"));
                        }
                        self.keys[self.key_count] = JsonKey {
                            object_id: node,
                            start: key.start,
                            len: key.end - key.start,
                        };
                        self.key_count += 1;
                        self.required(b':')?;
                        let value = self.value(depth + 1, node, false)?;
                        if root {
                            match &self.decoded[key] {
                                b"name" if value.kind == 3 => self.name = Some(value.decoded),
                                b"arguments" if value.kind == 1 => {
                                    self.args = Some(value.start..value.end)
                                }
                                _ => {
                                    return Err(bad(
                                        "tool envelope permits only name string and arguments object",
                                    ));
                                }
                            }
                        }
                        members += 1;
                        if self.eat(b'}') {
                            break;
                        }
                        self.required(b',')?;
                    }
                }
                if root && members != 2 {
                    return Err(bad("tool envelope requires exactly two members"));
                }
                1
            }
            Some(b'[') => {
                if root {
                    return Err(bad("tool envelope must be an object"));
                }
                self.pos += 1;
                if !self.eat(b']') {
                    loop {
                        self.value(depth + 1, node, false)?;
                        if self.eat(b']') {
                            break;
                        }
                        self.required(b',')?;
                    }
                }
                2
            }
            Some(b'"') => {
                if root {
                    return Err(bad("tool envelope must be an object"));
                }
                decoded = self.string()?;
                3
            }
            Some(b't') | Some(b'f') | Some(b'n') => {
                if root {
                    return Err(bad("tool envelope must be an object"));
                }
                let literal: &[u8] = match self.input[self.pos] {
                    b't' => b"true",
                    b'f' => b"false",
                    _ => b"null",
                };
                if self.input.get(self.pos..self.pos + literal.len()) != Some(literal) {
                    return Err(bad("invalid JSON literal"));
                }
                self.pos += literal.len();
                4
            }
            Some(b'-' | b'0'..=b'9') => {
                if root {
                    return Err(bad("tool envelope must be an object"));
                }
                self.number()?;
                5
            }
            _ => return Err(bad("invalid JSON value")),
        };
        self.nodes[node] = JsonNode {
            kind,
            parent,
            start,
            len: self.pos - start,
        };
        Ok(Val {
            kind,
            start,
            end: self.pos,
            decoded,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/slab_parser_tests.rs"]
mod tests;

// W2_SLAB_BACKED_DECODER_V1

// W2_PARSER_PLACEMENT_PARITY_V1
#[cfg(test)]
#[path = "../../tests/unit/parser_parity.rs"]
mod parity;
