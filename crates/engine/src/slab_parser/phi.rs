//! Phi's trained `functools[...]` grammar, using only admitted slab storage.
use super::parser::commit_phi_json_call;
use crate::{
    error::EngineError,
    parser_allocation::{FrozenParserAllocation, ParserAllocation},
};

const OPEN: &[u8] = b"functools[";
const RESERVED: &[&[u8]] = &[
    b"<|",
    b"<tool_call",
    b"</tool_call",
    b"<think>",
    b"</think>",
];
fn bad(message: &'static str) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "Phi tool parsing",
        message: message.into(),
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Text,
    First,
    Next,
    Object,
    AfterObject,
    Complete,
}

/// Incremental function-list parser. Only a successfully finished generation
/// exposes proposals; any failure poisons this owner, including text-sink errors.
/// Drop on cancellation. JSON values retain their exact types and spellings.
pub struct PhiToolParser {
    allocation: ParserAllocation,
    mode: Mode,
    carry: usize,
    call: usize,
    depth: usize,
    quoted: bool,
    escaped: bool,
    received: usize,
    max_output: usize,
    failed: bool,
    terminal_list: bool,
}
impl PhiToolParser {
    /// Take ownership of a fixed, already-admitted slab and total output bound.
    pub fn new(mut allocation: ParserAllocation, max_output: usize) -> Result<Self, EngineError> {
        let w = allocation.workspaces();
        let longest = RESERVED
            .iter()
            .map(|s| s.len())
            .chain([OPEN.len()])
            .max()
            .unwrap();
        if max_output == 0 || w.carry.len() < longest || w.stack.is_empty() || w.stack.len() > 128 {
            return Err(bad("invalid output/carry/depth capacity"));
        }
        Ok(Self {
            allocation,
            mode: Mode::Text,
            carry: 0,
            call: 0,
            depth: 0,
            quoted: false,
            escaped: false,
            received: 0,
            max_output,
            failed: false,
            terminal_list: false,
        })
    }
    /// End this profile turn at the first complete nonempty function list.
    /// Select before supplying output. A non-whitespace suffix within the same
    /// token is an error, never silently discarded or published as an answer.
    pub fn with_terminal_list(mut self) -> Self {
        self.terminal_list = true;
        self
    }

    /// Whether the configured protocol end condition has completed successfully.
    /// Proposals remain private until `finish`, even at this boundary.
    pub fn is_sequence_complete(&self) -> bool {
        self.terminal_list && !self.failed && self.mode == Mode::Complete
    }
    /// Consume a valid UTF-8 fragment and stream only ordinary assistant text.
    /// Any parsing or sink failure permanently prevents batch publication.
    pub fn push(
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
    fn append(&mut self, byte: u8) -> Result<(), EngineError> {
        let w = self.allocation.workspaces();
        if self.call >= w.call.len() {
            return Err(bad("call byte capacity exceeded"));
        }
        w.call[self.call] = byte;
        self.call += 1;
        Ok(())
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
            return Err(bad("total output capacity exceeded"));
        }
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let byte = bytes[i];
            match self.mode {
                Mode::Complete => {
                    if !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
                        return Err(bad("non-whitespace after terminal function list"));
                    }
                    i += 1;
                }
                Mode::First | Mode::Next => {
                    if matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
                        i += 1;
                        continue;
                    }
                    if byte != b'{' {
                        return Err(bad("function list requires a nonempty sequence of objects"));
                    }
                    self.call = 0;
                    self.depth = 1;
                    self.quoted = false;
                    self.escaped = false;
                    self.append(byte)?;
                    self.mode = Mode::Object;
                    i += 1;
                }
                Mode::Object => {
                    self.append(byte)?;
                    i += 1;
                    if self.quoted {
                        if self.escaped {
                            self.escaped = false;
                        } else if byte == b'\\' {
                            self.escaped = true;
                        } else if byte == b'"' {
                            self.quoted = false;
                        }
                    } else {
                        match byte {
                            b'"' => self.quoted = true,
                            b'{' | b'[' => {
                                self.depth += 1;
                                if self.depth > self.allocation.workspaces().stack.len() {
                                    return Err(bad("JSON depth capacity exceeded"));
                                }
                            }
                            b'}' | b']' => {
                                self.depth = self
                                    .depth
                                    .checked_sub(1)
                                    .ok_or_else(|| bad("unmatched JSON closer"))?;
                                if self.depth == 0 {
                                    commit_phi_json_call(&mut self.allocation, self.call)?;
                                    self.mode = Mode::AfterObject;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Mode::AfterObject => {
                    match byte {
                        b',' => self.mode = Mode::Next,
                        b']' => {
                            self.mode = if self.terminal_list {
                                Mode::Complete
                            } else {
                                Mode::Text
                            };
                        }
                        b' ' | b'\t' | b'\n' | b'\r' => {}
                        _ => return Err(bad("expected function-list comma or closing bracket")),
                    }
                    i += 1;
                }
                Mode::Text => {
                    if self.carry == 0 {
                        if byte == b'f' || byte == b'<' {
                            self.allocation.workspaces().carry[0] = byte;
                            self.carry = 1;
                            i += 1;
                        } else {
                            let start = i;
                            while i < bytes.len() && bytes[i] != b'f' && bytes[i] != b'<' {
                                i += 1;
                            }
                            emit(&text[start..i])?;
                        }
                        continue;
                    }
                    let (opening, reserved, prefix) = {
                        let w = self.allocation.workspaces();
                        w.carry[self.carry] = byte;
                        let candidate = &w.carry[..self.carry + 1];
                        (
                            candidate == OPEN,
                            RESERVED.iter().any(|s| candidate.starts_with(s)),
                            OPEN.starts_with(candidate)
                                || RESERVED.iter().any(|s| s.starts_with(candidate)),
                        )
                    };
                    if reserved {
                        return Err(bad("reserved control syntax in assistant output"));
                    }
                    if opening {
                        self.carry = 0;
                        self.mode = Mode::First;
                        i += 1;
                    } else if prefix {
                        self.carry += 1;
                        i += 1;
                    } else {
                        let w = self.allocation.workspaces();
                        let prefix = std::str::from_utf8(&w.carry[..self.carry])
                            .map_err(|_| bad("invalid delimiter carry"))?;
                        emit(prefix)?;
                        self.carry = 0;
                    }
                }
            }
        }
        Ok(())
    }
    /// Flush ordinary trailing text prefixes, then transfer the complete batch.
    /// No partially parsed or previously failed generation can be frozen.
    pub fn finish(
        mut self,
        mut emit: impl FnMut(&str) -> Result<(), EngineError>,
    ) -> Result<FrozenParserAllocation, EngineError> {
        if self.failed || !matches!(self.mode, Mode::Text | Mode::Complete) {
            return Err(bad("failed or incomplete function list"));
        }
        if self.carry > 0 {
            let w = self.allocation.workspaces();
            let pending = &w.carry[..self.carry];
            if pending == b"functools" || pending.starts_with(b"<") {
                return Err(bad("incomplete structural marker"));
            }
            emit(std::str::from_utf8(pending).map_err(|_| bad("invalid delimiter carry"))?)?;
        }
        Ok(self.allocation.freeze())
    }
}
