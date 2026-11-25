//! Incremental trained XML envelopes with a separate, non-executable reasoning sink.
use super::{XmlToolSchema, commit_xml_tool_call};
use crate::{
    error::EngineError,
    parser_allocation::{FrozenParserAllocation, ParserAllocation},
};

const OPEN: &[u8] = b"<tool_call>";
const CLOSE: &[u8] = b"</tool_call>";
const THINK: &[u8] = b"<think>";
const END_THINK: &[u8] = b"</think>";
const RESERVED: &[&[u8]] = &[
    b"<|",
    b"<tool_response",
    b"</tool_response",
    b"<function=",
    b"</function",
    b"<parameter=",
    b"</parameter",
];

fn bad(message: &'static str) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "streaming XML tools",
        message: message.into(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Text,
    Reasoning,
    Call,
}

/// Streaming tool parser that owns one previously admitted fixed slab.
///
/// Text and reasoning go to separate callbacks. Tool-like text inside reasoning
/// cannot create a proposal. Calls are accessible only after successful finish,
/// so a later malformed call or sink failure invalidates the complete batch.
/// Drop this owner on cancellation. Its schema and owner layout must be counted
/// separately if the caller allocates them; the parser never grows its storage.
pub struct XmlToolParser<S> {
    allocation: ParserAllocation,
    schema: S,
    mode: Mode,
    carry: usize,
    call: usize,
    received: usize,
    max_output: usize,
    after_call: bool,
    tool_call_mode: crate::types::ToolCallMode,
    failed: bool,
}

impl<S: XmlToolSchema> XmlToolParser<S> {
    /// Take ownership of exact caller-admitted storage and immutable schema lookup.
    pub fn new(
        mut allocation: ParserAllocation,
        schema: S,
        max_output: usize,
    ) -> Result<Self, EngineError> {
        let w = allocation.workspaces();
        let longest = RESERVED
            .iter()
            .map(|s| s.len())
            .chain([OPEN.len(), CLOSE.len(), THINK.len(), END_THINK.len()])
            .max()
            .unwrap_or(0);
        if max_output == 0 || w.carry.len() < longest || w.stack.len() > 128 {
            return Err(bad("invalid output/carry/depth capacity"));
        }
        Ok(Self {
            allocation,
            schema,
            mode: Mode::Text,
            carry: 0,
            call: 0,
            received: 0,
            max_output,
            after_call: false,
            tool_call_mode: crate::types::ToolCallMode::Batch,
            failed: false,
        })
    }

    /// Select the tool-call boundary before decoding any input.
    pub(crate) fn with_tool_call_mode(mut self, mode: crate::types::ToolCallMode) -> Self {
        self.tool_call_mode = mode;
        self
    }

    /// True only after a complete schema-validated call ends this generation.
    pub(crate) fn is_sequence_complete(&self) -> bool {
        self.tool_call_mode == crate::types::ToolCallMode::Single && self.after_call && !self.failed
    }

    /// Consume a valid UTF-8 fragment. Both callbacks may fail; either failure
    /// permanently poisons this parser and prevents proposal publication.
    pub fn push(
        &mut self,
        text: &str,
        mut emit_text: impl FnMut(&str) -> Result<(), EngineError>,
        mut emit_reasoning: impl FnMut(&str) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        if self.failed {
            return Err(bad("parser is poisoned"));
        }
        let result = self.process(text, &mut emit_text, &mut emit_reasoning);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn process(
        &mut self,
        text: &str,
        emit_text: &mut impl FnMut(&str) -> Result<(), EngineError>,
        emit_reasoning: &mut impl FnMut(&str) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        self.received = self
            .received
            .checked_add(text.len())
            .ok_or_else(|| bad("output size overflow"))?;
        if self.received > self.max_output {
            return Err(bad("cumulative output capacity exceeded"));
        }
        let input = text.as_bytes();
        let mut i = 0;
        while i < input.len() {
            if self.is_sequence_complete() {
                // A sampled token is indivisible. Permit trailing XML whitespace,
                // but never discard a second call, partial marker or other data.
                if input[i..]
                    .iter()
                    .all(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
                {
                    break;
                }
                return Err(bad("unexpected data after the single tool call"));
            }
            if self.mode == Mode::Call {
                if self.carry == 0 {
                    if input[i] == b'<' {
                        self.allocation.workspaces().carry[0] = b'<';
                        self.carry = 1;
                    } else {
                        self.append_call(input[i])?;
                    }
                    i += 1;
                } else if CLOSE[self.carry] == input[i] {
                    self.allocation.workspaces().carry[self.carry] = input[i];
                    self.carry += 1;
                    i += 1;
                    if self.carry == CLOSE.len() {
                        commit_xml_tool_call(&mut self.allocation, self.call, &self.schema)?;
                        self.mode = Mode::Text;
                        self.carry = 0;
                        self.call = 0;
                        self.after_call = true;
                    }
                } else {
                    for k in 0..self.carry {
                        let byte = self.allocation.workspaces().carry[k];
                        self.append_call(byte)?;
                    }
                    self.carry = 0;
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
                    let part = &text[start..i];
                    if self.mode == Mode::Reasoning {
                        emit_reasoning(part)?;
                    } else if self.after_call {
                        if !part
                            .bytes()
                            .all(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
                        {
                            return Err(bad("text after tool call"));
                        }
                    } else {
                        emit_text(part)?;
                    }
                }
                continue;
            }
            let (open, think, end_think, rejected, prefix) = {
                let w = self.allocation.workspaces();
                w.carry[self.carry] = input[i];
                let candidate = &w.carry[..self.carry + 1];
                if self.mode == Mode::Reasoning {
                    (
                        false,
                        candidate == THINK,
                        candidate == END_THINK,
                        false,
                        THINK.starts_with(candidate) || END_THINK.starts_with(candidate),
                    )
                } else {
                    (
                        candidate == OPEN,
                        candidate == THINK,
                        candidate == END_THINK,
                        RESERVED.iter().any(|s| candidate.starts_with(s))
                            || candidate.starts_with(b"</tool_call")
                            || (candidate.starts_with(b"<tool_call")
                                && candidate.len() > 10
                                && candidate != OPEN),
                        OPEN.starts_with(candidate)
                            || CLOSE.starts_with(candidate)
                            || THINK.starts_with(candidate)
                            || END_THINK.starts_with(candidate)
                            || RESERVED.iter().any(|s| s.starts_with(candidate)),
                    )
                }
            };
            if rejected {
                return Err(bad("reserved or malformed control syntax"));
            }
            if end_think {
                if self.mode != Mode::Reasoning {
                    return Err(bad("unmatched reasoning close"));
                }
                self.mode = Mode::Text;
                self.carry = 0;
                i += 1;
            } else if think {
                if self.mode == Mode::Reasoning || self.after_call {
                    return Err(bad("nested or late reasoning"));
                }
                self.mode = Mode::Reasoning;
                self.carry = 0;
                i += 1;
            } else if open {
                self.mode = Mode::Call;
                self.carry = 0;
                i += 1;
            } else if prefix {
                self.carry += 1;
                i += 1;
            } else {
                let w = self.allocation.workspaces();
                let part = std::str::from_utf8(&w.carry[..self.carry])
                    .map_err(|_| bad("non-ASCII marker carry"))?;
                if self.mode == Mode::Reasoning {
                    emit_reasoning(part)?;
                } else if self.after_call {
                    return Err(bad("text after tool call"));
                } else {
                    emit_text(part)?;
                }
                self.carry = 0;
            }
        }
        Ok(())
    }

    fn append_call(&mut self, byte: u8) -> Result<(), EngineError> {
        let w = self.allocation.workspaces();
        let slot = w
            .call
            .get_mut(self.call)
            .ok_or_else(|| bad("call capacity exceeded"))?;
        *slot = byte;
        self.call += 1;
        Ok(())
    }

    /// Release immutable untrusted proposals only for a fully valid generation.
    /// This does not authorize tool execution or bypass downstream schema checks.
    pub fn finish(self) -> Result<FrozenParserAllocation, EngineError> {
        if self.failed {
            return Err(bad("parser is poisoned"));
        }
        if self.mode != Mode::Text || self.carry != 0 {
            return Err(bad("incomplete XML/reasoning boundary"));
        }
        Ok(self.allocation.freeze())
    }
}
