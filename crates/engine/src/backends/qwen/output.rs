//! Qwen incremental output decoding.
//! With a parser configured, tool-call blocks are parsed incrementally into
//! untrusted proposals; without one, output stays text-only.

use super::{allocation, execution_error, internal};
use crate::{
    error::EngineError,
    rendering::PinnedTokenizer,
    runtime::TextDecoder,
    types::{GenerationOutputBlock, ToolCallProposal},
};
use crate::{parser_allocation::FrozenParserAllocation, slab_parser::SlabParser};
use qwen3::tool_calls::{ParsedBlock, ToolCallParser, ToolParseError};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub(super) struct Qwen3TextDecoder {
    pub(super) tokenizer: Arc<PinnedTokenizer>,
    pub(super) special_tokens: Arc<BTreeSet<u32>>,
    pub(super) byte_decoder: Arc<BTreeMap<char, u8>>,
    pub(super) pending: Vec<u8>,
    pub(super) max_token_bytes: usize,
    pub(super) finished: bool,
    /// Present only when tool-call parsing is enabled for this generation.
    pub(super) parser: Option<ToolCallParser>,
    pub(super) slab: Option<SlabParser>,
    pub(super) completed_slab: Option<FrozenParserAllocation>,
}

fn text_blocks(text: String) -> Vec<GenerationOutputBlock> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![GenerationOutputBlock::Text(text)]
    }
}

fn parse_error(error: ToolParseError) -> EngineError {
    execution_error("tool call parsing", error.to_string())
}

fn output_blocks(parsed: Vec<ParsedBlock>) -> Result<Vec<GenerationOutputBlock>, EngineError> {
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(parsed.len())
        .map_err(|_| allocation("parsed output blocks", None))?;
    for block in parsed {
        blocks.push(match block {
            ParsedBlock::Text(text) => GenerationOutputBlock::Text(text),
            ParsedBlock::ToolCall(call) => {
                GenerationOutputBlock::ToolCall(ToolCallProposal::new(call.name, call.arguments)?)
            }
        });
    }
    Ok(blocks)
}

impl TextDecoder for Qwen3TextDecoder {
    fn push(&mut self, token: u32) -> Result<String, EngineError> {
        if self.finished {
            return Err(internal("token supplied after decoder completion"));
        }
        let spelling = self
            .tokenizer
            .tokenizer()
            .id_to_token(token)
            .ok_or_else(|| {
                execution_error(
                    "detokenize",
                    "model selected an ID absent from the tokenizer",
                )
            })?;
        if spelling.len() > self.max_token_bytes {
            return Err(execution_error(
                "detokenize",
                "token exceeds the pinned decoder bound",
            ));
        }
        // W2_QWEN_PROTOCOL_SPECIAL_TOKENS_V1
        // Protocol delimiters are syntax, not ByteLevel alphabet data.
        if self.special_tokens.contains(&token) {
            if matches!(spelling.as_str(), "<tool_call>" | "</tool_call>") {
                if !self.pending.is_empty() {
                    return Err(execution_error(
                        "detokenize",
                        "tool delimiter interrupts an incomplete UTF-8 sequence",
                    ));
                }
                return Ok(spelling);
            }
            return Ok(String::new());
        }
        self.pending
            .try_reserve_exact(spelling.chars().count())
            .map_err(|_| allocation("incremental UTF-8 bytes", Some(spelling.len())))?;
        for ch in spelling.chars() {
            self.pending
                .push(*self.byte_decoder.get(&ch).ok_or_else(|| {
                    execution_error(
                        "detokenize",
                        "token is outside the verified ByteLevel alphabet",
                    )
                })?);
        }
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => {
                return Err(execution_error(
                    "detokenize",
                    "model output contains invalid UTF-8 bytes",
                ));
            }
        };
        if valid == 0 {
            return Ok(String::new());
        }
        let suffix = self.pending.split_off(valid);
        let complete = std::mem::replace(&mut self.pending, suffix);
        String::from_utf8(complete).map_err(|_| internal("validated UTF-8 conversion failed"))
    }

    fn finish(&mut self) -> Result<String, EngineError> {
        if self.finished {
            return Ok(String::new());
        }
        self.finished = true;
        if !self.pending.is_empty() {
            return Err(execution_error(
                "detokenize",
                "generation ended with an incomplete UTF-8 sequence",
            ));
        }
        Ok(String::new())
    }

    fn push_blocks(&mut self, token: u32) -> Result<Vec<GenerationOutputBlock>, EngineError> {
        let text = self.push(token)?;
        if let Some(parser) = self.slab.as_mut() {
            let mut emitted = String::new();
            parser.push(&text, |part| {
                emitted.push_str(part);
                Ok(())
            })?;
            return Ok(text_blocks(emitted));
        }
        let Some(parser) = self.parser.as_mut() else {
            return Ok(text_blocks(text));
        };
        if text.is_empty() {
            return Ok(Vec::new());
        }
        output_blocks(parser.push_stream(text.as_bytes()).map_err(parse_error)?)
    }

    fn finish_blocks(&mut self) -> Result<Vec<GenerationOutputBlock>, EngineError> {
        let tail = self.finish()?;
        if let Some(mut parser) = self.slab.take() {
            let mut emitted = String::new();
            parser.push(&tail, |part| {
                emitted.push_str(part);
                Ok(())
            })?;
            self.completed_slab = Some(parser.finish()?);
            return Ok(text_blocks(emitted));
        }
        let Some(mut parser) = self.parser.take() else {
            return Ok(text_blocks(tail));
        };
        let mut parsed = Vec::new();
        if !tail.is_empty() {
            parsed.extend(parser.push_stream(tail.as_bytes()).map_err(parse_error)?);
        }
        parsed.extend(parser.finish().map_err(parse_error)?);
        output_blocks(parsed)
    }
    fn take_parser_proposals(&mut self) -> Option<FrozenParserAllocation> {
        self.completed_slab.take()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/backends_qwen_output_protocol_token.rs"]
mod protocol_token_tests;

#[cfg(test)]
#[path = "../../../tests/unit/backends_qwen_output_structured_output.rs"]
mod structured_output_tests;

/// Allocate the decoder owner with exactly the Layout included in output planning.
pub(super) fn box_slab_decoder(
    value: Qwen3TextDecoder,
) -> Result<Box<dyn TextDecoder>, EngineError> {
    let layout = std::alloc::Layout::new::<Qwen3TextDecoder>();
    // SAFETY: The type is non-zero-sized and Layout supplies its alignment.
    // Runtime has admitted this owner layout before invoking the constructor.
    let raw = unsafe { std::alloc::alloc(layout) };
    let Some(pointer) = std::ptr::NonNull::new(raw.cast::<Qwen3TextDecoder>()) else {
        // value (including its slab reservation) drops on this error path.
        return Err(allocation(
            "allocate slab decoder owner",
            Some(layout.size()),
        ));
    };
    // SAFETY: pointer is uniquely owned, properly aligned, and allocated by
    // the global allocator for this exact type. Initialize once, then let Box
    // run the destructor and deallocate with the matching Layout.
    unsafe {
        pointer.as_ptr().write(value);
        Ok(Box::from_raw(pointer.as_ptr()))
    }
}
