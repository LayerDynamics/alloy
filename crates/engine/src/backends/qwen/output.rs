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
mod w2_protocol_token_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    fn decoder() -> Qwen3TextDecoder {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("<tool_call>".to_owned(), 0),
                    ("</tool_call>".to_owned(), 1),
                    ("<|im_end|>".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<|im_end|>".to_owned())
            .build()
            .expect("fixture vocabulary");
        let tokenizer = Tokenizer::new(model);
        let source = tokenizer.to_string(false).expect("tokenizer JSON");
        let pin: [u8; 32] = Sha256::digest(source.as_bytes()).into();
        let tokenizer = PinnedTokenizer::from_pinned_json(source.as_bytes(), pin, 16 * 1024)
            .expect("pinned tokenizer");
        Qwen3TextDecoder {
            tokenizer: Arc::new(tokenizer),
            special_tokens: Arc::new([0, 1, 2].into_iter().collect()),
            byte_decoder: Arc::new(BTreeMap::new()),
            pending: Vec::new(),
            max_token_bytes: 128,
            finished: false,
            parser: None,
            slab: None,
            completed_slab: None,
        }
    }

    #[test]
    fn declared_tool_delimiters_survive_detokenization() {
        let mut decoder = decoder();
        assert_eq!(decoder.push(0).expect("opening"), "<tool_call>");
        assert_eq!(decoder.push(1).expect("closing"), "</tool_call>");
        assert!(decoder.finish().expect("finish").is_empty());
    }

    #[test]
    fn ordinary_special_tokens_remain_hidden() {
        let mut decoder = decoder();
        assert!(decoder.push(2).expect("EOS spelling").is_empty());
    }

    #[test]
    fn delimiter_cannot_cross_an_incomplete_utf8_character() {
        for token in [0, 1] {
            let mut decoder = decoder();
            decoder.pending.push(0xe9);
            assert!(decoder.push(token).is_err());
            assert_eq!(decoder.pending, vec![0xe9]);
        }
    }

    #[test]
    fn protocol_spelling_must_fit_the_token_bound() {
        let mut decoder = decoder();
        decoder.max_token_bytes = 1;
        assert!(decoder.push(0).is_err());
    }

    #[test]
    fn protocol_tokens_are_rejected_after_completion() {
        let mut decoder = decoder();
        decoder.finish().expect("finish");
        assert!(decoder.push(0).is_err());
    }
}

#[cfg(test)]
mod w2_structured_output_tests {
    use super::*;
    use qwen3::tool_calls::ParserLimits;
    use sha2::{Digest, Sha256};
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    const OPEN: u32 = 0;
    const CLOSE: u32 = 1;
    const FIRST_CHARACTER: u32 = 3;
    const CALL: &str = r#"{"name":"fs.read","arguments":{}}"#;

    fn character_id(character: char) -> u32 {
        FIRST_CHARACTER + (u32::from(character) - 0x20)
    }

    fn ids(text: &str) -> Vec<u32> {
        text.chars().map(character_id).collect()
    }

    fn decoder(parsing: bool) -> Qwen3TextDecoder {
        let mut entries: Vec<(String, u32)> = vec![
            ("<tool_call>".to_owned(), OPEN),
            ("</tool_call>".to_owned(), CLOSE),
            ("<|im_end|>".to_owned(), 2),
        ];
        let mut bytes = BTreeMap::new();
        for byte in 0x20_u8..=0x7e {
            let character = char::from(byte);
            entries.push((character.to_string(), character_id(character)));
            bytes.insert(character, byte);
        }
        let model = WordLevel::builder()
            .vocab(entries.into_iter().collect())
            .unk_token("<|im_end|>".to_owned())
            .build()
            .expect("fixture vocabulary");
        let tokenizer = Tokenizer::new(model);
        let source = tokenizer.to_string(false).expect("tokenizer JSON");
        let pin: [u8; 32] = Sha256::digest(source.as_bytes()).into();
        let tokenizer = PinnedTokenizer::from_pinned_json(source.as_bytes(), pin, 64 * 1024)
            .expect("pinned tokenizer");
        Qwen3TextDecoder {
            tokenizer: Arc::new(tokenizer),
            special_tokens: Arc::new([0, 1, 2].into_iter().collect()),
            byte_decoder: Arc::new(bytes),
            pending: Vec::new(),
            max_token_bytes: 128,
            finished: false,
            parser: parsing.then(|| ToolCallParser::new(ParserLimits::default()).expect("parser")),
            slab: None,
            completed_slab: None,
        }
    }

    fn push_all(decoder: &mut Qwen3TextDecoder, tokens: &[u32]) -> Vec<GenerationOutputBlock> {
        let mut blocks = Vec::new();
        for token in tokens {
            blocks.extend(decoder.push_blocks(*token).expect("token"));
        }
        blocks
    }

    fn merged(blocks: Vec<GenerationOutputBlock>) -> Vec<GenerationOutputBlock> {
        let mut result: Vec<GenerationOutputBlock> = Vec::new();
        for block in blocks {
            match (result.last_mut(), block) {
                (
                    Some(GenerationOutputBlock::Text(previous)),
                    GenerationOutputBlock::Text(text),
                ) => {
                    previous.push_str(&text);
                }
                (_, block) => result.push(block),
            }
        }
        result
    }

    fn call_tokens() -> Vec<u32> {
        let mut tokens = vec![OPEN];
        tokens.extend(ids(CALL));
        tokens.push(CLOSE);
        tokens
    }

    #[test]
    fn text_streams_before_generation_completes() {
        let mut decoder = decoder(true);
        let blocks = merged(push_all(&mut decoder, &ids("hello")));
        assert_eq!(
            blocks,
            vec![GenerationOutputBlock::Text("hello".to_owned())]
        );
        assert!(decoder.finish_blocks().expect("finish").is_empty());
    }

    #[test]
    fn proposal_appears_only_when_its_closing_delimiter_arrives() {
        let mut decoder = decoder(true);
        let tokens = call_tokens();
        let (close, body) = tokens.split_last().expect("tokens");
        assert!(push_all(&mut decoder, body).is_empty());
        let blocks = decoder.push_blocks(*close).expect("close");
        assert!(matches!(
            blocks.as_slice(),
            [GenerationOutputBlock::ToolCall(proposal)] if proposal.name() == "fs.read"
        ));
        assert!(decoder.finish_blocks().expect("finish").is_empty());
    }

    #[test]
    fn text_and_call_order_is_preserved() {
        let mut decoder = decoder(true);
        let mut tokens = ids("a");
        tokens.extend(call_tokens());
        tokens.extend(ids("b"));
        let mut blocks = push_all(&mut decoder, &tokens);
        blocks.extend(decoder.finish_blocks().expect("finish"));
        let blocks = merged(blocks);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], GenerationOutputBlock::Text("a".to_owned()));
        assert!(matches!(&blocks[1], GenerationOutputBlock::ToolCall(_)));
        assert_eq!(blocks[2], GenerationOutputBlock::Text("b".to_owned()));
    }

    #[test]
    fn malformed_call_fails_at_its_closing_delimiter() {
        let mut decoder = decoder(true);
        let mut body = vec![OPEN];
        body.extend(ids("{broken}"));
        assert!(push_all(&mut decoder, &body).is_empty());
        assert!(decoder.push_blocks(CLOSE).is_err());
    }

    #[test]
    fn truncated_call_is_rejected_at_completion() {
        let mut decoder = decoder(true);
        let mut tokens = vec![OPEN];
        tokens.extend(ids(r#"{"name":"#));
        assert!(push_all(&mut decoder, &tokens).is_empty());
        assert!(decoder.finish_blocks().is_err());
    }

    #[test]
    fn disabled_parser_keeps_text_only_behavior() {
        let mut decoder = decoder(false);
        assert_eq!(
            decoder.push_blocks(OPEN).expect("delimiter"),
            vec![GenerationOutputBlock::Text("<tool_call>".to_owned())],
        );
        assert!(decoder.finish_blocks().expect("finish").is_empty());
    }
}

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
