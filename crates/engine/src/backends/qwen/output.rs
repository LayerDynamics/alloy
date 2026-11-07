//! Qwen incremental output decoding.
//! Behavior-preserving extraction; structured parsing is not wired yet.

use super::{allocation, execution_error, internal};
use crate::{error::EngineError, rendering::PinnedTokenizer, runtime::TextDecoder};
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
