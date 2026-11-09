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
