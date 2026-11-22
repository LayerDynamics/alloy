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
        gemma_decoder: false,
        tokenizer: Arc::new(tokenizer),
        special_tokens: Arc::new([0, 1, 2].into_iter().collect()),
        byte_decoder: Arc::new(bytes),
        pending: Vec::new(),
        max_token_bytes: 128,
        finished: false,
        parser: parsing.then(|| ToolCallParser::new(ParserLimits::default()).expect("parser")),
        slab: None,
        xml: None,
        ministral: None,
        phi: None,
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
            (Some(GenerationOutputBlock::Text(previous)), GenerationOutputBlock::Text(text)) => {
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

#[test]
fn ministral_protocol_decoder_distinguishes_sampled_ids_and_ordinary_marker_text() {
    use crate::{
        admission::AdmissionController,
        parser_allocation::{ParserAllocation, ParserAllocationLimits},
        slab_parser::ministral3::{self, MinistralToolParser},
    };

    let mut decoder = decoder(false);
    decoder.special_tokens = Arc::new([ministral3::EOS, ministral3::CALL, ministral3::ARGS].into());
    let limits = ParserAllocationLimits {
        carry_bytes: 1,
        call_bytes: 128,
        decoded_bytes: 128,
        depth: 4,
        keys: 8,
        nodes: 16,
        proposal_bytes: 128,
        calls: 2,
    };
    let bytes = ParserAllocation::required_bytes(limits).unwrap();
    let controller = AdmissionController::new(bytes).unwrap();
    let mut grant = controller.reserve_grant("decoder", &[bytes]).unwrap();
    decoder.ministral = Some(MinistralToolParser::new(
        ParserAllocation::from_grant(&mut grant, limits).unwrap(),
        512,
    ));
    let literal = "[TOOL_CALLS]fs.read[ARGS]{}";
    assert_eq!(
        merged(push_all(&mut decoder, &ids(literal))),
        vec![GenerationOutputBlock::Text(literal.into())]
    );
    let mut tokens = vec![ministral3::CALL];
    tokens.extend(ids("fs.read"));
    tokens.push(ministral3::ARGS);
    tokens.extend(ids("{}"));
    assert!(push_all(&mut decoder, &tokens).is_empty());
    assert!(decoder.take_parser_proposals().is_none());
    decoder.observe_eos(ministral3::EOS).unwrap();
    assert!(decoder.take_parser_proposals().is_none());
    assert!(decoder.finish_blocks().unwrap().is_empty());
    let calls = decoder.take_parser_proposals().unwrap();
    assert_eq!(calls.proposal_count(), 1);
    let proposal = calls.proposal(0).unwrap();
    assert_eq!((proposal.name, proposal.arguments_json), ("fs.read", "{}"));
}
