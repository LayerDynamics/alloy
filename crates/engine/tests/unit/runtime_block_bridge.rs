use super::blocks_to_text;
use crate::types::{GenerationOutputBlock, ToolCallProposal};
use serde_json::Map;

#[test]
fn text_blocks_concatenate_for_legacy_sessions() {
    let blocks = vec![
        GenerationOutputBlock::Text("a".to_owned()),
        GenerationOutputBlock::Text("雪".to_owned()),
    ];
    assert_eq!(blocks_to_text(blocks).expect("text"), "a雪");
}

#[test]
fn legacy_text_view_rejects_instead_of_dropping_a_call() {
    let proposal = ToolCallProposal::new("fs.read", Map::new()).expect("proposal");
    let blocks = vec![GenerationOutputBlock::ToolCall(proposal)];
    assert!(blocks_to_text(blocks).is_err());
}
