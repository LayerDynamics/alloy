use super::{GenerationOutputBlock, ToolCallProposal};
use serde_json::{Map, Value, json};

#[test]
fn proposal_preserves_nested_arguments_and_authority_shaped_data() {
    let value = json!({
        "path": "résumé/雪.txt",
        "nested": [null, true, 7, {"x": "retained"}],
        "approved": true,
        "call_id": "argument-data-not-an-invocation-id"
    });
    let arguments = value.as_object().expect("object").clone();
    let proposal = ToolCallProposal::new("fs.read", arguments.clone()).expect("proposal");
    assert_eq!(proposal.name(), "fs.read");
    assert_eq!(proposal.arguments(), &arguments);
    let (name, restored) = proposal.into_parts();
    assert_eq!(name, "fs.read");
    assert_eq!(Value::Object(restored), value);
}

#[test]
fn proposal_rejects_empty_padded_and_control_names() {
    for name in ["", " fs.read", "fs.read ", "fs\nread", "fs\0read"] {
        assert!(ToolCallProposal::new(name, Map::new()).is_err());
    }
}

#[test]
fn unknown_name_is_only_a_proposal_not_registry_resolution() {
    let proposal = ToolCallProposal::new("not.registered", Map::new()).expect("proposal");
    assert_eq!(proposal.name(), "not.registered");
    assert!(proposal.arguments().is_empty());
}

#[test]
fn output_blocks_keep_text_and_complete_proposals_distinct() {
    let proposal = ToolCallProposal::new("fs.read", Map::new()).expect("proposal");
    let blocks = [
        GenerationOutputBlock::Text("Reading.".to_owned()),
        GenerationOutputBlock::ToolCall(proposal.clone()),
    ];
    assert!(matches!(&blocks[0], GenerationOutputBlock::Text(text) if text == "Reading."));
    assert!(matches!(&blocks[1], GenerationOutputBlock::ToolCall(call) if call == &proposal));
}
