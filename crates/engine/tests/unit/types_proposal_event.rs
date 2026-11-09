use super::{FinishReason, GenerationEvent, ToolCallProposal, Usage};
use serde_json::Map;

#[test]
fn proposal_event_is_nonterminal_and_carries_no_identity() {
    let proposal = ToolCallProposal::new("fs.read", Map::new()).expect("proposal");
    let event = GenerationEvent::ToolCallProposed { proposal };
    assert!(!event.is_terminal());
    assert!(matches!(
        event,
        GenerationEvent::ToolCallProposed { proposal } if proposal.name() == "fs.read"
    ));
    assert!(
        GenerationEvent::GenerationCompleted {
            finish_reason: FinishReason::EndOfSequence,
            usage: Usage::default(),
        }
        .is_terminal()
    );
}
