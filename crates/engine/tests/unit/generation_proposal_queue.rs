use super::*;
use serde_json::{Map, json};
use std::task::Waker;

fn shared(event_capacity: usize, budget: usize, delta: usize) -> Arc<Shared> {
    Arc::new(Shared {
        state: Mutex::new(ChannelState {
            text: VecDeque::new(),
            proposals: VecDeque::new(),
            slab_proposals: None,
            buffered_text_bytes: 0,
            proposal_reserved: 0,
            terminal: None,
            worker_finished: false,
            receiver_dropped: false,
            consumer_waker: None,
            worker_thread: None,
        }),
        limits: GenerationLimits::new(event_capacity, budget, delta).expect("limits"),
    })
}

fn held(name: &str) -> HeldProposal {
    let proposal = ToolCallProposal::new(name, Map::new()).expect("proposal");
    let charge = proposal_charge(&proposal);
    HeldProposal { proposal, charge }
}

fn reserve(shared: &Shared, held: &HeldProposal) {
    assert!(matches!(
        try_reserve_proposal(shared, held.charge).expect("reservation"),
        ProposalReservation::Reserved
    ));
}

fn poll(stream: &mut EventStream) -> Option<GenerationEvent> {
    let mut context = Context::from_waker(Waker::noop());
    match Pin::new(stream).poll_next(&mut context) {
        Poll::Ready(event) => event,
        Poll::Pending => panic!("stream unexpectedly pending"),
    }
}

struct TextOnlySession;

impl GenerationSession for TextOnlySession {
    fn prefill(
        &mut self,
        _: &RenderedContext,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn decode_next(
        &mut self,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<TokenStep, EngineError> {
        Err(EngineError::Internal {
            message: "unused".to_owned(),
        })
    }

    fn finish_text(&mut self) -> Result<String, EngineError> {
        Ok(" tail".to_owned())
    }

    fn usage(&self) -> Usage {
        Usage::default()
    }
}

#[test]
fn default_finish_blocks_preserves_text_only_sessions() {
    let mut session = TextOnlySession;
    assert_eq!(
        session.finish_blocks().expect("finish"),
        vec![GenerationOutputBlock::Text(" tail".to_owned())],
    );
}

#[test]
fn charge_grows_with_nested_argument_data() {
    let small = ToolCallProposal::new("fs.read", Map::new()).expect("small");
    let payload = "x".repeat(4096);
    let value = json!({"items": [payload, {"nested": [1, 2, 3]}]});
    let large = ToolCallProposal::new("fs.read", value.as_object().expect("object").clone())
        .expect("large");
    assert!(proposal_charge(&large) >= proposal_charge(&small) + 4096);
}

#[test]
fn reservation_stays_below_the_text_budget_headroom() {
    let channel = shared(4, 1000, 400);
    assert!(matches!(
        try_reserve_proposal(&channel, 500).expect("first"),
        ProposalReservation::Reserved
    ));
    assert!(matches!(
        try_reserve_proposal(&channel, 500).expect("second"),
        ProposalReservation::TooLarge {
            required_bytes: 1000,
            available_bytes: 600
        }
    ));
    let state = lock(&channel.state);
    assert_eq!(state.proposal_reserved, 500);
    assert_eq!(state.buffered_text_bytes, 500);
}

#[test]
fn reservation_waits_while_queued_text_holds_the_budget() {
    let channel = shared(4, 1000, 400);
    lock(&channel.state).buffered_text_bytes = 800;
    assert!(matches!(
        try_reserve_proposal(&channel, 250).expect("reservation"),
        ProposalReservation::Wait
    ));
    assert_eq!(lock(&channel.state).proposal_reserved, 0);
}

#[test]
fn proposals_follow_text_and_precede_the_terminal_event() {
    let channel = shared(8, 4096, 1024);
    {
        let mut state = lock(&channel.state);
        state.text.push_back("hi".to_owned());
        state.buffered_text_bytes = state.text[0].capacity();
    }
    for name in ["a", "b"] {
        let item = held(name);
        reserve(&channel, &item);
        assert!(matches!(
            try_enqueue_proposal(&channel, item),
            ProposalEnqueue::Queued(_)
        ));
    }
    {
        let mut state = lock(&channel.state);
        state.terminal = Some(GenerationEvent::GenerationCompleted {
            finish_reason: FinishReason::EndOfSequence,
            usage: Usage::default(),
        });
        state.worker_finished = true;
    }

    let mut stream = EventStream {
        shared: Arc::clone(&channel),
        exhausted: false,
    };
    assert!(matches!(
        poll(&mut stream),
        Some(GenerationEvent::TextDelta { text }) if text == "hi"
    ));
    for expected in ["a", "b"] {
        match poll(&mut stream) {
            Some(GenerationEvent::ToolCallProposed { proposal }) => {
                assert_eq!(proposal.name(), expected);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(matches!(
        poll(&mut stream),
        Some(GenerationEvent::GenerationCompleted { .. })
    ));
    assert!(poll(&mut stream).is_none());

    let state = lock(&channel.state);
    assert_eq!(state.buffered_text_bytes, 0);
    assert_eq!(state.proposal_reserved, 0);
}

#[test]
fn full_queue_returns_the_unpublished_proposal() {
    let channel = shared(2, 4096, 1024);
    lock(&channel.state).text.push_back("hi".to_owned());
    assert!(matches!(
        try_enqueue_proposal(&channel, held("a")),
        ProposalEnqueue::Queued(_)
    ));
    match try_enqueue_proposal(&channel, held("b")) {
        ProposalEnqueue::Full(returned) => assert_eq!(returned.proposal.name(), "b"),
        _ => panic!("queue should be full"),
    }
}

#[test]
fn dropping_the_stream_discards_queued_proposals_and_reservations() {
    let channel = shared(4, 4096, 1024);
    let item = held("a");
    reserve(&channel, &item);
    assert!(matches!(
        try_enqueue_proposal(&channel, item),
        ProposalEnqueue::Queued(_)
    ));

    drop(EventStream {
        shared: Arc::clone(&channel),
        exhausted: false,
    });

    assert!(matches!(
        try_enqueue_proposal(&channel, held("b")),
        ProposalEnqueue::ReceiverDropped
    ));
    let state = lock(&channel.state);
    assert!(state.proposals.is_empty());
    assert_eq!(state.proposal_reserved, 0);
    assert_eq!(state.buffered_text_bytes, 0);
}
