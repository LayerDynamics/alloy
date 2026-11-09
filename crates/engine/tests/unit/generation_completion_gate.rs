//! Completion-gate integration through the production run_worker/driver,
//! scheduler arbitration, incremental Qwen parser, and EventStream.
//! Inference is scripted to inject failures; this is not native/model evidence.
use super::*;
use crate::{
    api::tool_output_test_execution_context,
    rendering::{self, PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions},
    scheduler::Scheduler,
    types::{ArtifactId, ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId},
};
use qwen3::tool_calls::{ParsedBlock, ParserLimits, ToolCallParser};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    sync::{
        Condvar, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

const TIMEOUT: Duration = Duration::from_secs(10);

fn failure(message: impl Into<String>) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "W2 scripted inference",
        message: message.into(),
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn rendered() -> RenderedContext {
    let model = WordLevel::builder()
        .vocab([("input".to_owned(), 0)].into_iter().collect())
        .unk_token("input".to_owned())
        .build()
        .expect("vocabulary");
    let source = Tokenizer::new(model)
        .to_string(false)
        .expect("tokenizer JSON");
    let tokenizer =
        PinnedTokenizer::from_pinned_json(source.as_bytes(), digest(source.as_bytes()), 64 * 1024)
            .expect("pinned tokenizer");
    let profile = ModelProfileId::new("w2-gate-profile").expect("profile");
    let artifact = ArtifactId::new("w2-gate-artifact").expect("artifact");
    let snapshot = ContextSnapshot::new(
        ContextSnapshotId::new("w2-gate-snapshot").expect("snapshot"),
        profile.clone(),
        vec![Message {
            role: MessageRole::User,
            content: "input".to_owned(),
        }],
    )
    .expect("context");
    let template = "{{ messages[0].content }}";
    let binding = RenderingBinding {
        profile_id: &profile,
        artifact_id: &artifact,
        template,
        template_sha256: digest(template.as_bytes()),
        artifact_manifest_sha256: [7; 32],
    };
    let options = RenderingOptions {
        add_generation_prompt: false,
        enable_thinking: None,
        add_special_tokens: false,
        keep_trailing_newline: true,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: &[],
    };
    rendering::render_and_count(
        &snapshot,
        &binding,
        &tokenizer,
        &options,
        RenderingLimits::new(4, 1024, 1024, 64 * 1024, 128).expect("render limits"),
    )
    .expect("production rendering/tokenization")
}

#[derive(Default)]
struct Gate {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}
impl Gate {
    fn enter(&self) {
        let mut state = self.state.lock().expect("gate");
        state.0 = true;
        self.changed.notify_all();
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, TIMEOUT, |s| !s.1)
            .expect("wait for release");
        assert!(!timeout.timed_out() || state.1, "finish gate timed out");
    }
    fn wait_entered(&self) {
        let state = self.state.lock().expect("gate");
        let (state, timeout) = self
            .changed
            .wait_timeout_while(state, TIMEOUT, |s| !s.0)
            .expect("wait for finish");
        assert!(
            !timeout.timed_out() || state.0,
            "worker did not reach finish"
        );
    }
    fn release(&self) {
        self.state.lock().expect("gate").1 = true;
        self.changed.notify_all();
    }
}

#[derive(Clone, Copy, Default)]
struct Faults {
    final_usage: bool,
    cleanup: bool,
}

struct ScriptedSession {
    fragments: Vec<String>,
    index: usize,
    parser: Option<ToolCallParser>,
    usage: Usage,
    finished: Cell<bool>,
    finish_usage_reads: Cell<usize>,
    gate: Arc<Gate>,
    faults: Faults,
    dropped: Arc<AtomicUsize>,
    // Probe witnesses request-local ownership, not allocator byte accounting.
    _owned: Arc<()>,
}
impl ScriptedSession {
    fn blocks(parsed: Vec<ParsedBlock>) -> Result<Vec<GenerationOutputBlock>, EngineError> {
        parsed
            .into_iter()
            .map(|block| match block {
                ParsedBlock::Text(text) => Ok(GenerationOutputBlock::Text(text)),
                ParsedBlock::ToolCall(call) => Ok(GenerationOutputBlock::ToolCall(
                    ToolCallProposal::new(call.name, call.arguments)?,
                )),
            })
            .collect()
    }
}
impl GenerationSession for ScriptedSession {
    fn prefill(
        &mut self,
        context: &RenderedContext,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<(), EngineError> {
        self.usage.input_tokens = context.token_count();
        Ok(())
    }
    fn decode_next(
        &mut self,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<TokenStep, EngineError> {
        Err(failure("driver used text-only decode"))
    }
    fn decode_next_blocks(
        &mut self,
        _: &GenerationOptions,
        cancellation: &CancellationToken,
    ) -> Result<BlockStep, EngineError> {
        if cancellation.is_cancelled() {
            return Err(failure("cancelled scripted decode"));
        }
        let fragment = self
            .fragments
            .get(self.index)
            .ok_or_else(|| failure("script exhausted"))?;
        self.index += 1;
        self.usage.output_tokens += 1;
        let parsed = self
            .parser
            .as_mut()
            .expect("live parser")
            .push_stream(fragment.as_bytes())
            .map_err(|error| failure(error.to_string()))?;
        Ok(BlockStep {
            token_id: 0,
            blocks: Self::blocks(parsed)?,
            end_of_sequence: self.index == self.fragments.len(),
        })
    }
    fn finish_text(&mut self) -> Result<String, EngineError> {
        Err(failure("driver used text-only finish"))
    }
    fn finish_blocks(&mut self) -> Result<Vec<GenerationOutputBlock>, EngineError> {
        self.gate.enter();
        self.finished.set(true);
        let parsed = self
            .parser
            .take()
            .expect("parser")
            .finish()
            .map_err(|error| failure(error.to_string()))?;
        Self::blocks(parsed)
    }
    fn usage(&self) -> Usage {
        let mut usage = self.usage;
        if self.finished.get() {
            let reads = self.finish_usage_reads.get();
            self.finish_usage_reads.set(reads + 1);
            // First read is driver finish validation; the next is final reconciliation.
            if self.faults.final_usage && reads >= 1 {
                usage.output_tokens += 1;
            }
        }
        usage
    }
}
impl Drop for ScriptedSession {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
        assert!(!self.faults.cleanup, "injected W2 cleanup failure");
    }
}

struct WakeThread(thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

struct Run {
    stream: Option<EventStream>,
    shared: Arc<Shared>,
    worker: Option<thread::JoinHandle<()>>,
    scheduler: Scheduler,
    request: RequestId,
    gate: Arc<Gate>,
    dropped: Arc<AtomicUsize>,
    owned: Weak<()>,
}
impl Run {
    fn start(fragments: Vec<String>, faults: Faults, paused: bool, capacity: usize) -> Self {
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = RequestId::new("w2-gate-request").expect("request");
        let execution = tool_output_test_execution_context(
            scheduler.try_acquire(request.clone()).expect("permit"),
        );
        let gate = Arc::new(Gate::default());
        if !paused {
            gate.release();
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let resource = Arc::new(());
        let owned = Arc::downgrade(&resource);
        let session = ScriptedSession {
            fragments,
            index: 0,
            parser: Some(ToolCallParser::new(ParserLimits::default()).expect("parser")),
            usage: Usage::default(),
            finished: Cell::new(false),
            finish_usage_reads: Cell::new(0),
            gate: Arc::clone(&gate),
            faults,
            dropped: Arc::clone(&dropped),
            _owned: resource,
        };
        let limits = GenerationLimits::new(capacity, 64 * 1024, 1024).expect("limits");
        let shared = Arc::new(Shared {
            state: Mutex::new(ChannelState {
                text: VecDeque::with_capacity(capacity),
                proposals: VecDeque::new(),
                buffered_text_bytes: 0,
                proposal_reserved: 0,
                slab_proposals: None,
                terminal: None,
                worker_finished: false,
                receiver_dropped: false,
                consumer_waker: None,
                worker_thread: None,
            }),
            limits,
        });
        let worker_shared = Arc::clone(&shared);
        let input = rendered();
        let worker = thread::spawn(move || {
            run_worker(
                worker_shared,
                execution,
                input,
                GenerationOptions::greedy(32).expect("options"),
                Box::new(session),
            )
        });
        Self {
            stream: Some(EventStream {
                shared: Arc::clone(&shared),
                exhausted: false,
            }),
            shared,
            worker: Some(worker),
            scheduler,
            request,
            gate,
            dropped,
            owned,
        }
    }
    fn poll(&mut self, context: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        Pin::new(self.stream.as_mut().expect("stream")).poll_next(context)
    }
    fn next(&mut self) -> Option<GenerationEvent> {
        let deadline = Instant::now() + TIMEOUT;
        let waker = Waker::from(Arc::new(WakeThread(thread::current())));
        let mut context = Context::from_waker(&waker);
        loop {
            match self.poll(&mut context) {
                Poll::Ready(event) => return event,
                Poll::Pending => {
                    let now = Instant::now();
                    assert!(now < deadline, "event delivery timed out");
                    thread::park_timeout(deadline.saturating_duration_since(now));
                }
            }
        }
    }
    fn join(&mut self) {
        let deadline = Instant::now() + TIMEOUT;
        while !self.worker.as_ref().expect("worker").is_finished() {
            assert!(Instant::now() < deadline, "worker failed to stop");
            thread::yield_now();
        }
        self.worker
            .take()
            .expect("worker")
            .join()
            .expect("worker panic");
        assert_eq!(self.dropped.load(Ordering::SeqCst), 1);
        assert!(
            self.owned.upgrade().is_none(),
            "request-local resource leaked"
        );
        assert!(!self.scheduler.is_busy().expect("scheduler state"));
    }
    fn wait_publication_blocked(&self) {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            {
                let state = lock(&self.shared.state);
                if state.proposals.len() == 1 && !state.worker_finished {
                    assert_eq!(self.shared.limits.event_capacity, 1);
                    assert!(state.proposal_reserved > 0);
                    return;
                }
            }
            assert!(Instant::now() < deadline, "publication never filled queue");
            thread::yield_now();
        }
    }
}
impl Drop for Run {
    fn drop(&mut self) {
        self.gate.release();
        drop(self.stream.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn call(name: &str) -> String {
    format!("<tool_call>{{\"name\":\"{name}\",\"arguments\":{{}}}}</tool_call>")
}

/// This counter is incremented on EVERY observed proposal, not only after
/// terminal success. A leaked event therefore fails the negative tests.
#[derive(Default)]
struct EffectSink {
    executions: AtomicUsize,
}
impl EffectSink {
    fn observe(&self, event: &GenerationEvent) {
        if matches!(event, GenerationEvent::ToolCallProposed { .. }) {
            self.executions.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn count(&self) -> usize {
        self.executions.load(Ordering::SeqCst)
    }
}

fn expect_rejection(run: &mut Run, expected: &str) {
    let sink = EffectSink::default();
    let mut proposals = 0;
    let mut terminals = 0;
    let mut diagnostic = String::new();
    while let Some(event) = run.next() {
        sink.observe(&event);
        if matches!(&event, GenerationEvent::ToolCallProposed { .. }) {
            proposals += 1;
        }
        if event.is_terminal() {
            terminals += 1;
            match event {
                GenerationEvent::RequestFailed { error, .. } => diagnostic = format!("{error:?}"),
                other => panic!("rejected generation succeeded: {other:?}"),
            }
            assert!(run.next().is_none(), "event after terminal");
            break;
        }
    }
    assert_eq!(terminals, 1);
    assert_eq!(proposals, 0);
    assert_eq!(sink.count(), 0, "rejected generation reached effect sink");
    assert!(
        diagnostic.contains(expected),
        "wrong rejection: {diagnostic}"
    );
    run.join();
}

#[test]
fn qwen3_malformed_tool_calls_never_execute() {
    let mut run = Run::start(
        vec![
            "lead".into(),
            call("first"),
            "<tool_call>{broken}</tool_call>".into(),
        ],
        Faults::default(),
        false,
        8,
    );
    expect_rejection(&mut run, "W2 scripted inference");
}

#[test]
fn truncated_call_after_valid_proposal_never_executes() {
    let mut run = Run::start(
        vec![call("first"), "<tool_call>{".into()],
        Faults::default(),
        false,
        8,
    );
    expect_rejection(&mut run, "W2 scripted inference");
}

#[test]
fn cancellation_after_a_proposal_publishes_zero_proposals() {
    let mut run = Run::start(vec![call("first")], Faults::default(), true, 8);
    run.gate.wait_entered();
    run.scheduler.cancel(&run.request).expect("cancel");
    assert!(run.scheduler.is_busy().expect("cleanup still owns slot"));
    run.gate.release();
    expect_rejection(&mut run, "Cancelled");
}

#[test]
fn invalid_final_usage_publishes_zero_proposals() {
    let mut run = Run::start(
        vec![call("first")],
        Faults {
            final_usage: true,
            ..Faults::default()
        },
        false,
        8,
    );
    expect_rejection(&mut run, "successful final usage");
}

#[test]
fn cleanup_failure_publishes_zero_proposals() {
    let mut run = Run::start(
        vec![call("first")],
        Faults {
            cleanup: true,
            ..Faults::default()
        },
        false,
        8,
    );
    expect_rejection(&mut run, "release generation session");
}

#[test]
fn text_streams_before_completion_but_proposals_do_not() {
    let mut run = Run::start(
        vec!["lead".into(), call("first"), "tail".into()],
        Faults::default(),
        true,
        8,
    );
    run.gate.wait_entered();
    assert!(run.scheduler.is_busy().expect("active"));
    assert!(matches!(run.next(), Some(GenerationEvent::TextDelta { text }) if text == "lead"));
    assert!(matches!(run.next(), Some(GenerationEvent::TextDelta { text }) if text == "tail"));
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    assert!(
        run.poll(&mut context).is_pending(),
        "proposal escaped before finish"
    );
    run.gate.release();
    let sink = EffectSink::default();
    let event = run.next().expect("proposal");
    sink.observe(&event);
    assert!(
        matches!(event, GenerationEvent::ToolCallProposed { proposal } if proposal.name() == "first")
    );
    assert!(matches!(
        run.next(),
        Some(GenerationEvent::GenerationCompleted { .. })
    ));
    assert!(run.next().is_none());
    assert_eq!(sink.count(), 1, "positive control must reach sink");
    run.join();
}

#[test]
fn multiple_proposals_preserve_order_under_backpressure() {
    let mut run = Run::start(
        vec![call("a"), call("b"), call("c")],
        Faults::default(),
        false,
        1,
    );
    run.wait_publication_blocked();
    let sink = EffectSink::default();
    for expected in ["a", "b", "c"] {
        let event = run.next().expect("proposal");
        sink.observe(&event);
        assert!(
            matches!(event, GenerationEvent::ToolCallProposed { proposal } if proposal.name() == expected)
        );
    }
    assert!(matches!(
        run.next(),
        Some(GenerationEvent::GenerationCompleted { .. })
    ));
    assert!(run.next().is_none());
    assert_eq!(sink.count(), 3);
    run.join();
    let state = lock(&run.shared.state);
    assert_eq!(state.buffered_text_bytes, 0);
    assert_eq!(state.proposal_reserved, 0);
}

#[test]
fn dropping_stream_unblocks_proposal_publication_and_releases_resources() {
    let mut run = Run::start(
        vec![call("a"), call("b"), call("c")],
        Faults::default(),
        false,
        1,
    );
    run.wait_publication_blocked();
    drop(run.stream.take());
    run.join();
    let state = lock(&run.shared.state);
    assert!(state.receiver_dropped);
    assert!(state.worker_finished);
    assert!(state.proposals.is_empty());
    assert_eq!(state.buffered_text_bytes, 0);
    assert_eq!(state.proposal_reserved, 0);
}
