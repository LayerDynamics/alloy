use super::*;
use crate::{
    admission::AdmissionController,
    api::tool_output_test_execution_context,
    parser_allocation::{ParserAllocation, ParserAllocationLimits},
    scheduler::Scheduler,
    slab_parser::SlabParser,
    types::{ArtifactId, ContextSnapshotId, ModelProfileId, RendererIdentity, TokenizerId},
};
use std::{
    cell::Cell,
    sync::Condvar,
    time::{Duration, Instant},
};
const WAIT: Duration = Duration::from_secs(10);
struct Gate {
    entered: std::sync::atomic::AtomicBool,
    open: Mutex<bool>,
    changed: Condvar,
}
impl Gate {
    fn release(&self) {
        *self.open.lock().expect("gate") = true;
        self.changed.notify_all();
    }
    fn wait(&self) {
        self.entered
            .store(true, std::sync::atomic::Ordering::Release);
        let guard = self.open.lock().expect("gate");
        let (guard, timeout) = self
            .changed
            .wait_timeout_while(guard, WAIT, |open| !*open)
            .expect("wait");
        assert!(*guard || !timeout.timed_out(), "finish gate timeout");
    }
}
struct Session {
    parser: Option<SlabParser>,
    ready: Option<FrozenParserAllocation>,
    parts: Vec<String>,
    index: usize,
    usage: Usage,
    finished: Cell<bool>,
    reads: Cell<usize>,
    bad_usage: bool,
    bad_cleanup: bool,
    gate: Arc<Gate>,
}
impl GenerationSession for Session {
    fn prefill(
        &mut self,
        r: &RenderedContext,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<(), EngineError> {
        self.usage.input_tokens = r.token_count();
        Ok(())
    }
    fn decode_next(
        &mut self,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<TokenStep, EngineError> {
        Err(EngineError::Internal {
            message: "wrong decoder path".into(),
        })
    }
    fn decode_next_blocks(
        &mut self,
        _: &GenerationOptions,
        _: &CancellationToken,
    ) -> Result<BlockStep, EngineError> {
        let mut text = String::new();
        self.usage.output_tokens += 1;
        self.parser
            .as_mut()
            .expect("parser")
            .push(&self.parts[self.index], |part| {
                text.push_str(part);
                Ok(())
            })?;
        self.index += 1;
        Ok(BlockStep {
            token_id: 0,
            blocks: text_blocks(text),
            end_of_sequence: self.index == self.parts.len(),
        })
    }
    fn finish_text(&mut self) -> Result<String, EngineError> {
        Err(EngineError::Internal {
            message: "wrong finish path".into(),
        })
    }
    fn finish_blocks(&mut self) -> Result<Vec<GenerationOutputBlock>, EngineError> {
        self.gate.wait();
        self.ready = Some(self.parser.take().expect("parser").finish()?);
        self.finished.set(true);
        Ok(Vec::new())
    }
    fn take_parser_proposals(&mut self) -> Option<FrozenParserAllocation> {
        self.ready.take()
    }
    fn usage(&self) -> Usage {
        let mut usage = self.usage;
        if self.finished.get() {
            let r = self.reads.get();
            self.reads.set(r + 1);
            if self.bad_usage && r > 0 {
                usage.output_tokens += 1;
            }
        }
        usage
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        assert!(!self.bad_cleanup, "injected cleanup failure");
    }
}
struct Run {
    scheduler: Scheduler,
    request: RequestId,
    stream: Option<EventStream>,
    shared: Arc<Shared>,
    worker: Option<thread::JoinHandle<()>>,
    admission: AdmissionController,
    gate: Arc<Gate>,
}
impl Run {
    fn new(parts: Vec<String>, paused: bool, bad_usage: bool, bad_cleanup: bool) -> Self {
        let capacities = ParserAllocationLimits {
            carry_bytes: 32,
            call_bytes: 2048,
            decoded_bytes: 2048,
            depth: 16,
            keys: 128,
            nodes: 256,
            proposal_bytes: 4096,
            calls: 8,
        };
        let size = ParserAllocation::required_bytes(capacities).expect("size");
        let admission = AdmissionController::new(size).expect("admission");
        let mut grant = admission.reserve_grant("slab", &[size]).expect("grant");
        let storage = ParserAllocation::from_grant(&mut grant, capacities).expect("storage");
        let gate = Arc::new(Gate {
            entered: std::sync::atomic::AtomicBool::new(false),
            open: Mutex::new(!paused),
            changed: Condvar::new(),
        });
        let session = Session {
            parser: Some(SlabParser::new(storage, 65536).expect("parser")),
            ready: None,
            parts,
            index: 0,
            usage: Usage::default(),
            finished: Cell::new(false),
            reads: Cell::new(0),
            bad_usage,
            bad_cleanup,
            gate: Arc::clone(&gate),
        };
        let scheduler = Scheduler::new(8).expect("scheduler");
        let request = RequestId::new("slab-test").expect("id");
        let execution = tool_output_test_execution_context(
            scheduler.try_acquire(request.clone()).expect("permit"),
        );
        // Logical input fixture only: not a tokenizer/rendering acceptance claim.
        let rendered = RenderedContext {
            snapshot_id: ContextSnapshotId::new("slab-snapshot").expect("id"),
            profile_id: ModelProfileId::new("slab-profile").expect("id"),
            artifact_id: ArtifactId::new("slab-artifact").expect("id"),
            text: "input".into(),
            token_ids: vec![1],
            render_hash: [0; 32],
            template_hash: [0; 32],
            tool_schema_hash: None,
            tokenizer_id: TokenizerId::new("fixture").expect("id"),
            renderer: RendererIdentity {
                implementation: "fixture".into(),
                version: "1".into(),
            },
        };
        let shared = Arc::new(Shared {
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
            limits: GenerationLimits::new(1, 8192, 1024).expect("limits"),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || {
            run_worker(
                worker_shared,
                execution,
                rendered,
                GenerationOptions::greedy(32).expect("options"),
                Box::new(session),
            )
        });
        Self {
            scheduler,
            request,
            stream: Some(EventStream {
                shared: Arc::clone(&shared),
                exhausted: false,
            }),
            shared,
            worker: Some(worker),
            admission,
            gate,
        }
    }
    fn next(&mut self) -> Option<GenerationEvent> {
        let deadline = Instant::now() + WAIT;
        let waker = Waker::from(Arc::new(WorkerWake {
            thread: thread::current(),
        }));
        let mut cx = Context::from_waker(&waker);
        loop {
            match Pin::new(self.stream.as_mut().expect("stream")).poll_next(&mut cx) {
                Poll::Ready(event) => return event,
                Poll::Pending => {
                    let now = Instant::now();
                    assert!(now < deadline, "event timeout");
                    thread::park_timeout(deadline.saturating_duration_since(now));
                }
            }
        }
    }
    fn join(&mut self) {
        let deadline = Instant::now() + WAIT;
        while !self.worker.as_ref().expect("worker").is_finished() {
            assert!(Instant::now() < deadline, "worker timeout");
            thread::yield_now();
        }
        self.worker
            .take()
            .expect("worker")
            .join()
            .expect("worker panic");
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
fn failure(run: &mut Run) {
    let mut effects = 0;
    let mut failed = false;
    while let Some(event) = run.next() {
        match event {
            GenerationEvent::ToolCallBatchProposed { proposals } => {
                effects += proposals.proposal_count()
            }
            GenerationEvent::ToolCallProposed { .. } => effects += 1,
            GenerationEvent::RequestFailed { .. } => failed = true,
            GenerationEvent::GenerationCompleted { .. } => panic!("failure succeeded"),
            _ => {}
        }
    }
    run.join();
    assert!(failed);
    assert_eq!(effects, 0);
    assert_eq!(
        run.admission.snapshot().expect("released").reserved_bytes(),
        0
    );
}
#[test]
fn successful_batch_outlives_decoder_worker_and_stream() {
    let mut run = Run::new(vec![call("a"), call("b"), call("c")], false, false, false);
    let owner = match run.next() {
        Some(GenerationEvent::ToolCallBatchProposed { proposals }) => proposals,
        other => panic!("wrong event: {other:?}"),
    };
    assert_eq!(owner.proposal_count(), 3);
    for (i, name) in ["a", "b", "c"].into_iter().enumerate() {
        assert_eq!(owner.proposal(i).expect("proposal").name, name);
    }
    assert!(matches!(
        run.next(),
        Some(GenerationEvent::GenerationCompleted { .. })
    ));
    assert!(run.next().is_none());
    run.join();
    drop(run.stream.take());
    assert_eq!(
        run.admission
            .snapshot()
            .expect("consumer retained")
            .reserved_bytes(),
        owner.reserved_bytes()
    );
    drop(owner);
    assert_eq!(
        run.admission.snapshot().expect("released").reserved_bytes(),
        0
    );
}
#[test]
fn malformed_later_slab_call_publishes_zero_effects() {
    let mut run = Run::new(
        vec![call("a"), "<tool_call>{broken}</tool_call>".into()],
        false,
        false,
        false,
    );
    failure(&mut run);
}
#[test]
fn slab_final_usage_failure_publishes_zero_effects() {
    let mut run = Run::new(vec![call("a")], false, true, false);
    failure(&mut run);
}
#[test]
fn slab_cleanup_failure_publishes_zero_effects() {
    let mut run = Run::new(vec![call("a")], false, false, true);
    failure(&mut run);
}
#[test]
fn text_is_available_while_slab_finish_is_blocked() {
    let mut run = Run::new(vec!["lead".into(), call("a")], true, false, false);
    assert!(matches!(run.next(),Some(GenerationEvent::TextDelta{text}) if text=="lead"));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(
        Pin::new(run.stream.as_mut().expect("stream"))
            .poll_next(&mut cx)
            .is_pending()
    );
    run.gate.release();
    let owner = match run.next() {
        Some(GenerationEvent::ToolCallBatchProposed { proposals }) => proposals,
        _ => panic!("missing batch"),
    };
    assert!(matches!(
        run.next(),
        Some(GenerationEvent::GenerationCompleted { .. })
    ));
    drop(owner);
    run.join();
}
#[test]
fn stream_drop_unblocks_slab_publication_and_releases_reservation() {
    let mut run = Run::new(vec!["lead".into(), call("a")], false, false, false);
    let deadline = Instant::now() + WAIT;
    loop {
        if lock(&run.shared.state).text.len() == 1 {
            break;
        }
        assert!(Instant::now() < deadline, "queue timeout");
        thread::yield_now();
    }
    drop(run.stream.take());
    run.join();
    assert_eq!(
        run.admission.snapshot().expect("released").reserved_bytes(),
        0
    );
}
#[test]
fn cancellation_after_a_valid_slab_call_publishes_zero_effects() {
    let mut run = Run::new(vec![call("a")], true, false, false);
    let deadline = Instant::now() + WAIT;
    while !run.gate.entered.load(std::sync::atomic::Ordering::Acquire) {
        assert!(Instant::now() < deadline, "finish gate was not reached");
        thread::yield_now();
    }
    assert_eq!(
        run.scheduler.cancel(&run.request).expect("cancel"),
        crate::types::CancelOutcome::Requested
    );
    run.gate.release();
    failure(&mut run);
}
#[test]
fn truncated_call_after_a_valid_slab_call_publishes_zero_effects() {
    let mut run = Run::new(vec![call("a"), "<tool_call>{".into()], false, false, false);
    failure(&mut run);
}
