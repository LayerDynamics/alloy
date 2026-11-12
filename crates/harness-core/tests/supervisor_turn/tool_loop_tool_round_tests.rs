// Scripted engine + actual supervisor/context/registry/store. No builtin dispatch.
use super::*;
use engine::{
    admission::AdmissionController,
    parser_allocation::{ParserAllocation, ParserAllocationLimits},
    types::{ToolCallProposal, ToolContextLimits},
};
use harness_core::supervisor::{RoundProposals, ToolRoundConfig, ToolRoundRequest};
use std::time::Instant;
use tool_registry::{
    call::ToolInvocation,
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
    registry::{ToolImplementation, ToolRegistry},
    result::{InlineToolResult, ToolResult},
};
#[derive(Clone, Copy)]
enum Mode {
    Normal,
    Pending,
    WrongId,
}
struct RoundQuietTransport;
impl TelemetryTransport for RoundQuietTransport {
    fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
        Ok(())
    }
}
struct CounterTool(Arc<AtomicUsize>);
impl ToolImplementation for CounterTool {
    fn execute(&self, invocation: &ToolInvocation) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::Inline(InlineToolResult {
            call_id: invocation.call.call_id.clone(),
            content: json!("not permitted in round-only tests"),
            is_error: true,
        })
    }
}
struct RoundEngine {
    inner: ReplyEngine,
    store: Arc<Mutex<SessionStore>>,
    batches: Mutex<VecDeque<VecDeque<GenerationEvent>>>,
    requests: Mutex<Vec<engine::types::RequestId>>,
    cancels: Mutex<Vec<engine::types::RequestId>>,
    rendered: Mutex<Vec<ContextSnapshot>>,
    mode: Mode,
}
impl Engine for RoundEngine {
    fn profile_context_budget(
        &self,
        _: &engine::types::ModelProfileId,
    ) -> Result<Option<engine::api::ProfileContextBudget>, engine::error::EngineError> {
        Ok(Some(engine::api::ProfileContextBudget {
            limit: 1_000_000,
            output_reserve: 512,
            safety_margin: 64,
        }))
    }

    fn api_version(&self) -> u32 {
        self.inner.api_version()
    }
    fn render_and_count(
        &self,
        context: ContextSnapshot,
    ) -> EngineFuture<'_, engine::types::RenderedContext> {
        assert!(
            self.store.try_lock().is_ok(),
            "no store lock across rendering"
        );
        assert!(
            context.tool_context().is_some(),
            "patch-2 builder must be used"
        );
        self.rendered.lock().unwrap().push(context.clone());
        self.inner.render_and_count(context)
    }
    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        count: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        self.inner.estimate_memory(profile, count)
    }
    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        assert!(
            self.store.try_lock().is_ok(),
            "no store lock across generate"
        );
        self.requests
            .lock()
            .unwrap()
            .push(request.request_id().clone());
        let id = if matches!(self.mode, Mode::WrongId) {
            engine::types::RequestId::new("foreign-handle").unwrap()
        } else {
            request.request_id().clone()
        };
        let events = self.batches.lock().unwrap().pop_front().unwrap_or_default();
        let pending = matches!(self.mode, Mode::Pending);
        Box::pin(async move {
            Ok(Generation::from_stream(
                id,
                Box::pin(RoundStream {
                    events,
                    pending,
                    _waker: None,
                }),
            ))
        })
    }
    fn cancel(&self, id: engine::types::RequestId) -> EngineFuture<'_, CancelOutcome> {
        assert!(
            self.store.try_lock().is_ok(),
            "no store lock across cancellation"
        );
        self.cancels.lock().unwrap().push(id);
        Box::pin(async { Ok(CancelOutcome::Requested) })
    }
    fn release_session_state(&self, id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        self.inner.release_session_state(id)
    }
    fn capabilities(&self, id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        self.inner.capabilities(id)
    }
}
struct RoundStream {
    events: VecDeque<GenerationEvent>,
    pending: bool,
    _waker: Option<std::task::Waker>,
}
impl GenerationEventStream for RoundStream {
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        let this = self.get_mut();
        if let Some(event) = this.events.pop_front() {
            return Poll::Ready(Some(event));
        }
        if this.pending {
            this._waker = Some(cx.waker().clone());
            Poll::Pending
        } else {
            Poll::Ready(None)
        }
    }
}
struct Fixture {
    store: Arc<Mutex<SessionStore>>,
    supervisor: Supervisor<RoundEngine>,
    registry: ToolRegistry,
    seq: i64,
    executions: Arc<AtomicUsize>,
    _dir: tempfile::TempDir,
}
fn fixture(batches: Vec<Vec<GenerationEvent>>, mode: Mode) -> Fixture {
    let dir = tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("round.sqlite")).unwrap();
    store
        .create_session("round-session", "harness-profile")
        .unwrap();
    store
        .append_event(
            "round-session",
            EventKind::UserMessage,
            json!({"content":"hello"}),
            json!([]),
        )
        .unwrap();
    let seq = store
        .append_event("round-session", EventKind::RunStarted, json!({}), json!([]))
        .unwrap()
        .seq;
    let store = Arc::new(Mutex::new(store));
    let executions = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry.register(ToolDefinition { id:"fixture.read".into(), version:"1.0.0".into(), description:"Read fixture".into(),
        input_schema:json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}),
        origin:ToolOrigin::Builtin, policy:ToolPolicy { effect:ToolEffect::Read, approval:ApprovalRequirement::Never,
            timeout_ms:5000, max_output_bytes:65536 } }, Arc::new(CounterTool(Arc::clone(&executions)))).unwrap();
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram36Gb,
        "https://telemetry.example.test/v1",
        Box::new(RoundQuietTransport),
    ));
    let supervisor = Supervisor::new(
        RoundEngine {
            inner: ReplyEngine::new(),
            store: Arc::clone(&store),
            batches: Mutex::new(batches.into_iter().map(VecDeque::from).collect()),
            requests: Mutex::new(Vec::new()),
            cancels: Mutex::new(Vec::new()),
            rendered: Mutex::new(Vec::new()),
            mode,
        },
        telemetry,
    );
    Fixture {
        store,
        supervisor,
        registry,
        seq,
        executions,
        _dir: dir,
    }
}
fn config() -> ToolRoundConfig {
    ToolRoundConfig {
        context_limits: ToolContextLimits {
            max_tools: 8,
            max_history_blocks: 32,
            max_json_bytes: 16384,
        },
        max_output_tokens: 16,
        max_rounds: 4,
        max_proposals: 4,
        max_proposal_bytes: 2048,
        max_slab_bytes: 65536,
        max_text_bytes: 1024,
        max_warning_bytes: 256,
        max_events: 32,
        deadline: Instant::now() + Duration::from_secs(5),
    }
}
fn request(fixture: &Fixture, round: u64) -> ToolRoundRequest<'_> {
    ToolRoundRequest {
        session_id: "round-session",
        turn_seq: fixture.seq,
        round,
    }
}
fn text(value: &str) -> GenerationEvent {
    GenerationEvent::TextDelta { text: value.into() }
}
fn done() -> GenerationEvent {
    GenerationEvent::GenerationCompleted {
        finish_reason: FinishReason::EndOfSequence,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    }
}
fn legacy(name: &str) -> GenerationEvent {
    GenerationEvent::ToolCallProposed {
        proposal: ToolCallProposal::new(
            name,
            json!({"path":"fixture.txt","approved":true,"call_id":"model-data"})
                .as_object()
                .unwrap()
                .clone(),
        )
        .unwrap(),
    }
}
fn slab() -> (AdmissionController, GenerationEvent) {
    let limits = ParserAllocationLimits {
        carry_bytes: 32,
        call_bytes: 512,
        decoded_bytes: 128,
        depth: 8,
        keys: 16,
        nodes: 32,
        proposal_bytes: 512,
        calls: 4,
    };
    let bytes = ParserAllocation::required_bytes(limits).unwrap();
    let admission = AdmissionController::new(bytes).unwrap();
    let mut grant = admission.reserve_grant("round-test", &[bytes]).unwrap();
    let mut storage = ParserAllocation::from_grant(&mut grant, limits).unwrap();
    let name = b"fixture.read";
    let args = br#"{"path":"fixture.txt"}"#;
    {
        let work = storage.workspaces();
        work.decoded[..name.len()].copy_from_slice(name);
        work.call[..args.len()].copy_from_slice(args);
    }
    storage
        .commit_validated_call(name.len(), 0..name.len(), args.len(), 0..args.len())
        .unwrap();
    (
        admission,
        GenerationEvent::ToolCallBatchProposed {
            proposals: storage.freeze(),
        },
    )
}
fn assert_no_effects(fixture: &Fixture) {
    assert_eq!(fixture.executions.load(Ordering::SeqCst), 0);
    let events = fixture
        .store
        .lock()
        .unwrap()
        .list_events("round-session")
        .unwrap();
    assert_eq!(
        events.len(),
        3,
        "a round-only API must not append turn/tool completion or admission"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event.kind, EventKind::ToolStarted | EventKind::RunCompleted))
    );
}

#[test]
fn tool_loop_round_uses_canonical_context_and_withholds_engine_terminals() {
    let f = fixture(vec![vec![text("answer"), done()]], Mode::Normal);
    let mut forwarded = Vec::new();
    let outcome = f
        .supervisor
        .generate_tool_round(
            &f.store,
            &f.registry,
            request(&f, 0),
            config(),
            |_| {
                assert!(f.store.try_lock().is_ok());
                Ok(())
            },
            |event| {
                assert!(f.store.try_lock().is_ok());
                assert!(!event.is_terminal());
                forwarded.push(matches!(event, GenerationEvent::TextDelta { .. }));
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(outcome.assistant_text, "answer");
    assert_eq!(outcome.input_tokens, 1);
    assert!(outcome.proposals.is_empty());
    assert_eq!(forwarded, [true]);
    assert!(
        f.supervisor.engine().rendered.lock().unwrap()[0]
            .tool_context()
            .is_some()
    );
    assert!(f.supervisor.engine().cancels.lock().unwrap().is_empty());
    assert_no_effects(&f);
}
#[test]
fn tool_loop_round_legacy_proposals_are_untrusted_and_ids_change_by_round() {
    let f = fixture(
        vec![
            vec![legacy("unregistered.tool"), done()],
            vec![legacy("fixture.read"), done()],
        ],
        Mode::Normal,
    );
    let first = f
        .supervisor
        .generate_tool_round(
            &f.store,
            &f.registry,
            request(&f, 0),
            config(),
            |_| Ok(()),
            |_| panic!("proposals/terminals must not be forwarded"),
        )
        .unwrap();
    let second = f
        .supervisor
        .generate_tool_round(
            &f.store,
            &f.registry,
            request(&f, 1),
            config(),
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
    assert_ne!(first.request_id, second.request_id);
    assert_eq!(first.request_id.as_str().len(), 68);
    match first.proposals {
        RoundProposals::Legacy(calls) => {
            assert_eq!(calls[0].name(), "unregistered.tool");
            assert_eq!(calls[0].arguments()["approved"], true);
            assert_eq!(calls[0].arguments()["call_id"], "model-data");
        }
        _ => panic!("legacy owner expected"),
    }
    assert_no_effects(&f);
}
#[test]
fn tool_loop_round_retains_original_slab_reservation_until_outcome_drop() {
    let (admission, batch) = slab();
    let bytes = admission.snapshot().unwrap().reserved_bytes();
    let f = fixture(
        vec![vec![
            batch,
            GenerationEvent::RuntimeWarning {
                code: "warning".into(),
                message: "still owned".into(),
            },
            done(),
        ]],
        Mode::Normal,
    );
    let outcome = f
        .supervisor
        .generate_tool_round(
            &f.store,
            &f.registry,
            request(&f, 0),
            config(),
            |_| Ok(()),
            |_| {
                assert_eq!(admission.snapshot().unwrap().reserved_bytes(), bytes);
                Ok(())
            },
        )
        .unwrap();
    match &outcome.proposals {
        RoundProposals::Slab(owner) => {
            assert_eq!(owner.reserved_bytes(), bytes);
            assert_eq!(
                owner.proposal(0).unwrap().arguments_json,
                r#"{"path":"fixture.txt"}"#
            );
        }
        _ => panic!("original slab expected"),
    }
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), bytes);
    drop(outcome);
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    assert_no_effects(&f);
}
#[test]
fn tool_loop_round_failure_eof_and_output_limit_drop_proposals_without_effects() {
    for ending in 0..4 {
        let (admission, batch) = slab();
        let mut events = vec![batch];
        match ending {
            0 => {}
            1 => events.push(GenerationEvent::RequestFailed {
                error: engine::error::EngineError::Internal {
                    message: "scripted failure".into(),
                },
                partial_usage: Usage::default(),
            }),
            2 => events.push(GenerationEvent::GenerationCompleted {
                finish_reason: FinishReason::OutputLimit,
                usage: Usage::default(),
            }),
            _ => events.push(GenerationEvent::RequestFailed {
                error: engine::error::EngineError::Cancelled {
                    request_id: "scripted".into(),
                },
                partial_usage: Usage::default(),
            }),
        }
        let f = fixture(vec![events], Mode::Normal);
        assert!(
            f.supervisor
                .generate_tool_round(
                    &f.store,
                    &f.registry,
                    request(&f, 0),
                    config(),
                    |_| Ok(()),
                    |_| Ok(())
                )
                .is_err()
        );
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        assert_eq!(f.supervisor.engine().cancels.lock().unwrap().len(), 1);
        assert_no_effects(&f);
    }
}
#[test]
fn tool_loop_round_rejects_raw_deltas_mixed_delivery_and_late_text() {
    for kind in 0..4 {
        let (admission, batch) = slab();
        let events = match kind {
            0 => vec![batch, legacy("fixture.read"), done()],
            1 => vec![legacy("fixture.read"), batch, done()],
            2 => vec![batch, text("late text"), done()],
            _ => vec![
                batch,
                GenerationEvent::ToolCallDelta {
                    call_id: "model-id".into(),
                    delta: "{}".into(),
                },
                done(),
            ],
        };
        let f = fixture(vec![events], Mode::Normal);
        assert!(
            f.supervisor
                .generate_tool_round(
                    &f.store,
                    &f.registry,
                    request(&f, 0),
                    config(),
                    |_| Ok(()),
                    |_| Ok(())
                )
                .is_err()
        );
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        assert_no_effects(&f);
    }
}
#[test]
fn tool_loop_round_enforces_text_warning_event_proposal_and_slab_limits() {
    for kind in 0..7 {
        let (admission, batch) = slab();
        let mut selected = config();
        let events = match kind {
            0 => {
                selected.max_text_bytes = 1;
                vec![text("too long"), done(), batch]
            }
            1 => {
                selected.max_warning_bytes = 1;
                vec![
                    GenerationEvent::RuntimeWarning {
                        code: "too".into(),
                        message: "large".into(),
                    },
                    batch,
                    done(),
                ]
            }
            2 => {
                selected.max_events = 1;
                vec![text("x"), done(), batch]
            }
            3 => {
                selected.max_proposals = 1;
                vec![
                    legacy("fixture.read"),
                    legacy("fixture.read"),
                    batch,
                    done(),
                ]
            }
            4 => {
                selected.max_proposal_bytes = "fixture.read".len() + 1;
                vec![batch, done()]
            }
            5 => {
                selected.max_slab_bytes = 1;
                vec![batch, done()]
            }
            _ => {
                selected.max_proposal_bytes = "fixture.read".len() + 1;
                vec![legacy("fixture.read"), batch, done()]
            }
        };
        let f = fixture(vec![events], Mode::Normal);
        assert!(
            f.supervisor
                .generate_tool_round(
                    &f.store,
                    &f.registry,
                    request(&f, 0),
                    selected,
                    |_| Ok(()),
                    |_| Ok(())
                )
                .is_err(),
            "limit case {kind}"
        );
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        assert_no_effects(&f);
    }
}
#[test]
fn tool_loop_round_deadline_and_callback_failure_drop_owned_proposals() {
    for kind in 0..3 {
        let (admission, batch) = slab();
        let events = if kind == 0 {
            vec![batch]
        } else {
            vec![
                batch,
                GenerationEvent::RuntimeWarning {
                    code: "w".into(),
                    message: "after batch".into(),
                },
                done(),
            ]
        };
        let f = fixture(
            vec![events],
            if kind == 0 {
                Mode::Pending
            } else {
                Mode::Normal
            },
        );
        let mut selected = config();
        selected.deadline = Instant::now() + Duration::from_secs(if kind == 2 { 5 } else { 1 });
        let result = f.supervisor.generate_tool_round(
            &f.store,
            &f.registry,
            request(&f, 0),
            selected,
            |_| Ok(()),
            |_| {
                if kind == 1 {
                    thread::sleep(Duration::from_millis(1050));
                    Ok(())
                } else {
                    Err(harness_core::HarnessError::Invalid(
                        "callback failed".into(),
                    ))
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        assert_eq!(f.supervisor.engine().cancels.lock().unwrap().len(), 1);
        assert_no_effects(&f);
    }
}
#[test]
fn tool_loop_round_pause_or_closed_turn_cannot_publish_proposals() {
    for pause in [false, true] {
        let (admission, batch) = slab();
        let f = fixture(
            vec![vec![
                batch,
                GenerationEvent::RuntimeWarning {
                    code: "w".into(),
                    message: "pause point".into(),
                },
                done(),
            ]],
            Mode::Normal,
        );
        let result = f.supervisor.generate_tool_round(
            &f.store,
            &f.registry,
            request(&f, 0),
            config(),
            |_| Ok(()),
            |_| {
                let mut store = f.store.lock().unwrap();
                if pause {
                    store.set_status("round-session", harness_core::SessionStatus::Paused)?;
                } else {
                    store.append_event(
                        "round-session",
                        EventKind::RunCompleted,
                        json!({}),
                        json!([]),
                    )?;
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        assert_eq!(f.executions.load(Ordering::SeqCst), 0);
        assert_eq!(f.supervisor.engine().cancels.lock().unwrap().len(), 1);
    }
}
#[test]
fn tool_loop_round_bad_identity_or_closed_budget_never_cancels_foreign_request() {
    let f = fixture(vec![vec![text("answer"), done()]], Mode::WrongId);
    assert!(
        f.supervisor
            .generate_tool_round(
                &f.store,
                &f.registry,
                request(&f, 0),
                config(),
                |_| Ok(()),
                |_| Ok(())
            )
            .is_err()
    );
    let engine = f.supervisor.engine();
    assert_eq!(
        engine.cancels.lock().unwrap()[0],
        engine.requests.lock().unwrap()[0]
    );
    assert_ne!(engine.cancels.lock().unwrap()[0].as_str(), "foreign-handle");
    assert_no_effects(&f);
    for kind in 0..4 {
        let f = fixture(vec![], Mode::Normal);
        let mut selected = config();
        let mut round_request = request(&f, 0);
        match kind {
            0 => round_request.round = selected.max_rounds,
            1 => selected.deadline = Instant::now(),
            2 => round_request.turn_seq += 1,
            _ => selected.max_proposals = 0,
        }
        assert!(
            f.supervisor
                .generate_tool_round(
                    &f.store,
                    &f.registry,
                    round_request,
                    selected,
                    |_| Ok(()),
                    |_| Ok(())
                )
                .is_err()
        );
        assert!(f.supervisor.engine().requests.lock().unwrap().is_empty());
        assert!(f.supervisor.engine().cancels.lock().unwrap().is_empty());
        assert_no_effects(&f);
    }
}
