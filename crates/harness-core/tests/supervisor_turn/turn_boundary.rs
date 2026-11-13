use super::*;

// Tests the actual supervisor/store/renderer with controlled inference events.
// This is patch-1 boundary evidence, NOT complete P2-12 or real-model acceptance.
#[derive(Clone, Copy)]
enum W6StreamMode {
    Complete,
    Truncated,
    Failed,
    LegacyToolDelta,
}
struct W6BoundaryEngine {
    inner: ReplyEngine,
    mode: W6StreamMode,
    seen: Arc<Mutex<Vec<String>>>,
}
impl Engine for W6BoundaryEngine {
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
        self.inner.render_and_count(context)
    }
    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        self.inner.estimate_memory(profile, input)
    }
    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let id = request.request_id().clone();
        self.seen.lock().unwrap().push(id.as_str().to_owned());
        let mut events = VecDeque::from([GenerationEvent::TextDelta {
            text: "boundary answer".into(),
        }]);
        match self.mode {
            W6StreamMode::Complete => events.push_back(GenerationEvent::GenerationCompleted {
                finish_reason: FinishReason::EndOfSequence,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                },
            }),
            W6StreamMode::Truncated => {}
            W6StreamMode::Failed => events.push_back(GenerationEvent::RequestFailed {
                error: engine::error::EngineError::Internal {
                    message: "fixture failure".into(),
                },
                partial_usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            }),
            W6StreamMode::LegacyToolDelta => {
                events.push_back(GenerationEvent::ToolCallDelta {
                    call_id: "untrusted".into(),
                    delta: "{}".into(),
                });
                events.push_back(GenerationEvent::GenerationCompleted {
                    finish_reason: FinishReason::EndOfSequence,
                    usage: Usage {
                        input_tokens: 1,
                        output_tokens: 2,
                    },
                });
            }
        }
        Box::pin(async move {
            Ok(Generation::from_stream(
                id,
                Box::pin(StaticStream { events }),
            ))
        })
    }
    fn cancel(&self, id: engine::types::RequestId) -> EngineFuture<'_, CancelOutcome> {
        self.inner.cancel(id)
    }
    fn release_session_state(&self, id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        self.inner.release_session_state(id)
    }
    fn capabilities(&self, id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        self.inner.capabilities(id)
    }
}
struct W6QuietTransport;
impl TelemetryTransport for W6QuietTransport {
    fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
        Ok(())
    }
}
fn tool_loop_boundary(
    mode: W6StreamMode,
) -> (Supervisor<W6BoundaryEngine>, Arc<Mutex<Vec<String>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram36Gb,
        "https://telemetry.example.test/v1",
        Box::new(W6QuietTransport),
    ));
    (
        Supervisor::new(
            W6BoundaryEngine {
                inner: ReplyEngine::new(),
                mode,
                seen: Arc::clone(&seen),
            },
            telemetry,
        ),
        seen,
    )
}
#[test]
fn tool_loop_equal_token_turns_have_distinct_generation_ids() {
    let dir = tempdir().unwrap();
    let store = Mutex::new(SessionStore::open(dir.path().join("identity.sqlite")).unwrap());
    let (supervisor, seen) = tool_loop_boundary(W6StreamMode::Complete);
    supervisor
        .create_session(&store, "same-size", "harness-profile")
        .unwrap();
    let first = supervisor
        .run_turn(&store, "same-size", "hello", 16)
        .unwrap();
    let second = supervisor
        .run_turn(&store, "same-size", "world", 16)
        .unwrap();
    assert_eq!(first.input_tokens, second.input_tokens);
    assert_ne!(first.request_id, second.request_id);
    assert_eq!(
        seen.lock().unwrap().as_slice(),
        &[
            first.request_id.as_str().to_owned(),
            second.request_id.as_str().to_owned()
        ]
    );
}
#[test]
fn tool_loop_restart_and_long_session_preserve_bounded_unique_generation_identity() {
    let dir = tempdir().unwrap();
    let db = dir.path().join("restart.sqlite");
    let session = "long / unicode-é".repeat(30);
    let first_id = {
        let store = Mutex::new(SessionStore::open(&db).unwrap());
        let (supervisor, _) = tool_loop_boundary(W6StreamMode::Complete);
        supervisor
            .create_session(&store, &session, "harness-profile")
            .unwrap();
        supervisor
            .run_turn(&store, &session, "hello", 16)
            .unwrap()
            .request_id
    };
    let store = Mutex::new(SessionStore::open(&db).unwrap());
    let (supervisor, _) = tool_loop_boundary(W6StreamMode::Complete);
    let next = supervisor
        .run_turn(&store, &session, "hello", 16)
        .unwrap()
        .request_id;
    assert_ne!(first_id, next);
    assert_eq!(first_id.as_str().len(), 68);
    assert_eq!(next.as_str().len(), 68);
}
#[test]
fn tool_loop_truncated_failed_and_legacy_tool_streams_never_persist_success() {
    for mode in [
        W6StreamMode::Truncated,
        W6StreamMode::Failed,
        W6StreamMode::LegacyToolDelta,
    ] {
        let dir = tempdir().unwrap();
        let store = Mutex::new(SessionStore::open(dir.path().join("boundary.sqlite")).unwrap());
        let (supervisor, _) = tool_loop_boundary(mode);
        supervisor
            .create_session(&store, "boundary", "harness-profile")
            .unwrap();
        let error = supervisor
            .run_turn(&store, "boundary", "hello", 16)
            .unwrap_err();
        if matches!(mode, W6StreamMode::Truncated) {
            assert!(error.to_string().contains("successful terminal event"));
        }
        let events = store.lock().unwrap().list_events("boundary").unwrap();
        assert!(!events.iter().any(|e| matches!(
            e.kind,
            EventKind::AssistantMessage | EventKind::ToolStarted
        ) || (e.kind == EventKind::RunCompleted
            && e.payload["status"] != "failed")));
        assert!(
            events
                .iter()
                .any(|e| e.kind == EventKind::RunCompleted && e.payload["status"] == "failed")
        );
    }
}
#[test]
fn tool_loop_callback_failure_cannot_persist_a_completed_turn() {
    let dir = tempdir().unwrap();
    let store = Mutex::new(SessionStore::open(dir.path().join("callback.sqlite")).unwrap());
    let (supervisor, _) = tool_loop_boundary(W6StreamMode::Complete);
    supervisor
        .create_session(&store, "callback", "harness-profile")
        .unwrap();
    assert!(
        supervisor
            .run_turn_streaming(
                &store,
                "callback",
                "hello",
                16,
                |_| Ok(()),
                |_| Err(harness_core::HarnessError::Invalid(
                    "callback failure".into()
                ))
            )
            .is_err()
    );
    let events = store.lock().unwrap().list_events("callback").unwrap();
    assert!(!events.iter().any(|e| e.kind == EventKind::AssistantMessage
        || (e.kind == EventKind::RunCompleted && e.payload["status"] != "failed")));
    assert!(
        events
            .iter()
            .any(|e| e.kind == EventKind::RunCompleted && e.payload["status"] == "failed")
    );
}
