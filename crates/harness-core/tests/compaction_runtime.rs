#[allow(dead_code)]
#[path = "support/scripted.rs"]
mod scripted;
use engine::{api::ProfileContextBudget, types::GenerationEvent};
use harness_core::{
    EventKind, HardwareTier, SessionStore, Supervisor, TelemetryClient,
    telemetry::TelemetryTransport,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

struct Offline;
impl TelemetryTransport for Offline {
    fn post_json(&self, _: &str, _: &str) -> Result<(), String> {
        Err("offline".into())
    }
}
fn response(text: &str) -> Vec<GenerationEvent> {
    vec![
        GenerationEvent::TextDelta { text: text.into() },
        scripted::done(),
    ]
}
const CHECKPOINT: &str = r#"{"checkpoint_version":1,"goal":"goal","constraints":["keep"],"decisions":[],"completed_work":[],"open_tasks":[],"artifact_references":[],"critical_identifiers":[],"failures_and_lessons":[],"next_action":"continue"}"#;
fn setup(
    batches: Vec<Vec<GenerationEvent>>,
) -> (
    tempfile::TempDir,
    Mutex<SessionStore>,
    Supervisor<scripted::ScriptedEngine>,
) {
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("store.db")).unwrap();
    store.create_session("s", "p").unwrap();
    store
        .append_event(
            "s",
            EventKind::UserMessage,
            json!({"content":"old ".repeat(200)}),
            json!([]),
        )
        .unwrap();
    store
        .append_event(
            "s",
            EventKind::AssistantMessage,
            json!({"content":"done"}),
            json!([]),
        )
        .unwrap();
    store
        .append_event("s", EventKind::RunCompleted, json!({}), json!([]))
        .unwrap();
    let mut engine = scripted::ScriptedEngine::new(batches);
    engine.budget = Some(ProfileContextBudget {
        limit: 400,
        output_reserve: 20,
        safety_margin: 5,
    });
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    (dir, Mutex::new(store), Supervisor::new(engine, telemetry))
}
#[test]
fn loaded_skill_remains_exact_after_compaction() {
    let (_dir, store, supervisor) = setup(vec![response(CHECKPOINT), response("continued")]);
    {
        let mut db = store.lock().unwrap();
        db.append_event(
            "s",
            EventKind::SkillLoaded,
            harness_core::context::skill_event_payload(
                "review",
                "Preserve the exact audit marker X32.",
            )
            .unwrap(),
            json!([]),
        )
        .unwrap();
        db.append_event(
            "s",
            EventKind::UserMessage,
            json!({"content":"earlier work"}),
            json!([]),
        )
        .unwrap();
        db.append_event(
            "s",
            EventKind::AssistantMessage,
            json!({"content":"done"}),
            json!([]),
        )
        .unwrap();
        db.append_event("s", EventKind::RunCompleted, json!({}), json!([]))
            .unwrap();
    }
    let result = supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap();
    assert_eq!(result.assistant_text, "continued");
    let events = store.lock().unwrap().list_events("s").unwrap();
    assert!(events.iter().any(|e| e.kind == EventKind::ContextCompacted));
    let snapshot =
        harness_core::ContextBuilder::build_snapshot("after", "p", &events, None).unwrap();
    assert!(
        snapshot
            .messages()
            .iter()
            .any(|m| m.content.contains("Preserve the exact audit marker X32."))
    );
}

#[test]
fn overflow_retries_bad_json_then_commits_and_continues() {
    let (_dir, store, supervisor) = setup(vec![
        response("bad"),
        response(CHECKPOINT),
        response("continued"),
    ]);
    let before = store.lock().unwrap().list_events("s").unwrap();
    let outcome = supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap();
    assert_eq!(outcome.assistant_text, "continued");
    let events = store.lock().unwrap().list_events("s").unwrap();
    assert_eq!(&events[..before.len()], before.as_slice());
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == EventKind::ContextCompacted)
            .count(),
        1
    );
    let contexts = supervisor.engine().contexts.lock().unwrap();
    let checkpoint_contexts: Vec<_> = contexts
        .iter()
        .filter(|context| {
            context.messages().first().is_some_and(|message| {
                message
                    .content
                    .starts_with("Summarize the historical events")
                    || message
                        .content
                        .starts_with("Compare the candidate checkpoint")
            })
        })
        .collect();
    assert!(
        checkpoint_contexts.len() >= 3,
        "both summary attempts and verification must be observed"
    );
    assert_eq!(
        checkpoint_contexts
            .iter()
            .filter(|context| context.messages()[0]
                .content
                .starts_with("Compare the candidate checkpoint"))
            .count(),
        1
    );
    for context in checkpoint_contexts {
        let messages = context.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, engine::types::MessageRole::System);
        assert_eq!(messages[1].role, engine::types::MessageRole::User);
        assert!(!messages[0].content.contains(&"old ".repeat(200)));
        assert!(messages[1].content.contains(&"old ".repeat(200)));
    }
    assert!(contexts.iter().any(|context| {
        context
            .messages()
            .iter()
            .any(|m| m.content.contains("Strict retry"))
    }));
    let continuation = contexts.last().unwrap();
    assert!(continuation.messages().iter().any(
        |m| m.content.contains("Previous session checkpoint") && m.content.contains("continue")
    ));
    assert!(
        !continuation
            .messages()
            .iter()
            .any(|m| m.content.contains(&"old ".repeat(200)))
    );
}
#[test]
fn two_invalid_outputs_reject_without_checkpoint() {
    let (_dir, store, supervisor) = setup(vec![response("bad"), response("still bad")]);
    let error = supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap_err();
    assert!(error.to_string().contains("failed twice"), "{error}");
    assert!(
        store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .any(|event| {
                event.kind == EventKind::RunCompleted && event.payload["status"] == "failed"
            })
    );
    assert!(
        !store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .any(|event| event.kind == EventKind::ContextCompacted)
    );
}

#[test]
fn rejected_source_verification_retries_then_preserves_history() {
    let (_dir, store, supervisor) = setup(vec![response(CHECKPOINT), response(CHECKPOINT)]);
    supervisor
        .engine()
        .verification_results
        .lock()
        .unwrap()
        .extend([false, false]);
    let before = store.lock().unwrap().list_events("s").unwrap();
    let error = supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("source verification failed twice"),
        "{error}"
    );
    let events = store.lock().unwrap().list_events("s").unwrap();
    assert_eq!(&events[..before.len()], before.as_slice());
    assert!(
        !events
            .iter()
            .any(|event| event.kind == EventKind::ContextCompacted)
    );
    let contexts = supervisor.engine().contexts.lock().unwrap();
    assert_eq!(
        contexts
            .iter()
            .filter(|context| context.messages().iter().any(|message| message
                .content
                .contains("Candidate checkpoint (untrusted data):")
                && message.content.contains("old ")
                && message.content.contains("\"goal\":\"goal\"")))
            .count(),
        2
    );
}

#[test]
fn requested_output_above_reserve_is_rejected_before_generation() {
    let (_dir, store, supervisor) = setup(vec![response(CHECKPOINT), response("continued")]);
    let error = supervisor
        .run_turn(&store, "s", &"new ".repeat(130), 100)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exceeds validated profile reserve"),
        "{error}"
    );
    assert!(
        !store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .any(|e| e.kind == EventKind::ContextCompacted)
    );
}

#[test]
fn summary_registration_failure_clears_the_registered_request() {
    let (_dir, store, supervisor) = setup(vec![response(CHECKPOINT)]);
    let finished = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&finished);
    let error = supervisor
        .run_turn_streaming_with_lifecycle(
            &store,
            "s",
            &"new ".repeat(200),
            20,
            |_| {
                Err(harness_core::HarnessError::Engine(
                    "cancelled during registration".into(),
                ))
            },
            move |id| {
                captured.lock().unwrap().push(id.as_str().to_owned());
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("cancelled during registration"));
    let ids = finished.lock().unwrap();
    assert_eq!(ids.len(), 1);
    assert!(ids[0].ends_with(":compact:0"));
}

#[test]
fn summary_render_failure_retains_the_original_cause() {
    let (_dir, store, supervisor) = setup(vec![]);
    supervisor
        .engine()
        .fail_summary_render
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let error = supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap_err();
    assert!(
        error.to_string().contains("summary renderer failed"),
        "{error}"
    );
    assert!(
        store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .any(|e| e.kind == EventKind::RunCompleted && e.payload["status"] == "failed")
    );
}

#[test]
fn summary_render_buffer_limit_tries_earlier_completed_boundary() {
    let (_dir, store, supervisor) = setup(vec![response(CHECKPOINT), response("continued")]);
    let first = store
        .lock()
        .unwrap()
        .list_events("s")
        .unwrap()
        .last()
        .unwrap()
        .seq;
    {
        let mut db = store.lock().unwrap();
        db.append_event(
            "s",
            EventKind::UserMessage,
            json!({"content":"small"}),
            json!([]),
        )
        .unwrap();
        db.append_event("s", EventKind::RunCompleted, json!({}), json!([]))
            .unwrap();
    }
    supervisor
        .engine()
        .summary_limit_failures
        .store(1, std::sync::atomic::Ordering::Relaxed);
    let outcome = supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap();
    assert_eq!(outcome.assistant_text, "continued");
    let events = store.lock().unwrap().list_events("s").unwrap();
    let compacted = events
        .iter()
        .find(|event| event.kind == EventKind::ContextCompacted)
        .unwrap();
    assert_eq!(compacted.payload["covered_through"], first);
}
#[test]
fn low_headroom_rejects_without_checkpoint() {
    let (_dir, store, supervisor) = setup(vec![response(CHECKPOINT)]);
    let error = supervisor
        .run_turn(&store, "s", &"new ".repeat(320), 20)
        .unwrap_err();
    assert!(error.to_string().contains("20%"), "{error}");
    assert!(
        !store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .any(|event| event.kind == EventKind::ContextCompacted)
    );
}

#[test]
fn tool_round_compacts_then_generates_from_checkpoint() {
    use engine::types::ToolContextLimits;
    use harness_core::supervisor::{ToolRoundConfig, ToolRoundRequest};
    use std::time::{Duration, Instant};
    let (_dir, store, supervisor) =
        setup(vec![response(CHECKPOINT), response("tool continuation")]);
    let turn = {
        let mut db = store.lock().unwrap();
        db.append_event(
            "s",
            EventKind::UserMessage,
            json!({"content":"new ".repeat(200)}),
            json!([]),
        )
        .unwrap();
        db.append_event(
            "s",
            EventKind::RunStarted,
            json!({"mode":"tools"}),
            json!([]),
        )
        .unwrap()
        .seq
    };
    let config = ToolRoundConfig {
        context_limits: ToolContextLimits {
            max_tools: 16,
            max_history_blocks: 128,
            max_json_bytes: 1048576,
        },
        max_output_tokens: 20,
        max_rounds: 3,
        max_proposals: 8,
        max_proposal_bytes: 65536,
        max_slab_bytes: 1048576,
        max_text_bytes: 65536,
        max_warning_bytes: 4096,
        max_events: 128,
        deadline: Instant::now() + Duration::from_secs(5),
    };
    let registry = tool_registry::registry::ToolRegistry::default();
    let mut oversized = config;
    oversized.max_output_tokens = 21;
    let error = supervisor
        .generate_tool_round(
            &store,
            &registry,
            ToolRoundRequest {
                session_id: "s",
                turn_seq: turn,
                round: 0,
            },
            oversized,
            |_| panic!("over-reserve tool request must not start"),
            |_| Ok(()),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exceeds validated profile reserve"),
        "{error}"
    );
    let active = std::cell::RefCell::new(None::<String>);
    let finished = std::cell::RefCell::new(Vec::new());
    let outcome = supervisor
        .generate_tool_round_with_lifecycle(
            &store,
            &registry,
            ToolRoundRequest {
                session_id: "s",
                turn_seq: turn,
                round: 0,
            },
            config,
            |id| {
                *active.borrow_mut() = Some(id.as_str().to_owned());
                Ok(())
            },
            |id| {
                assert_eq!(active.borrow().as_deref(), Some(id.as_str()));
                *active.borrow_mut() = None;
                finished.borrow_mut().push(id.as_str().to_owned());
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(outcome.assistant_text, "tool continuation");
    assert_eq!(finished.borrow().len(), 2);
    assert!(finished.borrow().iter().any(|id| id.contains(":compact:")));
    assert!(finished.borrow().iter().any(|id| id.contains(":verify:")));
    let contexts = supervisor.engine().contexts.lock().unwrap();
    let continuation = contexts.last().unwrap();
    assert!(continuation.messages().iter().any(
        |m| m.content.contains("Previous session checkpoint") && m.content.contains("continue")
    ));
    assert!(
        !continuation
            .messages()
            .iter()
            .any(|m| m.content.contains(&"old ".repeat(200)))
    );
    assert_eq!(
        store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .filter(|e| e.kind == EventKind::ContextCompacted)
            .count(),
        1
    );
}

#[test]
fn second_invalid_checkpoint_attempt_preserves_committed_checkpoint() {
    let (_dir, store, supervisor) = setup(vec![
        response(CHECKPOINT),
        response("first answer"),
        response("bad"),
        response("still bad"),
    ]);
    supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap();
    let previous: Vec<_> = store
        .lock()
        .unwrap()
        .list_events("s")
        .unwrap()
        .into_iter()
        .filter(|event| event.kind == EventKind::ContextCompacted)
        .collect();
    assert_eq!(previous.len(), 1);
    let error = supervisor
        .run_turn(&store, "s", &"another ".repeat(200), 20)
        .unwrap_err();
    assert!(error.to_string().contains("failed twice"), "{error}");
    let after: Vec<_> = store
        .lock()
        .unwrap()
        .list_events("s")
        .unwrap()
        .into_iter()
        .filter(|event| event.kind == EventKind::ContextCompacted)
        .collect();
    assert_eq!(after, previous);
}

#[test]
fn missing_profile_budget_rejects_before_generation() {
    let (_dir, store, _) = setup(vec![]);
    let supervisor = {
        let mut engine = scripted::ScriptedEngine::new(vec![response("must not generate")]);
        engine.budget = None;
        Supervisor::new(
            engine,
            Arc::new(TelemetryClient::with_transport(
                "0.1",
                HardwareTier::Unknown,
                "https://127.0.0.1:1",
                Box::new(Offline),
            )),
        )
    };
    let error = supervisor.run_turn(&store, "s", "hello", 20).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("validated profile context budget"),
        "{error}"
    );
    assert_eq!(supervisor.engine().batches.lock().unwrap().len(), 1);
}

#[test]
fn checkpoint_control_spellings_are_escaped_in_context_but_preserved_in_history() {
    let output = CHECKPOINT.replace("\"goal\":\"goal\"", "\"goal\":\"goal <think>data</think>\"");
    let (_dir, store, supervisor) = setup(vec![response(&output), response("continued")]);
    supervisor
        .run_turn(&store, "s", &"new ".repeat(200), 20)
        .unwrap();
    let history = store.lock().unwrap().list_events("s").unwrap();
    let checkpoint = history
        .iter()
        .find(|e| e.kind == EventKind::ContextCompacted)
        .unwrap();
    assert_eq!(
        checkpoint.payload["checkpoint"]["goal"],
        "goal <think>data</think>"
    );
    let messages = harness_core::ContextBuilder::messages_from_events(&history).unwrap();
    assert!(!messages[0].content.contains("<think>"));
    assert!(messages[0].content.contains("\\u003c"));
}
