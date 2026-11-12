//! Opt-in real Qwen compaction acceptance; run with ALLOY_REAL_MODEL pointing to the pinned artifact.
use alloyd::{BootRequest, boot_qwen3};
use engine::{
    memory_guard::{MemoryGuardConfig, MemoryGuardTier},
    tiered_kv::{HotCacheLimit, TieredKvConfig},
};
use harness_core::{
    ContextBuilder, EventKind, HardwareTier, SessionStore, Supervisor, TelemetryClient,
    telemetry::TelemetryTransport,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

struct Offline;
impl TelemetryTransport for Offline {
    fn post_json(&self, _: &str, _: &str) -> Result<(), String> {
        Err("offline".into())
    }
}

#[test]
#[ignore = "requires local pinned Qwen3 model and native MLX"]
fn real_model_compaction_preserves_history_and_continues() {
    let artifact = PathBuf::from(std::env::var("ALLOY_REAL_MODEL").expect("set ALLOY_REAL_MODEL"));
    let dir = tempfile::tempdir().unwrap();
    let boot = boot_qwen3(BootRequest {
        artifact_dir: artifact,
        profile_id: "qwen3-4b-4bit".into(),
        corpus_models_json: Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/corpus/models.json"),
        ),
        memory_guard: MemoryGuardConfig {
            tier: MemoryGuardTier::Balanced,
            custom_ceiling_bytes: 0,
            enabled: true,
        },
        admission_budget_override_bytes: None,
        profile_context_limit: Some(2048),
        prefer_metal: true,
        tools_enabled: true,
        tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
    })
    .expect("boot real Qwen");
    let mut db = SessionStore::open(dir.path().join("sessions.sqlite")).unwrap();
    db.create_session("real-compact", &boot.profile_id).unwrap();
    let old = format!(
        "Goal: refactor COBALT authentication. Constraint: keep the public API unchanged. Decision: use trait objects. Completed work: extracted the auth trait. Open task: migrate middleware. Artifact reference: art:12. Critical identifier: PR-42. Failure and lesson: bearer whitespace must be handled. Next action: migrate middleware. {}",
        "historic ".repeat(1000)
    );
    db.append_event(
        "real-compact",
        EventKind::UserMessage,
        json!({"content":old}),
        json!([]),
    )
    .unwrap();
    db.append_event(
        "real-compact",
        EventKind::AssistantMessage,
        json!({"content":"Understood: COBALT, no public API changes, next migrate middleware."}),
        json!([]),
    )
    .unwrap();
    db.append_event(
        "real-compact",
        EventKind::RunCompleted,
        json!({"status":"completed"}),
        json!([]),
    )
    .unwrap();
    let prior = db.list_events("real-compact").unwrap();
    let prompt = format!(
        "{} What is the project codename? Reply with the codename.",
        "current ".repeat(500)
    );
    let snapshot =
        ContextBuilder::build_snapshot("before", &boot.profile_id, &prior, Some(&prompt)).unwrap();
    let tokens = ContextBuilder::render_and_count(&boot.engine, snapshot)
        .unwrap()
        .token_count();
    assert!(
        tokens + 512 + 64 > 2048,
        "fixture must overflow, got {tokens}"
    );
    let store = Mutex::new(db);
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let supervisor = Supervisor::new(boot.engine, telemetry);
    let result = supervisor
        .run_turn(&store, "real-compact", &prompt, 128)
        .expect("compact and continue");
    assert!(
        result
            .assistant_text
            .to_ascii_uppercase()
            .contains("COBALT"),
        "{}",
        result.assistant_text
    );
    let events = store.lock().unwrap().list_events("real-compact").unwrap();
    assert_eq!(&events[..prior.len()], prior.as_slice());
    let compacted: Vec<_> = events
        .iter()
        .filter(|e| e.kind == EventKind::ContextCompacted)
        .collect();
    assert_eq!(compacted.len(), 1);
    let checkpoint = &compacted[0].payload["checkpoint"];
    for (field, marker) in [
        ("goal", "cobalt"),
        ("constraints", "api"),
        ("decisions", "trait"),
        ("completed_work", "trait"),
        ("open_tasks", "middleware"),
        ("artifact_references", "art:12"),
        ("critical_identifiers", "pr-42"),
        ("failures_and_lessons", "whitespace"),
        ("next_action", "middleware"),
    ] {
        assert!(
            checkpoint[field]
                .to_string()
                .to_ascii_lowercase()
                .contains(marker),
            "checkpoint {field} lost {marker}: {checkpoint}"
        );
    }
    let final_count = events.len();
    let db = store.into_inner().unwrap();
    drop(db);
    let reopened = SessionStore::open(dir.path().join("sessions.sqlite")).unwrap();
    assert_eq!(reopened.list_events("real-compact").unwrap(), events);
    let resumed = Mutex::new(reopened);
    let followup = supervisor
        .run_turn(
            &resumed,
            "real-compact",
            "What is the project codename? Answer briefly.",
            32,
        )
        .expect("continue after reopen");
    assert!(
        followup
            .assistant_text
            .to_ascii_uppercase()
            .contains("COBALT"),
        "{}",
        followup.assistant_text
    );
    println!(
        "real_compaction: initial_tokens={tokens}, continued_tokens={}, original_events={}, final_events={final_count}, resumed_tokens={}",
        result.input_tokens,
        prior.len(),
        followup.input_tokens
    );
}

#[test]
#[ignore = "requires pinned Qwen artifact; timing must use an optimized build"]
fn real_model_render_32k_latency() {
    let artifact = PathBuf::from(std::env::var("ALLOY_REAL_MODEL").expect("set ALLOY_REAL_MODEL"));
    let boot = boot_qwen3(BootRequest {
        artifact_dir: artifact,
        profile_id: "qwen3-4b-4bit".into(),
        corpus_models_json: Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/corpus/models.json"),
        ),
        memory_guard: MemoryGuardConfig {
            tier: MemoryGuardTier::Balanced,
            custom_ceiling_bytes: 0,
            enabled: true,
        },
        admission_budget_override_bytes: None,
        profile_context_limit: None,
        prefer_metal: true,
        tools_enabled: false,
        tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
    })
    .expect("boot Qwen renderer");
    let doc = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/corpus/fixtures/documents/document-32k.txt"),
    )
    .unwrap();
    let snapshot = engine::types::ContextSnapshot::new(
        engine::types::ContextSnapshotId::new("timing-32k").unwrap(),
        engine::types::ModelProfileId::new(&boot.profile_id).unwrap(),
        vec![engine::types::Message {
            role: engine::types::MessageRole::User,
            content: doc,
        }],
    )
    .unwrap();
    let warm = ContextBuilder::render_and_count(&boot.engine, snapshot.clone()).unwrap();
    assert!(
        (32_000..=34_000).contains(&warm.token_count()),
        "unexpected token count {}",
        warm.token_count()
    );
    let mut elapsed = Vec::new();
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let rendered = ContextBuilder::render_and_count(&boot.engine, snapshot.clone()).unwrap();
        elapsed.push(start.elapsed());
        assert_eq!(rendered.token_ids(), warm.token_ids());
    }
    let worst = *elapsed.iter().max().unwrap();
    println!(
        "render_32k: tokens={}, warm_samples_ms={:?}, worst_ms={:.3}",
        warm.token_count(),
        elapsed
            .iter()
            .map(|d| d.as_secs_f64() * 1000.)
            .collect::<Vec<_>>(),
        worst.as_secs_f64() * 1000.
    );
    assert!(
        worst < std::time::Duration::from_millis(150),
        "32k render exceeded 150 ms"
    );
}
