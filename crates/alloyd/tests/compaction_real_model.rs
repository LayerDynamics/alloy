//! Opt-in real Qwen compaction acceptance; run with ALLOY_REAL_MODEL pointing to the pinned artifact.
#[path = "support/recording_engine.rs"]
mod recording_engine;
use alloyd::{BootRequest, boot_qwen3};
use engine::{
    api::Engine,
    memory_guard::{MemoryGuardConfig, MemoryGuardTier},
    tiered_kv::{HotCacheLimit, TieredKvConfig},
    types::{ArtifactId, GenerationEvent},
};
use harness_core::{
    ContextBuilder, EventKind, HardwareTier, SessionStore, Supervisor, TelemetryClient,
    telemetry::TelemetryTransport,
};
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

struct TestWake(std::thread::Thread);
impl std::task::Wake for TestWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn wait_future<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::from(Arc::new(TestWake(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
#[ignore = "requires local pinned Qwen3 model and native MLX"]
fn real_rpc_skill_load_changes_next_generation() {
    let root = tempfile::tempdir().unwrap();
    let boot = alloyd::boot::boot_qwen3_with_context_limit(
        BootRequest {
            artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
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
            prefer_metal: true,
            tools_enabled: false,
            tiered_kv: TieredKvConfig::new(
                std::env::var_os("ALLOY_TEST_COLD")
                    .is_none()
                    .then(|| root.path().join("cache")),
                HotCacheLimit::Bytes(0),
            ),
        },
        Some(2048),
    )
    .unwrap();
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let service = alloyd::DaemonService::open(
        root.path().join("sessions.sqlite"),
        boot.engine,
        telemetry,
        16,
        1,
        0,
    )
    .unwrap();
    let rpc = |method: &str, params: serde_json::Value| -> Vec<serde_json::Value> {
        let req: alloyd::RpcRequest = serde_json::from_value(serde_json::json!({"jsonrpc":"2.0","id":1,"api_version":alloyd::API_VERSION,"method":method,"params":params})).unwrap();
        let mut output = Vec::new();
        service.handle(req, &mut output).unwrap();
        std::str::from_utf8(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let created = rpc(
        "session.create",
        serde_json::json!({"profile":"qwen3-4b-4bit"}),
    );
    let session = created[0]["result"]["session_id"].as_str().unwrap();
    let mut outputs = Vec::new();
    for (value, word) in [
        ("The secret color is COBALT.", "COBALT"),
        ("The secret color is AMBER.", "AMBER"),
    ] {
        let loaded = rpc(
            "session.skill.load",
            serde_json::json!({"session_id":session,"name":"answer","instructions":value}),
        );
        assert!(loaded[0].get("error").is_none(), "{loaded:?}");
        let sent = rpc(
            "session.send",
            serde_json::json!({"session_id":session,"content":"What color is specified?"}),
        );
        assert_eq!(sent.last().unwrap()["result"]["ok"], true, "{sent:?}");
        let text = sent
            .iter()
            .filter_map(|event| event["params"]["body"]["text"].as_str())
            .collect::<String>();
        eprintln!("skill={word} answer={text}");
        outputs.push((word, text));
    }
    for (word, text) in outputs {
        assert!(text.to_ascii_uppercase().contains(word), "{word}: {text}");
    }
    let caps = rpc(
        "models.capabilities",
        serde_json::json!({"artifact_id":"qwen3-4b-4bit"}),
    );
    assert_eq!(
        caps[0]["result"]["supports_prefix_reuse"],
        std::env::var_os("ALLOY_TEST_COLD").is_none(),
        "{caps:?}"
    );
}

#[test]
#[ignore = "requires local pinned Qwen3 model and native MLX"]
fn real_skill_change_invalidates_stale_cache_and_matches_cold() {
    let root = tempfile::tempdir().unwrap();
    let make_request = |disk: Option<PathBuf>| BootRequest {
        artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
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
        prefer_metal: true,
        tools_enabled: true,
        tiered_kv: TieredKvConfig::new(disk, HotCacheLimit::Bytes(0)),
    };
    let boot = alloyd::boot::boot_qwen3_with_context_limit(
        make_request(Some(root.path().join("cache"))),
        Some(2048),
    )
    .unwrap();
    let mut db = SessionStore::open(root.path().join("warm.sqlite")).unwrap();
    db.create_session("skill", &boot.profile_id).unwrap();
    db.append_event(
        "skill",
        EventKind::SkillLoaded,
        harness_core::context::skill_event_payload("answer", "The secret color is COBALT.")
            .unwrap(),
        serde_json::json!([]),
    )
    .unwrap();
    let initial = ContextBuilder::build_snapshot(
        "initial",
        &boot.profile_id,
        &db.list_events("skill").unwrap(),
        Some("What color is specified?"),
    )
    .unwrap();
    let initial = ContextBuilder::render_and_count(&boot.engine, initial).unwrap();
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let supervisor = Supervisor::new(boot.engine, telemetry);
    let store = Mutex::new(db);
    let first = supervisor
        .run_turn(&store, "skill", "What color is specified?", 16)
        .unwrap();
    assert!(
        first.assistant_text.to_ascii_uppercase().contains("COBALT"),
        "{}",
        first.assistant_text
    );
    {
        let mut db = store.lock().unwrap();
        db.append_event(
            "skill",
            EventKind::SkillLoaded,
            harness_core::context::skill_event_payload("answer", "The secret color is AMBER.")
                .unwrap(),
            serde_json::json!([]),
        )
        .unwrap();
    }
    let db = store.into_inner().unwrap();
    let changed = ContextBuilder::build_snapshot(
        "changed",
        "qwen3-4b-4bit",
        &db.list_events("skill").unwrap(),
        Some("What color is specified now?"),
    )
    .unwrap();
    let changed = ContextBuilder::render_and_count(supervisor.engine(), changed).unwrap();
    assert_ne!(initial.render_hash(), changed.render_hash());
    let common = initial
        .token_ids()
        .iter()
        .zip(changed.token_ids())
        .take_while(|(a, b)| a == b)
        .count();
    assert!(common < initial.token_count());
    drop(db);
    std::fs::copy(
        root.path().join("warm.sqlite"),
        root.path().join("cold.sqlite"),
    )
    .unwrap();
    let warm_store = Mutex::new(SessionStore::open(root.path().join("warm.sqlite")).unwrap());
    let before = supervisor.engine().runtime().tiered_kv_snapshot();
    let warm = supervisor
        .run_turn(&warm_store, "skill", "What color is specified now?", 16)
        .unwrap();
    let after = supervisor.engine().runtime().tiered_kv_snapshot();
    let restored = after.restored_tokens - before.restored_tokens;
    assert!(
        restored <= common as u64,
        "restored {restored} exceeds unchanged prefix {common}"
    );
    assert!(
        warm.assistant_text.to_ascii_uppercase().contains("AMBER"),
        "{}",
        warm.assistant_text
    );
    let caps = wait_future(
        supervisor
            .engine()
            .capabilities(ArtifactId::new("qwen3-4b-4bit").unwrap()),
    )
    .unwrap();
    assert!(caps.supports_prefix_reuse);
    let rpc_telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let service = alloyd::DaemonService::open(
        root.path().join("rpc.sqlite"),
        supervisor.engine().clone(),
        rpc_telemetry,
        16,
        1,
        0,
    )
    .unwrap();
    let request: alloyd::RpcRequest = serde_json::from_value(serde_json::json!({"jsonrpc":"2.0","id":1,"api_version":alloyd::API_VERSION,"method":"models.capabilities","params":{"artifact_id":"qwen3-4b-4bit"}})).unwrap();
    let mut response = Vec::new();
    service.handle(request, &mut response).unwrap();
    let response: serde_json::Value = serde_json::from_slice(&response).unwrap();
    assert_eq!(
        response["result"]["supports_prefix_reuse"], true,
        "{response}"
    );
    drop(service);
    drop(supervisor);
    let cold_boot =
        alloyd::boot::boot_qwen3_with_context_limit(make_request(None), Some(2048)).unwrap();
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let cold = Supervisor::new(cold_boot.engine, telemetry);
    let cold_store = Mutex::new(SessionStore::open(root.path().join("cold.sqlite")).unwrap());
    let baseline = cold
        .run_turn(&cold_store, "skill", "What color is specified now?", 16)
        .unwrap();
    assert_eq!(warm.input_tokens, baseline.input_tokens);
    assert_eq!(warm.assistant_text, baseline.assistant_text);
    let caps = wait_future(
        cold.engine()
            .capabilities(ArtifactId::new("qwen3-4b-4bit").unwrap()),
    )
    .unwrap();
    assert!(!caps.supports_prefix_reuse);
    println!(
        "real_skill_change: unchanged_prefix={common}, restored_tokens={restored}, response={:?}",
        warm.assistant_text
    );
}

#[test]
#[ignore = "requires local pinned Qwen3 model and native MLX"]
fn real_model_generated_history_has_verified_canonical_bytes() {
    let root = tempfile::tempdir().unwrap();
    let boot = alloyd::boot::boot_qwen3_with_context_limit(
        BootRequest {
            artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
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
            prefer_metal: true,
            tools_enabled: true,
            tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Percent(100)),
        },
        Some(2048),
    )
    .expect("boot pinned Qwen");
    let mut db = SessionStore::open(root.path().join("sessions.sqlite")).unwrap();
    db.create_session("real-prefix", &boot.profile_id).unwrap();
    let prompt = "Reply with exactly the word COBALT.";
    let snapshot =
        ContextBuilder::build_snapshot("real-prefix", &boot.profile_id, &[], Some(prompt)).unwrap();
    let rendered = ContextBuilder::render_and_count(&boot.engine, snapshot).unwrap();
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let supervisor = Supervisor::new(boot.engine, telemetry);
    let result = supervisor
        .run_turn(&Mutex::new(db), "real-prefix", prompt, 16)
        .unwrap();
    assert_eq!(result.input_tokens, rendered.token_count());
    assert!(!result.assistant_text.is_empty());
    let base = format!("{}{}", rendered.text(), result.assistant_text);
    let mut published = None;
    for suffix in ["", "<|im_end|>", "<|endoftext|>"] {
        let bytes = format!("{base}{suffix}").into_bytes();
        if let Ok(id) = wait_future(
            supervisor
                .engine()
                .runtime()
                .retain_session_state(result.request_id.clone(), bytes),
        ) {
            published = Some(id);
            break;
        }
    }
    let id = published.expect("actual generated history has no verified canonical byte roundtrip");
    let state = wait_future(supervisor.engine().runtime().lookup_session_state(id))
        .unwrap()
        .unwrap();
    assert!(state.logical_tokens > result.input_tokens);
    println!(
        "real_generated_prefix: input_tokens={}, executed_tokens={}, state={id}",
        result.input_tokens, state.logical_tokens
    );
}

#[test]
#[ignore = "requires local pinned Qwen3 model and native MLX"]
fn real_model_ssd_prefix_survives_process_restart() {
    const ROOT: &str = "ALLOY_TEST_REAL_SSD_ROOT";
    const PHASE: &str = "ALLOY_TEST_REAL_SSD_PHASE";
    if let Ok(phase) = std::env::var(PHASE) {
        let root = PathBuf::from(std::env::var_os(ROOT).unwrap());
        let boot = alloyd::boot::boot_qwen3_with_context_limit(
            BootRequest {
                artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
                profile_id: "qwen3-4b-4bit".into(),
                corpus_models_json: Some(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../../bench/corpus/models.json"),
                ),
                memory_guard: MemoryGuardConfig {
                    tier: MemoryGuardTier::Balanced,
                    custom_ceiling_bytes: 0,
                    enabled: true,
                },
                admission_budget_override_bytes: None,
                prefer_metal: true,
                tools_enabled: true,
                tiered_kv: TieredKvConfig::new(Some(root.join("cache")), HotCacheLimit::Bytes(0)),
            },
            Some(2048),
        )
        .expect("boot pinned Qwen");
        let mut db = SessionStore::open(root.join(format!("{phase}.sqlite"))).unwrap();
        db.create_session("real-ssd", &boot.profile_id).unwrap();
        let telemetry = Arc::new(TelemetryClient::with_transport(
            "0.1",
            HardwareTier::Unknown,
            "https://127.0.0.1:1",
            Box::new(Offline),
        ));
        let supervisor = Supervisor::new(boot.engine, telemetry);
        let prompt = format!(
            "{} Reply with exactly the word COBALT.",
            "Consider this independent cache validation context. ".repeat(12)
        );
        let result = supervisor
            .run_turn(&Mutex::new(db), "real-ssd", &prompt, 16)
            .expect("real Qwen turn");
        let metrics = supervisor.engine().runtime().tiered_kv_snapshot();
        if phase == "seed" {
            assert!(metrics.spill_count > 0, "{metrics:?}");
            assert_eq!(metrics.restore_count, 0);
        } else {
            assert!(metrics.restore_count > 0, "{metrics:?}");
            assert!(metrics.restored_tokens > 0, "{metrics:?}");
        }
        std::fs::write(
            root.join(format!("{phase}.json")),
            serde_json::to_vec(&serde_json::json!({
                "text": result.assistant_text,
                "input_tokens": result.input_tokens,
                "restore_count": metrics.restore_count,
                "restored_tokens": metrics.restored_tokens,
            }))
            .unwrap(),
        )
        .unwrap();
        return;
    }
    let root = tempfile::tempdir().unwrap();
    for phase in ["seed", "restore"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "real_model_ssd_prefix_survives_process_restart",
                "--nocapture",
            ])
            .env(PHASE, phase)
            .env(ROOT, root.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child {phase} failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let read = |phase: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(root.path().join(format!("{phase}.json"))).unwrap())
            .unwrap()
    };
    let cold = read("seed");
    let restored = read("restore");
    assert_eq!(cold["text"], restored["text"]);
    assert_eq!(cold["input_tokens"], restored["input_tokens"]);
    assert!(restored["restored_tokens"].as_u64().unwrap() > 0);
    println!("real_ssd_restart: cold={cold}, restored={restored}");
}

#[test]
#[ignore = "requires local pinned Qwen3 model and native MLX"]
fn real_model_compaction_preserves_history_and_continues() {
    let artifact = PathBuf::from(std::env::var("ALLOY_REAL_MODEL").expect("set ALLOY_REAL_MODEL"));
    let dir = tempfile::tempdir().unwrap();
    let boot = alloyd::boot::boot_qwen3_with_context_limit(
        BootRequest {
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
            prefer_metal: true,
            tools_enabled: true,
            tiered_kv: TieredKvConfig::new(
                Some(dir.path().join("compaction-kv")),
                HotCacheLimit::Bytes(0),
            ),
        },
        Some(2048),
    )
    .expect("boot real Qwen");
    let mut db = SessionStore::open(dir.path().join("sessions.sqlite")).unwrap();
    db.create_session("real-compact", &boot.profile_id).unwrap();
    let old = format!(
        "Goal: refactor COBALT authentication. Constraint: keep the public API unchanged. Decision: use trait objects. Completed work: extracted the auth trait. Open task: migrate middleware. Artifact reference: art:12. Critical identifier: PR-42. Failure and lesson: bearer whitespace must be handled. Next action: migrate middleware. {}",
        "historic ".repeat(1000)
    );
    let store = Mutex::new(db);
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let supervisor = Supervisor::new(
        recording_engine::RecordingEngine::new(boot.engine),
        telemetry,
    );
    let initial = supervisor
        .run_turn(&store, "real-compact", &old, 64)
        .expect("initial real-model turn");
    assert!(!initial.assistant_text.is_empty());
    let prior = store.lock().unwrap().list_events("real-compact").unwrap();
    assert!(
        prior
            .iter()
            .any(|event| event.kind == EventKind::AssistantMessage)
    );
    let prompt = format!(
        "{} What is the project codename? Reply with the codename.",
        "current ".repeat(500)
    );
    let snapshot =
        ContextBuilder::build_snapshot("before", &boot.profile_id, &prior, Some(&prompt)).unwrap();
    let tokens = ContextBuilder::render_and_count(supervisor.engine(), snapshot)
        .unwrap()
        .token_count();
    assert!(
        tokens + 512 + 64 > 2048,
        "fixture must overflow, got {tokens}"
    );
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
    // The recorder captures the actual request BEFORE generation, including
    // the committed checkpoint and identical decoding options. The cold oracle
    // below consumes this request, never a history containing its warm answer.
    let recorded = supervisor.engine().requests.lock().unwrap();
    let (transition_request, before_transition) = recorded.last().unwrap().clone();
    assert_eq!(transition_request.request_id(), &result.request_id);
    let transition_render = transition_request.expected_render().unwrap();
    assert_eq!(transition_render.token_count(), result.input_tokens);
    let common_prefix = recorded[..recorded.len() - 1]
        .iter()
        .map(|(request, _)| {
            let previous = request.expected_render().unwrap().token_ids();
            let common = previous
                .iter()
                .zip(transition_render.token_ids())
                .take_while(|(a, b)| a == b)
                .count();
            // Every earlier generated prompt diverges before its end. Thus
            // generated suffix tokens cannot extend this common-prefix bound.
            assert!(common < previous.len().min(transition_render.token_count()));
            common
        })
        .max()
        .unwrap();
    drop(recorded);
    let restored_at_transition = supervisor
        .engine()
        .runtime()
        .tiered_kv_snapshot()
        .restored_tokens
        - before_transition.restored_tokens;
    assert!(
        restored_at_transition <= common_prefix as u64,
        "compaction restored {restored_at_transition} beyond matching prefix {common_prefix}"
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
    drop(reopened);
    std::fs::copy(
        dir.path().join("sessions.sqlite"),
        dir.path().join("cold.sqlite"),
    )
    .unwrap();
    let reopened = SessionStore::open(dir.path().join("sessions.sqlite")).unwrap();
    let resumed = Mutex::new(reopened);
    let before_restore = supervisor
        .engine()
        .runtime()
        .tiered_kv_snapshot()
        .restore_count;
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
    let after_restore = supervisor.engine().runtime().tiered_kv_snapshot();
    assert!(
        after_restore.restore_count > before_restore,
        "compacted session did not restore KV: {after_restore:?}"
    );
    drop(supervisor);
    let cold_boot = alloyd::boot::boot_qwen3_with_context_limit(
        BootRequest {
            artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
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
            prefer_metal: true,
            tools_enabled: true,
            tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
        },
        Some(2048),
    )
    .expect("boot cold baseline");
    let cold_render =
        ContextBuilder::render_and_count(&cold_boot.engine, transition_request.context().clone())
            .unwrap();
    assert_eq!(cold_render, *transition_request.expected_render().unwrap());
    let cold_generation = wait_future(cold_boot.engine.generate(transition_request)).unwrap();
    let mut stream = cold_generation.into_events();
    let mut cold_transition = String::new();
    let mut completed = false;
    while let Some(event) = wait_future(std::future::poll_fn(|cx| stream.as_mut().poll_next(cx))) {
        match event {
            GenerationEvent::TextDelta { text } => cold_transition.push_str(&text),
            GenerationEvent::GenerationCompleted { usage, .. } => {
                assert_eq!(usage.input_tokens, result.input_tokens);
                completed = true;
            }
            GenerationEvent::RuntimeWarning { .. } => {}
            other => panic!("unexpected cold transition event: {other:?}"),
        }
    }
    assert!(completed, "cold transition did not complete");
    assert_eq!(
        result.assistant_text, cold_transition,
        "first post-compaction continuation differs from the exact cold request"
    );
    assert_eq!(
        cold_boot
            .engine
            .runtime()
            .tiered_kv_snapshot()
            .restored_tokens,
        0
    );
    println!(
        "real_compaction_transition: matching_prefix={common_prefix}, restored_tokens={restored_at_transition}, cold_output={cold_transition:?}"
    );
    let cold_store = Mutex::new(SessionStore::open(dir.path().join("cold.sqlite")).unwrap());
    let cold_telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let cold_supervisor = Supervisor::new(cold_boot.engine, cold_telemetry);
    let cold = cold_supervisor
        .run_turn(
            &cold_store,
            "real-compact",
            "What is the project codename? Answer briefly.",
            32,
        )
        .expect("cold compacted continuation");
    assert_eq!(followup.input_tokens, cold.input_tokens);
    assert_eq!(followup.assistant_text, cold.assistant_text);
    println!(
        "real_compaction: initial_tokens={tokens}, continued_tokens={}, original_events={}, final_events={final_count}, resumed_tokens={}, restored_tokens={}, restore_count={}, cold_output={:?}",
        result.input_tokens,
        prior.len(),
        followup.input_tokens,
        after_restore.restored_tokens,
        after_restore.restore_count,
        cold.assistant_text,
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
