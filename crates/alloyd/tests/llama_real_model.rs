//! Real Llama conformance. Run each frozen profile with ALLOY_REAL_MODEL/PROFILE.
use alloyd::{BootRequest, DaemonService, RpcRequest};
use engine::{
    memory_guard::MemoryGuardConfig,
    tiered_kv::{HotCacheLimit, TieredKvConfig},
};
use harness_core::{HardwareTier, TelemetryClient, telemetry::TelemetryTransport};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc};
struct Offline;
impl TelemetryTransport for Offline {
    fn post_json(&self, _: &str, _: &str) -> Result<(), String> {
        Err("offline".into())
    }
}
#[test]
#[ignore = "requires pinned local Llama artifact and native Metal"]
fn llama3_tool_stop_context_cache() {
    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    assert!(matches!(
        profile.as_str(),
        "llama-3.1-8b-4bit" | "llama-3.1-8b-8bit"
    ));
    let root = tempfile::tempdir().unwrap();
    let boot = alloyd::boot::boot_qwen3_with_context_limit(
        BootRequest {
            artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
            profile_id: profile.clone(),
            corpus_models_json: None,
            memory_guard: MemoryGuardConfig::balanced(),
            admission_budget_override_bytes: None,
            prefer_metal: true,
            tools_enabled: true,
            tiered_kv: TieredKvConfig::new(Some(root.path().join("kv")), HotCacheLimit::Bytes(0)),
        },
        Some(16384),
    )
    .unwrap();
    let service = DaemonService::open(
        root.path().join("session.sqlite"),
        boot.engine,
        Arc::new(TelemetryClient::with_transport(
            "test",
            HardwareTier::Unknown,
            "https://127.0.0.1:1",
            Box::new(Offline),
        )),
        128,
        0,
        0,
    )
    .unwrap();
    let config = json!({"read_root":root.path(),"write_root":root.path(),"shell_executables":[std::fs::canonicalize("/bin/echo").unwrap()],"shell_cwd_roots":[root.path()],"http_destinations":[{"origin":"http://fixture.invalid/","address":"127.0.0.1"}],"plugins":[],"policy":{"rules":[{"tool_id":"fs.write","allowed_arguments":[{"path":"marker.txt","content":"COBALT"}],"require_approval":false,"timeout_ms":1000,"max_output_bytes":1024}]}});
    let config_path = root.path().join("tools.json");
    std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let artifacts = service
        .open_artifacts(&root.path().join("artifacts"))
        .unwrap();
    let service = service.with_tools(alloyd::tool_config::load(&config_path, artifacts).unwrap());
    let rpc = |method: &str, params: Value| {
        let request: RpcRequest = serde_json::from_value(
            json!({"jsonrpc":"2.0","id":1,"api_version":1,"method":method,"params":params}),
        )
        .unwrap();
        let mut bytes = Vec::new();
        service.handle(request, &mut bytes).unwrap();
        std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect::<Vec<_>>()
    };
    let create = rpc("session.create", json!({"profile":profile}));
    let session = create[0]["result"]["session_id"].as_str().unwrap();
    let events = rpc(
        "session.send",
        json!({"session_id":session,"content":"Use fs.write exactly once to write marker.txt with exactly the content COBALT. After the tool succeeds, reply DONE. Do not use any other tool."}),
    );
    eprintln!(
        "tool_generation={}",
        serde_json::to_string(&events).unwrap()
    );
    for event in &events {
        assert!(event.get("error").is_none(), "{event}");
        assert_ne!(event["params"]["event"], "request_failed", "{event}");
    }
    assert_eq!(
        std::fs::read_to_string(root.path().join("marker.txt")).expect("authorized tool effect"),
        "COBALT"
    );

    eprintln!(
        "llama tool profile={profile} events={}",
        serde_json::to_string(&events).unwrap()
    );

    let matching = |events: &[Value], name: &str| {
        events
            .iter()
            .filter(|e| e["params"]["event"] == name)
            .count()
    };
    assert_eq!(matching(&events, "tool_requested"), 1);
    assert_eq!(matching(&events, "tool_call_started"), 1);
    assert!(
        events
            .iter()
            .any(|e| e["params"]["event"] == "tool_result"
                && e["params"]["body"]["is_error"] == false)
    );
    assert!(
        events
            .iter()
            .any(|e| e["params"]["event"] == "generation_completed"
                && e["params"]["body"]["finish_reason"] == "end_of_sequence")
    );
    assert_eq!(events.last().unwrap()["result"]["assistant_text"], "DONE");
    let caps = rpc("models.capabilities", json!({"artifact_id":profile}));
    assert_eq!(caps[0]["result"]["family"], "llama3");
    assert_eq!(caps[0]["result"]["backend"]["kind"], "mlx_metal");
    assert_eq!(
        caps[0]["result"]["weight_representations"],
        json!([{"kind":if profile.ends_with("8bit") {"affine8"} else {"affine4"}, "group_size":64}])
    );
    assert_eq!(
        caps[0]["result"]["cache_modes"][0]["representation"],
        "llama3.mlx-f32.full.bhtd.v1"
    );
    let denied = rpc(
        "session.send",
        json!({"session_id":session,"content":"Now use fs.write to write forbidden.txt with content BLOCKED."}),
    );
    assert!(!root.path().join("forbidden.txt").exists());
    assert_eq!(matching(&denied, "tool_requested"), 1);
    assert_eq!(matching(&denied, "tool_call_started"), 0);
    assert!(denied.iter().any(|e| e["params"]["event"] == "tool_result"
        && e["params"]["body"]["state"] == "Rejected"
        && e["params"]["body"]["reason"] == "policy:ArgumentsNotAllowed"));
    assert!(
        denied
            .iter()
            .any(|e| e["params"]["event"] == "generation_completed")
    );
    assert!(
        denied
            .iter()
            .all(|e| e.get("error").is_none() && e["params"]["event"] != "request_failed")
    );
    eprintln!("denied_events={}", serde_json::to_string(&denied).unwrap());
}

#[test]
#[ignore = "requires pinned local Llama artifact and native Metal"]
fn llama3_long_context_stream_cancel_memory() {
    use engine::types::{FinishReason, GenerationEvent};
    use harness_core::{ContextBuilder, SessionStore, Supervisor};
    use std::sync::Mutex;
    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let root = tempfile::tempdir().unwrap();
    let request = |budget| BootRequest {
        artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
        profile_id: profile.clone(),
        corpus_models_json: None,
        memory_guard: MemoryGuardConfig::balanced(),
        admission_budget_override_bytes: budget,
        prefer_metal: true,
        tools_enabled: false,
        tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
    };
    let telemetry = || {
        Arc::new(TelemetryClient::with_transport(
            "test",
            HardwareTier::Unknown,
            "https://127.0.0.1:1",
            Box::new(Offline),
        ))
    };
    // The 8 GB simulation constrains admission, not the host's physical RAM.
    let small = alloyd::boot::boot_qwen3_with_context_limit(
        request(Some(8 * 1024 * 1024 * 1024)),
        Some(40960),
    )
    .unwrap();
    let mut small_db = SessionStore::open(root.path().join("small.sqlite")).unwrap();
    small_db.create_session("small", &profile).unwrap();
    let small_supervisor = Supervisor::new(small.engine, telemetry());
    let result = small_supervisor.run_turn(
        &Mutex::new(small_db),
        "small",
        "Reply with exactly COBALT.",
        8,
    );
    // Either successful bounded inference or an explicit memory refusal is valid;
    // no execution failure, crash, fallback, or silent truncation is accepted.
    match result {
        Ok(outcome) => {
            assert!(outcome.assistant_text.contains("COBALT"));
            eprintln!("8gb_simulated={profile}: executed");
        }
        Err(error) => {
            let error = error.to_string();
            assert!(
                error.contains("bytes") && error.contains("available"),
                "{error}"
            );
            eprintln!("8gb_simulated={profile}: refused: {error}");
        }
    }
    drop(small_supervisor);
    let boot = alloyd::boot::boot_qwen3_with_context_limit(request(None), Some(40960)).unwrap();
    let mut db = SessionStore::open(root.path().join("long.sqlite")).unwrap();
    for session in ["long8", "long32", "cancel", "after_cancel"] {
        db.create_session(session, &profile).unwrap();
    }
    let store = Mutex::new(db);
    let supervisor = Supervisor::new(boot.engine, telemetry());
    mlx_ffi::memory::reset_peak_memory().unwrap();
    for (document, session, minimum) in [
        ("document-8k.txt", "long8", 7500),
        ("document-32k.txt", "long32", 32000),
    ] {
        let mut document = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../bench/corpus/fixtures/documents")
                .join(document),
        )
        .unwrap();
        let mut middle = document.len() / 2;
        while !document.is_char_boundary(middle) {
            middle += 1;
        }
        document.insert_str(
            middle,
            "\nARCHIVE RECORD: The vault access code is COBALT-7392.\n",
        );
        let prompt = format!(
            "Read this archive and retrieve the vault access code.\n{document}\nWhat is the vault access code in the ARCHIVE RECORD? Reply with only the code."
        );
        let snapshot =
            ContextBuilder::build_snapshot(session, &profile, &[], Some(&prompt)).unwrap();
        let rendered = ContextBuilder::render_and_count(supervisor.engine(), snapshot).unwrap();
        assert!(rendered.token_count() >= minimum);
        let mut streamed = String::new();
        let mut ended = false;
        let start = std::time::Instant::now();
        let outcome = supervisor
            .run_turn_streaming(
                &store,
                session,
                &prompt,
                32,
                |_| Ok(()),
                |event| {
                    match event {
                        GenerationEvent::TextDelta { text } => streamed.push_str(text),
                        GenerationEvent::GenerationCompleted { finish_reason, .. } => {
                            assert_eq!(*finish_reason, FinishReason::EndOfSequence);
                            ended = true;
                        }
                        GenerationEvent::RequestFailed { error, .. } => panic!("{error}"),
                        _ => {}
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(outcome.input_tokens, rendered.token_count());
        assert!(ended);
        assert_eq!(streamed, outcome.assistant_text);
        assert!(streamed.contains("COBALT-7392"), "{streamed}");
        for marker in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"] {
            assert!(!streamed.contains(marker));
        }
        eprintln!(
            "long_context={profile} tokens={} seconds={:.2} peak_mlx_bytes={} answer={streamed:?}",
            outcome.input_tokens,
            start.elapsed().as_secs_f64(),
            mlx_ffi::memory::peak_memory_bytes().unwrap()
        );
    }
    eprintln!("native_memory={:?}", mlx_ffi::memory::snapshot().unwrap());
    check_cancellation_recovery(&supervisor, &store, &profile);
}

fn check_cancellation_recovery<E: engine::api::Engine>(
    supervisor: &harness_core::Supervisor<E>,
    store: &std::sync::Mutex<harness_core::SessionStore>,
    profile: &str,
) {
    use engine::types::GenerationEvent;
    store
        .lock()
        .unwrap()
        .create_session("baseline", profile)
        .unwrap();
    let expected = supervisor
        .run_turn(store, "baseline", "Reply with exactly COBALT.", 16)
        .unwrap();
    assert!(!expected.assistant_text.is_empty());
    eprintln!("cancel_baseline={:?}", expected.assistant_text);
    let id = std::cell::RefCell::new(None);
    let mut cancelled = false;
    let result = supervisor.run_turn_streaming(
        store,
        "cancel",
        "Count from 1 to 1000, one integer per line.",
        256,
        |request| {
            *id.borrow_mut() = Some(request.clone());
            Ok(())
        },
        |event| {
            if matches!(event, GenerationEvent::TextDelta { .. }) && !cancelled {
                supervisor.cancel(id.borrow().clone().unwrap())?;
                cancelled = true;
            }
            Ok(())
        },
    );
    assert!(cancelled, "must cancel after an actual streamed token");
    assert!(result.is_err(), "cancellation must not complete normally");
    let error = result.err().unwrap().to_string();
    assert!(error.to_lowercase().contains("cancel"), "{error}");
    let resumed = supervisor
        .run_turn(store, "after_cancel", "Reply with exactly COBALT.", 16)
        .unwrap();
    eprintln!("cancel_following={:?}", resumed.assistant_text);
    assert_eq!(
        resumed.assistant_text, expected.assistant_text,
        "cancellation changed a fresh session's response"
    );
    eprintln!("stream_cancel={profile}: {error}; next request succeeded");
}

#[test]
#[ignore = "requires pinned local Llama artifact and native Metal"]
fn llama3_stream_cancel_recovery() {
    use harness_core::{SessionStore, Supervisor};
    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let root = tempfile::tempdir().unwrap();
    let boot = alloyd::boot::boot_qwen3_with_context_limit(
        BootRequest {
            artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
            profile_id: profile.clone(),
            corpus_models_json: None,
            memory_guard: MemoryGuardConfig::balanced(),
            admission_budget_override_bytes: None,
            prefer_metal: true,
            tools_enabled: false,
            tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
        },
        Some(4096),
    )
    .unwrap();
    let mut db = SessionStore::open(root.path().join("session.sqlite")).unwrap();
    for session in ["cancel", "after_cancel"] {
        db.create_session(session, &profile).unwrap();
    }
    let store = std::sync::Mutex::new(db);
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "test",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    check_cancellation_recovery(&Supervisor::new(boot.engine, telemetry), &store, &profile);
}
