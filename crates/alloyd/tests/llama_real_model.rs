//! Real shared native conformance. Run each frozen profile with ALLOY_REAL_MODEL/PROFILE.
#[path = "support/block_on.rs"]
mod executor;
#[path = "support/recording_engine.rs"]
mod recording_engine;
use alloyd::{BootRequest, DaemonService, RpcRequest};
use engine::{
    memory_guard::MemoryGuardConfig,
    tiered_kv::{HotCacheLimit, TieredKvConfig},
};
use harness_core::{HardwareTier, TelemetryClient, telemetry::TelemetryTransport};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc};
struct Offline;

fn gemma3_long_document() -> (String, Value) {
    use sha2::{Digest, Sha256};
    let source = include_str!("../../../bench/corpus/fixtures/documents/document-32k.txt");
    let document = include_str!("fixtures/gemma3_long_context/document.txt");
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/gemma3_long_context/manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        manifest["source_sha256"].as_str().unwrap()
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(document.as_bytes())),
        manifest["fixture_sha256"].as_str().unwrap()
    );
    assert!(source.starts_with(document));
    assert!(document.ends_with('\n'));
    assert_eq!(document.len() as u64, manifest["fixture_bytes"]);
    assert!((32000..=32100).contains(&manifest["prompt_tokens"].as_u64().unwrap()));
    assert!(
        manifest["prompt_tokens"].as_u64().unwrap()
            + manifest["output_reserve"].as_u64().unwrap()
            + manifest["safety_margin"].as_u64().unwrap()
            <= manifest["context_limit"].as_u64().unwrap()
    );
    (document.to_owned(), manifest)
}

#[test]
fn gemma3_long_context_fixture_integrity() {
    gemma3_long_document();
}

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
        "llama-3.1-8b-4bit"
            | "llama-3.1-8b-8bit"
            | "ministral-3-3b-instruct-2512-4bit"
            | "ministral-3-3b-instruct-2512-8bit"
            | "smollm3-3b-4bit"
            | "phi4-mini-instruct-4bit"
            | "gemma-3-4b-4bit"
            | "gemma-3-12b-4bit"
            | "gemma-4-e4b-it-4bit"
            | "ornith-1.5-9b-4bit"
            | "qwen3.5-0.8b-4bit"
            | "qwen3-4b-4bit"
            | "qwen3-14b-4bit"
            | "qwen3-coder-30b-a3b-instruct-4bit"
    ));
    let root = tempfile::tempdir().unwrap();
    let boot = alloyd::boot::boot_native_for_qualification(
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
        recording_engine::RecordingEngine::new(boot.engine),
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
        json!({"session_id":session,"content":"Use fs.write exactly once to write marker.txt with exactly the content COBALT. After the tool succeeds, reply with exactly the four uppercase letters DONE, without punctuation, whitespace, or any other text. Do not use any other tool."}),
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
    let proposals: Vec<_> = events
        .iter()
        .filter(|e| e["params"]["event"] == "tool_requested")
        .map(|e| &e["params"]["body"])
        .collect();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0]["tool_id"], "fs.write");
    assert_eq!(
        proposals[0]["arguments"],
        json!({"path":"marker.txt","content":"COBALT"})
    );
    assert_eq!(matching(&events, "tool_call_started"), 1);
    assert!(
        events
            .iter()
            .any(|e| e["params"]["event"] == "tool_result"
                && e["params"]["body"]["is_error"] == false)
    );
    let first_result = events
        .iter()
        .position(|e| e["params"]["event"] == "tool_result")
        .unwrap();
    assert!(
        events[..first_result].iter().all(|e| {
            e["params"]["event"] != "text_delta"
                || e["params"]["body"]["text"]
                    .as_str()
                    .is_some_and(|text| text.trim().is_empty())
        }),
        "assistant must wait for the actual result before answering"
    );
    assert!(
        events
            .iter()
            .any(|e| e["params"]["event"] == "generation_completed"
                && e["params"]["body"]["finish_reason"] == "end_of_sequence")
    );
    assert_eq!(events.last().unwrap()["result"]["assistant_text"], "DONE");
    let caps = rpc("models.capabilities", json!({"artifact_id":profile}));
    let dense_qwen = matches!(profile.as_str(), "qwen3-4b-4bit" | "qwen3-14b-4bit");
    let coder = profile == "qwen3-coder-30b-a3b-instruct-4bit";
    assert_eq!(
        caps[0]["result"]["family"],
        if profile.starts_with("ornith") || profile.starts_with("qwen3.5") {
            "qwen35"
        } else if coder {
            "qwen3-coder"
        } else if dense_qwen {
            "qwen3"
        } else if profile.starts_with("ministral-3") {
            "ministral3"
        } else if profile.starts_with("smollm") {
            "smollm3"
        } else if profile.starts_with("phi4") {
            "phi"
        } else if profile.starts_with("gemma-4") {
            "gemma4"
        } else if profile.starts_with("gemma-3") {
            "gemma3"
        } else {
            "llama3"
        }
    );
    assert_eq!(caps[0]["result"]["backend"]["kind"], "mlx_metal");
    assert_eq!(
        caps[0]["result"]["weight_representations"],
        if dense_qwen {
            json!([{"kind":"f32"},{"kind":"affine4","group_size":64}])
        } else if coder {
            json!([{"kind":"affine4","group_size":64},{"kind":"affine8","group_size":64}])
        } else {
            json!([{"kind":if profile.ends_with("8bit") {"affine8"} else {"affine4"}, "group_size":64}])
        }
    );
    assert_eq!(
        caps[0]["result"]["cache_modes"][0]["representation"],
        if profile.starts_with("ornith") || profile.starts_with("qwen3.5") {
            "qwen35.mlx-f32.full-kv-and-recurrent.v1"
        } else if coder {
            "qwen3-coder.mlx-f32.full.bhtd.v1"
        } else if dense_qwen {
            "qwen3.mlx-f32.full.bhtd.v1"
        } else if profile.starts_with("ministral-3") {
            "ministral3.mlx-f32.full.bhtd.v1"
        } else if profile.starts_with("smollm") {
            "smollm3.mlx-f32.full.bhtd.v1"
        } else if profile.starts_with("phi4") {
            "phi.mlx-f32.full.bhtd.su-long-fixed.v1"
        } else if profile.starts_with("gemma-4") {
            "gemma4.mlx-f32.shared-full-and-sliding.v1"
        } else if profile.starts_with("gemma-3") {
            "gemma3.mlx-f32.full-and-sliding.v1"
        } else {
            "llama3.mlx-f32.full.bhtd.v1"
        }
    );
    if profile.starts_with("gemma-") {
        assert_eq!(caps[0]["result"]["cache_modes"][0]["kind"], "mixed");
        assert_eq!(caps[0]["result"]["supports_prefix_reuse"], false);
    }
    let denied = rpc(
        "session.send",
        json!({"session_id":session,"content":"Now use fs.write to write forbidden.txt with content BLOCKED."}),
    );
    eprintln!("denied_events={}", serde_json::to_string(&denied).unwrap());
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
    let rejected_answer = denied.last().unwrap()["result"]["assistant_text"]
        .as_str()
        .unwrap();
    let lower = rejected_answer.to_ascii_lowercase();
    assert!(
        ["reject", "denied", "not allowed", "not permitted", "policy"]
            .iter()
            .any(|word| lower.contains(word)),
        "missing rejection explanation: {rejected_answer:?}"
    );
    assert_ne!(
        rejected_answer.trim(),
        "DONE",
        "rejection cannot be reported as success"
    );
    assert!(
        denied
            .iter()
            .all(|e| e.get("error").is_none() && e["params"]["event"] != "request_failed")
    );
}

#[test]
#[ignore = "requires pinned local Llama artifact and native Metal"]
fn llama3_long_context_stream_cancel_memory() {
    use engine::api::Engine;
    use engine::types::{FinishReason, GenerationEvent};
    use harness_core::{ContextBuilder, SessionStore, Supervisor};
    use std::sync::Mutex;
    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let context_limit = if profile.starts_with("gemma-3") {
        32768
    } else {
        40960
    };
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
    let small = alloyd::boot::boot_native_for_qualification(
        request(Some(8 * 1024 * 1024 * 1024)),
        Some(context_limit),
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
    let boot =
        alloyd::boot::boot_native_for_qualification(request(None), Some(context_limit)).unwrap();
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
        let gemma_long = profile.starts_with("gemma-3") && session == "long32";
        let (mut document, manifest) = if gemma_long {
            let (document, manifest) = gemma3_long_document();
            (document, Some(manifest))
        } else {
            (
                std::fs::read_to_string(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../../bench/corpus/fixtures/documents")
                        .join(document),
                )
                .unwrap(),
                None,
            )
        };
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
        if let Some(manifest) = manifest {
            let budget = supervisor
                .engine()
                .profile_context_budget(&engine::types::ModelProfileId::new(&profile).unwrap())
                .unwrap()
                .unwrap();
            assert_eq!(rendered.token_count() as u64, manifest["prompt_tokens"]);
            assert_eq!(budget.limit as u64, manifest["context_limit"]);
            assert_eq!(budget.output_reserve as u64, manifest["output_reserve"]);
            assert_eq!(budget.safety_margin as u64, manifest["safety_margin"]);
            assert!(
                rendered.token_count() + budget.output_reserve + budget.safety_margin
                    <= budget.limit
            );
        }
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
        for marker in [
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end_of_text|>",
            "<|im_start|>",
            "<|im_end|>",
        ] {
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
    let boot = alloyd::boot::boot_native_for_qualification(
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

fn boundary_message(profile: &str, excess: usize) -> (String, &'static str, &'static str) {
    // Publisher-rendered chat bytes and little-endian token IDs.
    let (lines, padding_tokens, text_sha, ids_sha) = match profile {
        "gemma-4-e4b-it-4bit" => (
            660,
            42,
            "0eedf262bb67cf1171840e996f7761308d6b0df4f399505cbe9952446431591b",
            "b6c4d99312044c6eaf657b3f1351afb2821aec77e78d8feb74882f67c36c17e2",
        ),
        "ornith-1.5-9b-4bit" | "qwen3.5-0.8b-4bit" => (
            587,
            92,
            "d39c26f5e515d28d809b4e3e2f9420e440d813a69cad7a9af357991f5c7e8625",
            "c480eec4cba2bc348f1d5d70169695400705f589ec4c943ad38ee4b94909f593",
        ),
        "gemma-3-4b-4bit" | "gemma-3-12b-4bit" => (
            22,
            21,
            "a340a1c63c01b2c52d37b6f0e3cb5a24e782319503853ea6018ebf1731fd197c",
            "5ec31a1953f782c233a7c0f89f073645792dc2748fa5d170bf69d9fca2739f3d",
        ),
        "phi4-mini-instruct-4bit" => (
            665,
            25,
            "3128ae4f04ae795e59a1d791d47ea9718d21943c803a95ac85c176cd3f0016a2",
            "5fef375ea405c5ed0f8a2d9ab32b163bec60a9c26f98571acc09daaacbf2a32a",
        ),
        "ministral-3-3b-instruct-2512-4bit" | "ministral-3-3b-instruct-2512-8bit" => (
            538,
            29,
            "6094342011ee1d2c9a858a8506f0535a3ae87aeef9accdd4238ced660a01e5aa",
            "065053bd8112d87f6a4a07419d5164ae93430d9c6928ec21d051940e6054581c",
        ),
        "llama-3.1-8b-4bit" | "llama-3.1-8b-8bit" => (
            653,
            38,
            "5f93c485c402ef82875d84d57fbd56cf8a179101121828845eb060bd24c18ed1",
            "89098b6e1d42b4ea5d950e06715da31ef2bc0ccefd508ad7d982538420b7cb82",
        ),
        "smollm3-3b-4bit" => (
            648,
            48,
            "42b3913c4ae6ef8f17a115ccf831d00a8d45d5b078dd499ef51eb4efc2ccc02d",
            "49470014a7f7ab37e6d3809d885edae59292ddc3eabf82966f6fa5ed0d46116f",
        ),
        "qwen3-4b-4bit" | "qwen3-14b-4bit" => (
            636,
            34,
            "d68b3de817e64d190042f25f6565140a7f3993c9dd9dead229c42fbc760ce2ac",
            "e87efc68c9e5b75ba52d26a02d9c46ea8c4baf544d1c4072591ffe3f74dda7e2",
        ),
        "qwen3-coder-30b-a3b-instruct-4bit" => (
            636,
            38,
            "56d8c9e83b6ef1a1ee04cea1078a5cb2d108e5c8b4ba62423329c8b4b12f3b8b",
            "2b3d8b4bee9969632dd979569ed9fe2cd1dd6eb1ddaf763ae5bd3849c0f1b352",
        ),
        other => panic!("no pinned boundary input for {other}"),
    };
    let mut archive = if profile.starts_with("gemma-3-") {
        gemma3_long_document().0
    } else {
        include_str!("../../../bench/corpus/fixtures/documents/document-32k.txt").to_owned()
    };
    let mut middle = archive.len() / 2;
    while !archive.is_char_boundary(middle) {
        middle += 1;
    }
    archive.insert_str(
        middle,
        "\nARCHIVE RECORD: The vault access code is COBALT-7392.\n",
    );
    archive.push('\n');
    archive.extend(
        include_str!("../../../bench/corpus/fixtures/documents/document-8k.txt")
            .split_inclusive('\n')
            .take(lines),
    );
    let padding = if profile.starts_with("ministral-") {
        " entry"
    } else {
        " ledger"
    };
    let message = format!(
        "Read this archive and retrieve the vault access code.\n{archive}\n{}\nWhat is the vault access code in the ARCHIVE RECORD? Reply with only the code.",
        padding.repeat(padding_tokens + excess)
    );
    (message, text_sha, ids_sha)
}

fn boundary_context_limit(profile: &str) -> usize {
    if profile.starts_with("gemma-3-") {
        32_768
    } else {
        40_960
    }
}

fn boundary_boot(profile: &str, context_limit: usize, prefer_metal: bool) -> alloyd::BootedEngine {
    alloyd::boot::boot_native_for_qualification(
        BootRequest {
            artifact_dir: PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").unwrap()),
            profile_id: profile.to_owned(),
            corpus_models_json: None,
            memory_guard: MemoryGuardConfig::balanced(),
            admission_budget_override_bytes: None,
            prefer_metal,
            tools_enabled: true,
            tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
        },
        Some(context_limit),
    )
    .unwrap()
}

fn check_boundary_hashes(render: &engine::types::RenderedContext, text_sha: &str, ids_sha: &str) {
    use sha2::{Digest, Sha256};
    assert_eq!(
        format!("{:x}", Sha256::digest(render.text().as_bytes())),
        text_sha
    );
    let mut hash = Sha256::new();
    for id in render.token_ids() {
        hash.update(id.to_le_bytes());
    }
    assert_eq!(format!("{:x}", hash.finalize()), ids_sha);
}

#[test]
#[ignore = "requires a pinned native model tokenizer"]
fn native_boundary_prompt_parity() {
    use harness_core::ContextBuilder;
    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let limit = boundary_context_limit(&profile);
    let boot = boundary_boot(&profile, limit, false);
    for excess in [0, 1] {
        let (message, text_sha, ids_sha) = boundary_message(&profile, excess);
        let snapshot =
            ContextBuilder::build_snapshot("boundary", &profile, &[], Some(&message)).unwrap();
        let render = ContextBuilder::render_and_count(&boot.engine, snapshot).unwrap();
        assert_eq!(render.token_count() + 512 + 64, limit + excess);
        if excess == 0 {
            check_boundary_hashes(&render, text_sha, ids_sha);
        }
    }
}

#[test]
#[ignore = "requires the pinned profile and physical host memory policy"]
fn native_boundary_memory_estimate() {
    use engine::{api::Engine, types::ModelProfileId};
    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let limit = boundary_context_limit(&profile);
    let boot = boundary_boot(&profile, limit, true);
    let input_tokens = limit - 512 - 64;
    let estimate = executor::block_on(
        boot.engine
            .estimate_memory(ModelProfileId::new(&profile).unwrap(), input_tokens),
    )
    .unwrap();
    println!(
        "profile={profile} boundary_input={input_tokens} required={} available={} breakdown={:?}",
        estimate.reserved_bytes(),
        estimate.available_bytes(),
        estimate.breakdown()
    );
    assert!(
        estimate.fits(),
        "advertised boundary exceeds the actual admission budget"
    );
}

#[test]
#[ignore = "requires the pinned profile checkpoint and native Metal"]
fn native_validated_context_boundary() {
    use engine::api::Engine;
    use engine::types::ModelProfileId;
    use harness_core::{ContextBuilder, HarnessError, SessionStore, Supervisor};
    use std::sync::Mutex;

    let profile = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let context_limit = boundary_context_limit(&profile);
    let boot = boundary_boot(&profile, context_limit, true);
    let budget = boot
        .engine
        .profile_context_budget(&ModelProfileId::new(&profile).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(
        (budget.limit, budget.output_reserve, budget.safety_margin),
        (context_limit, 512, 64)
    );
    let supervisor = Supervisor::new(
        recording_engine::RecordingEngine::new(boot.engine),
        Arc::new(TelemetryClient::with_transport(
            "test",
            HardwareTier::Unknown,
            "https://127.0.0.1:1",
            Box::new(Offline),
        )),
    );
    let root = tempfile::tempdir().unwrap();
    let mut db = SessionStore::open(root.path().join("boundary.sqlite")).unwrap();
    for id in ["overflow", "boundary"] {
        db.create_session(id, &profile).unwrap();
    }
    let store = Mutex::new(db);
    for (id, excess) in [("overflow", 1), ("boundary", 0)] {
        let (message, text_sha, ids_sha) = boundary_message(&profile, excess);
        let snapshot = ContextBuilder::build_snapshot(id, &profile, &[], Some(&message)).unwrap();
        let render = ContextBuilder::render_and_count(supervisor.engine(), snapshot).unwrap();
        assert_eq!(
            render.token_count() + budget.output_reserve + budget.safety_margin,
            budget.limit + excess
        );
        let result = supervisor.run_turn(&store, id, &message, 32);
        let requests = supervisor.engine().requests.lock().unwrap();
        if excess == 1 {
            let Err(HarnessError::Invalid(reason)) = result else {
                panic!("expected overflow: {result:?}")
            };
            assert!(
                reason.contains("no completed history fits a checkpoint"),
                "{reason}"
            );
            assert!(
                requests.is_empty(),
                "oversized fresh turn must never enter native generation"
            );
            println!(
                "profile={profile} refused_tokens={} reason={reason}",
                render.token_count()
            );
        } else {
            let outcome = result.unwrap();
            assert_eq!(outcome.input_tokens, render.token_count());
            assert_eq!(requests.len(), 1);
            let actual = requests[0].0.expected_render().unwrap();
            assert_eq!(actual.token_ids(), render.token_ids());
            check_boundary_hashes(actual, text_sha, ids_sha);
            assert_eq!(outcome.assistant_text, "COBALT-7392");
            println!(
                "profile={profile} context={} input_tokens={} output_reserve={} safety_margin={} answer={:?}",
                budget.limit,
                outcome.input_tokens,
                budget.output_reserve,
                budget.safety_margin,
                outcome.assistant_text
            );
        }
    }
}
