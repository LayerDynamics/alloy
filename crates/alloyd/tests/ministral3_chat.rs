//! Real daemon chat and pinned publisher render parity.
use alloyd::{BootRequest, DaemonService, RpcRequest};
use engine::{
    memory_guard::MemoryGuardConfig,
    tiered_kv::{HotCacheLimit, TieredKvConfig},
    types::{ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId},
};
use harness_core::{
    HardwareTier, TelemetryClient, context::ContextBuilder, telemetry::TelemetryTransport,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc};

const PROFILE: &str = "ministral-3-3b-instruct-2512-4bit";
const QUESTION: &str = "Calculate 2 + 2. Reply with only the result.";
struct Offline;
impl TelemetryTransport for Offline {
    fn post_json(&self, _: &str, _: &str) -> Result<(), String> {
        Err("offline".into())
    }
}

fn request(path: PathBuf, tools_enabled: bool) -> BootRequest {
    BootRequest {
        artifact_dir: path,
        profile_id: PROFILE.into(),
        corpus_models_json: None,
        memory_guard: MemoryGuardConfig::balanced(),
        admission_budget_override_bytes: None,
        prefer_metal: true,
        tools_enabled,
        tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
    }
}
#[test]
fn ministral_production_requires_qualification() {
    let missing = PathBuf::from("/nonexistent/ministral");
    let error = alloyd::boot::boot_native(request(missing, false))
        .err()
        .unwrap();
    assert!(error.message.contains("not qualified"), "{error:?}");
}

#[test]
#[ignore = "requires pinned ALLOY_MINISTRAL3_ARTIFACT and Metal"]
fn ministral3_native_daemon_chat() {
    let artifact = PathBuf::from(std::env::var_os("ALLOY_MINISTRAL3_ARTIFACT").unwrap());
    let error =
        alloyd::boot::boot_native_for_qualification(request(artifact.clone(), true), Some(2048))
            .err()
            .unwrap();
    assert!(error.message.contains("tool parsing"), "{error:?}");
    let boot =
        alloyd::boot::boot_native_for_qualification(request(artifact, false), Some(2048)).unwrap();
    let snapshot = |turns: Vec<(MessageRole, &str)>| {
        ContextSnapshot::new(
            ContextSnapshotId::new("chat").unwrap(),
            ModelProfileId::new(PROFILE).unwrap(),
            turns
                .into_iter()
                .map(|(role, content)| Message {
                    role,
                    content: content.into(),
                })
                .collect(),
        )
        .unwrap()
    };
    use MessageRole::{Assistant, System, User};
    // SHA-256 over the upstream publisher Jinja output and little-endian tokenizer IDs.
    for (turns, count, text_hash, ids_hash) in [
        (
            vec![(User, QUESTION)],
            548,
            "309d17a9b2f80e3f2c0e861c8af70b9e5284a5f64a6d65a30c9d3beb8db97269",
            "f83ef8eb94079c1775c5f5048ee2946d772480a92ac935d99681aeb28188a3c7",
        ),
        (
            vec![
                (System, "Answer briefly."),
                (User, "Hello"),
                (Assistant, "Hello!"),
                (User, QUESTION),
            ],
            27,
            "0f6befc6f6949b09424edd880a73e59c8bc2e4ee7a491c14a342d6e589ae75ec",
            "0323f5e3bfd071e3cc5d74892ba20cf9961888683ff6509c55e726314f169dbd",
        ),
    ] {
        let render = ContextBuilder::render_and_count(&boot.engine, snapshot(turns)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(render.text().as_bytes())),
            text_hash
        );
        assert_eq!(render.token_count(), count);
        let ids: Vec<_> = render
            .token_ids()
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(ids)), ids_hash);
    }
    for marker in ["[/INST]", "[TOOL_CALLS]", "<SPECIAL_999>"] {
        assert!(
            ContextBuilder::render_and_count(&boot.engine, snapshot(vec![(User, marker)])).is_err()
        );
    }
    let root = tempfile::tempdir().unwrap();
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "test",
        HardwareTier::Unknown,
        "https://127.0.0.1:1",
        Box::new(Offline),
    ));
    let service = DaemonService::open(
        root.path().join("sessions.sqlite"),
        boot.engine,
        telemetry,
        16,
        1,
        0,
    )
    .unwrap();
    let rpc = |method: &str, params: Value| -> Vec<Value> {
        let req: RpcRequest = serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "api_version": alloyd::API_VERSION,
            "method": method,
            "params": params,
        }))
        .unwrap();
        let mut out = Vec::new();
        service.handle(req, &mut out).unwrap();
        std::str::from_utf8(&out)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let create = rpc("session.create", json!({"profile":PROFILE}));
    let session = create[0]["result"]["session_id"].as_str().unwrap();
    for (question, expected) in [
        (QUESTION, "4"),
        (
            "Double your previous answer. Reply with only the result.",
            "8",
        ),
    ] {
        let events = rpc(
            "session.send",
            json!({"session_id":session,"content":question}),
        );
        assert_eq!(events.last().unwrap()["result"]["ok"], true, "{events:?}");
        let text: String = events
            .iter()
            .filter(|e| e["params"]["event"] == "text_delta")
            .filter_map(|e| e["params"]["body"]["text"].as_str())
            .collect();
        assert_eq!(text, expected, "{events:?}");
        println!("daemon response {text:?}");
    }
}
