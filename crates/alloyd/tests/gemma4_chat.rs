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

const PROFILE: &str = "gemma-4-e4b-it-4bit";
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
fn gemma4_production_requires_qualification() {
    let request = request(PathBuf::from("/nonexistent/gemma4"), false);
    let error = alloyd::boot::boot_native(request).err().unwrap();
    assert!(error.message.contains("not qualified"), "{error:?}");
}

#[test]
#[ignore = "requires pinned ALLOY_GEMMA4_ARTIFACT and Metal"]
fn gemma4_daemon_text_session() {
    let artifact = PathBuf::from(std::env::var_os("ALLOY_GEMMA4_ARTIFACT").unwrap());
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
            22,
            "5153adbd672f41dfec4b9cda638d7fca83d88bfa4358cdc0b5d73d54cca18a10",
            "a2c477de413ce5fa50196469d1603488fd7b578a9d417ca9d7dbc72adf170359",
        ),
        (
            vec![
                (System, "Answer briefly."),
                (User, "Hello"),
                (Assistant, "Hello!"),
                (User, QUESTION),
            ],
            43,
            "5bfc9a5dab161ed7619d08242704862196231736705b908de86a58ae55104220",
            "fce4aff29db09badadcc653ded6d928cd532f1a6af8b658b1aea10f09609a130",
        ),
        (
            vec![(User, "Saved context."), (User, QUESTION)],
            30,
            "b3d73b7e767543e518e90ef65738008decb467ec3adcb99e79d683a9479f3dff",
            "ab80c2d49d4555e87111adbe1138fc27c2191f36b0a01e57fae011cce5c3ce80",
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
    for marker in ["<turn|>", "<|tool_call>", "<|think|>", "<bos>", "<audio|>"] {
        assert!(
            ContextBuilder::render_and_count(&boot.engine, snapshot(vec![(User, marker)])).is_err()
        );
    }
    ContextBuilder::render_and_count(
        &boot.engine,
        snapshot(vec![(User, "Explain the |> operator.")]),
    )
    .unwrap();
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

#[test]
#[ignore = "requires pinned ALLOY_GEMMA4_ARTIFACT and Metal"]
fn gemma4_publisher_tool_prompt_parity() {
    use engine::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    let artifact = PathBuf::from(std::env::var_os("ALLOY_GEMMA4_ARTIFACT").unwrap());
    let boot =
        alloyd::boot::boot_native_for_qualification(request(artifact, true), Some(2048)).unwrap();
    let tool = RenderingTool::new(
        "fs.read".into(),
        "1".into(),
        "Read a file.".into(),
        json!({"type":"object","properties":{"path":{"type":"string","description":"Relative path."},"flags":{"type":"array","items":{"type":"string","enum":["a","b"]}}},"required":["path"]}),
    )
    .unwrap();
    let mut history = Vec::new();
    for (id, path) in [("call1", "a"), ("call2", "雪")] {
        history.push(ToolHistoryBlock::AssistantCall {
            call_id: id.into(),
            tool_id: "fs.read".into(),
            arguments: serde_json::Map::from_iter([("path".into(), json!(path))]),
        });
    }
    for (id, content) in [("call1", "first"), ("call2", "second")] {
        history.push(ToolHistoryBlock::ToolResult {
            call_id: id.into(),
            content: content.into(),
            is_error: false,
        });
    }

    for (blocks, count, bytes_hash, tokens_hash) in [
        (
            vec![],
            126,
            "a707fa051be45a1c430fb4651f814cd413b08307f4a0a121a29e34357660b36b",
            "862527a50edfad326105968a4dc1082e4bb38a526062fa23a90dc9b81eacd542",
        ),
        (
            history,
            212,
            "73331079c94b97e8ca23c70ab3ec27297bf8d2de14e3a6a3cd6a7cee47bd2f6e",
            "5e00843425aae905f07e764850951b90ad8c72ecfc5609089452b69f3427394b",
        ),
    ] {
        let context = ToolContext::new(
            vec![tool.clone()],
            blocks,
            ToolContextLimits {
                max_tools: 1,
                max_history_blocks: 4,
                max_json_bytes: 4096,
            },
        )
        .unwrap();
        let snapshot = ContextSnapshot::new(
            ContextSnapshotId::new("tool-parity").unwrap(),
            ModelProfileId::new(PROFILE).unwrap(),
            vec![
                Message {
                    role: MessageRole::System,
                    content: "Answer briefly.".into(),
                },
                Message {
                    role: MessageRole::User,
                    content: "Read both files.".into(),
                },
            ],
        )
        .unwrap()
        .with_tool_context(context);
        let render = ContextBuilder::render_and_count(&boot.engine, snapshot).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(render.text().as_bytes())),
            bytes_hash,
            "{}",
            render.text()
        );
        assert_eq!(render.token_count(), count);
        let ids: Vec<_> = render
            .token_ids()
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(ids)), tokens_hash);
    }
    // The protocol's string delimiter contains a quote that JSON serialization escapes.
    // Reject it before serialization, including when it occurs in untrusted result data.
    let unsafe_context = ToolContext::new(
        vec![tool],
        vec![
            ToolHistoryBlock::AssistantCall {
                call_id: "c".into(),
                tool_id: "fs.read".into(),
                arguments: Default::default(),
            },
            ToolHistoryBlock::ToolResult {
                call_id: "c".into(),
                content: r#"<|"|>"#.into(),
                is_error: false,
            },
        ],
        ToolContextLimits {
            max_tools: 1,
            max_history_blocks: 2,
            max_json_bytes: 4096,
        },
    )
    .unwrap();
    let snapshot = ContextSnapshot::new(
        ContextSnapshotId::new("quote-control").unwrap(),
        ModelProfileId::new(PROFILE).unwrap(),
        vec![Message {
            role: MessageRole::User,
            content: "Read a file.".into(),
        }],
    )
    .unwrap()
    .with_tool_context(unsafe_context);
    let error = ContextBuilder::render_and_count(&boot.engine, snapshot).unwrap_err();
    assert!(
        error.to_string().contains("reserved structural syntax"),
        "{error}"
    );
}
