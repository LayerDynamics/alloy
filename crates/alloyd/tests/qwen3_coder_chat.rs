//! Native Qwen3-Coder registration and publisher tokenizer/template parity.
use alloyd::BootRequest;
use engine::{
    memory_guard::MemoryGuardConfig,
    tiered_kv::{HotCacheLimit, TieredKvConfig},
    types::{ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId},
};
use harness_core::context::ContextBuilder;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROFILE: &str = "qwen3-coder-30b-a3b-instruct-4bit";

fn request(tools_enabled: bool) -> BootRequest {
    BootRequest {
        artifact_dir: PathBuf::from(std::env::var_os("ALLOY_QWEN3_CODER_ARTIFACT").unwrap()),
        profile_id: PROFILE.into(),
        corpus_models_json: None,
        memory_guard: MemoryGuardConfig::balanced(),
        admission_budget_override_bytes: None,
        prefer_metal: false,
        tools_enabled,
        tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
    }
}

#[test]
#[ignore = "requires pinned ALLOY_QWEN3_CODER_ARTIFACT"]
fn qwen3_coder_native_registration() {
    let inspect = |tools| {
        alloyd::boot::inspect_qualification_identity(
            &request(tools),
            Some(2048),
            &"1".repeat(40),
            &"2".repeat(64),
        )
        .unwrap()
        .0
    };
    let text = inspect(false);
    let tools = inspect(true);
    assert_eq!(text.family, "qwen3-coder");
    assert_eq!(
        text.tokenizer_sha256,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    );
    assert_ne!(text.tool_protocol_sha256, tools.tool_protocol_sha256);
}

#[test]
#[ignore = "requires pinned ALLOY_QWEN3_CODER_ARTIFACT"]
fn qwen3_coder_publisher_prompt_parity() {
    use engine::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    let boot = alloyd::boot::boot_native_for_qualification(request(true), Some(2048)).unwrap();
    let tool = RenderingTool::new(
        "fs.read".into(),
        "1".into(),
        "Read a file. ".into(),
        json!({"type":"object","properties":{"path":{"type":"string","enum":["a","雪"],"minLength":1}},"required":["path"]}),
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

    for (with_tools, blocks, count, bytes_hash, tokens_hash) in [
        (
            false,
            vec![],
            20,
            "7ede7a8a8bbe7ed3345ceff0e1494620fb597ef20003918676a462143af5a200",
            "0e6b07b5838c18aa24053c9a19c595ac1d8d78cf540a02493f83a9a655954c50",
        ),
        (
            true,
            vec![],
            280,
            "e1646b673f10dda09db212a562e8c4e23a4545f8ab141c27b4b49b61878001f9",
            "659e3e2e6d16bf2901498930efc52c661fd61261a50f92f76f4b0b6db78b8ff2",
        ),
        (
            true,
            history,
            377,
            "e4894fdc04de9ee7a8197372b04898a63e3aadc6b9f56fc4b8b53d1c617a3c6b",
            "b4d9bb5b7933cfd9ed38caa31e5410eac00d5a83f9b6de9ff125c7cb857a4329",
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
        .unwrap();
        let snapshot = if with_tools {
            snapshot.with_tool_context(context)
        } else {
            snapshot
        };
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
}
