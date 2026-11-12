//! Reproducible render/count measurement; see Phase 3 plan for boundary.
use engine::{
    rendering::{
        PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions,
        qwen3_projected_token_upper_bound, qwen3_projected_tool_token_upper_bound,
        render_and_count,
    },
    types::{
        ArtifactId, ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId,
        RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock,
    },
};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, time::Instant};

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let tokenizer_path = std::env::args()
        .nth(1)
        .ok_or("usage: context_32k TOKENIZER_JSON")?;
    let source = fs::read(tokenizer_path)?;
    let tokenizer_hash = digest(&source);
    assert_eq!(
        hex(&tokenizer_hash),
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer pin"
    );
    let tokenizer = PinnedTokenizer::from_pinned_json(&source, tokenizer_hash, source.len())?;
    for text in ["", "ASCII and spaces", "Ǖ\u{344}é中🦀", "<|im_start|>\n"] {
        let tokens = tokenizer
            .tokenizer()
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .len();
        let bound = qwen3_projected_token_upper_bound(&tokenizer, text.len())?;
        assert!(tokens <= bound, "projected bound failed for {text:?}");
    }
    let projection_profile = ModelProfileId::new("qwen3-projection")?;
    let projection_artifact = ArtifactId::new("qwen3-projection")?;
    let tool_limits = ToolContextLimits {
        max_tools: 4,
        max_history_blocks: 8,
        max_json_bytes: 65536,
    };
    let tool = RenderingTool::new(
        "fixture.read".into(),
        "1".into(),
        "Read fixture".into(),
        serde_json::json!({"type":"object"}),
    )?;
    let make_context = |content: String| -> Result<ContextSnapshot, Box<dyn Error>> {
        let history = vec![
            ToolHistoryBlock::AssistantCall {
                call_id: "call-1".into(),
                tool_id: "fixture.read".into(),
                arguments: serde_json::Map::new(),
            },
            ToolHistoryBlock::ToolResult {
                call_id: "call-1".into(),
                content,
                is_error: false,
            },
        ];
        Ok(ContextSnapshot::new(
            ContextSnapshotId::new("projection")?,
            projection_profile.clone(),
            vec![Message {
                role: MessageRole::User,
                content: "read fixture".into(),
            }],
        )?
        .with_tool_context(ToolContext::new(vec![tool.clone()], history, tool_limits)?))
    };
    let tool_template = qwen3::template::TOOL_CHAT_TEMPLATE;
    let tool_binding = RenderingBinding {
        profile_id: &projection_profile,
        artifact_id: &projection_artifact,
        template: tool_template,
        template_sha256: digest(tool_template.as_bytes()),
        artifact_manifest_sha256: [0; 32],
    };
    let tool_options = RenderingOptions {
        add_generation_prompt: true,
        enable_thinking: Some(false),
        add_special_tokens: false,
        keep_trailing_newline: false,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: &[
            "<|",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ],
    };
    let tool_render = |context: &ContextSnapshot| {
        render_and_count(
            context,
            &tool_binding,
            &tokenizer,
            &tool_options,
            RenderingLimits::new(16, 65536, 65536, 65536, 40960).unwrap(),
        )
    };
    let skeleton = tool_render(&make_context(String::new())?)?;
    for value in [
        serde_json::json!("simple"),
        serde_json::json!("quotes \" slashes \\\\ newline\nǕ\u{344}é中🦀"),
        serde_json::json!({"text":"x".repeat(4000)}),
        serde_json::json!("\u{0}\u{1}\u{1f}\\\"".repeat(128)),
        serde_json::json!({"call_id": format!("gen-{}-call-{}", "f".repeat(64), usize::MAX), "artifact_ref": format!("art1_{}", "f".repeat(32)), "is_error": false}),
    ] {
        let content = serde_json::to_string(&value)?;
        let actual = tool_render(&make_context(content.clone())?)?;
        assert!(
            actual.text().len() - skeleton.text().len() <= content.len() * 2,
            "serialized result exceeds projected rendered byte bound"
        );
        let upper =
            qwen3_projected_tool_token_upper_bound(&tokenizer, &skeleton, 1, content.len() * 2)?;
        assert!(
            actual.token_count() <= upper,
            "actual tokens exceed projected bound"
        );
    }
    // Exercise cumulative bounds for two projected results separated by an
    // actual assistant call, including Unicode and JSON escaping.
    let make_batch = |contents: [&str; 2]| -> Result<ContextSnapshot, Box<dyn Error>> {
        let mut history = Vec::new();
        for (index, content) in contents.into_iter().enumerate() {
            let call_id = format!("batch-{index}");
            history.push(ToolHistoryBlock::AssistantCall {
                call_id: call_id.clone(),
                tool_id: "fixture.read".into(),
                arguments: serde_json::Map::new(),
            });
            history.push(ToolHistoryBlock::ToolResult {
                call_id,
                content: content.into(),
                is_error: false,
            });
        }
        Ok(ContextSnapshot::new(
            ContextSnapshotId::new("batch-projection")?,
            projection_profile.clone(),
            vec![Message {
                role: MessageRole::User,
                content: "read both".into(),
            }],
        )?
        .with_tool_context(ToolContext::new(vec![tool.clone()], history, tool_limits)?))
    };
    let batch_skeleton = tool_render(&make_batch(["", ""])?)?;
    let first = serde_json::to_string(&serde_json::json!("Ǖ\u{344}\\\"".repeat(50)))?;
    let second = serde_json::to_string(&serde_json::json!({"result":"x".repeat(1200)}))?;
    let batch_actual = tool_render(&make_batch([&first, &second])?)?;
    let batch_upper = qwen3_projected_tool_token_upper_bound(
        &tokenizer,
        &batch_skeleton,
        2,
        (first.len() + second.len()) * 2,
    )?;
    assert!(
        batch_actual.token_count() <= batch_upper,
        "multi-result tokens exceed projected bound"
    );
    println!(
        "pinned_qwen_projected_tool_samples=5 status=passed skeleton_tokens={}",
        skeleton.token_count()
    );
    let document = fs::read("bench/corpus/fixtures/documents/document-32k.txt")?;
    assert_eq!(
        hex(&digest(&document)),
        "393bc22d01056d7e6f5fbba865071725867679aa00521f84780802c660cb0c75",
        "corpus pin"
    );
    let profile = ModelProfileId::new("qwen3-4b-4bit")?;
    let artifact = ArtifactId::new("qwen3-4b-4bit")?;
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("32k-measurement")?,
        profile.clone(),
        vec![Message {
            role: MessageRole::User,
            content: String::from_utf8(document)?,
        }],
    )?;
    let template = qwen3::template::CHAT_TEMPLATE;
    let binding = RenderingBinding {
        profile_id: &profile,
        artifact_id: &artifact,
        template,
        template_sha256: digest(template.as_bytes()),
        artifact_manifest_sha256: [0; 32],
    };
    let options = RenderingOptions {
        add_generation_prompt: true,
        enable_thinking: Some(false),
        add_special_tokens: false,
        keep_trailing_newline: false,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: &[
            "<|",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ],
    };
    let limits = RenderingLimits::new(8, 1_000_000, 100_000, 1_000_000, 65_536)?;
    let run = || render_and_count(&context, &binding, &tokenizer, &options, limits);
    let warm = run()?;
    assert!(
        warm.token_count() >= 32768 && warm.token_count() < 33000,
        "unexpected token count: {}",
        warm.token_count()
    );
    println!(
        "tokenizer_sha256={} corpus_sha256={} template_sha256={} rendered_tokens={} render_hash={}",
        hex(&tokenizer_hash),
        hex(&digest(context.messages()[0].content.as_bytes())),
        hex(&binding.template_sha256),
        warm.token_count(),
        hex(warm.render_hash())
    );
    let mut max = 0u128;
    for iteration in 0..10 {
        let start = Instant::now();
        let rendered = run()?;
        let micros = start.elapsed().as_micros();
        assert_eq!(rendered.token_ids(), warm.token_ids());
        max = max.max(micros);
        println!("iteration={iteration} elapsed_us={micros}");
    }
    println!(
        "max_elapsed_us={max} boundary=render_and_count_including_hash_and_tokenization_excluding_artifact_load"
    );
    assert!(max < 150_000, "32k render/count exceeds 150 ms");
    Ok(())
}
