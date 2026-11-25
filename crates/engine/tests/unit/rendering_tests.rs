use super::*;
use crate::{
    error::ErrorKind,
    types::{ContextSnapshotId, Message, MessageRole},
};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit};

#[test]
fn coder_nested_nul_matches_python_repr() {
    // Python str() uses repr() for strings nested in containers. NUL must be
    // spelled as \\x00 there, while a top-level string remains unchanged.
    for (input, expected) in [
        (serde_json::json!(["\0"]), "['\\x00']"),
        (
            serde_json::json!({"\0": ["a\0b"]}),
            "{'\\x00': ['a\\x00b']}",
        ),
        (serde_json::json!("a\0b"), "a\0b"),
    ] {
        assert_eq!(
            hf_python_string(minijinja::Value::from_serialize(input)).unwrap(),
            expected
        );
    }
}

fn limits() -> RenderingLimits {
    RenderingLimits::new(16, 4096, 4096, 4096, 128).expect("limits")
}

fn options() -> RenderingOptions<'static> {
    RenderingOptions {
        add_generation_prompt: false,
        enable_thinking: None,
        add_special_tokens: false,
        keep_trailing_newline: true,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: &[],
    }
}

fn profile() -> ModelProfileId {
    ModelProfileId::new("render-test").expect("profile")
}

fn artifact() -> ArtifactId {
    ArtifactId::new("render-test-artifact").expect("artifact")
}

fn snapshot(content: &str) -> ContextSnapshot {
    ContextSnapshot::new(
        ContextSnapshotId::new("snapshot-a").expect("snapshot"),
        profile(),
        vec![Message {
            role: MessageRole::User,
            content: content.to_owned(),
        }],
    )
    .expect("context")
}

fn binding<'a>(
    profile: &'a ModelProfileId,
    artifact: &'a ArtifactId,
    template: &'a str,
) -> RenderingBinding<'a> {
    // A deterministic binding fixture for renderer tests, not proof
    // that a production model artifact has been verified.
    RenderingBinding {
        profile_id: profile,
        artifact_id: artifact,
        template,
        template_sha256: sha256(template.as_bytes()),
        artifact_manifest_sha256: [7; 32],
    }
}

fn pinned_tokenizer() -> PinnedTokenizer {
    // Exercise the actual Hugging Face tokenizer implementation.
    // This small vocabulary is a tokenizer-unit-test fixture, not
    // a model execution implementation.
    let model = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".to_owned(), 0),
                ("hello".to_owned(), 1),
                ("world".to_owned(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".to_owned())
        .build()
        .expect("word-level model");

    let mut tokenizer = Tokenizer::new(model);

    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));

    let source = tokenizer.to_string(false).expect("tokenizer JSON");

    PinnedTokenizer::from_pinned_json(source.as_bytes(), sha256(source.as_bytes()), 16 * 1024)
        .expect("pinned tokenizer")
}

#[test]
fn canonical_prefix_requires_matching_lossless_tokenizer_roundtrip() {
    let tokenizer = pinned_tokenizer();
    verify_token_prefix_bytes(&tokenizer, &[1, 2], b"hello world").expect("matching prefix");
    assert!(verify_token_prefix_bytes(&tokenizer, &[1, 2], b"hello hello").is_err());
    assert!(verify_token_prefix_bytes(&tokenizer, &[1, 2], &[0xff]).is_err());
    // Hugging Face decode skips an unknown ID, so even equal decoded bytes
    // must be rejected when encoding cannot recover the executed sequence.
    assert!(verify_token_prefix_bytes(&tokenizer, &[999], b"").is_err());
}

#[test]
#[ignore = "requires the pinned local Qwen3-4B tokenizer"]
fn canonical_prefix_real_qwen_prompt_and_special_tokens_roundtrip() {
    let directory = std::env::var_os("ALLOY_REAL_MODEL").expect("set ALLOY_REAL_MODEL");
    let source = std::fs::read(std::path::Path::new(&directory).join("tokenizer.json"))
        .expect("read Qwen tokenizer");
    let expected = [
        0xae, 0xb1, 0x33, 0x07, 0xa7, 0x1a, 0xcd, 0x8f, 0xe8, 0x18, 0x61, 0xd9, 0x4a, 0xd5, 0x4a,
        0xb6, 0x89, 0xdf, 0x77, 0x33, 0x18, 0x80, 0x9e, 0xed, 0x3c, 0xbe, 0x79, 0x4b, 0x44, 0x92,
        0xda, 0xe4,
    ];
    let tokenizer = PinnedTokenizer::from_pinned_json(&source, expected, 16 * 1024 * 1024)
        .expect("verified Qwen tokenizer");
    let text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>";
    let encoding = tokenizer
        .tokenizer()
        .encode(text, false)
        .expect("encode prompt");
    verify_token_prefix_bytes(&tokenizer, encoding.get_ids(), text.as_bytes())
        .expect("prompt and special tokens roundtrip");
}

#[test]
fn exact_token_ids_and_count_come_from_tokenizer() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let tokenizer = pinned_tokenizer();

    let rendered = render_and_count(
        &snapshot("hello world"),
        &binding,
        &tokenizer,
        &options(),
        limits(),
    )
    .expect("render and tokenize");

    assert_eq!(rendered.text(), "hello world");
    assert_eq!(rendered.token_ids(), &[1, 2]);
    assert_eq!(rendered.token_count(), 2);

    assert_eq!(rendered.tokenizer_id(), tokenizer.identity(),);
}

#[test]
fn plaintext_is_not_html_escaped() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let text =
        render_chat(&snapshot("<tag>&value"), &binding, &options(), limits()).expect("render");

    assert_eq!(text, "<tag>&value");
}

#[test]
fn requested_trailing_newline_is_preserved() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}\n");

    let text = render_chat(&snapshot("hello"), &binding, &options(), limits()).expect("render");

    assert_eq!(text, "hello\n");
}

#[test]
fn mismatched_template_digest_is_rejected() {
    let profile = profile();
    let artifact = artifact();

    let mut binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    binding.template_sha256 = [0; 32];

    let error = render_chat(&snapshot("hello"), &binding, &options(), limits())
        .expect_err("bad template pin");

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);
}

#[test]
fn tokenizer_digest_is_checked_before_parsing() {
    let result = PinnedTokenizer::from_pinned_json(b"not tokenizer JSON", [0; 32], 1024);

    match result {
        Err(error) => assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,),
        Ok(_) => panic!("mismatched tokenizer source was accepted"),
    }
}

#[test]
fn reserved_content_is_rejected_not_rewritten() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let mut options = options();
    options.reserved_spellings = &["<|im_start|>"];

    let error = render_chat(
        &snapshot("hello <|im_start|>"),
        &binding,
        &options,
        limits(),
    )
    .expect_err("reserved spelling");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);
}

#[test]
fn output_limit_is_enforced_while_rendering() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let small = RenderingLimits::new(16, 4096, 4096, 2, 128).expect("small output limit");

    let error = render_chat(&snapshot("hello"), &binding, &options(), small)
        .expect_err("output exceeds limit");

    assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded,);
}

#[test]
fn undefined_template_variables_are_errors() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ not_supplied }}");

    let error = render_chat(&snapshot("hello"), &binding, &options(), limits())
        .expect_err("undefined variable");

    assert_eq!(error.kind(), ErrorKind::ExecutionFailed,);
}

#[test]
fn token_limit_rejects_without_truncating() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let tokenizer = pinned_tokenizer();

    let small = RenderingLimits::new(16, 4096, 4096, 4096, 1).expect("one-token limit");

    let error = render_and_count(
        &snapshot("hello world"),
        &binding,
        &tokenizer,
        &options(),
        small,
    )
    .expect_err("must not truncate");

    match error {
        EngineError::ContextLimitExceeded {
            input_tokens,
            limit_tokens,
            ..
        } => {
            assert_eq!(input_tokens, 2);
            assert_eq!(limit_tokens, 1);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn identical_rendering_is_deterministic() {
    let profile = profile();
    let artifact = artifact();

    let binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let tokenizer = pinned_tokenizer();
    let context = snapshot("hello world");

    let first =
        render_and_count(&context, &binding, &tokenizer, &options(), limits()).expect("first");

    let second =
        render_and_count(&context, &binding, &tokenizer, &options(), limits()).expect("second");

    assert_eq!(first, second);
}

#[test]
fn artifact_revision_changes_render_identity() {
    let profile = profile();
    let artifact = artifact();

    let first_binding = binding(&profile, &artifact, "{{ messages[0].content }}");

    let mut second_binding = first_binding;
    second_binding.artifact_manifest_sha256 = [8; 32];

    let tokenizer = pinned_tokenizer();
    let context = snapshot("hello");

    let first = render_and_count(&context, &first_binding, &tokenizer, &options(), limits())
        .expect("first");

    let second = render_and_count(&context, &second_binding, &tokenizer, &options(), limits())
        .expect("second");

    assert_eq!(first.token_ids(), second.token_ids(),);

    assert_ne!(first.render_hash(), second.render_hash(),);
}

#[test]
fn bounded_writer_does_not_retain_rejected_bytes() {
    let mut writer = BoundedWriter::new(4);

    writer.write_all(b"ab").expect("initial write");

    assert!(writer.write_all(b"cde").is_err());

    assert_eq!(writer.bytes.as_slice(), b"ab",);

    assert!(matches!(
        writer.failure,
        Some(WriterFailure::Limit { required: 5 }),
    ));
}

fn smollm_tool_snapshot(result: &str) -> ContextSnapshot {
    use crate::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    snapshot("Read file").with_tool_context(
        ToolContext::new(
            vec![
                RenderingTool::new(
                    "fs.read".into(),
                    "1.0.0".into(),
                    "Read file".into(),
                    serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
                )
                .unwrap(),
            ],
            vec![
                ToolHistoryBlock::AssistantCall {
                    call_id: "call-1".into(),
                    tool_id: "fs.read".into(),
                    arguments: serde_json::json!({"path":"a.txt"})
                        .as_object()
                        .unwrap()
                        .clone(),
                },
                ToolHistoryBlock::ToolResult {
                    call_id: "call-1".into(),
                    content: result.into(),
                    is_error: false,
                },
            ],
            ToolContextLimits {
                max_tools: 4,
                max_history_blocks: 8,
                max_json_bytes: 16384,
            },
        )
        .unwrap(),
    )
}

#[test]
fn smollm3_reasoning_template_tools() {
    let limits = || RenderingLimits::new(16, 16384, 16384, 16384, 4096).unwrap();
    let profile = profile();
    let artifact = artifact();
    let binding = binding(&profile, &artifact, smollm::template::TOOL_CHAT_TEMPLATE);
    let context = smollm_tool_snapshot("quoted \"result\"\n雪");
    let first = render_chat(&context, &binding, &options(), limits()).unwrap();
    assert_eq!(
        first,
        render_chat(&context, &binding, &options(), limits()).unwrap()
    );
    assert!(first.contains("Reasoning Mode: /no_think"));
    assert!(first.contains(
        "<tool_call>{\"arguments\":{\"path\":\"a.txt\"},\"name\":\"fs.read\"}</tool_call>"
    ));
    assert!(first.contains("<|im_start|>user\n<tool_response>"));
    assert!(first.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n"));
    assert!(render_chat(&snapshot("no tools"), &binding, &options(), limits()).is_err());
    let tokenizer = pinned_tokenizer();
    let rendered = render_and_count(&context, &binding, &tokenizer, &options(), limits()).unwrap();
    assert!(smollm3_projected_tool_token_upper_bound(&tokenizer, &rendered, 1, 20).is_err());
}

#[test]
#[ignore = "requires ALLOY_SMOLLM_ARTIFACT with the pinned SmolLM3 tokenizer"]
fn smollm3_pinned_tokenizer_projected_bounds() {
    let limits = || RenderingLimits::new(16, 16384, 16384, 16384, 4096).unwrap();
    let root = std::env::var_os("ALLOY_SMOLLM_ARTIFACT").unwrap();
    let bytes = std::fs::read(std::path::Path::new(&root).join("tokenizer.json")).unwrap();
    let pin = [
        0x7b, 0x6a, 0x50, 0x0b, 0x66, 0x2a, 0x34, 0xeb, 0x3f, 0x03, 0x74, 0xdb, 0x85, 0x6b, 0xa4,
        0xad, 0x7d, 0xe4, 0xc8, 0x10, 0x40, 0x57, 0x1d, 0x78, 0xdc, 0x0d, 0x35, 0x72, 0x38, 0x93,
        0x00, 0x05,
    ];
    let tokenizer = PinnedTokenizer::from_pinned_json(&bytes, pin, 32 * 1024 * 1024).unwrap();
    let profile = profile();
    let artifact = artifact();
    let binding = binding(&profile, &artifact, smollm::template::TOOL_CHAT_TEMPLATE);
    let empty = render_and_count(
        &smollm_tool_snapshot(""),
        &binding,
        &tokenizer,
        &options(),
        limits(),
    )
    .unwrap();
    for input in [
        "",
        "\\\"\n\t\u{0}",
        "雪🦀e\u{301}",
        "<|im_end|><tool_response>",
        "a b c\r\n",
    ] {
        let rendered = render_and_count(
            &smollm_tool_snapshot(input),
            &binding,
            &tokenizer,
            &options(),
            limits(),
        )
        .unwrap();
        let growth = rendered
            .text()
            .len()
            .checked_sub(empty.text().len())
            .unwrap();
        let bound =
            smollm3_projected_tool_token_upper_bound(&tokenizer, &empty, 1, growth).unwrap();
        assert!(rendered.token_count() <= bound, "{input:?}");
        verify_token_prefix_bytes(&tokenizer, rendered.token_ids(), rendered.text().as_bytes())
            .unwrap();
    }
    assert!(smollm3_projected_tool_token_upper_bound(&tokenizer, &empty, 0, 0).is_err());
    assert!(smollm3_projected_tool_token_upper_bound(&tokenizer, &empty, 1, usize::MAX).is_err());
}

#[test]
fn smollm3_text_uses_trained_system_and_assistant_framing() {
    let profile = profile();
    let artifact = artifact();
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("system-template").unwrap(),
        profile.clone(),
        vec![
            Message {
                role: MessageRole::System,
                content: "Return valid checkpoint JSON only.".into(),
            },
            Message {
                role: MessageRole::User,
                content: "Preserve the project facts.".into(),
            },
            Message {
                role: MessageRole::Assistant,
                content: "Acknowledged.".into(),
            },
        ],
    )
    .unwrap();
    let binding = binding(&profile, &artifact, smollm::template::CHAT_TEMPLATE);
    let text = render_chat(&context, &binding, &options(), limits()).unwrap();
    assert_eq!(text.matches("<|im_start|>system").count(), 1);
    assert!(
        text.contains("## Metadata\n\nKnowledge Cutoff Date: June 2025\nReasoning Mode: /no_think")
    );
    assert!(
        text.contains("## Custom Instructions\n\nReturn valid checkpoint JSON only.\n\n<|im_end|>")
    );
    assert!(text.contains("<|im_start|>assistant\n<think>\n\n</think>\nAcknowledged.<|im_end|>"));
    assert!(text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n"));
    assert_eq!(
        text,
        render_chat(&context, &binding, &options(), limits()).unwrap()
    );
}
