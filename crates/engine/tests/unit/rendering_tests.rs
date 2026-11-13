use super::*;
use crate::{
    error::ErrorKind,
    types::{ContextSnapshotId, Message, MessageRole},
};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit};

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
