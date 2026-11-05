//! Public-API tests for exact rendering and tokenization.
//!
//! Uses the actual MiniJinja renderer and Hugging Face tokenizer.
//! No renderer or tokenizer implementation is mocked.
//!
//! These tests validate the shared rendering module directly. They do not
//! prove that every backend delegates its rendering path to that module.

use engine::{
    error::{EngineError, ErrorKind},
    rendering::{
        PinnedTokenizer, RENDERER_VERSION, RenderingBinding, RenderingLimits, RenderingOptions,
        render_and_count, render_chat,
    },
    types::{ArtifactId, ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

const CONTENT_TEMPLATE: &str = "{{ messages[0].content }}";

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn expect_failure<T>(result: Result<T, EngineError>, message: &str) -> EngineError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

fn profile_id() -> ModelProfileId {
    ModelProfileId::new("rendering-integration").expect("profile ID")
}

fn artifact_id() -> ArtifactId {
    ArtifactId::new("rendering-artifact").expect("artifact ID")
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

fn limits() -> RenderingLimits {
    RenderingLimits::new(8, 4096, 4096, 4096, 128).expect("rendering limits")
}

fn context(snapshot: &str, profile: &ModelProfileId, messages: Vec<Message>) -> ContextSnapshot {
    ContextSnapshot::new(
        ContextSnapshotId::new(snapshot).expect("snapshot ID"),
        profile.clone(),
        messages,
    )
    .expect("context")
}

fn user_context(content: &str) -> ContextSnapshot {
    context(
        "rendering-snapshot",
        &profile_id(),
        vec![Message {
            role: MessageRole::User,
            content: content.to_owned(),
        }],
    )
}

fn binding<'a>(
    profile: &'a ModelProfileId,
    artifact: &'a ArtifactId,
    template: &'a str,
) -> RenderingBinding<'a> {
    // Fixture identity, not a production artifact-verification report.
    RenderingBinding {
        profile_id: profile,
        artifact_id: artifact,
        template,
        template_sha256: digest(template.as_bytes()),
        artifact_manifest_sha256: [7; 32],
    }
}

fn tokenizer_source(pretty: bool) -> Vec<u8> {
    let configuration = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "WhitespaceSplit"
        },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
                "assistant": 3,
                "café": 4
            },
            "unk_token": "[UNK]"
        }
    });

    if pretty {
        serde_json::to_vec_pretty(&configuration).expect("pretty tokenizer JSON")
    } else {
        serde_json::to_vec(&configuration).expect("tokenizer JSON")
    }
}

fn pinned_tokenizer() -> PinnedTokenizer {
    let source = tokenizer_source(false);

    PinnedTokenizer::from_pinned_json(&source, digest(&source), 16 * 1024)
        .expect("actual pinned tokenizer")
}

#[test]
fn exact_text_token_ids_and_identities_are_returned() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let tokenizer = pinned_tokenizer();
    let context = user_context("hello world");

    let rendered = render_and_count(&context, &binding, &tokenizer, &options(), limits())
        .expect("render and tokenize");

    assert_eq!(rendered.text(), "hello world");
    assert_eq!(rendered.token_ids(), &[1, 2]);
    assert_eq!(rendered.token_count(), 2);

    assert_eq!(rendered.snapshot_id(), context.snapshot_id(),);

    assert_eq!(rendered.profile_id(), &profile,);

    assert_eq!(rendered.artifact_id(), &artifact,);

    assert_eq!(rendered.tokenizer_id(), tokenizer.identity(),);

    assert_eq!(
        rendered.template_hash(),
        &digest(CONTENT_TEMPLATE.as_bytes()),
    );

    assert_eq!(rendered.renderer().implementation, "minijinja",);

    assert_eq!(rendered.renderer().version, RENDERER_VERSION,);
}

#[test]
fn message_order_and_roles_are_preserved() {
    let profile = profile_id();
    let artifact = artifact_id();

    let template = concat!(
        "{% for message in messages %}",
        "{{ message.role }}:{{ message.content }}\n",
        "{% endfor %}",
    );

    let binding = binding(&profile, &artifact, template);

    let context = context(
        "ordered-messages",
        &profile,
        vec![
            Message {
                role: MessageRole::System,
                content: "instructions".to_owned(),
            },
            Message {
                role: MessageRole::User,
                content: "hello".to_owned(),
            },
            Message {
                role: MessageRole::Assistant,
                content: "world".to_owned(),
            },
        ],
    );

    let before = context.clone();

    let rendered =
        render_chat(&context, &binding, &options(), limits()).expect("ordered rendering");

    assert_eq!(
        rendered,
        "system:instructions\nuser:hello\nassistant:world\n",
    );

    assert_eq!(context, before);
}

#[test]
fn plaintext_and_unicode_are_not_html_escaped_or_rewritten() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let input = "<tag>& café";

    let rendered = render_chat(&user_context(input), &binding, &options(), limits())
        .expect("plaintext rendering");

    assert_eq!(rendered, input);
}

#[test]
fn generation_prompt_option_is_explicit_and_affects_tokenization() {
    let profile = profile_id();
    let artifact = artifact_id();

    let template = concat!(
        "{{ messages[0].content }}",
        "{% if add_generation_prompt %}",
        " assistant",
        "{% endif %}",
    );

    let binding = binding(&profile, &artifact, template);

    let tokenizer = pinned_tokenizer();
    let context = user_context("hello");

    let without_prompt = render_and_count(&context, &binding, &tokenizer, &options(), limits())
        .expect("without generation prompt");

    let mut enabled = options();
    enabled.add_generation_prompt = true;

    let with_prompt = render_and_count(&context, &binding, &tokenizer, &enabled, limits())
        .expect("with generation prompt");

    assert_eq!(without_prompt.text(), "hello");
    assert_eq!(without_prompt.token_ids(), &[1]);

    assert_eq!(with_prompt.text(), "hello assistant",);
    assert_eq!(with_prompt.token_ids(), &[1, 3],);

    assert_ne!(without_prompt.render_hash(), with_prompt.render_hash(),);
}

#[test]
fn thinking_variable_distinguishes_omitted_false_and_true() {
    let profile = profile_id();
    let artifact = artifact_id();

    let template = concat!(
        "{% if enable_thinking is defined %}",
        "{% if enable_thinking %}on{% else %}off{% endif %}",
        "{% else %}omitted{% endif %}",
    );

    let binding = binding(&profile, &artifact, template);

    let context = user_context("hello");

    for (value, expected) in [(None, "omitted"), (Some(false), "off"), (Some(true), "on")] {
        let mut selected = options();
        selected.enable_thinking = value;

        let rendered =
            render_chat(&context, &binding, &selected, limits()).expect("thinking option");

        assert_eq!(rendered, expected);
    }
}

#[test]
fn mismatched_profile_is_rejected() {
    let profile = ModelProfileId::new("different-profile").expect("profile");

    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let error = render_chat(&user_context("hello"), &binding, &options(), limits())
        .expect_err("profile mismatch");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);
}

#[test]
fn template_pin_is_checked_before_compilation() {
    let profile = profile_id();
    let artifact = artifact_id();

    let mut binding = binding(&profile, &artifact, "{% if %}");

    binding.template_sha256 = [0; 32];

    let error = render_chat(&user_context("hello"), &binding, &options(), limits())
        .expect_err("bad pin before invalid syntax");

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);
}

#[test]
fn pinned_invalid_template_is_not_silently_replaced() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, "{% if %}");

    let error = render_chat(&user_context("hello"), &binding, &options(), limits())
        .expect_err("invalid template");

    match error {
        EngineError::ExecutionFailed { stage, .. } => {
            assert_eq!(stage, "compile chat template",);
        }

        other => {
            panic!("unexpected error: {other:?}");
        }
    }
}

#[test]
fn undefined_variables_and_explicit_template_exceptions_fail() {
    let profile = profile_id();
    let artifact = artifact_id();

    for template in [
        "{{ unavailable_variable }}",
        "{{ raise_exception('unsupported content') }}",
    ] {
        let binding = binding(&profile, &artifact, template);

        let error = render_chat(&user_context("hello"), &binding, &options(), limits())
            .expect_err("template must fail explicitly");

        assert_eq!(error.kind(), ErrorKind::ExecutionFailed,);
    }
}

#[test]
fn reserved_spellings_are_rejected_without_mutating_input() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let context = user_context("hello <|im_start|> world");

    let before = context.clone();

    let mut selected = options();
    selected.reserved_spellings = &["<|im_start|>", "<|im_end|>"];

    let error = render_chat(&context, &binding, &selected, limits())
        .expect_err("reserved structural spelling");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);

    assert_eq!(context, before);
}

#[test]
fn cumulative_input_limit_counts_utf8_bytes() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let limited = RenderingLimits::new(8, 3, 4096, 4096, 128).expect("limits");

    // Two characters, but four UTF-8 bytes.
    let error = render_chat(&user_context("éé"), &binding, &options(), limited)
        .expect_err("UTF-8 byte limit");

    match error {
        EngineError::MemoryLimitExceeded {
            required_bytes,
            available_bytes,
            ..
        } => {
            assert_eq!(required_bytes, 4);
            assert_eq!(available_bytes, 3);
        }

        other => {
            panic!("unexpected error: {other:?}");
        }
    }
}

#[test]
fn rendered_output_limit_rejects_template_expansion() {
    let profile = profile_id();
    let artifact = artifact_id();

    let template = concat!(
        "{% for item in range(32) %}",
        "{{ messages[0].content }}",
        "{% endfor %}",
    );

    let binding = binding(&profile, &artifact, template);

    let limited = RenderingLimits::new(8, 4096, 4096, 16, 128).expect("limits");

    let error = render_chat(&user_context("hello"), &binding, &options(), limited)
        .expect_err("render expansion exceeds bound");

    assert_eq!(error.kind(), ErrorKind::MemoryLimitExceeded,);
}

#[test]
fn token_limit_rejects_instead_of_truncating() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let tokenizer = pinned_tokenizer();

    let limited = RenderingLimits::new(8, 4096, 4096, 4096, 1).expect("one-token limit");

    let error = render_and_count(
        &user_context("hello world"),
        &binding,
        &tokenizer,
        &options(),
        limited,
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

        other => {
            panic!("unexpected error: {other:?}");
        }
    }
}

#[test]
fn tokenizer_pin_is_checked_before_json_parsing() {
    let error = expect_failure(
        PinnedTokenizer::from_pinned_json(b"not tokenizer JSON", [0; 32], 1024),
        "bad tokenizer pin accepted",
    );

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);
}

#[test]
fn configured_tokenizer_truncation_and_padding_are_rejected() {
    let source = tokenizer_source(false);

    let mut truncated = Tokenizer::from_bytes(&source).expect("tokenizer");

    truncated
        .with_truncation(Some(TruncationParams {
            max_length: 1,
            ..TruncationParams::default()
        }))
        .expect("truncation configuration");

    let truncated_source = truncated
        .to_string(false)
        .expect("truncated tokenizer JSON");

    let error = expect_failure(
        PinnedTokenizer::from_pinned_json(
            truncated_source.as_bytes(),
            digest(truncated_source.as_bytes()),
            16 * 1024,
        ),
        "configured truncation accepted",
    );

    assert_eq!(error.kind(), ErrorKind::Unsupported,);

    let mut padded = Tokenizer::from_bytes(&source).expect("tokenizer");

    padded.with_padding(Some(PaddingParams::default()));

    let padded_source = padded.to_string(false).expect("padded tokenizer JSON");

    let error = expect_failure(
        PinnedTokenizer::from_pinned_json(
            padded_source.as_bytes(),
            digest(padded_source.as_bytes()),
            16 * 1024,
        ),
        "configured padding accepted",
    );

    assert_eq!(error.kind(), ErrorKind::Unsupported,);
}

#[test]
fn tracing_snapshot_identity_is_not_part_of_render_identity() {
    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let tokenizer = pinned_tokenizer();

    let first_context = context(
        "trace-one",
        &profile,
        vec![Message {
            role: MessageRole::User,
            content: "hello world".to_owned(),
        }],
    );

    let second_context = context("trace-two", &profile, first_context.messages().to_vec());

    let first = render_and_count(&first_context, &binding, &tokenizer, &options(), limits())
        .expect("first");

    let second = render_and_count(&second_context, &binding, &tokenizer, &options(), limits())
        .expect("second");

    assert_ne!(first.snapshot_id(), second.snapshot_id(),);

    assert_eq!(first.token_ids(), second.token_ids(),);

    assert_eq!(first.render_hash(), second.render_hash(),);
}

#[test]
fn changed_manifest_changes_render_identity_without_changing_text() {
    let profile = profile_id();
    let artifact = artifact_id();

    let first_binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let mut second_binding = first_binding;
    second_binding.artifact_manifest_sha256 = [8; 32];

    let tokenizer = pinned_tokenizer();
    let context = user_context("hello");

    let first = render_and_count(&context, &first_binding, &tokenizer, &options(), limits())
        .expect("first");

    let second = render_and_count(&context, &second_binding, &tokenizer, &options(), limits())
        .expect("second");

    assert_eq!(first.text(), second.text());

    assert_eq!(first.token_ids(), second.token_ids(),);

    assert_ne!(first.render_hash(), second.render_hash(),);
}

#[test]
fn tokenizer_source_identity_changes_even_when_behavior_is_equivalent() {
    let compact = tokenizer_source(false);
    let pretty = tokenizer_source(true);

    assert_ne!(compact, pretty);

    let first = PinnedTokenizer::from_pinned_json(&compact, digest(&compact), 16 * 1024)
        .expect("compact tokenizer");

    let second = PinnedTokenizer::from_pinned_json(&pretty, digest(&pretty), 16 * 1024)
        .expect("pretty tokenizer");

    assert_ne!(first.identity(), second.identity(),);

    let profile = profile_id();
    let artifact = artifact_id();

    let binding = binding(&profile, &artifact, CONTENT_TEMPLATE);

    let context = user_context("hello world");

    let first_render =
        render_and_count(&context, &binding, &first, &options(), limits()).expect("first render");

    let second_render =
        render_and_count(&context, &binding, &second, &options(), limits()).expect("second render");

    assert_eq!(first_render.token_ids(), second_render.token_ids(),);

    assert_ne!(first_render.render_hash(), second_render.render_hash(),);
}
