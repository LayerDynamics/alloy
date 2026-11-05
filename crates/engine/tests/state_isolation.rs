//! Cache-layer state-isolation integration tests.
//!
//! Combines the public shared renderer, actual Hugging Face tokenization,
//! engine Qwen3SessionCache, and real Qwen3 model execution.
//!
//! Tests exercise interleaved independent caches, reset/error/drop
//! isolation, loaded-model ownership, and metadata compatibility changes.
//!
//! These tests do not establish:
//!
//! - A runtime registry of canonical SessionStateId entries.
//! - Runtime cross-request prefix-reuse wiring.
//! - Concurrent inference on one native worker.
//! - Production artifact or numerical conformance.
//!
//! Synthetic fixture weights are executed by the actual model. No model,
//! cache, or tokenizer implementation is mocked.

use engine::{
    cache::Qwen3SessionCache,
    error::ErrorKind,
    rendering::{
        PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions, render_and_count,
    },
    types::{
        ArtifactId, ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId,
        RenderedContext,
    },
};
use qwen3::{
    config::Qwen3Config,
    fixture::generate_fixture,
    model::{ModelError, Qwen3Model},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const TEMPLATE: &str = "{{ messages[0].content }}";

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn profile_id() -> ModelProfileId {
    ModelProfileId::new("state-profile").expect("profile ID")
}

fn artifact_id() -> ArtifactId {
    ArtifactId::new("state-artifact").expect("artifact ID")
}

fn loaded_model() -> Arc<Qwen3Model> {
    let config = Qwen3Config::tiny_fixture();
    let weights = generate_fixture(&config);

    Arc::new(Qwen3Model::load(config, weights).expect("actual fixture model"))
}

fn tokenizer(pretty: bool) -> PinnedTokenizer {
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
                "alpha": 1,
                "beta": 2,
                "gamma": 3,
                "delta": 4,
                "epsilon": 5
            },
            "unk_token": "[UNK]"
        }
    });

    let bytes = if pretty {
        serde_json::to_vec_pretty(&configuration).expect("pretty tokenizer JSON")
    } else {
        serde_json::to_vec(&configuration).expect("tokenizer JSON")
    };

    PinnedTokenizer::from_pinned_json(&bytes, digest(&bytes), 16 * 1024)
        .expect("actual pinned tokenizer")
}

fn render(
    profile: &ModelProfileId,
    artifact: &ArtifactId,
    tokenizer: &PinnedTokenizer,
    template: &str,
    content: &str,
) -> RenderedContext {
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    let snapshot_id = ContextSnapshotId::new(format!(
        "state-snapshot-{}",
        NEXT.fetch_add(1, Ordering::Relaxed),
    ))
    .expect("snapshot ID");

    let context = ContextSnapshot::new(
        snapshot_id,
        profile.clone(),
        vec![Message {
            role: MessageRole::User,
            content: content.to_owned(),
        }],
    )
    .expect("context");

    // Fixture bindings identify this in-memory model test. This is not
    // a production checkpoint-verification report.
    let binding = RenderingBinding {
        profile_id: profile,
        artifact_id: artifact,
        template,
        template_sha256: digest(template.as_bytes()),
        artifact_manifest_sha256: [1; 32],
    };

    let options = RenderingOptions {
        add_generation_prompt: false,
        enable_thinking: None,
        add_special_tokens: false,
        keep_trailing_newline: true,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: &[],
    };

    let limits = RenderingLimits::new(4, 1024, 1024, 1024, 32).expect("rendering limits");

    render_and_count(&context, &binding, tokenizer, &options, limits)
        .expect("actual rendering and tokenization")
}

fn cache(model: Arc<Qwen3Model>) -> Qwen3SessionCache {
    Qwen3SessionCache::new(model, artifact_id(), 16, 1024 * 1024).expect("actual cache allocation")
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite());
        assert!(expected.is_finite());

        let tolerance = 2e-2 + 1e-2 * expected.abs();

        assert!(
            (actual - expected).abs() <= tolerance,
            "logit {index}: actual {actual}, expected {expected}",
        );
    }
}

#[test]
fn interleaved_caches_do_not_mix_token_history_or_kv() {
    let model = loaded_model();

    let mut first = cache(Arc::clone(&model));
    let mut second = cache(Arc::clone(&model));

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    let first_context = render(&profile, &artifact, &tokenizer, TEMPLATE, "alpha beta");

    let second_context = render(&profile, &artifact, &tokenizer, TEMPLATE, "gamma delta");

    first.prefill(&first_context).expect("first prefill");

    second.prefill(&second_context).expect("second prefill");

    let first_logits = first.decode_token(3).expect("first continuation");

    let first_full = model.forward(&[1, 2, 3]).expect("first full forward");

    assert_close(first_logits, first_full.last().expect("first logits"));

    assert_eq!(first.retained_tokens(), &[1, 2, 3],);

    assert_eq!(second.retained_tokens(), &[3, 4],);

    let second_logits = second.decode_token(1).expect("second continuation");

    let second_full = model.forward(&[3, 4, 1]).expect("second full forward");

    assert_close(second_logits, second_full.last().expect("second logits"));

    assert_eq!(first.retained_tokens(), &[1, 2, 3],);

    assert_eq!(second.retained_tokens(), &[3, 4, 1],);
}

#[test]
fn clearing_one_cache_does_not_invalidate_another_prefix() {
    let model = loaded_model();

    let mut first = cache(Arc::clone(&model));
    let mut second = cache(Arc::clone(&model));

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    first
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "alpha beta",
        ))
        .expect("first prefill");

    second
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "gamma delta",
        ))
        .expect("second prefill");

    first.clear();

    assert!(first.is_empty());
    assert!(first.last_logits().is_none());

    let extension = render(
        &profile,
        &artifact,
        &tokenizer,
        TEMPLATE,
        "gamma delta alpha",
    );

    let result = second
        .prefill(&extension)
        .expect("second prefix remains reusable");

    assert_eq!(result.reused_prefix_tokens(), 2,);

    assert_eq!(result.processed_input_tokens(), 1,);

    let full = model
        .forward(extension.token_ids())
        .expect("full second context");

    assert_close(result.logits(), full.last().expect("last logits"));
}

#[test]
fn invalid_decode_clears_only_the_affected_cache() {
    let model = loaded_model();

    let mut first = cache(Arc::clone(&model));
    let mut second = cache(Arc::clone(&model));

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    first
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "alpha beta",
        ))
        .expect("first prefill");

    second
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "gamma delta",
        ))
        .expect("second prefill");

    assert!(first.decode_token(u32::MAX).is_err());

    assert!(first.is_empty());
    assert!(first.last_logits().is_none());

    assert_eq!(second.retained_tokens(), &[3, 4],);

    let extension = render(
        &profile,
        &artifact,
        &tokenizer,
        TEMPLATE,
        "gamma delta epsilon",
    );

    let result = second
        .prefill(&extension)
        .expect("unaffected cache extension");

    assert_eq!(result.reused_prefix_tokens(), 2,);

    let full = model
        .forward(extension.token_ids())
        .expect("full unaffected context");

    assert_close(result.logits(), full.last().expect("last logits"));
}

#[test]
fn dropping_one_cache_does_not_destroy_another_cache_state() {
    let model = loaded_model();

    let mut first = cache(Arc::clone(&model));
    let mut second = cache(Arc::clone(&model));

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    first
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "alpha beta",
        ))
        .expect("first prefill");

    second
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "gamma delta",
        ))
        .expect("second prefill");

    drop(first);

    let actual = second.decode_token(5).expect("remaining cache decode");

    let full = model.forward(&[3, 4, 5]).expect("full remaining context");

    assert_close(actual, full.last().expect("last logits"));

    assert_eq!(second.retained_tokens(), &[3, 4, 5],);
}

#[test]
fn cache_keeps_its_loaded_model_alive_until_cache_drop() {
    let model = loaded_model();

    let weak = Arc::downgrade(&model);

    let mut session = cache(Arc::clone(&model));

    drop(model);

    assert!(weak.upgrade().is_some());

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    session
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "alpha beta",
        ))
        .expect("cache-owned model remains usable");

    drop(session);

    assert!(weak.upgrade().is_none());
}

#[test]
fn another_loaded_instance_cannot_mutate_foreign_native_cache() {
    let owner = loaded_model();
    let foreign = loaded_model();

    let mut native = owner.new_cache(16, 1024 * 1024).expect("native cache");

    owner
        .prefill_cached(&[1, 2], &mut native)
        .expect("owner prefill");

    let before_length = native.len();

    let before_memory = native.tracked_memory_bytes().expect("before accounting");

    assert!(matches!(
        foreign.prefill_cached(&[3, 4], &mut native,),
        Err(ModelError::InvalidConfig("cache belongs to another model")),
    ));

    assert!(matches!(
        foreign.decode_token(3, &mut native,),
        Err(ModelError::InvalidConfig("cache belongs to another model")),
    ));

    assert_eq!(native.len(), before_length,);

    assert_eq!(
        native.tracked_memory_bytes().expect("after rejection"),
        before_memory,
    );

    // Check actual retained values, not just lengths.
    let actual = owner
        .decode_token(3, &mut native)
        .expect("owner continuation");

    let full = owner.forward(&[1, 2, 3]).expect("full owner context");

    assert_close(&actual, full.last().expect("last logits"));
}

#[test]
fn changed_template_identity_rebuilds_even_with_identical_tokens() {
    let model = loaded_model();
    let mut session = cache(Arc::clone(&model));

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    let first = render(&profile, &artifact, &tokenizer, TEMPLATE, "alpha beta");

    session.prefill(&first).expect("initial prefill");

    let changed = render(
        &profile,
        &artifact,
        &tokenizer,
        "{# changed template identity #}{{ messages[0].content }}",
        "alpha beta",
    );

    assert_eq!(first.token_ids(), changed.token_ids(),);

    assert_ne!(first.template_hash(), changed.template_hash(),);

    let result = session.prefill(&changed).expect("rebuild");

    assert_eq!(result.reused_prefix_tokens(), 0,);

    assert_eq!(result.processed_input_tokens(), changed.token_count(),);

    let full = model
        .forward(changed.token_ids())
        .expect("full changed context");

    assert_close(result.logits(), full.last().expect("last logits"));
}

#[test]
fn changed_tokenizer_identity_rebuilds_even_with_identical_tokens() {
    let model = loaded_model();
    let mut session = cache(Arc::clone(&model));

    let compact = tokenizer(false);
    let pretty = tokenizer(true);

    assert_ne!(compact.identity(), pretty.identity(),);

    let profile = profile_id();
    let artifact = artifact_id();

    let first = render(&profile, &artifact, &compact, TEMPLATE, "alpha beta");

    session.prefill(&first).expect("initial prefill");

    let changed = render(&profile, &artifact, &pretty, TEMPLATE, "alpha beta");

    assert_eq!(first.token_ids(), changed.token_ids(),);

    let result = session
        .prefill(&changed)
        .expect("tokenizer identity rebuild");

    assert_eq!(result.reused_prefix_tokens(), 0,);

    assert_eq!(result.processed_input_tokens(), changed.token_count(),);
}

#[test]
fn changed_profile_identity_does_not_silently_reuse_state() {
    let model = loaded_model();
    let mut session = cache(model);

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    session
        .prefill(&render(
            &profile,
            &artifact,
            &tokenizer,
            TEMPLATE,
            "alpha beta",
        ))
        .expect("initial profile");

    let other_profile = ModelProfileId::new("other-state-profile").expect("other profile");

    let changed = render(
        &other_profile,
        &artifact,
        &tokenizer,
        TEMPLATE,
        "alpha beta gamma",
    );

    let result = session.prefill(&changed).expect("profile rebuild");

    assert_eq!(result.reused_prefix_tokens(), 0,);

    assert_eq!(result.processed_input_tokens(), 3,);
}

#[test]
fn wrong_artifact_clears_only_the_targeted_cache() {
    let model = loaded_model();

    let mut first = cache(Arc::clone(&model));
    let mut second = cache(Arc::clone(&model));

    let tokenizer = tokenizer(false);
    let profile = profile_id();
    let artifact = artifact_id();

    let original = render(&profile, &artifact, &tokenizer, TEMPLATE, "alpha beta");

    first.prefill(&original).expect("first prefill");

    second.prefill(&original).expect("second prefill");

    let wrong_artifact = ArtifactId::new("different-state-artifact").expect("different artifact");

    let wrong = render(
        &profile,
        &wrong_artifact,
        &tokenizer,
        TEMPLATE,
        "alpha beta gamma",
    );

    let error = first.prefill(&wrong).expect_err("artifact mismatch");

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);

    assert!(first.is_empty());
    assert!(first.last_logits().is_none());

    let valid_extension = render(
        &profile,
        &artifact,
        &tokenizer,
        TEMPLATE,
        "alpha beta gamma",
    );

    let result = second.prefill(&valid_extension).expect("unaffected cache");

    assert_eq!(result.reused_prefix_tokens(), 2,);

    assert_eq!(result.processed_input_tokens(), 1,);

    let full = model
        .forward(valid_extension.token_ids())
        .expect("full valid context");

    assert_close(result.logits(), full.last().expect("last logits"));
}
