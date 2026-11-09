use super::*;
use crate::types::{ContextSnapshotId, ModelProfileId, RendererIdentity, TokenizerId};
use qwen3::{config::Qwen3Config, fixture::generate_fixture};

fn loaded_model() -> Arc<Qwen3Model> {
    let config = Qwen3Config::tiny_fixture();
    let weights = generate_fixture(&config);

    Arc::new(Qwen3Model::load(config, weights).expect("actual fixture-weight model must load"))
}

fn context(tokens: &[u32]) -> RenderedContext {
    // These fixtures test the cache interface and actual model
    // execution, not rendering or tokenizer conformance.
    RenderedContext {
        snapshot_id: ContextSnapshotId::new("cache-test").expect("snapshot ID"),
        profile_id: ModelProfileId::new("tiny-qwen3").expect("profile ID"),
        artifact_id: ArtifactId::new("tiny-qwen3").expect("artifact ID"),
        text: String::new(),
        token_ids: tokens.to_vec(),
        render_hash: [0; 32],
        template_hash: [1; 32],
        tool_schema_hash: None,
        tokenizer_id: TokenizerId::new("tiny-tokenizer").expect("tokenizer ID"),
        renderer: RendererIdentity {
            implementation: "minijinja".to_owned(),
            version: "2.12.0".to_owned(),
        },
    }
}

fn cache(model: Arc<Qwen3Model>) -> Qwen3SessionCache {
    Qwen3SessionCache::new(
        model,
        ArtifactId::new("tiny-qwen3").expect("artifact ID"),
        16,
        1024 * 1024,
    )
    .expect("actual KV allocation")
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (&got, &want) in actual.iter().zip(expected) {
        assert!(got.is_finite());
        assert!(want.is_finite());
        assert!(
            (got - want).abs() <= 2e-2 + 1e-2 * want.abs(),
            "got {got}, expected {want}"
        );
    }
}

/// Phase-1 acceptance name kept for issue #4 discoverability.
///
/// The original plan called for a stub that reported unimplemented.
/// The engine shipped a real single-session cache instead; this test
/// exercises that real interface.
#[test]
fn cache_stub_interface() {
    let model = loaded_model();
    let mut session = cache(model.clone());

    assert!(session.is_empty());
    assert_eq!(session.capacity_tokens(), 16);
    assert_eq!(session.artifact_id().as_str(), "tiny-qwen3");

    let first = session
        .prefill(&context(&[3, 41]))
        .expect("initial prefill");
    assert_eq!(first.reused_prefix_tokens(), 0);
    assert_eq!(first.processed_input_tokens(), 2);
    assert_eq!(session.len(), 2);
    assert_eq!(session.retained_tokens(), &[3, 41]);
    assert!(session.last_logits().is_some());

    let extended = session
        .prefill(&context(&[3, 41, 7, 12]))
        .expect("suffix prefill");
    assert_eq!(extended.reused_prefix_tokens(), 2);
    assert_eq!(extended.processed_input_tokens(), 2);

    let full = model.forward(&[3, 41, 7, 12]).expect("real full forward");
    assert_close(extended.logits(), full.last().expect("last logits"));

    session.clear();
    assert!(session.is_empty());
    assert!(session.last_logits().is_none());
}

#[test]
fn tool_schema_change_invalidates_actual_retained_prefix() {
    let model = loaded_model();
    let mut retained = cache(model.clone());
    let mut first = context(&[3, 41]);
    first.tool_schema_hash = Some([7; 32]);
    let initial = retained.prefill(&first).expect("initial schema");
    assert_eq!(initial.reused_prefix_tokens(), 0);
    let same = retained.prefill(&first).expect("unchanged schema");
    assert_eq!(same.reused_prefix_tokens(), 2);
    assert_eq!(same.processed_input_tokens(), 0);

    // Exact token IDs and every other identity stay unchanged.
    let mut changed = first.clone();
    changed.tool_schema_hash = Some([8; 32]);
    let rebuilt = retained.prefill(&changed).expect("changed schema");
    assert_eq!(rebuilt.reused_prefix_tokens(), 0);
    assert_eq!(rebuilt.processed_input_tokens(), 2);
    let full = model.forward(&[3, 41]).expect("full forward");
    assert_close(rebuilt.logits(), full.last().expect("last logits"));
}

#[test]
fn adding_and_removing_tool_schemas_invalidates_actual_retained_prefix() {
    let model = loaded_model();
    let mut retained = cache(model);
    let text_only = context(&[3, 41]);
    retained.prefill(&text_only).expect("text-only prefill");
    let mut with_tools = text_only.clone();
    with_tools.tool_schema_hash = Some([7; 32]);
    let added = retained.prefill(&with_tools).expect("added schemas");
    assert_eq!(added.reused_prefix_tokens(), 0);
    assert_eq!(added.processed_input_tokens(), 2);
    let removed = retained.prefill(&text_only).expect("removed schemas");
    assert_eq!(removed.reused_prefix_tokens(), 0);
    assert_eq!(removed.processed_input_tokens(), 2);
    let repeated = retained
        .prefill(&text_only)
        .expect("stable text-only context");
    assert_eq!(repeated.reused_prefix_tokens(), 2);
    assert_eq!(repeated.processed_input_tokens(), 0);
}

#[test]
fn actual_prefix_state_is_reused_for_an_extension() {
    let model = loaded_model();
    let mut cache = cache(model.clone());

    let first = cache.prefill(&context(&[3, 41])).expect("initial prefill");

    assert_eq!(first.reused_prefix_tokens(), 0);
    assert_eq!(first.processed_input_tokens(), 2);

    let extended = cache
        .prefill(&context(&[3, 41, 7, 12]))
        .expect("suffix prefill");

    assert_eq!(extended.reused_prefix_tokens(), 2);
    assert_eq!(extended.processed_input_tokens(), 2);

    let full = model.forward(&[3, 41, 7, 12]).expect("real full forward");

    assert_close(extended.logits(), full.last().expect("last logits"));
}

#[test]
fn identical_input_reuses_real_retained_logits() {
    let model = loaded_model();
    let mut cache = cache(model.clone());

    cache.prefill(&context(&[3, 41])).expect("initial");

    let repeated = cache.prefill(&context(&[3, 41])).expect("repeat");

    assert_eq!(repeated.reused_prefix_tokens(), 2);
    assert_eq!(repeated.processed_input_tokens(), 0);

    let full = model.forward(&[3, 41]).expect("real full forward");

    assert_close(repeated.logits(), full.last().expect("last logits"));
}

#[test]
fn divergent_or_shortened_input_rebuilds() {
    let model = loaded_model();
    let mut cache = cache(model.clone());

    cache.prefill(&context(&[3, 41, 7])).expect("initial");

    let shorter = cache.prefill(&context(&[3, 41])).expect("shorter");

    assert_eq!(shorter.reused_prefix_tokens(), 0);
    assert_eq!(shorter.processed_input_tokens(), 2);

    let divergent = cache.prefill(&context(&[3, 12, 7])).expect("divergent");

    assert_eq!(divergent.reused_prefix_tokens(), 0);
    assert_eq!(divergent.processed_input_tokens(), 3);

    let full = model.forward(&[3, 12, 7]).expect("real full forward");

    assert_close(divergent.logits(), full.last().expect("last logits"));
}

#[test]
fn template_identity_change_invalidates_reuse() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("initial");

    let mut changed = context(&[3, 41, 7]);
    changed.template_hash = [2; 32];

    let result = cache.prefill(&changed).expect("changed template");

    assert_eq!(result.reused_prefix_tokens(), 0);
    assert_eq!(result.processed_input_tokens(), 3);
}

#[test]
fn profile_identity_change_invalidates_reuse() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("initial");

    let mut changed = context(&[3, 41, 7]);
    changed.profile_id = ModelProfileId::new("another-profile").expect("profile ID");

    let result = cache.prefill(&changed).expect("changed profile");

    assert_eq!(result.reused_prefix_tokens(), 0);
    assert_eq!(result.processed_input_tokens(), 3);
}

#[test]
fn tokenizer_identity_change_invalidates_reuse() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("initial");

    let mut changed = context(&[3, 41, 7]);
    changed.tokenizer_id = TokenizerId::new("another-tokenizer").expect("tokenizer ID");

    let result = cache.prefill(&changed).expect("changed tokenizer");

    assert_eq!(result.reused_prefix_tokens(), 0);
    assert_eq!(result.processed_input_tokens(), 3);
}

#[test]
fn renderer_identity_change_invalidates_reuse() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("initial");

    let mut changed = context(&[3, 41, 7]);
    changed.renderer.version = "different-version".to_owned();

    let result = cache.prefill(&changed).expect("changed renderer");

    assert_eq!(result.reused_prefix_tokens(), 0);
    assert_eq!(result.processed_input_tokens(), 3);
}

#[test]
fn actual_cached_decode_matches_full_execution() {
    let model = loaded_model();
    let mut cache = cache(model.clone());

    cache.prefill(&context(&[3, 41])).expect("prefill");

    let actual = cache.decode_token(7).expect("real cached decode");

    let full = model.forward(&[3, 41, 7]).expect("real full forward");

    assert_close(actual, full.last().expect("last logits"));
    assert_eq!(cache.retained_tokens(), &[3, 41, 7]);
}

#[test]
fn decoded_tokens_can_be_reused_by_a_later_prefill() {
    let model = loaded_model();
    let mut cache = cache(model.clone());

    cache.prefill(&context(&[3, 41])).expect("prefill");
    cache.decode_token(7).expect("decode");

    let extended = cache
        .prefill(&context(&[3, 41, 7, 12]))
        .expect("extend decoded prefix");

    assert_eq!(extended.reused_prefix_tokens(), 3);
    assert_eq!(extended.processed_input_tokens(), 1);

    let full = model.forward(&[3, 41, 7, 12]).expect("real full forward");

    assert_close(extended.logits(), full.last().expect("last logits"));
}

#[test]
fn invalid_decode_clears_actual_state() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("prefill");

    assert!(cache.decode_token(u32::MAX).is_err());
    assert!(cache.is_empty());
    assert!(cache.last_logits().is_none());
    assert_eq!(cache.kv.len(), 0);
}

#[test]
fn invalid_prefill_clears_actual_state() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("initial");

    assert!(cache.prefill(&context(&[3, 41, u32::MAX])).is_err());
    assert!(cache.is_empty());
    assert!(cache.last_logits().is_none());
    assert_eq!(cache.kv.len(), 0);
}

#[test]
fn wrong_artifact_is_rejected_and_state_is_cleared() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3, 41])).expect("prefill");

    let mut wrong = context(&[3, 41, 7]);
    wrong.artifact_id = ArtifactId::new("different-artifact").expect("artifact ID");

    assert!(cache.prefill(&wrong).is_err());
    assert!(cache.is_empty());
    assert!(cache.last_logits().is_none());
    assert_eq!(cache.kv.len(), 0);
}

#[test]
fn empty_or_over_capacity_prefill_clears_state() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3])).expect("initial");

    assert!(cache.prefill(&context(&[])).is_err());
    assert!(cache.is_empty());
    assert!(cache.last_logits().is_none());

    cache.prefill(&context(&[3])).expect("initial again");

    assert!(cache.prefill(&context(&[3; 17])).is_err());
    assert!(cache.is_empty());
    assert!(cache.last_logits().is_none());
    assert_eq!(cache.kv.len(), 0);
}

#[test]
fn decode_at_capacity_clears_state() {
    let model = loaded_model();
    let mut cache = cache(model);

    cache.prefill(&context(&[3; 16])).expect("fill capacity");

    assert!(cache.decode_token(7).is_err());
    assert!(cache.is_empty());
    assert!(cache.last_logits().is_none());
    assert_eq!(cache.kv.len(), 0);
}

#[test]
fn clear_allows_a_fresh_real_prefill() {
    let model = loaded_model();
    let mut cache = cache(model.clone());

    cache.prefill(&context(&[3, 41])).expect("initial");

    cache.clear();

    assert!(cache.is_empty());
    assert_eq!(cache.kv.len(), 0);
    assert!(cache.last_logits().is_none());
    assert_eq!(cache.capacity_tokens(), 16);

    let fresh = cache.prefill(&context(&[7, 12])).expect("fresh prefill");

    assert_eq!(fresh.reused_prefix_tokens(), 0);
    assert_eq!(fresh.processed_input_tokens(), 2);

    let full = model.forward(&[7, 12]).expect("real full forward");

    assert_close(fresh.logits(), full.last().expect("last logits"));
}
