//! Profile arithmetic for projected model-visible context growth.
use engine::{
    error::EngineError,
    registry::ModelProfile,
    types::{ArtifactId, GenerationOptions, ModelProfileId},
};

#[test]
fn projected_growth_reserves_output_and_margin_and_rejects_overflow() {
    let profile = ModelProfile::new(
        ModelProfileId::new("projection").unwrap(),
        ArtifactId::new("artifact").unwrap(),
        100,
        20,
        5,
        GenerationOptions::greedy(20).unwrap(),
    )
    .unwrap();
    assert!(profile.check_context_with_growth(50, 25).is_ok());
    assert!(matches!(
        profile.check_context_with_growth(50, 26),
        Err(EngineError::ContextLimitExceeded {
            input_tokens: 76,
            output_reserve_tokens: 20,
            safety_margin_tokens: 5,
            limit_tokens: 100
        })
    ));
    assert!(matches!(
        profile.check_context_with_growth(usize::MAX, 1),
        Err(EngineError::ContextLimitExceeded {
            input_tokens: usize::MAX,
            ..
        })
    ));
    assert!(profile.check_context(75).is_ok());
    assert!(profile.check_context(76).is_err());
}

#[test]
fn pinned_normalizer_canonical_decomposition_is_bounded_by_input_bytes() {
    // Tokenizers 0.23.2 uses these exact Unicode 9 tables for its NFC
    // normalizer. Composition cannot increase the canonical decomposition's
    // scalar count, and every resulting scalar takes at most four UTF-8 bytes.
    // Validate the per-scalar premise exhaustively; it applies additively to
    // every input string before canonical reordering and composition.
    assert_eq!(unicode_normalization_alignments::UNICODE_VERSION, (9, 0, 0));
    for value in 0..=char::MAX as u32 {
        let Some(character) = char::from_u32(value) else {
            continue;
        };
        let mut decomposed = 0;
        unicode_normalization_alignments::char::decompose_canonical(character, |_| decomposed += 1);
        assert!(
            decomposed <= 2 * character.len_utf8(),
            "U+{value:04X} expands to {decomposed} scalars"
        );
    }
}

#[test]
fn projected_unknown_tokenizer_rejects() {
    use engine::rendering::{PinnedTokenizer, qwen3_projected_token_upper_bound};
    use sha2::{Digest, Sha256};
    let source = serde_json::to_vec(&serde_json::json!({
        "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
        "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
        "post_processor":null, "decoder":null,
        "model":{"type":"WordLevel", "vocab":{"[UNK]":0}, "unk_token":"[UNK]"}
    }))
    .unwrap();
    let tokenizer =
        PinnedTokenizer::from_pinned_json(&source, Sha256::digest(&source).into(), 16384).unwrap();
    let error = qwen3_projected_token_upper_bound(&tokenizer, 1).unwrap_err();
    assert!(matches!(error, EngineError::Unsupported { .. }), "{error}");
}
