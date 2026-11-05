//! Real single-session Qwen3 KV and prefix cache.
//!
//! This module retains actual Qwen3Cache state produced by model execution.
//! It never synthesizes cache hits or substitutes stored output text for
//! model execution.
//!
//! Reuse requires:
//!
//! - The same loaded model instance.
//! - Matching artifact, profile, tokenizer, template, and renderer identity.
//! - An exact retained-token prefix.
//!
//! The current Qwen3 cache does not expose truncation. Requests that shorten
//! or diverge from the retained prefix therefore rebuild from the beginning.
//!
//! This module does not implement the multi-session state registry or the
//! Appendix B.1 SessionStateId hashing algorithm.
//!
//! The engine manifest must declare the local Qwen3 dependency:
//!
//! ```toml
//! qwen3 = { path = "../families/qwen3" }
//! ```

use crate::{
    api::ENGINE_API_VERSION,
    error::EngineError,
    types::{ArtifactId, ModelProfileId, RenderedContext, RendererIdentity, TokenizerId},
};
use qwen3::{
    model::{ModelError, Qwen3Cache, Qwen3Model},
    register::IMPL_VERSION,
};
use std::sync::Arc;

/// Stable name of the actual cache representation used by this adapter.
///
/// Change this identifier when retained KV layout or interpretation changes.
/// The Qwen3 implementation version must also change for such changes.
///
/// The current Qwen3 reference implementation retains full host-side f64
/// keys and values. This identifier does not describe a quantized or
/// Metal-resident KV representation.
pub const QWEN3_CACHE_REPRESENTATION: &str = "qwen3.host-f64.full.v1";

/// Identities that must match before retained KV state can be reused.
///
/// Token-prefix equality is checked separately against the actual token IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Compatibility {
    artifact_id: ArtifactId,
    profile_id: ModelProfileId,
    tokenizer_id: TokenizerId,
    template_hash: [u8; 32],
    renderer: RendererIdentity,
    family_impl_version: u32,
    engine_api_version: u32,
    cache_representation: &'static str,
}

impl Compatibility {
    fn from_context(context: &RenderedContext) -> Self {
        Self {
            artifact_id: context.artifact_id().clone(),
            profile_id: context.profile_id().clone(),
            tokenizer_id: context.tokenizer_id().clone(),
            template_hash: *context.template_hash(),
            renderer: context.renderer().clone(),
            family_impl_version: IMPL_VERSION,
            engine_api_version: ENGINE_API_VERSION,
            cache_representation: QWEN3_CACHE_REPRESENTATION,
        }
    }
}

/// Outcome of a real prefill or prefix-cache lookup.
///
/// Logits correspond to the position following the final token of the
/// requested input sequence. They are borrowed from retained execution
/// output rather than copied into another buffer.
#[derive(Debug)]
pub struct PrefillResult<'a> {
    logits: &'a [f32],
    reused_prefix_tokens: usize,
    processed_input_tokens: usize,
}

impl<'a> PrefillResult<'a> {
    /// Borrow logits for the position following the supplied input.
    pub fn logits(&self) -> &'a [f32] {
        self.logits
    }

    /// Number of token positions whose actual retained KV state was reused.
    pub fn reused_prefix_tokens(&self) -> usize {
        self.reused_prefix_tokens
    }

    /// Number of input tokens actually processed during this call.
    pub fn processed_input_tokens(&self) -> usize {
        self.processed_input_tokens
    }
}

/// Actual Qwen3 KV state owned by one engine session.
///
/// The loaded model is retained through Arc, so it cannot be dropped while
/// this cache uses it. Mutable access serializes operations on this state.
///
/// The constructor's KV budget is passed to Qwen3's real cache allocator.
/// It does not cover model weights, token bookkeeping, final logits, or
/// execution temporaries. Those remain part of engine-wide admission.
///
/// This is a family-specific adapter, not the shared cache implementation
/// for every model family.
pub struct Qwen3SessionCache {
    model: Arc<Qwen3Model>,
    artifact_id: ArtifactId,
    capacity_tokens: usize,
    kv: Qwen3Cache,
    tokens: Vec<u32>,
    compatibility: Option<Compatibility>,
    last_logits: Option<Vec<f32>>,
}

impl Qwen3SessionCache {
    /// Allocate real K/V buffers for a single loaded model.
    ///
    /// The artifact identifier must be supplied by the verified model-loading
    /// path. This constructor does not verify checkpoint hashes itself.
    pub fn new(
        model: Arc<Qwen3Model>,
        artifact_id: ArtifactId,
        capacity_tokens: usize,
        max_kv_bytes: usize,
    ) -> Result<Self, EngineError> {
        if capacity_tokens == 0 {
            return Err(EngineError::invalid_request(
                "capacity_tokens",
                "cache capacity must be greater than zero",
            ));
        }

        let token_bytes = capacity_tokens
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| EngineError::Internal {
                message: "cache token-bookkeeping size overflow".to_owned(),
            })?;

        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(capacity_tokens)
            .map_err(|_| EngineError::AllocationFailed {
                operation: "reserve cache token bookkeeping",
                requested_bytes: u64::try_from(token_bytes).ok(),
            })?;

        let kv: Qwen3Cache = model
            .new_cache(capacity_tokens, max_kv_bytes)
            .map_err(|error| map_model_error(error, "allocate KV cache"))?;

        let cache = Self {
            model,
            artifact_id,
            capacity_tokens,
            kv,
            tokens,
            compatibility: None,
            last_logits: None,
        };

        cache.check_consistency()?;

        Ok(cache)
    }

    /// Return the model artifact to which this cache is bound.
    pub fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    /// Return the maximum number of retained token positions.
    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    /// Return the number of positions represented by actual retained KV state.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether no token positions are currently retained.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Borrow the exact token IDs represented by retained KV state.
    pub fn retained_tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Borrow final-position logits if the cache contains a valid prefix.
    pub fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    /// Invalidate retained context while retaining allocated KV capacity.
    ///
    /// This is not a full memory-release operation. Drop this cache to
    /// release its owned KV buffers and token-bookkeeping allocation.
    ///
    /// The retained logits allocation is released by this operation.
    pub fn clear(&mut self) {
        self.kv.reset();
        self.tokens.clear();
        self.compatibility = None;
        self.last_logits = None;
    }

    /// Prefill the requested context using actual retained K/V when compatible.
    ///
    /// If identities match and the input extends the complete retained token
    /// sequence, only the missing suffix is executed.
    ///
    /// If the entire input is already retained, the actual final-position
    /// logits retained from that execution are returned without re-execution.
    ///
    /// Shortened, divergent, or identity-incompatible inputs rebuild from an
    /// empty cache because the underlying cache does not expose truncation.
    ///
    /// Every returned error invalidates this cache.
    pub fn prefill(&mut self, context: &RenderedContext) -> Result<PrefillResult<'_>, EngineError> {
        if let Err(error) = self.check_consistency() {
            self.clear();
            return Err(error);
        }

        if context.artifact_id() != &self.artifact_id {
            self.clear();
            return Err(EngineError::ArtifactValidationFailed {
                artifact_id: Some(context.artifact_id().as_str().to_owned()),
                message: "rendered context belongs to a different artifact".to_owned(),
            });
        }

        let input = context.token_ids();

        if input.is_empty() {
            self.clear();
            return Err(EngineError::invalid_request(
                "token_ids",
                "prefill input must not be empty",
            ));
        }

        if input.len() > self.capacity_tokens {
            self.clear();
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: input.len(),
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: self.capacity_tokens,
            });
        }

        let compatibility = Compatibility::from_context(context);

        let compatible = self.compatibility.as_ref() == Some(&compatibility);
        let extends_retained_prefix = compatible && input.starts_with(&self.tokens);

        if !extends_retained_prefix {
            self.clear();
        }

        let reused_prefix_tokens = self.tokens.len();
        let suffix = &input[reused_prefix_tokens..];
        let processed_input_tokens = suffix.len();

        if !suffix.is_empty() {
            // The real Qwen3 method returns owned Vec<f32> logits.
            // Do not copy them through a slice or add an unnecessary
            // to_vec conversion.
            let logits: Vec<f32> = match self.model.prefill_cached(suffix, &mut self.kv) {
                Ok(logits) => logits,
                Err(error) => {
                    self.clear();
                    return Err(map_model_error(error, "cached prefill"));
                }
            };

            if let Err(error) = validate_logits(&logits, "cached prefill") {
                self.clear();
                return Err(error);
            }

            // Capacity was reserved in the constructor, and the complete
            // input was checked against capacity_tokens above.
            self.tokens.extend_from_slice(suffix);
            self.compatibility = Some(compatibility);
            self.last_logits = Some(logits);
        }

        if let Err(error) = self.check_consistency() {
            self.clear();
            return Err(error);
        }

        // Check before borrowing logits for the returned result so an
        // error can still invalidate the complete mutable cache.
        if self.last_logits.is_none() {
            self.clear();
            return Err(EngineError::Internal {
                message: "prefill completed without final-position logits".to_owned(),
            });
        }

        // The preceding checks establish that this is Some. Keep the
        // conversion fallible rather than introducing a panic.
        let logits = self
            .last_logits
            .as_deref()
            .ok_or_else(|| EngineError::Internal {
                message: "validated prefill logits are unavailable".to_owned(),
            })?;

        Ok(PrefillResult {
            logits,
            reused_prefix_tokens,
            processed_input_tokens,
        })
    }

    /// Process one actual next token and append its resulting K/V state.
    ///
    /// The token must be selected by generation from the preceding logits
    /// or supplied as an admitted continuation.
    ///
    /// This method does not sample tokens or decode token IDs into text.
    /// Every returned error invalidates this cache.
    pub fn decode_token(&mut self, token: u32) -> Result<&[f32], EngineError> {
        if let Err(error) = self.check_consistency() {
            self.clear();
            return Err(error);
        }

        if self.tokens.is_empty() {
            self.clear();
            return Err(EngineError::invalid_request(
                "cache",
                "prefill a context before decoding a continuation",
            ));
        }

        if self.tokens.len() >= self.capacity_tokens {
            let required = match self.tokens.len().checked_add(1) {
                Some(required) => required,
                None => {
                    self.clear();
                    return Err(EngineError::Internal {
                        message: "cache position overflow".to_owned(),
                    });
                }
            };

            self.clear();

            return Err(EngineError::ContextLimitExceeded {
                input_tokens: required,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: self.capacity_tokens,
            });
        }

        // Cached decode returns an owned Vec<f32>, matching last_logits.
        let logits: Vec<f32> = match self.model.decode_token(token, &mut self.kv) {
            Ok(logits) => logits,
            Err(error) => {
                self.clear();
                return Err(map_model_error(error, "cached decode"));
            }
        };

        if let Err(error) = validate_logits(&logits, "cached decode") {
            self.clear();
            return Err(error);
        }

        self.tokens.push(token);
        self.last_logits = Some(logits);

        if let Err(error) = self.check_consistency() {
            self.clear();
            return Err(error);
        }

        if self.last_logits.is_none() {
            self.clear();
            return Err(EngineError::Internal {
                message: "decode completed without final-position logits".to_owned(),
            });
        }

        self.last_logits
            .as_deref()
            .ok_or_else(|| EngineError::Internal {
                message: "validated decode logits are unavailable".to_owned(),
            })
    }

    fn check_consistency(&self) -> Result<(), EngineError> {
        let empty = self.tokens.is_empty();

        if self.kv.len() != self.tokens.len()
            || self.tokens.len() > self.capacity_tokens
            || empty != self.compatibility.is_none()
            || empty != self.last_logits.is_none()
        {
            return Err(EngineError::Internal {
                message: "retained token metadata disagrees with actual KV state".to_owned(),
            });
        }

        if let Some(compatibility) = &self.compatibility {
            if compatibility.artifact_id != self.artifact_id {
                return Err(EngineError::Internal {
                    message: "retained compatibility belongs to another artifact".to_owned(),
                });
            }
        }

        if let Some(logits) = &self.last_logits {
            validate_logits(logits, "retained logits")?;
        }

        Ok(())
    }
}

fn validate_logits(logits: &[f32], stage: &'static str) -> Result<(), EngineError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::ExecutionFailed {
            stage,
            message: "model returned empty or non-finite logits".to_owned(),
        });
    }

    Ok(())
}

fn map_model_error(error: ModelError, stage: &'static str) -> EngineError {
    match error {
        ModelError::MemoryLimit { required, limit } => {
            match (u64::try_from(required), u64::try_from(limit)) {
                (Ok(required_bytes), Ok(available_bytes)) => EngineError::MemoryLimitExceeded {
                    operation: stage,
                    required_bytes,
                    available_bytes,
                },
                _ => EngineError::Internal {
                    message: "model memory sizes cannot be represented as u64".to_owned(),
                },
            }
        }

        ModelError::AllocationFailed => EngineError::AllocationFailed {
            operation: stage,
            requested_bytes: None,
        },

        ModelError::ContextTooLong { maximum, got } => EngineError::ContextLimitExceeded {
            input_tokens: got,
            output_reserve_tokens: 0,
            safety_margin_tokens: 0,
            limit_tokens: maximum,
        },

        ModelError::EmptyInput => {
            EngineError::invalid_request("token_ids", "input must not be empty")
        }

        ModelError::InvalidToken {
            position,
            token,
            vocab_size,
        } => EngineError::invalid_request(
            "token_ids",
            format!(
                "token {token} at position {position} is outside \
                 vocabulary size {vocab_size}"
            ),
        ),

        ModelError::InvalidConfig(message) => EngineError::invalid_request("cache", message),

        ModelError::DimensionOverflow(name) => EngineError::Internal {
            message: format!("model dimension arithmetic overflow during {stage}: {name}"),
        },

        other => EngineError::ExecutionFailed {
            stage,
            message: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
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
}
