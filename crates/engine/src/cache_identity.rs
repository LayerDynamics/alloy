//! Canonical session-state identity construction.
//!
//! This module computes a SessionStateId from the Appendix B.1 fields:
//!
//! 1. Artifact identifier.
//! 2. Family implementation version.
//! 3. Optional LoRA identifier.
//! 4. Exact tokenizer-produced prefix bytes.
//! 5. Cache representation identifier.
//! 6. Engine API version.
//!
//! Each field is preceded by its byte length encoded as u64 little-endian.
//! The complete framed stream is hashed with ordinary, unkeyed BLAKE3.
//! There are no separators, terminators, JSON wrappers, or additional fields.
//!
//! Concrete encoding conventions:
//!
//! - Artifact and cache-representation identifiers use their UTF-8 bytes.
//! - Family versions use decimal "major.minor.patch" without extra padding.
//! - Engine API versions use ordinary decimal text without extra padding.
//! - No LoRA is represented by an empty field.
//! - A present LoRA identifier must be nonempty and uses its UTF-8 bytes.
//! - Token-prefix bytes are supplied exactly by the tokenizer integration.
//!   They are not reconstructed from rendered text or lossy decoded output.
//!   The prefix payload may contain arbitrary bytes.
//!
//! These conventions must remain pinned wherever identities are exchanged
//! or retained. An incompatible encoding change requires an explicit
//! compatibility/versioning decision.
//!
//! An identity is a lookup key, not proof that reusable KV state exists.
//! Before reuse, the cache owner must additionally establish:
//!
//! - The actual loaded-model ownership relationship.
//! - Verified artifact identity.
//! - Tokenizer, template, renderer, and profile compatibility.
//! - Exact token-ID prefix compatibility.
//! - Availability of the corresponding retained KV positions.
//! - Exclusive execution ownership or the applicable lease discipline.
//!
//! CacheIdentityInputs::for_retained_cache derives identity metadata from
//! actual native objects and validates their structural ownership
//! relationship. It does not verify artifact contents, tokenizer-byte
//! provenance, exact token-history compatibility, or memory admission.
//!
//! Computing or matching an identity does not allocate a native cache,
//! publish session state, acquire a lease, authorize reuse, release native
//! resources, or enable prefix-reuse capability reporting.
//!
//! The worker-owned state registry, reservations, publication commands,
//! lookup commands, leases, busy-state protection, and acknowledged native
//! destruction belong to the runtime, not this module.

use crate::{
    api::ENGINE_API_VERSION,
    error::EngineError,
    family::{FamilyCache, LoadedModel},
    registry::FamilyVersion,
    types::{ArtifactId, CacheRepresentationId, SessionStateId},
};

const FIELD_COUNT: usize = 6;
const LENGTH_PREFIX_BYTES: u64 = 8;

/// Borrowed inputs to canonical session-state identity construction.
///
/// Token-prefix bytes must come from the actual tokenizer integration.
/// This type does not verify how the caller obtained those bytes.
///
/// Profile and tracing identifiers are deliberately not added to the
/// Appendix B.1 field stream. Their relevant compatibility properties
/// must still be checked before actual cache reuse.
///
/// Public fields preserve the existing explicit-input API. Constructing
/// this value directly does not prove native ownership or state existence.
#[derive(Clone, Copy, Debug)]
pub struct CacheIdentityInputs<'a> {
    /// Verified artifact's immutable registry identity.
    ///
    /// A mutable directory path or an arbitrary unverified label is not
    /// evidence that artifact contents remain compatible.
    pub artifact_id: &'a ArtifactId,

    /// Version of the actual family implementation that produced the KV.
    pub family_impl_version: FamilyVersion,

    /// Identity of the applied LoRA adapter, or None when no adapter applies.
    ///
    /// Some("") is rejected because the empty field represents None.
    ///
    /// Supplying Some does not apply an adapter or prove that one was
    /// applied. The caller must obtain this value from execution metadata.
    pub lora_id: Option<&'a str>,

    /// Exact tokenizer-produced byte stream for the retained token prefix.
    ///
    /// Do not substitute rendered text, generated display text, a render
    /// hash, or an independently chosen token-ID serialization.
    ///
    /// The caller must establish that this payload corresponds to the
    /// exact executed token sequence associated with the retained state.
    pub token_prefix_bytes: &'a [u8],

    /// Identity of the actual retained KV representation and its semantics.
    pub cache_representation: &'a CacheRepresentationId,

    /// Engine contract version associated with this state.
    pub engine_api_version: u32,
}

impl<'a> CacheIdentityInputs<'a> {
    /// Construct inputs using this crate's current engine API version.
    ///
    /// Other inputs remain explicit. Validation occurs when computing
    /// the identity or its canonical serialized length.
    ///
    /// This constructor does not establish native ownership or prove
    /// that a corresponding retained allocation exists.
    pub fn for_current_engine(
        artifact_id: &'a ArtifactId,
        family_impl_version: FamilyVersion,
        lora_id: Option<&'a str>,
        token_prefix_bytes: &'a [u8],
        cache_representation: &'a CacheRepresentationId,
    ) -> Self {
        Self {
            artifact_id,
            family_impl_version,
            lora_id,
            token_prefix_bytes,
            cache_representation,
            engine_api_version: ENGINE_API_VERSION,
        }
    }

    /// Derive current-engine identity inputs from actual native objects.
    ///
    /// Validates that:
    ///
    /// - The cache belongs to the supplied loaded-model instance.
    /// - The cache representation agrees with its declared cache mode.
    /// - The actual model reports support for that cache mode.
    /// - Logical positions do not exceed the admitted cache capacity.
    /// - Physically retained positions do not exceed logical positions.
    ///
    /// Artifact identity and family implementation version come from the
    /// loaded model. Cache representation comes from the actual cache.
    ///
    /// This method does not verify artifact contents, apply or verify a
    /// LoRA adapter, validate tokenizer-byte provenance, compare exact
    /// executed token IDs, acquire admission, or establish lease ownership.
    ///
    /// Rotating caches may retain fewer physical positions than their
    /// logical length. That is not rejected here; the runtime must decide
    /// whether the required positions are actually available before reuse.
    ///
    /// The returned value borrows identity metadata from the supplied
    /// objects. It neither retains native ownership nor publishes state.
    pub fn for_retained_cache(
        model: &'a dyn LoadedModel,
        cache: &'a dyn FamilyCache,
        lora_id: Option<&'a str>,
        token_prefix_bytes: &'a [u8],
    ) -> Result<Self, EngineError> {
        validate_native_binding(model, cache)?;

        let inputs = Self::for_current_engine(
            model.artifact_id(),
            model.registration().implementation_version(),
            lora_id,
            token_prefix_bytes,
            cache.representation(),
        );

        validate_inputs(&inputs)?;
        Ok(inputs)
    }
}

/// Compute the canonical BLAKE3 session-state identity.
///
/// Input is fed incrementally into the hasher. No second allocation
/// proportional to the token-prefix length is constructed.
///
/// A successful result establishes only the identity of the supplied
/// fields. It does not establish that compatible execution state exists.
pub fn compute_session_state_id(
    inputs: &CacheIdentityInputs<'_>,
) -> Result<SessionStateId, EngineError> {
    let mut hasher = blake3::Hasher::new();

    visit_canonical_bytes(inputs, |bytes| {
        hasher.update(bytes);
    })?;

    Ok(SessionStateId::from_digest(*hasher.finalize().as_bytes()))
}

/// Check whether an identity matches the supplied canonical inputs.
///
/// Invalid inputs return an error rather than being reported as an
/// ordinary identity mismatch.
///
/// A true result proves only canonical digest agreement. It does not
/// prove native state existence, tokenizer provenance, prefix
/// compatibility, retained-position availability, or lease ownership.
pub fn matches_session_state_id(
    state_id: &SessionStateId,
    inputs: &CacheIdentityInputs<'_>,
) -> Result<bool, EngineError> {
    Ok(compute_session_state_id(inputs)? == *state_id)
}

/// Return the complete canonical serialized length in bytes.
///
/// Includes all six eight-byte length prefixes.
///
/// This does not serialize or hash the token-prefix payload and does not
/// acquire a memory reservation. Small version strings are formatted as
/// part of constructing the canonical fields.
pub fn canonical_serialized_len(inputs: &CacheIdentityInputs<'_>) -> Result<u64, EngineError> {
    visit_canonical_bytes(inputs, |_| {})
}

fn validate_inputs(inputs: &CacheIdentityInputs<'_>) -> Result<(), EngineError> {
    if inputs.lora_id.is_some_and(str::is_empty) {
        return Err(EngineError::invalid_request(
            "lora_id",
            "a present LoRA identifier must not be empty; use None for no adapter",
        ));
    }

    if inputs.engine_api_version == 0 {
        return Err(EngineError::invalid_request(
            "engine_api_version",
            "engine API version must be greater than zero",
        ));
    }

    Ok(())
}

fn validate_native_binding(
    model: &dyn LoadedModel,
    cache: &dyn FamilyCache,
) -> Result<(), EngineError> {
    if cache.owner() != model.instance_id() {
        return Err(internal(
            "canonical state cache belongs to a different loaded model",
        ));
    }

    if cache.representation() != &cache.mode().representation {
        return Err(internal(
            "canonical state cache representation disagrees with its mode",
        ));
    }

    if !model.capabilities().cache_modes.contains(cache.mode()) {
        return Err(internal(
            "canonical state cache mode is not supported by its loaded model",
        ));
    }

    if cache.len() > cache.capacity_tokens() {
        return Err(internal(
            "canonical state logical positions exceed cache capacity",
        ));
    }

    if cache.retained_tokens() > cache.len() {
        return Err(internal(
            "canonical state retained positions exceed logical positions",
        ));
    }

    Ok(())
}

/// Visit exactly the framed canonical stream.
///
/// All input validation and length arithmetic finish before any bytes
/// are passed to the sink. The second pass uses prevalidated lengths.
fn visit_canonical_bytes(
    inputs: &CacheIdentityInputs<'_>,
    mut sink: impl FnMut(&[u8]),
) -> Result<u64, EngineError> {
    validate_inputs(inputs)?;

    let family_version = inputs.family_impl_version.to_string();
    let engine_version = inputs.engine_api_version.to_string();
    let lora = inputs.lora_id.unwrap_or("");

    let fields: [&[u8]; FIELD_COUNT] = [
        inputs.artifact_id.as_str().as_bytes(),
        family_version.as_bytes(),
        lora.as_bytes(),
        inputs.token_prefix_bytes,
        inputs.cache_representation.as_str().as_bytes(),
        engine_version.as_bytes(),
    ];

    let mut lengths = [0_u64; FIELD_COUNT];
    let mut total = 0_u64;

    for (index, field) in fields.iter().enumerate() {
        let length = field_len(field)?;
        lengths[index] = length;
        total = checked_framed_len(total, length)?;
    }

    for (field, length) in fields.into_iter().zip(lengths) {
        sink(&length.to_le_bytes());
        sink(field);
    }

    Ok(total)
}

fn checked_framed_len(total: u64, field_length: u64) -> Result<u64, EngineError> {
    total
        .checked_add(LENGTH_PREFIX_BYTES)
        .and_then(|value| value.checked_add(field_length))
        .ok_or_else(|| internal("canonical cache identity length overflow"))
}

fn field_len(field: &[u8]) -> Result<u64, EngineError> {
    u64::try_from(field.len())
        .map_err(|_| internal("cache identity field length cannot be represented as u64"))
}

fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::ErrorKind,
        family::{BufferMemoryUsage, ModelInstanceId},
        registry::FamilyRegistration,
        types::{
            CacheKind, CacheMode, Capabilities, DecodingCapability, ExecutionBackend,
            WeightRepresentation,
        },
    };
    use std::{any::Any, num::NonZeroUsize};

    fn artifact() -> ArtifactId {
        ArtifactId::new("artifact-a").expect("artifact identity")
    }

    fn representation() -> CacheRepresentationId {
        CacheRepresentationId::new("kv.full.v1").expect("cache representation")
    }

    fn inputs<'a>(
        artifact: &'a ArtifactId,
        representation: &'a CacheRepresentationId,
        prefix: &'a [u8],
    ) -> CacheIdentityInputs<'a> {
        CacheIdentityInputs::for_current_engine(
            artifact,
            FamilyVersion::new(1, 2, 3),
            None,
            prefix,
            representation,
        )
    }

    fn encoded(inputs: &CacheIdentityInputs<'_>) -> Vec<u8> {
        let mut output = Vec::new();

        let reported_len = visit_canonical_bytes(inputs, |bytes| {
            output.extend_from_slice(bytes);
        })
        .expect("canonical encoding");

        assert_eq!(
            u64::try_from(output.len()).expect("encoded length"),
            reported_len,
        );

        output
    }

    #[test]
    fn current_engine_constructor_uses_api_constant() {
        let artifact = artifact();
        let representation = representation();
        let inputs = inputs(&artifact, &representation, b"prefix");

        assert_eq!(inputs.engine_api_version, ENGINE_API_VERSION);
    }

    #[test]
    fn canonical_encoding_has_exact_field_order_and_framing() {
        let artifact = artifact();
        let representation = representation();

        let inputs = CacheIdentityInputs {
            artifact_id: &artifact,
            family_impl_version: FamilyVersion::new(1, 2, 3),
            lora_id: Some("adapter-a"),
            token_prefix_bytes: b"abc",
            cache_representation: &representation,
            engine_api_version: 7,
        };

        let expected_fields: [&[u8]; FIELD_COUNT] = [
            b"artifact-a",
            b"1.2.3",
            b"adapter-a",
            b"abc",
            b"kv.full.v1",
            b"7",
        ];

        let mut expected = Vec::new();
        for field in expected_fields {
            expected.extend_from_slice(&(field.len() as u64).to_le_bytes());
            expected.extend_from_slice(field);
        }

        assert_eq!(encoded(&inputs), expected);
        assert_eq!(
            canonical_serialized_len(&inputs).expect("length"),
            expected.len() as u64,
        );

        let identity = compute_session_state_id(&inputs).expect("identity");
        assert_eq!(identity.as_bytes(), blake3::hash(&expected).as_bytes());
    }

    #[test]
    fn identical_inputs_produce_identical_ids() {
        let artifact = artifact();
        let representation = representation();
        let inputs = inputs(&artifact, &representation, b"exact-prefix");

        assert_eq!(
            compute_session_state_id(&inputs).expect("first"),
            compute_session_state_id(&inputs).expect("second"),
        );
    }

    #[test]
    fn changing_artifact_changes_identity() {
        let artifact = artifact();
        let other = ArtifactId::new("artifact-b").expect("other artifact");
        let representation = representation();

        let original = inputs(&artifact, &representation, b"prefix");
        let mut changed = original;
        changed.artifact_id = &other;

        assert_ne!(
            compute_session_state_id(&original).expect("original"),
            compute_session_state_id(&changed).expect("changed"),
        );
    }

    #[test]
    fn changing_family_version_changes_identity() {
        let artifact = artifact();
        let representation = representation();

        let original = inputs(&artifact, &representation, b"prefix");
        let mut changed = original;
        changed.family_impl_version = FamilyVersion::new(1, 2, 4);

        assert_ne!(
            compute_session_state_id(&original).expect("original"),
            compute_session_state_id(&changed).expect("changed"),
        );
    }

    #[test]
    fn changing_lora_changes_identity() {
        let artifact = artifact();
        let representation = representation();

        let original = inputs(&artifact, &representation, b"prefix");
        let mut changed = original;
        changed.lora_id = Some("adapter-a");

        assert_ne!(
            compute_session_state_id(&original).expect("original"),
            compute_session_state_id(&changed).expect("changed"),
        );
    }

    #[test]
    fn absent_lora_is_distinct_from_literal_null_identifier() {
        let artifact = artifact();
        let representation = representation();

        let absent = inputs(&artifact, &representation, b"prefix");
        let mut present = absent;
        present.lora_id = Some("null");

        assert_ne!(
            compute_session_state_id(&absent).expect("absent"),
            compute_session_state_id(&present).expect("present"),
        );
    }

    #[test]
    fn absent_lora_is_encoded_as_an_empty_field() {
        let artifact = artifact();
        let representation = representation();
        let inputs = inputs(&artifact, &representation, b"prefix");
        let bytes = encoded(&inputs);

        let artifact_field = 8 + artifact.as_str().len();
        let version_field = 8 + b"1.2.3".len();
        let lora_offset = artifact_field + version_field;

        assert_eq!(&bytes[lora_offset..lora_offset + 8], &0_u64.to_le_bytes(),);
    }

    #[test]
    fn empty_present_lora_is_rejected() {
        let artifact = artifact();
        let representation = representation();
        let mut inputs = inputs(&artifact, &representation, b"prefix");
        inputs.lora_id = Some("");

        let error = compute_session_state_id(&inputs).expect_err("ambiguous empty adapter");
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    }

    #[test]
    fn changing_exact_prefix_changes_identity() {
        let artifact = artifact();
        let representation = representation();

        let original = inputs(&artifact, &representation, b"prefix");
        let mut changed = original;
        changed.token_prefix_bytes = b"prefix!";

        assert_ne!(
            compute_session_state_id(&original).expect("original"),
            compute_session_state_id(&changed).expect("changed"),
        );
    }

    #[test]
    fn arbitrary_tokenizer_bytes_are_not_lossily_reencoded() {
        let artifact = artifact();
        let representation = representation();

        let binary = inputs(&artifact, &representation, &[0x00, 0xff, 0x80]);
        let replacement_text = inputs(&artifact, &representation, "\0\u{fffd}\u{fffd}".as_bytes());

        assert_ne!(
            compute_session_state_id(&binary).expect("binary prefix"),
            compute_session_state_id(&replacement_text).expect("replacement text"),
        );
    }

    #[test]
    fn changing_cache_representation_changes_identity() {
        let artifact = artifact();
        let representation = representation();
        let other = CacheRepresentationId::new("kv.full.v2").expect("other representation");

        let original = inputs(&artifact, &representation, b"prefix");
        let mut changed = original;
        changed.cache_representation = &other;

        assert_ne!(
            compute_session_state_id(&original).expect("original"),
            compute_session_state_id(&changed).expect("changed"),
        );
    }

    #[test]
    fn changing_engine_version_changes_identity() {
        let artifact = artifact();
        let representation = representation();

        let original = inputs(&artifact, &representation, b"prefix");
        let mut changed = original;
        changed.engine_api_version = ENGINE_API_VERSION.checked_add(1).expect("next version");

        assert_ne!(
            compute_session_state_id(&original).expect("original"),
            compute_session_state_id(&changed).expect("changed"),
        );
    }

    #[test]
    fn field_lengths_prevent_boundary_ambiguity() {
        let artifact = artifact();
        let representation = representation();

        let mut first = inputs(&artifact, &representation, b"c");
        first.lora_id = Some("ab");

        let mut second = inputs(&artifact, &representation, b"bc");
        second.lora_id = Some("a");

        assert_ne!(encoded(&first), encoded(&second));
        assert_ne!(
            compute_session_state_id(&first).expect("first"),
            compute_session_state_id(&second).expect("second"),
        );
    }

    #[test]
    fn empty_prefix_has_a_valid_distinct_identity() {
        let artifact = artifact();
        let representation = representation();

        let empty = inputs(&artifact, &representation, b"");
        let nonempty = inputs(&artifact, &representation, b"x");

        assert_ne!(
            compute_session_state_id(&empty).expect("empty prefix"),
            compute_session_state_id(&nonempty).expect("nonempty prefix"),
        );
    }

    #[test]
    fn zero_engine_api_version_is_rejected() {
        let artifact = artifact();
        let representation = representation();
        let mut inputs = inputs(&artifact, &representation, b"prefix");
        inputs.engine_api_version = 0;

        let error = compute_session_state_id(&inputs).expect_err("invalid API version");
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    }

    #[test]
    fn matching_identity_returns_true() {
        let artifact = artifact();
        let representation = representation();
        let inputs = inputs(&artifact, &representation, b"prefix");
        let id = compute_session_state_id(&inputs).expect("identity");

        assert!(matches_session_state_id(&id, &inputs).expect("match"));
    }

    #[test]
    fn different_identity_returns_false() {
        let artifact = artifact();
        let representation = representation();
        let original = inputs(&artifact, &representation, b"prefix");
        let id = compute_session_state_id(&original).expect("identity");

        let changed = inputs(&artifact, &representation, b"other-prefix");
        assert!(!matches_session_state_id(&id, &changed).expect("mismatch"));
    }

    #[test]
    fn matching_does_not_hide_invalid_inputs() {
        let artifact = artifact();
        let representation = representation();
        let mut inputs = inputs(&artifact, &representation, b"prefix");
        let id = compute_session_state_id(&inputs).expect("identity");
        inputs.engine_api_version = 0;

        let error = matches_session_state_id(&id, &inputs)
            .expect_err("invalid inputs are not an ordinary mismatch");
        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
    }

    #[test]
    fn invalid_inputs_emit_no_canonical_bytes() {
        let artifact = artifact();
        let representation = representation();
        let mut inputs = inputs(&artifact, &representation, b"prefix");
        inputs.lora_id = Some("");

        let mut calls = 0;
        let result = visit_canonical_bytes(&inputs, |_| {
            calls += 1;
        });

        assert!(result.is_err());
        assert_eq!(calls, 0);
        assert!(canonical_serialized_len(&inputs).is_err());
    }

    #[test]
    fn canonical_length_counts_utf8_bytes_not_characters() {
        let artifact = artifact();
        let representation = representation();
        let mut inputs = inputs(&artifact, &representation, &[0x00, 0xff]);
        inputs.lora_id = Some("adapter-\u{03bb}");

        let expected = 6 * 8
            + artifact.as_str().len()
            + b"1.2.3".len()
            + "adapter-\u{03bb}".len()
            + 2
            + representation.as_str().len()
            + ENGINE_API_VERSION.to_string().len();

        assert_eq!(
            canonical_serialized_len(&inputs).expect("length"),
            expected as u64,
        );
    }

    #[test]
    fn framed_length_arithmetic_rejects_overflow() {
        assert!(checked_framed_len(u64::MAX, 0).is_err());
        assert!(checked_framed_len(0, u64::MAX).is_err());
        assert_eq!(
            checked_framed_len(0, u64::MAX - 8).expect("largest framed length"),
            u64::MAX,
        );
    }

    fn cache_mode() -> CacheMode {
        CacheMode {
            representation: representation(),
            kind: CacheKind::Full,
        }
    }

    fn capabilities(mode: CacheMode) -> Capabilities {
        Capabilities {
            task: "text-generation".to_owned(),
            family: "identity-test".to_owned(),
            backend: ExecutionBackend::HostCpu,
            weight_representations: vec![WeightRepresentation::F32],
            cache_modes: vec![mode],
            context_limit_tokens: 32,
            supports_streaming: false,
            supports_incremental_kv: true,
            supports_prefix_reuse: false,
            decoding_modes: vec![DecodingCapability::Greedy],
        }
    }

    /// Minimal native-object test double.
    ///
    /// Registration is intentionally unavailable: tests using this double
    /// exercise native-binding validation, which must not load or execute
    /// a model or require registration access.
    struct TestModel {
        instance: ModelInstanceId,
        artifact: ArtifactId,
        capabilities: Capabilities,
    }

    impl LoadedModel for TestModel {
        fn instance_id(&self) -> &ModelInstanceId {
            &self.instance
        }

        fn artifact_id(&self) -> &ArtifactId {
            &self.artifact
        }

        fn registration(&self) -> &FamilyRegistration {
            panic!("native binding validation must not access registration")
        }

        fn capabilities(&self) -> &Capabilities {
            &self.capabilities
        }

        fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
            panic!("native binding validation must not inspect admission")
        }

        fn autoregressive(&mut self) -> Option<&mut dyn crate::family::AutoregressiveModel> {
            None
        }
    }

    struct TestCache {
        owner: ModelInstanceId,
        representation: CacheRepresentationId,
        mode: CacheMode,
        logical: usize,
        retained: usize,
        capacity: usize,
    }

    impl FamilyCache for TestCache {
        fn owner(&self) -> &ModelInstanceId {
            &self.owner
        }

        fn representation(&self) -> &CacheRepresentationId {
            &self.representation
        }

        fn mode(&self) -> &CacheMode {
            &self.mode
        }

        fn len(&self) -> usize {
            self.logical
        }

        fn retained_tokens(&self) -> usize {
            self.retained
        }

        fn capacity_tokens(&self) -> usize {
            self.capacity
        }

        fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
            panic!("native binding validation must not inspect admission")
        }

        fn supports_truncation(&self) -> bool {
            false
        }

        fn reset(&mut self) -> Result<(), EngineError> {
            panic!("identity validation must not mutate native state")
        }

        fn truncate(&mut self, _token_count: usize) -> Result<(), EngineError> {
            panic!("identity validation must not mutate native state")
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn native_pair() -> (TestModel, TestCache) {
        let instance = ModelInstanceId::new();
        let mode = cache_mode();

        let model = TestModel {
            instance: instance.clone(),
            artifact: artifact(),
            capabilities: capabilities(mode.clone()),
        };

        let cache = TestCache {
            owner: instance,
            representation: mode.representation.clone(),
            mode,
            logical: 4,
            retained: 4,
            capacity: 16,
        };

        (model, cache)
    }

    #[test]
    fn native_binding_accepts_actual_owner_and_supported_layout() {
        let (model, cache) = native_pair();
        validate_native_binding(&model, &cache).expect("valid native binding");

        assert_eq!(cache.logical, 4);
        assert_eq!(cache.retained, 4);
    }

    #[test]
    fn native_binding_rejects_foreign_model_instance() {
        let (model, mut cache) = native_pair();
        cache.owner = ModelInstanceId::new();

        assert!(validate_native_binding(&model, &cache).is_err());
        assert!(CacheIdentityInputs::for_retained_cache(&model, &cache, None, b"prefix",).is_err());
    }

    #[test]
    fn native_binding_rejects_representation_mode_disagreement() {
        let (model, mut cache) = native_pair();
        cache.representation = CacheRepresentationId::new("kv.other.v1").expect("other layout");

        assert!(validate_native_binding(&model, &cache).is_err());
    }

    #[test]
    fn native_binding_rejects_unsupported_cache_mode() {
        let (mut model, cache) = native_pair();
        model.capabilities.cache_modes.clear();

        assert!(validate_native_binding(&model, &cache).is_err());
    }

    #[test]
    fn native_binding_rejects_logical_length_over_capacity() {
        let (model, mut cache) = native_pair();
        cache.logical = cache.capacity + 1;

        assert!(validate_native_binding(&model, &cache).is_err());
    }

    #[test]
    fn native_binding_rejects_retained_length_over_logical_length() {
        let (model, mut cache) = native_pair();
        cache.retained = cache.logical + 1;

        assert!(validate_native_binding(&model, &cache).is_err());
    }

    #[test]
    fn rotating_cache_binding_does_not_claim_full_prefix_availability() {
        let (mut model, mut cache) = native_pair();
        let mode = CacheMode {
            representation: CacheRepresentationId::new("kv.rotating.v1")
                .expect("rotating representation"),
            kind: CacheKind::Rotating {
                window_tokens: NonZeroUsize::new(2).expect("window"),
            },
        };

        model.capabilities.cache_modes = vec![mode.clone()];
        cache.representation = mode.representation.clone();
        cache.mode = mode;
        cache.logical = 8;
        cache.retained = 2;

        validate_native_binding(&model, &cache).expect("valid rotating ownership");

        assert!(cache.retained_tokens() < cache.len());
    }

    #[test]
    fn cloned_instance_identity_is_the_same_native_owner() {
        let (model, mut cache) = native_pair();
        cache.owner = model.instance_id().clone();

        validate_native_binding(&model, &cache)
            .expect("cloned ownership identity denotes same instance");
    }
}
