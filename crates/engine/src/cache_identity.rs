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
//!
//! Computing an identity does not allocate a cache, publish session state,
//! authorize reuse, or enable prefix-reuse capability reporting.

use crate::{
    api::ENGINE_API_VERSION,
    error::EngineError,
    registry::FamilyVersion,
    types::{ArtifactId, CacheRepresentationId, SessionStateId},
};

/// Borrowed inputs to canonical session-state identity construction.
///
/// Token-prefix bytes must come from the actual tokenizer integration.
/// This type does not verify how the caller obtained those bytes.
///
/// Profile and tracing identifiers are deliberately not added to the
/// Appendix B.1 field stream. Their relevant compatibility properties
/// must still be checked before any actual cache reuse.
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
    pub lora_id: Option<&'a str>,

    /// Exact tokenizer-produced byte stream for the retained token prefix.
    ///
    /// Do not substitute rendered text, generated display text, a render
    /// hash, or an independently chosen token-ID serialization.
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

/// Return the complete canonical serialized length in bytes.
///
/// Includes all six eight-byte length prefixes.
///
/// This does not serialize or hash the token-prefix payload and does not
/// acquire a memory reservation.
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

/// Visit exactly the framed canonical stream.
///
/// Preflight all lengths before passing any bytes to the sink.
fn visit_canonical_bytes(
    inputs: &CacheIdentityInputs<'_>,
    mut sink: impl FnMut(&[u8]),
) -> Result<u64, EngineError> {
    validate_inputs(inputs)?;

    let family_version = inputs.family_impl_version.to_string();

    let engine_version = inputs.engine_api_version.to_string();

    let lora = inputs.lora_id.unwrap_or("");

    let fields: [&[u8]; 6] = [
        inputs.artifact_id.as_str().as_bytes(),
        family_version.as_bytes(),
        lora.as_bytes(),
        inputs.token_prefix_bytes,
        inputs.cache_representation.as_str().as_bytes(),
        engine_version.as_bytes(),
    ];

    let mut total = 0_u64;

    for field in fields {
        let length = field_len(field)?;

        total = total
            .checked_add(std::mem::size_of::<u64>() as u64)
            .and_then(|value| value.checked_add(length))
            .ok_or_else(|| EngineError::Internal {
                message: "canonical cache identity length overflow".to_owned(),
            })?;
    }

    for field in fields {
        let length = field_len(field)?;

        sink(&length.to_le_bytes());
        sink(field);
    }

    Ok(total)
}

fn field_len(field: &[u8]) -> Result<u64, EngineError> {
    u64::try_from(field.len()).map_err(|_| EngineError::Internal {
        message: "cache identity field length cannot be represented as u64".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

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

        assert_eq!(inputs.engine_api_version, ENGINE_API_VERSION,);
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

        let expected_fields: [&[u8]; 6] = [
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

        assert_eq!(identity.as_bytes(), blake3::hash(&expected).as_bytes(),);
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
    fn empty_present_lora_is_rejected() {
        let artifact = artifact();
        let representation = representation();

        let mut inputs = inputs(&artifact, &representation, b"prefix");

        inputs.lora_id = Some("");

        let error = compute_session_state_id(&inputs).expect_err("ambiguous empty adapter");

        assert_eq!(error.kind(), ErrorKind::InvalidRequest,);
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

        // Unframed LoRA-plus-prefix bytes would both be "abc".
        assert_ne!(encoded(&first), encoded(&second),);

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

        assert_eq!(error.kind(), ErrorKind::InvalidRequest,);
    }
}
