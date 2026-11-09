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
#[path = "../tests/unit/cache_identity_tests.rs"]
mod tests;
