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
