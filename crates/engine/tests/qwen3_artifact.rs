//! Public-API integration tests for actual Qwen3 artifact handling.
//!
//! The fixture is generated locally from deterministic Qwen3 weights and
//! serialized into a real safetensors checkpoint.
//!
//! Tests cover:
//!
//! - Valid pinned artifact construction and registry resolution.
//! - Real verified loading.
//! - Cached prefill/decode consistency against full forward execution.
//! - Actual cache allocation reporting.
//! - Digest and length mismatch rejection.
//! - Verification of bytes loaded after an earlier verification pass.
//! - Pinned metadata mutation rejection.
//! - Invalid template pins and unsafe component names.
//! - Indexed checkpoint inventory validation.
//! - Scheduler-compatible cancellation observation.
//!
//! This suite does not establish production-checkpoint conformance,
//! independent reference-logit accuracy, Metal execution, quantized
//! residency, or generated-text quality.
//!
//! The tokenizer is a real, small BPE/ByteLevel fixture. It is not a
//! production Qwen3 tokenizer or a claim of production tokenizer parity.

use engine::{
    backends::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits},
    cancellation::CancellationSource,
    error::{EngineError, ErrorKind},
    family::{ExecutionControl, LoadedModel, ModelFamily, ModelLoadRequest},
    generation::GenerationLimits,
    registry::{ModelProfile, Registry},
    runtime::{RuntimeBackend, RuntimePlan},
    types::{
        ArtifactId, ExecutionBackend, GenerationOptions, ModelProfileId, RequestId,
        WeightRepresentation,
    },
};
use qwen3::{
    config::Qwen3Config,
    fixture::{Qwen3Weights, generate_fixture},
    model::Qwen3Model,
    template::CHAT_TEMPLATE,
};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    future::Future,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, pre_tokenizers::byte_level::ByteLevel};

const OUTPUT_TOKENS: usize = 4;
const SAFETY_MARGIN_TOKENS: usize = 8;
const IM_START_ID: u32 = 256;
const IM_END_ID: u32 = 257;

/// Own only directories created by this test process.
struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);

        loop {
            let path = std::env::temp_dir().join(format!(
                "alloy-qwen-artifact-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));

            match std::fs::create_dir(&path) {
                Ok(()) => {
                    return Self(path.canonicalize().expect("canonical test directory"));
                }

                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    continue;
                }

                Err(error) => {
                    panic!("create test directory: {error}");
                }
            }
        }
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ThreadWake {
    thread: thread::Thread,
}

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

/// Poll actual futures without introducing an async-runtime dependency.
///
/// The timeout is a test deadline, not simulated execution.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);

    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));

    let mut context = Context::from_waker(&waker);

    let deadline = Instant::now()
        .checked_add(Duration::from_secs(30))
        .expect("test deadline");

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,

            Poll::Pending => {
                let now = Instant::now();

                assert!(now < deadline, "artifact test future exceeded its deadline",);

                thread::park_timeout(deadline.saturating_duration_since(now));
            }
        }
    }
}

fn expect_failure<T>(result: Result<T, EngineError>, message: &str) -> EngineError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

fn artifact_id() -> ArtifactId {
    ArtifactId::new("qwen3-artifact-fixture").expect("artifact ID")
}

fn profile_id() -> ModelProfileId {
    ModelProfileId::new("qwen3-artifact-profile").expect("profile ID")
}

fn control(name: &str) -> (CancellationSource, ExecutionControl) {
    let source = CancellationSource::new();

    let control = ExecutionControl::new(RequestId::new(name).expect("request ID"), source.token());

    (source, control)
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn backend_limits() -> Qwen3BackendLimits {
    Qwen3BackendLimits {
        max_files: 16,
        max_config_bytes: 1024 * 1024,
        max_tokenizer_bytes: 1024 * 1024,
        max_messages: 16,
        max_input_bytes: 4096,
        max_rendered_bytes: 16 * 1024,
        max_token_bytes: 128,
        verification_chunk_bytes: 4096,
        context_limit_tokens: 512,
        prefill_chunk_tokens: NonZeroUsize::new(2).expect("prefill chunk"),
        loader_headroom_bytes: 64 * 1024,
        execution_headroom_bytes: 64 * 1024,
        generation: GenerationLimits::new(8, 4096, 1024).expect("generation limits"),
    }
}

struct TensorRecord {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn push_tensor(records: &mut Vec<TensorRecord>, name: String, shape: Vec<usize>, values: &[f32]) {
    let expected = shape
        .iter()
        .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension))
        .expect("fixture tensor element count");

    assert_eq!(values.len(), expected, "fixture shape mismatch for {name}",);

    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    records.push(TensorRecord { name, shape, bytes });
}

fn write_checkpoint(directory: &Path, config: &Qwen3Config, weights: &Qwen3Weights) -> Vec<String> {
    let mut records = Vec::new();

    push_tensor(
        &mut records,
        "model.embed_tokens.weight".to_owned(),
        vec![config.vocab_size, config.hidden_size],
        &weights.embed_tokens,
    );

    for (index, layer) in weights.layers.iter().enumerate() {
        let prefix = format!("model.layers.{index}");

        push_tensor(
            &mut records,
            format!("{prefix}.input_layernorm.weight"),
            vec![config.hidden_size],
            &layer.input_layernorm,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![config.q_dim(), config.hidden_size],
            &layer.q_proj,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![config.kv_dim(), config.hidden_size],
            &layer.k_proj,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![config.kv_dim(), config.hidden_size],
            &layer.v_proj,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![config.head_dim],
            &layer.q_norm,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![config.head_dim],
            &layer.k_norm,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![config.hidden_size, config.q_dim()],
            &layer.o_proj,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![config.hidden_size],
            &layer.post_attention_layernorm,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![config.intermediate_size, config.hidden_size],
            &layer.gate_proj,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.mlp.up_proj.weight"),
            vec![config.intermediate_size, config.hidden_size],
            &layer.up_proj,
        );

        push_tensor(
            &mut records,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![config.hidden_size, config.intermediate_size],
            &layer.down_proj,
        );
    }

    push_tensor(
        &mut records,
        "model.norm.weight".to_owned(),
        vec![config.hidden_size],
        &weights.final_layernorm,
    );

    if let Some(head) = &weights.lm_head {
        push_tensor(
            &mut records,
            "lm_head.weight".to_owned(),
            vec![config.vocab_size, config.hidden_size],
            head,
        );
    }

    let names = records.iter().map(|record| record.name.clone()).collect();

    let views: Vec<_> = records
        .iter()
        .map(|record| {
            (
                record.name.as_str(),
                TensorView::new(Dtype::F32, record.shape.clone(), &record.bytes)
                    .expect("valid fixture tensor"),
            )
        })
        .collect();

    let bytes = serialize(views, &None).expect("serialize actual safetensors");

    std::fs::write(directory.join("model.safetensors"), bytes).expect("write checkpoint");

    names
}

fn tokenizer_json() -> Vec<u8> {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();

    alphabet.sort_unstable();

    assert_eq!(alphabet.len(), 256);

    let mut vocabulary = Map::new();

    for (index, character) in alphabet.into_iter().enumerate() {
        vocabulary.insert(character.to_string(), Value::from(index as u64));
    }

    vocabulary.insert("<|im_start|>".to_owned(), Value::from(IM_START_ID));

    vocabulary.insert("<|im_end|>".to_owned(), Value::from(IM_END_ID));

    let source = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {
                "id": IM_START_ID,
                "content": "<|im_start|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            },
            {
                "id": IM_END_ID,
                "content": "<|im_end|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocabulary,
            "merges": []
        }
    });

    let bytes = serde_json::to_vec(&source).expect("tokenizer JSON");

    // Validate the fixture using the actual tokenizer library before
    // handing it to the backend's independent construction path.
    let tokenizer = Tokenizer::from_bytes(&bytes).expect("actual BPE/ByteLevel tokenizer");

    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(IM_START_ID),);

    assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(IM_END_ID),);

    bytes
}

struct ArtifactFixture {
    directory: TestDirectory,
    config: Qwen3Config,
    specification: Qwen3Artifact,
    tensor_names: Vec<String>,
}

impl ArtifactFixture {
    fn new() -> Self {
        let directory = TestDirectory::new();

        let mut config = Qwen3Config::tiny_fixture();

        // Preserve the tiny architecture while providing one vocabulary
        // entry for every byte-alphabet symbol plus both special tokens.
        // This is not the unchanged canonical reference-logit fixture.
        config.vocab_size = 258;

        let weights = generate_fixture(&config);

        let tensor_names = write_checkpoint(directory.path(), &config, &weights);

        let config_json = json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_layers,
            "num_attention_heads": config.q_heads,
            "num_key_value_heads": config.kv_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_context,
            "tie_word_embeddings": config.tie_embeddings,
            "hidden_act": "silu",
            "rms_norm_eps": 0.000001,
            "attention_bias": false,
            "rope_scaling": null,
            "torch_dtype": "float32",
            "eos_token_id": IM_END_ID
        });

        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config_json).expect("configuration JSON"),
        )
        .expect("write configuration");

        std::fs::write(directory.path().join("tokenizer.json"), tokenizer_json())
            .expect("write tokenizer");

        let mut files = BTreeMap::new();

        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            let bytes =
                std::fs::read(directory.path().join(name)).expect("read known fixture bytes");

            files.insert(
                name.to_owned(),
                FilePin {
                    bytes: bytes.len() as u64,
                    sha256: digest(&bytes),
                },
            );
        }

        let specification = Qwen3Artifact {
            id: artifact_id(),
            directory: directory.path().to_path_buf(),
            files,
            template_sha256: digest(CHAT_TEMPLATE.as_bytes()),
        };

        Self {
            directory,
            config,
            specification,
            tensor_names,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }

    fn backend(&self) -> Qwen3Backend {
        Qwen3Backend::new(vec![self.specification.clone()], backend_limits())
            .expect("actual Qwen3 backend")
    }

    fn registry(&self, backend: &Qwen3Backend) -> Registry {
        let profile = ModelProfile::new(
            profile_id(),
            artifact_id(),
            self.config.max_context,
            OUTPUT_TOKENS,
            SAFETY_MARGIN_TOKENS,
            GenerationOptions::greedy(OUTPUT_TOKENS).expect("greedy options"),
        )
        .expect("profile");

        backend
            .registry(vec![profile])
            .expect("artifact and profile registration")
    }

    fn plan(
        &self,
        backend: &Qwen3Backend,
        registry: &Registry,
        input_tokens: usize,
    ) -> RuntimePlan {
        let profile = profile_id();

        block_on(backend.plan(registry, &profile, input_tokens)).expect("actual backend plan")
    }

    fn write_and_repin(&mut self, name: &str, bytes: &[u8]) {
        std::fs::write(self.path(name), bytes).expect("write fixture component");

        self.specification.files.insert(
            name.to_owned(),
            FilePin {
                bytes: bytes.len() as u64,
                sha256: digest(bytes),
            },
        );
    }

    fn write_index(&mut self, omit_last_tensor: bool) {
        let mut weight_map = Map::new();

        let count = if omit_last_tensor {
            self.tensor_names
                .len()
                .checked_sub(1)
                .expect("nonempty tensor inventory")
        } else {
            self.tensor_names.len()
        };

        for name in self.tensor_names.iter().take(count) {
            weight_map.insert(name.clone(), Value::String("model.safetensors".to_owned()));
        }

        let index = serde_json::to_vec(&json!({
            "weight_map": weight_map
        }))
        .expect("index JSON");

        self.write_and_repin("model.safetensors.index.json", &index);
    }
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

fn load_fixture(
    backend: &Qwen3Backend,
    registry: &Registry,
    plan: &RuntimePlan,
    control: &ExecutionControl,
) -> Result<Box<dyn LoadedModel>, EngineError> {
    let artifact = registry
        .artifact(&artifact_id())
        .expect("registered artifact");

    backend.load(ModelLoadRequest::new(artifact, plan.load_limits, control))
}

#[test]
fn pinned_fixture_registers_the_real_family_and_profile() {
    let fixture = ArtifactFixture::new();
    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let profile = profile_id();

    let resolved = registry
        .resolve_profile(&profile)
        .expect("resolved profile");

    assert_eq!(resolved.artifact().id(), &artifact_id(),);

    assert_eq!(resolved.artifact().directory(), fixture.directory.path(),);

    assert_eq!(resolved.family().family(), backend.registration().family(),);

    assert_eq!(
        resolved.profile().context_limit_tokens(),
        fixture.config.max_context,
    );
}

#[test]
fn verified_fixture_loads_and_cached_execution_matches_full_forward() {
    let fixture = ArtifactFixture::new();
    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let prompt = [65_u32, 66, 67];

    let plan = fixture.plan(&backend, &registry, prompt.len());

    let (_source, control) = control("artifact-real-execution");

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    backend
        .verify_artifact(artifact, &control)
        .expect("actual file verification");

    let mut loaded =
        load_fixture(&backend, &registry, &plan, &control).expect("actual verified model load");

    assert_eq!(loaded.artifact_id(), &artifact_id(),);

    assert_eq!(loaded.capabilities().backend, ExecutionBackend::HostCpu,);

    assert!(
        loaded
            .capabilities()
            .weight_representations
            .contains(&WeightRepresentation::F32),
    );

    let model_memory = loaded.memory_usage().expect("resident memory report");

    assert!(model_memory.allocated_bytes() > 0);

    assert!(model_memory.used_bytes() <= model_memory.allocated_bytes(),);

    // The reference is another real model using the same deterministic
    // weights. This checks cached/full execution consistency, not an
    // independent external numerical conformance vector.
    let reference =
        Qwen3Model::from_weights(fixture.config.clone(), generate_fixture(&fixture.config))
            .expect("reference full-forward model");

    let model = loaded
        .autoregressive()
        .expect("actual autoregressive interface");

    assert_eq!(model.vocabulary_size(), fixture.config.vocab_size,);

    let mut cache = model
        .allocate_cache(&plan.cache, &control)
        .expect("actual cache allocation");

    assert_eq!(cache.owner(), model.instance_id(),);

    assert_eq!(cache.len(), 0);

    let empty_memory = cache.memory_usage().expect("empty cache accounting");

    assert!(empty_memory.allocated_bytes() > 0);

    let prefill = model
        .prefill(&prompt, cache.as_mut(), &control)
        .expect("actual cached prefill");

    assert_eq!(prefill.processed_tokens(), prompt.len(),);

    assert_eq!(cache.len(), prompt.len(),);

    let full = reference.forward(&prompt).expect("actual full forward");

    assert_close(
        prefill.logits().values(),
        full.last().expect("final prompt logits"),
    );

    let continuation = 68_u32;

    let decoded = model
        .decode(continuation, cache.as_mut(), &control)
        .expect("actual cached decode");

    assert_eq!(cache.len(), prompt.len() + 1,);

    let full_continuation = reference
        .forward(&[65, 66, 67, continuation])
        .expect("full continuation forward");

    assert_close(
        decoded.values(),
        full_continuation.last().expect("final continuation logits"),
    );

    let populated_memory = cache.memory_usage().expect("populated cache accounting");

    assert_eq!(
        populated_memory.allocated_bytes(),
        empty_memory.allocated_bytes(),
    );

    assert!(populated_memory.used_bytes() > empty_memory.used_bytes(),);

    cache.reset().expect("actual cache reset");

    assert_eq!(cache.len(), 0);

    let reset_memory = cache.memory_usage().expect("reset cache accounting");

    assert_eq!(
        reset_memory.allocated_bytes(),
        empty_memory.allocated_bytes(),
    );

    assert_eq!(reset_memory.used_bytes(), empty_memory.used_bytes(),);
}

#[test]
fn changed_weight_bytes_are_rejected_after_prior_verification() {
    let fixture = ArtifactFixture::new();
    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let plan = fixture.plan(&backend, &registry, 3);

    let (_source, control) = control("artifact-changed-after-verification");

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    backend
        .verify_artifact(artifact, &control)
        .expect("initial verification");

    let path = fixture.path("model.safetensors");

    let mut bytes = std::fs::read(&path).expect("checkpoint bytes");

    let original_length = bytes.len();

    // Change the least-significant byte of the final f32 payload value.
    // The file length and safetensors structure remain unchanged.
    let position = bytes.len().checked_sub(4).expect("checkpoint payload");

    bytes[position] ^= 1;

    std::fs::write(&path, &bytes).expect("mutate checkpoint");

    assert_eq!(bytes.len(), original_length);

    // Deliberately do not rerun verify_artifact here. The loader must
    // independently verify the exact bytes it consumes.
    let error = expect_failure(
        load_fixture(&backend, &registry, &plan, &control),
        "loader accepted bytes changed after verification",
    );

    assert_ne!(error.kind(), ErrorKind::Cancelled,);

    let verification_error = backend
        .verify_artifact(artifact, &control)
        .expect_err("new verification must reject mutation");

    assert_eq!(
        verification_error.kind(),
        ErrorKind::ArtifactValidationFailed,
    );
}

#[test]
fn changed_weight_length_is_rejected() {
    let fixture = ArtifactFixture::new();
    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let (_source, control) = control("artifact-length-change");

    let path = fixture.path("model.safetensors");

    let mut bytes = std::fs::read(&path).expect("checkpoint bytes");

    bytes.push(0);

    std::fs::write(&path, bytes).expect("append checkpoint byte");

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    let error = backend
        .verify_artifact(artifact, &control)
        .expect_err("length mismatch");

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);
}

#[test]
fn changed_pinned_metadata_is_rejected() {
    let fixture = ArtifactFixture::new();
    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let (_source, control) = control("artifact-metadata-change");

    let path = fixture.path("config.json");

    let mut bytes = std::fs::read(&path).expect("configuration bytes");

    // Appended whitespace keeps JSON valid but changes its pinned
    // source length and digest.
    bytes.push(b'\n');

    std::fs::write(&path, bytes).expect("change configuration");

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    let error = backend
        .verify_artifact(artifact, &control)
        .expect_err("pinned metadata changed");

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);
}

#[test]
fn wrong_compiled_template_pin_is_rejected() {
    let fixture = ArtifactFixture::new();

    let mut specification = fixture.specification.clone();

    specification.template_sha256 = [0; 32];

    let error = expect_failure(
        Qwen3Backend::new(vec![specification], backend_limits()),
        "wrong template pin accepted",
    );

    assert_eq!(error.kind(), ErrorKind::ArtifactValidationFailed,);
}

#[test]
fn missing_tokenizer_pin_is_rejected() {
    let fixture = ArtifactFixture::new();

    let mut specification = fixture.specification.clone();

    specification.files.remove("tokenizer.json");

    let _error = expect_failure(
        Qwen3Backend::new(vec![specification], backend_limits()),
        "unpinned tokenizer accepted",
    );
}

#[test]
fn duplicate_artifact_identity_is_rejected() {
    let fixture = ArtifactFixture::new();

    let error = expect_failure(
        Qwen3Backend::new(
            vec![fixture.specification.clone(), fixture.specification.clone()],
            backend_limits(),
        ),
        "duplicate artifact identity accepted",
    );

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);
}

#[test]
fn unsafe_component_name_is_rejected() {
    let fixture = ArtifactFixture::new();

    let mut specification = fixture.specification.clone();

    specification.files.insert(
        "../escape.safetensors".to_owned(),
        FilePin {
            bytes: 1,
            sha256: [0; 32],
        },
    );

    let _error = expect_failure(
        Qwen3Backend::new(vec![specification], backend_limits()),
        "component path escape accepted",
    );
}

#[test]
fn indexed_checkpoint_with_exact_inventory_loads() {
    let mut fixture = ArtifactFixture::new();

    fixture.write_index(false);

    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let plan = fixture.plan(&backend, &registry, 3);

    let (_source, control) = control("artifact-valid-index");

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    backend
        .verify_artifact(artifact, &control)
        .expect("indexed artifact verification");

    let loaded =
        load_fixture(&backend, &registry, &plan, &control).expect("exact indexed checkpoint loads");

    assert_eq!(loaded.artifact_id(), &artifact_id(),);
}

#[test]
fn indexed_checkpoint_with_missing_inventory_entry_is_rejected() {
    let mut fixture = ArtifactFixture::new();

    // Pin a deliberately malformed but authentic test index: one actual
    // tensor is absent from weight_map.
    fixture.write_index(true);

    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let plan = fixture.plan(&backend, &registry, 3);

    let (_source, control) = control("artifact-invalid-index");

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    // Integrity and structural validity are separate requirements.
    backend
        .verify_artifact(artifact, &control)
        .expect("pins match the malformed fixture");

    let _error = expect_failure(
        load_fixture(&backend, &registry, &plan, &control),
        "incomplete index inventory accepted",
    );
}

#[test]
fn cancellation_reaches_verification_and_loading() {
    let fixture = ArtifactFixture::new();
    let backend = fixture.backend();
    let registry = fixture.registry(&backend);

    let plan = fixture.plan(&backend, &registry, 3);

    let (source, control) = control("artifact-cancelled-load");

    source.request_cancel();

    let artifact = registry.artifact(&artifact_id()).expect("artifact");

    let verification_error = backend
        .verify_artifact(artifact, &control)
        .expect_err("cancelled verification");

    assert_eq!(verification_error.kind(), ErrorKind::Cancelled,);

    let loading_error = expect_failure(
        load_fixture(&backend, &registry, &plan, &control),
        "cancelled loading succeeded",
    );

    assert_eq!(loading_error.kind(), ErrorKind::Cancelled,);
}
