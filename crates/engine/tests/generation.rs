//! Real-backend generation integration tests.
//!
//! Lifecycle tests use the production Qwen chat renderer.
//!
//! Prefix-reuse tests use:
//!
//! - A real locally written safetensors checkpoint.
//! - The actual Qwen3 loader and MLX model.
//! - Actual family-owned KV allocations.
//! - The shared pinned-template renderer and HF ByteLevel tokenizer.
//! - EngineCore::generate, FamilyRuntime, Scheduler, and admission.
//! - An observing adapter that records actual model calls and logits.
//!
//! The observing adapter never substitutes model outputs.
//!
//! Prefix tests use a pinned append-friendly test template so a subsequent
//! request can reproduce the entire previously executed sequence, including
//! generated tokens. They do not manufacture RenderedContext values.
//!
//! Prefix tests also select a fixed cache capacity through the real Qwen
//! planner. This separates reuse behavior from allocation replacement.
//!
//! Lifecycle fixtures use a zero output head for predictable "!" output.
//! Prefix fixtures use a nonzero two-row output head. Their generated
//! tokens remain printable ASCII while their logits depend on execution.
//!
//! These synthetic-checkpoint tests execute real MLX operations. Prefix and
//! lifecycle fixtures use MLX CPU. Runtime-policy fixtures cover the unified
//! CPU/Metal production path on Apple Silicon and MLX CPU on other hosts.
//! They do not establish trained-model conformance.

use engine::{
    admission::AdmissionController,
    api::{Engine, EngineCore, EngineFuture},
    backends::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits},
    error::{EngineError, ErrorKind},
    family::{
        AutoregressiveModel, BufferMemoryUsage, CacheAllocationRequest, ExecutionControl,
        FamilyCache, LoadedModel, ModelFamily, ModelInstanceId, ModelLoadRequest, PrefillOutput,
        TokenLogits,
    },
    generation::GenerationLimits,
    registry::{ArtifactRegistration, FamilyRegistration, ModelProfile, Registry},
    rendering::{self, PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions},
    runtime::{FamilyRuntime, RetentionPolicy, RuntimeBackend, RuntimePlan, TextDecoder},
    scheduler::Scheduler,
    types::{
        ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, ContextSnapshotId, FinishReason,
        Generation, GenerationEvent, GenerationOptions, GenerationRequest, GenerationStream,
        Message, MessageRole, ModelProfileId, RenderedContext, RequestId, Usage,
    },
};
use qwen3::{
    config::Qwen3Config,
    fixture::{Qwen3Weights, generate_fixture},
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
    future::{Future, poll_fn},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, pre_tokenizers::byte_level::ByteLevel};

const TEST_TIMEOUT: Duration = Duration::from_secs(30);
const IM_START_ID: u32 = 256;
const IM_END_ID: u32 = 257;
// The real loader's bounded metadata workspace brings fixture startup to
// roughly 20 MiB. Pressure tests still replace this baseline with exact
// production-plan budgets when exercising rejection and reclamation.
const BUDGET_BYTES: u64 = 32 * 1024 * 1024;

const TEMPLATE_A: &str = "{{ messages[0].content }}";
const TEMPLATE_B: &str = "{% set body = messages[0].content %}{{ body }}";

const RENDER_QWEN: usize = 0;
const RENDER_TEST_A: usize = 1;
const RENDER_TEST_B: usize = 2;
const RENDER_TEST_TOKENIZER_B: usize = 3;

fn discard_policy() -> RetentionPolicy {
    RetentionPolicy {
        retain_model: false,
        retain_cache_buffers: false,
    }
}

fn retained_policy() -> RetentionPolicy {
    RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: true,
    }
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);

        loop {
            let path = std::env::temp_dir().join(format!(
                "alloy-generation-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));

            match std::fs::create_dir(&path) {
                Ok(()) => {
                    return Self(path.canonicalize().expect("canonical fixture directory"));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    continue;
                }
                Err(error) => panic!("create fixture directory: {error}"),
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

fn deadline() -> Instant {
    Instant::now()
        .checked_add(TEST_TIMEOUT)
        .expect("test deadline")
}

fn block_on_until<F: Future>(future: F, deadline: Instant) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(ThreadWake {
        thread: thread::current(),
    }));
    let mut context = Context::from_waker(&waker);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                let now = Instant::now();
                assert!(now < deadline, "generation test exceeded its deadline",);
                thread::park_timeout(deadline.saturating_duration_since(now));
            }
        }
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    block_on_until(future, deadline())
}

fn next_event(stream: &mut GenerationStream, deadline: Instant) -> Option<GenerationEvent> {
    block_on_until(
        poll_fn(|context| stream.as_mut().poll_next(context)),
        deadline,
    )
}

fn expect_failure<T>(result: Result<T, EngineError>, message: &str) -> EngineError {
    match result {
        Ok(_) => panic!("{message}"),
        Err(error) => error,
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn profile_id() -> ModelProfileId {
    ModelProfileId::new("generation-profile").expect("profile ID")
}

fn alternate_profile_id() -> ModelProfileId {
    ModelProfileId::new("generation-profile-alternate").expect("alternate profile")
}

fn artifact_id() -> ArtifactId {
    ArtifactId::new("generation-artifact").expect("artifact ID")
}

fn context(snapshot: &str, profile: ModelProfileId, content: &str) -> ContextSnapshot {
    ContextSnapshot::new(
        ContextSnapshotId::new(snapshot).expect("snapshot ID"),
        profile,
        vec![Message {
            role: MessageRole::User,
            content: content.to_owned(),
        }],
    )
    .expect("context")
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
        .expect("fixture tensor size");

    assert_eq!(values.len(), expected, "fixture shape mismatch for {name}");

    records.push(TensorRecord {
        name,
        shape,
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    });
}

fn write_checkpoint(directory: &Path, config: &Qwen3Config, weights: &Qwen3Weights) {
    let mut records = Vec::new();

    push_tensor(
        &mut records,
        "model.embed_tokens.weight".to_owned(),
        vec![config.vocab_size, config.hidden_size],
        weights
            .embed_tokens
            .as_f32()
            .expect("dense fixture embedding"),
    );

    for (index, layer) in weights.layers.iter().enumerate() {
        let prefix = format!("model.layers.{index}");

        for (suffix, shape, values) in [
            (
                "input_layernorm.weight",
                vec![config.hidden_size],
                layer.input_layernorm.as_slice(),
            ),
            (
                "self_attn.q_proj.weight",
                vec![config.q_dim(), config.hidden_size],
                layer.q_proj.as_f32().expect("dense fixture q_proj"),
            ),
            (
                "self_attn.k_proj.weight",
                vec![config.kv_dim(), config.hidden_size],
                layer.k_proj.as_f32().expect("dense fixture k_proj"),
            ),
            (
                "self_attn.v_proj.weight",
                vec![config.kv_dim(), config.hidden_size],
                layer.v_proj.as_f32().expect("dense fixture v_proj"),
            ),
            (
                "self_attn.q_norm.weight",
                vec![config.head_dim],
                layer.q_norm.as_slice(),
            ),
            (
                "self_attn.k_norm.weight",
                vec![config.head_dim],
                layer.k_norm.as_slice(),
            ),
            (
                "self_attn.o_proj.weight",
                vec![config.hidden_size, config.q_dim()],
                layer.o_proj.as_f32().expect("dense fixture o_proj"),
            ),
            (
                "post_attention_layernorm.weight",
                vec![config.hidden_size],
                layer.post_attention_layernorm.as_slice(),
            ),
            (
                "mlp.gate_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
                layer.gate_proj.as_f32().expect("dense fixture gate_proj"),
            ),
            (
                "mlp.up_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
                layer.up_proj.as_f32().expect("dense fixture up_proj"),
            ),
            (
                "mlp.down_proj.weight",
                vec![config.hidden_size, config.intermediate_size],
                layer.down_proj.as_f32().expect("dense fixture down_proj"),
            ),
        ] {
            push_tensor(&mut records, format!("{prefix}.{suffix}"), shape, values);
        }
    }

    push_tensor(
        &mut records,
        "model.norm.weight".to_owned(),
        vec![config.hidden_size],
        &weights.final_layernorm,
    );

    push_tensor(
        &mut records,
        "lm_head.weight".to_owned(),
        vec![config.vocab_size, config.hidden_size],
        weights
            .lm_head
            .as_ref()
            .expect("untied fixture head")
            .as_f32()
            .expect("dense fixture head"),
    );

    let views: Vec<_> = records
        .iter()
        .map(|record| {
            (
                record.name.as_str(),
                TensorView::new(Dtype::F32, record.shape.clone(), &record.bytes)
                    .expect("fixture tensor"),
            )
        })
        .collect();

    let bytes = serialize(views, None).expect("safetensors serialization");
    std::fs::write(directory.join("model.safetensors"), bytes).expect("write checkpoint");
}

fn special_token(id: u32, content: &str) -> Value {
    json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
    })
}

fn tokenizer_bytes(eos_is_zero: bool) -> Vec<u8> {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();

    assert_eq!(alphabet.len(), 256);
    assert_eq!(alphabet[0], '!');
    assert_eq!(alphabet[1], '"');

    let mut vocabulary = Map::new();
    for (index, character) in alphabet.into_iter().enumerate() {
        vocabulary.insert(character.to_string(), Value::from(index as u64));
    }
    vocabulary.insert("<|im_start|>".to_owned(), Value::from(IM_START_ID));
    vocabulary.insert("<|im_end|>".to_owned(), Value::from(IM_END_ID));

    let mut specials = vec![
        special_token(IM_START_ID, "<|im_start|>"),
        special_token(IM_END_ID, "<|im_end|>"),
    ];
    if eos_is_zero {
        specials.push(special_token(0, "!"));
    }

    let source = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": specials,
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
    let tokenizer = Tokenizer::from_bytes(&bytes).expect("fixture tokenizer");
    assert_eq!(tokenizer.id_to_token(0).as_deref(), Some("!"));
    assert_eq!(tokenizer.id_to_token(1).as_deref(), Some("\""));
    bytes
}

#[derive(Clone, Debug, Default)]
struct RequestWork {
    prefill_tokens: Vec<u32>,
    prefill_calls: usize,
    decode_tokens: Vec<u32>,
    decode_logits: Vec<Vec<f32>>,
}

#[derive(Default)]
struct Observations {
    loads: AtomicUsize,
    cache_allocations: AtomicUsize,
    requests: Mutex<BTreeMap<RequestId, RequestWork>>,
}

impl Observations {
    fn work(&self, request: &str) -> RequestWork {
        self.requests
            .lock()
            .expect("observation mutex")
            .get(&RequestId::new(request).expect("request ID"))
            .cloned()
            .unwrap_or_default()
    }

    fn record_prefill(&self, request: &RequestId, tokens: &[u32]) {
        let mut requests = self.requests.lock().expect("observation mutex");
        let work = requests.entry(request.clone()).or_default();
        work.prefill_calls += 1;
        work.prefill_tokens.extend_from_slice(tokens);
    }

    fn record_decode(&self, request: &RequestId, token: u32, logits: &TokenLogits) {
        let mut requests = self.requests.lock().expect("observation mutex");
        let work = requests.entry(request.clone()).or_default();
        work.decode_tokens.push(token);
        work.decode_logits.push(logits.values().to_vec());
    }
}

/// Observes actual native operations without supplying their results.
struct ObservedModel {
    inner: Box<dyn LoadedModel>,
    vocabulary: usize,
    observations: Arc<Observations>,
}

impl ObservedModel {
    fn executable(&mut self) -> Result<&mut (dyn AutoregressiveModel + '_), EngineError> {
        match self.inner.autoregressive() {
            Some(model) => Ok(model),
            None => Err(EngineError::unsupported(
                "test observation",
                "actual loaded model is not autoregressive",
            )),
        }
    }
}

impl LoadedModel for ObservedModel {
    fn instance_id(&self) -> &ModelInstanceId {
        self.inner.instance_id()
    }

    fn artifact_id(&self) -> &ArtifactId {
        self.inner.artifact_id()
    }

    fn registration(&self) -> &FamilyRegistration {
        self.inner.registration()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        self.inner.memory_usage()
    }

    fn autoregressive(&mut self) -> Option<&mut dyn AutoregressiveModel> {
        Some(self)
    }
}

impl AutoregressiveModel for ObservedModel {
    fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }

    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        let cache = self.executable()?.allocate_cache(request, control)?;
        self.observations
            .cache_allocations
            .fetch_add(1, Ordering::Relaxed);
        Ok(cache)
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        let output = self.executable()?.prefill(tokens, cache, control)?;
        self.observations
            .record_prefill(control.request_id(), tokens);
        Ok(output)
    }

    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        let logits = self.executable()?.decode(token, cache, control)?;
        self.observations
            .record_decode(control.request_id(), token, &logits);
        Ok(logits)
    }
}

/// Delegates production operations to Qwen.
///
/// Controlled rendering is used only by prefix-isolation tests. Lifecycle
/// tests select RENDER_QWEN and use the production Qwen renderer.
struct ObservedBackend {
    inner: Qwen3Backend,
    observations: Arc<Observations>,
    render_selection: Arc<AtomicUsize>,
    tokenizer_a: PinnedTokenizer,
    tokenizer_b: PinnedTokenizer,
    manifest_sha256: [u8; 32],
    vocabulary: usize,
    fixed_capacity: bool,
}

impl ModelFamily for ObservedBackend {
    fn registration(&self) -> &FamilyRegistration {
        self.inner.registration()
    }

    fn load(&self, request: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        let inner = self.inner.load(request)?;
        self.observations.loads.fetch_add(1, Ordering::Relaxed);

        Ok(Box::new(ObservedModel {
            inner,
            vocabulary: self.vocabulary,
            observations: Arc::clone(&self.observations),
        }))
    }
}

impl RuntimeBackend for ObservedBackend {
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        Box::pin(async move {
            let selection = self.render_selection.load(Ordering::Acquire);
            if selection == RENDER_QWEN {
                return self.inner.render_and_count(registry, context).await;
            }

            let resolved = registry.resolve_profile(context.profile_id())?;
            let template = if selection == RENDER_TEST_B {
                TEMPLATE_B
            } else {
                TEMPLATE_A
            };
            let tokenizer = if selection == RENDER_TEST_TOKENIZER_B {
                &self.tokenizer_b
            } else {
                &self.tokenizer_a
            };

            let binding = RenderingBinding {
                profile_id: context.profile_id(),
                artifact_id: resolved.artifact().id(),
                template,
                template_sha256: digest(template.as_bytes()),
                artifact_manifest_sha256: self.manifest_sha256,
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
            let limits = RenderingLimits::new(
                16,
                4096,
                4096,
                16 * 1024,
                resolved.profile().context_limit_tokens(),
            )?;

            let rendered =
                rendering::render_and_count(&context, &binding, tokenizer, &options, limits)?;
            resolved.profile().check_context(rendered.token_count())?;
            Ok(rendered)
        })
    }

    fn plan<'a>(
        &'a self,
        registry: &'a Registry,
        profile: &'a ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, RuntimePlan> {
        Box::pin(async move {
            if !self.fixed_capacity {
                return self.inner.plan(registry, profile, input_tokens).await;
            }

            let profile_config = registry.profile(profile)?;
            profile_config.check_context(input_tokens)?;

            let maximum_input = profile_config
                .context_limit_tokens()
                .checked_sub(profile_config.max_output_tokens())
                .and_then(|value| value.checked_sub(profile_config.safety_margin_tokens()))
                .ok_or_else(|| EngineError::Internal {
                    message: "fixture profile reserve exceeds context".to_owned(),
                })?;

            // Obtain capacity and byte bounds from the real backend.
            // Do not invent a cache-byte multiplier in the test.
            self.inner.plan(registry, profile, maximum_input).await
        })
    }

    fn verify_artifact(
        &self,
        artifact: &ArtifactRegistration,
        control: &ExecutionControl,
    ) -> Result<(), EngineError> {
        self.inner.verify_artifact(artifact, control)
    }

    fn decoder(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        self.inner.decoder(registry, profile)
    }

    fn eos_tokens(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Vec<u32>, EngineError> {
        self.inner.eos_tokens(registry, profile)
    }
}

struct TestEnvironment {
    engine: EngineCore<FamilyRuntime>,
    admission: AdmissionController,
    scheduler: Scheduler,
    observations: Arc<Observations>,
    render_selection: Arc<AtomicUsize>,
    context: ContextSnapshot,
    max_output: usize,
    expected_input_tokens: usize,
    output_reservation_bytes: u64,
    weights_reservation_bytes: u64,
    _directory: TestDirectory,
}

impl TestEnvironment {
    fn new(max_output: usize, event_capacity: usize, eos_is_zero: bool) -> Self {
        Self::build(
            max_output,
            event_capacity,
            eos_is_zero,
            discard_policy(),
            false,
            engine::backends::Qwen3Execution::MlxCpu,
        )
    }

    fn prefix(retention: RetentionPolicy) -> Self {
        Self::prefix_with_execution(retention, engine::backends::Qwen3Execution::MlxCpu)
    }

    fn prefix_with_execution(
        retention: RetentionPolicy,
        execution: engine::backends::Qwen3Execution,
    ) -> Self {
        Self::build(4, 8, false, retention, true, execution)
    }

    fn build(
        max_output: usize,
        event_capacity: usize,
        eos_is_zero: bool,
        retention: RetentionPolicy,
        prefix_fixture: bool,
        execution: engine::backends::Qwen3Execution,
    ) -> Self {
        assert!(!prefix_fixture || !eos_is_zero);

        let directory = TestDirectory::new();
        let mut config = Qwen3Config::tiny_fixture();
        config.vocab_size = 258;
        config.tie_embeddings = false;

        let mut weights = generate_fixture(&config);
        let head = weights
            .lm_head
            .as_mut()
            .expect("untied output head")
            .as_f32_mut()
            .expect("dense fixture head");

        if prefix_fixture {
            let row = head[..config.hidden_size].to_vec();
            assert!(row.iter().any(|value| *value != 0.0));
            head.fill(0.0);

            // Opposite nonzero rows restrict greedy selection to token
            // 0 ("!") or token 1 ("\""), while preserving real,
            // context-dependent logits.
            for (index, value) in row.into_iter().enumerate() {
                head[index] = value;
                head[config.hidden_size + index] = -value;
            }
        } else {
            head.fill(0.0);
        }

        write_checkpoint(directory.path(), &config, &weights);

        let eos = if eos_is_zero { 0 } else { IM_END_ID };
        let configuration = json!({
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
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "rms_norm_eps": 0.000001,
            "attention_bias": false,
            "rope_scaling": null,
            "torch_dtype": "float32",
            "eos_token_id": eos
        });

        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&configuration).expect("configuration JSON"),
        )
        .expect("write configuration");

        let source_a = tokenizer_bytes(eos_is_zero);
        std::fs::write(directory.path().join("tokenizer.json"), &source_a)
            .expect("write tokenizer");

        // Same tokenizer semantics, different exact pinned source identity.
        let tokenizer_value: Value = serde_json::from_slice(&source_a).expect("tokenizer value");
        let source_b = serde_json::to_vec_pretty(&tokenizer_value).expect("pretty tokenizer");

        let tokenizer_a =
            PinnedTokenizer::from_pinned_json(&source_a, digest(&source_a), 1024 * 1024)
                .expect("pinned tokenizer A");
        let tokenizer_b =
            PinnedTokenizer::from_pinned_json(&source_b, digest(&source_b), 1024 * 1024)
                .expect("pinned tokenizer B");
        assert_ne!(tokenizer_a.identity(), tokenizer_b.identity());

        let mut files = BTreeMap::new();
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            let bytes = std::fs::read(directory.path().join(name)).expect("fixture bytes");
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
        let manifest_sha256 = specification.manifest_sha256().expect("manifest identity");
        let generation_limits = GenerationLimits::new(event_capacity, 4096, 1024).expect("limits");

        let inner = Qwen3Backend::new_with_execution(
            vec![specification],
            Qwen3BackendLimits {
                max_files: 16,
                max_config_bytes: 1024 * 1024,
                max_tokenizer_bytes: 1024 * 1024,
                max_messages: 16,
                max_input_bytes: 4096,
                max_rendered_bytes: 16 * 1024,
                max_token_bytes: 128,
                verification_chunk_bytes: 4096,
                context_limit_tokens: config.max_context,
                prefill_chunk_tokens: NonZeroUsize::new(2).expect("chunk"),
                loader_headroom_bytes: 64 * 1024,
                execution_headroom_bytes: 64 * 1024,
                generation: generation_limits,
            },
            execution,
        )
        .expect("Qwen backend");

        let make_profile = |id| {
            ModelProfile::new(
                id,
                artifact_id(),
                config.max_context,
                max_output,
                8,
                GenerationOptions::greedy(max_output).expect("options"),
            )
            .expect("profile")
        };

        let registry = inner
            .registry(vec![
                make_profile(profile_id()),
                make_profile(alternate_profile_id()),
            ])
            .expect("registry");

        let observations = Arc::new(Observations::default());
        let render_selection = Arc::new(AtomicUsize::new(if prefix_fixture {
            RENDER_TEST_A
        } else {
            RENDER_QWEN
        }));

        let backend = ObservedBackend {
            inner,
            observations: Arc::clone(&observations),
            render_selection: Arc::clone(&render_selection),
            tokenizer_a,
            tokenizer_b,
            manifest_sha256,
            vocabulary: config.vocab_size,
            fixed_capacity: prefix_fixture,
        };

        let context = context("generation-context", profile_id(), "hello");
        let rendered =
            block_on(backend.render_and_count(&registry, context.clone())).expect("render");

        let selected_profile = profile_id();
        let plan = block_on(backend.plan(&registry, &selected_profile, rendered.token_count()))
            .expect("plan");

        let admission = AdmissionController::new(BUDGET_BYTES).expect("admission");
        let scheduler = Scheduler::new(64).expect("scheduler");
        let backend: Arc<dyn RuntimeBackend> = Arc::new(backend);

        let runtime = FamilyRuntime::new(
            vec![backend],
            admission.clone(),
            generation_limits,
            retention,
        )
        .expect("runtime");

        let engine = EngineCore::new(registry, scheduler.clone(), runtime);

        Self {
            engine,
            admission,
            scheduler,
            observations,
            render_selection,
            context,
            max_output,
            expected_input_tokens: rendered.token_count(),
            output_reservation_bytes: plan.output_bytes,
            weights_reservation_bytes: plan.load_limits.max_resident_bytes(),
            _directory: directory,
        }
    }

    fn select_rendering(&self, selection: usize) {
        self.wait_idle();
        self.render_selection.store(selection, Ordering::Release);
    }

    fn request_with(&self, name: &str, context: ContextSnapshot) -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(name).expect("request ID"),
            context.profile_id().clone(),
            context,
            GenerationOptions::greedy(self.max_output).expect("options"),
        )
        .expect("request")
    }

    fn request(&self, name: &str) -> GenerationRequest {
        self.request_with(name, self.context.clone())
    }

    fn start(&self, name: &str) -> Generation {
        block_on(self.engine.generate(self.request(name))).expect("accepted generation")
    }

    fn render(&self, context: ContextSnapshot) -> RenderedContext {
        block_on(self.engine.render_and_count(context)).expect("render context")
    }

    fn run(&self, name: &str, context: ContextSnapshot) -> Transcript {
        let generation = block_on(self.engine.generate(self.request_with(name, context)))
            .expect("accepted generation");
        let mut events = generation.into_events();
        let transcript = drain(&mut events);
        drop(events);
        self.wait_idle();
        transcript
    }

    fn wait_idle(&self) {
        let deadline = deadline();
        while self.scheduler.is_busy().expect("scheduler status") {
            assert!(
                Instant::now() < deadline,
                "worker did not relinquish execution ownership",
            );
            thread::yield_now();
        }
    }

    fn wait_unreserved(&self) {
        let deadline = deadline();
        loop {
            let snapshot = self.admission.snapshot().expect("snapshot");
            if snapshot.reserved_bytes() == 0 {
                assert_eq!(snapshot.active_reservations(), 0);
                return;
            }
            assert!(
                Instant::now() < deadline,
                "reservations remained after cleanup: {snapshot:?}",
            );
            thread::yield_now();
        }
    }

    fn work(&self, request: &str) -> RequestWork {
        self.observations.work(request)
    }

    fn loads(&self) -> usize {
        self.observations.loads.load(Ordering::Relaxed)
    }

    fn cache_allocations(&self) -> usize {
        self.observations.cache_allocations.load(Ordering::Relaxed)
    }
}

struct Transcript {
    text: String,
    terminal: GenerationEvent,
}

fn drain(stream: &mut GenerationStream) -> Transcript {
    let deadline = deadline();
    let mut text = String::new();
    let mut terminal = None;

    while let Some(event) = next_event(stream, deadline) {
        assert!(
            terminal.is_none(),
            "event emitted after terminal event: {event:?}",
        );

        if event.is_terminal() {
            terminal = Some(event);
        } else {
            match event {
                GenerationEvent::TextDelta { text: delta } => {
                    text.push_str(&delta);
                }
                other => panic!("unexpected fixture event: {other:?}"),
            }
        }
    }

    Transcript {
        text,
        terminal: terminal.expect("one terminal event"),
    }
}

fn completed_usage(transcript: &Transcript, expected_reason: FinishReason) -> Usage {
    match &transcript.terminal {
        GenerationEvent::GenerationCompleted {
            finish_reason,
            usage,
        } => {
            assert_eq!(*finish_reason, expected_reason);
            *usage
        }
        other => panic!("expected completion, got {other:?}"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite());
        assert!(expected.is_finite());
        let tolerance = 2e-5 + 1e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "logit {index}: actual={actual}, expected={expected}",
        );
    }
}

fn assert_matches_cold(
    actual: &Transcript,
    actual_work: &RequestWork,
    cold: &Transcript,
    cold_work: &RequestWork,
) {
    assert_eq!(actual.text, cold.text);
    assert_eq!(
        completed_usage(actual, FinishReason::OutputLimit),
        completed_usage(cold, FinishReason::OutputLimit),
    );
    assert_eq!(actual_work.decode_tokens, cold_work.decode_tokens);
    assert_eq!(
        actual_work.decode_logits.len(),
        cold_work.decode_logits.len(),
    );
    assert!(!actual_work.decode_logits.is_empty());

    for (actual, cold) in actual_work
        .decode_logits
        .iter()
        .zip(&cold_work.decode_logits)
    {
        assert_close(actual, cold);
    }

    assert!(
        cold_work
            .decode_logits
            .iter()
            .flatten()
            .any(|value| value.abs() > 1e-7),
        "prefix oracle must not use an all-zero logits fixture",
    );
}

/// Produce a context whose exact input tokens equal all positions
/// executed by a completed seed request.
fn completed_context(
    environment: &TestEnvironment,
    seed_name: &str,
    snapshot_name: &str,
) -> ContextSnapshot {
    let seed_context = context("prefix-seed-context", profile_id(), "hello");
    let rendered = environment.render(seed_context.clone());
    let seed = environment.run(seed_name, seed_context);
    completed_usage(&seed, FinishReason::OutputLimit);

    let seed_work = environment.work(seed_name);
    assert_eq!(seed_work.prefill_tokens, rendered.token_ids());

    let body = format!("{}{}", rendered.text(), seed.text);
    let next_context = context(snapshot_name, profile_id(), &body);
    let next_rendered = environment.render(next_context.clone());

    let mut exact_executed = rendered.token_ids().to_vec();
    exact_executed.extend_from_slice(&seed_work.decode_tokens);

    assert_eq!(
        next_rendered.token_ids(),
        exact_executed,
        "the next context must reproduce actual completed token history",
    );

    next_context
}

#[test]
fn counted_render_is_used_for_generation_and_stale_identity_is_rejected() {
    let environment = TestEnvironment::prefix(discard_policy());
    let snapshot = environment.context.clone();
    let counted = environment.render(snapshot.clone());
    let request = environment
        .request_with("counted-render", snapshot.clone())
        .with_expected_render(counted.clone())
        .expect("bind render");
    let generation = block_on(environment.engine.generate(request)).expect("generation");
    let mut events = generation.into_events();
    let transcript = drain(&mut events);
    assert_eq!(
        completed_usage(&transcript, FinishReason::OutputLimit).input_tokens,
        counted.token_count()
    );
    drop(events);
    environment.wait_idle();
    assert_eq!(
        environment.work("counted-render").prefill_tokens,
        counted.token_ids()
    );

    environment.select_rendering(RENDER_TEST_B);
    let request = environment
        .request_with("stale-render", snapshot)
        .with_expected_render(counted)
        .expect("bind old render");
    let error = expect_failure(
        block_on(environment.engine.generate(request)),
        "stale rendering accepted",
    );
    assert!(error.to_string().contains("context rendering changed"));
    assert_eq!(environment.work("stale-render").prefill_tokens.len(), 0);
    environment.wait_unreserved();
}

#[test]
fn reused_snapshot_id_cannot_hide_changed_context_from_counted_render() {
    let environment = TestEnvironment::prefix(discard_policy());
    let original = environment.context.clone();
    let counted = environment.render(original.clone());
    let changed = context(
        original.snapshot_id().as_str(),
        original.profile_id().clone(),
        "different content",
    );
    let request = environment
        .request_with("changed-context", changed)
        .with_expected_render(counted)
        .expect("same tracing id is structurally valid");
    let error = expect_failure(
        block_on(environment.engine.generate(request)),
        "changed context accepted",
    );
    assert!(error.to_string().contains("context rendering changed"));
    assert_eq!(environment.work("changed-context").prefill_tokens.len(), 0);
    environment.wait_unreserved();
}

#[test]
fn real_generation_streams_exact_output_and_measured_usage() {
    let environment = TestEnvironment::new(4, 8, false);
    let generation = environment.start("real-output-limit");
    assert_eq!(generation.request_id().as_str(), "real-output-limit");

    let mut events = generation.into_events();
    let transcript = drain(&mut events);
    assert_eq!(transcript.text, "!!!!");

    let usage = completed_usage(&transcript, FinishReason::OutputLimit);
    assert_eq!(usage.input_tokens, environment.expected_input_tokens);
    assert_eq!(usage.output_tokens, 4);

    for _ in 0..4 {
        assert!(next_event(&mut events, deadline()).is_none());
    }

    drop(events);
    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn real_eos_stops_generation_and_is_counted_without_display_text() {
    let environment = TestEnvironment::new(8, 8, true);
    let mut events = environment.start("real-eos").into_events();
    let transcript = drain(&mut events);

    assert!(transcript.text.is_empty());
    let usage = completed_usage(&transcript, FinishReason::EndOfSequence);
    assert_eq!(usage.input_tokens, environment.expected_input_tokens);
    assert_eq!(usage.output_tokens, 1);

    drop(events);
    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn bounded_stream_keeps_execution_owned_until_output_can_progress() {
    let environment = TestEnvironment::new(8, 1, false);
    let generation = environment.start("backpressured-generation");

    let error = expect_failure(
        block_on(
            environment
                .engine
                .generate(environment.request("competing-generation")),
        ),
        "competing generation bypassed execution ownership",
    );
    assert_eq!(error.kind(), ErrorKind::Busy);

    let mut events = generation.into_events();
    let transcript = drain(&mut events);
    assert_eq!(transcript.text, "!".repeat(8));
    assert_eq!(
        completed_usage(&transcript, FinishReason::OutputLimit).output_tokens,
        8,
    );

    drop(events);
    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn cancellation_terminates_generation_without_draining_for_capacity() {
    let environment = TestEnvironment::new(32, 1, false);
    let generation = environment.start("cancel-backpressured");
    let request = generation.request_id().clone();

    assert_eq!(
        block_on(environment.engine.cancel(request.clone())).expect("cancel"),
        CancelOutcome::Requested,
    );

    let mut events = generation.into_events();
    let transcript = drain(&mut events);

    match transcript.terminal {
        GenerationEvent::RequestFailed {
            error,
            partial_usage,
        } => {
            assert_eq!(error.kind(), ErrorKind::Cancelled);
            assert!(partial_usage.input_tokens <= environment.expected_input_tokens);
            assert!(partial_usage.output_tokens <= 32);

            match error {
                EngineError::Cancelled { request_id } => {
                    assert_eq!(request_id, request.as_str());
                }
                other => panic!("unexpected cancellation error: {other:?}"),
            }
        }
        other => panic!("expected cancellation, got {other:?}"),
    }

    drop(events);
    environment.wait_idle();
    environment.wait_unreserved();

    assert_eq!(
        block_on(environment.engine.cancel(request)).expect("finished cancel"),
        CancelOutcome::AlreadyFinished,
    );
}

#[test]
fn dropping_the_stream_cancels_work_and_allows_a_fresh_request() {
    let environment = TestEnvironment::new(8, 1, false);
    drop(environment.start("dropped-stream").into_events());

    environment.wait_idle();
    environment.wait_unreserved();

    let transcript = environment.run("request-after-disconnect", environment.context.clone());
    assert_eq!(transcript.text, "!".repeat(8));
    assert_eq!(
        completed_usage(&transcript, FinishReason::OutputLimit).output_tokens,
        8,
    );
    environment.wait_unreserved();
}

#[test]
fn output_reservation_outlives_worker_until_stream_is_dropped() {
    let environment = TestEnvironment::new(4, 8, false);
    let mut events = environment.start("undrained-output").into_events();

    environment.wait_idle();
    let snapshot = environment.admission.snapshot().expect("snapshot");
    assert_eq!(
        snapshot.reserved_bytes(),
        environment.output_reservation_bytes,
    );
    assert_eq!(snapshot.active_reservations(), 1);

    let transcript = drain(&mut events);
    assert_eq!(transcript.text, "!!!!");

    assert_eq!(
        environment
            .admission
            .snapshot()
            .expect("exhausted accounting")
            .reserved_bytes(),
        environment.output_reservation_bytes,
    );

    drop(events);
    environment.wait_unreserved();
}

#[test]
fn completed_request_cannot_emit_a_second_terminal_after_cancel() {
    let environment = TestEnvironment::new(4, 8, false);
    let generation = environment.start("completed-before-cancel");
    let request = generation.request_id().clone();
    let mut events = generation.into_events();

    let transcript = drain(&mut events);
    completed_usage(&transcript, FinishReason::OutputLimit);

    assert_eq!(
        block_on(environment.engine.cancel(request)).expect("cancel completed"),
        CancelOutcome::AlreadyFinished,
    );

    for _ in 0..4 {
        assert!(next_event(&mut events, deadline()).is_none());
    }

    drop(events);
    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn retained_identical_executed_context_skips_prefill_and_matches_cold() {
    let warm = TestEnvironment::prefix(retained_policy());
    let next = completed_context(&warm, "identical-seed", "identical-next");
    let expected_input = warm.render(next.clone()).token_count();

    let actual = warm.run("identical-hit", next.clone());
    let actual_work = warm.work("identical-hit");

    assert_eq!(actual_work.prefill_calls, 0);
    assert!(actual_work.prefill_tokens.is_empty());
    assert_eq!(actual_work.decode_tokens.len(), warm.max_output);
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);
    assert_eq!(
        completed_usage(&actual, FinishReason::OutputLimit).input_tokens,
        expected_input,
    );

    let cold = TestEnvironment::prefix(discard_policy());
    let reference = cold.run("identical-cold", next);
    let cold_work = cold.work("identical-cold");

    assert_eq!(cold_work.prefill_tokens.len(), expected_input);
    assert_matches_cold(&actual, &actual_work, &reference, &cold_work);
    cold.wait_unreserved();
}

#[test]
fn retained_extension_executes_only_missing_suffix_and_matches_cold() {
    let warm = TestEnvironment::prefix(retained_policy());
    let completed = completed_context(&warm, "extension-seed", "extension-base");
    let retained_ids = warm.render(completed.clone()).token_ids().to_vec();

    let extended_body = format!("{}xy", completed.messages()[0].content);
    let extended = context("extension-next", profile_id(), &extended_body);
    let rendered = warm.render(extended.clone());

    assert!(rendered.token_ids().starts_with(&retained_ids));
    let suffix = rendered.token_ids()[retained_ids.len()..].to_vec();
    assert!(!suffix.is_empty());

    let actual = warm.run("extension-hit", extended.clone());
    let actual_work = warm.work("extension-hit");

    assert_eq!(actual_work.prefill_tokens, suffix);
    assert!(actual_work.prefill_tokens.len() < rendered.token_count());
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);

    let cold = TestEnvironment::prefix(discard_policy());
    let reference = cold.run("extension-cold", extended);
    let cold_work = cold.work("extension-cold");

    assert_eq!(cold_work.prefill_tokens, rendered.token_ids());
    assert_matches_cold(&actual, &actual_work, &reference, &cold_work);
    cold.wait_unreserved();
}

#[test]
fn divergent_tokens_invalidate_retained_kv_without_replacing_buffers() {
    let warm = TestEnvironment::prefix(retained_policy());
    let completed = completed_context(&warm, "divergent-seed", "divergent-base");
    let old = warm.render(completed.clone());

    let mut body = completed.messages()[0].content.clone();
    body.replace_range(0..1, "j");
    let changed = context("divergent-next", profile_id(), &body);
    let rendered = warm.render(changed.clone());

    assert_eq!(old.token_count(), rendered.token_count());
    assert_ne!(old.token_ids(), rendered.token_ids());

    let actual = warm.run("divergent-rebuild", changed.clone());
    let actual_work = warm.work("divergent-rebuild");

    assert_eq!(actual_work.prefill_tokens, rendered.token_ids());
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);

    let cold = TestEnvironment::prefix(discard_policy());
    let reference = cold.run("divergent-cold", changed);
    assert_matches_cold(
        &actual,
        &actual_work,
        &reference,
        &cold.work("divergent-cold"),
    );
    cold.wait_unreserved();
}

#[test]
fn shortened_context_rebuilds_instead_of_reusing_unrestored_positions() {
    let warm = TestEnvironment::prefix(retained_policy());
    let completed = completed_context(&warm, "shorter-seed", "shorter-base");
    let old = warm.render(completed);

    let shorter = context("shorter-next", profile_id(), "hel");
    let rendered = warm.render(shorter.clone());
    assert!(old.token_ids().starts_with(rendered.token_ids()));
    assert!(rendered.token_count() < old.token_count());

    let actual = warm.run("shorter-rebuild", shorter.clone());
    let actual_work = warm.work("shorter-rebuild");

    assert_eq!(actual_work.prefill_tokens, rendered.token_ids());
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);

    let cold = TestEnvironment::prefix(discard_policy());
    let reference = cold.run("shorter-cold", shorter);
    assert_matches_cold(
        &actual,
        &actual_work,
        &reference,
        &cold.work("shorter-cold"),
    );
    cold.wait_unreserved();
}

#[test]
fn template_identity_change_invalidates_even_when_token_ids_are_identical() {
    let warm = TestEnvironment::prefix(retained_policy());
    let next = completed_context(&warm, "template-seed", "template-next");
    let before = warm.render(next.clone());

    warm.select_rendering(RENDER_TEST_B);
    let after = warm.render(next.clone());

    assert_eq!(before.text(), after.text());
    assert_eq!(before.token_ids(), after.token_ids());
    assert_ne!(before.template_hash(), after.template_hash());

    let actual = warm.run("template-rebuild", next.clone());
    let actual_work = warm.work("template-rebuild");

    assert_eq!(actual_work.prefill_tokens, after.token_ids());
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);

    let cold = TestEnvironment::prefix(discard_policy());
    cold.select_rendering(RENDER_TEST_B);
    let reference = cold.run("template-cold", next);
    assert_matches_cold(
        &actual,
        &actual_work,
        &reference,
        &cold.work("template-cold"),
    );
    cold.wait_unreserved();
}

#[test]
fn tokenizer_identity_change_invalidates_even_when_token_ids_are_identical() {
    let warm = TestEnvironment::prefix(retained_policy());
    let next = completed_context(&warm, "tokenizer-seed", "tokenizer-next");
    let before = warm.render(next.clone());

    warm.select_rendering(RENDER_TEST_TOKENIZER_B);
    let after = warm.render(next.clone());

    assert_eq!(before.text(), after.text());
    assert_eq!(before.token_ids(), after.token_ids());
    assert_ne!(before.tokenizer_id(), after.tokenizer_id());

    let actual = warm.run("tokenizer-rebuild", next.clone());
    let actual_work = warm.work("tokenizer-rebuild");

    assert_eq!(actual_work.prefill_tokens, after.token_ids());
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);

    let cold = TestEnvironment::prefix(discard_policy());
    cold.select_rendering(RENDER_TEST_TOKENIZER_B);
    let reference = cold.run("tokenizer-cold", next);
    assert_matches_cold(
        &actual,
        &actual_work,
        &reference,
        &cold.work("tokenizer-cold"),
    );
    cold.wait_unreserved();
}

#[test]
fn profile_identity_change_invalidates_even_when_token_ids_are_identical() {
    let warm = TestEnvironment::prefix(retained_policy());
    let original = completed_context(&warm, "profile-seed", "profile-original");
    let before = warm.render(original.clone());

    let changed = context(
        "profile-next",
        alternate_profile_id(),
        &original.messages()[0].content,
    );
    let after = warm.render(changed.clone());

    assert_eq!(before.token_ids(), after.token_ids());
    assert_ne!(before.profile_id(), after.profile_id());

    let actual = warm.run("profile-rebuild", changed.clone());
    let actual_work = warm.work("profile-rebuild");

    assert_eq!(actual_work.prefill_tokens, after.token_ids());
    assert_eq!(warm.loads(), 1);
    assert_eq!(warm.cache_allocations(), 1);

    let cold = TestEnvironment::prefix(discard_policy());
    let reference = cold.run("profile-cold", changed);
    assert_matches_cold(
        &actual,
        &actual_work,
        &reference,
        &cold.work("profile-cold"),
    );
    cold.wait_unreserved();
}

#[test]
fn model_only_retention_keeps_weights_but_rebuilds_cache() {
    let environment = TestEnvironment::prefix(RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: false,
    });

    let first = environment.run("model-only-first", environment.context.clone());
    completed_usage(&first, FinishReason::OutputLimit);

    let after_first = environment.admission.snapshot().expect("snapshot");
    assert_eq!(
        after_first.reserved_bytes(),
        environment.weights_reservation_bytes,
    );
    assert_eq!(after_first.active_reservations(), 1);

    let second_context = context("model-only-second-context", profile_id(), "hello");
    let expected = environment.render(second_context.clone());
    let second = environment.run("model-only-second", second_context);
    completed_usage(&second, FinishReason::OutputLimit);

    assert_eq!(
        environment.work("model-only-second").prefill_tokens,
        expected.token_ids(),
    );
    assert_eq!(environment.loads(), 1);
    assert_eq!(environment.cache_allocations(), 2);

    let after_second = environment.admission.snapshot().expect("snapshot");
    assert_eq!(
        after_second.reserved_bytes(),
        environment.weights_reservation_bytes,
    );
    assert_eq!(after_second.active_reservations(), 1);
}

#[test]
fn discard_policy_cannot_report_reuse_for_matching_executed_history() {
    let environment = TestEnvironment::prefix(discard_policy());
    let next = completed_context(&environment, "discard-seed", "discard-next-context");
    environment.wait_unreserved();

    let expected = environment.render(next.clone());
    let transcript = environment.run("discard-second", next);
    completed_usage(&transcript, FinishReason::OutputLimit);

    assert_eq!(
        environment.work("discard-second").prefill_tokens,
        expected.token_ids(),
    );
    assert_eq!(environment.loads(), 2);
    assert_eq!(environment.cache_allocations(), 2);
    environment.wait_unreserved();
}

#[path = "support/runtime_policies.rs"]
mod runtime_policies;

#[path = "support/state_lifecycle.rs"]
mod state_lifecycle;
