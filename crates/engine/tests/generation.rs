//! Real-backend generation integration tests.
//!
//! Uses:
//!
//! - An actual locally written safetensors checkpoint.
//! - The real Qwen3 backend and model execution.
//! - The actual BPE/ByteLevel tokenizer and compiled chat template.
//! - EngineCore, FamilyRuntime, Scheduler, and AdmissionController.
//! - The actual bounded generation event stream.
//!
//! The fixture uses an untied zero-valued output projection. Actual model
//! execution therefore produces tied zero logits and real greedy selection
//! chooses token ID 0. The tokenizer maps that ID to "!". This makes
//! lifecycle assertions deterministic without substituting model output.
//!
//! This is not a production model, a trained-response test, an independent
//! numerical conformance test, or a Metal execution test.

use engine::{
    admission::AdmissionController,
    api::{Engine, EngineCore},
    backends::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits},
    error::{EngineError, ErrorKind},
    generation::GenerationLimits,
    registry::ModelProfile,
    runtime::{FamilyRuntime, RetentionPolicy, RuntimeBackend},
    scheduler::Scheduler,
    types::{
        ArtifactId, CancelOutcome, ContextSnapshot, ContextSnapshotId, FinishReason, Generation,
        GenerationEvent, GenerationOptions, GenerationRequest, GenerationStream, Message,
        MessageRole, ModelProfileId, RequestId, Usage,
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
        Arc,
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
const BUDGET_BYTES: u64 = 16 * 1024 * 1024;

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

                Err(error) => {
                    panic!("create fixture directory: {error}");
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

fn artifact_id() -> ArtifactId {
    ArtifactId::new("generation-artifact").expect("artifact ID")
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

    assert_eq!(values.len(), expected, "fixture shape mismatch for {name}",);

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
        &weights.embed_tokens,
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
                layer.q_proj.as_slice(),
            ),
            (
                "self_attn.k_proj.weight",
                vec![config.kv_dim(), config.hidden_size],
                layer.k_proj.as_slice(),
            ),
            (
                "self_attn.v_proj.weight",
                vec![config.kv_dim(), config.hidden_size],
                layer.v_proj.as_slice(),
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
                layer.o_proj.as_slice(),
            ),
            (
                "post_attention_layernorm.weight",
                vec![config.hidden_size],
                layer.post_attention_layernorm.as_slice(),
            ),
            (
                "mlp.gate_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
                layer.gate_proj.as_slice(),
            ),
            (
                "mlp.up_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
                layer.up_proj.as_slice(),
            ),
            (
                "mlp.down_proj.weight",
                vec![config.hidden_size, config.intermediate_size],
                layer.down_proj.as_slice(),
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
        weights.lm_head.as_ref().expect("untied fixture head"),
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

    let bytes = serialize(views, &None).expect("real safetensors serialization");

    std::fs::write(directory.join("model.safetensors"), bytes).expect("write real checkpoint");
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

    let tokenizer = Tokenizer::from_bytes(&bytes).expect("actual fixture tokenizer");

    assert_eq!(tokenizer.id_to_token(0).as_deref(), Some("!"),);

    bytes
}

struct TestEnvironment {
    engine: EngineCore<FamilyRuntime>,
    admission: AdmissionController,
    scheduler: Scheduler,
    context: ContextSnapshot,
    max_output: usize,
    expected_input_tokens: usize,
    output_reservation_bytes: u64,

    // Keep fixture files alive until the engine handle is dropped.
    _directory: TestDirectory,
}

impl TestEnvironment {
    fn new(max_output: usize, event_capacity: usize, eos_is_zero: bool) -> Self {
        let directory = TestDirectory::new();

        let mut config = Qwen3Config::tiny_fixture();

        config.vocab_size = 258;
        config.tie_embeddings = false;

        let mut weights = generate_fixture(&config);

        // Real zero-valued projection: greedy argmax deterministically
        // selects ID 0 after actual model computation.
        weights
            .lm_head
            .as_mut()
            .expect("untied output projection")
            .fill(0.0);

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
            serde_json::to_vec(&configuration).expect("config JSON"),
        )
        .expect("write config");

        std::fs::write(
            directory.path().join("tokenizer.json"),
            tokenizer_bytes(eos_is_zero),
        )
        .expect("write tokenizer");

        let mut files = BTreeMap::new();

        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            let bytes = std::fs::read(directory.path().join(name)).expect("known fixture bytes");

            files.insert(
                name.to_owned(),
                FilePin {
                    bytes: bytes.len() as u64,
                    sha256: digest(&bytes),
                },
            );
        }

        let generation_limits =
            GenerationLimits::new(event_capacity, 4096, 1024).expect("generation limits");

        let backend = Qwen3Backend::new(
            vec![Qwen3Artifact {
                id: artifact_id(),
                directory: directory.path().to_path_buf(),
                files,
                template_sha256: digest(CHAT_TEMPLATE.as_bytes()),
            }],
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
                prefill_chunk_tokens: NonZeroUsize::new(2).expect("prefill chunk"),
                loader_headroom_bytes: 64 * 1024,
                execution_headroom_bytes: 64 * 1024,
                generation: generation_limits,
            },
        )
        .expect("real backend");

        let profile = ModelProfile::new(
            profile_id(),
            artifact_id(),
            config.max_context,
            max_output,
            8,
            GenerationOptions::greedy(max_output).expect("default generation"),
        )
        .expect("profile");

        let context = ContextSnapshot::new(
            ContextSnapshotId::new("generation-context").expect("snapshot ID"),
            profile_id(),
            vec![Message {
                role: MessageRole::User,
                content: "hello".to_owned(),
            }],
        )
        .expect("context");

        let registry = backend
            .registry(vec![profile.clone()])
            .expect("planning registry");

        let rendered = block_on(backend.render_and_count(&registry, context.clone()))
            .expect("real rendering and tokenization");

        let profile_identity = profile_id();

        let plan = block_on(backend.plan(&registry, &profile_identity, rendered.token_count()))
            .expect("actual memory plan");

        let expected_input_tokens = rendered.token_count();

        let output_reservation_bytes = plan.output_bytes;

        let admission = AdmissionController::new(BUDGET_BYTES).expect("admission");

        let scheduler = Scheduler::new(32).expect("scheduler");

        let engine = backend
            .into_engine(
                vec![profile],
                admission.clone(),
                scheduler.clone(),
                RetentionPolicy {
                    retain_model: false,
                    retain_cache_buffers: false,
                },
            )
            .expect("real engine");

        Self {
            engine,
            admission,
            scheduler,
            context,
            max_output,
            expected_input_tokens,
            output_reservation_bytes,
            _directory: directory,
        }
    }

    fn request(&self, name: &str) -> GenerationRequest {
        GenerationRequest::new(
            RequestId::new(name).expect("request ID"),
            profile_id(),
            self.context.clone(),
            GenerationOptions::greedy(self.max_output).expect("generation options"),
        )
        .expect("request")
    }

    fn start(&self, name: &str) -> Generation {
        block_on(self.engine.generate(self.request(name))).expect("accepted generation")
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
            let snapshot = self.admission.snapshot().expect("admission snapshot");

            if snapshot.reserved_bytes() == 0 {
                assert_eq!(snapshot.active_reservations(), 0,);
                return;
            }

            assert!(
                Instant::now() < deadline,
                "reservations remained after cleanup: {snapshot:?}",
            );

            thread::yield_now();
        }
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

    loop {
        match next_event(stream, deadline) {
            Some(event) => {
                assert!(
                    terminal.is_none(),
                    "event emitted after terminal event: {event:?}",
                );

                if event.is_terminal() {
                    terminal = Some(event);
                    continue;
                }

                match event {
                    GenerationEvent::TextDelta { text: delta } => {
                        text.push_str(&delta);
                    }

                    other => {
                        panic!("unexpected fixture event: {other:?}");
                    }
                }
            }

            None => break,
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
            assert_eq!(*finish_reason, expected_reason,);

            *usage
        }

        other => {
            panic!("expected successful terminal event, got {other:?}");
        }
    }
}

#[test]
fn real_generation_streams_exact_output_and_measured_usage() {
    let environment = TestEnvironment::new(4, 8, false);

    let generation = environment.start("real-output-limit");

    assert_eq!(generation.request_id().as_str(), "real-output-limit",);

    let mut events = generation.into_events();

    let transcript = drain(&mut events);

    assert_eq!(transcript.text, "!!!!",);

    let usage = completed_usage(&transcript, FinishReason::OutputLimit);

    assert_eq!(usage.input_tokens, environment.expected_input_tokens,);

    assert_eq!(usage.output_tokens, 4,);

    for _ in 0..4 {
        assert!(next_event(&mut events, deadline(),).is_none());
    }

    drop(events);

    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn real_eos_stops_generation_and_is_counted_without_display_text() {
    let environment = TestEnvironment::new(8, 8, true);

    let generation = environment.start("real-eos");

    let mut events = generation.into_events();

    let transcript = drain(&mut events);

    assert!(transcript.text.is_empty());

    let usage = completed_usage(&transcript, FinishReason::EndOfSequence);

    assert_eq!(usage.input_tokens, environment.expected_input_tokens,);

    assert_eq!(usage.output_tokens, 1,);

    drop(events);

    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn bounded_stream_keeps_execution_owned_until_output_can_progress() {
    let environment = TestEnvironment::new(8, 1, false);

    let generation = environment.start("backpressured-generation");

    // Eight non-EOS text events cannot complete into a one-event queue
    // before its consumer begins draining it.
    let error = expect_failure(
        block_on(
            environment
                .engine
                .generate(environment.request("competing-generation")),
        ),
        "competing generation bypassed execution ownership",
    );

    assert_eq!(error.kind(), ErrorKind::Busy,);

    let mut events = generation.into_events();

    let transcript = drain(&mut events);

    assert_eq!(transcript.text, "!".repeat(8),);

    let usage = completed_usage(&transcript, FinishReason::OutputLimit);

    assert_eq!(usage.output_tokens, 8);

    drop(events);

    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn cancellation_terminates_generation_without_draining_for_capacity() {
    let environment = TestEnvironment::new(32, 1, false);

    let generation = environment.start("cancel-backpressured");

    let request = generation.request_id().clone();

    // With no consumer polling and only one queue position, this
    // non-EOS fixture cannot have completed all 32 tokens.
    assert_eq!(
        block_on(environment.engine.cancel(request.clone(),),).expect("cancel"),
        CancelOutcome::Requested,
    );

    let mut events = generation.into_events();

    let transcript = drain(&mut events);

    match transcript.terminal {
        GenerationEvent::RequestFailed {
            error,
            partial_usage,
        } => {
            assert_eq!(error.kind(), ErrorKind::Cancelled,);

            assert!(partial_usage.input_tokens <= environment.expected_input_tokens);

            assert!(partial_usage.output_tokens <= 32);

            match error {
                EngineError::Cancelled { request_id } => {
                    assert_eq!(request_id, request.as_str(),);
                }

                other => {
                    panic!("unexpected cancellation error: {other:?}");
                }
            }
        }

        other => {
            panic!("expected cancelled terminal event, got {other:?}");
        }
    }

    drop(events);

    environment.wait_idle();
    environment.wait_unreserved();

    assert_eq!(
        block_on(environment.engine.cancel(request),).expect("finished cancellation"),
        CancelOutcome::AlreadyFinished,
    );
}

#[test]
fn dropping_the_stream_cancels_work_and_allows_a_fresh_request() {
    let environment = TestEnvironment::new(8, 1, false);

    let generation = environment.start("dropped-stream");

    drop(generation.into_events());

    environment.wait_idle();
    environment.wait_unreserved();

    let generation = environment.start("request-after-disconnect");

    let mut events = generation.into_events();

    let transcript = drain(&mut events);

    assert_eq!(transcript.text, "!".repeat(8),);

    let usage = completed_usage(&transcript, FinishReason::OutputLimit);

    assert_eq!(usage.output_tokens, 8);

    drop(events);

    environment.wait_idle();
    environment.wait_unreserved();
}

#[test]
fn output_reservation_outlives_worker_until_stream_is_dropped() {
    let environment = TestEnvironment::new(4, 8, false);

    let generation = environment.start("undrained-output");

    let mut events = generation.into_events();

    // Queue capacity exceeds the four fixture output events. Model work
    // can finish without the consumer having polled any events.
    environment.wait_idle();

    let snapshot = environment
        .admission
        .snapshot()
        .expect("undrained output accounting");

    assert_eq!(
        snapshot.reserved_bytes(),
        environment.output_reservation_bytes,
    );

    assert_eq!(snapshot.active_reservations(), 1,);

    let transcript = drain(&mut events);

    assert_eq!(transcript.text, "!!!!",);

    // The exhausted stream still owns its allocated channel until drop.
    let exhausted_snapshot = environment
        .admission
        .snapshot()
        .expect("exhausted stream accounting");

    assert_eq!(
        exhausted_snapshot.reserved_bytes(),
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
        block_on(environment.engine.cancel(request),).expect("cancel completed request"),
        CancelOutcome::AlreadyFinished,
    );

    for _ in 0..4 {
        assert!(next_event(&mut events, deadline(),).is_none());
    }

    drop(events);

    environment.wait_idle();
    environment.wait_unreserved();
}
