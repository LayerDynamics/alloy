//! Actual Gemma3 family operations through engine contracts; no model mocks.
use engine::{
    admission::AdmissionController,
    api::Engine,
    backends::{
        FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution,
        qwen::{GEMMA3_CACHE_REPRESENTATION, GEMMA3_TEXT_TEMPLATE},
    },
    cancellation::CancellationSource,
    family::{ExecutionControl, LoadedModel, ModelFamily, ModelLoadRequest},
    generation::GenerationLimits,
    registry::{ModelProfile, Registry},
    runtime::{RetentionPolicy, RuntimeBackend},
    scheduler::Scheduler,
    types::{
        ArtifactId, ContextSnapshot, ContextSnapshotId, GenerationEvent, GenerationOptions,
        GenerationRequest, Message, MessageRole, ModelProfileId, RequestId,
    },
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    future::Future,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

struct WakeThread(thread::Thread);
impl Wake for WakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn wait<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(WakeThread(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(240);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        assert!(Instant::now() < deadline, "gemma3 test deadline");
        thread::park_timeout(Duration::from_millis(10));
    }
}
fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn control() -> (CancellationSource, ExecutionControl) {
    let source = CancellationSource::new();
    let control = ExecutionControl::new(RequestId::new("gemma3-test").unwrap(), source.token());
    (source, control)
}
fn profile_id() -> ModelProfileId {
    ModelProfileId::new("gemma3-test").unwrap()
}
fn profile(context: usize) -> ModelProfile {
    ModelProfile::new(
        profile_id(),
        ArtifactId::new("gemma3").unwrap(),
        context,
        32,
        8,
        GenerationOptions::greedy(32).unwrap(),
    )
    .unwrap()
}
fn limits(context: usize) -> Qwen3BackendLimits {
    Qwen3BackendLimits {
        max_files: 32,
        max_config_bytes: 2 << 20,
        max_tokenizer_bytes: 40 << 20,
        max_messages: 64,
        max_input_bytes: 1 << 20,
        max_rendered_bytes: 2 << 20,
        max_token_bytes: 64 << 10,
        verification_chunk_bytes: 1 << 20,
        context_limit_tokens: context,
        prefill_chunk_tokens: NonZeroUsize::new(32).unwrap(),
        loader_headroom_bytes: 64 << 20,
        execution_headroom_bytes: 64 << 20,
        generation: GenerationLimits::new(64, 64 << 10, 16 << 10).unwrap(),
    }
}
struct Fixture {
    root: PathBuf,
    specification: Qwen3Artifact,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
impl Fixture {
    fn new() -> Self {
        Self::with_context(128)
    }
    fn with_context(context: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = loop {
            let p = std::env::temp_dir().join(format!(
                "alloy-gemma3-engine-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&p) {
                Ok(()) => break p,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("fixture directory: {e}"),
            }
        };
        let mut config: Value = serde_json::from_str(include_str!(
            "../../inference-infra/tests/fixtures/gemma3_model/config.json"
        ))
        .unwrap();
        config["eos_token_id"] = json!([97, 99]);
        config["max_position_embeddings"] = json!(context);
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("model.safetensors"),
            include_bytes!("../../inference-infra/tests/fixtures/gemma3_model/model.safetensors"),
        )
        .unwrap();
        let mut vocab: BTreeMap<String, u32> = (33u8..=126)
            .enumerate()
            .map(|(i, c)| ((c as char).to_string(), i as u32))
            .collect();
        vocab.insert("▁".into(), 94);
        vocab.insert("\n".into(), 95);
        let specials = [
            (96, "<bos>"),
            (97, "<eos>"),
            (98, "<start_of_turn>"),
            (99, "<end_of_turn>"),
        ];
        for (id, spelling) in [
            (100, "<0xE9>"),
            (101, "<0x9B>"),
            (102, "<0xAA>"),
            (103, "<0xFF>"),
            (104, "<0x61>"),
            (105, "雪"),
            (106, "▁hello"),
            (107, "<0xe2>"),
            (108, "<0x82>"),
            (109, "<0xac>"),
        ] {
            vocab.insert(spelling.into(), id);
        }
        for (id, content) in specials {
            vocab.insert(content.into(), id);
        }
        let added: Vec<_> = specials.into_iter().map(|(id, content)|
            json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect();
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,
            "normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},"post_processor":null,
            "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
            "decoder":{"type":"Sequence","decoders":[
                {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                {"type":"ByteFallback"},{"type":"Fuse"}]},
            "model":{"type":"BPE","vocab":vocab,"merges":[],"dropout":null,"unk_token":null,
                     "continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":true,"ignore_merges":false}});
        std::fs::write(
            root.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            br#"{"eos_token":"<eos>"}"#,
        )
        .unwrap();
        let files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(root.join(name)).unwrap();
            (
                name.into(),
                FilePin {
                    bytes: bytes.len() as u64,
                    sha256: hash(&bytes),
                },
            )
        })
        .collect();
        let specification = Qwen3Artifact {
            id: ArtifactId::new("gemma3").unwrap(),
            directory: root.clone(),
            files,
            template_sha256: hash(GEMMA3_TEXT_TEMPLATE.as_bytes()),
        };
        Self {
            root,
            specification,
        }
    }
    fn backend(&self, execution: Qwen3Execution) -> Qwen3Backend {
        Qwen3Backend::new_gemma3(vec![self.specification.clone()], limits(128), execution).unwrap()
    }
}

fn load(
    backend: &Qwen3Backend,
    registry: &Registry,
    control: &ExecutionControl,
) -> Box<dyn LoadedModel> {
    let plan = wait(backend.plan(registry, &profile_id(), 16)).unwrap();
    let artifact = registry
        .artifact(&ArtifactId::new("gemma3").unwrap())
        .unwrap();
    backend.verify_artifact(artifact, control).unwrap();
    backend
        .load(ModelLoadRequest::new(artifact, plan.load_limits, control))
        .unwrap()
}
fn executions() -> Vec<Qwen3Execution> {
    let mut result = vec![Qwen3Execution::MlxCpu];
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        result.push(Qwen3Execution::MlxMetal);
    }
    result
}

fn context(messages: Vec<Message>) -> ContextSnapshot {
    ContextSnapshot::new(
        ContextSnapshotId::new("gemma-history").unwrap(),
        profile_id(),
        messages,
    )
    .unwrap()
}
fn message(role: MessageRole, content: &str) -> Message {
    Message {
        role,
        content: content.into(),
    }
}

#[test]
fn gemma3_engine_transition_memory_is_reserved_once() {
    let fixture = Fixture::new();
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend
        .registry(vec![
            ModelProfile::new(
                profile_id(),
                ArtifactId::new("gemma3").unwrap(),
                128,
                1,
                1,
                GenerationOptions::greedy(1).unwrap(),
            )
            .unwrap(),
        ])
        .unwrap();
    // Independent arithmetic from the frozen fixture geometry: two local
    // layers (window 4), one global layer, two KV heads of width 32, F32.
    // Exercise both sides of the sliding-window boundary and chunk boundary.
    for input in [1usize, 2, 3, 16, 31, 32, 33, 126] {
        let plan = wait(backend.plan(&registry, &profile_id(), input)).unwrap();
        let capacity = (input + 2) as u64;
        let chunk = input.min(32) as u64;
        let kv = (2 * capacity.min(4) + capacity) * 2 * 32 * 2 * 4;
        let layers = chunk * (192 * 16 + 256 * 8 + (4 * 32 + 2 * 2 * 32) * 8) * 4;
        let attention = chunk * capacity * 4 * 4 * 4;
        let native_work = layers + attention + 32 * 16 + 3 * kv;
        let native_output = chunk * 128 * 4 * 3;
        let bookkeeping = (64 << 10) * 8 + (16 << 10);
        let headroom = 64 << 20;
        assert_eq!(plan.cache.max_cache_bytes(), 2 * kv + 2 * headroom);
        assert_eq!(
            plan.cache.max_cache_bytes() + plan.temporary_bytes,
            native_work + native_output + bookkeeping + 2 * headroom,
            "each transition buffer must be covered exactly once at capacity {capacity}"
        );
        let components = [
            plan.cache.max_cache_bytes(),
            plan.temporary_bytes,
            plan.headroom_bytes,
            plan.output_bytes,
        ];
        let total = components.iter().sum();
        let exact = AdmissionController::new(total).unwrap();
        let grant = exact
            .reserve_grant("Gemma transition", &components)
            .unwrap();
        drop(grant);
        assert!(
            AdmissionController::new(total - 1)
                .unwrap()
                .reserve_grant("Gemma transition", &components)
                .is_err()
        );
    }
}

#[test]
fn gemma3_engine_render_and_decode_contract() {
    let fixture = Fixture::new();
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend.registry(vec![profile(128)]).unwrap();
    let rendered = wait(backend.render_and_count(
        &registry,
        context(vec![
            message(MessageRole::System, " Be concise. "),
            message(MessageRole::User, " Hi "),
            message(MessageRole::Assistant, " Hello "),
            message(MessageRole::User, " Next "),
        ]),
    ))
    .unwrap();
    assert_eq!(
        rendered.text(),
        "<bos><start_of_turn>user\n Be concise. \n\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>\n<start_of_turn>user\nNext<end_of_turn>\n<start_of_turn>model\n"
    );
    assert_eq!(rendered.token_ids()[0], 96);
    assert_eq!(
        rendered.token_ids().iter().filter(|&&id| id == 96).count(),
        1
    );
    assert_eq!(
        backend.eos_tokens(&registry, &profile_id()).unwrap(),
        [97, 99]
    );
    let tokenizer = tokenizers::Tokenizer::from_file(fixture.root.join("tokenizer.json")).unwrap();
    // Includes valid multibyte UTF-8, a malformed run after valid ASCII,
    // incomplete tails, skipped special tokens, lower-case hex and whitespace.
    for ids in [
        vec![106, 105],
        vec![100, 101, 102],
        vec![100, 101, 102, 106],
        vec![104, 103, 106],
        vec![100],
        vec![100, 96, 101, 102],
        vec![107, 108, 109, 94],
    ] {
        let mut decoder = backend.decoder(&registry, &profile_id()).unwrap();
        let mut text = String::new();
        for &id in &ids {
            text.push_str(&decoder.push(id).unwrap());
        }
        text.push_str(&decoder.finish().unwrap());
        assert_eq!(text, tokenizer.decode(&ids, true).unwrap(), "{ids:?}");
        assert_eq!(decoder.finish().unwrap(), "");
        assert!(decoder.push(0).is_err());
    }
    assert_eq!(
        backend.registration().capabilities().cache_modes[0].kind,
        engine::types::CacheKind::Mixed
    );
    assert_eq!(
        backend.registration().capabilities().cache_modes[0]
            .representation
            .as_str(),
        GEMMA3_CACHE_REPRESENTATION
    );
    assert!(!backend.registration().capabilities().supports_prefix_reuse);
}
#[test]
fn gemma3_engine_grouped_history_matches_publisher_template() {
    let fixture = Fixture::with_context(4096);
    let backend = Qwen3Backend::new_gemma3(
        vec![fixture.specification.clone()],
        limits(4096),
        Qwen3Execution::MlxCpu,
    )
    .unwrap();
    let registry = backend.registry(vec![profile(4096)]).unwrap();
    let input: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/gemma3_text_render/input.json")).unwrap();
    let messages = input
        .iter()
        .map(|value| {
            message(
                match value["role"].as_str().unwrap() {
                    "system" => MessageRole::System,
                    "user" => MessageRole::User,
                    "assistant" => MessageRole::Assistant,
                    _ => panic!("fixture role"),
                },
                value["content"].as_str().unwrap(),
            )
        })
        .collect();
    let rendered = wait(backend.render_and_count(&registry, context(messages))).unwrap();
    assert_eq!(
        rendered.text(),
        include_str!("fixtures/gemma3_text_render/expected.txt")
    );
    assert_eq!(
        rendered.token_ids().iter().filter(|&&id| id == 96).count(),
        1
    );
    assert_eq!(rendered.text().matches("<start_of_turn>user\n").count(), 2);
    assert!(rendered.text().contains("PR-42; art:12."));
    assert!(rendered.text().contains("COBALT\nSaved note: 雪."));
}

#[test]
fn gemma3_engine_rejects_invalid_tokenizer_and_tools() {
    let mut fixture = Fixture::new();
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend.registry(vec![profile(128)]).unwrap();
    for text in [
        "<bos>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<start_of_image>",
        "<image_soft_token>",
    ] {
        assert!(
            wait(
                backend
                    .render_and_count(&registry, context(vec![message(MessageRole::User, text)]))
            )
            .is_err()
        );
    }
    for messages in [
        vec![message(MessageRole::Assistant, "Hi")],
        vec![
            message(MessageRole::User, "Hi"),
            message(MessageRole::System, "Unsupported late system"),
            message(MessageRole::User, "Next"),
        ],
    ] {
        assert!(wait(backend.render_and_count(&registry, context(messages))).is_err());
    }
    assert!(
        backend
            .with_tool_call_parsing(true)
            .parser_allocation_limits(&registry, &profile_id())
            .is_err()
    );
    // The converted real dialect omits this field. Registry and native loader
    // must use the same validated family default; invalid explicit values fail.
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(fixture.root.join("config.json")).unwrap()).unwrap();
    config
        .as_object_mut()
        .unwrap()
        .remove("max_position_embeddings");
    let bytes = serde_json::to_vec(&config).unwrap();
    std::fs::write(fixture.root.join("config.json"), &bytes).unwrap();
    fixture.specification.files.insert(
        "config.json".into(),
        FilePin {
            bytes: bytes.len() as u64,
            sha256: hash(&bytes),
        },
    );
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend.registry(vec![profile(128)]).unwrap();
    assert_eq!(
        registry
            .artifact(&ArtifactId::new("gemma3").unwrap())
            .unwrap()
            .declared_context_limit_tokens(),
        32768
    );
    config["max_position_embeddings"] = json!(0);
    std::fs::write(
        fixture.root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut invalid_registry = Registry::new();
    invalid_registry
        .register_family(backend.registration().clone())
        .unwrap();
    assert!(
        invalid_registry
            .register_artifact(ArtifactId::new("invalid").unwrap(), &fixture.root)
            .is_err()
    );
    std::fs::write(fixture.root.join("config.json"), bytes).unwrap();
    let original: Value =
        serde_json::from_slice(&std::fs::read(fixture.root.join("tokenizer.json")).unwrap())
            .unwrap();
    for mutation in 0..4 {
        let mut value = original.clone();
        match mutation {
            0 => value["decoder"]["decoders"][0]["content"] = json!("_"),
            1 => value["decoder"]["decoders"]
                .as_array_mut()
                .unwrap()
                .reverse(),
            2 => value["model"]["byte_fallback"] = json!(false),
            _ => {
                value["decoder"] = json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true})
            }
        }
        let bytes = serde_json::to_vec(&value).unwrap();
        std::fs::write(fixture.root.join("tokenizer.json"), &bytes).unwrap();
        fixture.specification.files.insert(
            "tokenizer.json".into(),
            FilePin {
                bytes: bytes.len() as u64,
                sha256: hash(&bytes),
            },
        );
        assert!(
            Qwen3Backend::new_gemma3(
                vec![fixture.specification.clone()],
                limits(128),
                Qwen3Execution::MlxCpu
            )
            .is_err()
        );
    }
}
#[test]
fn gemma3_engine_native_cache_and_cancellation() {
    let fixture = Fixture::new();
    for execution in executions() {
        let backend = fixture.backend(execution);
        let registry = backend.registry(vec![profile(128)]).unwrap();
        let (source, control) = control();
        let plan = wait(backend.plan(&registry, &profile_id(), 16)).unwrap();
        let mut loaded = load(&backend, &registry, &control);
        let mut second = load(&backend, &registry, &control);
        let model = loaded.autoregressive().unwrap();
        let mut cache = model.allocate_cache(&plan.cache, &control).unwrap();
        assert!(!cache.supports_truncation());
        assert!(cache.kv_block_spec().is_none());
        assert!(cache.truncate(0).is_err());
        let short =
            engine::family::CacheAllocationRequest::new(plan.cache.mode().clone(), 128, 1).unwrap();
        assert!(model.allocate_cache(&short, &control).is_err());
        let prompt = [3, 41, 7, 120, 6, 12, 19, 22];
        let first = model.prefill(&prompt, cache.as_mut(), &control).unwrap();
        assert_eq!(first.logits().values().len(), 128);
        assert_eq!(cache.len(), 8);
        assert_eq!(cache.retained_tokens(), 4);
        let memory = cache.memory_usage().unwrap();
        assert!(memory.allocated_bytes() >= memory.used_bytes());
        assert!(memory.used_bytes() > 0);
        let decoded = model
            .decode(19, cache.as_mut(), &control)
            .unwrap()
            .values()
            .to_vec();
        assert!(
            second
                .autoregressive()
                .unwrap()
                .decode(19, cache.as_mut(), &control)
                .is_err()
        );
        assert_eq!(cache.len(), 9);
        source.request_cancel();
        assert!(model.decode(19, cache.as_mut(), &control).is_err());
        assert_eq!(cache.len(), 9);
        let source = CancellationSource::new();
        let control =
            ExecutionControl::new(RequestId::new("gemma-recover").unwrap(), source.token());
        cache.reset().unwrap();
        assert_eq!(cache.memory_usage().unwrap().used_bytes(), 0);
        assert_eq!(
            model
                .prefill(&prompt, cache.as_mut(), &control)
                .unwrap()
                .logits()
                .values(),
            first.logits().values()
        );
        assert_eq!(
            model.decode(19, cache.as_mut(), &control).unwrap().values(),
            decoded
        );
    }
}

#[test]
fn gemma3_engine_structured_output_exhaustion_releases_workspace() {
    use engine::structured_output::{JsonFieldType, JsonOutputSchema};
    let fixture = Fixture::new();
    for execution in executions() {
        let admission = AdmissionController::new(1 << 30).unwrap();
        let scheduler = Scheduler::new(8).unwrap();
        let engine = fixture
            .backend(execution)
            .into_engine(
                vec![profile(128)],
                admission.clone(),
                scheduler.clone(),
                RetentionPolicy {
                    retain_model: false,
                    retain_cache_buffers: false,
                },
            )
            .unwrap();
        let schema =
            JsonOutputSchema::new(vec![("v".into(), JsonFieldType::UnsignedConstant(1))]).unwrap();
        assert!(
            GenerationRequest::new(
                RequestId::new("json-tools").unwrap(),
                profile_id(),
                tools_context(true, None),
                GenerationOptions::greedy(32)
                    .unwrap()
                    .with_json_schema(schema.clone()),
            )
            .is_err()
        );
        // This untrained numerical fixture selects newlines. It establishes
        // real native exhaustion/cleanup/re-entry, not language conformance.
        // Successful constrained checkpoints are checked with the frozen models.
        for (name, limit) in [("partial", 1), ("exhausted", 32), ("repeat", 32)] {
            let request = GenerationRequest::new(
                RequestId::new(name).unwrap(),
                profile_id(),
                context(vec![message(MessageRole::User, "JSON")]),
                GenerationOptions::greedy(limit)
                    .unwrap()
                    .with_json_schema(schema.clone()),
            )
            .unwrap();
            let mut events = wait(engine.generate(request)).unwrap().into_events();
            let mut text = String::new();
            let mut terminal = false;
            while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
                assert!(!terminal);
                match event {
                    GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
                    GenerationEvent::RequestFailed { error, .. } => {
                        assert!(error.to_string().contains("incomplete structured output"));
                        assert_eq!(text, "\n".repeat(limit), "{name}");
                        terminal = true;
                    }
                    other => panic!("unexpected structured event {other:?}"),
                }
            }
            assert!(terminal);
            // The stream itself owns the ordinary admitted output slab until
            // dropped, even after its terminal event has been drained.
            drop(events);
            let deadline = Instant::now() + Duration::from_secs(10);
            while scheduler.is_busy().unwrap() {
                assert!(Instant::now() < deadline);
                thread::yield_now();
            }
            assert_eq!(
                admission.snapshot().unwrap().reserved_bytes(),
                0,
                "constraint reservation must be released"
            );
        }
    }
}

#[test]
fn gemma3_engine_structured_output_admission_counts_fixed_workspace() {
    use engine::structured_output::{JSON_WORKSPACE_BYTES, JsonFieldType, JsonOutputSchema};
    let fixture = Fixture::new();
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend.registry(vec![profile(128)]).unwrap();
    let context = context(vec![message(MessageRole::User, "JSON")]);
    let rendered = wait(backend.render_and_count(&registry, context.clone())).unwrap();
    let plan = wait(backend.plan(&registry, &profile_id(), rendered.token_count())).unwrap();
    let plain = plan.load_limits.max_load_peak_bytes()
        + plan.cache.max_cache_bytes()
        + plan.temporary_bytes
        + plan.output_bytes
        + plan.headroom_bytes;
    let admission = AdmissionController::new(plain + JSON_WORKSPACE_BYTES - 1).unwrap();
    let engine = backend
        .into_engine(
            vec![profile(128)],
            admission.clone(),
            Scheduler::new(8).unwrap(),
            RetentionPolicy {
                retain_model: false,
                retain_cache_buffers: false,
            },
        )
        .unwrap();
    assert_eq!(
        wait(engine.estimate_memory(profile_id(), rendered.token_count()))
            .unwrap()
            .reserved_bytes(),
        plain
    );
    let request = GenerationRequest::new(
        RequestId::new("json-no-room").unwrap(),
        profile_id(),
        context,
        GenerationOptions::greedy(32).unwrap().with_json_schema(
            JsonOutputSchema::new(vec![("v".into(), JsonFieldType::UnsignedConstant(1))]).unwrap(),
        ),
    )
    .unwrap();
    assert!(matches!(
        wait(engine.generate(request)),
        Err(engine::error::EngineError::MemoryLimitExceeded { .. })
    ));
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
}

#[test]
fn gemma3_engine_json_strings_preserve_tool_syntax_as_data() {
    use engine::structured_output::{JsonFieldType, JsonOutputSchema};
    let fixture = Fixture::new();
    let mut generous = limits(128);
    generous.max_token_bytes = 512 << 10; // daemon policy, not actual vocabulary size
    let backend = Qwen3Backend::new_gemma3(
        vec![fixture.specification.clone()],
        generous,
        Qwen3Execution::MlxCpu,
    )
    .unwrap()
    .with_parser_allocation_limits(parser_limits())
    .unwrap();
    let registry = backend.registry(vec![profile(128)]).unwrap();
    let (_, control) = control();
    let schema =
        JsonOutputSchema::new(vec![("s".into(), JsonFieldType::String { nonempty: true })])
            .unwrap();
    let mut constraint = backend
        .json_constraint(&registry, &profile_id(), schema, &control)
        .unwrap();
    let mut decoder = backend.json_decoder(&registry, &profile_id()).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(fixture.root.join("tokenizer.json")).unwrap();
    let raw = r#"{"s":"<tool_call>untrusted</tool_call> 雪"}"#;
    let tokens = tokenizer.encode(raw, false).unwrap();
    let mut text = String::new();
    for &token in tokens.get_ids() {
        constraint.consume(token).unwrap();
        for block in decoder.push_blocks(token).unwrap() {
            match block {
                engine::types::GenerationOutputBlock::Text(delta) => text.push_str(&delta),
                other => panic!("JSON emitted protocol block: {other:?}"),
            }
        }
        assert!(!decoder.is_sequence_complete());
    }
    constraint.finish().unwrap();
    for block in decoder.finish_blocks().unwrap() {
        match block {
            engine::types::GenerationOutputBlock::Text(delta) => text.push_str(&delta),
            other => panic!("JSON emitted protocol block: {other:?}"),
        }
    }
    assert!(decoder.take_parser_proposals().is_none());
    assert_eq!(text, raw);
}

#[test]
#[ignore = "requires pinned ALLOY_GEMMA3_SMALL_ARTIFACT and Metal"]
fn gemma3_real_4b_engine_streaming() {
    real_engine(true);
}
#[test]
#[ignore = "requires pinned ALLOY_GEMMA3_ARTIFACT and Metal"]
fn gemma3_real_12b_engine_streaming() {
    real_engine(false);
}
fn real_engine(small: bool) {
    let backend = real_backend(small);
    let output_limit = 32;
    let configured_profile = ModelProfile::new(
        profile_id(),
        ArtifactId::new("gemma3").unwrap(),
        512,
        128,
        8,
        GenerationOptions::greedy(32).unwrap(),
    )
    .unwrap();
    let context = context(vec![message(
        MessageRole::User,
        "Calculate 2 + 2. Reply with only the result.",
    )]);
    let registry = backend.registry(vec![configured_profile.clone()]).unwrap();
    let counted = wait(backend.render_and_count(&registry, context.clone())).unwrap();
    let expected_input = counted.token_count();
    assert_eq!(
        counted.text(),
        "<bos><start_of_turn>user\nCalculate 2 + 2. Reply with only the result.<end_of_turn>\n<start_of_turn>model\n"
    );
    let directory = PathBuf::from(
        std::env::var_os(if small {
            "ALLOY_GEMMA3_SMALL_ARTIFACT"
        } else {
            "ALLOY_GEMMA3_ARTIFACT"
        })
        .unwrap(),
    );
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json")).unwrap();
    assert_eq!(
        tokenizer.encode(counted.text(), false).unwrap().get_ids(),
        counted.token_ids()
    );
    for text in [
        "雪 café 🙂",
        " leading and  double spaces\n",
        "देवनागरी",
        "a\u{0301}",
    ] {
        let ids = tokenizer.encode(text, false).unwrap();
        let mut decoder = backend.decoder(&registry, &profile_id()).unwrap();
        let mut decoded = String::new();
        for &id in ids.get_ids() {
            decoded.push_str(&decoder.push(id).unwrap());
        }
        decoded.push_str(&decoder.finish().unwrap());
        assert_eq!(decoded, tokenizer.decode(ids.get_ids(), true).unwrap());
        assert_eq!(decoded, text);
    }
    for spellings in [
        vec!["<0xE9>", "<0x9B>", "<0xAA>"],
        vec!["<0x61>", "<0xFF>"],
        vec!["<0xE9>"],
    ] {
        let ids: Vec<_> = spellings
            .iter()
            .map(|s| tokenizer.token_to_id(s).unwrap())
            .collect();
        let mut decoder = backend.decoder(&registry, &profile_id()).unwrap();
        let mut decoded = String::new();
        for &id in &ids {
            decoded.push_str(&decoder.push(id).unwrap());
        }
        decoded.push_str(&decoder.finish().unwrap());
        assert_eq!(decoded, tokenizer.decode(&ids, true).unwrap());
    }
    let admission = AdmissionController::new(36 * 1024 * 1024 * 1024).unwrap();
    let scheduler = Scheduler::new(8).unwrap();
    let engine = backend
        .into_engine(
            vec![configured_profile],
            admission.clone(),
            scheduler.clone(),
            RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: true,
            },
        )
        .unwrap();
    let mut outputs = Vec::new();
    let mut retained_bytes = None;
    for name in ["gemma3-cold", "gemma3-repeat"] {
        let request = GenerationRequest::new(
            RequestId::new(name).unwrap(),
            profile_id(),
            context.clone(),
            GenerationOptions::greedy(output_limit).unwrap(),
        )
        .unwrap();
        let mut events = wait(engine.generate(request)).unwrap().into_events();
        let mut text = String::new();
        let mut completed = false;
        while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
            assert!(!completed, "event after terminal");
            match event {
                GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
                GenerationEvent::GenerationCompleted {
                    finish_reason,
                    usage,
                } => {
                    assert_eq!(usage.input_tokens, expected_input);
                    assert_eq!(finish_reason, engine::types::FinishReason::EndOfSequence);
                    eprintln!("Gemma3 {name}: {finish_reason:?} {usage:?}");
                    completed = true;
                }
                other => panic!("unexpected Gemma3 event {other:?}"),
            }
        }
        assert!(completed);
        assert_eq!(text.trim(), "4", "Gemma3 streamed output {text:?}");
        outputs.push(text);
        let deadline = Instant::now() + Duration::from_secs(30);
        while scheduler.is_busy().unwrap() {
            assert!(Instant::now() < deadline);
            thread::yield_now();
        }
        let reserved = admission.snapshot().unwrap().reserved_bytes();
        assert!(reserved > 0, "retained native state must remain accounted");
        if let Some(previous) = retained_bytes {
            assert_eq!(
                reserved, previous,
                "repeated generation leaked a reservation"
            );
        }
        retained_bytes = Some(reserved);
        eprintln!("Gemma3 small={small} {name}: retained reservation={reserved}");
        if name == "gemma3-cold" {
            let id = RequestId::new("gemma3-cancel").unwrap();
            let cancel_context = ContextSnapshot::new(
                ContextSnapshotId::new("cancel-context").unwrap(),
                profile_id(),
                vec![message(
                    MessageRole::User,
                    "List all integers from 1 to 200, one integer per line.",
                )],
            )
            .unwrap();
            let request = GenerationRequest::new(
                id.clone(),
                profile_id(),
                cancel_context,
                GenerationOptions::greedy(128).unwrap(),
            )
            .unwrap();
            let mut events = wait(engine.generate(request)).unwrap().into_events();
            let mut requested = false;
            let mut terminal = false;
            while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
                assert!(!terminal, "event after cancellation terminal");
                match event {
                    GenerationEvent::TextDelta { .. } => {
                        if !requested {
                            assert_eq!(
                                wait(engine.cancel(id.clone())).unwrap(),
                                engine::types::CancelOutcome::Requested
                            );
                            requested = true;
                        }
                    }
                    GenerationEvent::RequestFailed {
                        error: engine::error::EngineError::Cancelled { .. },
                        ..
                    } => terminal = true,
                    other => panic!("unexpected cancellation event {other:?}"),
                }
            }
            assert!(requested && terminal);
            let deadline = Instant::now() + Duration::from_secs(30);
            while scheduler.is_busy().unwrap() {
                assert!(Instant::now() < deadline);
                thread::yield_now();
            }
            eprintln!(
                "Gemma3 small={small}: live generation cancelled after output, next request must reproduce cold output"
            );
        }
    }
    assert_eq!(outputs[0], outputs[1]);
    drop(engine);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let snapshot = admission.snapshot().unwrap();
        if snapshot.reserved_bytes() == 0 {
            assert_eq!(snapshot.active_reservations(), 0);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "shutdown leaked reservations: {snapshot:?}"
        );
        thread::yield_now();
    }
}
fn real_backend(small: bool) -> Qwen3Backend {
    real_backend_with_context(small, 512)
}
fn real_backend_with_context(small: bool, context_limit: usize) -> Qwen3Backend {
    let frozen: Value = serde_json::from_str(if small {
        include_str!("../../inference-infra/tests/fixtures/gemma3_4b_profile.json")
    } else {
        include_str!("../../inference-infra/tests/fixtures/gemma3_12b_profile.json")
    })
    .unwrap();
    let files = frozen["artifact"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| {
            let hex = file["sha256"].as_str().unwrap();
            let mut sha256 = [0; 32];
            for (i, b) in sha256.iter_mut().enumerate() {
                *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
            }
            (
                file["path"].as_str().unwrap().into(),
                FilePin {
                    bytes: file["bytes"].as_u64().unwrap(),
                    sha256,
                },
            )
        })
        .collect();
    Qwen3Backend::new_gemma3(
        vec![Qwen3Artifact {
            id: ArtifactId::new("gemma3").unwrap(),
            directory: PathBuf::from(
                std::env::var_os(if small {
                    "ALLOY_GEMMA3_SMALL_ARTIFACT"
                } else {
                    "ALLOY_GEMMA3_ARTIFACT"
                })
                .expect("artifact"),
            ),
            files,
            template_sha256: hash(GEMMA3_TEXT_TEMPLATE.as_bytes()),
        }],
        limits(context_limit),
        Qwen3Execution::MlxMetal,
    )
    .unwrap()
}
fn parser_limits() -> engine::parser_allocation::ParserAllocationLimits {
    engine::parser_allocation::ParserAllocationLimits {
        carry_bytes: 64,
        call_bytes: 8192,
        decoded_bytes: 8192,
        depth: 64,
        keys: 512,
        nodes: 1024,
        proposal_bytes: 16384,
        calls: 8,
    }
}
fn tool_fixture_context() -> ContextSnapshot {
    use engine::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    let input: Value =
        serde_json::from_str(include_str!("fixtures/gemma3_tool_render/input.json")).unwrap();
    let role = |s: &str| match s {
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        _ => panic!("role"),
    };
    let messages = input["base_messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| Message {
            role: role(v["role"].as_str().unwrap()),
            content: v["content"].as_str().unwrap().into(),
        })
        .collect();
    let tools = input["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let f = &v["function"];
            RenderingTool::new(
                f["name"].as_str().unwrap().into(),
                f["version"].as_str().unwrap().into(),
                f["description"].as_str().unwrap().into(),
                f["parameters"].clone(),
            )
            .unwrap()
        })
        .collect();
    let history = input["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| match v["kind"].as_str().unwrap() {
            "call" => ToolHistoryBlock::AssistantCall {
                call_id: v["call_id"].as_str().unwrap().into(),
                tool_id: v["tool_id"].as_str().unwrap().into(),
                arguments: v["arguments"].as_object().unwrap().clone(),
            },
            "result" => ToolHistoryBlock::ToolResult {
                call_id: v["call_id"].as_str().unwrap().into(),
                content: v["content"].as_str().unwrap().into(),
                is_error: v["is_error"].as_bool().unwrap(),
            },
            "text" => ToolHistoryBlock::Text(Message {
                role: role(v["role"].as_str().unwrap()),
                content: v["content"].as_str().unwrap().into(),
            }),
            _ => panic!("history kind"),
        })
        .collect();
    ContextSnapshot::new(
        ContextSnapshotId::new("tool-template").unwrap(),
        profile_id(),
        messages,
    )
    .unwrap()
    .with_tool_context(
        ToolContext::new(
            tools,
            history,
            ToolContextLimits {
                max_tools: 8,
                max_history_blocks: 16,
                max_json_bytes: 32768,
            },
        )
        .unwrap(),
    )
}

#[test]
fn gemma3_engine_tool_template_and_admitted_decoder() {
    use engine::parser_allocation::ParserAllocation;
    let fixture = Fixture::with_context(4096);
    let backend = Qwen3Backend::new_gemma3(
        vec![fixture.specification.clone()],
        limits(4096),
        Qwen3Execution::MlxCpu,
    )
    .unwrap()
    .with_parser_allocation_limits(parser_limits())
    .unwrap();
    let registry = backend.registry(vec![profile(4096)]).unwrap();
    let context = tool_fixture_context();
    let expected_hash = *context.tool_context().unwrap().schema_hash();
    let rendered = wait(backend.render_and_count(&registry, context)).unwrap();
    assert_eq!(
        rendered.text(),
        include_str!("fixtures/gemma3_tool_render/expected.txt")
    );
    assert_eq!(rendered.tool_schema_hash(), Some(&expected_hash));
    assert!(
        backend
            .decoder_with_parser_allocation(&registry, &profile_id(), None)
            .is_err()
    );
    let caps = parser_limits();
    let required = ParserAllocation::required_bytes(caps).unwrap();
    let admission = AdmissionController::new(required * 2).unwrap();
    let mut wrong = caps;
    wrong.calls += 1;
    let mut grant = admission
        .reserve_grant(
            "wrong Gemma3 slab",
            &[ParserAllocation::required_bytes(wrong).unwrap()],
        )
        .unwrap();
    let allocation = ParserAllocation::from_grant(&mut grant, wrong).unwrap();
    assert!(
        backend
            .decoder_with_parser_allocation(&registry, &profile_id(), Some(allocation))
            .is_err()
    );
    drop(grant);
    let tokenizer = tokenizers::Tokenizer::from_file(fixture.root.join("tokenizer.json")).unwrap();
    for raw in [
        r#"<tool_call>{"name":"fs.read","arguments":{"path":"README.md","nested":[1,true]}}</tool_call>"#,
        "```xml\n<tool_call>{\"name\":\"fs.read\",\"arguments\":{\"path\":\"README.md\",\"nested\":[1,true]}}</tool_call>\n```",
        r#"<tool_call>{"name":"fs.read","arguments":false}</tool_call>"#,
        r#"<tool_call>{"name":"fs.read","arguments":{}}</tool_call>suffix"#,
    ] {
        let mut grant = admission.reserve_grant("Gemma3 slab", &[required]).unwrap();
        let allocation = ParserAllocation::from_grant(&mut grant, caps).unwrap();
        let mut decoder = backend
            .decoder_with_parser_allocation(&registry, &profile_id(), Some(allocation))
            .unwrap();
        let mut text = String::new();
        assert!(!decoder.is_sequence_complete());
        let mut failed = false;
        let ids = tokenizer.encode(raw, false).unwrap();
        assert_eq!(tokenizer.decode(ids.get_ids(), false).unwrap(), raw);
        for &id in ids.get_ids() {
            match decoder.push_blocks(id) {
                Ok(blocks) => {
                    for block in blocks {
                        if let engine::types::GenerationOutputBlock::Text(t) = block {
                            text.push_str(&t)
                        } else {
                            panic!("proposal published before finish")
                        }
                    }
                }
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        if raw.contains("false") || raw.contains("suffix") {
            assert!(!decoder.is_sequence_complete());
            assert!(failed || decoder.finish_blocks().is_err());
            assert!(decoder.take_parser_proposals().is_none());
        } else {
            assert!(!failed);
            assert!(decoder.is_sequence_complete());
            for block in decoder.finish_blocks().unwrap() {
                if let engine::types::GenerationOutputBlock::Text(t) = block {
                    text.push_str(&t)
                }
            }
            assert_eq!(text, "");
            let batch = decoder.take_parser_proposals().unwrap();
            assert_eq!(batch.proposal_count(), 1);
            let call = batch.proposal(0).unwrap();
            assert_eq!(call.name, "fs.read");
            assert_eq!(
                serde_json::from_str::<Value>(call.arguments_json).unwrap(),
                json!({"path":"README.md","nested":[1,true]})
            );
        }
        drop(decoder);
        drop(grant);
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    }
}

fn tools_context(read: bool, result: Option<(&str, bool)>) -> ContextSnapshot {
    use engine::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    let tools = vec![
        RenderingTool::new("fs.read".into(), "1.0.0".into(), "Read a text file.".into(), json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false})).unwrap(),
        RenderingTool::new("fs.write".into(), "1.0.0".into(), "Write a text file.".into(), json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false})).unwrap(),
    ];
    let mut history = vec![];
    if let Some((content, is_error)) = result {
        history.push(ToolHistoryBlock::AssistantCall {
            call_id: "call-1".into(),
            tool_id: "fs.write".into(),
            arguments: json!({"path":"note.txt","content":"COBALT"})
                .as_object()
                .unwrap()
                .clone(),
        });
        history.push(ToolHistoryBlock::ToolResult {
            call_id: "call-1".into(),
            content: content.into(),
            is_error,
        });
    }
    context(vec![message(MessageRole::User, if read {"Read README.md using fs.read."} else {"Use fs.write to write exactly COBALT to note.txt. After the tool succeeds reply with exactly DONE."})]).with_tool_context(
        ToolContext::new(tools,history,ToolContextLimits {max_tools:8,max_history_blocks:16,max_json_bytes:32768}).unwrap())
}
#[test]
#[ignore = "requires pinned ALLOY_GEMMA3_SMALL_ARTIFACT tokenizer"]
fn gemma3_engine_tool_projection_bounds() {
    let backend = real_backend_with_context(true, 4096)
        .with_parser_allocation_limits(parser_limits())
        .unwrap();
    let registry = backend.registry(vec![profile(4096)]).unwrap();
    let empty =
        wait(backend.render_and_count(&registry, tools_context(false, Some(("", false))))).unwrap();
    for content in [
        "snow 雪",
        "\u{0344}\u{0344}",
        "\n\t\u{1}quoted \" text",
        "ASCII",
        "      ",
    ] {
        let actual =
            wait(backend.render_and_count(&registry, tools_context(false, Some((content, false)))))
                .unwrap();
        let bound = backend
            .projected_token_upper_bound(
                &registry,
                engine::api::ProjectedToolRender {
                    rendered: &empty,
                    result_count: 1,
                    input_bytes: actual.text().len(),
                    additional_rendered_bytes: actual.text().len() - empty.text().len(),
                    additional_ascii_escape_bytes: 0,
                },
            )
            .unwrap();
        assert!(
            actual.token_count() <= bound,
            "actual={}, bound={bound}",
            actual.token_count()
        );
        assert!(
            bound < 4096,
            "bounded small result must fit the selected profile"
        );
    }
    for count in [0, 2] {
        assert!(
            backend
                .projected_token_upper_bound(
                    &registry,
                    engine::api::ProjectedToolRender {
                        rendered: &empty,
                        result_count: count,
                        input_bytes: empty.text().len(),
                        additional_rendered_bytes: 0,
                        additional_ascii_escape_bytes: 0
                    }
                )
                .is_err()
        );
    }
}
#[test]
#[ignore = "requires frozen ALLOY_GEMMA3_SMALL_ARTIFACT and Metal"]
fn gemma3_real_4b_engine_tools() {
    real_tools(true);
}
#[test]
#[ignore = "requires frozen ALLOY_GEMMA3_ARTIFACT and Metal"]
fn gemma3_real_12b_engine_tools() {
    real_tools(false);
}
fn real_tools(small: bool) {
    let backend = real_backend_with_context(small, 4096)
        .with_parser_allocation_limits(parser_limits())
        .unwrap();
    let profile = ModelProfile::new(
        profile_id(),
        ArtifactId::new("gemma3").unwrap(),
        4096,
        192,
        8,
        GenerationOptions::greedy(192).unwrap(),
    )
    .unwrap();
    let admission = AdmissionController::new(36 * 1024 * 1024 * 1024).unwrap();
    let scheduler = Scheduler::new(8).unwrap();
    let engine = backend
        .into_engine(
            vec![profile],
            admission.clone(),
            scheduler.clone(),
            RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: true,
            },
        )
        .unwrap();
    let cases = [
        ("read", true, None),
        ("write", false, None),
        (
            "success",
            false,
            Some(("File written successfully.", false)),
        ),
        (
            "denied",
            false,
            Some((
                "Tool policy rejected fs.write. The requested path is outside the allowed workspace. No file was written.",
                true,
            )),
        ),
    ];
    for (name, read, result) in cases {
        let context = tools_context(read, result);
        let rendered = wait(engine.render_and_count(context.clone())).unwrap();
        eprintln!(
            "Gemma3 small={small} {name} prompt_sha256={:x} tokens={}",
            Sha256::digest(rendered.text().as_bytes()),
            rendered.token_count()
        );
        let request = GenerationRequest::new(
            RequestId::new(name).unwrap(),
            profile_id(),
            context,
            GenerationOptions::greedy(192).unwrap(),
        )
        .unwrap();
        let mut events = wait(engine.generate(request)).unwrap().into_events();
        let mut text = String::new();
        let mut calls = vec![];
        let mut completed = false;
        while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
            assert!(!completed, "event after terminal");
            match event {
                GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
                GenerationEvent::ToolCallBatchProposed { proposals } => {
                    for i in 0..proposals.proposal_count() {
                        let call = proposals.proposal(i).unwrap();
                        calls.push((
                            call.name.to_owned(),
                            serde_json::from_str::<Value>(call.arguments_json).unwrap(),
                        ));
                    }
                }
                GenerationEvent::GenerationCompleted {
                    finish_reason,
                    usage,
                } => {
                    assert_eq!(finish_reason, engine::types::FinishReason::EndOfSequence);
                    eprintln!("Gemma3 small={small} {name}: {usage:?}");
                    completed = true;
                }
                other => panic!("unexpected Gemma3 tool event {other:?}"),
            }
        }
        assert!(completed);
        eprintln!("Gemma3 small={small} {name}: text={text:?}, calls={calls:?}");
        if result.is_none() {
            assert!(text.trim().is_empty(), "premature answer: {text:?}");
            assert_eq!(
                calls,
                vec![(
                    if read { "fs.read" } else { "fs.write" }.to_owned(),
                    if read {
                        json!({"path":"README.md"})
                    } else {
                        json!({"path":"note.txt","content":"COBALT"})
                    }
                )]
            );
        } else {
            assert!(
                calls.is_empty(),
                "must not retry a completed or rejected call"
            );
            if name == "success" {
                assert_eq!(text.trim(), "DONE");
            } else {
                let lower = text.to_lowercase();
                assert!(
                    lower.contains("reject")
                        && lower.contains("outside")
                        && lower.contains("allowed workspace")
                        && lower.contains("no file was written"),
                    "incorrect error interpretation: {text:?}"
                );
                assert!(!text.contains("DONE"));
            }
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        while scheduler.is_busy().unwrap() {
            assert!(Instant::now() < deadline);
            thread::yield_now();
        }
    }
    drop(engine);
    let deadline = Instant::now() + Duration::from_secs(30);
    while admission.snapshot().unwrap().reserved_bytes() != 0 {
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    assert_eq!(admission.snapshot().unwrap().active_reservations(), 0);
}
