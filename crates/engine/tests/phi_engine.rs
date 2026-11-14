//! Actual Phi family operations through engine contracts; no model mocks.
use engine::{
    admission::AdmissionController,
    api::Engine,
    backends::{
        FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution,
        qwen::{PHI_CACHE_REPRESENTATION, PHI_TEXT_TEMPLATE},
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
        assert!(Instant::now() < deadline, "phi test deadline");
        thread::park_timeout(Duration::from_millis(10));
    }
}
fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn control() -> (CancellationSource, ExecutionControl) {
    let source = CancellationSource::new();
    let control = ExecutionControl::new(RequestId::new("phi-test").unwrap(), source.token());
    (source, control)
}
fn profile_id() -> ModelProfileId {
    ModelProfileId::new("phi-test").unwrap()
}
fn profile(context: usize) -> ModelProfile {
    ModelProfile::new(
        profile_id(),
        ArtifactId::new("phi").unwrap(),
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
        max_config_bytes: 1 << 20,
        max_tokenizer_bytes: 32 << 20,
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
                "alloy-phi-engine-{}-{}",
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
            "../../inference-infra/tests/fixtures/phi_model/config.json"
        ))
        .unwrap();
        config["eos_token_id"] = json!(126);
        config["max_position_embeddings"] = json!(context);
        config["sliding_window"] = json!(context);
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("model.safetensors"),
            include_bytes!("../../inference-infra/tests/fixtures/phi_model/model.safetensors"),
        )
        .unwrap();
        let mut vocab: BTreeMap<String, u32> = (33u8..=126)
            .enumerate()
            .map(|(i, c)| ((c as char).to_string(), i as u32))
            .collect();
        vocab.insert("Ġ".into(), 94);
        vocab.insert("Ċ".into(), 95);
        let specials = [
            (123, "<|system|>"),
            (124, "<|user|>"),
            (125, "<|assistant|>"),
            (126, "<|end|>"),
            (127, "<|endoftext|>"),
        ];
        for (id, content) in specials {
            vocab.insert(content.into(), id);
        }
        let added: Vec<_> = specials.into_iter().map(|(id, content)|
            json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect();
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,
            "normalizer":null,"post_processor":null,
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "model":{"type":"BPE","vocab":vocab,"merges":[],"dropout":null,"unk_token":null,
                     "continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false}});
        std::fs::write(
            root.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            br#"{"eos_token":"<|endoftext|>"}"#,
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
            id: ArtifactId::new("phi").unwrap(),
            directory: root.clone(),
            files,
            template_sha256: hash(PHI_TEXT_TEMPLATE.as_bytes()),
        };
        Self {
            root,
            specification,
        }
    }
    fn backend(&self, execution: Qwen3Execution) -> Qwen3Backend {
        Qwen3Backend::new_phi(vec![self.specification.clone()], limits(128), execution).unwrap()
    }
}

fn load(
    backend: &Qwen3Backend,
    registry: &Registry,
    control: &ExecutionControl,
) -> Box<dyn LoadedModel> {
    let plan = wait(backend.plan(registry, &profile_id(), 16)).unwrap();
    let artifact = registry.artifact(&ArtifactId::new("phi").unwrap()).unwrap();
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
#[test]
fn phi_engine_native_cache_lifecycle() {
    let fixture = Fixture::new();
    for execution in executions() {
        let backend = fixture.backend(execution);
        let registry = backend.registry(vec![profile(128)]).unwrap();
        let (_, control) = control();
        let plan = wait(backend.plan(&registry, &profile_id(), 16)).unwrap();
        let mut loaded = load(&backend, &registry, &control);
        assert_eq!(loaded.registration().family(), "phi");
        let model = loaded.autoregressive().unwrap();
        let mut cache = model.allocate_cache(&plan.cache, &control).unwrap();
        assert!(cache.supports_truncation());
        assert!(cache.kv_block_spec().is_some());
        let prompt = [3, 41, 7, 120];
        let first = model.prefill(&prompt, cache.as_mut(), &control).unwrap();
        assert_eq!(first.logits().values().len(), 128);
        let decoded = model
            .decode(19, cache.as_mut(), &control)
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(cache.len(), 5);
        assert!(cache.memory_usage().unwrap().used_bytes() > 0);
        cache.truncate(4).unwrap();
        assert_eq!(cache.len(), 4);
        assert_eq!(
            decoded,
            model.decode(19, cache.as_mut(), &control).unwrap().values()
        );
        let block = cache.export_kv_block(0, cache.len()).unwrap();
        let continued = model
            .decode(13, cache.as_mut(), &control)
            .unwrap()
            .values()
            .to_vec();
        let mut second_load = load(&backend, &registry, &control);
        let second = second_load.autoregressive().unwrap();
        let mut restored = second.allocate_cache(&plan.cache, &control).unwrap();
        assert_eq!(restored.restore_kv_block(&block).unwrap(), 5);
        assert_eq!(
            continued,
            second
                .decode(13, restored.as_mut(), &control)
                .unwrap()
                .values()
        );
        cache.reset().unwrap();
        assert_eq!(cache.len(), 0);
        let rebuilt = model.prefill(&prompt, cache.as_mut(), &control).unwrap();
        assert_eq!(first.logits().values(), rebuilt.logits().values());
        assert_eq!(
            decoded,
            model.decode(19, cache.as_mut(), &control).unwrap().values()
        );
        cache.reset().unwrap();
        let edited = [7, 41, 7];
        let changed = model.prefill(&edited, cache.as_mut(), &control).unwrap();
        let mut cold = model.allocate_cache(&plan.cache, &control).unwrap();
        assert_eq!(
            changed.logits().values(),
            model
                .prefill(&edited, cold.as_mut(), &control)
                .unwrap()
                .logits()
                .values()
        );
    }
}
#[test]
fn phi_engine_rejects_foreign_cancelled_and_unimplemented_tools() {
    let fixture = Fixture::new();
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend.registry(vec![profile(128)]).unwrap();
    let (_, control) = control();
    let plan = wait(backend.plan(&registry, &profile_id(), 16)).unwrap();
    let mut first = load(&backend, &registry, &control);
    let mut second = load(&backend, &registry, &control);
    let mut cache = first
        .autoregressive()
        .unwrap()
        .allocate_cache(&plan.cache, &control)
        .unwrap();
    assert!(
        second
            .autoregressive()
            .unwrap()
            .prefill(&[3], cache.as_mut(), &control)
            .is_err()
    );
    assert_eq!(cache.len(), 0);
    let source = CancellationSource::new();
    source.request_cancel();
    let cancelled = ExecutionControl::new(RequestId::new("cancel-hybrid").unwrap(), source.token());
    assert!(
        first
            .autoregressive()
            .unwrap()
            .prefill(&[3], cache.as_mut(), &cancelled)
            .is_err()
    );
    assert_eq!(cache.len(), 0);
    first
        .autoregressive()
        .unwrap()
        .prefill(&[3], cache.as_mut(), &control)
        .unwrap();
    assert_eq!(cache.len(), 1);
    let tools = fixture
        .backend(Qwen3Execution::MlxCpu)
        .with_tool_call_parsing(true);
    assert!(
        tools
            .parser_allocation_limits(&registry, &profile_id())
            .is_err()
    );
}
#[test]
fn phi_engine_text_template_and_position_identity() {
    let fixture = Fixture::new();
    let backend = fixture.backend(Qwen3Execution::MlxCpu);
    let registry = backend.registry(vec![profile(128)]).unwrap();
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("history").unwrap(),
        profile_id(),
        vec![
            Message {
                role: MessageRole::System,
                content: " Be concise. ".into(),
            },
            Message {
                role: MessageRole::User,
                content: " Hi ".into(),
            },
            Message {
                role: MessageRole::Assistant,
                content: " Hello ".into(),
            },
            Message {
                role: MessageRole::User,
                content: " Next ".into(),
            },
        ],
    )
    .unwrap();
    let rendered = wait(backend.render_and_count(&registry, context)).unwrap();
    assert_eq!(
        rendered.text(),
        "<|system|> Be concise. <|end|><|user|> Hi <|end|><|assistant|> Hello <|end|><|user|> Next <|end|><|assistant|>"
    );
    assert_eq!(
        backend.eos_tokens(&registry, &profile_id()).unwrap(),
        [126, 127]
    );
    assert_eq!(
        backend.registration().capabilities().cache_modes[0]
            .representation
            .as_str(),
        PHI_CACHE_REPRESENTATION
    );
    let mut tools = fixture
        .backend(Qwen3Execution::MlxCpu)
        .with_tool_call_parsing(true);
    assert!(
        tools
            .parser_allocation_limits(&registry, &profile_id())
            .is_err()
    );
    tools = tools.with_tool_call_parsing(false);
    assert!(
        tools
            .parser_allocation_limits(&registry, &profile_id())
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "requires frozen ALLOY_PHI_ARTIFACT and Metal"]
fn phi_real_engine_streaming() {
    let frozen: Value = serde_json::from_str(include_str!(
        "../../inference-infra/tests/fixtures/phi4_mini_profile.json"
    ))
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
    let backend = Qwen3Backend::new_phi(
        vec![Qwen3Artifact {
            id: ArtifactId::new("phi").unwrap(),
            directory: PathBuf::from(std::env::var_os("ALLOY_PHI_ARTIFACT").expect("artifact")),
            files,
            template_sha256: hash(PHI_TEXT_TEMPLATE.as_bytes()),
        }],
        limits(512),
        Qwen3Execution::MlxMetal,
    )
    .unwrap();
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("real-phi").unwrap(),
        profile_id(),
        vec![Message {
            role: MessageRole::User,
            content: "Calculate 2 + 2. Reply with only the result.".into(),
        }],
    )
    .unwrap();
    let registry = backend.registry(vec![profile(512)]).unwrap();
    let counted = wait(backend.render_and_count(&registry, context.clone())).unwrap();
    let expected_input = counted.token_count();
    assert_eq!(
        counted.text(),
        "<|user|>Calculate 2 + 2. Reply with only the result.<|end|><|assistant|>"
    );
    let admission = AdmissionController::new(8 * 1024 * 1024 * 1024).unwrap();
    let scheduler = Scheduler::new(8).unwrap();
    let engine = backend
        .into_engine(
            vec![profile(512)],
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
    for name in ["phi-cold", "phi-repeat"] {
        let request = GenerationRequest::new(
            RequestId::new(name).unwrap(),
            profile_id(),
            context.clone(),
            GenerationOptions::greedy(32).unwrap(),
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
                    eprintln!("Phi {name}: {finish_reason:?} {usage:?}");
                    completed = true;
                }
                other => panic!("unexpected Phi event {other:?}"),
            }
        }
        assert!(completed);
        assert_eq!(text.trim(), "4", "Phi streamed output {text:?}");
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
