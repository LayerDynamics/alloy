//! Actual hybrid family operations through engine contracts; no model mocks.
use engine::{
    admission::AdmissionController,
    api::Engine,
    backends::{
        FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution,
        qwen::{ORNITH_TEXT_TEMPLATE, QWEN35_TEXT_TEMPLATE},
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
        assert!(Instant::now() < deadline, "hybrid test deadline");
        thread::park_timeout(Duration::from_millis(10));
    }
}
fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn control() -> (CancellationSource, ExecutionControl) {
    let source = CancellationSource::new();
    let control = ExecutionControl::new(RequestId::new("hybrid-test").unwrap(), source.token());
    (source, control)
}
fn profile_id() -> ModelProfileId {
    ModelProfileId::new("hybrid-test").unwrap()
}
fn profile(context: usize) -> ModelProfile {
    ModelProfile::new(
        profile_id(),
        ArtifactId::new("hybrid").unwrap(),
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
    fn new(ornith: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = loop {
            let p = std::env::temp_dir().join(format!(
                "alloy-hybrid-engine-{}-{}",
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
            "../../inference-infra/tests/fixtures/qwen35_model/config.json"
        ))
        .unwrap();
        config["text_config"]["eos_token_id"] = json!(255);
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.join("model.safetensors"),
            include_bytes!("../../inference-infra/tests/fixtures/qwen35_model/model.safetensors"),
        )
        .unwrap();
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .collect();
        alphabet.sort_unstable();
        let mut vocab: BTreeMap<String, u32> = alphabet
            .into_iter()
            .take(254)
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        vocab.insert("<|im_start|>".into(), 254);
        vocab.insert("<|im_end|>".into(), 255);
        let added: Vec<_> = [(254, "<|im_start|>"), (255, "<|im_end|>")].into_iter().map(|(id, content)|
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
            br#"{"eos_token":"<|im_end|>"}"#,
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
            id: ArtifactId::new("hybrid").unwrap(),
            directory: root.clone(),
            files,
            template_sha256: hash(
                if ornith {
                    ORNITH_TEXT_TEMPLATE
                } else {
                    QWEN35_TEXT_TEMPLATE
                }
                .as_bytes(),
            ),
        };
        Self {
            root,
            specification,
        }
    }
    fn backend(&self, execution: Qwen3Execution, ornith: bool) -> Qwen3Backend {
        if ornith {
            Qwen3Backend::new_ornith(vec![self.specification.clone()], limits(128), execution)
                .unwrap()
        } else {
            Qwen3Backend::new_qwen35(vec![self.specification.clone()], limits(128), execution)
                .unwrap()
        }
    }
}
fn load(
    backend: &Qwen3Backend,
    registry: &Registry,
    control: &ExecutionControl,
) -> Box<dyn LoadedModel> {
    let plan = wait(backend.plan(registry, &profile_id(), 16)).unwrap();
    let artifact = registry
        .artifact(&ArtifactId::new("hybrid").unwrap())
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
#[test]
fn qwen35_engine_native_cache_lifecycle() {
    let fixture = Fixture::new(false);
    for execution in executions() {
        let backend = fixture.backend(execution, false);
        let registry = backend.registry(vec![profile(128)]).unwrap();
        let (_, control) = control();
        let plan = wait(backend.plan(&registry, &profile_id(), 16)).unwrap();
        let mut loaded = load(&backend, &registry, &control);
        assert_eq!(loaded.registration().family(), "qwen35");
        let model = loaded.autoregressive().unwrap();
        let mut cache = model.allocate_cache(&plan.cache, &control).unwrap();
        assert!(!cache.supports_truncation());
        assert!(cache.kv_block_spec().is_none());
        let prompt = [3, 41, 7, 200];
        let first = model.prefill(&prompt, cache.as_mut(), &control).unwrap();
        assert_eq!(first.logits().values().len(), 256);
        let decoded = model
            .decode(19, cache.as_mut(), &control)
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(cache.len(), 5);
        assert!(cache.memory_usage().unwrap().used_bytes() > 0);
        assert!(cache.truncate(2).is_err());
        assert_eq!(cache.len(), 5);
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
fn qwen35_engine_rejects_foreign_and_cancelled_state() {
    let fixture = Fixture::new(false);
    let backend = fixture.backend(Qwen3Execution::MlxCpu, false);
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
        .backend(Qwen3Execution::MlxCpu, false)
        .with_tool_call_parsing(true);
    assert!(
        tools
            .parser_allocation_limits(&registry, &profile_id())
            .is_err()
    );
}
#[test]
fn qwen35_engine_text_template_contract() {
    for ornith in [false, true] {
        let fixture = Fixture::new(ornith);
        let backend = fixture.backend(Qwen3Execution::MlxCpu, ornith);
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
        let history = if ornith {
            "<think>\n\n</think>\n\n"
        } else {
            ""
        };
        assert_eq!(
            rendered.text(),
            format!(
                "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n{history}Hello<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        );
        assert_eq!(backend.eos_tokens(&registry, &profile_id()).unwrap(), [255]);
    }
}

#[test]
#[ignore = "requires ALLOY_HYBRID_ARTIFACT and Metal; ALLOY_HYBRID_PROFILE=ornith selects the mandatory 9B artifact"]
fn qwen35_real_engine_streaming() {
    let ornith = std::env::var("ALLOY_HYBRID_PROFILE").is_ok_and(|s| s == "ornith");
    let frozen: Value = serde_json::from_str(if ornith {
        include_str!("../../inference-infra/tests/fixtures/ornith_1_5_9b_profile.json")
    } else {
        include_str!("../../inference-infra/tests/fixtures/qwen35_0_8b_profile.json")
    })
    .unwrap();
    let root = PathBuf::from(std::env::var_os("ALLOY_HYBRID_ARTIFACT").expect("artifact"));
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
    let specification = Qwen3Artifact {
        id: ArtifactId::new("hybrid").unwrap(),
        directory: root,
        files,
        template_sha256: hash(
            if ornith {
                ORNITH_TEXT_TEMPLATE
            } else {
                QWEN35_TEXT_TEMPLATE
            }
            .as_bytes(),
        ),
    };
    let backend = if ornith {
        Qwen3Backend::new_ornith(vec![specification], limits(512), Qwen3Execution::MlxMetal)
    } else {
        Qwen3Backend::new_qwen35(vec![specification], limits(512), Qwen3Execution::MlxMetal)
    }
    .unwrap();
    let scheduler = Scheduler::new(8).unwrap();
    let small_budget = 8 * 1024 * 1024 * 1024;
    let admission = AdmissionController::new(small_budget).unwrap();
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
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("real-hybrid").unwrap(),
        profile_id(),
        vec![Message {
            role: MessageRole::User,
            content: "Calculate 2 + 2. Answer briefly.".into(),
        }],
    )
    .unwrap();
    if ornith {
        let request = GenerationRequest::new(
            RequestId::new("hybrid-small-tier-refusal").unwrap(),
            profile_id(),
            context.clone(),
            GenerationOptions::greedy(32).unwrap(),
        )
        .unwrap();
        let error = match wait(engine.generate(request)) {
            Err(error) => error,
            Ok(_) => panic!("Ornith must not fit the frozen 8 GiB startup plan"),
        };
        assert!(matches!(
            error,
            engine::error::EngineError::MemoryLimitExceeded {
                reason: engine::error::MemoryLimitReason::ReservationBudget { budget_bytes },
                required_bytes,
                available_bytes,
                ..
            } if budget_bytes == small_budget && required_bytes > small_budget
                && available_bytes == small_budget
        ));
        assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        assert!(!scheduler.is_busy().unwrap());
        eprintln!("{} 8 GiB tier: {error}", frozen["id"]);
        admission.resize_budget(36 * 1024 * 1024 * 1024).unwrap();
    }
    let mut outputs = Vec::new();
    for name in ["hybrid-cold", "hybrid-repeat"] {
        let request = GenerationRequest::new(
            RequestId::new(name).unwrap(),
            profile_id(),
            context.clone(),
            GenerationOptions::greedy(32).unwrap(),
        )
        .unwrap();
        let generation = wait(engine.generate(request)).unwrap();
        let mut events = generation.into_events();
        let mut text = String::new();
        let mut completed = false;
        while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
            assert!(!completed, "event after terminal");
            match event {
                GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
                GenerationEvent::GenerationCompleted { .. } => completed = true,
                other => panic!("unexpected real hybrid event {other:?}"),
            }
        }
        assert!(completed && !text.trim().is_empty());
        eprintln!("{} {name}: {text:?}", frozen["id"]);
        outputs.push(text);
        let deadline = Instant::now() + Duration::from_secs(30);
        while scheduler.is_busy().unwrap() {
            assert!(Instant::now() < deadline);
            thread::yield_now();
        }
    }
    assert_eq!(outputs[0], outputs[1]);
}
