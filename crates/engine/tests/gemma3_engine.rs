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
        limits(512),
        Qwen3Execution::MlxMetal,
    )
    .unwrap()
}
