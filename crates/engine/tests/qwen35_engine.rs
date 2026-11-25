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
        Self::with_context(ornith, 128)
    }
    fn with_context(ornith: bool, context: usize) -> Self {
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
        config["text_config"]["max_position_embeddings"] = json!(context);
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
        self.backend_with_context(execution, ornith, 128)
    }
    fn backend_with_context(
        &self,
        execution: Qwen3Execution,
        ornith: bool,
        context: usize,
    ) -> Qwen3Backend {
        if ornith {
            Qwen3Backend::new_ornith(vec![self.specification.clone()], limits(context), execution)
                .unwrap()
        } else {
            Qwen3Backend::new_qwen35(vec![self.specification.clone()], limits(context), execution)
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
        serde_json::from_str(include_str!("fixtures/qwen35_tool_render/input.json")).unwrap();
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
fn qwen35_engine_tool_template_contract() {
    for ornith in [false, true] {
        let fixture = Fixture::with_context(ornith, 4096);
        let backend = fixture
            .backend_with_context(Qwen3Execution::MlxCpu, ornith, 4096)
            .with_parser_allocation_limits(parser_limits())
            .unwrap();
        let registry = backend.registry(vec![profile(4096)]).unwrap();
        let context = tool_fixture_context();
        let expected_hash = *context.tool_context().unwrap().schema_hash();
        let rendered = wait(backend.render_and_count(&registry, context)).unwrap();
        let expected = if ornith {
            include_str!("fixtures/qwen35_tool_render/ornith.txt")
        } else {
            include_str!("fixtures/qwen35_tool_render/upstream.txt")
        };
        assert_eq!(rendered.text(), expected);
        assert_eq!(rendered.tool_schema_hash(), Some(&expected_hash));
        assert!(
            backend
                .decoder_with_parser_allocation(&registry, &profile_id(), None)
                .is_err()
        );
    }
}
#[test]
fn qwen35_engine_xml_decoder_uses_advertised_schema() {
    use engine::parser_allocation::ParserAllocation;
    for ornith in [false, true] {
        let fixture = Fixture::new(ornith);
        let backend = fixture
            .backend(Qwen3Execution::MlxCpu, ornith)
            .with_parser_allocation_limits(parser_limits())
            .unwrap();
        let registry = backend.registry(vec![profile(128)]).unwrap();
        let tokenizer =
            tokenizers::Tokenizer::from_file(fixture.root.join("tokenizer.json")).unwrap();
        for (text, schema, valid) in [
            (
                "<think>inert <tool_call>not executable</tool_call></think>answer<tool_call><function=fs.test><parameter=s>0012</parameter><parameter=i>7</parameter></function></tool_call>",
                true,
                true,
            ),
            (
                "<tool_call><function=fs.test><parameter=unknown>x</parameter></function></tool_call>",
                true,
                false,
            ),
            (
                "<tool_call><function=fs.test></function></tool_call>",
                false,
                false,
            ),
            ("plain answer", false, true),
        ] {
            let bytes = ParserAllocation::required_bytes(parser_limits()).unwrap();
            let admission = AdmissionController::new(bytes).unwrap();
            let mut grant = admission.reserve_grant("XML decoder", &[bytes]).unwrap();
            let slab = ParserAllocation::from_grant(&mut grant, parser_limits()).unwrap();
            let tools =
                schema.then(|| Arc::new(tool_fixture_context().tool_context().unwrap().clone()));
            let mut decoder = backend
                .decoder_with_tool_context(&registry, &profile_id(), Some(slab), tools)
                .unwrap();
            let tokens = tokenizer.encode(text, false).unwrap();
            let mut visible = String::new();
            let result = (|| {
                for &token in tokens.get_ids() {
                    for block in decoder.push_blocks(token)? {
                        match block {
                            engine::types::GenerationOutputBlock::Text(text) => {
                                visible.push_str(&text)
                            }
                            _ => panic!("early proposal"),
                        }
                    }
                    assert!(decoder.take_parser_proposals().is_none());
                }
                decoder.finish_blocks()?;
                Ok::<_, engine::error::EngineError>(())
            })();
            assert_eq!(result.is_ok(), valid, "{text}: {result:?}");
            if valid {
                let owner = decoder.take_parser_proposals().unwrap();
                assert_eq!(visible, if schema { "answer" } else { "plain answer" });
                assert_eq!(owner.proposal_count(), usize::from(schema));
                if schema {
                    assert_eq!(
                        owner.proposal(0).unwrap().arguments_json,
                        "{\"s\":\"0012\",\"i\":7}"
                    );
                }
            }
            drop(decoder);
            assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
        }
    }
}

#[test]
fn qwen35_engine_retention_capabilities() {
    use engine::tiered_kv::{HotCacheLimit, TieredKvConfig};
    let mut fixture = Fixture::new(false);
    // This numerical fixture is not a trained UTF-8 language model. Stop after
    // actual prefill/selection, before decoding an arbitrary byte token.
    let path = fixture.root.join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["text_config"]["eos_token_id"] = json!((0..256).collect::<Vec<_>>());
    let bytes = serde_json::to_vec(&config).unwrap();
    std::fs::write(path, &bytes).unwrap();
    fixture.specification.files.insert(
        "config.json".into(),
        FilePin {
            bytes: bytes.len() as u64,
            sha256: hash(&bytes),
        },
    );
    let path = fixture.root.join("tokenizer.json");
    let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    // EOS validation correctly requires declared special tokens. This fixture
    // makes every output a declared stop; it exercises retention, not language.
    tokenizer["added_tokens"] = Value::Array(
        tokenizer["model"]["vocab"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(content, id)| {
                json!({"id":id,"content":content,"single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":true})
            })
            .collect(),
    );
    let bytes = serde_json::to_vec(&tokenizer).unwrap();
    std::fs::write(path, &bytes).unwrap();
    fixture.specification.files.insert(
        "tokenizer.json".into(),
        FilePin {
            bytes: bytes.len() as u64,
            sha256: hash(&bytes),
        },
    );
    for (disk, hot) in [(false, false), (true, false), (false, true), (true, true)] {
        let backend = fixture.backend(Qwen3Execution::MlxCpu, false);
        let engine = backend
            .into_engine(
                vec![profile(128)],
                AdmissionController::new(1 << 30).unwrap(),
                Scheduler::new(8).unwrap(),
                RetentionPolicy {
                    retain_model: true,
                    retain_cache_buffers: true,
                },
            )
            .unwrap();
        engine
            .runtime()
            .configure_tiered_kv(TieredKvConfig::new(
                disk.then(|| fixture.root.join(format!("cache-{hot}"))),
                HotCacheLimit::Bytes(if hot { 1 << 28 } else { 0 }),
            ))
            .unwrap();
        let context = ContextSnapshot::new(
            ContextSnapshotId::new("retention").unwrap(),
            profile_id(),
            vec![Message {
                role: MessageRole::User,
                content: "hi".into(),
            }],
        )
        .unwrap();
        let request = GenerationRequest::new(
            RequestId::new("retention").unwrap(),
            profile_id(),
            context,
            GenerationOptions::greedy(1).unwrap(),
        )
        .unwrap();
        let mut events = wait(engine.generate(request)).unwrap().into_events();
        let mut completed = false;
        while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
            match event {
                GenerationEvent::GenerationCompleted { .. } => completed = true,
                GenerationEvent::TextDelta { .. } | GenerationEvent::RuntimeWarning { .. } => {}
                other => panic!("unexpected retention event: {other:?}"),
            }
        }
        assert!(completed);
        let caps = wait(engine.capabilities(ArtifactId::new("hybrid").unwrap())).unwrap();
        assert_eq!(caps.supports_prefix_reuse, hot, "disk={disk} hot={hot}");
        let metrics = engine.runtime().tiered_kv_snapshot();
        assert_eq!(metrics.spill_count, 0);
        assert_eq!(metrics.restore_count, 0);
        if !hot {
            assert!(metrics.hot_evictions > 0, "{metrics:?}");
        }
    }
}

#[test]
#[ignore = "requires ALLOY_HYBRID_ARTIFACT and Metal; ALLOY_HYBRID_PROFILE=ornith selects the mandatory 9B artifact"]
fn qwen35_real_engine_streaming() {
    real_engine(false);
}
#[test]
#[ignore = "requires ALLOY_HYBRID_ARTIFACT and Metal; ALLOY_HYBRID_PROFILE=ornith selects Ornith"]
fn qwen35_real_engine_tool_proposals() {
    real_engine(true);
}
fn real_engine(tool_mode: bool) {
    let context_limit = if tool_mode { 4096 } else { 512 };
    let output_limit = if tool_mode { 192 } else { 32 };
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
        Qwen3Backend::new_ornith(
            vec![specification],
            limits(context_limit),
            Qwen3Execution::MlxMetal,
        )
    } else {
        Qwen3Backend::new_qwen35(
            vec![specification],
            limits(context_limit),
            Qwen3Execution::MlxMetal,
        )
    }
    .unwrap();
    let backend = if tool_mode {
        backend
            .with_parser_allocation_limits(parser_limits())
            .unwrap()
    } else {
        backend
    };
    let configured_profile = ModelProfile::new(
        profile_id(),
        ArtifactId::new("hybrid").unwrap(),
        context_limit,
        output_limit,
        8,
        GenerationOptions::greedy(output_limit).unwrap(),
    )
    .unwrap();
    let context = if tool_mode {
        real_tools_context(None)
    } else {
        ContextSnapshot::new(
            ContextSnapshotId::new("real-hybrid").unwrap(),
            profile_id(),
            vec![Message {
                role: MessageRole::User,
                content: "Calculate 2 + 2. Answer briefly.".into(),
            }],
        )
        .unwrap()
    };
    if tool_mode {
        let registry = backend.registry(vec![configured_profile.clone()]).unwrap();
        let empty =
            wait(backend.render_and_count(&registry, real_tools_context(Some("")))).unwrap();
        for content in [
            "snow 雪",
            "\u{0344}\u{0344}\u{0344}",
            "\n\t\u{1}quoted \" text",
            "ASCII",
        ] {
            let actual =
                wait(backend.render_and_count(&registry, real_tools_context(Some(content))))
                    .unwrap();
            let bound = backend
                .projected_token_upper_bound(
                    &registry,
                    engine::api::ProjectedToolRender {
                        rendered: &empty,
                        result_count: 2,
                        input_bytes: actual.text().len(),
                        additional_rendered_bytes: actual.text().len() - empty.text().len(),
                        additional_ascii_escape_bytes: 0,
                    },
                )
                .unwrap();
            assert!(actual.token_count() <= bound);
            assert!(
                backend
                    .projected_token_upper_bound(
                        &registry,
                        engine::api::ProjectedToolRender {
                            rendered: &actual,
                            result_count: 2,
                            input_bytes: actual.text().len(),
                            additional_rendered_bytes: 0,
                            additional_ascii_escape_bytes: 0,
                        }
                    )
                    .is_err()
            );
        }
        assert!(
            backend
                .projected_token_upper_bound(
                    &registry,
                    engine::api::ProjectedToolRender {
                        rendered: &empty,
                        result_count: 3,
                        input_bytes: empty.text().len(),
                        additional_rendered_bytes: 0,
                        additional_ascii_escape_bytes: 0,
                    }
                )
                .is_err()
        );
    }
    let scheduler = Scheduler::new(8).unwrap();
    let small_budget = 8 * 1024 * 1024 * 1024;
    let admission = AdmissionController::new(small_budget).unwrap();
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
    if ornith {
        let request = GenerationRequest::new(
            RequestId::new("hybrid-small-tier-refusal").unwrap(),
            profile_id(),
            context.clone(),
            GenerationOptions::greedy(output_limit).unwrap(),
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
            GenerationOptions::greedy(output_limit).unwrap(),
        )
        .unwrap();
        let generation = wait(engine.generate(request)).unwrap();
        let mut events = generation.into_events();
        let mut text = String::new();
        let mut completed = false;
        let mut proposals = 0;
        while let Some(event) = wait(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
            assert!(!completed, "event after terminal");
            match event {
                GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
                GenerationEvent::GenerationCompleted {
                    finish_reason,
                    usage,
                } => {
                    eprintln!(
                        "{} {name} completion: {finish_reason:?}, {usage:?}",
                        frozen["id"]
                    );
                    completed = true;
                }
                GenerationEvent::ToolCallProposed { proposal } if tool_mode => {
                    eprintln!("{} {name} proposal: {:?}", frozen["id"], proposal);
                    assert_eq!(proposal.name(), "fs.read");
                    assert_eq!(
                        proposal.arguments(),
                        json!({"path":"README.md"}).as_object().unwrap()
                    );
                    proposals += 1;
                }
                GenerationEvent::ToolCallBatchProposed { proposals: batch } if tool_mode => {
                    for index in 0..batch.proposal_count() {
                        let proposal = batch.proposal(index).unwrap();
                        eprintln!(
                            "{} {name} proposal: {} {}",
                            frozen["id"], proposal.name, proposal.arguments_json
                        );
                        assert_eq!(proposal.name, "fs.read");
                        assert_eq!(
                            serde_json::from_str::<Value>(proposal.arguments_json).unwrap(),
                            json!({"path":"README.md"})
                        );
                        proposals += 1;
                    }
                }
                other => panic!("unexpected real hybrid event {other:?}"),
            }
        }
        assert!(completed);
        if tool_mode {
            assert_eq!(
                proposals, 1,
                "{} {name} visible output: {text:?}",
                frozen["id"]
            );
        } else {
            assert!(!text.trim().is_empty());
        }
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

fn real_tools_context(result: Option<&str>) -> ContextSnapshot {
    use engine::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    let tool = RenderingTool::new("fs.read".into(), "1.0.0".into(), "Read a UTF-8 file by its relative path.".into(),
        json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false})).unwrap();
    let mut history = Vec::new();
    if let Some(content) = result {
        for id in ["c1", "c2"] {
            history.push(ToolHistoryBlock::AssistantCall {
                call_id: id.into(),
                tool_id: "fs.read".into(),
                arguments: json!({"path":"README.md"}).as_object().unwrap().clone(),
            });
        }
        for id in ["c1", "c2"] {
            history.push(ToolHistoryBlock::ToolResult {
                call_id: id.into(),
                content: content.into(),
                is_error: false,
            });
        }
    }
    ContextSnapshot::new(
        ContextSnapshotId::new("real-hybrid-tools").unwrap(),
        profile_id(),
        vec![Message {
            role: MessageRole::User,
            content: "Call fs.read with path README.md. Do not answer from memory.".into(),
        }],
    )
    .unwrap()
    .with_tool_context(
        ToolContext::new(
            vec![tool],
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
fn qwen35_qualification_identity_tracks_native_settings() {
    let revision = "a".repeat(40);
    let dependencies = "b".repeat(64);
    let mut identities = Vec::new();
    for ornith in [false, true] {
        let fixture = Fixture::new(ornith);
        let backend = fixture.backend(Qwen3Execution::MlxCpu, ornith);
        let inspect = |backend: &Qwen3Backend, context| {
            backend
                .qualification_identity(&profile(context), &revision, &dependencies)
                .unwrap()
        };
        let selected = inspect(&backend, 128);
        assert_eq!(selected.family, "qwen35");
        assert_eq!(selected.profile_sha256, profile(128).qualification_sha256());
        assert_ne!(
            selected.profile_sha256,
            inspect(&backend, 96).profile_sha256
        );
        assert!(
            backend
                .qualification_identity(&profile(128), "unreviewed", &dependencies)
                .is_err()
        );
        let expected = fixture.specification.files["tokenizer.json"].sha256;
        assert_eq!(
            selected.tokenizer_sha256,
            format!("{:x}", sha2::digest::Output::<Sha256>::from(expected))
        );
        let enabled = backend
            .with_parser_allocation_limits(parser_limits())
            .unwrap();
        let tools = inspect(&enabled, 128);
        assert_ne!(selected.tool_protocol_sha256, tools.tool_protocol_sha256);
        assert_eq!(
            selected.artifact_manifest_sha256,
            tools.artifact_manifest_sha256
        );
        let mut limits = parser_limits();
        limits.call_bytes += 1;
        let modified = enabled.with_parser_allocation_limits(limits).unwrap();
        assert_ne!(
            tools.tool_protocol_sha256,
            inspect(&modified, 128).tool_protocol_sha256
        );
        identities.push(tools);
    }
    assert_ne!(identities[0].template_sha256, identities[1].template_sha256);
    assert_ne!(
        identities[0].tool_protocol_sha256,
        identities[1].tool_protocol_sha256
    );
}
