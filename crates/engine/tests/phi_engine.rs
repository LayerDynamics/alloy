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
fn phi_engine_rejects_foreign_cancelled_and_unbounded_tools() {
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
    real_engine(false);
}
#[test]
#[ignore = "requires frozen ALLOY_PHI_ARTIFACT and Metal"]
fn phi_real_engine_tool_proposals() {
    real_engine(true);
}
fn real_engine(tool_mode: bool) {
    let context_limit = if tool_mode { 4096 } else { 512 };
    let output_limit = if tool_mode { 192 } else { 32 };
    let backend = real_backend(context_limit);
    let backend = if tool_mode {
        backend
            .with_parser_allocation_limits(parser_limits())
            .unwrap()
    } else {
        backend
    };
    let configured_profile = ModelProfile::new(
        profile_id(),
        ArtifactId::new("phi").unwrap(),
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
            ContextSnapshotId::new("real-phi").unwrap(),
            profile_id(),
            vec![Message {
                role: MessageRole::User,
                content: "Calculate 2 + 2. Reply with only the result.".into(),
            }],
        )
        .unwrap()
    };
    let registry = backend.registry(vec![configured_profile.clone()]).unwrap();
    let counted = wait(backend.render_and_count(&registry, context.clone())).unwrap();
    let expected_input = counted.token_count();
    if std::env::var_os("ALLOY_TEST_TRACE_REQUESTS").is_some() {
        eprintln!(
            "Phi rendered={:?} tokens={:?}",
            counted.text(),
            counted.token_ids()
        );
    }
    if !tool_mode {
        assert_eq!(
            counted.text(),
            "<|user|>Calculate 2 + 2. Reply with only the result.<|end|><|assistant|>"
        );
    }
    let admission = AdmissionController::new(8 * 1024 * 1024 * 1024).unwrap();
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
    for name in ["phi-cold", "phi-repeat"] {
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
        let mut proposals = 0;
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
                    if tool_mode {
                        // Frozen artifact selects 22 actual tokens through the
                        // closing list; its otherwise-next token is EOS. The
                        // structural boundary must not execute that extra token.
                        assert_eq!(usage.output_tokens, 22);
                    }
                    eprintln!("Phi {name}: {finish_reason:?} {usage:?}");
                    completed = true;
                }
                GenerationEvent::ToolCallBatchProposed { proposals: batch } if tool_mode => {
                    for i in 0..batch.proposal_count() {
                        let call = batch.proposal(i).unwrap();
                        eprintln!("Phi {name} proposal: {} {}", call.name, call.arguments_json);
                        assert_eq!(call.name, "fs.read");
                        assert_eq!(
                            serde_json::from_str::<Value>(call.arguments_json).unwrap(),
                            json!({"path":"README.md"})
                        );
                        proposals += 1;
                    }
                }
                other => panic!("unexpected Phi event {other:?}"),
            }
        }
        assert!(completed);
        if tool_mode {
            assert_eq!(proposals, 1, "Phi tool output {text:?}");
            assert!(text.is_empty(), "tool turn included an answer: {text:?}");
        } else {
            assert_eq!(text.trim(), "4", "Phi streamed output {text:?}");
        }
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
        serde_json::from_str(include_str!("fixtures/phi_tool_render/input.json")).unwrap();
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
fn phi_engine_trained_tool_template_and_admitted_decoder() {
    use engine::parser_allocation::ParserAllocation;
    let fixture = Fixture::with_context(4096);
    let backend = Qwen3Backend::new_phi(
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
        include_str!("fixtures/phi_tool_render/expected.txt")
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
            "wrong Phi slab",
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
        r#"functools[{"name":"fs.read","arguments":{"path":"README.md","nested":[1,true]}}]"#,
        r#"functools[{"name":"fs.read","arguments":{}},{"name":"bad","arguments":false}]"#,
    ] {
        let mut grant = admission.reserve_grant("Phi slab", &[required]).unwrap();
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
        if raw.contains("false") {
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
        ContextSnapshotId::new("real-phi-tools").unwrap(),
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

fn real_backend(context_limit: usize) -> Qwen3Backend {
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
    Qwen3Backend::new_phi(
        vec![Qwen3Artifact {
            id: ArtifactId::new("phi").unwrap(),
            directory: PathBuf::from(std::env::var_os("ALLOY_PHI_ARTIFACT").expect("artifact")),
            files,
            template_sha256: hash(PHI_TEXT_TEMPLATE.as_bytes()),
        }],
        limits(context_limit),
        Qwen3Execution::MlxMetal,
    )
    .unwrap()
}

#[test]
#[ignore = "requires frozen ALLOY_PHI_ARTIFACT tokenizer metadata"]
fn phi_engine_tool_projection_bounds() {
    let backend = real_backend(4096)
        .with_parser_allocation_limits(parser_limits())
        .unwrap();
    let registry = backend.registry(vec![profile(4096)]).unwrap();
    let empty = wait(backend.render_and_count(&registry, real_tools_context(Some("")))).unwrap();
    for content in [
        "snow 雪",
        "\u{0344}\u{0344}",
        "\n\t\u{1}quoted \" text",
        "ASCII",
    ] {
        let actual =
            wait(backend.render_and_count(&registry, real_tools_context(Some(content)))).unwrap();
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
    }
    assert!(
        backend
            .projected_token_upper_bound(
                &registry,
                engine::api::ProjectedToolRender {
                    rendered: &empty,
                    result_count: 0,
                    input_bytes: empty.text().len(),
                    additional_rendered_bytes: 0,
                    additional_ascii_escape_bytes: 0,
                }
            )
            .is_err()
    );
    assert!(
        backend
            .projected_token_upper_bound(
                &registry,
                engine::api::ProjectedToolRender {
                    rendered: &empty,
                    result_count: 2,
                    input_bytes: empty.text().len(),
                    additional_rendered_bytes: usize::MAX,
                    additional_ascii_escape_bytes: 0,
                }
            )
            .is_err()
    );
}
