use super::*;
use crate::{
    api::Engine,
    types::{ContextSnapshotId, GenerationOptions, Message, MessageRole},
};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "alloy-qwen-shared-renderer-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path.canonicalize().expect("canonical directory")),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create fixture directory: {error}"),
            }
        }
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                let now = Instant::now();
                assert!(now < deadline, "renderer wiring test timed out");
                thread::park_timeout(deadline.saturating_duration_since(now));
            }
        }
    }
}

struct TensorRecord {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn add_f32_tensor(
    records: &mut Vec<TensorRecord>,
    name: String,
    shape: Vec<usize>,
    values: &[f32],
) {
    let count = shape
        .iter()
        .try_fold(1_usize, |n, &dimension| n.checked_mul(dimension))
        .expect("tensor element count");
    assert_eq!(count, values.len(), "fixture tensor {name}");
    records.push(TensorRecord {
        name,
        dtype: Dtype::F32,
        shape,
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    });
}

/// Group size for tiny wiring-test Affine4 checkpoints. Must divide every
/// matrix width in `Qwen3Config::tiny_fixture` (32 and 64). Production
/// planning still estimates with [`AFFINE4_GROUP_SIZE`] (64).
const WIRING_AFFINE4_GROUP: usize = 32;

fn quantize_affine4(
    values: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert_eq!(values.len(), rows.saturating_mul(cols));
    assert!(
        cols > 0 && cols.is_multiple_of(8) && group_size > 0 && cols.is_multiple_of(group_size)
    );
    let groups = cols / group_size;
    let mut packed = vec![0u32; rows * (cols / 8)];
    let mut scales = vec![0f32; rows * groups];
    let mut biases = vec![0f32; rows * groups];
    for row in 0..rows {
        for group in 0..groups {
            let start = row * cols + group * group_size;
            let slice = &values[start..start + group_size];
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
            let bias = min;
            scales[row * groups + group] = scale;
            biases[row * groups + group] = bias;
            for (offset, &value) in slice.iter().enumerate() {
                let code = if scale > 0.0 {
                    ((value - bias) / scale).round().clamp(0.0, 15.0) as u32
                } else {
                    0
                };
                let element = group * group_size + offset;
                let word = row * (cols / 8) + element / 8;
                let nibble = element % 8;
                packed[word] |= code << (4 * nibble);
            }
        }
    }
    (packed, scales, biases)
}

fn add_affine4_matrix(
    records: &mut Vec<TensorRecord>,
    weight_name: &str,
    rows: usize,
    cols: usize,
    values: &[f32],
) {
    let (packed, scales, biases) = quantize_affine4(values, rows, cols, WIRING_AFFINE4_GROUP);
    let groups = cols / WIRING_AFFINE4_GROUP;
    let base = weight_name
        .strip_suffix(".weight")
        .expect("matrix weight name");
    records.push(TensorRecord {
        name: weight_name.to_owned(),
        dtype: Dtype::U32,
        shape: vec![rows, cols / 8],
        bytes: packed.iter().flat_map(|word| word.to_le_bytes()).collect(),
    });
    records.push(TensorRecord {
        name: format!("{base}.scales"),
        dtype: Dtype::F32,
        shape: vec![rows, groups],
        bytes: scales
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    });
    records.push(TensorRecord {
        name: format!("{base}.biases"),
        dtype: Dtype::F32,
        shape: vec![rows, groups],
        bytes: biases
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    });
}

// Serialize Affine4-resident checkpoints for wiring/execution tests.
// Norms stay float32; every projection/embed/lm_head is packed U32.
fn checkpoint_bytes(config: &Qwen3Config) -> Vec<u8> {
    checkpoint_bytes_selected(config, false)
}

fn checkpoint_bytes_selected(config: &Qwen3Config, zero_head: bool) -> Vec<u8> {
    let mut weights = qwen3::fixture::generate_fixture(config);
    if zero_head {
        weights
            .lm_head
            .as_mut()
            .expect("untied test output head")
            .as_f32_mut()
            .expect("fixture f32")
            .fill(0.0);
    }
    let mut records = Vec::new();
    add_affine4_matrix(
        &mut records,
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
        weights.embed_tokens.as_f32().expect("fixture f32"),
    );
    for (index, layer) in weights.layers.iter().enumerate() {
        let prefix = format!("model.layers.{index}");
        add_f32_tensor(
            &mut records,
            format!("{prefix}.input_layernorm.weight"),
            vec![config.hidden_size],
            &layer.input_layernorm,
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.self_attn.q_proj.weight"),
            config.q_dim(),
            config.hidden_size,
            layer.q_proj.as_f32().expect("fixture f32"),
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.self_attn.k_proj.weight"),
            config.kv_dim(),
            config.hidden_size,
            layer.k_proj.as_f32().expect("fixture f32"),
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.self_attn.v_proj.weight"),
            config.kv_dim(),
            config.hidden_size,
            layer.v_proj.as_f32().expect("fixture f32"),
        );
        add_f32_tensor(
            &mut records,
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![config.head_dim],
            &layer.q_norm,
        );
        add_f32_tensor(
            &mut records,
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![config.head_dim],
            &layer.k_norm,
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.self_attn.o_proj.weight"),
            config.hidden_size,
            config.q_dim(),
            layer.o_proj.as_f32().expect("fixture f32"),
        );
        add_f32_tensor(
            &mut records,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![config.hidden_size],
            &layer.post_attention_layernorm,
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.mlp.gate_proj.weight"),
            config.intermediate_size,
            config.hidden_size,
            layer.gate_proj.as_f32().expect("fixture f32"),
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.mlp.up_proj.weight"),
            config.intermediate_size,
            config.hidden_size,
            layer.up_proj.as_f32().expect("fixture f32"),
        );
        add_affine4_matrix(
            &mut records,
            &format!("{prefix}.mlp.down_proj.weight"),
            config.hidden_size,
            config.intermediate_size,
            layer.down_proj.as_f32().expect("fixture f32"),
        );
    }
    add_f32_tensor(
        &mut records,
        "model.norm.weight".to_owned(),
        vec![config.hidden_size],
        &weights.final_layernorm,
    );
    if let Some(head) = &weights.lm_head {
        add_affine4_matrix(
            &mut records,
            "lm_head.weight",
            config.vocab_size,
            config.hidden_size,
            head.as_f32().expect("fixture f32"),
        );
    }
    let views: Vec<_> = records
        .iter()
        .map(|record| {
            (
                record.name.as_str(),
                TensorView::new(record.dtype, record.shape.clone(), &record.bytes).expect("tensor"),
            )
        })
        .collect();
    serialize(views, None).expect("actual safetensors checkpoint")
}

fn tokenizer_bytes() -> Vec<u8> {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    assert_eq!(alphabet.len(), 256);
    let mut vocabulary = serde_json::Map::new();
    for (id, character) in alphabet.into_iter().enumerate() {
        vocabulary.insert(character.to_string(), Value::from(id as u64));
    }
    vocabulary.insert("<|im_start|>".to_owned(), Value::from(256));
    vocabulary.insert("<|im_end|>".to_owned(), Value::from(257));
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            {"id":256,"content":"<|im_start|>","single_word":false,
             "lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":257,"content":"<|im_end|>","single_word":false,
             "lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type":"ByteLevel","add_prefix_space":false,
                          "trim_offsets":true,"use_regex":true},
        "post_processor": null,
        "decoder": {"type":"ByteLevel","add_prefix_space":false,
                    "trim_offsets":true,"use_regex":true},
        "model": {"type":"BPE","dropout":null,"unk_token":null,
                  "continuing_subword_prefix":null,"end_of_word_suffix":null,
                  "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                  "vocab":vocabulary,"merges":[]}
    }))
    .expect("tokenizer JSON")
}

struct Fixture {
    directory: Directory,
    specification: Qwen3Artifact,
    limits: Qwen3BackendLimits,
    profile: ModelProfile,
    context: ContextSnapshot,
}

impl Fixture {
    fn new() -> Self {
        let directory = Directory::new();
        let mut config = Qwen3Config::tiny_fixture();
        config.vocab_size = 258;
        let configuration = serde_json::to_vec(&serde_json::json!({
            "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
            "hidden_size":config.hidden_size, "num_hidden_layers":config.num_layers,
            "num_attention_heads":config.q_heads, "num_key_value_heads":config.kv_heads,
            "head_dim":config.head_dim, "intermediate_size":config.intermediate_size,
            "vocab_size":config.vocab_size, "rope_theta":config.rope_theta,
            "max_position_embeddings":config.max_context,
            "tie_word_embeddings":config.tie_embeddings, "hidden_act":"silu",
            "rms_norm_eps":0.000001, "attention_bias":false,
            "rope_scaling":null, "torch_dtype":"float32", "eos_token_id":257
        }))
        .expect("config JSON");
        let mut files = BTreeMap::new();
        for (name, bytes) in [
            ("config.json", configuration),
            ("tokenizer.json", tokenizer_bytes()),
            ("model.safetensors", checkpoint_bytes(&config)),
        ] {
            std::fs::write(directory.0.join(name), &bytes).expect("fixture component");
            files.insert(
                name.to_owned(),
                FilePin {
                    bytes: bytes.len() as u64,
                    sha256: digest(&bytes),
                },
            );
        }
        let artifact = ArtifactId::new("shared-renderer-artifact").expect("artifact");
        let profile_id = ModelProfileId::new("shared-renderer-profile").expect("profile");
        let profile = ModelProfile::new(
            profile_id.clone(),
            artifact.clone(),
            config.max_context,
            4,
            8,
            GenerationOptions::greedy(4).expect("options"),
        )
        .expect("profile");
        let context = ContextSnapshot::new(
            ContextSnapshotId::new("shared-renderer-snapshot").expect("snapshot"),
            profile_id,
            vec![Message {
                role: MessageRole::User,
                content: "hello".to_owned(),
            }],
        )
        .expect("context");
        let specification = Qwen3Artifact {
            id: artifact,
            directory: directory.0.clone(),
            files,
            template_sha256: digest(CHAT_TEMPLATE.as_bytes()),
        };
        let limits = Qwen3BackendLimits {
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
            generation: GenerationLimits::new(8, 4096, 1024).expect("buffers"),
        };
        Self {
            directory,
            specification,
            limits,
            profile,
            context,
        }
    }

    fn engine(&self) -> EngineCore<FamilyRuntime> {
        Qwen3Backend::new(vec![self.specification.clone()], self.limits.clone())
            .expect("actual backend")
            .into_engine(
                vec![self.profile.clone()],
                AdmissionController::new(16 * 1024 * 1024).expect("admission"),
                Scheduler::new(8).expect("scheduler"),
                RetentionPolicy {
                    retain_model: false,
                    retain_cache_buffers: false,
                },
            )
            .expect("actual engine")
    }
}

fn explicit_parser_capacities() -> ParserAllocationLimits {
    ParserAllocationLimits {
        carry_bytes: 32,
        call_bytes: 128,
        decoded_bytes: 128,
        depth: 8,
        keys: 16,
        nodes: 32,
        proposal_bytes: 256,
        calls: 4,
    }
}

fn parser_capacity_tuple(limits: ParserAllocationLimits) -> [usize; 8] {
    [
        limits.carry_bytes,
        limits.call_bytes,
        limits.decoded_bytes,
        limits.depth,
        limits.keys,
        limits.nodes,
        limits.proposal_bytes,
        limits.calls,
    ]
}

#[test]
fn qwen_parser_hooks_text_only_plan_and_decoder_are_unchanged() {
    let fixture = Fixture::new();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend");
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    let profile = fixture.context.profile_id();
    assert!(
        backend
            .parser_allocation_limits(&registry, profile)
            .expect("metadata")
            .is_none()
    );
    let rendered =
        block_on(backend.render_and_count(&registry, fixture.context.clone())).expect("render");
    let plan = block_on(backend.plan(&registry, profile, rendered.token_count())).expect("plan");
    assert_eq!(
        plan.output_bytes,
        output_bytes(fixture.limits.generation).expect("legacy output")
    );
    assert!(
        backend
            .decoder_with_parser_allocation(&registry, profile, None)
            .is_ok()
    );
}

#[test]
fn qwen_parser_hooks_flag_alone_does_not_invent_capacities() {
    let fixture = Fixture::new();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_tool_call_parsing(true);
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    let profile = fixture.context.profile_id();
    assert!(
        backend
            .parser_allocation_limits(&registry, profile)
            .is_err()
    );
    assert!(block_on(backend.plan(&registry, profile, 1)).is_err());
    // The non-admitted constructor cannot bypass the runtime hook.
    assert!(backend.decoder(&registry, profile).is_err());
}

#[test]
fn qwen_parser_hooks_snapshot_all_explicit_capacities_and_plan_slab_once() {
    let fixture = Fixture::new();
    let capacities = explicit_parser_capacities();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_parser_allocation_limits(capacities)
        .expect("capacities");
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    let profile = fixture.context.profile_id();
    let actual = backend
        .parser_allocation_limits(&registry, profile)
        .expect("metadata")
        .expect("selected");
    assert_eq!(
        parser_capacity_tuple(actual),
        parser_capacity_tuple(capacities)
    );
    let rendered =
        block_on(backend.render_and_count(&registry, fixture.context.clone())).expect("render");
    let plan = block_on(backend.plan(&registry, profile, rendered.token_count())).expect("plan");
    let expected = fixture
        .limits
        .output_memory_plan(
            Some(capacities),
            &[std::alloc::Layout::new::<Qwen3TextDecoder>()],
        )
        .expect("memory plan");
    assert_eq!(plan.output_bytes, expected.total_bytes());
    assert_eq!(
        plan.output_bytes - output_bytes(fixture.limits.generation).expect("legacy"),
        ParserAllocation::required_bytes(capacities).expect("slab")
            + std::mem::size_of::<Qwen3TextDecoder>() as u64
    );
}

#[test]
fn qwen_parser_hooks_disabling_tools_clears_capacities() {
    let fixture = Fixture::new();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_parser_allocation_limits(explicit_parser_capacities())
        .expect("capacities")
        .with_tool_call_parsing(false);
    assert!(
        backend
            .selected_parser_capacities()
            .expect("selection")
            .is_none()
    );
    assert_eq!(
        backend.planned_output_bytes().expect("plan"),
        output_bytes(fixture.limits.generation).expect("legacy")
    );
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    assert!(
        backend
            .decoder_with_parser_allocation(&registry, fixture.context.profile_id(), None)
            .is_ok()
    );
}

#[test]
fn qwen_parser_hooks_invalid_or_overflowing_capacities_are_rejected() {
    for invalid in [0, usize::MAX] {
        let fixture = Fixture::new();
        let mut capacities = explicit_parser_capacities();
        capacities.nodes = invalid;
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
                .expect("backend");
        assert!(backend.with_parser_allocation_limits(capacities).is_err());
    }
}

#[test]
fn qwen_parser_hooks_missing_slab_cannot_fall_back_to_legacy_parsing() {
    let fixture = Fixture::new();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_parser_allocation_limits(explicit_parser_capacities())
        .expect("capacities");
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    assert!(
        backend
            .decoder_with_parser_allocation(&registry, fixture.context.profile_id(), None)
            .is_err()
    );
    assert!(
        backend
            .decoder(&registry, fixture.context.profile_id())
            .is_err()
    );
}

#[test]
fn qwen_parser_hooks_supplied_slab_decodes_and_outlives_the_decoder() {
    let fixture = Fixture::new();
    let capacities = explicit_parser_capacities();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_parser_allocation_limits(capacities)
        .expect("capacities");
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    let bytes = ParserAllocation::required_bytes(capacities).expect("size");
    let owner_bytes = std::mem::size_of::<Qwen3TextDecoder>() as u64;
    let admission = AdmissionController::new(bytes + owner_bytes).expect("admission");
    let mut grant = admission
        .reserve_grant("slab and decoder owner", &[bytes, owner_bytes])
        .expect("grant");
    let decoder_owner = grant
        .take("decoder owner", owner_bytes)
        .expect("partition")
        .expect("positive");
    let slab = ParserAllocation::from_grant(&mut grant, capacities).expect("allocation");
    let mut decoder = backend
        .decoder_with_parser_allocation(&registry, fixture.context.profile_id(), Some(slab))
        .expect("storage-backed decoder");
    let data = backend
        .data(
            registry
                .resolve_profile(fixture.context.profile_id())
                .expect("profile")
                .artifact(),
        )
        .expect("data");
    let tokens = data
        .tokenizer
        .tokenizer()
        .encode(
            "before<tool_call>{\"name\":\"fs.read\",\"arguments\":{}}</tool_call>after",
            false,
        )
        .expect("tokens");
    let mut text = String::new();
    for &token in tokens.get_ids() {
        for block in decoder.push_blocks(token).expect("decode") {
            match block {
                crate::types::GenerationOutputBlock::Text(delta) => text.push_str(&delta),
                _ => panic!("slab decoding must not materialize owned proposal maps"),
            }
        }
    }
    decoder.finish_blocks().expect("finish");
    let batch = decoder.take_parser_proposals().expect("owned batch");
    assert!(decoder.take_parser_proposals().is_none());
    drop(decoder);
    drop(decoder_owner);
    drop(grant);
    assert_eq!(text, "beforeafter");
    assert_eq!(batch.proposal_count(), 1);
    assert_eq!(batch.proposal(0).expect("call").name, "fs.read");
    assert_eq!(batch.proposal(0).expect("call").arguments_json, "{}");
    assert_eq!(
        admission
            .snapshot()
            .expect("consumer retained")
            .reserved_bytes(),
        bytes
    );
    drop(batch);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}

#[test]
fn qwen_parser_hooks_wrong_slab_size_is_rejected_and_released() {
    let fixture = Fixture::new();
    let capacities = explicit_parser_capacities();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_parser_allocation_limits(capacities)
        .expect("capacities");
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    let mut different = capacities;
    different.calls += 1;
    let bytes = ParserAllocation::required_bytes(different).expect("bytes");
    let admission = AdmissionController::new(bytes).expect("admission");
    let mut grant = admission.reserve_grant("parser", &[bytes]).expect("grant");
    let allocation = ParserAllocation::from_grant(&mut grant, different).expect("slab");
    match backend.decoder_with_parser_allocation(
        &registry,
        fixture.context.profile_id(),
        Some(allocation),
    ) {
        Err(error) => assert!(
            error
                .to_string()
                .contains("differs from the planned slab size")
        ),
        Ok(_) => panic!("wrong allocation accepted"),
    }
    drop(grant);
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}

#[test]
fn tool_context_reaches_the_production_qwen_renderer() {
    use crate::types::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};
    let mut fixture = Fixture::new();
    // Rendering needs more context than the tiny execution fixture. Change
    // and re-pin the actual config; do not bypass profile/context checks.
    let path = fixture.directory.0.join("config.json");
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(&path).expect("config")).expect("JSON");
    config["max_position_embeddings"] = serde_json::json!(8192);
    let bytes = serde_json::to_vec(&config).expect("config JSON");
    std::fs::write(&path, &bytes).expect("config write");
    fixture.specification.files.insert(
        "config.json".to_owned(),
        FilePin {
            bytes: bytes.len() as u64,
            sha256: digest(&bytes),
        },
    );
    fixture.limits.context_limit_tokens = 8192;
    fixture.profile = ModelProfile::new(
        fixture.context.profile_id().clone(),
        fixture.specification.id.clone(),
        8192,
        4,
        8,
        GenerationOptions::greedy(4).expect("options"),
    )
    .expect("profile");
    let metadata = RenderingTool::new(
        "fs.read".into(),
        "1.0.0".into(),
        "Read file".into(),
        serde_json::json!({"type":"object","properties":{"path":{"type":"string"}}}),
    )
    .expect("metadata");
    let history = vec![
        ToolHistoryBlock::AssistantCall {
            call_id: "call-1".into(),
            tool_id: "fs.read".into(),
            arguments: serde_json::json!({"path":"file.txt"})
                .as_object()
                .expect("object")
                .clone(),
        },
        ToolHistoryBlock::ToolResult {
            call_id: "call-1".into(),
            content: "result data".into(),
            is_error: false,
        },
    ];
    let tool_context = ToolContext::new(
        vec![metadata],
        history,
        ToolContextLimits {
            max_tools: 4,
            max_history_blocks: 8,
            max_json_bytes: 4096,
        },
    )
    .expect("tool context");
    let context = fixture.context.clone().with_tool_context(tool_context);
    assert!(
        block_on(fixture.engine().render_and_count(context.clone())).is_err(),
        "text-only backend accepted tools"
    );
    let engine = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend")
        .with_tool_call_parsing(true)
        .into_engine(
            vec![fixture.profile.clone()],
            AdmissionController::new(16 * 1024 * 1024).expect("admission"),
            Scheduler::new(8).expect("scheduler"),
            RetentionPolicy {
                retain_model: false,
                retain_cache_buffers: false,
            },
        )
        .expect("engine");
    let rendered = block_on(engine.render_and_count(context)).expect("production tool rendering");
    assert!(rendered.text().contains("\"name\":\"fs.read\""));
    assert!(rendered.text().contains("result data"));
    assert!(rendered.tool_schema_hash().is_some());
    let source = std::fs::read(fixture.directory.0.join("tokenizer.json")).expect("tokenizer");
    let tokenizer = PinnedTokenizer::from_pinned_json(
        &source,
        fixture.specification.files["tokenizer.json"].sha256,
        fixture.limits.max_tokenizer_bytes,
    )
    .expect("pin");
    let expected = tokenizer
        .tokenizer()
        .encode(rendered.text(), false)
        .expect("encode");
    assert_eq!(rendered.token_ids(), expected.get_ids());
}

#[test]
fn engine_rendering_matches_shared_renderer_and_legacy_prompt_text() {
    let fixture = Fixture::new();
    let engine = fixture.engine();
    let actual = block_on(engine.render_and_count(fixture.context.clone()))
        .expect("production Engine rendering");
    let bytes = std::fs::read(fixture.directory.0.join("tokenizer.json"))
        .expect("known fixture tokenizer source");
    let tokenizer = PinnedTokenizer::from_pinned_json(
        &bytes,
        fixture.specification.files["tokenizer.json"].sha256,
        fixture.limits.max_tokenizer_bytes,
    )
    .expect("shared pinned tokenizer");
    let binding = RenderingBinding {
        profile_id: fixture.context.profile_id(),
        artifact_id: &fixture.specification.id,
        template: CHAT_TEMPLATE,
        template_sha256: fixture.specification.template_sha256,
        artifact_manifest_sha256: fixture.specification.manifest_sha256().expect("manifest"),
    };
    // Intentionally spell the expected Qwen policy independently of
    // qwen3_rendering_options, including the prior whitespace defaults.
    let options = RenderingOptions {
        add_generation_prompt: true,
        enable_thinking: Some(false),
        add_special_tokens: false,
        keep_trailing_newline: false,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: QWEN3_RESERVED_SPELLINGS,
    };
    let expected = rendering::render_and_count(
        &fixture.context,
        &binding,
        &tokenizer,
        &options,
        RenderingLimits::new(
            fixture.limits.max_messages,
            fixture.limits.max_input_bytes,
            CHAT_TEMPLATE.len(),
            fixture.limits.max_rendered_bytes,
            fixture.profile.context_limit_tokens(),
        )
        .expect("render limits"),
    )
    .expect("shared render");
    // Includes the shared tokenizer ID and render hash. The removed
    // private implementation cannot satisfy this equality.
    assert_eq!(actual, expected);
    assert_eq!(actual.tokenizer_id(), tokenizer.identity());

    // Preserve the actual previously selected model-visible prompt,
    // not just equivalence between two new-code call sites.
    let mut legacy = minijinja::Environment::new();
    legacy.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    legacy
        .add_template("qwen3-text", CHAT_TEMPLATE)
        .expect("legacy template");
    let legacy_text = legacy
        .get_template("qwen3-text")
        .expect("template")
        .render(minijinja::context! {
            messages => vec![minijinja::context! { role => "user", content => "hello" }]
        })
        .expect("prior prompt contract");
    assert_eq!(actual.text(), legacy_text.as_str());
    assert!(
        engine
            .active_request_id()
            .expect("execution owner")
            .is_none()
    );
}

#[test]
fn engine_reserved_syntax_errors_come_from_shared_renderer() {
    let fixture = Fixture::new();
    let engine = fixture.engine();
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("reserved-syntax").expect("snapshot"),
        fixture.context.profile_id().clone(),
        vec![Message {
            role: MessageRole::User,
            content: "hello <think>".to_owned(),
        }],
    )
    .expect("context");
    let error = block_on(engine.render_and_count(context)).expect_err("reserved syntax");
    match error {
        EngineError::InvalidRequest { field, .. } => assert_eq!(field, "message.content"),
        other => panic!("expected shared validation error, got {other:?}"),
    }
}

#[test]
fn engine_output_bounds_use_shared_writer_errors() {
    let mut fixture = Fixture::new();
    fixture.limits.max_rendered_bytes = 1;
    let engine = fixture.engine();
    let error =
        block_on(engine.render_and_count(fixture.context.clone())).expect_err("output bound");
    match error {
        EngineError::MemoryLimitExceeded {
            operation,
            required_bytes,
            available_bytes,
            reason,
        } => {
            assert_eq!(operation, "render chat template output");
            assert!(required_bytes > 1);
            assert_eq!(available_bytes, 1);
            assert_eq!(reason, MemoryLimitReason::ContextRenderingBufferLimit);
        }
        other => panic!("expected shared writer error, got {other:?}"),
    }
}

#[test]
fn manifest_fingerprint_tracks_relevant_pins_not_local_directory() {
    let fixture = Fixture::new();
    let original = fixture.specification.manifest_sha256().expect("manifest");
    let mut relocated = fixture.specification.clone();
    relocated.directory = PathBuf::from("another-local-directory");
    assert_eq!(original, relocated.manifest_sha256().expect("relocated"));
    let mut changed = fixture.specification.clone();
    changed
        .files
        .get_mut("tokenizer.json")
        .expect("tokenizer pin")
        .sha256[0] ^= 1;
    assert_ne!(original, changed.manifest_sha256().expect("changed pin"));
    let mut changed = fixture.specification.clone();
    changed
        .files
        .get_mut("model.safetensors")
        .expect("weight pin")
        .bytes += 1;
    assert_ne!(original, changed.manifest_sha256().expect("changed length"));
    let mut changed = fixture.specification.clone();
    changed.template_sha256[0] ^= 1;
    assert_ne!(
        original,
        changed.manifest_sha256().expect("changed template")
    );
}

fn executable_fixture() -> Fixture {
    let mut fixture = Fixture::new();
    let config_path = fixture.directory.0.join("config.json");
    let mut value: Value =
        serde_json::from_slice(&std::fs::read(&config_path).expect("config")).expect("JSON");
    value["tie_word_embeddings"] = Value::Bool(false);
    let config = parse_config(&value).expect("untied config");
    for (name, bytes) in [
        (
            "config.json",
            serde_json::to_vec(&value).expect("config JSON"),
        ),
        (
            "model.safetensors",
            checkpoint_bytes_selected(&config, true),
        ),
    ] {
        std::fs::write(fixture.directory.0.join(name), &bytes).expect("fixture update");
        fixture.specification.files.insert(
            name.to_owned(),
            FilePin {
                bytes: bytes.len() as u64,
                sha256: digest(&bytes),
            },
        );
    }
    fixture
}

fn execute_engine_path(execution: Qwen3Execution) {
    execute_engine_path_with_pressure(execution, None);
}

fn execute_engine_path_with_pressure(
    execution: Qwen3Execution,
    pressure: Option<crate::pressure::PressurePolicy>,
) {
    let fixture = executable_fixture();
    let backend = Qwen3Backend::new_with_execution(
        vec![fixture.specification.clone()],
        fixture.limits.clone(),
        execution,
    )
    .expect("MLX backend");
    let admission = AdmissionController::new(16 * 1024 * 1024).expect("budget");
    let scheduler = Scheduler::new(8).expect("scheduler");
    let retention = RetentionPolicy {
        retain_model: true,
        retain_cache_buffers: false,
    };
    let engine = match pressure {
        Some(policy) => backend.into_pressure_engine(
            vec![fixture.profile.clone()],
            admission,
            scheduler,
            retention,
            policy,
        ),
        None => backend.into_engine(
            vec![fixture.profile.clone()],
            admission,
            scheduler,
            retention,
        ),
    }
    .expect("engine");
    let rendered = block_on(engine.render_and_count(fixture.context.clone())).expect("render");
    let request = crate::types::GenerationRequest::new(
        crate::types::RequestId::new("mlx-adapter-execution").expect("ID"),
        fixture.context.profile_id().clone(),
        fixture.context.clone(),
        GenerationOptions::greedy(4).expect("options"),
    )
    .expect("request");
    let generation = block_on(engine.generate(request)).expect("admitted MLX execution");
    let mut events = generation.into_events();
    let mut text = String::new();
    let mut terminal = false;
    while let Some(event) = block_on(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
        assert!(!terminal, "event after completion");
        match event {
            crate::types::GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
            crate::types::GenerationEvent::GenerationCompleted {
                finish_reason,
                usage,
            } => {
                assert_eq!(finish_reason, crate::types::FinishReason::OutputLimit);
                assert_eq!(usage.input_tokens, rendered.token_count());
                assert_eq!(usage.output_tokens, 4);
                terminal = true;
            }
            other => panic!("unexpected MLX execution event: {other:?}"),
        }
    }
    assert!(terminal);
    assert_eq!(text, "!!!!");
    drop(events);
    let caps = block_on(engine.capabilities(fixture.specification.id.clone()))
        .expect("loaded capabilities");
    assert_eq!(caps.backend, execution.backend());
    assert_eq!(
        caps.weight_representations,
        vec![
            WeightRepresentation::F32,
            WeightRepresentation::Affine4 {
                group_size: NonZeroUsize::new(64).expect("64"),
            },
        ]
    );
    assert_eq!(
        caps.cache_modes[0].representation.as_str(),
        QWEN3_MLX_CACHE_REPRESENTATION
    );
    assert!(!caps.supports_prefix_reuse);
}

#[test]
fn engine_generate_executes_with_explicit_admission_pressure_policy() {
    execute_engine_path_with_pressure(
        Qwen3Execution::MlxCpu,
        Some(
            crate::pressure::PressurePolicy::new(crate::pressure::PressureOptions {
                allocator_ceiling_bytes: None,
                allow_allocator_cache_clear: true,
                allow_idle_state_release: true,
                allow_idle_model_unload: true,
                max_reclamation_actions: 5,
            })
            .expect("explicit pressure policy"),
        ),
    );
}

#[test]
fn pressure_enabled_startup_reaches_worker_after_failed_advisory_estimate() {
    let fixture = executable_fixture();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend");
    let policy = crate::pressure::PressurePolicy::new(crate::pressure::PressureOptions {
        allocator_ceiling_bytes: None,
        allow_allocator_cache_clear: false,
        allow_idle_state_release: false,
        allow_idle_model_unload: false,
        max_reclamation_actions: 0,
    })
    .expect("policy");
    let engine = backend
        .into_pressure_engine(
            vec![fixture.profile.clone()],
            AdmissionController::new(1).expect("tiny budget"),
            Scheduler::new(8).expect("scheduler"),
            RetentionPolicy {
                retain_model: false,
                retain_cache_buffers: false,
            },
            policy,
        )
        .expect("engine");
    let request = crate::types::GenerationRequest::new(
        crate::types::RequestId::new("pressure-advisory-deferral").expect("ID"),
        fixture.context.profile_id().clone(),
        fixture.context.clone(),
        GenerationOptions::greedy(4).expect("options"),
    )
    .expect("request");
    match block_on(engine.generate(request)) {
        Err(EngineError::MemoryLimitExceeded { operation, .. }) => {
            // The old API fail-fast operation is "generation startup
            // estimate". This operation is emitted by the native policy.
            assert_eq!(operation, "startup demand exceeds admission budget");
        }
        other => panic!("expected native pressure rejection, got {other:?}"),
    }
}

#[test]
fn pressure_releases_idle_state_without_reloading_model_under_ledger_pressure() {
    let fixture = executable_fixture();
    let backend = Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
        .expect("backend");
    let registry = backend
        .registry(vec![fixture.profile.clone()])
        .expect("registry");
    let rendered =
        block_on(backend.render_and_count(&registry, fixture.context.clone())).expect("render");
    let plan = block_on(backend.plan(
        &registry,
        fixture.context.profile_id(),
        rendered.token_count(),
    ))
    .expect("plan");
    let admission = AdmissionController::new(16 * 1024 * 1024).expect("budget");
    let scheduler = Scheduler::new(16).expect("scheduler");
    let policy = crate::pressure::PressurePolicy::new(crate::pressure::PressureOptions {
        allocator_ceiling_bytes: None,
        allow_allocator_cache_clear: true,
        allow_idle_state_release: true,
        allow_idle_model_unload: false,
        max_reclamation_actions: 5,
    })
    .expect("state-only pressure policy");
    let backend: Arc<dyn RuntimeBackend> = Arc::new(backend);
    let runtime = FamilyRuntime::new(
        vec![backend],
        admission.clone(),
        fixture.limits.generation,
        RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: true,
        },
    )
    .expect("runtime")
    .with_pressure_policy(policy);
    let engine = EngineCore::new(registry, scheduler, runtime.clone());
    let run = |name: &str| {
        let request = crate::types::GenerationRequest::new(
            crate::types::RequestId::new(name).expect("ID"),
            fixture.context.profile_id().clone(),
            fixture.context.clone(),
            GenerationOptions::greedy(4).expect("options"),
        )
        .expect("request");
        let mut events = block_on(engine.generate(request))
            .expect("pressure-admitted execution")
            .into_events();
        let mut text = String::new();
        let mut terminal = false;
        while let Some(event) = block_on(std::future::poll_fn(|cx| events.as_mut().poll_next(cx))) {
            assert!(!terminal, "event after terminal");
            match event {
                crate::types::GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
                crate::types::GenerationEvent::GenerationCompleted { usage, .. } => {
                    assert_eq!(usage.input_tokens, rendered.token_count());
                    assert_eq!(usage.output_tokens, 4);
                    terminal = true;
                }
                other => panic!("pressure execution failed: {other:?}"),
            }
        }
        assert!(terminal);
        assert_eq!(text, "!!!!");
        drop(events);
    };
    run("pressure-retained-first");
    let source =
        std::fs::read(fixture.directory.0.join("tokenizer.json")).expect("tokenizer source");
    let tokenizer = tokenizers::Tokenizer::from_bytes(&source).expect("tokenizer");
    let mut executed = rendered.token_ids().to_vec();
    executed.extend_from_slice(&[0, 0, 0, 0]);
    let decoded = tokenizer
        .decode(&executed, false)
        .expect("lossless fixture decode");
    assert_eq!(
        tokenizer
            .encode(decoded.as_str(), false)
            .expect("roundtrip")
            .get_ids(),
        executed.as_slice()
    );
    let prefix = decoded.into_bytes();
    let first_id = block_on(runtime.retain_session_state(
        crate::types::RequestId::new("pressure-retained-first").expect("ID"),
        prefix.clone(),
    ))
    .expect("first publication");
    let first = block_on(runtime.lookup_session_state(first_id))
        .expect("lookup")
        .expect("first state");
    let warm_required = sum(&[
        plan.temporary_bytes,
        plan.headroom_bytes,
        plan.output_bytes,
        product(as_u64(plan.cache.capacity_tokens()).expect("capacity"), 4).expect("history"),
    ])
    .expect("warm requirement");
    let available = admission.snapshot().expect("idle ledger").available_bytes();
    assert!(available >= warm_required && warm_required > 0);
    let fill = available
        .checked_sub(warm_required - 1)
        .expect("fill amount");
    let unrelated = admission
        .reserve("pressure fixture unrelated owner", fill)
        .expect("fill ledger");
    assert_eq!(
        admission
            .snapshot()
            .expect("pressured ledger")
            .available_bytes(),
        warm_required - 1
    );
    // A fresh advisory estimate cannot fit. The native state-only
    // action must release real guards; model unload is forbidden.
    run("pressure-retained-second");
    let second_id = block_on(runtime.retain_session_state(
        crate::types::RequestId::new("pressure-retained-second").expect("ID"),
        prefix,
    ))
    .expect("second publication");
    let second = block_on(runtime.lookup_session_state(second_id))
        .expect("lookup")
        .expect("second state");
    assert_eq!(second.model_instance_id, first.model_instance_id);
    assert_eq!(second.request_id.as_str(), "pressure-retained-second");
    assert_eq!(second.reused_input_tokens, 0);
    drop(unrelated);
    assert_eq!(
        block_on(engine.release_session_state(second_id)).expect("release"),
        crate::types::ReleaseOutcome::Released
    );
    assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
}
#[test]
fn engine_generate_executes_mlx_cpu_with_new_cache_identity() {
    execute_engine_path(Qwen3Execution::MlxCpu);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn engine_generate_executes_metal_with_unified_mlx_weights_and_kv() {
    execute_engine_path(Qwen3Execution::MlxMetal);
}

#[test]
fn mlx_planning_counts_native_weights_and_kv_update_overlap() {
    for tied in [true, false] {
        let mut config = Qwen3Config::tiny_fixture();
        config.tie_embeddings = tied;
        let dense = qwen3::fixture::generate_fixture(&config);
        let dense_payload = weight_memory(&dense).expect("f32 accounting");
        let headers = config.num_layers * std::mem::size_of::<LayerWeights>();
        let dense_matrices = dense_payload.used_bytes() - headers as u64;
        let host = host_affine4_bound(&config).expect("host affine4");
        let resident = resident_bound(&config).expect("mlx resident");
        // Quantized host estimate must beat a full f32 duplicate.
        assert!(host < dense_matrices, "host={host} dense={dense_matrices}");
        // Every matrix, including tied/untied vocabulary weights, stays packed.
        assert_eq!(resident, host);
        // Still well under dense when projections dominate.
        assert!(
            resident < dense_matrices,
            "resident={resident} dense={dense_matrices}"
        );
        let plan = cache_spec(&config, 16).payload_plan().expect("cache plan");
        assert_eq!(
            cache_bound(&config, 16).expect("cache allowance"),
            plan.resident_bytes * 2
        );
        assert!(
            logits_payload_bound(&config).expect("output allowance")
                >= config.vocab_size as u64 * 12
        );
    }
}
