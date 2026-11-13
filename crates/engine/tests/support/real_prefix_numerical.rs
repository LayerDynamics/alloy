//! Trained Qwen SSD reuse compared with the actual eager cold forward path.
use super::*;
use engine::backends::Qwen3Execution;
use engine::tiered_kv::{HotCacheLimit, TieredKvConfig};

#[test]
#[ignore = "requires the pinned local Qwen3-4B model and Metal"]
fn real_prefix_restored_logits_match_cold() {
    let directory = PathBuf::from(std::env::var_os("ALLOY_REAL_MODEL").expect("ALLOY_REAL_MODEL"));
    let corpus: Value =
        serde_json::from_str(include_str!("../../../../bench/corpus/models.json")).unwrap();
    let profile = corpus["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "qwen3-4b-4bit")
        .unwrap();
    let mut files = BTreeMap::new();
    for item in profile["artifact"]["files"].as_array().unwrap() {
        let name = item["path"].as_str().unwrap();
        let hex = item["sha256"].as_str().unwrap();
        let mut hash = [0u8; 32];
        for (i, byte) in hash.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        files.insert(
            name.to_owned(),
            FilePin {
                bytes: std::fs::metadata(directory.join(name)).unwrap().len(),
                sha256: hash,
            },
        );
    }
    let specification = Qwen3Artifact {
        id: artifact_id(),
        directory: directory.clone(),
        files,
        template_sha256: digest(CHAT_TEMPLATE.as_bytes()),
    };
    let manifest_sha256 = specification.manifest_sha256().unwrap();
    let config: Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    let vocabulary = config["vocab_size"].as_u64().unwrap() as usize;
    let context_limit = config["max_position_embeddings"].as_u64().unwrap() as usize;
    let tokenizer_source = std::fs::read(directory.join("tokenizer.json")).unwrap();
    let tokenizer_hash = digest(&tokenizer_source);
    let tokenizer_a =
        PinnedTokenizer::from_pinned_json(&tokenizer_source, tokenizer_hash, 16 * 1024 * 1024)
            .unwrap();
    let tokenizer_b =
        PinnedTokenizer::from_pinned_json(&tokenizer_source, tokenizer_hash, 16 * 1024 * 1024)
            .unwrap();
    let generation = GenerationLimits::new(64, 64 * 1024, 16 * 1024).unwrap();
    let inner = Qwen3Backend::new_with_execution(
        vec![specification],
        Qwen3BackendLimits {
            max_files: 64,
            max_config_bytes: 8 * 1024 * 1024,
            max_tokenizer_bytes: 16 * 1024 * 1024,
            max_messages: 256,
            max_input_bytes: 1024 * 1024,
            max_rendered_bytes: 2 * 1024 * 1024,
            max_token_bytes: 512 * 1024,
            verification_chunk_bytes: 1024 * 1024,
            context_limit_tokens: context_limit,
            prefill_chunk_tokens: NonZeroUsize::new(256).unwrap(),
            loader_headroom_bytes: 64 * 1024 * 1024,
            execution_headroom_bytes: 64 * 1024 * 1024,
            generation,
        },
        Qwen3Execution::MlxMetal,
    )
    .unwrap();
    let registry = inner
        .registry(vec![
            ModelProfile::new(
                profile_id(),
                artifact_id(),
                2048,
                4,
                64,
                GenerationOptions::greedy(4).unwrap(),
            )
            .unwrap(),
        ])
        .unwrap();
    let observations = Arc::new(Observations::default());
    let backend: Arc<dyn RuntimeBackend> = Arc::new(ObservedBackend {
        inner,
        observations: Arc::clone(&observations),
        render_selection: Arc::new(AtomicUsize::new(RENDER_QWEN)),
        tokenizer_a,
        tokenizer_b,
        manifest_sha256,
        vocabulary,
        fixed_capacity: false,
    });
    let admission = AdmissionController::new(16 * 1024 * 1024 * 1024).unwrap();
    let scheduler = Scheduler::new(16).unwrap();
    let runtime =
        FamilyRuntime::new(vec![backend], admission, generation, retained_policy()).unwrap();
    let cache = TestDirectory::new();
    runtime
        .configure_tiered_kv(TieredKvConfig::new(
            Some(cache.path().join("ssd")),
            HotCacheLimit::Bytes(0),
        ))
        .unwrap();
    let engine = EngineCore::new(registry, scheduler.clone(), runtime);
    let snapshot = context(
        "real-prefix-numerical",
        profile_id(),
        "Explain why exact token prefixes matter in one sentence.",
    );
    let rendered = block_on(engine.render_and_count(snapshot.clone())).unwrap();
    let run = |name: &str| {
        let request = GenerationRequest::new(
            RequestId::new(name).unwrap(),
            profile_id(),
            snapshot.clone(),
            GenerationOptions::greedy(4).unwrap(),
        )
        .unwrap();
        let generation = block_on(engine.generate(request)).unwrap();
        let mut events = generation.into_events();
        let transcript = drain(&mut events);
        drop(events);
        let until = deadline();
        while scheduler.is_busy().unwrap() {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        transcript
    };
    let cold = run("real-numerical-cold");
    let cold_work = observations.work("real-numerical-cold");
    assert_eq!(cold_work.prefill_tokens, rendered.token_ids());
    assert!(engine.runtime().tiered_kv_snapshot().spill_count > 0);
    let warm = run("real-numerical-restored");
    let warm_work = observations.work("real-numerical-restored");
    let metrics = engine.runtime().tiered_kv_snapshot();
    assert!(metrics.restore_count > 0);
    assert_eq!(metrics.restored_tokens, (rendered.token_count() - 1) as u64);
    assert_eq!(
        warm_work.prefill_tokens,
        &rendered.token_ids()[rendered.token_count() - 1..]
    );
    assert_eq!(warm.text, cold.text);
    assert_eq!(warm_work.decode_tokens, cold_work.decode_tokens);
    assert!(!warm_work.decode_tokens.is_empty());
    let mut max_abs = 0.0f64;
    let mut max_relative = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut compare = |actual: &[f32], expected: &[f32]| {
        assert!(!actual.is_empty());
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            let diff = f64::from(a) - f64::from(b);
            assert!(
                diff.abs() <= 2e-2 + 1e-2 * f64::from(b).abs(),
                "elementwise logit mismatch: actual={a}, expected={b}"
            );
            max_abs = max_abs.max(diff.abs());
            let relative = if b == 0.0 {
                if a == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                diff.abs() / f64::from(b).abs()
            };
            max_relative = max_relative.max(relative);
            sum_sq += diff * diff;
            ref_sq += f64::from(b) * f64::from(b);
        }
    };
    compare(
        &warm_work.final_prefill_logits,
        &cold_work.final_prefill_logits,
    );
    assert_eq!(warm_work.decode_logits.len(), cold_work.decode_logits.len());
    for (actual, expected) in warm_work.decode_logits.iter().zip(&cold_work.decode_logits) {
        compare(actual, expected);
    }
    assert!(ref_sq > 0.0, "reference logits must not all be zero");
    let relative_l2 = sum_sq.sqrt() / ref_sq.sqrt();
    println!(
        "real_prefix_numerical: input_tokens={}, restored_tokens={}, decode_tokens={:?}, max_abs={max_abs}, relative_l2={relative_l2}, max_relative={max_relative}",
        rendered.token_count(),
        metrics.restored_tokens,
        warm_work.decode_tokens
    );
}
