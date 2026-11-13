//! M10 acceptance through real Qwen MLX weights, KV, and generation.
//!
//! The checkpoint fixture uses a nonzero output head. Its decode-logit
//! oracle therefore detects lost or incorrectly restored attention state.
//! Apple Silicon uses the production Metal path; other hosts use MLX CPU.
//! Each recreated runtime opens the same pinned checkpoint files and SSD
//! directory, while loading a new actual model instance.

use super::*;
use engine::tiered_kv::{HotCacheLimit, TieredKvConfig, TieredKvSnapshot};
use safetensors::SafeTensors;
use std::fs;

const TIERED_BUDGET: u64 = 64 * 1024 * 1024;
const TIERED_TRANSFER: u64 = 9 * 1024 * 1024;
const TIERED_BODY: &str = "Restore native attention keys and values across blocks and preserve exact continuation logits.";

/// Keep the exact pinned files alive across independently owned runtimes.
struct TieredFixtures {
    checkpoints: Vec<TestEnvironment>,
}

impl TieredFixtures {
    fn new() -> Self {
        Self {
            checkpoints: (0..3)
                .map(|_| TestEnvironment::prefix(retained_policy()))
                .collect(),
        }
    }

    fn environment(&self) -> PolicyEnvironment {
        let context_limit = Qwen3Config::tiny_fixture().max_context;
        let generation_limits = GenerationLimits::new(8, 4096, 1024).expect("generation limits");
        let execution = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Qwen3Execution::MlxMetal
        } else {
            Qwen3Execution::MlxCpu
        };
        let inner = Qwen3Backend::new_with_execution(
            vec![
                fixture_specification(&self.checkpoints[0], artifact_a()),
                fixture_specification(&self.checkpoints[1], artifact_b()),
                fixture_specification(&self.checkpoints[2], artifact_c()),
            ],
            Qwen3BackendLimits {
                max_files: 16,
                max_config_bytes: 1024 * 1024,
                max_tokenizer_bytes: 1024 * 1024,
                max_messages: 16,
                max_input_bytes: 4096,
                max_rendered_bytes: 16 * 1024,
                max_token_bytes: 128,
                verification_chunk_bytes: 4096,
                context_limit_tokens: context_limit,
                prefill_chunk_tokens: NonZeroUsize::new(2).unwrap(),
                loader_headroom_bytes: 64 * 1024,
                execution_headroom_bytes: 64 * 1024,
                generation: generation_limits,
            },
            execution,
        )
        .expect("real tiered Qwen backend");
        let make_profile = |profile, artifact, output| {
            ModelProfile::new(
                profile,
                artifact,
                context_limit,
                output,
                POLICY_MARGIN,
                GenerationOptions::greedy(output).unwrap(),
            )
            .unwrap()
        };
        let registry = inner
            .registry(vec![
                make_profile(profile_a(), artifact_a(), POLICY_OUTPUT),
                make_profile(profile_b(), artifact_b(), POLICY_OUTPUT),
                make_profile(profile_c(), artifact_c(), POLICY_OUTPUT),
                make_profile(profile_large_cache(), artifact_a(), POLICY_OUTPUT + 4),
            ])
            .unwrap();
        let plan = |profile: ModelProfileId| {
            let rendered = block_on(
                inner.render_and_count(&registry, request_context(profile.clone(), TIERED_BODY)),
            )
            .unwrap();
            block_on(inner.plan(&registry, &profile, rendered.token_count())).unwrap()
        };
        let plan_a = plan(profile_a());
        let plan_b = plan(profile_b());
        let plan_c = plan(profile_c());
        let plan_large = plan(profile_large_cache());
        let admission = AdmissionController::new(TIERED_BUDGET).unwrap();
        let scheduler = Scheduler::new(64).unwrap();
        let control = Arc::new(PolicyControl::new(admission.clone()));
        let observations = Arc::new(Observations::default());
        let backend: Arc<dyn RuntimeBackend> = Arc::new(PolicyBackend {
            inner,
            control: Arc::clone(&control),
            observations: Arc::clone(&observations),
        });
        let runtime = FamilyRuntime::new(
            vec![backend],
            admission.clone(),
            generation_limits,
            retained_policy(),
        )
        .unwrap()
        .with_model_pool(None)
        .unwrap();
        PolicyEnvironment {
            runtime: runtime.clone(),
            engine: EngineCore::new(registry, scheduler.clone(), runtime),
            admission,
            scheduler,
            control,
            observations,
            plan_a,
            plan_b,
            plan_c,
            plan_large,
            // TieredFixtures owns the exact files shared by these runtimes.
            _fixtures: Vec::new(),
        }
    }

    fn publish(
        &self,
        environment: &PolicyEnvironment,
        name: &str,
    ) -> (
        engine::types::SessionStateId,
        engine::runtime::RetainedSessionState,
    ) {
        let rendered = render(environment, profile_a(), TIERED_BODY);
        let transcript = environment.run(name, profile_a(), TIERED_BODY);
        completed_usage(&transcript, FinishReason::OutputLimit);
        let work = environment.observations.work(name);
        assert_eq!(work.decode_tokens.len(), POLICY_OUTPUT);
        let mut tokens = rendered.token_ids().to_vec();
        tokens.extend_from_slice(&work.decode_tokens);
        let tokenizer =
            Tokenizer::from_file(self.checkpoints[0]._directory.path().join("tokenizer.json"))
                .unwrap();
        let prefix = tokenizer.decode(&tokens, false).unwrap();
        assert_eq!(
            tokenizer.encode(prefix.as_str(), false).unwrap().get_ids(),
            tokens.as_slice(),
        );
        let id = block_on(
            environment
                .runtime
                .retain_session_state(RequestId::new(name).unwrap(), prefix.into_bytes()),
        )
        .unwrap();
        let state = block_on(environment.runtime.lookup_session_state(id))
            .unwrap()
            .expect("actual retained native publication");
        (id, state)
    }
}

fn configuration(path: &Path, hot_limit: HotCacheLimit) -> TieredKvConfig {
    let mut config = TieredKvConfig::new(Some(path.to_path_buf()), hot_limit);
    config.transfer_bytes = TIERED_TRANSFER;
    config
}

fn configure(environment: &PolicyEnvironment, config: TieredKvConfig) -> TieredKvSnapshot {
    environment.runtime.configure_tiered_kv(config).unwrap();
    settled_metrics(environment)
}

fn settled_metrics(environment: &PolicyEnvironment) -> TieredKvSnapshot {
    environment.wait_idle();
    // A worker barrier ensures completion and its final metadata publication
    // are visible before checking the nonblocking snapshot.
    block_on(environment.runtime.model_residency()).unwrap();
    environment.runtime.tiered_kv_snapshot()
}

fn render(environment: &PolicyEnvironment, profile: ModelProfileId, body: &str) -> RenderedContext {
    block_on(
        environment
            .engine
            .render_and_count(request_context(profile, body)),
    )
    .unwrap()
}

fn prompt_with_partial_last_block(environment: &PolicyEnvironment) -> (String, RenderedContext) {
    let mut body = TIERED_BODY.to_owned();
    loop {
        let rendered = render(environment, profile_a(), &body);
        assert!(
            rendered.token_count() > 64,
            "fixture must cross multiple SSD blocks"
        );
        if !(rendered.token_count() - 1).is_multiple_of(32) {
            return (body, rendered);
        }
        // The byte-level tokenizer makes one additional ASCII byte a
        // deterministic adjustment to the partial-block restore boundary.
        body.push('x');
        assert!(body.len() <= TIERED_BODY.len() + 2);
    }
}

fn entries(path: &Path) -> Vec<PathBuf> {
    if !path.exists() {
        return Vec::new();
    }
    let mut directories: Vec<_> = fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("kv-v1-")
        })
        .collect();
    directories.sort();
    directories
}

fn assert_only_weights_and_transfer(environment: &PolicyEnvironment) {
    let metrics = settled_metrics(environment);
    let models = environment.runtime.residency_snapshot().models;
    assert!(!models.is_empty());
    let weight_bytes: u64 = models
        .iter()
        .map(|model| model.weights_reserved_bytes)
        .sum();
    assert!(weight_bytes > 0);
    assert!(models.iter().all(|model| model.kv_reserved_bytes == 0));
    assert_eq!(metrics.hot_reserved_bytes, 0);
    assert_eq!(metrics.protected_reserved_bytes, 0);
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        weight_bytes + metrics.transfer_reserved_bytes,
        "idle history, logits, request workspaces, and KV must be released",
    );
}

fn drop_runtime_and_assert_unreserved(environment: PolicyEnvironment) {
    environment.wait_idle();
    let admission = environment.admission.clone();
    drop(environment);
    let limit = deadline();
    loop {
        let accounting = admission.snapshot().unwrap();
        if accounting.reserved_bytes() == 0 {
            assert_eq!(accounting.active_reservations(), 0);
            return;
        }
        assert!(
            Instant::now() < limit,
            "native runtime drop retained an admitted allocation: {accounting:?}",
        );
        thread::yield_now();
    }
}

#[test]
fn tiered_finish_persists_native_blocks_and_zero_hot_cap_retains_only_weights() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let environment = fixtures.environment();
    let initial = configure(
        &environment,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    assert!(initial.enabled && initial.disk_enabled);
    assert_eq!(initial.transfer_reserved_bytes, TIERED_TRANSFER);
    assert_eq!(environment.snapshot().reserved_bytes(), TIERED_TRANSFER);
    let (body, rendered) = prompt_with_partial_last_block(&environment);
    let first = environment.run("tiered-save-first", profile_a(), &body);
    completed_usage(&first, FinishReason::OutputLimit);
    let first_work = environment.observations.work("tiered-save-first");
    assert_eq!(first_work.prefill_tokens, rendered.token_ids());
    let instance = pool_instance(&environment, &artifact_a());
    let metrics = settled_metrics(&environment);
    assert_eq!(metrics.spill_count, 1);
    assert_eq!(metrics.cold_entries, 1);
    assert_eq!(metrics.restore_count, 0);
    assert_eq!(metrics.hot_limit_bytes, 0);
    assert!(metrics.hot_evictions >= 1);
    assert!(metrics.last_error.is_none(), "{:?}", metrics.last_error);
    assert_only_weights_and_transfer(&environment);
    let events = environment.control.events();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == "cache-destroyed" && event.artifact == artifact_a())
            .count(),
        1,
        "zero hot retention must destroy the actual native cache",
    );
    assert!(events.iter().all(|event| event.kind != "model-destroyed"));

    let snapshots = entries(directory.path());
    assert_eq!(snapshots.len(), 1);
    let manifest: Value =
        serde_json::from_slice(&fs::read(snapshots[0].join("manifest.json")).unwrap()).unwrap();
    let tokens = manifest["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), rendered.token_count() + POLICY_OUTPUT);
    let blocks = manifest["blocks"].as_array().unwrap();
    assert!(blocks.len() >= 3);
    let config = Qwen3Config::tiny_fixture();
    let mut scalar_count = 0;
    let mut nonzero = false;
    for (index, record) in blocks.iter().enumerate() {
        let bytes = fs::read(snapshots[0].join(format!("{index:08}.safetensors"))).unwrap();
        assert_eq!(bytes.len() as u64, record["bytes"].as_u64().unwrap());
        let tensors = SafeTensors::deserialize(&bytes).expect("actual safetensors KV block");
        assert_eq!(tensors.len(), config.num_layers * 2);
        let block_tokens = record["tokens"].as_u64().unwrap() as usize;
        assert!((1..=32).contains(&block_tokens));
        for layer in 0..config.num_layers {
            for kind in ["keys", "values"] {
                let tensor = tensors.tensor(&format!("layer.{layer}.{kind}")).unwrap();
                assert_eq!(tensor.dtype(), Dtype::F32);
                assert_eq!(
                    tensor.shape(),
                    [1, config.kv_heads, block_tokens, config.head_dim]
                );
                for &value in tensor.data().as_chunks::<4>().0 {
                    let value = f32::from_le_bytes(value);
                    assert!(value.is_finite());
                    nonzero |= value != 0.0;
                    scalar_count += 1;
                }
            }
        }
    }
    assert!(
        nonzero,
        "checkpoint must contain real nonzero native attention state"
    );
    assert_eq!(
        scalar_count,
        tokens.len() * config.num_layers * config.kv_heads * config.head_dim * 2,
    );
    assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".tmp-")
    }));

    let second = environment.run("tiered-save-second", profile_a(), &body);
    let second_work = environment.observations.work("tiered-save-second");
    assert_matches_cold(&second, &second_work, &first, &first_work);
    assert_eq!(
        second_work.prefill_tokens,
        &rendered.token_ids()[rendered.token_count() - 1..]
    );
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&instance));
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        2
    );
    assert_only_weights_and_transfer(&environment);
    drop_runtime_and_assert_unreserved(environment);
}

#[test]
fn tiered_recreated_runtime_restores_partial_block_into_new_owner_and_matches_fresh_logits() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let source = fixtures.environment();
    configure(
        &source,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    let (body, rendered) = prompt_with_partial_last_block(&source);
    let original = source.run("tiered-restart-seed", profile_a(), &body);
    completed_usage(&original, FinishReason::OutputLimit);
    let old_instance = pool_instance(&source, &artifact_a());
    assert_eq!(settled_metrics(&source).spill_count, 1);
    drop_runtime_and_assert_unreserved(source);

    let fresh = fixtures.environment();
    let cold = fresh.run("tiered-restart-oracle", profile_a(), &body);
    let cold_work = fresh.observations.work("tiered-restart-oracle");
    assert_eq!(cold_work.prefill_tokens, rendered.token_ids());
    drop_runtime_and_assert_unreserved(fresh);

    let restored = fixtures.environment();
    let discovered = configure(
        &restored,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    assert_eq!(discovered.cold_entries, 1);
    assert_eq!(discovered.spill_count, 0);
    let actual = restored.run("tiered-restart-restored", profile_a(), &body);
    let work = restored.observations.work("tiered-restart-restored");
    assert_matches_cold(&actual, &work, &cold, &cold_work);
    assert_eq!(
        work.prefill_tokens,
        &rendered.token_ids()[rendered.token_count() - 1..]
    );
    let metrics = settled_metrics(&restored);
    assert_eq!(metrics.restore_count, 1);
    assert_eq!(metrics.restored_tokens, (rendered.token_count() - 1) as u64);
    assert!(metrics.last_error.is_none(), "{:?}", metrics.last_error);
    assert!(!pool_instance(&restored, &artifact_a()).same_instance(&old_instance));
    assert_eq!(restored.observations.loads.load(Ordering::Relaxed), 1);
    assert_eq!(
        restored
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        1
    );
    assert_only_weights_and_transfer(&restored);
    drop_runtime_and_assert_unreserved(restored);
}

#[test]
fn tiered_process_restart_restores_prefix_and_matches_fresh() {
    const PHASE: &str = "ALLOY_TEST_PREFIX_RESTART_PHASE";
    const ROOT: &str = "ALLOY_TEST_PREFIX_RESTART_ROOT";
    if let Ok(phase) = std::env::var(PHASE) {
        let root = std::path::PathBuf::from(std::env::var_os(ROOT).unwrap());
        let fixtures = TieredFixtures::new();
        let environment = fixtures.environment();
        configure(
            &environment,
            configuration(&root.join("cache"), HotCacheLimit::Bytes(0)),
        );
        let (body, rendered) = prompt_with_partial_last_block(&environment);
        let transcript = environment.run("process-restart", profile_a(), &body);
        let work = environment.observations.work("process-restart");
        let metrics = settled_metrics(&environment);
        if phase == "seed" {
            assert_eq!(work.prefill_tokens, rendered.token_ids());
            assert_eq!(metrics.spill_count, 1);
            assert_eq!(metrics.restore_count, 0);
        } else {
            assert_eq!(
                work.prefill_tokens,
                &rendered.token_ids()[rendered.token_count() - 1..]
            );
            assert_eq!(metrics.restore_count, 1);
            assert_eq!(metrics.restored_tokens, (rendered.token_count() - 1) as u64);
        }
        fs::write(
            root.join(format!("{phase}.json")),
            serde_json::to_vec(&serde_json::json!({
                "text": transcript.text,
                "tokens": work.decode_tokens,
                "logits": work.decode_logits,
            }))
            .unwrap(),
        )
        .unwrap();
        drop_runtime_and_assert_unreserved(environment);
        return;
    }

    let directory = TestDirectory::new();
    for phase in ["seed", "restore"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "runtime_policies::tiered::tiered_process_restart_restores_prefix_and_matches_fresh", "--nocapture"])
            .env(PHASE, phase)
            .env(ROOT, directory.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "child {phase} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let read = |phase: &str| -> Value {
        serde_json::from_slice(&fs::read(directory.path().join(format!("{phase}.json"))).unwrap())
            .unwrap()
    };
    let cold = read("seed");
    let restored = read("restore");
    assert_eq!(cold["text"], restored["text"]);
    assert_eq!(cold["tokens"], restored["tokens"]);
    let cold_logits: Vec<Vec<f32>> = serde_json::from_value(cold["logits"].clone()).unwrap();
    let restored_logits: Vec<Vec<f32>> =
        serde_json::from_value(restored["logits"].clone()).unwrap();
    assert!(!cold_logits.is_empty());
    assert_eq!(cold_logits.len(), restored_logits.len());
    for (actual, expected) in restored_logits.iter().zip(&cold_logits) {
        assert_close(actual, expected);
    }
}

proptest::proptest! {
    #![proptest_config(cache_property_config(8))]
    #[test]
    fn tiered_property_restart_edits_and_corruption_match_cold(change in 0u8..4, suffix in "[a-z]{1,3}") {
        let fixtures = TieredFixtures::new();
        let directory = TestDirectory::new();
        let source = fixtures.environment();
        configure(&source, configuration(directory.path(), HotCacheLimit::Bytes(0)));
        let (body, original_rendered) = prompt_with_partial_last_block(&source);
        let seeded = source.run("property-ssd-seed", profile_a(), &body);
        completed_usage(&seeded, FinishReason::OutputLimit);
        assert_eq!(settled_metrics(&source).spill_count, 1);
        drop_runtime_and_assert_unreserved(source);

        if change == 1 {
            let block = entries(directory.path())[0].join("00000001.safetensors");
            let mut bytes = fs::read(&block).unwrap();
            *bytes.last_mut().unwrap() ^= 0x80;
            fs::write(block, bytes).unwrap();
        }
        let input = match change {
            2 => format!("{suffix}{}", &body[1..]),
            3 => body[..body.len() / 2].to_owned(),
            _ => body,
        };
        let cold_env = fixtures.environment();
        let expected = cold_env.run("property-ssd-cold", profile_a(), &input);
        let cold_work = cold_env.observations.work("property-ssd-cold");
        drop_runtime_and_assert_unreserved(cold_env);

        let restored = fixtures.environment();
        configure(&restored, configuration(directory.path(), HotCacheLimit::Bytes(0)));
        let rendered = render(&restored, profile_a(), &input);
        let actual = restored.run("property-ssd-restore", profile_a(), &input);
        let work = restored.observations.work("property-ssd-restore");
        assert_matches_cold(&actual, &work, &expected, &cold_work);
        let metrics = settled_metrics(&restored);
        if change == 0 {
            assert_eq!(metrics.restore_count, 1);
            assert_eq!(work.prefill_tokens, &rendered.token_ids()[rendered.token_count() - 1..]);
        } else if change == 1 {
            assert_eq!(metrics.restore_count, 0);
            assert!(metrics.last_error.is_some());
            assert_eq!(work.prefill_tokens, rendered.token_ids());
        } else {
            let common = original_rendered.token_ids().iter().zip(rendered.token_ids())
                .take_while(|(a, b)| a == b).count().min(rendered.token_count() - 1);
            assert_eq!(metrics.restored_tokens, common as u64);
            assert_eq!(metrics.restore_count, u64::from(common != 0));
            assert_eq!(work.prefill_tokens, &rendered.token_ids()[common..]);
        }
        drop_runtime_and_assert_unreserved(restored);
    }
}

#[test]
fn tiered_artifact_and_profile_identity_mismatch_falls_back_to_full_real_prefill() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let source = fixtures.environment();
    configure(
        &source,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    let original = source.run("tiered-identity-seed", profile_a(), TIERED_BODY);
    completed_usage(&original, FinishReason::OutputLimit);
    assert_eq!(settled_metrics(&source).spill_count, 1);
    drop_runtime_and_assert_unreserved(source);

    let fresh = fixtures.environment();
    let rendered = render(&fresh, profile_b(), TIERED_BODY);
    let cold = fresh.run("tiered-identity-oracle", profile_b(), TIERED_BODY);
    let cold_work = fresh.observations.work("tiered-identity-oracle");
    drop_runtime_and_assert_unreserved(fresh);
    let mismatch = fixtures.environment();
    assert_eq!(
        configure(
            &mismatch,
            configuration(directory.path(), HotCacheLimit::Bytes(0))
        )
        .cold_entries,
        1,
    );
    let actual = mismatch.run("tiered-identity-miss", profile_b(), TIERED_BODY);
    let work = mismatch.observations.work("tiered-identity-miss");
    assert_matches_cold(&actual, &work, &cold, &cold_work);
    assert_eq!(work.prefill_tokens, rendered.token_ids());
    let metrics = settled_metrics(&mismatch);
    assert_eq!(metrics.restore_count, 0);
    assert_eq!(metrics.restored_tokens, 0);
    assert!(
        metrics.last_error.is_none(),
        "identity mismatch is a normal miss"
    );
    assert_only_weights_and_transfer(&mismatch);
    drop_runtime_and_assert_unreserved(mismatch);
}

#[test]
fn tiered_corrupt_block_or_manifest_rolls_back_partial_restore_and_prefills_safely() {
    let fixtures = TieredFixtures::new();
    for corruption in 0..6 {
        let directory = TestDirectory::new();
        let source = fixtures.environment();
        configure(
            &source,
            configuration(directory.path(), HotCacheLimit::Bytes(0)),
        );
        let (body, rendered) = prompt_with_partial_last_block(&source);
        let cold = source.run("tiered-corrupt-seed", profile_a(), &body);
        let cold_work = source.observations.work("tiered-corrupt-seed");
        assert_eq!(cold_work.prefill_tokens, rendered.token_ids());
        assert_eq!(settled_metrics(&source).spill_count, 1);
        drop_runtime_and_assert_unreserved(source);
        let snapshots = entries(directory.path());
        assert_eq!(snapshots.len(), 1);
        // Block and block-hash corruption is found after the first block
        // imports; identity, layout, and prefix changes must instead miss.
        let path = snapshots[0].join("00000001.safetensors");
        if corruption == 0 {
            let length = fs::metadata(&path).unwrap().len();
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(length / 2)
                .unwrap();
        } else if corruption == 1 {
            let mut bytes = fs::read(&path).unwrap();
            *bytes.last_mut().unwrap() ^= 0x80;
            fs::write(&path, bytes).unwrap();
        } else {
            let path = snapshots[0].join("manifest.json");
            let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            match corruption {
                2 => manifest["blocks"][1]["sha256"] = Value::String("0".repeat(64)),
                3 => manifest["identity"] = Value::String("0".repeat(64)),
                4 => {
                    manifest["spec"]["head_dim"] =
                        Value::from(manifest["spec"]["head_dim"].as_u64().unwrap() + 1)
                }
                5 => {
                    manifest["tokens"][0] = Value::from(manifest["tokens"][0].as_u64().unwrap() ^ 1)
                }
                _ => unreachable!(),
            }
            fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        }
        let restored = fixtures.environment();
        configure(
            &restored,
            configuration(directory.path(), HotCacheLimit::Bytes(0)),
        );
        let actual = restored.run("tiered-corrupt-fallback", profile_a(), &body);
        let work = restored.observations.work("tiered-corrupt-fallback");
        assert_matches_cold(&actual, &work, &cold, &cold_work);
        assert_eq!(work.prefill_tokens, rendered.token_ids());
        let first_prefill = restored
            .control
            .events()
            .into_iter()
            .find(|event| event.kind == "prefill-entered")
            .unwrap();
        assert_eq!(first_prefill.logical_before, Some(0));
        let metrics = settled_metrics(&restored);
        assert_eq!(metrics.restore_count, 0);
        assert_eq!(metrics.restored_tokens, 0);
        if corruption < 3 {
            assert!(
                metrics.last_error.is_some(),
                "block corruption must be observable"
            );
        }
        assert_eq!(
            restored
                .observations
                .cache_allocations
                .load(Ordering::Relaxed),
            1
        );
        assert_only_weights_and_transfer(&restored);
        drop_runtime_and_assert_unreserved(restored);
    }
}

#[test]
fn tiered_hot_only_never_creates_ssd_or_reserves_transfer_workspace() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let disabled_path = directory.path().join("must-remain-absent");
    let environment = fixtures.environment();
    let mut config = configuration(&disabled_path, HotCacheLimit::Percent(0));
    config.hot_cache_only = true;
    let initial = configure(&environment, config);
    assert!(initial.enabled);
    assert!(!initial.disk_enabled);
    assert_eq!(initial.transfer_reserved_bytes, 0);
    assert_eq!(environment.snapshot().reserved_bytes(), 0);
    assert!(!disabled_path.exists());
    let rendered = render(&environment, profile_a(), TIERED_BODY);
    let first = environment.run("tiered-hot-only-first", profile_a(), TIERED_BODY);
    let first_work = environment.observations.work("tiered-hot-only-first");
    let second = environment.run("tiered-hot-only-second", profile_a(), TIERED_BODY);
    let second_work = environment.observations.work("tiered-hot-only-second");
    assert_matches_cold(&second, &second_work, &first, &first_work);
    assert_eq!(first_work.prefill_tokens, rendered.token_ids());
    assert_eq!(second_work.prefill_tokens, rendered.token_ids());
    let metrics = settled_metrics(&environment);
    assert_eq!(metrics.cold_entries, 0);
    assert_eq!(metrics.spill_count, 0);
    assert_eq!(metrics.restore_count, 0);
    assert_eq!(metrics.transfer_reserved_bytes, 0);
    assert!(metrics.last_error.is_none());
    assert!(!disabled_path.exists());
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        2
    );
    assert_only_weights_and_transfer(&environment);
    drop_runtime_and_assert_unreserved(environment);
}

#[test]
fn tiered_hot_only_positive_cap_reuses_native_ram_prefix_without_ssd() {
    let directory = TestDirectory::new();
    let disabled_path = directory
        .path()
        .join("positive-hot-only-must-remain-absent");
    let execution = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Qwen3Execution::MlxMetal
    } else {
        Qwen3Execution::MlxCpu
    };
    // The existing prefix fixture uses a pinned append-friendly renderer
    // and the real planner's fixed capacity. It can reproduce all executed
    // tokens, including output, without replacing a growing allocation.
    let warm = TestEnvironment::prefix_with_execution(retained_policy(), execution);
    let runtime = warm.engine.runtime();
    let mut config = configuration(&disabled_path, HotCacheLimit::Bytes(1024 * 1024));
    config.hot_cache_only = true;
    runtime.configure_tiered_kv(config).unwrap();
    assert_eq!(warm.admission.snapshot().unwrap().reserved_bytes(), 0);
    assert!(!disabled_path.exists());

    let completed = completed_context(
        &warm,
        "tiered-positive-hot-seed",
        "tiered-positive-hot-base",
    );
    let retained_ids = warm.render(completed.clone()).token_ids().to_vec();
    let before = block_on(runtime.model_residency()).unwrap();
    assert_eq!(before.len(), 1);
    let owner = before[0].instance.clone();
    let before_metrics = runtime.tiered_kv_snapshot();
    assert!(before_metrics.hot_reserved_bytes > 0);
    assert!(before_metrics.hot_reserved_bytes <= before_metrics.hot_limit_bytes);
    let expected_backend = match execution {
        Qwen3Execution::MlxCpu => engine::types::ExecutionBackend::MlxCpu,
        Qwen3Execution::MlxMetal => engine::types::ExecutionBackend::MlxMetal,
    };
    assert_eq!(
        block_on(warm.engine.capabilities(artifact_id()))
            .unwrap()
            .backend,
        expected_backend,
        "Apple Silicon must execute the real Metal model without fallback",
    );

    let extended = context(
        "tiered-positive-hot-extension",
        profile_id(),
        &format!("{}xy", completed.messages()[0].content),
    );
    let rendered = warm.render(extended.clone());
    assert!(rendered.token_ids().starts_with(&retained_ids));
    let suffix = &rendered.token_ids()[retained_ids.len()..];
    assert!(!suffix.is_empty() && suffix.len() < rendered.token_count());
    let actual = warm.run("tiered-positive-hot-hit", extended.clone());
    let work = warm.work("tiered-positive-hot-hit");
    assert_eq!(work.prefill_tokens, suffix);
    assert_eq!(work.prefill_calls, suffix.len().div_ceil(2));
    assert_eq!(warm.loads(), 1);
    assert_eq!(
        warm.cache_allocations(),
        1,
        "RAM reuse must keep the actual KV allocation"
    );
    let after = block_on(runtime.model_residency()).unwrap();
    assert_eq!(after.len(), 1);
    assert!(after[0].instance.same_instance(&owner));

    let cold = TestEnvironment::prefix_with_execution(discard_policy(), execution);
    let reference = cold.run("tiered-positive-hot-oracle", extended);
    let cold_work = cold.work("tiered-positive-hot-oracle");
    assert_eq!(cold_work.prefill_tokens, rendered.token_ids());
    assert_matches_cold(&actual, &work, &reference, &cold_work);
    cold.wait_unreserved();

    let metrics = runtime.tiered_kv_snapshot();
    assert!(metrics.enabled && !metrics.disk_enabled);
    assert_eq!(metrics.hot_limit_bytes, 1024 * 1024);
    assert_eq!(
        metrics.hot_reserved_bytes,
        before_metrics.hot_reserved_bytes
    );
    assert_eq!(metrics.protected_reserved_bytes, 0);
    assert_eq!(metrics.hot_evictions, 0);
    assert_eq!(metrics.cold_entries, 0);
    assert_eq!(metrics.spill_count, 0);
    assert_eq!(metrics.restore_count, 0);
    assert_eq!(metrics.restored_tokens, 0);
    assert_eq!(metrics.transfer_reserved_bytes, 0);
    assert!(metrics.last_error.is_none());
    assert!(
        !disabled_path.exists(),
        "hot-only execution must never create SSD state"
    );
    let admission = warm.admission.clone();
    drop(warm);
    let limit = deadline();
    loop {
        let accounting = admission.snapshot().unwrap();
        if accounting.reserved_bytes() == 0 {
            assert_eq!(accounting.active_reservations(), 0);
            break;
        }
        assert!(
            Instant::now() < limit,
            "hot-only native ownership did not drain: {accounting:?}"
        );
        thread::yield_now();
    }
}

#[test]
fn tiered_zero_hot_cap_preserves_unrelated_leased_publication_and_both_models_weights() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let environment = fixtures.environment();
    configure(
        &environment,
        configuration(directory.path(), HotCacheLimit::Bytes(TIERED_BUDGET)),
    );
    let (state_id, publication) = fixtures.publish(&environment, "tiered-leased-a");
    let lease = block_on(environment.runtime.lease_session_state(state_id))
        .unwrap()
        .unwrap();
    configure(
        &environment,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    instance_lease_assert_unchanged(&environment, &publication);
    let transcript = environment.run("tiered-unrelated-b", profile_b(), TIERED_BODY);
    completed_usage(&transcript, FinishReason::OutputLimit);
    instance_lease_assert_unchanged(&environment, &publication);
    let metrics = settled_metrics(&environment);
    assert_eq!(metrics.hot_limit_bytes, 0);
    assert_eq!(
        metrics.protected_reserved_bytes,
        publication.cache_reserved_bytes
    );
    assert_eq!(metrics.hot_reserved_bytes, publication.cache_reserved_bytes);
    assert!(metrics.hot_evictions >= 1);
    assert_eq!(metrics.transfer_reserved_bytes, TIERED_TRANSFER);
    let residents = environment.runtime.residency_snapshot().models;
    assert_eq!(residents.len(), 2);
    let a = residents
        .iter()
        .find(|model| model.artifact == artifact_a())
        .unwrap();
    let b = residents
        .iter()
        .find(|model| model.artifact == artifact_b())
        .unwrap();
    assert!(a.leased && a.weights_reserved_bytes > 0);
    assert_eq!(a.kv_reserved_bytes, publication.cache_reserved_bytes);
    assert!(b.weights_reserved_bytes > 0);
    assert_eq!(b.kv_reserved_bytes, 0);
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
    drop(lease);
    assert!(matches!(
        block_on(environment.engine.release_session_state(state_id)).unwrap(),
        engine::types::ReleaseOutcome::Released | engine::types::ReleaseOutcome::AlreadyAbsent,
    ));
    // Zero-hot maintenance may reclaim the newly unleased publication
    // before the explicit release command reaches the worker.
    assert!(
        block_on(environment.runtime.lookup_session_state(state_id))
            .unwrap()
            .is_none()
    );
    assert_only_weights_and_transfer(&environment);
    drop_runtime_and_assert_unreserved(environment);
}

#[test]
fn tiered_startup_spill_keeps_unrelated_published_model_eligible_for_weight_eviction() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let environment = fixtures.environment();
    // This environment intentionally has no startup PressurePolicy. Its
    // ordinary shared-ledger reclamation must be sufficient on its own.
    configure(
        &environment,
        configuration(directory.path(), HotCacheLimit::Bytes(TIERED_BUDGET)),
    );
    let (state_id, publication) = fixtures.publish(&environment, "tiered-startup-published-a");
    assert!(publication.weights_reserved_bytes > 0);
    assert!(publication.cache_reserved_bytes > 0);
    let before = settled_metrics(&environment);
    assert_eq!(before.hot_evictions, 0);
    assert_eq!(before.protected_reserved_bytes, 0);
    assert_eq!(before.spill_count, 1);

    // B fits by itself, including the existing store's transfer allowance
    // and independently retained logits. Even after A's KV/history/logits
    // are released, keeping A's weights would still make B impossible.
    let b_startup = pending_startup_reservation(&environment.plan_b);
    let budget = sum(&[TIERED_TRANSFER, startup_budget(&environment.plan_b)]);
    assert!(environment.snapshot().reserved_bytes() <= budget);
    assert!(
        sum(&[
            TIERED_TRANSFER,
            publication.weights_reserved_bytes,
            b_startup
        ]) > budget,
        "fixture must require actual A weight destruction after the KV spill",
    );
    environment.admission.resize_budget(budget).unwrap();
    assert!(
        block_on(environment.runtime.lookup_session_state(state_id))
            .unwrap()
            .is_some(),
        "hot-limit maintenance must leave the idle publication for startup reclamation",
    );
    let checkpoint = environment.control.events().len();

    let transcript = environment.run("tiered-startup-replacement-b", profile_b(), TIERED_BODY);
    completed_usage(&transcript, FinishReason::OutputLimit);
    let work = environment
        .observations
        .work("tiered-startup-replacement-b");
    assert_eq!(
        work.prefill_tokens,
        render(&environment, profile_b(), TIERED_BODY).token_ids(),
    );
    assert_eq!(work.decode_tokens.len(), POLICY_OUTPUT);
    let events = environment.control.events();
    let events = &events[checkpoint..];
    let cache_drop = event_index(events, "cache-destroyed", &artifact_a());
    let model_drop = event_index(events, "model-destroyed", &artifact_a());
    let load_b = event_index(events, "load-entered", &artifact_b());
    assert!(cache_drop < model_drop && model_drop < load_b);
    assert_eq!(
        events[load_b].snapshot.reserved_bytes(),
        sum(&[TIERED_TRANSFER, b_startup]),
        "B's loader must see only its one startup grant and the storage allowance",
    );
    assert!(
        block_on(environment.runtime.lookup_session_state(state_id))
            .unwrap()
            .is_none(),
    );
    let metrics = settled_metrics(&environment);
    assert_eq!(metrics.hot_evictions, 1);
    assert_eq!(metrics.transfer_reserved_bytes, TIERED_TRANSFER);
    assert!(metrics.last_error.is_none(), "{:?}", metrics.last_error);
    let models = environment.runtime.residency_snapshot().models;
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].artifact, artifact_b());
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
    drop_runtime_and_assert_unreserved(environment);
}

#[test]
fn tiered_pressure_reclaims_both_idle_caches_before_unloading_any_model() {
    use engine::memory_guard::PressureLevel;
    use engine::runtime::IdleModelReclaim;

    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let environment = fixtures.environment();
    configure(
        &environment,
        configuration(directory.path(), HotCacheLimit::Bytes(TIERED_BUDGET)),
    );
    completed_usage(
        &environment.run("tiered-pressure-warm-a", profile_a(), TIERED_BODY),
        FinishReason::OutputLimit,
    );
    let a = pool_instance(&environment, &artifact_a());
    completed_usage(
        &environment.run("tiered-pressure-warm-b", profile_b(), TIERED_BODY),
        FinishReason::OutputLimit,
    );
    let b = pool_instance(&environment, &artifact_b());
    let before = settled_metrics(&environment);
    assert_eq!(before.spill_count, 2);
    assert_eq!(before.hot_evictions, 0);
    assert_eq!(before.protected_reserved_bytes, 0);
    let models = environment.runtime.residency_snapshot().models;
    assert_eq!(models.len(), 2);
    let a_model = models
        .iter()
        .find(|model| model.artifact == artifact_a())
        .unwrap();
    let b_model = models
        .iter()
        .find(|model| model.artifact == artifact_b())
        .unwrap();
    assert!(a_model.kv_reserved_bytes > 0 && b_model.kv_reserved_bytes > 0);
    assert_eq!(
        before.hot_reserved_bytes,
        sum(&[a_model.kv_reserved_bytes, b_model.kv_reserved_bytes]),
    );
    let checkpoint = environment.control.events().len();

    // The observation fixture keeps pressure high even after a reclamation.
    // Every tensor and ownership transition still uses the real Qwen backend.
    let (enforcer, source) = pool_pressure_enforcer(environment.admission.clone(), false);
    let first = pool_pressure_after_grace(&environment, &enforcer);
    assert_eq!(first.tick.status.pressure, PressureLevel::Soft);
    assert!(matches!(first.model, IdleModelReclaim::DeferredForKv));
    assert!(environment.admission.admissions_paused());
    let after_first = settled_metrics(&environment);
    assert_eq!(after_first.hot_evictions, 1);
    assert_eq!(after_first.hot_reserved_bytes, b_model.kv_reserved_bytes);
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| event.kind != "model-destroyed"),
        "the first KV release must not also unload model weights",
    );

    let second = block_on(environment.runtime.enforce_memory_tick(enforcer.clone())).unwrap();
    assert_eq!(second.tick.status.pressure, PressureLevel::Soft);
    assert!(matches!(second.model, IdleModelReclaim::DeferredForKv));
    let after_second = settled_metrics(&environment);
    assert_eq!(after_second.hot_evictions, 2);
    assert_eq!(after_second.hot_reserved_bytes, 0);
    assert_eq!(after_second.protected_reserved_bytes, 0);
    assert_eq!(after_second.transfer_reserved_bytes, TIERED_TRANSFER);
    assert!(
        after_second.last_error.is_none(),
        "{:?}",
        after_second.last_error
    );
    assert!(pool_instance(&environment, &artifact_a()).same_instance(&a));
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    let weights_and_transfer = sum(&[
        TIERED_TRANSFER,
        a_model.weights_reserved_bytes,
        b_model.weights_reserved_bytes,
    ]);
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        weights_and_transfer
    );
    assert!(
        environment.control.events()[checkpoint..]
            .iter()
            .all(|event| event.kind != "model-destroyed"),
        "both eligible caches must be released before any model is unloaded",
    );

    // Only the following fresh, still-pressured tick may reach model LRU.
    let third = block_on(environment.runtime.enforce_memory_tick(enforcer.clone())).unwrap();
    match third.model {
        IdleModelReclaim::Evicted(instance) => assert!(instance.same_instance(&a)),
        other => panic!("expected actual oldest-model eviction after KV exhaustion: {other:?}"),
    }
    assert!(pool_instance(&environment, &artifact_b()).same_instance(&b));
    let events = environment.control.events();
    let events = &events[checkpoint..];
    let cache_a = event_index(events, "cache-destroyed", &artifact_a());
    let cache_b = event_index(events, "cache-destroyed", &artifact_b());
    let model_a = event_index(events, "model-destroyed", &artifact_a());
    assert!(cache_a < cache_b && cache_b < model_a);
    assert_eq!(
        events[model_a].snapshot.reserved_bytes(),
        weights_and_transfer
    );
    assert_eq!(
        environment.snapshot().reserved_bytes(),
        sum(&[TIERED_TRANSFER, b_model.weights_reserved_bytes]),
    );
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 2);
    source.set_normal();
    let recovered = block_on(environment.runtime.enforce_memory_tick(enforcer)).unwrap();
    assert!(matches!(recovered.model, IdleModelReclaim::NotRequested));
    assert!(!environment.admission.admissions_paused());
    drop_runtime_and_assert_unreserved(environment);
}

#[test]
fn tiered_snapshot_quota_failure_preserves_generation_and_releases_kv_reservations() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let environment = fixtures.environment();
    let mut config = configuration(directory.path(), HotCacheLimit::Bytes(0));
    // A positive cap is a valid store configuration but cannot hold even
    // one real snapshot. Optional SSD failure must not fail native output.
    config.max_disk_bytes = 1;
    let initial = configure(&environment, config);
    assert!(initial.disk_enabled);
    assert_eq!(initial.transfer_reserved_bytes, TIERED_TRANSFER);
    let transcript = environment.run("tiered-quota-failure", profile_a(), TIERED_BODY);
    completed_usage(&transcript, FinishReason::OutputLimit);
    let work = environment.observations.work("tiered-quota-failure");
    assert_eq!(
        work.prefill_tokens,
        render(&environment, profile_a(), TIERED_BODY).token_ids(),
    );
    assert_eq!(work.decode_tokens.len(), POLICY_OUTPUT);
    let metrics = settled_metrics(&environment);
    assert_eq!(metrics.spill_count, 0);
    assert_eq!(metrics.restore_count, 0);
    assert_eq!(metrics.cold_entries, 0);
    assert_eq!(metrics.hot_evictions, 1);
    assert!(
        metrics.last_error.as_deref().is_some_and(|error| {
            error.contains("KV snapshot exceeds configured SSD cache cap")
        }),
        "the quota miss must be observable: {:?}",
        metrics.last_error,
    );
    assert!(entries(directory.path()).is_empty());
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        1,
    );
    assert_only_weights_and_transfer(&environment);
    drop_runtime_and_assert_unreserved(environment);
}

#[test]
fn tiered_actual_staged_write_failure_cleans_partial_files_and_preserves_native_generation() {
    let fixtures = TieredFixtures::new();
    let oracle = fixtures.environment();
    let rendered = render(&oracle, profile_a(), TIERED_BODY);
    assert!(
        rendered.token_count() > 64,
        "persistence must write several real blocks"
    );
    let reference = oracle.run("tiered-io-oracle", profile_a(), TIERED_BODY);
    let reference_work = oracle.observations.work("tiered-io-oracle");
    drop_runtime_and_assert_unreserved(oracle);

    // Both failures come from real create_new calls after at least one
    // real safetensors block has been synced. Directory collisions work
    // identically for privileged and unprivileged test processes.
    for collision in ["00000001.safetensors", "manifest.json"] {
        let directory = TestDirectory::new();
        let environment = fixtures.environment();
        configure(
            &environment,
            configuration(directory.path(), HotCacheLimit::Bytes(TIERED_BUDGET)),
        );
        let root = directory.path().to_path_buf();
        let admission = environment.admission.clone();
        let (evidence_tx, evidence_rx) = mpsc::channel();
        *environment.control.kv_export_observer.lock().unwrap() = Some((
            32,
            Box::new(move |cache, block| {
                assert_eq!(block.start, 32);
                assert_eq!(block.tokens, 32);
                block
                    .payload_plan()
                    .expect("second real native block must be valid");
                let staged: Vec<_> = fs::read_dir(&root)
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .filter(|path| {
                        path.file_name()
                            .unwrap()
                            .to_string_lossy()
                            .starts_with(".tmp-kv-v1-")
                    })
                    .collect();
                assert_eq!(
                    staged.len(),
                    1,
                    "the actual writer must already own staging"
                );
                let temporary = staged[0].clone();
                let bytes = fs::read(temporary.join("00000000.safetensors"))
                    .expect("first real block must have been written before the fault");
                let tensors =
                    SafeTensors::deserialize(&bytes).expect("complete first native block");
                assert_eq!(tensors.len(), block.spec.layers * 2);
                let mut nonzero = false;
                for layer in 0..block.spec.layers {
                    for kind in ["keys", "values"] {
                        let tensor = tensors.tensor(&format!("layer.{layer}.{kind}")).unwrap();
                        assert_eq!(tensor.dtype(), Dtype::F32);
                        assert_eq!(
                            tensor.shape(),
                            [
                                block.spec.batch,
                                block.spec.kv_heads,
                                32,
                                block.spec.head_dim
                            ],
                        );
                        for &value in tensor.data().as_chunks::<4>().0 {
                            let value = f32::from_le_bytes(value);
                            assert!(value.is_finite());
                            nonzero |= value != 0.0;
                        }
                    }
                }
                assert!(
                    nonzero,
                    "the staged file must contain actual nonzero Qwen KV"
                );
                let usage = cache
                    .memory_usage()
                    .expect("export must leave native KV healthy");
                assert!(usage.allocated_bytes() > 0 && usage.used_bytes() > 0);
                fs::create_dir(temporary.join(collision))
                    .expect("inject deterministic I/O collision");
                evidence_tx
                    .send((
                        temporary,
                        cache.owner().clone(),
                        cache.len(),
                        usage.allocated_bytes(),
                        admission.snapshot().unwrap(),
                    ))
                    .unwrap();
            }),
        ));

        let actual = environment.run("tiered-io-failure", profile_a(), TIERED_BODY);
        let work = environment.observations.work("tiered-io-failure");
        assert_matches_cold(&actual, &work, &reference, &reference_work);
        let (temporary, owner, logical_tokens, native_bytes, during_export) = evidence_rx
            .try_recv()
            .expect("fault must occur after actual block export and write");
        assert_eq!(logical_tokens, rendered.token_count() + POLICY_OUTPUT);
        assert!(during_export.reserved_bytes() >= TIERED_TRANSFER + native_bytes);
        let metrics = settled_metrics(&environment);
        assert_eq!(metrics.spill_count, 0);
        assert_eq!(metrics.cold_entries, 0);
        assert_eq!(metrics.restore_count, 0);
        assert_eq!(metrics.hot_evictions, 0);
        assert_eq!(metrics.transfer_reserved_bytes, TIERED_TRANSFER);
        assert!(metrics.hot_reserved_bytes >= native_bytes);
        assert!(
            metrics
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("tiered KV storage")),
            "the failed real filesystem write must be reported: {:?}",
            metrics.last_error,
        );
        assert!(
            !temporary.exists(),
            "partial native snapshot and collision must be cleaned"
        );
        assert!(
            entries(directory.path()).is_empty(),
            "failed snapshots must not be published"
        );
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp-")
        }));
        assert!(pool_instance(&environment, &artifact_a()).same_instance(&owner));
        assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
        assert_eq!(
            environment
                .observations
                .cache_allocations
                .load(Ordering::Relaxed),
            1
        );
        assert!(
            environment.control.events().iter().all(|event| {
                event.kind != "cache-destroyed" && event.kind != "model-destroyed"
            }),
            "optional I/O failure must retain healthy native ownership"
        );

        // Reuse the surviving native allocation after the one-shot fault.
        // The shorter prompt requires full logical prefill, but no reload
        // or reallocation, and the next actual filesystem publish succeeds.
        let recovered = environment.run("tiered-io-recovery", profile_a(), TIERED_BODY);
        let recovered_work = environment.observations.work("tiered-io-recovery");
        assert_matches_cold(&recovered, &recovered_work, &reference, &reference_work);
        assert_eq!(recovered_work.prefill_tokens, rendered.token_ids());
        assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
        assert_eq!(
            environment
                .observations
                .cache_allocations
                .load(Ordering::Relaxed),
            1
        );
        assert!(pool_instance(&environment, &artifact_a()).same_instance(&owner));
        assert_eq!(settled_metrics(&environment).spill_count, 1);
        assert_eq!(entries(directory.path()).len(), 1);

        let mut release = configuration(directory.path(), HotCacheLimit::Bytes(0));
        release.hot_cache_only = true;
        let released = configure(&environment, release);
        assert_eq!(released.hot_reserved_bytes, 0);
        assert_eq!(released.transfer_reserved_bytes, 0);
        assert_only_weights_and_transfer(&environment);
        let control = Arc::clone(&environment.control);
        assert_eq!(
            control
                .events()
                .iter()
                .filter(|event| event.kind == "cache-destroyed")
                .count(),
            1
        );
        drop_runtime_and_assert_unreserved(environment);
        assert_eq!(
            control
                .events()
                .iter()
                .filter(|event| event.kind == "model-destroyed")
                .count(),
            1
        );
    }
}

#[test]
fn tiered_cancel_after_native_restore_releases_request_kv_and_drop_releases_transfer_owner() {
    let fixtures = TieredFixtures::new();
    let directory = TestDirectory::new();
    let source = fixtures.environment();
    configure(
        &source,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    let (body, rendered) = prompt_with_partial_last_block(&source);
    let cold = source.run("tiered-cancel-seed", profile_a(), &body);
    let cold_work = source.observations.work("tiered-cancel-seed");
    assert_eq!(settled_metrics(&source).spill_count, 1);
    drop_runtime_and_assert_unreserved(source);

    let environment = fixtures.environment();
    configure(
        &environment,
        configuration(directory.path(), HotCacheLimit::Bytes(0)),
    );
    assert_eq!(environment.snapshot().reserved_bytes(), TIERED_TRANSFER);
    let release = environment.control.execution_gate.arm();
    let generation = block_on(environment.engine.generate(environment.request(
        "tiered-cancel-restored",
        profile_a(),
        &body,
    )))
    .unwrap();
    environment.control.execution_gate.wait_entered();
    let restored_prefix = environment
        .control
        .events()
        .into_iter()
        .find(|event| event.kind == "prefill-entered")
        .unwrap();
    assert_eq!(
        restored_prefix.logical_before,
        Some(rendered.token_count() - 1)
    );
    let blocked = environment.snapshot();
    assert!(blocked.reserved_bytes() > TIERED_TRANSFER);
    assert_eq!(
        block_on(
            environment
                .engine
                .cancel(RequestId::new("tiered-cancel-restored").unwrap())
        )
        .unwrap(),
        CancelOutcome::Requested,
    );
    assert_eq!(environment.snapshot(), blocked);
    assert!(environment.scheduler.is_busy().unwrap());
    release.release();
    let mut stream = generation.into_events();
    let cancelled = drain(&mut stream);
    assert!(matches!(
        cancelled.terminal,
        GenerationEvent::RequestFailed {
            error: EngineError::Cancelled { .. },
            ..
        },
    ));
    drop(stream);
    drop(release);
    let metrics = settled_metrics(&environment);
    assert_eq!(metrics.transfer_reserved_bytes, TIERED_TRANSFER);
    assert_eq!(metrics.restore_count, 1);
    assert_eq!(
        metrics.spill_count, 0,
        "cancelled native work must not publish a snapshot"
    );
    assert_eq!(metrics.cold_entries, 1);
    assert_only_weights_and_transfer(&environment);
    let events = environment.control.events();
    assert!(
        event_index(&events, "prefill-returned", &artifact_a())
            < event_index(&events, "cache-destroyed", &artifact_a()),
        "cancellation must keep native KV alive until execution returns",
    );

    let recovered = environment.run("tiered-cancel-recovery", profile_a(), &body);
    let recovered_work = environment.observations.work("tiered-cancel-recovery");
    assert_matches_cold(&recovered, &recovered_work, &cold, &cold_work);
    assert_eq!(environment.observations.loads.load(Ordering::Relaxed), 1);
    assert_eq!(
        environment
            .observations
            .cache_allocations
            .load(Ordering::Relaxed),
        2
    );
    assert_only_weights_and_transfer(&environment);
    // The admitted transfer allowance belongs to the live store. It stays
    // reserved across cancellation, and releases only after its owner and
    // all native workers are destroyed.
    drop_runtime_and_assert_unreserved(environment);
}
