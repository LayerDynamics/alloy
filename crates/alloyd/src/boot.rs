//! Production boot: real Qwen3 `EngineCore` from a corpus artifact directory.

use crate::rpc::RpcError;
use engine::admission::AdmissionController;
use engine::api::EngineCore;
use engine::backends::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution};
use engine::generation::GenerationLimits;
use engine::memory_enforcer::MemoryEnforcer;
use engine::memory_guard::{
    self, CeilingBreakdown, MemoryGuardConfig, should_apply_wired_limit,
    wired_limit_suggestion_bytes,
};
use engine::parser_allocation::ParserAllocationLimits;
use engine::platform;
use engine::registry::ModelProfile;
use engine::runtime::{FamilyRuntime, RetentionPolicy};
use engine::scheduler::Scheduler;
use engine::tiered_kv::TieredKvConfig;
use engine::types::{ArtifactId, GenerationOptions, ModelProfileId};
use mlx_ffi::memory as mlx_memory;
use qwen3::template::CHAT_TEMPLATE;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// Result of booting a real Qwen3 engine for `alloyd`.
pub struct BootedEngine {
    /// Live engine handle.
    pub engine: EngineCore<FamilyRuntime>,
    /// Profile id registered for `session.create`.
    pub profile_id: String,
    /// Models counted as resident after successful construction.
    pub models_resident: u64,
    /// Effective hard ceiling used as the admission ledger budget.
    pub memory_reserved: u64,
    /// Memory-guard tier name (`safe` / `balanced` / `aggressive` / `custom`).
    pub memory_guard_tier: String,
    /// Binding component of the hard ceiling, when known.
    pub ceiling_binding: Option<String>,
    /// Static / dynamic / metal / hard breakdown for status logs.
    pub ceiling: CeilingBreakdown,
    /// Live soft/hard enforcer (clone into the poll thread).
    pub memory_enforcer: MemoryEnforcer,
}

/// Boot requirements for the Phase-1 production daemon.
pub struct BootRequest {
    /// Local directory containing the mlx-community Qwen3-4B-4bit artifact.
    pub artifact_dir: PathBuf,
    /// Profile id (must be `qwen3-4b-4bit` for Phase 1).
    pub profile_id: String,
    /// Optional path to `bench/corpus/models.json` for hash verification.
    pub corpus_models_json: Option<PathBuf>,
    /// oMLX-method memory-guard configuration (default balanced).
    pub memory_guard: MemoryGuardConfig,
    /// Optional absolute admission override for tests. When set, used as the
    /// custom ceiling path instead of live host math.
    pub admission_budget_override_bytes: Option<u64>,
    /// Prefer Metal when the platform supports it.
    pub prefer_metal: bool,
    /// Enable bounded Qwen tool-call parsing only when the daemon has a tools configuration.
    pub tools_enabled: bool,
    /// Bounded hot KV retention and optional persistent SSD snapshots.
    pub tiered_kv: TieredKvConfig,
}

impl BootRequest {
    /// Validate Phase-1 profile naming.
    pub fn validate_profile(&self) -> Result<(), RpcError> {
        if !matches!(
            self.profile_id.as_str(),
            "qwen3-4b-4bit"
                | "llama-3.1-8b-4bit"
                | "llama-3.1-8b-8bit"
                | "ministral-3-3b-instruct-2512-4bit"
                | "ministral-3-3b-instruct-2512-8bit"
                | "smollm3-3b-4bit"
                | "phi4-mini-instruct-4bit"
                | "gemma-3-4b-4bit"
                | "gemma-4-e4b-it-4bit"
                | "gemma-3-12b-4bit"
                | "ornith-1.5-9b-4bit"
                | "qwen3.5-0.8b-4bit"
        ) {
            return Err(RpcError::invalid_params(format!(
                "unsupported native profile {}",
                self.profile_id
            )));
        }
        Ok(())
    }
}

/// Construct a real Qwen3 engine from a local corpus artifact directory.
pub fn boot_qwen3(request: BootRequest) -> Result<BootedEngine, RpcError> {
    boot_native(request)
}

/// Boot the explicitly requested native profile through the shared pipeline.
pub fn boot_native(request: BootRequest) -> Result<BootedEngine, RpcError> {
    boot_qwen3_with_context_limit(request, None)
}

/// Boot with an optional smaller validated context limit for constrained profiles.
pub fn boot_qwen3_with_context_limit(
    request: BootRequest,
    context_override: Option<usize>,
) -> Result<BootedEngine, RpcError> {
    request.validate_profile()?;
    if matches!(
        request.profile_id.as_str(),
        "smollm3-3b-4bit" | "phi4-mini-instruct-4bit"
    ) || is_ministral_profile(&request.profile_id)
        || is_hybrid_profile(&request.profile_id)
        || is_gemma_profile(&request.profile_id)
    {
        return Err(RpcError::invalid_params(format!(
            "{} is not qualified for production: native agent conformance has not passed",
            request.profile_id
        )));
    }
    boot_native_for_qualification(request, context_override)
}

/// Construct an experimental native engine solely for profile qualification.
/// This does not establish support or make a profile available through the CLI.
/// Callers must retain failed observations; production uses `boot_native`.
pub fn boot_native_for_qualification(
    request: BootRequest,
    context_override: Option<usize>,
) -> Result<BootedEngine, RpcError> {
    let (backend, profile) = prepare_native_profile(&request, context_override)?;
    let (budget_bytes, ceiling, guard_config) = resolve_admission_budget(&request)?;
    maybe_apply_wired_limit(&request, &ceiling)?;

    let admission = AdmissionController::new(budget_bytes)
        .map_err(|error| RpcError::internal(error.to_string()))?;
    let memory_enforcer = MemoryEnforcer::new(guard_config, admission.clone());
    let scheduler = Scheduler::new(32).map_err(|error| RpcError::internal(error.to_string()))?;
    let engine = backend
        .into_engine_with_memory_enforcer(
            vec![profile],
            admission,
            scheduler,
            RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: true,
            },
            Some(memory_enforcer.clone()),
        )
        .map_err(|error| RpcError::internal(format!("construct engine: {error}")))?;
    engine
        .runtime()
        .configure_tiered_kv(request.tiered_kv.clone())
        .map_err(|error| RpcError::internal(format!("configure tiered KV: {error}")))?;

    Ok(BootedEngine {
        engine,
        profile_id: request.profile_id,
        models_resident: 1,
        memory_reserved: budget_bytes,
        memory_guard_tier: guard_config.tier.as_str().to_owned(),
        ceiling_binding: ceiling.binding.map(|binding| binding.as_str().to_owned()),
        ceiling,
        memory_enforcer,
    })
}

/// Inspect the same registered profile, tokenizer and templates that daemon boot selects.
/// The trusted runner supplies its independently verified candidate and dependency digest.
/// This does not load weights, execute fixtures, or authorize generation.
pub fn inspect_qualification_identity(
    request: &BootRequest,
    context_override: Option<usize>,
    candidate: &str,
    dependencies_sha256: &str,
) -> Result<(engine::conformance::QualificationIdentity, ModelProfile), RpcError> {
    let (backend, profile) = prepare_native_profile(request, context_override)?;
    let identity = backend
        .qualification_identity(&profile, candidate, dependencies_sha256)
        .map_err(|error| RpcError::invalid_params(error.to_string()))?;
    Ok((identity, profile))
}

fn prepare_native_profile(
    request: &BootRequest,
    context_override: Option<usize>,
) -> Result<(Qwen3Backend, ModelProfile), RpcError> {
    request.validate_profile()?;
    if !request.artifact_dir.is_dir() {
        return Err(RpcError::internal(format!(
            "artifact directory does not exist: {}",
            request.artifact_dir.display()
        )));
    }

    let llama = request.profile_id.starts_with("llama-");
    let ministral = is_ministral_profile(&request.profile_id);
    let smollm = request.profile_id == "smollm3-3b-4bit";
    let phi = request.profile_id == "phi4-mini-instruct-4bit";
    let gemma4 = request.profile_id == "gemma-4-e4b-it-4bit";
    let gemma = is_gemma_profile(&request.profile_id);
    let hybrid = is_hybrid_profile(&request.profile_id);
    let ornith = request.profile_id == "ornith-1.5-9b-4bit";
    let files = pin_artifact_files(
        &request.artifact_dir,
        request.corpus_models_json.as_deref(),
        &request.profile_id,
    )?;
    let artifact_id = ArtifactId::new(request.profile_id.clone())
        .map_err(|error| RpcError::internal(error.to_string()))?;
    let specification = Qwen3Artifact {
        id: artifact_id.clone(),
        directory: request.artifact_dir.clone(),
        files,
        template_sha256: Sha256::digest(
            if ministral {
                engine::backends::qwen::MINISTRAL3_TEXT_TEMPLATE
            } else if gemma4 {
                engine::backends::qwen::GEMMA4_TEXT_TEMPLATE
            } else if gemma {
                engine::backends::qwen::GEMMA3_TEXT_TEMPLATE
            } else if phi {
                engine::backends::qwen::PHI_TEXT_TEMPLATE
            } else if ornith {
                engine::backends::qwen::ORNITH_TEXT_TEMPLATE
            } else if hybrid {
                engine::backends::qwen::QWEN35_TEXT_TEMPLATE
            } else if smollm {
                smollm::template::CHAT_TEMPLATE
            } else if llama {
                llama3::template::CHAT_TEMPLATE
            } else {
                CHAT_TEMPLATE
            }
            .as_bytes(),
        )
        .into(),
    };

    let config_bytes = fs::read(request.artifact_dir.join("config.json"))
        .map_err(|error| RpcError::internal(format!("read config.json: {error}")))?;
    let config: Value = serde_json::from_slice(&config_bytes)
        .map_err(|error| RpcError::internal(format!("parse config.json: {error}")))?;
    let context_limit = if ministral {
        inference_infra::decoder::ministral3::MinistralConfig::from_json(&config)
            .map_err(|e| RpcError::invalid_params(e.to_string()))?
            .context
    } else if gemma4 {
        inference_infra::decoder::gemma4::Gemma4TextConfig::from_json(&config)
            .map_err(|e| RpcError::invalid_params(e.to_string()))?
            .context()
    } else if gemma {
        gemma_context_limit(&config)?
    } else if hybrid {
        hybrid_context_limit(&config)?
    } else {
        config
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .unwrap_or(40_960) as usize
    };

    let profile_context_limit = context_override.unwrap_or(context_limit);
    if profile_context_limit == 0 || profile_context_limit > context_limit {
        return Err(RpcError::invalid_params(
            "profile context limit exceeds model context",
        ));
    }

    let limits = Qwen3BackendLimits {
        max_files: 64,
        max_config_bytes: 8 * 1024 * 1024,
        max_tokenizer_bytes: if gemma {
            40 * 1024 * 1024
        } else if smollm || hybrid || ministral {
            32 * 1024 * 1024
        } else {
            16 * 1024 * 1024
        },
        max_messages: 256,
        max_input_bytes: 1024 * 1024,
        max_rendered_bytes: 2 * 1024 * 1024,
        max_token_bytes: 512 * 1024,
        verification_chunk_bytes: 1024 * 1024,
        context_limit_tokens: context_limit,
        // Tool schemas increase prompt length; bound the actual prefill work
        // at 32 for hybrid/mixed state and 64 for the other native tool profiles.
        // The base Qwen3 text path stays at 256 without relaxing admission.
        prefill_chunk_tokens: NonZeroUsize::new(if hybrid || gemma {
            32
        } else if request.tools_enabled || llama || smollm || phi || ministral {
            64
        } else {
            256
        })
        .expect("chunk"),
        loader_headroom_bytes: 64 * 1024 * 1024,
        execution_headroom_bytes: 64 * 1024 * 1024,
        generation: GenerationLimits::new(64, 64 * 1024, 16 * 1024)
            .map_err(|error| RpcError::internal(error.to_string()))?,
    };

    let execution = if request.prefer_metal {
        Qwen3Execution::MlxMetal
    } else {
        Qwen3Execution::MlxCpu
    };

    let backend = (if ministral {
        Qwen3Backend::new_ministral3(vec![specification], limits, execution)
    } else if gemma4 {
        Qwen3Backend::new_gemma4(vec![specification], limits, execution)
    } else if gemma {
        Qwen3Backend::new_gemma3(vec![specification], limits, execution)
    } else if phi {
        Qwen3Backend::new_phi(vec![specification], limits, execution)
    } else if ornith {
        Qwen3Backend::new_ornith(vec![specification], limits, execution)
    } else if hybrid {
        Qwen3Backend::new_qwen35(vec![specification], limits, execution)
    } else if smollm {
        Qwen3Backend::new_smollm3(vec![specification], limits, execution)
    } else if llama {
        Qwen3Backend::new_llama3(vec![specification], limits, execution)
    } else {
        Qwen3Backend::new_with_execution(vec![specification], limits, execution)
    })
    .map_err(|error| RpcError::internal(format!("construct Qwen3 backend: {error}")))?;
    let backend = if request.tools_enabled {
        // Match the daemon tool loop's 32-proposal and 1 MiB payload bounds.
        // These allocations are included in the engine's admission plan.
        backend
            .with_parser_allocation_limits(ParserAllocationLimits {
                carry_bytes: 64,
                call_bytes: 1024 * 1024,
                decoded_bytes: 1024 * 1024,
                depth: 64,
                keys: 8192,
                nodes: 16384,
                proposal_bytes: 1024 * 1024,
                calls: 32,
            })
            .map_err(|error| RpcError::internal(format!("configure Qwen3 tool parser: {error}")))?
    } else {
        backend
    };

    let profile = ModelProfile::new(
        ModelProfileId::new(request.profile_id.clone())
            .map_err(|error| RpcError::internal(error.to_string()))?,
        artifact_id,
        profile_context_limit,
        512,
        64,
        GenerationOptions::greedy(512).map_err(|error| RpcError::internal(error.to_string()))?,
    )
    .map_err(|error| RpcError::internal(error.to_string()))?;

    Ok((backend, profile))
}

fn resolve_admission_budget(
    request: &BootRequest,
) -> Result<(u64, CeilingBreakdown, MemoryGuardConfig), RpcError> {
    if let Some(override_bytes) = request.admission_budget_override_bytes {
        let config = MemoryGuardConfig::custom(override_bytes)
            .map_err(|error| RpcError::internal(error.to_string()))?;
        let host =
            platform::discover_host_memory_snapshot().unwrap_or(memory_guard::HostMemorySnapshot {
                physical_memory_bytes: override_bytes.saturating_add(2 * 1024 * 1024 * 1024),
                free_bytes: 0,
                inactive_bytes: 0,
                active_file_bytes: 0,
                other_apps_active_bytes: 0,
                phys_footprint_bytes: 0,
                graphics_footprint_bytes: 0,
                iogpu_wired_limit_bytes: 0,
                recommended_working_set_bytes: override_bytes,
                release_credit_bytes: 0,
            });
        let ceiling = memory_guard::ceiling_breakdown(config, host, 0)
            .map_err(|error| RpcError::internal(error.to_string()))?;
        // Test override wins as the ledger budget even when metal binds lower;
        // live guard path (no override) uses hard_limit exclusively.
        return Ok((override_bytes, ceiling, config));
    }

    let host = platform::discover_host_memory_snapshot()
        .map_err(|error| RpcError::internal(format!("discover host memory: {error}")))?;
    let mlx_active = mlx_memory::active_memory_bytes().unwrap_or(0) as u64;
    let alloy_usage = memory_guard::usage_bytes(mlx_active, host);
    let ceiling = memory_guard::ceiling_breakdown(request.memory_guard, host, alloy_usage)
        .map_err(|error| RpcError::internal(error.to_string()))?;
    if ceiling.hard_limit_bytes == 0 {
        return Err(RpcError::internal(
            "memory guard hard ceiling is zero; check --memory-guard / host probes",
        ));
    }
    Ok((ceiling.hard_limit_bytes, ceiling, request.memory_guard))
}

fn maybe_apply_wired_limit(
    request: &BootRequest,
    ceiling: &CeilingBreakdown,
) -> Result<(), RpcError> {
    let Ok(host) = platform::discover_host_memory_snapshot() else {
        return Ok(());
    };
    if !should_apply_wired_limit(host) {
        let suggestion =
            wired_limit_suggestion_bytes(ceiling.static_bytes, host.physical_memory_bytes);
        eprintln!(
            "alloyd: iogpu.wired_limit_mb unset; leaving Apple Metal cap active. \
             To raise: sudo sysctl iogpu.wired_limit_mb={}",
            suggestion / (1024 * 1024)
        );
        return Ok(());
    }
    let desired = wired_limit_suggestion_bytes(ceiling.static_bytes, host.physical_memory_bytes);
    let applied = desired.min(host.iogpu_wired_limit_bytes);
    match mlx_memory::set_wired_limit_bytes(applied as usize) {
        Ok(previous) => {
            eprintln!(
                "alloyd: Metal wired limit {} -> {} (target {}, sysctl cap {})",
                previous, applied, desired, host.iogpu_wired_limit_bytes
            );
        }
        Err(error) => {
            eprintln!("alloyd: mlx_set_wired_limit failed: {error}");
        }
    }
    let _ = request;
    Ok(())
}

fn pin_artifact_files(
    directory: &Path,
    corpus_models_json: Option<&Path>,
    profile_id: &str,
) -> Result<BTreeMap<String, FilePin>, RpcError> {
    let expected = load_expected_hashes(corpus_models_json, profile_id)?;
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| RpcError::internal(format!("read artifact dir: {error}")))?
    {
        let entry = entry.map_err(|error| RpcError::internal(error.to_string()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        // Skip non-model noise; still pin everything the backend needs.
        if name.starts_with('.') || name == "README.md" {
            continue;
        }
        let (byte_len, digest) = hash_file(&path)
            .map_err(|error| RpcError::internal(format!("hash {}: {error}", path.display())))?;
        if is_frozen_profile(profile_id) && !expected.contains_key(&name) {
            return Err(RpcError::internal(format!(
                "unlisted frozen profile input: {name}"
            )));
        }
        if let Some(expected_hex) = expected.get(&name) {
            let actual_hex = hex_encode(&digest);
            if &actual_hex != expected_hex {
                return Err(RpcError::internal(format!(
                    "artifact file {name} sha256 mismatch (corpus pin {expected_hex}, disk {actual_hex})"
                )));
            }
        }
        files.insert(
            name,
            FilePin {
                bytes: byte_len,
                sha256: digest,
            },
        );
    }
    if !files.contains_key("config.json")
        || !files.contains_key("tokenizer.json")
        || !files.keys().any(|name| name.ends_with(".safetensors"))
    {
        return Err(RpcError::internal(
            "artifact directory missing config.json, tokenizer.json, or safetensors weights",
        ));
    }
    if is_frozen_profile(profile_id) {
        for name in expected
            .keys()
            .filter(|n| !n.starts_with('.') && n.as_str() != "README.md")
        {
            if !files.contains_key(name) {
                return Err(RpcError::internal(format!(
                    "missing frozen profile input: {name}"
                )));
            }
        }
    }
    Ok(files)
}

fn is_hybrid_profile(profile: &str) -> bool {
    matches!(profile, "ornith-1.5-9b-4bit" | "qwen3.5-0.8b-4bit")
}

fn is_ministral_profile(profile: &str) -> bool {
    matches!(
        profile,
        "ministral-3-3b-instruct-2512-4bit" | "ministral-3-3b-instruct-2512-8bit"
    )
}

fn is_gemma_profile(profile: &str) -> bool {
    matches!(
        profile,
        "gemma-3-4b-4bit" | "gemma-3-12b-4bit" | "gemma-4-e4b-it-4bit"
    )
}

fn is_frozen_profile(profile: &str) -> bool {
    profile.starts_with("llama-")
        || matches!(profile, "smollm3-3b-4bit" | "phi4-mini-instruct-4bit")
        || is_ministral_profile(profile)
        || is_hybrid_profile(profile)
        || is_gemma_profile(profile)
}

fn gemma_context_limit(config: &Value) -> Result<usize, RpcError> {
    if let Some(text) = config.get("text_config")
        && let Some(outer) = config.get("max_position_embeddings")
        && text.get("max_position_embeddings") != Some(outer)
    {
        return Err(RpcError::invalid_params(
            "conflicting outer and text context limits",
        ));
    }
    inference_infra::decoder::gemma3::Gemma3Config::from_json(config)
        .map(|config| config.context)
        .map_err(|error| RpcError::invalid_params(format!("Gemma3 configuration: {error}")))
}

fn hybrid_context_limit(config: &Value) -> Result<usize, RpcError> {
    let text = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| RpcError::invalid_params("hybrid text_config must be an object"))?;
    if let Some(outer) = config.get("max_position_embeddings")
        && text.get("max_position_embeddings") != Some(outer)
    {
        return Err(RpcError::invalid_params(
            "conflicting outer and text context limits",
        ));
    }
    text.get("max_position_embeddings")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            RpcError::invalid_params(
                "hybrid max_position_embeddings must be a positive supported integer",
            )
        })
}

fn load_expected_hashes(
    corpus_models_json: Option<&Path>,
    profile_id: &str,
) -> Result<BTreeMap<String, String>, RpcError> {
    let frozen: Option<&[u8]> = match profile_id {
        "ministral-3-3b-instruct-2512-8bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/ministral3_8bit_profile.json"
        )),
        "ministral-3-3b-instruct-2512-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/ministral3_profile.json"
        )),
        "gemma-4-e4b-it-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/gemma4_e4b_profile.json"
        )),
        "gemma-3-4b-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/gemma3_4b_profile.json"
        )),
        "gemma-3-12b-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/gemma3_12b_profile.json"
        )),
        "phi4-mini-instruct-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/phi4_mini_profile.json"
        )),
        "smollm3-3b-4bit" => Some(include_bytes!("../../families/smollm/fixture/profile.json")),
        "ornith-1.5-9b-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/ornith_1_5_9b_profile.json"
        )),
        "qwen3.5-0.8b-4bit" => Some(include_bytes!(
            "../../inference-infra/tests/fixtures/qwen35_0_8b_profile.json"
        )),
        _ => None,
    };
    if let Some(bytes) = frozen {
        let profile: Value = serde_json::from_slice(bytes)
            .map_err(|e| RpcError::internal(format!("parse frozen profile: {e}")))?;
        return profile["artifact"]["files"]
            .as_array()
            .ok_or_else(|| RpcError::internal("frozen profile missing files"))?
            .iter()
            .map(|entry| {
                Ok((
                    entry["path"]
                        .as_str()
                        .ok_or_else(|| RpcError::internal("missing file path"))?
                        .to_owned(),
                    entry["sha256"]
                        .as_str()
                        .ok_or_else(|| RpcError::internal("missing file hash"))?
                        .to_owned(),
                ))
            })
            .collect();
    }
    // Llama runtime inputs must match the immutable corpus compiled into this
    // source revision, even when an external corpus path is supplied.
    let bytes = if profile_id.starts_with("llama-") {
        include_bytes!("../../../bench/corpus/models.json").to_vec()
    } else if let Some(path) = corpus_models_json {
        fs::read(path)
            .map_err(|error| RpcError::internal(format!("read corpus models.json: {error}")))?
    } else {
        return Ok(BTreeMap::new());
    };
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| RpcError::internal(format!("parse models.json: {error}")))?;
    let profiles = root
        .get("profiles")
        .and_then(Value::as_array)
        .ok_or_else(|| RpcError::internal("models.json missing profiles[]"))?;
    for profile in profiles {
        if profile.get("id").and_then(Value::as_str) != Some(profile_id) {
            continue;
        }
        let mut map = BTreeMap::new();
        let files = profile
            .pointer("/artifact/files")
            .and_then(Value::as_array)
            .ok_or_else(|| RpcError::internal("qwen3-4b-4bit missing artifact.files"))?;
        for file in files {
            let name = file
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| RpcError::internal("artifact file missing path"))?;
            let hash = file.get("sha256").and_then(Value::as_str).ok_or_else(|| {
                RpcError::internal(format!("artifact file {name} missing sha256"))
            })?;
            // Only leaf filenames are pinned by Qwen3Artifact.
            let leaf = Path::new(name)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(name);
            map.insert(leaf.to_owned(), hash.to_owned());
        }
        return Ok(map);
    }
    Err(RpcError::internal(
        "qwen3-4b-4bit profile not found in corpus models.json",
    ))
}

/// Stream-hash a file without retaining its full contents in memory.
fn hash_file(path: &Path) -> Result<(u64, [u8; 32]), String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(|error| error.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total = total.saturating_add(n as u64);
    }
    Ok((total, hasher.finalize().into()))
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod llama_pins {
    use super::*;
    #[test]
    fn llama_cannot_replace_frozen_hashes_with_external_manifest() {
        let bogus = Path::new("/nonexistent/untrusted-corpus.json");
        for profile in ["llama-3.1-8b-4bit", "llama-3.1-8b-8bit"] {
            let pins = load_expected_hashes(Some(bogus), profile).unwrap();
            assert!(!pins.is_empty());
            assert_eq!(pins, load_expected_hashes(None, profile).unwrap());
        }
        let four = load_expected_hashes(None, "llama-3.1-8b-4bit").unwrap();
        let eight = load_expected_hashes(None, "llama-3.1-8b-8bit").unwrap();
        assert_ne!(four["config.json"], eight["config.json"]);
        assert!(load_expected_hashes(None, "llama-unsupported").is_err());
    }
    #[test]
    fn llama_missing_and_modified_inputs_fail_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        assert!(pin_artifact_files(dir.path(), None, "llama-3.1-8b-4bit").is_err());
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert!(pin_artifact_files(dir.path(), None, "llama-3.1-8b-4bit").is_err());
    }
}

#[cfg(test)]
mod qualification_boot_tests {
    use super::*;
    #[test]
    fn gemma_profiles_ignore_external_pin_overrides() {
        for profile in ["gemma-3-4b-4bit", "gemma-3-12b-4bit"] {
            let pins =
                load_expected_hashes(Some(Path::new("/nonexistent/untrusted.json")), profile)
                    .unwrap();
            assert_eq!(pins, load_expected_hashes(None, profile).unwrap());
            assert_eq!(
                pins["tokenizer.json"],
                "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
            );
            let dir = tempfile::tempdir().unwrap();
            assert!(pin_artifact_files(dir.path(), None, profile).is_err());
            fs::write(dir.path().join("config.json"), b"{}").unwrap();
            assert!(pin_artifact_files(dir.path(), None, profile).is_err());
        }
    }

    #[test]
    fn unqualified_gemma_cannot_boot_production_or_override_context() {
        for profile in ["gemma-3-4b-4bit", "gemma-3-12b-4bit"] {
            assert_unqualified_production_refused(profile);
        }
    }

    #[test]
    fn gemma_context_uses_validated_native_metadata() {
        use serde_json::json;
        let mut config: Value = serde_json::from_slice(include_bytes!(
            "../../inference-infra/tests/fixtures/gemma3_model/config.json"
        ))
        .unwrap();
        config
            .as_object_mut()
            .unwrap()
            .remove("max_position_embeddings");
        assert_eq!(gemma_context_limit(&config).unwrap(), 32768);
        for invalid in [json!(0), json!(-1), json!(1.5), json!("1024")] {
            config["max_position_embeddings"] = invalid;
            assert!(gemma_context_limit(&config).is_err());
        }
        config["max_position_embeddings"] = json!(1024);
        assert_eq!(gemma_context_limit(&config).unwrap(), 1024);
        let mut wrapped = json!({"model_type":"gemma3", "text_config": config});
        assert_eq!(gemma_context_limit(&wrapped).unwrap(), 1024);
        wrapped["max_position_embeddings"] = json!(2048);
        assert!(gemma_context_limit(&wrapped).is_err());
        wrapped["max_position_embeddings"] = json!(1024);
        assert_eq!(gemma_context_limit(&wrapped).unwrap(), 1024);
    }

    #[test]
    fn phi_profile_ignores_external_pin_overrides() {
        let profile = "phi4-mini-instruct-4bit";
        let pins =
            load_expected_hashes(Some(Path::new("/nonexistent/untrusted.json")), profile).unwrap();
        assert_eq!(pins, load_expected_hashes(None, profile).unwrap());
        assert_eq!(pins.len(), 6);
        assert_eq!(
            pins["model.safetensors"],
            "9dcfcdc0a579494283f2d2317ad686123782ce461d1111690ede7c904ff3f87b"
        );
        let dir = tempfile::tempdir().unwrap();
        assert!(pin_artifact_files(dir.path(), None, profile).is_err());
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert!(pin_artifact_files(dir.path(), None, profile).is_err());
    }

    #[test]
    fn unqualified_phi_cannot_boot_production_or_override_context() {
        assert_unqualified_production_refused("phi4-mini-instruct-4bit");
    }

    #[test]
    fn hybrid_profiles_ignore_external_pin_overrides() {
        for profile in ["ornith-1.5-9b-4bit", "qwen3.5-0.8b-4bit"] {
            let pins =
                load_expected_hashes(Some(Path::new("/nonexistent/untrusted.json")), profile)
                    .unwrap();
            assert_eq!(pins, load_expected_hashes(None, profile).unwrap());
            assert!(pins.contains_key("chat_template.jinja"));
            assert!(pins.contains_key("model.safetensors"));
            let dir = tempfile::tempdir().unwrap();
            assert!(pin_artifact_files(dir.path(), None, profile).is_err());
            fs::write(dir.path().join("config.json"), b"{}").unwrap();
            assert!(pin_artifact_files(dir.path(), None, profile).is_err());
        }
    }

    #[test]
    fn hybrid_context_requires_positive_consistent_metadata() {
        use serde_json::json;
        assert_eq!(
            hybrid_context_limit(&json!({"text_config":{"max_position_embeddings":262144}}))
                .unwrap(),
            262144
        );
        assert_eq!(hybrid_context_limit(&json!({"max_position_embeddings":1024,"text_config":{"max_position_embeddings":1024}})).unwrap(), 1024);
        for invalid in [
            json!({}),
            json!({"max_position_embeddings":40960}),
            json!({"text_config":null}),
            json!({"text_config":{}}),
            json!({"text_config":{"max_position_embeddings":0}}),
            json!({"text_config":{"max_position_embeddings":-1}}),
            json!({"text_config":{"max_position_embeddings":1.5}}),
            json!({"text_config":{"max_position_embeddings":"1024"}}),
            json!({"max_position_embeddings":1024,"text_config":{"max_position_embeddings":2048}}),
        ] {
            assert!(hybrid_context_limit(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn unqualified_hybrids_cannot_boot_production() {
        for profile in ["ornith-1.5-9b-4bit", "qwen3.5-0.8b-4bit"] {
            assert_unqualified_production_refused(profile);
        }
    }

    #[test]
    fn unqualified_smollm_cannot_boot_production_or_override_context() {
        assert_unqualified_production_refused("smollm3-3b-4bit");
    }

    fn assert_unqualified_production_refused(profile: &str) {
        for tools_enabled in [false, true] {
            for mode in 0..3 {
                let request = BootRequest {
                    artifact_dir: PathBuf::from("/nonexistent/not-loaded"),
                    profile_id: profile.into(),
                    corpus_models_json: None,
                    memory_guard: MemoryGuardConfig::balanced(),
                    admission_budget_override_bytes: None,
                    prefer_metal: true,
                    tools_enabled,
                    tiered_kv: TieredKvConfig::new(
                        None,
                        engine::tiered_kv::HotCacheLimit::Bytes(0),
                    ),
                };
                let result = match mode {
                    0 => boot_native(request),
                    1 => boot_qwen3(request),
                    _ => boot_qwen3_with_context_limit(request, Some(2048)),
                };
                let error = result.err().expect("unqualified profile must be refused");
                assert!(
                    error.message.contains("not qualified for production"),
                    "{error:?}"
                );
            }
        }
    }
}
