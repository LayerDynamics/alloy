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
use engine::platform;
use engine::registry::ModelProfile;
use engine::runtime::{FamilyRuntime, RetentionPolicy};
use engine::scheduler::Scheduler;
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
}

impl BootRequest {
    /// Validate Phase-1 profile naming.
    pub fn validate_profile(&self) -> Result<(), RpcError> {
        if self.profile_id != "qwen3-4b-4bit" {
            return Err(RpcError::invalid_params(format!(
                "Phase-1 production profile must be qwen3-4b-4bit (got {})",
                self.profile_id
            )));
        }
        Ok(())
    }
}

/// Construct a real Qwen3 engine from a local corpus artifact directory.
pub fn boot_qwen3(request: BootRequest) -> Result<BootedEngine, RpcError> {
    request.validate_profile()?;
    if !request.artifact_dir.is_dir() {
        return Err(RpcError::internal(format!(
            "artifact directory does not exist: {}",
            request.artifact_dir.display()
        )));
    }

    let files = pin_artifact_files(&request.artifact_dir, request.corpus_models_json.as_deref())?;
    let artifact_id =
        ArtifactId::new("qwen3-4b-4bit").map_err(|error| RpcError::internal(error.to_string()))?;
    let specification = Qwen3Artifact {
        id: artifact_id.clone(),
        directory: request.artifact_dir.clone(),
        files,
        template_sha256: Sha256::digest(CHAT_TEMPLATE.as_bytes()).into(),
    };

    let config_bytes = fs::read(request.artifact_dir.join("config.json"))
        .map_err(|error| RpcError::internal(format!("read config.json: {error}")))?;
    let config: Value = serde_json::from_slice(&config_bytes)
        .map_err(|error| RpcError::internal(format!("parse config.json: {error}")))?;
    let context_limit = config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .unwrap_or(40_960) as usize;

    let limits = Qwen3BackendLimits {
        max_files: 64,
        max_config_bytes: 8 * 1024 * 1024,
        max_tokenizer_bytes: 16 * 1024 * 1024,
        max_messages: 256,
        max_input_bytes: 1024 * 1024,
        max_rendered_bytes: 2 * 1024 * 1024,
        max_token_bytes: 512 * 1024,
        verification_chunk_bytes: 1024 * 1024,
        context_limit_tokens: context_limit,
        // Start at the adaptive floor so cold-start admission does not
        // reserve workspace for a 1024-token chunk before shrinking can run.
        prefill_chunk_tokens: NonZeroUsize::new(256).expect("chunk"),
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

    let backend = Qwen3Backend::new_with_execution(vec![specification], limits, execution)
        .map_err(|error| RpcError::internal(format!("construct Qwen3 backend: {error}")))?;

    let profile = ModelProfile::new(
        ModelProfileId::new(request.profile_id.clone())
            .map_err(|error| RpcError::internal(error.to_string()))?,
        artifact_id,
        context_limit,
        512,
        64,
        GenerationOptions::greedy(512).map_err(|error| RpcError::internal(error.to_string()))?,
    )
    .map_err(|error| RpcError::internal(error.to_string()))?;

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
) -> Result<BTreeMap<String, FilePin>, RpcError> {
    let expected = load_expected_hashes(corpus_models_json)?;
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
    Ok(files)
}

fn load_expected_hashes(
    corpus_models_json: Option<&Path>,
) -> Result<BTreeMap<String, String>, RpcError> {
    let Some(path) = corpus_models_json else {
        return Ok(BTreeMap::new());
    };
    let bytes = fs::read(path)
        .map_err(|error| RpcError::internal(format!("read corpus models.json: {error}")))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| RpcError::internal(format!("parse models.json: {error}")))?;
    let profiles = root
        .get("profiles")
        .and_then(Value::as_array)
        .ok_or_else(|| RpcError::internal("models.json missing profiles[]"))?;
    for profile in profiles {
        if profile.get("id").and_then(Value::as_str) != Some("qwen3-4b-4bit") {
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
