//! Production boot: real Qwen3 `EngineCore` from a corpus artifact directory.

use crate::rpc::RpcError;
use engine::admission::AdmissionController;
use engine::api::EngineCore;
use engine::backends::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution};
use engine::generation::GenerationLimits;
use engine::registry::ModelProfile;
use engine::runtime::{FamilyRuntime, RetentionPolicy};
use engine::scheduler::Scheduler;
use engine::types::{ArtifactId, GenerationOptions, ModelProfileId};
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
    /// Admission budget bytes configured for this process.
    pub memory_reserved: u64,
}

/// Boot requirements for the Phase-1 production daemon.
pub struct BootRequest {
    /// Local directory containing the mlx-community Qwen3-4B-4bit artifact.
    pub artifact_dir: PathBuf,
    /// Profile id (must be `qwen3-4b-4bit` for Phase 1).
    pub profile_id: String,
    /// Optional path to `bench/corpus/models.json` for hash verification.
    pub corpus_models_json: Option<PathBuf>,
    /// Engine-managed admission budget.
    pub admission_budget_bytes: u64,
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

    let config_bytes = fs::read(request.artifact_dir.join("config.json")).map_err(|error| {
        RpcError::internal(format!("read config.json: {error}"))
    })?;
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
        prefill_chunk_tokens: NonZeroUsize::new(128).expect("chunk"),
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

    let admission = AdmissionController::new(request.admission_budget_bytes)
        .map_err(|error| RpcError::internal(error.to_string()))?;
    let scheduler =
        Scheduler::new(32).map_err(|error| RpcError::internal(error.to_string()))?;
    let engine = backend
        .into_engine(
            vec![profile],
            admission,
            scheduler,
            RetentionPolicy {
                retain_model: true,
                retain_cache_buffers: true,
            },
        )
        .map_err(|error| RpcError::internal(format!("construct engine: {error}")))?;

    Ok(BootedEngine {
        engine,
        profile_id: request.profile_id,
        models_resident: 1,
        memory_reserved: request.admission_budget_bytes,
    })
}

fn pin_artifact_files(
    directory: &Path,
    corpus_models_json: Option<&Path>,
) -> Result<BTreeMap<String, FilePin>, RpcError> {
    let expected = load_expected_hashes(corpus_models_json)?;
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(|error| {
        RpcError::internal(format!("read artifact dir: {error}"))
    })? {
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
        let (byte_len, digest) = hash_file(&path).map_err(|error| {
            RpcError::internal(format!("hash {}: {error}", path.display()))
        })?;
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
    let bytes = fs::read(path).map_err(|error| {
        RpcError::internal(format!("read corpus models.json: {error}"))
    })?;
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
            let hash = file
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| RpcError::internal(format!("artifact file {name} missing sha256")))?;
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
