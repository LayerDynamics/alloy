//! Opt-in M9 acceptance against the two distinct frozen Qwen3 checkpoints.
//!
//! Run through `scripts/m9_large_model_acceptance.sh` on Apple Silicon. The
//! observer only delegates production calls and records ownership boundaries;
//! it does not provide model output, native buffers, memory observations, or
//! admission decisions. `--validate-manifest` and unit tests are structural
//! checks and can never produce an acceptance success record.

use engine::{
    admission::AdmissionController,
    api::{Engine, EngineCore, EngineFuture},
    backends::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution},
    error::{EngineError, ErrorKind},
    family::{
        AutoregressiveModel, BufferMemoryUsage, CacheAllocationRequest, ExecutionControl,
        FamilyCache, LoadedModel, ModelFamily, ModelInstanceId, ModelLoadRequest, PrefillOutput,
        TokenLogits,
    },
    generation::GenerationLimits,
    memory_enforcer::MemoryEnforcer,
    memory_guard::{self, HostMemorySnapshot, MemoryGuardConfig},
    platform,
    pressure::{PressureOptions, PressurePolicy},
    registry::{ArtifactRegistration, FamilyRegistration, ModelProfile, Registry},
    runtime::{
        FamilyRuntime, MemoryEnforcementPump, RetentionPolicy, RuntimeBackend, RuntimePlan,
        RuntimeResidency, TextDecoder,
    },
    scheduler::Scheduler,
    types::{
        ArtifactId, Capabilities, ContextSnapshot, ContextSnapshotId, ExecutionBackend,
        GenerationEvent, GenerationOptions, GenerationRequest, Message, MessageRole,
        ModelProfileId, RenderedContext, RequestId,
    },
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    future::{Future, poll_fn},
    io::{self, Read, Write},
    num::NonZeroUsize,
    path::{Component, Path, PathBuf},
    pin::pin,
    process::{self, Command},
    sync::{Arc, Mutex, OnceLock},
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const CORPUS: &str = include_str!("../../../bench/corpus/models.json");
const CORPUS_SHA256: &str = "dc2eac48c5522fa53360855950eba4527dc6c21dc943f1d1c9282bd8535b2534";
const MODEL_A: &str = "qwen3-4b-4bit";
const MODEL_B: &str = "qwen3-14b-4bit";
const A_SHA256: &str = "95d369e2b999429fd8f02efdd9c0916718e653cb0a62d58e7f6a2770fee447a5";
const B_SHA256: &str = "166bc507b0d8185c0f1aedf41eef136a6d8c7757387a59ff9adb5e7ecdf467d9";
const MIB: u64 = 1024 * 1024;
const OUTPUT_TOKENS: usize = 8;
const PROMPT: &str = "Count from 1 to 8, separated by commas, with no introduction.";
const BUILD_COMMIT: Option<&str> = option_env!("ALLOY_M9_BUILD_COMMIT");
type AcceptanceResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn require(condition: bool, message: impl Into<String>) -> AcceptanceResult<()> {
    if condition {
        Ok(())
    } else {
        Err(io::Error::other(message.into()).into())
    }
}

fn engine_failure(message: impl Into<String>) -> EngineError {
    EngineError::Internal {
        message: message.into(),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(text: &str) -> AcceptanceResult<[u8; 32]> {
    require(
        text.len() == 64
            && text
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "SHA-256 must be 64 lowercase hexadecimal characters",
    )?;
    let mut bytes = [0; 32];
    for (i, chunk) in text.as_bytes().chunks_exact(2).enumerate() {
        bytes[i] = u8::from_str_radix(std::str::from_utf8(chunk)?, 16)?;
    }
    Ok(bytes)
}

#[derive(Clone, Debug)]
struct FrozenProfile {
    id: String,
    artifact_repository: String,
    artifact_revision: String,
    source_repository: String,
    source_revision: String,
    artifact_sha256: String,
    files: BTreeMap<String, [u8; 32]>,
}

fn text_at<'a>(value: &'a Value, key: &str) -> AcceptanceResult<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other(format!("missing string {key}")).into())
}

fn aggregate(files: &BTreeMap<String, [u8; 32]>) -> [u8; 32] {
    let mut hash = Sha256::new();
    for (path, sha) in files {
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update(hex(sha).as_bytes());
        hash.update([0]);
    }
    hash.finalize().into()
}

fn frozen_profiles() -> AcceptanceResult<[FrozenProfile; 2]> {
    require(
        hex(&Sha256::digest(CORPUS.as_bytes())) == CORPUS_SHA256,
        "embedded frozen corpus integrity changed",
    )?;
    let corpus: Value = serde_json::from_str(CORPUS)?;
    require(
        corpus["schema_version"] == 1
            && corpus["status"] == "frozen"
            && corpus["corpus_id"] == "alloy-phase-0",
        "unexpected corpus identity",
    )?;
    let profiles = corpus["profiles"]
        .as_array()
        .ok_or_else(|| io::Error::other("missing frozen profiles"))?;
    let parse = |id: &str, expected: &str| -> AcceptanceResult<FrozenProfile> {
        let matches: Vec<_> = profiles.iter().filter(|p| p["id"] == id).collect();
        require(
            matches.len() == 1,
            format!("expected one frozen profile {id}"),
        )?;
        let p = matches[0];
        require(
            p["family"] == "qwen3"
                && p["quantization"]["nominal_bits"] == 4
                && p["quantization"]["mode"] == "affine"
                && p["quantization"]["group_size"] == 64,
            "unexpected Qwen3 quantization identity",
        )?;
        let artifact = &p["artifact"];
        require(
            artifact["format"] == "mlx-safetensors" && artifact["sha256"] == expected,
            "unexpected pinned artifact identity",
        )?;
        let mut files = BTreeMap::new();
        for f in artifact["files"]
            .as_array()
            .ok_or_else(|| io::Error::other("missing pinned files"))?
        {
            let name = text_at(f, "path")?;
            require(
                !name.is_empty()
                    && Path::new(name)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_))),
                "unsafe artifact-relative path",
            )?;
            require(
                files
                    .insert(name.to_owned(), digest(text_at(f, "sha256")?)?)
                    .is_none(),
                "duplicate artifact path",
            )?;
        }
        require(
            !files.is_empty() && hex(&aggregate(&files)) == expected,
            format!("frozen aggregate mismatch for {id}"),
        )?;
        Ok(FrozenProfile {
            id: id.into(),
            artifact_repository: text_at(artifact, "repository")?.into(),
            artifact_revision: text_at(artifact, "revision")?.into(),
            source_repository: text_at(&p["source"], "repository")?.into(),
            source_revision: text_at(&p["source"], "revision")?.into(),
            artifact_sha256: expected.into(),
            files,
        })
    };
    let a = parse(MODEL_A, A_SHA256)?;
    let b = parse(MODEL_B, B_SHA256)?;
    require(
        a.source_repository != b.source_repository
            && a.source_revision != b.source_revision
            && a.artifact_repository != b.artifact_repository
            && a.artifact_sha256 != b.artifact_sha256,
        "acceptance requires two distinct checkpoints",
    )?;
    Ok([a, b])
}

#[derive(Clone, Debug, Default)]
struct Observations {
    live: BTreeSet<String>,
    loader_entries: BTreeMap<String, Vec<u64>>,
    destroyed: BTreeMap<String, Vec<u64>>,
    prefill_calls: BTreeMap<String, usize>,
    decode_calls: BTreeMap<String, usize>,
    maximum_live_models: usize,
}

struct Journal {
    output: File,
    sequence: u64,
    error: Option<String>,
}

struct Evidence {
    journal: Mutex<Journal>,
    observations: Mutex<Observations>,
    admission: OnceLock<AdmissionController>,
    started: Instant,
    timeout: Duration,
}

impl Evidence {
    fn new(path: &Path, timeout: Duration) -> AcceptanceResult<Self> {
        Ok(Self {
            journal: Mutex::new(Journal {
                output: OpenOptions::new().write(true).create_new(true).open(path)?,
                sequence: 0,
                error: None,
            }),
            observations: Mutex::new(Observations::default()),
            admission: OnceLock::new(),
            started: Instant::now(),
            timeout,
        })
    }

    fn observations(&self) -> AcceptanceResult<Observations> {
        self.observations
            .lock()
            .map(|s| s.clone())
            .map_err(|_| io::Error::other("observation lock poisoned").into())
    }

    fn ledger(&self) -> AcceptanceResult<Value> {
        match self.admission.get() {
            None => Ok(Value::Null),
            Some(admission) => {
                let s = admission.snapshot()?;
                Ok(
                    json!({"budget_bytes": s.budget_bytes(), "reserved_bytes": s.reserved_bytes(), "available_bytes": s.available_bytes(), "active_reservations": s.active_reservations()}),
                )
            }
        }
    }

    fn emit(&self, event: &str, data: Value) -> AcceptanceResult<u64> {
        let ledger = self.ledger()?;
        let observations = self.observations()?;
        let mut journal = self
            .journal
            .lock()
            .map_err(|_| io::Error::other("journal lock poisoned"))?;
        require(
            journal.error.is_none(),
            format!("earlier evidence write failed: {:?}", journal.error),
        )?;
        journal.sequence += 1;
        let sequence = journal.sequence;
        let record = json!({
            "schema_version": 1, "acceptance": "m9-real-large-model", "sequence": sequence,
            "elapsed_ms": self.started.elapsed().as_millis(),
            "unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            "event": event, "data": data, "ledger": ledger,
            "native_model_handle_count": observations.live.len(),
            "native_model_artifacts": observations.live,
        });
        let result = (|| -> io::Result<()> {
            serde_json::to_writer(&mut journal.output, &record)?;
            journal.output.write_all(b"\n")?;
            journal.output.flush()
        })();
        if let Err(error) = result {
            journal.error = Some(error.to_string());
            return Err(error.into());
        }
        Ok(sequence)
    }

    fn check_deadline(&self) -> AcceptanceResult<()> {
        require(
            self.started.elapsed() < self.timeout,
            "acceptance deadline exceeded before native execution",
        )
    }
}

fn command(program: &str, args: &[&str]) -> AcceptanceResult<String> {
    let result = Command::new(program).args(args).output()?;
    require(
        result.status.success(),
        format!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&result.stderr)
        ),
    )?;
    Ok(String::from_utf8(result.stdout)?.trim().to_owned())
}

fn provenance(root: &Path, expected: &str) -> AcceptanceResult<Value> {
    require(
        expected.len() == 40 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
        "expected commit must be a full Git SHA",
    )?;
    require(
        BUILD_COMMIT == Some(expected),
        "binary was not built with ALLOY_M9_BUILD_COMMIT matching this source; use the launch script",
    )?;
    let compiled_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    require(
        root == compiled_root,
        "runtime checkout differs from the checkout that built this driver",
    )?;
    let root_text = root
        .to_str()
        .ok_or_else(|| io::Error::other("source root is not UTF-8"))?;
    let commit = command("git", &["-C", root_text, "rev-parse", "HEAD"])?;
    require(
        commit == expected,
        "source HEAD changed from the accepted commit",
    )?;
    let dirty = command(
        "git",
        &[
            "-C",
            root_text,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        ],
    )?;
    require(
        dirty.is_empty(),
        format!("source checkout is not clean: {dirty}"),
    )?;
    let current_manifest = fs::read(root.join("bench/corpus/models.json"))?;
    require(
        hex(&Sha256::digest(&current_manifest)) == CORPUS_SHA256,
        "runtime frozen manifest differs from the embedded manifest",
    )?;
    Ok(json!({
        "commit_sha": commit, "build_commit_sha": BUILD_COMMIT, "source_root": root,
        "source_tree_clean": true, "corpus_path": "bench/corpus/models.json",
        "corpus_sha256": CORPUS_SHA256,
        "driver_sha256": hex(&Sha256::digest(fs::read(root.join("crates/engine/examples/m9_large_model_acceptance.rs"))?)),
        "script_sha256": hex(&Sha256::digest(fs::read(root.join("scripts/m9_large_model_acceptance.sh"))?)),
        "rustc": command("rustc", &["--version"] )?,
    }))
}

fn verify_profile(
    profile: &FrozenProfile,
    directory: &Path,
    evidence: &Evidence,
) -> AcceptanceResult<Qwen3Artifact> {
    let directory = directory.canonicalize()?;
    require(
        directory.is_dir(),
        format!("{} is not a model directory", directory.display()),
    )?;
    evidence.emit("artifact_verification_started", json!({"artifact_id": profile.id, "directory": directory, "expected_aggregate_sha256": profile.artifact_sha256, "file_count": profile.files.len()}))?;
    let mut pins = BTreeMap::new();
    let mut measured_hashes = BTreeMap::new();
    let mut total_bytes = 0u64;
    let mut buffer = vec![0u8; MIB as usize];
    for (name, expected) in &profile.files {
        evidence.check_deadline()?;
        let path = directory.join(name);
        // HF snapshot directories may use symlinks into their immutable blob
        // store; bytes, rather than local directory spelling, establish identity.
        let mut file = File::open(&path)?;
        let metadata = file.metadata()?;
        require(
            metadata.is_file(),
            format!("{} is not a regular file", path.display()),
        )?;
        let mut hash = Sha256::new();
        let mut bytes = 0u64;
        loop {
            evidence.check_deadline()?;
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            bytes = bytes
                .checked_add(count as u64)
                .ok_or_else(|| io::Error::other("artifact size overflow"))?;
        }
        let actual: [u8; 32] = hash.finalize().into();
        require(
            actual == *expected && bytes == metadata.len(),
            format!("pinned hash/size verification failed: {}", path.display()),
        )?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| io::Error::other("artifact total overflow"))?;
        measured_hashes.insert(name.clone(), actual);
        pins.insert(
            name.clone(),
            FilePin {
                bytes,
                sha256: actual,
            },
        );
        evidence.emit("artifact_file_verified", json!({"artifact_id": profile.id, "path": name, "bytes": bytes, "sha256": hex(&actual), "matches_frozen_pin": true}))?;
    }
    let actual_aggregate = hex(&aggregate(&measured_hashes));
    require(
        actual_aggregate == profile.artifact_sha256,
        "verified artifact aggregate mismatch",
    )?;
    evidence.emit("artifact_verified", json!({
        "artifact_id": profile.id, "directory": directory, "artifact_repository": profile.artifact_repository,
        "artifact_revision": profile.artifact_revision, "source_repository": profile.source_repository,
        "source_revision": profile.source_revision, "total_file_bytes": total_bytes,
        "aggregate_sha256": actual_aggregate, "all_frozen_files_verified": true,
        "aggregate_contract": "sha256(sorted UTF-8 path + NUL + lowercase file sha256 + NUL)",
    }))?;
    Ok(Qwen3Artifact {
        id: ArtifactId::new(&profile.id)?,
        directory,
        files: pins,
        template_sha256: Sha256::digest(qwen3::template::CHAT_TEMPLATE.as_bytes()).into(),
    })
}

/// Mirrors the documented production startup components for fresh model+KV
/// with retain_model=true and retain_cache_buffers=false (no retained history).
/// The loader boundary compares this sum with the actual production ledger.
fn startup_bytes(plan: &RuntimePlan) -> AcceptanceResult<u64> {
    [
        plan.output_bytes,
        plan.temporary_bytes,
        plan.headroom_bytes,
        plan.load_limits.max_load_peak_bytes(),
        plan.cache.max_cache_bytes(),
    ]
    .into_iter()
    .try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| io::Error::other("startup arithmetic overflow").into())
    })
}

fn constrained_budget(a: u64, b: u64) -> AcceptanceResult<u64> {
    require(
        a > 0 && b >= a,
        "14B startup must be at least the positive 4B startup",
    )?;
    Ok(b)
}

fn plan_json(plan: &RuntimePlan) -> AcceptanceResult<Value> {
    Ok(json!({
        "source_bytes": plan.load_limits.max_source_bytes(),
        "resident_limit_bytes": plan.load_limits.max_resident_bytes(),
        "load_peak_bytes": plan.load_limits.max_load_peak_bytes(),
        "cache_bytes": plan.cache.max_cache_bytes(), "cache_capacity_tokens": plan.cache.capacity_tokens(),
        "temporary_bytes": plan.temporary_bytes, "output_bytes": plan.output_bytes,
        "headroom_bytes": plan.headroom_bytes, "prefill_chunk_tokens": plan.prefill_chunk_tokens.get(),
        "startup_bytes": startup_bytes(plan)?, "retained_token_history_bytes": 0,
    }))
}

fn host_json(host: HostMemorySnapshot) -> Value {
    json!({
        "physical_memory_bytes": host.physical_memory_bytes, "free_bytes": host.free_bytes,
        "inactive_bytes": host.inactive_bytes, "active_file_bytes": host.active_file_bytes,
        "other_apps_active_bytes": host.other_apps_active_bytes,
        "phys_footprint_bytes": host.phys_footprint_bytes,
        "graphics_footprint_bytes": host.graphics_footprint_bytes,
        "iogpu_wired_limit_bytes": host.iogpu_wired_limit_bytes,
        "recommended_working_set_bytes": host.recommended_working_set_bytes,
        "release_credit_bytes": host.release_credit_bytes,
    })
}

fn native_snapshot() -> AcceptanceResult<Value> {
    let s = mlx_ffi::memory::snapshot()?;
    Ok(
        json!({"active_bytes": s.active_bytes, "cache_bytes": s.cache_bytes, "peak_bytes": s.peak_bytes, "memory_limit_bytes": s.memory_limit_bytes}),
    )
}

struct ObservedBackend {
    real: Qwen3Backend,
    evidence: Arc<Evidence>,
    startup: BTreeMap<String, u64>,
}

impl ModelFamily for ObservedBackend {
    fn registration(&self) -> &FamilyRegistration {
        self.real.registration()
    }

    fn load(&self, request: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        let artifact = request.artifact().id().as_str().to_owned();
        let observed = self
            .evidence
            .observations()
            .map_err(|e| engine_failure(e.to_string()))?;
        let ledger = self
            .evidence
            .admission
            .get()
            .ok_or_else(|| engine_failure("missing shared admission"))?
            .snapshot()?;
        let expected = self
            .startup
            .get(&artifact)
            .copied()
            .ok_or_else(|| engine_failure("unexpected artifact in loader"))?;
        // Fail before allocating anything if either old native ownership or its
        // ledger charge survived. Exact equality also detects double admission.
        if !observed.live.is_empty() || ledger.reserved_bytes() != expected {
            return Err(engine_failure(format!(
                "loader boundary invariant: live={:?}, reserved={}, expected={expected}",
                observed.live,
                ledger.reserved_bytes()
            )));
        }
        let sequence = self.evidence.emit("native_loader_enter", json!({
            "artifact_id": artifact, "expected_startup_bytes": expected,
            "actual_reserved_bytes": ledger.reserved_bytes(), "native_models_before_loader": 0,
            "startup_charged_exactly_once": true, "mlx": native_snapshot().map_err(|e| engine_failure(e.to_string()))?,
        })).map_err(|e| engine_failure(e.to_string()))?;
        self.evidence
            .observations
            .lock()
            .map_err(|_| engine_failure("observation lock poisoned"))?
            .loader_entries
            .entry(artifact.clone())
            .or_default()
            .push(sequence);
        let mut loaded = self.real.load(request)?;
        if loaded.capabilities().backend != ExecutionBackend::MlxMetal {
            return Err(engine_failure("real acceptance loaded a non-Metal model"));
        }
        let usage = loaded.memory_usage()?;
        let vocabulary_size = loaded
            .autoregressive()
            .ok_or_else(|| engine_failure("real Qwen3 model is not autoregressive"))?
            .vocabulary_size();
        {
            let mut observations = self
                .evidence
                .observations
                .lock()
                .map_err(|_| engine_failure("observation lock poisoned"))?;
            if !observations.live.insert(artifact.clone()) {
                return Err(engine_failure("duplicate live model handle"));
            }
            observations.maximum_live_models = observations
                .maximum_live_models
                .max(observations.live.len());
        }
        let wrapped = ObservedModel {
            real: Some(loaded),
            artifact: artifact.clone(),
            vocabulary_size,
            evidence: self.evidence.clone(),
        };
        self.evidence.emit("native_model_loaded", json!({
            "artifact_id": artifact, "execution_backend": "MlxMetal",
            "allocated_bytes": usage.allocated_bytes(), "used_bytes": usage.used_bytes(),
            "weight_representations": format!("{:?}", wrapped.capabilities().weight_representations),
            "mlx": native_snapshot().map_err(|e| engine_failure(e.to_string()))?,
        })).map_err(|e| engine_failure(e.to_string()))?;
        Ok(Box::new(wrapped))
    }
}

impl RuntimeBackend for ObservedBackend {
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        self.real.render_and_count(registry, context)
    }
    fn plan<'a>(
        &'a self,
        registry: &'a Registry,
        profile: &'a ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, RuntimePlan> {
        self.real.plan(registry, profile, input_tokens)
    }
    fn verify_artifact(
        &self,
        artifact: &ArtifactRegistration,
        control: &ExecutionControl,
    ) -> Result<(), EngineError> {
        self.real.verify_artifact(artifact, control)
    }
    fn decoder(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        self.real.decoder(registry, profile)
    }
    fn incremental_mlx_bytes(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        plan: &RuntimePlan,
        residency: RuntimeResidency,
    ) -> Result<Option<u64>, EngineError> {
        self.real
            .incremental_mlx_bytes(registry, profile, plan, residency)
    }
    fn eos_tokens(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Vec<u32>, EngineError> {
        self.real.eos_tokens(registry, profile)
    }
}

struct ObservedModel {
    real: Option<Box<dyn LoadedModel>>,
    artifact: String,
    vocabulary_size: usize,
    evidence: Arc<Evidence>,
}

impl ObservedModel {
    fn real(&self) -> &dyn LoadedModel {
        self.real
            .as_deref()
            .expect("live observer owns its real model")
    }
    fn autoregressive_real(&mut self) -> Result<&mut dyn AutoregressiveModel, EngineError> {
        self.real
            .as_deref_mut()
            .and_then(LoadedModel::autoregressive)
            .ok_or_else(|| engine_failure("real autoregressive model disappeared"))
    }
    fn record_logits(&self, operation: &str, logits: &TokenLogits) -> Result<(), EngineError> {
        let mut hash = Sha256::new();
        for value in logits.values() {
            hash.update(value.to_bits().to_le_bytes());
        }
        self.evidence.emit(operation, json!({"artifact_id": self.artifact, "vocabulary_size": logits.vocabulary_size(), "logits_sha256": hex(&hash.finalize()), "source": "real Qwen3 Metal forward result"})).map_err(|e| engine_failure(e.to_string()))?;
        let mut observations = self
            .evidence
            .observations
            .lock()
            .map_err(|_| engine_failure("observation lock poisoned"))?;
        let counts = if operation == "native_prefill_completed" {
            &mut observations.prefill_calls
        } else {
            &mut observations.decode_calls
        };
        *counts.entry(self.artifact.clone()).or_default() += 1;
        Ok(())
    }
}

impl LoadedModel for ObservedModel {
    fn instance_id(&self) -> &ModelInstanceId {
        self.real().instance_id()
    }
    fn artifact_id(&self) -> &ArtifactId {
        self.real().artifact_id()
    }
    fn registration(&self) -> &FamilyRegistration {
        self.real().registration()
    }
    fn capabilities(&self) -> &Capabilities {
        self.real().capabilities()
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        self.real().memory_usage()
    }
    fn autoregressive(&mut self) -> Option<&mut dyn AutoregressiveModel> {
        Some(self)
    }
}

impl AutoregressiveModel for ObservedModel {
    fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }
    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        self.autoregressive_real()?.allocate_cache(request, control)
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        let output = self
            .autoregressive_real()?
            .prefill(tokens, cache, control)?;
        self.record_logits("native_prefill_completed", output.logits())?;
        Ok(output)
    }
    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        let logits = self.autoregressive_real()?.decode(token, cache, control)?;
        self.record_logits("native_decode_completed", &logits)?;
        Ok(logits)
    }
}

impl Drop for ObservedModel {
    fn drop(&mut self) {
        // Actual native model destruction completes before the evidence event.
        // Resident's reservation may release just after this Drop returns;
        // the next loader entry verifies that ledger release also completed.
        drop(self.real.take());
        if let Ok(mut observations) = self.evidence.observations.lock() {
            observations.live.remove(&self.artifact);
        }
        if let Ok(sequence) = self.evidence.emit("native_model_destroyed", json!({"artifact_id": self.artifact, "real_model_drop_completed": true, "mlx": native_snapshot().ok()})) {
            if let Ok(mut observations) = self.evidence.observations.lock() { observations.destroyed.entry(self.artifact.clone()).or_default().push(sequence); }
        }
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
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park_timeout(Duration::from_millis(100)),
        }
    }
}

struct Monitor {
    engine: EngineCore<FamilyRuntime>,
    pump: MemoryEnforcementPump,
    evidence: Arc<Evidence>,
    next_tick: Instant,
    cancellation_started: Option<Instant>,
}

impl Monitor {
    fn new(
        engine: EngineCore<FamilyRuntime>,
        enforcer: MemoryEnforcer,
        evidence: Arc<Evidence>,
    ) -> Self {
        Self {
            pump: MemoryEnforcementPump::new(engine.clone(), enforcer),
            engine,
            evidence,
            next_tick: Instant::now(),
            cancellation_started: None,
        }
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> AcceptanceResult<()> {
        if self.evidence.started.elapsed() >= self.evidence.timeout {
            if self.cancellation_started.is_none() {
                let target = self.engine.active_cancellation_target()?;
                let cancellation = target.map(|target| target.request_cancel());
                self.evidence.emit(
                    "deadline_cancellation_requested",
                    json!({"result": format!("{cancellation:?}"), "cleanup_grace_seconds": 120}),
                )?;
                self.cancellation_started = Some(Instant::now());
            }
            if self
                .cancellation_started
                .is_some_and(|since| since.elapsed() >= Duration::from_secs(120))
            {
                let _ = self.evidence.emit("acceptance_failed", json!({"reason": "native operation did not acknowledge cooperative timeout within 120 seconds", "process_exit": 124, "acceptance_passed": false}));
                process::exit(124);
            }
        }
        if Instant::now() >= self.next_tick {
            self.next_tick = Instant::now() + Duration::from_secs(1);
            let poll = self.pump.poll_tick(cx)?;
            let status = poll.tick.status;
            self.evidence.emit(
                "live_memory_tick",
                json!({
                    "pressure": status.pressure.as_str(), "usage_bytes": status.usage_bytes,
                    "hard_limit_bytes": status.hard_limit_bytes, "soft_bytes": status.soft_bytes,
                    "hard_bytes": status.hard_bytes, "abort_limit_bytes": status.abort_limit_bytes,
                    "admit_line_bytes": status.admit_line_bytes, "binding": status.binding,
                    "admissions_paused": status.admissions_paused, "emergency": status.emergency,
                    "request_cache_reclaim": poll.tick.request_cache_reclaim,
                    "request_model_eviction": poll.tick.request_model_eviction,
                    "cancellation": format!("{:?}", poll.cancellation),
                    "native_action_completed": poll.completed.as_ref().map(|result| match result {
                        Ok(report) => json!({
                            "model": format!("{:?}", report.model),
                            "allocator_reclaim_completed": report.allocator_cache.is_some(),
                            "allocator_error": report.allocator_error,
                            "pressure": report.tick.status.pressure.as_str(),
                        }),
                        Err(error) => json!({"error": error.to_string()}),
                    }),
                    "native_command_pending": self.pump.has_pending(),
                }),
            )?;
        }
        Ok(())
    }

    fn wait<F: Future>(&mut self, future: F) -> AcceptanceResult<F::Output> {
        let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut future = pin!(future);
        loop {
            self.poll(&mut cx)?;
            if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
                require(
                    self.cancellation_started.is_none(),
                    "acceptance timed out; native cleanup acknowledged",
                )?;
                return Ok(value);
            }
            thread::park_timeout(Duration::from_millis(100));
        }
    }

    fn idle_for(&mut self, duration: Duration) -> AcceptanceResult<()> {
        let until = Instant::now() + duration;
        self.wait(poll_fn(|_| {
            if Instant::now() >= until {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }))
    }
}

fn context(profile: &ModelProfileId, name: &str) -> AcceptanceResult<ContextSnapshot> {
    Ok(ContextSnapshot::new(
        ContextSnapshotId::new(name)?,
        profile.clone(),
        vec![Message {
            role: MessageRole::User,
            content: PROMPT.into(),
        }],
    )?)
}

fn request(profile: &ModelProfileId, name: &str) -> AcceptanceResult<GenerationRequest> {
    Ok(GenerationRequest::new(
        RequestId::new(name)?,
        profile.clone(),
        context(profile, name)?,
        GenerationOptions::greedy(OUTPUT_TOKENS)?,
    )?)
}

fn generate(
    engine: &EngineCore<FamilyRuntime>,
    monitor: &mut Monitor,
    profile: &ModelProfileId,
    name: &str,
) -> AcceptanceResult<Result<(), ErrorKind>> {
    monitor.evidence.emit("generation_requested", json!({"request_id": name, "profile_id": profile.as_str(), "prompt": PROMPT, "maximum_output_tokens": OUTPUT_TOKENS}))?;
    let generation = match monitor.wait(engine.generate(request(profile, name)?))? {
        Ok(generation) => generation,
        Err(error) => {
            monitor.evidence.emit("generation_refused", json!({"request_id": name, "error_kind": format!("{:?}", error.kind()), "message": error.to_string()}))?;
            return Ok(Err(error.kind()));
        }
    };
    let mut events = generation.into_events();
    let mut result = None;
    let mut text = String::new();
    while let Some(event) = monitor.wait(poll_fn(|cx| events.as_mut().poll_next(cx)))? {
        require(
            result.is_none(),
            "generation emitted an event after its terminal event",
        )?;
        match event {
            GenerationEvent::TextDelta { text: delta } => text.push_str(&delta),
            GenerationEvent::RuntimeWarning { code, message } => {
                monitor.evidence.emit(
                    "runtime_warning",
                    json!({"request_id": name, "code": code, "message": message}),
                )?;
            }
            GenerationEvent::ToolCallDelta { .. } => {
                return Err(
                    io::Error::other("unexpected tool call in text-only Qwen3 acceptance").into(),
                );
            }
            GenerationEvent::GenerationCompleted {
                finish_reason,
                usage,
            } => {
                require(
                    usage.input_tokens > 0 && usage.output_tokens >= 2,
                    "real generation must consume input and produce at least two tokens",
                )?;
                monitor.evidence.emit("generation_completed", json!({"request_id": name, "finish_reason": format!("{finish_reason:?}"), "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "text": text, "text_sha256": hex(&Sha256::digest(text.as_bytes()))}))?;
                result = Some(Ok(()));
            }
            GenerationEvent::RequestFailed {
                error,
                partial_usage,
            } => {
                monitor.evidence.emit("generation_failed", json!({"request_id": name, "error_kind": format!("{:?}", error.kind()), "message": error.to_string(), "input_tokens": partial_usage.input_tokens, "output_tokens": partial_usage.output_tokens}))?;
                result = Some(Err(error.kind()));
            }
        }
    }
    // Exhaust and drop the stream before observing idle ledger state: the
    // bounded output reservation belongs to the stream's lifetime as well.
    drop(events);
    result.ok_or_else(|| io::Error::other("generation ended without a terminal event").into())
}

fn run(
    root: &Path,
    expected_commit: &str,
    directories: &[PathBuf; 2],
    evidence: Arc<Evidence>,
) -> AcceptanceResult<()> {
    require(
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
        "REAL acceptance requires native macOS aarch64 and MlxMetal; CPU/Linux results cannot pass",
    )?;
    evidence.emit("source_verified", provenance(root, expected_commit)?)?;
    let device = platform::discover()?;
    let host = platform::discover_host_memory_snapshot()?;
    evidence.emit("host_verified", json!({
        "os": env::consts::OS, "architecture": env::consts::ARCH,
        "uname": command("uname", &["-a"] )?, "macos_version": command("sw_vers", &["-productVersion"] )?,
        "macos_build": command("sw_vers", &["-buildVersion"] )?,
        "hardware_model": command("sysctl", &["-n", "hw.model"] )?,
        "cpu_brand": command("sysctl", &["-n", "machdep.cpu.brand_string"] )?,
        "physical_memory_bytes": device.physical_memory_bytes(),
        "recommended_working_set_bytes": device.recommended_working_set_bytes(),
        "host_memory": host_json(host), "native_backend_required": "MlxMetal",
    }))?;
    let frozen = frozen_profiles()?;
    let a = verify_profile(&frozen[0], &directories[0], &evidence)?;
    let b = verify_profile(&frozen[1], &directories[1], &evidence)?;
    require(
        a.directory != b.directory,
        "two artifact IDs cannot share one directory",
    )?;
    let generation_limits = GenerationLimits::new(64, 64 * 1024, 16 * 1024)?;
    let limits = Qwen3BackendLimits {
        max_files: 64,
        max_config_bytes: 8 * 1024 * 1024,
        max_tokenizer_bytes: 16 * 1024 * 1024,
        max_messages: 256,
        max_input_bytes: 1024 * 1024,
        max_rendered_bytes: 2 * 1024 * 1024,
        max_token_bytes: 512 * 1024,
        verification_chunk_bytes: MIB as usize,
        context_limit_tokens: 512,
        prefill_chunk_tokens: NonZeroUsize::new(16).unwrap(),
        loader_headroom_bytes: 64 * MIB,
        execution_headroom_bytes: 64 * MIB,
        generation: generation_limits,
    };
    let real = Qwen3Backend::new_with_execution(vec![a, b], limits, Qwen3Execution::MlxMetal)?;
    let ids = [ModelProfileId::new(MODEL_A)?, ModelProfileId::new(MODEL_B)?];
    let profiles = ids
        .iter()
        .map(|id| {
            ModelProfile::new(
                id.clone(),
                ArtifactId::new(id.as_str())?,
                512,
                OUTPUT_TOKENS,
                8,
                GenerationOptions::greedy(OUTPUT_TOKENS)?,
            )
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let registry = real.registry(profiles)?;
    let rendered_a = block_on(real.render_and_count(&registry, context(&ids[0], "m9-plan-a")?))?;
    let rendered_b = block_on(real.render_and_count(&registry, context(&ids[1], "m9-plan-b")?))?;
    let plan_a = block_on(real.plan(&registry, &ids[0], rendered_a.token_count()))?;
    let plan_b = block_on(real.plan(&registry, &ids[1], rendered_b.token_count()))?;
    let startup_a = startup_bytes(&plan_a)?;
    let startup_b = startup_bytes(&plan_b)?;
    let budget = constrained_budget(startup_a, startup_b)?;
    // The initial safe host limit is independent of the test's constraint.
    // A machine that cannot safely fit the production 14B startup must fail
    // preflight instead of shrinking real plans or changing to a tiny fixture.
    let fresh_host = platform::discover_host_memory_snapshot()?;
    let initial_mlx = mlx_ffi::memory::snapshot()?;
    let usage = memory_guard::usage_bytes(initial_mlx.active_bytes as u64, fresh_host);
    let safe = memory_guard::ceiling_breakdown(MemoryGuardConfig::balanced(), fresh_host, usage)?;
    evidence.emit("budget_planned", json!({
        "model_a": {"artifact_id": MODEL_A, "input_tokens": rendered_a.token_count(), "plan": plan_json(&plan_a)?},
        "model_b": {"artifact_id": MODEL_B, "input_tokens": rendered_b.token_count(), "plan": plan_json(&plan_b)?},
        "shared_budget_bytes": budget, "selection": "max(individual fresh startup), exactly the 14B startup",
        "measured_safe_ceiling_bytes": safe.hard_limit_bytes, "safe_reserve_bytes": safe.reserve_bytes,
        "safe_static_bytes": safe.static_bytes, "safe_dynamic_bytes": safe.dynamic_bytes,
        "safe_metal_cap_bytes": safe.metal_cap_bytes, "safe_binding": safe.binding.map(|b| b.as_str()),
        "host_memory": host_json(fresh_host), "retain_model": true, "retain_cache_buffers": false,
        "host_memory_observation_source": "production native probes",
    }))?;
    require(
        safe.hard_limit_bytes > 0 && budget <= safe.hard_limit_bytes,
        format!(
            "safe host preflight rejected: production 14B startup {budget} bytes exceeds measured balanced ceiling {} bytes; free memory or use a larger Apple Silicon Mac",
            safe.hard_limit_bytes
        ),
    )?;
    let admission = AdmissionController::new(budget)?;
    evidence
        .admission
        .set(admission.clone())
        .map_err(|_| io::Error::other("shared admission already initialized"))?;
    let enforcer = MemoryEnforcer::new(MemoryGuardConfig::custom(budget)?, admission.clone());
    let initial_tick = enforcer.tick()?;
    require(
        initial_tick.status.hard_limit_bytes == budget && !initial_tick.status.admissions_paused,
        "live enforcer cannot safely retain the planned constrained budget",
    )?;
    let previous_memory_limit = mlx_ffi::memory::set_memory_limit_bytes(usize::try_from(budget)?)?;
    let previous_cache_limit = mlx_ffi::memory::set_cache_limit_bytes(256 * MIB as usize)?;
    mlx_ffi::memory::reset_peak_memory()?;
    evidence.emit("native_limits_configured", json!({"memory_guideline_bytes": budget, "allocator_cache_limit_bytes": 256 * MIB, "previous_memory_guideline_bytes": previous_memory_limit, "previous_allocator_cache_limit_bytes": previous_cache_limit, "live_guard_tier": "custom", "one_admission_controller": true}))?;
    let backend: Arc<dyn RuntimeBackend> = Arc::new(ObservedBackend {
        real,
        evidence: evidence.clone(),
        startup: BTreeMap::from([(MODEL_A.into(), startup_a), (MODEL_B.into(), startup_b)]),
    });
    let runtime = FamilyRuntime::new(
        vec![backend],
        admission.clone(),
        generation_limits,
        RetentionPolicy {
            retain_model: true,
            retain_cache_buffers: false,
        },
    )?
    .with_model_pool(Some(Duration::from_secs(300)))?
    .with_memory_enforcer(enforcer.clone())
    .with_prefill_min_chunk_tokens(NonZeroUsize::new(1).unwrap())
    .with_pressure_policy(PressurePolicy::new(PressureOptions {
        allocator_ceiling_bytes: Some(budget),
        allow_allocator_cache_clear: true,
        allow_idle_state_release: true,
        allow_idle_model_unload: true,
        max_reclamation_actions: 5,
    })?);
    let engine = EngineCore::new(registry, Scheduler::new(16)?, runtime);
    let mut monitor = Monitor::new(engine.clone(), enforcer, evidence.clone());
    require(
        generate(&engine, &mut monitor, &ids[0], "m9-large-a")?.is_ok(),
        "real 4B generation failed",
    )?;
    let residents = monitor.wait(engine.runtime().model_residency())??;
    require(
        residents.len() == 1 && residents[0].artifact.as_str() == MODEL_A && !residents[0].pinned,
        "A did not remain resident and unpinned after generation",
    )?;
    let a_instance = residents[0].instance.clone();
    let idle_a_bytes = residents[0].committed_weight_bytes;
    let idle_ledger = admission.snapshot()?;
    require(
        idle_a_bytes > 0 && idle_ledger.reserved_bytes() == idle_a_bytes,
        "idle A must retain exactly its real weight reservation with output and KV released",
    )?;
    require(
        idle_a_bytes
            .checked_add(startup_b)
            .is_some_and(|combined| combined > budget),
        "budget does not force A eviction for B",
    )?;
    evidence.emit("model_a_idle", json!({"artifact_id": MODEL_A, "committed_weight_bytes": idle_a_bytes, "b_startup_bytes": startup_b, "combined_bytes": idle_a_bytes + startup_b, "budget_bytes": budget, "eviction_required": true}))?;
    monitor.idle_for(Duration::from_secs(2))?;
    let pin = monitor
        .wait(engine.runtime().pin_model(a_instance.clone()))??
        .ok_or_else(|| io::Error::other("A disappeared before the pin scenario"))?;
    evidence.emit("model_a_pinned", json!({"artifact_id": MODEL_A}))?;
    let pinned_result = generate(&engine, &mut monitor, &ids[1], "m9-large-b-pinned")?;
    require(
        pinned_result == Err(ErrorKind::MemoryLimitExceeded),
        format!("pinned A must refuse B with MemoryLimitExceeded; got {pinned_result:?}"),
    )?;
    let residents = monitor.wait(engine.runtime().model_residency())??;
    let observations = evidence.observations()?;
    require(
        residents.len() == 1
            && residents[0].artifact.as_str() == MODEL_A
            && residents[0].pinned
            && residents[0].instance.same_instance(&a_instance),
        "pinned B refusal did not preserve A's actual instance",
    )?;
    require(
        admission.snapshot()?.reserved_bytes() == idle_a_bytes
            && observations.live == BTreeSet::from([MODEL_A.into()])
            && !observations.loader_entries.contains_key(MODEL_B),
        "pinned refusal leaked a reservation or entered B's native loader",
    )?;
    evidence.emit("pinned_refusal_verified", json!({"error_kind": "MemoryLimitExceeded", "model_a_same_instance": true, "model_a_preserved": true, "b_loader_calls": 0, "reservation_bytes": idle_a_bytes}))?;
    drop(pin);
    let residents = monitor.wait(engine.runtime().model_residency())??;
    require(
        residents.len() == 1
            && !residents[0].pinned
            && residents[0].instance.same_instance(&a_instance),
        "unpin did not preserve A as an eligible idle model",
    )?;
    evidence.emit("model_a_unpinned", json!({"artifact_id": MODEL_A}))?;
    require(
        generate(&engine, &mut monitor, &ids[1], "m9-large-b-after-unpin")?.is_ok(),
        "real 14B generation failed after unpin",
    )?;
    let residents = monitor.wait(engine.runtime().model_residency())??;
    require(
        residents.len() == 1 && residents[0].artifact.as_str() == MODEL_B,
        "B was not the sole retained native model",
    )?;
    require(
        admission.snapshot()?.reserved_bytes() == residents[0].committed_weight_bytes,
        "idle B retained extra admission charges",
    )?;
    let observations = evidence.observations()?;
    let a_drops = observations
        .destroyed
        .get(MODEL_A)
        .ok_or_else(|| io::Error::other("missing actual A destruction evidence"))?;
    let b_loads = observations
        .loader_entries
        .get(MODEL_B)
        .ok_or_else(|| io::Error::other("missing B native loader evidence"))?;
    require(
        a_drops.len() == 1
            && b_loads.len() == 1
            && a_drops[0] < b_loads[0]
            && observations.maximum_live_models == 1,
        "A destruction did not precede the only B loader, or native models overlapped",
    )?;
    for id in [MODEL_A, MODEL_B] {
        require(
            observations.prefill_calls.get(id).copied().unwrap_or(0) > 0
                && observations.decode_calls.get(id).copied().unwrap_or(0) > 0,
            format!("{id} lacks real prefill/decode evidence"),
        )?;
    }
    evidence.emit("eviction_before_loader_verified", json!({"a_destroy_sequence": a_drops[0], "b_loader_sequence": b_loads[0], "maximum_native_model_handles": observations.maximum_live_models, "b_idle_weight_bytes": residents[0].committed_weight_bytes, "loader_admission_exactly_once": true, "mlx": native_snapshot()?}))?;
    // Stop the poller and release the final worker sender. Native shutdown is
    // asynchronous; wait for actual ownership destruction and an empty ledger.
    drop(monitor);
    drop(engine);
    let shutdown_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let ledger = admission.snapshot()?;
        if evidence.observations()?.live.is_empty()
            && ledger.reserved_bytes() == 0
            && ledger.active_reservations() == 0
        {
            break;
        }
        require(
            Instant::now() < shutdown_deadline,
            "engine shutdown left native ownership or admission charges after 30 seconds",
        )?;
        thread::park_timeout(Duration::from_millis(50));
    }
    evidence.emit("shutdown_verified", json!({"native_model_handles": 0, "reserved_bytes": 0, "active_reservations": 0, "mlx": native_snapshot()?}))?;
    provenance(root, expected_commit)?;
    evidence.emit(
        "acceptance_passed",
        json!({
            "commit_sha": expected_commit, "artifacts": [MODEL_A, MODEL_B],
            "artifact_sha256": [A_SHA256, B_SHA256], "execution_backend": "MlxMetal",
            "source_and_manifest_verified": true, "real_prefill_and_decode_both_models": true,
            "pinned_a_refused_b_and_preserved_a": true, "unpin_allowed_b": true,
            "actual_a_destruction_before_b_loader": true, "no_double_admission": true,
            "maximum_native_model_handles": 1, "shutdown_ledger_empty": true,
        }),
    )?;
    Ok(())
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    if args == ["--validate-manifest"] {
        match frozen_profiles() {
            Ok(profiles) => {
                println!(
                    "{}",
                    json!({"structural_manifest_validation": "passed", "corpus_sha256": CORPUS_SHA256, "artifacts": profiles.map(|p| p.id), "native_acceptance_executed": false})
                );
                return;
            }
            Err(error) => {
                eprintln!("manifest validation failed: {error}");
                process::exit(1);
            }
        }
    }
    if !args.is_empty() {
        eprintln!(
            "usage: m9_large_model_acceptance [--validate-manifest]; launch real acceptance with scripts/m9_large_model_acceptance.sh"
        );
        process::exit(2);
    }
    let setup = (|| -> AcceptanceResult<_> {
        let root = PathBuf::from(env::var_os("ALLOY_M9_SOURCE_ROOT").ok_or_else(|| {
            io::Error::other("ALLOY_M9_SOURCE_ROOT is required; use the launch script")
        })?)
        .canonicalize()?;
        let expected = env::var("ALLOY_M9_EXPECTED_COMMIT")?;
        let directories = [
            PathBuf::from(
                env::var_os("ALLOY_M9_MODEL_A_DIR")
                    .ok_or_else(|| io::Error::other("ALLOY_M9_MODEL_A_DIR missing"))?,
            ),
            PathBuf::from(
                env::var_os("ALLOY_M9_MODEL_B_DIR")
                    .ok_or_else(|| io::Error::other("ALLOY_M9_MODEL_B_DIR missing"))?,
            ),
        ];
        let output = PathBuf::from(
            env::var_os("ALLOY_M9_OUTPUT")
                .ok_or_else(|| io::Error::other("ALLOY_M9_OUTPUT missing"))?,
        );
        let seconds: u64 = env::var("ALLOY_M9_TIMEOUT_SECONDS")
            .unwrap_or_else(|_| "1800".into())
            .parse()?;
        require(
            (60..=7200).contains(&seconds),
            "timeout must be between 60 and 7200 seconds",
        )?;
        let evidence = Arc::new(Evidence::new(&output, Duration::from_secs(seconds))?);
        evidence.emit("acceptance_started", json!({"expected_commit_sha": expected, "output": output, "timeout_seconds": seconds, "native_acceptance_required": true}))?;
        Ok((root, expected, directories, evidence))
    })();
    let (root, expected, directories, evidence) = match setup {
        Ok(value) => value,
        Err(error) => {
            eprintln!("acceptance setup failed: {error}");
            process::exit(2);
        }
    };
    if let Err(error) = run(&root, &expected, &directories, evidence.clone()) {
        let _ = evidence.emit(
            "acceptance_failed",
            json!({"reason": error.to_string(), "acceptance_passed": false}),
        );
        eprintln!("M9 real large-model acceptance failed: {error}");
        process::exit(1);
    }
    println!(
        "M9 real Qwen3 4B -> 14B Metal acceptance passed; JSONL: {}",
        env::var("ALLOY_M9_OUTPUT").unwrap_or_default()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_corpus_resolves_two_distinct_large_checkpoints() {
        let [a, b] = frozen_profiles().unwrap();
        assert_eq!(a.id, MODEL_A);
        assert_eq!(b.id, MODEL_B);
        assert_ne!(a.artifact_sha256, b.artifact_sha256);
        assert_ne!(a.source_repository, b.source_repository);
        assert_eq!(a.files.len(), 11);
        assert_eq!(b.files.len(), 12);
    }
    #[test]
    fn aggregate_detects_a_changed_file_pin() {
        let mut a = frozen_profiles().unwrap()[0].clone();
        a.files.get_mut("config.json").unwrap()[0] ^= 1;
        assert_ne!(hex(&aggregate(&a.files)), a.artifact_sha256);
    }
    #[test]
    fn budget_fits_both_individual_startups_and_forces_positive_idle_eviction() {
        assert_eq!(constrained_budget(100, 300).unwrap(), 300);
        assert!(constrained_budget(0, 300).is_err());
        assert!(constrained_budget(301, 300).is_err());
        assert!(1u64 + constrained_budget(100, 300).unwrap() > 300);
    }
    #[test]
    fn sha256_parser_rejects_noncanonical_pins() {
        assert!(digest(&"A".repeat(64)).is_err());
        assert!(digest("abc").is_err());
        assert_eq!(digest(&"0".repeat(64)).unwrap(), [0; 32]);
    }
}
