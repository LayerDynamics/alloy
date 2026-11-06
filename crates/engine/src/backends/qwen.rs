//! Qwen3 unified-memory MLX adapter for the family-independent runtime.
//!
//! Verified safetensors are decoded to f32 and consumed into one set of
//! MLX weight arrays. Full f32 MLX K/V is shared by supported CPU and Metal
//! operations; it is not the legacy host-reference f64 cache.
//!
//! new() selects MLX CPU for portable, deterministic setup. Use
//! new_with_execution(..., Qwen3Execution::MlxMetal) to request Metal.
//! A requested Metal device is checked on the native worker and is never
//! silently replaced with CPU execution. Streams and arrays stay there.
//!
//! Shared pinned rendering, artifact checks, and incremental ByteLevel
//! decoding retain their existing contracts.
//!
//! Planning counts decoded/native loading overlap, full-cache functional
//! update overlap, and conservative f32 graph/workspace payload estimates.
//! These are engine-accounted allowances, NOT measured process RSS or a
//! hard MLX allocator limit. Library/allocator overhead requires separately
//! provisioned headroom; RAM discovery and allocator-pressure integration
//! are not implemented by this adapter.
//!
//! No resident quantized-weight, rotating-cache, sampling, thinking-mode,
//! or cross-request prefix-reuse capability is advertised here.

use crate::{
    admission::AdmissionController,
    api::{EngineCore, EngineFuture},
    error::EngineError,
    family::{
        AutoregressiveModel, BufferMemoryUsage, CacheAllocationRequest, ExecutionControl,
        FamilyCache, LoadedModel, ModelFamily, ModelInstanceId, ModelLoadLimits, ModelLoadRequest,
        PrefillOutput, TokenLogits,
    },
    generation::GenerationLimits,
    registry::{ArtifactRegistration, FamilyRegistration, FamilyVersion, ModelProfile, Registry},
    rendering::{self, PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions},
    runtime::{FamilyRuntime, RetentionPolicy, RuntimeBackend, RuntimePlan, TextDecoder},
    scheduler::Scheduler,
    types::{
        ArtifactId, CacheKind, CacheMode, CacheRepresentationId, Capabilities, ContextSnapshot,
        DecodingCapability, ExecutionBackend, ModelProfileId, RenderedContext,
        WeightRepresentation,
    },
};
use inference_infra::kv_cache::{KvCacheError, KvCacheSpec, MlxKvCache, MlxModelId};
use mlx_ffi::{Device, DeviceType, Stream, ops::OperationError};
use qwen3::{
    config::Qwen3Config,
    fixture::{LayerWeights, Qwen3Weights},
    loader::{CheckpointVerification, LoadOptions, load_weights_verified},
    mlx_model::{MlxQwen3Model, MlxQwenError},
    model::ModelError,
    template::CHAT_TEMPLATE,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{self, Read},
    num::NonZeroUsize,
    path::{Component, Path, PathBuf},
    rc::Rc,
    sync::Arc,
};
// Qwen policy remains in the adapter; rendering mechanics and identities
// are owned exclusively by crate::rendering.
const QWEN3_RESERVED_SPELLINGS: &[&str] = &[
    "<|",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];

/// Qwen MLX full-cache identity, distinct from the legacy host f64 cache.
pub const QWEN3_MLX_CACHE_REPRESENTATION: &str = "qwen3.mlx-f32.full.bhtd.v1";

/// Execution placement for one unified-memory MLX model instance.
///
/// Both selections use the same f32 array layout, not separate CPU/GPU
/// model representations. Selection does not mutate the global device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3Execution {
    /// Execute transformer operations through an explicit MLX CPU stream.
    MlxCpu,
    /// Execute transformer operations through an explicit Metal stream.
    /// Unavailability is an error; there is no automatic CPU fallback.
    MlxMetal,
}

impl Qwen3Execution {
    fn backend(self) -> ExecutionBackend {
        match self {
            Self::MlxCpu => ExecutionBackend::MlxCpu,
            Self::MlxMetal => ExecutionBackend::MlxMetal,
        }
    }
}

/// A file identity obtained through the caller's trusted artifact policy.
/// Do not create these pins from an untrusted manifest and call that verification.
#[derive(Clone, Debug)]
pub struct FilePin {
    /// Exact source-file length.
    pub bytes: u64,
    /// SHA-256 of the complete file contents.
    pub sha256: [u8; 32],
}

/// Explicitly pinned local artifact and template selection.
#[derive(Clone, Debug)]
pub struct Qwen3Artifact {
    /// Identifier that profiles will reference.
    pub id: ArtifactId,
    /// Existing artifact directory; canonicalized during construction.
    pub directory: PathBuf,
    /// Basenames only. Include config.json, tokenizer.json, every consumed
    /// weight shard/index, and any selected EOS configuration file.
    pub files: BTreeMap<String, FilePin>,
    /// Trusted digest of qwen3::template::CHAT_TEMPLATE for this profile path.
    pub template_sha256: [u8; 32],
}

impl Qwen3Artifact {
    /// Fingerprint the trusted manifest record used for render identity.
    ///
    /// Encoding: the NUL-terminated domain below, u64-LE file count, then
    /// each BTreeMap entry in UTF-8 name order as u64-LE name byte length,
    /// UTF-8 name, u64-LE pinned file size, and the 32-byte SHA-256 pin.
    /// Finally append the 32-byte trusted compiled-template digest.
    ///
    /// The artifact ID is framed separately by the shared renderer. Local
    /// directory paths are intentionally excluded. This function performs
    /// no I/O and does not itself verify that files match these pins.
    pub fn manifest_sha256(&self) -> Result<[u8; 32], EngineError> {
        let mut hash = Sha256::new();
        hash.update(b"alloy.qwen3.artifact-manifest.v1\0");
        hash.update(as_u64(self.files.len())?.to_le_bytes());
        for (name, pin) in &self.files {
            hash.update(as_u64(name.len())?.to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(pin.bytes.to_le_bytes());
            hash.update(pin.sha256);
        }
        hash.update(self.template_sha256);
        Ok(hash.finalize().into())
    }
}

/// Explicit bounds; no model, output, or memory policy is silently selected.
#[derive(Clone, Debug)]
pub struct Qwen3BackendLimits {
    /// Maximum number of pinned files per artifact.
    pub max_files: usize,
    /// Maximum bytes in a JSON configuration file.
    pub max_config_bytes: usize,
    /// Maximum tokenizer.json source bytes.
    pub max_tokenizer_bytes: usize,
    /// Maximum structured messages accepted for rendering.
    pub max_messages: usize,
    /// Maximum cumulative message-content bytes.
    pub max_input_bytes: usize,
    /// Maximum rendered template bytes, enforced while writing.
    pub max_rendered_bytes: usize,
    /// Maximum tokenizer vocabulary spelling bytes for a token.
    pub max_token_bytes: usize,
    /// Chunk size for streaming source-file verification.
    pub verification_chunk_bytes: usize,
    /// Caller-selected validated context ceiling for this execution path.
    pub context_limit_tokens: usize,
    /// Maximum chunk handed to family prefill. The adapter additionally
    /// checks cancellation between transformer layers and before KV commit.
    pub prefill_chunk_tokens: NonZeroUsize,
    /// Applied separately to resident weight-buffer padding and loader
    /// metadata/allocator/control overhead. Both allowances are reserved.
    pub loader_headroom_bytes: u64,
    /// Applied to request overhead and separately to each of the two
    /// KV allocation versions. All allowances are explicitly reserved.
    pub execution_headroom_bytes: u64,
    /// Exact limits used to size output admission.
    pub generation: GenerationLimits,
}

impl Qwen3BackendLimits {
    fn validate(&self) -> Result<(), EngineError> {
        if [
            self.max_files,
            self.max_config_bytes,
            self.max_tokenizer_bytes,
            self.max_messages,
            self.max_input_bytes,
            self.max_rendered_bytes,
            self.max_token_bytes,
            self.verification_chunk_bytes,
            self.context_limit_tokens,
        ]
        .contains(&0)
        {
            return Err(EngineError::invalid_request(
                "qwen3 limits",
                "all size/count limits must be positive",
            ));
        }
        output_bytes(self.generation)?;
        Ok(())
    }
}

struct ArtifactData {
    specification: Qwen3Artifact,
    config: Qwen3Config,
    tokenizer: Arc<PinnedTokenizer>,
    special_tokens: Arc<BTreeSet<u32>>,
    byte_decoder: Arc<BTreeMap<char, u8>>,
    manifest_sha256: [u8; 32],
    eos: Vec<u32>,
    source_bytes: u64,
    resident_bound: u64,
    decoded_bound: u64,
    checksums: BTreeMap<PathBuf, [u8; 32]>,
}

/// Real Qwen3 backend. Normal metadata/factories may cross threads;
/// actual model and cache instances are created by ModelFamily::load.
pub struct Qwen3Backend {
    registration: FamilyRegistration,
    artifacts: BTreeMap<ArtifactId, ArtifactData>,
    limits: Qwen3BackendLimits,
    execution: Qwen3Execution,
}

impl Qwen3Backend {
    /// Validate pinned metadata and tokenizer configuration without loading
    /// model weights. This synchronous constructor can perform substantial
    /// tokenizer parsing; call it during setup, not on a cancellation loop.
    /// Selects MLX CPU. Use new_with_execution to request Metal explicitly.
    pub fn new(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
    ) -> Result<Self, EngineError> {
        Self::new_with_execution(specifications, limits, Qwen3Execution::MlxCpu)
    }

    /// Construct with explicit unified-memory MLX execution placement.
    /// Native streams/devices are created only during worker-side loading.
    pub fn new_with_execution(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        limits.validate()?;
        if specifications.is_empty() {
            return Err(EngineError::invalid_request(
                "artifacts",
                "must not be empty",
            ));
        }
        let registration = registration(limits.context_limit_tokens, execution)?;
        let mut artifacts = BTreeMap::new();
        let inverse = Arc::new(byte_decoder_map());
        for mut specification in specifications {
            if artifacts.contains_key(&specification.id) {
                return Err(EngineError::invalid_request(
                    "artifact_id",
                    "duplicate pinned artifact",
                ));
            }
            specification.directory = specification.directory.canonicalize().map_err(|e| {
                EngineError::checkpoint_io(
                    "canonicalize Qwen3 artifact",
                    &specification.directory,
                    e,
                )
            })?;
            if !specification.directory.is_dir() {
                return Err(validation(
                    Some(&specification.id),
                    "artifact path must be a directory",
                ));
            }
            if specification.files.is_empty() || specification.files.len() > limits.max_files {
                return Err(validation(
                    Some(&specification.id),
                    "pinned file count is outside its limit",
                ));
            }
            if digest(CHAT_TEMPLATE.as_bytes()) != specification.template_sha256 {
                return Err(validation(
                    Some(&specification.id),
                    "trusted template digest does not match the compiled Qwen3 template",
                ));
            }
            let mut source_bytes = 0;
            let mut checksums = BTreeMap::new();
            let mut has_weights = false;
            for (name, pin) in &specification.files {
                let path = component_path(&specification, name)?;
                source_bytes = sum(&[source_bytes, pin.bytes])?;
                if name.ends_with(".safetensors") {
                    has_weights = true;
                }
                checksums.insert(path, pin.sha256);
            }
            if !has_weights {
                return Err(validation(
                    Some(&specification.id),
                    "no pinned safetensors weights",
                ));
            }
            let config_bytes =
                read_metadata(&specification, "config.json", limits.max_config_bytes)?;
            let config_value =
                parse_json(&specification.directory.join("config.json"), &config_bytes)?;
            let config = parse_config(&config_value)?;
            // Undeclared representation uses the conservative dense path.
            // Quantized declarations retain the existing packed estimates.
            let declares_quantization = ["quantization", "quantization_config"]
                .iter()
                .any(|key| config_value.get(*key).is_some_and(|value| !value.is_null()));
            let (decoded_bound, resident_payload_bound) = if declares_quantization {
                (decoded_weight_bound(&config)?, resident_bound(&config)?)
            } else {
                (
                    dense_decoded_weight_bound(&config)?,
                    dense_weight_payload_bound(&config)?,
                )
            };
            let resident_bound = sum(&[resident_payload_bound, limits.loader_headroom_bytes])?;
            let tokenizer_bytes =
                read_metadata(&specification, "tokenizer.json", limits.max_tokenizer_bytes)?;
            let tokenizer_value = parse_json(
                &specification.directory.join("tokenizer.json"),
                &tokenizer_bytes,
            )?;
            if tokenizer_value
                .pointer("/model/type")
                .and_then(Value::as_str)
                != Some("BPE")
                || tokenizer_value
                    .pointer("/decoder/type")
                    .and_then(Value::as_str)
                    != Some("ByteLevel")
            {
                return Err(EngineError::unsupported(
                    "tokenizer",
                    "this adapter supports the actual BPE/ByteLevel Qwen3 tokenizer contract only",
                ));
            }
            let tokenizer_pin = specification.files.get("tokenizer.json").ok_or_else(|| {
                validation(
                    Some(&specification.id),
                    "tokenizer.json lacks a trusted pin",
                )
            })?;
            let tokenizer = Arc::new(PinnedTokenizer::from_pinned_json(
                &tokenizer_bytes,
                tokenizer_pin.sha256,
                limits.max_tokenizer_bytes,
            )?);
            if tokenizer.tokenizer().get_encode_special_tokens() {
                return Err(EngineError::unsupported(
                    "tokenizer options",
                    "special-token splitting is not enabled by this Qwen3 profile",
                ));
            }
            let mut specials = BTreeSet::new();
            if let Some(added) = tokenizer_value
                .get("added_tokens")
                .and_then(Value::as_array)
            {
                for entry in added {
                    if entry.get("special").and_then(Value::as_bool) == Some(true) {
                        let token = entry
                            .get("id")
                            .and_then(Value::as_u64)
                            .and_then(|id| u32::try_from(id).ok())
                            .ok_or_else(|| {
                                validation(Some(&specification.id), "invalid special-token ID")
                            })?;
                        specials.insert(token);
                    }
                }
            }
            for (spelling, id) in tokenizer.tokenizer().get_vocab(true) {
                if u64::from(id) >= as_u64(config.vocab_size)?
                    || spelling.len() > limits.max_token_bytes
                {
                    return Err(validation(
                        Some(&specification.id),
                        "tokenizer vocabulary exceeds the configured token/model bounds",
                    ));
                }
                if !specials.contains(&id) && !spelling.chars().all(|ch| inverse.contains_key(&ch))
                {
                    return Err(EngineError::unsupported(
                        "ByteLevel token",
                        "non-special token contains an unsupported byte-alphabet character",
                    ));
                }
            }
            let eos_value = if specification.files.contains_key("generation_config.json") {
                let bytes = read_metadata(
                    &specification,
                    "generation_config.json",
                    limits.max_config_bytes,
                )?;
                let generation = parse_json(
                    &specification.directory.join("generation_config.json"),
                    &bytes,
                )?;
                generation
                    .get("eos_token_id")
                    .cloned()
                    .or_else(|| config_value.get("eos_token_id").cloned())
            } else {
                config_value.get("eos_token_id").cloned()
            };
            let eos = parse_eos(eos_value.as_ref())?;
            for &token in &eos {
                if u64::from(token) >= as_u64(config.vocab_size)?
                    || tokenizer.tokenizer().id_to_token(token).is_none()
                    || !specials.contains(&token)
                {
                    return Err(validation(
                        Some(&specification.id),
                        "EOS configuration does not identify declared special tokens inside the model vocabulary",
                    ));
                }
            }
            let manifest_sha256 = specification.manifest_sha256()?;
            let data = ArtifactData {
                specification,
                config,
                tokenizer,
                special_tokens: Arc::new(specials),
                byte_decoder: Arc::clone(&inverse),
                manifest_sha256,
                eos,
                source_bytes,
                resident_bound,
                decoded_bound,
                checksums,
            };
            artifacts.insert(data.specification.id.clone(), data);
        }
        Ok(Self {
            registration,
            artifacts,
            limits,
            execution,
        })
    }

    /// Construct a registry whose family metadata matches this executable
    /// adapter, register its actual local artifacts, and add caller profiles.
    pub fn registry(&self, profiles: Vec<ModelProfile>) -> Result<Registry, EngineError> {
        let mut registry = Registry::new();
        registry.register_family(self.registration.clone())?;
        for artifact in self.artifacts.values() {
            registry.register_artifact(
                artifact.specification.id.clone(),
                &artifact.specification.directory,
            )?;
        }
        for profile in profiles {
            registry.register_profile(profile)?;
        }
        Ok(registry)
    }

    // ALLOY_UNIFIED_PLATFORM_FACTORY_V1

    /// Construct an engine using actual unified-memory observations.
    ///
    /// Returns the derived plan for diagnostics and explicit allocator
    /// configuration. Does not automatically change process-wide MLX
    /// settings; configure the returned plan during coordinated setup
    /// before starting generation.
    ///
    /// Discovery requires an actual unified-memory Metal device. CPU and
    /// Metal operation placement still use this same shared budget.
    pub fn into_unified_engine(
        self,
        profiles: Vec<ModelProfile>,
        scheduler: Scheduler,
        retention: RetentionPolicy,
        policy: crate::platform::UnifiedMemoryPolicy,
    ) -> Result<
        (
            EngineCore<FamilyRuntime>,
            crate::platform::UnifiedBudgetPlan,
        ),
        EngineError,
    > {
        let info = crate::platform::discover().map_err(|error| EngineError::ExecutionFailed {
            stage: "discover unified-memory platform",
            message: error.to_string(),
        })?;

        let plan = info
            .plan(policy)
            .map_err(|error| EngineError::ExecutionFailed {
                stage: "derive unified-memory budget",
                message: error.to_string(),
            })?;

        let admission = plan.admission_controller()?;
        let engine = self.into_engine(profiles, admission, scheduler, retention)?;

        Ok((engine, plan))
    }

    /// Construct with explicit bounded startup-pressure policy.
    ///
    /// This does not change process-wide MLX limits. The supplied ledger
    /// and allocator ceiling must come from centrally coordinated setup.
    pub fn into_pressure_engine(
        self,
        profiles: Vec<ModelProfile>,
        admission: AdmissionController,
        scheduler: Scheduler,
        retention: RetentionPolicy,
        pressure: crate::pressure::PressurePolicy,
    ) -> Result<EngineCore<FamilyRuntime>, EngineError> {
        let registry = self.registry(profiles)?;
        let limits = self.limits.generation;
        let backend: Arc<dyn RuntimeBackend> = Arc::new(self);
        let runtime = FamilyRuntime::new(vec![backend], admission, limits, retention)?
            .with_pressure_policy(pressure);
        Ok(EngineCore::new(registry, scheduler, runtime))
    }

    /// Connect this real backend to FamilyRuntime and EngineCore.
    /// The caller supplies budget, scheduling, output, and retention policy.
    pub fn into_engine(
        self,
        profiles: Vec<ModelProfile>,
        admission: AdmissionController,
        scheduler: Scheduler,
        retention: RetentionPolicy,
    ) -> Result<EngineCore<FamilyRuntime>, EngineError> {
        self.into_engine_with_memory_enforcer(profiles, admission, scheduler, retention, None)
    }

    /// Like [`Self::into_engine`], attaching a live memory enforcer for
    /// prefill admit-line / adaptive chunk shrink (M5–M6).
    pub fn into_engine_with_memory_enforcer(
        self,
        profiles: Vec<ModelProfile>,
        admission: AdmissionController,
        scheduler: Scheduler,
        retention: RetentionPolicy,
        memory_enforcer: Option<crate::memory_enforcer::MemoryEnforcer>,
    ) -> Result<EngineCore<FamilyRuntime>, EngineError> {
        let registry = self.registry(profiles)?;
        let generation_limits = self.limits.generation;
        let backend: Arc<dyn RuntimeBackend> = Arc::new(self);
        let mut runtime =
            FamilyRuntime::new(vec![backend], admission, generation_limits, retention)?;
        if let Some(enforcer) = memory_enforcer {
            runtime = runtime.with_memory_enforcer(enforcer);
        }
        Ok(EngineCore::new(registry, scheduler, runtime))
    }

    fn data(&self, artifact: &ArtifactRegistration) -> Result<&ArtifactData, EngineError> {
        let data = self
            .artifacts
            .get(artifact.id())
            .ok_or_else(|| EngineError::NotFound {
                resource: "pinned Qwen3 artifact",
                id: artifact.id().as_str().to_owned(),
            })?;
        if artifact.directory() != data.specification.directory.as_path()
            || artifact.family() != self.registration.family()
            || artifact.declared_context_limit_tokens() != data.config.max_context
        {
            return Err(validation(
                Some(artifact.id()),
                "registry artifact disagrees with pinned Qwen3 metadata",
            ));
        }
        Ok(data)
    }
}

impl ModelFamily for Qwen3Backend {
    fn registration(&self) -> &FamilyRegistration {
        &self.registration
    }

    fn load(&self, request: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        let control = request.control();
        control.check_cancelled()?;
        let data = self.data(request.artifact())?;
        let decoded_bound = data.decoded_bound;
        // Headroom covers group-size variance (planning assumes 64; some
        // fixtures/checkpoints use smaller groups with larger scale tables).
        let weight_charge_limit = sum(&[decoded_bound, self.limits.loader_headroom_bytes])?;
        let required_peak = sum(&[
            data.source_bytes,
            decoded_bound,
            data.resident_bound,
            self.limits.loader_headroom_bytes,
        ])?;
        if request.limits().max_source_bytes() < data.source_bytes
            || request.limits().max_resident_bytes() < data.resident_bound
            || request.limits().max_load_peak_bytes() < required_peak
        {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "MLX Qwen loading overlap",
                required_bytes: required_peak,
                available_bytes: request.limits().max_load_peak_bytes(),
            });
        }

        let cpu_device = Device::new(DeviceType::Cpu, 0)
            .map_err(|e| execution_error("create MLX CPU device", e.to_string()))?;
        if !cpu_device
            .is_available()
            .map_err(|e| execution_error("query MLX CPU", e.to_string()))?
        {
            return Err(EngineError::unsupported(
                "MLX CPU",
                "requested device is unavailable",
            ));
        }
        let loading_stream = Stream::new(&cpu_device)
            .map_err(|e| execution_error("create MLX loading stream", e.to_string()))?;
        let execution_stream = match self.execution {
            Qwen3Execution::MlxCpu => Rc::new(loading_stream),
            Qwen3Execution::MlxMetal => {
                if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                    return Err(EngineError::unsupported(
                        "MLX Metal",
                        "requires an Apple-Silicon macOS worker",
                    ));
                }
                let gpu = Device::new(DeviceType::Gpu, 0)
                    .map_err(|e| execution_error("create MLX Metal device", e.to_string()))?;
                if !gpu
                    .is_available()
                    .map_err(|e| execution_error("query MLX Metal", e.to_string()))?
                {
                    return Err(EngineError::unsupported(
                        "MLX Metal",
                        "requested device is unavailable; no CPU fallback",
                    ));
                }
                // Array construction copies decoded source buffers; the
                // model subsequently evaluates its arrays on this worker.
                // The unused CPU setup stream has no submitted operations.
                drop(loading_stream);
                Rc::new(
                    Stream::new(&gpu)
                        .map_err(|e| execution_error("create MLX Metal stream", e.to_string()))?,
                )
            }
        };
        control.check_cancelled()?;
        let cancelled = || control.is_cancelled();
        let verification = CheckpointVerification {
            sha256: &data.checksums,
            cancelled: &cancelled,
        };
        let result = load_weights_verified(
            &data.config,
            &data.specification.directory,
            LoadOptions {
                max_file_bytes: as_usize(request.limits().max_source_bytes())?,
                max_weight_bytes: as_usize(weight_charge_limit)?,
            },
            &verification,
        );
        control.check_cancelled()?;
        let weights = result.map_err(|e| model_error(e, "load MLX Qwen weights"))?;
        let decoded = weight_memory(&weights)?;
        if decoded.allocated_bytes() > weight_charge_limit {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "decoded MLX Qwen weight buffers",
                required_bytes: decoded.allocated_bytes(),
                available_bytes: weight_charge_limit,
            });
        }
        let model = MlxQwen3Model::from_weights(
            data.config.clone(),
            weights,
            request.limits().max_resident_bytes(),
            execution_stream.as_ref(),
        )
        .map_err(|e| mlx_model_error(e, control, "construct MLX Qwen model"))?;
        control.check_cancelled()?;
        let payload = model.weight_payload_bytes();
        let measured = model
            .weight_buffer_bytes()
            .map_err(|e| mlx_model_error(e, control, "measure MLX Qwen weights"))?;
        if measured > request.limits().max_resident_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "measured MLX Qwen weight buffers",
                required_bytes: measured,
                available_bytes: request.limits().max_resident_bytes(),
            });
        }
        let mut capabilities = self.registration.capabilities().clone();
        capabilities.context_limit_tokens = capabilities
            .context_limit_tokens
            .min(data.config.max_context);
        Ok(Box::new(Qwen3Loaded {
            model,
            stream: execution_stream,
            artifact: request.artifact().id().clone(),
            instance: ModelInstanceId::new(),
            registration: self.registration.clone(),
            capabilities,
            memory: BufferMemoryUsage::new(measured, payload)?,
            vocabulary: data.config.vocab_size,
            max_prefill_tokens: self.limits.prefill_chunk_tokens.get(),
            logits_payload_bytes: logits_payload_bound(&data.config)?,
        }))
    }
}

impl RuntimeBackend for Qwen3Backend {
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        Box::pin(async move {
            let resolved = registry.resolve_profile(context.profile_id())?;
            let data = self.data(resolved.artifact())?;
            let profile = resolved.profile();
            let binding = RenderingBinding {
                profile_id: context.profile_id(),
                artifact_id: &data.specification.id,
                template: CHAT_TEMPLATE,
                template_sha256: data.specification.template_sha256,
                artifact_manifest_sha256: data.manifest_sha256,
            };
            let limits = RenderingLimits::new(
                self.limits.max_messages,
                self.limits.max_input_bytes,
                CHAT_TEMPLATE.len(),
                self.limits.max_rendered_bytes,
                profile.context_limit_tokens(),
            )?;
            let rendered = rendering::render_and_count(
                &context,
                &binding,
                data.tokenizer.as_ref(),
                &qwen3_rendering_options(),
                limits,
            )
            .map_err(|error| match error {
                // Keep the profile's actual reserve/margin in context errors.
                EngineError::ContextLimitExceeded { input_tokens, .. } => {
                    EngineError::ContextLimitExceeded {
                        input_tokens,
                        output_reserve_tokens: profile.max_output_tokens(),
                        safety_margin_tokens: profile.safety_margin_tokens(),
                        limit_tokens: profile.context_limit_tokens(),
                    }
                }
                other => other,
            })?;
            profile.check_context(rendered.token_count())?;
            let vocabulary = as_u64(data.config.vocab_size)?;
            if rendered
                .token_ids()
                .iter()
                .any(|&id| u64::from(id) >= vocabulary)
            {
                return Err(validation(
                    Some(resolved.artifact().id()),
                    "tokenizer produced an ID outside the loaded model vocabulary",
                ));
            }
            Ok(rendered)
        })
    }

    fn plan<'a>(
        &'a self,
        registry: &'a Registry,
        profile: &'a ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'a, RuntimePlan> {
        Box::pin(async move {
            let resolved = registry.resolve_profile(profile)?;
            let data = self.data(resolved.artifact())?;
            resolved.profile().check_context(input_tokens)?;
            let capacity = input_tokens
                .checked_add(resolved.profile().max_output_tokens())
                .and_then(|v| v.checked_add(resolved.profile().safety_margin_tokens()))
                .ok_or_else(|| internal("Qwen3 context arithmetic overflow"))?;
            let load_peak = sum(&[
                data.source_bytes,
                data.decoded_bound,
                data.resident_bound,
                self.limits.loader_headroom_bytes,
            ])?;
            Ok(RuntimePlan {
                load_limits: ModelLoadLimits::new(
                    data.source_bytes,
                    data.resident_bound,
                    load_peak,
                )?,
                cache: CacheAllocationRequest::new(
                    cache_mode()?,
                    capacity,
                    sum(&[
                        cache_bound(&data.config, capacity)?,
                        product(self.limits.execution_headroom_bytes, 2)?,
                    ])?,
                )?,
                temporary_bytes: workspace_bound(
                    &data.config,
                    capacity,
                    self.limits.prefill_chunk_tokens.get(),
                    self.limits.max_token_bytes,
                )?,
                output_bytes: output_bytes(self.limits.generation)?,
                headroom_bytes: self.limits.execution_headroom_bytes,
                prefill_chunk_tokens: self.limits.prefill_chunk_tokens,
            })
        })
    }

    fn verify_artifact(
        &self,
        artifact: &ArtifactRegistration,
        control: &ExecutionControl,
    ) -> Result<(), EngineError> {
        let data = self.data(artifact)?;
        for (name, pin) in &data.specification.files {
            control.check_cancelled()?;
            let path = component_path(&data.specification, name)?;
            let mut file = File::open(&path)
                .map_err(|e| EngineError::checkpoint_io("open pinned artifact", &path, e))?;
            let metadata = file
                .metadata()
                .map_err(|e| EngineError::checkpoint_io("stat pinned artifact", &path, e))?;
            if !metadata.is_file() || metadata.len() != pin.bytes {
                return Err(validation(
                    Some(artifact.id()),
                    "pinned file type or length changed",
                ));
            }
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(self.limits.verification_chunk_bytes)
                .map_err(|_| {
                    allocation(
                        "artifact hash buffer",
                        Some(self.limits.verification_chunk_bytes),
                    )
                })?;
            buffer.resize(self.limits.verification_chunk_bytes, 0);
            let mut hash = Sha256::new();
            let mut total = 0_u64;
            loop {
                control.check_cancelled()?;
                let count = file
                    .read(&mut buffer)
                    .map_err(|e| EngineError::checkpoint_io("hash pinned artifact", &path, e))?;
                if count == 0 {
                    break;
                }
                total = sum(&[total, as_u64(count)?])?;
                if total > pin.bytes {
                    return Err(validation(
                        Some(artifact.id()),
                        "pinned file grew while hashing",
                    ));
                }
                hash.update(&buffer[..count]);
            }
            let observed: [u8; 32] = hash.finalize().into();
            if total != pin.bytes || observed != pin.sha256 {
                return Err(validation(
                    Some(artifact.id()),
                    "pinned file digest changed",
                ));
            }
        }
        control.check_cancelled()
    }

    fn decoder(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        let resolved = registry.resolve_profile(profile)?;
        let data = self.data(resolved.artifact())?;
        Ok(Box::new(Qwen3TextDecoder {
            tokenizer: Arc::clone(&data.tokenizer),
            special_tokens: Arc::clone(&data.special_tokens),
            byte_decoder: Arc::clone(&data.byte_decoder),
            pending: Vec::new(),
            max_token_bytes: self.limits.max_token_bytes,
            finished: false,
        }))
    }

    fn incremental_mlx_bytes(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        plan: &RuntimePlan,
        residency: crate::runtime::RuntimeResidency,
    ) -> Result<Option<u64>, EngineError> {
        let resolved = registry.resolve_profile(profile)?;
        self.data(resolved.artifact())?;
        if !residency.reusable_model && residency.reusable_cache_buffer_bytes != 0 {
            return Err(EngineError::invalid_request(
                "residency",
                "foreign cache cannot offset MLX demand",
            ));
        }
        let weights = if residency.reusable_model {
            0
        } else {
            plan.load_limits.max_resident_bytes()
        };
        // The cache allowance contains old/new overlap and padding. Deduct
        // only the measured compatible storage already counted as active.
        let cache = plan
            .cache
            .max_cache_bytes()
            .saturating_sub(residency.reusable_cache_buffer_bytes);
        // Workspace/headroom is deliberately treated as a conservative
        // native upper allowance, even though some small components are
        // host bookkeeping. Host checkpoint/decoded-weight loading peaks
        // and output-queue reservations are NOT counted as native weights.
        Ok(Some(sum(&[
            weights,
            cache,
            plan.temporary_bytes,
            plan.headroom_bytes,
        ])?))
    }

    fn eos_tokens(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Vec<u32>, EngineError> {
        Ok(self
            .data(registry.resolve_profile(profile)?.artifact())?
            .eos
            .clone())
    }
}

struct Qwen3Loaded {
    model: MlxQwen3Model,
    stream: Rc<Stream>,
    artifact: ArtifactId,
    instance: ModelInstanceId,
    registration: FamilyRegistration,
    capabilities: Capabilities,
    // Tracks float32 tensor payload, not allocator cache or process RSS.
    memory: BufferMemoryUsage,
    vocabulary: usize,
    max_prefill_tokens: usize,
    logits_payload_bytes: u64,
}

impl LoadedModel for Qwen3Loaded {
    fn instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }
    fn artifact_id(&self) -> &ArtifactId {
        &self.artifact
    }
    fn registration(&self) -> &FamilyRegistration {
        &self.registration
    }
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        Ok(self.memory)
    }
    fn autoregressive(&mut self) -> Option<&mut dyn AutoregressiveModel> {
        Some(self)
    }
}

impl Drop for Qwen3Loaded {
    fn drop(&mut self) {
        // Normal model operations already synchronize on every return.
        // This is best-effort teardown, not native device-failure recovery.
        let _ = self.stream.synchronize();
    }
}

struct Qwen3NativeCache {
    inner: MlxKvCache,
    owner: ModelInstanceId,
    mlx_owner: MlxModelId,
    mode: CacheMode,
    capacity: usize,
    stream: Rc<Stream>,
}

impl FamilyCache for Qwen3NativeCache {
    fn owner(&self) -> &ModelInstanceId {
        &self.owner
    }
    fn representation(&self) -> &CacheRepresentationId {
        &self.mode.representation
    }
    fn mode(&self) -> &CacheMode {
        &self.mode
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn retained_tokens(&self) -> usize {
        self.inner.len()
    }
    fn capacity_tokens(&self) -> usize {
        self.capacity
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        // Measured current backing buffers, not allocator cache or
        // older external views. Used bytes remain logical KV payload.
        let allocated = self
            .inner
            .buffer_bytes()
            .map_err(|e| cache_error(e, "measure committed MLX Qwen KV buffers"))?;
        let spec = self.inner.spec();
        let used = product(
            product(
                product(
                    product(as_u64(spec.layers)?, as_u64(spec.kv_heads)?)?,
                    as_u64(spec.head_dim)?,
                )?,
                as_u64(self.inner.len())?,
            )?,
            8,
        )?;
        BufferMemoryUsage::new(allocated, used)
    }
    fn supports_truncation(&self) -> bool {
        true
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner
            .reset(&self.mlx_owner)
            .map_err(|e| cache_error(e, "reset MLX Qwen KV"))
    }
    fn truncate(&mut self, tokens: usize) -> Result<(), EngineError> {
        self.inner
            .truncate(&self.mlx_owner, tokens)
            .map_err(|e| cache_error(e, "truncate MLX Qwen KV"))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Drop for Qwen3NativeCache {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl Qwen3Loaded {
    fn owned_cache<'a>(
        &self,
        cache: &'a mut dyn FamilyCache,
    ) -> Result<&'a mut Qwen3NativeCache, EngineError> {
        if cache.owner() != &self.instance || cache.mode() != &cache_mode()? {
            return Err(EngineError::invalid_request(
                "cache",
                "cache belongs to another model or representation",
            ));
        }
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen3NativeCache>()
            .ok_or_else(|| {
                EngineError::invalid_request("cache", "cache has a different family type")
            })?;
        if !cache.inner.owner().same_instance(self.model.instance_id())
            || !cache.mlx_owner.same_instance(self.model.instance_id())
        {
            return Err(EngineError::invalid_request(
                "cache",
                "native MLX cache belongs to another model load",
            ));
        }
        Ok(cache)
    }
}

impl AutoregressiveModel for Qwen3Loaded {
    fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }
    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        control.check_cancelled()?;
        if request.mode() != &cache_mode()? {
            return Err(EngineError::unsupported(
                "cache mode",
                "only full MLX float32 KV is implemented",
            ));
        }
        if request.capacity_tokens() > self.capabilities.context_limit_tokens {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: request.capacity_tokens(),
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: self.capabilities.context_limit_tokens,
            });
        }
        let required = cache_bound(self.model.config(), request.capacity_tokens())?;
        if required > request.max_cache_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "MLX Qwen KV update overlap",
                required_bytes: required,
                available_bytes: request.max_cache_bytes(),
            });
        }
        let inner = self
            .model
            .new_cache(
                request.capacity_tokens(),
                request.max_cache_bytes(),
                self.stream.as_ref(),
            )
            .map_err(|e| mlx_model_error(e, control, "allocate MLX Qwen KV"))?;
        control.check_cancelled()?;
        Ok(Box::new(Qwen3NativeCache {
            inner,
            owner: self.instance.clone(),
            mlx_owner: self.model.instance_id().clone(),
            mode: request.mode().clone(),
            capacity: request.capacity_tokens(),
            stream: Rc::clone(&self.stream),
        }))
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        let cache = self.owned_cache(cache)?;
        control.check_cancelled()?;
        if tokens.is_empty() || tokens.len() > self.max_prefill_tokens {
            return Err(EngineError::invalid_request(
                "tokens",
                "prefill must be nonempty and fit the planned chunk bound",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| internal("prefill position overflow"))?;
        if end > cache.capacity {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.capacity,
            });
        }
        let result = self
            .model
            .prefill_cancellable(
                tokens,
                &mut cache.inner,
                self.logits_payload_bytes,
                self.stream.as_ref(),
                || control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "MLX Qwen prefill"))
            .and_then(|output| TokenLogits::new(output.into_values()))
            .and_then(|logits| PrefillOutput::new(tokens.len(), logits));
        if result.is_err() {
            let _ = cache.inner.reset(&cache.mlx_owner);
        }
        result
    }
    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        let cache = self.owned_cache(cache)?;
        control.check_cancelled()?;
        if cache.inner.is_empty() {
            return Err(EngineError::invalid_request(
                "cache",
                "prefill before decoding a continuation",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(1)
            .ok_or_else(|| internal("decode position overflow"))?;
        if end > cache.capacity {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.capacity,
            });
        }
        // Use the cancellable one-token path, not the noncancellable
        // convenience decode method. It has identical model semantics.
        let result = self
            .model
            .prefill_cancellable(
                &[token],
                &mut cache.inner,
                self.logits_payload_bytes,
                self.stream.as_ref(),
                || control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "MLX Qwen decode"))
            .and_then(|output| TokenLogits::new(output.into_values()));
        if result.is_err() {
            let _ = cache.inner.reset(&cache.mlx_owner);
        }
        result
    }
}

struct Qwen3TextDecoder {
    tokenizer: Arc<PinnedTokenizer>,
    special_tokens: Arc<BTreeSet<u32>>,
    byte_decoder: Arc<BTreeMap<char, u8>>,
    pending: Vec<u8>,
    max_token_bytes: usize,
    finished: bool,
}

impl TextDecoder for Qwen3TextDecoder {
    fn push(&mut self, token: u32) -> Result<String, EngineError> {
        if self.finished {
            return Err(internal("token supplied after decoder completion"));
        }
        if self.special_tokens.contains(&token) {
            return Ok(String::new());
        }
        let spelling = self
            .tokenizer
            .tokenizer()
            .id_to_token(token)
            .ok_or_else(|| {
                execution_error(
                    "detokenize",
                    "model selected an ID absent from the tokenizer",
                )
            })?;
        if spelling.len() > self.max_token_bytes {
            return Err(execution_error(
                "detokenize",
                "token exceeds the pinned decoder bound",
            ));
        }
        self.pending
            .try_reserve_exact(spelling.chars().count())
            .map_err(|_| allocation("incremental UTF-8 bytes", Some(spelling.len())))?;
        for ch in spelling.chars() {
            self.pending
                .push(*self.byte_decoder.get(&ch).ok_or_else(|| {
                    execution_error(
                        "detokenize",
                        "token is outside the verified ByteLevel alphabet",
                    )
                })?);
        }
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => {
                return Err(execution_error(
                    "detokenize",
                    "model output contains invalid UTF-8 bytes",
                ));
            }
        };
        if valid == 0 {
            return Ok(String::new());
        }
        let suffix = self.pending.split_off(valid);
        let complete = std::mem::replace(&mut self.pending, suffix);
        String::from_utf8(complete).map_err(|_| internal("validated UTF-8 conversion failed"))
    }

    fn finish(&mut self) -> Result<String, EngineError> {
        if self.finished {
            return Ok(String::new());
        }
        self.finished = true;
        if !self.pending.is_empty() {
            return Err(execution_error(
                "detokenize",
                "generation ended with an incomplete UTF-8 sequence",
            ));
        }
        Ok(String::new())
    }
}

fn registration(
    context_limit: usize,
    execution: Qwen3Execution,
) -> Result<FamilyRegistration, EngineError> {
    FamilyRegistration::new(
        // Numerical execution and KV layout differ from host-reference
        // v2: canonical state identity must use the new adapter version.
        FamilyVersion::new(qwen3::register::IMPL_VERSION, 3, 0),
        qwen3::register::ARCHITECTURES
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        Capabilities {
            task: "text-generation".to_owned(),
            family: "qwen3".to_owned(),
            backend: execution.backend(),
            weight_representations: vec![
                WeightRepresentation::F32,
                WeightRepresentation::Affine4 {
                    group_size: NonZeroUsize::new(64).expect("64"),
                },
            ],
            cache_modes: vec![cache_mode()?],
            context_limit_tokens: context_limit,
            supports_streaming: true,
            supports_incremental_kv: true,
            supports_prefix_reuse: false,
            decoding_modes: vec![DecodingCapability::Greedy],
        },
    )
}

fn cache_mode() -> Result<CacheMode, EngineError> {
    Ok(CacheMode {
        representation: CacheRepresentationId::new(QWEN3_MLX_CACHE_REPRESENTATION)?,
        kind: CacheKind::Full,
    })
}

fn component_path(spec: &Qwen3Artifact, name: &str) -> Result<PathBuf, EngineError> {
    let path = Path::new(name);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(validation(
            Some(&spec.id),
            "manifest paths must be simple basenames",
        ));
    }
    let supplied = spec.directory.join(path);
    let resolved = supplied
        .canonicalize()
        .map_err(|e| EngineError::checkpoint_io("canonicalize pinned component", &supplied, e))?;
    if resolved.parent() != Some(spec.directory.as_path()) {
        return Err(validation(
            Some(&spec.id),
            "artifact component escapes its directory",
        ));
    }
    Ok(resolved)
}

fn read_metadata(spec: &Qwen3Artifact, name: &str, limit: usize) -> Result<Vec<u8>, EngineError> {
    let pin = spec
        .files
        .get(name)
        .ok_or_else(|| validation(Some(&spec.id), "required metadata lacks a trusted pin"))?;
    if pin.bytes > as_u64(limit)? {
        return Err(EngineError::MemoryLimitExceeded {
            operation: "read Qwen3 metadata",
            required_bytes: pin.bytes,
            available_bytes: as_u64(limit)?,
        });
    }
    let path = component_path(spec, name)?;
    let mut file = File::open(&path)
        .map_err(|e| EngineError::checkpoint_io("open Qwen3 metadata", &path, e))?;
    let meta = file
        .metadata()
        .map_err(|e| EngineError::checkpoint_io("stat Qwen3 metadata", &path, e))?;
    if !meta.is_file() || meta.len() != pin.bytes {
        return Err(validation(
            Some(&spec.id),
            "metadata length or type disagrees with its pin",
        ));
    }
    let length = as_usize(pin.bytes)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| allocation("Qwen3 metadata", Some(length)))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .map_err(|e| EngineError::checkpoint_io("read Qwen3 metadata", &path, e))?;
    let mut probe = [0_u8; 1];
    if file
        .read(&mut probe)
        .map_err(|e| EngineError::checkpoint_io("check Qwen3 metadata length", &path, e))?
        != 0
        || digest(&bytes) != pin.sha256
    {
        return Err(validation(
            Some(&spec.id),
            "metadata changed or failed its trusted digest",
        ));
    }
    Ok(bytes)
}

fn parse_json(path: &Path, bytes: &[u8]) -> Result<Value, EngineError> {
    serde_json::from_slice(bytes)
        .map_err(|e| EngineError::checkpoint_format("json", Some(path.to_owned()), e))
}

fn parse_config(value: &Value) -> Result<Qwen3Config, EngineError> {
    if value.get("model_type").and_then(Value::as_str) != Some("qwen3") {
        return Err(EngineError::unsupported(
            "architecture",
            "only dense Qwen3 causal LM is implemented",
        ));
    }
    let architectures = value
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| validation(None, "missing architectures"))?;
    if architectures.len() != 1 || architectures[0].as_str() != Some("Qwen3ForCausalLM") {
        return Err(EngineError::unsupported(
            "architecture",
            "expected Qwen3ForCausalLM",
        ));
    }
    for key in ["rope_scaling"] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return Err(EngineError::unsupported(
                "model configuration",
                format!("{key} is not implemented"),
            ));
        }
    }
    for key in ["attention_bias", "mlp_bias", "use_sliding_window"] {
        if value.get(key).is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(EngineError::unsupported(
                "model configuration",
                format!("{key} must be false when present"),
            ));
        }
    }
    if value
        .get("hidden_act")
        .is_some_and(|v| v.as_str() != Some("silu"))
    {
        return Err(EngineError::unsupported(
            "hidden_act",
            "only silu is implemented",
        ));
    }
    if value.get("rms_norm_eps").is_some_and(|v| {
        v.as_f64()
            .is_none_or(|n| !n.is_finite() || (n - 1e-6).abs() > 1e-12)
    }) {
        return Err(EngineError::unsupported(
            "rms_norm_eps",
            "only 1e-6 is implemented",
        ));
    }
    let dimension = |name: &str| -> Result<usize, EngineError> {
        value
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                validation(
                    None,
                    format!("missing or invalid positive dimension: {name}"),
                )
            })
    };
    let theta = value
        .get("rope_theta")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| validation(None, "missing rope_theta"))? as f32;
    if !theta.is_finite() || theta <= 0.0 {
        return Err(validation(None, "rope_theta is not representable as f32"));
    }
    let config = Qwen3Config {
        hidden_size: dimension("hidden_size")?,
        num_layers: dimension("num_hidden_layers")?,
        q_heads: dimension("num_attention_heads")?,
        kv_heads: dimension("num_key_value_heads")?,
        head_dim: dimension("head_dim")?,
        intermediate_size: dimension("intermediate_size")?,
        vocab_size: dimension("vocab_size")?,
        max_context: dimension("max_position_embeddings")?,
        rope_theta: theta,
        tie_embeddings: value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .ok_or_else(|| validation(None, "tie_word_embeddings must be explicit"))?,
    };
    if config.q_heads % config.kv_heads != 0
        || config.head_dim % 2 != 0
        || as_u64(config.vocab_size)? > u64::from(u32::MAX) + 1
    {
        return Err(validation(
            None,
            "invalid attention dimensions or token-ID range",
        ));
    }
    for dimension in [
        config.hidden_size,
        config.num_layers,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.vocab_size,
        config.max_context,
    ] {
        if i32::try_from(dimension).is_err() {
            return Err(validation(None, "MLX model dimension exceeds C int"));
        }
    }
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    if q > i32::MAX as u64 || kv > i32::MAX as u64 {
        return Err(validation(None, "MLX projection width exceeds C int"));
    }
    resident_bound(&config)?;
    cache_bound(&config, 1)?;
    Ok(config)
}

fn parse_eos(value: Option<&Value>) -> Result<Vec<u32>, EngineError> {
    let value = value.ok_or_else(|| {
        validation(
            None,
            "EOS must be explicit in pinned config or generation_config",
        )
    })?;
    let raw = match value {
        Value::Array(values) => values.clone(),
        Value::Number(_) => vec![value.clone()],
        _ => {
            return Err(validation(
                None,
                "EOS must be an integer or an explicit integer array",
            ));
        }
    };
    let mut eos = Vec::new();
    for value in raw {
        let id = value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| validation(None, "invalid EOS token ID"))?;
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    Ok(eos)
}

fn qwen3_rendering_options() -> RenderingOptions<'static> {
    RenderingOptions {
        add_generation_prompt: true,
        enable_thinking: Some(false),
        add_special_tokens: false,
        // Match the previous Environment defaults for this pinned template.
        keep_trailing_newline: false,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: QWEN3_RESERVED_SPELLINGS,
    }
}

fn byte_decoder_map() -> BTreeMap<char, u8> {
    let mut retained = Vec::new();
    retained.extend(33_u16..=126);
    retained.extend(161_u16..=172);
    retained.extend(174_u16..=255);
    let mut map = BTreeMap::new();
    let mut extra = 0_u32;
    for byte in 0_u16..=255 {
        let code = if retained.contains(&byte) {
            u32::from(byte)
        } else {
            let code = 256 + extra;
            extra += 1;
            code
        };
        if let Some(character) = char::from_u32(code) {
            map.insert(character, byte as u8);
        }
    }
    map
}

/// Default Affine4 group size used for resident estimates when the checkpoint
/// has not been opened yet (MLX Qwen3-4bit community artifacts use 64).
const AFFINE4_GROUP_SIZE: u64 = 64;

fn layer_projection_elements(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let i = as_u64(config.intermediate_size)?;
    sum(&[
        product(2, product(q, h)?)?,  // q, o
        product(2, product(kv, h)?)?, // k, v
        product(3, product(i, h)?)?,  // gate, up, down
    ])
}

fn norm_bytes(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let head = as_u64(config.head_dim)?;
    let per_layer = sum(&[
        product(2, h)?,    // input + post attention
        product(2, head)?, // q/k norm
    ])?;
    let elements = sum(&[
        product(as_u64(config.num_layers)?, per_layer)?,
        h, // final layernorm
    ])?;
    product(elements, 4)
}

/// Host-side Affine4 payload estimate (packed codes + scales/biases for every
/// matrix, including embed/lm_head). Used for loader charge before MLX upload.
fn host_affine4_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let v = as_u64(config.vocab_size)?;
    let matrix_elements = sum(&[
        product(v, h)?, // embed
        product(
            as_u64(config.num_layers)?,
            layer_projection_elements(config)?,
        )?,
        if config.tie_embeddings {
            0
        } else {
            product(v, h)?
        },
    ])?;
    let matrix_bytes = affine4_matrix_bytes(matrix_elements, AFFINE4_GROUP_SIZE)?;
    let workspace = product(matrix_bytes / 32 + 1, 1)?;
    sum(&[norm_bytes(config)?, matrix_bytes, workspace])
}

/// MLX resident estimate after upload: projections stay Affine4; embed/lm_head
/// are dequantized once to dense float32 (oMLX-method quantized-resident).
fn resident_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let v = as_u64(config.vocab_size)?;
    let proj_elements = product(
        as_u64(config.num_layers)?,
        layer_projection_elements(config)?,
    )?;
    let proj_bytes = affine4_matrix_bytes(proj_elements, AFFINE4_GROUP_SIZE)?;
    let embed_bytes = product(product(v, h)?, 4)?;
    let lm_head_bytes = if config.tie_embeddings {
        0
    } else {
        embed_bytes
    };
    let workspace = product(proj_bytes / 32 + 1, 1)?;
    sum(&[
        norm_bytes(config)?,
        proj_bytes,
        embed_bytes,
        lm_head_bytes,
        workspace,
    ])
}

fn affine4_matrix_bytes(elements: u64, group_size: u64) -> Result<u64, EngineError> {
    if group_size == 0 {
        return Err(EngineError::invalid_request(
            "group_size",
            "affine4 group size must be positive",
        ));
    }
    // packed u32: 8 nibbles per word → elements/8 words * 4 bytes = elements/2
    let packed = elements / 2;
    // scales + biases: 2 * (elements / group_size) * 4 bytes
    let params = product(elements / group_size, 8)?;
    sum(&[packed, params])
}

// Dense matrix payload, shared by host decoding and MLX residency.
fn dense_weight_payload_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let embedding = product(as_u64(config.vocab_size)?, as_u64(config.hidden_size)?)?;
    let projections = product(
        as_u64(config.num_layers)?,
        layer_projection_elements(config)?,
    )?;
    let elements = sum(&[
        embedding,
        projections,
        if config.tie_embeddings { 0 } else { embedding },
    ])?;
    sum(&[product(elements, 4)?, norm_bytes(config)?])
}

fn dense_decoded_weight_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    sum(&[
        dense_weight_payload_bound(config)?,
        product(
            as_u64(config.num_layers)?,
            as_u64(std::mem::size_of::<LayerWeights>())?,
        )?,
    ])
}

fn decoded_weight_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    // Loader charges host Affine4 (or f32) buffers before MLX upload.
    sum(&[
        host_affine4_bound(config)?,
        product(
            as_u64(config.num_layers)?,
            as_u64(std::mem::size_of::<LayerWeights>())?,
        )?,
    ])
}

fn weight_memory_add_f32(
    allocated: &mut u64,
    used: &mut u64,
    values: &Vec<f32>,
) -> Result<(), EngineError> {
    *allocated = sum(&[*allocated, product(as_u64(values.capacity())?, 4)?])?;
    *used = sum(&[*used, product(as_u64(values.len())?, 4)?])?;
    Ok(())
}

fn weight_memory_add_bytes(
    allocated: &mut u64,
    used: &mut u64,
    bytes: u64,
) -> Result<(), EngineError> {
    *allocated = sum(&[*allocated, bytes])?;
    *used = sum(&[*used, bytes])?;
    Ok(())
}

fn weight_memory(weights: &Qwen3Weights) -> Result<BufferMemoryUsage, EngineError> {
    let mut allocated = product(
        as_u64(weights.layers.capacity())?,
        as_u64(std::mem::size_of::<LayerWeights>())?,
    )?;
    let mut used = product(
        as_u64(weights.layers.len())?,
        as_u64(std::mem::size_of::<LayerWeights>())?,
    )?;
    weight_memory_add_bytes(
        &mut allocated,
        &mut used,
        as_u64(weights.embed_tokens.resident_bytes())?,
    )?;
    weight_memory_add_f32(&mut allocated, &mut used, &weights.final_layernorm)?;
    if let Some(head) = &weights.lm_head {
        weight_memory_add_bytes(&mut allocated, &mut used, as_u64(head.resident_bytes())?)?;
    }
    for layer in &weights.layers {
        weight_memory_add_f32(&mut allocated, &mut used, &layer.input_layernorm)?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.q_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.k_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.v_proj.resident_bytes())?,
        )?;
        weight_memory_add_f32(&mut allocated, &mut used, &layer.q_norm)?;
        weight_memory_add_f32(&mut allocated, &mut used, &layer.k_norm)?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.o_proj.resident_bytes())?,
        )?;
        weight_memory_add_f32(&mut allocated, &mut used, &layer.post_attention_layernorm)?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.gate_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.up_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.down_proj.resident_bytes())?,
        )?;
    }
    BufferMemoryUsage::new(allocated, used)
}

fn cache_spec(config: &Qwen3Config, capacity: usize) -> KvCacheSpec {
    KvCacheSpec {
        layers: config.num_layers,
        batch: 1,
        kv_heads: config.kv_heads,
        capacity_tokens: capacity,
        head_dim: config.head_dim,
    }
}

fn cache_bound(config: &Qwen3Config, capacity: usize) -> Result<u64, EngineError> {
    cache_spec(config, capacity)
        .payload_plan()
        .map(|plan| plan.update_overlap_bytes)
        .map_err(|e| cache_error(e, "plan MLX Qwen KV overlap"))
}

fn logits_payload_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    // The model's last-row API admits projection, possible contiguous
    // output, and host result. Four rows' bytes leave capacity headroom.
    product(as_u64(config.vocab_size)?, 16)
}

/// Conservative f32 graph-payload allowance for the selected chunk bound.
/// Counts per-layer projection/normalization/MLP intermediates, possible
/// attention score/softmax buffers, and prefix-layout temporaries.
///
/// KV old/new overlap is separately reserved in cache_bound, not here.
/// MLX allocator internals, kernels' hidden workspace, caches and alignment
/// are not a certified hard bound: execution_headroom_bytes must be sized
/// and calibrated for the supported MLX pin/device/model. This adapter does
/// not claim OS OOM prevention or allocator-pressure discovery.
fn workspace_bound(
    config: &Qwen3Config,
    capacity: usize,
    chunk_tokens: usize,
    token_bytes: usize,
) -> Result<u64, EngineError> {
    let layers = as_u64(config.num_layers)?;
    let t = as_u64(chunk_tokens.min(capacity).max(1))?;
    let s = as_u64(capacity)?;
    let h = as_u64(config.hidden_size)?;
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let i = as_u64(config.intermediate_size)?;
    let per_layer_elements = sum(&[
        product(12, product(t, h)?)?,
        product(12, product(t, q)?)?,
        product(12, product(t, kv)?)?,
        product(10, product(t, i)?)?,
        product(8, product(product(as_u64(config.q_heads)?, t)?, s)?)?,
        product(4, product(s, kv)?)?,
    ])?;
    sum(&[
        product(product(layers, per_layer_elements)?, 4)?,
        product(product(t, h)?, 16)?,
        logits_payload_bound(config)?,
        product(as_u64(token_bytes)?, 8)?,
        product(sum(&[layers, 64])?, 256)?,
        32,
    ])
}

fn output_bytes(limits: GenerationLimits) -> Result<u64, EngineError> {
    sum(&[
        as_u64(limits.max_buffered_text_bytes())?,
        as_u64(limits.max_text_delta_bytes())?,
        product(
            as_u64(limits.event_capacity())?,
            as_u64(std::mem::size_of::<String>())?,
        )?,
    ])
}
fn sum(values: &[u64]) -> Result<u64, EngineError> {
    values.iter().try_fold(0_u64, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| internal("Qwen3 size arithmetic overflow"))
    })
}
fn product(a: u64, b: u64) -> Result<u64, EngineError> {
    a.checked_mul(b)
        .ok_or_else(|| internal("Qwen3 size arithmetic overflow"))
}
fn as_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value).map_err(|_| internal("Qwen3 size exceeds u64"))
}
fn as_usize(value: u64) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| internal("Qwen3 size exceeds platform usize"))
}
fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn allocation(operation: &'static str, bytes: Option<usize>) -> EngineError {
    EngineError::AllocationFailed {
        operation,
        requested_bytes: bytes.and_then(|v| u64::try_from(v).ok()),
    }
}
fn validation(id: Option<&ArtifactId>, message: impl Into<String>) -> EngineError {
    EngineError::ArtifactValidationFailed {
        artifact_id: id.map(|id| id.as_str().to_owned()),
        message: message.into(),
    }
}
fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        message: message.to_owned(),
    }
}
fn execution_error(stage: &'static str, message: impl Into<String>) -> EngineError {
    EngineError::ExecutionFailed {
        stage,
        message: message.into(),
    }
}
fn cancelled_error(control: &ExecutionControl) -> EngineError {
    EngineError::Cancelled {
        request_id: control.request_id().as_str().to_owned(),
    }
}

fn cache_error(error: KvCacheError, stage: &'static str) -> EngineError {
    match error {
        KvCacheError::PayloadLimit {
            required_bytes,
            limit_bytes,
        } => EngineError::MemoryLimitExceeded {
            operation: stage,
            required_bytes,
            available_bytes: limit_bytes,
        },
        KvCacheError::ForeignModel => {
            EngineError::invalid_request("cache", "native cache belongs to another model")
        }
        KvCacheError::InvalidInput(message) => EngineError::invalid_request("cache", message),
        KvCacheError::AllocationFailed(_) => allocation(stage, None),
        other => execution_error(stage, other.to_string()),
    }
}

fn mlx_model_error(
    error: MlxQwenError,
    control: &ExecutionControl,
    stage: &'static str,
) -> EngineError {
    match error {
        MlxQwenError::Cancelled | MlxQwenError::Model(ModelError::Cancelled) => {
            cancelled_error(control)
        }
        MlxQwenError::PayloadLimit {
            operation,
            required_bytes,
            limit_bytes,
        } => EngineError::MemoryLimitExceeded {
            operation,
            required_bytes,
            available_bytes: limit_bytes,
        },
        MlxQwenError::AllocationFailed(_) => allocation(stage, None),
        MlxQwenError::Operation(OperationError::AllocationFailed(_)) => allocation(stage, None),
        MlxQwenError::Operation(OperationError::InvalidInput(message)) => {
            EngineError::invalid_request("MLX operation", message)
        }
        MlxQwenError::Cache(error) => cache_error(error, stage),
        MlxQwenError::Model(error) => model_error(error, stage),
        other => execution_error(stage, other.to_string()),
    }
}

fn model_error(error: ModelError, stage: &'static str) -> EngineError {
    match error {
        ModelError::MemoryLimit { required, limit } => {
            match (u64::try_from(required), u64::try_from(limit)) {
                (Ok(required_bytes), Ok(available_bytes)) => EngineError::MemoryLimitExceeded {
                    operation: stage,
                    required_bytes,
                    available_bytes,
                },
                _ => internal("model memory sizes exceed u64"),
            }
        }
        ModelError::AllocationFailed => allocation(stage, None),
        ModelError::ContextTooLong { maximum, got } => EngineError::ContextLimitExceeded {
            input_tokens: got,
            output_reserve_tokens: 0,
            safety_margin_tokens: 0,
            limit_tokens: maximum,
        },
        ModelError::InvalidToken {
            position,
            token,
            vocab_size,
        } => EngineError::invalid_request(
            "tokens",
            format!("token {token} at position {position} exceeds vocabulary {vocab_size}"),
        ),
        ModelError::EmptyInput => EngineError::invalid_request("tokens", "must not be empty"),
        ModelError::CheckpointIo {
            operation,
            path,
            kind,
            message,
        } => EngineError::checkpoint_io(operation, path, io::Error::new(kind, message)),
        ModelError::CheckpointFormat { format, message } => EngineError::checkpoint_format(
            format,
            None,
            io::Error::new(io::ErrorKind::InvalidData, message),
        ),
        ModelError::CheckpointValidation(message) | ModelError::Checkpoint(message) => {
            validation(None, message)
        }
        ModelError::InvalidConfig(message) => {
            EngineError::unsupported("Qwen3 configuration", message)
        }
        ModelError::DimensionOverflow(_) => internal("Qwen3 dimension arithmetic overflow"),
        other => execution_error(stage, other.to_string()),
    }
}

#[cfg(test)]
mod shared_rendering_wiring_tests {
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
        assert!(cols > 0 && cols % 8 == 0 && group_size > 0 && cols % group_size == 0);
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
                    TensorView::new(record.dtype, record.shape.clone(), &record.bytes)
                        .expect("tensor"),
                )
            })
            .collect();
        serialize(views, &None).expect("actual safetensors checkpoint")
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
            } => {
                assert_eq!(operation, "render chat template output");
                assert!(required_bytes > 1);
                assert_eq!(available_bytes, 1);
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
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
            while let Some(event) =
                block_on(std::future::poll_fn(|cx| events.as_mut().poll_next(cx)))
            {
                assert!(!terminal, "event after terminal");
                match event {
                    crate::types::GenerationEvent::TextDelta { text: delta } => {
                        text.push_str(&delta)
                    }
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
            // MLX dequants embed/lm_head to f32, so resident ≥ host Affine4.
            assert!(resident >= host, "resident={resident} host={host}");
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
}

// ALLOY_MEASURED_BUFFERS_V1

// ALLOY_PRESSURE_STARTUP_INTEGRATION_V1

#[cfg(test)]
mod dense_weight_planning_regressions {
    use super::*;

    #[test]
    fn dense_fixture_payload_and_host_storage_fit_the_dense_bounds() {
        for tied in [true, false] {
            let mut config = Qwen3Config::tiny_fixture();
            config.vocab_size = 258;
            config.tie_embeddings = tied;

            let weights = qwen3::fixture::generate_fixture(&config);
            let measured = weight_memory(&weights).expect("fixture accounting");
            let headers = product(
                as_u64(weights.layers.len()).expect("layer count"),
                as_u64(std::mem::size_of::<LayerWeights>()).expect("layer size"),
            )
            .expect("headers");
            let payload = measured.used_bytes() - headers;

            assert_eq!(
                dense_weight_payload_bound(&config).expect("dense payload"),
                payload,
            );
            assert!(
                dense_decoded_weight_bound(&config).expect("dense host bound")
                    >= measured.allocated_bytes(),
            );
        }
    }
}
