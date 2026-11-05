//! Actual Qwen3 host-reference adapter for the family-independent runtime.
//!
//! This adapter loads verified safetensors into the existing real Qwen3Model.
//! Weights reside in f32 buffers; cached attention uses full host f64 K/V.
//! It does not claim MLX/Metal execution, resident quantized weights,
//! sampling, rotating caches, thinking-mode rendering, or prefix reuse.
//!
//! Supported tokenizer contract: a pinned Hugging Face BPE tokenizer.json
//! with the ByteLevel decoder, no truncation/padding, and explicitly declared
//! special tokens. Incremental decoding preserves incomplete UTF-8 bytes and
//! rejects invalid/truncated sequences instead of inserting replacements.
//!
//! The renderer is the existing versioned Qwen3 non-thinking CHAT_TEMPLATE.
//! Its expected SHA-256 is supplied by the trusted caller, not inferred as
//! trustworthy from mutable artifact contents. Production rendering and
//! tokenization delegate to crate::rendering; this adapter has no private
//! template engine, output writer, tokenizer identity, or render-hash path.
//! The incremental decoder shares the same immutable PinnedTokenizer.
//!
//! Tokenizer/configuration objects are bounded control-plane metadata.
//! Their library-internal heap storage is not reported as model tensors.
//! The caller must exclude an appropriate control-plane allowance from the
//! engine-managed allocation budget, as required by admission.rs.

use crate::{
    admission::AdmissionController,
    api::{EngineCore, EngineFuture},
    cache::QWEN3_CACHE_REPRESENTATION,
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
use qwen3::{
    config::Qwen3Config,
    fixture::{LayerWeights, Qwen3Weights},
    loader::{CheckpointVerification, LoadOptions, load_weights_verified},
    model::{ModelError, Qwen3Cache, Qwen3Model},
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
    /// checks cancellation before each actual token execution.
    pub prefill_chunk_tokens: NonZeroUsize,
    /// Extra allowance for loader metadata/allocator/control structures.
    pub loader_headroom_bytes: u64,
    /// Extra request-local allowance beyond the explicit workspace formula.
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
    checksums: BTreeMap<PathBuf, [u8; 32]>,
}

/// Real Qwen3 backend. Normal metadata/factories may cross threads;
/// actual model and cache instances are created by ModelFamily::load.
pub struct Qwen3Backend {
    registration: FamilyRegistration,
    artifacts: BTreeMap<ArtifactId, ArtifactData>,
    limits: Qwen3BackendLimits,
}

impl Qwen3Backend {
    /// Validate pinned metadata and tokenizer configuration without loading
    /// model weights. This synchronous constructor can perform substantial
    /// tokenizer parsing; call it during setup, not on a cancellation loop.
    pub fn new(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
    ) -> Result<Self, EngineError> {
        limits.validate()?;
        if specifications.is_empty() {
            return Err(EngineError::invalid_request(
                "artifacts",
                "must not be empty",
            ));
        }
        let registration = registration(limits.context_limit_tokens)?;
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
            let resident_bound = resident_bound(&config)?;
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
                checksums,
            };
            artifacts.insert(data.specification.id.clone(), data);
        }
        Ok(Self {
            registration,
            artifacts,
            limits,
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

    /// Connect this real backend to FamilyRuntime and EngineCore.
    /// The caller supplies budget, scheduling, output, and retention policy.
    pub fn into_engine(
        self,
        profiles: Vec<ModelProfile>,
        admission: AdmissionController,
        scheduler: Scheduler,
        retention: RetentionPolicy,
    ) -> Result<EngineCore<FamilyRuntime>, EngineError> {
        let registry = self.registry(profiles)?;
        let generation_limits = self.limits.generation;
        let backend: Arc<dyn RuntimeBackend> = Arc::new(self);
        let runtime = FamilyRuntime::new(vec![backend], admission, generation_limits, retention)?;
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
                max_weight_bytes: as_usize(request.limits().max_resident_bytes())?,
            },
            &verification,
        );
        control.check_cancelled()?;
        let weights = result.map_err(|e| model_error(e, "load Qwen3 weights"))?;
        let measured = weight_memory(&weights)?;
        if measured.allocated_bytes() > request.limits().max_resident_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "Qwen3 resident buffers",
                required_bytes: measured.allocated_bytes(),
                available_bytes: request.limits().max_resident_bytes(),
            });
        }
        let model = Qwen3Model::from_weights(data.config.clone(), weights)
            .map_err(|e| model_error(e, "construct Qwen3 model"))?;
        control.check_cancelled()?;
        let mut capabilities = self.registration.capabilities().clone();
        capabilities.context_limit_tokens = capabilities
            .context_limit_tokens
            .min(data.config.max_context);
        Ok(Box::new(Qwen3Loaded {
            model,
            artifact: request.artifact().id().clone(),
            instance: ModelInstanceId::new(),
            registration: self.registration.clone(),
            capabilities,
            memory: measured,
            vocabulary: data.config.vocab_size,
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
                    cache_bound(&data.config, capacity)?,
                )?,
                temporary_bytes: workspace_bound(
                    &data.config,
                    capacity,
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
    model: Qwen3Model,
    artifact: ArtifactId,
    instance: ModelInstanceId,
    registration: FamilyRegistration,
    capabilities: Capabilities,
    memory: BufferMemoryUsage,
    vocabulary: usize,
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

struct Qwen3NativeCache {
    inner: Qwen3Cache,
    owner: ModelInstanceId,
    mode: CacheMode,
    capacity: usize,
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
        let (allocated, used) = self
            .inner
            .tracked_memory_bytes()
            .map_err(|e| model_error(e, "measure Qwen3 KV buffers"))?;
        BufferMemoryUsage::new(as_u64(allocated)?, as_u64(used)?)
    }
    fn supports_truncation(&self) -> bool {
        false
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner.reset();
        Ok(())
    }
    fn truncate(&mut self, _token_count: usize) -> Result<(), EngineError> {
        Err(EngineError::unsupported(
            "cache truncation",
            "Qwen3 full cache must be reset and re-prefilled",
        ))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
        cache
            .as_any_mut()
            .downcast_mut::<Qwen3NativeCache>()
            .ok_or_else(|| {
                EngineError::invalid_request("cache", "cache has a different family type")
            })
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
                "only actual full host-f64 KV is implemented",
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
        let inner = self
            .model
            .new_cache(
                request.capacity_tokens(),
                as_usize(request.max_cache_bytes())?,
            )
            .map_err(|e| model_error(e, "allocate Qwen3 KV"))?;
        let cache = Qwen3NativeCache {
            inner,
            owner: self.instance.clone(),
            mode: request.mode().clone(),
            capacity: request.capacity_tokens(),
        };
        let actual = cache.memory_usage()?.allocated_bytes();
        if actual > request.max_cache_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "allocated Qwen3 KV buffers",
                required_bytes: actual,
                available_bytes: request.max_cache_bytes(),
            });
        }
        control.check_cancelled()?;
        Ok(Box::new(cache))
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        let cache = self.owned_cache(cache)?;
        control.check_cancelled()?;
        if tokens.is_empty() {
            return Err(EngineError::invalid_request(
                "tokens",
                "prefill chunk must not be empty",
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
        let result = (|| {
            let mut last = None;
            for &token in tokens {
                control.check_cancelled()?;
                let logits = self
                    .model
                    .decode_token(token, &mut cache.inner)
                    .map_err(|e| model_error(e, "Qwen3 prefill"))?;
                last = Some(TokenLogits::new(logits)?);
            }
            let logits = last.ok_or_else(|| internal("prefill produced no logits"))?;
            PrefillOutput::new(tokens.len(), logits)
        })();
        if result.is_err() {
            cache.inner.reset();
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
        let result = self
            .model
            .decode_token(token, &mut cache.inner)
            .map_err(|e| model_error(e, "Qwen3 decode"))
            .and_then(TokenLogits::new);
        if result.is_err() {
            cache.inner.reset();
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

fn registration(context_limit: usize) -> Result<FamilyRegistration, EngineError> {
    FamilyRegistration::new(
        // Shared rendering changes tokenizer/render identities. Keep the
        // adapter implementation version distinct from the legacy path.
        FamilyVersion::new(qwen3::register::IMPL_VERSION, 2, 0),
        qwen3::register::ARCHITECTURES
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        Capabilities {
            task: "text-generation".to_owned(),
            family: "qwen3".to_owned(),
            backend: ExecutionBackend::HostCpu,
            weight_representations: vec![WeightRepresentation::F32],
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
        representation: CacheRepresentationId::new(QWEN3_CACHE_REPRESENTATION)?,
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

fn resident_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let v = as_u64(config.vocab_size)?;
    let i = as_u64(config.intermediate_size)?;
    let layer = sum(&[
        product(2, h)?,
        product(2, as_u64(config.head_dim)?)?,
        product(2, product(q, h)?)?,
        product(2, product(kv, h)?)?,
        product(3, product(i, h)?)?,
    ])?;
    let elements = sum(&[
        product(v, h)?,
        product(as_u64(config.num_layers)?, layer)?,
        h,
        if config.tie_embeddings {
            0
        } else {
            product(v, h)?
        },
    ])?;
    sum(&[
        product(elements, 4)?,
        product(
            as_u64(config.num_layers)?,
            as_u64(std::mem::size_of::<LayerWeights>())?,
        )?,
    ])
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
    let mut account = |values: &Vec<f32>| -> Result<(), EngineError> {
        allocated = sum(&[allocated, product(as_u64(values.capacity())?, 4)?])?;
        used = sum(&[used, product(as_u64(values.len())?, 4)?])?;
        Ok(())
    };
    account(&weights.embed_tokens)?;
    account(&weights.final_layernorm)?;
    if let Some(head) = &weights.lm_head {
        account(head)?;
    }
    for layer in &weights.layers {
        for tensor in [
            &layer.input_layernorm,
            &layer.q_proj,
            &layer.k_proj,
            &layer.v_proj,
            &layer.q_norm,
            &layer.k_norm,
            &layer.o_proj,
            &layer.post_attention_layernorm,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
        ] {
            account(tensor)?;
        }
    }
    BufferMemoryUsage::new(allocated, used)
}

fn cache_bound(config: &Qwen3Config, capacity: usize) -> Result<u64, EngineError> {
    let layers = as_u64(config.num_layers)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    sum(&[
        product(product(product(layers, as_u64(capacity)?)?, kv)?, 16)?,
        product(layers, as_u64(std::mem::size_of::<(Vec<f64>, Vec<f64>)>())?)?,
    ])
}

/// Conservative bound for the existing per-token host reference routine.
/// Counts pending layer K/V, overlapping projection/normalization/MLP
/// rows, logits, attention scores, Vec headers, and decoder byte buffers.
/// It is not an estimate of MLX memory and is tied to this implementation.
fn workspace_bound(
    config: &Qwen3Config,
    capacity: usize,
    token_bytes: usize,
) -> Result<u64, EngineError> {
    let layers = as_u64(config.num_layers)?;
    let h = as_u64(config.hidden_size)?;
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let rows = sum(&[
        product(2, product(layers, kv)?)?,
        product(8, h)?,
        product(8, q)?,
        product(8, kv)?,
        product(8, as_u64(config.intermediate_size)?)?,
        product(4, as_u64(config.vocab_size)?)?,
        product(2, as_u64(capacity)?)?,
    ])?;
    sum(&[
        product(rows, 8)?,
        product(
            sum(&[layers, 32])?,
            as_u64(std::mem::size_of::<(Vec<f64>, Vec<f64>)>())?,
        )?,
        product(as_u64(token_bytes)?, 8)?,
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
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn add_tensor(
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
            shape,
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        });
    }

    // Serialize actual deterministic fixture weights. The rendering tests
    // do not execute inference, but do not use a fake checkpoint either.
    fn checkpoint_bytes(config: &Qwen3Config) -> Vec<u8> {
        let weights = qwen3::fixture::generate_fixture(config);
        let mut records = Vec::new();
        add_tensor(
            &mut records,
            "model.embed_tokens.weight".to_owned(),
            vec![config.vocab_size, config.hidden_size],
            &weights.embed_tokens,
        );
        for (index, layer) in weights.layers.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            for (suffix, shape, values) in [
                (
                    "input_layernorm.weight",
                    vec![config.hidden_size],
                    layer.input_layernorm.as_slice(),
                ),
                (
                    "self_attn.q_proj.weight",
                    vec![config.q_dim(), config.hidden_size],
                    layer.q_proj.as_slice(),
                ),
                (
                    "self_attn.k_proj.weight",
                    vec![config.kv_dim(), config.hidden_size],
                    layer.k_proj.as_slice(),
                ),
                (
                    "self_attn.v_proj.weight",
                    vec![config.kv_dim(), config.hidden_size],
                    layer.v_proj.as_slice(),
                ),
                (
                    "self_attn.q_norm.weight",
                    vec![config.head_dim],
                    layer.q_norm.as_slice(),
                ),
                (
                    "self_attn.k_norm.weight",
                    vec![config.head_dim],
                    layer.k_norm.as_slice(),
                ),
                (
                    "self_attn.o_proj.weight",
                    vec![config.hidden_size, config.q_dim()],
                    layer.o_proj.as_slice(),
                ),
                (
                    "post_attention_layernorm.weight",
                    vec![config.hidden_size],
                    layer.post_attention_layernorm.as_slice(),
                ),
                (
                    "mlp.gate_proj.weight",
                    vec![config.intermediate_size, config.hidden_size],
                    layer.gate_proj.as_slice(),
                ),
                (
                    "mlp.up_proj.weight",
                    vec![config.intermediate_size, config.hidden_size],
                    layer.up_proj.as_slice(),
                ),
                (
                    "mlp.down_proj.weight",
                    vec![config.hidden_size, config.intermediate_size],
                    layer.down_proj.as_slice(),
                ),
            ] {
                add_tensor(&mut records, format!("{prefix}.{suffix}"), shape, values);
            }
        }
        add_tensor(
            &mut records,
            "model.norm.weight".to_owned(),
            vec![config.hidden_size],
            &weights.final_layernorm,
        );
        if let Some(head) = &weights.lm_head {
            add_tensor(
                &mut records,
                "lm_head.weight".to_owned(),
                vec![config.vocab_size, config.hidden_size],
                head,
            );
        }
        let views: Vec<_> = records
            .iter()
            .map(|record| {
                (
                    record.name.as_str(),
                    TensorView::new(Dtype::F32, record.shape.clone(), &record.bytes)
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
}
