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
//! trustworthy from mutable artifact contents.
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
    runtime::{FamilyRuntime, RetentionPolicy, RuntimeBackend, RuntimePlan, TextDecoder},
    scheduler::Scheduler,
    types::{
        ArtifactId, CacheKind, CacheMode, CacheRepresentationId, Capabilities, ContextSnapshot,
        DecodingCapability, ExecutionBackend, ModelProfileId, RenderedContext, RendererIdentity,
        TokenizerId, WeightRepresentation,
    },
};
use minijinja::{Environment, UndefinedBehavior, context};
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
    io::{self, Read, Write},
    num::NonZeroUsize,
    path::{Component, Path, PathBuf},
    sync::Arc,
};
use tokenizers::Tokenizer;

const TOKENIZERS_VERSION: &str = "0.23.2";
const RENDERER_VERSION: &str = "2.12.0";

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
    tokenizer: Arc<Tokenizer>,
    special_tokens: Arc<BTreeSet<u32>>,
    byte_decoder: Arc<BTreeMap<char, u8>>,
    tokenizer_id: TokenizerId,
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
            let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(|source| {
                EngineError::CheckpointFormat {
                    format: "tokenizer.json",
                    path: Some(specification.directory.join("tokenizer.json")),
                    source,
                }
            })?;
            if tokenizer.get_truncation().is_some()
                || tokenizer.get_padding().is_some()
                || tokenizer.get_encode_special_tokens()
            {
                return Err(EngineError::unsupported(
                    "tokenizer options",
                    "truncation, padding, and special-token splitting are not enabled by this profile",
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
            for (spelling, id) in tokenizer.get_vocab(true) {
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
                    || tokenizer.id_to_token(token).is_none()
                    || !specials.contains(&token)
                {
                    return Err(validation(
                        Some(&specification.id),
                        "EOS configuration does not identify declared special tokens inside the model vocabulary",
                    ));
                }
            }
            let mut identity = Sha256::new();
            identity.update(b"alloy.qwen3.tokenizer.v1\0");
            identity.update(TOKENIZERS_VERSION.as_bytes());
            identity.update(digest(&tokenizer_bytes));
            let tokenizer_id = TokenizerId::new(format!("sha256:{:x}", identity.finalize()))?;
            let data = ArtifactData {
                specification,
                config,
                tokenizer: Arc::new(tokenizer),
                special_tokens: Arc::new(specials),
                byte_decoder: Arc::clone(&inverse),
                tokenizer_id,
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
            let text = render_bounded(&context, &self.limits)?;
            let encoding = data
                .tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| execution_error("tokenize", e.to_string()))?;
            let ids = encoding.get_ids();
            resolved.profile().check_context(ids.len())?;
            if ids
                .iter()
                .any(|&id| u64::from(id) >= as_u64(data.config.vocab_size).unwrap_or(0))
            {
                return Err(validation(
                    Some(resolved.artifact().id()),
                    "tokenizer produced an ID outside the loaded model vocabulary",
                ));
            }
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(ids.len())
                .map_err(|_| allocation("rendered token IDs", ids.len().checked_mul(4)))?;
            tokens.extend_from_slice(ids);
            let mut hash = Sha256::new();
            hash.update(b"alloy.qwen3.render.v1\0");
            for value in [
                context.profile_id().as_str(),
                data.specification.id.as_str(),
                data.tokenizer_id.as_str(),
                RENDERER_VERSION,
                text.as_str(),
            ] {
                hash.update(as_u64(value.len())?.to_le_bytes());
                hash.update(value.as_bytes());
            }
            hash.update(data.specification.template_sha256);
            for (name, pin) in &data.specification.files {
                hash.update(as_u64(name.len())?.to_le_bytes());
                hash.update(name.as_bytes());
                hash.update(pin.bytes.to_le_bytes());
                hash.update(pin.sha256);
            }
            hash.update(as_u64(tokens.len())?.to_le_bytes());
            for token in &tokens {
                hash.update(token.to_le_bytes());
            }
            Ok(RenderedContext {
                snapshot_id: context.snapshot_id().clone(),
                profile_id: context.profile_id().clone(),
                artifact_id: data.specification.id.clone(),
                text,
                token_ids: tokens,
                render_hash: hash.finalize().into(),
                template_hash: data.specification.template_sha256,
                tokenizer_id: data.tokenizer_id.clone(),
                renderer: RendererIdentity {
                    implementation: "minijinja".to_owned(),
                    version: RENDERER_VERSION.to_owned(),
                },
            })
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
    tokenizer: Arc<Tokenizer>,
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
        let spelling = self.tokenizer.id_to_token(token).ok_or_else(|| {
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
        FamilyVersion::new(qwen3::register::IMPL_VERSION, 1, 0),
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

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
    failed: bool,
    allocation_failed: bool,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let new_size = self.bytes.len().checked_add(bytes.len());
        if new_size.is_none_or(|size| size > self.limit) {
            self.failed = true;
            return Err(io::Error::other(
                "rendered output exceeds its explicit bound",
            ));
        }
        if self.bytes.try_reserve_exact(bytes.len()).is_err() {
            self.allocation_failed = true;
            return Err(io::Error::other("rendered output allocation failed"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn render_bounded(
    context: &ContextSnapshot,
    limits: &Qwen3BackendLimits,
) -> Result<String, EngineError> {
    if context.messages().len() > limits.max_messages {
        return Err(EngineError::invalid_request(
            "messages",
            "message count exceeds its explicit limit",
        ));
    }
    let mut total = 0_usize;
    let mut messages = Vec::new();
    messages
        .try_reserve_exact(context.messages().len())
        .map_err(|_| allocation("rendering messages", None))?;
    for message in context.messages() {
        total = total
            .checked_add(message.content.len())
            .ok_or_else(|| internal("input byte count overflow"))?;
        if total > limits.max_input_bytes {
            return Err(EngineError::invalid_request(
                "messages",
                "input byte limit exceeded",
            ));
        }
        if message.content.contains("<|")
            || [
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
            ]
            .iter()
            .any(|control| message.content.contains(control))
        {
            return Err(EngineError::invalid_request(
                "messages",
                "message body contains reserved Qwen3 control syntax",
            ));
        }
        messages
            .push(context! { role => message.role.as_str(), content => message.content.as_str() });
    }
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment
        .add_template("qwen3-text", CHAT_TEMPLATE)
        .map_err(|e| execution_error("compile chat template", e.to_string()))?;
    let template = environment
        .get_template("qwen3-text")
        .map_err(|e| execution_error("get chat template", e.to_string()))?;
    let mut writer = BoundedWriter {
        bytes: Vec::new(),
        limit: limits.max_rendered_bytes,
        failed: false,
        allocation_failed: false,
    };
    let result = template.render_to_write(context! { messages => messages }, &mut writer);
    if writer.allocation_failed {
        return Err(allocation("rendered output", None));
    }
    if writer.failed {
        return Err(EngineError::invalid_request(
            "rendered context",
            "rendered byte limit exceeded",
        ));
    }
    result.map_err(|e| execution_error("render chat template", e.to_string()))?;
    String::from_utf8(writer.bytes).map_err(|_| internal("template emitted invalid UTF-8"))
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
