//! Qwen3 unified-memory MLX adapter for the family-independent runtime.
//!
//! Verified safetensors retain dense f32 or packed Affine4 matrices in one
//! set of MLX weight arrays, including vocabulary weights. Full f32 MLX
//! K/V is shared by supported CPU and Metal operations; it is not the
//! legacy host-reference f64 cache.
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
//! Affine4 residency and greedy decoding are advertised. Rotating caches,
//! stochastic sampling, thinking mode, and public prefix reuse are not.

mod output;
use self::output::Qwen3TextDecoder;

use crate::{
    admission::AdmissionController,
    api::{EngineCore, EngineFuture},
    error::{EngineError, MemoryLimitReason},
    family::{
        AutoregressiveModel, BufferMemoryUsage, CacheAllocationRequest, ExecutionControl,
        FamilyCache, LoadedModel, ModelFamily, ModelInstanceId, ModelLoadLimits, ModelLoadRequest,
        PrefillOutput, TokenLogits,
    },
    generation::GenerationLimits,
    parser_allocation::{ParserAllocation, ParserAllocationLimits},
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

include!("qwen/artifact.rs");

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
        self.output_memory_plan(None, &[])?;
        Ok(())
    }
}

/// Real Qwen3 backend. Normal metadata/factories may cross threads;
/// actual model and cache instances are created by ModelFamily::load.
pub struct Qwen3Backend {
    registration: FamilyRegistration,
    artifacts: BTreeMap<ArtifactId, ArtifactData>,
    limits: Qwen3BackendLimits,
    execution: Qwen3Execution,
    tool_call_parsing: bool,
    // W2_QWEN_PARSER_CAPACITY_HOOKS_V1
    parser_capacities: Option<ParserAllocationLimits>,
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
            tool_call_parsing: false,
            parser_capacities: None,
        })
    }

    /// Enable or disable Qwen tool-call parsing for new generations.
    ///
    /// Disabled by default, which preserves the text-only decoder. When
    /// enabled, `<tool_call>` blocks are parsed incrementally into untrusted
    /// proposals that the generation driver publishes only after the whole
    /// generation succeeds. Reserved control syntax such as `<think>` in
    /// model output is then a protocol error instead of display text.
    /// Generation also requires explicit capacities through
    /// `with_parser_allocation_limits`; this flag alone selects no limits.
    /// Rendering tool context does not allocate parser storage.
    pub fn with_tool_call_parsing(mut self, enabled: bool) -> Self {
        self.tool_call_parsing = enabled;
        if !enabled {
            self.parser_capacities = None;
        }
        self
    }

    /// Enable tool parsing with explicit fixed parser/slab capacities.
    ///
    /// Validates the layout and output-plan arithmetic without allocating
    /// parser storage or acquiring admission. No defaults are selected.
    /// Disabling tool parsing clears this configuration.
    ///
    /// Until the storage-backed decoder is implemented, generation with this
    /// configuration fails closed at decoder construction. It never falls
    /// back to the legacy Map-based tool parser.
    pub fn with_parser_allocation_limits(
        mut self,
        limits: ParserAllocationLimits,
    ) -> Result<Self, EngineError> {
        ParserAllocation::required_bytes(limits).map_err(|error| {
            EngineError::invalid_request("parser_allocation_limits", error.to_string())
        })?;
        self.limits.output_memory_plan(Some(limits), &[])?;
        self.parser_capacities = Some(limits);
        self.tool_call_parsing = true;
        Ok(self)
    }

    fn selected_parser_capacities(&self) -> Result<Option<ParserAllocationLimits>, EngineError> {
        match (self.tool_call_parsing, self.parser_capacities) {
            (false, None) => Ok(None),
            (true, Some(limits)) => Ok(Some(limits)),
            (true, None) => Err(EngineError::invalid_request(
                "parser_allocation_limits",
                "tool generation requires explicit parser capacities via with_parser_allocation_limits",
            )),
            (false, Some(_)) => Err(internal("disabled tool parsing retained parser capacities")),
        }
    }

    fn planned_output_bytes(&self) -> Result<u64, EngineError> {
        Ok(self
            .limits
            .output_memory_plan(self.selected_parser_capacities()?, &[])?
            .total_bytes())
    }

    fn tool_parser_limits(&self) -> qwen3::tool_calls::ParserLimits {
        let defaults = qwen3::tool_calls::ParserLimits::default();
        // Output cannot exceed context tokens times the largest token spelling.
        let output = self
            .limits
            .max_token_bytes
            .saturating_mul(self.limits.context_limit_tokens)
            .max(1);
        qwen3::tool_calls::ParserLimits {
            max_output_bytes: output,
            max_call_bytes: defaults.max_call_bytes.min(output),
            max_calls: defaults.max_calls,
        }
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
            FamilyRuntime::new(vec![backend], admission, generation_limits, retention)?
                .with_model_pool(Some(std::time::Duration::from_secs(300)))?;
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

include!("qwen/loading.rs");

impl RuntimeBackend for Qwen3Backend {
    fn cache_manifest_digest(
        &self,
        artifact: &ArtifactRegistration,
    ) -> Result<Option<[u8; 32]>, EngineError> {
        Ok(Some(self.data(artifact)?.manifest_sha256))
    }
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        Box::pin(async move {
            let resolved = registry.resolve_profile(context.profile_id())?;
            let data = self.data(resolved.artifact())?;
            let profile = resolved.profile();
            let tool_mode = context.tool_context().is_some();
            if tool_mode && !self.tool_call_parsing {
                return Err(EngineError::unsupported(
                    "tool context",
                    "enable tool-call parsing before rendering tool context",
                ));
            }
            let template = if tool_mode {
                qwen3::template::TOOL_CHAT_TEMPLATE
            } else {
                CHAT_TEMPLATE
            };
            let template_hash = if tool_mode {
                digest(template.as_bytes())
            } else {
                data.specification.template_sha256
            };
            let binding = RenderingBinding {
                profile_id: context.profile_id(),
                artifact_id: &data.specification.id,
                template,
                template_sha256: template_hash,
                artifact_manifest_sha256: data.manifest_sha256,
            };
            let limits = RenderingLimits::new(
                self.limits.max_messages,
                self.limits.max_input_bytes,
                template.len(),
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
            let load_peak = loading_peak_bound(data, &self.limits)?;
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
                // Prefill cannot process more tokens than this request
                // supplies. Keep a one-token allowance for decode work.
                temporary_bytes: workspace_bound(
                    &data.config,
                    capacity,
                    self.limits
                        .prefill_chunk_tokens
                        .get()
                        .min(input_tokens.max(1)),
                    self.limits.max_token_bytes,
                )?,
                output_bytes: self.planned_output_bytes()?,
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
        if self.tool_call_parsing {
            return Err(EngineError::unsupported(
                "admitted parser storage",
                "tool generation requires the storage-backed decoder construction hook",
            ));
        }
        let parser = if self.tool_call_parsing {
            Some(
                qwen3::tool_calls::ToolCallParser::new(self.tool_parser_limits())
                    .map_err(|error| execution_error("tool call parser", error.to_string()))?,
            )
        } else {
            None
        };
        Ok(Box::new(Qwen3TextDecoder {
            tokenizer: Arc::clone(&data.tokenizer),
            special_tokens: Arc::clone(&data.special_tokens),
            byte_decoder: Arc::clone(&data.byte_decoder),
            pending: Vec::new(),
            max_token_bytes: self.limits.max_token_bytes,
            finished: false,
            parser,
        }))
    }

    fn parser_allocation_limits(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
    ) -> Result<Option<ParserAllocationLimits>, EngineError> {
        let resolved = registry.resolve_profile(profile)?;
        self.data(resolved.artifact())?;
        self.selected_parser_capacities()
    }

    fn decoder_with_parser_allocation(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        allocation: Option<ParserAllocation>,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        let resolved = registry.resolve_profile(profile)?;
        self.data(resolved.artifact())?;
        match (self.selected_parser_capacities()?, allocation) {
            (None, None) => self.decoder(registry, profile),
            (Some(limits), Some(allocation)) => {
                let expected = ParserAllocation::required_bytes(limits).map_err(|error| {
                    EngineError::invalid_request("parser_allocation_limits", error.to_string())
                })?;
                if allocation.reserved_bytes() != expected {
                    return Err(internal(
                        "Qwen parser allocation differs from the planned slab size",
                    ));
                }
                // Intentionally no fallback, wrapper, or unused side slab.
                // The decoder patch must replace this rejection with a
                // constructor that actually parses in the supplied storage.
                Err(EngineError::unsupported(
                    "admitted parser storage",
                    "Qwen storage-backed decoder integration is not implemented yet",
                ))
            }
            _ => Err(internal(
                "Qwen parser allocation presence differs from backend configuration",
            )),
        }
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

include!("qwen/cache.rs");

fn registration(
    context_limit: usize,
    execution: Qwen3Execution,
) -> Result<FamilyRegistration, EngineError> {
    FamilyRegistration::new(
        // Packed vocabulary lookup and quantized output projection change
        // numerical execution. Old native/disk state must miss this version.
        FamilyVersion::new(qwen3::register::IMPL_VERSION, 4, 0),
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

include!("qwen/memory.rs");

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
            reason: MemoryLimitReason::BufferLimit,
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
            reason: MemoryLimitReason::BufferLimit,
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
                    reason: MemoryLimitReason::BufferLimit,
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
                    TensorView::new(record.dtype, record.shape.clone(), &record.bytes)
                        .expect("tensor"),
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
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
        let plan =
            block_on(backend.plan(&registry, profile, rendered.token_count())).expect("plan");
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
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
        let plan =
            block_on(backend.plan(&registry, profile, rendered.token_count())).expect("plan");
        let expected = fixture
            .limits
            .output_memory_plan(Some(capacities), &[])
            .expect("memory plan");
        assert_eq!(plan.output_bytes, expected.total_bytes());
        assert_eq!(
            plan.output_bytes - output_bytes(fixture.limits.generation).expect("legacy"),
            ParserAllocation::required_bytes(capacities).expect("slab")
        );
    }

    #[test]
    fn qwen_parser_hooks_disabling_tools_clears_capacities() {
        let fixture = Fixture::new();
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
    fn qwen_parser_hooks_supplied_slab_is_rejected_and_released_until_decoder_lands() {
        let fixture = Fixture::new();
        let capacities = explicit_parser_capacities();
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
                .expect("backend")
                .with_parser_allocation_limits(capacities)
                .expect("capacities");
        let registry = backend
            .registry(vec![fixture.profile.clone()])
            .expect("registry");
        let bytes = ParserAllocation::required_bytes(capacities).expect("bytes");
        let admission = AdmissionController::new(bytes).expect("admission");
        let mut grant = admission.reserve_grant("parser", &[bytes]).expect("grant");
        let allocation = ParserAllocation::from_grant(&mut grant, capacities).expect("slab");
        assert_eq!(admission.snapshot().expect("live").reserved_bytes(), bytes);
        let result = backend.decoder_with_parser_allocation(
            &registry,
            fixture.context.profile_id(),
            Some(allocation),
        );
        match result {
            Err(error) => assert!(
                error
                    .to_string()
                    .contains("storage-backed decoder integration")
            ),
            Ok(_) => panic!("decoder must not claim storage-backed parsing yet"),
        }
        drop(grant);
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
        assert_eq!(
            admission
                .snapshot()
                .expect("released")
                .active_reservations(),
            0
        );
    }

    #[test]
    fn qwen_parser_hooks_wrong_slab_size_is_rejected_and_released() {
        let fixture = Fixture::new();
        let capacities = explicit_parser_capacities();
        let backend =
            Qwen3Backend::new(vec![fixture.specification.clone()], fixture.limits.clone())
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
        let rendered =
            block_on(engine.render_and_count(context)).expect("production tool rendering");
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
                assert_eq!(reason, MemoryLimitReason::BufferLimit);
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
