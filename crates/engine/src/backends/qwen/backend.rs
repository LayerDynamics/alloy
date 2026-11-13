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

#[path = "output.rs"]
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

include!("artifact.rs");

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
    /// Runtime construction supplies the admitted slab. Tool JSON validation
    /// uses that fixed storage and completed proposals transfer as one owned
    /// batch. There is no fallback to the legacy Map-based tool parser.
    pub fn with_parser_allocation_limits(
        mut self,
        limits: ParserAllocationLimits,
    ) -> Result<Self, EngineError> {
        ParserAllocation::required_bytes(limits).map_err(|error| {
            EngineError::invalid_request("parser_allocation_limits", error.to_string())
        })?;
        self.limits.output_memory_plan(
            Some(limits),
            &[std::alloc::Layout::new::<Qwen3TextDecoder>()],
        )?;
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
        let selected = self.selected_parser_capacities()?;
        let owner = [std::alloc::Layout::new::<Qwen3TextDecoder>()];
        let extra = if selected.is_some() {
            owner.as_slice()
        } else {
            &[]
        };
        Ok(self
            .limits
            .output_memory_plan(selected, extra)?
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

include!("loading.rs");

impl RuntimeBackend for Qwen3Backend {
    fn projected_token_upper_bound(
        &self,
        registry: &Registry,
        projection: crate::api::ProjectedToolRender<'_>,
    ) -> Result<usize, EngineError> {
        let rendered = projection.rendered;
        let artifact = registry.resolve_profile(rendered.profile_id())?.artifact();
        let data = self.data(artifact)?;
        if rendered.artifact_id() != artifact.id()
            || rendered.tokenizer_id() != data.tokenizer.identity()
            || rendered.template_hash() != &digest(qwen3::template::TOOL_CHAT_TEMPLATE.as_bytes())
        {
            return Err(EngineError::invalid_request(
                "projection",
                "rendered tool context identity does not match the selected Qwen profile",
            ));
        }
        let rendered_bytes = rendered
            .text()
            .len()
            .checked_add(projection.additional_rendered_bytes)
            .ok_or_else(|| {
                EngineError::invalid_request("projection", "rendered byte bound overflow")
            })?;
        let required = rendered_bytes.max(projection.input_bytes);
        let limit = self
            .limits
            .max_rendered_bytes
            .min(self.limits.max_input_bytes);
        if required > limit {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "projected tool context bytes",
                required_bytes: u64::try_from(required).unwrap_or(u64::MAX),
                available_bytes: u64::try_from(limit).unwrap_or(u64::MAX),
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        rendering::qwen3_projected_tool_token_upper_bound(
            &data.tokenizer,
            rendered,
            projection.result_count,
            projection.additional_rendered_bytes,
            projection.additional_ascii_escape_bytes,
        )
    }
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
                self.limits.max_rendered_bytes,
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
            // Count overflow here; generation admission enforces profile limits.
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
            slab: None,
            completed_slab: None,
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
                let data = self.data(resolved.artifact())?;
                let slab = crate::slab_parser::SlabParser::new(
                    allocation,
                    self.tool_parser_limits().max_output_bytes,
                )?;
                output::box_slab_decoder(Qwen3TextDecoder {
                    tokenizer: Arc::clone(&data.tokenizer),
                    special_tokens: Arc::clone(&data.special_tokens),
                    byte_decoder: Arc::clone(&data.byte_decoder),
                    pending: Vec::new(),
                    max_token_bytes: self.limits.max_token_bytes,
                    finished: false,
                    parser: None,
                    slab: Some(slab),
                    completed_slab: None,
                })
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

include!("cache.rs");

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

include!("memory.rs");

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
#[path = "../../../tests/unit/backends_qwen_shared_rendering_wiring.rs"]
mod shared_rendering_wiring_tests;

// ALLOY_MEASURED_BUFFERS_V1

// ALLOY_PRESSURE_STARTUP_INTEGRATION_V1

#[cfg(test)]
#[path = "../../../tests/unit/backends_qwen_dense_weight_planning_regressions.rs"]
mod dense_weight_planning_regressions;
