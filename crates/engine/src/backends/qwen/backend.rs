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
//! stochastic sampling and thinking mode are not.

#[path = "constraints.rs"]
mod constraints;
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
        DecodingCapability, ExecutionBackend, ModelProfileId, RenderedContext, TokenizerId,
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

include!("architecture.rs");
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
    family: BackendFamily,
    registration: FamilyRegistration,
    artifacts: BTreeMap<ArtifactId, ArtifactData>,
    limits: Qwen3BackendLimits,
    execution: Qwen3Execution,
    tool_call_parsing: bool,
    // W2_QWEN_PARSER_CAPACITY_HOOKS_V1
    parser_capacities: Option<ParserAllocationLimits>,
}

impl Qwen3Backend {
    /// Construct the native Ministral 3 text backend for qualification runs.
    pub fn new_ministral3(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Ministral3)
    }

    /// Native Gemma 3 text adapter with mixed local/full attention.
    /// Construction does not qualify tools or production agent behavior.
    pub fn new_gemma3(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Gemma3)
    }
    /// Gemma 4 E4B text decoder with per-layer embeddings and shared KV.
    pub fn new_gemma4(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Gemma4)
    }

    /// Native Phi fused decoder with immutable long-factor SuScaledRoPE.
    /// Construction is not a production qualification.
    pub fn new_phi(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Phi)
    }
    /// Native hybrid decoder with the upstream trained non-thinking text template.
    /// Construction alone does not qualify a production agent profile.
    pub fn new_qwen35(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Qwen35)
    }
    /// Native hybrid decoder with Ornith's distinct assistant-history framing.
    pub fn new_ornith(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Ornith)
    }

    /// Construct the native SmolLM3 adapter with validated NoPE layer semantics.
    pub fn new_smollm3(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::SmolLm3)
    }
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
        Self::new_family(specifications, limits, execution, BackendFamily::Qwen3)
    }

    /// Explicit native Llama factory. Configs never fall back to Qwen.
    pub fn new_llama3(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
    ) -> Result<Self, EngineError> {
        Self::new_family(specifications, limits, execution, BackendFamily::Llama3)
    }

    fn new_family(
        specifications: Vec<Qwen3Artifact>,
        limits: Qwen3BackendLimits,
        execution: Qwen3Execution,
        family: BackendFamily,
    ) -> Result<Self, EngineError> {
        limits.validate()?;
        if specifications.is_empty() {
            return Err(EngineError::invalid_request(
                "artifacts",
                "must not be empty",
            ));
        }
        let registration = registration_for(family, limits.context_limit_tokens, execution)?;
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
            if digest(family_template(family, false).as_bytes()) != specification.template_sha256 {
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
            let config = match family {
                BackendFamily::Ministral3 => BackendConfig::Transformer(
                    inference_infra::decoder::ministral3::MinistralConfig::from_json(&config_value)
                        .and_then(|config| config.decoder_config())
                        .map_err(|e| model_error(e, "validate Ministral architecture"))?,
                ),
                BackendFamily::Gemma4 => BackendConfig::Gemma4(
                    inference_infra::decoder::gemma4::Gemma4TextConfig::from_affine4_json(
                        &config_value,
                    )
                    .map_err(|e| model_error(e, "validate Gemma 4 config"))?,
                ),
                BackendFamily::Gemma3 => BackendConfig::Gemma3(
                    inference_infra::decoder::gemma3::Gemma3Config::from_json(&config_value)
                        .map_err(|e| model_error(e, "validate Gemma3 config"))?,
                ),
                BackendFamily::Phi => BackendConfig::Phi(
                    inference_infra::decoder::phi::PhiConfig::from_json(&config_value)
                        .map_err(|e| model_error(e, "validate Phi config"))?,
                ),
                BackendFamily::Qwen3 => BackendConfig::Transformer(parse_config(&config_value)?),
                BackendFamily::Llama3 => BackendConfig::Transformer(
                    llama3::config::parse_config(&config_value)
                        .map_err(|e| model_error(e, "validate Llama config"))?,
                ),
                BackendFamily::SmolLm3 => BackendConfig::Transformer(
                    smollm::config::parse_config(&config_value)
                        .map_err(|e| model_error(e, "validate SmolLM3 config"))?,
                ),
                BackendFamily::Qwen35 | BackendFamily::Ornith => BackendConfig::Hybrid(
                    inference_infra::decoder::qwen35::config::Qwen35Config::from_json(
                        &config_value,
                    )
                    .map_err(|e| model_error(e, "validate native hybrid config"))?,
                ),
            };
            // Undeclared representation uses the conservative dense path.
            // Quantized declarations retain the existing packed estimates.
            let declares_quantization = ["quantization", "quantization_config"]
                .iter()
                .any(|key| config_value.get(*key).is_some_and(|value| !value.is_null()));
            let (decoded_bound, resident_payload_bound) = match &config {
                BackendConfig::Gemma4(c) => {
                    let p = c
                        .weight_payload_plan()
                        .map_err(|e| model_error(e, "plan Gemma 4 weights"))?;
                    (p.payload_bytes, p.payload_bytes)
                }
                BackendConfig::Gemma3(c) => {
                    let p = c
                        .weight_payload_plan()
                        .map_err(|e| execution_error("plan Gemma3 weights", e.to_string()))?;
                    (p.payload_bytes, p.payload_bytes)
                }
                BackendConfig::Phi(c) => {
                    let p = c
                        .weight_payload_plan()
                        .map_err(|e| execution_error("plan Phi weights", e.to_string()))?;
                    (p.payload_bytes, p.payload_bytes)
                }
                BackendConfig::Hybrid(c) => {
                    let p = c
                        .weight_payload_plan()
                        .map_err(|e| execution_error("plan hybrid weights", e.to_string()))?;
                    (p.payload_bytes, p.payload_bytes)
                }
                BackendConfig::Transformer(c) if declares_quantization => {
                    (decoded_weight_bound(c)?, resident_bound(c)?)
                }
                BackendConfig::Transformer(c) => (
                    dense_decoded_weight_bound(c)?,
                    dense_weight_payload_bound(c)?,
                ),
            };
            let resident_bound = sum(&[resident_payload_bound, limits.loader_headroom_bytes])?;
            let tokenizer_bytes =
                read_metadata(&specification, "tokenizer.json", limits.max_tokenizer_bytes)?;
            let tokenizer_value = parse_json(
                &specification.directory.join("tokenizer.json"),
                &tokenizer_bytes,
            )?;
            let gemma_decoder = matches!(family, BackendFamily::Gemma3 | BackendFamily::Gemma4);
            output::validate_tokenizer_decoder(&tokenizer_value, gemma_decoder)?;
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
            let mut max_token_spelling_bytes = 0;
            for (spelling, id) in tokenizer.tokenizer().get_vocab(true) {
                max_token_spelling_bytes = max_token_spelling_bytes.max(spelling.len());
                if u64::from(id) >= as_u64(config.vocabulary())?
                    || spelling.len() > limits.max_token_bytes
                {
                    return Err(validation(
                        Some(&specification.id),
                        "tokenizer vocabulary exceeds the configured token/model bounds",
                    ));
                }
                if !gemma_decoder
                    && !specials.contains(&id)
                    && !spelling.chars().all(|ch| inverse.contains_key(&ch))
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
            let mut eos = if family.hybrid() || gemma_decoder {
                parse_eos(
                    eos_value
                        .as_ref()
                        .or_else(|| config_value.pointer("/text_config/eos_token_id")),
                )?
            } else {
                parse_eos(eos_value.as_ref())?
            };
            if family.hybrid() || family == BackendFamily::Phi || gemma_decoder {
                let bytes = read_metadata(
                    &specification,
                    "tokenizer_config.json",
                    limits.max_config_bytes,
                )?;
                let metadata: Value = parse_json(
                    &specification.directory.join("tokenizer_config.json"),
                    &bytes,
                )?;
                let spelling = metadata["eos_token"]
                    .as_str()
                    .or_else(|| metadata["eos_token"]["content"].as_str())
                    .ok_or_else(|| {
                        validation(Some(&specification.id), "missing pinned tokenizer EOS")
                    })?;
                let token = tokenizer.tokenizer().token_to_id(spelling).ok_or_else(|| {
                    validation(Some(&specification.id), "tokenizer EOS spelling is absent")
                })?;
                if !eos.contains(&token) {
                    eos.push(token);
                }
            }
            if family == BackendFamily::Ministral3 {
                if eos != [2]
                    || [
                        ("<s>", 1),
                        ("</s>", 2),
                        ("[INST]", 3),
                        ("[/INST]", 4),
                        ("[AVAILABLE_TOOLS]", 5),
                        ("[/AVAILABLE_TOOLS]", 6),
                        ("[TOOL_RESULTS]", 7),
                        ("[/TOOL_RESULTS]", 8),
                        ("[TOOL_CALLS]", crate::slab_parser::ministral3::CALL),
                        ("[ARGS]", crate::slab_parser::ministral3::ARGS),
                        ("[SYSTEM_PROMPT]", 17),
                        ("[/SYSTEM_PROMPT]", 18),
                    ]
                    .into_iter()
                    .any(|(name, id)| {
                        tokenizer.tokenizer().token_to_id(name) != Some(id)
                            || !specials.contains(&id)
                    })
                {
                    return Err(validation(
                        Some(&specification.id),
                        "unexpected Ministral control tokens",
                    ));
                }
                let reserved = family_rendering_options(family).reserved_spellings;
                for id in &specials {
                    let spelling = tokenizer.tokenizer().id_to_token(*id).ok_or_else(|| {
                        validation(
                            Some(&specification.id),
                            "missing Ministral control spelling",
                        )
                    })?;
                    if !reserved.iter().any(|pattern| spelling.contains(pattern)) {
                        return Err(validation(
                            Some(&specification.id),
                            "unprotected Ministral control token",
                        ));
                    }
                }
            }
            if family == BackendFamily::Gemma4 {
                let reserved = family_rendering_options(family).reserved_spellings;
                if eos != [1, 106, 50]
                    || [
                        ("<bos>", 2),
                        ("<eos>", 1),
                        ("<|turn>", 105),
                        ("<turn|>", 106),
                        ("<|tool_response>", 50),
                        ("<|tool_call>", 48),
                        ("<tool_call|>", 49),
                        (r#"<|"|>"#, 52),
                    ]
                    .into_iter()
                    .any(|(name, id)| {
                        tokenizer.tokenizer().token_to_id(name) != Some(id)
                            || !specials.contains(&id)
                    })
                    || specials.iter().any(|id| {
                        tokenizer
                            .tokenizer()
                            .id_to_token(*id)
                            .is_none_or(|s| !reserved.iter().any(|marker| s.contains(marker)))
                    })
                {
                    return Err(validation(
                        Some(&specification.id),
                        "unexpected Gemma 4 control tokens",
                    ));
                }
            }
            if family == BackendFamily::Gemma3 {
                for spelling in ["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>"] {
                    let id = tokenizer
                        .tokenizer()
                        .token_to_id(spelling)
                        .filter(|id| specials.contains(id))
                        .ok_or_else(|| {
                            validation(Some(&specification.id), "missing Gemma3 special token")
                        })?;
                    if matches!(spelling, "<eos>" | "<end_of_turn>") && !eos.contains(&id) {
                        eos.push(id);
                    }
                }
            }
            for &token in &eos {
                if u64::from(token) >= as_u64(config.vocabulary())?
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
                gemma_decoder,
                max_token_spelling_bytes,
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
            family,
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
        mut self,
        profiles: Vec<ModelProfile>,
        admission: AdmissionController,
        scheduler: Scheduler,
        retention: RetentionPolicy,
        memory_enforcer: Option<crate::memory_enforcer::MemoryEnforcer>,
    ) -> Result<EngineCore<FamilyRuntime>, EngineError> {
        if self.family == BackendFamily::Gemma4 && admission.snapshot()?.budget_bytes() <= 8 << 30 {
            // E4B projects all prompt rows into its 262,144-token vocabulary.
            // Four rows bound this workspace during cold startup on the small tier.
            self.limits.prefill_chunk_tokens =
                self.limits.prefill_chunk_tokens.min(NonZeroUsize::new(4).unwrap());
        }
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
            || artifact.declared_context_limit_tokens() != data.config.context()
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
include!("hybrid_loading.rs");
include!("phi_loading.rs");
include!("gemma3_loading.rs");
include!("gemma4_loading.rs");

impl RuntimeBackend for Qwen3Backend {
    fn supports_json_output(&self) -> bool {
        true
    }

    fn json_constraint(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        schema: crate::structured_output::JsonOutputSchema,
        control: &ExecutionControl,
    ) -> Result<crate::structured_output::JsonConstraint, EngineError> {
        let resolved = registry.resolve_profile(profile)?;
        let data = self.data(resolved.artifact())?;
        if data.max_token_spelling_bytes > 64 << 10 {
            return Err(EngineError::unsupported(
                "structured output",
                "tokenizer spelling scratch exceeds 64 KiB",
            ));
        }
        constraints::build(data, schema, control)
    }

    fn verify_retained_prefix_bytes(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        tokenizer: &TokenizerId,
        tokens: &[u32],
        bytes: &[u8],
    ) -> Result<(), EngineError> {
        let resolved = registry.resolve_profile(profile)?;
        let data = self.data(resolved.artifact())?;
        if data.tokenizer.identity() != tokenizer {
            return Err(EngineError::invalid_request(
                "token_prefix_bytes",
                "retained state uses a different tokenizer",
            ));
        }
        rendering::verify_token_prefix_bytes(&data.tokenizer, tokens, bytes)
    }
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
            || rendered.template_hash() != &digest(family_template(self.family, true).as_bytes())
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
        if self.family == BackendFamily::Gemma4 {
            return rendering::gemma4_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
            );
        }
        if self.family == BackendFamily::Ministral3 {
            return rendering::ministral3_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
            );
        }
        if self.family == BackendFamily::Gemma3 {
            return rendering::gemma3_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
            );
        }
        if self.family == BackendFamily::Phi {
            return rendering::phi_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
            );
        }
        if self.family.hybrid() {
            return rendering::qwen35_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
                projection.additional_ascii_escape_bytes,
            );
        }
        if self.family == BackendFamily::SmolLm3 {
            return rendering::smollm3_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
            );
        }
        if self.family == BackendFamily::Llama3 {
            return rendering::llama3_projected_tool_token_upper_bound(
                &data.tokenizer,
                rendered,
                projection.result_count,
                projection.additional_rendered_bytes,
            );
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
            let template = family_template(self.family, tool_mode);
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
                &family_rendering_options(self.family),
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
            let vocabulary = as_u64(data.config.vocabulary())?;
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
            // Full f32 KV overlap dominates long Llama requests. Bound the
            // per-layer attention workspace further once capacity exceeds 16k.
            let chunk_tokens = data
                .config
                .chunk_limit(self.limits.prefill_chunk_tokens.get(), capacity);
            let prefill_chunk_tokens =
                NonZeroUsize::new(chunk_tokens).expect("nonzero configured chunk");
            Ok(RuntimePlan {
                load_limits: ModelLoadLimits::new(
                    data.source_bytes,
                    data.resident_bound,
                    load_peak,
                )?,
                cache: CacheAllocationRequest::new(
                    family_cache_mode(self.family)?,
                    capacity,
                    sum(&[
                        data.config.cache_bound(capacity, chunk_tokens)?,
                        product(self.limits.execution_headroom_bytes, 2)?,
                    ])?,
                )?,
                // Prefill cannot process more tokens than this request
                // supplies. Keep a one-token allowance for decode work.
                temporary_bytes: data.config.workspace_bound(
                    capacity,
                    chunk_tokens.min(input_tokens.max(1)),
                    self.limits.max_token_bytes,
                )?,
                output_bytes: self.planned_output_bytes()?,
                headroom_bytes: self.limits.execution_headroom_bytes,
                prefill_chunk_tokens,
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
        if self.tool_call_parsing {
            return Err(EngineError::unsupported(
                "admitted parser storage",
                "tool generation requires the storage-backed decoder construction hook",
            ));
        }
        self.json_decoder(registry, profile)
    }

    fn json_decoder(
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
            gemma_decoder: data.gemma_decoder,
            pending: Vec::new(),
            max_token_bytes: self.limits.max_token_bytes,
            finished: false,
            parser: None,
            slab: None,
            xml: None,
            token_protocol: None,
            phi: None,
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

    fn decoder_with_tool_context(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        allocation: Option<ParserAllocation>,
        tools: Option<Arc<crate::types::ToolContext>>,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        if !self.family.hybrid() || !self.tool_call_parsing {
            return self.decoder_with_parser_allocation(registry, profile, allocation);
        }
        let resolved = registry.resolve_profile(profile)?;
        let data = self.data(resolved.artifact())?;
        let limits = self
            .selected_parser_capacities()?
            .ok_or_else(|| internal("missing hybrid parser capacities"))?;
        let expected = ParserAllocation::required_bytes(limits)
            .map_err(|e| EngineError::invalid_request("parser capacities", e.to_string()))?;
        let allocation = allocation.ok_or_else(|| {
            EngineError::invalid_request(
                "parser allocation",
                "hybrid tools require the admitted slab",
            )
        })?;
        if allocation.reserved_bytes() != expected {
            return Err(internal(
                "hybrid parser allocation differs from planned size",
            ));
        }
        let xml = crate::slab_parser::XmlToolParser::new(
            allocation,
            output::HybridToolSchema(tools),
            self.tool_parser_limits().max_output_bytes,
        )?;
        output::box_slab_decoder(Qwen3TextDecoder {
            tokenizer: Arc::clone(&data.tokenizer),
            special_tokens: Arc::clone(&data.special_tokens),
            byte_decoder: Arc::clone(&data.byte_decoder),
            gemma_decoder: data.gemma_decoder,
            pending: Vec::new(),
            max_token_bytes: self.limits.max_token_bytes,
            finished: false,
            parser: None,
            slab: None,
            xml: Some(xml),
            token_protocol: None,
            phi: None,
            completed_slab: None,
        })
    }

    fn decoder_with_parser_allocation(
        &self,
        registry: &Registry,
        profile: &ModelProfileId,
        allocation: Option<ParserAllocation>,
    ) -> Result<Box<dyn TextDecoder>, EngineError> {
        if self.family.hybrid() && self.tool_call_parsing {
            return Err(EngineError::invalid_request(
                "tool schemas",
                "use the generation-bound decoder hook",
            ));
        }
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
                let max_output = self.tool_parser_limits().max_output_bytes;
                let (slab, phi, token_protocol) = match self.family {
                    BackendFamily::Gemma4 => (
                        None,
                        None,
                        Some(output::TokenProtocol::Gemma4 {
                            parser: crate::slab_parser::gemma4::Gemma4ToolParser::new(
                                allocation, max_output,
                            ),
                            eos: None,
                        }),
                    ),
                    BackendFamily::Ministral3 => (
                        None,
                        None,
                        Some(output::TokenProtocol::Ministral(
                            crate::slab_parser::ministral3::MinistralToolParser::new(
                                allocation, max_output,
                            ),
                        )),
                    ),
                    BackendFamily::Phi => (
                        None,
                        Some(
                            crate::slab_parser::PhiToolParser::new(allocation, max_output)?
                                .with_terminal_list(),
                        ),
                        None,
                    ),
                    _ => {
                        let parser = crate::slab_parser::SlabParser::new(allocation, max_output)?;
                        let parser = if self.family == BackendFamily::Gemma3 {
                            parser.with_terminal_call()
                        } else {
                            parser
                        };
                        (Some(parser), None, None)
                    }
                };
                output::box_slab_decoder(Qwen3TextDecoder {
                    tokenizer: Arc::clone(&data.tokenizer),
                    special_tokens: Arc::clone(&data.special_tokens),
                    byte_decoder: Arc::clone(&data.byte_decoder),
                    gemma_decoder: data.gemma_decoder,
                    pending: Vec::new(),
                    max_token_bytes: self.limits.max_token_bytes,
                    finished: false,
                    parser: None,
                    slab,
                    xml: None,
                    phi,
                    token_protocol,
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

fn registration_for(
    family: BackendFamily,
    context_limit: usize,
    execution: Qwen3Execution,
) -> Result<FamilyRegistration, EngineError> {
    let hybrid = family.hybrid();
    let llama = family == BackendFamily::Llama3;
    let smollm = family == BackendFamily::SmolLm3;
    let ministral = family == BackendFamily::Ministral3;
    let phi = family == BackendFamily::Phi;
    let gemma = family == BackendFamily::Gemma3;
    let mut representations = vec![
        WeightRepresentation::F32,
        WeightRepresentation::Affine4 {
            group_size: NonZeroUsize::new(64).unwrap(),
        },
    ];
    if llama || smollm || hybrid || phi || gemma || ministral {
        representations.push(WeightRepresentation::Affine8 {
            group_size: NonZeroUsize::new(64).unwrap(),
        });
    }
    let gemma4 = family == BackendFamily::Gemma4;
    if gemma4 {
        representations = vec![WeightRepresentation::Affine4 {
            group_size: NonZeroUsize::new(64).unwrap(),
        }];
    }
    FamilyRegistration::new(
        // Packed vocabulary lookup and quantized output projection change
        // numerical execution. Old native/disk state must miss this version.
        FamilyVersion::new(
            if hybrid || phi || gemma || gemma4 || ministral {
                1
            } else if smollm {
                smollm::register::IMPL_VERSION
            } else if llama {
                llama3::register::IMPL_VERSION
            } else {
                qwen3::register::IMPL_VERSION
            },
            5,
            0,
        ),
        (if ministral {
            &["Mistral3ForConditionalGeneration"][..]
        } else if gemma4 {
            &["Gemma4ForConditionalGeneration"][..]
        } else if gemma {
            &["Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"][..]
        } else if phi {
            &["Phi3ForCausalLM"][..]
        } else if hybrid {
            &["Qwen3_5ForConditionalGeneration"][..]
        } else if smollm {
            smollm::register::ARCHITECTURES
        } else if llama {
            llama3::register::ARCHITECTURES
        } else {
            qwen3::register::ARCHITECTURES
        })
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
        Capabilities {
            task: "text-generation".to_owned(),
            family: if ministral {
                "ministral3"
            } else if gemma4 {
                "gemma4"
            } else if gemma {
                "gemma3"
            } else if phi {
                "phi"
            } else if hybrid {
                "qwen35"
            } else if smollm {
                "smollm3"
            } else if llama {
                "llama3"
            } else {
                "qwen3"
            }
            .to_owned(),
            backend: execution.backend(),
            weight_representations: representations,
            cache_modes: vec![family_cache_mode(family)?],
            context_limit_tokens: context_limit,
            supports_streaming: true,
            supports_incremental_kv: true,
            supports_prefix_reuse: !gemma && !gemma4,
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

fn family_rendering_options(family: BackendFamily) -> RenderingOptions<'static> {
    let mut options = qwen3_rendering_options();
    if family == BackendFamily::Ministral3 {
        options.reserved_spellings = &[
            "<s>",
            "</s>",
            "<unk>",
            "<pad>",
            "<SPECIAL_",
            "[INST]",
            "[/INST]",
            "[AVAILABLE_TOOLS]",
            "[/AVAILABLE_TOOLS]",
            "[TOOL_RESULTS]",
            "[/TOOL_RESULTS]",
            "[TOOL_CALLS]",
            "[IMG]",
            "[IMG_BREAK]",
            "[IMG_END]",
            "[PREFIX]",
            "[MIDDLE]",
            "[SUFFIX]",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "[TOOL_CONTENT]",
            "[AUDIO]",
            "[BEGIN_AUDIO]",
            "[ARGS]",
            "[CALL_ID]",
            "[THINK]",
            "[/THINK]",
        ];
    }
    if family == BackendFamily::Gemma4 {
        options.reserved_spellings = &[
            "<bos>",
            "<eos>",
            "<pad>",
            "<unk>",
            "<mask>",
            "<|tool>",
            "<tool|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            "<|\"|>",
            "<|think|>",
            "<|channel>",
            "<channel|>",
            "<|turn>",
            "<turn|>",
            "<|image>",
            "<|audio>",
            "<|image|>",
            "<|audio|>",
            "<image|>",
            "<audio|>",
            "<|video|>",
        ];
    }
    if family == BackendFamily::Gemma3 {
        // These are tokenizer control delimiters, never user text boundaries.
        options.reserved_spellings = &[
            "<bos>",
            "<eos>",
            "<pad>",
            "<unk>",
            "<start_of_",
            "<end_of_",
            "<image>",
            "<image_soft_token>",
            "<unused",
            "<mask>",
            "<0x",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ];
    }
    options
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

fn family_template(family: BackendFamily, tools: bool) -> &'static str {
    match (family, tools) {
        (BackendFamily::Ministral3, false) => MINISTRAL3_TEXT_TEMPLATE,
        (BackendFamily::Ministral3, true) => MINISTRAL3_TOOL_TEMPLATE,
        (BackendFamily::Gemma4, false) => GEMMA4_TEXT_TEMPLATE,
        (BackendFamily::Gemma4, true) => GEMMA4_TOOL_TEMPLATE,
        (BackendFamily::Gemma3, false) => GEMMA3_TEXT_TEMPLATE,
        (BackendFamily::Gemma3, true) => GEMMA3_TOOL_TEMPLATE,
        (BackendFamily::Phi, false) => PHI_TEXT_TEMPLATE,
        (BackendFamily::Phi, true) => PHI_TOOL_TEMPLATE,
        (BackendFamily::Ornith, false) => ORNITH_TEXT_TEMPLATE,
        (BackendFamily::Ornith, true) => ORNITH_TOOL_TEMPLATE,
        (BackendFamily::Qwen35, false) => QWEN35_TEXT_TEMPLATE,
        (BackendFamily::Qwen35, true) => QWEN35_TOOL_TEMPLATE,
        (BackendFamily::Qwen3, false) => CHAT_TEMPLATE,
        (BackendFamily::Qwen3, true) => qwen3::template::TOOL_CHAT_TEMPLATE,
        (BackendFamily::Llama3, false) => llama3::template::CHAT_TEMPLATE,
        (BackendFamily::Llama3, true) => llama3::template::TOOL_CHAT_TEMPLATE,
        (BackendFamily::SmolLm3, false) => smollm::template::CHAT_TEMPLATE,
        (BackendFamily::SmolLm3, true) => smollm::template::TOOL_CHAT_TEMPLATE,
    }
}
fn family_cache_mode(family: BackendFamily) -> Result<CacheMode, EngineError> {
    if family == BackendFamily::Ministral3 {
        return Ok(CacheMode {
            representation: CacheRepresentationId::new("ministral3.mlx-f32.full.bhtd.v1")?,
            kind: CacheKind::Full,
        });
    }
    if family == BackendFamily::Gemma4 {
        return Ok(CacheMode {
            representation: CacheRepresentationId::new(
                "gemma4.mlx-f32.shared-full-and-sliding.v1",
            )?,
            kind: CacheKind::Mixed,
        });
    }
    if family == BackendFamily::Gemma3 {
        return Ok(CacheMode {
            representation: CacheRepresentationId::new(GEMMA3_CACHE_REPRESENTATION)?,
            kind: CacheKind::Mixed,
        });
    }
    if family == BackendFamily::Phi {
        return Ok(CacheMode {
            representation: CacheRepresentationId::new(PHI_CACHE_REPRESENTATION)?,
            kind: CacheKind::Full,
        });
    }
    if family.hybrid() {
        return Ok(CacheMode {
            representation: CacheRepresentationId::new(QWEN35_CACHE_REPRESENTATION)?,
            kind: CacheKind::Full,
        });
    }
    if family == BackendFamily::SmolLm3 {
        return Ok(CacheMode {
            representation: CacheRepresentationId::new(smollm::register::CACHE_REPRESENTATION)?,
            kind: CacheKind::Full,
        });
    }
    if family == BackendFamily::Llama3 {
        Ok(CacheMode {
            representation: CacheRepresentationId::new(llama3::register::CACHE_REPRESENTATION)?,
            kind: CacheKind::Full,
        })
    } else {
        cache_mode()
    }
}
