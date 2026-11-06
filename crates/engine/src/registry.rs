//! Model-family, artifact, and profile registry.
//!
//! Families register explicit architecture identifiers and capabilities.
//! Artifacts are inspected from actual local config.json files. Profiles
//! bind registered artifacts to bounded execution settings.
//!
//! Configuration inspection is not weight verification or model loading.
//! A concrete engine must verify artifact integrity, load the real model,
//! and perform admission before generation.
//!
//! Registry mutations require exclusive access. Once shared, callers may
//! protect this registry with their chosen synchronization mechanism.

use crate::{
    error::{EngineError, MemoryLimitReason},
    types::{
        ArtifactId, Capabilities, DecodingCapability, DecodingMode, GenerationOptions,
        ModelProfileId,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// Maximum accepted config.json size.
///
/// This bounds the source bytes, not all allocations performed by JSON
/// parsing or the total memory occupied by the registry.
pub const MAX_CONFIG_BYTES: usize = 1024 * 1024;

/// Maximum length of an architecture or family identifier.
pub const MAX_REGISTRY_NAME_BYTES: usize = 128;

/// Maximum number of architecture identifiers in one registration or config.
pub const MAX_ARCHITECTURES: usize = 32;

/// Semantic version of a model-family implementation.
///
/// Numerical behavior and cache-layout changes require a deliberate
/// version change. This value is separate from the engine API version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FamilyVersion {
    /// Major implementation version.
    pub major: u32,
    /// Minor implementation version.
    pub minor: u32,
    /// Patch implementation version.
    pub patch: u32,
}

impl FamilyVersion {
    /// Construct an implementation version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl fmt::Display for FamilyVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Immutable metadata supplied by an actual family adapter.
///
/// Constructing this metadata does not implement an adapter. The engine
/// integration must associate the registered family with its real loader
/// and execution implementation.
#[derive(Clone, Debug)]
pub struct FamilyRegistration {
    implementation_version: FamilyVersion,
    architectures: Vec<String>,
    capabilities: Capabilities,
}

impl FamilyRegistration {
    /// Validate and construct a family registration.
    ///
    /// Capability values must describe the actual implementation. This
    /// constructor checks structural consistency, not execution behavior.
    pub fn new(
        implementation_version: FamilyVersion,
        architectures: Vec<String>,
        capabilities: Capabilities,
    ) -> Result<Self, EngineError> {
        validate_name(&capabilities.family, "family")?;

        if capabilities.task.is_empty() {
            return Err(EngineError::invalid_request(
                "task",
                "family task must not be empty",
            ));
        }

        if capabilities.context_limit_tokens == 0 {
            return Err(EngineError::invalid_request(
                "context_limit_tokens",
                "family context limit must be greater than zero",
            ));
        }

        if capabilities.weight_representations.is_empty() {
            return Err(EngineError::invalid_request(
                "weight_representations",
                "family must declare at least one weight representation",
            ));
        }

        if capabilities.decoding_modes.is_empty() {
            return Err(EngineError::invalid_request(
                "decoding_modes",
                "family must declare at least one decoding mode",
            ));
        }

        if capabilities
            .decoding_modes
            .iter()
            .enumerate()
            .any(|(index, mode)| capabilities.decoding_modes[..index].contains(mode))
        {
            return Err(EngineError::invalid_request(
                "decoding_modes",
                "duplicate decoding capability",
            ));
        }

        if capabilities.supports_prefix_reuse && !capabilities.supports_incremental_kv {
            return Err(EngineError::invalid_request(
                "supports_prefix_reuse",
                "prefix reuse requires incremental KV support",
            ));
        }

        if capabilities.supports_incremental_kv && capabilities.cache_modes.is_empty() {
            return Err(EngineError::invalid_request(
                "cache_modes",
                "incremental KV support requires a declared cache mode",
            ));
        }

        if !capabilities.supports_incremental_kv && !capabilities.cache_modes.is_empty() {
            return Err(EngineError::invalid_request(
                "cache_modes",
                "cache modes cannot be advertised without incremental KV support",
            ));
        }

        let mut representations = BTreeSet::new();
        for mode in &capabilities.cache_modes {
            if !representations.insert(mode.representation.as_str()) {
                return Err(EngineError::invalid_request(
                    "cache_modes",
                    "duplicate cache representation identifier",
                ));
            }
        }

        validate_architectures(&architectures)?;

        Ok(Self {
            implementation_version,
            architectures,
            capabilities,
        })
    }

    /// Return the registered family name.
    pub fn family(&self) -> &str {
        &self.capabilities.family
    }

    /// Return the family implementation version.
    pub fn implementation_version(&self) -> FamilyVersion {
        self.implementation_version
    }

    /// Borrow the architecture identifiers implemented by this family.
    pub fn architectures(&self) -> &[String] {
        &self.architectures
    }

    /// Borrow the implementation's declared capabilities.
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
}

/// Configuration inspected from an existing local artifact directory.
///
/// This record is explicitly not a weight-integrity validation report.
/// The directory may change after inspection; execution must verify the
/// artifact against its manifest before loading.
#[derive(Clone, Debug)]
pub struct ArtifactRegistration {
    id: ArtifactId,
    directory: PathBuf,
    architectures: Vec<String>,
    family: String,
    declared_context_limit_tokens: usize,
}

impl ArtifactRegistration {
    /// Return the artifact's registry identifier.
    pub fn id(&self) -> &ArtifactId {
        &self.id
    }

    /// Return the canonical local artifact directory.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Borrow the architectures declared by config.json.
    pub fn architectures(&self) -> &[String] {
        &self.architectures
    }

    /// Return the uniquely selected family.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Return max_position_embeddings declared by config.json.
    ///
    /// This is a declared limit, not evidence of validated long-context
    /// behavior or sufficient memory for that context.
    pub fn declared_context_limit_tokens(&self) -> usize {
        self.declared_context_limit_tokens
    }
}

/// Bounded execution settings associated with a registered artifact.
#[derive(Clone, Debug)]
pub struct ModelProfile {
    id: ModelProfileId,
    artifact_id: ArtifactId,
    context_limit_tokens: usize,
    max_output_tokens: usize,
    safety_margin_tokens: usize,
    default_generation: GenerationOptions,
}

impl ModelProfile {
    /// Construct structurally valid profile settings.
    ///
    /// Registry insertion additionally checks the selected artifact and
    /// family capabilities. This constructor does not admit execution.
    pub fn new(
        id: ModelProfileId,
        artifact_id: ArtifactId,
        context_limit_tokens: usize,
        max_output_tokens: usize,
        safety_margin_tokens: usize,
        default_generation: GenerationOptions,
    ) -> Result<Self, EngineError> {
        if context_limit_tokens == 0 {
            return Err(EngineError::invalid_request(
                "context_limit_tokens",
                "profile context limit must be greater than zero",
            ));
        }

        if max_output_tokens == 0 {
            return Err(EngineError::invalid_request(
                "max_output_tokens",
                "profile output allowance must be greater than zero",
            ));
        }

        let reserved = max_output_tokens
            .checked_add(safety_margin_tokens)
            .ok_or_else(|| {
                EngineError::invalid_request(
                    "max_output_tokens",
                    "output reserve and safety margin overflow",
                )
            })?;

        if reserved >= context_limit_tokens {
            return Err(EngineError::invalid_request(
                "context_limit_tokens",
                "output reserve and safety margin leave no input capacity",
            ));
        }

        if default_generation.max_output_tokens() > max_output_tokens {
            return Err(EngineError::invalid_request(
                "default_generation",
                "default output limit exceeds the profile output allowance",
            ));
        }

        Ok(Self {
            id,
            artifact_id,
            context_limit_tokens,
            max_output_tokens,
            safety_margin_tokens,
            default_generation,
        })
    }

    /// Return the profile identifier.
    pub fn id(&self) -> &ModelProfileId {
        &self.id
    }

    /// Return the selected artifact identifier.
    pub fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    /// Return the profile's bounded context limit.
    pub fn context_limit_tokens(&self) -> usize {
        self.context_limit_tokens
    }

    /// Return the maximum permitted output-token allowance.
    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    /// Return the context safety margin.
    pub fn safety_margin_tokens(&self) -> usize {
        self.safety_margin_tokens
    }

    /// Borrow the profile's default generation options.
    pub fn default_generation(&self) -> &GenerationOptions {
        &self.default_generation
    }

    /// Check input context against the profile's maximum output reserve.
    ///
    /// This does not perform memory admission or acquire a reservation.
    pub fn check_context(&self, input_tokens: usize) -> Result<(), EngineError> {
        let total = input_tokens
            .checked_add(self.max_output_tokens)
            .and_then(|value| value.checked_add(self.safety_margin_tokens));

        if total.is_none_or(|value| value > self.context_limit_tokens) {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens,
                output_reserve_tokens: self.max_output_tokens,
                safety_margin_tokens: self.safety_margin_tokens,
                limit_tokens: self.context_limit_tokens,
            });
        }

        Ok(())
    }
}

/// Borrowed profile, artifact, and family selected by registry lookup.
///
/// Resolving this metadata does not load a model or authorize generation.
#[derive(Debug)]
pub struct ResolvedProfile<'a> {
    profile: &'a ModelProfile,
    artifact: &'a ArtifactRegistration,
    family: &'a FamilyRegistration,
}

impl<'a> ResolvedProfile<'a> {
    /// Return the selected profile.
    pub fn profile(&self) -> &'a ModelProfile {
        self.profile
    }

    /// Return the selected artifact.
    pub fn artifact(&self) -> &'a ArtifactRegistration {
        self.artifact
    }

    /// Return the selected family implementation metadata.
    pub fn family(&self) -> &'a FamilyRegistration {
        self.family
    }
}

/// Deterministic family, artifact, and profile registry.
///
/// Existing entries are never silently replaced. Construct a replacement
/// registry or add an explicit lifecycle operation when replacement and
/// eviction semantics are implemented.
#[derive(Debug, Default)]
pub struct Registry {
    families: BTreeMap<String, FamilyRegistration>,
    architecture_owners: BTreeMap<String, String>,
    artifacts: BTreeMap<ArtifactId, ArtifactRegistration>,
    directory_owners: BTreeMap<PathBuf, ArtifactId>,
    profiles: BTreeMap<ModelProfileId, ModelProfile>,
}

impl Registry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a family and claim its architecture identifiers.
    ///
    /// All conflicts are checked before insertion. A failed registration
    /// leaves the registry unchanged.
    pub fn register_family(&mut self, registration: FamilyRegistration) -> Result<(), EngineError> {
        let family = registration.family().to_owned();

        if self.families.contains_key(&family) {
            return Err(EngineError::invalid_request(
                "family",
                format!("family is already registered: {family}"),
            ));
        }

        for architecture in registration.architectures() {
            if let Some(owner) = self.architecture_owners.get(architecture) {
                return Err(EngineError::invalid_request(
                    "architectures",
                    format!("architecture {architecture} is already owned by family {owner}"),
                ));
            }
        }

        for architecture in registration.architectures() {
            self.architecture_owners
                .insert(architecture.clone(), family.clone());
        }

        self.families.insert(family, registration);
        Ok(())
    }

    /// Return a registered family by name.
    pub fn family(&self, family: &str) -> Result<&FamilyRegistration, EngineError> {
        self.families
            .get(family)
            .ok_or_else(|| EngineError::NotFound {
                resource: "family",
                id: family.to_owned(),
            })
    }

    /// Select a unique registered family from declared architectures.
    ///
    /// Unknown architecture names do not create support. At least one
    /// must match, and matches belonging to different families are rejected
    /// as ambiguous rather than selecting whichever appears first.
    pub fn family_for_architectures(
        &self,
        architectures: &[String],
    ) -> Result<&FamilyRegistration, EngineError> {
        validate_architectures(architectures)?;

        let mut selected: Option<&str> = None;

        for architecture in architectures {
            let Some(owner) = self.architecture_owners.get(architecture) else {
                continue;
            };

            match selected {
                None => selected = Some(owner.as_str()),
                Some(existing) if existing == owner => {}
                Some(_) => {
                    return Err(EngineError::invalid_request(
                        "architectures",
                        "declared architectures match multiple registered families",
                    ));
                }
            }
        }

        let family = selected.ok_or_else(|| {
            EngineError::unsupported(
                "architecture",
                format!(
                    "no registered family implements: {}",
                    architectures.join(", ")
                ),
            )
        })?;

        self.family(family)
    }

    /// Inspect an actual artifact directory and register its configuration.
    ///
    /// Reads a bounded config.json and selects a unique registered family.
    /// This does not verify weight hashes, tokenizer files, or numerical
    /// conformance and does not load weights.
    ///
    /// Duplicate artifact IDs and directory aliases are rejected.
    pub fn register_artifact(
        &mut self,
        id: ArtifactId,
        directory: impl AsRef<Path>,
    ) -> Result<(), EngineError> {
        if self.artifacts.contains_key(&id) {
            return Err(EngineError::invalid_request(
                "artifact_id",
                format!("artifact is already registered: {id}"),
            ));
        }

        let supplied = directory.as_ref();
        let canonical = supplied.canonicalize().map_err(|source| {
            EngineError::checkpoint_io("canonicalize artifact directory", supplied, source)
        })?;

        let metadata = std::fs::metadata(&canonical).map_err(|source| {
            EngineError::checkpoint_io("inspect artifact directory", &canonical, source)
        })?;

        if !metadata.is_dir() {
            return Err(artifact_validation(
                &id,
                "artifact path must be a directory",
            ));
        }

        if let Some(owner) = self.directory_owners.get(&canonical) {
            return Err(EngineError::invalid_request(
                "artifact_directory",
                format!("directory is already registered under artifact {owner}"),
            ));
        }

        let config_path = canonical.join("config.json");
        let resolved_config = config_path.canonicalize().map_err(|source| {
            EngineError::checkpoint_io("canonicalize artifact config", &config_path, source)
        })?;

        if resolved_config.parent() != Some(canonical.as_path()) {
            return Err(artifact_validation(
                &id,
                "config.json resolves outside the artifact directory",
            ));
        }

        let bytes = read_config(&resolved_config)?;
        let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|source| {
            EngineError::checkpoint_format("json", Some(resolved_config.clone()), source)
        })?;

        let object = config
            .as_object()
            .ok_or_else(|| artifact_validation(&id, "config.json must contain an object"))?;

        let declared = object
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                artifact_validation(&id, "config.json must declare an architectures array")
            })?;

        if declared.is_empty() || declared.len() > MAX_ARCHITECTURES {
            return Err(artifact_validation(
                &id,
                "architectures array is empty or exceeds the configured limit",
            ));
        }

        let architectures: Vec<String> = declared
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| artifact_validation(&id, "each architecture must be a string"))
            })
            .collect::<Result<_, _>>()?;

        let family = self
            .family_for_architectures(&architectures)?
            .family()
            .to_owned();

        let declared_context_limit_tokens = object
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                artifact_validation(
                    &id,
                    "max_position_embeddings must be a positive supported integer",
                )
            })?;

        let registration = ArtifactRegistration {
            id: id.clone(),
            directory: canonical.clone(),
            architectures,
            family,
            declared_context_limit_tokens,
        };

        self.directory_owners.insert(canonical, id.clone());
        self.artifacts.insert(id, registration);
        Ok(())
    }

    /// Return an inspected artifact by identifier.
    ///
    /// The returned record is not proof that the artifact has passed
    /// integrity verification or actual loading.
    pub fn artifact(&self, id: &ArtifactId) -> Result<&ArtifactRegistration, EngineError> {
        self.artifacts.get(id).ok_or_else(|| EngineError::NotFound {
            resource: "artifact",
            id: id.as_str().to_owned(),
        })
    }

    /// Register a profile against an existing artifact and family.
    ///
    /// Rejects context limits beyond either the declared artifact limit
    /// or the family implementation's limit.
    pub fn register_profile(&mut self, profile: ModelProfile) -> Result<(), EngineError> {
        if self.profiles.contains_key(profile.id()) {
            return Err(EngineError::invalid_request(
                "profile_id",
                format!("profile is already registered: {}", profile.id()),
            ));
        }

        let artifact = self.artifact(profile.artifact_id())?;
        let family = self.family(artifact.family())?;

        let maximum = artifact
            .declared_context_limit_tokens()
            .min(family.capabilities().context_limit_tokens);

        if profile.context_limit_tokens() > maximum {
            return Err(EngineError::invalid_request(
                "context_limit_tokens",
                format!(
                    "profile context limit {} exceeds artifact/family limit {maximum}",
                    profile.context_limit_tokens()
                ),
            ));
        }

        validate_decoding(profile.default_generation(), family.capabilities())?;

        self.profiles.insert(profile.id().clone(), profile);
        Ok(())
    }

    /// Return a registered profile.
    pub fn profile(&self, id: &ModelProfileId) -> Result<&ModelProfile, EngineError> {
        self.profiles.get(id).ok_or_else(|| EngineError::NotFound {
            resource: "profile",
            id: id.as_str().to_owned(),
        })
    }

    /// Resolve a profile, its artifact, and its unique family.
    ///
    /// Resolution does not acquire execution ownership or memory.
    pub fn resolve_profile(&self, id: &ModelProfileId) -> Result<ResolvedProfile<'_>, EngineError> {
        let profile = self.profile(id)?;
        let artifact = self.artifact(profile.artifact_id())?;
        let family = self.family(artifact.family())?;

        Ok(ResolvedProfile {
            profile,
            artifact,
            family,
        })
    }

    /// Validate requested options against profile and family limits.
    ///
    /// This checks output allowance and decoding-mode support. It does
    /// not replace real context counting or memory admission.
    pub fn validate_generation_options(
        &self,
        profile_id: &ModelProfileId,
        options: &GenerationOptions,
    ) -> Result<(), EngineError> {
        let resolved = self.resolve_profile(profile_id)?;

        if options.max_output_tokens() > resolved.profile().max_output_tokens() {
            return Err(EngineError::invalid_request(
                "max_output_tokens",
                "requested output exceeds the profile output allowance",
            ));
        }

        validate_decoding(options, resolved.family().capabilities())
    }

    /// Iterate over registered families in deterministic name order.
    pub fn families(&self) -> impl Iterator<Item = &FamilyRegistration> {
        self.families.values()
    }

    /// Iterate over inspected artifacts in deterministic identifier order.
    pub fn artifacts(&self) -> impl Iterator<Item = &ArtifactRegistration> {
        self.artifacts.values()
    }

    /// Iterate over registered profiles in deterministic identifier order.
    pub fn profiles(&self) -> impl Iterator<Item = &ModelProfile> {
        self.profiles.values()
    }
}

fn validate_name(value: &str, field: &'static str) -> Result<(), EngineError> {
    if value.is_empty() || value.len() > MAX_REGISTRY_NAME_BYTES {
        return Err(EngineError::invalid_request(
            field,
            format!(
                "name must contain between 1 and \
                 {MAX_REGISTRY_NAME_BYTES} bytes"
            ),
        ));
    }

    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(EngineError::invalid_request(
            field,
            "name contains unsupported characters",
        ));
    }

    Ok(())
}

fn validate_architectures(architectures: &[String]) -> Result<(), EngineError> {
    if architectures.is_empty() || architectures.len() > MAX_ARCHITECTURES {
        return Err(EngineError::invalid_request(
            "architectures",
            format!("must declare between 1 and {MAX_ARCHITECTURES} architectures"),
        ));
    }

    let mut seen = BTreeSet::new();

    for architecture in architectures {
        validate_name(architecture, "architectures")?;

        if !seen.insert(architecture.as_str()) {
            return Err(EngineError::invalid_request(
                "architectures",
                "duplicate architecture identifier",
            ));
        }
    }

    Ok(())
}

fn validate_decoding(
    options: &GenerationOptions,
    capabilities: &Capabilities,
) -> Result<(), EngineError> {
    let required = match options.decoding() {
        DecodingMode::Greedy => DecodingCapability::Greedy,
        DecodingMode::Sampling { .. } => DecodingCapability::Sampling,
    };

    if !capabilities.decoding_modes.contains(&required) {
        return Err(EngineError::unsupported(
            "decoding mode",
            format!(
                "family {} does not implement {required:?}",
                capabilities.family
            ),
        ));
    }

    Ok(())
}

fn artifact_validation(artifact_id: &ArtifactId, message: impl Into<String>) -> EngineError {
    EngineError::ArtifactValidationFailed {
        artifact_id: Some(artifact_id.as_str().to_owned()),
        message: message.into(),
    }
}

fn read_config(path: &Path) -> Result<Vec<u8>, EngineError> {
    let mut file = File::open(path)
        .map_err(|source| EngineError::checkpoint_io("open artifact config", path, source))?;

    let metadata = file
        .metadata()
        .map_err(|source| EngineError::checkpoint_io("inspect artifact config", path, source))?;

    if !metadata.is_file() {
        return Err(EngineError::ArtifactValidationFailed {
            artifact_id: None,
            message: "config.json must be a regular file".to_owned(),
        });
    }

    if metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err(EngineError::MemoryLimitExceeded {
            operation: "read artifact config",
            required_bytes: metadata.len(),
            available_bytes: MAX_CONFIG_BYTES as u64,
            reason: MemoryLimitReason::BufferLimit,
        });
    }

    let length =
        usize::try_from(metadata.len()).map_err(|_| EngineError::ArtifactValidationFailed {
            artifact_id: None,
            message: "config.json length is not representable".to_owned(),
        })?;

    let mut bytes = Vec::new();

    bytes
        .try_reserve_exact(length)
        .map_err(|_| EngineError::AllocationFailed {
            operation: "read artifact config",
            requested_bytes: Some(metadata.len()),
        })?;

    bytes.resize(length, 0);

    file.read_exact(&mut bytes)
        .map_err(|source| EngineError::checkpoint_io("read artifact config", path, source))?;

    let mut probe = [0u8; 1];
    let additional = file.read(&mut probe).map_err(|source| {
        EngineError::checkpoint_io("check artifact config length", path, source)
    })?;

    if additional != 0 {
        return Err(EngineError::ArtifactValidationFailed {
            artifact_id: None,
            message: "config.json changed length during inspection".to_owned(),
        });
    }

    Ok(bytes)
}
