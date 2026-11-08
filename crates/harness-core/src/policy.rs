//! Default-deny, exact-argument policy preparation. No execution or approval grants.
//!
//! Rules are daemon configuration, never model metadata. This first policy mode
//! uses exact canonical argument allowlists, not path prefixes, DNS decisions,
//! filesystem canonicalization, or a sandbox. W4 must enforce resource policy at
//! actual operation boundaries; W6 must apply durable authorization before effects.

use crate::approval::{
    BindingError, BindingInputs, CONTRACT_VERSION, CanonicalizationError, InvocationBinding,
    canonical_json, identity_digest,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
use tool_registry::{
    RegistryError,
    call::ToolInvocation,
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin},
    registry::ToolRegistry,
};

/// Daemon-owned exact-match rule. Limits may only tighten registered limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRule {
    /// Registry identifier; no executable path or implementation selection.
    pub tool_id: String,
    /// Exact argument objects; empty allowlists are configuration errors.
    pub allowed_arguments: Vec<Map<String, Value>>,
    /// Require an explicit prompt even if registration does not require one.
    /// For registration `policy`, false is an explicit daemon-policy resolution;
    /// it never overrides `always` or destructive-effect prompting.
    pub require_approval: bool,
    /// Positive upper duration; effective value is min(rule, registration).
    pub timeout_ms: u64,
    /// Positive upper output bytes; effective value is min(rule, registration).
    pub max_output_bytes: u64,
}

/// Strict typed daemon configuration; no file loader or client RPC is added.
///
/// Its future loader must reject duplicate raw keys and bound input before
/// deserialization. Programmatic construction does not authorize any invocation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    /// One rule per registry ID; an empty vector is valid default-deny policy.
    pub rules: Vec<PolicyRule>,
}

#[derive(Serialize)]
struct CompiledRule {
    allowed_arguments: BTreeSet<String>,
    require_approval: bool,
    timeout_ms: u64,
    max_output_bytes: u64,
}

/// Immutable compiled policy. Mutating the source configuration cannot change it.
pub struct DaemonPolicy {
    rules: BTreeMap<String, CompiledRule>,
    revision: String,
}

/// Internal denial reason; not a new tool-result error category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyDenial {
    /// No explicit rule allows the registered identifier.
    NoMatchingRule,
    /// Arguments do not exactly match an allowed canonical object.
    ArgumentsNotAllowed,
    /// Builtin registration contradicts frozen effect/approval floors.
    BuiltinContractMismatch,
}

/// Preparation outcome only; neither successful variant permits effects.
#[derive(Debug)]
pub enum PolicyDecision {
    /// Denied before creating executable authority.
    Deny(PolicyDenial),
    /// Exact identity ready for later durable policy admission, not execution.
    AllowWithoutPrompt(Box<InvocationBinding>),
    /// Exact identity requiring later persisted explicit bound approval.
    RequireApproval(Box<InvocationBinding>),
}

/// Typed policy preparation failure; no wire enum expansion or hidden execution.
#[derive(Debug)]
pub enum PolicyError {
    /// Invalid or ambiguous daemon configuration.
    InvalidConfiguration(&'static str),
    /// Unknown tool or invalid registered-schema arguments/context.
    Registry(RegistryError),
    /// Canonical policy/arguments cannot fit their bounded identity encoding.
    Canonicalization(CanonicalizationError),
    /// Config metadata serialization failure.
    Serialization(serde_json::Error),
    /// Immutable binding construction failed.
    Binding(BindingError),
}
impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => write!(f, "invalid daemon policy: {reason}"),
            Self::Registry(error) => write!(f, "policy registry resolution: {error}"),
            Self::Canonicalization(error) => write!(f, "policy canonicalization: {error}"),
            Self::Serialization(error) => write!(f, "policy serialization: {error}"),
            Self::Binding(error) => write!(f, "policy binding: {error}"),
        }
    }
}
impl Error for PolicyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Registry(error) => Some(error),
            Self::Canonicalization(error) => Some(error),
            Self::Serialization(error) => Some(error),
            Self::Binding(error) => Some(error),
            Self::InvalidConfiguration(_) => None,
        }
    }
}
impl From<RegistryError> for PolicyError {
    fn from(error: RegistryError) -> Self {
        Self::Registry(error)
    }
}
impl From<CanonicalizationError> for PolicyError {
    fn from(error: CanonicalizationError) -> Self {
        Self::Canonicalization(error)
    }
}
impl From<serde_json::Error> for PolicyError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}
impl From<BindingError> for PolicyError {
    fn from(error: BindingError) -> Self {
        Self::Binding(error)
    }
}

impl DaemonPolicy {
    /// Compile strict exact-argument rules, reject duplicates/zero limits, and hash
    /// the complete normalized configuration. No registry or tool is executed.
    /// Rule order and argument allowlist order/duplicates do not affect identity.
    /// Large configs fail closed at the canonical identity encoder's ceiling.
    pub fn new(config: PolicyConfig) -> Result<Self, PolicyError> {
        let mut rules = BTreeMap::new();
        for rule in config.rules {
            if rule.tool_id.is_empty() || rule.tool_id.trim() != rule.tool_id {
                return Err(PolicyError::InvalidConfiguration("invalid tool_id"));
            }
            if rule.timeout_ms == 0 || rule.max_output_bytes == 0 {
                return Err(PolicyError::InvalidConfiguration("limits must be positive"));
            }
            if rule.allowed_arguments.is_empty() {
                return Err(PolicyError::InvalidConfiguration(
                    "empty argument allowlist",
                ));
            }
            let mut allowed = BTreeSet::new();
            for arguments in rule.allowed_arguments {
                let bytes = canonical_json(&Value::Object(arguments))?;
                let text = String::from_utf8(bytes)
                    .map_err(|_| PolicyError::InvalidConfiguration("canonical UTF-8 failure"))?;
                allowed.insert(text);
            }
            let compiled = CompiledRule {
                allowed_arguments: allowed,
                require_approval: rule.require_approval,
                timeout_ms: rule.timeout_ms,
                max_output_bytes: rule.max_output_bytes,
            };
            if rules.insert(rule.tool_id, compiled).is_some() {
                return Err(PolicyError::InvalidConfiguration("duplicate tool_id"));
            }
        }
        let normalized = serde_json::json!({
            "contract_version": CONTRACT_VERSION,
            "policy_mode": "exact_arguments_v1",
            "rules": &rules,
        });
        let revision = identity_digest(&canonical_json(&normalized)?);
        Ok(Self { rules, revision })
    }
    /// Complete normalized configuration identity; not an approval credential.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Resolve and schema-validate via the real registry, then prepare a binding.
    ///
    /// Epoch and creation time are daemon inputs, never client authority fields.
    /// This pure preparation does not check a live clock, reserve/persist a call,
    /// create an AuthorizedInvocation, prompt a human, or execute an implementation.
    pub fn evaluate(
        &self,
        registry: &ToolRegistry,
        invocation: ToolInvocation,
        daemon_epoch: &str,
        created_at_unix_ms: i64,
    ) -> Result<PolicyDecision, PolicyError> {
        let resolved = registry.resolve(invocation)?;
        let definition = resolved.definition();
        if !builtin_floor_holds(definition) {
            return Ok(PolicyDecision::Deny(PolicyDenial::BuiltinContractMismatch));
        }
        let Some(rule) = self.rules.get(&definition.id) else {
            return Ok(PolicyDecision::Deny(PolicyDenial::NoMatchingRule));
        };
        let arguments =
            canonical_json(&Value::Object(resolved.invocation().call.arguments.clone()))?;
        let arguments = std::str::from_utf8(&arguments)
            .map_err(|_| PolicyError::InvalidConfiguration("canonical UTF-8 failure"))?;
        if !rule.allowed_arguments.contains(arguments) {
            return Ok(PolicyDecision::Deny(PolicyDenial::ArgumentsNotAllowed));
        }
        let prompt = rule.require_approval
            || definition.policy.approval == ApprovalRequirement::Always
            || definition.policy.effect == ToolEffect::Destructive;
        let binding = InvocationBinding::from_resolved(
            &resolved,
            BindingInputs {
                daemon_epoch,
                policy_revision: &self.revision,
                effective_approval: if prompt {
                    ApprovalRequirement::Always
                } else {
                    ApprovalRequirement::Never
                },
                timeout_ms: rule.timeout_ms.min(definition.policy.timeout_ms),
                max_output_bytes: rule
                    .max_output_bytes
                    .min(definition.policy.max_output_bytes),
                created_at_unix_ms,
            },
        )?;
        Ok(if prompt {
            PolicyDecision::RequireApproval(Box::new(binding))
        } else {
            PolicyDecision::AllowWithoutPrompt(Box::new(binding))
        })
    }
}

fn builtin_floor_holds(definition: &ToolDefinition) -> bool {
    use ApprovalRequirement::{Always, Never};
    use ToolEffect::{Destructive, Write};
    if definition.origin != ToolOrigin::Builtin {
        return true;
    }
    match definition.id.as_str() {
        "shell.run" => {
            definition.policy.effect == Destructive && definition.policy.approval == Always
        }
        "fs.write" => {
            matches!(definition.policy.effect, Write | Destructive)
                && definition.policy.approval != Never
        }
        "http.fetch" => definition.policy.approval != Never,
        _ => true,
    }
}
