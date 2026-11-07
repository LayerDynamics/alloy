//! Registry-owned tool definitions and executable implementations.
//!
//! Registration is an explicit daemon operation. Deserializing a tool call
//! cannot create a registration or select an arbitrary executable.
//!
//! Resolution checks identity and arguments but does not authorize execution.
//! A resolved invocation is an intermediate state for the harness policy
//! layer, not an execution permit.

pub use crate::error::RegistryError;
use crate::{
    call::ToolInvocation,
    definition::ToolDefinition,
    result::ToolResult,
    validation::{ValidatedDefinition, ValidationError, validate_definition},
};

use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

/// Interface owned by a registered tool implementation.
///
/// This is the W1 implementation boundary. W4 must extend the execution
/// contract with cancellation, deadlines, and bounded output collection
/// before production builtins are connected.
///
/// The registry does not invoke this method during registration or
/// resolution. Implementing this trait grants no approval or policy bypass.
pub trait ToolImplementation: Send + Sync {
    /// Perform an invocation after harness authorization.
    ///
    /// Implementations must preserve the originating call identity in the
    /// result. Execution errors are represented in the returned tool result.
    fn execute(&self, invocation: &ToolInvocation) -> ToolResult;
}

struct RegisteredTool {
    definition: ValidatedDefinition,

    // Retained by the registry, never exposed through untrusted metadata.
    // W4 will consume this field through the authorized execution boundary.
    _implementation: Arc<dyn ToolImplementation>,
}

/// Explicitly populated collection of validated tools.
///
/// BTreeMap provides stable identifier ordering when definitions are
/// enumerated. It does not by itself establish a canonical schema hash.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, RegisteredTool>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one definition with its executable implementation.
    ///
    /// Invalid definitions and duplicate identifiers leave the registry
    /// unchanged. Registration never executes the implementation.
    pub fn register(
        &mut self,
        definition: ToolDefinition,
        implementation: Arc<dyn ToolImplementation>,
    ) -> Result<(), RegistryError> {
        let definition = validate_definition(definition)?;
        let id = definition.definition().id.clone();

        match self.tools.entry(id) {
            Entry::Occupied(entry) => Err(RegistryError::DuplicateId(entry.key().clone())),
            Entry::Vacant(entry) => {
                entry.insert(RegisteredTool {
                    definition,
                    _implementation: implementation,
                });
                Ok(())
            }
        }
    }

    /// Number of registered identifiers.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry has no registrations.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Inspect a registered definition without permitting mutation.
    pub fn definition(&self, id: &str) -> Option<&ToolDefinition> {
        self.tools
            .get(id)
            .map(|entry| entry.definition.definition())
    }

    /// Enumerate definitions in identifier order.
    pub fn definitions(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools
            .values()
            .map(|entry| entry.definition.definition())
    }

    /// Resolve an invocation and validate its arguments without executing it.
    ///
    /// The harness supplies context. All association identifiers must be
    /// nonempty and free of surrounding whitespace.
    pub fn resolve(
        &self,
        invocation: ToolInvocation,
    ) -> Result<ResolvedInvocation<'_>, RegistryError> {
        for (field, value) in [
            ("session_id", invocation.context.session_id.as_str()),
            ("turn_id", invocation.context.turn_id.as_str()),
            ("generation_id", invocation.context.generation_id.as_str()),
        ] {
            if value.is_empty() || value.trim() != value {
                return Err(ValidationError::InvalidIdentity(field).into());
            }
        }

        let tool = self
            .tools
            .get(&invocation.call.tool_id)
            .ok_or_else(|| RegistryError::UnknownTool(invocation.call.tool_id.clone()))?;

        tool.definition.validate_call(&invocation.call)?;

        Ok(ResolvedInvocation {
            definition: tool.definition.definition(),
            invocation,
        })
    }
}

/// Immutable validated request linked to its registered definition.
///
/// Fields are private so callers cannot replace validated arguments or
/// metadata after resolution. This type grants no approval and contains
/// no public executable handle.
pub struct ResolvedInvocation<'registry> {
    definition: &'registry ToolDefinition,
    invocation: ToolInvocation,
}

impl ResolvedInvocation<'_> {
    /// Registered metadata used during validation.
    pub fn definition(&self) -> &ToolDefinition {
        self.definition
    }

    /// Validated request and its daemon-supplied association.
    pub fn invocation(&self) -> &ToolInvocation {
        &self.invocation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        call::{ToolCall, ToolCallContext},
        definition::{ApprovalRequirement, ToolEffect, ToolOrigin, ToolPolicy},
        result::InlineToolResult,
    };
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTool {
        executions: Arc<AtomicUsize>,
    }

    impl ToolImplementation for CountingTool {
        fn execute(&self, invocation: &ToolInvocation) -> ToolResult {
            self.executions.fetch_add(1, Ordering::SeqCst);

            ToolResult::Inline(InlineToolResult {
                call_id: invocation.call.call_id.clone(),
                content: json!({ "ok": true }),
                is_error: false,
            })
        }
    }

    fn implementation(executions: &Arc<AtomicUsize>) -> Arc<dyn ToolImplementation> {
        Arc::new(CountingTool {
            executions: Arc::clone(executions),
        })
    }

    fn definition(id: &str) -> ToolDefinition {
        ToolDefinition {
            id: id.into(),
            version: "1.0.0".into(),
            description: "Read an allowed file.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            origin: ToolOrigin::Builtin,
            policy: ToolPolicy {
                effect: ToolEffect::Read,
                approval: ApprovalRequirement::Never,
                timeout_ms: 5_000,
                max_output_bytes: 65_536,
            },
        }
    }

    fn invocation(id: &str, arguments: Value) -> ToolInvocation {
        ToolInvocation {
            context: ToolCallContext {
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                generation_id: "generation-1".into(),
            },
            call: ToolCall {
                call_id: "call-1".into(),
                tool_id: id.into(),
                arguments: arguments.as_object().expect("object").clone(),
            },
        }
    }

    #[test]
    fn registry_schema_normalization() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();

        for (id, origin) in [
            ("builtin.read", ToolOrigin::Builtin),
            ("plugin.read", ToolOrigin::Plugin),
            ("mcp.read", ToolOrigin::Mcp),
        ] {
            let mut candidate = definition(id);
            candidate.origin = origin;

            registry
                .register(candidate, implementation(&executions))
                .expect("registration");
        }

        let duplicate = registry.register(definition("builtin.read"), implementation(&executions));
        assert!(matches!(duplicate, Err(RegistryError::DuplicateId(_))));

        let mut invalid = definition("invalid.read");
        invalid.input_schema["properties"]["path"]["type"] = json!(42);

        assert!(matches!(
            registry.register(invalid, implementation(&executions)),
            Err(RegistryError::Validation(ValidationError::InvalidSchema(_)))
        ));

        assert_eq!(registry.len(), 3);
        assert!(registry.definition("invalid.read").is_none());
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn tool_arguments_rejected_before_execution() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();

        registry
            .register(definition("fs.read"), implementation(&executions))
            .expect("registration");

        for arguments in [
            json!({}),
            json!({ "path": 42 }),
            json!({ "path": "fixture.txt", "extra": true }),
        ] {
            assert!(registry.resolve(invocation("fs.read", arguments)).is_err());
        }

        assert!(matches!(
            registry.resolve(invocation("unknown.tool", json!({ "path": "fixture.txt" }))),
            Err(RegistryError::UnknownTool(_))
        ));

        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn successful_resolution_is_not_execution() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();

        registry
            .register(definition("fs.read"), implementation(&executions))
            .expect("registration");

        let resolved = registry
            .resolve(invocation("fs.read", json!({ "path": "fixture.txt" })))
            .expect("resolved");

        assert_eq!(resolved.definition().id, "fs.read");
        assert_eq!(resolved.invocation().call.call_id, "call-1");
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }
}
