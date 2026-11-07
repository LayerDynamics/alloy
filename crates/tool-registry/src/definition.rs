//! Tool registration metadata.
//!
//! These types describe a candidate registration. Deserialization does not
//! validate its JSON Schema, register an implementation, or authorize a call.
//! The registry must validate definitions before accepting them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Descriptive metadata for one registered tool implementation.
///
/// `version` must be a valid semantic version. `input_schema` must be a
/// supported, valid JSON Schema. Both are checked at registration time.
///
/// The executable implementation is stored separately by the registry;
/// metadata cannot name a filesystem executable or manufacture a registration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDefinition {
    /// Unique registry identifier, such as `fs.read`.
    pub id: String,

    /// Semantic version of this tool's contract.
    pub version: String,

    /// Human- and model-readable description.
    pub description: String,

    /// Argument schema, serialized using the B.2 `inputSchema` field name.
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,

    /// Registration source; not an executable location or authorization.
    pub origin: ToolOrigin,

    /// Declared effect, approval requirement, and execution limits.
    pub policy: ToolPolicy,
}

/// Category of the component supplying a registration.
///
/// Provider-specific identity belongs in registry-owned provenance rather
/// than a model-supplied path. The MCP variant reserves the contract category;
/// this crate does not implement MCP transport or discovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOrigin {
    Builtin,
    Plugin,
    Mcp,
}

/// Declared policy requirements for a tool.
///
/// Registration validation must reject zero limits. Authorization remains
/// daemon-owned: these declarations do not grant access to any resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPolicy {
    /// Effect classification used by daemon policy.
    pub effect: ToolEffect,

    /// When explicit user approval is required.
    pub approval: ApprovalRequirement,

    /// Maximum execution duration in milliseconds.
    pub timeout_ms: u64,

    /// Maximum inline output size in bytes.
    pub max_output_bytes: u64,
}

/// Effect classification. Declaration order is not an authorization ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffect {
    Read,
    Write,
    Destructive,
}

/// Approval requirement declared by the registration.
///
/// `Never` skips the human approval prompt, not argument validation or
/// resource-scope checks. `Policy` delegates the approval requirement to
/// daemon policy. `Always` requires an explicit bound approval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalRequirement {
    Never,
    Policy,
    Always,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn definition_json() -> Value {
        json!({
            "id": "fs.read",
            "version": "1.0.0",
            "description": "Read a bounded range from an allowed file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            },
            "origin": "builtin",
            "policy": {
                "effect": "read",
                "approval": "never",
                "timeout_ms": 5_000,
                "max_output_bytes": 65_536
            }
        })
    }

    #[test]
    fn definition_roundtrips_with_b2_field_names() {
        let expected = definition_json();
        let definition: ToolDefinition =
            serde_json::from_value(expected.clone()).expect("valid metadata");

        assert_eq!(definition.id, "fs.read");
        assert_eq!(definition.origin, ToolOrigin::Builtin);
        assert_eq!(definition.policy.effect, ToolEffect::Read);
        assert_eq!(definition.policy.approval, ApprovalRequirement::Never);

        let actual = serde_json::to_value(definition).expect("serialize");
        assert_eq!(actual, expected);
        assert!(actual.get("input_schema").is_none());
    }

    #[test]
    fn unknown_definition_fields_are_rejected() {
        let mut value = definition_json();
        value["executable"] = json!("/bin/sh");

        assert!(serde_json::from_value::<ToolDefinition>(value).is_err());
    }

    #[test]
    fn unknown_policy_fields_are_rejected() {
        let mut value = definition_json();
        value["policy"]["auto_approve"] = json!(true);

        assert!(serde_json::from_value::<ToolDefinition>(value).is_err());
    }

    #[test]
    fn missing_policy_is_rejected() {
        let mut value = definition_json();
        value.as_object_mut().expect("object").remove("policy");

        assert!(serde_json::from_value::<ToolDefinition>(value).is_err());
    }

    #[test]
    fn unknown_effect_is_rejected() {
        let mut value = definition_json();
        value["policy"]["effect"] = json!("safe");

        assert!(serde_json::from_value::<ToolDefinition>(value).is_err());
    }

    #[test]
    fn origin_categories_have_explicit_wire_values() {
        for (origin, expected) in [
            (ToolOrigin::Builtin, "builtin"),
            (ToolOrigin::Plugin, "plugin"),
            (ToolOrigin::Mcp, "mcp"),
        ] {
            assert_eq!(
                serde_json::to_value(origin).expect("serialize"),
                json!(expected)
            );
        }
    }
}

#[cfg(test)]
mod definition_contract_tests {
    use super::*;
    use serde_json::{Value, json};

    fn fixture() -> Value {
        json!({
            "id": "shell.run",
            "version": "1.0.0",
            "description": "Run an exact executable and argument array.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "executable": { "type": "string" },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "cwd": { "type": "string" }
                },
                "required": ["executable", "args", "cwd"],
                "additionalProperties": false
            },
            "origin": "builtin",
            "policy": {
                "effect": "destructive",
                "approval": "always",
                "timeout_ms": 10000,
                "max_output_bytes": 65536
            }
        })
    }

    #[test]
    fn shell_definition_preserves_its_declared_policy() {
        let definition: ToolDefinition = serde_json::from_value(fixture()).expect("definition");

        assert_eq!(definition.id, "shell.run");
        assert_eq!(definition.policy.effect, ToolEffect::Destructive);
        assert_eq!(definition.policy.approval, ApprovalRequirement::Always);
        assert_eq!(definition.policy.timeout_ms, 10_000);
        assert_eq!(definition.policy.max_output_bytes, 65_536);
    }

    #[test]
    fn every_definition_field_is_required() {
        for field in [
            "id",
            "version",
            "description",
            "inputSchema",
            "origin",
            "policy",
        ] {
            let mut value = fixture();
            value.as_object_mut().expect("object").remove(field);

            assert!(
                serde_json::from_value::<ToolDefinition>(value).is_err(),
                "missing field accepted: {field}"
            );
        }
    }

    #[test]
    fn every_policy_field_is_required() {
        for field in ["effect", "approval", "timeout_ms", "max_output_bytes"] {
            let mut value = fixture();
            value["policy"]
                .as_object_mut()
                .expect("policy")
                .remove(field);

            assert!(
                serde_json::from_value::<ToolDefinition>(value).is_err(),
                "missing policy field accepted: {field}"
            );
        }
    }

    #[test]
    fn input_schema_uses_exact_b2_wire_name() {
        let original = fixture();
        let definition: ToolDefinition =
            serde_json::from_value(original.clone()).expect("definition");

        let serialized = serde_json::to_value(definition).expect("serialize");

        assert_eq!(serialized, original);
        assert!(serialized.get("input_schema").is_none());

        let mut wrong_name = fixture();
        let schema = wrong_name
            .as_object_mut()
            .expect("object")
            .remove("inputSchema")
            .expect("schema");
        wrong_name["input_schema"] = schema;

        assert!(serde_json::from_value::<ToolDefinition>(wrong_name).is_err());
    }

    #[test]
    fn unsigned_limits_reject_negative_fractional_and_string_values() {
        for field in ["timeout_ms", "max_output_bytes"] {
            for invalid in [
                json!(-1),
                json!(1.5),
                json!("1000"),
                json!(true),
                Value::Null,
            ] {
                let mut value = fixture();
                value["policy"][field] = invalid;

                assert!(
                    serde_json::from_value::<ToolDefinition>(value).is_err(),
                    "invalid limit accepted for {field}"
                );
            }
        }
    }

    #[test]
    fn zero_limits_deserialize_but_are_not_registration_validation() {
        let mut value = fixture();
        value["policy"]["timeout_ms"] = json!(0);
        value["policy"]["max_output_bytes"] = json!(0);

        let definition: ToolDefinition = serde_json::from_value(value).expect("typed metadata");

        assert_eq!(definition.policy.timeout_ms, 0);
        assert_eq!(definition.policy.max_output_bytes, 0);

        // validation.rs is responsible for rejecting these at registration.
    }

    #[test]
    fn effect_wire_values_are_explicit() {
        for (effect, wire) in [
            (ToolEffect::Read, "read"),
            (ToolEffect::Write, "write"),
            (ToolEffect::Destructive, "destructive"),
        ] {
            assert_eq!(
                serde_json::to_value(effect).expect("serialize"),
                json!(wire)
            );
            assert_eq!(
                serde_json::from_value::<ToolEffect>(json!(wire)).expect("deserialize"),
                effect
            );
        }
    }

    #[test]
    fn approval_wire_values_are_explicit() {
        for (approval, wire) in [
            (ApprovalRequirement::Never, "never"),
            (ApprovalRequirement::Policy, "policy"),
            (ApprovalRequirement::Always, "always"),
        ] {
            assert_eq!(
                serde_json::to_value(approval).expect("serialize"),
                json!(wire)
            );
            assert_eq!(
                serde_json::from_value::<ApprovalRequirement>(json!(wire)).expect("deserialize"),
                approval
            );
        }
    }

    #[test]
    fn unknown_enum_values_do_not_fall_back_to_defaults() {
        assert!(serde_json::from_value::<ToolEffect>(json!("safe")).is_err());
        assert!(serde_json::from_value::<ApprovalRequirement>(json!("auto_approve")).is_err());
        assert!(serde_json::from_value::<ToolOrigin>(json!("filesystem")).is_err());
    }

    #[test]
    fn clone_preserves_schema_and_is_independent_of_source_mutation() {
        let original: ToolDefinition = serde_json::from_value(fixture()).expect("definition");
        let mut changed = original.clone();

        changed.id = "different.tool".into();
        changed.input_schema["properties"]["cwd"]["type"] = json!("integer");
        changed.policy.timeout_ms = 1;

        assert_eq!(original.id, "shell.run");
        assert_eq!(original.input_schema["properties"]["cwd"]["type"], "string");
        assert_eq!(original.policy.timeout_ms, 10_000);
    }
}
