//! Registration and argument validation.
//!
//! A validated definition is not authorization to execute. This module
//! resolves no executable paths and performs no tool effects.
//!
//! Schema policy: Draft 2020-12, object-shaped arguments, explicit format
//! validation, and no external resource retrieval.

pub use crate::error::ValidationError;
use crate::{call::ToolCall, definition::ToolDefinition};

use jsonschema::{Retrieve, Uri, Validator};
use serde_json::Value;
use std::error::Error;

/// Immutable definition paired with its compiled argument validator.
///
/// Keeping both fields private prevents callers from changing metadata
/// without recompiling and revalidating the schema.
pub struct ValidatedDefinition {
    definition: ToolDefinition,
    validator: Validator,
}

impl ValidatedDefinition {
    /// Inspect the accepted definition without permitting mutation.
    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    /// Check the request identity and arguments.
    ///
    /// Success does not grant policy approval or permission to execute.
    pub fn validate_call(&self, call: &ToolCall) -> Result<(), ValidationError> {
        validate_identity("call_id", &call.call_id)?;
        validate_identity("tool_id", &call.tool_id)?;

        if call.tool_id != self.definition.id {
            return Err(ValidationError::ToolMismatch);
        }

        let arguments = Value::Object(call.arguments.clone());

        self.validator
            .validate(&arguments)
            .map_err(|error| ValidationError::InvalidArguments(error.to_string()))
    }
}

/// Validate registration metadata and compile its argument schema.
///
/// Proposed W1 schema policy:
/// - Draft 2020-12 only.
/// - Root schema explicitly declares `"type": "object"`.
/// - Root schema explicitly declares `"additionalProperties": false`.
/// - External references cannot retrieve network or filesystem resources.
///
/// Unknown tool IDs and duplicate registrations are registry concerns.
pub fn validate_definition(
    definition: ToolDefinition,
) -> Result<ValidatedDefinition, ValidationError> {
    validate_identity("tool id", &definition.id)?;

    semver::Version::parse(&definition.version)
        .map_err(|error| ValidationError::InvalidVersion(error.to_string()))?;

    if definition.description.trim().is_empty() {
        return Err(ValidationError::EmptyDescription);
    }

    if definition.policy.timeout_ms == 0 {
        return Err(ValidationError::InvalidLimit("timeout_ms"));
    }

    if definition.policy.max_output_bytes == 0 {
        return Err(ValidationError::InvalidLimit("max_output_bytes"));
    }

    let schema = &definition.input_schema;
    let object = schema
        .as_object()
        .ok_or_else(|| ValidationError::InvalidSchema("root must be a JSON object".into()))?;

    if object.get("type").and_then(Value::as_str) != Some("object") {
        return Err(ValidationError::InvalidSchema(
            "root must explicitly declare type=object".into(),
        ));
    }

    if object.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(ValidationError::InvalidSchema(
            "root must explicitly declare additionalProperties=false".into(),
        ));
    }

    if let Some(draft) = object.get("$schema")
        && draft.as_str() != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err(ValidationError::InvalidSchema(
            "only JSON Schema Draft 2020-12 is supported".into(),
        ));
    }

    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .should_ignore_unknown_formats(false)
        .with_retriever(DenyExternalReferences)
        .build(schema)
        .map_err(|error| ValidationError::InvalidSchema(error.to_string()))?;

    Ok(ValidatedDefinition {
        definition,
        validator,
    })
}

fn validate_identity(field: &'static str, value: &str) -> Result<(), ValidationError> {
    if value.is_empty() || value.trim() != value {
        return Err(ValidationError::InvalidIdentity(field));
    }

    Ok(())
}

/// Fail closed if schema compilation requests an external resource.
struct DenyExternalReferences;

impl Retrieve for DenyExternalReferences {
    fn retrieve(&self, _uri: &Uri<String>) -> Result<Value, Box<dyn Error + Send + Sync>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "external schema retrieval is disabled",
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::{ApprovalRequirement, ToolEffect, ToolOrigin, ToolPolicy};
    use serde_json::json;

    fn definition() -> ToolDefinition {
        ToolDefinition {
            id: "fs.read".into(),
            version: "1.0.0".into(),
            description: "Read an allowed file.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "minLength": 1 }
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

    fn call(arguments: Value) -> ToolCall {
        ToolCall {
            call_id: "call-1".into(),
            tool_id: "fs.read".into(),
            arguments: arguments.as_object().expect("object").clone(),
        }
    }

    #[test]
    fn valid_arguments_pass() {
        let validated = validate_definition(definition()).expect("definition");
        assert!(
            validated
                .validate_call(&call(json!({ "path": "fixture.txt" })))
                .is_ok()
        );
    }

    #[test]
    fn missing_extra_and_wrong_type_arguments_fail() {
        let validated = validate_definition(definition()).expect("definition");

        for arguments in [
            json!({}),
            json!({ "path": 42 }),
            json!({ "path": "" }),
            json!({ "path": "fixture.txt", "extra": true }),
        ] {
            assert!(matches!(
                validated.validate_call(&call(arguments)),
                Err(ValidationError::InvalidArguments(_))
            ));
        }
    }

    #[test]
    fn invalid_versions_fail() {
        for version in ["", "1.0", "^1.0.0", "1.0.01"] {
            let mut candidate = definition();
            candidate.version = version.into();

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidVersion(_))
            ));
        }
    }

    #[test]
    fn invalid_schema_fails_registration() {
        let mut candidate = definition();
        candidate.input_schema["properties"]["path"]["type"] = json!(42);

        assert!(matches!(
            validate_definition(candidate),
            Err(ValidationError::InvalidSchema(_))
        ));
    }

    #[test]
    fn external_references_fail_registration() {
        for reference in ["https://example.invalid/schema.json", "file:///etc/passwd"] {
            let mut candidate = definition();
            candidate.input_schema["properties"]["path"] = json!({ "$ref": reference });

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidSchema(_))
            ));
        }
    }

    #[test]
    fn local_schema_references_work() {
        let mut candidate = definition();
        candidate.input_schema["$defs"] = json!({
            "path": { "type": "string", "minLength": 1 }
        });
        candidate.input_schema["properties"]["path"] = json!({ "$ref": "#/$defs/path" });

        let validated = validate_definition(candidate).expect("local reference");

        assert!(
            validated
                .validate_call(&call(json!({ "path": "fixture.txt" })))
                .is_ok()
        );
    }

    #[test]
    fn zero_limits_fail_registration() {
        let mut candidate = definition();
        candidate.policy.timeout_ms = 0;
        assert!(matches!(
            validate_definition(candidate),
            Err(ValidationError::InvalidLimit("timeout_ms"))
        ));

        let mut candidate = definition();
        candidate.policy.max_output_bytes = 0;
        assert!(matches!(
            validate_definition(candidate),
            Err(ValidationError::InvalidLimit("max_output_bytes"))
        ));
    }

    #[test]
    fn mismatched_tool_and_empty_call_identity_fail() {
        let validated = validate_definition(definition()).expect("definition");
        let mut request = call(json!({ "path": "fixture.txt" }));

        request.tool_id = "shell.run".into();
        assert_eq!(
            validated.validate_call(&request),
            Err(ValidationError::ToolMismatch)
        );

        request.tool_id = "fs.read".into();
        request.call_id.clear();
        assert_eq!(
            validated.validate_call(&request),
            Err(ValidationError::InvalidIdentity("call_id"))
        );
    }

    #[test]
    fn definition_rejects_empty_and_whitespace_tool_ids() {
        for id in ["", " ", "\t", " fs.read", "fs.read ", "\nfs.read"] {
            let mut candidate = definition();
            candidate.id = id.into();

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidIdentity("tool id"))
            ));
        }
    }

    #[test]
    fn definition_rejects_blank_descriptions() {
        for description in ["", " ", "\t\n", "\u{2003}"] {
            let mut candidate = definition();
            candidate.description = description.into();

            assert_eq!(
                validate_definition(candidate).err(),
                Some(ValidationError::EmptyDescription)
            );
        }
    }

    #[test]
    fn complete_semantic_versions_are_accepted() {
        for version in [
            "0.0.0",
            "1.2.3",
            "1.2.3-alpha.1",
            "1.2.3+build.42",
            "1.2.3-rc.1+build.42",
        ] {
            let mut candidate = definition();
            candidate.version = version.into();

            let validated = validate_definition(candidate).expect("valid semantic version");

            assert_eq!(validated.definition().version, version);
        }
    }

    #[test]
    fn version_requirements_and_noncanonical_versions_are_rejected() {
        for version in [
            "v1.2.3", "1", "1.2", "01.2.3", "1.02.3", "1.2.03", "1.2.3-01", ">=1.2.3", "~1.2.3",
            "*", " 1.2.3", "1.2.3 ",
        ] {
            let mut candidate = definition();
            candidate.version = version.into();

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidVersion(_))
            ));
        }
    }

    #[test]
    fn schema_root_must_be_an_object_schema() {
        for schema in [
            Value::Null,
            json!(true),
            json!(false),
            json!([]),
            json!("schema"),
            json!(42),
        ] {
            let mut candidate = definition();
            candidate.input_schema = schema;

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidSchema(_))
            ));
        }
    }

    #[test]
    fn root_type_must_explicitly_be_object() {
        for root_type in [
            json!("array"),
            json!("string"),
            json!("integer"),
            json!(["object", "null"]),
            Value::Null,
        ] {
            let mut candidate = definition();
            candidate.input_schema["type"] = root_type;

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidSchema(_))
            ));
        }

        let mut candidate = definition();
        candidate
            .input_schema
            .as_object_mut()
            .expect("schema object")
            .remove("type");

        assert!(matches!(
            validate_definition(candidate),
            Err(ValidationError::InvalidSchema(_))
        ));
    }

    #[test]
    fn root_additional_properties_must_explicitly_be_false() {
        for additional_properties in [
            json!(true),
            json!({ "type": "string" }),
            json!("false"),
            Value::Null,
        ] {
            let mut candidate = definition();
            candidate.input_schema["additionalProperties"] = additional_properties;

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidSchema(_))
            ));
        }

        let mut candidate = definition();
        candidate
            .input_schema
            .as_object_mut()
            .expect("schema object")
            .remove("additionalProperties");

        assert!(matches!(
            validate_definition(candidate),
            Err(ValidationError::InvalidSchema(_))
        ));
    }

    #[test]
    fn explicit_draft_202012_is_accepted() {
        let mut candidate = definition();
        candidate.input_schema["$schema"] = json!("https://json-schema.org/draft/2020-12/schema");

        assert!(validate_definition(candidate).is_ok());
    }

    #[test]
    fn unsupported_or_malformed_draft_declarations_are_rejected() {
        for draft in [
            json!("http://json-schema.org/draft-07/schema#"),
            json!("https://json-schema.org/draft/2019-09/schema"),
            json!("not-a-schema-uri"),
            json!(42),
            Value::Null,
        ] {
            let mut candidate = definition();
            candidate.input_schema["$schema"] = draft;

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidSchema(_))
            ));
        }
    }

    #[test]
    fn invalid_schema_keyword_values_are_rejected() {
        for property_schema in [
            json!({ "type": "not-a-type" }),
            json!({ "type": "string", "minLength": -1 }),
            json!({ "type": "string", "pattern": "[" }),
        ] {
            let mut candidate = definition();
            candidate.input_schema["properties"]["path"] = property_schema;

            assert!(matches!(
                validate_definition(candidate),
                Err(ValidationError::InvalidSchema(_))
            ));
        }
    }

    #[test]
    fn unknown_formats_are_rejected_during_registration() {
        let mut candidate = definition();
        candidate.input_schema["properties"]["path"] = json!({
            "type": "string",
            "format": "alloy-unregistered-format"
        });

        assert!(matches!(
            validate_definition(candidate),
            Err(ValidationError::InvalidSchema(_))
        ));
    }

    #[test]
    fn known_formats_are_enforced_on_arguments() {
        let mut candidate = definition();
        candidate.input_schema["properties"]["path"] = json!({
            "type": "string",
            "format": "ipv4"
        });

        let validated = validate_definition(candidate).expect("IPv4 schema");

        assert!(
            validated
                .validate_call(&call(json!({ "path": "192.0.2.1" })))
                .is_ok()
        );

        assert!(matches!(
            validated.validate_call(&call(json!({ "path": "999.0.0.1" }))),
            Err(ValidationError::InvalidArguments(_))
        ));
    }

    #[test]
    fn integer_bounds_are_inclusive_and_enforced() {
        let mut candidate = definition();
        candidate.input_schema = json!({
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 65536
                }
            },
            "required": ["length"],
            "additionalProperties": false
        });

        let validated = validate_definition(candidate).expect("bounded schema");

        for length in [1, 65536] {
            assert!(
                validated
                    .validate_call(&call(json!({ "length": length })))
                    .is_ok()
            );
        }

        for arguments in [
            json!({ "length": 0 }),
            json!({ "length": -1 }),
            json!({ "length": 65537 }),
            json!({ "length": 1.5 }),
            json!({ "length": "4096" }),
        ] {
            assert!(matches!(
                validated.validate_call(&call(arguments)),
                Err(ValidationError::InvalidArguments(_))
            ));
        }
    }

    #[test]
    fn nested_object_constraints_are_enforced_when_declared() {
        let mut candidate = definition();
        candidate.input_schema = json!({
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "minLength": 1
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            },
            "required": ["request"],
            "additionalProperties": false
        });

        let validated = validate_definition(candidate).expect("nested schema");

        assert!(
            validated
                .validate_call(&call(json!({
                    "request": { "path": "fixture.txt" }
                })))
                .is_ok()
        );

        for arguments in [
            json!({ "request": {} }),
            json!({ "request": { "path": 42 } }),
            json!({
                "request": {
                    "path": "fixture.txt",
                    "approved": true
                }
            }),
        ] {
            assert!(matches!(
                validated.validate_call(&call(arguments)),
                Err(ValidationError::InvalidArguments(_))
            ));
        }
    }

    #[test]
    fn call_identity_validation_rejects_whitespace() {
        let validated = validate_definition(definition()).expect("definition");

        for id in ["", " ", " call-1", "call-1 ", "\ncall-1"] {
            let mut request = call(json!({ "path": "fixture.txt" }));
            request.call_id = id.into();

            assert_eq!(
                validated.validate_call(&request),
                Err(ValidationError::InvalidIdentity("call_id"))
            );
        }

        for id in ["", " ", " fs.read", "fs.read "] {
            let mut request = call(json!({ "path": "fixture.txt" }));
            request.tool_id = id.into();

            assert_eq!(
                validated.validate_call(&request),
                Err(ValidationError::InvalidIdentity("tool_id"))
            );
        }
    }

    #[test]
    fn validation_does_not_mutate_call_arguments() {
        let validated = validate_definition(definition()).expect("definition");
        let request = call(json!({
            "path": "résumé/雪/$(touch NEVER_RUN).txt"
        }));
        let original = request.clone();

        validated.validate_call(&request).expect("valid arguments");

        assert_eq!(request, original);
    }

    #[test]
    fn compiled_definition_is_independent_of_the_original_clone() {
        let original = definition();
        let validated = validate_definition(original.clone()).expect("definition");

        let mut changed = original;
        changed.id = "shell.run".into();
        changed.input_schema["properties"]["path"]["type"] = json!("integer");

        assert_eq!(validated.definition().id, "fs.read");

        assert!(
            validated
                .validate_call(&call(json!({ "path": "fixture.txt" })))
                .is_ok()
        );

        assert!(matches!(
            validated.validate_call(&call(json!({ "path": 42 }))),
            Err(ValidationError::InvalidArguments(_))
        ));
    }

    #[test]
    fn external_retriever_returns_permission_denied() {
        let uri = "https://example.invalid/schema.json"
            .parse::<Uri<String>>()
            .expect("URI");

        let error = DenyExternalReferences
            .retrieve(&uri)
            .expect_err("retrieval must fail");

        let io_error = error.downcast_ref::<std::io::Error>().expect("I/O error");

        assert_eq!(io_error.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn validation_errors_have_readable_diagnostics() {
        let errors = [
            ValidationError::InvalidIdentity("call_id"),
            ValidationError::InvalidVersion("incomplete version".into()),
            ValidationError::EmptyDescription,
            ValidationError::InvalidLimit("timeout_ms"),
            ValidationError::InvalidSchema("invalid keyword".into()),
            ValidationError::ToolMismatch,
            ValidationError::InvalidArguments("missing path".into()),
        ];

        for error in errors {
            let diagnostic = error.to_string();

            assert!(!diagnostic.trim().is_empty());
            assert!(
                !diagnostic.contains('\n'),
                "unexpected multiline diagnostic: {diagnostic}"
            );
        }
    }
}
