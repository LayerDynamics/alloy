//! W4 builtin definitions and scoped execution primitives.
//! Limits are explicit daemon inputs; no new default time/byte limits are frozen.
//! Schema validity is not filesystem/network resource authorization.
/// Scoped file-read primitives; no live dispatch is enabled.
#[cfg(unix)]
pub mod fs;

/// Owned shell process primitives; no live dispatch enabled.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod shell;

use crate::{
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
    error::ValidationError,
    validation::validate_definition,
};
use serde_json::json;

/// Produce and schema-validate the five builtin definitions without registering
/// placeholder implementations or enabling effects. Versions/schemas below are
/// a W4 candidate requiring review before binding real effects in later steps.
pub fn definitions(
    timeout_ms: u64,
    max_output_bytes: u64,
) -> Result<Vec<ToolDefinition>, ValidationError> {
    use ApprovalRequirement::{Always, Never, Policy};
    use ToolEffect::{Destructive, Read, Write};
    let path = json!({"type":"string","minLength":1});
    let schemas = [
        (
            "fs.read",
            Read,
            Never,
            "Read an approved bounded file range.",
            json!({"path":path.clone(),"offset":{"type":"integer","minimum":0},"length":{"type":"integer","minimum":1}}),
            vec!["path", "offset", "length"],
        ),
        (
            "fs.list",
            Read,
            Never,
            "List an approved directory deterministically.",
            json!({"path":path.clone(),"max_entries":{"type":"integer","minimum":1}}),
            vec!["path", "max_entries"],
        ),
        (
            "fs.write",
            Write,
            Policy,
            "Write exact UTF-8 contents to an approved destination.",
            json!({"path":path.clone(),"content":{"type":"string"}}),
            vec!["path", "content"],
        ),
        (
            "shell.run",
            Destructive,
            Always,
            "Run an approved executable with exact argv and cwd.",
            json!({"executable":path.clone(),"args":{"type":"array","items":{"type":"string"}},"cwd":path}),
            vec!["executable", "args", "cwd"],
        ),
        (
            "http.fetch",
            Read,
            Policy,
            "Fetch a bounded approved GET or HEAD response.",
            json!({"url":{"type":"string","minLength":1},"method":{"type":"string","enum":["GET","HEAD"]}}),
            vec!["url", "method"],
        ),
    ];
    let mut result = Vec::new();
    for (id, effect, approval, description, properties, required) in schemas {
        let definition = ToolDefinition {
            id: id.into(),
            version: "1.0.0".into(),
            description: description.into(),
            input_schema: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
            origin: ToolOrigin::Builtin,
            policy: ToolPolicy {
                effect,
                approval,
                timeout_ms,
                max_output_bytes,
            },
        };
        validate_definition(definition.clone())?;
        result.push(definition);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::ToolCall;
    use serde_json::Value;
    fn call(id: &str, value: Value) -> ToolCall {
        ToolCall {
            call_id: "call-fixture".into(),
            tool_id: id.into(),
            arguments: value.as_object().unwrap().clone(),
        }
    }
    #[test]
    fn five_definitions_preserve_frozen_defaults() {
        let definitions = definitions(1000, 65536).unwrap();
        let actual: Vec<_> = definitions
            .iter()
            .map(|d| (d.id.as_str(), d.policy.effect, d.policy.approval))
            .collect();
        assert_eq!(
            actual,
            vec![
                ("fs.read", ToolEffect::Read, ApprovalRequirement::Never),
                ("fs.list", ToolEffect::Read, ApprovalRequirement::Never),
                ("fs.write", ToolEffect::Write, ApprovalRequirement::Policy),
                (
                    "shell.run",
                    ToolEffect::Destructive,
                    ApprovalRequirement::Always
                ),
                ("http.fetch", ToolEffect::Read, ApprovalRequirement::Policy)
            ]
        );
        assert!(
            definitions
                .iter()
                .all(|d| d.policy.timeout_ms == 1000 && d.policy.max_output_bytes == 65536)
        );
    }
    #[test]
    fn zero_limits_fail_without_returning_definitions() {
        assert!(definitions(0, 1).is_err());
        assert!(definitions(1, 0).is_err());
    }
    #[test]
    fn valid_arguments_validate_and_authority_fields_do_not() {
        let valid = [
            json!({"path":"a","offset":0,"length":1}),
            json!({"path":"a","max_entries":1}),
            json!({"path":"a","content":""}),
            json!({"executable":"/bin/fixture","args":["$(literal)"],"cwd":"/fixture"}),
            json!({"url":"http://fixture.invalid/","method":"GET"}),
        ];
        for (definition, arguments) in definitions(1000, 65536).unwrap().into_iter().zip(valid) {
            let id = definition.id.clone();
            let validator = validate_definition(definition).unwrap();
            assert!(
                validator
                    .validate_call(&call(&id, arguments.clone()))
                    .is_ok()
            );
            let mut changed = arguments;
            changed["approved"] = json!(true);
            assert!(validator.validate_call(&call(&id, changed)).is_err());
            assert!(validator.validate_call(&call(&id, json!({}))).is_err());
        }
    }
    #[test]
    fn unsupported_http_method_and_shell_string_reject() {
        let defs = definitions(1000, 65536).unwrap();
        let http = validate_definition(defs[4].clone()).unwrap();
        assert!(
            http.validate_call(&call(
                "http.fetch",
                json!({"url":"https://fixture.invalid/","method":"POST"})
            ))
            .is_err()
        );
        let shell = validate_definition(defs[3].clone()).unwrap();
        assert!(
            shell
                .validate_call(&call(
                    "shell.run",
                    json!({"executable":"/bin/fixture","args":"echo raw","cwd":"/fixture"})
                ))
                .is_err()
        );
    }
}
