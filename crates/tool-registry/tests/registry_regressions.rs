//! Regression coverage for registry ownership and public contracts.

use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tool_registry::{
    call::{ToolCall, ToolInvocation},
    definition::{ToolDefinition, ToolOrigin},
    registry::{RegistryError, ToolImplementation, ToolRegistry},
    result::{InlineToolResult, ToolResult},
    validation::ValidationError,
};

fn definition() -> ToolDefinition {
    serde_json::from_value(json!({
        "id": "fs.read",
        "version": "1.0.0",
        "description": "Read an allowed file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1
                }
            },
            "required": ["path"],
            "additionalProperties": false
        },
        "origin": "builtin",
        "policy": {
            "effect": "read",
            "approval": "never",
            "timeout_ms": 5000,
            "max_output_bytes": 65536
        }
    }))
    .expect("definition fixture")
}

fn invocation() -> ToolInvocation {
    serde_json::from_value(json!({
        "context": {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "generation_id": "generation-1"
        },
        "call": {
            "call_id": "call-1",
            "tool_id": "fs.read",
            "arguments": {
                "path": "fixtures/notes.txt"
            }
        }
    }))
    .expect("invocation fixture")
}

struct Probe {
    executions: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl ToolImplementation for Probe {
    fn execute(&self, invocation: &ToolInvocation) -> ToolResult {
        self.executions.fetch_add(1, Ordering::SeqCst);

        ToolResult::Inline(InlineToolResult {
            call_id: invocation.call.call_id.clone(),
            content: json!({ "text": "probe output" }),
            is_error: false,
        })
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn probe(executions: &Arc<AtomicUsize>, drops: &Arc<AtomicUsize>) -> Arc<dyn ToolImplementation> {
    Arc::new(Probe {
        executions: Arc::clone(executions),
        drops: Arc::clone(drops),
    })
}

#[test]
fn registry_rejected_duplicate_drops_only_the_new_implementation() {
    let executions = Arc::new(AtomicUsize::new(0));
    let original_drops = Arc::new(AtomicUsize::new(0));
    let rejected_drops = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();

    registry
        .register(definition(), probe(&executions, &original_drops))
        .expect("first registration");

    let mut duplicate = definition();
    duplicate.version = "2.0.0".into();

    assert!(matches!(
        registry.register(duplicate, probe(&executions, &rejected_drops)),
        Err(RegistryError::DuplicateId(_))
    ));

    assert_eq!(registry.len(), 1);
    assert_eq!(
        registry.definition("fs.read").expect("original").version,
        "1.0.0"
    );
    assert_eq!(original_drops.load(Ordering::SeqCst), 0);
    assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
    assert_eq!(executions.load(Ordering::SeqCst), 0);

    drop(registry);

    assert_eq!(original_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn registry_invalid_registration_does_not_reserve_the_identifier() {
    let executions = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();

    let mut invalid = definition();
    invalid.input_schema["properties"]["path"]["type"] = json!(42);

    assert!(matches!(
        registry.register(invalid, probe(&executions, &drops)),
        Err(RegistryError::Validation(ValidationError::InvalidSchema(_)))
    ));

    assert!(registry.is_empty());
    assert!(registry.definition("fs.read").is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    registry
        .register(definition(), probe(&executions, &drops))
        .expect("valid retry with same identifier");

    assert_eq!(registry.len(), 1);
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[test]
fn registry_preserves_each_session_association_without_executing() {
    let executions = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();

    registry
        .register(definition(), probe(&executions, &drops))
        .expect("registration");

    for (session, turn, generation, call_id) in [
        ("session-a", "turn-a", "generation-a", "call-a"),
        ("session-b", "turn-b", "generation-b", "call-b"),
    ] {
        let mut request = invocation();
        request.context.session_id = session.into();
        request.context.turn_id = turn.into();
        request.context.generation_id = generation.into();
        request.call.call_id = call_id.into();

        let resolved = registry.resolve(request).expect("resolution");
        let actual = resolved.invocation();

        assert_eq!(actual.context.session_id, session);
        assert_eq!(actual.context.turn_id, turn);
        assert_eq!(actual.context.generation_id, generation);
        assert_eq!(actual.call.call_id, call_id);
    }

    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[test]
fn registry_rejects_each_missing_context_identity() {
    let executions = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();

    registry
        .register(definition(), probe(&executions, &drops))
        .expect("registration");

    for field in ["session_id", "turn_id", "generation_id"] {
        let mut wire = serde_json::to_value(invocation()).expect("JSON");
        wire["context"][field] = json!("");

        let request: ToolInvocation = serde_json::from_value(wire).expect("typed request");

        assert!(matches!(
            registry.resolve(request),
            Err(RegistryError::Validation(
                ValidationError::InvalidIdentity(actual)
            )) if actual == field
        ));
    }

    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[test]
fn definition_rejects_unknown_origin_and_missing_policy_limits() {
    let original = serde_json::to_value(definition()).expect("JSON");

    let mut unknown_origin = original.clone();
    unknown_origin["origin"] = json!("filesystem");
    assert!(serde_json::from_value::<ToolDefinition>(unknown_origin).is_err());

    for field in ["timeout_ms", "max_output_bytes"] {
        let mut candidate = original.clone();
        candidate["policy"]
            .as_object_mut()
            .expect("policy")
            .remove(field);

        assert!(serde_json::from_value::<ToolDefinition>(candidate).is_err());
    }
}

#[test]
fn definition_origin_category_does_not_create_a_registration() {
    let registry = ToolRegistry::new();

    for origin in [ToolOrigin::Builtin, ToolOrigin::Plugin, ToolOrigin::Mcp] {
        let mut candidate = definition();
        candidate.origin = origin;

        let wire = serde_json::to_vec(&candidate).expect("serialize");
        let _: ToolDefinition = serde_json::from_slice(&wire).expect("deserialize");

        assert!(registry.is_empty());
        assert!(registry.definition("fs.read").is_none());
    }
}

#[test]
fn call_requires_explicit_request_identity() {
    let original = serde_json::to_value(invocation().call).expect("JSON");

    for field in ["call_id", "tool_id"] {
        let mut candidate = original.clone();
        candidate.as_object_mut().expect("call").remove(field);

        assert!(serde_json::from_value::<ToolCall>(candidate).is_err());
    }
}

#[test]
fn call_deserialization_preserves_nested_json_values() {
    let arguments = json!({
        "text": "line one\n雪",
        "enabled": false,
        "optional": null,
        "count": 42,
        "nested": {
            "items": ["a", "b"]
        }
    });

    let wire = json!({
        "call_id": "call-1",
        "tool_id": "fixture.tool",
        "arguments": arguments
    });

    let decoded: ToolCall = serde_json::from_value(wire).expect("call");

    assert_eq!(Value::Object(decoded.arguments), arguments);
}

#[test]
fn result_rejects_extra_fields_in_both_payload_variants() {
    for mut candidate in [
        json!({
            "call_id": "call-1",
            "content": { "text": "result" },
            "is_error": false
        }),
        json!({
            "call_id": "call-1",
            "artifact_ref": "artifact-1",
            "is_error": false
        }),
    ] {
        candidate["approved"] = json!(true);
        assert!(serde_json::from_value::<ToolResult>(candidate).is_err());
    }
}

#[test]
fn result_requires_boolean_error_status() {
    for is_error in [json!("false"), json!(0), Value::Null] {
        let wire = json!({
            "call_id": "call-1",
            "content": "result",
            "is_error": is_error
        });

        assert!(serde_json::from_value::<ToolResult>(wire).is_err());
    }
}

#[test]
fn result_error_status_is_preserved_for_both_payload_variants() {
    for wire in [
        json!({
            "call_id": "failed-inline",
            "content": {
                "kind": "execution",
                "message": "Process failed",
                "details": { "exit_code": 1 }
            },
            "is_error": true
        }),
        json!({
            "call_id": "failed-artifact",
            "artifact_ref": "artifact-stderr-1",
            "is_error": true
        }),
    ] {
        let result: ToolResult = serde_json::from_value(wire.clone()).expect("result");

        assert!(result.is_error());
        assert_eq!(serde_json::to_value(result).expect("serialize"), wire);
    }
}
