//! Parsed tool calls and their daemon-owned context.
//!
//! These types represent requests, not permission to execute. The registry
//! must resolve the tool identifier and validate arguments; the harness
//! must then apply policy and any required bound approval.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A complete structured request for a registered tool.
///
/// The daemon assigns `call_id`; it must not trust a model-generated ID as
/// a globally unique execution identity.
///
/// Arguments are represented as an object so arrays, strings, numbers,
/// booleans, and null cannot deserialize as an argument envelope. Individual
/// fields still require validation against the registered input schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    /// Daemon-scoped identity linking intent, approval, and result events.
    pub call_id: String,

    /// Identifier resolved against an existing registry entry.
    pub tool_id: String,

    /// Untrusted arguments awaiting registered-schema validation.
    pub arguments: Map<String, Value>,
}

/// Daemon-supplied association for a tool invocation.
///
/// Construct this from the active session and generation, not from
/// model-generated content. These identifiers bind lifecycle records to
/// the user turn that owns the request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCallContext {
    /// Session that owns the invocation.
    pub session_id: String,

    /// User turn within that session.
    pub turn_id: String,

    /// Model generation that produced the structured call.
    pub generation_id: String,
}

/// A parsed request associated with its daemon-owned context.
///
/// This is not an authorized or validated execution handle. Registry
/// validation and harness authorization must happen before execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolInvocation {
    /// Association supplied by the daemon.
    pub context: ToolCallContext,

    /// Complete parsed request with a daemon-assigned call identity.
    pub call: ToolCall,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call_json() -> Value {
        json!({
            "call_id": "call-1",
            "tool_id": "fs.read",
            "arguments": {
                "path": "fixtures/input.txt"
            }
        })
    }

    #[test]
    fn call_roundtrips_without_changing_arguments() {
        let expected = call_json();
        let call: ToolCall = serde_json::from_value(expected.clone()).expect("object arguments");

        assert_eq!(call.call_id, "call-1");
        assert_eq!(call.tool_id, "fs.read");
        assert_eq!(
            call.arguments.get("path"),
            Some(&json!("fixtures/input.txt"))
        );
        assert_eq!(serde_json::to_value(call).expect("serialize"), expected);
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        for arguments in [
            Value::Null,
            json!([]),
            json!("file.txt"),
            json!(42),
            json!(true),
        ] {
            let mut value = call_json();
            value["arguments"] = arguments;

            assert!(serde_json::from_value::<ToolCall>(value).is_err());
        }
    }

    #[test]
    fn missing_arguments_are_rejected() {
        let mut value = call_json();
        value.as_object_mut().expect("object").remove("arguments");

        assert!(serde_json::from_value::<ToolCall>(value).is_err());
    }

    #[test]
    fn call_envelope_rejects_claimed_authority() {
        for (field, claimed_value) in [
            ("executable", json!("/bin/sh")),
            ("origin", json!("builtin")),
            ("approved", json!(true)),
            ("policy", json!({ "approval": "never" })),
            ("version", json!("1.0.0")),
        ] {
            let mut value = call_json();
            value[field] = claimed_value;

            assert!(
                serde_json::from_value::<ToolCall>(value).is_err(),
                "unexpected authority field accepted: {field}"
            );
        }
    }

    #[test]
    fn invocation_preserves_context_and_call_identity() {
        let expected = json!({
            "context": {
                "session_id": "session-1",
                "turn_id": "turn-1",
                "generation_id": "generation-1"
            },
            "call": call_json()
        });

        let invocation: ToolInvocation =
            serde_json::from_value(expected.clone()).expect("invocation");

        assert_eq!(invocation.context.session_id, "session-1");
        assert_eq!(invocation.context.turn_id, "turn-1");
        assert_eq!(invocation.context.generation_id, "generation-1");
        assert_eq!(invocation.call.call_id, "call-1");

        assert_eq!(
            serde_json::to_value(invocation).expect("serialize"),
            expected
        );
    }
}

#[cfg(test)]
mod call_contract_tests {
    use super::*;
    use serde_json::{Value, json};

    fn call_fixture() -> Value {
        json!({
            "call_id": "call-1",
            "tool_id": "shell.run",
            "arguments": {
                "executable": "/usr/bin/printf",
                "args": ["%s", "résumé 雪\n$(touch NEVER_RUN)"],
                "cwd": "/tmp"
            }
        })
    }

    fn invocation_fixture() -> Value {
        json!({
            "context": {
                "session_id": "session-1",
                "turn_id": "turn-1",
                "generation_id": "generation-1"
            },
            "call": call_fixture()
        })
    }

    #[test]
    fn each_call_field_is_required() {
        for field in ["call_id", "tool_id", "arguments"] {
            let mut value = call_fixture();
            value.as_object_mut().expect("call").remove(field);

            assert!(
                serde_json::from_value::<ToolCall>(value).is_err(),
                "missing call field accepted: {field}"
            );
        }
    }

    #[test]
    fn identities_must_deserialize_as_strings() {
        for field in ["call_id", "tool_id"] {
            for invalid in [Value::Null, json!(42), json!(true), json!([]), json!({})] {
                let mut value = call_fixture();
                value[field] = invalid;

                assert!(
                    serde_json::from_value::<ToolCall>(value).is_err(),
                    "non-string identity accepted: {field}"
                );
            }
        }
    }

    #[test]
    fn empty_argument_object_is_preserved_for_schema_validation() {
        let mut value = call_fixture();
        value["arguments"] = json!({});

        let call: ToolCall = serde_json::from_value(value).expect("object envelope");

        assert!(call.arguments.is_empty());

        // Whether arguments are required is decided by the registered schema.
    }

    #[test]
    fn arbitrary_nested_argument_values_are_preserved_exactly() {
        let arguments = json!({
            "text": "résumé 雪\n$(touch NEVER_RUN)",
            "boolean": false,
            "null": null,
            "integer": 42,
            "fraction": 1.5,
            "array": ["a", 7, null],
            "object": {
                "nested": {
                    "value": "retained"
                }
            }
        });

        let mut value = call_fixture();
        value["arguments"] = arguments.clone();

        let call: ToolCall = serde_json::from_value(value).expect("call");
        let bytes = serde_json::to_vec(&call).expect("serialize");
        let restored: ToolCall = serde_json::from_slice(&bytes).expect("deserialize");

        assert_eq!(restored, call);
        assert_eq!(Value::Object(restored.arguments), arguments);
    }

    #[test]
    fn authority_shaped_argument_is_data_not_an_envelope_field() {
        let mut value = call_fixture();
        value["arguments"] = json!({
            "approved": true
        });

        let call: ToolCall = serde_json::from_value(value).expect("object arguments");

        assert_eq!(call.arguments.get("approved"), Some(&json!(true)));

        // The registered schema and harness policy decide what is allowed.
        // Merely deserializing this value grants no authorization.
    }

    #[test]
    fn every_context_field_is_required() {
        for field in ["session_id", "turn_id", "generation_id"] {
            let mut value = invocation_fixture();
            value["context"]
                .as_object_mut()
                .expect("context")
                .remove(field);

            assert!(
                serde_json::from_value::<ToolInvocation>(value).is_err(),
                "missing context field accepted: {field}"
            );
        }
    }

    #[test]
    fn unknown_context_and_invocation_fields_are_rejected() {
        let mut unknown_context = invocation_fixture();
        unknown_context["context"]["approved"] = json!(true);

        assert!(serde_json::from_value::<ToolInvocation>(unknown_context).is_err());

        let mut unknown_invocation = invocation_fixture();
        unknown_invocation["executable"] = json!("/bin/sh");

        assert!(serde_json::from_value::<ToolInvocation>(unknown_invocation).is_err());
    }

    #[test]
    fn invocation_requires_both_context_and_call() {
        for field in ["context", "call"] {
            let mut value = invocation_fixture();
            value.as_object_mut().expect("invocation").remove(field);

            assert!(
                serde_json::from_value::<ToolInvocation>(value).is_err(),
                "missing invocation field accepted: {field}"
            );
        }
    }

    #[test]
    fn cloning_a_call_does_not_share_mutable_arguments() {
        let original: ToolCall = serde_json::from_value(call_fixture()).expect("call");
        let mut changed = original.clone();

        changed.call_id = "call-2".into();
        changed.arguments.insert("cwd".into(), json!("/elsewhere"));

        assert_eq!(original.call_id, "call-1");
        assert_eq!(original.arguments.get("cwd"), Some(&json!("/tmp")));
        assert_eq!(changed.call_id, "call-2");
    }

    #[test]
    fn invocation_roundtrip_preserves_all_association_ids() {
        let original = invocation_fixture();
        let invocation: ToolInvocation =
            serde_json::from_value(original.clone()).expect("invocation");

        assert_eq!(invocation.context.session_id, "session-1");
        assert_eq!(invocation.context.turn_id, "turn-1");
        assert_eq!(invocation.context.generation_id, "generation-1");
        assert_eq!(invocation.call.call_id, "call-1");
        assert_eq!(invocation.call.tool_id, "shell.run");

        assert_eq!(
            serde_json::to_value(invocation).expect("serialize"),
            original
        );
    }
}
