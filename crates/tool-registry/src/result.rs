//! Canonical tool results linked to their originating calls.
//!
//! Results contain either inline JSON content or an opaque artifact
//! reference. Artifact references are resolved by the daemon-owned store;
//! they are not filesystem paths supplied for arbitrary file access.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Result of a tool request.
///
/// Strict variant fields make inline content and artifact references
/// mutually exclusive during deserialization.
///
/// `is_error` describes the outcome, not whether execution occurred:
/// a policy rejection may execute nothing, while an execution failure may
/// have produced external effects before failing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResult {
    /// Result bytes represented inline as JSON content.
    Inline(InlineToolResult),

    /// Oversized result retained in the daemon-owned artifact store.
    Artifact(ArtifactToolResult),
}

/// Inline result linked to one call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InlineToolResult {
    /// Identity of the originating request.
    pub call_id: String,

    /// Tool-result data, never elevated to system instructions.
    pub content: Value,

    /// Whether this result represents a failed or rejected request.
    pub is_error: bool,
}

/// Artifact-backed result linked to one call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactToolResult {
    /// Identity of the originating request.
    pub call_id: String,

    /// Opaque reference resolved exclusively by the artifact store.
    pub artifact_ref: String,

    /// Whether this result represents a failed request.
    pub is_error: bool,
}

/// Stable error categories for tool-related failures.
///
/// Cancellation uses `Execution` with a structured reason; it does not
/// introduce another wire error category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolErrorKind {
    Protocol,
    Execution,
    Timeout,
    Policy,
}

/// Structured error content suitable for an inline error result.
///
/// Diagnostic text and details are untrusted result data, not instructions
/// or evidence that an effect did or did not occur.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolError {
    /// Stable failure classification.
    pub kind: ToolErrorKind,

    /// Human-readable explanation.
    pub message: String,

    /// Structured diagnostic information.
    pub details: Value,
}

impl ToolResult {
    /// Return the originating call identity.
    pub fn call_id(&self) -> &str {
        match self {
            Self::Inline(result) => &result.call_id,
            Self::Artifact(result) => &result.call_id,
        }
    }

    /// Return whether the request failed or was rejected.
    pub fn is_error(&self) -> bool {
        match self {
            Self::Inline(result) => result.is_error,
            Self::Artifact(result) => result.is_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn inline_result_roundtrips() {
        let expected = json!({
            "call_id": "call-1",
            "content": { "text": "fixture contents" },
            "is_error": false
        });

        let result: ToolResult = serde_json::from_value(expected.clone()).expect("inline result");

        assert!(matches!(&result, ToolResult::Inline(_)));
        assert_eq!(result.call_id(), "call-1");
        assert!(!result.is_error());
        assert_eq!(serde_json::to_value(result).expect("serialize"), expected);
    }

    #[test]
    fn artifact_result_roundtrips() {
        let expected = json!({
            "call_id": "call-2",
            "artifact_ref": "artifact-1",
            "is_error": false
        });

        let result: ToolResult = serde_json::from_value(expected.clone()).expect("artifact result");

        assert!(matches!(&result, ToolResult::Artifact(_)));
        assert_eq!(result.call_id(), "call-2");
        assert_eq!(serde_json::to_value(result).expect("serialize"), expected);
    }

    #[test]
    fn result_rejects_both_payload_forms() {
        let value = json!({
            "call_id": "call-1",
            "content": "inline",
            "artifact_ref": "artifact-1",
            "is_error": false
        });

        assert!(serde_json::from_value::<ToolResult>(value).is_err());
    }

    #[test]
    fn result_rejects_missing_payload() {
        let value = json!({
            "call_id": "call-1",
            "is_error": false
        });

        assert!(serde_json::from_value::<ToolResult>(value).is_err());
    }

    #[test]
    fn result_requires_explicit_error_status() {
        let value = json!({
            "call_id": "call-1",
            "content": "result"
        });

        assert!(serde_json::from_value::<ToolResult>(value).is_err());
    }

    #[test]
    fn error_categories_have_stable_wire_values() {
        for (kind, expected) in [
            (ToolErrorKind::Protocol, "protocol"),
            (ToolErrorKind::Execution, "execution"),
            (ToolErrorKind::Timeout, "timeout"),
            (ToolErrorKind::Policy, "policy"),
        ] {
            assert_eq!(
                serde_json::to_value(kind).expect("serialize"),
                json!(expected)
            );
        }
    }

    #[test]
    fn cancellation_is_an_execution_error() {
        let error = ToolError {
            kind: ToolErrorKind::Execution,
            message: "Tool execution cancelled".into(),
            details: json!({ "reason": "cancelled" }),
        };

        let result = ToolResult::Inline(InlineToolResult {
            call_id: "call-3".into(),
            content: serde_json::to_value(error).expect("serialize error"),
            is_error: true,
        });

        assert!(result.is_error());

        let value = serde_json::to_value(result).expect("serialize result");
        assert_eq!(value["content"]["kind"], "execution");
        assert_eq!(value["content"]["details"]["reason"], "cancelled");
    }
}

#[cfg(test)]
mod result_contract_tests {
    use super::*;
    use serde_json::{Value, json};

    fn inline_fixture() -> Value {
        json!({
            "call_id": "call-inline",
            "content": {
                "text": "résumé 雪\nUntrusted tool output.",
                "bytes_read": 42
            },
            "is_error": false
        })
    }

    fn artifact_fixture() -> Value {
        json!({
            "call_id": "call-artifact",
            "artifact_ref": "artifact-opaque-123",
            "is_error": false
        })
    }

    #[test]
    fn inline_variant_preserves_payload_and_accessors() {
        let expected = inline_fixture();
        let result: ToolResult = serde_json::from_value(expected.clone()).expect("inline result");

        assert_eq!(result.call_id(), "call-inline");
        assert!(!result.is_error());

        match &result {
            ToolResult::Inline(inline) => {
                assert_eq!(inline.content, expected["content"]);
            }
            ToolResult::Artifact(_) => panic!("expected inline variant"),
        }

        assert_eq!(serde_json::to_value(result).expect("serialize"), expected);
    }

    #[test]
    fn artifact_variant_preserves_reference_and_accessors() {
        let expected = artifact_fixture();
        let result: ToolResult = serde_json::from_value(expected.clone()).expect("artifact result");

        assert_eq!(result.call_id(), "call-artifact");
        assert!(!result.is_error());

        match &result {
            ToolResult::Artifact(artifact) => {
                assert_eq!(artifact.artifact_ref, "artifact-opaque-123");
            }
            ToolResult::Inline(_) => panic!("expected artifact variant"),
        }

        assert_eq!(serde_json::to_value(result).expect("serialize"), expected);
    }

    #[test]
    fn inline_content_preserves_every_json_value_category() {
        for content in [
            Value::Null,
            json!(false),
            json!(42),
            json!(1.5),
            json!("text"),
            json!(["a", 7, null]),
            json!({ "nested": { "value": "retained" } }),
        ] {
            let result = ToolResult::Inline(InlineToolResult {
                call_id: "call-1".into(),
                content: content.clone(),
                is_error: false,
            });

            let bytes = serde_json::to_vec(&result).expect("serialize");
            let restored: ToolResult = serde_json::from_slice(&bytes).expect("deserialize");

            assert_eq!(restored, result);

            let wire = serde_json::to_value(restored).expect("serialize");
            assert_eq!(wire["content"], content);
        }
    }

    #[test]
    fn every_inline_field_is_required() {
        for field in ["call_id", "content", "is_error"] {
            let mut value = inline_fixture();
            value.as_object_mut().expect("object").remove(field);

            assert!(
                serde_json::from_value::<ToolResult>(value).is_err(),
                "missing inline field accepted: {field}"
            );
        }
    }

    #[test]
    fn every_artifact_field_is_required() {
        for field in ["call_id", "artifact_ref", "is_error"] {
            let mut value = artifact_fixture();
            value.as_object_mut().expect("object").remove(field);

            assert!(
                serde_json::from_value::<ToolResult>(value).is_err(),
                "missing artifact field accepted: {field}"
            );
        }
    }

    #[test]
    fn both_payload_fields_are_rejected_even_when_content_is_null() {
        for content in [Value::Null, json!("inline"), json!({})] {
            let value = json!({
                "call_id": "call-1",
                "content": content,
                "artifact_ref": "artifact-1",
                "is_error": false
            });

            assert!(serde_json::from_value::<ToolResult>(value).is_err());
        }
    }

    #[test]
    fn unknown_fields_are_rejected_for_both_variants() {
        for fixture in [inline_fixture(), artifact_fixture()] {
            for field in ["approved", "role", "executable", "policy"] {
                let mut value = fixture.clone();
                value[field] = json!(true);

                assert!(
                    serde_json::from_value::<ToolResult>(value).is_err(),
                    "unexpected result field accepted: {field}"
                );
            }
        }
    }

    #[test]
    fn call_identity_must_be_a_string() {
        for fixture in [inline_fixture(), artifact_fixture()] {
            for invalid in [Value::Null, json!(42), json!(true), json!([]), json!({})] {
                let mut value = fixture.clone();
                value["call_id"] = invalid;

                assert!(serde_json::from_value::<ToolResult>(value).is_err());
            }
        }
    }

    #[test]
    fn artifact_reference_must_be_a_string() {
        for invalid in [Value::Null, json!(42), json!(true), json!([]), json!({})] {
            let mut value = artifact_fixture();
            value["artifact_ref"] = invalid;

            assert!(serde_json::from_value::<ToolResult>(value).is_err());
        }
    }

    #[test]
    fn error_flag_requires_an_explicit_boolean() {
        for fixture in [inline_fixture(), artifact_fixture()] {
            for invalid in [Value::Null, json!(0), json!(1), json!("false"), json!([])] {
                let mut value = fixture.clone();
                value["is_error"] = invalid;

                assert!(serde_json::from_value::<ToolResult>(value).is_err());
            }
        }
    }

    #[test]
    fn error_accessor_preserves_both_boolean_states_for_both_variants() {
        for fixture in [inline_fixture(), artifact_fixture()] {
            for expected in [false, true] {
                let mut value = fixture.clone();
                value["is_error"] = json!(expected);

                let result: ToolResult = serde_json::from_value(value).expect("result");

                assert_eq!(result.is_error(), expected);
            }
        }
    }

    #[test]
    fn structured_errors_roundtrip_for_every_category() {
        for (kind, wire_kind) in [
            (ToolErrorKind::Protocol, "protocol"),
            (ToolErrorKind::Execution, "execution"),
            (ToolErrorKind::Timeout, "timeout"),
            (ToolErrorKind::Policy, "policy"),
        ] {
            let error = ToolError {
                kind,
                message: "Request failed".into(),
                details: json!({
                    "reason": "fixture_failure",
                    "retryable": false
                }),
            };

            let wire = serde_json::to_value(&error).expect("serialize");
            assert_eq!(wire["kind"], wire_kind);

            let restored: ToolError = serde_json::from_value(wire).expect("deserialize");

            assert_eq!(restored, error);
        }
    }

    #[test]
    fn structured_error_requires_all_fields() {
        let original = json!({
            "kind": "execution",
            "message": "Tool failed",
            "details": { "reason": "cancelled" }
        });

        for field in ["kind", "message", "details"] {
            let mut value = original.clone();
            value.as_object_mut().expect("error object").remove(field);

            assert!(
                serde_json::from_value::<ToolError>(value).is_err(),
                "missing error field accepted: {field}"
            );
        }
    }

    #[test]
    fn structured_error_rejects_unknown_fields_and_categories() {
        let mut value = json!({
            "kind": "execution",
            "message": "Tool failed",
            "details": {}
        });
        value["approved"] = json!(true);

        assert!(serde_json::from_value::<ToolError>(value).is_err());

        for kind in ["cancelled", "unknown", "success", "Execution"] {
            assert!(serde_json::from_value::<ToolErrorKind>(json!(kind)).is_err());
        }
    }

    #[test]
    fn cancellation_uses_execution_kind_and_retains_reason() {
        let error = ToolError {
            kind: ToolErrorKind::Execution,
            message: "Cancelled while running tool".into(),
            details: json!({
                "reason": "cancelled",
                "stage": "execution"
            }),
        };

        let result = ToolResult::Inline(InlineToolResult {
            call_id: "call-cancelled".into(),
            content: serde_json::to_value(error).expect("error content"),
            is_error: true,
        });

        let wire = serde_json::to_value(&result).expect("result");

        assert_eq!(result.call_id(), "call-cancelled");
        assert!(result.is_error());
        assert_eq!(wire["content"]["kind"], "execution");
        assert_eq!(wire["content"]["details"]["reason"], "cancelled");
    }

    #[test]
    fn cloned_inline_result_has_independent_content() {
        let original = ToolResult::Inline(InlineToolResult {
            call_id: "call-1".into(),
            content: json!({ "text": "original" }),
            is_error: false,
        });

        let mut changed = original.clone();

        if let ToolResult::Inline(inline) = &mut changed {
            inline.call_id = "call-2".into();
            inline.content["text"] = json!("changed");
            inline.is_error = true;
        }

        assert_eq!(original.call_id(), "call-1");
        assert!(!original.is_error());

        let wire = serde_json::to_value(original).expect("serialize");
        assert_eq!(wire["content"]["text"], "original");

        assert_eq!(changed.call_id(), "call-2");
        assert!(changed.is_error());
    }
}
