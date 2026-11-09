//! JSON-RPC 2.0 codec, `api_version` gate, and Phase-1 error registry.

#[path = "slab_transport.rs"]
mod slab_transport;
pub use slab_transport::{
    RpcTransportLimits, measure_slab_session_event, write_slab_session_event,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Phase-1 daemon API version accepted on every request.
pub const API_VERSION: u32 = 1;

/// Application error kind carried in `error.data.kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Request `api_version` is not supported.
    UnsupportedApiVersion,
    /// Session id is unknown.
    SessionNotFound,
    /// Session exists but is not active for send.
    SessionNotActive,
    /// Parameters failed validation.
    InvalidParams,
    /// Engine or harness failure.
    EngineFailed,
    /// Method exists in B.6 but is out of Phase-1 scope.
    UnsupportedInPhase1,
}

impl ErrorKind {
    /// Wire string for `error.data.kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedApiVersion => "unsupported_api_version",
            Self::SessionNotFound => "session_not_found",
            Self::SessionNotActive => "session_not_active",
            Self::InvalidParams => "invalid_params",
            Self::EngineFailed => "engine_failed",
            Self::UnsupportedInPhase1 => "unsupported_in_phase1",
        }
    }
}

/// JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// JSON-RPC error code.
    pub code: i64,
    /// Human-readable message.
    pub message: String,
    /// Optional structured data including `kind`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    /// Standard invalid request (-32600).
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: message.into(),
            data: None,
        }
    }

    /// Method not found (-32601).
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
            data: None,
        }
    }

    /// Invalid params (-32602).
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
            data: Some(json!({ "kind": ErrorKind::InvalidParams.as_str() })),
        }
    }

    /// Internal error (-32603).
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: -32603,
            message: message.into(),
            data: None,
        }
    }

    /// Application error with Phase-1 kind (-32000 range).
    pub fn application(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            code: -32000,
            message: message.into(),
            data: Some(json!({ "kind": kind.as_str() })),
        }
    }

    /// Preserve the engine cause inside the existing API v1 error envelope.
    pub fn engine(error: &engine::error::EngineError) -> Self {
        Self {
            code: -32000,
            message: error.to_string(),
            data: Some(json!({
                "kind": ErrorKind::EngineFailed.as_str(),
                "engine_error": error.diagnostic(),
            })),
        }
    }
}

/// Incoming JSON-RPC request line.
#[derive(Debug, Clone, Deserialize)]
pub struct RpcRequest {
    /// Must be `"2.0"`.
    pub jsonrpc: String,
    /// Request id (null for notifications — Phase 1 clients always send ids).
    pub id: Option<Value>,
    /// Method name (B.6).
    pub method: String,
    /// Parameter object.
    #[serde(default)]
    pub params: Value,
    /// Required Phase-1 API version.
    pub api_version: Option<u32>,
}

/// Outgoing JSON-RPC response line.
#[derive(Debug, Clone, Serialize)]
pub struct RpcResponse {
    /// Always `"2.0"`.
    pub jsonrpc: &'static str,
    /// Echoed request id.
    pub id: Value,
    /// Success payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    /// Success response.
    pub fn result(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Error response.
    pub fn error(id: Value, error: RpcError) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// Server-push notification (no `id`).
#[derive(Debug, Clone, Serialize)]
pub struct RpcNotification {
    /// Always `"2.0"`.
    pub jsonrpc: &'static str,
    /// Notification method (`session.event`).
    pub method: &'static str,
    /// Event payload.
    pub params: Value,
}

impl RpcNotification {
    /// Build a `session.event` notification.
    pub fn session_event(event: &str, session_id: &str, body: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            method: "session.event",
            params: json!({
                "session_id": session_id,
                "event": event,
                "body": body,
            }),
        }
    }
}

/// Parse one NDJSON request line.
pub fn parse_request_line(line: &str) -> Result<RpcRequest, RpcError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(RpcError::invalid_request("empty line"));
    }
    harness_core::plugin_host::decode_configuration(trimmed.as_bytes())
        .map_err(|error| RpcError::invalid_request(format!("malformed JSON-RPC request: {error}")))
}

/// Validate JSON-RPC version and Phase-1 `api_version`.
pub fn validate_request(request: &RpcRequest) -> Result<(), RpcError> {
    if request.jsonrpc != "2.0" {
        return Err(RpcError::invalid_request("jsonrpc must be \"2.0\""));
    }
    match request.api_version {
        Some(API_VERSION) => Ok(()),
        Some(other) => Err(RpcError::application(
            ErrorKind::UnsupportedApiVersion,
            format!("unsupported api_version {other}; daemon supports {API_VERSION}"),
        )),
        None => Err(RpcError::application(
            ErrorKind::UnsupportedApiVersion,
            "api_version is required on every request",
        )),
    }
}

/// Serialize a value as one NDJSON line (with trailing newline).
pub fn encode_line(value: &impl Serialize) -> Result<String, RpcError> {
    let mut line = serde_json::to_string(value)
        .map_err(|error| RpcError::internal(format!("serialize response: {error}")))?;
    line.push('\n');
    Ok(line)
}

/// Map engine generation events to B.6 wire event names.
pub fn wire_event_name(event: &engine::types::GenerationEvent) -> &'static str {
    use engine::types::GenerationEvent;
    match event {
        GenerationEvent::TextDelta { .. } => "text_delta",
        GenerationEvent::ToolCallDelta { .. } => "tool_call_delta",
        GenerationEvent::ToolCallProposed { .. } => "tool_call_proposed",
        GenerationEvent::ToolCallBatchProposed { .. } => "tool_call_batch_proposed",
        GenerationEvent::RuntimeWarning { .. } => "runtime_warning",
        GenerationEvent::GenerationCompleted { .. } => "generation_completed",
        GenerationEvent::RequestFailed { .. } => "request_failed",
    }
}

/// Compatibility/fixture JSON body for a session event.
///
/// This helper materializes a Value tree. Production slab-batch notifications
/// must use write_slab_session_event, which borrows the slab directly.
/// Non-batch callers retain their existing encoding behavior.
pub fn wire_event_body(event: &engine::types::GenerationEvent) -> Value {
    use engine::types::GenerationEvent;
    match event {
        GenerationEvent::TextDelta { text } => json!({ "text": text }),
        GenerationEvent::ToolCallDelta { call_id, delta } => {
            json!({ "call_id": call_id, "delta": delta })
        }
        GenerationEvent::ToolCallProposed { proposal } => {
            json!({
                "name": proposal.name(),
                "arguments": proposal.arguments(),
            })
        }
        GenerationEvent::ToolCallBatchProposed { proposals } => {
            // RPC materializes a transport response; this is not
            // parser storage. Arguments stay validated JSON TEXT, not a second
            // unaccounted argument-object tree. Consumers must budget decoding.
            let calls: Vec<_> = (0..proposals.proposal_count()).filter_map(|index| proposals.proposal(index))
                .map(|proposal| json!({"name":proposal.name,"arguments_json":proposal.arguments_json})).collect();
            json!({"encoding":"validated_json_text","calls":calls})
        }
        GenerationEvent::RuntimeWarning { code, message } => {
            json!({ "code": code, "message": message })
        }
        GenerationEvent::GenerationCompleted {
            finish_reason,
            usage,
        } => json!({
            "finish_reason": match finish_reason {
                engine::types::FinishReason::EndOfSequence => "end_of_sequence",
                engine::types::FinishReason::OutputLimit => "output_limit",
            },
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
        }),
        GenerationEvent::RequestFailed {
            error,
            partial_usage,
        } => json!({
            "message": error.to_string(),
            "engine_error": error.diagnostic(),
            "partial_usage": {
                "input_tokens": partial_usage.input_tokens,
                "output_tokens": partial_usage.output_tokens,
            }
        }),
    }
}

// W2_RPC_PROPOSAL_CODEC_V1
#[cfg(test)]
mod w2_proposal_wire_tests {
    use super::*;
    use engine::types::{GenerationEvent, ToolCallProposal};

    #[test]
    fn proposal_wire_payload_preserves_data_without_inventing_authority() {
        let arguments = json!({
            "path": "file.txt",
            "nested": [null, true, {"x": 7}],
            "approved": true,
            "call_id": "argument-data-only"
        });
        let proposal =
            ToolCallProposal::new("fs.read", arguments.as_object().expect("object").clone())
                .expect("proposal");
        let event = GenerationEvent::ToolCallProposed { proposal };

        assert_eq!(wire_event_name(&event), "tool_call_proposed");
        assert!(!event.is_terminal());

        let body = wire_event_body(&event);
        assert_eq!(
            body,
            json!({
                "name": "fs.read",
                "arguments": arguments,
            })
        );
        assert!(body.get("call_id").is_none());
        assert!(body.get("approved").is_none());
        assert!(body.get("delta").is_none());

        let notification =
            RpcNotification::session_event(wire_event_name(&event), "session-test", body.clone());
        let encoded = encode_line(&notification).expect("encode");
        let decoded: Value = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded["method"], "session.event");
        assert_eq!(decoded["params"]["event"], "tool_call_proposed");
        assert_eq!(decoded["params"]["body"], body);
    }
}

#[cfg(test)]
mod w2_slab_wire_tests {
    use super::*;
    use engine::{
        admission::AdmissionController,
        parser_allocation::{ParserAllocation, ParserAllocationLimits},
        types::GenerationEvent,
    };
    #[test]
    fn batch_codec_preserves_validated_json_text_without_assigning_authority() {
        let limits = ParserAllocationLimits {
            carry_bytes: 32,
            call_bytes: 128,
            decoded_bytes: 128,
            depth: 8,
            keys: 16,
            nodes: 32,
            proposal_bytes: 256,
            calls: 4,
        };
        let bytes = ParserAllocation::required_bytes(limits).expect("layout");
        let admission = AdmissionController::new(bytes).expect("admission");
        let mut grant = admission.reserve_grant("slab", &[bytes]).expect("grant");
        let mut storage = ParserAllocation::from_grant(&mut grant, limits).expect("storage");
        let args = br#"{"call_id":"argument-data","approved":true}"#;
        {
            let work = storage.workspaces();
            work.decoded[..7].copy_from_slice(b"fs.read");
            work.call[..args.len()].copy_from_slice(args);
        }
        storage
            .commit_validated_call(7, 0..7, args.len(), 0..args.len())
            .expect("commit validated fixture");
        let event = GenerationEvent::ToolCallBatchProposed {
            proposals: storage.freeze(),
        };
        assert_eq!(wire_event_name(&event), "tool_call_batch_proposed");
        let body = wire_event_body(&event);
        assert_eq!(body["encoding"], "validated_json_text");
        assert_eq!(body["calls"][0]["name"], "fs.read");
        assert_eq!(
            body["calls"][0]["arguments_json"],
            std::str::from_utf8(args).expect("UTF-8")
        );
        assert!(body["calls"][0].get("call_id").is_none());
        assert!(body["calls"][0].get("approved").is_none());
        assert_eq!(
            admission.snapshot().expect("owner live").reserved_bytes(),
            bytes
        );
        drop(event);
        assert_eq!(
            admission
                .snapshot()
                .expect("owner released")
                .reserved_bytes(),
            0
        );
    }
}
