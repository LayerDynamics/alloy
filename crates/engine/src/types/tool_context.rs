//! Model-facing tool metadata and linked history, not a tool registry.
//! Callers project schemas from validated registrations. This module neither
//! registers implementations, validates arguments against schemas, nor grants
//! identity/approval/authority. Limits bound accepted serialized input, not
//! transient serde/MiniJinja allocations.
use super::{EngineError, Message};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Model-visible projection of a validated registered definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderingTool {
    id: String,
    version: String,
    description: String,
    input_schema: Value,
    registered_policy: Option<Value>,
}
impl RenderingTool {
    /// Copy exact metadata from a validated registration, not model output.
    /// This checks shape only; registry/schema validation is still required.
    pub fn new(
        id: String,
        version: String,
        description: String,
        input_schema: Value,
    ) -> Result<Self, EngineError> {
        super::validate_identifier(&id, "tool.id")?;
        if version.is_empty() || version.trim() != version || version.chars().any(char::is_control)
        {
            return Err(invalid(
                "tool.version",
                "missing or malformed registration version",
            ));
        }
        if !input_schema.is_object() && !input_schema.is_boolean() {
            return Err(invalid(
                "tool.inputSchema",
                "schema must be an object or boolean",
            ));
        }
        Ok(Self {
            id,
            version,
            description,
            input_schema: canonical(input_schema),
            registered_policy: None,
        })
    }
    /// Attach descriptive registered policy metadata. This is model-facing
    /// information only; daemon policy and approvals remain authoritative.
    pub fn with_registered_policy(mut self, policy: Value) -> Result<Self, EngineError> {
        if !policy.is_object() {
            return Err(invalid(
                "tool.registered_policy",
                "policy must be an object",
            ));
        }
        self.registered_policy = Some(canonical(policy));
        Ok(self)
    }
    fn value(&self) -> Value {
        let mut value = json!({"type":"function", "function": {
            "name": self.id, "version": self.version,
            "description": self.description, "parameters": self.input_schema,
        }});
        if let Some(policy) = &self.registered_policy {
            value["function"]["registered_policy"] = policy.clone();
            value["function"]["local_policy_enforced_separately"] = Value::Bool(true);
        }
        value
    }
}

/// Ordered tail appended to the snapshot text messages.
/// Call IDs are daemon-bound historical data; the model cannot mint them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolHistoryBlock {
    /// Ordinary text, kept under its original role.
    Text(Message),
    /// Validated, daemon-bound assistant call from an earlier generation.
    AssistantCall {
        /// Daemon call identifier.
        call_id: String,
        /// Registered tool identifier.
        tool_id: String,
        /// Exact historical argument object.
        arguments: serde_json::Map<String, Value>,
    },
    /// Result linked to an earlier assistant call, never a system message.
    ToolResult {
        /// Identifier of the call receiving this result.
        call_id: String,
        /// Inline result, or a serialized artifact-reference description.
        content: String,
        /// Whether the historical result represents an error.
        is_error: bool,
    },
}

/// Explicit limits for accepted tool-context metadata/history.
#[derive(Clone, Copy, Debug)]
pub struct ToolContextLimits {
    /// Maximum number of definitions advertised to the model.
    pub max_tools: usize,
    /// Maximum historical blocks appended to the base messages.
    pub max_history_blocks: usize,
    /// Maximum compact serialized tools/history bytes combined.
    pub max_json_bytes: usize,
}

/// Immutable model-visible tool definitions and ordered, linked history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolContext {
    tools: Vec<RenderingTool>,
    history: Vec<ToolHistoryBlock>,
    schema_hash: [u8; 32],
    input_bytes: usize,
}
impl ToolContext {
    /// Construct a new validated context with a projected historical tail.
    ///
    /// This does not mutate the original snapshot or authorize the projected
    /// calls. Normal size, identity and call/result linkage checks apply to
    /// the complete combined context.
    pub fn with_appended_history(
        &self,
        blocks: impl IntoIterator<Item = ToolHistoryBlock>,
        limits: ToolContextLimits,
    ) -> Result<Self, EngineError> {
        let mut history = self.history.clone();
        for block in blocks {
            if history.len() >= limits.max_history_blocks {
                return Err(invalid("tool_context", "history count limit exceeded"));
            }
            history.push(block);
        }
        Self::new(self.tools.clone(), history, limits)
    }
    /// Freeze schema order and validate historical call/result linkage.
    /// No registration, execution, policy, or approval is performed here.
    pub fn new(
        mut tools: Vec<RenderingTool>,
        history: Vec<ToolHistoryBlock>,
        limits: ToolContextLimits,
    ) -> Result<Self, EngineError> {
        if limits.max_tools == 0 || limits.max_history_blocks == 0 || limits.max_json_bytes == 0 {
            return Err(invalid(
                "tool_context.limits",
                "all limits must be positive",
            ));
        }
        if tools.len() > limits.max_tools || history.len() > limits.max_history_blocks {
            return Err(invalid(
                "tool_context",
                "tool or history count limit exceeded",
            ));
        }
        tools.sort_by(|a, b| a.id.cmp(&b.id));
        if tools.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err(invalid("tool_context.tools", "duplicate tool identifier"));
        }
        let mut calls = BTreeSet::new();
        let mut results = BTreeSet::new();
        for block in &history {
            match block {
                ToolHistoryBlock::Text(_) => {}
                ToolHistoryBlock::AssistantCall {
                    call_id, tool_id, ..
                } => {
                    super::validate_identifier(call_id, "tool.call_id")?;
                    super::validate_identifier(tool_id, "tool.id")?;
                    if !calls.insert(call_id.as_str()) {
                        return Err(invalid("tool.call_id", "duplicate historical call"));
                    }
                }
                ToolHistoryBlock::ToolResult { call_id, .. } => {
                    if !calls.contains(call_id.as_str()) || !results.insert(call_id.as_str()) {
                        return Err(invalid(
                            "tool.call_id",
                            "unknown, future, or duplicate result link",
                        ));
                    }
                }
            }
        }
        drop(calls);
        drop(results);
        let tools_value = Value::Array(tools.iter().map(RenderingTool::value).collect());
        let tools_json = serialize(&canonical(tools_value))?;
        let history_json = serialize(&Value::Array(history.iter().map(history_value).collect()))?;
        let input_bytes = tools_json
            .len()
            .checked_add(history_json.len())
            .ok_or_else(|| invalid("tool_context", "serialized byte count overflow"))?;
        if input_bytes > limits.max_json_bytes {
            return Err(invalid("tool_context", "serialized byte limit exceeded"));
        }
        let mut hash = Sha256::new();
        hash.update(b"alloy.tool-schemas.v1\0");
        hash.update(tools_json.as_bytes());
        Ok(Self {
            tools,
            history,
            schema_hash: hash.finalize().into(),
            input_bytes,
        })
    }
    /// Schema/metadata fingerprint used by rendering and cache compatibility.
    pub fn schema_hash(&self) -> &[u8; 32] {
        &self.schema_hash
    }
    /// Number of historical blocks appended by this context.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    /// Serialized tool/history bytes charged to rendering input limits.
    pub fn input_bytes(&self) -> usize {
        self.input_bytes
    }
    /// Exact deterministic model-visible function schemas.
    pub(crate) fn tools_json(&self) -> Result<String, EngineError> {
        serialize(&canonical(Value::Array(
            self.tools.iter().map(RenderingTool::value).collect(),
        )))
    }
    /// Reject reserved control syntax in all model-facing data before rendering.
    pub(crate) fn validate_spellings(&self, reserved: &[&str]) -> Result<(), EngineError> {
        for value in self
            .tools
            .iter()
            .map(RenderingTool::value)
            .chain(self.history.iter().map(history_value))
        {
            let text = serialize(&value)?;
            if reserved.iter().any(|token| text.contains(token)) {
                return Err(invalid(
                    "tool_context",
                    "data contains reserved structural syntax",
                ));
            }
        }
        Ok(())
    }
    /// Values are separate role-bearing messages, never injected into system content.
    pub(crate) fn messages(&self) -> Vec<Value> {
        self.history.iter().map(history_value).collect()
    }
}
fn history_value(block: &ToolHistoryBlock) -> Value {
    match block {
        ToolHistoryBlock::Text(message) => json!({
            "kind":"text", "role":message.role.as_str(), "content":message.content,
        }),
        ToolHistoryBlock::AssistantCall {
            call_id,
            tool_id,
            arguments,
        } => {
            let call = canonical(json!({"name":tool_id, "arguments":arguments}));
            // Serialization of an in-memory JSON value cannot contain invalid UTF-8.
            json!({"kind":"call", "role":"assistant", "content":"", "call_id":call_id,
                "call_json":call.to_string()})
        }
        ToolHistoryBlock::ToolResult {
            call_id,
            content,
            is_error,
        } => {
            let result =
                canonical(json!({"call_id":call_id, "content":content, "is_error":is_error}));
            json!({"kind":"result", "role":"tool", "content":result.to_string(), "call_id":call_id})
        }
    }
}
fn serialize(value: &Value) -> Result<String, EngineError> {
    serde_json::to_string(value).map_err(|error| invalid("tool_context", error.to_string()))
}
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(entries) => {
            let sorted: BTreeMap<_, _> = entries
                .into_iter()
                .map(|(k, v)| (k, canonical(v)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical).collect()),
        other => other,
    }
}
fn invalid(field: &'static str, message: impl Into<String>) -> EngineError {
    EngineError::invalid_request(field, message)
}
