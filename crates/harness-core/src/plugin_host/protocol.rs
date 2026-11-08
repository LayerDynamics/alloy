//! Strict V1 hook envelopes. No response contains execution authority.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) const VERSION: u32 = 1;
pub(crate) const BEFORE: &str = "tool.before_execute";
pub(crate) const AFTER: &str = "tool.after_execute";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Envelope<T> {
    pub jsonrpc: String,
    pub id: u64,
    pub result: T,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Handshake {
    pub protocol_version: u32,
    pub plugin_id: String,
    pub plugin_version: String,
    pub hooks: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BeforeReply {
    pub decision: Decision,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub annotations: BTreeMap<String, Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AfterReply {
    #[serde(default)]
    pub annotations: BTreeMap<String, Value>,
}
/// The only decision available to a before hook; this is not authorization.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Permit later daemon checks to continue.
    Allow,
    /// Stop this invocation before durable start.
    Veto,
}
/// Immutable, daemon-derived snapshot supplied after validation and approval.
/// Constructing this data does not grant execution authority. W6 must bind it to
/// its approved receipt and must not hold store/session/native locks across IPC.
#[derive(Clone, Debug, Serialize)]
pub struct HookInvocation {
    /// Unique daemon invocation identity.
    pub invocation_id: String,
    /// Owning session.
    pub session_id: String,
    /// Canonical call identity.
    pub call_id: String,
    /// Exact registered definition (including version, origin and effect policy).
    pub definition: tool_registry::definition::ToolDefinition,
    /// Registered schema fingerprint.
    pub schema_hash: String,
    /// Validated canonical arguments.
    pub arguments: serde_json::Map<String, Value>,
    /// Bound approval/policy snapshot; not a model-issued approval flag.
    pub approval: Value,
    /// Durable deadline representation for plugin diagnostics.
    pub deadline_unix_ms: u64,
}
