//! W5 supervised two-hook host. This module does not authorize/start tools.
//! W6 must invoke before hooks after approval but BEFORE durable tool.started;
//! invoke after hooks only for attempted execution, retaining the original result.
mod process;
mod protocol;
mod requests;
pub mod sandbox;
mod strict_json;
use process::Process;
use protocol::{AFTER, AfterReply, BEFORE, BeforeReply, Handshake, VERSION};
pub use protocol::{Decision, HookInvocation};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::{Duration, Instant},
};
use tool_registry::{execution::CancellationToken, result::ToolResult};

/// Internal host error, not a new serialized tool error category.
#[derive(Debug)]
pub struct HostError(String);
impl HostError {
    pub(crate) fn new(message: &str) -> Self {
        Self(message.into())
    }
    pub(crate) fn io(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}
impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for HostError {}
/// Trusted daemon-selected transport/annotation limits.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Entire request/response line ceiling, including newline.
    pub message_bytes: usize,
    /// Retained stderr ceiling; excess per exchange terminates the plugin.
    pub diagnostic_bytes: usize,
    /// Maximum total serialized annotation bytes.
    pub annotation_bytes: usize,
    /// Maximum veto reason UTF-8 bytes.
    pub reason_bytes: usize,
    /// Per-exchange timeout, capped by the invocation's monotonic deadline.
    pub timeout: Duration,
}
/// Daemon configuration, never accepted from model output.
#[derive(Clone, Debug)]
pub struct PluginConfig {
    /// Expected handshake plugin ID (ASCII identifier).
    pub id: String,
    /// Expected exact plugin version.
    pub version: String,
    /// Absolute executable, launched without shell interpolation.
    pub program: PathBuf,
    /// Explicit arguments.
    pub args: Vec<String>,
    /// Absolute working directory.
    pub cwd: PathBuf,
    /// Explicit trust/sandbox policy.
    pub sandbox: sandbox::Sandbox,
}
/// Auditable receipt. An annotation never replaces any part of a tool result.
#[derive(Debug)]
pub struct HookReceipt {
    /// Expected plugin identity.
    pub plugin_id: String,
    /// Exact expected version.
    pub plugin_version: String,
    /// Owning invocation.
    pub invocation_id: String,
    /// Hook name.
    pub hook: &'static str,
    /// Valid bounded annotations only.
    pub annotations: BTreeMap<String, Value>,
    /// Error/warning or veto reason.
    pub diagnostic: Option<String>,
}
/// Before decision is a constraint, not an execution permit.
#[derive(Debug)]
pub struct BeforeOutcome {
    /// Allow means hooks did not veto; all daemon checks still apply.
    pub decision: Decision,
    /// Ordered receipts through the first failure/veto.
    pub receipts: Vec<HookReceipt>,
}
struct Plugin {
    config: PluginConfig,
    process: Process,
    hooks: BTreeSet<String>,
}
/// Owns child lifetimes in explicit configuration order; no automatic retries.
pub struct PluginHost {
    plugins: Vec<Plugin>,
    limits: Limits,
}
impl PluginHost {
    /// Handshake all plugins atomically before publishing a host. Failure drops
    /// and reaps already-started children. No tool is offered during registration.
    pub fn start(
        configs: Vec<PluginConfig>,
        limits: Limits,
        deadline: Instant,
        cancel: &CancellationToken,
    ) -> Result<Self, HostError> {
        if configs.len() > 64
            || limits.message_bytes > 8 * 1024 * 1024
            || limits.diagnostic_bytes > 1024 * 1024
            || limits.annotation_bytes > limits.message_bytes
            || limits.reason_bytes > limits.message_bytes
            || limits.message_bytes < 2
            || limits.diagnostic_bytes == 0
            || limits.annotation_bytes == 0
            || limits.timeout.is_zero()
        {
            return Err(HostError::new("invalid plugin limits"));
        }
        let mut ids = BTreeSet::new();
        for c in &configs {
            if c.id.is_empty()
                || !c
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                || c.version.is_empty()
                || !ids.insert(c.id.clone())
            {
                return Err(HostError::new("invalid/duplicate plugin identity"));
            }
        }
        let mut host = Self {
            plugins: Vec::new(),
            limits,
        };
        for config in configs {
            if cancel.is_cancelled() || Instant::now() >= deadline {
                return Err(HostError::new("registration cancelled/expired"));
            }
            let mut process = Process::spawn(&config, limits)?;
            let handshake: Handshake = process.rpc("initialize", serde_json::json!({
                "protocol_version":VERSION, "plugin_id":config.id, "plugin_version":config.version,
                "hooks":[BEFORE,AFTER]
            }), deadline, cancel)?;
            let hooks: BTreeSet<_> = handshake.hooks.iter().cloned().collect();
            if handshake.protocol_version != VERSION
                || handshake.plugin_id != config.id
                || handshake.plugin_version != config.version
                || hooks.len() != handshake.hooks.len()
                || hooks.iter().any(|h| h != BEFORE && h != AFTER)
            {
                return Err(HostError::new("plugin handshake mismatch"));
            }
            host.plugins.push(Plugin {
                config,
                process,
                hooks,
            });
        }
        Ok(host)
    }
    /// Run declared before hooks serially. Failure/veto stops later hooks.
    /// Caller must map a veto/failure to the existing policy result, zero effects.
    pub fn before(
        &mut self,
        call: &HookInvocation,
        deadline: Instant,
        cancel: &CancellationToken,
    ) -> BeforeOutcome {
        let mut receipts = Vec::new();
        for plugin in &mut self.plugins {
            if !plugin.hooks.contains(BEFORE) {
                continue;
            }
            let response: Result<BeforeReply, HostError> =
                plugin
                    .process
                    .rpc(BEFORE, requests::before(call), deadline, cancel);
            let mut receipt = receipt(plugin, call, BEFORE);
            match response.and_then(|r| {
                validate(&r.annotations, &plugin.config.id, self.limits)?;
                if r.reason.len() > self.limits.reason_bytes {
                    return Err(HostError::new("reason byte limit"));
                }
                Ok(r)
            }) {
                Ok(r) => {
                    receipt.annotations = r.annotations;
                    if !r.reason.is_empty() {
                        receipt.diagnostic = Some(r.reason);
                    }
                    let veto = r.decision == Decision::Veto;
                    receipts.push(receipt);
                    if veto {
                        return BeforeOutcome {
                            decision: Decision::Veto,
                            receipts,
                        };
                    }
                }
                Err(error) => {
                    plugin.process.stop();
                    receipt.diagnostic = Some(error.to_string());
                    receipts.push(receipt);
                    return BeforeOutcome {
                        decision: Decision::Veto,
                        receipts,
                    };
                }
            }
        }
        if cancel.is_cancelled() || Instant::now() >= deadline {
            return BeforeOutcome {
                decision: Decision::Veto,
                receipts,
            };
        }
        BeforeOutcome {
            decision: Decision::Allow,
            receipts,
        }
    }
    /// Annotate one actual attempted execution; errors become warning receipts.
    /// No executor callback exists here, so this method cannot rerun an effect.
    /// W6 must call once and persist receipts with the unchanged original result.
    pub fn after(
        &mut self,
        call: &HookInvocation,
        result: &ToolResult,
        duration_ms: u64,
        deadline: Instant,
        cancel: &CancellationToken,
    ) -> Vec<HookReceipt> {
        let mut receipts = Vec::new();
        for plugin in &mut self.plugins {
            if !plugin.hooks.contains(AFTER) {
                continue;
            }
            let response: Result<AfterReply, HostError> = plugin.process.rpc(
                AFTER,
                requests::after(call, result, duration_ms),
                deadline,
                cancel,
            );
            let mut receipt = receipt(plugin, call, AFTER);
            match response.and_then(|r| {
                validate(&r.annotations, &plugin.config.id, self.limits)?;
                Ok(r)
            }) {
                Ok(r) => receipt.annotations = r.annotations,
                Err(error) => {
                    plugin.process.stop();
                    receipt.diagnostic = Some(error.to_string());
                }
            }
            receipts.push(receipt);
        }
        receipts
    }
    /// Bounded retained diagnostics, including stopped children, in plugin order.
    pub fn diagnostics(&self) -> Vec<(&str, &[u8])> {
        self.plugins
            .iter()
            .map(|p| (p.config.id.as_str(), p.process.diagnostics()))
            .collect()
    }
}
fn receipt(plugin: &Plugin, call: &HookInvocation, hook: &'static str) -> HookReceipt {
    HookReceipt {
        plugin_id: plugin.config.id.clone(),
        plugin_version: plugin.config.version.clone(),
        invocation_id: call.invocation_id.clone(),
        hook,
        annotations: BTreeMap::new(),
        diagnostic: None,
    }
}
fn validate(values: &BTreeMap<String, Value>, id: &str, limits: Limits) -> Result<(), HostError> {
    let prefix = format!("{id}.");
    if values
        .keys()
        .any(|k| !k.starts_with(&prefix) || k.len() == prefix.len())
        || serde_json::to_vec(values)
            .map_err(|_| HostError::new("annotation encoding"))?
            .len()
            > limits.annotation_bytes
    {
        return Err(HostError::new("annotation namespace/byte limit"));
    }
    Ok(())
}
