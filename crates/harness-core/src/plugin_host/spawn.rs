//! Plugin-specific policy adaptation for the shared process owner.
use super::{HostError, PluginConfig, sandbox};
pub(super) use alloy_process_owner::spawn::Child;
use std::{path::Path, time::Instant};
use tool_registry::execution::CancellationToken;
type Spawned = alloy_process_owner::spawn::Spawned;

pub(super) fn prepare_helper(
    configs: &[PluginConfig],
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Result<Option<std::path::PathBuf>, HostError> {
    let audits: Vec<_> = configs
        .iter()
        .map(|config| alloy_process_owner::spawn::AuditConfig {
            program: config.program.clone(),
            write_trees: match &config.sandbox {
                sandbox::Sandbox::MacOs { write_trees, .. } => write_trees.clone(),
                sandbox::Sandbox::Trusted => Vec::new(),
            },
        })
        .collect();
    alloy_process_owner::spawn::prepare_helper(&audits, helper, deadline, &|| cancel.is_cancelled())
        .map_err(HostError::io)
}

pub(super) fn spawn(
    config: &PluginConfig,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &CancellationToken,
    external_reaper: bool,
) -> Result<Spawned, HostError> {
    let command = sandbox::command(config)?;
    alloy_process_owner::spawn::spawn(
        command,
        helper,
        deadline,
        &|| cancel.is_cancelled(),
        external_reaper,
    )
    .map_err(HostError::io)
}
