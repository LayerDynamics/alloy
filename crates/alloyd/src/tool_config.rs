//! Strict, bounded daemon tool configuration. No model/client chooses scopes or plugins.
use engine::types::ToolContextLimits;
use harness_core::{
    artifact::ArtifactStore,
    plugin_host::{self, PluginConfig, PluginHost, sandbox::Sandbox},
    policy::{DaemonPolicy, PolicyConfig},
    supervisor::{ToolLoopConfig, ToolRoundConfig, ToolRuntime},
    tool_execution::{BuiltinImplementations, ReceiptExecutor},
};
use serde::Deserialize;
use std::{
    io::Read,
    net::IpAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tool_registry::{
    builtins::{
        fs::{FileListTool, FileReadTool, FileWriteTool, ReadScope, WriteScope},
        http::{HttpDestination, HttpFetchTool, HttpScope},
        shell::{ShellRunTool, ShellScope},
    },
    execution::CancellationToken,
};

/// Frozen bounded coordinator defaults, independent of model context admission.
pub fn loop_config(max_output_tokens: usize) -> ToolLoopConfig {
    ToolLoopConfig {
        round: ToolRoundConfig {
            context_limits: ToolContextLimits {
                max_tools: 64,
                max_history_blocks: 512,
                max_json_bytes: 1024 * 1024,
            },
            max_output_tokens,
            max_rounds: 8,
            max_proposals: 32,
            max_proposal_bytes: 1024 * 1024,
            max_slab_bytes: 8 * 1024 * 1024,
            max_text_bytes: 1024 * 1024,
            max_warning_bytes: 65536,
            max_events: 65536,
            deadline: Instant::now() + Duration::from_secs(120),
        },
        max_calls: 32,
        inline_bytes: 8192,
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    read_root: PathBuf,
    write_root: PathBuf,
    shell_executables: Vec<PathBuf>,
    shell_cwd_roots: Vec<PathBuf>,
    http_destinations: Vec<Destination>,
    policy: PolicyConfig,
    plugins: Vec<Plugin>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Destination {
    origin: String,
    address: IpAddr,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plugin {
    id: String,
    version: String,
    program: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
    sandbox: Trust,
}
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum Trust {
    Trusted,
    MacOs {
        read_files: Vec<PathBuf>,
        read_trees: Vec<PathBuf>,
        write_trees: Vec<PathBuf>,
    },
}
/// Load an explicit local configuration (at most 1 MiB), reject duplicate/unknown keys,
/// construct all five real scoped implementations and handshake ordered plugins.
/// A missing/invalid configuration never silently enables execution.
pub fn load(path: &Path, artifacts: ArtifactStore) -> Result<ToolRuntime, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 1_048_576 {
        return Err("tool configuration exceeds 1 MiB".into());
    }
    let config: Config = plugin_host::decode_configuration(&bytes).map_err(|e| e.to_string())?;
    let policy = DaemonPolicy::new(config.policy).map_err(|e| e.to_string())?;
    let ioerr = |e: &dyn std::fmt::Display| e.to_string();
    let read = FileReadTool::new(
        ReadScope::open(&config.read_root).map_err(|e| ioerr(&e))?,
        8192,
    )
    .map_err(|e| ioerr(&e))?;
    let list = FileListTool::new(ReadScope::open(&config.read_root).map_err(|e| ioerr(&e))?);
    let write = FileWriteTool::new(
        WriteScope::open(&config.write_root, 1024 * 1024).map_err(|e| ioerr(&e))?,
    );
    let shell = ShellRunTool::new(
        ShellScope::new(config.shell_executables, config.shell_cwd_roots, 65536)
            .map_err(|e| ioerr(&e))?,
    );
    let destinations = config
        .http_destinations
        .into_iter()
        .map(|d| HttpDestination::new(&d.origin, d.address).map_err(|e| ioerr(&e)))
        .collect::<Result<Vec<_>, _>>()?;
    let http = HttpFetchTool::new(
        HttpScope::new(destinations, 4, 1024 * 1024, 8192, Duration::from_secs(30))
            .map_err(|e| ioerr(&e))?,
    );
    let executor = ReceiptExecutor::with_builtins(
        30_000,
        1024 * 1024,
        BuiltinImplementations {
            fs_read: Arc::new(read),
            fs_list: Arc::new(list),
            fs_write: Arc::new(write),
            shell_run: Arc::new(shell),
            http_fetch: Arc::new(http),
        },
    )
    .map_err(|e| ioerr(&e))?;
    // A policy typo is a startup failure, never a model-visible phantom registration.
    if policy_rules_unknown(&bytes, executor.registry())? {
        return Err("policy references an unregistered tool".into());
    }
    let plugins = config
        .plugins
        .into_iter()
        .map(|p| PluginConfig {
            id: p.id,
            version: p.version,
            program: p.program,
            args: p.args,
            cwd: p.cwd,
            sandbox: match p.sandbox {
                Trust::Trusted => Sandbox::Trusted,
                Trust::MacOs {
                    read_files,
                    read_trees,
                    write_trees,
                } => Sandbox::MacOs {
                    read_files,
                    read_trees,
                    write_trees,
                },
            },
        })
        .collect();
    let host = PluginHost::start(
        plugins,
        plugin_host::Limits {
            message_bytes: 2 * 1024 * 1024,
            diagnostic_bytes: 65536,
            annotation_bytes: 65536,
            reason_bytes: 4096,
            timeout: Duration::from_secs(5),
        },
        Instant::now() + Duration::from_secs(30),
        &CancellationToken::default(),
    )
    .map_err(|e| ioerr(&e))?;
    Ok(ToolRuntime {
        executor,
        policy,
        artifacts,
        plugins: Mutex::new(host),
    })
}
fn policy_rules_unknown(
    bytes: &[u8],
    registry: &tool_registry::registry::ToolRegistry,
) -> Result<bool, String> {
    let c: Config = plugin_host::decode_configuration(bytes).map_err(|e| e.to_string())?;
    Ok(c.policy
        .rules
        .iter()
        .any(|r| !registry.definitions().any(|d| d.id == r.tool_id)))
}
