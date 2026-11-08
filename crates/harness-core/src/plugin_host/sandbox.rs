//! Explicit macOS sandbox policy; absence never silently becomes unsandboxed.
use super::{HostError, PluginConfig};
use std::{path::PathBuf, process::Command};

/// Daemon-owned process trust policy, never selected by a model/plugin.
#[derive(Clone, Debug)]
pub enum Sandbox {
    /// Explicitly trusted local process. This is NOT a sandbox guarantee.
    Trusted,
    /// macOS deny-default Seatbelt profile with declared file resources.
    /// Runtime read roots must include the executable and dynamic loader needs.
    MacOs {
        /// Exact readable files (canonical existing paths).
        read_files: Vec<PathBuf>,
        /// Readable trees (canonical existing directories).
        read_trees: Vec<PathBuf>,
        /// Writable trees (canonical existing directories).
        write_trees: Vec<PathBuf>,
    },
}
fn literal(path: &std::path::Path) -> Result<String, HostError> {
    let path = path.canonicalize().map_err(HostError::io)?;
    let text = path
        .to_str()
        .ok_or_else(|| HostError::new("non-UTF8 sandbox path"))?;
    if text.chars().any(char::is_control) {
        return Err(HostError::new("control character in sandbox path"));
    }
    Ok(format!(
        "\"{}\"",
        text.replace('\\', "\\\\").replace('"', "\\\"")
    ))
}
/// Render a deny-default profile. Network access is not granted.
pub fn profile(policy: &Sandbox) -> Result<String, HostError> {
    let Sandbox::MacOs {
        read_files,
        read_trees,
        write_trees,
    } = policy
    else {
        return Err(HostError::new("trusted policy has no sandbox profile"));
    };
    let mut text = String::from(
        "(version 1)\n(deny default)\n(allow process-exec process-fork)\n(allow file-read-data (literal \"/\"))\n(allow sysctl-read (sysctl-name \"security.mac.lockdown_mode_state\") (sysctl-name \"kern.bootargs\"))\n",
    );
    for p in read_files {
        text.push_str(&format!("(allow file-read* (literal {}))\n", literal(p)?));
    }
    for p in read_trees {
        text.push_str(&format!("(allow file-read* (subpath {}))\n", literal(p)?));
    }
    for p in write_trees {
        text.push_str(&format!(
            "(allow file-read* file-write* (subpath {}))\n",
            literal(p)?
        ));
    }
    Ok(text)
}
pub(crate) fn command(config: &PluginConfig) -> Result<Command, HostError> {
    if !config.program.is_absolute() || !config.cwd.is_absolute() {
        return Err(HostError::new("plugin executable and cwd must be absolute"));
    }
    let mut cmd = match &config.sandbox {
        Sandbox::Trusted => Command::new(&config.program),
        Sandbox::MacOs { .. } => {
            if !cfg!(target_os = "macos") {
                return Err(HostError::new("macOS sandbox unavailable on this platform"));
            }
            let mut command = Command::new("/usr/bin/sandbox-exec");
            command
                .arg("-p")
                .arg(profile(&config.sandbox)?)
                .arg(&config.program);
            command
        }
    };
    cmd.args(&config.args).current_dir(&config.cwd).env_clear();
    Ok(cmd)
}
