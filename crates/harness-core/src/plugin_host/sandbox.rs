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
        "(version 1)\n(deny default)\n(allow process-exec)\n(deny process-fork)\n(allow file-read-data (literal \"/\"))\n(allow sysctl-read (sysctl-name \"security.mac.lockdown_mode_state\") (sysctl-name \"kern.bootargs\") (sysctl-name \"hw.pagesize_compat\"))\n",
    );
    // Content grants remain scoped. Metadata grants cover exact ancestor nodes
    // for both the configured spelling and the resolved path, not whole trees.
    let mut ancestors = std::collections::BTreeSet::new();
    for p in read_files
        .iter()
        .chain(read_trees.iter())
        .chain(write_trees.iter())
    {
        if !p.is_absolute() {
            return Err(HostError::new("sandbox resource must be absolute"));
        }
        let resolved = p.canonicalize().map_err(HostError::io)?;
        for spelling in [p.as_path(), resolved.as_path()] {
            for ancestor in spelling.ancestors() {
                ancestors.insert(ancestor.to_path_buf());
            }
        }
    }
    for p in ancestors {
        let s = p
            .to_str()
            .ok_or_else(|| HostError::new("non-UTF8 resource ancestor"))?;
        if s.chars().any(char::is_control) {
            return Err(HostError::new("invalid resource ancestor"));
        }
        let quoted = format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
        text.push_str(&format!("(allow file-read-metadata (literal {quoted}))\n"));
    }
    for p in read_files {
        text.push_str(&format!("(allow file-read* (literal {}))\n", literal(p)?));
    }
    for p in read_trees {
        if !p.is_dir() {
            return Err(HostError::new("read tree is not a directory"));
        }
        text.push_str(&format!("(allow file-read* (subpath {}))\n", literal(p)?));
    }
    for p in write_trees {
        if !p.is_dir() {
            return Err(HostError::new("write tree is not a directory"));
        }
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
    let program = config.program.canonicalize().map_err(HostError::io)?;
    let cwd = config.cwd.canonicalize().map_err(HostError::io)?;
    if !program.is_file() || !cwd.is_dir() {
        return Err(HostError::new("invalid plugin program/cwd"));
    }
    let mut cmd = match &config.sandbox {
        Sandbox::Trusted => Command::new(&program),
        Sandbox::MacOs { .. } => {
            if !cfg!(target_os = "macos") {
                return Err(HostError::new("macOS sandbox unavailable on this platform"));
            }
            if let Sandbox::MacOs { write_trees, .. } = &config.sandbox {
                for tree in write_trees {
                    let tree = tree.canonicalize().map_err(HostError::io)?;
                    if program.starts_with(&tree) {
                        return Err(HostError::new(
                            "plugin executable is inside a plugin-writable tree",
                        ));
                    }
                }
            }
            let mut policy = profile(&config.sandbox)?;
            policy.push_str(&format!(
                "(deny process-exec (require-not (literal {})))\n",
                literal(&program)?
            ));
            let mut command = Command::new("/usr/bin/sandbox-exec");
            command.arg("-p").arg(policy).arg(&program);
            command
        }
    };
    cmd.args(&config.args).current_dir(cwd).env_clear();
    Ok(cmd)
}
