//! W4 shell process primitive, NOT authorization or a sandbox.
//!
//! A future harness bridge must require unchanged W3 destructive/always approval
//! and a durable StartedInvocation before calling run. No live dispatch is wired.
//! Configured executables/cwd trees must be trusted against OS-level mutation.
//! Environment is cleared; stdin is null. argv is unchanged and no shell is
//! inserted. An explicitly approved shell executable can of course interpret its
//! own arguments. Process groups do not contain descendants that escape via
//! setsid/setpgid. Only the direct child is reaped here; this is not a subreaper.
//! Do not retain store/session locks or the native generation slot across run.
//! No other waiter/SIGCHLD handler may reap the owned child. Blocking spawn,
//! kernel I/O, sink calls and final wait are not universally interruptible.

use crate::execution::{ExecutionContext, ExecutionError, OutputSink};
use rustix::{
    fs::{OFlags, fcntl_getfl, fcntl_setfl},
    process::{Pid, Signal, WaitId, WaitIdOptions, kill_process_group, waitid},
};
use std::{
    fmt, fs,
    io::{self, Read},
    os::unix::{
        fs::MetadataExt,
        process::{CommandExt, ExitStatusExt},
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    time::Duration,
};

/// Exact requested executable/argv/cwd, not an execution permit.
#[derive(Debug)]
pub struct ShellRequest {
    executable: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
}
impl ShellRequest {
    /// Absolute executable/cwd paths and NUL-free arguments are required. No
    /// canonicalization or expansion is performed on the approved request.
    pub fn new(executable: &str, args: Vec<String>, cwd: &str) -> Result<Self, ShellError> {
        if !Path::new(executable).is_absolute()
            || !Path::new(cwd).is_absolute()
            || executable.contains('\0')
            || cwd.contains('\0')
            || args.iter().any(|a| a.contains('\0'))
        {
            return Err(ShellError::Policy(
                "absolute paths and NUL-free argv required",
            ));
        }
        Ok(Self {
            executable: executable.into(),
            args,
            cwd: cwd.into(),
        })
    }
}

/// Internal failure, not a tool-result wire error category.
#[derive(Debug)]
pub enum ShellError {
    Policy(&'static str),
    Execution(ExecutionError),
    Io(io::Error),
}
impl fmt::Display for ShellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(s) => write!(f, "shell policy: {s}"),
            Self::Execution(e) => write!(f, "shell execution: {e}"),
            Self::Io(e) => write!(f, "shell I/O: {e}"),
        }
    }
}
impl std::error::Error for ShellError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for ShellError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<ExecutionError> for ShellError {
    fn from(e: ExecutionError) -> Self {
        Self::Execution(e)
    }
}
fn os(e: rustix::io::Errno) -> ShellError {
    ShellError::Io(e.into())
}

/// Failure preserves attempted execution and cleanup diagnostics; never retry
/// automatically or claim zero effects merely because output was aborted.
#[derive(Debug)]
pub struct ShellFailure {
    pub started: bool,
    pub pid: Option<u32>,
    pub cause: ShellError,
    pub cleanup_errors: Vec<String>,
}
impl fmt::Display for ShellFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "shell started={}: {}; cleanup={:?}",
            self.started, self.cause, self.cleanup_errors
        )
    }
}
impl std::error::Error for ShellFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Process exit and exact captured-byte counts. A nonzero/signal exit is not a
/// successful tool outcome: the future result adapter must use success=false.
#[derive(Debug)]
pub struct ShellReport {
    pub pid: u32,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout_bytes: u64,
    pub stderr_bytes: u64,
    pub output_bytes: u64,
}

/// Trusted executable allowlist, cwd roots, and invocation-size bound, separate
/// from approval and the effective execution deadline/output ceiling.
pub struct ShellScope {
    executables: Vec<PathBuf>,
    cwd_roots: Vec<PathBuf>,
    max_invocation_bytes: usize,
}
impl ShellScope {
    /// Canonicalize trusted configuration only. Requests must already name the
    /// canonical paths exactly; aliases are rejected, not silently rewritten.
    pub fn new(
        executables: Vec<PathBuf>,
        cwd_roots: Vec<PathBuf>,
        max_invocation_bytes: usize,
    ) -> io::Result<Self> {
        if executables.is_empty() || cwd_roots.is_empty() || max_invocation_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "nonempty scope and positive invocation limit required",
            ));
        }
        let executables = executables
            .into_iter()
            .map(fs::canonicalize)
            .collect::<io::Result<Vec<_>>>()?;
        let cwd_roots = cwd_roots
            .into_iter()
            .map(fs::canonicalize)
            .collect::<io::Result<Vec<_>>>()?;
        for p in &executables {
            let m = fs::metadata(p)?;
            if !m.is_file() || m.mode() & 0o6000 != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "regular non-setid executable required",
                ));
            }
        }
        if cwd_roots.iter().any(|p| !p.is_dir()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "directory roots required",
            ));
        }
        Ok(Self {
            executables,
            cwd_roots,
            max_invocation_bytes,
        })
    }
    fn validate(&self, r: &ShellRequest) -> Result<(), ShellError> {
        let mut size = r
            .executable
            .as_os_str()
            .len()
            .checked_add(r.cwd.as_os_str().len())
            .ok_or(ShellError::Policy("invocation size overflow"))?;
        for arg in &r.args {
            size = size
                .checked_add(arg.len())
                .and_then(|n| n.checked_add(1))
                .ok_or(ShellError::Policy("invocation size overflow"))?;
        }
        if size > self.max_invocation_bytes {
            return Err(ShellError::Policy(
                "invocation exceeds configured input limit",
            ));
        }
        let executable = fs::canonicalize(&r.executable)?;
        let cwd = fs::canonicalize(&r.cwd)?;
        if executable.as_os_str() != r.executable.as_os_str()
            || cwd.as_os_str() != r.cwd.as_os_str()
        {
            return Err(ShellError::Policy(
                "requested paths must already be canonical",
            ));
        }
        if !self.executables.contains(&executable)
            || !self.cwd_roots.iter().any(|root| cwd.starts_with(root))
        {
            return Err(ShellError::Policy(
                "executable or cwd outside configured scope",
            ));
        }
        let m = fs::metadata(&executable)?;
        if !m.is_file() || m.mode() & 0o6000 != 0 || !cwd.is_dir() {
            return Err(ShellError::Policy(
                "regular non-setid executable and directory required",
            ));
        }
        Ok(())
    }
    /// Spawn only after independent immutable approval/durable-start checks.
    /// Output is NDJSON: stream-tagged data byte arrays, then exit_code/signal.
    /// The shared byte limit includes every serialized frame and newline.
    /// Failure aborts the sink, kills the owned process group and reaps the
    /// direct child. Escaped descendants require a stronger OS sandbox.
    pub fn run(
        &self,
        request: &ShellRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ShellReport, ShellFailure> {
        let mut owned = None;
        let mut pid = None;
        let result = self.run_inner(request, context, output, &mut owned, &mut pid);
        match result {
            Ok(report) => Ok(report),
            Err(cause) => {
                output.abort();
                let cleanup_errors = owned.as_mut().map_or_else(Vec::new, OwnedChild::cleanup);
                Err(ShellFailure {
                    started: pid.is_some(),
                    pid,
                    cause,
                    cleanup_errors,
                })
            }
        }
    }
    fn run_inner(
        &self,
        r: &ShellRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
        owned: &mut Option<OwnedChild>,
        pid: &mut Option<u32>,
    ) -> Result<ShellReport, ShellError> {
        context.check()?;
        self.validate(r)?;
        context.check()?;
        let child = Command::new(&r.executable)
            .args(&r.args)
            .current_dir(&r.cwd)
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()?;
        *pid = Some(child.id());
        *owned = Some(OwnedChild::new(child)?);
        let child = owned
            .as_mut()
            .ok_or(ShellError::Policy("missing child ownership"))?;
        let mut stdout = child
            .child
            .stdout
            .take()
            .ok_or(ShellError::Policy("missing stdout pipe"))?;
        let mut stderr = child
            .child
            .stderr
            .take()
            .ok_or(ShellError::Policy("missing stderr pipe"))?;
        fcntl_setfl(
            &stdout,
            fcntl_getfl(&stdout).map_err(os)? | OFlags::NONBLOCK,
        )
        .map_err(os)?;
        fcntl_setfl(
            &stderr,
            fcntl_getfl(&stderr).map_err(os)? | OFlags::NONBLOCK,
        )
        .map_err(os)?;
        let mut capture = Capture {
            used: 0,
            stdout: 0,
            stderr: 0,
        };
        let mut out_done = false;
        let mut err_done = false;
        loop {
            context.check()?;
            let a = pump(
                &mut stdout,
                "stdout",
                &mut out_done,
                context,
                output,
                &mut capture,
            )?;
            let b = pump(
                &mut stderr,
                "stderr",
                &mut err_done,
                context,
                output,
                &mut capture,
            )?;
            let exited = child.exited()?;
            if exited {
                child.stop_group()?;
            }
            if exited && out_done && err_done {
                break;
            }
            if !a && !b {
                std::thread::sleep(context.remaining()?.min(Duration::from_millis(2)));
            }
        }
        let status = child.reap()?;
        context.check()?;
        capture.emit(
            serde_json::json!({"exit_code":status.code(),"signal":status.signal()}),
            context,
            output,
        )?;
        context.check()?;
        Ok(ShellReport {
            pid: child.child.id(),
            success: status.success(),
            exit_code: status.code(),
            signal: status.signal(),
            stdout_bytes: capture.stdout,
            stderr_bytes: capture.stderr,
            output_bytes: capture.used,
        })
    }
}

struct Capture {
    used: u64,
    stdout: u64,
    stderr: u64,
}
impl Capture {
    fn emit(
        &mut self,
        value: serde_json::Value,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<(), ShellError> {
        let mut frame = serde_json::to_vec(&value).map_err(io::Error::other)?;
        frame.push(b'\n');
        let total = self
            .used
            .checked_add(frame.len() as u64)
            .ok_or(ExecutionError::OutputLimit)?;
        if total > context.max_output_bytes() {
            return Err(ExecutionError::OutputLimit.into());
        }
        context.check()?;
        output.write_chunk(&frame)?;
        self.used = total;
        Ok(())
    }
}
fn pump<R: Read>(
    reader: &mut R,
    stream: &str,
    done: &mut bool,
    context: &ExecutionContext,
    output: &mut dyn OutputSink,
    capture: &mut Capture,
) -> Result<bool, ShellError> {
    if *done {
        return Ok(false);
    }
    context.check()?;
    let mut bytes = [0u8; 4096];
    match reader.read(&mut bytes) {
        Ok(0) => {
            *done = true;
            Ok(true)
        }
        Ok(n) => {
            context.check()?;
            capture.emit(
                serde_json::json!({"stream":stream,"data":&bytes[..n]}),
                context,
                output,
            )?;
            if stream == "stdout" {
                capture.stdout += n as u64;
            } else {
                capture.stderr += n as u64;
            }
            Ok(true)
        }
        Err(e)
            if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::Interrupted =>
        {
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

// Retain the unreaped leader until all group signals have been sent. try_wait
// would reap early and permit PID reuse before a later group signal.
struct OwnedChild {
    child: Child,
    pid: Pid,
    active: bool,
    group_stopped: bool,
    identity_lost: bool,
}
impl OwnedChild {
    fn new(mut child: Child) -> io::Result<Self> {
        let id = i32::try_from(child.id())
            .ok()
            .filter(|id| *id > 1)
            .and_then(Pid::from_raw);
        match id {
            Some(pid) => Ok(Self {
                child,
                pid,
                active: true,
                group_stopped: false,
                identity_lost: false,
            }),
            None => {
                let _ = child.kill();
                let _ = child.wait();
                Err(io::Error::other("invalid owned process-group ID"))
            }
        }
    }
    fn exited(&mut self) -> Result<bool, ShellError> {
        match waitid(
            WaitId::Pid(self.pid),
            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
        ) {
            Ok(status) => Ok(status.is_some()),
            Err(rustix::io::Errno::CHILD) => {
                self.identity_lost = true;
                Err(ShellError::Policy(
                    "owned child was reaped by another waiter",
                ))
            }
            Err(rustix::io::Errno::INTR) => Ok(false),
            Err(e) => Err(os(e)),
        }
    }
    fn stop_group(&mut self) -> Result<(), ShellError> {
        if self.group_stopped {
            return Ok(());
        }
        match kill_process_group(self.pid, Signal::KILL) {
            Ok(()) | Err(rustix::io::Errno::SRCH) => {
                self.group_stopped = true;
                Ok(())
            }
            Err(rustix::io::Errno::PERM) => {
                // Darwin can reject a zombie-only group. An exited leader alone
                // is insufficient evidence: a live, unsignalable member may
                // remain. Keep the leader unreaped while verifying the group.
                #[cfg(target_os = "macos")]
                {
                    if self.exited()? && macos_zombie_only_group(self.child.id())? {
                        self.group_stopped = true;
                        return Ok(());
                    }
                }
                Err(os(rustix::io::Errno::PERM))
            }
            Err(error) => Err(os(error)),
        }
    }
    fn reap(&mut self) -> io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.active = false;
        Ok(status)
    }
    fn cleanup(&mut self) -> Vec<String> {
        if !self.active {
            return Vec::new();
        }
        if self.identity_lost {
            self.active = false;
            return vec!["child ownership lost; no potentially reused PID was signalled".into()];
        }
        let mut errors = Vec::new();
        if let Err(e) = self.stop_group() {
            errors.push(e.to_string());
        }
        match self.child.kill() {
            Err(e) if e.kind() != io::ErrorKind::InvalidInput => {
                errors.push(e.to_string());
            }
            _ => {}
        }
        if let Err(e) = self.child.wait() {
            errors.push(e.to_string());
        }
        self.active = false;
        errors
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

// Read-only Darwin inspection for EPERM after a confirmed, unreaped leader
// exit. No fallback swallows errors or signals a PID/group after reaping it.
// Snapshot visibility and the trusted-OS/no-other-waiter assumptions are part
// of this candidate's boundary; this is not kernel-atomic group containment.
#[cfg(target_os = "macos")]
fn macos_zombie_only_group(leader: u32) -> Result<bool, ShellError> {
    const MAX_SNAPSHOT: usize = 1024 * 1024;
    let command = Command::new("/bin/ps")
        .args(["-A", "-o", "pid=,pgid=,stat="])
        .env_clear()
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut probe = MacPsProbe {
        child: command,
        active: true,
    };
    let mut stdout = probe
        .child
        .stdout
        .take()
        .ok_or(ShellError::Policy("missing Darwin process-table pipe"))?;
    fcntl_setfl(
        &stdout,
        fcntl_getfl(&stdout).map_err(os)? | OFlags::NONBLOCK,
    )
    .map_err(os)?;
    let start = std::time::Instant::now();
    let mut bytes = Vec::new();
    let mut eof = false;
    let mut status = None;
    loop {
        if start.elapsed() >= Duration::from_secs(1) {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Darwin process-group inspection exceeded cleanup budget",
            )
            .into());
        }
        let mut progressed = false;
        if !eof {
            let mut buffer = [0u8; 4096];
            match stdout.read(&mut buffer) {
                Ok(0) => {
                    eof = true;
                    progressed = true;
                }
                Ok(n) => {
                    if n > MAX_SNAPSHOT.saturating_sub(bytes.len()) {
                        return Err(ShellError::Policy(
                            "Darwin process snapshot exceeds byte bound",
                        ));
                    }
                    bytes
                        .try_reserve(n)
                        .map_err(|_| ExecutionError::AllocationFailed)?;
                    bytes.extend_from_slice(&buffer[..n]);
                    progressed = true;
                }
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock
                        || error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        if status.is_none() {
            match probe.child.try_wait() {
                Ok(Some(exit)) => {
                    probe.active = false;
                    status = Some(exit);
                }
                Ok(None) => (),
                Err(error) => {
                    if error.raw_os_error()
                        == io::Error::from(rustix::io::Errno::CHILD).raw_os_error()
                    {
                        probe.active = false;
                    }
                    return Err(error.into());
                }
            }
        }
        if eof && status.is_some() {
            break;
        }
        if !progressed {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    if !status.is_some_and(|exit| exit.success()) {
        return Err(ShellError::Policy("Darwin process-table inspection failed"));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| ShellError::Policy("non-UTF-8 Darwin process snapshot"))?;
    zombie_only_snapshot(text, leader)
}

// This trusted /bin/ps probe cannot spawn arbitrary descendants. Its guard only
// kills/reaps its own still-unreaped direct child; it never recurses into group
// inspection or signals a process after try_wait has reaped it.
#[cfg(target_os = "macos")]
struct MacPsProbe {
    child: Child,
    active: bool,
}
#[cfg(target_os = "macos")]
impl Drop for MacPsProbe {
    fn drop(&mut self) {
        if self.active {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.active = false;
        }
    }
}

#[cfg(any(target_os = "macos", test))]
fn zombie_only_snapshot(text: &str, leader: u32) -> Result<bool, ShellError> {
    let mut leader_seen = false;
    let mut all_zombies = true;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = line.split_whitespace();
        let invalid = || ShellError::Policy("malformed Darwin process snapshot");
        let pid = fields
            .next()
            .ok_or_else(invalid)?
            .parse::<u32>()
            .map_err(|_| invalid())?;
        let pgid = fields
            .next()
            .ok_or_else(invalid)?
            .parse::<u32>()
            .map_err(|_| invalid())?;
        let state = fields.next().ok_or_else(invalid)?;
        if fields.next().is_some() {
            return Err(invalid());
        }
        if pgid == leader {
            if !state.starts_with('Z') {
                all_zombies = false;
            }
            if pid == leader {
                leader_seen = state.starts_with('Z');
            }
        }
    }
    Ok(leader_seen && all_zombies)
}

#[cfg(test)]
mod group_teardown_tests {
    use super::*;

    #[test]
    fn zombie_only_group_requires_observed_zombie_leader() {
        assert!(zombie_only_snapshot("100 100 Z\n101 100 Z+\n7 7 S\n", 100).unwrap());
        assert!(!zombie_only_snapshot("101 100 Z\n", 100).unwrap());
        assert!(!zombie_only_snapshot("", 100).unwrap());
    }

    #[test]
    fn live_group_member_does_not_justify_ignoring_permission_denial() {
        assert!(!zombie_only_snapshot("100 100 Z\n101 100 S\n", 100).unwrap());
        assert!(!zombie_only_snapshot("100 100 R\n", 100).unwrap());
    }

    #[test]
    fn malformed_snapshot_fails_closed() {
        for snapshot in ["PID PGID STAT", "100 100", "100 100 Z extra", "x 100 Z"] {
            assert!(zombie_only_snapshot(snapshot, 100).is_err());
        }
    }

    #[test]
    fn unrelated_live_groups_do_not_change_zombie_group_observation() {
        assert!(zombie_only_snapshot("1 1 S\n100 100 Z\n200 200 R\n", 100).unwrap());
    }

    #[test]
    fn exited_unreaped_child_group_can_be_settled_then_reaped() {
        let child = Command::new("/usr/bin/true")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut child = OwnedChild::new(child).unwrap();
        let start = std::time::Instant::now();
        while !child.exited().unwrap() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "fixture did not exit"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        child.stop_group().unwrap();
        assert!(child.reap().unwrap().success());
        assert!(!child.active);
    }
}

/// Concrete shell.run backend bound to trusted executable/cwd configuration.
/// Configuration is not approval or a sandbox. Only the receipt-owned harness
/// path supports execution; legacy/generic entrypoints fail closed.
pub struct ShellRunTool {
    scope: ShellScope,
}
impl ShellRunTool {
    /// Bind a separately configured executable/cwd/input-size scope.
    pub fn new(scope: ShellScope) -> Self {
        Self { scope }
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArguments {
    executable: String,
    args: Vec<String>,
    cwd: String,
}
impl crate::registry::ToolImplementation for ShellRunTool {
    fn execute(&self, invocation: &crate::call::ToolInvocation) -> crate::result::ToolResult {
        crate::result::ToolResult::Inline(crate::result::InlineToolResult {
            call_id: invocation.call.call_id.clone(),
            is_error: true,
            content: serde_json::json!({"kind":"policy","message":"Receipt-owned shell execution required",
                "details":{"reason":"shell_receipt_required","started":false,"pid":null}}),
        })
    }
    fn execute_with_context(
        &self,
        invocation: &crate::call::ToolInvocation,
        _context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> crate::result::ToolResult {
        output.abort();
        self.execute(invocation)
    }
    fn execute_shell_run(
        &self,
        invocation: &crate::call::ToolInvocation,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ShellReport, ShellFailure> {
        let request = (|| {
            context.check()?;
            if invocation.call.tool_id != "shell.run" {
                return Err(ShellError::Policy("shell.run identity required"));
            }
            let args: ShellArguments = serde_json::from_value(serde_json::Value::Object(
                invocation.call.arguments.clone(),
            ))
            .map_err(|_| ShellError::Policy("invalid shell arguments"))?;
            ShellRequest::new(&args.executable, args.args, &args.cwd)
        })();
        match request {
            Ok(request) => self.scope.run(&request, context, output),
            Err(cause) => {
                output.abort();
                Err(ShellFailure {
                    started: false,
                    pid: None,
                    cause,
                    cleanup_errors: Vec::new(),
                })
            }
        }
    }
}
