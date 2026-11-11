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
//! An isolated native owner retains the target through group cleanup. Blocking spawn,
//! kernel I/O, sink calls and final wait are not universally interruptible.

use crate::execution::{ExecutionContext, ExecutionError, OutputSink};
use rustix::fs::{OFlags, fcntl_getfl, fcntl_setfl};
use std::{
    fmt, fs,
    io::{self, Read},
    os::unix::{fs::MetadataExt, process::ExitStatusExt},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
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
                    started: pid.is_some()
                        || matches!(&cause, ShellError::Io(error) if error.get_ref().is_some_and(|source|source.is::<alloy_process_owner::spawn::CommittedLaunchError>())),
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
        let mut command = Command::new(&r.executable);
        command
            .args(&r.args)
            .current_dir(&r.cwd)
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let helper_path = alloy_process_owner::spawn::default_helper()?;
        let helper = alloy_process_owner::spawn::prepare_helper(
            &[alloy_process_owner::spawn::AuditConfig {
                program: r.executable.clone(),
                write_trees: vec![],
            }],
            Some(&helper_path),
            std::time::Instant::now() + context.remaining()?,
            &|| context.check().is_err(),
        )?;
        let (child, stdin, mut stdout, mut stderr) = alloy_process_owner::spawn::spawn_shell(
            command,
            helper.as_deref(),
            std::time::Instant::now() + context.remaining()?,
            &|| context.check().is_err(),
        )?;
        drop(stdin);
        *pid = Some(child.id());
        *owned = Some(OwnedChild::new(child));
        let child = owned
            .as_mut()
            .ok_or(ShellError::Policy("missing child ownership"))?;
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

// Only the isolated native owner can signal or reap the target/group.
struct OwnedChild {
    child: alloy_process_owner::spawn::Child,
    status: Option<ExitStatus>,
    active: bool,
}
impl OwnedChild {
    fn new(child: alloy_process_owner::spawn::Child) -> Self {
        Self {
            child,
            status: None,
            active: true,
        }
    }
    fn exited(&mut self) -> Result<bool, ShellError> {
        self.status = self.child.try_wait()?;
        Ok(self.status.is_some())
    }
    fn reap(&mut self) -> io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.active = false;
        Ok(status)
    }
    fn cleanup(&mut self) -> Vec<String> {
        if !self.active {
            return vec![];
        }
        let mut errors = vec![];
        if let Err(e) = self.child.kill() {
            errors.push(e.to_string());
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
