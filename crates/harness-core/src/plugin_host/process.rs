//! Owned nonblocking child transport; no detached reader/writer threads.
use super::spawn::Child;
use super::{HostError, Limits, PluginConfig, spawn};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    io::{self, Read, Write},
    os::fd::AsRawFd,
    process::{ChildStderr, ChildStdin, ChildStdout},
    thread,
    time::{Duration, Instant},
};
use tool_registry::execution::CancellationToken;

unsafe extern "C" {
    fn fcntl(fd: std::ffi::c_int, cmd: std::ffi::c_int, ...) -> std::ffi::c_int;
    fn kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
}
fn nonblocking(fd: std::ffi::c_int) -> Result<(), HostError> {
    #[cfg(target_os = "linux")]
    const NONBLOCK: std::ffi::c_int = 0x800;
    #[cfg(target_os = "macos")]
    const NONBLOCK: std::ffi::c_int = 0x4;
    // SAFETY: fd is borrowed from a live owned pipe; F_GETFL takes no varargs.
    let flags = unsafe { fcntl(fd, 3) };
    if flags < 0 {
        return Err(HostError::io(io::Error::last_os_error()));
    }
    // SAFETY: F_SETFL accepts one int; unrelated descriptor flags are preserved.
    if unsafe { fcntl(fd, 4, flags | NONBLOCK) } < 0 {
        return Err(HostError::io(io::Error::last_os_error()));
    }
    Ok(())
}
#[derive(Serialize)]
struct Request<'a, T> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    params: T,
}
struct LimitedWriter {
    bytes: Vec<u8>,
    max: usize,
}
impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.max.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("request byte limit"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| io::Error::other("request allocation failed"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) struct Process {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    diagnostics: Vec<u8>,
    limits: Limits,
    next_id: u64,
    dead: bool,
}
impl Process {
    pub fn spawn(config: &PluginConfig, limits: Limits) -> Result<Self, HostError> {
        let (child, input, output, stderr) = spawn::spawn(config)?;
        let owned = Self {
            child,
            input: Some(input),
            output: Some(output),
            stderr: Some(stderr),
            diagnostics: Vec::new(),
            limits,
            next_id: 0,
            dead: false,
        };
        for fd in [
            owned.input.as_ref().unwrap().as_raw_fd(),
            owned.output.as_ref().unwrap().as_raw_fd(),
            owned.stderr.as_ref().unwrap().as_raw_fd(),
        ] {
            nonblocking(fd)?;
        }
        Ok(owned)
    }
    pub fn stop(&mut self) {
        if self.dead {
            return;
        }
        self.dead = true;
        self.input.take();
        // SAFETY: process_group(0) created a child-owned group. Negative PID
        // targets that group, not the daemon. SIGKILL is 9 on Linux and macOS.
        unsafe {
            kill(-(self.child.id() as i32), 9);
        }
        let _ = self.child.kill();
        loop {
            match self.child.wait() {
                Ok(_) => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    let note = b"; child wait failed: cleanup not established";
                    let keep = note.len().min(
                        self.limits
                            .diagnostic_bytes
                            .saturating_sub(self.diagnostics.len()),
                    );
                    self.diagnostics.extend_from_slice(&note[..keep]);
                    break;
                }
            }
        }
        self.output.take();
        self.stderr.take();
    }
    pub fn diagnostics(&self) -> &[u8] {
        &self.diagnostics
    }
    fn reject_pending_output(&mut self) -> Result<(), HostError> {
        let mut byte = [0u8; 1];
        loop {
            match self.output.as_mut().unwrap().read(&mut byte) {
                Ok(0) => return Err(HostError::new("plugin stdout closed")),
                Ok(_) => return Err(HostError::new("unsolicited plugin output")),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(e) => return Err(HostError::io(e)),
            }
        }
    }
    pub fn rpc<T: DeserializeOwned>(
        &mut self,
        method: &str,
        params: impl Serialize,
        deadline: Instant,
        cancel: &CancellationToken,
    ) -> Result<T, HostError> {
        let result = self.exchange(method, params, deadline, cancel);
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                // Preserve available startup diagnostics before stop closes pipes.
                let mut buffer = [0u8; 4096];
                loop {
                    let remaining = self
                        .limits
                        .diagnostic_bytes
                        .saturating_sub(self.diagnostics.len());
                    if remaining == 0 {
                        break;
                    }
                    let take = remaining.min(buffer.len());
                    let Some(stderr) = self.stderr.as_mut() else {
                        break;
                    };
                    match stderr.read(&mut buffer[..take]) {
                        Ok(0) => break,
                        Ok(n) => self.diagnostics.extend_from_slice(&buffer[..n]),
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
                self.stop();
                Err(HostError::new(&format!(
                    "{}; retained plugin stderr: {}",
                    error,
                    String::from_utf8_lossy(&self.diagnostics)
                )))
            }
        }
    }
    fn exchange<T: DeserializeOwned>(
        &mut self,
        method: &str,
        params: impl Serialize,
        outer_deadline: Instant,
        cancel: &CancellationToken,
    ) -> Result<T, HostError> {
        if self.dead {
            return Err(HostError::new("plugin is stopped"));
        }
        if cancel.is_cancelled() || Instant::now() >= outer_deadline {
            return Err(HostError::new("hook cancelled/expired before encoding"));
        }
        // A child must not send a frame between requests, including a partial
        // frame. Check the nonblocking pipe before issuing the next request.
        self.reject_pending_output()?;
        let deadline = Instant::now()
            .checked_add(self.limits.timeout)
            .ok_or_else(|| HostError::new("timeout overflow"))?
            .min(outer_deadline);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| HostError::new("request id overflow"))?;
        let mut writer = LimitedWriter {
            bytes: Vec::new(),
            max: self.limits.message_bytes - 1,
        };
        serde_json::to_writer(
            &mut writer,
            &Request {
                jsonrpc: "2.0",
                id: self.next_id,
                method,
                params,
            },
        )
        .map_err(|_| HostError::new("request encoding/byte limit"))?;
        writer.bytes.push(b'\n');
        self.reject_pending_output()?;
        let mut written = 0;
        let mut received = Vec::new();
        let mut stderr_total = 0usize;
        loop {
            if cancel.is_cancelled() {
                return Err(HostError::new("hook cancelled"));
            }
            if Instant::now() >= deadline {
                return Err(HostError::new("hook deadline"));
            }
            if written < writer.bytes.len() {
                match self.input.as_mut().unwrap().write(&writer.bytes[written..]) {
                    Ok(0) => return Err(HostError::new("plugin stdin closed")),
                    Ok(n) => written += n,
                    Err(e)
                        if matches!(
                            e.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                        ) => {}
                    Err(e) => return Err(HostError::io(e)),
                }
            }
            let mut buffer = [0u8; 4096];
            match self.stderr.as_mut().unwrap().read(&mut buffer) {
                Ok(n) => {
                    stderr_total = stderr_total.saturating_add(n);
                    let keep = n.min(
                        self.limits
                            .diagnostic_bytes
                            .saturating_sub(self.diagnostics.len()),
                    );
                    self.diagnostics.extend_from_slice(&buffer[..keep]);
                    if stderr_total > self.limits.diagnostic_bytes {
                        return Err(HostError::new("plugin diagnostic byte limit"));
                    }
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(HostError::io(e)),
            }
            match self.output.as_mut().unwrap().read(&mut buffer) {
                Ok(0) => return Err(HostError::new("plugin stdout closed")),
                Ok(n) => {
                    if n > self.limits.message_bytes.saturating_sub(received.len()) {
                        return Err(HostError::new("plugin response byte limit"));
                    }
                    received.extend_from_slice(&buffer[..n]);
                    if let Some(end) = received.iter().position(|b| *b == b'\n') {
                        if end + 1 != received.len() || written != writer.bytes.len() {
                            return Err(HostError::new("unsolicited or multiple response frames"));
                        }
                        let reply: super::protocol::Envelope<T> =
                            super::strict_json::decode(&received[..end])?;
                        if reply.jsonrpc != "2.0" || reply.id != self.next_id {
                            return Err(HostError::new("response identity mismatch"));
                        }
                        if cancel.is_cancelled() || Instant::now() >= deadline {
                            return Err(HostError::new("hook cancelled or expired"));
                        }
                        return Ok(reply.result);
                    }
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(e) => return Err(HostError::io(e)),
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        self.stop();
    }
}
