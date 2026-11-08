//! Scoped W4 file-read primitive. This module does not grant call authorization.
//!
//! A trusted caller supplies the root capability; a future harness bridge must
//! also require the unchanged W3 StartedInvocation. No live dispatch is enabled.
//! All request components are normal relative names, opened with NOFOLLOW from
//! pinned directory descriptors. Symlinks (even in-root) and hardlinked files are
//! conservatively rejected. Root setup trusts the configured root and ancestors;
//! hostile mount changes and concurrent modifications of file contents are not
//! snapshot-isolation guarantees. Local regular-file syscalls may still block in
//! the kernel; cooperative deadline checks cannot promise universal interruption.

use crate::execution::{ExecutionContext, ExecutionError, OutputSink};
use rustix::fs::{Mode, OFlags, open, openat};
use std::{
    fmt,
    fs::File,
    io,
    os::unix::fs::{FileExt, MetadataExt},
    path::Path,
};

/// Internal read failure, not an extension of the tool-result wire error enum.
#[derive(Debug)]
pub enum ReadError {
    /// The path/range or actual object violates the scoped read contract.
    Policy(&'static str),
    /// Cancellation, deadline or sink rejection.
    Execution(ExecutionError),
    /// Operating-system failure; a partial prefix is not complete output.
    Io(io::Error),
}
impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(reason) => write!(f, "file read policy: {reason}"),
            Self::Execution(error) => write!(f, "file read execution: {error}"),
            Self::Io(error) => write!(f, "file read I/O: {error}"),
        }
    }
}
impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}
impl From<io::Error> for ReadError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<ExecutionError> for ReadError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}
fn os(error: rustix::io::Errno) -> ReadError {
    ReadError::Io(error.into())
}

/// Validated relative path and bounded byte range; not an execution permit.
#[derive(Debug)]
pub struct ReadRequest {
    path: String,
    offset: u64,
    length: u64,
}
impl ReadRequest {
    /// Reject absolute, empty, dot, parent, NUL and empty path components and
    /// range overflow. No normalization, defaults or shell expansion occurs.
    pub fn new(path: &str, offset: u64, length: u64) -> Result<Self, ReadError> {
        if path.is_empty()
            || path.contains('\0')
            || path.starts_with('/')
            || path
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(ReadError::Policy("normal relative path required"));
        }
        if length == 0 || offset.checked_add(length).is_none() {
            return Err(ReadError::Policy("invalid or overflowing byte range"));
        }
        Ok(Self {
            path: path.into(),
            offset,
            length,
        })
    }
}

/// Complete result of reading at most the requested range, including short EOF.
/// It does not claim a stable filesystem snapshot or complete-file retrieval.
#[derive(Debug, PartialEq, Eq)]
pub struct ReadReport {
    /// Actual payload bytes accepted by the sink.
    pub bytes_read: u64,
}

/// Pinned trusted directory capability; configuration is not human approval.
/// No caller-supplied path can replace this root during request resolution.
pub struct ReadScope {
    root: File,
    device: u64,
}
impl ReadScope {
    /// Configure a trusted local root. Resolve configured aliases once, then pin
    /// the directory. Request paths never use ambient path resolution afterward.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ReadError> {
        let root = std::fs::canonicalize(root)?;
        let file = File::from(
            open(
                &root,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(os)?,
        );
        let device = file.metadata()?.dev();
        Ok(Self { root: file, device })
    }
    /// Stream a validated range into the sink after independent durable-start
    /// authorization by the harness. The sink owns no retained store borrow.
    /// Failure aborts the sink so an accepted prefix cannot be finalized as a
    /// complete artifact. Only Ok permits the later inline/artifact finalization.
    pub fn read_into(
        &self,
        request: &ReadRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ReadReport, ReadError> {
        let result = self.read_inner(request, context, output);
        if result.is_err() {
            output.abort();
        }
        result
    }
    fn read_inner(
        &self,
        request: &ReadRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ReadReport, ReadError> {
        context.check()?;
        if request.length > context.max_output_bytes() {
            return Err(ReadError::Policy(
                "requested range exceeds approved output limit",
            ));
        }
        let mut directory = self.root.try_clone()?;
        let mut components = request.path.split('/').peekable();
        let file = loop {
            context.check()?;
            let name = components.next().ok_or(ReadError::Policy("empty path"))?;
            let last = components.peek().is_none();
            let mut flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
            if !last {
                flags |= OFlags::DIRECTORY;
            }
            let opened = File::from(openat(&directory, name, flags, Mode::empty()).map_err(os)?);
            let metadata = opened.metadata()?;
            if metadata.dev() != self.device {
                return Err(ReadError::Policy("cross-device read rejected"));
            }
            if last {
                if !metadata.is_file() || metadata.nlink() != 1 {
                    return Err(ReadError::Policy("single-link regular file required"));
                }
                break opened;
            }
            if !metadata.is_dir() {
                return Err(ReadError::Policy("directory component required"));
            }
            directory = opened;
        };
        context.check()?;
        let mut buffer = [0u8; 64 * 1024];
        let mut accepted = 0u64;
        while accepted < request.length {
            context.check()?;
            let count = (request.length - accepted).min(buffer.len() as u64) as usize;
            let n = match file.read_at(&mut buffer[..count], request.offset + accepted) {
                Ok(n) => n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            };
            context.check()?;
            if n == 0 {
                break;
            }
            output.write_chunk(&buffer[..n])?;
            accepted += n as u64;
        }
        context.check()?;
        Ok(ReadReport {
            bytes_read: accepted,
        })
    }
}
