//! Scoped W4 read/list primitives. This module does not grant call authorization.
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

/// Validated directory path and maximum complete entry count; not approval.
/// "." names the pinned scope root; all other paths are normal relative names.
#[derive(Debug)]
pub struct ListRequest {
    path: String,
    max_entries: u64,
}
impl ListRequest {
    /// No normalization, recursive traversal, implicit defaults or pagination.
    /// Exceeding max_entries fails the whole listing rather than choosing an
    /// arbitrary filesystem-order prefix and claiming it is complete.
    pub fn new(path: &str, max_entries: u64) -> Result<Self, ReadError> {
        if max_entries == 0 {
            return Err(ReadError::Policy("max_entries must be positive"));
        }
        let path = if path == "." {
            path.to_owned()
        } else {
            ReadRequest::new(path, 0, 1)?.path
        };
        Ok(Self { path, max_entries })
    }
}

/// Complete bounded listing; valid only after the final context check succeeds.
#[derive(Debug, PartialEq, Eq)]
pub struct ListReport {
    /// Number of non-dot directory entries; directories/symlinks are names only.
    pub entries: u64,
    /// Exact serialized UTF-8 JSON byte count accepted by the sink.
    pub output_bytes: u64,
}

impl ReadScope {
    /// List a policy-approved directory after independent durable-start
    /// authorization. Output is JSON {"entries":[{"name_bytes":[...],
    /// "name_utf8":string_or_null},...]}, sorted lexicographically by raw name
    /// bytes. No entry target is opened or followed, even for symlinks.
    ///
    /// Count and serialized-byte limits bound collection before sorting. Any
    /// failure aborts the sink. Only Ok permits later artifact/inline completion.
    /// Like readdir, this observes a live directory, not a transactional snapshot;
    /// reproducible ordering is guaranteed for the names actually observed, not
    /// stable membership under concurrent filesystem changes.
    pub fn list_into(
        &self,
        request: &ListRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ListReport, ReadError> {
        let result = self.list_inner(request, context, output);
        if result.is_err() {
            output.abort();
        }
        result
    }

    fn list_directory(
        &self,
        request: &ListRequest,
        context: &ExecutionContext,
    ) -> Result<File, ReadError> {
        context.check()?;
        let mut directory = self.root.try_clone()?;
        if request.path != "." {
            for component in request.path.split('/') {
                context.check()?;
                let opened = File::from(
                    openat(
                        &directory,
                        component,
                        OFlags::RDONLY
                            | OFlags::DIRECTORY
                            | OFlags::NOFOLLOW
                            | OFlags::CLOEXEC
                            | OFlags::NONBLOCK,
                        Mode::empty(),
                    )
                    .map_err(os)?,
                );
                let metadata = opened.metadata()?;
                if !metadata.is_dir() || metadata.dev() != self.device {
                    return Err(ReadError::Policy("same-device scoped directory required"));
                }
                directory = opened;
            }
        }
        context.check()?;
        Ok(directory)
    }

    fn list_inner(
        &self,
        request: &ListRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ListReport, ReadError> {
        const PREFIX: &[u8] = b"{\"entries\":[";
        const SUFFIX: &[u8] = b"]}";
        context.check()?;
        let mut size = (PREFIX.len() + SUFFIX.len()) as u64;
        if size > context.max_output_bytes() {
            return Err(ExecutionError::OutputLimit.into());
        }
        let directory = self.list_directory(request, context)?;
        // read_from creates an independent cursor; do not enumerate via an
        // ambient pathname or share the root descriptor's directory offset.
        let mut iterator = rustix::fs::Dir::read_from(&directory).map_err(os)?;
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        loop {
            context.check()?;
            let item = match iterator.next() {
                Some(item) => item.map_err(os)?,
                None => break,
            };
            context.check()?;
            let name = item.file_name().to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            if entries.len() as u64 >= request.max_entries {
                return Err(ReadError::Policy("directory exceeds requested max_entries"));
            }
            let encoded = encode_list_name(name)?;
            let comma = u64::from(!entries.is_empty());
            size = size
                .checked_add(encoded.len() as u64)
                .and_then(|n| n.checked_add(comma))
                .ok_or(ExecutionError::OutputLimit)?;
            if size > context.max_output_bytes() {
                return Err(ExecutionError::OutputLimit.into());
            }
            entries
                .try_reserve(1)
                .map_err(|_| ExecutionError::AllocationFailed)?;
            entries.push((name.to_vec(), encoded));
        }
        context.check()?;
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        context.check()?;
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(ReadError::Policy("duplicate observed directory name"));
        }
        output.write_chunk(PREFIX)?;
        for (index, (_, encoded)) in entries.iter().enumerate() {
            context.check()?;
            if index != 0 {
                output.write_chunk(b",")?;
            }
            context.check()?;
            output.write_chunk(encoded)?;
        }
        context.check()?;
        output.write_chunk(SUFFIX)?;
        context.check()?;
        Ok(ListReport {
            entries: entries.len() as u64,
            output_bytes: size,
        })
    }
}

// Pure representation boundary, independently testable on filesystems which
// cannot construct every valid Unix byte-string filename.
fn encode_list_name(name: &[u8]) -> Result<Vec<u8>, ReadError> {
    serde_json::to_vec(&serde_json::json!({
        "name_bytes": name,
        "name_utf8": std::str::from_utf8(name).ok(),
    }))
    .map_err(|error| ReadError::Io(io::Error::other(error)))
}

#[cfg(test)]
mod list_name_encoding_tests {
    use super::encode_list_name;

    #[test]
    fn non_utf8_name_bytes_are_preserved() {
        let raw = [b'x', 0xff];
        let bytes = encode_list_name(&raw).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let decoded: Vec<u8> = serde_json::from_value(value["name_bytes"].clone()).unwrap();
        assert_eq!(decoded, raw);
        assert!(value["name_utf8"].is_null());
    }

    #[test]
    fn escaped_name_encoding_is_lossless() {
        let name = "a\"\n\\";
        let bytes = encode_list_name(name.as_bytes()).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let decoded: Vec<u8> = serde_json::from_value(value["name_bytes"].clone()).unwrap();
        assert_eq!(decoded, name.as_bytes());
        assert_eq!(value["name_utf8"].as_str(), Some(name));
    }
}
