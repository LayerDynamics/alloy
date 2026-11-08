//! Scoped W4 filesystem primitives. This module does not grant call authorization.
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

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod scoped_write {
    use super::*;
    use rustix::fs::{AtFlags, RenameFlags, renameat_with, statat, unlinkat};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Exact UTF-8 contents and unchanged relative destination; not approval.
    #[derive(Debug)]
    pub struct WriteRequest {
        path: String,
        content: String,
    }
    impl WriteRequest {
        /// This candidate creates new files only. Internal staging names are
        /// reserved; existing destinations are never replaced.
        pub fn new(path: &str, content: String) -> Result<Self, ReadError> {
            let path = ReadRequest::new(path, 0, 1)?.path;
            if path
                .split('/')
                .any(|part| part.starts_with(".alloy-write-"))
            {
                return Err(ReadError::Policy("reserved staging path component"));
            }
            Ok(Self { path, content })
        }
    }

    /// Publication state, distinct from whether staging effects were attempted.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum WritePublication {
        /// Destination was not published; temporary staging may have occurred.
        NotPublished,
        /// The publication syscall failed; do not infer no effect or retry.
        Unknown,
        /// Destination was published, even if result delivery later failed.
        Published,
    }

    /// Internal failure that preserves effect status and cleanup diagnostics.
    #[derive(Debug)]
    pub struct WriteFailure {
        /// Destination publication status, not a new tool-result wire category.
        pub publication: WritePublication,
        /// Original failure.
        pub cause: ReadError,
        /// Cleanup failed; an uncommitted staging file may remain.
        pub cleanup_error: Option<io::Error>,
    }
    impl fmt::Display for WriteFailure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "file write {:?}: {}", self.publication, self.cause)?;
            if let Some(error) = &self.cleanup_error {
                write!(f, "; staging cleanup failed: {error}")?;
            }
            Ok(())
        }
    }
    impl std::error::Error for WriteFailure {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.cause)
        }
    }

    /// Confirmed publication of an exact UTF-8 payload, not crash durability.
    #[derive(Debug, PartialEq, Eq)]
    pub struct WriteReport {
        /// Exact relative destination supplied by the caller.
        pub path: String,
        /// Content byte count, not Unicode scalar count.
        pub bytes_written: u64,
    }

    /// Separately configured writable root and input ceiling, not execution
    /// authorization. Read roots are not implicitly promoted to writable roots.
    ///
    /// Requires a local trusted directory tree: untrusted OS processes must not
    /// be able to replace staging names or mutate ancestors during publication.
    /// Scope handles pin directory objects, not their current ambient locations.
    /// A future harness bridge must independently require immutable W3 approval
    /// and StartedInvocation before calling this primitive. Live dispatch stays
    /// disabled. Cancellation is cooperative around potentially blocking I/O.
    pub struct WriteScope {
        scope: ReadScope,
        max_content_bytes: u64,
    }
    impl WriteScope {
        /// Trusted daemon configuration, separate from max_output_bytes.
        pub fn open(root: impl AsRef<Path>, max_content_bytes: u64) -> Result<Self, ReadError> {
            if max_content_bytes == 0 {
                return Err(ExecutionError::InvalidLimits.into());
            }
            Ok(Self {
                scope: ReadScope::open(root)?,
                max_content_bytes,
            })
        }

        /// Stage, sync and atomically publish a new file with no replacement.
        /// No automatic parent creation, overwrite, chmod of existing objects,
        /// shell expansion, or fallback to an overwriting rename is permitted.
        /// Output JSON identifies path/bytes_written, not an artifact reference.
        /// A Published/Unknown failure must not trigger an automatic retry.
        /// Successful file sync does not promise directory/power-loss durability.
        pub fn write_into(
            &self,
            request: &WriteRequest,
            context: &ExecutionContext,
            output: &mut dyn OutputSink,
        ) -> Result<WriteReport, WriteFailure> {
            let mut publication = WritePublication::NotPublished;
            let mut staged = None;
            let result = self.write_inner(request, context, output, &mut publication, &mut staged);
            let cleanup_error = staged.as_mut().and_then(|stage| stage.cleanup().err());
            match result {
                Ok(report) => Ok(report),
                Err(cause) => {
                    output.abort();
                    Err(WriteFailure {
                        publication,
                        cause,
                        cleanup_error,
                    })
                }
            }
        }

        fn write_inner(
            &self,
            request: &WriteRequest,
            context: &ExecutionContext,
            output: &mut dyn OutputSink,
            publication: &mut WritePublication,
            staged: &mut Option<StagedWrite>,
        ) -> Result<WriteReport, ReadError> {
            context.check()?;
            let bytes = request.content.as_bytes();
            if bytes.len() as u64 > self.max_content_bytes {
                return Err(ReadError::Policy(
                    "content exceeds configured write-input limit",
                ));
            }
            let encoded = serde_json::to_vec(&serde_json::json!({
                "path": &request.path, "bytes_written": bytes.len() as u64,
            }))
            .map_err(|error| ReadError::Io(io::Error::other(error)))?;
            if encoded.len() as u64 > context.max_output_bytes() {
                return Err(ExecutionError::OutputLimit.into());
            }
            let (parent, name) = request
                .path
                .rsplit_once('/')
                .unwrap_or((".", &request.path));
            let parent = self
                .scope
                .list_directory(&ListRequest::new(parent, 1)?, context)?;
            match statat(&parent, name, AtFlags::SYMLINK_NOFOLLOW) {
                Ok(_) => {
                    return Err(ReadError::Policy(
                        "destination already exists; overwrite disabled",
                    ));
                }
                Err(rustix::io::Errno::NOENT) => (),
                Err(error) => return Err(os(error)),
            }
            context.check()?;
            *staged = Some(StagedWrite::new(parent, context)?);
            let stage = staged
                .as_mut()
                .ok_or(ReadError::Policy("missing staging ownership"))?;
            write_contents(&mut stage.file, bytes, context)?;
            context.check()?;
            stage.file.sync_all()?;
            context.check()?;
            let held = rustix::fs::fstat(&stage.file).map_err(os)?;
            let named = statat(
                &stage.parent,
                stage.name.as_str(),
                AtFlags::SYMLINK_NOFOLLOW,
            )
            .map_err(os)?;
            if named.st_dev != held.st_dev || named.st_ino != held.st_ino || named.st_nlink != 1 {
                return Err(ReadError::Policy("staging identity changed"));
            }
            context.check()?;
            stage.publish(name, context, publication)?;
            context.check()?;
            output.write_chunk(&encoded)?;
            context.check()?;
            Ok(WriteReport {
                path: request.path.clone(),
                bytes_written: bytes.len() as u64,
            })
        }
    }

    struct StagedWrite {
        parent: File,
        file: File,
        name: String,
        armed: bool,
    }
    impl StagedWrite {
        fn new(parent: File, context: &ExecutionContext) -> Result<Self, ReadError> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            for _ in 0..64 {
                context.check()?;
                let name = format!(
                    ".alloy-write-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                );
                match openat(
                    &parent,
                    name.as_str(),
                    OFlags::WRONLY
                        | OFlags::CREATE
                        | OFlags::EXCL
                        | OFlags::NOFOLLOW
                        | OFlags::CLOEXEC,
                    Mode::RUSR | Mode::WUSR,
                ) {
                    Ok(fd) => {
                        return Ok(Self {
                            parent,
                            file: File::from(fd),
                            name,
                            armed: true,
                        });
                    }
                    Err(rustix::io::Errno::EXIST) => continue,
                    Err(error) => return Err(os(error)),
                }
            }
            Err(ReadError::Policy("bounded staging-name retries exhausted"))
        }
        fn publish(
            &mut self,
            name: &str,
            context: &ExecutionContext,
            publication: &mut WritePublication,
        ) -> Result<(), ReadError> {
            context.check()?;
            *publication = WritePublication::Unknown;
            match renameat_with(
                &self.parent,
                self.name.as_str(),
                &self.parent,
                name,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => {
                    self.armed = false;
                    *publication = WritePublication::Published;
                    Ok(())
                }
                Err(rustix::io::Errno::EXIST) => {
                    *publication = WritePublication::NotPublished;
                    Err(ReadError::Policy(
                        "destination appeared; overwrite disabled",
                    ))
                }
                Err(error) => Err(os(error)),
            }
        }
        fn cleanup(&mut self) -> io::Result<()> {
            if !self.armed {
                return Ok(());
            }
            match unlinkat(&self.parent, self.name.as_str(), AtFlags::empty()) {
                Ok(()) | Err(rustix::io::Errno::NOENT) => {
                    self.armed = false;
                    Ok(())
                }
                Err(error) => Err(error.into()),
            }
        }
    }
    impl Drop for StagedWrite {
        fn drop(&mut self) {
            let _ = self.cleanup();
        }
    }

    fn write_contents<W: io::Write>(
        writer: &mut W,
        bytes: &[u8],
        context: &ExecutionContext,
    ) -> Result<(), ReadError> {
        let mut offset = 0;
        while offset < bytes.len() {
            context.check()?;
            let end = offset + (bytes.len() - offset).min(64 * 1024);
            match writer.write(&bytes[offset..end]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "staging write returned zero",
                    )
                    .into());
                }
                Ok(n) => offset += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        context.check()?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{
            definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
            execution::CancellationToken,
        };
        fn context(token: CancellationToken) -> ExecutionContext {
            ExecutionContext::new(
                &ToolPolicy {
                    effect: ToolEffect::Write,
                    approval: ApprovalRequirement::Policy,
                    timeout_ms: 60000,
                    max_output_bytes: 4096,
                },
                60000,
                4096,
                token,
            )
            .unwrap()
        }
        struct CancelWriter {
            token: CancellationToken,
            calls: usize,
        }
        impl io::Write for CancelWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                self.token.cancel();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        #[test]
        fn write_cancellation_stops_between_chunks() {
            let token = CancellationToken::default();
            let c = context(token.clone());
            let mut writer = CancelWriter { token, calls: 0 };
            assert!(write_contents(&mut writer, &vec![1; 150000], &c).is_err());
            assert_eq!(writer.calls, 1);
        }
        struct ShortWriter {
            calls: usize,
            bytes: Vec<u8>,
        }
        impl io::Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                if self.calls == 1 {
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                let n = bytes.len().min(3);
                self.bytes.extend_from_slice(&bytes[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        #[test]
        fn short_and_interrupted_writes_preserve_exact_content() {
            let c = context(CancellationToken::default());
            let mut writer = ShortWriter {
                calls: 0,
                bytes: Vec::new(),
            };
            write_contents(&mut writer, b"exact bytes", &c).unwrap();
            assert_eq!(writer.bytes, b"exact bytes");
        }

        struct TempDirectory(std::path::PathBuf);
        impl TempDirectory {
            fn new() -> Self {
                static NEXT: AtomicU64 = AtomicU64::new(0);
                loop {
                    let path = std::env::temp_dir().join(format!(
                        "alloy-w4-write-unit-{}-{}",
                        std::process::id(),
                        NEXT.fetch_add(1, Ordering::Relaxed)
                    ));
                    match std::fs::create_dir(&path) {
                        Ok(()) => return Self(path),
                        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                        Err(error) => panic!("fixture: {error}"),
                    }
                }
            }
        }
        impl Drop for TempDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        #[test]
        fn publication_does_not_replace_destination_created_after_staging() {
            let dir = TempDirectory::new();
            let c = context(CancellationToken::default());
            let mut stage = StagedWrite::new(File::open(&dir.0).unwrap(), &c).unwrap();
            io::Write::write_all(&mut stage.file, b"staged").unwrap();
            std::fs::write(dir.0.join("target"), b"late destination").unwrap();
            let mut publication = WritePublication::NotPublished;
            assert!(stage.publish("target", &c, &mut publication).is_err());
            assert_eq!(publication, WritePublication::NotPublished);
            assert_eq!(
                std::fs::read(dir.0.join("target")).unwrap(),
                b"late destination"
            );
            stage.cleanup().unwrap();
            assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
        }
        #[test]
        fn cancellation_before_publication_allows_staging_cleanup() {
            let dir = TempDirectory::new();
            let token = CancellationToken::default();
            let c = context(token.clone());
            let mut stage = StagedWrite::new(File::open(&dir.0).unwrap(), &c).unwrap();
            io::Write::write_all(&mut stage.file, b"partial staging").unwrap();
            token.cancel();
            let mut publication = WritePublication::NotPublished;
            assert!(stage.publish("target", &c, &mut publication).is_err());
            assert_eq!(publication, WritePublication::NotPublished);
            stage.cleanup().unwrap();
            assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
        }
    }
}

/// Linux/macOS scoped create-only write primitives; no live dispatch enabled.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use scoped_write::{WriteFailure, WritePublication, WriteReport, WriteRequest, WriteScope};

/// Concrete fs.read implementation bound to a daemon-configured pinned scope.
/// This object is not authorization: register it through the receipt executor.
/// Its legacy entrypoint fails closed. This W4-02.01 candidate supports complete
/// bounded UTF-8 inline results only; large/binary output requires the pending
/// W4-02.02 artifact adapter and is explicitly rejected rather than truncated.
pub struct FileReadTool {
    scope: ReadScope,
    max_inline_bytes: usize,
}
impl FileReadTool {
    /// Configure a scope and positive serialized-inline ceiling at trusted setup.
    /// The request cannot supply or replace either value. No default is invented.
    pub fn new(scope: ReadScope, max_inline_bytes: usize) -> Result<Self, ReadError> {
        if max_inline_bytes == 0 {
            return Err(ExecutionError::InvalidLimits.into());
        }
        Ok(Self {
            scope,
            max_inline_bytes,
        })
    }
    fn read_content(
        &self,
        invocation: &crate::call::ToolInvocation,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<serde_json::Value, ReadAdapterFailure> {
        context
            .check()
            .map_err(|e| read_adapter_execution(e, false))?;
        if invocation.call.tool_id != "fs.read" {
            return Err(ReadAdapterFailure {
                kind: crate::result::ToolErrorKind::Protocol,
                reason: "tool_id_mismatch",
                attempted: false,
            });
        }
        let args: FileReadArguments =
            serde_json::from_value(serde_json::Value::Object(invocation.call.arguments.clone()))
                .map_err(|_| ReadAdapterFailure {
                    kind: crate::result::ToolErrorKind::Protocol,
                    reason: "invalid_read_arguments",
                    attempted: false,
                })?;
        let request = ReadRequest::new(&args.path, args.offset, args.length)
            .map_err(|e| read_adapter_failure(e, false))?;
        let maximum = context.max_output_bytes().min(self.max_inline_bytes as u64);
        if request.length > maximum {
            return Err(ReadAdapterFailure {
                kind: crate::result::ToolErrorKind::Execution,
                reason: "artifact_output_required",
                attempted: false,
            });
        }
        let mut buffer = crate::execution::BoundedOutput::new(context, maximum as usize)
            .map_err(|e| read_adapter_execution(e, false))?;
        let report = self
            .scope
            .read_into(&request, context, &mut buffer)
            .map_err(|e| read_adapter_failure(e, true))?;
        let bytes = buffer
            .finish()
            .map_err(|e| read_adapter_execution(e, true))?;
        let text = String::from_utf8(bytes).map_err(|_| ReadAdapterFailure {
            kind: crate::result::ToolErrorKind::Execution,
            reason: "artifact_output_required",
            attempted: true,
        })?;
        context
            .check()
            .map_err(|e| read_adapter_execution(e, true))?;
        let content = serde_json::json!({"text": text, "bytes_read": report.bytes_read,
            "offset": request.offset});
        let mut writer = InlineReadOutput {
            output,
            context,
            maximum,
            accepted: 0,
            failure: None,
        };
        if serde_json::to_writer(&mut writer, &content).is_err() {
            return Err(read_adapter_execution(
                writer.failure.unwrap_or(ExecutionError::IncompleteOutput),
                true,
            ));
        }
        context
            .check()
            .map_err(|e| read_adapter_execution(e, true))?;
        Ok(content)
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileReadArguments {
    path: String,
    offset: u64,
    length: u64,
}
struct ReadAdapterFailure {
    kind: crate::result::ToolErrorKind,
    reason: &'static str,
    attempted: bool,
}
fn read_adapter_execution(error: ExecutionError, attempted: bool) -> ReadAdapterFailure {
    use crate::result::ToolErrorKind;
    let (kind, reason) = match error {
        ExecutionError::Cancelled => (ToolErrorKind::Execution, "cancelled"),
        ExecutionError::DeadlineExceeded => (ToolErrorKind::Timeout, "deadline_exceeded"),
        ExecutionError::OutputLimit => (ToolErrorKind::Execution, "output_limit"),
        ExecutionError::InvalidLimits => (ToolErrorKind::Execution, "invalid_execution_limits"),
        ExecutionError::AllocationFailed => (ToolErrorKind::Execution, "allocation_failed"),
        ExecutionError::IncompleteOutput => (ToolErrorKind::Execution, "incomplete_output"),
    };
    ReadAdapterFailure {
        kind,
        reason,
        attempted,
    }
}
fn read_adapter_failure(error: ReadError, attempted: bool) -> ReadAdapterFailure {
    match error {
        ReadError::Execution(error) => read_adapter_execution(error, attempted),
        ReadError::Policy(_) => ReadAdapterFailure {
            kind: crate::result::ToolErrorKind::Policy,
            reason: "read_scope_or_range_rejected",
            attempted,
        },
        ReadError::Io(_) => ReadAdapterFailure {
            kind: crate::result::ToolErrorKind::Execution,
            reason: "file_read_failed",
            attempted,
        },
    }
}
fn read_adapter_error(call: &str, failure: ReadAdapterFailure) -> crate::result::ToolResult {
    crate::result::ToolResult::Inline(crate::result::InlineToolResult {
        call_id: call.into(),
        is_error: true,
        content: serde_json::json!({"kind":failure.kind,"message":"Scoped file read did not produce a complete inline result",
            "details":{"reason":failure.reason,"scope_operation_attempted":failure.attempted,"output_complete":false}}),
    })
}
impl crate::registry::ToolImplementation for FileReadTool {
    fn execute_file_read(
        &self,
        invocation: &crate::call::ToolInvocation,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<ReadReport, ReadError> {
        let result = (|| {
            context.check()?;
            if invocation.call.tool_id != "fs.read" {
                return Err(ReadError::Policy("fs.read identity required"));
            }
            let args: FileReadArguments = serde_json::from_value(serde_json::Value::Object(
                invocation.call.arguments.clone(),
            ))
            .map_err(|_| ReadError::Policy("invalid fs.read arguments"))?;
            let request = ReadRequest::new(&args.path, args.offset, args.length)?;
            self.scope.read_into(&request, context, output)
        })();
        if result.is_err() {
            output.abort();
        }
        result
    }

    fn execute(&self, invocation: &crate::call::ToolInvocation) -> crate::result::ToolResult {
        read_adapter_error(
            &invocation.call.call_id,
            ReadAdapterFailure {
                kind: crate::result::ToolErrorKind::Policy,
                reason: "durable_context_required",
                attempted: false,
            },
        )
    }
    fn execute_with_context(
        &self,
        invocation: &crate::call::ToolInvocation,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> crate::result::ToolResult {
        match self.read_content(invocation, context, output) {
            Ok(content) => crate::result::ToolResult::Inline(crate::result::InlineToolResult {
                call_id: invocation.call.call_id.clone(),
                content,
                is_error: false,
            }),
            Err(failure) => {
                output.abort();
                read_adapter_error(&invocation.call.call_id, failure)
            }
        }
    }
}
struct InlineReadOutput<'a> {
    output: &'a mut dyn OutputSink,
    context: &'a ExecutionContext,
    maximum: u64,
    accepted: u64,
    failure: Option<ExecutionError>,
}
impl InlineReadOutput<'_> {
    fn fail(&mut self, error: ExecutionError) -> io::Error {
        self.failure = Some(error);
        self.output.abort();
        io::Error::other("inline read output failed")
    }
}
impl io::Write for InlineReadOutput<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(error) = self.failure {
            return Err(self.fail(error));
        }
        if let Err(error) = self.context.check() {
            return Err(self.fail(error));
        }
        if bytes.len() as u64 > self.maximum.saturating_sub(self.accepted) {
            return Err(self.fail(ExecutionError::OutputLimit));
        }
        if let Err(error) = self.output.write_chunk(bytes) {
            return Err(self.fail(error));
        }
        self.accepted += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
