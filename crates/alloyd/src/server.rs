//! Unix-domain NDJSON accept loop.

use crate::rpc::{RpcError, encode_line, parse_request_line};
use crate::service::DaemonService;
use engine::api::Engine;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

/// Daemon bind configuration.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Filesystem path for the UDS listener.
    pub socket_path: PathBuf,
    /// SQLite session database path.
    pub db_path: PathBuf,
}

impl DaemonConfig {
    /// Default paths under `~/.alloy/` when HOME is available.
    pub fn default_paths() -> Result<Self, RpcError> {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| RpcError::internal("HOME is required to derive ~/.alloy paths"))?;
        let root = PathBuf::from(home).join(".alloy");
        Ok(Self {
            socket_path: root.join("alloyd.sock"),
            db_path: root.join("sessions.sqlite"),
        })
    }
}

/// Bind a Unix listener at `socket_path`, replacing a stale socket file only when safe.
pub fn bind_listener(socket_path: &Path) -> Result<UnixListener, RpcError> {
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| RpcError::internal(format!("create socket parent: {error}")))?;
    }
    if socket_path.exists() {
        match UnixStream::connect(socket_path) {
            Ok(_live) => {
                return Err(RpcError::internal(format!(
                    "socket {} is already in use by a live alloyd",
                    socket_path.display()
                )));
            }
            Err(_) => {
                std::fs::remove_file(socket_path)
                    .map_err(|error| RpcError::internal(format!("remove stale socket: {error}")))?;
            }
        }
    }
    let listener = UnixListener::bind(socket_path)
        .map_err(|error| RpcError::internal(format!("bind {}: {error}", socket_path.display())))?;
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
        .map_err(|error| RpcError::internal(format!("chmod socket: {error}")))?;
    Ok(listener)
}

/// Accept connections forever, spawning one thread per client.
pub fn serve_listener<E: Engine + Send + Sync + 'static>(
    listener: UnixListener,
    service: Arc<DaemonService<E>>,
) -> Result<(), RpcError> {
    for connection in listener.incoming() {
        let stream = match connection {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("alloyd accept error (continuing): {error}");
                continue;
            }
        };
        let service = Arc::clone(&service);
        thread::spawn(move || {
            if let Err(error) = handle_connection(stream, &service) {
                eprintln!("alloyd connection error: {}", error.message);
            }
        });
    }
    Ok(())
}

/// Serve a single already-connected stream (tests + accept loop).
pub fn handle_connection<E: Engine>(
    stream: UnixStream,
    service: &DaemonService<E>,
) -> Result<(), RpcError> {
    if let Some(limits) = service.rpc_transport_limits() {
        stream
            .set_write_timeout(Some(limits.write_timeout()))
            .map_err(|error| RpcError::internal(format!("set socket write timeout: {error}")))?;
    }
    let reader_stream = stream
        .try_clone()
        .map_err(|error| RpcError::internal(format!("clone stream: {error}")))?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = ConnectionWriter::new(stream);
    let mut line = String::new();

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| RpcError::internal(format!("read: {error}")))?;
        if read == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        match parse_request_line(&line) {
            Ok(request) => {
                if let Err(error) = service.handle(request, &mut writer) {
                    // Best-effort error frame when dispatch itself fails mid-write.
                    let _ = write_fatal(&mut writer, error);
                    break;
                }
            }
            Err(error) => {
                let response = crate::rpc::RpcResponse::error(serde_json::Value::Null, error);
                let encoded = encode_line(&response)?;
                writer
                    .write_all(encoded.as_bytes())
                    .map_err(|error| RpcError::internal(format!("write: {error}")))?;
                writer
                    .flush()
                    .map_err(|error| RpcError::internal(format!("flush: {error}")))?;
            }
        }
    }
    Ok(())
}

fn write_fatal(writer: &mut impl Write, error: RpcError) -> Result<(), RpcError> {
    let response = crate::rpc::RpcResponse::error(serde_json::Value::Null, error);
    let line = encode_line(&response)?;
    writer
        .write_all(line.as_bytes())
        .map_err(|error| RpcError::internal(format!("write fatal: {error}")))?;
    writer
        .flush()
        .map_err(|error| RpcError::internal(format!("flush fatal: {error}")))?;
    Ok(())
}

// W2_RPC_CONNECTION_WRITE_OWNERSHIP_V1
// A partially written NDJSON frame cannot be followed by an error response:
// concatenating them would create a corrupt line. No output buffer is owned.
struct ConnectionWriter<W> {
    inner: W,
    poisoned: bool,
}
impl<W> ConnectionWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            poisoned: false,
        }
    }
}
impl<W: Write> Write for ConnectionWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        use std::io::{Error, ErrorKind};
        if self.poisoned {
            return Err(Error::new(
                ErrorKind::BrokenPipe,
                "connection output is poisoned",
            ));
        }
        if bytes.is_empty() {
            return Ok(0);
        }
        match self.inner.write(bytes) {
            Ok(0) => {
                self.poisoned = true;
                Err(Error::new(
                    ErrorKind::WriteZero,
                    "connection made no write progress",
                ))
            }
            Ok(count) if count <= bytes.len() => Ok(count),
            Ok(_) => {
                self.poisoned = true;
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "connection returned an invalid byte count",
                ))
            }
            Err(error) => {
                // write_all retries Interrupted; no output was reported by
                // this call. Other failures make subsequent framing unsafe.
                if error.kind() != ErrorKind::Interrupted {
                    self.poisoned = true;
                }
                Err(error)
            }
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if self.poisoned {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "connection output is poisoned",
            ));
        }
        match self.inner.flush() {
            Ok(()) => Ok(()),
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tool_output_connection_writer_tests {
    use super::*;
    use std::io::{self, ErrorKind};
    struct PartialFailure {
        bytes: Vec<u8>,
        allowance: usize,
    }
    impl Write for PartialFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.allowance == 0 {
                return Err(io::Error::new(ErrorKind::BrokenPipe, "injected disconnect"));
            }
            let count = self.allowance.min(bytes.len());
            self.bytes.extend_from_slice(&bytes[..count]);
            self.allowance -= count;
            Ok(count)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn partial_frame_failure_prevents_appending_a_fatal_response() {
        let mut writer = ConnectionWriter::new(PartialFailure {
            bytes: Vec::new(),
            allowance: 3,
        });
        assert!(writer.write_all(b"longer-frame").is_err());
        let before = writer.inner.bytes.clone();
        assert_eq!(before, b"lon");
        assert!(
            write_fatal(
                &mut writer,
                RpcError::internal("cannot append an error frame")
            )
            .is_err()
        );
        assert_eq!(writer.inner.bytes, before);
        assert!(writer.poisoned);
    }
    #[test]
    fn interrupted_write_is_retried_without_poisoning() {
        struct Interrupted {
            first: bool,
            bytes: Vec<u8>,
        }
        impl Write for Interrupted {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.first {
                    self.first = false;
                    return Err(io::Error::new(ErrorKind::Interrupted, "retry"));
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut writer = ConnectionWriter::new(Interrupted {
            first: true,
            bytes: Vec::new(),
        });
        writer.write_all(b"valid\n").expect("retry");
        writer.flush().expect("flush");
        assert_eq!(writer.inner.bytes, b"valid\n");
        assert!(!writer.poisoned);
    }
    #[test]
    fn zero_progress_poisoning_stops_all_later_writes() {
        struct Zero {
            calls: usize,
        }
        impl Write for Zero {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                Ok(0)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut writer = ConnectionWriter::new(Zero { calls: 0 });
        assert!(writer.write_all(b"x").is_err());
        assert!(writer.write_all(b"y").is_err());
        assert_eq!(writer.inner.calls, 1);
        assert!(writer.poisoned);
    }
    #[test]
    fn flush_failure_prevents_future_frame_writes() {
        struct BadFlush {
            bytes: Vec<u8>,
        }
        impl Write for BadFlush {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                self.bytes.extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::new(ErrorKind::BrokenPipe, "flush failed"))
            }
        }
        let mut writer = ConnectionWriter::new(BadFlush { bytes: Vec::new() });
        writer.write_all(b"first\n").expect("write");
        assert!(writer.flush().is_err());
        assert!(writer.write_all(b"second\n").is_err());
        assert_eq!(writer.inner.bytes, b"first\n");
    }
    #[test]
    fn failed_slab_transport_retains_source_until_owner_drop_and_cannot_append() {
        use crate::rpc::{RpcTransportLimits, write_slab_session_event};
        use engine::{
            admission::AdmissionController,
            parser_allocation::{ParserAllocation, ParserAllocationLimits},
        };
        use std::time::Duration;
        let limits = ParserAllocationLimits {
            carry_bytes: 32,
            call_bytes: 128,
            decoded_bytes: 128,
            depth: 8,
            keys: 16,
            nodes: 32,
            proposal_bytes: 256,
            calls: 4,
        };
        let size = ParserAllocation::required_bytes(limits).expect("size");
        let admission = AdmissionController::new(size).expect("admission");
        let mut grant = admission.reserve_grant("parser", &[size]).expect("grant");
        let mut storage = ParserAllocation::from_grant(&mut grant, limits).expect("storage");
        {
            let w = storage.workspaces();
            w.decoded[0] = b'a';
            w.call[..2].copy_from_slice(b"{}");
        }
        storage
            .commit_validated_call(1, 0..1, 2, 0..2)
            .expect("fixture");
        let batch = storage.freeze();
        let mut writer = ConnectionWriter::new(PartialFailure {
            bytes: Vec::new(),
            allowance: 3,
        });
        let policy = RpcTransportLimits::new(4096, Duration::from_secs(1)).expect("policy");
        assert!(write_slab_session_event(&mut writer, "s", &batch, policy).is_err());
        assert!(writer.poisoned);
        let partial = writer.inner.bytes.clone();
        assert!(write_fatal(&mut writer, RpcError::internal("disconnect")).is_err());
        assert_eq!(writer.inner.bytes, partial);
        assert_eq!(
            admission
                .snapshot()
                .expect("source remains owned")
                .reserved_bytes(),
            size
        );
        drop(batch);
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    }
}
