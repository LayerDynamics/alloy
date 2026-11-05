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
        let home = std::env::var_os("HOME").ok_or_else(|| {
            RpcError::internal("HOME is required to derive ~/.alloy paths")
        })?;
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
        std::fs::create_dir_all(parent).map_err(|error| {
            RpcError::internal(format!("create socket parent: {error}"))
        })?;
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
                std::fs::remove_file(socket_path).map_err(|error| {
                    RpcError::internal(format!("remove stale socket: {error}"))
                })?;
            }
        }
    }
    let listener = UnixListener::bind(socket_path).map_err(|error| {
        RpcError::internal(format!("bind {}: {error}", socket_path.display()))
    })?;
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600)).map_err(
        |error| RpcError::internal(format!("chmod socket: {error}")),
    )?;
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
    let reader_stream = stream
        .try_clone()
        .map_err(|error| RpcError::internal(format!("clone stream: {error}")))?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = stream;
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

fn write_fatal(writer: &mut UnixStream, error: RpcError) -> Result<(), RpcError> {
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
