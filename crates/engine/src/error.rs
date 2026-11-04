//! Structured errors for engine operations.
//!
//! Error categories are machine-readable. I/O and parser failures retain
//! their underlying sources instead of requiring callers to parse messages.

use std::{
    error::Error,
    fmt, io,
    path::{Path, PathBuf},
};

/// Machine-readable classification of an engine failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An identifier, option, or request is invalid.
    InvalidRequest,
    /// A requested profile, artifact, request, or state was not found.
    NotFound,
    /// The selected implementation does not support the requested behavior.
    Unsupported,
    /// The requested operation conflicts with active work.
    Busy,
    /// The request exceeds the validated model context limit.
    ContextLimitExceeded,
    /// The request exceeds an applicable memory budget.
    MemoryLimitExceeded,
    /// A fallible allocation failed.
    AllocationFailed,
    /// An artifact failed integrity or compatibility validation.
    ArtifactValidationFailed,
    /// Reading a checkpoint failed at the filesystem or I/O layer.
    CheckpointIo,
    /// A checkpoint representation could not be parsed.
    CheckpointFormat,
    /// Generation was cancelled.
    Cancelled,
    /// Actual model execution failed.
    ExecutionFailed,
    /// An internal invariant or arithmetic constraint failed.
    Internal,
}

/// The engine's structured error type.
///
/// Errors are deliberately not `Clone` or `PartialEq`: underlying I/O
/// and parser errors are retained as real error sources.
#[derive(Debug)]
#[non_exhaustive]
pub enum EngineError {
    /// Invalid request content or configuration.
    InvalidRequest {
        /// Request field or input category responsible for the failure.
        field: &'static str,
        /// Explanation of the invalid value.
        message: String,
    },

    /// A requested resource does not exist.
    NotFound {
        /// Resource category, such as `profile`, `artifact`, or `request`.
        resource: &'static str,
        /// Identifier that could not be resolved.
        id: String,
    },

    /// A requested operation or option is unsupported.
    Unsupported {
        /// Unsupported capability or operation.
        feature: &'static str,
        /// Explanation of the implementation limit.
        message: String,
    },

    /// An operation conflicts with active work.
    Busy {
        /// Identifier of the conflicting request, if known.
        active_request_id: Option<String>,
        /// Explanation of the conflict.
        message: String,
    },

    /// The requested context reservation exceeds the validated limit.
    ContextLimitExceeded {
        /// Number of rendered input tokens.
        input_tokens: usize,
        /// Reserved maximum number of output tokens.
        output_reserve_tokens: usize,
        /// Additional context safety margin.
        safety_margin_tokens: usize,
        /// Validated context limit for the selected profile.
        limit_tokens: usize,
    },

    /// A required reservation exceeds the applicable memory budget.
    MemoryLimitExceeded {
        /// Operation or reservation category.
        operation: &'static str,
        /// Required bytes.
        required_bytes: u64,
        /// Available bytes within the applicable budget.
        available_bytes: u64,
    },

    /// A fallible allocation could not be satisfied.
    AllocationFailed {
        /// Allocation purpose.
        operation: &'static str,
        /// Requested bytes, when they can be determined.
        requested_bytes: Option<u64>,
    },

    /// An artifact failed verification or model compatibility checks.
    ArtifactValidationFailed {
        /// Artifact identifier, when available.
        artifact_id: Option<String>,
        /// Validation failure explanation.
        message: String,
    },

    /// A checkpoint filesystem operation failed.
    CheckpointIo {
        /// Filesystem operation that failed.
        operation: &'static str,
        /// Path involved in the operation.
        path: PathBuf,
        /// Original I/O error.
        source: io::Error,
    },

    /// A checkpoint or associated metadata could not be parsed.
    CheckpointFormat {
        /// Representation being parsed, such as `json` or `safetensors`.
        format: &'static str,
        /// Source path, if the representation came from a file.
        path: Option<PathBuf>,
        /// Original parsing error.
        source: Box<dyn Error + Send + Sync + 'static>,
    },

    /// A generation request was cancelled.
    Cancelled {
        /// Identifier of the cancelled request.
        request_id: String,
    },

    /// Model execution failed after a valid request was accepted.
    ExecutionFailed {
        /// Execution stage, such as `prefill`, `decode`, or `detokenize`.
        stage: &'static str,
        /// Failure explanation.
        message: String,
    },

    /// An engine invariant or checked calculation failed.
    Internal {
        /// Failure explanation.
        message: String,
    },
}

impl EngineError {
    /// Return the machine-readable error category.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::InvalidRequest { .. } => ErrorKind::InvalidRequest,
            Self::NotFound { .. } => ErrorKind::NotFound,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::Busy { .. } => ErrorKind::Busy,
            Self::ContextLimitExceeded { .. } => ErrorKind::ContextLimitExceeded,
            Self::MemoryLimitExceeded { .. } => ErrorKind::MemoryLimitExceeded,
            Self::AllocationFailed { .. } => ErrorKind::AllocationFailed,
            Self::ArtifactValidationFailed { .. } => ErrorKind::ArtifactValidationFailed,
            Self::CheckpointIo { .. } => ErrorKind::CheckpointIo,
            Self::CheckpointFormat { .. } => ErrorKind::CheckpointFormat,
            Self::Cancelled { .. } => ErrorKind::Cancelled,
            Self::ExecutionFailed { .. } => ErrorKind::ExecutionFailed,
            Self::Internal { .. } => ErrorKind::Internal,
        }
    }

    /// Construct an invalid-request error.
    pub fn invalid_request(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            field,
            message: message.into(),
        }
    }

    /// Construct an unsupported-feature error.
    pub fn unsupported(feature: &'static str, message: impl Into<String>) -> Self {
        Self::Unsupported {
            feature,
            message: message.into(),
        }
    }

    /// Preserve an original checkpoint I/O failure.
    pub fn checkpoint_io(
        operation: &'static str,
        path: impl Into<PathBuf>,
        source: io::Error,
    ) -> Self {
        Self::CheckpointIo {
            operation,
            path: path.into(),
            source,
        }
    }

    /// Preserve an original checkpoint parser failure.
    pub fn checkpoint_format<E>(format: &'static str, path: Option<PathBuf>, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::CheckpointFormat {
            format,
            path,
            source: Box::new(source),
        }
    }

    /// Return the original I/O error kind when this is an I/O failure.
    ///
    /// This does not classify other failures as I/O failures merely because
    /// their human-readable messages mention a filesystem operation.
    pub fn io_kind(&self) -> Option<io::ErrorKind> {
        match self {
            Self::CheckpointIo { source, .. } => Some(source.kind()),
            _ => None,
        }
    }

    /// Return the checkpoint path associated with this error, if present.
    pub fn checkpoint_path(&self) -> Option<&Path> {
        match self {
            Self::CheckpointIo { path, .. } => Some(path.as_path()),
            Self::CheckpointFormat { path, .. } => path.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { field, message } => {
                write!(f, "invalid request field {field}: {message}")
            }

            Self::NotFound { resource, id } => {
                write!(f, "{resource} not found: {id}")
            }

            Self::Unsupported { feature, message } => {
                write!(f, "unsupported {feature}: {message}")
            }

            Self::Busy {
                active_request_id,
                message,
            } => {
                if let Some(id) = active_request_id {
                    write!(f, "engine busy with request {id}: {message}")
                } else {
                    write!(f, "engine busy: {message}")
                }
            }

            Self::ContextLimitExceeded {
                input_tokens,
                output_reserve_tokens,
                safety_margin_tokens,
                limit_tokens,
            } => {
                write!(
                    f,
                    "context limit exceeded: input={input_tokens}, \
                     output reserve={output_reserve_tokens}, \
                     safety margin={safety_margin_tokens}, \
                     limit={limit_tokens}"
                )
            }

            Self::MemoryLimitExceeded {
                operation,
                required_bytes,
                available_bytes,
            } => {
                write!(
                    f,
                    "{operation} requires {required_bytes} bytes; \
                     {available_bytes} bytes available"
                )
            }

            Self::AllocationFailed {
                operation,
                requested_bytes,
            } => {
                if let Some(bytes) = requested_bytes {
                    write!(
                        f,
                        "allocation failed during {operation}: \
                         requested {bytes} bytes"
                    )
                } else {
                    write!(f, "allocation failed during {operation}")
                }
            }

            Self::ArtifactValidationFailed {
                artifact_id,
                message,
            } => {
                if let Some(id) = artifact_id {
                    write!(f, "artifact {id} failed validation: {message}")
                } else {
                    write!(f, "artifact validation failed: {message}")
                }
            }

            Self::CheckpointIo {
                operation,
                path,
                source,
            } => {
                write!(
                    f,
                    "checkpoint I/O failure during {operation} at {}: {source}",
                    path.display()
                )
            }

            Self::CheckpointFormat {
                format,
                path,
                source,
            } => {
                if let Some(path) = path {
                    write!(
                        f,
                        "invalid checkpoint {format} at {}: {source}",
                        path.display()
                    )
                } else {
                    write!(f, "invalid checkpoint {format}: {source}")
                }
            }

            Self::Cancelled { request_id } => {
                write!(f, "request cancelled: {request_id}")
            }

            Self::ExecutionFailed { stage, message } => {
                write!(f, "model execution failed during {stage}: {message}")
            }

            Self::Internal { message } => {
                write!(f, "internal engine error: {message}")
            }
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CheckpointIo { source, .. } => Some(source),
            Self::CheckpointFormat { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
