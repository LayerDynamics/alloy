//! Structured errors for engine operations.
//!
//! Error categories are machine-readable. I/O and parser failures retain
//! their underlying sources instead of requiring callers to parse messages.

use std::{
    error::Error,
    fmt, io,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use crate::{
    family::ModelInstanceId,
    memory_guard::CeilingBreakdown,
    types::{ArtifactId, SessionStateId},
};

/// The memory constraint responsible for a refusal.
///
/// This is captured at the origin. Handlers must not infer the constraint
/// from the human-readable operation or assume every refusal is an OOM.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryLimitReason {
    /// The shared reservation ledger cannot grant the requested bytes.
    /// Compare demand with the whole budget to distinguish an intrinsically
    /// oversized request from capacity occupied by existing reservations.
    ReservationBudget {
        /// Complete ledger budget, not its currently available remainder.
        budget_bytes: u64,
    },
    /// A fail-fast advisory estimate exceeds the reported available capacity.
    /// The advisory interface does not provide the complete ledger budget.
    AdvisoryEstimate,
    /// Planned native allocations exceed the configured allocator ceiling.
    AllocatorCeiling {
        /// Applicable native-allocation ceiling at the observation.
        ceiling_bytes: u64,
    },
    /// The enforcer paused admissions; byte availability may still suffice.
    AdmissionsPaused,
    /// A bounded loader, cache, output, or other buffer allowance was exceeded.
    BufferLimit,
    /// A fresh host observation rejected planned startup before native loading.
    HostPreflight {
        /// Artifact whose startup plan was rejected.
        artifact_id: ArtifactId,
        /// Complete observed ceiling breakdown, not measured model consumption.
        ceiling: CeilingBreakdown,
    },
}

impl MemoryLimitReason {
    /// Stable diagnostic name, independent of display wording.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReservationBudget { .. } => "reservation_budget",
            Self::AdvisoryEstimate => "advisory_estimate",
            Self::AllocatorCeiling { .. } => "allocator_ceiling",
            Self::AdmissionsPaused => "admissions_paused",
            Self::BufferLimit => "buffer_limit",
            Self::HostPreflight { .. } => "host_preflight",
        }
    }
}

/// One observed ownership guarantee blocking a particular operation.
///
/// These values contain diagnostic identities and counts only. They do not
/// own model buffers, reservations, residency pins, or session-state leases.
/// Counts are snapshots: a concurrent final handle drop may immediately make
/// them stale. They never authorize release or eviction of any resource.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProtectedResource {
    /// Explicit residency handles protect this exact model load.
    ModelPin {
        /// Human-recognizable artifact identity.
        artifact_id: ArtifactId,
        /// Exact loaded instance, distinct from a later load of that artifact.
        instance_id: ModelInstanceId,
        /// Positive number of pin handles observed without acquiring one.
        handles: NonZeroUsize,
    },
    /// Live leases protect the addressed canonical state and its model owner.
    SessionStateLease {
        /// Human-recognizable artifact identity.
        artifact_id: ArtifactId,
        /// Exact model load owning the state.
        instance_id: ModelInstanceId,
        /// Exact canonical publication protected by the leases.
        state_id: SessionStateId,
        /// Positive number of lease handles observed without acquiring one.
        handles: NonZeroUsize,
    },
}

impl ProtectedResource {
    /// Produce non-owning diagnostic data; never inspect the native worker.
    pub fn diagnostic(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Self::ModelPin {
                artifact_id,
                instance_id,
                handles,
            } => json!({
                "protection": "model_pin",
                "artifact_id": artifact_id.as_str(),
                "instance_id": format!("model-{}", instance_id.diagnostic_id()),
                "handles": handles.get(),
            }),
            Self::SessionStateLease {
                artifact_id,
                instance_id,
                state_id,
                handles,
            } => json!({
                "protection": "session_state_lease",
                "artifact_id": artifact_id.as_str(),
                "instance_id": format!("model-{}", instance_id.diagnostic_id()),
                "state_id": state_id.to_string(),
                "handles": handles.get(),
            }),
        }
    }
}

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
    /// Active/pending execution or a scheduler slot prevents the operation.
    Busy,
    /// Identified residency pins or state leases prevent the operation.
    ProtectedResources,
    /// The request exceeds the validated model context limit.
    ContextLimitExceeded,
    /// A typed memory constraint or admission pause prevents the request.
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

impl ErrorKind {
    /// Stable machine-readable diagnostic name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::NotFound => "not_found",
            Self::Unsupported => "unsupported",
            Self::Busy => "busy",
            Self::ProtectedResources => "protected_resources",
            Self::ContextLimitExceeded => "context_limit_exceeded",
            Self::MemoryLimitExceeded => "memory_limit_exceeded",
            Self::AllocationFailed => "allocation_failed",
            Self::ArtifactValidationFailed => "artifact_validation_failed",
            Self::CheckpointIo => "checkpoint_io",
            Self::CheckpointFormat => "checkpoint_format",
            Self::Cancelled => "cancelled",
            Self::ExecutionFailed => "execution_failed",
            Self::Internal => "internal",
        }
    }
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

    /// Active/pending execution or scheduler ownership prevents an operation.
    /// Idle residency pins and canonical state leases use ProtectedResources.
    Busy {
        /// Identifier of the conflicting request, if known.
        active_request_id: Option<String>,
        /// Explanation of the conflict.
        message: String,
    },

    /// Existing ownership guarantees prevent the requested operation.
    ///
    /// Resources are operation-specific observed blockers, not a list of all
    /// residents. A model pin alone does not prevent reclaiming its idle KV.
    /// The runtime supplies at least one blocker and preserves all ownership.
    ProtectedResources {
        /// Operation name, such as `startup pressure reclamation`.
        operation: &'static str,
        /// Exact observed pin/lease blockers; none of these values is a lease.
        resources: Vec<ProtectedResource>,
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

    /// A typed memory constraint or admission pause prevents the operation.
    MemoryLimitExceeded {
        /// Typed constraint; distinguishes a pause from inadequate bytes.
        reason: MemoryLimitReason,
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
            Self::ProtectedResources { .. } => ErrorKind::ProtectedResources,
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

    /// Serialize a captured engine cause for evidence and RPC handlers.
    ///
    /// This performs no native queries and acquires no ownership. The
    /// human-readable message may evolve; `kind`, `reason`, and structured
    /// fields are the machine-readable contract. Instance identifiers are
    /// process-local diagnostic tokens, never release/eviction capabilities.
    pub fn diagnostic(&self) -> serde_json::Value {
        use serde_json::json;
        let mut report = json!({"kind": self.kind().as_str(), "message": self.to_string()});
        match self {
            Self::Busy {
                active_request_id, ..
            } => {
                report["active_request_id"] = json!(active_request_id);
            }
            Self::ProtectedResources {
                operation,
                resources,
            } => {
                report["operation"] = json!(operation);
                report["resources"] = resources
                    .iter()
                    .map(ProtectedResource::diagnostic)
                    .collect();
            }
            Self::MemoryLimitExceeded {
                reason,
                operation,
                required_bytes,
                available_bytes,
            } => {
                report["reason"] = json!(reason.as_str());
                report["operation"] = json!(operation);
                report["required_bytes"] = json!(required_bytes);
                report["available_bytes"] = json!(available_bytes);
                match reason {
                    MemoryLimitReason::ReservationBudget { budget_bytes } => {
                        report["budget_bytes"] = json!(budget_bytes);
                    }
                    MemoryLimitReason::AllocatorCeiling { ceiling_bytes } => {
                        report["ceiling_bytes"] = json!(ceiling_bytes);
                    }
                    MemoryLimitReason::HostPreflight {
                        artifact_id,
                        ceiling,
                    } => {
                        report["artifact_id"] = json!(artifact_id.as_str());
                        report["ceiling"] = json!({
                            "reserve_bytes": ceiling.reserve_bytes,
                            "static_bytes": ceiling.static_bytes,
                            "dynamic_bytes": ceiling.dynamic_bytes,
                            "metal_cap_bytes": ceiling.metal_cap_bytes,
                            "hard_limit_bytes": ceiling.hard_limit_bytes,
                            "binding": ceiling.binding.map(|binding| binding.as_str()),
                        });
                    }
                    MemoryLimitReason::AdmissionsPaused
                    | MemoryLimitReason::AdvisoryEstimate
                    | MemoryLimitReason::BufferLimit => {}
                }
            }
            _ => {}
        }
        report
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

            Self::ProtectedResources {
                operation,
                resources,
            } => {
                write!(f, "{operation} blocked by protected resources")?;
                for resource in resources {
                    match resource {
                        ProtectedResource::ModelPin {
                            artifact_id,
                            instance_id,
                            handles,
                        } => {
                            write!(
                                f,
                                "; {artifact_id} model-{} has {} residency pin handle(s)",
                                instance_id.diagnostic_id(),
                                handles
                            )?;
                        }
                        ProtectedResource::SessionStateLease {
                            artifact_id,
                            instance_id,
                            state_id,
                            handles,
                        } => {
                            write!(
                                f,
                                "; {artifact_id} model-{} state {state_id} has {} lease handle(s)",
                                instance_id.diagnostic_id(),
                                handles
                            )?;
                        }
                    }
                }
                Ok(())
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
                     limit={limit_tokens}; start a new session or select a profile with a larger context"
                )
            }

            Self::MemoryLimitExceeded {
                reason,
                operation,
                required_bytes,
                available_bytes,
            } => {
                if matches!(reason, MemoryLimitReason::AdmissionsPaused) {
                    return write!(
                        f,
                        "admissions paused by memory pressure during {operation}; requested {required_bytes} bytes, {available_bytes} reservation bytes currently available"
                    );
                }
                if let MemoryLimitReason::HostPreflight {
                    artifact_id,
                    ceiling,
                } = reason
                {
                    return write!(
                        f,
                        "startup preflight rejected {artifact_id}: planned {required_bytes} bytes exceeds observed safe ceiling {available_bytes} bytes (binding {}); native loading did not start",
                        ceiling
                            .binding
                            .map(|binding| binding.as_str())
                            .unwrap_or("none")
                    );
                }
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

#[cfg(test)]
#[path = "../tests/unit/error_tests.rs"]
mod tests;
