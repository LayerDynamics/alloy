//! Harness-core error type.

use std::fmt;

/// Result alias for harness-core operations.
pub type HarnessResult<T> = Result<T, HarnessError>;

/// Internal failures from sessions, durable tool lifecycle, artifacts, and supervision.
/// These categories do not extend the tool-result wire error vocabulary.
#[derive(Debug)]
pub enum HarnessError {
    /// Storage, schema, ownership, artifact I/O, or integrity failure.
    /// This category alone does not establish retry safety or commit outcome.
    Store(String),
    /// Invalid input, lifecycle state, binding, or serialized metadata.
    /// Never infer execution authority or retry safety from an error message.
    Invalid(String),
    /// Engine render/generate failure.
    Engine(String),
    /// Structured engine failure, preserved through the daemon boundary.
    EngineFailure(engine::error::EngineError),
    /// Telemetry flush or schema failure (never fatal to a session turn).
    Telemetry(String),
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(message) => write!(f, "store: {message}"),
            Self::Invalid(message) => write!(f, "invalid: {message}"),
            Self::Engine(message) => write!(f, "engine: {message}"),
            Self::EngineFailure(error) => write!(f, "engine: {error}"),
            Self::Telemetry(message) => write!(f, "telemetry: {message}"),
        }
    }
}

impl std::error::Error for HarnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EngineFailure(error) => Some(error),
            _ => None,
        }
    }
}

impl From<engine::error::EngineError> for HarnessError {
    fn from(error: engine::error::EngineError) -> Self {
        Self::EngineFailure(error)
    }
}

impl From<rusqlite::Error> for HarnessError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Store(error.to_string())
    }
}

impl From<serde_json::Error> for HarnessError {
    fn from(error: serde_json::Error) -> Self {
        Self::Invalid(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::HarnessError;
    use engine::error::EngineError;
    use std::{error::Error, io};

    #[test]
    fn structured_engine_failure_preserves_its_original_source() {
        let error = HarnessError::from(EngineError::checkpoint_io(
            "read",
            "fixture.safetensors",
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "fixture permission failure",
            ),
        ));
        let engine = error
            .source()
            .and_then(|source| source.downcast_ref::<EngineError>())
            .expect("original structured engine failure");
        assert_eq!(engine.io_kind(), Some(io::ErrorKind::PermissionDenied));
        assert!(engine.source().unwrap().is::<io::Error>());
    }
}
