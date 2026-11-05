//! Harness-core error type.

use std::fmt;

/// Result alias for harness-core operations.
pub type HarnessResult<T> = Result<T, HarnessError>;

/// Errors from the session store, context builder, supervisor, or telemetry.
#[derive(Debug)]
pub enum HarnessError {
    /// SQLite or schema failure.
    Store(String),
    /// Invalid session or event input.
    Invalid(String),
    /// Engine render/generate failure.
    Engine(String),
    /// Telemetry flush or schema failure (never fatal to a session turn).
    Telemetry(String),
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(message) => write!(f, "store: {message}"),
            Self::Invalid(message) => write!(f, "invalid: {message}"),
            Self::Engine(message) => write!(f, "engine: {message}"),
            Self::Telemetry(message) => write!(f, "telemetry: {message}"),
        }
    }
}

impl std::error::Error for HarnessError {}

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
