//! Library errors for registration and call validation.
//!
//! These are Rust API errors, not serialized tool-result payloads.

use std::{error::Error, fmt};

/// A registration or call validation failure.
///
/// Keep this internal contract separate from tool-result wire errors.
/// The harness can translate these failures into protocol results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// A required identity is empty or contains surrounding whitespace.
    InvalidIdentity(&'static str),

    /// The tool version is not a complete semantic version.
    InvalidVersion(String),

    /// The description contains no non-whitespace text.
    EmptyDescription,

    /// An execution limit is zero.
    InvalidLimit(&'static str),

    /// The input schema violates the registry schema contract.
    InvalidSchema(String),

    /// The request names a different tool.
    ToolMismatch,

    /// Arguments do not satisfy the registered schema.
    InvalidArguments(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentity(field) => write!(f, "invalid {field}"),
            Self::InvalidVersion(reason) => {
                write!(f, "invalid tool semantic version: {reason}")
            }
            Self::EmptyDescription => write!(f, "tool description is empty"),
            Self::InvalidLimit(field) => write!(f, "{field} must be positive"),
            Self::InvalidSchema(reason) => write!(f, "invalid input schema: {reason}"),
            Self::ToolMismatch => write!(f, "call does not match registered tool"),
            Self::InvalidArguments(reason) => {
                write!(f, "arguments do not match input schema: {reason}")
            }
        }
    }
}

impl Error for ValidationError {}

/// Failure to register or resolve a tool.
///
/// These are registry errors, not serialized tool-result error categories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// This identifier is already registered, regardless of version.
    DuplicateId(String),

    /// No registered definition owns this identifier.
    UnknownTool(String),

    /// Registration metadata or call arguments failed validation.
    Validation(ValidationError),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "duplicate tool identifier: {id}"),
            Self::UnknownTool(id) => write!(f, "unknown tool identifier: {id}"),
            Self::Validation(error) => write!(f, "{error}"),
        }
    }
}

impl Error for RegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            Self::DuplicateId(_) | Self::UnknownTool(_) => None,
        }
    }
}

impl From<ValidationError> for RegistryError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_error_converts_into_registry_error() {
        let error = ValidationError::InvalidIdentity("call_id");
        let wrapped = RegistryError::from(error.clone());

        assert_eq!(wrapped, RegistryError::Validation(error));
    }

    #[test]
    fn wrapped_validation_error_is_available_as_source() {
        let error =
            RegistryError::Validation(ValidationError::InvalidArguments("missing path".into()));

        let source = error.source().expect("validation source");
        assert!(source.downcast_ref::<ValidationError>().is_some());
        assert_eq!(
            source.to_string(),
            "arguments do not match input schema: missing path"
        );
    }

    #[test]
    fn leaf_errors_have_no_source() {
        for error in [
            RegistryError::DuplicateId("fs.read".into()),
            RegistryError::UnknownTool("unknown.tool".into()),
        ] {
            assert!(error.source().is_none());
        }

        let error = ValidationError::EmptyDescription;
        assert!(error.source().is_none());
    }

    #[test]
    fn registry_diagnostics_identify_the_affected_tool() {
        assert_eq!(
            RegistryError::DuplicateId("fs.read".into()).to_string(),
            "duplicate tool identifier: fs.read"
        );

        assert_eq!(
            RegistryError::UnknownTool("shell.run".into()).to_string(),
            "unknown tool identifier: shell.run"
        );
    }

    #[test]
    fn public_error_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ValidationError>();
        assert_send_sync::<RegistryError>();
    }
}
