//! Canonical tool contracts, registration, and validation for Alloy.
//!
//! This crate defines the shared boundary between model-requested tool
//! calls and daemon-owned executable implementations.
//!
//! # Responsibilities
//!
//! - Define tool metadata, origins, effect classes, and approval requirements.
//! - Represent canonical tool calls and their linked results.
//! - Register executable implementations under unique tool identifiers.
//! - Validate registration schemas and call arguments before execution.
//!
//! # Trust boundary
//!
//! Model-generated names, origins, versions, arguments, and paths are
//! untrusted input. They cannot create registrations or select arbitrary
//! executable implementations.
//!
//! A call must resolve to an existing registry entry and pass argument
//! validation before it can be offered for authorization and execution.
//! Schema validation alone does not grant authorization.
//!
//! # Ownership boundaries
//!
//! The registry owns registered executable implementations. The harness
//! owns policy evaluation, durable approval decisions, lifecycle persistence,
//! and orchestration. Model-family crates parse structured calls, and
//! clients display state and relay explicit user decisions.
//!
//! These modules establish the Phase 2 W1 contracts. Builtin execution and
//! cancellation are added in W4; they must use the same registry boundary.

#![forbid(unsafe_code)]

pub mod call;
pub mod definition;
pub mod error;
pub mod registry;
pub mod result;
pub mod validation;

pub use error::{RegistryError, ValidationError};
