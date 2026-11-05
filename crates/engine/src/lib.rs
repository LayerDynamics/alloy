//! Alloy's versioned in-process engine contract.
//!
//! The public API defines the boundary between the harness and actual
//! model execution. Shared types and structured errors are provided by
//! separate modules.
//!
//! Defining this contract does not implement model loading, rendering,
//! tokenization, admission, scheduling, or generation.

#![deny(missing_docs)]

pub mod admission;
pub mod api;
pub mod backends;
pub mod cache;
pub mod cache_identity;
pub mod cancellation;
pub mod error;
pub mod family;
pub mod generation;
pub mod registry;
pub mod runtime;
pub mod scheduler;
pub mod types;
pub mod rendering;