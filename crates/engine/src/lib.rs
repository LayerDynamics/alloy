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
/// Live soft/hard enforcer poll tick (pool-drain grace, admission pause).
pub mod memory_enforcer;
/// oMLX-method live memory guard (tiers, ceiling, watermarks).
pub mod memory_guard;
/// Unified CPU/Metal memory discovery and shared budgeting.
pub mod platform;
/// Prefill admit-line checks and adaptive chunk shrink (M5–M6).
pub mod prefill_admit;
/// Bounded unified-memory pressure and reclamation decisions.
pub mod pressure;
pub mod registry;
pub mod rendering;
pub mod residency_pin;
pub mod residency_policy;
pub mod runtime;
pub mod scheduler;
/// Bounded hot KV retention and atomic safetensors persistence.
pub mod tiered_kv;
pub mod types;

#[cfg(test)]
mod w3_acceptance;
