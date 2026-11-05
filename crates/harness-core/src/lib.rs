//! Phase-1 harness core: supervisor skeleton, context builder, session store,
//! and FR-35 telemetry client.
//!
//! This crate sits between the daemon JSON-RPC surface and the in-process
//! engine. Phase 1 covers a single-session turn loop with no tools, policy,
//! MCP, skills, or compaction.

#![deny(missing_docs)]

pub mod context;
pub mod error;
pub mod store;
pub mod supervisor;
pub mod telemetry;

pub use context::ContextBuilder;
pub use error::{HarnessError, HarnessResult};
pub use store::{EventKind, EventRecord, SessionRecord, SessionStatus, SessionStore};
pub use supervisor::{Supervisor, TurnOutcome};
pub use telemetry::{HardwareTier, TelemetryClient, TelemetryPayload};
