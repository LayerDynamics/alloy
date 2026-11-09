//! Harness core: supervision, durable sessions, W3 policy/approvals and artifacts.
//!
//! The supervisor supports both text turns and explicitly configured canonical
//! tool turns. Tool turns use durable policy receipts, scoped builtin adapters,
//! ordered plugins and a separately responsive approval/control connection.

#![deny(missing_docs)]

pub mod approval;
pub mod artifact;
pub mod context;
pub mod error;
/// Supervised V1 two-hook plugin host on the supported native platforms.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod plugin_host;
pub mod policy;
pub mod store;
pub mod supervisor;
pub mod telemetry;
pub mod tool_execution;

pub use context::ContextBuilder;
pub use error::{HarnessError, HarnessResult};
pub use store::{EventKind, EventRecord, SessionRecord, SessionStatus, SessionStore};
pub use supervisor::{Supervisor, TurnOutcome};
pub use telemetry::{HardwareTier, TelemetryClient, TelemetryPayload};
