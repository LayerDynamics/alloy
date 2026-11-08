//! Harness core: supervision, durable sessions, W3 policy/approvals and artifacts.
//!
//! The active supervisor remains text-only. These reusable daemon APIs do not
//! enable builtin execution, live approval RPC or the tool loop. The W5 plugin
//! host is opt-in; the active supervisor does not construct or call it.

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
