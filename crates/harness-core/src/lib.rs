//! Harness core: supervision, durable sessions, W3 policy/approvals and artifacts.
//!
//! The active supervisor remains text-only. These reusable daemon APIs do not
//! enable builtin execution, plugin hosting, live approval RPC or the tool loop.

#![deny(missing_docs)]

pub mod approval;
pub mod artifact;
pub mod context;
pub mod error;
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
