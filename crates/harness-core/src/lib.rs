//! Harness core: session supervision, event storage, telemetry, and W3 preparation.
//!
//! The active supervisor remains text-only. Policy/binding preparation does not
//! enable effects, durable approval grants, hooks, or the integrated tool loop.

#![deny(missing_docs)]

pub mod approval;
pub mod context;
pub mod error;
pub mod policy;
pub mod store;
pub mod supervisor;
pub mod telemetry;

pub use context::ContextBuilder;
pub use error::{HarnessError, HarnessResult};
pub use store::{EventKind, EventRecord, SessionRecord, SessionStatus, SessionStore};
pub use supervisor::{Supervisor, TurnOutcome};
pub use telemetry::{HardwareTier, TelemetryClient, TelemetryPayload};
