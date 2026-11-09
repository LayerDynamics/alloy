//! Family-independent text-generation execution.
//!
//! This module runs actual model execution through GenerationSession.
//! It provides:
//!
//! - A worker thread for blocking model operations.
//! - Scheduler-owned execution lifetime.
//! - Cooperative cancellation.
//! - Bounded text-event buffering.
//! - Cancellation-aware producer backpressure.
//! - Validated, monotonic token-usage accounting.
//! - Terminal-event arbitration through ExecutionContext::finish.
//! - Rust panic containment around family execution and cleanup.
//!
//! Native process crashes are not recoverable through catch_unwind.
//!
//! A concrete family adapter must implement GenerationSession using its
//! actual model, cache, token selection, and incremental tokenizer decoder.
//! This module does not supply a simulated or fallback implementation.

mod stream;

pub use stream::*;
