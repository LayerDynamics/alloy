//! Shared native process ownership primitives for Alloy supervised children.

/// Native supervisor launch and cleanup interfaces.
pub mod spawn;

/// Ensure consumers link the platform supervisor implementation.
pub fn link_native_supervisor() {}
