//! Shared native process ownership primitives for Alloy supervised children.

/// Native supervisor launch and cleanup interfaces.
pub mod spawn;

#[cfg(target_os = "macos")]
mod macos_service;

/// Ensure consumers link the platform supervisor implementation.
pub fn link_native_supervisor() {}
