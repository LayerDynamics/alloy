//! Executable adapters for the shared family-independent runtime.

/// Unified-memory MLX Qwen3 execution, verified loading, and pinned rendering.
pub mod qwen;

pub use qwen::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits, Qwen3Execution};
