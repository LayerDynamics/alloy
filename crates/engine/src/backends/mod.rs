//! Executable adapters for the shared family-independent runtime.

/// Actual Qwen3 host-reference loader, cache, renderer, and tokenizer adapter.
pub mod qwen;

pub use qwen::{FilePin, Qwen3Artifact, Qwen3Backend, Qwen3BackendLimits};
