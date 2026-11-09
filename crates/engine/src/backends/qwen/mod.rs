//! Qwen3 unified-memory MLX adapter for the family-independent runtime.
//!
//! Verified safetensors retain dense f32 or packed Affine4 matrices in one
//! set of MLX weight arrays, including vocabulary weights. Full f32 MLX
//! K/V is shared by supported CPU and Metal operations; it is not the
//! legacy host-reference f64 cache.
//!
//! new() selects MLX CPU for portable, deterministic setup. Use
//! new_with_execution(..., Qwen3Execution::MlxMetal) to request Metal.
//! A requested Metal device is checked on the native worker and is never
//! silently replaced with CPU execution. Streams and arrays stay there.
//!
//! Shared pinned rendering, artifact checks, and incremental ByteLevel
//! decoding retain their existing contracts.
//!
//! Planning counts decoded/native loading overlap, full-cache functional
//! update overlap, and conservative f32 graph/workspace payload estimates.
//! These are engine-accounted allowances, NOT measured process RSS or a
//! hard MLX allocator limit. Library/allocator overhead requires separately
//! provisioned headroom; RAM discovery and allocator-pressure integration
//! are not implemented by this adapter.
//!
//! Affine4 residency and greedy decoding are advertised. Rotating caches,
//! stochastic sampling, thinking mode, and public prefix reuse are not.

mod backend;

pub use backend::*;
