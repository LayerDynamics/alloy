//! Native SmolLM3 dense attention with an explicit per-layer NoPE schedule.
pub mod config;
pub mod register;
pub mod template;
pub use inference_infra::decoder::{fixture, loader, mlx_model};
