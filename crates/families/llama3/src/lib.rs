//! Native Llama 3.x over the shared dense decoder, loader and KV cache.
pub mod attention;
pub mod config;
pub mod register;
pub mod rope;
pub mod template;
pub use inference_infra::decoder::{fixture, loader, mlx_model};
