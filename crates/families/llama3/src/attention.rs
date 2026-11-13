//! Llama uses full causal grouped-query attention without Q/K RMSNorm.
pub use mlx_ffi::attention::{AttentionMask, grouped_query_attention_scaled};
