use inference_infra::capabilities::{Backend, BackendCapabilities, QuantMode};

/// Architecture identifiers this family serves (B.5 `architectures`).
pub const ARCHITECTURES: &[&str] = &["Qwen3ForCausalLM"];

/// Bump on changes to forward-pass behavior, templates, or capabilities.
pub const IMPL_VERSION: u32 = 2;

/// Registration record describing this family's implemented execution path.
#[derive(Clone, Debug)]
pub struct FamilyRegistration {
    pub architectures: Vec<String>,
    pub impl_version: u32,
    pub capabilities: BackendCapabilities,
}

/// Register the host-side reference implementation, not the default MLX device.
///
/// The Result signature is retained for compatibility with existing callers.
/// Cached decoding and sampling are not currently implemented by this family.
pub fn register() -> Result<FamilyRegistration, mlx_ffi::Error> {
    Ok(FamilyRegistration {
        architectures: ARCHITECTURES.iter().map(|s| s.to_string()).collect(),
        impl_version: IMPL_VERSION,
        capabilities: BackendCapabilities {
            backend: Backend::Cpu,
            supports_rope: true,
            supports_attention: true,
            supports_kv_cache: false,
            supports_sampling: false,
            quantization_modes: vec![QuantMode::Fp32],
            max_context_len: 32_768,
        },
    })
}
