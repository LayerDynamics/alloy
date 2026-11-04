use inference_infra::capabilities::BackendCapabilities;

/// Architecture identifiers this family serves (B.5 `architectures`).
pub const ARCHITECTURES: &[&str] = &["Qwen3ForCausalLM"];

/// Bump on any change to forward-pass behavior or template handling.
pub const IMPL_VERSION: u32 = 1;

/// B.5 registration record handed to the engine's family registry (W3).
#[derive(Clone, Debug)]
pub struct FamilyRegistration {
    pub architectures: Vec<String>,
    pub impl_version: u32,
    pub capabilities: BackendCapabilities,
}

/// Register the family against the live MLX device.
pub fn register() -> Result<FamilyRegistration, mlx_ffi::Error> {
    Ok(FamilyRegistration {
        architectures: ARCHITECTURES.iter().map(|s| s.to_string()).collect(),
        impl_version: IMPL_VERSION,
        capabilities: BackendCapabilities::detect()?,
    })
}
