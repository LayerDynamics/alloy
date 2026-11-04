use mlx_ffi::{Device, DeviceType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMode {
    Fp32,
    I4,
}

/// What the family crate (B.5 `capabilities`) reads at registration time.
#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    pub backend: Backend,
    pub supports_rope: bool,
    pub supports_attention: bool,
    pub supports_kv_cache: bool,
    pub supports_sampling: bool,
    pub quantization_modes: Vec<QuantMode>,
    pub max_context_len: usize,
}

impl BackendCapabilities {
    /// Query the process-wide default MLX device and map its type, so a
    /// Metal device is never reported as CPU (review comment 7).
    pub fn detect() -> Result<Self, mlx_ffi::Error> {
        let device = Device::get_default()?;
        let backend = match device.kind()? {
            DeviceType::Cpu => Backend::Cpu,
            DeviceType::Gpu => Backend::Metal,
        };
        Ok(BackendCapabilities {
            backend,
            // Honest values for the kernels implemented in `ops` today;
            // flip flags only as real backends land.
            supports_rope: true,
            supports_attention: true,
            supports_kv_cache: true,
            supports_sampling: true,
            quantization_modes: vec![QuantMode::Fp32],
            max_context_len: 32_768,
        })
    }
}
