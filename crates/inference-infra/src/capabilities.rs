#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend { Cpu, Metal }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMode { Fp32, I4 }

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
    pub fn detect() -> Self {
        // TODO(#2): query the vendored mlx-ffi for the active device when the
        // bindings are wired. These are the honest values for the CPU kernels
        // implemented in `ops` today; flip flags only as real backends land.
        BackendCapabilities {
            backend: Backend::Cpu,
            supports_rope: true,
            supports_attention: true,
            supports_kv_cache: true,
            supports_sampling: true,
            quantization_modes: vec![QuantMode::Fp32],
            max_context_len: 32_768,
        }
    }
}