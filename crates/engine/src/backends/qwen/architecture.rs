// Adapter dispatch is separate from the homogeneous decoder's family enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendFamily {
    Qwen3,
    Llama3,
    SmolLm3,
    Qwen35,
    Ornith,
}
impl BackendFamily {
    fn hybrid(self) -> bool {
        matches!(self, Self::Qwen35 | Self::Ornith)
    }
}
impl From<inference_infra::decoder::config::DecoderFamily> for BackendFamily {
    fn from(value: inference_infra::decoder::config::DecoderFamily) -> Self {
        use inference_infra::decoder::config::DecoderFamily as F;
        match value {
            F::Qwen3 => Self::Qwen3,
            F::Llama3 => Self::Llama3,
            F::SmolLm3 => Self::SmolLm3,
        }
    }
}

enum BackendConfig {
    Transformer(Qwen3Config),
    Hybrid(inference_infra::decoder::qwen35::config::Qwen35Config),
}
impl BackendConfig {
    fn vocabulary(&self) -> usize {
        match self {
            Self::Transformer(c) => c.vocab_size,
            Self::Hybrid(c) => c.vocabulary,
        }
    }
    fn context(&self) -> usize {
        match self {
            Self::Transformer(c) => c.max_context,
            Self::Hybrid(c) => c.context,
        }
    }
    fn chunk_limit(&self, configured: usize, capacity: usize) -> usize {
        match self {
            Self::Transformer(c) if !c.has_qk_norm() && capacity > 16_384 => configured.min(32),
            _ => configured,
        }
    }
    fn cache_bound(&self, capacity: usize) -> Result<u64, EngineError> {
        match self {
            Self::Transformer(c) => cache_bound(c, capacity),
            Self::Hybrid(c) => c
                .cache_payload_plan(capacity)
                .map(|p| p.total_overlap_bytes)
                .map_err(|e| execution_error("plan hybrid cache", e.to_string())),
        }
    }
    fn workspace_bound(
        &self,
        capacity: usize,
        chunk: usize,
        token_bytes: usize,
    ) -> Result<u64, EngineError> {
        match self {
            Self::Transformer(c) => workspace_bound(c, capacity, chunk, token_bytes),
            Self::Hybrid(c) => {
                let p = c
                    .step_payload_plan(chunk, capacity, false)
                    .map_err(|e| execution_error("plan hybrid step", e.to_string()))?;
                sum(&[
                    p.work_bytes,
                    product(p.output_bytes, 3)?,
                    product(as_u64(token_bytes)?, 8)?,
                    16 * 1024,
                ])
            }
        }
    }
    fn largest_upload(&self) -> Result<u64, EngineError> {
        match self {
            Self::Transformer(c) => largest_upload_buffer_bound(c),
            Self::Hybrid(c) => c
                .weight_payload_plan()
                .map(|p| p.largest_upload_bytes)
                .map_err(|e| execution_error("plan hybrid upload", e.to_string())),
        }
    }
}

/// Trained Ornith text serialization with explicit non-thinking generation.
pub const ORNITH_TEXT_TEMPLATE: &str = include_str!("ornith_text.jinja");
/// Trained upstream Qwen3.5 text serialization with explicit non-thinking generation.
pub const QWEN35_TEXT_TEMPLATE: &str = include_str!("qwen35_text.jinja");
/// Combined full-attention KV, convolution history and recurrent matrix state.
pub const QWEN35_CACHE_REPRESENTATION: &str = "qwen35.mlx-f32.full-kv-and-recurrent.v1";
