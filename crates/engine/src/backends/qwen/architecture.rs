// Adapter dispatch is separate from the homogeneous decoder's family enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendFamily {
    Qwen3,
    Qwen3Coder,
    Llama3,
    SmolLm3,
    Qwen35,
    Ornith,
    Phi,
    Gemma3,
    Gemma4,
    Ministral3,
}
impl BackendFamily {
    fn xml_tools(self) -> bool {
        self.hybrid() || self == Self::Qwen3Coder
    }

    fn hybrid(self) -> bool {
        matches!(self, Self::Qwen35 | Self::Ornith)
    }
}
fn dense_cache_mode(
    family: inference_infra::decoder::config::DecoderFamily,
) -> Result<CacheMode, EngineError> {
    use inference_infra::decoder::config::DecoderFamily;
    match family {
        DecoderFamily::Qwen3 => family_cache_mode(BackendFamily::Qwen3),
        DecoderFamily::Qwen3Moe => family_cache_mode(BackendFamily::Qwen3Coder),
        DecoderFamily::Llama3 => family_cache_mode(BackendFamily::Llama3),
        DecoderFamily::SmolLm3 => family_cache_mode(BackendFamily::SmolLm3),
        DecoderFamily::Ministral3 { .. } => family_cache_mode(BackendFamily::Ministral3),
    }
}

enum BackendConfig {
    Transformer(Qwen3Config),
    Hybrid(inference_infra::decoder::qwen35::config::Qwen35Config),
    Phi(inference_infra::decoder::phi::PhiConfig),
    Gemma4(inference_infra::decoder::gemma4::Gemma4TextConfig),
    Gemma3(inference_infra::decoder::gemma3::Gemma3Config),
}
impl BackendConfig {
    fn vocabulary(&self) -> usize {
        match self {
            Self::Transformer(c) => c.vocab_size,
            Self::Hybrid(c) => c.vocabulary,
            Self::Phi(c) => c.vocabulary,
            Self::Gemma3(c) => c.vocabulary,
            Self::Gemma4(c) => c.vocabulary(),
        }
    }
    fn context(&self) -> usize {
        match self {
            Self::Transformer(c) => c.max_context,
            Self::Hybrid(c) => c.context,
            Self::Phi(c) => c.context,
            Self::Gemma3(c) => c.context,
            Self::Gemma4(c) => c.context(),
        }
    }
    fn chunk_limit(&self, configured: usize, capacity: usize) -> usize {
        match self {
            Self::Transformer(c)
                if c.family != inference_infra::decoder::config::DecoderFamily::Qwen3
                    && capacity > 16_384 =>
            {
                configured.min(32)
            }
            Self::Phi(_) | Self::Gemma3(_) if capacity > 16_384 => configured.min(32),
            _ => configured,
        }
    }
    fn cache_bound(&self, capacity: usize, chunk: usize) -> Result<u64, EngineError> {
        match self {
            Self::Transformer(c) => cache_bound(c, capacity),
            Self::Phi(c) => phi_cache_bound(c, capacity),
            Self::Gemma4(c) => c
                .cache_overlap_bound(capacity, chunk)
                .map_err(|e| model_error(e, "plan Gemma 4 cache")),
            Self::Gemma3(c) => c
                .cache_payload(capacity)
                .map_err(|e| execution_error("plan Gemma3 cache", e.to_string()))
                .and_then(|bytes| product(bytes, 2)),
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
            Self::Gemma4(c) => sum(&[
                c.workspace_bytes(chunk, capacity)
                    .map_err(|e| model_error(e, "plan Gemma 4 workspace"))?,
                product(as_u64(token_bytes)?, 8)?,
                16 * 1024,
            ]),
            Self::Gemma3(c) => {
                let p = c
                    .step_payload_plan(chunk, capacity)
                    .map_err(|e| execution_error("plan Gemma3 step", e.to_string()))?;
                // The native step bound already includes three complete KV
                // payloads for transactional overlap. Two are reserved by
                // cache_bound; charge only the remainder as workspace.
                let workspace = p
                    .work_bytes
                    .checked_sub(self.cache_bound(capacity, chunk)?)
                    .ok_or_else(|| internal("Gemma3 step bound omits reserved cache overlap"))?;
                sum(&[
                    workspace,
                    p.output_bytes,
                    product(as_u64(token_bytes)?, 8)?,
                    16 * 1024,
                ])
            }
            Self::Phi(c) => {
                let p = c
                    .step_payload_plan(chunk, capacity)
                    .map_err(|e| execution_error("plan Phi step", e.to_string()))?;
                sum(&[
                    p.work_bytes,
                    p.output_bytes,
                    product(as_u64(token_bytes)?, 8)?,
                    16 * 1024,
                ])
            }
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
            Self::Gemma4(c) => c
                .weight_payload_plan()
                .map(|p| p.largest_upload_bytes)
                .map_err(|e| model_error(e, "plan Gemma 4 upload")),
            Self::Gemma3(c) => c
                .weight_payload_plan()
                .map(|p| p.largest_upload_bytes)
                .map_err(|e| execution_error("plan Gemma3 upload", e.to_string())),
            Self::Phi(c) => c
                .weight_payload_plan()
                .map(|p| p.largest_upload_bytes)
                .map_err(|e| execution_error("plan Phi upload", e.to_string())),
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
/// Trained Ornith function/parameter tool protocol with its assistant history.
pub const ORNITH_TOOL_TEMPLATE: &str = concat!(
    "{% set ornith = true %}",
    include_str!("hybrid_tools.jinja")
);
/// Trained upstream Qwen3.5 function/parameter tool protocol.
pub const QWEN35_TOOL_TEMPLATE: &str = concat!(
    "{% set ornith = false %}",
    include_str!("hybrid_tools.jinja")
);
/// Combined full-attention KV, convolution history and recurrent matrix state.
pub const QWEN35_CACHE_REPRESENTATION: &str = "qwen35.mlx-f32.full-kv-and-recurrent.v1";

/// Publisher Phi role framing; tool mode has a separate protocol contract.
pub const PHI_TEXT_TEMPLATE: &str = include_str!("phi_text.jinja");
/// Phi function-list protocol with publisher role/end framing.
pub const PHI_TOOL_TEMPLATE: &str = include_str!("phi_tools.jinja");
/// Fixed long-factor SuScaledRoPE is part of persistent state identity.
pub const PHI_CACHE_REPRESENTATION: &str = "phi.mlx-f32.full.bhtd.su-long-fixed.v1";

fn phi_cache_bound(
    c: &inference_infra::decoder::phi::PhiConfig,
    capacity: usize,
) -> Result<u64, EngineError> {
    c.cache_spec(capacity)
        .and_then(|spec| Ok(spec.payload_plan()?.update_overlap_bytes))
        .map_err(|e| execution_error("plan Phi KV overlap", e.to_string()))
}

/// Gemma 3 trained text turns with system instructions in the first user turn.
pub const GEMMA3_TEXT_TEMPLATE: &str = include_str!("gemma3_text.jinja");
/// Mixed local/full KV; absolute offsets survive local eviction.
pub const GEMMA3_CACHE_REPRESENTATION: &str = "gemma3.mlx-f32.full-and-sliding.v1";

/// Gemma one-call tool turns, including bounded standalone XML fences.
pub const GEMMA3_TOOL_TEMPLATE: &str = include_str!("gemma3_tools.jinja");

/// Trained Ministral chat boundaries for a text-only conversation.
pub const MINISTRAL3_TEXT_TEMPLATE: &str = include_str!("ministral3_text.jinja");

/// Distinct tool rendering identity for Ministral's trained protocol.
pub const MINISTRAL3_TOOL_TEMPLATE: &str = concat!(
    "{% set ministral_tools = true %}",
    include_str!("ministral3_text.jinja")
);

/// Publisher Gemma 4 text turns with thinking disabled.
pub const GEMMA4_TEXT_TEMPLATE: &str = include_str!("gemma4_text.jinja");

/// Publisher Gemma 4 declarations and linked call/result history.
pub const GEMMA4_TOOL_TEMPLATE: &str = include_str!("gemma4_tools.jinja");

/// Publisher Coder chat serialization in the native history representation.
pub const QWEN3_CODER_TEXT_TEMPLATE: &str = concat!(
    "{% set coder_tools = false %}",
    include_str!("qwen3_coder.jinja")
);
/// Coder chat serialization with the linked XML tool protocol enabled.
pub const QWEN3_CODER_TOOL_TEMPLATE: &str = concat!(
    "{% set coder_tools = true %}",
    include_str!("qwen3_coder.jinja")
);
