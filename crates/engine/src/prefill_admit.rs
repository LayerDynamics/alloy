//! Prefill admit-line checks and adaptive chunk shrink (M5–M6).
//!
//! Predict the peak of `usage + chunk_workspace`. When that exceeds the
//! live admit line, shrink the prefill chunk down to
//! [`crate::prefill_admit::DEFAULT_PREFILL_MIN_CHUNK_TOKENS`] (or a caller floor). If the floor
//! still does not fit, reject with a typed memory error that names the
//! binding component when known.

use crate::error::{EngineError, MemoryLimitReason};
use crate::types::DegradationOption;
use std::num::NonZeroUsize;

/// oMLX default floor for adaptive prefill chunk shrink.
pub const DEFAULT_PREFILL_MIN_CHUNK_TOKENS: usize = 256;

/// Inputs for one adaptive prefill decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillAdmitRequest {
    /// Backend / plan preferred chunk size.
    pub requested_chunk_tokens: NonZeroUsize,
    /// Smallest chunk the family will accept (default 256).
    pub min_chunk_tokens: NonZeroUsize,
    /// Current process usage against the live ceiling.
    pub usage_bytes: u64,
    /// Prefill admit line (`min(hard_ceiling, hard_watermark, prefill_headroom)`).
    pub admit_line_bytes: u64,
    /// Workspace bytes predicted for `requested_chunk_tokens`.
    pub workspace_bytes_at_requested: u64,
    /// Ceiling binding name for reject messages (`static` / `dynamic` / `metal_cap`).
    pub binding: Option<&'static str>,
}

/// Successful adaptive selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillAdmitDecision {
    /// Chunk size to use for this prefill.
    pub chunk_tokens: NonZeroUsize,
    /// Workspace bytes to reserve for that chunk (scaled from the request).
    pub workspace_bytes: u64,
    /// True when the chunk was reduced below the requested size.
    pub shrunk: bool,
}

/// Scale workspace linearly with chunk size (matches `workspace_bound`’s
/// dominant `t = chunk_tokens` terms for planning purposes).
pub fn scale_workspace_bytes(
    workspace_at_requested: u64,
    requested_chunk: NonZeroUsize,
    selected_chunk: NonZeroUsize,
) -> Result<u64, EngineError> {
    if selected_chunk == requested_chunk {
        return Ok(workspace_at_requested);
    }
    let requested = requested_chunk.get() as u128;
    let selected = selected_chunk.get() as u128;
    let scaled = (workspace_at_requested as u128)
        .checked_mul(selected)
        .and_then(|value| value.checked_div(requested))
        .ok_or_else(|| EngineError::Internal {
            message: "prefill workspace scale overflow".to_owned(),
        })?;
    u64::try_from(scaled).map_err(|_| EngineError::Internal {
        message: "prefill workspace scale exceeds u64".to_owned(),
    })
}

/// Choose a prefill chunk that fits under the admit line, or reject.
///
/// When `admit_line_bytes == 0` the guard is disabled and the requested
/// chunk is returned unchanged.
pub fn select_prefill_chunk(
    request: PrefillAdmitRequest,
) -> Result<PrefillAdmitDecision, EngineError> {
    let min = NonZeroUsize::new(
        request
            .min_chunk_tokens
            .get()
            .min(request.requested_chunk_tokens.get()),
    )
    .expect("min of two NonZeroUsize values is nonzero");
    let requested = request.requested_chunk_tokens;

    if request.admit_line_bytes == 0 {
        return Ok(PrefillAdmitDecision {
            chunk_tokens: requested,
            workspace_bytes: request.workspace_bytes_at_requested,
            shrunk: false,
        });
    }

    let mut chunk = requested;
    loop {
        let workspace =
            scale_workspace_bytes(request.workspace_bytes_at_requested, requested, chunk)?;
        let peak =
            request
                .usage_bytes
                .checked_add(workspace)
                .ok_or_else(|| EngineError::Internal {
                    message: "prefill peak arithmetic overflow".to_owned(),
                })?;
        if peak <= request.admit_line_bytes {
            return Ok(PrefillAdmitDecision {
                chunk_tokens: chunk,
                workspace_bytes: workspace,
                shrunk: chunk.get() < requested.get(),
            });
        }
        if chunk.get() <= min.get() {
            let binding = request.binding.unwrap_or("admit_line");
            return Err(prefill_admit_rejected(
                binding,
                peak,
                request.admit_line_bytes,
                min.get(),
            ));
        }
        let next = (chunk.get() / 2).max(min.get());
        chunk = NonZeroUsize::new(next).expect("min chunk is nonzero");
    }
}

fn prefill_admit_rejected(
    binding: &'static str,
    required_bytes: u64,
    available_bytes: u64,
    min_chunk_tokens: usize,
) -> EngineError {
    // DegradationOption::ReducePrefillChunk documents the recovery action;
    // callers that build MemoryEstimate can surface it explicitly.
    let _ = (DegradationOption::ReducePrefillChunk, min_chunk_tokens);
    EngineError::MemoryLimitExceeded {
        operation: match binding {
            "static" => "prefill admit line (binding=static)",
            "dynamic" => "prefill admit line (binding=dynamic)",
            "metal_cap" => "prefill admit line (binding=metal_cap)",
            _ => "prefill admit line",
        },
        required_bytes,
        available_bytes,
        reason: MemoryLimitReason::AllocatorCeiling {
            ceiling_bytes: available_bytes,
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/prefill_admit_tests.rs"]
mod tests;
