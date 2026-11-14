//! Checked logical buffer plans shared by native execution and engine admission.
use super::config::{Qwen35Config, invalid};
use crate::{decoder::mlx_model::MlxDecoderResult, kv_cache::KvCacheSpec};

fn add(a: u64, b: u64) -> MlxDecoderResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("hybrid payload overflow").into())
}
fn mul(a: u64, b: u64) -> MlxDecoderResult<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("hybrid payload overflow").into())
}

/// Full-cache overlap and recurrent-state overlap are independent components.
/// Allocator padding, weights and step work are not included here.
#[derive(Clone, Copy, Debug)]
pub struct CachePayloadPlan {
    pub full: KvCacheSpec,
    pub full_overlap_bytes: u64,
    pub recurrent_overlap_bytes: u64,
    pub total_overlap_bytes: u64,
}

/// Output and intermediate buffers are separately charged, with no allocation.
#[derive(Clone, Copy, Debug)]
pub struct StepPayloadPlan {
    pub output_bytes: u64,
    pub work_bytes: u64,
}

impl Qwen35Config {
    pub fn cache_payload_plan(&self, capacity: usize) -> MlxDecoderResult<CachePayloadPlan> {
        self.validate()?;
        if capacity == 0 || capacity > self.context {
            return Err(invalid("invalid Qwen3.5 cache capacity").into());
        }
        let recurrent_layers = self.linear_layers.iter().filter(|&&v| v).count();
        let full = KvCacheSpec {
            layers: self.linear_layers.len() - recurrent_layers,
            batch: 1,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            capacity_tokens: capacity,
        };
        let full_overlap_bytes = full.payload_plan()?.update_overlap_bytes;
        let recurrent_overlap_bytes = mul(
            mul(recurrent_layers as u64, self.linear.state_payload_bytes()?)?,
            2,
        )?;
        Ok(CachePayloadPlan {
            full,
            full_overlap_bytes,
            recurrent_overlap_bytes,
            total_overlap_bytes: add(full_overlap_bytes, recurrent_overlap_bytes)?,
        })
    }

    /// Plan a chunk ending at `end`, including its preceding cached positions.
    /// Layers evaluate sequentially; only one layer's intermediate work is live.
    pub fn step_payload_plan(
        &self,
        tokens: usize,
        end: usize,
        all_logits: bool,
    ) -> MlxDecoderResult<StepPayloadPlan> {
        self.validate()?;
        if tokens == 0 || end < tokens || end > self.context {
            return Err(invalid("invalid Qwen3.5 step geometry").into());
        }
        let t = tokens as u64;
        let width = add(
            add(
                mul(8, self.hidden as u64)?,
                mul(3, self.intermediate as u64)?,
            )?,
            mul(mul(8, self.query_heads as u64)?, self.head_dim as u64)?,
        )?;
        let scores = mul(mul(t, end as u64)?, self.query_heads as u64)?;
        let dense = mul(add(mul(t, width)?, scores)?, 4)?;
        // Include both the residual/MLP layer graph and recurrent block
        // temporaries. Counting state again here is deliberately conservative.
        let work_bytes = add(dense, self.linear.step_payload_bytes(tokens)?)?;
        let output_bytes = mul(
            mul(if all_logits { t } else { 1 }, self.vocabulary as u64)?,
            4,
        )?;
        Ok(StepPayloadPlan {
            output_bytes,
            work_bytes,
        })
    }
}
