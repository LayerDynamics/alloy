//! Checked tensor payload plans shared by native Phi execution and admission.
use super::{PhiConfig, config::invalid};
use crate::{
    decoder::{loader::TensorRequest, mlx_model::MlxDecoderResult},
    kv_cache::KvCacheSpec,
};

fn add(a: u64, b: u64) -> MlxDecoderResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Phi payload overflow").into())
}
fn mul(a: u64, b: u64) -> MlxDecoderResult<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("Phi payload overflow").into())
}

#[derive(Clone, Copy, Debug)]
pub struct WeightPayloadPlan {
    pub payload_bytes: u64,
    pub largest_upload_bytes: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct StepPayloadPlan {
    /// Full-position output projection, possible contiguous copy, host readback.
    pub output_bytes: u64,
    /// Conservative layer-local activations, masks and attention intermediates.
    /// Weights, KV append overlap and allocator padding are accounted separately.
    pub work_bytes: u64,
}
impl PhiConfig {
    pub fn weight_payload_plan(&self) -> MlxDecoderResult<WeightPayloadPlan> {
        let layout = self.tensor_layout()?;
        let mut payload_bytes = 0;
        let mut largest_upload_bytes = 0;
        for tensor in layout.tensors.values() {
            let (bytes, largest) = match tensor {
                TensorRequest::Float(shape) => {
                    let n = shape.iter().try_fold(1, |n, &d| mul(n, d as u64))?;
                    let bytes = mul(n, 4)?;
                    (bytes, bytes)
                }
                TensorRequest::Matrix { rows, cols } => {
                    let n = mul(*rows as u64, *cols as u64)?;
                    if let Some(group) = self.group_size {
                        let packing = 32 / self.quantization_bits as usize;
                        if !cols.is_multiple_of(group) || !cols.is_multiple_of(packing) {
                            return Err(invalid("Phi packed matrix alignment mismatch").into());
                        }
                        let packed = mul(n / packing as u64, 4)?;
                        let metadata = mul(n / group as u64, 4)?;
                        (add(packed, mul(metadata, 2)?)?, packed.max(metadata))
                    } else {
                        let bytes = mul(n, 4)?;
                        (bytes, bytes)
                    }
                }
            };
            payload_bytes = add(payload_bytes, bytes)?;
            largest_upload_bytes = largest_upload_bytes.max(largest);
        }
        Ok(WeightPayloadPlan {
            payload_bytes,
            largest_upload_bytes,
        })
    }
    pub fn cache_spec(&self, capacity: usize) -> MlxDecoderResult<KvCacheSpec> {
        self.validate()?;
        if capacity == 0 || capacity > self.context {
            return Err(invalid("invalid Phi cache capacity").into());
        }
        let spec = KvCacheSpec {
            layers: self.layers,
            batch: 1,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            capacity_tokens: capacity,
        };
        spec.payload_plan()?;
        Ok(spec)
    }
    pub fn step_payload_plan(
        &self,
        tokens: usize,
        end: usize,
    ) -> MlxDecoderResult<StepPayloadPlan> {
        self.validate()?;
        if tokens == 0 || tokens > end || end > self.context {
            return Err(invalid("invalid Phi step length").into());
        }
        let t = tokens as u64;
        let output_bytes = mul(mul(mul(t, self.vocabulary as u64)?, 4)?, 3)?;
        let layer_width = add(
            add(
                mul(self.hidden as u64, 16)?,
                mul(self.intermediate as u64, 8)?,
            )?,
            mul(self.qkv_width() as u64, 8)?,
        )?;
        let layer = mul(mul(t, layer_width)?, 4)?;
        let attention = mul(
            mul(mul(mul(t, end as u64)?, self.query_heads as u64)?, 4)?,
            4,
        )?;
        let rotary = mul(self.rotary_dim as u64, 16)?;
        Ok(StepPayloadPlan {
            output_bytes,
            work_bytes: add(add(layer, attention)?, rotary)?,
        })
    }
}
