//! Both native position transforms in their trained order.
use super::{MinistralConfig, config::invalid};
use crate::decoder::mlx_model::MlxDecoderResult;
use mlx_ffi::{Array, Stream, attention, ops};

/// Apply full-head YaRN to Q/K and the absolute-position multiplier only to Q.
/// Offsets come from the cache owner, never the current chunk length.
pub fn apply_position(
    query: &Array,
    key: &Array,
    config: &MinistralConfig,
    offset: usize,
    stream: &Stream,
) -> MlxDecoderResult<(Array, Array)> {
    config.validate()?;
    let q = query.shape();
    let k = key.shape();
    if q.len() != 4
        || k.len() != 4
        || q[0] != 1
        || k[0] != 1
        || q[1] != config.query_heads
        || k[1] != config.kv_heads
        || q[2] == 0
        || q[2] != k[2]
        || q[3] != config.head_dim
        || k[3] != config.head_dim
        || offset
            .checked_add(q[2])
            .is_none_or(|end| end > config.context)
    {
        return Err(invalid("invalid Ministral position input or absolute offset").into());
    }
    rotate_query_key(
        query,
        key,
        &config.yarn,
        config.query_scaling_beta,
        offset,
        stream,
    )
}

pub(crate) fn rotate_query_key(
    query: &Array,
    key: &Array,
    yarn: &attention::YarnParameters,
    query_beta: f32,
    offset: usize,
    stream: &Stream,
) -> MlxDecoderResult<(Array, Array)> {
    let frequencies = attention::yarn_frequencies(yarn, stream)?;
    let amplitude = yarn.rotary_amplitude();
    let rotate = |input: &Array| -> MlxDecoderResult<Array> {
        let scaled = if amplitude == 1.0 {
            input.clone()
        } else {
            ops::scale(input, amplitude, stream)?
        };
        Ok(attention::rope_with_frequencies(
            &scaled,
            &frequencies,
            offset,
            stream,
        )?)
    };
    let query = rotate(query)?;
    let key = rotate(key)?;
    let query =
        attention::position_query_scale(&query, offset, yarn.original_context, query_beta, stream)?;
    Ok((query, key))
}
