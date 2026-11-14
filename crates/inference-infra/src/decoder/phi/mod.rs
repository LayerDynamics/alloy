//! Phi native position operations; complete model qualification is separate.
pub mod config;
pub mod model;
pub mod planning;
pub mod weights;
use crate::decoder::mlx_model::{MlxDecoderError, MlxDecoderResult};
pub use config::{PhiConfig, PhiRopeRegime};
use mlx_ffi::{Array, Stream, attention, ops};

/// Native partial LongRoPE. Scale and rotate only the leading rotary dimensions;
/// the remaining features retain their exact values. Never switch regime based
/// on a chunk's length: the model/cache owner supplies the chosen regime.
pub fn partial_longrope(
    input: &Array,
    config: &PhiConfig,
    regime: PhiRopeRegime,
    offset: usize,
    stream: &Stream,
) -> MlxDecoderResult<Array> {
    config.validate()?;
    let shape = input.shape();
    if shape.len() != 4
        || shape[0] != 1
        || ![config.query_heads, config.kv_heads].contains(&shape[1])
        || shape[2] == 0
        || shape[3] != config.head_dim
        || offset
            .checked_add(shape[2])
            .is_none_or(|end| end > config.context)
    {
        return Err(MlxDecoderError::Model(config::invalid(
            "invalid Phi rotary input or position",
        )));
    }
    let mut leading = shape.clone();
    leading[3] = config.rotary_dim;
    let prefix = ops::slice(input, &[0, 0, 0, 0], &leading, stream)?;
    let prefix = ops::scale(&prefix, config.rotary_amplitude(), stream)?;
    let factors = match regime {
        PhiRopeRegime::Short => &config.short_factors,
        PhiRopeRegime::Long => &config.long_factors,
    };
    let frequencies =
        attention::longrope_frequencies(config.rotary_dim, config.theta, factors, stream)?;
    let rotated = attention::rope_with_frequencies(&prefix, &frequencies, offset, stream)?;
    if config.rotary_dim == config.head_dim {
        return Ok(rotated);
    }
    let tail = ops::slice(input, &[0, 0, 0, config.rotary_dim], &shape, stream)?;
    Ok(ops::concatenate_axis(&rotated, &tail, 3, stream)?)
}
