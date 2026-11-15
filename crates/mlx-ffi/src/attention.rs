//! Explicit-stream RoPE and grouped-query attention.
//!
//! Native operations use actual MLX graphs on the supplied stream.
//! There is no host arithmetic fallback in the production functions.
//!
//! Operations support float32 arrays with layout:
//!
//! - Q: [batch, query_heads, query_tokens, head_dim].
//! - K/V: [batch, kv_heads, kv_tokens, head_dim].
//!
//! RoPE uses split-half pairing, full head rotation, scale 1, and an
//! explicit nonnegative absolute position offset.
//!
//! CausalSuffix validates that keys comprise the cached prefix followed
//! by exactly the query chunk. MLX's lower-right causal alignment then
//! implements query i attending through prefix_tokens + i.
//! CausalOffset also supports arbitrary query/key lengths using a native
//! Boolean mask when the lower-right causal alignment does not apply.
//! Attention accepts either an explicit finite scale or 1 / sqrt(head_dim).
//!
//! These operations do not allocate or own a persistent KV cache, select
//! model weights, perform admission, or report engine capabilities.

use crate::{
    Array, DType, Error, Stream, bindings as sys,
    ops::{OperationError, OperationResult},
};
use std::{os::raw::c_int, ptr};

/// Attention visibility for a supplied Q/K/V sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMask {
    /// Every query can attend to every supplied key.
    ///
    /// Appropriate for single-token decode when the supplied cache
    /// contains exactly the permitted prefix plus the current token.
    Full,

    /// Causal query chunk following an already processed prefix.
    ///
    /// Requires kv_tokens == prefix_tokens + query_tokens.
    ///
    /// prefix_tokens = 0 represents ordinary fresh prefill.
    CausalSuffix {
        /// Number of cached positions preceding this query chunk.
        prefix_tokens: usize,
    },

    /// Query token i may attend to keys at positions <= offset + i.
    ///
    /// Unlike CausalSuffix, this does not require the supplied keys to end
    /// at the query chunk's final position. Visibility is capped by the
    /// number of supplied keys; even usize::MAX is a valid saturated
    /// offset and does not require overflowing position arithmetic.
    CausalOffset {
        /// Absolute key position visible to the first query token.
        offset: usize,
    },
    /// Causal suffix restricted to the most recent `window_tokens` positions,
    /// including the query's own position. Keys may begin at an evicted prefix.
    SlidingCausalSuffix {
        /// Retained keys before the current query chunk.
        prefix_tokens: usize,
        /// Maximum visible positions per query, including itself; nonzero.
        window_tokens: usize,
    },
}

struct PendingArray {
    raw: sys::mlx_array,
}

impl PendingArray {
    fn new() -> Self {
        Self {
            raw: sys::mlx_array {
                ctx: ptr::null_mut(),
            },
        }
    }

    fn finish(mut self, status: c_int) -> OperationResult<Array> {
        if status != 0 {
            return Err(OperationError::Mlx(Error::Status(status)));
        }

        let raw = std::mem::replace(
            &mut self.raw,
            sys::mlx_array {
                ctx: ptr::null_mut(),
            },
        );
        Array::from_raw(raw).map_err(OperationError::from)
    }
}

impl Drop for PendingArray {
    fn drop(&mut self) {
        if !self.raw.ctx.is_null() {
            // SAFETY: this guard exclusively owns the untransferred
            // native operation output.
            unsafe {
                sys::mlx_array_free(self.raw);
            }
        }
    }
}

fn require_f32(array: &Array) -> OperationResult<()> {
    if array.dtype() != DType::Float32 {
        return Err(OperationError::InvalidInput(
            "attention operations currently require float32 arrays",
        ));
    }
    Ok(())
}

fn shape4(array: &Array) -> OperationResult<[usize; 4]> {
    require_f32(array)?;
    let shape = array.shape();

    if shape.len() != 4 || shape.contains(&0) {
        return Err(OperationError::InvalidInput(
            "expected positive [batch, heads, tokens, head_dim] dimensions",
        ));
    }

    Ok([shape[0], shape[1], shape[2], shape[3]])
}

/// Apply full-head, split-half rotary position encoding.
///
/// Input/output layout is [batch, heads, tokens, head_dim].
///
/// head_dim must be even. The first input token uses position offset;
/// subsequent tokens use consecutive absolute positions.
///
/// This uses MLX's nontraditional pairing and position scale 1. It does
/// not implement extended-context scaling or custom frequency arrays.
///
/// The returned array is lazy. The caller owns evaluation and stream
/// synchronization.
pub fn rope(input: &Array, theta: f32, offset: usize, stream: &Stream) -> OperationResult<Array> {
    rope_impl(input, theta, None, 1.0, offset, stream)
}

/// Split-half RoPE with an explicit positive position scale, matching MLX's
/// native linear scaling graph (absolute positions are scaled before rotation).
pub fn rope_scaled(
    input: &Array,
    theta: f32,
    scale: f32,
    offset: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    rope_impl(input, theta, None, scale, offset, stream)
}

/// Split-half RoPE with explicit positive frequency divisors, one per head pair.
pub fn rope_with_frequencies(
    input: &Array,
    frequencies: &Array,
    offset: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    rope_impl(input, 1.0, Some(frequencies), 1.0, offset, stream)
}

fn rope_impl(
    input: &Array,
    theta: f32,
    frequencies: Option<&Array>,
    scale: f32,
    offset: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    let [_, _, tokens, head_dim] = shape4(input)?;
    if let Some(freqs) = frequencies {
        require_f32(freqs)?;
        if freqs.shape() != [head_dim / 2] {
            return Err(OperationError::InvalidInput(
                "RoPE frequency shape mismatch",
            ));
        }
    }

    if head_dim % 2 != 0 {
        return Err(OperationError::InvalidInput(
            "split-half RoPE requires an even head dimension",
        ));
    }
    if !theta.is_finite() || theta <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        return Err(OperationError::InvalidInput(
            "RoPE theta and scale must be finite and positive",
        ));
    }

    let dims = c_int::try_from(head_dim)
        .map_err(|_| OperationError::InvalidInput("RoPE head dimension exceeds C int"))?;
    let native_offset = c_int::try_from(offset)
        .map_err(|_| OperationError::InvalidInput("RoPE position offset exceeds C int"))?;

    let final_position = offset
        .checked_add(tokens - 1)
        .ok_or(OperationError::InvalidInput(
            "RoPE position arithmetic overflow",
        ))?;
    c_int::try_from(final_position)
        .map_err(|_| OperationError::InvalidInput("RoPE final position exceeds C int"))?;

    let base = sys::mlx_optional_float {
        value: theta,
        has_value: frequencies.is_none(),
    };
    let no_frequencies = sys::mlx_array {
        ctx: ptr::null_mut(),
    };
    let mut output = PendingArray::new();

    // SAFETY: input and stream are valid borrowed handles; output is
    // writable. Dimensions and offsets were validated. A null frequency
    // array is permitted because an explicit base is supplied.
    let status = unsafe {
        sys::mlx_fast_rope(
            &mut output.raw,
            input.as_raw(),
            dims,
            false,
            base,
            scale,
            native_offset,
            frequencies.map_or(no_frequencies, Array::as_raw),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Compute actual MLX grouped-query scaled dot-product attention.
///
/// Q: [batch, query_heads, query_tokens, head_dim].
/// K/V: [batch, kv_heads, kv_tokens, head_dim].
///
/// query_heads must be divisible by kv_heads. K/V are not repeated or
/// tiled on the host. Scale is 1 / sqrt(head_dim).
///
/// CausalSuffix uses validated lower-right causal alignment. For cached
/// prefill, K/V must include the old prefix and the current query chunk.
///
/// force_fused is deliberately false: MLX may choose its supported
/// implementation on the explicitly selected stream. This is not a
/// request to silently switch to a different execution device.
///
/// Returns a lazy [batch, query_heads, query_tokens, head_dim] array.
pub fn grouped_query_attention(
    query: &Array,
    key: &Array,
    value: &Array,
    mask: AttentionMask,
    stream: &Stream,
) -> OperationResult<Array> {
    let [_, _, _, head_dim] = shape4(query)?;
    grouped_query_attention_scaled(
        query,
        key,
        value,
        1.0 / (head_dim as f32).sqrt(),
        mask,
        stream,
    )
}

/// Compute lazy grouped-query attention with a caller-supplied scale.
///
/// Layout, GQA grouping and stream selection follow
/// [`grouped_query_attention`]. Any finite scale is supported, including
/// zero and negative values. Inputs are not read back or checked for finite
/// elements; callers requiring eager validation can use
/// [`crate::ops::all_finite`].
///
/// [`AttentionMask::CausalSuffix`] retains its strict cached-prefix
/// contract. [`AttentionMask::CausalOffset`] allows arbitrary query/key
/// lengths and computes its Boolean visibility mask in MLX. Neither path
/// constructs scores, masks or repeated KV heads on the host.
pub fn grouped_query_attention_scaled(
    query: &Array,
    key: &Array,
    value: &Array,
    scale: f32,
    mask: AttentionMask,
    stream: &Stream,
) -> OperationResult<Array> {
    let [batch, query_heads, query_tokens, head_dim] = shape4(query)?;
    let [key_batch, kv_heads, kv_tokens, key_dim] = shape4(key)?;
    let value_shape = shape4(value)?;

    if value_shape != [key_batch, kv_heads, kv_tokens, key_dim] {
        return Err(OperationError::InvalidInput(
            "attention key and value shapes must match",
        ));
    }
    if batch != key_batch || head_dim != key_dim {
        return Err(OperationError::InvalidInput(
            "attention batch sizes and head dimensions must match",
        ));
    }
    if query_heads % kv_heads != 0 {
        return Err(OperationError::InvalidInput(
            "query-head count must be divisible by KV-head count",
        ));
    }
    if !scale.is_finite() {
        return Err(OperationError::InvalidInput(
            "attention scale must be finite",
        ));
    }

    let mut visibility = None;
    let mask_mode = match mask {
        AttentionMask::Full => c"".as_ptr(),
        AttentionMask::SlidingCausalSuffix {
            prefix_tokens,
            window_tokens,
        } => {
            if window_tokens == 0 || prefix_tokens.checked_add(query_tokens) != Some(kv_tokens) {
                return Err(OperationError::InvalidInput(
                    "invalid sliding causal suffix",
                ));
            }
            let upper = causal_visibility(query_tokens, kv_tokens, prefix_tokens, stream)?;
            if window_tokens >= kv_tokens {
                visibility = Some(upper);
            } else {
                let diagonal = i32::try_from(prefix_tokens)
                    .ok()
                    .zip(i32::try_from(window_tokens).ok())
                    .map(|(p, w)| p - w)
                    .ok_or(OperationError::InvalidInput(
                        "sliding diagonal exceeds C int",
                    ))?;
                let rows = i32::try_from(query_tokens)
                    .map_err(|_| OperationError::InvalidInput("query count exceeds C int"))?;
                let columns = i32::try_from(kv_tokens)
                    .map_err(|_| OperationError::InvalidInput("key count exceeds C int"))?;
                let mut old = PendingArray::new();
                // SAFETY: dimensions/diagonal checked; output and stream are valid.
                let status = unsafe {
                    sys::mlx_tri(
                        &mut old.raw,
                        rows,
                        columns,
                        diagonal,
                        DType::Bool as u32,
                        stream.as_raw(),
                    )
                };
                let old = old.finish(status)?;
                let mut recent = PendingArray::new();
                // SAFETY: valid Boolean input and explicit stream.
                let status =
                    unsafe { sys::mlx_logical_not(&mut recent.raw, old.as_raw(), stream.as_raw()) };
                let recent = recent.finish(status)?;
                let mut allowed = PendingArray::new();
                // SAFETY: equal-shaped Boolean arrays and valid stream/output.
                let status = unsafe {
                    sys::mlx_logical_and(
                        &mut allowed.raw,
                        upper.as_raw(),
                        recent.as_raw(),
                        stream.as_raw(),
                    )
                };
                visibility = Some(allowed.finish(status)?);
            }
            c"".as_ptr()
        }
        AttentionMask::CausalSuffix { prefix_tokens } => {
            let expected =
                prefix_tokens
                    .checked_add(query_tokens)
                    .ok_or(OperationError::InvalidInput(
                        "causal attention position arithmetic overflow",
                    ))?;

            if expected != kv_tokens {
                return Err(OperationError::InvalidInput(
                    "causal keys must contain cached prefix plus query chunk",
                ));
            }

            c"causal".as_ptr()
        }
        AttentionMask::CausalOffset { offset } => {
            if offset >= kv_tokens - 1 {
                // Every key is already visible to the first query. This
                // also handles huge offsets without adding query indices.
                c"".as_ptr()
            } else if offset.checked_add(query_tokens) == Some(kv_tokens) {
                // Preserve MLX's fast lower-right causal representation
                // whenever it has exactly the requested visibility.
                c"causal".as_ptr()
            } else {
                visibility = Some(causal_visibility(query_tokens, kv_tokens, offset, stream)?);
                c"".as_ptr()
            }
        }
    };

    let no_array = sys::mlx_array {
        ctx: ptr::null_mut(),
    };
    let mask_array = visibility.as_ref().map_or(no_array, Array::as_raw);
    let mut output = PendingArray::new();

    // SAFETY: all arrays and the stream are valid borrowed handles.
    // Shapes and GQA grouping were validated. mask_mode points to a
    // static NUL-terminated string. An optional Boolean mask is retained
    // for the call and captured by MLX's graph. Optional sinks are absent.
    let status = unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            &mut output.raw,
            query.as_raw(),
            key.as_raw(),
            value.as_raw(),
            scale,
            mask_mode,
            mask_array,
            no_array,
            false,
            stream.as_raw(),
        )
    };

    output.finish(status)
}

fn causal_visibility(
    query_tokens: usize,
    kv_tokens: usize,
    offset: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    let rows = c_int::try_from(query_tokens)
        .map_err(|_| OperationError::InvalidInput("query-token count exceeds C int"))?;
    let columns = c_int::try_from(kv_tokens)
        .map_err(|_| OperationError::InvalidInput("key-token count exceeds C int"))?;
    let diagonal = c_int::try_from(offset)
        .map_err(|_| OperationError::InvalidInput("causal offset exceeds C int"))?;
    let mut output = PendingArray::new();

    // SAFETY: dimensions and diagonal fit C int, and the stream/output
    // are valid. MLX tri creates a Boolean [query_tokens, kv_tokens]
    // array with key_index <= query_index + offset. Attention broadcasts
    // it over batch and heads; every query can see at least key zero.
    let status = unsafe {
        sys::mlx_tri(
            &mut output.raw,
            rows,
            columns,
            diagonal,
            DType::Bool as u32,
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Native LongRoPE frequency denominators, preserving the upstream float32
/// exponentiation and multiplication graph on the caller's stream.
pub fn longrope_frequencies(
    dims: usize,
    base: f32,
    factors: &[f32],
    stream: &Stream,
) -> OperationResult<Array> {
    if dims == 0
        || !dims.is_multiple_of(2)
        || dims > 4096
        || factors.len() != dims / 2
        || !base.is_finite()
        || base <= 0.0
        || factors.iter().any(|n| !n.is_finite() || *n <= 0.0)
    {
        return Err(OperationError::InvalidInput("invalid LongRoPE frequencies"));
    }
    let shape = &[(dims / 2) as i32];
    let bases = Array::from_f32_slice(&vec![base; dims / 2], shape)?;
    let exponents = Array::from_f32_slice(
        &(0..dims / 2)
            .map(|i| (2 * i) as f32 / dims as f32)
            .collect::<Vec<_>>(),
        shape,
    )?;
    let factors = Array::from_f32_slice(factors, shape)?;
    let mut output = PendingArray::new();
    // SAFETY: live equal-shaped float32 arrays and explicit stream; owned output.
    let status = unsafe {
        sys::mlx_power(
            &mut output.raw,
            bases.as_raw(),
            exponents.as_raw(),
            stream.as_raw(),
        )
    };
    let frequencies = output.finish(status)?;
    let mut output = PendingArray::new();
    // SAFETY: live equal-shaped float32 arrays and explicit stream; owned output.
    let status = unsafe {
        sys::mlx_multiply(
            &mut output.raw,
            frequencies.as_raw(),
            factors.as_raw(),
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Construct Llama 3.1 frequency divisors using eager float32 MLX arithmetic.
/// Keeping this calculation on the selected stream avoids host libm rounding
/// differences becoming amplified by long rotary offsets.
pub fn llama3_rope_frequencies(
    dims: usize,
    base: f32,
    factor: f32,
    low: f32,
    high: f32,
    original: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    if dims == 0
        || !dims.is_multiple_of(2)
        || dims > i32::MAX as usize
        || ![base, factor, low, high]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        || factor < 1.0
        || high <= low
        || original == 0
    {
        return Err(OperationError::InvalidInput("invalid Llama rotary scaling"));
    }
    type Binary = unsafe extern "C" fn(
        *mut sys::mlx_array,
        sys::mlx_array,
        sys::mlx_array,
        sys::mlx_stream,
    ) -> c_int;
    let binary = |op: Binary, a: &Array, b: &Array| -> OperationResult<Array> {
        let mut out = PendingArray::new();
        // SAFETY: valid live borrowed operands and stream; output exclusively owned.
        let status = unsafe { op(&mut out.raw, a.as_raw(), b.as_raw(), stream.as_raw()) };
        out.finish(status)
    };
    let constant = |x: f32| {
        Array::from_f32_slice(&vec![x; dims / 2], &[(dims / 2) as i32])
            .map_err(OperationError::from)
    };
    let select = |condition: &Array, yes: &Array, no: &Array| -> OperationResult<Array> {
        let mut out = PendingArray::new();
        // SAFETY: valid same-shape native operands, Boolean condition, owned output.
        let status = unsafe {
            sys::mlx_where(
                &mut out.raw,
                condition.as_raw(),
                yes.as_raw(),
                no.as_raw(),
                stream.as_raw(),
            )
        };
        out.finish(status)
    };
    let exponents = Array::from_f32_slice(
        &(0..dims / 2)
            .map(|i| (2 * i) as f32 / dims as f32)
            .collect::<Vec<_>>(),
        &[(dims / 2) as i32],
    )?;
    let frequencies = binary(sys::mlx_power, &constant(base)?, &exponents)?;
    let wavelengths = binary(
        sys::mlx_multiply,
        &constant(std::f32::consts::TAU)?,
        &frequencies,
    )?;
    let low_wave = constant(original as f32 / low)?;
    let high_wave = constant(original as f32 / high)?;
    let low_mask = binary(sys::mlx_greater, &wavelengths, &low_wave)?;
    let scaled = select(
        &low_mask,
        &binary(sys::mlx_multiply, &frequencies, &constant(factor)?)?,
        &frequencies,
    )?;
    let smooth = binary(
        sys::mlx_divide,
        &binary(
            sys::mlx_subtract,
            &binary(sys::mlx_divide, &constant(original as f32)?, &wavelengths)?,
            &constant(low)?,
        )?,
        &constant(high - low)?,
    )?;
    let denominator = binary(
        sys::mlx_add,
        &binary(
            sys::mlx_divide,
            &binary(sys::mlx_subtract, &constant(1.0)?, &smooth)?,
            &constant(factor)?,
        )?,
        &smooth,
    )?;
    let interpolated = binary(sys::mlx_divide, &scaled, &denominator)?;
    let above_high = binary(sys::mlx_greater, &wavelengths, &high_wave)?;
    let below_low = binary(sys::mlx_less, &wavelengths, &low_wave)?;
    let within = select(&below_low, &interpolated, &scaled)?;
    select(&above_high, &within, &scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, DeviceType};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(actual.is_finite(), "non-finite result at {index}");
            assert!(expected.is_finite());

            let tolerance = 2e-4 + 2e-4 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual}, expected={expected}",
            );
        }
    }

    fn read_contiguous(array: &Array, stream: &Stream) -> Vec<f32> {
        let mut output = PendingArray::new();

        // SAFETY: valid input and stream; writable owned output.
        let status =
            unsafe { sys::mlx_contiguous(&mut output.raw, array.as_raw(), false, stream.as_raw()) };
        let contiguous = output.finish(status).expect("row-major output");

        contiguous.eval().expect("evaluate native graph");
        stream.synchronize().expect("finish explicit stream");
        contiguous
            .data_f32()
            .expect("evaluated f32 output")
            .to_vec()
    }

    fn data(elements: usize, shift: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| (((index * 7 + shift) % 29) as f32 - 14.0) / 16.0)
            .collect()
    }

    fn rope_reference(input: &[f32], shape: [usize; 4], theta: f32, offset: usize) -> Vec<f32> {
        let [batch, heads, tokens, width] = shape;
        let half = width / 2;
        let mut output = input.to_vec();

        for batch_index in 0..batch {
            for head in 0..heads {
                for token in 0..tokens {
                    let base = ((batch_index * heads + head) * tokens + token) * width;
                    let position = (offset + token) as f64;

                    for dimension in 0..half {
                        let frequency =
                            f64::from(theta).powf(-(2.0 * dimension as f64) / width as f64);
                        let (sin, cos) = (position * frequency).sin_cos();
                        let low = f64::from(input[base + dimension]);
                        let high = f64::from(input[base + half + dimension]);

                        output[base + dimension] = (low * cos - high * sin) as f32;
                        output[base + half + dimension] = (low * sin + high * cos) as f32;
                    }
                }
            }
        }

        output
    }

    fn attention_reference(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        query_shape: [usize; 4],
        key_shape: [usize; 4],
        mask: AttentionMask,
        scale: f32,
    ) -> Vec<f32> {
        let [batch, query_heads, query_tokens, width] = query_shape;
        let [_, kv_heads, kv_tokens, _] = key_shape;
        let group = query_heads / kv_heads;
        let scale = f64::from(scale);
        let mut output = vec![0.0; query.len()];

        for batch_index in 0..batch {
            for head in 0..query_heads {
                let kv_head = head / group;

                for token in 0..query_tokens {
                    let query_base =
                        ((batch_index * query_heads + head) * query_tokens + token) * width;

                    let key_count = match mask {
                        AttentionMask::Full => kv_tokens,
                        AttentionMask::CausalSuffix { prefix_tokens } => prefix_tokens + token + 1,
                        AttentionMask::SlidingCausalSuffix { prefix_tokens, .. } => {
                            prefix_tokens + token + 1
                        }
                        AttentionMask::CausalOffset { offset } => offset
                            .saturating_add(token)
                            .saturating_add(1)
                            .min(kv_tokens),
                    };
                    let first_key = match mask {
                        AttentionMask::SlidingCausalSuffix { window_tokens, .. } => {
                            key_count.saturating_sub(window_tokens)
                        }
                        _ => 0,
                    };

                    let mut scores = Vec::with_capacity(key_count);

                    for key_token in first_key..key_count {
                        let key_base =
                            ((batch_index * kv_heads + kv_head) * kv_tokens + key_token) * width;

                        let dot = (0..width)
                            .map(|dimension| {
                                f64::from(query[query_base + dimension])
                                    * f64::from(key[key_base + dimension])
                            })
                            .sum::<f64>();

                        scores.push(dot * scale);
                    }

                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                    for score in &mut scores {
                        *score = (*score - maximum).exp();
                    }
                    let denominator = scores.iter().sum::<f64>();

                    for dimension in 0..width {
                        let mut result = 0.0_f64;

                        for (relative_key, score) in scores.iter().enumerate() {
                            let key_token = first_key + relative_key;
                            let value_base = ((batch_index * kv_heads + kv_head) * kv_tokens
                                + key_token)
                                * width;

                            result +=
                                score / denominator * f64::from(value[value_base + dimension]);
                        }

                        output[query_base + dimension] = result as f32;
                    }
                }
            }
        }

        output
    }

    fn array(input: &[f32], shape: [usize; 4]) -> Array {
        let shape: Vec<i32> = shape
            .into_iter()
            .map(|value| i32::try_from(value).expect("test dimension"))
            .collect();

        Array::from_f32_slice(input, &shape).expect("native test array")
    }

    fn run_rope(stream: &Stream) {
        let shape = [2, 3, 4, 8];
        let input = data(shape.into_iter().product(), 1);
        let native = array(&input, shape);

        for offset in [0, 5] {
            let result = rope(&native, 10_000.0, offset, stream).expect("actual MLX RoPE");
            let actual = read_contiguous(&result, stream);
            let expected = rope_reference(&input, shape, 10_000.0, offset);

            assert_close(&actual, &expected);
        }
    }

    fn run_attention_case(
        stream: &Stream,
        query_tokens: usize,
        kv_tokens: usize,
        mask: AttentionMask,
    ) {
        let query_shape = [2, 4, query_tokens, 8];
        let key_shape = [2, 2, kv_tokens, 8];

        let query = data(query_shape.into_iter().product(), 3);
        let key = data(key_shape.into_iter().product(), 7);
        let value = data(key_shape.into_iter().product(), 11);

        let q = array(&query, query_shape);
        let k = array(&key, key_shape);
        let v = array(&value, key_shape);

        let result = grouped_query_attention(&q, &k, &v, mask, stream)
            .expect("actual MLX grouped-query attention");

        assert_eq!(result.shape(), query_shape.to_vec());

        let actual = read_contiguous(&result, stream);
        let expected = attention_reference(
            &query,
            &key,
            &value,
            query_shape,
            key_shape,
            mask,
            1.0 / (query_shape[3] as f32).sqrt(),
        );

        assert_close(&actual, &expected);
    }

    fn run_suite(kind: DeviceType) {
        let device = Device::new(kind, 0).expect("explicit device");
        assert!(
            device.is_available().expect("device availability"),
            "requested execution device is unavailable: {kind:?}",
        );

        let stream = Stream::new(&device).expect("explicit stream");
        assert_eq!(
            stream
                .device()
                .expect("stream device")
                .kind()
                .expect("kind"),
            kind,
        );

        run_rope(&stream);

        // Fresh causal prefill.
        run_attention_case(
            &stream,
            3,
            3,
            AttentionMask::CausalSuffix { prefix_tokens: 0 },
        );

        // Two new queries following three cached token positions.
        run_attention_case(
            &stream,
            2,
            5,
            AttentionMask::CausalSuffix { prefix_tokens: 3 },
        );

        // Single-token decode over the complete allowed cache.
        run_attention_case(&stream, 1, 5, AttentionMask::Full);
    }

    #[test]
    fn mlx_cpu_rope_and_gqa_match_reference() {
        run_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_rope_and_gqa_match_reference() {
        // No silent skip or CPU substitution on the Apple-Silicon lane.
        run_suite(DeviceType::Gpu);
    }

    fn run_scaled_attention_suite(kind: DeviceType) {
        let device = Device::new(kind, 0).expect("explicit device");
        assert!(device.is_available().expect("device availability"));
        let stream = Stream::new(&device).expect("explicit stream");
        assert_eq!(stream.device().expect("device").kind().expect("kind"), kind);

        // Three queries inside a longer key sequence, a query chunk
        // extending beyond the keys, and single-token decode. These
        // cases cannot all use lower-right causal alignment.
        for (query_tokens, kv_tokens) in [(3, 5), (6, 3), (1, 5)] {
            let query_shape = [2, 6, query_tokens, 8];
            let key_shape = [2, 2, kv_tokens, 8];
            let query = data(query_shape.into_iter().product(), 2);
            let key = data(key_shape.into_iter().product(), 5);
            let value = data(key_shape.into_iter().product(), 13);
            let q = array(&query, query_shape);
            let k = array(&key, key_shape);
            let v = array(&value, key_shape);

            for scale in [0.0, 0.37, -0.6, 2.75] {
                for mask in [
                    AttentionMask::Full,
                    AttentionMask::CausalOffset { offset: 0 },
                    AttentionMask::CausalOffset { offset: 1 },
                    AttentionMask::CausalOffset { offset: 2 },
                    AttentionMask::CausalOffset { offset: usize::MAX },
                ] {
                    let result = grouped_query_attention_scaled(&q, &k, &v, scale, mask, &stream)
                        .expect("native scaled GQA");
                    assert_eq!(result.shape(), query_shape);
                    assert!(!result.is_available().expect("lazy graph availability"));

                    let expected = attention_reference(
                        &query,
                        &key,
                        &value,
                        query_shape,
                        key_shape,
                        mask,
                        scale,
                    );
                    assert_close(&read_contiguous(&result, &stream), &expected);
                }
            }
        }
    }

    #[test]
    fn mlx_cpu_scaled_gqa_and_general_causal_offsets_match_reference() {
        run_scaled_attention_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_scaled_gqa_and_general_causal_offsets_match_reference() {
        // The required Apple-Silicon lane must execute the requested GPU.
        run_scaled_attention_suite(DeviceType::Gpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn scaled_gqa_consumes_shared_arrays_across_cpu_and_metal_streams() {
        let cpu_device = Device::new(DeviceType::Cpu, 0).expect("CPU");
        let metal_device = Device::new(DeviceType::Gpu, 0).expect("Metal");
        assert!(metal_device.is_available().expect("required Metal device"));
        let cpu = Stream::new(&cpu_device).expect("CPU stream");
        let metal = Stream::new(&metal_device).expect("Metal stream");
        assert_eq!(
            metal
                .device()
                .expect("Metal stream device")
                .kind()
                .expect("kind"),
            DeviceType::Gpu
        );

        let query_shape = [1, 4, 3, 8];
        let key_shape = [1, 2, 5, 8];
        let query = data(query_shape.into_iter().product(), 3);
        let key = data(key_shape.into_iter().product(), 7);
        let value = data(key_shape.into_iter().product(), 11);
        // Both devices consume these exact native array handles.
        let q = array(&query, query_shape);
        let k = array(&key, key_shape);
        let v = array(&value, key_shape);
        let mask = AttentionMask::CausalOffset { offset: 1 };
        let on_cpu =
            grouped_query_attention_scaled(&q, &k, &v, -0.25, mask, &cpu).expect("CPU graph");
        let on_metal =
            grouped_query_attention_scaled(&q, &k, &v, -0.25, mask, &metal).expect("Metal graph");
        let expected =
            attention_reference(&query, &key, &value, query_shape, key_shape, mask, -0.25);

        // Read each result on the other device's stream. The dependency
        // graph must complete the producing stream before host access.
        assert_close(&read_contiguous(&on_cpu, &metal), &expected);
        assert_close(&read_contiguous(&on_metal, &cpu), &expected);
        cpu.synchronize().expect("CPU completion");
        metal.synchronize().expect("Metal completion");
    }

    #[test]
    fn scaled_gqa_rejects_non_finite_scale() {
        let device = Device::new(DeviceType::Cpu, 0).expect("CPU");
        let stream = Stream::new(&device).expect("CPU stream");
        let q = array(&[0.0; 32], [1, 2, 2, 8]);
        let k = array(&[0.0; 24], [1, 1, 3, 8]);
        let v = array(&[0.0; 24], [1, 1, 3, 8]);

        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                grouped_query_attention_scaled(&q, &k, &v, scale, AttentionMask::Full, &stream),
                Err(OperationError::InvalidInput(_)),
            ));
        }
    }

    #[test]
    fn causal_suffix_rejects_incorrect_cached_prefix_length() {
        let device = Device::new(DeviceType::Cpu, 0).expect("CPU");
        let stream = Stream::new(&device).expect("CPU stream");

        let q = array(&[0.0; 32], [1, 2, 2, 8]);
        let k = array(&[0.0; 40], [1, 1, 5, 8]);
        let v = array(&[0.0; 40], [1, 1, 5, 8]);

        assert!(matches!(
            grouped_query_attention(
                &q,
                &k,
                &v,
                AttentionMask::CausalSuffix { prefix_tokens: 2 },
                &stream,
            ),
            Err(OperationError::InvalidInput(_)),
        ));
    }

    #[test]
    fn rope_rejects_odd_head_dimensions() {
        let device = Device::new(DeviceType::Cpu, 0).expect("CPU");
        let stream = Stream::new(&device).expect("CPU stream");
        let input = array(&[0.0; 6], [1, 1, 2, 3]);

        assert!(matches!(
            rope(&input, 10_000.0, 0, &stream),
            Err(OperationError::InvalidInput(_)),
        ));
    }
}
