//! Explicit-stream RoPE and grouped-query attention.
//!
//! Native operations use actual MLX graphs on the supplied stream.
//! There is no host arithmetic fallback in the production functions.
//!
//! This initial implementation supports float32 arrays with layout:
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
    let [_, _, tokens, head_dim] = shape4(input)?;

    if head_dim % 2 != 0 {
        return Err(OperationError::InvalidInput(
            "split-half RoPE requires an even head dimension",
        ));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(OperationError::InvalidInput(
            "RoPE theta must be finite and positive",
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
        has_value: true,
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
            1.0,
            native_offset,
            no_frequencies,
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

    let mask_mode = match mask {
        AttentionMask::Full => c"".as_ptr(),
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
    };

    let scale = 1.0 / (head_dim as f32).sqrt();
    let no_array = sys::mlx_array {
        ctx: ptr::null_mut(),
    };
    let mut output = PendingArray::new();

    // SAFETY: all arrays and the stream are valid borrowed handles.
    // Shapes and GQA grouping were validated. mask_mode points to a
    // static NUL-terminated string. Optional mask/sinks are absent.
    let status = unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            &mut output.raw,
            query.as_raw(),
            key.as_raw(),
            value.as_raw(),
            scale,
            mask_mode,
            no_array,
            no_array,
            false,
            stream.as_raw(),
        )
    };

    output.finish(status)
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
    ) -> Vec<f32> {
        let [batch, query_heads, query_tokens, width] = query_shape;
        let [_, kv_heads, kv_tokens, _] = key_shape;
        let group = query_heads / kv_heads;
        let scale = 1.0 / (width as f64).sqrt();
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
                    };

                    let mut scores = Vec::with_capacity(key_count);

                    for key_token in 0..key_count {
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

                        for (key_token, score) in scores.iter().enumerate() {
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
        let expected = attention_reference(&query, &key, &value, query_shape, key_shape, mask);

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
