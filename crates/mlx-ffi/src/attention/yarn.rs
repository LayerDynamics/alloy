//! Eager float32 YaRN and Ministral absolute-position scaling.
use super::*;

/// Full-head YaRN parameters. Validation precedes native allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnParameters {
    /// Even full rotary head width, bounded to 4096 features.
    pub dimensions: usize,
    /// Base used to construct the unscaled frequency divisors.
    pub theta: f32,
    /// Context extension factor, at least one.
    pub factor: f32,
    /// Original trained context length used by the correction range.
    pub original_context: usize,
    /// Upper rotation-count threshold for extrapolation.
    pub beta_fast: f32,
    /// Lower rotation-count threshold for interpolation.
    pub beta_slow: f32,
    /// Numerator coefficient in YaRN's rotary amplitude ratio.
    pub mscale: f32,
    /// Denominator coefficient in YaRN's rotary amplitude ratio.
    pub mscale_all_dim: f32,
}

impl YarnParameters {
    /// Reject unsupported dimensions and nonrepresentable intermediate values.
    pub fn validate(&self) -> OperationResult<()> {
        if self.dimensions == 0
            || self.dimensions > 4096
            || !self.dimensions.is_multiple_of(2)
            || !self.theta.is_finite()
            || self.theta <= 1.0
            || !self.factor.is_finite()
            || self.factor < 1.0
            || self.original_context == 0
            || self.original_context > 1 << 24
            || !self.beta_fast.is_finite()
            || !self.beta_slow.is_finite()
            || self.beta_slow <= 0.0
            || self.beta_fast < self.beta_slow
            || !self.mscale.is_finite()
            || !self.mscale_all_dim.is_finite()
            || self.mscale < 0.0
            || self.mscale_all_dim < 0.0
            // Bound the intermediate freq_inter * freq_extra, not just the
            // final divisor. Preserve the reference graph without overflow.
            || f64::from(self.factor) * f64::from(self.theta).powi(2) > f64::from(f32::MAX)
            || !self.rotary_amplitude().is_finite()
            || self.rotary_amplitude() <= 0.0
        {
            return Err(OperationError::InvalidInput("invalid YaRN parameters"));
        }
        let (low, high) = self.correction_range();
        if low > high {
            return Err(OperationError::InvalidInput(
                "invalid YaRN correction range",
            ));
        }
        Ok(())
    }

    /// Upstream computes these two scalar correction endpoints in float64.
    fn correction_range(&self) -> (f32, f32) {
        let dim = |rotations: f32| {
            self.dimensions as f64
                * (self.original_context as f64 / (f64::from(rotations) * std::f64::consts::TAU))
                    .ln()
                / (2.0 * f64::from(self.theta).ln())
        };
        (
            dim(self.beta_fast).floor().max(0.0) as f32,
            dim(self.beta_slow).ceil().min(self.dimensions as f64 - 1.0) as f32,
        )
    }

    /// Amplitude applied to both Q and K before rotation, distinct from the
    /// position-dependent scale applied only to Q afterwards.
    pub fn rotary_amplitude(&self) -> f32 {
        if self.factor <= 1.0 {
            1.0
        } else {
            let scale = |m: f32| 0.1 * f64::from(m) * f64::from(self.factor).ln() + 1.0;
            (scale(self.mscale) / scale(self.mscale_all_dim)) as f32
        }
    }
}

type Binary = unsafe extern "C" fn(
    *mut sys::mlx_array,
    sys::mlx_array,
    sys::mlx_array,
    sys::mlx_stream,
) -> c_int;
type Unary = unsafe extern "C" fn(*mut sys::mlx_array, sys::mlx_array, sys::mlx_stream) -> c_int;

fn binary(op: Binary, a: &Array, b: &Array, stream: &Stream) -> OperationResult<Array> {
    let mut output = PendingArray::new();
    // SAFETY: private callers supply live, broadcast-compatible float32 arrays.
    let status = unsafe { op(&mut output.raw, a.as_raw(), b.as_raw(), stream.as_raw()) };
    output.finish(status)
}
fn unary(op: Unary, a: &Array, stream: &Stream) -> OperationResult<Array> {
    let mut output = PendingArray::new();
    // SAFETY: live array and stream; exclusive owned output.
    let status = unsafe { op(&mut output.raw, a.as_raw(), stream.as_raw()) };
    output.finish(status)
}
fn scalar(value: f32) -> OperationResult<Array> {
    Ok(Array::from_f32_slice(&[value], &[])?)
}
fn arange(start: usize, end: usize, stream: &Stream) -> OperationResult<Array> {
    let mut output = PendingArray::new();
    // SAFETY: callers bound nonempty positions to exactly representable float32
    // integers and C dimensions before requesting this native array.
    let status = unsafe {
        sys::mlx_arange(
            &mut output.raw,
            start as f64,
            end as f64,
            1.0,
            DType::Float32 as u32,
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// YaRN frequency divisors with the pinned upstream eager float32 operation
/// ordering, including interpolation and the coincident-endpoint guard.
pub fn yarn_frequencies(parameters: &YarnParameters, stream: &Stream) -> OperationResult<Array> {
    parameters.validate()?;
    let p = parameters;
    let indices = arange(0, p.dimensions / 2, stream)?;
    let exponents = binary(
        sys::mlx_divide,
        &binary(sys::mlx_multiply, &indices, &scalar(2.0)?, stream)?,
        &scalar(p.dimensions as f32)?,
        stream,
    )?;
    let extra = binary(sys::mlx_power, &scalar(p.theta)?, &exponents, stream)?;
    let inter = binary(sys::mlx_multiply, &scalar(p.factor)?, &extra, stream)?;
    let (low, high) = p.correction_range();
    let high = if low == high {
        f64::from(high) + 0.001
    } else {
        f64::from(high)
    };
    let ramp = binary(
        sys::mlx_divide,
        &binary(sys::mlx_subtract, &indices, &scalar(low)?, stream)?,
        &scalar((high - f64::from(low)) as f32)?,
        stream,
    )?;
    let ramp = binary(
        sys::mlx_minimum,
        &binary(sys::mlx_maximum, &ramp, &scalar(0.0)?, stream)?,
        &scalar(1.0)?,
        stream,
    )?;
    let mask = binary(sys::mlx_subtract, &scalar(1.0)?, &ramp, stream)?;
    let complement = binary(sys::mlx_subtract, &scalar(1.0)?, &mask, stream)?;
    let numerator = binary(sys::mlx_multiply, &inter, &extra, stream)?;
    let denominator = binary(
        sys::mlx_add,
        &binary(sys::mlx_multiply, &inter, &mask, stream)?,
        &binary(sys::mlx_multiply, &extra, &complement, stream)?,
        stream,
    )?;
    binary(sys::mlx_divide, &numerator, &denominator, stream)
}

/// Scale query features at absolute positions, preserving shape. The scale is
/// `1 + beta * log(1 + floor(position / original_context))`. Keys must not pass
/// through this function. Native broadcasting only creates a [tokens, 1] scale.
pub fn position_query_scale(
    input: &Array,
    offset: usize,
    original_context: usize,
    beta: f32,
    stream: &Stream,
) -> OperationResult<Array> {
    let [_, _, tokens, _] = shape4(input)?;
    if original_context == 0
        || original_context > 1 << 24
        || !beta.is_finite()
        || beta < 0.0
        || offset.checked_add(tokens).is_none_or(|end| end > 1 << 24)
        || f64::from(beta) * (1.0 + ((1 << 24) as f64 / original_context as f64)).ln() + 1.0
            > f64::from(f32::MAX)
    {
        return Err(OperationError::InvalidInput(
            "invalid query position scaling",
        ));
    }
    let positions = arange(offset, offset + tokens, stream)?;
    let quotient = binary(
        sys::mlx_divide,
        &positions,
        &scalar(original_context as f32)?,
        stream,
    )?;
    let floor = unary(sys::mlx_floor, &quotient, stream)?;
    let log = unary(
        sys::mlx_log,
        &binary(sys::mlx_add, &scalar(1.0)?, &floor, stream)?,
        stream,
    )?;
    let scales = binary(
        sys::mlx_add,
        &scalar(1.0)?,
        &binary(sys::mlx_multiply, &scalar(beta)?, &log, stream)?,
        stream,
    )?;
    let scales = crate::ops::reshape(&scales, &[tokens, 1], stream)?;
    binary(sys::mlx_multiply, input, &scales, stream)
}
