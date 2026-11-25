//! Native primitives for mixed full/recurrent attention. State ownership and
//! memory admission stay with the model; these operations never persist state.
use super::*;
use crate::VectorArray;

/// Explicit right-aligned float32 broadcasting. A source axis must equal its
/// destination or be one; this never silently changes a non-singleton axis.
pub fn broadcast_to(input: &Array, shape: &[usize], stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    require_positive_shape(shape)?;
    shape
        .iter()
        .try_fold(1_usize, |total, &n| total.checked_mul(n))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(OperationError::InvalidInput("broadcast payload overflow"))?;
    let source = input.shape();
    if source.len() > shape.len()
        || source
            .iter()
            .rev()
            .zip(shape.iter().rev())
            .any(|(&a, &b)| a != 1 && a != b)
    {
        return Err(OperationError::InvalidInput(
            "incompatible broadcast dimensions",
        ));
    }
    let dimensions: Vec<_> = shape
        .iter()
        .copied()
        .map(native_dimension)
        .collect::<OperationResult<_>>()?;
    let mut output = PendingArray::new();
    // SAFETY: rank, dimensions and broadcast compatibility checked above.
    let status = unsafe {
        sys::mlx_broadcast_to(
            &mut output.raw,
            input.as_raw(),
            dimensions.as_ptr(),
            dimensions.len(),
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Concatenate two nonempty float32 tensors along one explicit axis.
pub fn concatenate_axis(
    left: &Array,
    right: &Array,
    axis: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    concatenate_arrays(&[left.clone(), right.clone()], axis, stream)
}

/// Join compatible F32 arrays in order without materializing intermediate joins.
pub fn concatenate_arrays(
    inputs: &[Array],
    axis: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    let first = inputs
        .first()
        .ok_or(OperationError::InvalidInput("empty concatenation"))?;
    let shape = first.shape();
    let mut length = 0usize;
    for input in inputs {
        require_f32(input)?;
        let part = input.shape();
        require_positive_shape(&part)?;
        if axis >= part.len()
            || part.len() != shape.len()
            || part
                .iter()
                .zip(&shape)
                .enumerate()
                .any(|(i, (a, b))| i != axis && a != b)
        {
            return Err(OperationError::InvalidInput(
                "incompatible concatenation dimensions",
            ));
        }
        length = length
            .checked_add(part[axis])
            .ok_or(OperationError::InvalidInput(
                "concatenation length overflow",
            ))?;
    }
    native_dimension(length)?;
    if inputs.len() == 1 {
        return Ok(first.clone());
    }
    let arrays = VectorArray::from_arrays(inputs)?;
    let mut output = PendingArray::new();
    // SAFETY: the owned vector retains all validated arrays and the axis is in range.
    let status = unsafe {
        sys::mlx_concatenate_axis(
            &mut output.raw,
            arrays.raw,
            native_dimension(axis)?,
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Multiply by a finite scalar, retaining the input shape.
pub fn scale(input: &Array, factor: f32, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    if !factor.is_finite() {
        return Err(OperationError::InvalidInput("nonfinite scale"));
    }
    let scalar = Array::from_f32_slice(&[factor], &[])?;
    let mut output = PendingArray::new();
    // SAFETY: a scalar broadcasts to every valid input shape.
    let status = unsafe {
        sys::mlx_multiply(
            &mut output.raw,
            input.as_raw(),
            scalar.as_raw(),
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// RMS normalization without a learned weight, matching the trained Q/K
/// normalization in Qwen3.5 recurrent attention.
pub fn rms_norm_unweighted(input: &Array, epsilon: f32, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    let shape = input.shape();
    if shape.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OperationError::InvalidInput(
            "invalid unweighted RMS normalization",
        ));
    }
    require_positive_shape(&shape)?;
    let mut output = PendingArray::new();
    // SAFETY: null optional weight is the MLX unweighted normalization API.
    let status = unsafe {
        sys::mlx_fast_rms_norm(
            &mut output.raw,
            input.as_raw(),
            sys::mlx_array {
                ctx: ptr::null_mut(),
            },
            epsilon,
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Elementwise sigmoid on float32 model tensors.
pub fn sigmoid(input: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    let mut output = PendingArray::new();
    // SAFETY: borrowed valid handles and an owned writable output slot.
    let status = unsafe { sys::mlx_sigmoid(&mut output.raw, input.as_raw(), stream.as_raw()) };
    output.finish(status)
}

/// Elementwise exponential on float32 model tensors.
pub fn exp(input: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    let mut output = PendingArray::new();
    // SAFETY: borrowed valid handles and an owned writable output slot.
    let status = unsafe { sys::mlx_exp(&mut output.raw, input.as_raw(), stream.as_raw()) };
    output.finish(status)
}

/// Stable softplus, log(exp(x) + 1), without overflowing exp(x) at large x.
pub fn softplus(input: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    let zero = Array::from_f32_slice(&[0.0], &[])?;
    let mut output = PendingArray::new();
    // SAFETY: scalar zero broadcasts to input; all handles stay alive.
    let status = unsafe {
        sys::mlx_logaddexp(
            &mut output.raw,
            input.as_raw(),
            zero.as_raw(),
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Valid depthwise convolution: input [B,T,C], weight [C,K,1], output
/// [B,T-K+1,C]. Caller prepends its K-1 convolution-state samples, so no implicit
/// padding or hidden mutable history is introduced here.
pub fn depthwise_conv1d(input: &Array, weight: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    require_f32(weight)?;
    let a = input.shape();
    let w = weight.shape();
    if a.len() != 3 || w.len() != 3 {
        return Err(OperationError::InvalidInput(
            "depthwise convolution requires rank three",
        ));
    }
    require_positive_shape(&a)?;
    require_positive_shape(&w)?;
    if w[0] != a[2] || w[2] != 1 || a[1] < w[1] {
        return Err(OperationError::InvalidInput(
            "invalid depthwise channels or kernel length",
        ));
    }
    let groups = native_dimension(a[2])?;
    let mut output = PendingArray::new();
    // SAFETY: shape checks establish one input channel per group and a valid
    // kernel extent. Native output is owned by PendingArray until transferred.
    let status = unsafe {
        sys::mlx_conv1d(
            &mut output.raw,
            input.as_raw(),
            weight.as_raw(),
            1,
            0,
            1,
            groups,
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Gated delta recurrence on float32 tensors. Q/K [B,T,Hk,Dk], V [B,T,Hv,Dv],
/// decay and beta [B,T,Hv], initial/final state [B,Hv,Dv,Dk]. Hv must be a
/// multiple of Hk. Returns (output [B,T,Hv,Dv], final state). Inputs remain
/// immutable; a rejected/cancelled caller can discard the new state atomically.
/// Decay and beta are already transformed gates, not projection logits.
pub fn gated_delta_update(
    queries: &Array,
    keys: &Array,
    values: &Array,
    decay: &Array,
    beta: &Array,
    initial_state: Option<&Array>,
    stream: &Stream,
) -> OperationResult<(Array, Array)> {
    for input in [queries, keys, values, decay, beta] {
        require_f32(input)?;
        require_positive_shape(&input.shape())?;
    }
    let q = queries.shape();
    let v = values.shape();
    if q.len() != 4 || v.len() != 4 || keys.shape() != q {
        return Err(OperationError::InvalidInput(
            "gated delta requires matching rank-four Q/K and rank-four V",
        ));
    }
    if q[0..2] != v[0..2]
        || !v[2].is_multiple_of(q[2])
        || decay.shape() != [q[0], q[1], v[2]]
        || beta.shape() != decay.shape()
    {
        return Err(OperationError::InvalidInput(
            "gated delta batch, sequence, heads or gate shape mismatch",
        ));
    }
    if let Some(state) = initial_state {
        require_f32(state)?;
        if state.shape() != [q[0], v[2], v[3], q[3]] {
            return Err(OperationError::InvalidInput(
                "gated delta initial state shape mismatch",
            ));
        }
    }
    if stream.device()?.kind()? == crate::DeviceType::Gpu && q[3].is_multiple_of(32) && q[3] <= 512
    {
        return super::sequential_delta::evaluate(
            queries,
            keys,
            values,
            decay,
            beta,
            initial_state,
            stream,
        );
    }
    // SAFETY: new returns an owned vector. The RAII wrapper also frees it if
    // the native operation fails after partially constructing its results.
    let mut output = VectorArray::from_raw(unsafe { sys::mlx_vector_array_new() })?;
    let null = sys::mlx_array {
        ctx: ptr::null_mut(),
    };
    // SAFETY: all ranks/dimensions and dtype contracts were checked above.
    // Null optional state means zero initialization, null mask means no mask.
    let status = unsafe {
        sys::mlx_fast_gated_delta_update(
            &mut output.raw,
            queries.as_raw(),
            keys.as_raw(),
            values.as_raw(),
            decay.as_raw(),
            beta.as_raw(),
            initial_state.map_or(null, Array::as_raw),
            null,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(OperationError::Mlx(Error::Status(status)));
    }
    if output.len() != 2 {
        return Err(OperationError::InvalidInput(
            "gated delta returned an unexpected result count",
        ));
    }
    Ok((output.get(0)?, output.get(1)?))
}
