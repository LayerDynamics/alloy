//! Native primitives for mixed full/recurrent attention. State ownership and
//! memory admission stay with the model; these operations never persist state.
use super::*;
use crate::VectorArray;

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
