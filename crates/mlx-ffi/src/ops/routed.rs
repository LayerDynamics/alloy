//! Token-wise top-k routing over packed affine expert banks.
use super::{
    Array, DType, OperationError, OperationResult, PendingArray, Stream, native_dimension, sys,
};

/// One stack of transposed affine expert matrices in checkpoint layout.
pub struct AffineExperts<'a> {
    /// Packed uint32 weights shaped [experts, output, packed input].
    pub packed: &'a Array,
    /// Per-group affine scales shaped [experts, output, groups].
    pub scales: &'a Array,
    /// Per-group affine offsets in the same shape and dtype as the scales.
    pub biases: &'a Array,
    /// Number of adjacent input-channel weights in each affine group.
    pub group_size: usize,
    /// Packed bits per weight (4 or 8).
    pub bits: usize,
}

// PendingArray takes ownership of any result, including a partial error result.
fn array(call: impl FnOnce(&mut sys::mlx_array) -> i32) -> OperationResult<Array> {
    let mut out = PendingArray::new();
    let status = call(&mut out.raw);
    out.finish(status)
}

impl AffineExperts<'_> {
    fn output_width(&self, experts: usize, input: usize, dtype: DType) -> OperationResult<usize> {
        let shape = self.packed.shape();
        let [count, width, words] = shape.as_slice() else {
            return Err(OperationError::InvalidInput(
                "expected a rank-three packed expert bank",
            ));
        };
        if *count != experts
            || *width == 0
            || self.packed.dtype() != DType::Uint32
            || !matches!(self.bits, 4 | 8)
            || !matches!(self.group_size, 32 | 64 | 128)
            || !input.is_multiple_of(self.group_size)
            || *words != input / (32 / self.bits)
            || self.scales.shape() != [experts, *width, input / self.group_size]
            || self.biases.shape() != self.scales.shape()
            || self.scales.dtype() != dtype
            || self.biases.dtype() != dtype
        {
            return Err(OperationError::InvalidInput(
                "incompatible affine expert tensors",
            ));
        }
        Ok(*width)
    }

    fn project(&self, input: &Array, indices: &Array, stream: &Stream) -> OperationResult<Array> {
        let none = sys::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        let group = sys::mlx_optional_int {
            value: native_dimension(self.group_size)?,
            has_value: true,
        };
        let bits = sys::mlx_optional_int {
            value: native_dimension(self.bits)?,
            has_value: true,
        };
        // SAFETY: all operands are borrowed live handles. MLX broadcasts its default
        // lhs indices against the in-range rhs indices produced by argpartition.
        array(|out| unsafe {
            sys::mlx_gather_qmm(
                out,
                input.as_raw(),
                self.packed.as_raw(),
                self.scales.as_raw(),
                self.biases.as_raw(),
                none,
                indices.as_raw(),
                true,
                group,
                bits,
                c"affine".as_ptr(),
                none,
                false,
                stream.as_raw(),
            )
        })
    }
}

/// Affine input and output projections of one routed SwiGLU layer.
pub struct SwitchGluExperts<'a> {
    /// Input projection feeding the SiLU nonlinearity.
    pub gate: AffineExperts<'a>,
    /// Input projection multiplied by the activated gate.
    pub up: AffineExperts<'a>,
    /// Projection from intermediate activations back to hidden width.
    pub down: AffineExperts<'a>,
}

/// Execute SwitchGLU for `[tokens, hidden]` activations and `[tokens, experts]`
/// router logits. All routing, three projections, and reductions stay on the stream.
pub fn routed_swiglu(
    input: &Array,
    logits: &Array,
    banks: &SwitchGluExperts<'_>,
    top_k: usize,
    normalize: bool,
    stream: &Stream,
) -> OperationResult<Array> {
    let shape = input.shape();
    let router = logits.shape();
    let ([tokens, hidden], [rows, experts]) = (shape.as_slice(), router.as_slice()) else {
        return Err(OperationError::InvalidInput(
            "expected rank-two tokens and router logits",
        ));
    };
    let dtype = input.dtype();
    if tokens != rows
        || *tokens == 0
        || *hidden == 0
        || top_k == 0
        || top_k > *experts
        || logits.dtype() != dtype
        || !matches!(dtype, DType::Float32 | DType::Float16 | DType::BFloat16)
    {
        return Err(OperationError::InvalidInput(
            "invalid top-k routing dimensions or dtype",
        ));
    }
    let mid = banks.gate.output_width(*experts, *hidden, dtype)?;
    if banks.up.output_width(*experts, *hidden, dtype)? != mid
        || banks.down.output_width(*experts, mid, dtype)? != *hidden
    {
        return Err(OperationError::InvalidInput(
            "incompatible SwitchGLU projection sizes",
        ));
    }
    let k = native_dimension(top_k)?;
    let stop = [native_dimension(*tokens)?, native_dimension(*experts)?];
    let start = [0, stop[1] - k];
    let s = stream.as_raw();
    // SAFETY: all arrays and the stream outlive their calls. The router is rank
    // two, each slice vector has two elements, and top_k is in 1..=experts.
    let (indices, scores) = unsafe {
        let probabilities = array(|out| sys::mlx_softmax_axis(out, logits.as_raw(), 1, true, s))?;
        let partition =
            array(|out| sys::mlx_argpartition_axis(out, probabilities.as_raw(), -k, 1, s))?;
        let indices = array(|out| {
            sys::mlx_slice(
                out,
                partition.as_raw(),
                start.as_ptr(),
                2,
                stop.as_ptr(),
                2,
                [1, 1].as_ptr(),
                2,
                s,
            )
        })?;
        let mut scores = array(|out| {
            sys::mlx_take_along_axis(out, probabilities.as_raw(), indices.as_raw(), 1, s)
        })?;
        if normalize {
            let sums = array(|out| sys::mlx_sum_axis(out, scores.as_raw(), 1, true, s))?;
            scores = array(|out| sys::mlx_divide(out, scores.as_raw(), sums.as_raw(), s))?;
        }
        (indices, scores)
    };
    // SAFETY: expanding [tokens,hidden] at axes 1 and 2 produces the
    // [tokens,1,1,hidden] matrix batches required by gather_qmm.
    let x = array(|out| unsafe {
        sys::mlx_expand_dims_axes(out, input.as_raw(), [1, 2].as_ptr(), 2, s)
    })?;
    let g = banks.gate.project(&x, &indices, stream)?;
    let u = banks.up.project(&x, &indices, stream)?;
    // SAFETY: g and u have identical [tokens,k,1,mid] layouts.
    let activated = unsafe {
        let sigmoid = array(|out| sys::mlx_sigmoid(out, g.as_raw(), s))?;
        let silu = array(|out| sys::mlx_multiply(out, g.as_raw(), sigmoid.as_raw(), s))?;
        array(|out| sys::mlx_multiply(out, silu.as_raw(), u.as_raw(), s))?
    };
    let projected = banks.down.project(&activated, &indices, stream)?;
    // SAFETY: projected is [tokens,k,1,hidden]. Each [tokens,k,1] score
    // weights one route before the expert axis is reduced.
    unsafe {
        let y = array(|out| sys::mlx_squeeze_axis(out, projected.as_raw(), 2, s))?;
        let w = array(|out| sys::mlx_expand_dims(out, scores.as_raw(), 2, s))?;
        let weighted = array(|out| sys::mlx_multiply(out, y.as_raw(), w.as_raw(), s))?;
        array(|out| sys::mlx_sum_axis(out, weighted.as_raw(), 1, false, s))
    }
}
