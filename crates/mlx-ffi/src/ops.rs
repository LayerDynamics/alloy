//! Explicit-stream MLX tensor operations for native model execution.
//!
//! Arithmetic operations build lazy MLX graphs without host tensor
//! arithmetic. Explicit validation and readback helpers are synchronization
//! boundaries: `all_finite` reads one reduced Boolean, and `to_vec_f32`
//! copies a requested output into a host vector.
//!
//! The caller selects execution streams explicitly. Arrays are not
//! duplicated into separate CPU-model and GPU-model representations.
//!
//! Dense linears use float32 weights. Affine4 linears use packed uint32
//! weights with `quantized_matmul` (scales/biases stay float32).
//! Token IDs supplied to embedding_lookup are uploaded as uint32 metadata.
//!
//! Cache helpers use:
//! [batch, heads, capacity_or_chunk_tokens, head_dim].
//!
//! Slice updates are functional: callers must retain the returned array.
//! No API here promises in-place allocation reuse or bounded allocator
//! pressure. Persistent cache ownership, admission, graph lifetime, and
//! cancellation/completion policy remain runtime responsibilities.

use crate::{Array, DType, Error, Stream, bindings as sys};
use std::{
    ffi::CString,
    fmt,
    os::raw::{c_int, c_void},
    ptr,
};

/// Failure constructing or evaluating an MLX tensor operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationError {
    /// Input metadata violates the supported operation contract.
    InvalidInput(&'static str),

    /// A fallible Rust metadata allocation failed.
    AllocationFailed(&'static str),

    /// The underlying MLX wrapper reported an error.
    Mlx(Error),
}

impl fmt::Display for OperationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => {
                write!(formatter, "invalid MLX operation input: {message}")
            }
            Self::AllocationFailed(operation) => {
                write!(formatter, "allocation failed during {operation}")
            }
            Self::Mlx(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for OperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mlx(error) => Some(error),
            Self::InvalidInput(_) | Self::AllocationFailed(_) => None,
        }
    }
}

impl From<Error> for OperationError {
    fn from(error: Error) -> Self {
        Self::Mlx(error)
    }
}

/// Result of an actual MLX tensor operation.
pub type OperationResult<T> = std::result::Result<T, OperationError>;

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
            // SAFETY: this guard exclusively owns an untransferred
            // operation output.
            unsafe {
                sys::mlx_array_free(self.raw);
            }
        }
    }
}

fn require_f32(array: &Array) -> OperationResult<()> {
    if array.dtype() != DType::Float32 {
        return Err(OperationError::InvalidInput(
            "this operation path requires float32 model tensors",
        ));
    }
    Ok(())
}

fn metadata_vec<T>(capacity: usize) -> OperationResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| OperationError::AllocationFailed("tensor-operation metadata"))?;
    Ok(values)
}

fn native_dimension(value: usize) -> OperationResult<c_int> {
    c_int::try_from(value)
        .map_err(|_| OperationError::InvalidInput("dimension or index exceeds C int"))
}

fn native_dimensions(shape: &[usize]) -> OperationResult<Vec<c_int>> {
    native_dimension(shape.len())?;
    let mut output = metadata_vec(shape.len())?;

    for &dimension in shape {
        output.push(native_dimension(dimension)?);
    }

    Ok(output)
}

fn element_count(shape: &[usize]) -> OperationResult<usize> {
    shape.iter().try_fold(1_usize, |total, &dimension| {
        total
            .checked_mul(dimension)
            .ok_or(OperationError::InvalidInput(
                "tensor element-count overflow",
            ))
    })
}

fn require_positive_shape(shape: &[usize]) -> OperationResult<()> {
    if shape.contains(&0) {
        return Err(OperationError::InvalidInput(
            "zero-sized dimensions are not supported by this operation",
        ));
    }
    native_dimension(shape.len())?;
    Ok(())
}

fn copy_handle(input: &Array) -> OperationResult<Array> {
    let mut output = PendingArray::new();

    // SAFETY: valid borrowed source and writable output. mlx_array_set
    // creates another owned array handle, not a host tensor copy.
    let status = unsafe { sys::mlx_array_set(&mut output.raw, input.as_raw()) };

    output.finish(status)
}

/// Compute input @ transpose(weight).
///
/// input: [rows, input_width].
/// weight: [output_width, input_width].
/// output: [rows, output_width].
///
/// Both inputs must be float32 with positive dimensions.
pub fn linear(input: &Array, weight: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    require_f32(weight)?;

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.len() != 2 || weight_shape.len() != 2 {
        return Err(OperationError::InvalidInput(
            "linear requires rank-two input and weight arrays",
        ));
    }
    require_positive_shape(&input_shape)?;
    require_positive_shape(&weight_shape)?;

    if input_shape[1] != weight_shape[1] {
        return Err(OperationError::InvalidInput(
            "linear input width does not match weight input width",
        ));
    }

    let mut transposed = PendingArray::new();

    // SAFETY: valid borrowed input/stream and writable output.
    let status =
        unsafe { sys::mlx_matrix_transpose(&mut transposed.raw, weight.as_raw(), stream.as_raw()) };
    let transposed = transposed.finish(status)?;

    let mut output = PendingArray::new();

    // SAFETY: valid handles; dimensions satisfy [rows, in] @ [in, out].
    let status = unsafe {
        sys::mlx_matmul(
            &mut output.raw,
            input.as_raw(),
            transposed.as_raw(),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Compute `input @ quantized(weight)^T` for affine packed weights.
///
/// `input`: `[rows, input_width]` float32.
/// `weight`: packed uint32 `[output_width, input_width * bits / 32]`.
/// `scales` / `biases`: `[output_width, groups]` float32.
/// `transpose`: true for the usual checkpoint layout `[out, in]`.
pub fn quantized_matmul(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> OperationResult<Array> {
    require_f32(input)?;
    if weight.dtype() != DType::Uint32 {
        return Err(OperationError::InvalidInput(
            "quantized_matmul weight must be uint32",
        ));
    }
    require_f32(scales)?;
    require_f32(biases)?;

    let input_shape = input.shape();
    let weight_shape = weight.shape();
    let scales_shape = scales.shape();
    let biases_shape = biases.shape();

    if input_shape.len() != 2 || weight_shape.len() != 2 {
        return Err(OperationError::InvalidInput(
            "quantized_matmul requires rank-two input and weight",
        ));
    }
    if scales_shape != biases_shape || scales_shape.len() != 2 {
        return Err(OperationError::InvalidInput(
            "quantized_matmul scales/biases must be rank-two and matching",
        ));
    }
    if group_size <= 0 || bits <= 0 {
        return Err(OperationError::InvalidInput(
            "quantized_matmul group_size and bits must be positive",
        ));
    }
    require_positive_shape(&input_shape)?;
    require_positive_shape(&weight_shape)?;
    require_positive_shape(&scales_shape)?;

    let mode = CString::new("affine")
        .map_err(|_| OperationError::InvalidInput("quantized_matmul mode string"))?;
    let group = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bit_opt = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };

    let mut output = PendingArray::new();
    // SAFETY: valid handles; mode NUL-terminated and lives for the call.
    let status = unsafe {
        sys::mlx_quantized_matmul(
            &mut output.raw,
            input.as_raw(),
            weight.as_raw(),
            scales.as_raw(),
            biases.as_raw(),
            transpose,
            group,
            bit_opt,
            mode.as_ptr(),
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Dequantize an affine packed weight matrix to float32.
pub fn dequantize(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> OperationResult<Array> {
    if weight.dtype() != DType::Uint32 {
        return Err(OperationError::InvalidInput(
            "dequantize weight must be uint32",
        ));
    }
    require_f32(scales)?;
    require_f32(biases)?;
    if group_size <= 0 || bits <= 0 {
        return Err(OperationError::InvalidInput(
            "dequantize group_size and bits must be positive",
        ));
    }

    let mode = CString::new("affine")
        .map_err(|_| OperationError::InvalidInput("dequantize mode string"))?;
    let group = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bit_opt = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };
    let null_global = sys::mlx_array {
        ctx: ptr::null_mut(),
    };
    let null_dtype = sys::mlx_optional_dtype {
        value: DType::Float32 as u32,
        has_value: false,
    };

    let mut output = PendingArray::new();
    // SAFETY: valid handles; null global_scale / dtype are documented allowed.
    let status = unsafe {
        sys::mlx_dequantize(
            &mut output.raw,
            weight.as_raw(),
            scales.as_raw(),
            biases.as_raw(),
            group,
            bit_opt,
            mode.as_ptr(),
            null_global,
            null_dtype,
            stream.as_raw(),
        )
    };
    output.finish(status)
}

/// Apply weighted RMSNorm over the final dimension.
///
/// weight must be a vector matching the final input dimension.
/// epsilon must be finite and positive.
pub fn rms_norm(
    input: &Array,
    weight: &Array,
    epsilon: f32,
    stream: &Stream,
) -> OperationResult<Array> {
    require_f32(input)?;
    require_f32(weight)?;

    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OperationError::InvalidInput(
            "RMSNorm epsilon must be finite and positive",
        ));
    }

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.is_empty() {
        return Err(OperationError::InvalidInput(
            "RMSNorm requires a non-scalar input",
        ));
    }
    require_positive_shape(&input_shape)?;

    if weight_shape.len() != 1 || weight_shape[0] != input_shape[input_shape.len() - 1] {
        return Err(OperationError::InvalidInput(
            "RMSNorm weight must match the final input dimension",
        ));
    }

    let mut output = PendingArray::new();

    // SAFETY: valid handles, matching normalization width, valid epsilon.
    let status = unsafe {
        sys::mlx_fast_rms_norm(
            &mut output.raw,
            input.as_raw(),
            weight.as_raw(),
            epsilon,
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Compute input * sigmoid(input) on the selected stream.
pub fn silu(input: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;

    let mut sigmoid = PendingArray::new();

    // SAFETY: valid borrowed input/stream and writable output.
    let status = unsafe { sys::mlx_sigmoid(&mut sigmoid.raw, input.as_raw(), stream.as_raw()) };
    let sigmoid = sigmoid.finish(status)?;

    multiply(input, &sigmoid, stream)
}

/// Multiply equal-shaped float32 tensors without implicit broadcasting.
pub fn multiply(left: &Array, right: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(left)?;
    require_f32(right)?;

    if left.shape() != right.shape() {
        return Err(OperationError::InvalidInput(
            "elementwise multiplication requires identical shapes",
        ));
    }

    let mut output = PendingArray::new();

    // SAFETY: valid handles and equal-shaped operands.
    let status = unsafe {
        sys::mlx_multiply(
            &mut output.raw,
            left.as_raw(),
            right.as_raw(),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Compute the SwiGLU activation product: SiLU(gate) * up.
///
/// Inputs are already-projected gate and up tensors of identical shape.
/// This function does not perform the gate/up/down linear projections.
pub fn swiglu(gate: &Array, up: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(gate)?;
    require_f32(up)?;

    if gate.shape() != up.shape() {
        return Err(OperationError::InvalidInput(
            "SwiGLU gate and up tensors must have identical shapes",
        ));
    }

    let activated = silu(gate, stream)?;
    multiply(&activated, up, stream)
}

/// Add equal-shaped float32 tensors without implicit broadcasting.
pub fn residual_add(input: &Array, residual: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    require_f32(residual)?;

    if input.shape() != residual.shape() {
        return Err(OperationError::InvalidInput(
            "residual addition requires identical shapes",
        ));
    }

    let mut output = PendingArray::new();

    // SAFETY: valid handles and equal-shaped operands.
    let status = unsafe {
        sys::mlx_add(
            &mut output.raw,
            input.as_raw(),
            residual.as_raw(),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Look up embedding rows for validated request token IDs.
///
/// weight: [vocabulary, embedding_width], float32.
/// tokens: nonempty request metadata.
/// output: [tokens.len(), embedding_width].
///
/// Token IDs are checked against vocabulary before the native gather.
/// Only token metadata is uploaded; weight contents are not read back.
pub fn embedding_lookup(weight: &Array, tokens: &[u32], stream: &Stream) -> OperationResult<Array> {
    require_f32(weight)?;
    let shape = weight.shape();

    if shape.len() != 2 {
        return Err(OperationError::InvalidInput(
            "embedding weight must have shape [vocabulary, width]",
        ));
    }
    require_positive_shape(&shape)?;

    if tokens.is_empty() {
        return Err(OperationError::InvalidInput(
            "embedding lookup requires nonempty token IDs",
        ));
    }

    for &token in tokens {
        let token = usize::try_from(token)
            .map_err(|_| OperationError::InvalidInput("token ID cannot fit usize"))?;
        if token >= shape[0] {
            return Err(OperationError::InvalidInput(
                "embedding token ID is outside the vocabulary",
            ));
        }
    }

    let token_count = native_dimension(tokens.len())?;
    let index_shape = [token_count];

    // SAFETY: tokens is a valid uint32 buffer with exactly the validated
    // shape's element count. mlx_array_new_data copies the input buffer.
    let indices = unsafe {
        sys::mlx_array_new_data(
            tokens.as_ptr().cast::<c_void>(),
            index_shape.as_ptr(),
            1,
            DType::Uint32 as u32,
        )
    };
    let indices = Array::from_raw(indices)?;

    let slice_sizes = [1, native_dimension(shape[1])?];
    let mut gathered = PendingArray::new();

    // SAFETY: valid handles and validated token IDs. Gather axis zero
    // selects full embedding rows. Slice-size metadata lives for the call.
    let status = unsafe {
        sys::mlx_gather_single(
            &mut gathered.raw,
            weight.as_raw(),
            indices.as_raw(),
            0,
            slice_sizes.as_ptr(),
            slice_sizes.len(),
            stream.as_raw(),
        )
    };
    let gathered = gathered.finish(status)?;

    // Gather retains the size-one sliced vocabulary dimension.
    reshape(&gathered, &[tokens.len(), shape[1]], stream)
}

/// Look up rows of an Affine4 embedding without expanding its vocabulary.
///
/// Packed weights remain uint32 `[vocabulary, width / 8]`; scales and biases
/// are float32 `[vocabulary, width / group_size]`. Only the requested rows
/// are gathered and dequantized on `stream`. The result is lazy float32
/// `[tokens.len(), width]`, and the original packed arrays remain shared.
pub fn quantized_embedding_lookup(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    tokens: &[u32],
    group_size: i32,
    stream: &Stream,
) -> OperationResult<Array> {
    if weight.dtype() != DType::Uint32 {
        return Err(OperationError::InvalidInput(
            "quantized embedding weight must be uint32",
        ));
    }
    require_f32(scales)?;
    require_f32(biases)?;
    let packed_shape = weight.shape();
    let parameter_shape = scales.shape();
    if packed_shape.len() != 2
        || parameter_shape.len() != 2
        || biases.shape() != parameter_shape
        || parameter_shape[0] != packed_shape[0]
        || group_size <= 0
    {
        return Err(OperationError::InvalidInput(
            "quantized embedding requires matching rank-two weights and parameters",
        ));
    }
    require_positive_shape(&packed_shape)?;
    require_positive_shape(&parameter_shape)?;
    let width = packed_shape[1]
        .checked_mul(8)
        .ok_or(OperationError::InvalidInput(
            "quantized embedding width overflow",
        ))?;
    let group = group_size as usize;
    if width % group != 0 || parameter_shape[1] != width / group {
        return Err(OperationError::InvalidInput(
            "quantized embedding parameters disagree with width and group size",
        ));
    }
    native_dimension(width)?;
    if tokens.is_empty()
        || tokens
            .iter()
            .any(|&token| u64::from(token) >= packed_shape[0] as u64)
    {
        return Err(OperationError::InvalidInput(
            "quantized embedding token IDs must be nonempty and inside the vocabulary",
        ));
    }
    let count = native_dimension(tokens.len())?;
    let indices = Array::from_u32_slice(tokens, &[count])?;
    let take_rows = |array: &Array| {
        let mut output = PendingArray::new();
        // SAFETY: valid arrays/stream; rank-two inputs share the vocabulary
        // dimension, and every uint32 index was checked before this call.
        let status = unsafe {
            sys::mlx_take_axis(
                &mut output.raw,
                array.as_raw(),
                indices.as_raw(),
                0,
                stream.as_raw(),
            )
        };
        output.finish(status)
    };
    let selected_weight = take_rows(weight)?;
    let selected_scales = take_rows(scales)?;
    let selected_biases = take_rows(biases)?;
    dequantize(
        &selected_weight,
        &selected_scales,
        &selected_biases,
        group_size,
        4,
        stream,
    )
}

/// Reshape a float32 tensor with an explicitly supplied shape.
///
/// The element count must remain unchanged. Dimensions must be positive;
/// an empty shape is permitted for a one-element scalar.
///
/// No inferred -1 dimensions or implicit evaluation are performed.
pub fn reshape(input: &Array, shape: &[usize], stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    require_positive_shape(shape)?;

    if element_count(shape)? != input.size() {
        return Err(OperationError::InvalidInput(
            "reshape must preserve the tensor element count",
        ));
    }

    let native_shape = native_dimensions(shape)?;
    let mut output = PendingArray::new();

    // SAFETY: valid input/stream and length-matched shape metadata.
    // The checked product equals the source element count.
    let status = unsafe {
        sys::mlx_reshape(
            &mut output.raw,
            input.as_raw(),
            native_shape.as_ptr(),
            native_shape.len(),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Permute float32 tensor axes.
///
/// axes must be a complete permutation of 0..input.ndim().
/// For example, [0, 2, 1, 3] converts [batch, tokens, heads, dim] into
/// [batch, heads, tokens, dim].
///
/// The permutation is composed from explicit-stream MLX moveaxis
/// operations. No tensor contents are copied through Rust.
pub fn transpose(input: &Array, axes: &[usize], stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;

    let rank = input.ndim();
    native_dimension(rank)?;

    if axes.len() != rank {
        return Err(OperationError::InvalidInput(
            "transpose permutation length must equal tensor rank",
        ));
    }

    let mut seen = metadata_vec(rank)?;
    seen.resize(rank, false);

    for &axis in axes {
        if axis >= rank || seen[axis] {
            return Err(OperationError::InvalidInput(
                "transpose axes must be a complete nonrepeating permutation",
            ));
        }
        seen[axis] = true;
    }

    let mut order = metadata_vec(rank)?;
    order.extend(0..rank);
    let mut current = copy_handle(input)?;

    for (destination, &desired_axis) in axes.iter().enumerate() {
        let source = order.iter().position(|&axis| axis == desired_axis).ok_or(
            OperationError::InvalidInput("transpose permutation bookkeeping disagrees"),
        )?;

        if source == destination {
            continue;
        }

        let mut output = PendingArray::new();

        // SAFETY: valid handles and validated in-range source/destination.
        let status = unsafe {
            sys::mlx_moveaxis(
                &mut output.raw,
                current.as_raw(),
                native_dimension(source)?,
                native_dimension(destination)?,
                stream.as_raw(),
            )
        };
        current = output.finish(status)?;

        let axis = order.remove(source);
        order.insert(destination, axis);
    }

    Ok(current)
}

fn slice_bounds(
    shape: &[usize],
    start: &[usize],
    stop: &[usize],
) -> OperationResult<(Vec<c_int>, Vec<c_int>, Vec<c_int>)> {
    if shape.is_empty() || start.len() != shape.len() || stop.len() != shape.len() {
        return Err(OperationError::InvalidInput(
            "slice bounds must cover every axis of a non-scalar tensor",
        ));
    }

    require_positive_shape(shape)?;
    let mut native_start = metadata_vec(shape.len())?;
    let mut native_stop = metadata_vec(shape.len())?;
    let mut strides = metadata_vec(shape.len())?;

    for axis in 0..shape.len() {
        if start[axis] >= stop[axis] || stop[axis] > shape[axis] {
            return Err(OperationError::InvalidInput(
                "slice must be nonempty and entirely inside the source",
            ));
        }

        native_start.push(native_dimension(start[axis])?);
        native_stop.push(native_dimension(stop[axis])?);
        strides.push(1);
    }

    Ok((native_start, native_stop, strides))
}

/// Extract a nonempty, unit-stride rectangular slice.
///
/// start is inclusive and stop is exclusive. Bounds must be supplied for
/// every axis. Out-of-range indices are rejected rather than clamped.
pub fn slice(
    input: &Array,
    start: &[usize],
    stop: &[usize],
    stream: &Stream,
) -> OperationResult<Array> {
    require_f32(input)?;
    let shape = input.shape();
    let (start, stop, strides) = slice_bounds(&shape, start, stop)?;
    let mut output = PendingArray::new();

    // SAFETY: valid handles and equal-length validated bounds/strides.
    let status = unsafe {
        sys::mlx_slice(
            &mut output.raw,
            input.as_raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

/// Return source with a rectangular region replaced by update.
///
/// Both arrays must be float32 and have the same rank. start supplies
/// one nonnegative index per axis. The update must fit completely.
///
/// This is a functional MLX operation. The source handle is unchanged.
/// Retain the returned array as the new cache/tensor version.
///
/// No in-place-storage, zero-allocation, or aliasing guarantee is made.
pub fn slice_update(
    source: &Array,
    update: &Array,
    start: &[usize],
    stream: &Stream,
) -> OperationResult<Array> {
    require_f32(source)?;
    require_f32(update)?;

    let source_shape = source.shape();
    let update_shape = update.shape();

    if source_shape.is_empty()
        || source_shape.len() != update_shape.len()
        || start.len() != source_shape.len()
    {
        return Err(OperationError::InvalidInput(
            "slice update requires matching non-scalar ranks and start indices",
        ));
    }

    require_positive_shape(&source_shape)?;
    require_positive_shape(&update_shape)?;

    let mut stop = metadata_vec(source_shape.len())?;
    for axis in 0..source_shape.len() {
        let end =
            start[axis]
                .checked_add(update_shape[axis])
                .ok_or(OperationError::InvalidInput(
                    "slice-update endpoint arithmetic overflow",
                ))?;

        if end > source_shape[axis] {
            return Err(OperationError::InvalidInput(
                "slice update exceeds source dimensions",
            ));
        }
        stop.push(end);
    }

    let (start, stop, strides) = slice_bounds(&source_shape, start, &stop)?;
    let mut output = PendingArray::new();

    // SAFETY: valid handles, same-rank float32 operands, and validated
    // bounds selecting exactly update.shape() with unit strides.
    let status = unsafe {
        sys::mlx_slice_update(
            &mut output.raw,
            source.as_raw(),
            update.as_raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_raw(),
        )
    };

    output.finish(status)
}

fn cache_shape(cache: &Array) -> OperationResult<[usize; 4]> {
    require_f32(cache)?;
    let shape = cache.shape();

    if shape.len() != 4 {
        return Err(OperationError::InvalidInput(
            "cache layout must be [batch, heads, tokens, head_dim]",
        ));
    }
    require_positive_shape(&shape)?;

    Ok([shape[0], shape[1], shape[2], shape[3]])
}

/// Extract cache positions [start_token, stop_token).
///
/// Cache layout: [batch, heads, capacity, head_dim].
///
/// This checks physical bounds only. A higher-level cache owner must
/// additionally ensure stop_token does not exceed valid logical length.
pub fn cache_slice(
    cache: &Array,
    start_token: usize,
    stop_token: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    let [batch, heads, capacity, width] = cache_shape(cache)?;

    if start_token >= stop_token || stop_token > capacity {
        return Err(OperationError::InvalidInput(
            "cache slice must be nonempty and within capacity",
        ));
    }

    slice(
        cache,
        &[0, 0, start_token, 0],
        &[batch, heads, stop_token, width],
        stream,
    )
}

/// Return a cache with update written at offset along its token axis.
///
/// cache: [batch, heads, capacity, head_dim].
/// update: [batch, heads, chunk_tokens, head_dim].
///
/// The caller must retain the returned array, track logical positions,
/// and commit K/V updates together at the cache-owner layer.
pub fn cache_update(
    cache: &Array,
    update: &Array,
    offset: usize,
    stream: &Stream,
) -> OperationResult<Array> {
    let source_shape = cache_shape(cache)?;
    let update_shape = cache_shape(update)?;

    if source_shape[0] != update_shape[0]
        || source_shape[1] != update_shape[1]
        || source_shape[3] != update_shape[3]
    {
        return Err(OperationError::InvalidInput(
            "cache update batch, head count, and head width must match",
        ));
    }

    let end = offset
        .checked_add(update_shape[2])
        .ok_or(OperationError::InvalidInput(
            "cache update position arithmetic overflow",
        ))?;

    if end > source_shape[2] {
        return Err(OperationError::InvalidInput(
            "cache update exceeds allocated token capacity",
        ));
    }

    slice_update(cache, update, &[0, 0, offset, 0], stream)
}

/// Check that every float32 element is finite on the selected stream.
///
/// MLX evaluates `isfinite` and an `all` reduction natively. Only the
/// resulting scalar Boolean is read on the host; the input tensor is not
/// copied into a Rust buffer. Empty arrays satisfy the predicate.
///
/// This is an explicit validation/synchronization boundary. Arithmetic
/// wrappers themselves remain lazy and do not call this helper implicitly.
pub fn all_finite(input: &Array, stream: &Stream) -> OperationResult<bool> {
    require_f32(input)?;
    let mut finite = PendingArray::new();

    // SAFETY: valid borrowed input/stream and writable owned output.
    let status = unsafe { sys::mlx_isfinite(&mut finite.raw, input.as_raw(), stream.as_raw()) };
    let finite = finite.finish(status)?;

    let mut reduced = PendingArray::new();
    // SAFETY: isfinite produces a Boolean array. Reducing every axis with
    // keepdims=false produces exactly one Boolean scalar.
    let status = unsafe { sys::mlx_all(&mut reduced.raw, finite.as_raw(), false, stream.as_raw()) };
    let reduced = reduced.finish(status)?;
    reduced.eval()?;
    stream.synchronize()?;

    let mut result = false;
    // SAFETY: reduced is an evaluated Boolean scalar, and result is a
    // valid writable target. The accessor copies the single value.
    let status = unsafe { sys::mlx_array_item_bool(&mut result, reduced.as_raw()) };
    if status != 0 {
        return Err(OperationError::Mlx(Error::Status(status)));
    }
    Ok(result)
}

/// Return a row-major contiguous float32 array on the selected stream.
///
/// This may allocate. It is not a promise of zero-copy conversion.
pub fn contiguous(input: &Array, stream: &Stream) -> OperationResult<Array> {
    require_f32(input)?;
    let mut output = PendingArray::new();

    // SAFETY: valid input/stream and writable output. false requires
    // row-major rather than allowing column-major contiguity.
    let status =
        unsafe { sys::mlx_contiguous(&mut output.raw, input.as_raw(), false, stream.as_raw()) };

    output.finish(status)
}

/// Evaluate and copy a float32 tensor into a Rust vector.
///
/// This is an explicit synchronization/readback boundary for tests,
/// diagnostics, or final outputs—not an intermediate execution kernel.
///
/// A contiguous array is obtained before accessing its data pointer.
pub fn to_vec_f32(input: &Array, stream: &Stream) -> OperationResult<Vec<f32>> {
    let input = contiguous(input, stream)?;
    input.eval()?;
    stream.synchronize()?;

    let data = input.data_f32()?;
    let mut output = metadata_vec(data.len())?;
    output.extend_from_slice(data);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, DeviceType};

    const EPSILON: f32 = 1e-6;
    const INPUT: [f32; 8] = [0.5, -1.0, 2.0, 0.25, -0.75, 0.125, 1.5, -2.0];
    const NORM_WEIGHT: [f32; 4] = [1.0, 0.75, 1.25, 0.5];
    const LINEAR_WEIGHT: [f32; 12] = [
        0.25, -0.5, 0.75, 1.0, -1.0, 0.125, 0.5, -0.25, 0.5, 0.25, -0.125, 0.75,
    ];
    const RESIDUAL: [f32; 6] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6];

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(actual.is_finite());
            assert!(expected.is_finite());
            let tolerance = 1e-4 + 1e-4 * expected.abs();

            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual}, expected={expected}",
            );
        }
    }

    fn stream(kind: DeviceType) -> Stream {
        let device = Device::new(kind, 0).expect("explicit device");
        assert!(
            device.is_available().expect("availability"),
            "requested device unavailable: {kind:?}",
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
        stream
    }

    fn reference() -> Vec<f32> {
        let mut output = Vec::with_capacity(6);

        for row in 0..2 {
            let start = row * 4;
            let mean_square = INPUT[start..start + 4]
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                / 4.0;
            let denominator = (mean_square + f64::from(EPSILON)).sqrt();

            let normalized: Vec<f64> = (0..4)
                .map(|index| {
                    f64::from(INPUT[start + index]) * f64::from(NORM_WEIGHT[index]) / denominator
                })
                .collect();

            for column in 0..3 {
                let projected = (0..4)
                    .map(|index| normalized[index] * f64::from(LINEAR_WEIGHT[column * 4 + index]))
                    .sum::<f64>();

                let activated = projected / (1.0 + (-projected).exp());
                output.push((activated + f64::from(RESIDUAL[row * 3 + column])) as f32);
            }
        }

        output
    }

    fn execute_graph(kind: DeviceType) -> Vec<f32> {
        let stream = stream(kind);
        let input = Array::from_f32_slice(&INPUT, &[2, 4]).expect("input");
        let norm_weight = Array::from_f32_slice(&NORM_WEIGHT, &[4]).expect("norm weight");
        let linear_weight = Array::from_f32_slice(&LINEAR_WEIGHT, &[3, 4]).expect("weight");
        let residual = Array::from_f32_slice(&RESIDUAL, &[2, 3]).expect("residual");

        let normalized = rms_norm(&input, &norm_weight, EPSILON, &stream).expect("MLX RMSNorm");
        let projected = linear(&normalized, &linear_weight, &stream).expect("MLX projection");
        let activated = silu(&projected, &stream).expect("MLX SiLU");
        let output = residual_add(&activated, &residual, &stream).expect("MLX residual");

        assert_eq!(output.shape(), vec![2, 3]);
        assert_eq!(output.dtype(), DType::Float32);

        let values = to_vec_f32(&output, &stream).expect("read output");
        assert_close(&values, &reference());
        values
    }

    fn execute_tensor_suite(kind: DeviceType) {
        let stream = stream(kind);

        let table_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let table = Array::from_f32_slice(&table_data, &[4, 4]).expect("table");
        let embedded = embedding_lookup(&table, &[2, 0, 2], &stream).expect("gather");

        assert_eq!(embedded.shape(), vec![3, 4]);
        assert_close(
            &to_vec_f32(&embedded, &stream).expect("embeddings"),
            &[
                8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0,
            ],
        );

        let sequence_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let sequence = Array::from_f32_slice(&sequence_data, &[12]).expect("sequence");
        let shaped = reshape(&sequence, &[2, 2, 3], &stream).expect("reshape");
        let transposed = transpose(&shaped, &[1, 0, 2], &stream).expect("transpose");

        assert_eq!(transposed.shape(), vec![2, 2, 3]);
        assert_close(
            &to_vec_f32(&transposed, &stream).expect("transposed output"),
            &[0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 9.0, 10.0, 11.0],
        );

        let gate_data = [-2.0_f32, -0.5, 0.0, 0.5, 1.0, 2.0];
        let up_data = [0.25_f32, -1.0, 3.0, 2.0, -0.75, 0.5];
        let gate = Array::from_f32_slice(&gate_data, &[2, 3]).expect("gate");
        let up = Array::from_f32_slice(&up_data, &[2, 3]).expect("up");
        let output = swiglu(&gate, &up, &stream).expect("SwiGLU");

        let expected: Vec<f32> = gate_data
            .iter()
            .zip(up_data)
            .map(|(&gate, up)| {
                let gate = f64::from(gate);
                (gate / (1.0 + (-gate).exp()) * f64::from(up)) as f32
            })
            .collect();

        assert_close(
            &to_vec_f32(&output, &stream).expect("SwiGLU output"),
            &expected,
        );

        let original = Array::from_f32_slice(&[0.0; 20], &[1, 2, 5, 2]).expect("cache");
        let first = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2])
            .expect("first chunk");
        let second =
            Array::from_f32_slice(&[9.0, 10.0, 11.0, 12.0], &[1, 2, 1, 2]).expect("second chunk");

        let updated = cache_update(&original, &first, 1, &stream).expect("update one");
        let updated = cache_update(&updated, &second, 3, &stream).expect("update two");
        let selected = cache_slice(&updated, 1, 4, &stream).expect("cache positions");

        assert_eq!(selected.shape(), vec![1, 2, 3, 2]);
        assert_close(
            &to_vec_f32(&selected, &stream).expect("cache contents"),
            &[
                1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0,
            ],
        );

        assert_close(
            &to_vec_f32(&original, &stream).expect("original version"),
            &[0.0; 20],
        );

        let unchanged_tail = cache_slice(&updated, 4, 5, &stream).expect("unmodified tail");
        assert_close(
            &to_vec_f32(&unchanged_tail, &stream).expect("tail"),
            &[0.0; 4],
        );
    }

    fn packed_embedding_suite(kind: DeviceType) {
        let stream = stream(kind);
        let words: Vec<u32> = (0..4)
            .flat_map(|row| {
                let word = (0..8).fold(0_u32, |word, column| {
                    word | (((row * 3 + column) % 16) as u32) << (column * 4)
                });
                [word; 8]
            })
            .collect();
        let scales_data = [0.25_f32, 0.5, 0.125, 0.75];
        let biases_data = [-1.0_f32, -0.5, 0.25, 1.0];
        let weight = Array::from_u32_slice(&words, &[4, 8]).expect("packed table");
        let scales = Array::from_f32_slice(&scales_data, &[4, 1]).expect("scales");
        let biases = Array::from_f32_slice(&biases_data, &[4, 1]).expect("biases");
        let tokens = [3_u32, 0, 3];
        let result = quantized_embedding_lookup(&weight, &scales, &biases, &tokens, 64, &stream)
            .expect("packed row lookup");
        assert_eq!(result.shape(), vec![3, 64]);
        assert_eq!(result.dtype(), DType::Float32);
        let expected: Vec<f32> = tokens
            .iter()
            .flat_map(|&row| {
                (0..64).map(move |column| {
                    ((row as usize * 3 + column % 8) % 16) as f32 * scales_data[row as usize]
                        + biases_data[row as usize]
                })
            })
            .collect();
        assert_close(
            &to_vec_f32(&result, &stream).expect("selected rows"),
            &expected,
        );
        assert_eq!(weight.dtype(), DType::Uint32);
        assert_eq!(weight.shape(), vec![4, 8]);
        assert!(quantized_embedding_lookup(&weight, &scales, &biases, &[], 64, &stream).is_err());
        assert!(quantized_embedding_lookup(&weight, &scales, &biases, &[4], 64, &stream).is_err());
        assert!(
            quantized_embedding_lookup(&weight, &scales, &biases, &[u32::MAX], 64, &stream)
                .is_err()
        );
        assert!(quantized_embedding_lookup(&weight, &scales, &biases, &[0], 0, &stream).is_err());
        assert!(quantized_embedding_lookup(&weight, &scales, &biases, &[0], 32, &stream).is_err());
        let wrong_scales = Array::from_f32_slice(&[1.0; 3], &[3, 1]).expect("wrong vocabulary");
        assert!(
            quantized_embedding_lookup(&weight, &wrong_scales, &wrong_scales, &[0], 64, &stream)
                .is_err()
        );
        assert!(quantized_embedding_lookup(&scales, &scales, &biases, &[0], 64, &stream).is_err());
    }

    #[test]
    fn mlx_cpu_quantized_embedding_uses_only_selected_rows() {
        packed_embedding_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_quantized_embedding_uses_only_selected_rows() {
        packed_embedding_suite(DeviceType::Gpu);
    }

    #[test]
    fn mlx_cpu_projection_graph_matches_reference() {
        execute_graph(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_projection_graph_matches_cpu_and_reference() {
        let metal = execute_graph(DeviceType::Gpu);
        let cpu = execute_graph(DeviceType::Cpu);
        assert_close(&metal, &cpu);
    }

    #[test]
    fn mlx_cpu_embedding_shapes_swiglu_and_cache_updates() {
        execute_tensor_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_embedding_shapes_swiglu_and_cache_updates() {
        execute_tensor_suite(DeviceType::Gpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn unified_weights_are_consumed_across_cpu_and_metal_streams() {
        let cpu = stream(DeviceType::Cpu);
        let metal = stream(DeviceType::Gpu);

        // Construct one weight array. Both streams consume this same
        // array handle; there is no second model upload.
        let weights = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
            .expect("shared weights");

        let on_cpu = embedding_lookup(&weights, &[2, 0], &cpu).expect("CPU lookup");
        let on_metal = embedding_lookup(&weights, &[2, 0], &metal).expect("Metal lookup");

        // A Metal operation consumes a CPU-produced graph result.
        // No intermediate readback or explicit transfer is performed.
        let squared = multiply(&on_cpu, &on_metal, &metal).expect("mixed graph");

        // Final readback is requested on the CPU stream. MLX must honor
        // the dependencies from both producing streams.
        assert_close(
            &to_vec_f32(&squared, &cpu).expect("mixed-stream output"),
            &[25.0, 36.0, 1.0, 4.0],
        );

        metal.synchronize().expect("Metal completion");
        cpu.synchronize().expect("CPU completion");
    }

    fn run_all_finite_suite(kind: DeviceType) {
        let stream = stream(kind);
        assert_eq!(
            stream
                .device()
                .expect("stream device")
                .kind()
                .expect("kind"),
            kind
        );
        let finite = Array::from_f32_slice(&[0.0, -0.0, f32::MIN, f32::MAX, 1.0, -2.0], &[2, 3])
            .expect("finite input");
        let transposed = transpose(&finite, &[1, 0], &stream).expect("strided input");
        assert!(all_finite(&transposed, &stream).expect("native finite reduction"));

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let source = Array::from_f32_slice(&[1.0, invalid, 2.0, 3.0, 4.0, 5.0], &[2, 3])
                .expect("mixed input");
            let included = slice(&source, &[0, 1], &[2, 2], &stream).expect("non-finite column");
            let excluded = slice(&source, &[0, 0], &[2, 1], &stream).expect("finite column");
            assert!(!all_finite(&included, &stream).expect("strided non-finite reduction"));
            assert!(all_finite(&excluded, &stream).expect("strided finite reduction"));
        }

        // Validate the produced lazy values, not just their finite inputs.
        let maximum = Array::from_f32_slice(&[f32::MAX], &[1]).expect("finite scalar input");
        let overflow = multiply(&maximum, &maximum, &stream).expect("lazy native overflow");
        assert!(!overflow.is_available().expect("lazy graph"));
        assert!(!all_finite(&overflow, &stream).expect("native graph validation"));

        let empty = Array::from_f32_slice(&[], &[0]).expect("empty input");
        assert!(all_finite(&empty, &stream).expect("empty reduction identity"));
        let integer = Array::from_u32_slice(&[1, 2], &[2]).expect("integer input");
        assert!(matches!(
            all_finite(&integer, &stream),
            Err(OperationError::InvalidInput(_)),
        ));
    }

    #[test]
    fn mlx_cpu_all_finite_reduces_native_values_and_strided_views() {
        run_all_finite_suite(DeviceType::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_metal_all_finite_reduces_native_values_and_strided_views() {
        run_all_finite_suite(DeviceType::Gpu);
    }

    #[test]
    fn linear_rejects_incompatible_widths() {
        let stream = stream(DeviceType::Cpu);
        let input = Array::from_f32_slice(&[1.0; 8], &[2, 4]).expect("input");
        let weight = Array::from_f32_slice(&[1.0; 9], &[3, 3]).expect("weight");

        assert!(matches!(
            linear(&input, &weight, &stream),
            Err(OperationError::InvalidInput(_)),
        ));
    }

    #[test]
    fn rms_norm_rejects_invalid_epsilon_and_weight_width() {
        let stream = stream(DeviceType::Cpu);
        let input = Array::from_f32_slice(&[1.0; 8], &[2, 4]).expect("input");
        let correct = Array::from_f32_slice(&[1.0; 4], &[4]).expect("weight");
        let wrong = Array::from_f32_slice(&[1.0; 3], &[3]).expect("wrong weight");

        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                rms_norm(&input, &correct, epsilon, &stream),
                Err(OperationError::InvalidInput(_)),
            ));
        }

        assert!(matches!(
            rms_norm(&input, &wrong, EPSILON, &stream),
            Err(OperationError::InvalidInput(_)),
        ));
    }

    #[test]
    fn residual_add_rejects_broadcasting() {
        let stream = stream(DeviceType::Cpu);
        let input = Array::from_f32_slice(&[1.0; 6], &[2, 3]).expect("input");
        let residual = Array::from_f32_slice(&[1.0; 3], &[3]).expect("residual");

        assert!(matches!(
            residual_add(&input, &residual, &stream),
            Err(OperationError::InvalidInput(_)),
        ));
    }

    #[test]
    fn embedding_rejects_empty_and_out_of_vocabulary_tokens() {
        let stream = stream(DeviceType::Cpu);
        let table = Array::from_f32_slice(&[1.0; 8], &[4, 2]).expect("table");

        assert!(embedding_lookup(&table, &[], &stream).is_err());
        assert!(embedding_lookup(&table, &[4], &stream).is_err());
        assert!(embedding_lookup(&table, &[u32::MAX], &stream).is_err());
    }

    #[test]
    fn reshape_and_transpose_reject_invalid_metadata() {
        let stream = stream(DeviceType::Cpu);
        let input = Array::from_f32_slice(&[1.0; 6], &[2, 3]).expect("input");

        assert!(reshape(&input, &[4, 2], &stream).is_err());
        assert!(reshape(&input, &[usize::MAX, 2], &stream).is_err());
        assert!(reshape(&input, &[0, 6], &stream).is_err());

        assert!(transpose(&input, &[0, 0], &stream).is_err());
        assert!(transpose(&input, &[0, 2], &stream).is_err());
        assert!(transpose(&input, &[0], &stream).is_err());
    }

    #[test]
    fn elementwise_operations_reject_broadcasting() {
        let stream = stream(DeviceType::Cpu);
        let input = Array::from_f32_slice(&[1.0; 6], &[2, 3]).expect("input");
        let vector = Array::from_f32_slice(&[1.0; 3], &[3]).expect("vector");

        assert!(multiply(&input, &vector, &stream).is_err());
        assert!(swiglu(&input, &vector, &stream).is_err());
    }

    #[test]
    fn cache_operations_reject_invalid_ranges_and_layouts() {
        let stream = stream(DeviceType::Cpu);
        let cache = Array::from_f32_slice(&[0.0; 20], &[1, 2, 5, 2]).expect("cache");
        let update = Array::from_f32_slice(&[1.0; 8], &[1, 2, 2, 2]).expect("update");
        let wrong_heads = Array::from_f32_slice(&[1.0; 4], &[1, 1, 2, 2]).expect("wrong heads");

        assert!(cache_slice(&cache, 0, 0, &stream).is_err());
        assert!(cache_slice(&cache, 4, 6, &stream).is_err());
        assert!(cache_update(&cache, &update, 4, &stream).is_err());
        assert!(cache_update(&cache, &update, usize::MAX, &stream).is_err());
        assert!(cache_update(&cache, &wrong_heads, 0, &stream).is_err());

        assert!(slice(&cache, &[0], &[1], &stream).is_err());
    }
}
