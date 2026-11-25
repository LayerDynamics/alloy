//! Sequential SIMD gated-delta evaluation matching the pinned mlx-lm numerical
//! path. The recurrence equations are shared with the independent scalar test.
//! Unlike MLX's architecture-selected chunk kernel, this keeps the reduction
//! order independent of sequence length. No host tensor arithmetic is used.
use super::*;
use crate::VectorArray;
use std::ffi::CStr;

struct Strings(sys::mlx_vector_string);
impl Strings {
    fn new(names: &[&CStr]) -> OperationResult<Self> {
        let mut pointers: Vec<_> = names.iter().map(|name| name.as_ptr()).collect();
        // SAFETY: live NUL-terminated strings, writable pointer list of given size.
        let raw = unsafe { sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len()) };
        if raw.ctx.is_null() {
            return Err(Error::NullHandle.into());
        }
        Ok(Self(raw))
    }
}
impl Drop for Strings {
    fn drop(&mut self) {
        // SAFETY: this wrapper exclusively owns the vector handle.
        unsafe {
            sys::mlx_vector_string_free(self.0);
        }
    }
}
struct Kernel(sys::mlx_fast_metal_kernel);
impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: this wrapper exclusively owns the kernel handle.
        unsafe {
            sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}
struct Config(sys::mlx_fast_metal_kernel_config);
impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: this wrapper exclusively owns the configuration handle.
        unsafe {
            sys::mlx_fast_metal_kernel_config_free(self.0);
        }
    }
}
fn status(value: i32) -> OperationResult<()> {
    if value == 0 {
        Ok(())
    } else {
        Err(Error::Status(value).into())
    }
}
const SOURCE: &CStr = c"
    uint head = thread_position_in_grid.z % Hv;
    uint batch = thread_position_in_grid.z / Hv;
    uint value_index = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    constexpr uint width = Dk / 32;
    uint key_head = head / (Hv / Hk);
    uint initial_offset = ((batch * Hv + head) * Dv + value_index) * Dk;
    float memory[width];
    for (uint j=0; j<width; ++j) {
        memory[j] = state[initial_offset + width * lane + j];
    }
    for (uint token=0; token<T; ++token) {
        uint key_offset = ((batch * T + token) * Hk + key_head) * Dk;
        uint gate_offset = (batch * T + token) * Hv + head;
        uint value_offset = gate_offset * Dv + value_index;
        float prediction = 0.0f;
        for (uint j=0; j<width; ++j) {
            uint channel = width * lane + j;
            memory[j] = memory[j] * decay[gate_offset];
            prediction += memory[j] * keys[key_offset + channel];
        }
        prediction = simd_sum(prediction);
        float correction = (values[value_offset] - prediction) * beta[gate_offset];
        float result = 0.0f;
        for (uint j=0; j<width; ++j) {
            uint channel = width * lane + j;
            memory[j] = memory[j] + keys[key_offset + channel] * correction;
            result += memory[j] * queries[key_offset + channel];
        }
        result = simd_sum(result);
        if (thread_index_in_simdgroup == 0) { output[value_offset] = result; }
    }
    for (uint j=0; j<width; ++j) {
        final_state[initial_offset + width * lane + j] = memory[j];
    }
";

/// Caller has validated float32 ranks, grouped-head geometry and state shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    initial: Option<&Array>,
    stream: &Stream,
) -> OperationResult<(Array, Array)> {
    let qs = q.shape();
    let vs = v.shape();
    let state_shape = [qs[0], vs[2], vs[3], qs[3]];
    for shape in [&qs[..], &vs[..], &state_shape[..]] {
        shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or(OperationError::InvalidInput(
                "sequential delta index overflow",
            ))?;
    }
    let zero;
    let state = if let Some(state) = initial {
        state
    } else {
        zero = broadcast_to(&Array::from_f32_slice(&[0.0], &[])?, &state_shape, stream)?;
        &zero
    };
    let t = Array::from_u32_slice(&[qs[1] as u32], &[])?;
    let inputs = VectorArray::from_arrays(&[
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        state.clone(),
        t,
    ])?;
    let input_names = Strings::new(&[
        c"queries", c"keys", c"values", c"decay", c"beta", c"state", c"T",
    ])?;
    let output_names = Strings::new(&[c"output", c"final_state"])?;
    // SAFETY: owned string vectors and constant terminated source/header. MLX
    // copies these into the kernel; contiguous input materialization is enabled.
    let raw = unsafe {
        sys::mlx_fast_metal_kernel_new(
            c"alloy_sequential_delta_f32".as_ptr(),
            input_names.0,
            output_names.0,
            SOURCE.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        )
    };
    if raw.ctx.is_null() {
        return Err(Error::NullHandle.into());
    }
    let kernel = Kernel(raw);
    // SAFETY: constructs an owned native config, checked before use.
    let raw = unsafe { sys::mlx_fast_metal_kernel_config_new() };
    if raw.ctx.is_null() {
        return Err(Error::NullHandle.into());
    }
    let config = Config(raw);
    for (name, value) in [
        (c"Dk", qs[3]),
        (c"Dv", vs[3]),
        (c"Hk", qs[2]),
        (c"Hv", vs[2]),
    ] {
        // SAFETY: valid owned config, static name and bounded positive dimension.
        status(unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config.0,
                name.as_ptr(),
                native_dimension(value)?,
            )
        })?;
    }
    for shape in [&vs[..], &state_shape[..]] {
        let dimensions: Vec<_> = shape
            .iter()
            .copied()
            .map(native_dimension)
            .collect::<OperationResult<_>>()?;
        // SAFETY: valid config and live rank-matched dimension buffer.
        status(unsafe {
            sys::mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                dimensions.as_ptr(),
                dimensions.len(),
                sys::mlx_dtype__MLX_FLOAT32,
            )
        })?;
    }
    // SAFETY: shapes and grid product bounded above; one simdgroup computes
    // each (batch,value-head,value-channel) row, with exactly 32 key lanes.
    unsafe {
        status(sys::mlx_fast_metal_kernel_config_set_grid(
            config.0,
            32,
            native_dimension(vs[3])?,
            native_dimension(qs[0] * vs[2])?,
        ))?;
        status(sys::mlx_fast_metal_kernel_config_set_thread_group(
            config.0, 32, 4, 1,
        ))?;
    }
    // SAFETY: owned result slot. RAII frees partial native output on error.
    let mut output = VectorArray::from_raw(unsafe { sys::mlx_vector_array_new() })?;
    // SAFETY: every input/output shape and handle satisfies the kernel contract.
    status(unsafe {
        sys::mlx_fast_metal_kernel_apply(
            &mut output.raw,
            kernel.0,
            inputs.as_raw(),
            config.0,
            stream.as_raw(),
        )
    })?;
    if output.len() != 2 {
        return Err(OperationError::InvalidInput(
            "sequential delta result count",
        ));
    }
    Ok((output.get(0)?, output.get(1)?))
}
