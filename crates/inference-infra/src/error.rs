use std::fmt;

/// Numerical and FFI contract violations surfaced by the kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelError {
    /// A required input had zero elements (e.g. attention with no K/V rows).
    Empty,
    /// An input contained NaN or an infinity where finite values are required.
    NonFinite,
    /// Logits contained no finite value, so no distribution or argmax exists.
    NoFiniteLogits,
    /// Constructed shape implies a different element count than the data.
    ShapeMismatch { expected: usize, got: usize },
    /// The requested cache capacity (max_len * head_dim) overflows usize.
    CapacityTooLarge { max_len: usize, head_dim: usize },
    /// Reserving a cache buffer failed (allocation would exhaust memory).
    AllocationFailed { elements: usize },
    /// An mlx-c call failed.
    Mlx(mlx_ffi::Error),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Empty => write!(f, "required input was empty"),
            KernelError::NonFinite => write!(f, "input contained NaN or infinity"),
            KernelError::NoFiniteLogits => write!(f, "logits contain no finite value"),
            KernelError::ShapeMismatch { expected, got } => {
                write!(f, "shape implies {expected} elements, got {got}")
            }
            KernelError::CapacityTooLarge { max_len, head_dim } => {
                write!(f, "cache capacity {max_len} x {head_dim} overflows usize")
            }
            KernelError::AllocationFailed { elements } => {
                write!(f, "failed to reserve {elements} f32 elements")
            }
            KernelError::Mlx(e) => write!(f, "mlx-c failure: {e}"),
        }
    }
}

impl std::error::Error for KernelError {}
