use crate::error::KernelError;
use mlx_ffi::Array;

/// MLX-backed `float32` tensor. Shape and data are private and only set
/// through [`Tensor::new`], which copies the buffer through mlx-c
/// (`mlx_array_new_data`) and evaluates it, so a caller cannot mutate
/// either after construction. The kernels' arithmetic stays host-side
/// until mlx-ffi grows op wrappers (see PR reply on comment 8).
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    array: Array,
}

impl Tensor {
    /// New tensor; `data` is copied into an MLX array and evaluated.
    pub fn new(shape: &[usize], data: Vec<f32>) -> Result<Self, KernelError> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(KernelError::ShapeMismatch {
                expected,
                got: data.len(),
            });
        }
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let array = Array::from_f32_slice(&data, &dims).map_err(KernelError::Mlx)?;
        array.eval().map_err(KernelError::Mlx)?;
        Ok(Tensor {
            shape: shape.to_vec(),
            array,
        })
    }

    /// Row-major shape, fixed at construction.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Contiguous `f32` data of the evaluated MLX array.
    pub fn data(&self) -> Result<&[f32], KernelError> {
        self.array.data_f32().map_err(KernelError::Mlx)
    }
}
