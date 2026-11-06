use crate::error::KernelError;
use mlx_ffi::{Array, DType, Device, Stream, ops};
use std::cell::OnceCell;

/// Immutable MLX-backed `float32` tensor.
///
/// [`Tensor::new`] copies and evaluates a host buffer. [`Tensor::from_array`]
/// adopts a native array, including a lazy or strided operation result,
/// without reading its contents. RoPE and attention build native graphs on
/// the selected CPU or Metal stream and return lazy tensors.
///
/// [`Tensor::data_on_stream`] is an explicit evaluation/readback boundary.
/// It retains a row-major contiguous native view before exposing a slice;
/// arbitrary native strides are never interpreted as contiguous memory.
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    array: Array,
    readback: OnceCell<Array>,
}

impl Tensor {
    /// New tensor; `data` is copied into an MLX array and evaluated.
    pub fn new(shape: &[usize], data: Vec<f32>) -> Result<Self, KernelError> {
        let expected = checked_elements(shape)?;
        if expected != data.len() {
            return Err(KernelError::ShapeMismatch {
                expected,
                got: data.len(),
            });
        }
        let dims: Vec<i32> = shape
            .iter()
            .map(|&dimension| {
                i32::try_from(dimension)
                    .map_err(|_| KernelError::InvalidInput("tensor dimension exceeds C int"))
            })
            .collect::<Result<_, _>>()?;
        let array = Array::from_f32_slice(&data, &dims)?;
        array.eval()?;
        Ok(Tensor {
            shape: shape.to_vec(),
            array,
            readback: OnceCell::new(),
        })
    }

    /// Adopt an owned native float32 array without evaluation or host copies.
    ///
    /// The array may be lazy or noncontiguous. Its shape and element count
    /// are validated before it can be used through this tensor interface.
    pub fn from_array(array: Array) -> Result<Self, KernelError> {
        if array.dtype() != DType::Float32 {
            return Err(KernelError::InvalidInput("tensor requires float32 data"));
        }
        let shape = array.shape();
        let expected = checked_elements(&shape)?;
        if expected != array.size() {
            return Err(KernelError::ShapeMismatch {
                expected,
                got: array.size(),
            });
        }
        Ok(Self {
            shape,
            array,
            readback: OnceCell::new(),
        })
    }

    /// Row-major shape, fixed at construction.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the native array for further explicit-stream operations.
    ///
    /// This does not evaluate the array or create a separate device copy.
    pub fn as_array(&self) -> &Array {
        &self.array
    }

    /// Evaluate and expose row-major data using the native default device.
    ///
    /// This convenience path uses that device's default stream. It does
    /// not substitute a CPU stream when the selected device is Metal.
    pub fn data(&self) -> Result<&[f32], KernelError> {
        self.data_on_stream(&default_stream()?)
    }

    /// Evaluate a contiguous native view on `stream` and borrow its data.
    ///
    /// The first call synchronizes the selected stream and retains the
    /// evaluated view for the tensor's lifetime. Later calls reuse that
    /// immutable view. MLX arrays can be consumed by both CPU and Metal
    /// streams; there is no separate host tensor representation here.
    pub fn data_on_stream(&self, stream: &Stream) -> Result<&[f32], KernelError> {
        if self.array.size() == 0 {
            return Ok(&[]);
        }
        let readback = match self.readback.get() {
            Some(readback) => readback,
            None => {
                let contiguous = ops::contiguous(&self.array, stream)?;
                contiguous.eval()?;
                stream.synchronize()?;
                self.readback.get_or_init(|| contiguous)
            }
        };
        readback.data_f32().map_err(KernelError::from)
    }
}

fn checked_elements(shape: &[usize]) -> Result<usize, KernelError> {
    i32::try_from(shape.len())
        .map_err(|_| KernelError::InvalidInput("tensor rank exceeds C int"))?;
    let elements = shape.iter().try_fold(1_usize, |count, &dimension| {
        i32::try_from(dimension)
            .map_err(|_| KernelError::InvalidInput("tensor dimension exceeds C int"))?;
        count
            .checked_mul(dimension)
            .ok_or(KernelError::InvalidInput(
                "tensor element count overflows usize",
            ))
    })?;
    let bytes =
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(KernelError::InvalidInput(
                "tensor byte count overflows usize",
            ))?;
    if bytes > isize::MAX as usize {
        return Err(KernelError::InvalidInput(
            "tensor byte count exceeds the maximum readable slice",
        ));
    }
    Ok(elements)
}

pub(crate) fn default_stream() -> Result<Stream, KernelError> {
    let device = Device::get_default()?;
    Stream::get_default(&device).map_err(KernelError::from)
}
