/// Thin owned tensor. The kernels' bodies go through mlx-ffi arrays on the
/// CPU device; this type is the crate's public shape/data contract.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "shape {:?} implies {} elements, got {}",
            shape,
            n,
            data.len()
        );
        Tensor { shape: shape.to_vec(), data }
    }
}