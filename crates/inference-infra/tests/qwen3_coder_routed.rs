//! Compare the first trained Qwen3-Coder routed MLP to the pinned eager upstream implementation.
use mlx_ffi::{Array, Device, DeviceType, Stream, ops};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

const SHARD_SHA: &str = "25c6d0a62e02224605774ce2e9dc9a8c95910cde33557b938e55a368da1b5212";

struct TrainedShard {
    file: File,
    offsets_start: u64,
    tensors: Value,
}

impl TrainedShard {
    fn open() -> Self {
        let root = PathBuf::from(std::env::var_os("ALLOY_QWEN3_CODER_ARTIFACT").unwrap());
        let mut file = File::open(root.join("model-00001-of-00004.safetensors")).unwrap();
        let mut sha = Sha256::new();
        std::io::copy(&mut file, &mut sha).unwrap();
        assert_eq!(format!("{:x}", sha.finalize()), SHARD_SHA);
        file.rewind().unwrap();
        let mut length = [0; 8];
        file.read_exact(&mut length).unwrap();
        let length = u64::from_le_bytes(length);
        assert!(length < 4 << 20);
        let mut header = vec![0; length as usize];
        file.read_exact(&mut header).unwrap();
        Self {
            file,
            offsets_start: length + 8,
            tensors: serde_json::from_slice(&header).unwrap(),
        }
    }

    fn read(&mut self, key: &str, dtype: &str, expected: &[i32]) -> Array {
        let header = &self.tensors[key];
        assert_eq!(header["dtype"], dtype);
        assert_eq!(header["shape"], serde_json::json!(expected));
        let offsets = &header["data_offsets"];
        let begin = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        let elements: usize = expected
            .iter()
            .map(|&d| usize::try_from(d).unwrap())
            .product();
        let size = elements * if dtype == "U32" { 4 } else { 2 };
        assert_eq!(end - begin, size as u64);
        self.file
            .seek(SeekFrom::Start(self.offsets_start + begin))
            .unwrap();
        let mut bytes = vec![0; size];
        self.file.read_exact(&mut bytes).unwrap();
        match dtype {
            "U32" => {
                let packed: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                Array::from_u32_slice(&packed, expected).unwrap()
            }
            "BF16" => {
                let values: Vec<_> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
                    .collect();
                Array::from_f32_slice(&values, expected).unwrap()
            }
            _ => unreachable!(),
        }
    }

    fn projection(&mut self, name: &str, leading: &[i32], width: i32, bits: i32) -> Projection {
        let prefix = format!("model.layers.0.mlp.{name}");
        let packed_shape = [leading, &[width / (32 / bits)]].concat();
        let meta_shape = [leading, &[width / 64]].concat();
        Projection {
            packed: self.read(&format!("{prefix}.weight"), "U32", &packed_shape),
            scales: self.read(&format!("{prefix}.scales"), "BF16", &meta_shape),
            biases: self.read(&format!("{prefix}.biases"), "BF16", &meta_shape),
        }
    }
}

struct Projection {
    packed: Array,
    scales: Array,
    biases: Array,
}

impl Projection {
    fn experts(&self) -> ops::AffineExperts<'_> {
        ops::AffineExperts {
            packed: &self.packed,
            scales: &self.scales,
            biases: &self.biases,
            group_size: 64,
            bits: 4,
        }
    }
}

#[test]
#[ignore = "requires pinned Qwen3-Coder checkpoint and eager layer reference"]
fn qwen3_coder_trained_routed_layer() {
    let bytes = fs::read(std::env::var_os("ALLOY_QWEN3_CODER_ROUTED_REFERENCE").unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "308fd3a7058432da34421ef028e18c4bc6af164e144cba9870407d99a6089b6e"
    );
    let expected: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(expected["shard_sha256"], SHARD_SHA);
    assert_eq!(expected["cases"].as_array().unwrap().len(), 2);
    let mut shard = TrainedShard::open();
    let router = shard.projection("gate", &[128], 2048, 8);
    let gate = shard.projection("switch_mlp.gate_proj", &[128, 768], 2048, 4);
    let up = shard.projection("switch_mlp.up_proj", &[128, 768], 2048, 4);
    let down = shard.projection("switch_mlp.down_proj", &[128, 2048], 768, 4);
    let banks = ops::SwitchGluExperts {
        gate: gate.experts(),
        up: up.experts(),
        down: down.experts(),
    };
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    for (i, tokens) in [1, 9].into_iter().enumerate() {
        let case = &expected["cases"][i];
        assert_eq!(case["tokens"], tokens);
        let inputs: Vec<_> = (0..tokens * 2048)
            .map(|j| ((j * 37) % 251 - 125) as f32 / 128.)
            .collect();
        let x = Array::from_f32_slice(&inputs, &[tokens, 2048]).unwrap();
        let logits = ops::quantized_matmul(
            &x,
            &router.packed,
            &router.scales,
            &router.biases,
            true,
            64,
            8,
            &stream,
        )
        .unwrap();
        let y = ops::routed_swiglu(&x, &logits, &banks, 8, true, &stream).unwrap();
        let actual = ops::to_vec_f32(&y, &stream).unwrap();
        let reference = case["output"].as_array().unwrap();
        assert_eq!(actual.len(), inputs.len());
        assert_eq!(actual.len(), reference.len());
        let (mut absolute, mut relative) = (0f64, 0f64);
        for (&observed, expected) in actual.iter().zip(reference) {
            let expected = expected.as_f64().unwrap();
            assert!(observed.is_finite() && expected.is_finite());
            let delta = (f64::from(observed) - expected).abs();
            absolute = absolute.max(delta);
            if delta != 0. {
                relative = relative.max(delta / expected.abs());
            }
        }
        eprintln!(
            "{tokens} tokens, {} activations: max_abs={absolute}, max_relative={relative}",
            actual.len()
        );
        assert!(absolute < 0.02 && relative < 0.01);
    }
}
