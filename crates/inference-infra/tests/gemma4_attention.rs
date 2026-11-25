//! Independent scalar expectations for the E4B text attention kernels.
use inference_infra::decoder::gemma4::Gemma4Attention;
use mlx_ffi::{Array, Device, DeviceType, Stream, ops};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONFIG: &[u8] = include_bytes!("fixtures/gemma4_e4b_config.json");
fn config() -> Value {
    serde_json::from_slice(CONFIG).unwrap()
}
fn streams() -> Vec<Stream> {
    let mut devices = vec![DeviceType::Cpu];
    if cfg!(target_os = "macos") {
        devices.push(DeviceType::Gpu);
    }
    devices
        .into_iter()
        .map(|d| Stream::new(&Device::new(d, 0).unwrap()).unwrap())
        .collect()
}
fn array(data: &[f32], heads: usize, tokens: usize, dim: usize) -> Array {
    Array::from_f32_slice(data, &[1, heads as i32, tokens as i32, dim as i32]).unwrap()
}

#[test]
fn gemma4_shared_kv_sliding_full() {
    assert_eq!(
        format!("{:x}", Sha256::digest(CONFIG)),
        "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
    );
    let s = &streams()[0];
    let c = config();
    let g = Gemma4Attention::from_config(&c, s).unwrap();
    assert_eq!(g.layers().len(), 42);
    for (i, l) in g.layers().iter().enumerate() {
        assert_eq!(l.sliding, i % 6 != 5);
        assert_eq!(l.head_dim, if l.sliding { 256 } else { 512 });
        assert_eq!(
            l.kv_owner,
            if i < 24 {
                i
            } else if l.sliding {
                22
            } else {
                23
            }
        );
    }
    for (key, value) in [
        ("num_kv_shared_layers", json!(42)),
        ("num_kv_shared_layers", json!(41)),
        ("num_attention_heads", json!(7)),
        ("global_head_dim", json!(511)),
        ("attention_k_eq_v", json!(true)),
        ("layer_types", json!(["full_attention"])),
        ("rope_traditional", json!(true)),
    ] {
        let mut bad = c.clone();
        bad["text_config"][key] = value;
        assert!(Gemma4Attention::from_config(&bad, s).is_err(), "{key}");
    }
    let mut bad = c;
    bad["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] = json!(0.123);
    assert!(Gemma4Attention::from_config(&bad, s).is_err());
}

#[test]
fn gemma4_partial_rotation() {
    for s in streams() {
        let g = Gemma4Attention::from_config(&config(), &s).unwrap();
        for (layer, d, rotated, base) in [(0, 256, 256, 10000_f64), (5, 512, 128, 1000000_f64)] {
            let data: Vec<_> = (0..2 * 2 * d)
                .map(|i| ((i * 13 % 53) as f32 - 26.0) / 32.0)
                .collect();
            let input = array(&data, 2, 2, d);
            for offset in [0, 127, 4096] {
                let out = g.rotate_owner_keys(layer, &input, offset, &s).unwrap();
                let got = ops::to_vec_f32(&out, &s).unwrap();
                for (h, head) in got.chunks_exact(d).enumerate() {
                    let row = &data[h * d..(h + 1) * d];
                    for j in 0..d / 2 {
                        let a = row[j] as f64;
                        let b = row[d / 2 + j] as f64;
                        if j >= rotated / 2 {
                            assert_eq!((head[j], head[d / 2 + j]), (row[j], row[d / 2 + j]));
                            continue;
                        }
                        let angle = (offset + h % 2) as f64 / base.powf(2.0 * j as f64 / d as f64);
                        let (sin, cos) = angle.sin_cos();
                        for (actual, expected) in [
                            (head[j], a * cos - b * sin),
                            (head[d / 2 + j], a * sin + b * cos),
                        ] {
                            assert!(
                                (f64::from(actual) - expected).abs() < 0.001,
                                "layer {layer}, offset {offset}, pair {j}: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
            assert!(g.rotate_owner_keys(layer, &input, 131071, &s).is_err());
            assert!(
                g.rotate_owner_keys(if layer == 0 { 24 } else { 29 }, &input, 0, &s)
                    .is_err()
            );
            let queries = array(&data.repeat(4), 8, 2, d);
            let owner = g.rotate_queries(layer, &queries, 512, &s).unwrap();
            let shared = g
                .rotate_queries(if layer == 0 { 24 } else { 29 }, &queries, 512, &s)
                .unwrap();
            assert_eq!(
                ops::to_vec_f32(&owner, &s).unwrap(),
                ops::to_vec_f32(&shared, &s).unwrap()
            );
        }
    }
}

#[test]
fn gemma4_attention_masks_and_scale() {
    // Two query tokens follow three cached positions. The sliding view starts
    // at position two (one retained key); the global view begins at zero.
    let mut cfg = config();
    cfg["text_config"]["sliding_window"] = json!(2);
    for s in streams() {
        let g = Gemma4Attention::from_config(&cfg, &s).unwrap();
        for (layer, d, first) in [(24, 256, 2), (29, 512, 0)] {
            let n = 5 - first;
            let mut q = vec![0.0; 8 * 2 * d];
            for head in q.chunks_exact_mut(d) {
                head[0] = 1.0;
            }
            let q = array(&q, 8, 2, d);
            let mut k = vec![0.0; 2 * n * d];
            let mut v = k.clone();
            for (index, (kr, vr)) in k.chunks_exact_mut(d).zip(v.chunks_exact_mut(d)).enumerate() {
                let pos = first + index % n;
                kr[0] = pos as f32 * 0.5;
                vr.fill((pos + index / n * 10) as f32);
            }
            let k = array(&k, 2, n, d);
            let v = array(&v, 2, n, d);
            let out = g.attend(layer, &q, &k, &v, 3, &s).unwrap();
            let got = ops::to_vec_f32(&out, &s).unwrap();
            for (i, row) in got.chunks_exact(d).enumerate() {
                let end = 3 + i % 2;
                let start = if first == 0 { 0 } else { end - 1 };
                let weights: Vec<_> = (start..=end).map(|p| (p as f64 * 0.5).exp()).collect();
                let expected = weights
                    .iter()
                    .zip(start..=end)
                    .map(|(w, p)| w * (p + i / 8 * 10) as f64)
                    .sum::<f64>()
                    / weights.iter().sum::<f64>();
                assert!(row.iter().all(|v| (f64::from(*v) - expected).abs() < 1e-4));
            }
            assert!(g.attend(layer, &q, &k, &v, 0, &s).is_err());
        }
    }
}
