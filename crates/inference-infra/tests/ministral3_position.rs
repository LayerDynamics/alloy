//! Native position evidence only; full Ministral model qualification is separate.
use inference_infra::decoder::ministral3::{MinistralConfig, apply_position};
use mlx_ffi::{Array, Device, DeviceType, Stream, attention, ops};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONFIG: &str = include_str!("fixtures/ministral3_position/config.json");
const REFERENCE: &str = include_str!("fixtures/ministral3_position/reference.json");
fn config() -> MinistralConfig {
    MinistralConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
}
fn fixture_config() -> MinistralConfig {
    let mut c = config();
    c.query_heads = 4;
    c.kv_heads = 2;
    c
}
fn devices() -> Vec<(DeviceType, &'static str)> {
    let mut devices = vec![(DeviceType::Cpu, "cpu")];
    if cfg!(target_os = "macos") {
        devices.push((DeviceType::Gpu, "gpu"));
    }
    devices
}
fn data(heads: usize, tokens: usize, query: bool) -> Vec<f32> {
    let (multiplier, modulus, shift) = if query { (17, 97, 48) } else { (11, 89, 44) };
    (0..heads * tokens * 128)
        .map(|i| ((i * multiplier % modulus) as f32 - shift as f32) / 64.0)
        .collect()
}
fn input(heads: usize, query: bool) -> Array {
    Array::from_f32_slice(&data(heads, 3, query), &[1, heads as i32, 3, 128]).unwrap()
}
fn values(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn strict_compare(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len());
    let (mut absolute, mut relative) = (0.0_f32, 0.0_f32);
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        absolute = absolute.max((a - b).abs());
        if b == 0.0 {
            assert_eq!(a, b, "zero reference {name}");
        } else {
            relative = relative.max((a - b).abs() / b.abs());
        }
    }
    assert!(
        absolute < 0.02 && relative < 0.01,
        "{name}: maximum absolute {absolute}, maximum pointwise relative {relative}"
    );
    println!("{name}: maximum absolute {absolute}, maximum pointwise relative {relative}");
}

#[test]
fn ministral3_pinned_geometry_and_position_contract() {
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/ministral3_position/manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(CONFIG)),
        "cdd4bfa0a702d0241faaa3b80d15d481d7f8abf2659fba0958b1e7f24a28055f"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(REFERENCE)),
        "785dd4df6e3efe9652fbde77535e0541bb994dd2f190f3b0bcfef4257cbed243"
    );
    assert_eq!(
        manifest["files"]["generate.py"],
        format!(
            "{:x}",
            Sha256::digest(include_str!("fixtures/ministral3_position/generate.py"))
        )
    );
    let c = config();
    assert_eq!(
        (c.hidden, c.intermediate, c.layers, c.vocabulary),
        (3072, 9216, 26, 131072)
    );
    assert_eq!(
        (
            c.query_heads,
            c.kv_heads,
            c.head_dim,
            c.q_width(),
            c.kv_width()
        ),
        (32, 8, 128, 4096, 1024)
    );
    assert_eq!((c.context, c.yarn.original_context), (262144, 16384));
    assert_eq!(
        (
            c.yarn.theta,
            c.yarn.factor,
            c.yarn.beta_fast,
            c.yarn.beta_slow
        ),
        (1e6, 16.0, 32.0, 1.0)
    );
    assert_eq!(c.query_scaling_beta, 0.1);
    assert_eq!(c.yarn.rotary_amplitude(), 1.0);
    assert_eq!((c.quantization_bits, c.group_size), (4, Some(64)));
    assert!(c.tied_embeddings);
}

#[test]
fn ministral3_rejects_unsupported_position_and_geometry() {
    let base: Value = serde_json::from_str(CONFIG).unwrap();
    for (pointer, bad) in [
        ("/model_type", json!("llama")),
        ("/architectures", json!(["LlamaForCausalLM"])),
        ("/text_config/model_type", json!("mistral")),
        ("/text_config/hidden_size", json!(0)),
        ("/text_config/head_dim", json!(127)),
        ("/text_config/num_attention_heads", json!(0)),
        ("/text_config/num_key_value_heads", json!(7)),
        ("/text_config/rms_norm_eps", json!(-1)),
        ("/text_config/tie_word_embeddings", json!(null)),
        ("/text_config/hidden_act", json!("gelu")),
        ("/text_config/sliding_window", json!(4096)),
        ("/text_config/num_hidden_layers", json!(5000)),
        ("/text_config/max_position_embeddings", json!(8192)),
        ("/text_config/rope_parameters/type", json!("default")),
        ("/text_config/rope_parameters/rope_type", json!("llama3")),
        ("/text_config/rope_parameters/factor", json!(0.5)),
        ("/text_config/rope_parameters/rope_theta", json!(1.0)),
        ("/text_config/rope_parameters/beta_slow", json!(33.0)),
        ("/text_config/rope_parameters/beta_fast", json!(1e300)),
        ("/text_config/rope_parameters/mscale", json!(-1)),
        ("/text_config/rope_parameters/mscale_all_dim", json!(-1)),
        (
            "/text_config/rope_parameters/llama_4_scaling_beta",
            json!(-1),
        ),
        (
            "/text_config/rope_parameters/original_max_position_embeddings",
            json!(0),
        ),
        ("/quantization/bits", json!(8)),
        ("/quantization/group_size", json!(32)),
    ] {
        let mut v = base.clone();
        *v.pointer_mut(pointer).unwrap() = bad;
        assert!(
            MinistralConfig::from_json(&v).is_err(),
            "accepted {pointer}"
        );
    }
    for (key, bad) in [
        ("attention_bias", json!(true)),
        ("layer_types", json!(["sliding_attention"])),
        ("rope_scaling", json!({"type":"linear"})),
        ("partial_rotary_factor", json!(0.5)),
        ("quantization", json!({"bits":4})),
        ("rope_theta", json!(10000)),
    ] {
        let mut v = base.clone();
        v["text_config"][key] = bad;
        assert!(MinistralConfig::from_json(&v).is_err(), "accepted {key}");
    }
    let mut v = base.clone();
    v["text_config"]["rope_parameters"]["attention_factor"] = json!(2.0);
    assert!(MinistralConfig::from_json(&v).is_err());
    let mut v = base;
    v["text_config"]["layer_types"] = json!(vec!["full_attention"; 26]);
    assert!(MinistralConfig::from_json(&v).is_ok());
    let mut c = config();
    c.query_heads = usize::MAX;
    assert!(c.validate().is_err());
    let mut c = config();
    c.yarn.theta = f32::MAX;
    assert!(c.validate().is_err());
    let mut c = config();
    c.query_scaling_beta = f32::NAN;
    assert!(c.validate().is_err());
}

#[test]
fn ministral3_yarn_matches_pinned_reference() {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    for (kind, name) in devices() {
        let device = Device::new(kind, 0).unwrap();
        assert!(device.is_available().unwrap());
        let stream = Stream::new(&device).unwrap();
        for case in reference["devices"][name].as_array().unwrap() {
            let mut c = fixture_config();
            c.yarn.mscale_all_dim = case["mscale_all_dim"].as_f64().unwrap() as f32;
            let offset = case["offset"].as_u64().unwrap() as usize;
            let frequencies = attention::yarn_frequencies(&c.yarn, &stream).unwrap();
            strict_compare(
                &ops::to_vec_f32(&frequencies, &stream).unwrap(),
                &values(&case["frequencies"]),
                "frequencies",
            );
            let (q, k) =
                apply_position(&input(4, true), &input(2, false), &c, offset, &stream).unwrap();
            strict_compare(
                &ops::to_vec_f32(&q, &stream).unwrap(),
                &values(&case["query"]),
                &format!("{name} query offset={offset}"),
            );
            strict_compare(
                &ops::to_vec_f32(&k, &stream).unwrap(),
                &values(&case["key"]),
                &format!("{name} key offset={offset}"),
            );
        }
    }
}

#[test]
fn ministral3_position_boundary_and_chunk_equivalence() {
    for (kind, _) in devices() {
        let device = Device::new(kind, 0).unwrap();
        let stream = Stream::new(&device).unwrap();
        let mut c = fixture_config();
        let q = input(4, true);
        let k = input(2, false);
        for amplitude in [1.0, 0.0] {
            c.yarn.mscale_all_dim = amplitude;
            for offset in [0, 16383, 16384, 32767, 262141] {
                let (qr, kr) = apply_position(&q, &k, &c, offset, &stream).unwrap();
                let allq = ops::to_vec_f32(&qr, &stream).unwrap();
                let allk = ops::to_vec_f32(&kr, &stream).unwrap();
                // Every split reproduces all Q/K positions across scale boundaries.
                for split in [1, 2] {
                    for (start, end) in [(0, split), (split, 3)] {
                        let qs =
                            ops::slice(&q, &[0, 0, start, 0], &[1, 4, end, 128], &stream).unwrap();
                        let ks =
                            ops::slice(&k, &[0, 0, start, 0], &[1, 2, end, 128], &stream).unwrap();
                        let (qs, ks) =
                            apply_position(&qs, &ks, &c, offset + start, &stream).unwrap();
                        for (array, whole, heads) in [(&qs, &allq, 4), (&ks, &allk, 2)] {
                            let chunk = ops::to_vec_f32(array, &stream).unwrap();
                            for head in 0..heads {
                                assert_eq!(
                                    &chunk[head * (end - start) * 128
                                        ..(head + 1) * (end - start) * 128],
                                    &whole[(head * 3 + start) * 128..(head * 3 + end) * 128],
                                    "chunk mismatch {kind:?} offset={offset}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // Independent scalar formula checks all positions, heads and features;
        // this directly proves Q-only scaling at and beyond the first boundary.
        for offset in [0, 16383, 16384, 32767, 262141] {
            let scaled = attention::position_query_scale(&q, offset, 16384, 0.1, &stream).unwrap();
            let scaled = ops::to_vec_f32(&scaled, &stream).unwrap();
            let original = data(4, 3, true);
            for h in 0..4 {
                for t in 0..3 {
                    for d in 0..128 {
                        let i = (h * 3 + t) * 128 + d;
                        let expected = f64::from(original[i])
                            * (1.0 + 0.1 * (1.0 + ((offset + t) / 16384) as f64).ln());
                        assert!((f64::from(scaled[i]) - expected).abs() < 2e-7);
                    }
                }
            }
        }
        // Independent f64 rotation at low offsets avoids relying solely on the
        // same upstream native RoPE implementation for pair ordering/amplitude.
        c.yarn.mscale_all_dim = 0.0;
        let (_, rotated) = apply_position(&q, &k, &c, 0, &stream).unwrap();
        let actual = ops::to_vec_f32(&rotated, &stream).unwrap();
        let original = data(2, 3, false);
        let low = (128.0 * (16384.0 / (32.0 * std::f64::consts::TAU)).ln() / (2.0 * 1e6_f64.ln()))
            .floor()
            .max(0.0);
        let high = (128.0 * (16384.0 / std::f64::consts::TAU).ln() / (2.0 * 1e6_f64.ln()))
            .ceil()
            .min(127.0);
        for h in 0..2 {
            for t in 0..3 {
                for pair in 0..64 {
                    let base = (h * 3 + t) * 128;
                    let mask = 1.0 - ((pair as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let inv = 1e6_f64.powf(-(pair as f64) / 64.0) * (mask + (1.0 - mask) / 16.0);
                    let (sin, cos) = (t as f64 * inv).sin_cos();
                    let amplitude = 1.0 + 0.1 * 16.0_f64.ln();
                    let a = f64::from(original[base + pair]);
                    let b = f64::from(original[base + pair + 64]);
                    assert!(
                        (f64::from(actual[base + pair]) - amplitude * (a * cos - b * sin)).abs()
                            < 1e-6
                    );
                    assert!(
                        (f64::from(actual[base + pair + 64]) - amplitude * (a * sin + b * cos))
                            .abs()
                            < 1e-6
                    );
                }
            }
        }
    }
}

#[test]
fn ministral3_position_rejects_invalid_inputs() {
    let device = Device::new(DeviceType::Cpu, 0).unwrap();
    let stream = Stream::new(&device).unwrap();
    let c = fixture_config();
    let q = input(4, true);
    let k = input(2, false);
    for offset in [usize::MAX, c.context, c.context - 2] {
        assert!(apply_position(&q, &k, &c, offset, &stream).is_err());
    }
    assert!(apply_position(&k, &q, &c, 0, &stream).is_err());
    let short = ops::slice(&k, &[0, 0, 0, 0], &[1, 2, 1, 128], &stream).unwrap();
    assert!(apply_position(&q, &short, &c, 0, &stream).is_err());
    let rank = Array::from_f32_slice(&[1.0; 128], &[128]).unwrap();
    assert!(apply_position(&rank, &k, &c, 0, &stream).is_err());
    for (offset, original, beta) in [
        (usize::MAX, 16384, 0.1),
        (0, 0, 0.1),
        (0, 16384, f32::NAN),
        (0, 16384, -0.1),
    ] {
        assert!(attention::position_query_scale(&q, offset, original, beta, &stream).is_err());
    }
    let mut bad = c.yarn;
    bad.factor = f32::INFINITY;
    assert!(attention::yarn_frequencies(&bad, &stream).is_err());
    let mut bad = c.yarn;
    bad.dimensions = usize::MAX;
    assert!(attention::yarn_frequencies(&bad, &stream).is_err());
}
