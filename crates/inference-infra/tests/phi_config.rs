//! Native position/geometry evidence only, not real Phi model qualification.
use inference_infra::decoder::phi::{PhiConfig, PhiRopeRegime, partial_longrope};
use mlx_ffi::{Array, Device, DeviceType, Stream, ops};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const PINNED: &str = include_str!("fixtures/phi4_mini_config.json");
fn config() -> PhiConfig {
    PhiConfig::from_json(&serde_json::from_str(PINNED).unwrap()).unwrap()
}

#[test]
fn phi4mini_pinned_geometry_and_longrope_parameters() {
    assert_eq!(
        format!("{:x}", Sha256::digest(PINNED.as_bytes())),
        "431eafdf55dc6d4a76dc3e9f22b0d06eac42d9ac6dc7cbd7ff8ca1e38f18cdb1"
    );
    let c = config();
    assert_eq!(
        (c.hidden, c.intermediate, c.layers, c.vocabulary),
        (3072, 8192, 32, 200064)
    );
    assert_eq!(
        (c.query_heads, c.kv_heads, c.head_dim, c.rotary_dim),
        (24, 8, 128, 96)
    );
    assert_eq!((c.qkv_width(), c.gate_up_width()), (5120, 16384));
    assert_eq!((c.context, c.original_context), (131072, 4096));
    assert_eq!((c.quantization_bits, c.group_size), (4, Some(64)));
    assert!(c.tied_embeddings);
    assert!((f64::from(c.rotary_amplitude()) - (17.0_f64 / 12.0).sqrt()).abs() < 1e-7);
    assert_eq!(c.frequency_denominators(PhiRopeRegime::Short)[0], 1.0);
    assert_ne!(
        c.frequency_denominators(PhiRopeRegime::Short),
        c.frequency_denominators(PhiRopeRegime::Long)
    );
}

#[test]
fn phi4mini_rejects_unsupported_or_inconsistent_architecture() {
    let base: Value = serde_json::from_str(PINNED).unwrap();
    for (pointer, bad) in [
        ("/model_type", json!("llama")),
        ("/architectures", json!(["PhiForCausalLM"])),
        ("/hidden_size", json!(3073)),
        ("/num_key_value_heads", json!(7)),
        ("/head_dim", json!(64)),
        ("/partial_rotary_factor", json!(0.751)),
        ("/partial_rotary_factor", json!(1.5)),
        ("/partial_rotary_factor", json!(0)),
        ("/rope_scaling/type", json!("linear")),
        ("/rope_scaling/long_factor", json!([1.0])),
        ("/rope_scaling/short_factor/0", json!(0)),
        ("/rope_scaling/long_factor/0", json!(1e300)),
        ("/rms_norm_eps", json!(-1)),
        ("/rope_theta", json!(0)),
        ("/attention_bias", json!(true)),
        ("/tie_word_embeddings", json!(null)),
        ("/num_hidden_layers", json!(5000)),
        ("/original_max_position_embeddings", json!(1)),
        ("/sliding_window", json!(4096)),
        ("/quantization/bits", json!(3)),
        ("/quantization/group_size", json!(32)),
        ("/full_attn_mod", json!(2)),
        ("/interpolate_factor", json!(2)),
    ] {
        let mut v = base.clone();
        if pointer == "/head_dim" {
            v["head_dim"] = bad;
        } else {
            *v.pointer_mut(pointer).unwrap() = bad;
        }
        assert!(PhiConfig::from_json(&v).is_err(), "accepted {pointer}: {v}");
    }
    let mut c = config();
    c.long_factors[0] = f32::NAN;
    assert!(c.validate().is_err());
    let mut c = config();
    c.head_dim = usize::MAX;
    assert!(c.validate().is_err());
    let mut v = base;
    v["rope_scaling"]["attention_factor"] = json!(2);
    assert!(PhiConfig::from_json(&v).is_err());
}

#[test]
fn phi4mini_longrope_boundary_math_preserves_partial_dimensions() {
    let c = config();
    let mut kinds = vec![DeviceType::Cpu];
    if cfg!(target_os = "macos") {
        kinds.push(DeviceType::Gpu);
    }
    for kind in kinds {
        let device = Device::new(kind, 0).unwrap();
        assert!(device.is_available().unwrap());
        let stream = Stream::new(&device).unwrap();
        let values: Vec<f32> = (0..c.kv_heads * 3 * c.head_dim)
            .map(|i| ((i * 17 % 97) as f32 - 48.0) / 64.0)
            .collect();
        let input =
            Array::from_f32_slice(&values, &[1, c.kv_heads as i32, 3, c.head_dim as i32]).unwrap();
        for regime in [PhiRopeRegime::Short, PhiRopeRegime::Long] {
            let output = partial_longrope(&input, &c, regime, 4095, &stream).unwrap();
            let actual = ops::to_vec_f32(&output, &stream).unwrap();
            // Independent f64 scalar trigonometry, no production frequency helper.
            let factors = match regime {
                PhiRopeRegime::Short => &c.short_factors,
                PhiRopeRegime::Long => &c.long_factors,
            };
            let amplitude = (1.0 + (131072.0_f64 / 4096.0).ln() / 4096.0_f64.ln()).sqrt();
            for head in 0..c.kv_heads {
                for t in 0..3 {
                    let start = (head * 3 + t) * c.head_dim;
                    for pair in 0..48 {
                        let angle = (4095 + t) as f64
                            / (f64::from(factors[pair]) * 10000.0_f64.powf(pair as f64 / 48.0));
                        let (sin, cos) = angle.sin_cos();
                        let a = f64::from(values[start + pair]);
                        let b = f64::from(values[start + pair + 48]);
                        for (index, expected) in [
                            (pair, amplitude * (a * cos - b * sin)),
                            (pair + 48, amplitude * (a * sin + b * cos)),
                        ] {
                            assert!(
                                (f64::from(actual[start + index]) - expected).abs() < 0.002,
                                "{kind:?} {regime:?} head={head} position={} dim={index}",
                                4095 + t
                            );
                        }
                    }
                    assert_eq!(
                        &actual[start + 96..start + 128],
                        &values[start + 96..start + 128],
                        "unrotated tail changed"
                    );
                }
            }
            // Chunking across the original-position boundary cannot change regime.
            for t in 0..3 {
                let chunk = ops::slice(
                    &input,
                    &[0, 0, t, 0],
                    &[1, c.kv_heads, t + 1, c.head_dim],
                    &stream,
                )
                .unwrap();
                let chunk = partial_longrope(&chunk, &c, regime, 4095 + t, &stream).unwrap();
                let chunk = ops::to_vec_f32(&chunk, &stream).unwrap();
                for head in 0..c.kv_heads {
                    assert_eq!(
                        &chunk[head * c.head_dim..(head + 1) * c.head_dim],
                        &actual[(head * 3 + t) * c.head_dim..(head * 3 + t + 1) * c.head_dim]
                    );
                }
            }
        }
        assert!(partial_longrope(&input, &c, PhiRopeRegime::Long, usize::MAX, &stream).is_err());
        assert!(partial_longrope(&input, &c, PhiRopeRegime::Long, c.context - 2, &stream).is_err());
        let wrong = Array::from_f32_slice(&[0.0; 128], &[1, 1, 1, 128]).unwrap();
        assert!(partial_longrope(&wrong, &c, PhiRopeRegime::Long, 0, &stream).is_err());
    }
}
