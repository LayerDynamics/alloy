use inference_infra::decoder::{
    loader::{CheckpointVerification, LoadOptions},
    mlx_model::MlxDecoderError,
    qwen35::{
        config::Qwen35Config,
        model::{ForwardOptions, MlxQwen35Model},
        weights::load_verified,
    },
};
use mlx_ffi::{Device, DeviceType, Stream};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{cell::Cell, collections::BTreeMap, path::PathBuf};
const TOKENS: &[u32] = &[3, 41, 7, 200, 128, 95, 12, 250];

#[test]
fn qwen35_admission_matches_execution() {
    for stream in streams() {
        let model = model(&stream);
        let c = model.config();
        let weight_plan = c.weight_payload_plan().unwrap();
        assert_eq!(weight_plan.payload_bytes, model.weight_payload_bytes());
        assert_eq!(weight_plan.largest_upload_bytes, (512 * 64 * 4) as u64);
        let pins = pins();
        let (mut mismatch, weights) = load_verified(
            &root(),
            LoadOptions {
                max_file_bytes: 8 << 20,
                max_weight_bytes: 8 << 20,
            },
            &CheckpointVerification {
                sha256: &pins,
                cancelled: &|| false,
            },
        )
        .unwrap();
        mismatch.group_size = Some(64);
        assert!(MlxQwen35Model::from_weights(mismatch, weights, 8 << 20, &stream).is_err());
        let cache_plan = c.cache_payload_plan(TOKENS.len()).unwrap();
        let linear = c.linear_layers.iter().filter(|&&v| v).count() as u64;
        let full = c.linear_layers.len() as u64 - linear;
        let history = (c.linear.convolution_kernel - 1)
            * (2 * c.linear.key_heads * c.linear.key_dim
                + c.linear.value_heads * c.linear.value_dim);
        let recurrent = c.linear.value_heads * c.linear.key_dim * c.linear.value_dim;
        assert_eq!(
            cache_plan.recurrent_overlap_bytes,
            linear * (history + recurrent) as u64 * 8
        );
        assert_eq!(
            cache_plan.full_overlap_bytes,
            full * c.kv_heads as u64 * c.head_dim as u64 * TOKENS.len() as u64 * 16
        );
        assert_eq!(
            cache_plan.total_overlap_bytes,
            cache_plan.full_overlap_bytes + cache_plan.recurrent_overlap_bytes
        );
        assert!(
            model
                .new_cache(TOKENS.len(), cache_plan.total_overlap_bytes - 1, &stream)
                .is_err()
        );
        for (tokens, end) in [(0, 1), (2, 1), (1, c.context + 1), (usize::MAX, usize::MAX)] {
            assert!(c.step_payload_plan(tokens, end, true).is_err());
        }
        assert!(c.cache_payload_plan(0).is_err());
        assert!(c.cache_payload_plan(c.context + 1).is_err());
        let step = c
            .step_payload_plan(TOKENS.len(), TOKENS.len(), true)
            .unwrap();
        assert_eq!(step.output_bytes, (TOKENS.len() * c.vocabulary * 4) as u64);
        let exact = ForwardOptions {
            all_logits: true,
            output_limit: step.output_bytes,
            work_limit: step.work_bytes,
        };
        let mut cache = model
            .new_cache(
                TOKENS.len(),
                cache_plan.total_overlap_bytes + (1 << 20),
                &stream,
            )
            .unwrap();
        for options in [
            ForwardOptions {
                output_limit: step.output_bytes - 1,
                ..exact
            },
            ForwardOptions {
                work_limit: step.work_bytes - 1,
                ..exact
            },
        ] {
            assert!(matches!(
                model.forward(TOKENS, &mut cache, options, &stream, &|| false),
                Err(MlxDecoderError::PayloadLimit { .. })
            ));
            assert!(cache.is_empty());
            assert!(model.is_healthy());
        }
        let actual = model
            .forward(TOKENS, &mut cache, exact, &stream, &|| false)
            .unwrap()
            .into_values();
        assert_eq!(actual.len(), TOKENS.len() * c.vocabulary);
        assert_eq!(cache.len(), TOKENS.len());
    }
}
const OPTIONS: ForwardOptions = ForwardOptions {
    all_logits: true,
    output_limit: 1 << 20,
    work_limit: 1 << 24,
};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen35_model")
        .canonicalize()
        .unwrap()
}
fn pins() -> BTreeMap<PathBuf, [u8; 32]> {
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/qwen35_model/manifest.json")).unwrap();
    manifest
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, hex)| {
            let text = hex.as_str().unwrap();
            let mut digest = [0; 32];
            for (i, v) in digest.iter_mut().enumerate() {
                *v = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap();
            }
            let path = root().join(name);
            assert_eq!(
                Sha256::digest(std::fs::read(&path).unwrap()).as_slice(),
                digest
            );
            (path, digest)
        })
        .collect()
}
fn streams() -> Vec<Stream> {
    let mut kinds = Vec::new();
    if cfg!(target_os = "macos") {
        kinds.push(DeviceType::Gpu);
    }
    kinds.push(DeviceType::Cpu);
    kinds
        .into_iter()
        .map(|k| {
            let d = Device::new(k, 0).unwrap();
            assert!(d.is_available().unwrap());
            Stream::new(&d).unwrap()
        })
        .collect()
}
fn model(stream: &Stream) -> MlxQwen35Model {
    let pins = pins();
    let (c, w) = load_verified(
        &root(),
        LoadOptions {
            max_file_bytes: 8 << 20,
            max_weight_bytes: 8 << 20,
        },
        &CheckpointVerification {
            sha256: &pins,
            cancelled: &|| false,
        },
    )
    .unwrap();
    MlxQwen35Model::from_weights(c, w, 8 << 20, stream).unwrap()
}
fn reference() -> Value {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    let bytes = {
        let bytes = include_bytes!("fixtures/qwen35_model/reference-linux-aarch64.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "ceca68a0fa7952ffe169206a4e7ff84164adea992c7504fc84139969fc3c2461"
        );
        bytes.as_slice()
    };
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    let bytes = include_bytes!("fixtures/qwen35_model/reference.json").as_slice();
    serde_json::from_slice(bytes).unwrap()
}
fn device_reference<'a>(reference: &'a Value, stream: &Stream) -> &'a Value {
    if stream.device().unwrap().kind().unwrap() == DeviceType::Cpu {
        &reference["cpu"]
    } else {
        reference
    }
}
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn fr8(actual: &[f32], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let mut abs = 0f64;
    let mut relative = 0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(a.is_finite() && e.is_finite());
        let d = (f64::from(a) - e).abs();
        abs = abs.max(d);
        let ratio = if d == 0.0 { 0.0 } else { d / e.abs() };
        if ratio > relative {
            relative = ratio;
            if ratio >= 0.01 {
                eprintln!("worst relative native={a:.12e} reference={e:.12e} abs={d:.12e}");
            }
        }
    }
    eprintln!("{label}: max_abs={abs}, max_relative={relative}");
    assert!(
        abs < 0.02 && relative < 0.01,
        "{label} failed unchanged strict FR8"
    );
}
#[test]
fn qwen35_reference_logits() {
    let reference = reference();
    assert_eq!(reference["compile"], false);
    assert_eq!(reference["mlx"], "0.32.4");
    assert_eq!(reference["mlx_lm"], "0.31.2");
    for stream in streams() {
        eprintln!(
            "native device {:?}",
            stream.device().unwrap().kind().unwrap()
        );
        let model = model(&stream);
        let mut cache = model.new_cache(16, 4 << 20, &stream).unwrap();
        let logits = model
            .forward(TOKENS, &mut cache, OPTIONS, &stream, &|| false)
            .unwrap();
        assert_eq!(
            (logits.rows(), logits.vocabulary_size(), cache.len()),
            (8, 256, 8)
        );
        fr8(
            logits.values(),
            &numbers(&device_reference(&reference, &stream)["full"]),
            "mixed full logits",
        );
        assert!(model.weight_buffer_bytes().unwrap() >= model.weight_payload_bytes());
    }
}
#[test]
fn qwen35_partial_rope_cached_continuation() {
    let reference = reference();
    for stream in streams() {
        let reference = device_reference(&reference, &stream);
        eprintln!(
            "native device {:?}",
            stream.device().unwrap().kind().unwrap()
        );
        let model = model(&stream);
        let mut cache = model.new_cache(16, 4 << 20, &stream).unwrap();
        let first = model
            .forward(&TOKENS[..4], &mut cache, OPTIONS, &stream, &|| false)
            .unwrap();
        fr8(
            first.values(),
            &numbers(&reference["prefill"]),
            "upstream cached prefill",
        );
        for (i, &token) in TOKENS[4..].iter().enumerate() {
            let out = model
                .forward(&[token], &mut cache, OPTIONS, &stream, &|| false)
                .unwrap();
            fr8(
                out.values(),
                &numbers(&reference["continuation"][i]),
                "upstream decode",
            );
        }
        for split in 1..TOKENS.len() {
            let mut cache = model.new_cache(16, 4 << 20, &stream).unwrap();
            model
                .forward(&TOKENS[..split], &mut cache, OPTIONS, &stream, &|| false)
                .unwrap();
            let tail = model
                .forward(&TOKENS[split..], &mut cache, OPTIONS, &stream, &|| false)
                .unwrap();
            fr8(
                tail.values(),
                &numbers(&reference["splits"][split - 1]),
                "upstream matching split",
            );
            // #51 cache equivalence is distinct from FR8 against the same
            // upstream execution shape. Preserve its established bound.
            for (&actual, expected) in tail
                .values()
                .iter()
                .zip(numbers(&reference["full"])[split * 256..].iter())
            {
                assert!((f64::from(actual) - expected).abs() <= 0.02 + 0.01 * expected.abs());
            }
        }
    }
}
#[test]
fn qwen35_model_state_transaction_and_reset() {
    let references = reference();
    for stream in streams() {
        let reference = device_reference(&references, &stream);
        let model = model(&stream);
        let other = crate::model(&stream);
        let mut cache = model.new_cache(16, 4 << 20, &stream).unwrap();
        model
            .forward(&TOKENS[..3], &mut cache, OPTIONS, &stream, &|| false)
            .unwrap();
        let bytes = cache.buffer_bytes().unwrap();
        for cancel_at in [0, 2, 5, 8] {
            let count = Cell::new(0);
            let cancelled = || {
                let n = count.get();
                count.set(n + 1);
                n >= cancel_at
            };
            assert!(matches!(
                model.forward(&TOKENS[3..], &mut cache, OPTIONS, &stream, &cancelled),
                Err(MlxDecoderError::Cancelled)
            ));
            assert_eq!(cache.len(), 3);
            assert_eq!(cache.buffer_bytes().unwrap(), bytes);
        }
        assert!(
            other
                .forward(&TOKENS[3..], &mut cache, OPTIONS, &stream, &|| false)
                .is_err()
        );
        assert!(cache.reset(&other).is_err());
        assert!(
            model
                .forward(&[256], &mut cache, OPTIONS, &stream, &|| false)
                .is_err()
        );
        assert!(
            model
                .forward(
                    &TOKENS[3..],
                    &mut cache,
                    ForwardOptions {
                        output_limit: 0,
                        ..OPTIONS
                    },
                    &stream,
                    &|| false
                )
                .is_err()
        );
        assert!(
            model
                .forward(
                    &TOKENS[3..],
                    &mut cache,
                    ForwardOptions {
                        work_limit: 0,
                        ..OPTIONS
                    },
                    &stream,
                    &|| false
                )
                .is_err()
        );
        assert_eq!(cache.len(), 3);
        let out = model
            .forward(&TOKENS[3..], &mut cache, OPTIONS, &stream, &|| false)
            .unwrap();
        fr8(
            out.values(),
            &numbers(&reference["splits"][2]),
            "upstream split after rejected/cancelled steps",
        );
        let mut cold = model.new_cache(16, 4 << 20, &stream).unwrap();
        model
            .forward(&TOKENS[..3], &mut cold, OPTIONS, &stream, &|| false)
            .unwrap();
        let cold_tail = model
            .forward(&TOKENS[3..], &mut cold, OPTIONS, &stream, &|| false)
            .unwrap();
        assert_eq!(
            out.values(),
            cold_tail.values(),
            "cancelled and rejected work must not alter committed state"
        );
        cache.reset(&model).unwrap();
        assert!(cache.is_empty());
        let reset = model
            .forward(
                TOKENS,
                &mut cache,
                ForwardOptions {
                    all_logits: false,
                    ..OPTIONS
                },
                &stream,
                &|| false,
            )
            .unwrap();
        fr8(
            reset.values(),
            &numbers(&reference["full"])[7 * 256..],
            "reset rebuild final logits",
        );
        assert!(model.new_cache(129, 4 << 20, &stream).is_err());
        assert!(model.new_cache(16, 0, &stream).is_err());
    }
}
#[test]
fn qwen35_config_and_tensor_inventory() {
    let original: Value =
        serde_json::from_str(include_str!("fixtures/qwen35_model/config.json")).unwrap();
    let c = Qwen35Config::from_json(&original).unwrap();
    assert_eq!(c.rotary_dim, 4);
    assert_eq!(c.linear_layers, [true, false, true, false]);
    assert!(
        c.tensor_layout()
            .unwrap()
            .tensors
            .contains_key("language_model.model.layers.0.linear_attn.conv1d.weight")
    );
    for (key, value) in [
        ("attention_bias", Value::Bool(true)),
        ("attn_output_gate", Value::Bool(false)),
        ("hidden_act", Value::String("gelu".into())),
        ("num_experts", Value::from(2)),
        ("head_dim", Value::from(15)),
        ("linear_num_value_heads", Value::from(0)),
        ("layer_types", serde_json::json!(["full_attention"])),
    ] {
        let mut bad = original.clone();
        bad["text_config"][key] = value;
        assert!(Qwen35Config::from_json(&bad).is_err(), "{key}");
    }
    let mut overridden = original.clone();
    overridden["text_config"]["layer_types"] = serde_json::json!([
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention"
    ]);
    assert_eq!(
        Qwen35Config::from_json(&overridden).unwrap().linear_layers,
        [false, true, false, true]
    );
    let mut bad = original.clone();
    bad["text_config"]["rope_parameters"]["type"] = Value::String("yarn".into());
    assert!(Qwen35Config::from_json(&bad).is_err());
}
