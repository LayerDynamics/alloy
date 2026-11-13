use inference_infra::decoder::{
    config::{DecoderConfig, DecoderFamily, Llama31Scaling},
    fixture::generate_fixture,
    mlx_model::MlxDecoderModel,
};
use mlx_ffi::{Device, DeviceType, Stream};
fn config() -> DecoderConfig {
    DecoderConfig {
        family: DecoderFamily::Llama3,
        rms_eps: 1e-5,
        rope_theta: 500000.0,
        rope_scaling: Some(Llama31Scaling {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        }),
        ..DecoderConfig::tiny_fixture()
    }
}
#[test]
fn llama31_rope_frequency_transitions() {
    let s = config().rope_scaling.unwrap();
    for (wavelength, expected_multiplier) in [
        (1024.0, 1.0),
        (2048.0, 1.0),
        (8192.0, 0.125),
        (16384.0, 0.125),
    ] {
        let f = std::f64::consts::TAU / wavelength;
        assert!((s.scale_inverse_frequency(f) / f - expected_multiplier).abs() < 1e-12);
    }
    let f = std::f64::consts::TAU / 4096.0;
    assert!((s.scale_inverse_frequency(f) / f - (2.0 / 3.0 / 8.0 + 1.0 / 3.0)).abs() < 1e-12);
}
#[test]
fn llama3_reference_logits() {
    let c = config();
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    let model =
        MlxDecoderModel::from_weights(c.clone(), generate_fixture(&c), 8 * 1024 * 1024, &stream)
            .unwrap();
    let tokens = [3, 41, 7, 200, 128, 95, 12, 250];
    let actual = model
        .forward(&tokens, 1024 * 1024, 8 * 256 * 16, &stream)
        .unwrap();
    let expected: Vec<f32> = include_str!("../fixture/reference_logits.txt")
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    assert_eq!(actual.values().len(), expected.len());
    let mut absolute = 0f32;
    let mut relative = 0f32;
    for (&a, &b) in actual.values().iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite());
        absolute = absolute.max((a - b).abs());
        relative = relative.max(if a == b { 0.0 } else { (a - b).abs() / b.abs() });
    }
    eprintln!("llama3 tiny max_abs={absolute} max_relative={relative}");
    assert!(absolute < 0.02 && relative < 0.01);
}
#[test]
fn llama3_rope_positions() {
    let c = config();
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    let model =
        MlxDecoderModel::from_weights(c.clone(), generate_fixture(&c), 8 * 1024 * 1024, &stream)
            .unwrap();
    let tokens = [3, 41, 7, 200, 128, 95, 12, 250];
    let full = model.forward(&tokens, 1024 * 1024, 32768, &stream).unwrap();
    let mut cache = model.new_cache(32, 1024 * 1024, &stream).unwrap();
    model
        .prefill(&tokens[..3], &mut cache, 32768, &stream)
        .unwrap();
    model
        .prefill(&tokens[3..7], &mut cache, 32768, &stream)
        .unwrap();
    let cached = model.decode(tokens[7], &mut cache, 32768, &stream).unwrap();
    for (&a, &b) in cached.values().iter().zip(full.last_row()) {
        assert!((a - b).abs() < 0.002);
    }
    assert_eq!(cache.len(), 8);
}
#[test]
fn llama31_instruct_stop_tokens() {
    use llama3::template::*;
    assert_eq!(END_OF_TEXT, 128001);
    assert_eq!(END_OF_TURN, 128009);
    assert_eq!(TOOL_HANDOFF, 128008);
    assert_ne!(END_OF_TURN, TOOL_HANDOFF);
    assert!(CHAT_TEMPLATE.contains("<|eot_id|>"));
    assert!(TOOL_CHAT_TEMPLATE.contains("</tool_call><|eom_id|>"));
}
#[test]
fn unsupported_architecture_and_rope_fail_closed() {
    let base = serde_json::json!({"model_type":"llama","architectures":["LlamaForCausalLM"],"hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"intermediate_size":64,"vocab_size":256,"max_position_embeddings":512,"rope_theta":500000.0,"rms_norm_eps":0.00001,"tie_word_embeddings":true,"hidden_act":"silu"});
    assert!(llama3::config::parse_config(&base).is_ok());
    for (key, bad) in [
        ("architectures", serde_json::json!(["Qwen3ForCausalLM"])),
        ("rope_scaling", serde_json::json!({"rope_type":"yarn"})),
        ("attention_bias", serde_json::json!(true)),
        ("num_key_value_heads", serde_json::json!(3)),
    ] {
        let mut v = base.clone();
        v[key] = bad;
        assert!(llama3::config::parse_config(&v).is_err(), "{key}");
    }
}
