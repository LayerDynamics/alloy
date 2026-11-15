use inference_infra::decoder::gemma3::Gemma3Config;
use serde_json::{Value, json};
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/gemma3_model/config.json")).unwrap()
}
#[test]
fn gemma3_layer_schedule_config_override() {
    let mut v = fixture();
    let c = Gemma3Config::from_json(&v).unwrap();
    assert_eq!(c.sliding, [true, true, false]);
    assert_ne!(c.q_width(), c.hidden);
    assert_eq!(c.embedding_multiplier(), 13.875);
    v["layer_types"] = json!(["full_attention", "sliding_attention", "full_attention"]);
    assert_eq!(
        Gemma3Config::from_json(&v).unwrap().sliding,
        [false, true, false]
    );
    v["layer_types"] = json!(["full_attention"]);
    assert!(Gemma3Config::from_json(&v).is_err());
    let wrapped = json!({"model_type":"gemma3","text_config":{"model_type":"gemma3_text","hidden_size":3840,"num_attention_heads":16,"num_key_value_heads":8,"num_hidden_layers":48,"intermediate_size":15360,"vocab_size":262208,"sliding_window":1024,"rope_scaling":{"rope_type":"linear","factor":8.0}},"quantization":{"bits":4,"group_size":64}});
    let c = Gemma3Config::from_json(&wrapped).unwrap();
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.q_width(), 4096);
    assert!(!c.tied_embeddings);
    assert_eq!(c.sliding.iter().filter(|&&local| !local).count(), 8);
    assert_eq!(
        c.cache_payload(32768).unwrap(),
        (8 * 32768 + 40 * 1024) * 8 * 256 * 8
    );
}
#[test]
fn gemma3_rejects_unsupported_equations() {
    for (key, value) in [
        ("attention_bias", json!(true)),
        ("hidden_activation", json!("silu")),
        ("head_dim", json!(31)),
        ("sliding_window", json!(0)),
        ("query_pre_attn_scalar", json!(-1)),
        ("sliding_window_pattern", json!(0)),
        ("rope_scaling", json!({"rope_type":"dynamic","factor":8})),
        ("final_logit_softcapping", json!(30.0)),
    ] {
        let mut v = fixture();
        v[key] = value;
        assert!(Gemma3Config::from_json(&v).is_err(), "{key}");
    }
    let mut v = fixture();
    v["quantization"]["bits"] = json!(8);
    assert_eq!(Gemma3Config::from_json(&v).unwrap().quantization_bits, 8);
    v["quantization_config"] = json!({"bits":4,"group_size":64});
    assert!(Gemma3Config::from_json(&v).is_err());
}
