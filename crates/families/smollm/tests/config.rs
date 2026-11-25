use serde_json::{Value, json};
use smollm::config::parse_config;
fn config() -> Value {
    json!({"model_type":"smollm3","architectures":["SmolLM3ForCausalLM"],
        "hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,
        "num_hidden_layers":4,"intermediate_size":64,"vocab_size":256,
        "max_position_embeddings":512,"rope_theta":5000000.0,"rms_norm_eps":0.000001,
        "hidden_act":"silu","pretraining_tp":2})
}
#[test]
fn smollm3_nope_schedule_override() {
    let mut value = config();
    let parsed = parse_config(&value).unwrap();
    assert!(parsed.tie_embeddings);
    assert_eq!(parsed.rope_layers.unwrap(), [true, true, true, false]);
    value["no_rope_layer_interval"] = json!(2);
    assert_eq!(
        parse_config(&value).unwrap().rope_layers.unwrap(),
        [true, false, true, false]
    );
    value["no_rope_layers"] = json!([0, 1, 0, 1]);
    assert_eq!(
        parse_config(&value).unwrap().rope_layers.unwrap(),
        [false, true, false, true]
    );
    value["no_rope_layer_interval"] = json!(0);
    assert!(
        parse_config(&value).is_ok(),
        "explicit schedule overrides unused interval"
    );
}
#[test]
fn smollm3_invalid_config_rejected() {
    for (key, bad) in [
        ("architectures", json!(["LlamaForCausalLM"])),
        ("no_rope_layers", json!([0, 1])),
        ("no_rope_layers", json!([0, 1, 2, 1])),
        ("no_rope_layers", json!([true, false, true, false])),
        ("no_rope_layer_interval", json!(0)),
        ("no_rope_layer_interval", json!(-1)),
        (
            "layer_types",
            json!([
                "full_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ]),
        ),
        ("attention_bias", json!(true)),
        ("tie_word_embeddings", json!(null)),
        ("rope_scaling", json!({"rope_type":"llama3"})),
        ("num_hidden_layers", json!(4097)),
        ("pretraining_tp", json!(3)),
    ] {
        let mut value = config();
        value[key] = bad;
        assert!(parse_config(&value).is_err(), "{key}: {value}");
    }
}
