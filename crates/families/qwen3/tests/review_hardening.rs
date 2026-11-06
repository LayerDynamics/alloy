use qwen3::{
    config::Qwen3Config,
    fixture::generate_fixture,
    loader::LoadOptions,
    model::{ModelError, Qwen3Model},
    template::{TemplateError, render_chat},
};
use safetensors::Dtype;
use std::{collections::BTreeMap, path::PathBuf};

type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "alloy-qwen-hardening-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("mkdir");
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn config() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 8,
        num_layers: 1,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 16,
        rope_theta: 10000.0,
        max_context: 8,
        tie_embeddings: true,
    }
}
fn tensors(c: &Qwen3Config) -> Tensors {
    let w = generate_fixture(c);
    let mut data = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>, values: &[f32]| {
        data.insert(
            name,
            (
                Dtype::F32,
                shape,
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        );
    };
    add(
        "model.embed_tokens.weight".into(),
        vec![c.vocab_size, c.hidden_size],
        w.embed_tokens.as_f32().expect("fixture f32"),
    );
    for (index, l) in w.layers.iter().enumerate() {
        let p = format!("model.layers.{index}");
        for (name, shape, values) in [
            (
                "input_layernorm",
                vec![c.hidden_size],
                l.input_layernorm.as_slice(),
            ),
            (
                "self_attn.q_proj",
                vec![c.q_dim(), c.hidden_size],
                l.q_proj.as_f32().expect("fixture f32"),
            ),
            (
                "self_attn.k_proj",
                vec![c.kv_dim(), c.hidden_size],
                l.k_proj.as_f32().expect("fixture f32"),
            ),
            (
                "self_attn.v_proj",
                vec![c.kv_dim(), c.hidden_size],
                l.v_proj.as_f32().expect("fixture f32"),
            ),
            ("self_attn.q_norm", vec![c.head_dim], l.q_norm.as_slice()),
            ("self_attn.k_norm", vec![c.head_dim], l.k_norm.as_slice()),
            (
                "self_attn.o_proj",
                vec![c.hidden_size, c.q_dim()],
                l.o_proj.as_f32().expect("fixture f32"),
            ),
            (
                "post_attention_layernorm",
                vec![c.hidden_size],
                l.post_attention_layernorm.as_slice(),
            ),
            (
                "mlp.gate_proj",
                vec![c.intermediate_size, c.hidden_size],
                l.gate_proj.as_f32().expect("fixture f32"),
            ),
            (
                "mlp.up_proj",
                vec![c.intermediate_size, c.hidden_size],
                l.up_proj.as_f32().expect("fixture f32"),
            ),
            (
                "mlp.down_proj",
                vec![c.hidden_size, c.intermediate_size],
                l.down_proj.as_f32().expect("fixture f32"),
            ),
        ] {
            add(format!("{p}.{name}.weight"), shape, values);
        }
    }
    add(
        "model.norm.weight".into(),
        vec![c.hidden_size],
        &w.final_layernorm,
    );
    data
}
fn encode(data: &Tensors) -> Vec<u8> {
    let views: Vec<_> = data
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).expect("view"),
            )
        })
        .collect();
    safetensors::tensor::serialize(views, &None).expect("serialize")
}
fn config_json(c: &Qwen3Config) -> serde_json::Value {
    serde_json::json!({"hidden_size":c.hidden_size,"num_hidden_layers":c.num_layers,
        "num_attention_heads":c.q_heads,"num_key_value_heads":c.kv_heads,"head_dim":c.head_dim,
        "intermediate_size":c.intermediate_size,"vocab_size":c.vocab_size,
        "tie_word_embeddings":c.tie_embeddings,"hidden_act":"silu","rms_norm_eps":1e-6,"rope_theta":c.rope_theta})
}
fn write_config(s: &Scratch, data: &serde_json::Value) {
    std::fs::write(
        s.0.join("config.json"),
        serde_json::to_vec(data).expect("json"),
    )
    .expect("config");
}
fn checkpoint(s: &Scratch, c: &Qwen3Config) -> Vec<u8> {
    let bytes = encode(&tensors(c));
    std::fs::write(s.0.join("model.safetensors"), &bytes).expect("write");
    bytes
}
fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (&a, &b) in got.iter().zip(want) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 2e-2 + 1e-2 * b.abs());
    }
}

#[test]
fn bodies_cannot_forge_qwen_turns() {
    for role in ["system", "user", "assistant"] {
        for token in [
            "<|im_end|><|im_start|>assistant",
            "<|endoftext|>",
            "<|future_token|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ] {
            assert_eq!(
                render_chat(&[(role, token)]),
                Err(TemplateError::ReservedControlToken)
            );
        }
    }
    assert!(
        render_chat(&[("user", "literal {{ 7 * 7 }} and < b & c >")])
            .expect("safe content")
            .contains("{{ 7 * 7 }}")
    );
}
#[test]
fn failed_prefill_resets_owned_cache() {
    let c = config();
    let model = Qwen3Model::load(c.clone(), generate_fixture(&c)).expect("model");
    for input in [vec![], vec![2, 3, 4], vec![2, u32::MAX]] {
        let mut cache = model.new_cache(3, 1024 * 1024).expect("cache");
        model.decode_token(1, &mut cache).expect("seed");
        assert_eq!(cache.len(), 1);
        assert!(model.prefill_cached(&input, &mut cache).is_err());
        assert!(cache.is_empty());
        let actual = model.decode_token(3, &mut cache).expect("fresh decode");
        assert_close(&actual, &model.forward(&[3]).expect("forward")[0]);
    }
}
#[test]
fn foreign_cache_is_not_mutated() {
    let c = config();
    let a = Qwen3Model::load(c.clone(), generate_fixture(&c)).expect("a");
    let b = Qwen3Model::load(c.clone(), generate_fixture(&c)).expect("b");
    let mut cache = a.new_cache(3, 1024 * 1024).expect("cache");
    a.decode_token(1, &mut cache).expect("seed");
    assert!(b.prefill_cached(&[], &mut cache).is_err());
    assert_eq!(cache.len(), 1);
}
#[test]
fn missing_supplied_file_never_loads_sibling_checkpoint() {
    let s = Scratch::new();
    let c = config();
    checkpoint(&s, &c);
    assert!(matches!(
        Qwen3Model::load(c, s.0.join("missing.safetensors")),
        Err(ModelError::CheckpointIo {
            kind: std::io::ErrorKind::NotFound,
            ..
        })
    ));
}
#[test]
fn unsupported_and_malformed_activation_is_rejected() {
    let s = Scratch::new();
    let c = config();
    checkpoint(&s, &c);
    for activation in [
        serde_json::json!("gelu"),
        serde_json::json!(null),
        serde_json::json!(12),
    ] {
        let mut data = config_json(&c);
        data["hidden_act"] = activation;
        write_config(&s, &data);
        assert!(matches!(
            Qwen3Model::load(c.clone(), s.0.as_path()),
            Err(ModelError::CheckpointValidation(_))
        ));
    }
    write_config(&s, &config_json(&c));
    assert!(Qwen3Model::load(c, s.0.as_path()).is_ok());
}
#[test]
fn present_numeric_settings_must_be_numbers() {
    let s = Scratch::new();
    let c = config();
    checkpoint(&s, &c);
    for field in ["rms_norm_eps", "rope_theta"] {
        for value in [
            serde_json::json!("10000"),
            serde_json::json!(null),
            serde_json::json!(false),
            serde_json::json!([]),
        ] {
            let mut data = config_json(&c);
            data[field] = value;
            write_config(&s, &data);
            assert!(matches!(
                Qwen3Model::load(c.clone(), s.0.as_path()),
                Err(ModelError::CheckpointValidation(_))
            ));
        }
    }
}
#[test]
fn shard_inventory_must_equal_weight_map() {
    let s = Scratch::new();
    let c = config();
    let data = tensors(&c);
    checkpoint(&s, &c);
    let map: serde_json::Map<String, serde_json::Value> = data
        .keys()
        .map(|name| (name.clone(), serde_json::json!("model.safetensors")))
        .collect();
    let mut index = serde_json::json!({"weight_map":map});
    std::fs::write(
        s.0.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).expect("json"),
    )
    .expect("index");
    assert!(Qwen3Model::load(c.clone(), s.0.as_path()).is_ok());
    index["weight_map"]
        .as_object_mut()
        .expect("map")
        .remove("model.norm.weight");
    std::fs::write(
        s.0.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).expect("json"),
    )
    .expect("index");
    assert!(matches!(
        Qwen3Model::load(c, s.0.as_path()),
        Err(ModelError::CheckpointValidation(_))
    ));
}
#[test]
fn reduced_defaults_bound_both_representations() {
    let options = LoadOptions::default();
    assert!(options.max_file_bytes <= 16 * 1024 * 1024);
    assert!(options.max_weight_bytes <= 16 * 1024 * 1024);
}
#[test]
fn raw_budget_includes_config_and_index() {
    let c = config();
    let s = Scratch::new();
    let bytes = checkpoint(&s, &c);
    write_config(&s, &config_json(&c));
    let options = LoadOptions {
        max_file_bytes: bytes.len(),
        max_weight_bytes: 1024 * 1024,
    };
    assert!(matches!(
        Qwen3Model::load_safetensors(c.clone(), s.0.as_path(), options),
        Err(ModelError::MemoryLimit { .. })
    ));
    std::fs::remove_file(s.0.join("config.json")).expect("remove config");
    let map: serde_json::Map<String, serde_json::Value> = tensors(&c)
        .keys()
        .map(|name| (name.clone(), serde_json::json!("model.safetensors")))
        .collect();
    std::fs::write(
        s.0.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map":map})).expect("json"),
    )
    .expect("index");
    assert!(matches!(
        Qwen3Model::load_safetensors(c, s.0.as_path(), options),
        Err(ModelError::MemoryLimit { .. })
    ));
}
#[test]
fn borrowed_pathbuf_and_bare_relative_file_load() {
    let s = Scratch::new();
    let c = config();
    checkpoint(&s, &c);
    let path = s.0.join("model.safetensors");
    let loaded = Qwen3Model::load(c, &path).expect("borrowed PathBuf");
    assert!(loaded.forward(&[1]).is_ok());
    let status = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "bare_relative_file_child", "--nocapture"])
        .current_dir(&s.0)
        .env("ALLOY_BARE_CHECKPOINT_CHILD", "1")
        .status()
        .expect("child");
    assert!(status.success());
}
#[test]
fn bare_relative_file_child() {
    if std::env::var("ALLOY_BARE_CHECKPOINT_CHILD").as_deref() != Ok("1") {
        return;
    }
    Qwen3Model::load(config(), "model.safetensors").expect("bare filename must load");
}
#[test]
fn standalone_quantized_group_is_inferred() {
    let s = Scratch::new();
    let c = config();
    let mut data = tensors(&c);
    let rows = c.vocab_size;
    let width = c.hidden_size;
    assert_eq!(width, 8);
    data.insert(
        "model.embed_tokens.weight".into(),
        (
            Dtype::U32,
            vec![rows, 1],
            (0..rows)
                .flat_map(|_| 0x76543210u32.to_le_bytes())
                .collect(),
        ),
    );
    data.insert(
        "model.embed_tokens.scales".into(),
        (
            Dtype::F32,
            vec![rows, 2],
            (0..rows * 2).flat_map(|_| 0.5f32.to_le_bytes()).collect(),
        ),
    );
    data.insert(
        "model.embed_tokens.biases".into(),
        (
            Dtype::F32,
            vec![rows, 2],
            (0..rows * 2).flat_map(|_| (-1f32).to_le_bytes()).collect(),
        ),
    );
    let quantized = encode(&data);
    std::fs::write(s.0.join("model.safetensors"), quantized).expect("quantized");
    // Affine4 stays packed for MLX; CPU Qwen3Model rejects it. Verify the
    // loader keeps resident Affine4 with the inferred group size.
    let loaded =
        qwen3::loader::load_weights(&c, s.0.as_path(), qwen3::loader::LoadOptions::default())
            .expect("infer group size four");
    match &loaded.embed_tokens {
        qwen3::fixture::MatrixWeight::Affine4 {
            rows: r,
            cols,
            group_size,
            packed,
            scales,
            biases,
        } => {
            assert_eq!((*r, *cols, *group_size), (rows, width, 4));
            assert_eq!(packed.len(), rows);
            assert_eq!(scales.len(), rows * 2);
            assert_eq!(biases.len(), rows * 2);
        }
        qwen3::fixture::MatrixWeight::F32(_) => panic!("expected Affine4 embed"),
    }
    assert!(
        Qwen3Model::from_weights(c.clone(), loaded).is_err(),
        "CPU reference path must reject Affine4"
    );
}

#[test]
fn verified_streaming_load_handles_cross_shard_triplets_with_exact_budgets() {
    use qwen3::loader::{CheckpointVerification, load_weights_verified};
    use sha2::{Digest, Sha256};

    let scratch = Scratch::new();
    let c = config();
    let mut data = tensors(&c);
    let original_payload: usize = data.values().map(|(_, _, bytes)| bytes.len()).sum();
    let dense_embed_bytes = c.vocab_size * c.hidden_size * 4;
    data.insert(
        "model.embed_tokens.weight".into(),
        (
            Dtype::U32,
            vec![c.vocab_size, 1],
            (0..c.vocab_size)
                .flat_map(|_| 0x7654_3210u32.to_le_bytes())
                .collect(),
        ),
    );
    data.insert(
        "model.embed_tokens.scales".into(),
        (
            Dtype::F16,
            vec![c.vocab_size, 2],
            (0..c.vocab_size * 2)
                .flat_map(|_| half::f16::from_f32(0.5).to_bits().to_le_bytes())
                .collect(),
        ),
    );
    data.insert(
        "model.embed_tokens.biases".into(),
        (
            Dtype::BF16,
            vec![c.vocab_size, 2],
            (0..c.vocab_size * 2)
                .flat_map(|_| half::bf16::from_f32(-1.0).to_bits().to_le_bytes())
                .collect(),
        ),
    );
    data.insert(
        "model.norm.weight".into(),
        (
            Dtype::BF16,
            vec![c.hidden_size],
            (0..c.hidden_size)
                .flat_map(|_| half::bf16::from_f32(1.25).to_bits().to_le_bytes())
                .collect(),
        ),
    );
    // A dense projection does not consume optional quantization
    // parameters. Their bytes still belong to the strict index and pin.
    data.insert(
        "model.layers.0.self_attn.o_proj.scales".into(),
        (Dtype::F32, vec![1], f32::NAN.to_le_bytes().to_vec()),
    );
    let mut shards = [Tensors::new(), Tensors::new(), Tensors::new()];
    let filenames = ["one.safetensors", "two.safetensors", "three.safetensors"];
    let mut weight_map = serde_json::Map::new();
    for (index, (name, tensor)) in data.into_iter().enumerate() {
        let shard = match name.as_str() {
            "model.embed_tokens.weight" => 0,
            "model.embed_tokens.scales" => 1,
            "model.embed_tokens.biases" => 2,
            _ => index % 3,
        };
        weight_map.insert(name.clone(), serde_json::json!(filenames[shard]));
        shards[shard].insert(name, tensor);
    }
    for (filename, shard) in filenames.iter().zip(&shards) {
        std::fs::write(scratch.0.join(filename), encode(shard)).expect("write shard");
    }
    write_config(&scratch, &config_json(&c));
    std::fs::write(
        scratch.0.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).expect("index"),
    )
    .expect("write index");
    let mut source_bytes = 0;
    let mut pins = BTreeMap::new();
    for entry in std::fs::read_dir(&scratch.0).expect("components") {
        let path = entry
            .expect("component")
            .path()
            .canonicalize()
            .expect("canonical path");
        let bytes = std::fs::read(&path).expect("pin input");
        source_bytes += bytes.len();
        pins.insert(path, Sha256::digest(bytes).into());
    }
    let payload_bytes = original_payload - dense_embed_bytes + c.vocab_size * (4 + 2 * 4 + 2 * 4);
    let options = LoadOptions {
        max_file_bytes: source_bytes,
        max_weight_bytes: payload_bytes,
    };
    let cancelled = || false;
    let verification = CheckpointVerification {
        sha256: &pins,
        cancelled: &cancelled,
    };
    let loaded = load_weights_verified(&c, &scratch.0, options, &verification)
        .expect("cross-shard mixed types fit exactly charged source and payload limits");
    match loaded.embed_tokens {
        qwen3::fixture::MatrixWeight::Affine4 {
            packed,
            scales,
            biases,
            rows,
            cols,
            group_size,
        } => {
            assert_eq!(packed, vec![0x7654_3210; c.vocab_size]);
            assert_eq!(scales, vec![0.5; c.vocab_size * 2]);
            assert_eq!(biases, vec![-1.0; c.vocab_size * 2]);
            assert_eq!((rows, cols, group_size), (c.vocab_size, c.hidden_size, 4));
        }
        _ => panic!("the loader must keep the embedding packed"),
    }
    assert_eq!(loaded.final_layernorm, vec![1.25; c.hidden_size]);
    assert!(loaded.layers[0].o_proj.as_f32().is_some());
    assert!(matches!(
        load_weights_verified(
            &c,
            &scratch.0,
            LoadOptions {
                max_file_bytes: source_bytes - 1,
                ..options
            },
            &verification
        ),
        Err(ModelError::MemoryLimit { .. })
    ));
    assert!(matches!(
        load_weights_verified(
            &c,
            &scratch.0,
            LoadOptions {
                max_weight_bytes: payload_bytes - 1,
                ..options
            },
            &verification
        ),
        Err(ModelError::MemoryLimit { .. })
    ));

    for path in pins.keys() {
        let mut wrong = pins.clone();
        wrong.insert(path.clone(), [0; 32]);
        let verification = CheckpointVerification {
            sha256: &wrong,
            cancelled: &cancelled,
        };
        assert!(
            matches!(load_weights_verified(&c, &scratch.0, options, &verification),
            Err(ModelError::CheckpointValidation(message)) if message.contains("SHA-256 mismatch"))
        );
        let mut missing = pins.clone();
        missing.remove(path);
        let verification = CheckpointVerification {
            sha256: &missing,
            cancelled: &cancelled,
        };
        assert!(
            matches!(load_weights_verified(&c, &scratch.0, options, &verification),
            Err(ModelError::CheckpointValidation(message)) if message.contains("no trusted digest"))
        );
    }
}

#[test]
fn tied_memory_and_disk_heads_are_rejected() {
    let c = config();
    let mut w = generate_fixture(&c);
    w.lm_head = Some(qwen3::fixture::MatrixWeight::f32(vec![
        0.0;
        c.vocab_size
            * c.hidden_size
    ]));
    assert!(Qwen3Model::load(c.clone(), w).is_err());
    let s = Scratch::new();
    let mut data = tensors(&c);
    let (_, shape, bytes) = data
        .get("model.embed_tokens.weight")
        .expect("embed")
        .clone();
    data.insert("lm_head.weight".into(), (Dtype::F32, shape, bytes));
    std::fs::write(s.0.join("model.safetensors"), encode(&data)).expect("write");
    assert!(matches!(
        Qwen3Model::load(c, s.0.as_path()),
        Err(ModelError::CheckpointValidation(_))
    ));
}
#[test]
fn checkpoint_failures_have_distinct_typed_categories() {
    let s = Scratch::new();
    let c = config();
    assert!(matches!(
        Qwen3Model::load(c.clone(), s.0.join("absent")),
        Err(ModelError::CheckpointIo {
            kind: std::io::ErrorKind::NotFound,
            ..
        })
    ));
    std::fs::write(s.0.join("model.safetensors"), b"corrupt").expect("bad file");
    assert!(matches!(
        Qwen3Model::load(c.clone(), s.0.as_path()),
        Err(ModelError::CheckpointFormat {
            format: "safetensors",
            ..
        })
    ));
    checkpoint(&s, &c);
    std::fs::write(s.0.join("config.json"), b"{").expect("bad json");
    assert!(matches!(
        Qwen3Model::load(c.clone(), s.0.as_path()),
        Err(ModelError::CheckpointFormat { format: "json", .. })
    ));
    let mut data = config_json(&c);
    data["hidden_act"] = serde_json::json!("gelu");
    write_config(&s, &data);
    assert!(matches!(
        Qwen3Model::load(c, s.0.as_path()),
        Err(ModelError::CheckpointValidation(_))
    ));
}
