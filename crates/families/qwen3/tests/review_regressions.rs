use qwen3::{
    config::Qwen3Config,
    fixture::generate_fixture,
    loader::LoadOptions,
    model::{ModelError, Qwen3Model, full_logits_bytes},
};
use std::{collections::BTreeMap, path::PathBuf};

fn model() -> Qwen3Model {
    let c = Qwen3Config::tiny_fixture();
    let w = generate_fixture(&c);
    Qwen3Model::load(c, w).expect("model")
}
fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (&a, &b) in got.iter().zip(want) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 2e-2 + 1e-2 * b.abs(), "{a} != {b}");
    }
}
#[test]
fn cached_decode_matches_full_forward() {
    let model = model();
    let tokens = [3, 41, 7, 200, 128, 95, 12, 250];
    let full = model.forward(&tokens).expect("full");
    let mut cache = model.new_cache(16, 1024 * 1024).expect("cache");
    close(
        &model
            .prefill_cached(&tokens[..3], &mut cache)
            .expect("prefill"),
        &full[2],
    );
    for (index, &token) in tokens.iter().enumerate().skip(3) {
        close(
            &model.decode_token(token, &mut cache).expect("decode"),
            &full[index],
        );
    }
    assert_eq!(cache.len(), tokens.len());
    cache.reset();
    assert!(cache.is_empty());
    close(
        &model
            .decode_token(tokens[0], &mut cache)
            .expect("reset decode"),
        &full[0],
    );
}
#[test]
fn cache_rejects_another_model_and_does_not_advance_on_invalid_token() {
    let first = model();
    let second = model();
    let mut cache = first.new_cache(2, 1024 * 1024).expect("cache");
    assert!(second.decode_token(3, &mut cache).is_err());
    assert!(first.decode_token(u32::MAX, &mut cache).is_err());
    assert_eq!(cache.len(), 0);
    first.decode_token(3, &mut cache).expect("first");
    first.decode_token(4, &mut cache).expect("second");
    assert!(first.decode_token(5, &mut cache).is_err());
    assert_eq!(cache.len(), 2);
}
#[test]
fn cache_and_logits_limits_are_enforced() {
    let model = model();
    assert!(matches!(
        model.new_cache(16, 1),
        Err(ModelError::MemoryLimit { .. })
    ));
    assert!(matches!(
        model.forward_with_output_limit(&[3, 41], 1),
        Err(ModelError::MemoryLimit { .. })
    ));
    assert!(full_logits_bytes(32768, 151936).expect("size") > 1024 * 1024 * 1024);
    assert!(full_logits_bytes(usize::MAX, usize::MAX).is_err());
}
#[test]
fn untied_head_is_required_and_controls_logits() {
    let mut c = Qwen3Config::tiny_fixture();
    c.tie_embeddings = false;
    let mut w = generate_fixture(&c);
    w.lm_head = None;
    assert!(Qwen3Model::load(c.clone(), w).is_err());
    let mut w = generate_fixture(&c);
    w.lm_head = Some(qwen3::fixture::MatrixWeight::f32(vec![
        0.0;
        c.vocab_size
            * c.hidden_size
    ]));
    let model = Qwen3Model::load(c, w).expect("untied model");
    assert!(
        model
            .forward(&[3, 41])
            .expect("forward")
            .iter()
            .flatten()
            .all(|&v| v == 0.0)
    );
}
#[test]
fn greedy_loop_matches_repeated_full_forward() {
    let model = model();
    let mut context = vec![3, 41];
    let mut expected = Vec::new();
    for _ in 0..3 {
        let all = model.forward(&context).expect("full");
        let logits = all.last().expect("last");
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        expected.push(best as u32);
        context.push(best as u32);
    }
    assert_eq!(
        model
            .generate_greedy(&[3, 41], 3, None, 1024 * 1024)
            .expect("generate"),
        expected
    );
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ID: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "alloy-qwen-review-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).expect("scratch");
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn checkpoint(untied: bool) -> (Qwen3Config, Vec<u8>) {
    let mut c = Qwen3Config::tiny_fixture();
    c.tie_embeddings = !untied;
    let w = generate_fixture(&c);
    let mut data: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>, values: &[f32]| {
        data.insert(
            name,
            (shape, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
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
    if let Some(head) = &w.lm_head {
        add(
            "lm_head.weight".into(),
            vec![c.vocab_size, c.hidden_size],
            head.as_f32().expect("fixture f32"),
        );
    }
    let views: Vec<_> = data
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("view"),
            )
        })
        .collect();
    (
        c,
        safetensors::tensor::serialize(views, &None).expect("serialize"),
    )
}
#[test]
fn safetensors_roundtrip_tied_and_untied() {
    for untied in [false, true] {
        let (c, bytes) = checkpoint(untied);
        let scratch = Scratch::new();
        let path = scratch.0.join("model.safetensors");
        std::fs::write(&path, bytes).expect("write");
        let expected = Qwen3Model::load(c.clone(), generate_fixture(&c))
            .expect("memory")
            .forward(&[3, 41])
            .expect("forward");
        let actual = Qwen3Model::load(c, path.as_path())
            .expect("disk")
            .forward(&[3, 41])
            .expect("forward");
        for (got, want) in actual.iter().zip(&expected) {
            close(got, want);
        }
    }
}
#[test]
fn safetensors_corruption_and_limits_return_errors() {
    let (c, bytes) = checkpoint(false);
    let scratch = Scratch::new();
    let path = scratch.0.join("model.safetensors");
    std::fs::write(&path, &bytes).expect("write");
    assert!(
        Qwen3Model::load_safetensors(
            c.clone(),
            &path,
            LoadOptions {
                max_file_bytes: 1,
                max_weight_bytes: 1
            }
        )
        .is_err()
    );
    assert!(
        Qwen3Model::load_safetensors(
            c.clone(),
            &path,
            LoadOptions {
                max_file_bytes: bytes.len(),
                max_weight_bytes: 1
            }
        )
        .is_err()
    );
    std::fs::write(&path, b"corrupted").expect("write corrupt");
    assert!(Qwen3Model::load(c, path.as_path()).is_err());
}

#[test]
fn safetensors_indexed_shards_load_and_unsafe_paths_fail() {
    let (c, bytes) = checkpoint(false);
    let scratch = Scratch::new();
    // Split the serialized fixture across two actual safetensors files.
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("tensors");
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut map = serde_json::Map::new();
    let mut names: Vec<_> = tensors.names().into_iter().cloned().collect();
    names.sort();
    for (index, name) in names.iter().enumerate() {
        let view = tensors.tensor(name).expect("view");
        let shard = if index % 2 == 0 {
            a.push((name.as_str(), view));
            "one.safetensors"
        } else {
            b.push((name.as_str(), view));
            "two.safetensors"
        };
        map.insert(name.clone(), serde_json::Value::String(shard.into()));
    }
    std::fs::write(
        scratch.0.join("one.safetensors"),
        safetensors::tensor::serialize(a, &None).expect("one"),
    )
    .expect("write one");
    std::fs::write(
        scratch.0.join("two.safetensors"),
        safetensors::tensor::serialize(b, &None).expect("two"),
    )
    .expect("write two");
    let index = scratch.0.join("model.safetensors.index.json");
    std::fs::write(
        &index,
        serde_json::to_vec(&serde_json::json!({"weight_map":map})).expect("index"),
    )
    .expect("write index");
    let loaded = Qwen3Model::load(c.clone(), scratch.0.as_path()).expect("shards");
    let expected = Qwen3Model::load(c.clone(), generate_fixture(&c)).expect("memory");
    close(
        &loaded.forward(&[3]).expect("disk")[0],
        &expected.forward(&[3]).expect("memory")[0],
    );
    std::fs::write(
        &index,
        b"{\"weight_map\":{\"model.embed_tokens.weight\":\"../escape.safetensors\"}}",
    )
    .expect("unsafe index");
    assert!(Qwen3Model::load(c, scratch.0.as_path()).is_err());
}
