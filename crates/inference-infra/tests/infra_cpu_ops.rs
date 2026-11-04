use inference_infra::capabilities::{Backend, BackendCapabilities};
use inference_infra::ops::{apply_rope, attention, greedy_token, sample_token, KvCache};
use inference_infra::tensor::Tensor;

const TOL: f32 = 1e-4;

/// This test is Linux-lane only: the CPU backend is the contract for W1.
#[test]
fn capabilities_report_cpu_backend() {
    let caps = BackendCapabilities::detect();
    assert_eq!(caps.backend, Backend::Cpu, "W1 targets the CPU backend");
    assert!(caps.supports_rope, "rope must be reported on CPU");
    assert!(caps.supports_attention, "attention must be reported on CPU");
    assert!(caps.supports_kv_cache, "kv cache must be reported on CPU");
    assert!(caps.supports_sampling, "sampling must be reported on CPU");
    assert!(caps.max_context_len >= 4096, "must cover the qwen3-4b-4b profile context");
}

#[test]
fn rope_rotates_by_expected_angle() {
    // head_dim = 2, one head, one token, theta_base = 10000.
    // For the single dim pair, theta = 1.0 rad at position 1.
    // (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    let x = Tensor::new(&[1, 1, 2], vec![1.0, 0.0]);
    let out = apply_rope(&x, 1, 10_000.0);
    assert_eq!(out.shape, vec![1, 1, 2]);
    assert!((out.data[0] - 1.0_f32.cos()).abs() < TOL, "got {}", out.data[0]);
    assert!((out.data[1] - 1.0_f32.sin()).abs() < TOL, "got {}", out.data[1]);

    // Second basis vector rotates the other way.
    let x = Tensor::new(&[1, 1, 2], vec![0.0, 1.0]);
    let out = apply_rope(&x, 1, 10_000.0);
    assert!((out.data[0] + 1.0_f32.sin()).abs() < TOL, "got {}", out.data[0]);
    assert!((out.data[1] - 1.0_f32.cos()).abs() < TOL, "got {}", out.data[1]);
}

#[test]
fn attention_single_head_matches_hand_computation() {
    // One query [1,0] against keys [[1,0],[0,1]], scale 1.0.
    // Scores (1, 0) -> softmax (0.73106, 0.26894).
    // Output = 0.73106*[1,0] + 0.26894*[0,1].
    let q = Tensor::new(&[1, 2], vec![1.0, 0.0]);
    let k = Tensor::new(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = Tensor::new(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let out = attention(&q, &k, &v, 1.0);
    assert_eq!(out.shape, vec![1, 2]);
    assert!((out.data[0] - 0.73106).abs() < TOL, "got {}", out.data[0]);
    assert!((out.data[1] - 0.26894).abs() < TOL, "got {}", out.data[1]);
}

#[test]
fn sampling_is_deterministic_for_seed() {
    let logits = [0.0, 1.0, 2.0];
    let a = sample_token(&logits, 42);
    let b = sample_token(&logits, 42);
    assert_eq!(a, b, "same seed must sample the same token");
    assert!(a < logits.len());
    // Greedy ignores the seed entirely.
    assert_eq!(greedy_token(&logits), 2);
}

#[test]
fn kv_cache_append_equals_concatenated() {
    // Append (k1, v1) then (k2, v2); the cache must equal the
    // concatenation of both, in order, with no rewrites of old slots.
    let k1 = Tensor::new(&[1, 2], vec![1.0, 0.0]);
    let v1 = Tensor::new(&[1, 2], vec![0.0, 1.0]);
    let k2 = Tensor::new(&[1, 2], vec![0.0, 1.0]);
    let v2 = Tensor::new(&[1, 2], vec![1.0, 0.0]);

    let mut cache = KvCache::new(2, 8);
    cache.append(&k1, &v1);
    cache.append(&k2, &v2);

    assert_eq!(cache.len(), 2);
    let keys = cache.keys();
    assert_eq!(keys.shape, vec![2, 2]);
    assert!((keys.data[0] - 1.0).abs() < TOL && keys.data[1].abs() < TOL);
    assert!(keys.data[2].abs() < TOL && (keys.data[3] - 1.0).abs() < TOL);

    let values = cache.values();
    assert_eq!(values.shape, vec![2, 2]);
    assert!(values.data[0].abs() < TOL && (values.data[1] - 1.0).abs() < TOL);
    assert!((values.data[2] - 1.0).abs() < TOL && values.data[3].abs() < TOL);
}