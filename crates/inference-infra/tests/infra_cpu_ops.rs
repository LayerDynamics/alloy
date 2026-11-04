use inference_infra::capabilities::{Backend, BackendCapabilities};
use inference_infra::error::KernelError;
use inference_infra::ops::{apply_rope, attention, greedy_token, sample_token, KvCache};
use inference_infra::tensor::Tensor;
use mlx_ffi::{Device, DeviceType};

const TOL: f32 = 1e-4;

fn t(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::new(shape, data).expect("tensor construction")
}

/// Comment 7: detect() must report the actual active device. On the Linux
/// lane that is CPU; on a Metal host it is Metal. Consistency, not a
/// hardcoded value, is the contract.
#[test]
fn capabilities_report_active_device() {
    let caps = BackendCapabilities::detect().expect("device query");
    let expected = match Device::get_default().expect("default device").kind().expect("kind") {
        DeviceType::Cpu => Backend::Cpu,
        DeviceType::Gpu => Backend::Metal,
    };
    assert_eq!(caps.backend, expected, "detect() must report the active MLX device");
    assert!(caps.supports_rope && caps.supports_attention && caps.supports_kv_cache && caps.supports_sampling);
    assert!(caps.max_context_len >= 4096);
}

#[test]
fn rope_rotates_by_expected_angle() {
    let x = t(&[1, 1, 2], vec![1.0, 0.0]);
    let out = apply_rope(&x, 1, 10_000.0).expect("rope");
    assert_eq!(out.shape(), &[1, 1, 2]);
    let d = out.data().expect("data");
    assert!((d[0] - 1.0_f32.cos()).abs() < TOL, "got {}", d[0]);
    assert!((d[1] - 1.0_f32.sin()).abs() < TOL, "got {}", d[1]);

    let x = t(&[1, 1, 2], vec![0.0, 1.0]);
    let out = apply_rope(&x, 1, 10_000.0).expect("rope");
    let d = out.data().expect("data");
    assert!((d[0] + 1.0_f32.sin()).abs() < TOL, "got {}", d[0]);
    assert!((d[1] - 1.0_f32.cos()).abs() < TOL, "got {}", d[1]);
}

#[test]
fn attention_single_head_matches_hand_computation() {
    let q = t(&[1, 2], vec![1.0, 0.0]);
    let k = t(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = t(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let out = attention(&q, &k, &v, 1.0).expect("attention");
    assert_eq!(out.shape(), &[1, 2]);
    let d = out.data().expect("data");
    assert!((d[0] - 0.73106).abs() < TOL, "got {}", d[0]);
    assert!((d[1] - 0.26894).abs() < TOL, "got {}", d[1]);
}

#[test]
fn attention_rejects_empty_kv() {
    let q = t(&[1, 2], vec![1.0, 0.0]);
    let k = t(&[0, 2], vec![]);
    let v = t(&[0, 2], vec![]);
    assert_eq!(attention(&q, &k, &v, 1.0).unwrap_err(), KernelError::Empty);
}

#[test]
fn attention_rejects_non_finite_inputs() {
    let q = t(&[1, 2], vec![f32::NAN, 0.0]);
    let k = t(&[1, 2], vec![1.0, 0.0]);
    let v = t(&[1, 2], vec![0.0, 1.0]);
    assert_eq!(attention(&q, &k, &v, 1.0).unwrap_err(), KernelError::NonFinite);
}

#[test]
fn sampling_is_deterministic_for_seed() {
    let logits = [0.0, 1.0, 2.0];
    let a = sample_token(&logits, 42).expect("sample");
    let b = sample_token(&logits, 42).expect("sample");
    assert_eq!(a, b, "same seed must sample the same token");
    assert!(a < logits.len());
    assert_eq!(greedy_token(&logits).expect("greedy"), 2);
}

/// Comment 10: determinism alone would pass a constant sampler.
/// softmax([0, 1, 2]) is roughly [0.0900, 0.2447, 0.6652]; over 1000
/// seeded draws the empirical frequency must track that distribution.
#[test]
fn sampling_follows_logit_distribution() {
    let logits = [0.0, 1.0, 2.0];
    let mut counts = [0usize; 3];
    for seed in 0..1000u64 {
        let tok = sample_token(&logits, seed).expect("sample");
        counts[tok] += 1;
    }
    assert_eq!(counts.iter().sum::<usize>(), 1000);
    let frac2 = counts[2] as f64 / 1000.0;
    assert!((frac2 - 0.6652).abs() < 0.05, "token-2 frequency {frac2}");
    assert!(counts[0] > 0 && counts[1] > 0, "unreachable tokens: {counts:?}");
}

#[test]
fn sampling_rejects_invalid_logits() {
    assert_eq!(sample_token(&[], 42), Err(KernelError::Empty));
    assert_eq!(sample_token(&[f32::NEG_INFINITY; 3], 42), Err(KernelError::NoFiniteLogits));
    assert_eq!(sample_token(&[f32::NAN, 1.0, 2.0], 42), Err(KernelError::NonFinite));
    assert_eq!(sample_token(&[f32::INFINITY, 1.0, 2.0], 42), Err(KernelError::NonFinite));
}

#[test]
fn greedy_rejects_invalid_logits() {
    assert_eq!(greedy_token(&[]), Err(KernelError::Empty));
    // Comment 9: NaN at index 0 must fail even when later logits are finite.
    assert_eq!(greedy_token(&[f32::NAN, 1.0, 2.0]), Err(KernelError::NonFinite));
    assert_eq!(greedy_token(&[f32::NEG_INFINITY; 3]), Err(KernelError::NoFiniteLogits));
}

#[test]
fn kv_cache_append_equals_concatenated() {
    let k1 = t(&[1, 2], vec![1.0, 0.0]);
    let v1 = t(&[1, 2], vec![0.0, 1.0]);
    let k2 = t(&[1, 2], vec![0.0, 1.0]);
    let v2 = t(&[1, 2], vec![1.0, 0.0]);

    let mut cache = KvCache::new(2, 8);
    cache.append(&k1, &v1).expect("append 1");
    cache.append(&k2, &v2).expect("append 2");

    assert_eq!(cache.len(), 2);
    let keys = cache.keys().expect("keys");
    assert_eq!(keys.shape(), &[2, 2]);
    let kd = keys.data().expect("data");
    assert!((kd[0] - 1.0).abs() < TOL && kd[1].abs() < TOL);
    assert!(kd[2].abs() < TOL && (kd[3] - 1.0).abs() < TOL);

    let values = cache.values().expect("values");
    assert_eq!(values.shape(), &[2, 2]);
    let vd = values.data().expect("data");
    assert!(vd[0].abs() < TOL && (vd[1] - 1.0).abs() < TOL);
    assert!((vd[2] - 1.0).abs() < TOL && vd[3].abs() < TOL);
}