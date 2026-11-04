use inference_infra::capabilities::{Backend, QuantMode};
use qwen3::register::{ARCHITECTURES, IMPL_VERSION, register};

#[test]
fn registration_reports_only_implemented_host_capabilities() {
    let registration = register().expect("host registration must succeed");
    assert_eq!(registration.impl_version, IMPL_VERSION);
    assert_eq!(
        registration.architectures,
        ARCHITECTURES
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );
    let capabilities = registration.capabilities;
    assert_eq!(capabilities.backend, Backend::Cpu);
    assert!(capabilities.supports_rope);
    assert!(capabilities.supports_attention);
    assert!(!capabilities.supports_kv_cache);
    assert!(!capabilities.supports_sampling);
    assert_eq!(capabilities.quantization_modes, vec![QuantMode::Fp32]);
    assert_eq!(capabilities.max_context_len, 32_768);
}

#[test]
fn registration_is_stable_across_calls() {
    let first = register().expect("first registration");
    let second = register().expect("second registration");
    assert_eq!(first.architectures, second.architectures);
    assert_eq!(first.impl_version, second.impl_version);
    assert_eq!(first.capabilities.backend, second.capabilities.backend);
    assert_eq!(
        first.capabilities.supports_kv_cache,
        second.capabilities.supports_kv_cache
    );
}
