use super::*;

fn policy() -> UnifiedMemoryPolicy {
    UnifiedMemoryPolicy {
        os_application_headroom_bytes: 4_000_000_000,
        control_plane_allowance_bytes: 500_000_000,
        allocator_headroom_bytes: 1_000_000_000,
        cache_limit_bytes: 250_000_000,
        engine_budget_override_bytes: None,
    }
}

#[test]
fn unified_budget_uses_one_pool_not_ram_plus_gpu_guidance() {
    let info = UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, true)
        .expect("observations");

    let plan = info.plan(policy()).expect("plan");

    assert_eq!(plan.shared_ceiling_bytes(), 27_000_000_000);
    assert_eq!(plan.engine_budget_bytes(), 25_500_000_000);
    assert_eq!(plan.mlx_memory_guideline_bytes(), 26_500_000_000);
    assert_eq!(plan.mlx_cache_limit_bytes(), 250_000_000);
}

#[test]
fn non_power_of_two_and_nonrounded_ram_counts_are_preserved() {
    let physical = 36_000_000_123;
    let info =
        UnifiedMemoryInfo::from_observations(physical, u64::MAX, true).expect("observations");

    let plan = info.plan(policy()).expect("plan");

    assert_eq!(plan.platform().physical_memory_bytes(), physical);
    assert_eq!(plan.shared_ceiling_bytes(), 32_000_000_123);
    assert_eq!(plan.engine_budget_bytes(), 30_500_000_123);
}

#[test]
fn discrete_memory_device_is_rejected() {
    assert!(matches!(
        UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, false,),
        Err(PlatformError::NotUnifiedMemory),
    ));
}

#[test]
fn missing_observations_are_not_fabricated() {
    assert!(UnifiedMemoryInfo::from_observations(0, 1, true).is_err());
    assert!(UnifiedMemoryInfo::from_observations(1, 0, true).is_err());
}

#[test]
fn smaller_override_is_used_exactly() {
    let info = UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, true)
        .expect("observations");

    let mut selected = policy();
    selected.engine_budget_override_bytes = Some(12_345_678_901);

    let plan = info.plan(selected).expect("override");
    assert_eq!(plan.engine_budget_bytes(), 12_345_678_901);
    assert_eq!(plan.mlx_memory_guideline_bytes(), 13_345_678_901);
}

#[test]
fn zero_or_oversized_overrides_are_rejected() {
    let info = UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, true)
        .expect("observations");

    for value in [0, 25_500_000_001] {
        let mut selected = policy();
        selected.engine_budget_override_bytes = Some(value);
        assert!(info.plan(selected).is_err());
    }
}

#[test]
fn cached_buffers_must_fit_the_same_allocator_allowance() {
    let info = UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, true)
        .expect("observations");

    let mut selected = policy();
    selected.cache_limit_bytes = selected.allocator_headroom_bytes + 1;

    assert!(info.plan(selected).is_err());
}

#[test]
fn excessive_headroom_and_arithmetic_overflow_are_rejected() {
    let info = UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, true)
        .expect("observations");

    let mut selected = policy();
    selected.os_application_headroom_bytes = 40_000_000_000;
    assert!(info.plan(selected).is_err());

    let mut selected = policy();
    selected.control_plane_allowance_bytes = u64::MAX;
    assert!(matches!(
        info.plan(selected),
        Err(PlatformError::ArithmeticOverflow),
    ));
}

#[test]
fn admission_uses_the_derived_shared_budget_without_reserving_it() {
    let info = UnifiedMemoryInfo::from_observations(36_000_000_000, 27_000_000_000, true)
        .expect("observations");

    let plan = info.plan(policy()).expect("plan");
    let admission = plan.admission_controller().expect("controller");
    let snapshot = admission.snapshot().expect("snapshot");

    assert_eq!(snapshot.budget_bytes(), plan.engine_budget_bytes());
    assert_eq!(snapshot.reserved_bytes(), 0);
    assert_eq!(snapshot.active_reservations(), 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_discovery_requires_actual_unified_metal_device() {
    let actual = discover().expect("native unified-memory platform");
    assert!(actual.physical_memory_bytes() > 0);
    assert!(actual.recommended_working_set_bytes() > 0);
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn unsupported_target_does_not_claim_unified_metal_discovery() {
    assert!(matches!(
        discover(),
        Err(PlatformError::UnsupportedPlatform),
    ));
}
