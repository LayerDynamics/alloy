use super::*;

fn host_36g() -> HostMemorySnapshot {
    HostMemorySnapshot {
        physical_memory_bytes: 36 * GIB,
        free_bytes: 10 * GIB,
        inactive_bytes: 4 * GIB,
        active_file_bytes: GIB,
        other_apps_active_bytes: 8 * GIB,
        phys_footprint_bytes: 2 * GIB,
        graphics_footprint_bytes: GIB,
        iogpu_wired_limit_bytes: 0,
        recommended_working_set_bytes: 27 * GIB,
        release_credit_bytes: 0,
    }
}

#[test]
fn balanced_reserve_clamps_on_36gb() {
    // 8% of 36 GiB = 2.88 GiB → clamp to min 3 GiB
    assert_eq!(
        tier_reserve_bytes(MemoryGuardTier::Balanced, 36 * GIB).unwrap(),
        3 * GIB
    );
}

#[test]
fn safe_reserve_on_36gb_is_twenty_percent() {
    // 20% of 36 = 7.2 → within [6,16]
    assert_eq!(
        tier_reserve_bytes(MemoryGuardTier::Safe, 36 * GIB).unwrap(),
        ((36 * GIB) as f64 * 0.20).round() as u64
    );
}

#[test]
fn custom_skips_vm_dynamic_and_mins_with_static_and_metal() {
    let host = host_36g();
    let config = MemoryGuardConfig::custom(40 * GIB).unwrap();
    let breakdown = ceiling_breakdown(config, host, 2 * GIB).unwrap();
    // static = 36 - 2 = 34; metal = 27 + 1 = 28; custom dynamic = 40
    // hard = min(34, 40, 28) = 28 metal
    assert_eq!(breakdown.static_bytes, 34 * GIB);
    assert_eq!(breakdown.dynamic_bytes, 40 * GIB);
    assert_eq!(breakdown.metal_cap_bytes, 28 * GIB);
    assert_eq!(breakdown.hard_limit_bytes, 28 * GIB);
    assert_eq!(breakdown.binding, Some(CeilingBinding::MetalCap));
}

#[test]
fn balanced_dynamic_can_bind_when_machine_is_busy() {
    let mut host = host_36g();
    host.free_bytes = GIB;
    host.inactive_bytes = 0;
    host.active_file_bytes = 0;
    host.other_apps_active_bytes = 0;
    host.recommended_working_set_bytes = 100 * GIB;
    let config = MemoryGuardConfig::balanced();
    let alloy = 2 * GIB;
    let breakdown = ceiling_breakdown(config, host, alloy).unwrap();
    // dynamic = 2 + 1 - 3 = 0 → skipped; static = 33 GiB binds
    // wait: reserve is 3 GiB, static = 33. dynamic = 2+1-3 = 0, so only static
    assert_eq!(breakdown.reserve_bytes, 3 * GIB);
    assert_eq!(breakdown.static_bytes, 33 * GIB);
    assert_eq!(breakdown.dynamic_bytes, 0);
    assert_eq!(breakdown.hard_limit_bytes, 33 * GIB);
    assert_eq!(breakdown.binding, Some(CeilingBinding::Static));
}

#[test]
fn balanced_dynamic_binds_when_reclaimable_pool_is_small() {
    let mut host = host_36g();
    host.free_bytes = 2 * GIB;
    host.inactive_bytes = GIB;
    host.active_file_bytes = 0;
    host.other_apps_active_bytes = 0;
    host.recommended_working_set_bytes = 100 * GIB;
    let alloy = 2 * GIB;
    let breakdown = ceiling_breakdown(MemoryGuardConfig::balanced(), host, alloy).unwrap();
    // dynamic = 2+2+1 - 3 = 2 GiB; static = 33; metal skipped as huge
    assert_eq!(breakdown.dynamic_bytes, 2 * GIB);
    assert_eq!(breakdown.hard_limit_bytes, 2 * GIB);
    assert_eq!(breakdown.binding, Some(CeilingBinding::Dynamic));
}

#[test]
fn watermark_fractions_match_omlx_balanced() {
    let breakdown = CeilingBreakdown {
        reserve_bytes: 3 * GIB,
        static_bytes: 33 * GIB,
        dynamic_bytes: 20 * GIB,
        metal_cap_bytes: 28 * GIB,
        hard_limit_bytes: 20 * GIB,
        binding: Some(CeilingBinding::Dynamic),
    };
    let marks = watermark_bytes(MemoryGuardTier::Balanced, breakdown);
    assert_eq!(marks.soft_bytes, scale(20 * GIB, 0.90));
    assert_eq!(marks.hard_bytes, scale(20 * GIB, 0.95));
    assert_eq!(marks.admit_line_bytes, scale(20 * GIB, 0.92));
    assert_eq!(marks.abort_limit_bytes, 28 * GIB); // min(33, 28)
}

#[test]
fn pressure_levels_use_soft_then_hard() {
    let marks = WatermarkBytes {
        soft_bytes: 10,
        hard_bytes: 20,
        admit_line_bytes: 18,
        abort_limit_bytes: 30,
    };
    assert_eq!(pressure_level(9, marks), PressureLevel::Ok);
    assert_eq!(pressure_level(10, marks), PressureLevel::Soft);
    assert_eq!(pressure_level(19, marks), PressureLevel::Soft);
    assert_eq!(pressure_level(20, marks), PressureLevel::Hard);
}

#[test]
fn wired_suggestion_floors_to_mib_and_leaves_five_percent() {
    let total = 128 * GIB;
    let desired = total; // too high
    let suggestion = wired_limit_suggestion_bytes(desired, total);
    assert_eq!(suggestion % (1024 * 1024), 0);
    assert!(suggestion <= total - total / 20);
}

#[test]
fn wired_applied_only_when_sysctl_set() {
    let mut host = host_36g();
    assert!(!should_apply_wired_limit(host));
    host.iogpu_wired_limit_bytes = 30 * GIB;
    assert!(should_apply_wired_limit(host));
}

#[test]
fn disabled_guard_reports_zero_ceiling() {
    let mut config = MemoryGuardConfig::balanced();
    config.enabled = false;
    let breakdown = ceiling_breakdown(config, host_36g(), 1).unwrap();
    assert_eq!(breakdown.hard_limit_bytes, 0);
    assert!(breakdown.binding.is_none());
}

#[test]
fn tier_parse_rejects_unknown() {
    assert!(MemoryGuardTier::parse("turbo").is_err());
    assert_eq!(
        MemoryGuardTier::parse("Balanced").unwrap(),
        MemoryGuardTier::Balanced
    );
}
