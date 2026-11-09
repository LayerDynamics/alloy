use super::*;

fn seconds(value: u64) -> Duration {
    Duration::from_secs(value)
}

fn idle(key: u64, last: u64) -> ResidencySnapshot {
    ResidencySnapshot {
        instance_key: key,
        loading: false,
        evicting: false,
        active_generations: 0,
        live_leases: 0,
        explicit_pins: 0,
        last_used: seconds(last),
        idle_since: Some(seconds(last)),
        idle_ttl: Some(seconds(10)),
    }
}

#[test]
fn ownership_and_pin_protections_override_expired_ttl_and_pressure() {
    for protection in 0..6 {
        let mut entry = idle(1, 0);
        match protection {
            0 => entry.loading = true,
            1 => entry.evicting = true,
            2 => entry.active_generations = 1,
            3 => entry.live_leases = 1,
            4 => entry.explicit_pins = 1,
            _ => entry.idle_since = None,
        }
        for trigger in [EvictionTrigger::IdleTtl, EvictionTrigger::MemoryPressure] {
            assert_eq!(
                select_idle_victim(&[entry.clone()], seconds(100), trigger),
                None,
            );
        }
    }
}

#[test]
fn ttl_expires_at_exact_boundary_from_idle_start() {
    let mut entry = idle(1, 1);
    entry.idle_since = Some(seconds(90));
    assert_eq!(
        select_idle_victim(&[entry.clone()], seconds(99), EvictionTrigger::IdleTtl),
        None,
    );
    assert_eq!(
        select_idle_victim(&[entry], seconds(100), EvictionTrigger::IdleTtl),
        Some(1),
    );
}

#[test]
fn disabled_ttl_still_allows_pressure_eviction() {
    let mut entry = idle(1, 0);
    entry.idle_ttl = None;
    assert_eq!(
        select_idle_victim(&[entry.clone()], seconds(100), EvictionTrigger::IdleTtl),
        None,
    );
    assert_eq!(
        select_idle_victim(&[entry], seconds(100), EvictionTrigger::MemoryPressure),
        Some(1),
    );
}

#[test]
fn zero_ttl_expires_immediately() {
    let mut entry = idle(1, 10);
    entry.idle_ttl = Some(Duration::ZERO);
    assert_eq!(
        select_idle_victim(&[entry], seconds(10), EvictionTrigger::IdleTtl),
        Some(1),
    );
}

#[test]
fn pressure_chooses_lru_then_instance_key_independent_of_input_order() {
    let mut entries = vec![idle(9, 5), idle(3, 5), idle(1, 8)];
    assert_eq!(
        select_idle_victim(&entries, seconds(10), EvictionTrigger::MemoryPressure),
        Some(3),
    );
    entries.reverse();
    assert_eq!(
        select_idle_victim(&entries, seconds(10), EvictionTrigger::MemoryPressure),
        Some(3),
    );
}

#[test]
fn ttl_selection_excludes_unexpired_instances() {
    let mut older = idle(1, 1);
    older.idle_ttl = Some(seconds(100));
    let newer = idle(2, 5);
    assert_eq!(
        select_idle_victim(&[older, newer], seconds(15), EvictionTrigger::IdleTtl),
        Some(2),
    );
}

#[test]
fn invalid_or_future_timestamps_fail_closed() {
    let mut future_idle = idle(1, 0);
    future_idle.idle_since = Some(seconds(20));
    let mut inconsistent = idle(2, 9);
    inconsistent.idle_since = Some(seconds(8));
    for entry in [future_idle, inconsistent] {
        assert!(!entry.is_evictable(seconds(10)));
    }
}

#[test]
fn selection_is_advisory_and_does_not_claim_the_instance() {
    let entries = [idle(1, 0)];
    for _ in 0..2 {
        assert_eq!(
            select_idle_victim(&entries, seconds(10), EvictionTrigger::MemoryPressure),
            Some(1),
        );
    }
    assert!(!entries[0].evicting);
}

#[test]
fn no_candidate_is_an_honest_result() {
    assert_eq!(
        select_idle_victim(&[], seconds(10), EvictionTrigger::MemoryPressure),
        None,
    );
}
