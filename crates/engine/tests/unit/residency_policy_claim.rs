use super::*;

fn idle() -> ResidencySnapshot {
    ResidencySnapshot {
        instance_key: 7,
        loading: false,
        evicting: false,
        active_generations: 0,
        live_leases: 0,
        explicit_pins: 0,
        last_used: Duration::ZERO,
        idle_since: Some(Duration::ZERO),
        idle_ttl: Some(Duration::from_secs(10)),
    }
}

fn claim(entry: &ResidencySnapshot) -> Result<(), EvictionClaimError> {
    validate_idle_eviction(
        7,
        Some(entry),
        Duration::from_secs(20),
        EvictionTrigger::MemoryPressure,
    )
}

#[test]
fn pin_lease_generation_or_load_after_selection_blocks_claim() {
    for protection in 0..4 {
        let mut entry = idle();
        let selected = select_idle_victim(
            &[entry.clone()],
            Duration::from_secs(20),
            EvictionTrigger::MemoryPressure,
        );
        assert_eq!(selected, Some(7));

        match protection {
            0 => entry.explicit_pins = 1,
            1 => entry.live_leases = 1,
            2 => entry.active_generations = 1,
            _ => entry.loading = true,
        }
        assert_eq!(claim(&entry), Err(EvictionClaimError::Protected));
    }
}

#[test]
fn missing_or_replaced_instance_rejects_stale_candidate() {
    assert_eq!(
        validate_idle_eviction(
            7,
            None,
            Duration::from_secs(20),
            EvictionTrigger::MemoryPressure,
        ),
        Err(EvictionClaimError::InstanceChanged),
    );

    let mut replacement = idle();
    replacement.instance_key = 8;
    assert_eq!(
        claim(&replacement),
        Err(EvictionClaimError::InstanceChanged),
    );
}

#[test]
fn already_claimed_eviction_is_not_authorized_twice() {
    let mut entry = idle();
    assert_eq!(claim(&entry), Ok(()));

    // Models the executor's serialized claim transition.
    entry.evicting = true;
    assert_eq!(claim(&entry), Err(EvictionClaimError::Protected));
}

#[test]
fn renewed_idle_interval_invalidates_a_stale_ttl_candidate() {
    let mut entry = idle();
    entry.last_used = Duration::from_secs(19);
    entry.idle_since = Some(Duration::from_secs(19));

    assert_eq!(
        validate_idle_eviction(
            7,
            Some(&entry),
            Duration::from_secs(20),
            EvictionTrigger::IdleTtl,
        ),
        Err(EvictionClaimError::TtlNotExpired),
    );
    assert_eq!(claim(&entry), Ok(()));
}
