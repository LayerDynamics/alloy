//! Deterministic selection of idle model eviction candidates.
//!
//! This module owns no native resources or admission reservations.
//! Selection is advisory: the executor must revalidate live ownership,
//! mark the instance evicting, destroy its resources, and release its
//! guards before reporting reclamation. Snapshot byte estimates must
//! never be credited as released memory.
//!
//! Times are caller-supplied monotonic durations from one shared origin.
//! No wall clock, sleeps, background thread, or allocator access is used.

use std::time::Duration;

/// Snapshot of one uniquely identified loaded or loading model instance.
#[derive(Clone, Debug)]
pub struct ResidencySnapshot {
    /// Unique native-instance key; not merely the model artifact name.
    pub instance_key: u64,
    /// Whether model loading is still in progress.
    pub loading: bool,
    /// Whether destruction has already been claimed by the executor.
    pub evicting: bool,
    /// Number of active generations protecting this instance.
    pub active_generations: usize,
    /// Number of live leases protecting this instance.
    pub live_leases: usize,
    /// Number of explicit residency pins.
    pub explicit_pins: usize,
    /// Last successful use, measured from the shared monotonic origin.
    pub last_used: Duration,
    /// Start of the current uninterrupted idle interval.
    pub idle_since: Option<Duration>,
    /// Optional per-instance idle TTL; zero means immediately expired.
    pub idle_ttl: Option<Duration>,
}

impl ResidencySnapshot {
    /// Whether this snapshot permits ordinary idle eviction at `now`.
    ///
    /// Future or inconsistent timestamps fail closed. Pins and leases
    /// remain protective even when the TTL has expired.
    pub fn is_evictable(&self, now: Duration) -> bool {
        !self.loading
            && !self.evicting
            && self.active_generations == 0
            && self.live_leases == 0
            && self.explicit_pins == 0
            && self
                .idle_since
                .is_some_and(|idle| self.last_used <= idle && idle <= now)
    }
}

/// Reason to consider ordinary idle-model eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionTrigger {
    /// Only instances whose idle TTL has expired are candidates.
    IdleTtl,
    /// Select an eligible idle LRU instance, regardless of TTL.
    MemoryPressure,
}

/// Select one eligible victim deterministically.
///
/// Ties in last-use time are broken by unique instance key. The caller
/// must supply unique keys and snapshots from the same clock origin.
///
/// Returning a key does not mutate ownership, reclaim memory, authorize
/// cancellation, or permit eviction without executor-side revalidation.
pub fn select_idle_victim(
    entries: &[ResidencySnapshot],
    now: Duration,
    trigger: EvictionTrigger,
) -> Option<u64> {
    entries
        .iter()
        .filter(|entry| entry.is_evictable(now))
        .filter(|entry| match trigger {
            EvictionTrigger::MemoryPressure => true,
            EvictionTrigger::IdleTtl => match (entry.idle_since, entry.idle_ttl) {
                (Some(idle), Some(ttl)) => {
                    now.checked_sub(idle).is_some_and(|elapsed| elapsed >= ttl)
                }
                _ => false,
            },
        })
        .min_by_key(|entry| (entry.last_used, entry.instance_key))
        .map(|entry| entry.instance_key)
}

/// Why a proposed idle eviction cannot currently be authorized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionClaimError {
    /// The original instance is absent or has been replaced.
    InstanceChanged,
    /// Current ownership or timestamps prohibit idle eviction.
    Protected,
    /// The instance is idle but its TTL has not expired or is disabled.
    TtlNotExpired,
}

/// Revalidate an advisory victim using a fresh executor-owned snapshot.
///
/// The executor must call this in the same serialized operation that
/// claims destruction. No command may acquire ownership between this
/// check and that transition.
///
/// Instance keys must never be reused for replacement allocations.
/// Success neither marks an entry evicting nor releases any resources.
/// This function does not re-run global LRU ordering; it validates the
/// safety and trigger eligibility of the originally selected instance.
pub fn validate_idle_eviction(
    selected_instance: u64,
    current: Option<&ResidencySnapshot>,
    now: Duration,
    trigger: EvictionTrigger,
) -> Result<(), EvictionClaimError> {
    let current = current
        .filter(|entry| entry.instance_key == selected_instance)
        .ok_or(EvictionClaimError::InstanceChanged)?;

    if !current.is_evictable(now) {
        return Err(EvictionClaimError::Protected);
    }

    if select_idle_victim(std::slice::from_ref(current), now, trigger).is_none() {
        return Err(EvictionClaimError::TtlNotExpired);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
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
}

#[cfg(test)]
mod claim_tests {
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
}
