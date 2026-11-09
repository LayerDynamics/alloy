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
#[path = "../tests/unit/residency_policy_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/residency_policy_claim.rs"]
mod claim_tests;
