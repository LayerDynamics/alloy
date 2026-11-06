//! Live oMLX-method memory enforcer (poll tick + soft/hard ladder).
//!
//! Recomputes the hard ceiling every tick, classifies pressure against soft
//! and hard watermarks, prefers non-destructive MLX pool drain for a bounded
//! grace window, then pauses admissions under soft/hard pressure.
//!
//! This module does **not** call Metal or clear the MLX cache itself. The
//! poll thread only sets `cache_reclaim_requested`; the native worker /
//! generation path must honor that flag at a safe boundary (same rule as
//! oMLX: the enforcer never touches Metal directly).
//!
//! Multi-model LRU unload and in-flight abort are PR4 (M9). Prefill chunk
//! shrink helpers live in [`crate::prefill_admit`] (M5–M6); this enforcer
//! publishes the admit line those helpers consume.

use crate::admission::AdmissionController;
use crate::error::EngineError;
use crate::memory_guard::{
    self, CeilingBreakdown, HostMemorySnapshot, MemoryGuardConfig, PressureLevel, WatermarkBytes,
};
use crate::platform;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Default poll interval while pressure is elevated or work is active.
pub const ACTIVE_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Grace polls spent on pool drain before destructive soft/hard actions.
pub const PRESSURE_RECLAIM_GRACE_POLLS_MAX: u32 = 5;

/// MLX cache pool must exceed this before grace drain is requested.
pub const POOL_RECLAIM_FLOOR_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Emergency margin above the abort limit (oMLX: 2 GiB).
pub const EMERGENCY_OVER_CEILING_MARGIN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Consecutive over-ceiling polls before emergency pressure.
pub const EMERGENCY_OVER_CEILING_POLLS: u32 = 2;

/// Snapshot published for status / doctor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnforcerStatus {
    /// Soft / hard / ok.
    pub pressure: PressureLevel,
    /// Whether new admissions are paused.
    pub admissions_paused: bool,
    /// Whether a cache reclaim was requested and not yet cleared.
    pub cache_reclaim_requested: bool,
    /// Latest hard ceiling.
    pub hard_limit_bytes: u64,
    /// Soft watermark.
    pub soft_bytes: u64,
    /// Hard watermark.
    pub hard_bytes: u64,
    /// Abort limit (`min(static, metal)`).
    pub abort_limit_bytes: u64,
    /// Prefill admit line (`min(hard_ceiling, hard_watermark, prefill_headroom)`).
    pub admit_line_bytes: u64,
    /// Observed usage used for the last classification.
    pub usage_bytes: u64,
    /// Ceiling binding name, when known.
    pub binding: Option<&'static str>,
    /// Whether the last tick classified emergency pressure.
    pub emergency: bool,
}

fn binding_from_code(code: u64) -> Option<&'static str> {
    match code {
        1 => Some("static"),
        2 => Some("dynamic"),
        3 => Some("metal_cap"),
        _ => None,
    }
}

/// Actions the caller should apply after one tick (beyond shared-state updates).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnforcementTick {
    /// Published status after the tick.
    pub status: EnforcerStatus,
    /// True when the caller should ask the native worker to drain the MLX pool.
    pub request_cache_reclaim: bool,
    /// True when soft/hard destructive steps were deferred for grace.
    pub deferred_for_pool_drain: bool,
    /// Previous pressure level (for logging transitions).
    pub previous_pressure: PressureLevel,
}

/// Errors from one enforcer tick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnforcerError {
    /// Host memory probe failed.
    HostProbe(String),
    /// Ceiling math failed.
    Guard(memory_guard::MemoryGuardError),
    /// Admission budget resize failed.
    Admission(String),
}

impl fmt::Display for EnforcerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostProbe(message) => write!(formatter, "host memory probe: {message}"),
            Self::Guard(error) => write!(formatter, "{error}"),
            Self::Admission(message) => write!(formatter, "admission: {message}"),
        }
    }
}

impl std::error::Error for EnforcerError {}

/// Shared live enforcer handle (clone freely across threads).
#[derive(Clone, Debug)]
pub struct MemoryEnforcer {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    config: MemoryGuardConfig,
    admission: AdmissionController,
    /// Optional test override for host snapshot (None → live probe).
    host_override: Mutex<Option<HostMemorySnapshot>>,
    /// Optional test override for MLX active bytes.
    mlx_active_override: Mutex<Option<u64>>,
    /// Optional test override for MLX cache-pool bytes.
    mlx_pool_override: Mutex<Option<u64>>,
    pressure: Mutex<PressureLevel>,
    cache_reclaim_requested: AtomicBool,
    over_ceiling_polls: AtomicU64,
    reclaim_grace_polls: AtomicU64,
    hard_limit_bytes: AtomicU64,
    soft_bytes: AtomicU64,
    hard_bytes: AtomicU64,
    abort_limit_bytes: AtomicU64,
    admit_line_bytes: AtomicU64,
    usage_bytes: AtomicU64,
    /// Encoded ceiling binding: 0=none, 1=static, 2=dynamic, 3=metal_cap.
    binding_code: AtomicU64,
    emergency: AtomicBool,
}

impl MemoryEnforcer {
    /// Construct an enforcer bound to an admission ledger.
    pub fn new(config: MemoryGuardConfig, admission: AdmissionController) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                admission,
                host_override: Mutex::new(None),
                mlx_active_override: Mutex::new(None),
                mlx_pool_override: Mutex::new(None),
                pressure: Mutex::new(PressureLevel::Ok),
                cache_reclaim_requested: AtomicBool::new(false),
                over_ceiling_polls: AtomicU64::new(0),
                reclaim_grace_polls: AtomicU64::new(0),
                hard_limit_bytes: AtomicU64::new(0),
                soft_bytes: AtomicU64::new(0),
                hard_bytes: AtomicU64::new(0),
                abort_limit_bytes: AtomicU64::new(0),
                admit_line_bytes: AtomicU64::new(0),
                usage_bytes: AtomicU64::new(0),
                binding_code: AtomicU64::new(0),
                emergency: AtomicBool::new(false),
            }),
        }
    }

    /// Admission controller this enforcer pauses / resizes.
    pub fn admission(&self) -> &AdmissionController {
        &self.inner.admission
    }

    /// Guard configuration.
    pub fn config(&self) -> MemoryGuardConfig {
        self.inner.config
    }

    /// Whether the generation / worker path should clear the MLX cache.
    pub fn cache_reclaim_requested(&self) -> bool {
        self.inner
            .cache_reclaim_requested
            .load(Ordering::Acquire)
    }

    /// Clear the reclaim request after the worker honors it.
    pub fn clear_cache_reclaim_request(&self) {
        self.inner
            .cache_reclaim_requested
            .store(false, Ordering::Release);
    }

    /// Latest status for daemon logs / doctor.
    pub fn status(&self) -> EnforcerStatus {
        let pressure = *self
            .inner
            .pressure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        EnforcerStatus {
            pressure,
            admissions_paused: self.inner.admission.admissions_paused(),
            cache_reclaim_requested: self.cache_reclaim_requested(),
            hard_limit_bytes: self.inner.hard_limit_bytes.load(Ordering::Relaxed),
            soft_bytes: self.inner.soft_bytes.load(Ordering::Relaxed),
            hard_bytes: self.inner.hard_bytes.load(Ordering::Relaxed),
            abort_limit_bytes: self.inner.abort_limit_bytes.load(Ordering::Relaxed),
            admit_line_bytes: self.inner.admit_line_bytes.load(Ordering::Relaxed),
            usage_bytes: self.inner.usage_bytes.load(Ordering::Relaxed),
            binding: binding_from_code(self.inner.binding_code.load(Ordering::Relaxed)),
            emergency: self.inner.emergency.load(Ordering::Relaxed),
        }
    }

    /// Adapt a planned prefill chunk against the latest admit line.
    pub fn select_prefill_chunk(
        &self,
        requested_chunk_tokens: NonZeroUsize,
        min_chunk_tokens: NonZeroUsize,
        workspace_bytes_at_requested: u64,
    ) -> Result<crate::prefill_admit::PrefillAdmitDecision, EngineError> {
        let status = self.status();
        crate::prefill_admit::select_prefill_chunk(crate::prefill_admit::PrefillAdmitRequest {
            requested_chunk_tokens,
            min_chunk_tokens,
            usage_bytes: status.usage_bytes,
            admit_line_bytes: status.admit_line_bytes,
            workspace_bytes_at_requested,
            binding: status.binding,
        })
    }

    /// Test helper: pin host / MLX observations.
    #[cfg(test)]
    pub fn set_observation_overrides(
        &self,
        host: Option<HostMemorySnapshot>,
        mlx_active: Option<u64>,
        mlx_pool: Option<u64>,
    ) {
        *self
            .inner
            .host_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = host;
        *self
            .inner
            .mlx_active_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = mlx_active;
        *self
            .inner
            .mlx_pool_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = mlx_pool;
    }

    /// One enforcement tick (oMLX `_check_and_enforce`).
    pub fn tick(&self) -> Result<EnforcementTick, EnforcerError> {
        let host = self.observe_host()?;
        let mlx_active = self.observe_mlx_active();
        let pool_bytes = self.observe_mlx_pool();
        let usage = memory_guard::usage_bytes(mlx_active, host);
        let ceiling = memory_guard::ceiling_breakdown(self.inner.config, host, usage)
            .map_err(EnforcerError::Guard)?;
        let marks = memory_guard::watermark_bytes(self.inner.config.tier, ceiling);

        self.publish_ceiling(ceiling, marks, usage);

        if ceiling.hard_limit_bytes > 0 {
            if let Err(error) = self
                .inner
                .admission
                .resize_budget(ceiling.hard_limit_bytes)
            {
                // Budget below reserved is expected during a large load; keep
                // the previous ledger budget and continue pressure handling.
                let _ = error;
            }
        }

        let previous = *self
            .inner
            .pressure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if ceiling.hard_limit_bytes == 0 {
            self.set_pressure(PressureLevel::Ok);
            self.inner.admission.set_admissions_paused(false);
            self.inner.over_ceiling_polls.store(0, Ordering::Relaxed);
            self.inner.reclaim_grace_polls.store(0, Ordering::Relaxed);
            self.inner.emergency.store(false, Ordering::Relaxed);
            return Ok(EnforcementTick {
                status: self.status_with_binding(ceiling),
                request_cache_reclaim: false,
                deferred_for_pool_drain: false,
                previous_pressure: previous,
            });
        }

        let emergency = self.is_emergency(usage, marks.abort_limit_bytes, ceiling.hard_limit_bytes);
        self.inner.emergency.store(emergency, Ordering::Relaxed);

        let new_level = memory_guard::pressure_level(usage, marks);
        if new_level == PressureLevel::Ok {
            self.inner.reclaim_grace_polls.store(0, Ordering::Relaxed);
        }
        self.set_pressure(new_level);

        // Prefer non-destructive pool drain for a bounded grace window.
        if new_level != PressureLevel::Ok
            && !emergency
            && self.inner.reclaim_grace_polls.load(Ordering::Relaxed)
                < u64::from(PRESSURE_RECLAIM_GRACE_POLLS_MAX)
            && pool_bytes > POOL_RECLAIM_FLOOR_BYTES
        {
            self.inner
                .cache_reclaim_requested
                .store(true, Ordering::Release);
            self.inner
                .reclaim_grace_polls
                .fetch_add(1, Ordering::Relaxed);
            // Soft still pauses admissions during grace (oMLX propagates
            // admission_paused whenever pressure != ok).
            self.inner
                .admission
                .set_admissions_paused(new_level != PressureLevel::Ok);
            return Ok(EnforcementTick {
                status: self.status_with_binding(ceiling),
                request_cache_reclaim: true,
                deferred_for_pool_drain: true,
                previous_pressure: previous,
            });
        }

        match new_level {
            PressureLevel::Ok => {
                self.inner.admission.set_admissions_paused(false);
            }
            PressureLevel::Soft | PressureLevel::Hard => {
                self.inner.admission.set_admissions_paused(true);
                // Hard also asks for a pool drain after grace is exhausted.
                if new_level == PressureLevel::Hard {
                    self.inner
                        .cache_reclaim_requested
                        .store(true, Ordering::Release);
                }
            }
        }

        Ok(EnforcementTick {
            status: self.status_with_binding(ceiling),
            request_cache_reclaim: new_level == PressureLevel::Hard
                || self.cache_reclaim_requested(),
            deferred_for_pool_drain: false,
            previous_pressure: previous,
        })
    }

    fn status_with_binding(&self, _ceiling: CeilingBreakdown) -> EnforcerStatus {
        self.status()
    }

    fn publish_ceiling(
        &self,
        ceiling: CeilingBreakdown,
        marks: WatermarkBytes,
        usage: u64,
    ) {
        self.inner
            .hard_limit_bytes
            .store(ceiling.hard_limit_bytes, Ordering::Relaxed);
        self.inner
            .soft_bytes
            .store(marks.soft_bytes, Ordering::Relaxed);
        self.inner
            .hard_bytes
            .store(marks.hard_bytes, Ordering::Relaxed);
        self.inner
            .abort_limit_bytes
            .store(marks.abort_limit_bytes, Ordering::Relaxed);
        self.inner
            .admit_line_bytes
            .store(marks.admit_line_bytes, Ordering::Relaxed);
        self.inner.usage_bytes.store(usage, Ordering::Relaxed);
        self.inner.binding_code.store(
            match ceiling.binding {
                None => 0,
                Some(memory_guard::CeilingBinding::Static) => 1,
                Some(memory_guard::CeilingBinding::Dynamic) => 2,
                Some(memory_guard::CeilingBinding::MetalCap) => 3,
            },
            Ordering::Relaxed,
        );
    }

    fn set_pressure(&self, level: PressureLevel) {
        *self
            .inner
            .pressure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = level;
    }

    fn is_emergency(&self, usage: u64, abort_limit: u64, hard_limit: u64) -> bool {
        let ceiling = if abort_limit > 0 {
            abort_limit
        } else {
            hard_limit
        };
        if ceiling == 0 || usage < ceiling {
            self.inner.over_ceiling_polls.store(0, Ordering::Relaxed);
            return false;
        }
        let polls = self.inner.over_ceiling_polls.fetch_add(1, Ordering::Relaxed) + 1;
        usage >= ceiling.saturating_add(EMERGENCY_OVER_CEILING_MARGIN_BYTES)
            || polls >= u64::from(EMERGENCY_OVER_CEILING_POLLS)
    }

    fn observe_host(&self) -> Result<HostMemorySnapshot, EnforcerError> {
        if let Some(host) = *self
            .inner
            .host_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Ok(host);
        }
        platform::discover_host_memory_snapshot()
            .map_err(|error| EnforcerError::HostProbe(error.to_string()))
    }

    fn observe_mlx_active(&self) -> u64 {
        if let Some(value) = *self
            .inner
            .mlx_active_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return value;
        }
        mlx_ffi::memory::active_memory_bytes()
            .ok()
            .map(|value| value as u64)
            .unwrap_or(0)
    }

    fn observe_mlx_pool(&self) -> u64 {
        if let Some(value) = *self
            .inner
            .mlx_pool_override
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return value;
        }
        mlx_ffi::memory::cache_memory_bytes()
            .ok()
            .map(|value| value as u64)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_guard::{HostMemorySnapshot, MemoryGuardTier};

    const GIB: u64 = 1024 * 1024 * 1024;

    fn host_busy() -> HostMemorySnapshot {
        HostMemorySnapshot {
            physical_memory_bytes: 36 * GIB,
            free_bytes: 1 * GIB,
            inactive_bytes: 0,
            active_file_bytes: 0,
            other_apps_active_bytes: 0,
            phys_footprint_bytes: 30 * GIB,
            graphics_footprint_bytes: 20 * GIB,
            iogpu_wired_limit_bytes: 0,
            recommended_working_set_bytes: 100 * GIB,
            release_credit_bytes: 0,
        }
    }

    fn enforcer_with(host: HostMemorySnapshot, usage_mlx: u64, pool: u64) -> MemoryEnforcer {
        let admission = AdmissionController::new(32 * GIB).expect("admission");
        let enforcer = MemoryEnforcer::new(MemoryGuardConfig::balanced(), admission);
        enforcer.set_observation_overrides(Some(host), Some(usage_mlx), Some(pool));
        enforcer
    }

    #[test]
    fn ok_pressure_unpauses_admissions() {
        let mut host = host_busy();
        host.phys_footprint_bytes = 1 * GIB;
        host.graphics_footprint_bytes = 0;
        host.free_bytes = 20 * GIB;
        host.inactive_bytes = 5 * GIB;
        let enforcer = enforcer_with(host, 1 * GIB, 0);
        enforcer.admission().set_admissions_paused(true);
        let tick = enforcer.tick().expect("tick");
        assert_eq!(tick.status.pressure, PressureLevel::Ok);
        assert!(!enforcer.admission().admissions_paused());
        assert!(!tick.deferred_for_pool_drain);
    }

    #[test]
    fn soft_pressure_pauses_and_defers_when_pool_large() {
        // H = usage + free - reserve = 20 + 5 - 3 = 22 GiB
        // soft = 0.90*22 = 19.8, hard = 0.95*22 = 20.9 → usage 20 is soft.
        let mut host = host_busy();
        host.free_bytes = 5 * GIB;
        host.inactive_bytes = 0;
        host.active_file_bytes = 0;
        host.other_apps_active_bytes = 0;
        host.phys_footprint_bytes = 20 * GIB;
        host.graphics_footprint_bytes = 0;
        host.recommended_working_set_bytes = 100 * GIB;
        let enforcer = enforcer_with(host, 20 * GIB, 3 * GIB);
        let tick = enforcer.tick().expect("tick");
        assert_eq!(tick.status.pressure, PressureLevel::Soft);
        assert!(enforcer.admission().admissions_paused());
        assert!(tick.deferred_for_pool_drain);
        assert!(tick.request_cache_reclaim);
        assert!(enforcer.cache_reclaim_requested());
    }

    #[test]
    fn hard_pressure_requests_reclaim_after_grace_exhausted() {
        // static = 33 GiB binds; soft=29.7 hard=31.35; usage 32 → hard. pool=0 skips grace.
        let mut host = host_busy();
        host.free_bytes = 5 * GIB;
        host.inactive_bytes = 0;
        host.active_file_bytes = 0;
        host.other_apps_active_bytes = 0;
        host.phys_footprint_bytes = 32 * GIB;
        host.graphics_footprint_bytes = 0;
        host.recommended_working_set_bytes = 100 * GIB;
        let enforcer = enforcer_with(host, 32 * GIB, 0);
        let tick = enforcer.tick().expect("tick");
        assert_eq!(tick.status.pressure, PressureLevel::Hard);
        assert!(enforcer.admission().admissions_paused());
        assert!(tick.request_cache_reclaim);
        assert!(!tick.deferred_for_pool_drain);
    }

    #[test]
    fn paused_admission_rejects_reserve() {
        let admission = AdmissionController::new(1024).expect("admission");
        admission.set_admissions_paused(true);
        let error = admission.reserve("load", 16).expect_err("paused");
        match error {
            crate::error::EngineError::MemoryLimitExceeded { operation, .. } => {
                assert!(operation.contains("paused"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn tier_name_exposed_via_config() {
        let admission = AdmissionController::new(1024).expect("admission");
        let enforcer = MemoryEnforcer::new(MemoryGuardConfig::balanced(), admission);
        assert_eq!(enforcer.config().tier, MemoryGuardTier::Balanced);
    }
}
