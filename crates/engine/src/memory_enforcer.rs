//! Live oMLX-method memory enforcer (poll tick + emergency cancellation).
//!
//! Recomputes the hard ceiling every tick, classifies pressure against soft
//! and hard watermarks, and pauses admissions immediately under pressure,
//! including the bounded grace window for non-destructive MLX pool drain.
//!
//! This module does **not** call Metal or clear the MLX cache itself. The
//! poll thread publishes allocator-drain requests and idle-model eviction
//! authorization; FamilyRuntime revalidates them on the native worker.
//!
//! Only emergency pressure requests cooperative execution cancellation.
//! The polling bridge captures the request and scheduler lease before the
//! observation, then signals that target without waiting on the native
//! command queue. Execution retains all ownership until native cleanup.
//! Prefill chunk shrink helpers live in [`crate::prefill_admit`] (M5–M6);
//! this enforcer publishes the admit line those helpers consume.

use crate::admission::AdmissionController;
use crate::error::EngineError;
use crate::memory_guard::{
    self, CeilingBreakdown, HostMemorySnapshot, MemoryGuardConfig, PressureLevel, WatermarkBytes,
};
use crate::platform;
use crate::scheduler::ExecutionCancellationTarget;
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
    /// Persistent pressure authorizes at most one idle-model eviction.
    pub request_model_eviction: bool,
    /// Emergency interruption of the execution captured before observation.
    ///
    /// Signal this target through the scheduler, never the native command
    /// queue. Requesting cancellation neither completes cleanup nor releases
    /// reservations or the execution slot. None also covers an idle scheduler.
    pub request_execution_cancellation: Option<ExecutionCancellationTarget>,
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
    observation_source: Option<Arc<dyn MemoryObservationSource>>,
    admission: AdmissionController,
    /// Optional test override for host snapshot (None → live probe).
    host_override: Mutex<Option<HostMemorySnapshot>>,
    /// Optional test override for MLX active bytes.
    mlx_active_override: Mutex<Option<u64>>,
    /// Optional test override for MLX cache-pool bytes.
    mlx_pool_override: Mutex<Option<u64>>,
    pressure: Mutex<PressureLevel>,
    cache_reclaim_requested: AtomicBool,
    model_eviction_requested: AtomicBool,
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

/// One host and allocator observation supplied to the enforcer.
#[derive(Clone, Copy, Debug)]
pub struct MemoryObservation {
    /// Host-memory counters used by the ceiling calculation.
    pub host: HostMemorySnapshot,
    /// Allocator active bytes; zero retains the host-footprint fallback.
    pub mlx_active_bytes: u64,
    /// Unused allocator-cache bytes, not resident model reservations.
    pub mlx_pool_bytes: u64,
}

/// Injectable observation source; never owns or reclaims native models.
///
/// Called on the enforcement executor. Production defaults to live probes;
/// alternate providers support platform adapters and deterministic tests.
pub trait MemoryObservationSource: fmt::Debug + Send + Sync + 'static {
    /// Obtain the next observation, without changing runtime ownership.
    fn observe(&self) -> Result<MemoryObservation, EnforcerError>;
}

impl MemoryEnforcer {
    /// Construct an enforcer using an explicit observation provider.
    pub fn with_observation_source(
        config: MemoryGuardConfig,
        admission: AdmissionController,
        source: Arc<dyn MemoryObservationSource>,
    ) -> Self {
        let mut enforcer = Self::new(config, admission);
        Arc::get_mut(&mut enforcer.inner)
            .expect("new enforcer has unique ownership")
            .observation_source = Some(source);
        enforcer
    }

    /// Construct an enforcer bound to an admission ledger.
    pub fn new(config: MemoryGuardConfig, admission: AdmissionController) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                observation_source: None,
                admission,
                host_override: Mutex::new(None),
                mlx_active_override: Mutex::new(None),
                mlx_pool_override: Mutex::new(None),
                pressure: Mutex::new(PressureLevel::Ok),
                cache_reclaim_requested: AtomicBool::new(false),
                model_eviction_requested: AtomicBool::new(false),
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
        self.inner.cache_reclaim_requested.load(Ordering::Acquire)
    }

    /// Latest completed poll authorizes destructive idle-model reclamation.
    /// A newer grace or normal-pressure poll revokes older queued actions.
    pub(crate) fn model_eviction_requested(&self) -> bool {
        self.inner.model_eviction_requested.load(Ordering::Acquire)
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

    /// Observe pressure and update reclamation/admission without an execution
    /// target. Use the live polling bridge for emergency cancellation.
    pub fn tick(&self) -> Result<EnforcementTick, EnforcerError> {
        self.tick_for_execution(None)
    }

    /// One enforcement tick bound to a scheduler snapshot taken before the
    /// memory observation. Ordinary soft/hard pressure never requests
    /// cancellation; emergency pressure bypasses allocator-drain grace.
    pub fn tick_for_execution(
        &self,
        execution: Option<ExecutionCancellationTarget>,
    ) -> Result<EnforcementTick, EnforcerError> {
        // Revoke old authority while obtaining a new observation. Probe
        // failure must not leave a stale destructive request authorized.
        self.inner
            .model_eviction_requested
            .store(false, Ordering::Release);
        let observation = self.observe_memory()?;
        let host = observation.host;
        let mlx_active = observation.mlx_active_bytes;
        let pool_bytes = observation.mlx_pool_bytes;
        let usage = memory_guard::usage_bytes(mlx_active, host);
        let ceiling = memory_guard::ceiling_breakdown(self.inner.config, host, usage)
            .map_err(EnforcerError::Guard)?;
        let marks = memory_guard::watermark_bytes(self.inner.config.tier, ceiling);

        self.publish_ceiling(ceiling, marks, usage);

        if ceiling.hard_limit_bytes > 0 {
            if let Err(error) = self.inner.admission.resize_budget(ceiling.hard_limit_bytes) {
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
            self.clear_cache_reclaim_request();
            self.set_pressure(PressureLevel::Ok);
            self.inner.admission.set_admissions_paused(false);
            self.inner.over_ceiling_polls.store(0, Ordering::Relaxed);
            self.inner.reclaim_grace_polls.store(0, Ordering::Relaxed);
            self.inner.emergency.store(false, Ordering::Relaxed);
            return Ok(EnforcementTick {
                status: self.status_with_binding(ceiling),
                request_cache_reclaim: false,
                request_model_eviction: false,
                request_execution_cancellation: None,
                deferred_for_pool_drain: false,
                previous_pressure: previous,
            });
        }

        let emergency = self.is_emergency(usage, marks.abort_limit_bytes, ceiling.hard_limit_bytes);
        self.inner.emergency.store(emergency, Ordering::Relaxed);

        let new_level = memory_guard::pressure_level(usage, marks);
        if new_level == PressureLevel::Ok {
            self.clear_cache_reclaim_request();
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
                request_model_eviction: false,
                request_execution_cancellation: None,
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

        self.inner
            .model_eviction_requested
            .store(new_level != PressureLevel::Ok, Ordering::Release);
        Ok(EnforcementTick {
            status: self.status_with_binding(ceiling),
            request_model_eviction: new_level != PressureLevel::Ok,
            request_execution_cancellation: execution.filter(|_| emergency),
            request_cache_reclaim: new_level == PressureLevel::Hard
                || self.cache_reclaim_requested(),
            deferred_for_pool_drain: false,
            previous_pressure: previous,
        })
    }

    fn observe_memory(&self) -> Result<MemoryObservation, EnforcerError> {
        match &self.inner.observation_source {
            Some(source) => source.observe(),
            None => Ok(MemoryObservation {
                host: self.observe_host()?,
                mlx_active_bytes: self.observe_mlx_active(),
                mlx_pool_bytes: self.observe_mlx_pool(),
            }),
        }
    }

    pub(crate) fn refresh_pressure(&self) -> Result<EnforcerStatus, EnforcerError> {
        let observation = self.observe_memory()?;
        let usage = memory_guard::usage_bytes(observation.mlx_active_bytes, observation.host);
        let ceiling = memory_guard::ceiling_breakdown(self.inner.config, observation.host, usage)
            .map_err(EnforcerError::Guard)?;
        let marks = memory_guard::watermark_bytes(self.inner.config.tier, ceiling);
        self.publish_ceiling(ceiling, marks, usage);
        if ceiling.hard_limit_bytes > 0 {
            let _ = self.inner.admission.resize_budget(ceiling.hard_limit_bytes);
        }
        let level = if ceiling.hard_limit_bytes == 0 {
            PressureLevel::Ok
        } else {
            memory_guard::pressure_level(usage, marks)
        };
        self.set_pressure(level);
        self.inner
            .admission
            .set_admissions_paused(level != PressureLevel::Ok);
        if level == PressureLevel::Ok {
            self.inner
                .model_eviction_requested
                .store(false, Ordering::Release);
            self.clear_cache_reclaim_request();
            self.inner.reclaim_grace_polls.store(0, Ordering::Relaxed);
            self.inner.over_ceiling_polls.store(0, Ordering::Relaxed);
            self.inner.emergency.store(false, Ordering::Relaxed);
        }
        Ok(self.status())
    }

    fn status_with_binding(&self, _ceiling: CeilingBreakdown) -> EnforcerStatus {
        self.status()
    }

    fn publish_ceiling(&self, ceiling: CeilingBreakdown, marks: WatermarkBytes, usage: u64) {
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
        let polls = self
            .inner
            .over_ceiling_polls
            .fetch_add(1, Ordering::Relaxed)
            + 1;
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
    use crate::scheduler::{RequestState, Scheduler};
    use crate::types::RequestId;

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

    fn fixed_ceiling_host(usage: u64) -> HostMemorySnapshot {
        HostMemorySnapshot {
            // Balanced static ceiling is 33 GiB; dynamic and Metal ceilings
            // exceed it, so these tests exercise the abort classifier itself.
            free_bytes: 10 * GIB,
            phys_footprint_bytes: usage,
            graphics_footprint_bytes: 0,
            ..host_busy()
        }
    }

    #[test]
    fn ordinary_soft_and_hard_pressure_never_request_cancellation() {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let request = RequestId::new("ordinary-pressure").expect("request");
        let _permit = scheduler.try_acquire(request.clone()).expect("permit");
        let target = scheduler.active_cancellation_target().expect("target");
        for usage in [30 * GIB, 32 * GIB] {
            let enforcer = enforcer_with(fixed_ceiling_host(usage), usage, 3 * GIB);
            // Cover both grace and post-grace policy, with the real target.
            for _ in 0..PRESSURE_RECLAIM_GRACE_POLLS_MAX + 2 {
                let tick = enforcer.tick_for_execution(target.clone()).expect("tick");
                assert_ne!(tick.status.pressure, PressureLevel::Ok);
                assert!(tick.status.admissions_paused);
                assert!(!tick.status.emergency);
                assert!(tick.request_execution_cancellation.is_none());
            }
        }
        assert_eq!(
            scheduler.request_state(&request).expect("state"),
            RequestState::Running
        );
    }

    #[test]
    fn consecutive_over_ceiling_polls_interrupt_the_captured_execution() {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let _permit = scheduler
            .try_acquire(RequestId::new("consecutive-pressure").expect("request"))
            .expect("permit");
        let target = scheduler.active_cancellation_target().expect("target");
        let enforcer = enforcer_with(fixed_ceiling_host(33 * GIB), 33 * GIB, 3 * GIB);

        let first = enforcer
            .tick_for_execution(target.clone())
            .expect("first tick");
        assert!(first.deferred_for_pool_drain);
        assert!(!first.status.emergency);
        assert!(first.request_execution_cancellation.is_none());

        let emergency = enforcer
            .tick_for_execution(target.clone())
            .expect("second tick");
        assert!(emergency.status.emergency);
        assert!(emergency.status.admissions_paused);
        assert!(!emergency.deferred_for_pool_drain);
        assert!(emergency.request_cache_reclaim);
        assert!(emergency.request_model_eviction);
        assert_eq!(emergency.request_execution_cancellation, target);
    }

    #[test]
    fn emergency_margin_requests_cancellation_on_the_first_poll() {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let _permit = scheduler
            .try_acquire(RequestId::new("margin-pressure").expect("request"))
            .expect("permit");
        let target = scheduler.active_cancellation_target().expect("target");
        let enforcer = enforcer_with(fixed_ceiling_host(35 * GIB), 35 * GIB, 3 * GIB);
        let tick = enforcer.tick_for_execution(target.clone()).expect("tick");
        assert!(tick.status.emergency);
        assert!(!tick.deferred_for_pool_drain);
        assert_eq!(tick.request_execution_cancellation, target);

        // Emergency pressure while idle still pauses admission and reclaims,
        // but must not manufacture a target for a request that starts later.
        let idle = enforcer.tick().expect("idle tick");
        assert!(idle.status.emergency);
        assert!(idle.status.admissions_paused);
        assert!(idle.request_execution_cancellation.is_none());
    }

    #[test]
    fn acceptable_observation_resets_emergency_history_and_action() {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let _permit = scheduler
            .try_acquire(RequestId::new("recovered-pressure").expect("request"))
            .expect("permit");
        let target = scheduler.active_cancellation_target().expect("target");
        let enforcer = enforcer_with(fixed_ceiling_host(35 * GIB), 35 * GIB, 0);
        assert!(
            enforcer
                .tick_for_execution(target.clone())
                .expect("emergency")
                .status
                .emergency
        );

        enforcer.set_observation_overrides(Some(fixed_ceiling_host(GIB)), Some(GIB), Some(0));
        let recovery = enforcer
            .tick_for_execution(target.clone())
            .expect("recovery");
        assert_eq!(recovery.status.pressure, PressureLevel::Ok);
        assert!(!recovery.status.admissions_paused);
        assert!(!recovery.status.emergency);
        assert!(recovery.request_execution_cancellation.is_none());

        enforcer.set_observation_overrides(
            Some(fixed_ceiling_host(33 * GIB)),
            Some(33 * GIB),
            Some(0),
        );
        let first = enforcer.tick_for_execution(target).expect("new first poll");
        assert!(!first.status.emergency);
        assert!(first.request_execution_cancellation.is_none());
    }

    #[test]
    fn disabled_guard_never_interrupts_an_execution() {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let _permit = scheduler
            .try_acquire(RequestId::new("disabled-pressure").expect("request"))
            .expect("permit");
        let mut config = MemoryGuardConfig::balanced();
        config.enabled = false;
        let enforcer = MemoryEnforcer::new(
            config,
            AdmissionController::new(32 * GIB).expect("admission"),
        );
        enforcer.set_observation_overrides(
            Some(fixed_ceiling_host(35 * GIB)),
            Some(35 * GIB),
            Some(0),
        );
        let tick = enforcer
            .tick_for_execution(scheduler.active_cancellation_target().expect("target"))
            .expect("tick");
        assert!(!tick.status.emergency);
        assert!(!tick.status.admissions_paused);
        assert!(tick.request_execution_cancellation.is_none());
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
