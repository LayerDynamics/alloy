//! Read-only live runtime metrics, independent of the native command queue.

use crate::rpc::RpcError;
use engine::api::EngineCore;
use engine::memory_enforcer::MemoryEnforcer;
use engine::runtime::FamilyRuntime;
use serde_json::{Value, json};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Source for the additive `daemon.metrics` snapshot RPC.
///
/// Implementations must read metadata without queuing native work or changing
/// admission, cancellation, model ownership, or memory-enforcement state.
pub trait MetricsSource: Send + Sync {
    /// Collect one JSON snapshot using the versioned metrics schema.
    fn snapshot(&self) -> Result<Value, RpcError>;
}

/// Runtime metrics over the daemon's existing engine and memory enforcer.
///
/// Residency and pressure observations are cached with explicit ages. Scheduler
/// and admission reads use their short control-path locks, never native calls.
pub struct LiveEngineMetrics {
    engine: EngineCore<FamilyRuntime>,
    enforcer: MemoryEnforcer,
}

impl LiveEngineMetrics {
    /// Bind the running engine and its existing enforcer over the same ledger.
    pub fn new(engine: EngineCore<FamilyRuntime>, enforcer: MemoryEnforcer) -> Self {
        Self { engine, enforcer }
    }
}

impl MetricsSource for LiveEngineMetrics {
    fn snapshot(&self) -> Result<Value, RpcError> {
        let runtime = self.engine.runtime();
        let admission = runtime
            .admission_snapshot()
            .map_err(|error| RpcError::internal(format!("read admission metrics: {error}")))?;
        let execution = self
            .engine
            .execution_snapshot()
            .map_err(|error| RpcError::internal(format!("read execution metrics: {error}")))?;
        let residency = runtime.residency_snapshot();
        let guard = self.enforcer.metrics_snapshot();
        let sample = guard.sample.as_ref();
        let models: Vec<_> = residency
            .models
            .iter()
            .map(|model| {
                json!({
                    "artifact_id": model.artifact.as_str(),
                    "weights_reserved_bytes": model.weights_reserved_bytes,
                    "kv_reserved_bytes": model.kv_reserved_bytes,
                    "pinned": model.pinned,
                    "leased": model.leased,
                })
            })
            .collect();
        Ok(json!({
            "schema_version": 1,
            "process_id": std::process::id(),
            "sampled_at_unix_ms": milliseconds(
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
            ),
            "admission": {
                "budget_bytes": admission.budget_bytes(),
                "reserved_bytes": admission.reserved_bytes(),
                "available_bytes": admission.available_bytes(),
                "active_reservations": admission.active_reservations(),
                "admissions_paused": runtime.admission_controller().admissions_paused(),
            },
            "execution": {
                "active_request_id": execution.active_request_id.as_ref().map(|id| id.as_str()),
                "cancellation_requested": execution.cancellation_requested,
            },
            "kv_cache": runtime.tiered_kv_snapshot(),
            "residency": {
                "worker_alive": residency.worker_alive,
                "sample_age_ms": milliseconds(residency.sample_age),
                "models_resident": models.len(),
                "models": models,
            },
            "memory_guard": {
                "tier": self.enforcer.config().tier.as_str(),
                "pressure": sample.map_or("unknown", |sample| sample.status.pressure.as_str()),
                "emergency": sample.is_some_and(|sample| sample.status.emergency),
                "usage_bytes": sample.map(|sample| sample.status.usage_bytes),
                "static_ceiling_bytes": sample.map(|sample| sample.ceiling.static_bytes),
                "dynamic_ceiling_bytes": sample.map(|sample| sample.ceiling.dynamic_bytes),
                "metal_ceiling_bytes": sample.map(|sample| sample.ceiling.metal_cap_bytes),
                "hard_limit_bytes": sample.map(|sample| sample.status.hard_limit_bytes),
                "soft_bytes": sample.map(|sample| sample.status.soft_bytes),
                "hard_bytes": sample.map(|sample| sample.status.hard_bytes),
                "abort_limit_bytes": sample.map(|sample| sample.status.abort_limit_bytes),
                "admit_line_bytes": sample.map(|sample| sample.status.admit_line_bytes),
                "binding": sample.and_then(|sample| sample.status.binding),
                "sample_age_ms": guard.sample_age.map(milliseconds),
                "last_error": guard.last_error,
                "cache_reclaim_requested": self.enforcer.cache_reclaim_requested(),
            },
        }))
    }
}

fn milliseconds(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}
