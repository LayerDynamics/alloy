//! FR-35 anonymous telemetry client: in-memory counts, TLS flush, offline drop.

use crate::error::{HarnessError, HarnessResult};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// Embedded copy of `docs/events/telemetry-v1.json` for conformance tests.
pub const TELEMETRY_SCHEMA_V1: &str = include_str!("../../../docs/events/telemetry-v1.json");

/// Hardware RAM tier label used in anonymous telemetry (not a power-of-two class).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareTier {
    /// 8 GB unified memory.
    Ram8Gb,
    /// 16 GB unified memory.
    Ram16Gb,
    /// 18 GB unified memory.
    Ram18Gb,
    /// 24 GB unified memory.
    Ram24Gb,
    /// 32 GB unified memory.
    Ram32Gb,
    /// 36 GB unified memory (first-class tier).
    Ram36Gb,
    /// 48 GB unified memory.
    Ram48Gb,
    /// 64 GB unified memory.
    Ram64Gb,
    /// 96 GB unified memory.
    Ram96Gb,
    /// 128 GB or larger unified memory.
    Ram128GbPlus,
    /// Tier could not be classified.
    Unknown,
}

impl HardwareTier {
    /// Map detected physical RAM bytes to a tier label.
    pub fn from_total_bytes(total_bytes: u64) -> Self {
        const GIB: u64 = 1024 * 1024 * 1024;
        let gib = total_bytes / GIB;
        match gib {
            0..=8 => Self::Ram8Gb,
            9..=16 => Self::Ram16Gb,
            17..=18 => Self::Ram18Gb,
            19..=24 => Self::Ram24Gb,
            25..=32 => Self::Ram32Gb,
            33..=36 => Self::Ram36Gb,
            37..=48 => Self::Ram48Gb,
            49..=64 => Self::Ram64Gb,
            65..=96 => Self::Ram96Gb,
            _ => Self::Ram128GbPlus,
        }
    }
}

/// Schema-conforming anonymous telemetry payload (no prompts, paths, or model ids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryPayload {
    /// Schema version string, always `"telemetry-v1"` for this payload shape.
    pub schema_version: String,
    /// Daemon / package version string (semver-ish; not a model id).
    pub daemon_version: String,
    /// Detected hardware tier.
    pub hardware_tier: HardwareTier,
    /// Sessions observed since process start (or since last successful flush reset).
    pub session_count: u64,
    /// Completed turns observed.
    pub turn_count: u64,
    /// Aggregate tokens/s (0 when unused in Phase 1).
    pub tokens_per_s: f64,
    /// Aggregate TTFT in milliseconds (0 when unused).
    pub ttft_ms: f64,
    /// Aggregate cache hit rate 0.0–1.0 (0 when unused).
    pub cache_hit_rate: f64,
    /// Aggregate tool latency in milliseconds (0 in Phase 1 — no tools).
    pub tool_latency_ms: f64,
    /// Aggregate reserved memory bytes.
    pub memory_reserved_bytes: u64,
    /// Aggregate actual memory bytes.
    pub memory_actual_bytes: u64,
}

impl TelemetryPayload {
    /// Build a payload from live counters.
    pub fn new(
        daemon_version: impl Into<String>,
        hardware_tier: HardwareTier,
        session_count: u64,
        turn_count: u64,
    ) -> Self {
        Self {
            schema_version: "telemetry-v1".to_owned(),
            daemon_version: daemon_version.into(),
            hardware_tier,
            session_count,
            turn_count,
            tokens_per_s: 0.0,
            ttft_ms: 0.0,
            cache_hit_rate: 0.0,
            tool_latency_ms: 0.0,
            memory_reserved_bytes: 0,
            memory_actual_bytes: 0,
        }
    }

    /// Reject payloads that violate Phase-1 anonymity invariants.
    pub fn assert_anonymous(&self) -> HarnessResult<()> {
        let json = serde_json::to_value(self)?;
        assert_value_anonymous(&json, "payload")?;
        Ok(())
    }
}

fn assert_value_anonymous(value: &serde_json::Value, path: &str) -> HarnessResult<()> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let lower = key.to_ascii_lowercase();
                if lower.contains("prompt")
                    || lower.contains("path")
                    || lower == "model"
                    || lower.contains("model_id")
                    || lower.contains("artifact")
                    || lower.contains("tokenizer")
                {
                    return Err(HarnessError::Telemetry(format!(
                        "forbidden telemetry field at {path}.{key}"
                    )));
                }
                assert_value_anonymous(child, &format!("{path}.{key}"))?;
            }
        }
        serde_json::Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_value_anonymous(child, &format!("{path}[{index}]"))?;
            }
        }
        serde_json::Value::String(text) => {
            if text.contains('/') && (text.starts_with('/') || text.contains(":\\")) {
                return Err(HarnessError::Telemetry(format!(
                    "telemetry string looks like a filesystem path at {path}"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

/// How the client performs an HTTPS flush (injectable for tests).
pub trait TelemetryTransport: Send + Sync {
    /// POST `body` to `endpoint`. Return Ok on 2xx; Err otherwise.
    fn post_json(&self, endpoint: &str, body: &str) -> Result<(), String>;
}

/// Default transport using `ureq` with TLS.
#[derive(Debug, Default)]
pub struct UreqTransport;

impl TelemetryTransport for UreqTransport {
    fn post_json(&self, endpoint: &str, body: &str) -> Result<(), String> {
        let response = ureq::post(endpoint)
            .set("Content-Type", "application/json")
            .send_string(body)
            .map_err(|error| error.to_string())?;
        let status = response.status();
        if (200..300).contains(&status) {
            Ok(())
        } else {
            Err(format!("HTTP {status}"))
        }
    }
}

struct Counters {
    session_count: u64,
    turn_count: u64,
    tokens_per_s: f64,
    ttft_ms: f64,
    cache_hit_rate: f64,
    tool_latency_ms: f64,
    memory_reserved_bytes: u64,
    memory_actual_bytes: u64,
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            session_count: 0,
            turn_count: 0,
            tokens_per_s: 0.0,
            ttft_ms: 0.0,
            cache_hit_rate: 0.0,
            tool_latency_ms: 0.0,
            memory_reserved_bytes: 0,
            memory_actual_bytes: 0,
        }
    }
}

/// In-memory FR-35 telemetry client.
pub struct TelemetryClient {
    daemon_version: String,
    hardware_tier: HardwareTier,
    endpoint: String,
    transport: Box<dyn TelemetryTransport>,
    counters: Mutex<Counters>,
}

impl TelemetryClient {
    /// Create a client that flushes to `endpoint` over the default TLS transport.
    pub fn new(
        daemon_version: impl Into<String>,
        hardware_tier: HardwareTier,
        endpoint: impl Into<String>,
    ) -> Self {
        Self::with_transport(
            daemon_version,
            hardware_tier,
            endpoint,
            Box::new(UreqTransport),
        )
    }

    /// Create a client with an injectable transport (tests).
    pub fn with_transport(
        daemon_version: impl Into<String>,
        hardware_tier: HardwareTier,
        endpoint: impl Into<String>,
        transport: Box<dyn TelemetryTransport>,
    ) -> Self {
        Self {
            daemon_version: daemon_version.into(),
            hardware_tier,
            endpoint: endpoint.into(),
            transport,
            counters: Mutex::new(Counters::default()),
        }
    }

    /// Record that a session was created.
    pub fn record_session(&self) {
        if let Ok(mut counters) = self.counters.lock() {
            counters.session_count = counters.session_count.saturating_add(1);
        }
    }

    /// Record that a turn completed.
    pub fn record_turn(&self) {
        if let Ok(mut counters) = self.counters.lock() {
            counters.turn_count = counters.turn_count.saturating_add(1);
        }
    }

    /// Snapshot the current payload without flushing.
    pub fn snapshot(&self) -> HarnessResult<TelemetryPayload> {
        let counters = self
            .counters
            .lock()
            .map_err(|_| HarnessError::Telemetry("counter lock poisoned".into()))?;
        let mut payload = TelemetryPayload::new(
            self.daemon_version.clone(),
            self.hardware_tier,
            counters.session_count,
            counters.turn_count,
        );
        payload.tokens_per_s = counters.tokens_per_s;
        payload.ttft_ms = counters.ttft_ms;
        payload.cache_hit_rate = counters.cache_hit_rate;
        payload.tool_latency_ms = counters.tool_latency_ms;
        payload.memory_reserved_bytes = counters.memory_reserved_bytes;
        payload.memory_actual_bytes = counters.memory_actual_bytes;
        payload.assert_anonymous()?;
        Ok(payload)
    }

    /// Flush counts over HTTPS/TLS.
    ///
    /// On failure the in-memory counts are dropped (offline drop — no on-disk
    /// queue). Errors never affect sessions; callers may log and ignore.
    pub fn flush(&self) -> HarnessResult<()> {
        let payload = self.snapshot()?;
        let body = serde_json::to_string(&payload)
            .map_err(|error| HarnessError::Telemetry(error.to_string()))?;
        match self.transport.post_json(&self.endpoint, &body) {
            Ok(()) => {
                if let Ok(mut counters) = self.counters.lock() {
                    *counters = Counters::default();
                }
                Ok(())
            }
            Err(error) => {
                // Offline drop: discard the batch; do not queue.
                if let Ok(mut counters) = self.counters.lock() {
                    *counters = Counters::default();
                }
                Err(HarnessError::Telemetry(format!(
                    "flush failed (counts dropped): {error}"
                )))
            }
        }
    }
}
