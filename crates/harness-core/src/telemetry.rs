//! FR-35 anonymous telemetry: bounded background TLS flush and offline drop.

use crate::error::{HarnessError, HarnessResult};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const FLUSH_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const FLUSH_WAIT_TIMEOUT: Duration = Duration::from_secs(12);

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
    /// Sessions observed since the previous flush attempt (or process start).
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

/// Default transport using `ureq` with TLS and finite I/O deadlines.
///
/// Redirects are not followed. The configured destination must use HTTPS.
/// The operating system's DNS resolver cannot be interrupted by `ureq`, so
/// transport work also remains isolated from generation and client shutdown.
#[derive(Debug, Default)]
pub struct UreqTransport;

impl TelemetryTransport for UreqTransport {
    fn post_json(&self, endpoint: &str, body: &str) -> Result<(), String> {
        post_json_with_agent(&https_agent(HTTP_TIMEOUT), endpoint, body)
    }
}

fn https_agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .https_only(true)
        .redirects(0)
        .timeout_connect(timeout)
        .timeout_read(timeout)
        .timeout_write(timeout)
        .timeout(timeout)
        .max_idle_connections(0)
        .build()
}

fn post_json_with_agent(agent: &ureq::Agent, endpoint: &str, body: &str) -> Result<(), String> {
    let response = agent
        .post(endpoint)
        .set("Content-Type", "application/json")
        .send_string(body)
        // Neither endpoint credentials nor a remote response body belongs in
        // a telemetry diagnostic. The worker further reduces this to a fixed
        // error class before logging or replying to a caller.
        .map_err(|error| format!("HTTPS request failed ({:?})", error.kind()))?;
    let status = response.status();
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(format!("HTTP {status}"))
    }
}

#[derive(Default)]
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

impl Counters {
    fn payload(&self, daemon_version: &str, hardware_tier: HardwareTier) -> TelemetryPayload {
        let mut payload = TelemetryPayload::new(
            daemon_version,
            hardware_tier,
            self.session_count,
            self.turn_count,
        );
        payload.tokens_per_s = self.tokens_per_s;
        payload.ttft_ms = self.ttft_ms;
        payload.cache_hit_rate = self.cache_hit_rate;
        payload.tool_latency_ms = self.tool_latency_ms;
        payload.memory_reserved_bytes = self.memory_reserved_bytes;
        payload.memory_actual_bytes = self.memory_actual_bytes;
        payload
    }

    fn is_empty(&self) -> bool {
        self.session_count == 0 && self.turn_count == 0
    }
}

struct Shared {
    daemon_version: String,
    hardware_tier: HardwareTier,
    counters: Mutex<Counters>,
    stopped: AtomicBool,
}

impl Shared {
    fn take_payload(&self) -> Result<Option<TelemetryPayload>, FlushFailure> {
        // Remove precisely this batch before any I/O. Counts recorded during
        // a slow/failed POST belong to the next batch and must not be erased.
        let counters = {
            let mut guard = self.counters.lock().map_err(|_| FlushFailure::Counters)?;
            std::mem::take(&mut *guard)
        };
        if counters.is_empty() {
            return Ok(None);
        }
        let payload = counters.payload(&self.daemon_version, self.hardware_tier);
        payload
            .assert_anonymous()
            .map_err(|_| FlushFailure::Payload)?;
        Ok(Some(payload))
    }

    fn drop_pending(&self) {
        if let Ok(mut counters) = self.counters.lock() {
            *counters = Counters::default();
        }
    }
}

#[derive(Clone, Copy)]
enum FlushFailure {
    Worker,
    Counters,
    Payload,
    Transport,
}

impl FlushFailure {
    fn message(self) -> &'static str {
        match self {
            Self::Worker => "worker unavailable (counts dropped)",
            Self::Counters => "counter state unavailable (counts dropped)",
            Self::Payload => "invalid anonymous payload (counts dropped)",
            Self::Transport => "HTTPS flush failed (counts dropped)",
        }
    }

    fn error(self) -> HarnessError {
        HarnessError::Telemetry(self.message().to_owned())
    }
}

type Diagnostic = Arc<dyn Fn(&'static str) + Send + Sync>;

struct WorkerOptions {
    interval: Duration,
    wait_timeout: Duration,
    diagnostic: Diagnostic,
}

impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            interval: FLUSH_INTERVAL,
            wait_timeout: FLUSH_WAIT_TIMEOUT,
            diagnostic: Arc::new(|message| eprintln!("alloy telemetry: {message}")),
        }
    }
}

enum FlushRequest {
    Background,
    Wait(SyncSender<HarnessResult<()>>),
}

/// In-memory FR-35 telemetry client with one owned transport worker.
///
/// Recording and [`TelemetryClient::request_flush`] never perform network I/O.
/// A capacity-one signal queue coalesces flush requests, and the worker also
/// checks pending counts at least every 24 hours while idle. Failed batches
/// are dropped, with a fixed diagnostic; nothing is persisted or retried.
pub struct TelemetryClient {
    shared: Arc<Shared>,
    requests: Option<SyncSender<FlushRequest>>,
    worker: Option<JoinHandle<()>>,
    wait_timeout: Duration,
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
    ///
    /// The transport runs only on this client's single worker, including for
    /// explicit [`TelemetryClient::flush`] calls. It should apply finite I/O
    /// timeouts; a blocked injected transport cannot block recording, turn
    /// completion, or client destruction.
    pub fn with_transport(
        daemon_version: impl Into<String>,
        hardware_tier: HardwareTier,
        endpoint: impl Into<String>,
        transport: Box<dyn TelemetryTransport>,
    ) -> Self {
        Self::with_options(
            daemon_version.into(),
            hardware_tier,
            endpoint.into(),
            transport,
            WorkerOptions::default(),
        )
    }

    fn with_options(
        daemon_version: String,
        hardware_tier: HardwareTier,
        endpoint: String,
        transport: Box<dyn TelemetryTransport>,
        options: WorkerOptions,
    ) -> Self {
        let shared = Arc::new(Shared {
            daemon_version,
            hardware_tier,
            counters: Mutex::new(Counters::default()),
            stopped: AtomicBool::new(false),
        });
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker_shared = Arc::clone(&shared);
        let diagnostic = Arc::clone(&options.diagnostic);
        let spawned = thread::Builder::new()
            .name("alloy-telemetry".to_owned())
            .spawn(move || {
                flush_worker(
                    worker_shared,
                    receiver,
                    endpoint,
                    transport,
                    options.interval,
                    diagnostic,
                );
            });
        let (requests, worker) = match spawned {
            Ok(worker) => (Some(sender), Some(worker)),
            Err(_) => {
                // Startup failure is visible but cannot make session startup
                // fail. A failed worker is not repeatedly respawned per turn.
                (options.diagnostic)(FlushFailure::Worker.message());
                (None, None)
            }
        };
        Self {
            shared,
            requests,
            worker,
            wait_timeout: options.wait_timeout,
        }
    }

    /// Record that a session was created.
    pub fn record_session(&self) {
        if self.requests.is_some()
            && let Ok(mut counters) = self.shared.counters.lock()
        {
            counters.session_count = counters.session_count.saturating_add(1);
        }
    }

    /// Record that a turn completed.
    pub fn record_turn(&self) {
        if self.requests.is_some()
            && let Ok(mut counters) = self.shared.counters.lock()
        {
            counters.turn_count = counters.turn_count.saturating_add(1);
        }
    }

    /// Snapshot pending counters without flushing or waiting on the transport.
    ///
    /// A batch already taken by the worker is not in this snapshot. Removing
    /// a batch from the counters is not a claim that it was delivered.
    pub fn snapshot(&self) -> HarnessResult<TelemetryPayload> {
        let counters = self
            .shared
            .counters
            .lock()
            .map_err(|_| FlushFailure::Counters.error())?;
        let payload = counters.payload(&self.shared.daemon_version, self.shared.hardware_tier);
        payload.assert_anonymous()?;
        Ok(payload)
    }

    /// Request a background flush without waiting for transport or queue space.
    ///
    /// Repeated requests coalesce behind at most one in-flight POST. This is
    /// the production turn-completion path; failures are logged by the worker
    /// without including payloads, endpoint URLs, or transport error text.
    pub fn request_flush(&self) {
        let Some(sender) = &self.requests else {
            self.shared.drop_pending();
            return;
        };
        match sender.try_send(FlushRequest::Background) {
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => self.shared.drop_pending(),
        }
    }

    /// Request a flush and wait for its outcome, for explicit administrative use.
    ///
    /// This never creates a second transport worker. A busy signal queue
    /// returns an error immediately; otherwise acknowledgment is bounded to
    /// 12 seconds. A wait timeout does not retry the submitted batch, which
    /// may still complete on the worker. Generation uses `request_flush`
    /// instead. Each attempted batch is dropped on failure; counts recorded
    /// during its I/O remain independent pending counters.
    pub fn flush(&self) -> HarnessResult<()> {
        let Some(sender) = &self.requests else {
            self.shared.drop_pending();
            return Err(FlushFailure::Worker.error());
        };
        let (reply, result) = mpsc::sync_channel(1);
        match sender.try_send(FlushRequest::Wait(reply)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(HarnessError::Telemetry(
                    "flush already pending; counters remain in memory".to_owned(),
                ));
            }
            Err(TrySendError::Disconnected(_)) => {
                self.shared.drop_pending();
                return Err(FlushFailure::Worker.error());
            }
        }
        match result.recv_timeout(self.wait_timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(HarnessError::Telemetry(
                "flush acknowledgment timed out; submitted batch will not be retried".to_owned(),
            )),
            Err(RecvTimeoutError::Disconnected) => Err(FlushFailure::Worker.error()),
        }
    }
}

impl Drop for TelemetryClient {
    fn drop(&mut self) {
        self.shared.stopped.store(true, Ordering::Release);
        self.requests.take();
        // Do not wait for a network call (or an injected transport). Normal
        // workers wake on sender disconnect; an in-flight worker exits when
        // its one call returns. No new worker or retry is created on shutdown.
        if let Some(worker) = self.worker.take()
            && worker.is_finished()
        {
            let _ = worker.join();
        }
    }
}

fn flush_worker(
    shared: Arc<Shared>,
    requests: Receiver<FlushRequest>,
    endpoint: String,
    transport: Box<dyn TelemetryTransport>,
    interval: Duration,
    diagnostic: Diagnostic,
) {
    let mut next_flush = Instant::now() + interval;
    loop {
        let request =
            match requests.recv_timeout(next_flush.saturating_duration_since(Instant::now())) {
                Ok(request) => Some(request),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            };
        if shared.stopped.load(Ordering::Acquire) {
            break;
        }
        // Schedule from the attempt's start, so transport latency does not
        // repeatedly push the idle deadline beyond 24 hours.
        next_flush = Instant::now() + interval;
        let result = flush_batch(&shared, transport.as_ref(), &endpoint);
        if let Err(failure) = result {
            diagnostic(failure.message());
        }
        if let Some(FlushRequest::Wait(reply)) = request {
            let _ = reply.try_send(result.map_err(FlushFailure::error));
        }
    }
}

fn flush_batch(
    shared: &Shared,
    transport: &dyn TelemetryTransport,
    endpoint: &str,
) -> Result<(), FlushFailure> {
    let Some(payload) = shared.take_payload()? else {
        return Ok(());
    };
    let body = serde_json::to_string(&payload).map_err(|_| FlushFailure::Payload)?;
    transport
        .post_json(endpoint, &body)
        .map_err(|_| FlushFailure::Transport)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;

    const TEST_WAIT: Duration = Duration::from_secs(5);
    type Post = (thread::ThreadId, TelemetryPayload);

    struct ReleaseOnDrop(Option<SyncSender<()>>);

    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.try_send(());
            }
        }
    }

    struct GatedTransport {
        posts: mpsc::Sender<Post>,
        release: Mutex<Receiver<()>>,
        calls: Arc<AtomicUsize>,
        dropped: SyncSender<()>,
        fail_first: bool,
    }

    impl TelemetryTransport for GatedTransport {
        fn post_json(&self, _endpoint: &str, body: &str) -> Result<(), String> {
            let number = self.calls.fetch_add(1, Ordering::SeqCst);
            self.posts
                .send((thread::current().id(), serde_json::from_str(body).unwrap()))
                .map_err(|_| "test receiver closed".to_owned())?;
            if number == 0 {
                self.release
                    .lock()
                    .unwrap()
                    .recv()
                    .map_err(|_| "test gate closed".to_owned())?;
                if self.fail_first {
                    return Err(
                        "PRIVATE_PAYLOAD https://user:PRIVATE_TOKEN@example.test/private"
                            .to_owned(),
                    );
                }
            }
            Ok(())
        }
    }

    impl Drop for GatedTransport {
        fn drop(&mut self) {
            let _ = self.dropped.try_send(());
        }
    }

    fn gated_transport(
        fail_first: bool,
    ) -> (
        Box<dyn TelemetryTransport>,
        Receiver<Post>,
        ReleaseOnDrop,
        Arc<AtomicUsize>,
        Receiver<()>,
    ) {
        let (posts, received) = mpsc::channel();
        let (release, wait) = mpsc::sync_channel(1);
        let (dropped, destroyed) = mpsc::sync_channel(1);
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Box::new(GatedTransport {
                posts,
                release: Mutex::new(wait),
                calls: Arc::clone(&calls),
                dropped,
                fail_first,
            }),
            received,
            ReleaseOnDrop(Some(release)),
            calls,
            destroyed,
        )
    }

    fn options() -> WorkerOptions {
        WorkerOptions {
            diagnostic: Arc::new(|_| {}),
            ..WorkerOptions::default()
        }
    }

    struct NotifyTransport(mpsc::Sender<TelemetryPayload>);

    impl TelemetryTransport for NotifyTransport {
        fn post_json(&self, _endpoint: &str, body: &str) -> Result<(), String> {
            self.0
                .send(serde_json::from_str(body).unwrap())
                .map_err(|_| "test receiver closed".to_owned())
        }
    }

    #[test]
    fn telemetry_idle_timer_flushes_session_counts_without_a_turn() {
        let (sent, received) = mpsc::channel();
        let client = TelemetryClient::with_options(
            "0.1.0".to_owned(),
            HardwareTier::Ram36Gb,
            "https://telemetry.example.test/v1".to_owned(),
            Box::new(NotifyTransport(sent)),
            WorkerOptions {
                interval: Duration::from_millis(20),
                ..options()
            },
        );
        // No explicit flush or completed turn wakes the worker.
        client.record_session();
        let payload = received
            .recv_timeout(TEST_WAIT)
            .expect("scheduled idle flush");
        assert_eq!(payload.session_count, 1);
        assert_eq!(payload.turn_count, 0);
        assert_eq!(client.snapshot().unwrap().session_count, 0);
        // Empty timer ticks do not create telemetry traffic.
        assert!(matches!(
            received.recv_timeout(Duration::from_millis(80)),
            Err(RecvTimeoutError::Timeout)
        ));
    }

    #[test]
    fn telemetry_coalesces_requests_and_preserves_counts_during_failed_post() {
        let (transport, posts, release, calls, dropped) = gated_transport(true);
        let (logged, diagnostics) = mpsc::channel();
        let client = TelemetryClient::with_options(
            "0.1.0".to_owned(),
            HardwareTier::Ram36Gb,
            "https://user:PRIVATE_TOKEN@example.test/private".to_owned(),
            transport,
            WorkerOptions {
                diagnostic: Arc::new(move |message| {
                    let _ = logged.send(message);
                }),
                ..options()
            },
        );
        client.record_session();
        client.record_turn();
        client.request_flush();
        let (first_worker, first) = posts.recv_timeout(TEST_WAIT).expect("first post blocked");
        assert_eq!((first.session_count, first.turn_count), (1, 1));

        client.record_session();
        for _ in 0..1_000 {
            client.record_turn();
            client.request_flush();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "no parallel transport");
        assert_eq!(client.snapshot().unwrap().turn_count, 1_000);
        drop(release);
        let (second_worker, second) = posts.recv_timeout(TEST_WAIT).expect("one next batch");
        assert_eq!(
            first_worker, second_worker,
            "one owned worker, not a per-turn task"
        );
        assert_eq!((second.session_count, second.turn_count), (1, 1_000));
        let diagnostic = diagnostics.recv_timeout(TEST_WAIT).expect("failure logged");
        assert_eq!(diagnostic, "HTTPS flush failed (counts dropped)");
        assert!(!diagnostic.contains("PRIVATE"));
        // Wait for completion without asking the transport to post empty data.
        client.flush().expect("worker settled");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "failed batch was never retried"
        );
        assert_eq!(client.snapshot().unwrap().turn_count, 0);
        drop(client);
        dropped
            .recv_timeout(TEST_WAIT)
            .expect("worker owns and drops transport");
    }

    #[test]
    fn telemetry_explicit_wait_and_drop_do_not_wait_for_a_blocked_transport() {
        let (transport, posts, release, calls, dropped) = gated_transport(false);
        let client = TelemetryClient::with_options(
            "0.1.0".to_owned(),
            HardwareTier::Ram36Gb,
            "https://telemetry.example.test/v1".to_owned(),
            transport,
            WorkerOptions {
                wait_timeout: Duration::from_millis(40),
                ..options()
            },
        );
        client.record_turn();
        client.request_flush();
        posts
            .recv_timeout(TEST_WAIT)
            .expect("transport entered its gate before the bounded wait starts");
        // The explicit flush is queued behind an already blocked POST.
        client.record_turn();
        let (done, returned) = mpsc::sync_channel(1);
        let caller = thread::spawn(move || {
            let result = client.flush();
            // The channel is sent only after Drop has also returned.
            drop(client);
            let _ = done.send(result);
        });
        let result = returned
            .recv_timeout(TEST_WAIT)
            .expect("bounded flush and client drop");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("acknowledgment timed out")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(release);
        dropped
            .recv_timeout(TEST_WAIT)
            .expect("in-flight worker exits after its call");
        caller.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "no shutdown retry");
    }

    struct SecretFailure;

    impl TelemetryTransport for SecretFailure {
        fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
            Err("PRIVATE_PAYLOAD https://user:PRIVATE_TOKEN@example.test/private".to_owned())
        }
    }

    #[test]
    fn telemetry_explicit_error_and_diagnostic_are_redacted() {
        let (logged, diagnostics) = mpsc::channel();
        let client = TelemetryClient::with_options(
            "0.1.0".to_owned(),
            HardwareTier::Ram36Gb,
            "https://user:PRIVATE_TOKEN@example.test/private".to_owned(),
            Box::new(SecretFailure),
            WorkerOptions {
                diagnostic: Arc::new(move |message| {
                    let _ = logged.send(message);
                }),
                ..options()
            },
        );
        client.record_turn();
        assert_eq!(
            client.flush().unwrap_err().to_string(),
            "telemetry: HTTPS flush failed (counts dropped)"
        );
        assert_eq!(
            diagnostics.recv_timeout(TEST_WAIT).unwrap(),
            "HTTPS flush failed (counts dropped)"
        );
        assert_eq!(client.snapshot().unwrap().turn_count, 0);
    }

    #[test]
    fn telemetry_https_transport_times_out_a_stalled_tls_peer() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("loopback listener");
        let endpoint = format!("https://{}/v1", listener.local_addr().unwrap());
        let (accepted, connected) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        let release = ReleaseOnDrop(Some(release));
        let peer = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept telemetry connection");
            let _ = accepted.send(());
            // Keep TCP open but never complete TLS. The client must time out
            // while this gate remains closed, not merely notice a closed peer.
            let _ = wait.recv_timeout(TEST_WAIT);
            drop(stream);
        });
        let (finished, result) = mpsc::sync_channel(1);
        let caller = thread::spawn(move || {
            let response =
                post_json_with_agent(&https_agent(Duration::from_millis(100)), &endpoint, "{}");
            let _ = finished.send(response);
        });
        connected
            .recv_timeout(TEST_WAIT)
            .expect("real TLS attempt connected");
        let error = result
            .recv_timeout(Duration::from_secs(2))
            .expect("TLS must time out before peer releases its gate")
            .expect_err("stalled TLS handshake must fail");
        assert!(error.starts_with("HTTPS request failed"));
        assert!(!error.contains("127.0.0.1"));
        drop(release);
        caller.join().unwrap();
        peer.join().unwrap();
    }

    #[test]
    fn telemetry_https_transport_rejects_cleartext() {
        let error = post_json_with_agent(
            &https_agent(Duration::from_millis(100)),
            "http://telemetry.example.test/v1",
            "{}",
        )
        .expect_err("HTTPS-only before any DNS or connection");
        assert!(error.contains("InsecureRequestHttpsOnly"), "{error}");
    }
}
