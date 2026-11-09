//! Daemon façade over harness-core supervisor + session store.

use crate::metrics::MetricsSource;
use crate::rpc::{
    API_VERSION, ErrorKind, RpcError, RpcNotification, RpcRequest, RpcResponse, encode_line,
    validate_request, wire_event_body, wire_event_name,
};
use crate::rpc::{RpcTransportLimits, write_slab_session_event};
use engine::api::{ENGINE_API_VERSION, Engine, EngineFuture};
use engine::error::EngineError;
use engine::rendering::{
    PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions, render_and_count,
};
use engine::types::{
    ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, DecodingCapability, ExecutionBackend,
    FinishReason, Generation, GenerationEvent, GenerationEventStream, GenerationRequest,
    MemoryBreakdown, MemoryEstimate, ModelProfileId, ReleaseOutcome, RequestId, SessionStateId,
    Usage, WeightRepresentation,
};
use harness_core::{HarnessError, SessionStatus, SessionStore, Supervisor, TelemetryClient};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

/// Per-session turn slot: present while a send is in progress.
struct InflightSlot {
    cancellation: tool_registry::execution::CancellationToken,
    request_id: Option<RequestId>,
    /// Set by pause before a request id exists; on_started must cancel immediately.
    cancel_requested: bool,
}

/// Shared daemon state for one process.
pub struct DaemonService<E: Engine> {
    supervisor: Supervisor<E>,
    store: Mutex<SessionStore>,
    next_session: AtomicU64,
    max_output_tokens: usize,
    /// One slot per session_id; rejects overlapping sends on the same session.
    inflight: Mutex<HashMap<String, InflightSlot>>,
    models_resident: u64,
    memory_reserved: u64,
    metrics: Option<Arc<dyn MetricsSource>>,
    notification_limits: Option<RpcTransportLimits>,
    tools: Option<harness_core::supervisor::ToolRuntime>,
}

impl<E: Engine> DaemonService<E> {
    /// Open (or create) the session database and bind supervisor.
    pub fn open(
        db_path: impl AsRef<Path>,
        engine: E,
        telemetry: Arc<TelemetryClient>,
        max_output_tokens: usize,
        models_resident: u64,
        memory_reserved: u64,
    ) -> Result<Self, RpcError> {
        let store = SessionStore::open(db_path)
            .map_err(|error| RpcError::internal(format!("open session store: {error}")))?;
        let next = seed_session_counter(&store)?;
        Ok(Self {
            supervisor: Supervisor::new(engine, telemetry),
            store: Mutex::new(store),
            next_session: AtomicU64::new(next),
            max_output_tokens,
            inflight: Mutex::new(HashMap::new()),
            models_resident,
            memory_reserved,
            metrics: None,
            notification_limits: None,
            tools: None,
        })
    }

    /// Bind an immutable scoped catalogue and policy before publishing the service.
    pub fn with_tools(mut self, runtime: harness_core::supervisor::ToolRuntime) -> Self {
        // Tool turns must not wait forever for a client that stops reading.
        // Explicit caller transport configuration still takes precedence.
        self.notification_limits.get_or_insert_with(|| {
            RpcTransportLimits::new(2 * 1024 * 1024, std::time::Duration::from_secs(5))
                .expect("positive bounded tool transport defaults")
        });
        self.tools = Some(runtime);
        self
    }

    /// Initialize the bounded artifact store in this service's own store epoch.
    pub fn open_artifacts(
        &self,
        root: &Path,
    ) -> Result<harness_core::artifact::ArtifactStore, RpcError> {
        harness_core::artifact::ArtifactStore::open(
            &mut *self
                .store
                .lock()
                .map_err(|_| RpcError::internal("store lock poisoned"))?,
            root,
            Default::default(),
        )
        .map_err(map_harness)
    }

    /// Attach read-only runtime metrics without changing legacy status fields.
    pub fn with_metrics_source(mut self, metrics: Arc<dyn MetricsSource>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Configure explicit slab-notification byte and per-write timeout limits.
    /// This does not enable tools or relax the supervisor completion gate.
    /// Without configuration, slab-batch transport is rejected before writing.
    pub fn with_rpc_transport_limits(mut self, limits: RpcTransportLimits) -> Self {
        self.notification_limits = Some(limits);
        self
    }

    /// Configured notification policy, also applied to server socket writes.
    pub fn rpc_transport_limits(&self) -> Option<RpcTransportLimits> {
        self.notification_limits
    }

    /// Handle one JSON-RPC request, writing stream notifications to `writer` as needed.
    pub fn handle<W: Write>(&self, request: RpcRequest, writer: &mut W) -> Result<(), RpcError> {
        let id = request.id.clone().unwrap_or(Value::Null);

        let response = match self.dispatch(&request, writer) {
            Ok(result) => RpcResponse::result(id, result),
            Err(error) => RpcResponse::error(id, error),
        };
        let line = encode_line(&response)?;
        writer
            .write_all(line.as_bytes())
            .map_err(|error| RpcError::internal(format!("write response: {error}")))?;
        writer
            .flush()
            .map_err(|error| RpcError::internal(format!("flush response: {error}")))?;
        Ok(())
    }

    fn dispatch<W: Write>(&self, request: &RpcRequest, writer: &mut W) -> Result<Value, RpcError> {
        validate_request(request)?;
        match request.method.as_str() {
            "session.create" => self.session_create(&request.params),
            "session.send" => self.session_send(&request.params, writer),
            "session.pause" => self.session_pause(&request.params),
            "session.resume" => self.session_resume(&request.params),
            "sessions.resume" => self.sessions_resume(&request.params),
            "daemon.status" => self.daemon_status(),
            "daemon.metrics" => self.daemon_metrics(),
            "session.approve" => self.session_approve(&request.params),
            "models.list"
            | "models.pull"
            | "models.verify"
            | "models.capabilities"
            | "skills.list"
            | "skills.show"
            | "mcp.add"
            | "mcp.list"
            | "automation.add"
            | "automation.list"
            | "automation.pause"
            | "daemon.doctor"
            | "daemon.stats" => Err(RpcError::application(
                ErrorKind::UnsupportedInPhase1,
                format!("{} is not implemented in Phase 1", request.method),
            )),
            other => Err(RpcError::method_not_found(other)),
        }
    }

    fn session_create(&self, params: &Value) -> Result<Value, RpcError> {
        let profile = params
            .get("profile")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("profile string required"))?;
        if profile.is_empty() {
            return Err(RpcError::invalid_params("profile must be non-empty"));
        }
        let n = self.next_session.fetch_add(1, Ordering::Relaxed);
        let session_id = format!("sess-{n}");
        self.supervisor
            .create_session(&self.store, &session_id, profile)
            .map_err(map_harness)?;
        Ok(json!({ "session_id": session_id, "api_version": API_VERSION }))
    }

    fn session_send<W: Write>(&self, params: &Value, writer: &mut W) -> Result<Value, RpcError> {
        let session_id = params
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("session_id string required"))?
            .to_owned();
        let content = params
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("content string required"))?
            .to_owned();

        {
            let store = self
                .store
                .lock()
                .map_err(|_| RpcError::internal("store lock poisoned"))?;
            match store.get_session(&session_id).map_err(map_harness)? {
                None => {
                    return Err(RpcError::application(
                        ErrorKind::SessionNotFound,
                        format!("unknown session_id: {session_id}"),
                    ));
                }
                Some(session) if session.status != SessionStatus::Active => {
                    return Err(RpcError::application(
                        ErrorKind::SessionNotActive,
                        format!("session is not active: {session_id}"),
                    ));
                }
                Some(_) => {}
            }
        }

        let cancellation = tool_registry::execution::CancellationToken::default();
        {
            let mut guard = self
                .inflight
                .lock()
                .map_err(|_| RpcError::internal("inflight lock poisoned"))?;
            if guard.contains_key(&session_id) {
                return Err(RpcError::application(
                    ErrorKind::SessionNotActive,
                    format!("session already has an in-flight turn: {session_id}"),
                ));
            }
            guard.insert(
                session_id.clone(),
                InflightSlot {
                    cancellation: cancellation.clone(),
                    request_id: None,
                    cancel_requested: false,
                },
            );
        }

        let on_started = |request_id: &RequestId| {
            let mut guard = self
                .inflight
                .lock()
                .map_err(|_| HarnessError::Store("inflight lock poisoned".into()))?;
            let slot = guard
                .get_mut(&session_id)
                .ok_or_else(|| HarnessError::Engine("missing inflight slot".into()))?;
            slot.request_id = Some(request_id.clone());
            if slot.cancel_requested {
                return Err(HarnessError::Engine("turn cancelled".into()));
            }
            Ok(())
        };
        let outcome = if let Some(runtime) = &self.tools {
            self.supervisor.run_tool_turn(
                &self.store,
                runtime,
                &session_id,
                &content,
                crate::tool_config::loop_config(self.max_output_tokens),
                &cancellation,
                on_started,
                |event| {
                    write_tool_notification(
                        writer,
                        &session_id,
                        event.event,
                        event.body,
                        self.notification_limits.expect("configured tool transport"),
                    )
                },
            )
        } else {
            self.supervisor.run_turn_streaming(
                &self.store,
                &session_id,
                &content,
                self.max_output_tokens,
                |request_id| {
                    let mut guard = self
                        .inflight
                        .lock()
                        .map_err(|_| HarnessError::Store("inflight lock poisoned".into()))?;
                    let Some(slot) = guard.get_mut(&session_id) else {
                        return Err(HarnessError::Engine(format!(
                            "inflight slot missing for session {session_id}"
                        )));
                    };
                    slot.request_id = Some(request_id.clone());
                    let cancel_now = slot.cancel_requested.then(|| request_id.clone());
                    drop(guard);
                    if let Some(id) = cancel_now {
                        self.supervisor.cancel(id)?;
                    }
                    Ok(())
                },
                |event| {
                    write_generation_notification(
                        writer,
                        &session_id,
                        event,
                        self.notification_limits,
                    )
                },
            )
        };
        {
            let mut guard = self
                .inflight
                .lock()
                .map_err(|_| RpcError::internal("inflight lock poisoned"))?;
            guard.remove(&session_id);
        }
        let outcome = outcome.map_err(map_harness)?;
        Ok(json!({
            "ok": true,
            "session_id": outcome.session_id,
            "assistant_text": outcome.assistant_text,
            "input_tokens": outcome.input_tokens,
        }))
    }

    fn session_approve(&self, params: &Value) -> Result<Value, RpcError> {
        let object = params
            .as_object()
            .ok_or_else(|| RpcError::invalid_params("approval object required"))?;
        if object.len() != 3
            || !object.contains_key("session_id")
            || !object.contains_key("call_id")
            || !object.contains_key("allow")
        {
            return Err(RpcError::invalid_params(
                "exact session_id, call_id and allow required",
            ));
        }
        let session = require_session_id(params)?;
        let call = params["call_id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| RpcError::invalid_params("call_id string required"))?;
        let allow = params["allow"]
            .as_bool()
            .ok_or_else(|| RpcError::invalid_params("explicit allow boolean required"))?;
        self.store
            .lock()
            .map_err(|_| RpcError::internal("store lock poisoned"))?
            .decide_tool(session, call, allow)
            .map_err(map_harness)?;
        Ok(json!({"ok":true,"session_id":session,"call_id":call,"allow":allow}))
    }

    fn session_pause(&self, params: &Value) -> Result<Value, RpcError> {
        let session_id = require_session_id(params)?;
        let inflight_request = {
            let mut guard = self
                .inflight
                .lock()
                .map_err(|_| RpcError::internal("inflight lock poisoned"))?;
            if let Some(slot) = guard.get_mut(session_id) {
                slot.cancel_requested = true;
                slot.cancellation.cancel();
                slot.request_id.clone()
            } else {
                None
            }
        };
        self.supervisor
            .pause_session(&self.store, session_id, inflight_request.as_ref())
            .map_err(map_harness)?;
        Ok(json!({ "ok": true, "session_id": session_id, "status": "paused" }))
    }

    fn session_resume(&self, params: &Value) -> Result<Value, RpcError> {
        let session_id = require_session_id(params)?;
        self.supervisor
            .resume_session(&self.store, session_id)
            .map_err(map_harness)?;
        Ok(json!({ "ok": true, "session_id": session_id, "status": "active" }))
    }

    fn sessions_resume(&self, params: &Value) -> Result<Value, RpcError> {
        let session_id = require_session_id(params)?;
        let store = self
            .store
            .lock()
            .map_err(|_| RpcError::internal("store lock poisoned"))?;
        let session = store
            .get_session(session_id)
            .map_err(map_harness)?
            .ok_or_else(|| {
                RpcError::application(
                    ErrorKind::SessionNotFound,
                    format!("unknown session_id: {session_id}"),
                )
            })?;
        let events = store.list_events(session_id).map_err(map_harness)?;
        let history: Vec<Value> = events
            .iter()
            .map(|event| {
                json!({
                    "seq": event.seq,
                    "kind": event.kind.as_str(),
                    "payload": event.payload,
                })
            })
            .collect();
        Ok(json!({
            "session_id": session.session_id,
            "status": match session.status {
                SessionStatus::Active => "active",
                SessionStatus::Paused => "paused",
                SessionStatus::Completed => "completed",
            },
            "profile_id": session.profile_id,
            "event_count": history.len(),
            "events": history,
        }))
    }

    fn daemon_status(&self) -> Result<Value, RpcError> {
        let store = self
            .store
            .lock()
            .map_err(|_| RpcError::internal("store lock poisoned"))?;
        let sessions_active = store.count_active_sessions().map_err(map_harness)?;
        Ok(json!({
            "api_version": API_VERSION,
            "models_resident": self.models_resident,
            "memory_reserved": self.memory_reserved,
            "sessions_active": sessions_active,
        }))
    }

    fn daemon_metrics(&self) -> Result<Value, RpcError> {
        self.metrics
            .as_ref()
            .ok_or_else(|| RpcError::internal("live runtime metrics are not configured"))?
            .snapshot()
    }
}

fn seed_session_counter(store: &SessionStore) -> Result<u64, RpcError> {
    let ids = store
        .list_session_ids()
        .map_err(|error| RpcError::internal(error.to_string()))?;
    let mut max = 0u64;
    for id in ids {
        if let Some(suffix) = id.strip_prefix("sess-")
            && let Ok(value) = suffix.parse::<u64>()
        {
            max = max.max(value);
        }
    }
    Ok(max.saturating_add(1).max(1))
}

fn require_session_id(params: &Value) -> Result<&str, RpcError> {
    params
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("session_id string required"))
}

fn map_harness(error: HarnessError) -> RpcError {
    match &error {
        HarnessError::Invalid(message) if message.contains("unknown session") => {
            RpcError::application(ErrorKind::SessionNotFound, message.clone())
        }
        HarnessError::Invalid(message) if message.contains("not active") => {
            RpcError::application(ErrorKind::SessionNotActive, message.clone())
        }
        HarnessError::Engine(message) => {
            RpcError::application(ErrorKind::EngineFailed, message.clone())
        }
        HarnessError::EngineFailure(error) => RpcError::engine(error),
        HarnessError::Invalid(message) => RpcError::invalid_params(message.clone()),
        other => RpcError::internal(other.to_string()),
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Linux-safe fixture engine for protocol tests.
///
/// Supports cooperative cancel via per-request flags (plus an optional shared
/// flag for test observation).
pub struct FixtureEngine {
    tokenizer: PinnedTokenizer,
    artifact: ArtifactId,
    reply: String,
    /// Shared flag flipped by any cancel — for tests that observe pause.
    cancelled: Arc<AtomicBool>,
    /// Per-request cancel flags keyed by `RequestId` string.
    per_request: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    /// Cancels that arrived before `generate` registered a flag.
    pending_cancels: Arc<Mutex<HashSet<String>>>,
    /// Optional delay between deltas so pause/cancel tests can interleave.
    delta_delay_ms: u64,
}

impl FixtureEngine {
    /// Build a WordLevel tokenizer fixture that replies with fixed text.
    pub fn new(reply: impl Into<String>) -> Self {
        Self::with_cancel(reply, Arc::new(AtomicBool::new(false)), 0)
    }

    /// Fixture with shared cancel flag and optional per-delta delay.
    pub fn with_cancel(
        reply: impl Into<String>,
        cancelled: Arc<AtomicBool>,
        delta_delay_ms: u64,
    ) -> Self {
        let source = serde_json::to_vec(&json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "WhitespaceSplit" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1, "world": 2, "test": 3 },
                "unk_token": "[UNK]"
            }
        }))
        .expect("tokenizer json");
        Self {
            tokenizer: PinnedTokenizer::from_pinned_json(&source, digest(&source), 16 * 1024)
                .expect("pinned tokenizer"),
            artifact: ArtifactId::new("alloyd-fixture").expect("artifact"),
            reply: reply.into(),
            cancelled,
            per_request: Arc::new(Mutex::new(HashMap::new())),
            pending_cancels: Arc::new(Mutex::new(HashSet::new())),
            delta_delay_ms,
        }
    }

    /// Shared cancel flag for tests / pause coordination.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }
}

impl Engine for FixtureEngine {
    fn api_version(&self) -> u32 {
        ENGINE_API_VERSION
    }

    fn render_and_count(
        &self,
        context: ContextSnapshot,
    ) -> EngineFuture<'_, engine::types::RenderedContext> {
        let profile = context.profile_id().clone();
        let template = "{% for message in messages %}{% if loop.last %}{{ message.content }}{% endif %}{% endfor %}";
        let binding = RenderingBinding {
            profile_id: &profile,
            artifact_id: &self.artifact,
            template,
            template_sha256: digest(template.as_bytes()),
            artifact_manifest_sha256: [2; 32],
        };
        let options = RenderingOptions {
            add_generation_prompt: false,
            enable_thinking: None,
            add_special_tokens: false,
            keep_trailing_newline: true,
            trim_blocks: false,
            lstrip_blocks: false,
            reserved_spellings: &[],
        };
        let limits = RenderingLimits::new(32, 8192, 8192, 8192, 256).expect("limits");
        let result = render_and_count(&context, &binding, &self.tokenizer, &options, limits);
        Box::pin(async move { result })
    }

    fn estimate_memory(
        &self,
        _profile: ModelProfileId,
        _input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        Box::pin(async {
            MemoryEstimate::new(
                MemoryBreakdown {
                    weights_bytes: 1,
                    kv_bytes: 1,
                    temporary_bytes: 1,
                    output_bytes: 1,
                    draft_bytes: 0,
                    headroom_bytes: 1,
                },
                1_000_000,
                Vec::new(),
            )
        })
    }

    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let request_id = request.request_id().clone();
        let reply = self.reply.clone();
        let shared = Arc::clone(&self.cancelled);
        let per_request = Arc::clone(&self.per_request);
        let pending_cancels = Arc::clone(&self.pending_cancels);
        let delay = self.delta_delay_ms;
        Box::pin(async move {
            let flag = Arc::new(AtomicBool::new(false));
            let key = request_id.as_str().to_owned();
            if let Ok(mut pending) = pending_cancels.lock()
                && pending.remove(&key)
            {
                flag.store(true, Ordering::SeqCst);
            }
            if let Ok(mut map) = per_request.lock() {
                map.insert(key, Arc::clone(&flag));
            }
            let mut events = VecDeque::new();
            for chunk in split_chunks(&reply) {
                events.push_back(GenerationEvent::TextDelta { text: chunk });
            }
            events.push_back(GenerationEvent::GenerationCompleted {
                finish_reason: FinishReason::EndOfSequence,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: events.len().max(1),
                },
            });
            // `shared` is only for test observation of cancel(); streams key off `flag`.
            let _ = shared;
            Ok(Generation::from_stream(
                request_id.clone(),
                Box::pin(CancellableStream {
                    events,
                    cancelled: flag,
                    request_id,
                    delay_ms: delay,
                    emitted: 0,
                    failed_emitted: false,
                }),
            ))
        })
    }

    fn cancel(&self, request_id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        self.cancelled.store(true, Ordering::SeqCst);
        let key = request_id.as_str().to_owned();
        let mut found = false;
        if let Ok(map) = self.per_request.lock()
            && let Some(flag) = map.get(&key)
        {
            flag.store(true, Ordering::SeqCst);
            found = true;
        }
        if !found && let Ok(mut pending) = self.pending_cancels.lock() {
            pending.insert(key);
        }
        Box::pin(async { Ok(CancelOutcome::Requested) })
    }

    fn release_session_state(&self, _state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        Box::pin(async { Ok(ReleaseOutcome::AlreadyAbsent) })
    }

    fn capabilities(&self, _artifact_id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        Box::pin(async {
            Ok(Capabilities {
                task: "text-generation".to_owned(),
                family: "fixture".to_owned(),
                backend: ExecutionBackend::HostCpu,
                weight_representations: vec![WeightRepresentation::F32],
                cache_modes: Vec::new(),
                context_limit_tokens: 256,
                supports_streaming: true,
                supports_incremental_kv: false,
                supports_prefix_reuse: false,
                decoding_modes: vec![DecodingCapability::Greedy],
            })
        })
    }
}

fn split_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![".".to_owned()];
    }
    text.split_whitespace()
        .enumerate()
        .map(|(index, word)| {
            if index == 0 {
                word.to_owned()
            } else {
                format!(" {word}")
            }
        })
        .collect()
}

struct CancellableStream {
    events: VecDeque<GenerationEvent>,
    cancelled: Arc<AtomicBool>,
    request_id: RequestId,
    delay_ms: u64,
    emitted: usize,
    failed_emitted: bool,
}

impl GenerationEventStream for CancellableStream {
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        let this = self.get_mut();
        if this.delay_ms > 0 && this.emitted > 0 && !this.failed_emitted {
            std::thread::sleep(std::time::Duration::from_millis(this.delay_ms));
        }
        if this.cancelled.load(Ordering::SeqCst) {
            if this.failed_emitted {
                return Poll::Ready(None);
            }
            this.failed_emitted = true;
            this.events.clear();
            return Poll::Ready(Some(GenerationEvent::RequestFailed {
                error: EngineError::Cancelled {
                    request_id: this.request_id.as_str().to_owned(),
                },
                partial_usage: Usage {
                    input_tokens: 1,
                    output_tokens: this.emitted,
                },
            }));
        }
        let next = this.events.pop_front();
        if next.is_some() {
            this.emitted = this.emitted.saturating_add(1);
        }
        Poll::Ready(next)
    }
}

fn write_tool_notification<W: Write>(
    writer: &mut W,
    session_id: &str,
    event: &'static str,
    body: Value,
    limits: RpcTransportLimits,
) -> Result<(), HarnessError> {
    let line = encode_line(&RpcNotification::session_event(event, session_id, body))
        .map_err(|error| HarnessError::Engine(error.message))?;
    if line.len() > limits.max_notification_bytes() {
        return Err(HarnessError::Engine(
            "tool notification byte limit exceeded".into(),
        ));
    }
    writer
        .write_all(line.as_bytes())
        .and_then(|()| writer.flush())
        .map_err(|error| HarnessError::Engine(error.to_string()))
}

fn write_generation_notification<W: Write>(
    writer: &mut W,
    session_id: &str,
    event: &GenerationEvent,
    limits: Option<RpcTransportLimits>,
) -> Result<(), HarnessError> {
    if let GenerationEvent::ToolCallBatchProposed { proposals } = event {
        let limits = limits.ok_or_else(|| {
            HarnessError::Engine("slab notifications require explicit RPC transport limits".into())
        })?;
        write_slab_session_event(writer, session_id, proposals, limits).map_err(|error| {
            HarnessError::Engine(format!(
                "stream slab notification failed: {}",
                error.message,
            ))
        })?;
        return Ok(());
    }
    let notification =
        RpcNotification::session_event(wire_event_name(event), session_id, wire_event_body(event));
    let line = encode_line(&notification).map_err(|error| {
        HarnessError::Engine(format!("encode session.event: {}", error.message))
    })?;
    writer
        .write_all(line.as_bytes())
        .and_then(|_| writer.flush())
        .map_err(|error| HarnessError::Engine(format!("stream notify write failed: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod w2_notification_dispatch_tests {
    use super::*;
    use engine::{
        admission::AdmissionController,
        parser_allocation::{ParserAllocation, ParserAllocationLimits},
    };
    use std::time::Duration;
    fn batch() -> GenerationEvent {
        let limits = ParserAllocationLimits {
            carry_bytes: 32,
            call_bytes: 128,
            decoded_bytes: 128,
            depth: 8,
            keys: 16,
            nodes: 32,
            proposal_bytes: 256,
            calls: 4,
        };
        let bytes = ParserAllocation::required_bytes(limits).expect("layout");
        let admission = AdmissionController::new(bytes).expect("admission");
        let mut grant = admission.reserve_grant("parser", &[bytes]).expect("grant");
        let mut storage = ParserAllocation::from_grant(&mut grant, limits).expect("storage");
        {
            let w = storage.workspaces();
            w.decoded[0] = b'a';
            w.call[..2].copy_from_slice(b"{}");
        }
        storage
            .commit_validated_call(1, 0..1, 2, 0..2)
            .expect("validated fixture");
        GenerationEvent::ToolCallBatchProposed {
            proposals: storage.freeze(),
        }
    }
    #[test]
    fn real_notification_dispatch_requires_explicit_slab_transport_limits() {
        let mut bytes = Vec::new();
        let event = batch();
        assert!(write_generation_notification(&mut bytes, "s", &event, None).is_err());
        assert!(bytes.is_empty());
    }
    #[test]
    fn real_notification_dispatch_routes_slab_batches_to_borrowed_writer() {
        let mut bytes = Vec::new();
        let event = batch();
        let limits = RpcTransportLimits::new(4096, Duration::from_secs(1)).expect("limits");
        write_generation_notification(&mut bytes, "s", &event, Some(limits)).expect("write");
        let wire: Value = serde_json::from_slice(&bytes).expect("JSON");
        assert_eq!(wire["params"]["event"], "tool_call_batch_proposed");
        assert_eq!(wire["params"]["body"]["calls"][0]["arguments_json"], "{}");
        let tiny =
            RpcTransportLimits::new(bytes.len() - 1, Duration::from_secs(1)).expect("tiny limit");
        let mut refused = Vec::new();
        assert!(write_generation_notification(&mut refused, "s", &event, Some(tiny)).is_err());
        assert!(refused.is_empty());
    }
    #[test]
    fn real_notification_dispatch_preserves_legacy_text_wire_format() {
        let mut bytes = Vec::new();
        let event = GenerationEvent::TextDelta {
            text: "hello".into(),
        };
        write_generation_notification(&mut bytes, "s", &event, None).expect("legacy write");
        let wire: Value = serde_json::from_slice(&bytes).expect("JSON");
        assert_eq!(wire["params"]["event"], "text_delta");
        assert_eq!(wire["params"]["body"], json!({"text":"hello"}));
    }
}

#[cfg(test)]
mod w6_tool_notification_tests {
    use super::*;
    #[test]
    fn tool_notification_limit_counts_escaped_envelope_before_any_write() {
        let body = json!({"text":"é\n\""});
        let encoded = encode_line(&RpcNotification::session_event(
            "text_delta",
            "s",
            body.clone(),
        ))
        .unwrap();
        let mut bytes = Vec::new();
        let short =
            RpcTransportLimits::new(encoded.len() - 1, std::time::Duration::from_secs(1)).unwrap();
        assert!(
            write_tool_notification(&mut bytes, "s", "text_delta", body.clone(), short).is_err()
        );
        assert!(bytes.is_empty(), "rejection must not emit a partial frame");
        let exact =
            RpcTransportLimits::new(encoded.len(), std::time::Duration::from_secs(1)).unwrap();
        write_tool_notification(&mut bytes, "s", "text_delta", body, exact).unwrap();
        assert_eq!(bytes, encoded.as_bytes());
    }
}
