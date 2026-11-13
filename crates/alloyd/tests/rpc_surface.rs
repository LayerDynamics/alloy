//! Named acceptance: `rpc_surface_v1`, `stream_events_order`.

use alloyd::{
    API_VERSION, DaemonService, ErrorKind, FixtureEngine, RpcRequest, bind_listener,
    handle_connection,
};
use engine::api::{Engine, EngineFuture};
use engine::registry::ModelProfile;
use engine::types::{
    ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, Generation, GenerationEvent,
    GenerationEventStream, GenerationOptions, GenerationRequest, GenerationStream, MemoryEstimate,
    ModelProfileId, ReleaseOutcome, RenderedContext, RequestId, SessionStateId,
};
use harness_core::{HardwareTier, TelemetryClient};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

#[derive(Default)]
struct StreamGateState {
    entered_profiles: BTreeSet<String>,
    released: bool,
}

#[derive(Default)]
struct StreamGate {
    state: Mutex<StreamGateState>,
    changed: Condvar,
}

impl StreamGate {
    fn release_on_drop(self: &Arc<Self>) -> ReleaseStreamGate {
        ReleaseStreamGate(Arc::clone(self))
    }

    fn wait_entered(&self, profiles: &[&str]) {
        let (state, _) = self
            .changed
            .wait_timeout_while(
                self.state.lock().expect("gate lock"),
                Duration::from_secs(5),
                |state| {
                    profiles
                        .iter()
                        .any(|profile| !state.entered_profiles.contains(*profile))
                },
            )
            .expect("wait for first streamed deltas");
        let entered = state.entered_profiles.clone();
        drop(state);
        assert!(
            profiles.iter().all(|profile| entered.contains(*profile)),
            "streams did not reach the gate after their first delta: expected {profiles:?}, got {entered:?}"
        );
    }

    fn hold_after_delta(&self, profile: &str) {
        let mut state = self.state.lock().expect("gate lock");
        state.entered_profiles.insert(profile.to_owned());
        self.changed.notify_all();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(5), |state| !state.released)
            .expect("wait for gate release");
        let released = state.released;
        drop(state);
        assert!(released, "stream gate was not released for {profile}");
    }
}

struct ReleaseStreamGate(Arc<StreamGate>);

impl Drop for ReleaseStreamGate {
    fn drop(&mut self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .released = true;
        self.0.changed.notify_all();
    }
}

/// Keep the real fixture cancellation behavior, but stop after its first delta
/// until the test has issued pause. No startup/completion timing is assumed.
struct GatedFixtureEngine {
    inner: FixtureEngine,
    gate: Arc<StreamGate>,
}

impl GatedFixtureEngine {
    fn new(reply: &str, cancelled: Arc<AtomicBool>, gate: Arc<StreamGate>) -> Self {
        Self {
            inner: FixtureEngine::with_cancel(reply, cancelled, 0),
            gate,
        }
    }
}

impl Engine for GatedFixtureEngine {
    fn profile_context_budget(
        &self,
        _: &engine::types::ModelProfileId,
    ) -> Result<Option<engine::api::ProfileContextBudget>, engine::error::EngineError> {
        Ok(Some(engine::api::ProfileContextBudget {
            limit: 1_000_000,
            output_reserve: 512,
            safety_margin: 64,
        }))
    }

    fn api_version(&self) -> u32 {
        self.inner.api_version()
    }

    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext> {
        self.inner.render_and_count(context)
    }

    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        self.inner.estimate_memory(profile, input_tokens)
    }

    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let profile = request.profile_id().as_str().to_owned();
        let gate = Arc::clone(&self.gate);
        let generated = self.inner.generate(request);
        Box::pin(async move {
            let generation = generated.await?;
            Ok(Generation::from_stream(
                generation.request_id().clone(),
                Box::pin(GatedStream {
                    inner: generation.into_events(),
                    gate,
                    profile,
                    emitted_delta: false,
                    passed_gate: false,
                }),
            ))
        })
    }

    fn cancel(&self, request_id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        self.inner.cancel(request_id)
    }

    fn release_session_state(&self, state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        self.inner.release_session_state(state_id)
    }

    fn capabilities(&self, artifact_id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        self.inner.capabilities(artifact_id)
    }
}

// Exercise RPC propagation using the same profile arithmetic as runtime admission.
struct TurnBoundaryEngine {
    inner: FixtureEngine,
    profile: ModelProfile,
}

impl Engine for TurnBoundaryEngine {
    fn profile_context_budget(
        &self,
        _: &ModelProfileId,
    ) -> Result<Option<engine::api::ProfileContextBudget>, engine::error::EngineError> {
        Ok(Some(engine::api::ProfileContextBudget {
            limit: self.profile.context_limit_tokens(),
            output_reserve: self.profile.max_output_tokens(),
            safety_margin: self.profile.safety_margin_tokens(),
        }))
    }
    fn api_version(&self) -> u32 {
        self.inner.api_version()
    }
    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext> {
        self.inner.render_and_count(context)
    }
    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        self.inner.estimate_memory(profile, input_tokens)
    }
    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let count = request
            .expected_render()
            .expect("supervisor binds counted render")
            .token_count();
        if let Err(error) = self.profile.check_context(count) {
            return Box::pin(async move { Err(error) });
        }
        self.inner.generate(request)
    }
    fn cancel(&self, id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        self.inner.cancel(id)
    }
    fn release_session_state(&self, id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        self.inner.release_session_state(id)
    }
    fn capabilities(&self, id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        self.inner.capabilities(id)
    }
}

struct GatedStream {
    inner: GenerationStream,
    gate: Arc<StreamGate>,
    profile: String,
    emitted_delta: bool,
    passed_gate: bool,
}

impl GenerationEventStream for GatedStream {
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        let this = self.get_mut();
        if this.emitted_delta && !this.passed_gate {
            // Supervisor writes and flushes the previous notification before
            // polling again. The request is registered and cannot finish here.
            this.gate.hold_after_delta(&this.profile);
            this.passed_gate = true;
        }
        let next = this.inner.as_mut().poll_next(cx);
        if matches!(&next, Poll::Ready(Some(GenerationEvent::TextDelta { .. }))) {
            this.emitted_delta = true;
        }
        next
    }
}

/// Own all fixture threads so a restart releases the first daemon's store.
struct TestServer {
    socket: PathBuf,
    stopping: Arc<AtomicBool>,
    accept: Option<thread::JoinHandle<()>>,
}

impl TestServer {
    fn stop(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        if let Some(accept) = self.accept.take() {
            self.stopping
                .store(true, std::sync::atomic::Ordering::Release);
            // Wake blocking accept. The loop checks stopping before spawning.
            let _ = UnixStream::connect(&self.socket);
            let result = accept.join();
            assert!(
                result.is_ok() || thread::panicking(),
                "test server thread panicked during shutdown"
            );
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn start_server_with_engine<E: Engine + 'static>(
    dir: &Path,
    socket_name: &str,
    engine: E,
    models_resident: u64,
    memory_reserved: u64,
) -> (PathBuf, PathBuf, TestServer) {
    let socket = dir.join(socket_name);
    let db = dir.join("sessions.sqlite");
    let telemetry = Arc::new(TelemetryClient::new(
        "0.1.0-test",
        HardwareTier::Ram36Gb,
        "https://127.0.0.1:1/drop",
    ));
    let service = Arc::new(
        DaemonService::open(&db, engine, telemetry, 32, models_resident, memory_reserved)
            .expect("service"),
    );
    let listener = bind_listener(&socket).expect("bind");
    let mode = std::fs::metadata(&socket)
        .expect("meta")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "socket must be mode 0600");
    let stopping = Arc::new(AtomicBool::new(false));
    let thread_stopping = Arc::clone(&stopping);
    let accept = thread::spawn(move || {
        let mut workers = Vec::new();
        let mut connections = Vec::new();
        for connection in listener.incoming() {
            let stream = connection.expect("accept");
            if thread_stopping.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("server read timeout");
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .expect("server write timeout");
            connections.push(stream.try_clone().expect("shutdown handle"));
            let service = Arc::clone(&service);
            workers.push(thread::spawn(move || {
                let _ = handle_connection(stream, service.as_ref());
            }));
        }
        // Teardown must also unblock readers on persistent connections.
        for connection in connections {
            let _ = connection.shutdown(std::net::Shutdown::Both);
        }
        let mut panicked = false;
        for worker in workers {
            panicked |= worker.join().is_err();
        }
        assert!(!panicked, "test connection worker panicked");
        // The last service owner drops as this thread exits, before join returns.
    });
    let server = TestServer {
        socket: socket.clone(),
        stopping,
        accept: Some(accept),
    };
    (socket, db, server)
}

fn start_server(dir: &Path) -> (PathBuf, PathBuf, TestServer) {
    start_server_with_engine(
        dir,
        "alloyd.sock",
        FixtureEngine::new("one two three"),
        0,
        0,
    )
}

fn call(socket: &std::path::Path, request: Value) -> Vec<Value> {
    let mut stream = UnixStream::connect(socket).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("read timeout");
    let line = format!("{request}\n");
    stream.write_all(line.as_bytes()).expect("write");
    stream.flush().expect("flush");

    let mut reader = BufReader::new(stream);
    let mut out = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let read = reader.read_line(&mut buf).expect("read");
        if read == 0 {
            break;
        }
        if buf.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(buf.trim()).expect("json");
        let is_final = value.get("id").is_some();
        out.push(value);
        if is_final {
            break;
        }
    }
    out
}

#[test]
fn rpc_skill_load_updates_session_and_persists_exact_content() {
    let dir = tempdir().unwrap();
    let (socket, _, server) = start_server(dir.path());
    let created = call(
        &socket,
        json!({"jsonrpc":"2.0","id":1,"api_version":API_VERSION,"method":"session.create","params":{"profile":"fixture-profile"}}),
    );
    let session = created[0]["result"]["session_id"].as_str().unwrap();
    for (id, text) in [(2, "first instructions"), (3, "updated instructions")] {
        let result = call(
            &socket,
            json!({"jsonrpc":"2.0","id":id,"api_version":API_VERSION,"method":"session.skill.load","params":{"session_id":session,"name":"audit","instructions":text}}),
        );
        assert_eq!(result[0]["result"]["name"], "audit", "{result:?}");
    }
    let sent = call(
        &socket,
        json!({"jsonrpc":"2.0","id":4,"api_version":API_VERSION,"method":"session.send","params":{"session_id":session,"content":"hello"}}),
    );
    assert_eq!(sent.last().unwrap()["result"]["ok"], true, "{sent:?}");
    let capabilities = call(
        &socket,
        json!({"jsonrpc":"2.0","id":6,"api_version":API_VERSION,"method":"models.capabilities","params":{"artifact_id":"alloyd-fixture"}}),
    );
    assert_eq!(
        capabilities[0]["result"]["supports_prefix_reuse"], false,
        "{capabilities:?}"
    );
    let history = call(
        &socket,
        json!({"jsonrpc":"2.0","id":5,"api_version":API_VERSION,"method":"sessions.resume","params":{"session_id":session}}),
    );
    let events = history[0]["result"]["events"].as_array().unwrap();
    let skills: Vec<_> = events
        .iter()
        .filter(|e| e["kind"] == "skill.loaded")
        .collect();
    assert_eq!(skills.len(), 2);
    assert_eq!(skills[1]["payload"]["instructions"], "updated instructions");
    server.stop();
}

#[test]
fn rpc_turn_context_boundary_reserves_output_and_margin() {
    let dir = tempdir().unwrap();
    let profile = ModelProfile::new(
        ModelProfileId::new("fixture-profile").unwrap(),
        ArtifactId::new("alloyd-fixture").unwrap(),
        35,
        32,
        1,
        GenerationOptions::greedy(32).unwrap(),
    )
    .unwrap();
    let (socket, _, server) = start_server_with_engine(
        dir.path(),
        "boundary.sock",
        TurnBoundaryEngine {
            inner: FixtureEngine::new("ok"),
            profile,
        },
        0,
        0,
    );
    for (id, content, fits) in [(1, "hello world", true), (2, "hello world test", false)] {
        let created = call(
            &socket,
            json!({"jsonrpc":"2.0", "id":id, "api_version":API_VERSION,
            "method":"session.create", "params":{"profile":"fixture-profile"}}),
        );
        let session = created[0]["result"]["session_id"].as_str().unwrap();
        let response = call(
            &socket,
            json!({"jsonrpc":"2.0", "id":id, "api_version":API_VERSION,
            "method":"session.send", "params":{"session_id":session,"content":content}}),
        );
        let final_response = response.last().unwrap();
        if fits {
            assert_eq!(final_response["result"]["ok"], true, "{response:?}");
        } else {
            assert!(
                final_response["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("start a new session or select a profile with a larger context"),
                "{response:?}"
            );
            assert!(
                !response
                    .iter()
                    .any(|line| line["params"]["event"] == "text_delta"),
                "{response:?}"
            );
        }
    }
    server.stop();
}

#[test]
fn rpc_surface_v1() {
    let dir = tempdir().expect("tempdir");
    let (socket, db, server) = start_server(dir.path());

    // api_version negotiation — missing version.
    let missing = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "daemon.status",
            "params": {}
        }),
    );
    assert_eq!(missing.len(), 1);
    let err = &missing[0]["error"];
    assert_eq!(err["code"], -32000);
    assert_eq!(
        err["data"]["kind"],
        ErrorKind::UnsupportedApiVersion.as_str()
    );

    // Wrong api_version.
    let wrong = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "api_version": 99,
            "method": "daemon.status",
            "params": {}
        }),
    );
    assert_eq!(
        wrong[0]["error"]["data"]["kind"],
        ErrorKind::UnsupportedApiVersion.as_str()
    );

    // Method not found.
    let unknown = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "api_version": API_VERSION,
            "method": "nope.missing",
            "params": {}
        }),
    );
    assert_eq!(unknown[0]["error"]["code"], -32601);

    // Phase-1 unsupported B.6 method.
    let unsupported = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "api_version": API_VERSION,
            "method": "skills.list",
            "params": {}
        }),
    );
    assert_eq!(
        unsupported[0]["error"]["data"]["kind"],
        ErrorKind::UnsupportedInPhase1.as_str()
    );

    // Invalid params.
    let bad_params = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "api_version": API_VERSION,
            "method": "session.create",
            "params": {}
        }),
    );
    assert_eq!(bad_params[0]["error"]["code"], -32602);

    // session.create
    let created = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "api_version": API_VERSION,
            "method": "session.create",
            "params": { "profile": "fixture-profile" }
        }),
    );
    assert!(created[0]["error"].is_null() || created[0].get("error").is_none());
    let session_id = created[0]["result"]["session_id"].as_str().expect("id");
    assert!(session_id.starts_with("sess-"));

    // session.send streams text_delta then generation_completed, then result.
    let streamed = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 7,
            "api_version": API_VERSION,
            "method": "session.send",
            "params": { "session_id": session_id, "content": "hello world" }
        }),
    );
    assert!(
        streamed.len() >= 3,
        "expected notifications + final response, got {streamed:?}"
    );
    let events: Vec<&str> = streamed
        .iter()
        .filter(|line| line.get("method") == Some(&json!("session.event")))
        .map(|line| line["params"]["event"].as_str().expect("event"))
        .collect();
    assert_eq!(events.first().copied(), Some("text_delta"));
    assert_eq!(events.last().copied(), Some("generation_completed"));
    let completed = streamed
        .iter()
        .find(|line| {
            line.get("method") == Some(&json!("session.event"))
                && line["params"]["event"] == "generation_completed"
        })
        .expect("generation_completed notify");
    assert_eq!(
        completed["params"]["body"]["finish_reason"],
        "end_of_sequence"
    );
    let final_response = streamed.last().expect("final");
    assert_eq!(final_response["id"], 7);
    assert_eq!(final_response["result"]["ok"], true);
    assert_eq!(final_response["result"]["assistant_text"], "one two three");

    let status_active = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": "status-1",
            "api_version": API_VERSION,
            "method": "daemon.status",
            "params": {}
        }),
    );
    assert_eq!(status_active[0]["result"]["sessions_active"], 1);

    // resume of already-active must fail (paused→active only).
    let resume_active = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": "resume-active",
            "api_version": API_VERSION,
            "method": "session.resume",
            "params": { "session_id": session_id }
        }),
    );
    assert!(
        resume_active[0].get("error").is_some(),
        "resume of active session must error: {resume_active:?}"
    );

    // session.pause / session.resume
    let paused = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 8,
            "api_version": API_VERSION,
            "method": "session.pause",
            "params": { "session_id": session_id }
        }),
    );
    assert_eq!(paused[0]["result"]["status"], "paused");

    let send_while_paused = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 9,
            "api_version": API_VERSION,
            "method": "session.send",
            "params": { "session_id": session_id, "content": "nope" }
        }),
    );
    assert_eq!(
        send_while_paused[0]["error"]["data"]["kind"],
        ErrorKind::SessionNotActive.as_str()
    );

    let resumed = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 10,
            "api_version": API_VERSION,
            "method": "session.resume",
            "params": { "session_id": session_id }
        }),
    );
    assert_eq!(resumed[0]["result"]["status"], "active");

    let resume_again = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": "resume-again",
            "api_version": API_VERSION,
            "method": "session.resume",
            "params": { "session_id": session_id }
        }),
    );
    assert!(
        resume_again[0].get("error").is_some(),
        "second resume while active must error: {resume_again:?}"
    );

    // sessions.resume after reopening the same DB (W5 store/daemon reopen stand-in).
    let old_n: u64 = session_id
        .strip_prefix("sess-")
        .expect("sess prefix")
        .parse()
        .expect("numeric id");
    assert!(
        harness_core::SessionStore::open(&db).is_err(),
        "first live daemon must retain exclusive ownership"
    );
    server.stop();
    drop(socket);
    let (socket2, _, _server2) = start_server_with_engine(
        dir.path(),
        "alloyd-reopen.sock",
        FixtureEngine::new("ignored"),
        1,
        8 * 1024 * 1024 * 1024,
    );

    let historical = call(
        &socket2,
        json!({
            "jsonrpc": "2.0",
            "id": 11,
            "api_version": API_VERSION,
            "method": "sessions.resume",
            "params": { "session_id": session_id }
        }),
    );
    assert_eq!(historical[0]["result"]["session_id"], session_id);
    assert!(historical[0]["result"]["event_count"].as_u64().unwrap() >= 3);

    let created_after = call(
        &socket2,
        json!({
            "jsonrpc": "2.0",
            "id": 12,
            "api_version": API_VERSION,
            "method": "session.create",
            "params": { "profile": "fixture-profile" }
        }),
    );
    let new_id = created_after[0]["result"]["session_id"]
        .as_str()
        .expect("new id");
    let new_n: u64 = new_id
        .strip_prefix("sess-")
        .expect("prefix")
        .parse()
        .expect("n");
    assert!(
        new_n > old_n,
        "reopened daemon must not reuse session ids ({new_id} vs {session_id})"
    );

    let status = call(
        &socket2,
        json!({
            "jsonrpc": "2.0",
            "id": 13,
            "api_version": API_VERSION,
            "method": "daemon.status",
            "params": {}
        }),
    );
    assert_eq!(status[0]["result"]["api_version"], API_VERSION);
    assert_eq!(status[0]["result"]["models_resident"], 1);
    assert_eq!(
        status[0]["result"]["memory_reserved"],
        8 * 1024 * 1024 * 1024u64
    );
}

#[test]
fn stream_events_order() {
    let dir = tempdir().expect("tempdir");
    let cancel = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(StreamGate::default());
    let release_gate = gate.release_on_drop();
    let (socket, _, _server) = start_server_with_engine(
        dir.path(),
        "stream.sock",
        GatedFixtureEngine::new(
            "alpha beta gamma delta",
            Arc::clone(&cancel),
            Arc::clone(&gate),
        ),
        0,
        0,
    );

    let created = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "api_version": API_VERSION,
            "method": "session.create",
            "params": { "profile": "fixture-profile" }
        }),
    );
    let session_id = created[0]["result"]["session_id"]
        .as_str()
        .expect("id")
        .to_owned();

    let socket_send = socket.clone();
    let session_for_send = session_id.clone();
    let send_handle = thread::spawn(move || {
        call(
            &socket_send,
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "api_version": API_VERSION,
                "method": "session.send",
                "params": { "session_id": session_for_send, "content": "hello" }
            }),
        )
    });

    gate.wait_entered(&["fixture-profile"]);
    assert!(!cancel.load(std::sync::atomic::Ordering::SeqCst));
    let paused = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "api_version": API_VERSION,
            "method": "session.pause",
            "params": { "session_id": session_id }
        }),
    );
    assert_eq!(paused[0]["result"]["status"], "paused");
    assert!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    assert!(
        !send_handle.is_finished(),
        "cancelled send must remain held until the stream gate is released"
    );
    drop(release_gate);

    let streamed = send_handle.join().expect("join send");
    let events: Vec<&str> = streamed
        .iter()
        .filter(|line| line.get("method") == Some(&json!("session.event")))
        .map(|line| line["params"]["event"].as_str().expect("event"))
        .collect();
    assert!(
        !events.is_empty(),
        "expected at least one streamed event before cancel: {streamed:?}"
    );
    assert_eq!(events.first().copied(), Some("text_delta"));
    assert!(
        events.contains(&"request_failed"),
        "pause-cancel must stream session.event request_failed: {streamed:?}"
    );
    assert_eq!(
        events
            .iter()
            .copied()
            .filter(|event| matches!(*event, "request_failed" | "generation_completed"))
            .collect::<Vec<_>>(),
        vec!["request_failed"],
        "cancelled request must have exactly one terminal event: {streamed:?}"
    );
    let pos = events
        .iter()
        .position(|event| *event == "request_failed")
        .expect("request_failed index");
    assert!(
        events[..pos].iter().all(|event| *event == "text_delta"),
        "only text_delta may precede request_failed: {events:?}"
    );
    assert!(
        streamed.last().and_then(|line| line.get("error")).is_some(),
        "cancelled send must end with an RPC error: {streamed:?}"
    );
}

#[test]
fn pause_cancels_only_target_session() {
    let dir = tempdir().expect("tempdir");
    let cancel_a = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(StreamGate::default());
    let release_gate = gate.release_on_drop();
    let (socket, _, _server) = start_server_with_engine(
        dir.path(),
        "multi.sock",
        GatedFixtureEngine::new(
            "alpha beta gamma delta epsilon",
            Arc::clone(&cancel_a),
            Arc::clone(&gate),
        ),
        0,
        0,
    );

    let create = |id, profile: &str| {
        call(
            &socket,
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "api_version": API_VERSION,
                "method": "session.create",
                "params": { "profile": profile }
            }),
        )[0]["result"]["session_id"]
            .as_str()
            .expect("id")
            .to_owned()
    };
    let sess_a = create(1, "fixture-a");
    let sess_b = create(2, "fixture-b");

    let socket_a = socket.clone();
    let a_id = sess_a.clone();
    let send_a = thread::spawn(move || {
        call(
            &socket_a,
            json!({
                "jsonrpc": "2.0",
                "id": 10,
                "api_version": API_VERSION,
                "method": "session.send",
                "params": { "session_id": a_id, "content": "hello" }
            }),
        )
    });
    let socket_b = socket.clone();
    let b_id = sess_b.clone();
    let send_b = thread::spawn(move || {
        call(
            &socket_b,
            json!({
                "jsonrpc": "2.0",
                "id": 11,
                "api_version": API_VERSION,
                "method": "session.send",
                "params": { "session_id": b_id, "content": "hello" }
            }),
        )
    });

    gate.wait_entered(&["fixture-a", "fixture-b"]);
    assert!(!cancel_a.load(std::sync::atomic::Ordering::SeqCst));
    let paused = call(
        &socket,
        json!({
            "jsonrpc": "2.0",
            "id": 12,
            "api_version": API_VERSION,
            "method": "session.pause",
            "params": { "session_id": sess_a }
        }),
    );
    assert_eq!(paused[0]["result"]["status"], "paused");
    assert!(cancel_a.load(std::sync::atomic::Ordering::SeqCst));
    assert!(
        !send_a.is_finished() && !send_b.is_finished(),
        "both requests must remain held until the stream gate is released"
    );
    drop(release_gate);

    let out_a = send_a.join().expect("join a");
    let out_b = send_b.join().expect("join b");

    assert!(
        out_a.last().and_then(|line| line.get("error")).is_some(),
        "paused session send must fail: {out_a:?}"
    );
    let events_a: Vec<&str> = out_a
        .iter()
        .filter(|line| line.get("method") == Some(&json!("session.event")))
        .map(|line| line["params"]["event"].as_str().expect("event"))
        .collect();
    assert!(
        !events_a.contains(&"generation_completed"),
        "paused session must not complete generation: {out_a:?}"
    );
    assert_eq!(
        events_a
            .iter()
            .copied()
            .filter(|event| matches!(*event, "request_failed" | "generation_completed"))
            .collect::<Vec<_>>(),
        vec!["request_failed"],
        "paused session must have exactly one failed terminal event: {out_a:?}"
    );

    let events_b: Vec<&str> = out_b
        .iter()
        .filter(|line| line.get("method") == Some(&json!("session.event")))
        .map(|line| line["params"]["event"].as_str().expect("event"))
        .collect();
    assert!(
        events_b.contains(&"generation_completed"),
        "untargeted session must complete: {out_b:?}"
    );
    assert_eq!(
        events_b
            .iter()
            .copied()
            .filter(|event| matches!(*event, "request_failed" | "generation_completed"))
            .collect::<Vec<_>>(),
        vec!["generation_completed"],
        "untargeted session must have exactly one successful terminal event: {out_b:?}"
    );
    assert_eq!(out_b.last().expect("b final")["result"]["ok"], true);
}

#[test]
fn live_socket_refuse() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("live.sock");
    let _listener = bind_listener(&socket).expect("first bind");
    let err = bind_listener(&socket).expect_err("second bind must refuse live socket");
    assert!(
        err.message.contains("already in use"),
        "unexpected refuse message: {}",
        err.message
    );
}

/// Writer that fails any `session.event` notify line; allows final RPC responses.
struct FailNotifyWriter {
    buf: Vec<u8>,
}

impl Write for FailNotifyWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let text = std::str::from_utf8(data).unwrap_or("");
        if text.contains("session.event") {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "notify write failed",
            ));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn notify_write_failure_surfaces_engine_failed() {
    let dir = tempdir().expect("tempdir");
    let db = dir.path().join("sessions.sqlite");
    let telemetry = Arc::new(TelemetryClient::new(
        "0.1.0-test",
        HardwareTier::Ram36Gb,
        "https://127.0.0.1:1/drop",
    ));
    let service = DaemonService::open(
        &db,
        FixtureEngine::new("one two three"),
        telemetry,
        32,
        0,
        0,
    )
    .expect("service");

    let create: RpcRequest = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "api_version": API_VERSION,
        "method": "session.create",
        "params": { "profile": "fixture-profile" }
    }))
    .expect("create req");
    let mut ok_writer = Vec::new();
    service.handle(create, &mut ok_writer).expect("create");
    let created: Value = serde_json::from_slice(
        ok_writer
            .split(|&b| b == b'\n')
            .next()
            .expect("create line"),
    )
    .expect("create json");
    let session_id = created["result"]["session_id"].as_str().expect("id");

    let send: RpcRequest = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": 2,
        "api_version": API_VERSION,
        "method": "session.send",
        "params": { "session_id": session_id, "content": "hello" }
    }))
    .expect("send req");
    let mut fail_writer = FailNotifyWriter { buf: Vec::new() };
    service
        .handle(send, &mut fail_writer)
        .expect("handle returns error response");
    let response: Value = serde_json::from_slice(&fail_writer.buf).expect("error response json");
    assert_eq!(
        response["error"]["data"]["kind"],
        ErrorKind::EngineFailed.as_str(),
        "notify write failure must map to engine_failed: {response:?}"
    );
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("stream notify write failed"),
        "message should name notify write failure: {response:?}"
    );
}
