use harness_core::{
    SessionStore,
    artifact::{ArtifactLimits, ArtifactStore},
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::{ReceiptExecutor, http_fetch::HttpExecution},
};
use serde_json::{Value, json};
use std::{io::Read, sync::Arc, time::Duration};
use tool_registry::{
    builtins::http::{HttpDestination, HttpFetchTool, HttpScope},
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::CancellationToken,
    result::ToolResult,
};
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tool-registry/tests/fixtures/http_server.rs"
    ));
}
use fixture::{Server, response};
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    artifacts: ArtifactStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    request: ToolInvocation,
}
impl Fixture {
    fn new(server: &Server, path: &str, method: &str, approval: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let artifacts = ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        let scope = HttpScope::new(
            vec![HttpDestination::new(&server.origin, "127.0.0.1".parse().unwrap()).unwrap()],
            2,
            4096,
            8192,
            Duration::from_secs(2),
        )
        .unwrap();
        let definition = tool_registry::builtins::definitions(10000, 65536)
            .unwrap()
            .into_iter()
            .find(|d| d.id == "http.fetch")
            .unwrap();
        let mut executor = ReceiptExecutor::new();
        executor
            .register(definition, Arc::new(HttpFetchTool::new(scope)))
            .unwrap();
        let arguments = json!({"url":format!("{}{path}",server.origin),"method":method})
            .as_object()
            .unwrap()
            .clone();
        let request = ToolInvocation {
            context: ToolCallContext {
                session_id: "s".into(),
                turn_id: "t".into(),
                generation_id: "g".into(),
            },
            call: ToolCall {
                call_id: "c".into(),
                tool_id: "http.fetch".into(),
                arguments: arguments.clone(),
            },
        };
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "http.fetch".into(),
                allowed_arguments: vec![arguments],
                require_approval: approval,
                timeout_ms: 10000,
                max_output_bytes: 65536,
            }],
        })
        .unwrap();
        Self {
            dir,
            store,
            artifacts,
            executor,
            policy,
            request,
        }
    }
    fn job(&mut self, token: CancellationToken, inline: usize) -> HttpExecution {
        let admission = self
            .store
            .admit_tool(self.executor.registry(), &self.policy, self.request.clone())
            .unwrap();
        if admission == IntentAdmission::AwaitingApproval {
            self.store.decide_tool("s", "c", true).unwrap();
        } else {
            assert_eq!(admission, IntentAdmission::Ready);
        }
        let claim = self.store.claim_tool("s", "c").unwrap();
        let receipt = self
            .store
            .start_tool(claim, self.executor.registry(), &self.policy)
            .unwrap();
        self.executor
            .prepare_http_fetch(&self.store, receipt, &self.artifacts, token, inline)
            .unwrap()
    }
    fn run(&mut self, inline: usize) -> ToolResult {
        let completed = self.job(CancellationToken::default(), inline).run();
        completed.complete(&mut self.store).unwrap()
    }
    fn bytes(&self, result: &ToolResult) -> Vec<u8> {
        match result {
            ToolResult::Inline(i) => i.content["ndjson"].as_str().unwrap().as_bytes().to_vec(),
            ToolResult::Artifact(a) => {
                let mut file = self
                    .artifacts
                    .lookup(&self.store, "s", "c", &a.artifact_ref)
                    .unwrap()
                    .verify()
                    .unwrap();
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).unwrap();
                bytes
            }
        }
    }
}
fn body(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_slice(line).unwrap();
        if value["type"] == "body" {
            result.extend(serde_json::from_value::<Vec<u8>>(value["data"].clone()).unwrap());
        }
    }
    result
}
#[test]
fn allowed_get_persists_exact_binary_inline_and_artifact() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    for inline in [65536, 8] {
        let server = Server::new(|_, stream, _| response(stream, 200, "", &[0, 255, 128]));
        let mut f = Fixture::new(&server, "/ok", "GET", false);
        let result = f.run(inline);
        assert!(
            !result.is_error(),
            "inline threshold={inline}; result={result:#?}"
        );
        assert_eq!(result.call_id(), "c");
        assert_eq!(body(&f.bytes(&result)), vec![0, 255, 128]);
        assert_eq!(server.calls.lock().unwrap().len(), 1);
        assert_eq!(
            f.store.tool_state("s", "c").unwrap().as_deref(),
            Some("Completed")
        );
        assert!(f.store.claim_tool("s", "c").is_err());
    }
}
#[test]
fn head_result_has_no_body() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b""));
    let mut f = Fixture::new(&server, "/ok", "HEAD", false);
    let result = f.run(65536);
    assert!(!result.is_error());
    assert!(body(&f.bytes(&result)).is_empty());
    assert!(server.calls.lock().unwrap()[0].starts_with("HEAD "));
}
#[test]
fn required_approval_and_denial_make_zero_requests() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let mut f = Fixture::new(&server, "/ok", "GET", true);
    assert_eq!(
        f.store
            .admit_tool(f.executor.registry(), &f.policy, f.request.clone())
            .unwrap(),
        IntentAdmission::AwaitingApproval
    );
    assert!(f.store.claim_tool("s", "c").is_err());
    assert!(server.calls.lock().unwrap().is_empty());
    f.store.decide_tool("s", "c", false).unwrap();
    assert!(f.store.claim_tool("s", "c").is_err());
    assert!(server.calls.lock().unwrap().is_empty());
}
#[test]
fn approved_http_request_executes_once() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let mut f = Fixture::new(&server, "/ok", "GET", true);
    assert!(!f.run(65536).is_error());
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}
#[test]
fn changed_url_or_method_never_reaches_transport() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    for (key, value) in [
        ("url", json!("http://example.invalid/changed")),
        ("method", json!("HEAD")),
    ] {
        let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
        let mut f = Fixture::new(&server, "/ok", "GET", false);
        let mut changed = f.request.clone();
        changed.call.arguments.insert(key.into(), value);
        assert_eq!(
            f.store
                .admit_tool(f.executor.registry(), &f.policy, changed)
                .unwrap(),
            IntentAdmission::Rejected
        );
        assert!(server.calls.lock().unwrap().is_empty());
    }
}
#[test]
fn redirect_denial_is_durable_without_contacting_denied_origin() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let denied = Server::new(|_, stream, _| response(stream, 200, "", b"secret"));
    let location = denied.origin.clone();
    let server = Server::new(move |_, stream, _| {
        response(
            stream,
            302,
            &format!("Location: {location}/secret\r\n"),
            b"",
        )
    });
    let mut f = Fixture::new(&server, "/redirect", "GET", false);
    let result = f.run(65536);
    assert!(result.is_error());
    assert!(denied.calls.lock().unwrap().is_empty());
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}
#[test]
fn complete_http_error_response_keeps_body_and_error_flag() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    for inline in [65536, 8] {
        let server = Server::new(|_, stream, _| response(stream, 404, "", b"not found"));
        let mut f = Fixture::new(&server, "/missing", "GET", false);
        let result = f.run(inline);
        assert!(result.is_error());
        assert_eq!(body(&f.bytes(&result)), b"not found");
    }
}
#[test]
fn precancellation_has_no_network_effect() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let mut f = Fixture::new(&server, "/ok", "GET", false);
    let token = CancellationToken::default();
    let job = f.job(token.clone(), 65536);
    token.cancel();
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert!(server.calls.lock().unwrap().is_empty());
}
#[test]
fn artifact_failure_preserves_attempt_count_without_replay() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let mut f = Fixture::new(&server, "/ok", "GET", false);
    let completed = f.job(CancellationToken::default(), 8).run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_http_artifact BEFORE INSERT ON events WHEN NEW.kind='artifact.created' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let error = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(error.outcome["requests_attempted"], 1);
    assert!(error.result.is_none());
    assert_eq!(server.calls.lock().unwrap().len(), 1);
    assert!(f.store.claim_tool("s", "c").is_err());
}
#[test]
fn completion_failure_recovers_unknown_without_reissuing_get() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let mut f = Fixture::new(&server, "/ok", "GET", false);
    let completed = f.job(CancellationToken::default(), 65536).run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_http_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let error = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(error.outcome["requests_attempted"], 1);
    assert!(error.result.is_some());
    db.execute_batch("DROP TRIGGER fail_http_completion;")
        .unwrap();
    drop(db);
    drop(f.artifacts);
    drop(f.store);
    let mut reopened = SessionStore::open(f.dir.path().join("db")).unwrap();
    assert_eq!(
        reopened.tool_state("s", "c").unwrap().as_deref(),
        Some("OutcomeUnknown")
    );
    assert!(reopened.claim_tool("s", "c").is_err());
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}
#[test]
fn completed_artifact_survives_reopen() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", &[0, 255, 128]));
    let mut f = Fixture::new(&server, "/ok", "GET", false);
    let result = f.run(8);
    let expected = f.bytes(&result);
    let ToolResult::Artifact(a) = result else {
        panic!("artifact required")
    };
    drop(f.artifacts);
    drop(f.store);
    let mut reopened = SessionStore::open(f.dir.path().join("db")).unwrap();
    let artifacts = ArtifactStore::open(
        &mut reopened,
        f.dir.path().join("artifacts"),
        ArtifactLimits::default(),
    )
    .unwrap();
    let mut verified = artifacts
        .lookup(&reopened, "s", "c", &a.artifact_ref)
        .unwrap()
        .verify()
        .unwrap();
    let mut restored = Vec::new();
    verified.read_to_end(&mut restored).unwrap();
    assert_eq!(restored, expected);
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}

#[test]
fn elapsed_receipt_deadline_prevents_network_attempt() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let mut f = Fixture::new(&server, "/ok", "GET", false);
    f.policy = DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "http.fetch".into(),
            allowed_arguments: vec![f.request.call.arguments.clone()],
            require_approval: false,
            timeout_ms: 20,
            max_output_bytes: 65536,
        }],
    })
    .unwrap();
    let job = f.job(CancellationToken::default(), 65536);
    std::thread::sleep(Duration::from_millis(40));
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert!(server.calls.lock().unwrap().is_empty());
}
