#![cfg(unix)]
//! Real FileReadTool + W3 transactions; not full W4-02.02/artifact acceptance.
use harness_core::{
    SessionStore,
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::ReceiptExecutor,
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tool_registry::{
    builtins::fs::{FileReadTool, ReadScope},
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, OutputSink},
    registry::ToolImplementation,
    result::ToolResult,
};
struct ObservedRead {
    inner: FileReadTool,
    count: Arc<AtomicUsize>,
}
impl ToolImplementation for ObservedRead {
    fn execute(&self, i: &ToolInvocation) -> ToolResult {
        self.inner.execute(i)
    }
    fn execute_with_context(
        &self,
        i: &ToolInvocation,
        c: &ExecutionContext,
        o: &mut dyn OutputSink,
    ) -> ToolResult {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_with_context(i, c, o)
    }
}
fn request(path: &str, offset: u64, length: u64) -> ToolInvocation {
    ToolInvocation {
        context: ToolCallContext {
            session_id: "s".into(),
            turn_id: "t".into(),
            generation_id: "g".into(),
        },
        call: ToolCall {
            call_id: "c".into(),
            tool_id: "fs.read".into(),
            arguments: json!({"path":path,"offset":offset,"length":length})
                .as_object()
                .unwrap()
                .clone(),
        },
    }
}
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    request: ToolInvocation,
    count: Arc<AtomicUsize>,
}
impl Fixture {
    fn new(data: &[u8], path: &str, offset: u64, length: u64, inline: usize, timeout: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("root")).unwrap();
        std::fs::write(dir.path().join("root/data"), data).unwrap();
        std::fs::write(dir.path().join("outside"), b"OUTSIDE_SECRET").unwrap();
        let inner =
            FileReadTool::new(ReadScope::open(dir.path().join("root")).unwrap(), inline).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let mut executor = ReceiptExecutor::new();
        let definition = tool_registry::builtins::definitions(10000, 4096)
            .unwrap()
            .into_iter()
            .find(|d| d.id == "fs.read")
            .unwrap();
        executor
            .register(
                definition,
                Arc::new(ObservedRead {
                    inner,
                    count: count.clone(),
                }),
            )
            .unwrap();
        let request = request(path, offset, length);
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "fs.read".into(),
                allowed_arguments: vec![request.call.arguments.clone()],
                require_approval: false,
                timeout_ms: timeout,
                max_output_bytes: 4096,
            }],
        })
        .unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        Self {
            dir,
            store,
            executor,
            policy,
            request,
            count,
        }
    }
    fn job(&mut self, token: CancellationToken) -> harness_core::tool_execution::StartedExecution {
        assert_eq!(
            self.store
                .admit_tool(self.executor.registry(), &self.policy, self.request.clone())
                .unwrap(),
            IntentAdmission::Ready
        );
        let claim = self.store.claim_tool("s", "c").unwrap();
        let receipt = self
            .store
            .start_tool(claim, self.executor.registry(), &self.policy)
            .unwrap();
        self.executor
            .prepare_started(&self.store, receipt, token)
            .unwrap()
    }
    fn zero(&self) {
        assert_eq!(self.count.load(Ordering::SeqCst), 0);
    }
}
fn output() -> BoundedOutput {
    let d = tool_registry::builtins::definitions(10000, 4096)
        .unwrap()
        .remove(0);
    let c = ExecutionContext::new(&d.policy, 10000, 4096, CancellationToken::default()).unwrap();
    BoundedOutput::new(&c, 4096).unwrap()
}
fn successful(f: &mut Fixture) -> Value {
    let job = f.job(CancellationToken::default());
    let mut out = output();
    let completed = job.run(&mut out);
    assert!(!completed.result().is_error(), "{:?}", completed.result());
    assert_eq!(completed.result().call_id(), "c");
    let content = match completed.result() {
        ToolResult::Inline(r) => r.content.clone(),
        _ => panic!("expected inline"),
    };
    let streamed: Value = serde_json::from_slice(&out.finish().unwrap()).unwrap();
    assert_eq!(streamed, content);
    completed.complete(&mut f.store).unwrap();
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    content
}
fn rejected(f: &mut Fixture) -> ToolResult {
    let job = f.job(CancellationToken::default());
    let mut out = output();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    assert!(out.finish().is_err());
    let result = completed.result().clone();
    completed.complete(&mut f.store).unwrap();
    result
}
#[test]
fn authorized_range_returns_and_persists_exact_content() {
    let mut f = Fixture::new(b"0123456789", "data", 2, 4, 4096, 5000);
    let content = successful(&mut f);
    assert_eq!(content["text"], "2345");
    assert_eq!(content["bytes_read"], 4);
    assert_eq!(content["offset"], 2);
    assert_eq!(
        f.request.call.arguments,
        json!({"path":"data","offset":2,"length":4})
            .as_object()
            .unwrap()
            .clone()
    );
}
#[test]
fn empty_and_short_eof_are_complete_not_truncation() {
    for (data, offset, length, text) in [(&b""[..], 0, 8, ""), (&b"0123456789"[..], 8, 8, "89")] {
        let mut f = Fixture::new(data, "data", offset, length, 4096, 5000);
        let content = successful(&mut f);
        assert_eq!(content["text"], text);
        assert_eq!(content["bytes_read"], text.len() as u64);
    }
}
#[test]
fn utf8_text_is_preserved_without_lossy_conversion() {
    let text = "résumé 雪\n";
    let mut f = Fixture::new(text.as_bytes(), "data", 0, text.len() as u64, 4096, 5000);
    assert_eq!(successful(&mut f)["text"], text);
}
#[test]
fn denied_policy_never_invokes_real_adapter() {
    let mut f = Fixture::new(b"data", "data", 0, 4, 4096, 5000);
    let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
    assert_eq!(
        f.store
            .admit_tool(f.executor.registry(), &deny, f.request.clone())
            .unwrap(),
        IntentAdmission::Rejected
    );
    assert!(f.store.claim_tool("s", "c").is_err());
    f.zero();
}
#[test]
fn changed_or_extra_arguments_never_invoke_real_adapter() {
    let mut f = Fixture::new(b"data", "data", 0, 4, 4096, 5000);
    let mut changed = f.request.clone();
    changed
        .call
        .arguments
        .insert("path".into(), json!("outside"));
    assert_eq!(
        f.store
            .admit_tool(f.executor.registry(), &f.policy, changed)
            .unwrap(),
        IntentAdmission::Rejected
    );
    let mut extra = f.request.clone();
    extra.call.call_id = "extra".into();
    extra.call.arguments.insert("approved".into(), json!(true));
    assert!(
        f.store
            .admit_tool(f.executor.registry(), &f.policy, extra)
            .is_err()
    );
    f.zero();
}
#[test]
fn traversal_and_absolute_paths_never_return_outside_data() {
    for path in ["../outside", "/etc/passwd", "a/../data"] {
        let mut f = Fixture::new(b"data", path, 0, 4, 4096, 5000);
        let _ = rejected(&mut f);
        assert_eq!(
            std::fs::read(f.dir.path().join("outside")).unwrap(),
            b"OUTSIDE_SECRET"
        );
    }
}
#[test]
fn symlink_targets_and_parent_components_are_rejected() {
    use std::os::unix::fs::symlink;
    for path in ["link", "parent/outside"] {
        let mut f = Fixture::new(b"data", path, 0, 4, 4096, 5000);
        symlink(f.dir.path().join("outside"), f.dir.path().join("root/link")).unwrap();
        symlink(f.dir.path(), f.dir.path().join("root/parent")).unwrap();
        let _ = rejected(&mut f);
        assert_eq!(
            std::fs::read(f.dir.path().join("outside")).unwrap(),
            b"OUTSIDE_SECRET"
        );
    }
}
#[test]
fn directory_target_is_rejected() {
    let mut f = Fixture::new(b"data", "sub", 0, 4, 4096, 5000);
    std::fs::create_dir(f.dir.path().join("root/sub")).unwrap();
    let _ = rejected(&mut f);
}
#[test]
fn precancelled_job_does_not_enter_adapter() {
    let mut f = Fixture::new(b"data", "data", 0, 4, 4096, 5000);
    let token = CancellationToken::default();
    let job = f.job(token.clone());
    token.cancel();
    let mut out = output();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    f.zero();
    completed.complete(&mut f.store).unwrap();
}
#[test]
fn elapsed_receipt_deadline_is_not_renewed() {
    let mut f = Fixture::new(b"data", "data", 0, 4, 4096, 50);
    let job = f.job(CancellationToken::default());
    std::thread::sleep(std::time::Duration::from_millis(100));
    let mut out = output();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    f.zero();
    completed.complete(&mut f.store).unwrap();
}
#[test]
fn inline_capacity_rejection_never_truncates_the_requested_range() {
    let mut f = Fixture::new(b"0123456789", "data", 0, 10, 8, 5000);
    let result = rejected(&mut f);
    if let ToolResult::Inline(r) = result {
        assert_eq!(r.content["details"]["reason"], "artifact_output_required");
        assert_eq!(r.content["details"]["scope_operation_attempted"], false);
    }
}
#[test]
fn binary_payload_requires_artifact_instead_of_lossy_success() {
    let mut f = Fixture::new(&[0, 255, 128], "data", 0, 3, 4096, 5000);
    let result = rejected(&mut f);
    if let ToolResult::Inline(r) = result {
        assert_eq!(r.content["details"]["reason"], "artifact_output_required");
    }
}
#[test]
fn serialized_json_expansion_cannot_finalize_a_partial_prefix() {
    let bytes = vec![b'\n'; 80];
    let mut f = Fixture::new(&bytes, "data", 0, 80, 128, 5000);
    let result = rejected(&mut f);
    if let ToolResult::Inline(r) = result {
        assert_eq!(r.content["details"]["reason"], "output_limit");
    }
}
#[test]
fn completion_failure_does_not_authorize_another_read() {
    let mut f = Fixture::new(b"data", "data", 0, 4, 4096, 5000);
    let job = f.job(CancellationToken::default());
    let mut out = output();
    let completed = job.run(&mut out);
    assert!(!completed.result().is_error());
    let probe = rusqlite::Connection::open(f.store.path()).unwrap();
    probe.execute_batch("CREATE TRIGGER fail_read_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'read_completion_failure'); END;").unwrap();
    assert!(completed.complete(&mut f.store).is_err());
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
}
#[test]
fn legacy_entrypoint_fails_closed() {
    let f = Fixture::new(b"data", "data", 0, 4, 4096, 5000);
    let tool =
        FileReadTool::new(ReadScope::open(f.dir.path().join("root")).unwrap(), 4096).unwrap();
    assert!(tool.execute(&f.request).is_error());
    f.zero();
}
