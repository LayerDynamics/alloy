#![cfg(any(target_os = "linux", target_os = "macos"))]
//! W4-04: real create-only writer through approval, durable start and completion.
use harness_core::{
    SessionStore,
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::{ReceiptExecutor, file_write::FileWriteExecution},
};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tool_registry::{
    builtins::fs::{
        FileWriteTool, ReadError, WriteFailure, WritePublication, WriteReport, WriteScope,
    },
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::{CancellationToken, ExecutionContext, ExecutionError, OutputSink},
    registry::ToolImplementation,
    result::ToolResult,
};
#[derive(Clone, Copy)]
enum Mode {
    Normal,
    CancelAfter,
    FailOutput,
    Unknown,
    Mismatch,
}
struct Reject;
impl OutputSink for Reject {
    fn write_chunk(&mut self, _: &[u8]) -> Result<(), ExecutionError> {
        Err(ExecutionError::OutputLimit)
    }
    fn abort(&mut self) {}
}
struct Observed {
    inner: FileWriteTool,
    count: Arc<AtomicUsize>,
    mode: Mode,
    token: CancellationToken,
}
impl ToolImplementation for Observed {
    fn execute(&self, i: &ToolInvocation) -> ToolResult {
        self.inner.execute(i)
    }
    fn execute_file_write(
        &self,
        i: &ToolInvocation,
        c: &ExecutionContext,
        o: &mut dyn OutputSink,
    ) -> Result<WriteReport, WriteFailure> {
        self.count.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::FailOutput) {
            return self.inner.execute_file_write(i, c, &mut Reject);
        }
        let mut report = self.inner.execute_file_write(i, c, o)?;
        match self.mode {
            Mode::CancelAfter => self.token.cancel(),
            // Inject uncertain status after a real effect to test conservative
            // boundary mapping, not to claim a kernel rename failure was observed.
            Mode::Unknown => {
                return Err(WriteFailure {
                    publication: WritePublication::Unknown,
                    cause: ReadError::Io(std::io::Error::other("injected uncertain outcome")),
                    cleanup_error: None,
                });
            }
            Mode::Mismatch => report.bytes_written += 1,
            _ => (),
        }
        Ok(report)
    }
}
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    request: ToolInvocation,
    count: Arc<AtomicUsize>,
    token: CancellationToken,
}
impl Fixture {
    fn new(path: &str, content: &str, input: u64, output: u64, mode: Mode) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("root")).unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let token = CancellationToken::default();
        let inner = FileWriteTool::new(WriteScope::open(dir.path().join("root"), input).unwrap());
        let definition = tool_registry::builtins::definitions(60000, output)
            .unwrap()
            .into_iter()
            .find(|d| d.id == "fs.write")
            .unwrap();
        let mut executor = ReceiptExecutor::new();
        executor
            .register(
                definition,
                Arc::new(Observed {
                    inner,
                    count: count.clone(),
                    mode,
                    token: token.clone(),
                }),
            )
            .unwrap();
        let arguments = json!({"path":path,"content":content})
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
                tool_id: "fs.write".into(),
                arguments: arguments.clone(),
            },
        };
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "fs.write".into(),
                allowed_arguments: vec![arguments],
                require_approval: true,
                timeout_ms: 60000,
                max_output_bytes: output,
            }],
        })
        .unwrap();
        Self {
            dir,
            store,
            executor,
            policy,
            request,
            count,
            token,
        }
    }
    fn admit(&mut self) {
        assert_eq!(
            self.store
                .admit_tool(self.executor.registry(), &self.policy, self.request.clone())
                .unwrap(),
            IntentAdmission::AwaitingApproval
        );
    }
    fn job(&mut self) -> FileWriteExecution {
        self.admit();
        self.store.decide_tool("s", "c", true).unwrap();
        let claim = self.store.claim_tool("s", "c").unwrap();
        let receipt = self
            .store
            .start_tool(claim, self.executor.registry(), &self.policy)
            .unwrap();
        self.executor
            .prepare_file_write(&self.store, receipt, self.token.clone())
            .unwrap()
    }
    fn run(&mut self) -> ToolResult {
        let completed = self.job().run();
        completed.complete(&mut self.store).unwrap()
    }
    fn no_staging(&self) {
        assert!(
            std::fs::read_dir(self.dir.path().join("root"))
                .unwrap()
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".alloy-write-"))
        );
    }
    fn zero(&self) {
        assert_eq!(self.count.load(Ordering::SeqCst), 0);
        assert!(!self.dir.path().join("root/target").exists());
        self.no_staging();
    }
}
fn publication(result: &ToolResult) -> &str {
    match result {
        ToolResult::Inline(i) => {
            if result.is_error() {
                i.content["details"]["publication"].as_str().unwrap()
            } else {
                i.content["publication"].as_str().unwrap()
            }
        }
        _ => panic!("inline write metadata required"),
    }
}
#[test]
fn approved_write_preserves_exact_utf8_and_completes_once() {
    let text = "résumé 雪\n";
    let mut f = Fixture::new("target", text, 4096, 4096, Mode::Normal);
    let result = f.run();
    assert!(!result.is_error());
    assert_eq!(result.call_id(), "c");
    assert_eq!(publication(&result), "published");
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        text.as_bytes()
    );
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    f.no_staging();
}
#[test]
fn waiting_for_approval_and_denial_have_zero_effects() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    f.admit();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.zero();
    f.store.decide_tool("s", "c", false).unwrap();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.zero();
}
#[test]
fn changed_path_content_and_authority_fields_never_start() {
    for (key, value) in [("path", json!("other")), ("content", json!("changed"))] {
        let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
        let mut request = f.request.clone();
        request.call.arguments.insert(key.into(), value);
        assert_eq!(
            f.store
                .admit_tool(f.executor.registry(), &f.policy, request)
                .unwrap(),
            IntentAdmission::Rejected
        );
        f.zero();
    }
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    let mut request = f.request.clone();
    request
        .call
        .arguments
        .insert("approved".into(), json!(true));
    assert!(
        f.store
            .admit_tool(f.executor.registry(), &f.policy, request)
            .is_err()
    );
    f.zero();
}
#[test]
fn policy_change_before_durable_start_blocks_effect() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    f.admit();
    f.store.decide_tool("s", "c", true).unwrap();
    let claim = f.store.claim_tool("s", "c").unwrap();
    let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
    assert!(
        f.store
            .start_tool(claim, f.executor.registry(), &deny)
            .is_err()
    );
    f.zero();
}
#[test]
fn traversal_absolute_and_reserved_paths_reject_before_publication() {
    for path in [
        "../outside",
        "/tmp/alloy-write-escape-fixture",
        "sub/../target",
        ".alloy-write-forbidden",
    ] {
        let mut f = Fixture::new(path, "data", 4096, 4096, Mode::Normal);
        let result = f.run();
        assert!(result.is_error());
        assert_eq!(publication(&result), "not_published");
        assert!(!f.dir.path().join("outside").exists());
        f.no_staging();
    }
}
#[test]
fn symlink_parent_and_destination_are_never_followed() {
    use std::os::unix::fs::symlink;
    for path in ["link/target", "target"] {
        let mut f = Fixture::new(path, "data", 4096, 4096, Mode::Normal);
        std::fs::create_dir(f.dir.path().join("outside")).unwrap();
        symlink(f.dir.path().join("outside"), f.dir.path().join("root/link")).unwrap();
        if path == "target" {
            symlink(
                f.dir.path().join("outside/target"),
                f.dir.path().join("root/target"),
            )
            .unwrap();
        }
        let result = f.run();
        assert!(result.is_error());
        assert_eq!(publication(&result), "not_published");
        assert!(!f.dir.path().join("outside/target").exists());
        f.no_staging();
    }
}
#[test]
fn existing_file_is_not_replaced() {
    let mut f = Fixture::new("target", "new", 4096, 4096, Mode::Normal);
    std::fs::write(f.dir.path().join("root/target"), b"original").unwrap();
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "not_published");
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        b"original"
    );
    f.no_staging();
}
#[test]
fn input_limit_rejects_before_staging() {
    let mut f = Fixture::new("target", "too big", 3, 4096, Mode::Normal);
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "not_published");
    assert!(!f.dir.path().join("root/target").exists());
    f.no_staging();
}
#[test]
fn successful_result_limit_is_preflighted_before_effect() {
    let mut f = Fixture::new("target", "data", 4096, 16, Mode::Normal);
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "not_published");
    f.zero();
}
#[test]
fn precancelled_job_never_enters_writer() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    let job = f.job();
    f.token.cancel();
    let completed = job.run();
    let result = completed.complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(publication(&result), "not_published");
    f.zero();
}
#[test]
fn expired_start_deadline_is_not_restarted() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    f.policy = DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "fs.write".into(),
            allowed_arguments: vec![f.request.call.arguments.clone()],
            require_approval: true,
            timeout_ms: 20,
            max_output_bytes: 4096,
        }],
    })
    .unwrap();
    let job = f.job();
    std::thread::sleep(std::time::Duration::from_millis(40));
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(publication(&result), "not_published");
    f.zero();
}
#[test]
fn cancellation_after_publication_preserves_published_status() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::CancelAfter);
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "published");
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        b"data"
    );
    f.no_staging();
}
#[test]
fn output_delivery_failure_preserves_published_effect() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::FailOutput);
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "published");
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        b"data"
    );
    f.no_staging();
}
#[test]
fn injected_uncertain_outcome_stays_unknown() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Unknown);
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "unknown");
    assert!(f.store.claim_tool("s", "c").is_err());
}
#[test]
fn malformed_backend_report_never_claims_no_effect() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Mismatch);
    let result = f.run();
    assert!(result.is_error());
    assert_eq!(publication(&result), "unknown");
    assert!(f.dir.path().join("root/target").exists());
}
#[test]
fn completion_failure_returns_actual_published_result_without_replay() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    let completed = f.job().run();
    assert_eq!(publication(completed.result()), "published");
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_write_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let failure = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(publication(&failure.result), "published");
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        b"data"
    );
}
#[test]
fn uncertain_outcome_survives_completion_failure() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Unknown);
    let completed = f.job().run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_unknown_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let failure = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(publication(&failure.result), "unknown");
    assert!(f.store.claim_tool("s", "c").is_err());
}
#[test]
fn recovery_after_failed_completion_never_reexecutes_write() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    let completed = f.job().run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_recovery_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let failure = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(publication(&failure.result), "published");
    // The completion failure has already been observed. Remove only the
    // injected fault so restart recovery can record OutcomeUnknown.
    db.execute_batch("DROP TRIGGER fail_recovery_completion;")
        .unwrap();
    drop(db);
    drop(f.store);
    let mut store = SessionStore::open(f.dir.path().join("db")).unwrap();
    assert_eq!(
        store.tool_state("s", "c").unwrap().as_deref(),
        Some("OutcomeUnknown")
    );
    assert!(store.claim_tool("s", "c").is_err());
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        b"data"
    );
}
#[test]
fn cancellation_between_run_and_completion_retains_effect() {
    let mut f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    let completed = f.job().run();
    f.token.cancel();
    let result = completed.complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(publication(&result), "published");
}
#[test]
fn legacy_and_generic_entrypoints_do_not_write() {
    let f = Fixture::new("target", "data", 4096, 4096, Mode::Normal);
    let tool = FileWriteTool::new(WriteScope::open(f.dir.path().join("root"), 4096).unwrap());
    assert!(tool.execute(&f.request).is_error());
    let policy = &f.executor.registry().definition("fs.write").unwrap().policy;
    let context = ExecutionContext::new(policy, 60000, 4096, CancellationToken::default()).unwrap();
    let mut output = tool_registry::execution::BoundedOutput::new(&context, 4096).unwrap();
    assert!(
        tool.execute_with_context(&f.request, &context, &mut output)
            .is_error()
    );
    assert!(output.finish().is_err());
    f.zero();
}

#[test]
fn empty_content_is_a_real_create_only_write() {
    let mut f = Fixture::new("target", "", 4096, 4096, Mode::Normal);
    let result = f.run();
    assert!(!result.is_error());
    assert_eq!(publication(&result), "published");
    assert_eq!(
        std::fs::read(f.dir.path().join("root/target")).unwrap(),
        b""
    );
    f.no_staging();
}
#[test]
fn cancellation_does_not_affect_another_write_operation() {
    let mut first = Fixture::new("target", "first", 4096, 4096, Mode::Normal);
    let job = first.job();
    first.token.cancel();
    let result = job.run().complete(&mut first.store).unwrap();
    assert!(result.is_error());
    first.zero();
    let mut second = Fixture::new("target", "second", 4096, 4096, Mode::Normal);
    let result = second.run();
    assert!(!result.is_error());
    assert_eq!(
        std::fs::read(second.dir.path().join("root/target")).unwrap(),
        b"second"
    );
}
