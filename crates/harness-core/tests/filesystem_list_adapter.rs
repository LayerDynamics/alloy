#![cfg(unix)]
//! Real scoped listing through W3 admission, start, output publication and completion.
use harness_core::{
    EventKind, SessionStore,
    artifact::{ArtifactLimits, ArtifactStore},
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::{ReceiptExecutor, file_list::FileListExecution},
};
use serde_json::{Value, json};
use std::{
    io::Read,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tool_registry::{
    builtins::fs::{FileListTool, ListReport, ReadError, ReadScope},
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::{CancellationToken, ExecutionContext, ExecutionError, OutputSink},
    registry::ToolImplementation,
    result::ToolResult,
};
struct Observed {
    inner: FileListTool,
    count: Arc<AtomicUsize>,
    cancel: Option<CancellationToken>,
    mismatch: bool,
}
struct CancelOutput<'a> {
    output: &'a mut dyn OutputSink,
    token: CancellationToken,
}
impl OutputSink for CancelOutput<'_> {
    fn write_chunk(&mut self, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.output.write_chunk(bytes)?;
        self.token.cancel();
        Ok(())
    }
    fn abort(&mut self) {
        self.output.abort();
    }
}
impl ToolImplementation for Observed {
    fn execute(&self, i: &ToolInvocation) -> ToolResult {
        self.inner.execute(i)
    }
    fn execute_file_list(
        &self,
        i: &ToolInvocation,
        c: &ExecutionContext,
        o: &mut dyn OutputSink,
    ) -> Result<ListReport, ReadError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        let mut report = if let Some(token) = &self.cancel {
            self.inner.execute_file_list(
                i,
                c,
                &mut CancelOutput {
                    output: o,
                    token: token.clone(),
                },
            )?
        } else {
            self.inner.execute_file_list(i, c, o)?
        };
        if self.mismatch {
            report.output_bytes += 1;
        }
        Ok(report)
    }
}
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    artifacts: ArtifactStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    request: ToolInvocation,
    count: Arc<AtomicUsize>,
}
impl Fixture {
    fn new(
        path: &str,
        entries: u64,
        maximum: u64,
        cancel: Option<CancellationToken>,
        mismatch: bool,
    ) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("root")).unwrap();
        for name in ["z", "a", "m"] {
            std::fs::write(dir.path().join("root").join(name), b"fixture").unwrap();
        }
        std::fs::create_dir(dir.path().join("root/empty")).unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let artifacts = ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let mut executor = ReceiptExecutor::new();
        let definition = tool_registry::builtins::definitions(60000, maximum)
            .unwrap()
            .into_iter()
            .find(|d| d.id == "fs.list")
            .unwrap();
        executor
            .register(
                definition,
                Arc::new(Observed {
                    inner: FileListTool::new(ReadScope::open(dir.path().join("root")).unwrap()),
                    count: count.clone(),
                    cancel,
                    mismatch,
                }),
            )
            .unwrap();
        let arguments = json!({"path":path,"max_entries":entries})
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
                tool_id: "fs.list".into(),
                arguments: arguments.clone(),
            },
        };
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "fs.list".into(),
                allowed_arguments: vec![arguments],
                require_approval: false,
                timeout_ms: 60000,
                max_output_bytes: maximum,
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
            count,
        }
    }
    fn job(&mut self, token: CancellationToken, inline: usize) -> FileListExecution {
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
            .prepare_file_list(&self.store, receipt, &self.artifacts, token, inline)
            .unwrap()
    }
    fn run(&mut self, inline: usize) -> ToolResult {
        let completed = self.job(CancellationToken::default(), inline).run();
        completed.complete(&mut self.store).unwrap()
    }
    fn value(&self, result: &ToolResult) -> Value {
        assert!(!result.is_error(), "{result:?}");
        assert_eq!(result.call_id(), "c");
        match result {
            ToolResult::Inline(i) => i.content.clone(),
            ToolResult::Artifact(a) => {
                let mut verified = self
                    .artifacts
                    .lookup(&self.store, "s", "c", &a.artifact_ref)
                    .unwrap()
                    .verify()
                    .unwrap();
                let mut bytes = Vec::new();
                verified.read_to_end(&mut bytes).unwrap();
                serde_json::from_slice(&bytes).unwrap()
            }
        }
    }
    fn no_output(&self) {
        assert!(
            self.store
                .list_events("s")
                .unwrap()
                .iter()
                .all(|e| e.kind != EventKind::ArtifactCreated)
        );
        assert_eq!(
            std::fs::read_dir(self.dir.path().join("artifacts/tmp"))
                .unwrap()
                .count(),
            0
        );
        assert_eq!(
            std::fs::read_dir(self.dir.path().join("artifacts/objects"))
                .unwrap()
                .count(),
            0
        );
    }
}
fn names(value: &Value) -> Vec<Vec<u8>> {
    value["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| serde_json::from_value(entry["name_bytes"].clone()).unwrap())
        .collect()
}
#[test]
fn authorized_listing_is_sorted_and_persists_once() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let result = f.run(4096);
    assert!(matches!(result, ToolResult::Inline(_)));
    assert_eq!(
        names(&f.value(&result)),
        vec![
            b"a".to_vec(),
            b"empty".to_vec(),
            b"m".to_vec(),
            b"z".to_vec()
        ]
    );
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert!(f.store.claim_tool("s", "c").is_err());
    f.no_output();
}
#[test]
fn artifact_and_inline_listing_have_identical_json() {
    let mut a = Fixture::new(".", 10, 4096, None, false);
    let inline = a.run(4096);
    let expected = a.value(&inline);
    let mut b = Fixture::new(".", 10, 4096, None, false);
    let artifact = b.run(8);
    assert!(matches!(artifact, ToolResult::Artifact(_)));
    assert_eq!(b.value(&artifact), expected);
}
#[test]
fn empty_listing_is_complete() {
    let mut f = Fixture::new("empty", 1, 4096, None, false);
    let result = f.run(4096);
    assert_eq!(f.value(&result), json!({"entries":[]}));
    f.no_output();
}
#[test]
fn policy_denial_never_enters_listing_backend() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
    assert_eq!(
        f.store
            .admit_tool(f.executor.registry(), &deny, f.request.clone())
            .unwrap(),
        IntentAdmission::Rejected
    );
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
    f.no_output();
}
#[test]
fn changed_arguments_and_authority_fields_are_rejected() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let mut changed = f.request.clone();
    changed.call.arguments.insert("path".into(), json!("empty"));
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
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
}
#[test]
fn traversal_absolute_and_symlink_directories_are_rejected() {
    use std::os::unix::fs::symlink;
    for path in ["../outside", "/tmp", "a/../empty", "link", "link/sub"] {
        let mut f = Fixture::new(path, 10, 4096, None, false);
        std::fs::create_dir(f.dir.path().join("outside")).unwrap();
        std::fs::write(f.dir.path().join("outside/SECRET"), b"secret").unwrap();
        symlink(f.dir.path().join("outside"), f.dir.path().join("root/link")).unwrap();
        let result = f.run(4096);
        assert!(result.is_error());
        f.no_output();
        assert_eq!(
            std::fs::read(f.dir.path().join("outside/SECRET")).unwrap(),
            b"secret"
        );
    }
}
#[test]
fn symlink_entry_is_a_name_not_a_followed_target() {
    use std::os::unix::fs::symlink;
    let mut f = Fixture::new(".", 10, 4096, None, false);
    symlink(
        "/nonexistent-fixture-target",
        f.dir.path().join("root/link"),
    )
    .unwrap();
    let result = f.run(4096);
    assert!(names(&f.value(&result)).contains(&b"link".to_vec()));
}
#[test]
fn entry_limit_rejects_whole_listing_not_arbitrary_prefix() {
    let mut f = Fixture::new(".", 3, 4096, None, false);
    let result = f.run(4096);
    assert!(result.is_error());
    f.no_output();
}
#[test]
fn serialized_output_limit_rejects_whole_listing() {
    let mut f = Fixture::new(".", 10, 16, None, false);
    let result = f.run(4096);
    assert!(result.is_error());
    f.no_output();
}
#[test]
fn precancelled_job_never_enters_backend() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let token = CancellationToken::default();
    let job = f.job(token.clone(), 4096);
    token.cancel();
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
    f.no_output();
}
#[test]
fn cancellation_during_stream_aborts_partial_listing() {
    let token = CancellationToken::default();
    let mut f = Fixture::new(".", 10, 4096, Some(token.clone()), false);
    let job = f.job(token, 4096);
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    f.no_output();
    let mut other = Fixture::new(".", 10, 4096, None, false);
    let result = other.run(4096);
    assert!(!result.is_error());
}
#[test]
fn elapsed_start_deadline_is_not_renewed() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    f.policy = DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "fs.list".into(),
            allowed_arguments: vec![f.request.call.arguments.clone()],
            require_approval: false,
            timeout_ms: 20,
            max_output_bytes: 4096,
        }],
    })
    .unwrap();
    let job = f.job(CancellationToken::default(), 4096);
    std::thread::sleep(std::time::Duration::from_millis(40));
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
    f.no_output();
}
#[test]
fn mismatched_byte_report_cannot_finalize_output() {
    let mut f = Fixture::new(".", 10, 4096, None, true);
    let result = f.run(4096);
    assert!(result.is_error());
    f.no_output();
}
#[test]
fn artifact_event_failure_never_reauthorizes_listing() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let completed = f.job(CancellationToken::default(), 8).run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_list_artifact BEFORE INSERT ON events WHEN NEW.kind='artifact.created' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    assert!(completed.complete(&mut f.store).is_err());
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
}
#[test]
fn completion_failure_never_replays_listing() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let completed = f.job(CancellationToken::default(), 8).run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_list_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    assert!(completed.complete(&mut f.store).is_err());
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
}
#[test]
fn persisted_listing_artifact_survives_reopen() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let result = f.run(8);
    let expected = f.value(&result);
    let ToolResult::Artifact(a) = result else {
        panic!("artifact required")
    };
    let dir = f.dir;
    drop(f.artifacts);
    drop(f.store);
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    let artifacts = ArtifactStore::open(
        &mut store,
        dir.path().join("artifacts"),
        ArtifactLimits::default(),
    )
    .unwrap();
    let mut verified = artifacts
        .lookup(&store, "s", "c", &a.artifact_ref)
        .unwrap()
        .verify()
        .unwrap();
    let mut bytes = Vec::new();
    verified.read_to_end(&mut bytes).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
    assert_eq!(
        store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert!(store.claim_tool("s", "c").is_err());
}
#[test]
fn legacy_and_generic_entrypoints_fail_closed() {
    let f = Fixture::new(".", 10, 4096, None, false);
    let tool = FileListTool::new(ReadScope::open(f.dir.path().join("root")).unwrap());
    assert!(tool.execute(&f.request).is_error());
    let policy = f
        .executor
        .registry()
        .definition("fs.list")
        .unwrap()
        .policy
        .clone();
    let context =
        ExecutionContext::new(&policy, 60000, 4096, CancellationToken::default()).unwrap();
    let mut output = tool_registry::execution::BoundedOutput::new(&context, 4096).unwrap();
    assert!(
        tool.execute_with_context(&f.request, &context, &mut output)
            .is_error()
    );
    assert!(output.finish().is_err());
    f.no_output();
}

#[test]
fn exact_serialized_limit_succeeds_and_one_less_rejects() {
    let mut baseline = Fixture::new(".", 10, 4096, None, false);
    let result = baseline.run(4096);
    let expected = baseline.value(&result);
    let exact = serde_json::to_vec(&expected).unwrap().len() as u64;
    let mut allowed = Fixture::new(".", 10, exact, None, false);
    let result = allowed.run(4096);
    assert_eq!(allowed.value(&result), expected);
    let mut denied = Fixture::new(".", 10, exact - 1, None, false);
    let result = denied.run(4096);
    assert!(result.is_error());
    denied.no_output();
}
#[test]
fn pinned_scope_cannot_be_replaced_by_ambient_root_name() {
    let mut f = Fixture::new(".", 10, 4096, None, false);
    std::fs::rename(f.dir.path().join("root"), f.dir.path().join("original")).unwrap();
    std::fs::create_dir(f.dir.path().join("root")).unwrap();
    std::fs::write(
        f.dir.path().join("root/EVIL"),
        b"not the approved directory",
    )
    .unwrap();
    let result = f.run(4096);
    let actual = names(&f.value(&result));
    assert!(actual.contains(&b"a".to_vec()));
    assert!(!actual.contains(&b"EVIL".to_vec()));
}
#[test]
fn unsupported_registered_backend_never_falls_back_to_legacy() {
    struct Legacy;
    impl ToolImplementation for Legacy {
        fn execute(&self, _: &ToolInvocation) -> ToolResult {
            panic!("legacy entrypoint must not run")
        }
    }
    let mut f = Fixture::new(".", 10, 4096, None, false);
    let definition = f.executor.registry().definition("fs.list").unwrap().clone();
    let mut executor = ReceiptExecutor::new();
    executor.register(definition, Arc::new(Legacy)).unwrap();
    f.executor = executor;
    let result = f.run(4096);
    assert!(result.is_error());
    f.no_output();
}
