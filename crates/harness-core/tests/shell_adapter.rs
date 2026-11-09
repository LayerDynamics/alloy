#![cfg(any(target_os = "linux", target_os = "macos"))]
//! Real self-hosted child processes through W3 destructive approval and receipts.
use harness_core::{
    SessionStore,
    artifact::{ArtifactLimits, ArtifactStore},
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::{ReceiptExecutor, shell_run::ShellExecution},
};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tool_registry::{
    builtins::shell::{ShellFailure, ShellReport, ShellRunTool, ShellScope},
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::{CancellationToken, ExecutionContext, OutputSink},
    registry::ToolImplementation,
    result::ToolResult,
};
struct Observed {
    inner: ShellRunTool,
    count: Arc<AtomicUsize>,
}
impl ToolImplementation for Observed {
    fn execute(&self, i: &ToolInvocation) -> ToolResult {
        self.inner.execute(i)
    }
    fn execute_shell_run(
        &self,
        i: &ToolInvocation,
        c: &ExecutionContext,
        o: &mut dyn OutputSink,
    ) -> Result<ShellReport, ShellFailure> {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_shell_run(i, c, o)
    }
}
#[test]
fn shell_child_fixture() {
    let args: Vec<_> = std::env::args().collect();
    let Some(mode) = args
        .windows(2)
        .find(|pair| pair[0] == "--skip")
        .map(|pair| pair[1].as_str())
    else {
        return;
    };
    match mode {
        "marker" => std::fs::write("marker", b"executed").unwrap(),
        "hang" => {
            std::fs::write("marker", b"started").unwrap();
            std::thread::sleep(std::time::Duration::from_secs(30));
        }
        "binary" => {
            std::io::stdout().write_all(&[0, 255, 128, 10]).unwrap();
            std::io::stderr().write_all(b"EXACT_STDERR").unwrap();
        }
        "exit7" => std::process::exit(7),
        "env" => {
            assert!(std::env::var_os("PATH").is_none());
            let mut bytes = Vec::new();
            std::io::stdin().read_to_end(&mut bytes).unwrap();
            assert!(bytes.is_empty());
        }
        "cwd" => std::io::stdout()
            .write_all(
                std::env::current_dir()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .as_bytes(),
            )
            .unwrap(),
        "flood" => loop {
            if std::io::stdout().write_all(&[b'x'; 8192]).is_err() {
                break;
            }
        },
        _ if mode.starts_with("echo:") => {
            std::io::stdout().write_all(&mode.as_bytes()[5..]).unwrap()
        }
        _ => panic!("unknown child mode"),
    }
}
struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    store: SessionStore,
    artifacts: ArtifactStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    request: ToolInvocation,
    count: Arc<AtomicUsize>,
}
impl Fixture {
    fn new(mode: &str, maximum: u64, timeout: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::create_dir(root.join("cwd")).unwrap();
        let executable = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
        let scope =
            ShellScope::new(vec![executable.clone()], vec![root.join("cwd")], 65536).unwrap();
        let mut store = SessionStore::open(root.join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let artifacts = ArtifactStore::open(
            &mut store,
            root.join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let mut executor = ReceiptExecutor::new();
        let definition = tool_registry::builtins::definitions(10000, maximum)
            .unwrap()
            .into_iter()
            .find(|d| d.id == "shell.run")
            .unwrap();
        executor
            .register(
                definition,
                Arc::new(Observed {
                    inner: ShellRunTool::new(scope),
                    count: count.clone(),
                }),
            )
            .unwrap();
        let arguments = json!({"executable":executable.to_str().unwrap(),
            "args":["--exact","shell_child_fixture","--nocapture","--test-threads=1","--skip",mode],
            "cwd":root.join("cwd").to_str().unwrap()})
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
                tool_id: "shell.run".into(),
                arguments: arguments.clone(),
            },
        };
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "shell.run".into(),
                allowed_arguments: vec![arguments],
                require_approval: false,
                timeout_ms: timeout,
                max_output_bytes: maximum,
            }],
        })
        .unwrap();
        Self {
            _dir: dir,
            root,
            store,
            artifacts,
            executor,
            policy,
            request,
            count,
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
    fn job(&mut self, token: CancellationToken, inline: usize) -> ShellExecution {
        self.admit();
        self.store.decide_tool("s", "c", true).unwrap();
        let claim = self.store.claim_tool("s", "c").unwrap();
        let receipt = self
            .store
            .start_tool(claim, self.executor.registry(), &self.policy)
            .unwrap();
        self.executor
            .prepare_shell_run(&self.store, receipt, &self.artifacts, token, inline)
            .unwrap()
    }
    fn run(&mut self, inline: usize) -> ToolResult {
        let completed = self.job(CancellationToken::default(), inline).run();
        completed.complete(&mut self.store).unwrap()
    }
    fn zero(&self) {
        assert_eq!(self.count.load(Ordering::SeqCst), 0);
        assert!(!self.root.join("cwd/marker").exists());
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
    fn no_artifact(&self) {
        assert_eq!(
            std::fs::read_dir(self.root.join("artifacts/tmp"))
                .unwrap()
                .count(),
            0
        );
        assert_eq!(
            std::fs::read_dir(self.root.join("artifacts/objects"))
                .unwrap()
                .count(),
            0
        );
    }
}
fn stream(bytes: &[u8], name: &str) -> Vec<u8> {
    let mut output = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
        let value: Value = serde_json::from_slice(line).unwrap();
        if value["stream"] == name {
            output.extend(serde_json::from_value::<Vec<u8>>(value["data"].clone()).unwrap());
        }
    }
    output
}
fn error_outcome(result: &ToolResult) -> &Value {
    let ToolResult::Inline(i) = result else {
        panic!("inline error metadata required")
    };
    &i.content["details"]["outcome"]
}
#[test]
fn destructive_default_waits_and_denial_spawns_nothing() {
    let mut f = Fixture::new("marker", 65536, 5000);
    f.admit();
    f.zero();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.store.decide_tool("s", "c", false).unwrap();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.zero();
}
#[test]
fn approved_exact_invocation_executes_and_persists_once() {
    let mut f = Fixture::new("marker", 65536, 5000);
    let result = f.run(65536);
    assert!(!result.is_error(), "{result:?}");
    assert_eq!(result.call_id(), "c");
    assert_eq!(
        std::fs::read(f.root.join("cwd/marker")).unwrap(),
        b"executed"
    );
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    f.no_artifact();
}
#[test]
fn changed_executable_argv_and_cwd_cannot_spawn() {
    for (key, value) in [
        ("executable", json!("/bin/sh")),
        ("args", json!(["-c", "touch marker"])),
        ("cwd", json!("/tmp")),
    ] {
        let mut f = Fixture::new("marker", 65536, 5000);
        let mut changed = f.request.clone();
        changed.call.arguments.insert(key.into(), value);
        assert_eq!(
            f.store
                .admit_tool(f.executor.registry(), &f.policy, changed)
                .unwrap(),
            IntentAdmission::Rejected
        );
        f.zero();
    }
}
#[test]
fn authority_fields_are_rejected_before_spawn() {
    let mut f = Fixture::new("marker", 65536, 5000);
    let mut changed = f.request.clone();
    changed
        .call
        .arguments
        .insert("approved".into(), json!(true));
    assert!(
        f.store
            .admit_tool(f.executor.registry(), &f.policy, changed)
            .is_err()
    );
    f.zero();
}
#[test]
fn wrong_session_and_duplicate_decisions_cannot_spawn() {
    let mut f = Fixture::new("marker", 65536, 5000);
    f.store.create_session("other", "fixture").unwrap();
    f.admit();
    assert!(f.store.decide_tool("other", "c", true).is_err());
    assert!(f.store.claim_tool("s", "c").is_err());
    f.store.decide_tool("s", "c", false).unwrap();
    assert!(f.store.decide_tool("s", "c", true).is_err());
    f.zero();
}
#[test]
fn changed_policy_before_start_prevents_spawn() {
    let mut f = Fixture::new("marker", 65536, 5000);
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
fn failed_start_transaction_has_no_marker_or_dispatch() {
    let mut f = Fixture::new("marker", 65536, 5000);
    f.admit();
    f.store.decide_tool("s", "c", true).unwrap();
    let claim = f.store.claim_tool("s", "c").unwrap();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_shell_start BEFORE INSERT ON events WHEN NEW.kind='tool.started' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    assert!(
        f.store
            .start_tool(claim, f.executor.registry(), &f.policy)
            .is_err()
    );
    f.zero();
}
#[test]
fn literal_argv_is_not_interpreted_by_an_implicit_shell() {
    let literal = "$(touch SHOULD_NOT_EXIST); * 'quoted'";
    let mut f = Fixture::new(&format!("echo:{literal}"), 65536, 5000);
    let result = f.run(65536);
    assert!(!result.is_error());
    assert!(String::from_utf8_lossy(&stream(&f.bytes(&result), "stdout")).contains(literal));
    assert!(!f.root.join("cwd/SHOULD_NOT_EXIST").exists());
}
#[test]
fn binary_streams_remain_distinct_inline_and_in_artifacts() {
    for inline in [65536, 8] {
        let mut f = Fixture::new("binary", 65536, 5000);
        let result = f.run(inline);
        assert!(!result.is_error());
        if inline == 8 {
            assert!(matches!(result, ToolResult::Artifact(_)));
        }
        let bytes = f.bytes(&result);
        assert!(
            stream(&bytes, "stdout")
                .windows(4)
                .any(|part| part == [0, 255, 128, 10])
        );
        assert_eq!(stream(&bytes, "stderr"), b"EXACT_STDERR");
    }
}
#[test]
fn nonzero_exit_is_error_with_complete_output() {
    for inline in [65536, 8] {
        let mut f = Fixture::new("exit7", 65536, 5000);
        let result = f.run(inline);
        assert!(result.is_error());
        let bytes = f.bytes(&result);
        let exit: Value = serde_json::from_slice(
            bytes
                .split(|b| *b == b'\n')
                .rfind(|line| !line.is_empty())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(exit["exit_code"], 7);
    }
}
#[test]
fn environment_is_cleared_stdin_closed_and_cwd_bound() {
    let mut env = Fixture::new("env", 65536, 5000);
    assert!(!env.run(65536).is_error());
    let mut cwd = Fixture::new("cwd", 65536, 5000);
    let result = cwd.run(65536);
    assert!(!result.is_error());
    assert!(
        String::from_utf8_lossy(&stream(&cwd.bytes(&result), "stdout"))
            .contains(cwd.root.join("cwd").to_str().unwrap())
    );
}
#[test]
fn precancelled_receipt_never_enters_backend() {
    let mut f = Fixture::new("marker", 65536, 5000);
    let token = CancellationToken::default();
    let job = f.job(token.clone(), 65536);
    token.cancel();
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    assert_eq!(error_outcome(&result)["started"], false);
    f.zero();
    f.no_artifact();
}
#[test]
fn delayed_run_does_not_renew_start_deadline() {
    let mut f = Fixture::new("marker", 65536, 20);
    let job = f.job(CancellationToken::default(), 65536);
    std::thread::sleep(std::time::Duration::from_millis(40));
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    f.zero();
    f.no_artifact();
}
#[test]
fn output_overflow_preserves_started_and_cleanup_status() {
    let mut f = Fixture::new("flood", 512, 5000);
    let result = f.run(512);
    assert!(result.is_error());
    assert_eq!(error_outcome(&result)["started"], true);
    assert_eq!(error_outcome(&result)["cleanup_errors"], json!([]));
    f.no_artifact();
}
#[test]
fn deadline_after_spawn_records_started_and_cleanup() {
    let mut f = Fixture::new("hang", 65536, 1000);
    let result = f.run(65536);
    assert!(result.is_error());
    assert_eq!(error_outcome(&result)["started"], true);
    assert_eq!(error_outcome(&result)["cleanup_errors"], json!([]));
    f.no_artifact();
}
#[test]
fn active_cancellation_stops_owned_work_without_cancelling_another_operation() {
    let mut f = Fixture::new("hang", 65536, 6000);
    let token = CancellationToken::default();
    let job = f.job(token.clone(), 65536);
    let marker = f.root.join("cwd/marker");
    let worker = std::thread::spawn(move || {
        let start = std::time::Instant::now();
        while !marker.exists() {
            assert!(start.elapsed() < std::time::Duration::from_secs(5));
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        token.cancel();
    });
    let result = job.run().complete(&mut f.store).unwrap();
    worker.join().unwrap();
    assert!(result.is_error());
    assert_eq!(error_outcome(&result)["started"], true);
    assert_eq!(error_outcome(&result)["cleanup_errors"], json!([]));
    f.no_artifact();
    let mut other = Fixture::new("marker", 65536, 5000);
    assert!(!other.run(65536).is_error());
}
#[test]
fn artifact_registration_failure_preserves_process_outcome() {
    let mut f = Fixture::new("marker", 65536, 5000);
    let completed = f.job(CancellationToken::default(), 8).run();
    assert_eq!(completed.outcome()["started"], true);
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_shell_artifact BEFORE INSERT ON events WHEN NEW.kind='artifact.created' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let error = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(error.outcome["started"], true);
    assert!(error.result.is_none());
    assert_eq!(
        std::fs::read(f.root.join("cwd/marker")).unwrap(),
        b"executed"
    );
    assert!(f.store.claim_tool("s", "c").is_err());
}
#[test]
fn completion_failure_preserves_result_and_does_not_replay() {
    let mut f = Fixture::new("marker", 65536, 5000);
    let completed = f.job(CancellationToken::default(), 65536).run();
    let db = rusqlite::Connection::open(f.store.path()).unwrap();
    db.execute_batch("CREATE TRIGGER fail_shell_complete BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    let error = completed.complete(&mut f.store).unwrap_err();
    assert_eq!(error.outcome["started"], true);
    assert!(error.result.is_some());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    assert!(f.store.claim_tool("s", "c").is_err());
    db.execute_batch("DROP TRIGGER fail_shell_complete;")
        .unwrap();
    drop(db);
    drop(f.artifacts);
    drop(f.store);
    let mut reopened = SessionStore::open(f.root.join("db")).unwrap();
    assert_eq!(
        reopened.tool_state("s", "c").unwrap().as_deref(),
        Some("OutcomeUnknown")
    );
    assert!(reopened.claim_tool("s", "c").is_err());
    assert_eq!(
        std::fs::read(f.root.join("cwd/marker")).unwrap(),
        b"executed"
    );
}
#[test]
fn completed_artifact_survives_reopen_with_no_new_execution() {
    let mut f = Fixture::new("binary", 65536, 5000);
    let result = f.run(8);
    let expected = f.bytes(&result);
    let ToolResult::Artifact(a) = result else {
        panic!("artifact required")
    };
    drop(f.artifacts);
    drop(f.store);
    let mut reopened = SessionStore::open(f.root.join("db")).unwrap();
    let artifacts = ArtifactStore::open(
        &mut reopened,
        f.root.join("artifacts"),
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
    assert!(reopened.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
}
#[test]
fn legacy_entrypoint_has_no_marker_effect() {
    let f = Fixture::new("marker", 65536, 5000);
    let executable = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
    let tool = ShellRunTool::new(
        ShellScope::new(vec![executable], vec![f.root.join("cwd")], 65536).unwrap(),
    );
    assert!(tool.execute(&f.request).is_error());
    f.zero();
}

#[test]
fn policy_approval_does_not_override_executable_scope() {
    let mut f = Fixture::new("marker", 65536, 5000);
    f.request.call.arguments.insert(
        "executable".into(),
        json!(std::fs::canonicalize("/bin/sh").unwrap().to_str().unwrap()),
    );
    f.request
        .call
        .arguments
        .insert("args".into(), json!(["-c", "touch marker"]));
    f.policy = DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "shell.run".into(),
            allowed_arguments: vec![f.request.call.arguments.clone()],
            require_approval: false,
            timeout_ms: 5000,
            max_output_bytes: 65536,
        }],
    })
    .unwrap();
    let result = f.run(65536);
    assert!(result.is_error());
    assert_eq!(error_outcome(&result)["started"], false);
    assert!(!f.root.join("cwd/marker").exists());
    f.no_artifact();
}
#[test]
fn unsupported_backend_cannot_fall_back_to_legacy_execution() {
    struct Legacy;
    impl ToolImplementation for Legacy {
        fn execute(&self, _: &ToolInvocation) -> ToolResult {
            panic!("legacy must not run")
        }
    }
    let mut f = Fixture::new("marker", 65536, 5000);
    let definition = f
        .executor
        .registry()
        .definition("shell.run")
        .unwrap()
        .clone();
    let mut executor = ReceiptExecutor::new();
    executor.register(definition, Arc::new(Legacy)).unwrap();
    f.executor = executor;
    let result = f.run(65536);
    assert!(result.is_error());
    assert_eq!(error_outcome(&result)["started"], false);
    assert!(!f.root.join("cwd/marker").exists());
    f.no_artifact();
}
