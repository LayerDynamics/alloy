#![cfg(any(target_os = "linux", target_os = "macos"))]
//! W4 builtin-only P2-09 coverage; W5/W6 pause orchestration remains pending.
use harness_core::{
    SessionStore,
    artifact::{ArtifactLimits, ArtifactStore},
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::{IntentAdmission, StartedInvocation},
    tool_execution::{BuiltinImplementations, ReceiptExecutor},
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tool_registry::{
    builtins::{
        fs::{
            FileListTool, FileReadTool, FileWriteTool, ListReport, ReadError, ReadReport,
            ReadScope, WriteFailure, WriteReport, WriteScope,
        },
        http::{HttpDestination, HttpFailure, HttpFetchTool, HttpReport, HttpScope},
        shell::{ShellFailure, ShellReport, ShellRunTool, ShellScope},
    },
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::{CancellationToken, ExecutionContext, ExecutionError, OutputSink},
    registry::ToolImplementation,
    result::ToolResult,
};
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tool-registry/tests/fixtures/http_server.rs"
    ));
}
use fixture::{Server, response};
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
struct Probe {
    inner: Arc<dyn ToolImplementation>,
    db: PathBuf,
    count: Arc<AtomicUsize>,
    cancel: Option<CancellationToken>,
}
impl Probe {
    fn check(&self) {
        let db = rusqlite::Connection::open(&self.db).unwrap();
        db.busy_timeout(Duration::from_secs(1)).unwrap();
        db.execute_batch("BEGIN IMMEDIATE; ROLLBACK;")
            .expect("execution retained a SQLite write transaction");
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}
macro_rules! forward {
    ($name:ident,$report:ty,$error:ty) => {
        fn $name(
            &self,
            i: &ToolInvocation,
            c: &ExecutionContext,
            o: &mut dyn OutputSink,
        ) -> Result<$report, $error> {
            self.check();
            if i.context.session_id == "a"
                && let Some(token) = &self.cancel
            {
                self.inner.$name(
                    i,
                    c,
                    &mut CancelOutput {
                        output: o,
                        token: token.clone(),
                    },
                )
            } else {
                self.inner.$name(i, c, o)
            }
        }
    };
}
impl ToolImplementation for Probe {
    fn execute(&self, _: &ToolInvocation) -> ToolResult {
        panic!("legacy execution forbidden")
    }
    forward!(execute_file_read, ReadReport, ReadError);
    forward!(execute_file_list, ListReport, ReadError);
    forward!(execute_file_write, WriteReport, WriteFailure);
    forward!(execute_shell_run, ShellReport, ShellFailure);
    forward!(execute_http_fetch, HttpReport, HttpFailure);
}
#[derive(Clone, Copy, Debug)]
enum Tool {
    Read,
    List,
    Write,
    Shell,
    Http,
}
impl Tool {
    fn id(self) -> &'static str {
        match self {
            Self::Read => "fs.read",
            Self::List => "fs.list",
            Self::Write => "fs.write",
            Self::Shell => "shell.run",
            Self::Http => "http.fetch",
        }
    }
}
struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    store: SessionStore,
    artifacts: ArtifactStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    token: CancellationToken,
    count: Arc<AtomicUsize>,
    origin: String,
}
impl Fixture {
    fn new(server: &Server, active: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::create_dir(root.join("root")).unwrap();
        for session in ["a", "b"] {
            std::fs::create_dir(root.join("root").join(session)).unwrap();
        }
        std::fs::write(root.join("root/data"), vec![b'x'; 150000]).unwrap();
        let mut store = SessionStore::open(root.join("db")).unwrap();
        for session in ["a", "b"] {
            store.create_session(session, "fixture").unwrap();
        }
        let artifacts = ArtifactStore::open(
            &mut store,
            root.join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        let token = CancellationToken::default();
        let count = Arc::new(AtomicUsize::new(0));
        let wrap = |inner: Arc<dyn ToolImplementation>| -> Arc<dyn ToolImplementation> {
            Arc::new(Probe {
                inner,
                db: root.join("db"),
                count: count.clone(),
                cancel: active.then(|| token.clone()),
            })
        };
        let exe = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
        let executor = ReceiptExecutor::with_builtins(
            10000,
            1048576,
            BuiltinImplementations {
                fs_read: wrap(Arc::new(
                    FileReadTool::new(ReadScope::open(root.join("root")).unwrap(), 4096).unwrap(),
                )),
                fs_list: wrap(Arc::new(FileListTool::new(
                    ReadScope::open(root.join("root")).unwrap(),
                ))),
                fs_write: wrap(Arc::new(FileWriteTool::new(
                    WriteScope::open(root.join("root"), 65536).unwrap(),
                ))),
                shell_run: wrap(Arc::new(ShellRunTool::new(
                    ShellScope::new(vec![exe], vec![root.join("root")], 65536).unwrap(),
                ))),
                http_fetch: wrap(Arc::new(HttpFetchTool::new(
                    HttpScope::new(
                        vec![
                            HttpDestination::new(&server.origin, "127.0.0.1".parse().unwrap())
                                .unwrap(),
                        ],
                        2,
                        4096,
                        8192,
                        Duration::from_secs(2),
                    )
                    .unwrap(),
                ))),
            },
        )
        .unwrap();
        let mut f = Self {
            _dir: dir,
            root,
            store,
            artifacts,
            executor,
            policy: DaemonPolicy::new(PolicyConfig::default()).unwrap(),
            token,
            count,
            origin: server.origin.clone(),
        };
        let rules = [Tool::Read, Tool::List, Tool::Write, Tool::Shell, Tool::Http]
            .into_iter()
            .map(|tool| PolicyRule {
                tool_id: tool.id().into(),
                allowed_arguments: vec![
                    f.request(tool, "a").call.arguments,
                    f.request(tool, "b").call.arguments,
                ],
                require_approval: matches!(tool, Tool::Write | Tool::Shell),
                timeout_ms: 10000,
                max_output_bytes: 1048576,
            })
            .collect();
        f.policy = DaemonPolicy::new(PolicyConfig { rules }).unwrap();
        f
    }
    fn request(&self, tool: Tool, session: &str) -> ToolInvocation {
        let args = match tool {
            Tool::Read => json!({"path":"data","offset":0,"length":150000}),
            Tool::List => json!({"path":".","max_entries":10}),
            Tool::Write => json!({"path":format!("{session}/target"),"content":"exact"}),
            Tool::Shell => {
                json!({"executable":std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap().to_str().unwrap(),
                "args":["--exact","cancellation_child_fixture","--nocapture","--test-threads=1","--skip",if session=="a"{"hang"}else{"marker"}],
                "cwd":self.root.join("root").join(session).to_str().unwrap()})
            }
            Tool::Http => json!({"url":format!("{}/{session}",self.origin),"method":"GET"}),
        };
        ToolInvocation {
            context: ToolCallContext {
                session_id: session.into(),
                turn_id: "t".into(),
                generation_id: "g".into(),
            },
            call: ToolCall {
                call_id: tool.id().into(),
                tool_id: tool.id().into(),
                arguments: args.as_object().unwrap().clone(),
            },
        }
    }
    fn receipt(&mut self, tool: Tool, session: &str) -> StartedInvocation {
        let request = self.request(tool, session);
        let admission = self
            .store
            .admit_tool(self.executor.registry(), &self.policy, request)
            .unwrap();
        if admission == IntentAdmission::AwaitingApproval {
            self.store.decide_tool(session, tool.id(), true).unwrap();
        } else {
            assert_eq!(admission, IntentAdmission::Ready);
        }
        let claim = self.store.claim_tool(session, tool.id()).unwrap();
        self.store
            .start_tool(claim, self.executor.registry(), &self.policy)
            .unwrap()
    }
    fn run(&mut self, tool: Tool, session: &str, token: CancellationToken) -> ToolResult {
        let receipt = self.receipt(tool, session);
        match tool {
            Tool::Read => {
                let done = self
                    .executor
                    .prepare_file_read(&self.store, receipt, &self.artifacts, token, 4096)
                    .unwrap()
                    .run();
                done.complete(&mut self.store).unwrap()
            }
            Tool::List => {
                let done = self
                    .executor
                    .prepare_file_list(&self.store, receipt, &self.artifacts, token, 4096)
                    .unwrap()
                    .run();
                done.complete(&mut self.store).unwrap()
            }
            Tool::Write => {
                let done = self
                    .executor
                    .prepare_file_write(&self.store, receipt, token)
                    .unwrap()
                    .run();
                done.complete(&mut self.store).unwrap()
            }
            Tool::Shell => {
                let done = self
                    .executor
                    .prepare_shell_run(&self.store, receipt, &self.artifacts, token, 4096)
                    .unwrap()
                    .run();
                done.complete(&mut self.store).unwrap()
            }
            Tool::Http => {
                let done = self
                    .executor
                    .prepare_http_fetch(&self.store, receipt, &self.artifacts, token, 4096)
                    .unwrap()
                    .run();
                done.complete(&mut self.store).unwrap()
            }
        }
    }
    fn no_partial(&self) {
        for sub in ["tmp", "objects"] {
            assert_eq!(
                std::fs::read_dir(self.root.join("artifacts").join(sub))
                    .unwrap()
                    .count(),
                0
            );
        }
        for session in ["a", "b"] {
            assert!(
                std::fs::read_dir(self.root.join("root").join(session))
                    .unwrap()
                    .all(|entry| !entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".alloy-write-"))
            );
        }
    }
}
#[test]
fn cancellation_child_fixture() {
    use std::io::Write;
    let args: Vec<_> = std::env::args().collect();
    let Some(mode) = args
        .windows(2)
        .find(|pair| pair[0] == "--skip")
        .map(|pair| pair[1].as_str())
    else {
        return;
    };
    std::fs::write("marker", b"executed").unwrap();
    std::io::stdout().write_all(b"READY\n").unwrap();
    std::io::stdout().flush().unwrap();
    if mode == "hang" {
        std::thread::sleep(Duration::from_secs(30));
    } else {
        assert_eq!(mode, "marker");
    }
}
#[test]
fn tool_cancel_releases_owned_resources() {
    let _http_guard = fixture::http_timing_test_guard();
    for active in [false, true] {
        for tool in [Tool::Read, Tool::List, Tool::Write, Tool::Shell, Tool::Http] {
            let server = Server::new(|_, stream, _| response(stream, 200, "", b"exact"));
            let mut f = Fixture::new(&server, active);
            if !active {
                f.token.cancel();
            }
            let result = f.run(tool, "a", f.token.clone());
            assert!(result.is_error(), "{tool:?} active={active}: {result:?}");
            assert_eq!(f.count.load(Ordering::SeqCst), active as usize);
            if active && matches!(tool, Tool::Write) {
                let ToolResult::Inline(result) = &result else {
                    panic!("inline error")
                };
                assert_eq!(result.content["details"]["publication"], "published");
                assert_eq!(
                    std::fs::read(f.root.join("root/a/target")).unwrap(),
                    b"exact"
                );
            }
            if !active {
                assert!(!f.root.join("root/a/target").exists());
                assert!(!f.root.join("root/a/marker").exists());
                assert!(server.calls.lock().unwrap().is_empty());
            }
            f.no_partial();
            let other = f.run(tool, "b", CancellationToken::default());
            assert!(!other.is_error(), "other session {tool:?}: {other:?}");
            assert_eq!(f.count.load(Ordering::SeqCst), active as usize + 1);
            match tool {
                Tool::Read => {
                    let ToolResult::Artifact(result) = &other else {
                        panic!("large read artifact required")
                    };
                    assert!(
                        f.artifacts
                            .lookup(&f.store, "a", tool.id(), &result.artifact_ref)
                            .is_err()
                    );
                    let mut artifact = f
                        .artifacts
                        .lookup(&f.store, "b", tool.id(), &result.artifact_ref)
                        .unwrap()
                        .verify()
                        .unwrap();
                    let mut bytes = Vec::new();
                    std::io::Read::read_to_end(&mut artifact, &mut bytes).unwrap();
                    assert_eq!(bytes, vec![b'x'; 150000]);
                }
                Tool::List => {
                    let ToolResult::Inline(result) = &other else {
                        panic!("small listing inline required")
                    };
                    assert_eq!(result.content["entries"].as_array().unwrap().len(), 3);
                }
                Tool::Write => assert_eq!(
                    std::fs::read(f.root.join("root/b/target")).unwrap(),
                    b"exact"
                ),
                Tool::Shell => assert_eq!(
                    std::fs::read(f.root.join("root/b/marker")).unwrap(),
                    b"executed"
                ),
                Tool::Http => assert!(
                    server
                        .calls
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|line| line.starts_with("GET /b "))
                ),
            }

            for session in ["a", "b"] {
                assert_eq!(
                    f.store.tool_state(session, tool.id()).unwrap().as_deref(),
                    Some("Completed")
                );
                assert!(f.store.claim_tool(session, tool.id()).is_err());
            }
        }
    }
}
#[test]
fn active_shell_does_not_block_second_session_or_sqlite_writer() {
    let _http_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"exact"));
    let mut f = Fixture::new(&server, false);
    let receipt = f.receipt(Tool::Shell, "a");
    let job = f
        .executor
        .prepare_shell_run(&f.store, receipt, &f.artifacts, f.token.clone(), 4096)
        .unwrap();
    let worker = std::thread::spawn(move || job.run());
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let started = std::time::Instant::now();
        while !f.root.join("root/a/marker").exists() {
            assert!(started.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(2));
        }
        let db = rusqlite::Connection::open(f.store.path()).unwrap();
        db.busy_timeout(Duration::from_secs(1)).unwrap();
        db.execute_batch("BEGIN IMMEDIATE; UPDATE sessions SET profile_id=profile_id WHERE session_id='b'; COMMIT;").unwrap();
        assert!(
            !f.run(Tool::Read, "b", CancellationToken::default())
                .is_error()
        );
    }));
    f.token.cancel();
    let done = worker.join().unwrap();
    assert!(done.complete(&mut f.store).unwrap().is_error());
    if let Err(error) = checked {
        std::panic::resume_unwind(error);
    }
}
#[test]
fn dropping_prepared_read_releases_output_and_database_lease() {
    let _http_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"exact"));
    let mut f = Fixture::new(&server, false);
    let receipt = f.receipt(Tool::Read, "a");
    let job = f
        .executor
        .prepare_file_read(
            &f.store,
            receipt,
            &f.artifacts,
            CancellationToken::default(),
            4096,
        )
        .unwrap();
    let path = f.store.path().to_owned();
    drop(f.artifacts);
    drop(f.store);
    assert!(SessionStore::open(&path).is_err());
    drop(job);
    let reopened = SessionStore::open(&path).unwrap();
    assert_eq!(
        reopened.tool_state("a", "fs.read").unwrap().as_deref(),
        Some("OutcomeUnknown")
    );
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
    assert_eq!(
        std::fs::read_dir(f.root.join("artifacts/tmp"))
            .unwrap()
            .count(),
        0
    );
}
