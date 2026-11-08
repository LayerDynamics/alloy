#![cfg(unix)]
use harness_core::{
    EventKind, SessionStore,
    artifact::{ArtifactLimits, ArtifactStore},
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::ReceiptExecutor,
};
use std::{io::Read, sync::Arc};
use tool_registry::{
    builtins::fs::{FileReadTool, ReadScope},
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::CancellationToken,
    result::ToolResult,
};
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    artifacts: ArtifactStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    request: ToolInvocation,
}
impl Fixture {
    fn new(data: &[u8], maximum: u64, length: u64, path: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("root")).unwrap();
        std::fs::write(dir.path().join("root/data"), data).unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let artifacts = ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        let mut executor = ReceiptExecutor::new();
        let definition = tool_registry::builtins::definitions(60000, maximum)
            .unwrap()
            .into_iter()
            .find(|d| d.id == "fs.read")
            .unwrap();
        executor
            .register(
                definition,
                Arc::new(
                    FileReadTool::new(ReadScope::open(dir.path().join("root")).unwrap(), 128)
                        .unwrap(),
                ),
            )
            .unwrap();
        let arguments = serde_json::json!({"path":path,"offset":0,"length":length})
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
                tool_id: "fs.read".into(),
                arguments: arguments.clone(),
            },
        };
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "fs.read".into(),
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
        }
    }
    fn job(
        &mut self,
        token: CancellationToken,
    ) -> harness_core::tool_execution::file_read::FileReadExecution {
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
            .prepare_file_read(&self.store, receipt, &self.artifacts, token, 128)
            .unwrap()
    }
    fn payload(&self, result: &ToolResult) -> Vec<u8> {
        match result {
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
            ToolResult::Inline(i) => i.content["text"].as_str().unwrap().as_bytes().to_vec(),
        }
    }
    fn no_artifacts(&self) {
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
    }
}
#[test]
fn complete_utf8_stays_inline_without_artifact() {
    let mut f = Fixture::new("résumé 雪".as_bytes(), 4096, 64, "data");
    let completed = f.job(CancellationToken::default()).run();
    let result = completed.complete(&mut f.store).unwrap();
    assert!(matches!(result, ToolResult::Inline(_)));
    assert!(!result.is_error());
    assert_eq!(f.payload(&result), "résumé 雪".as_bytes());
    f.no_artifacts();
}
#[test]
fn large_and_binary_reads_roundtrip_exact_bytes() {
    for bytes in [vec![b'x'; 150000], vec![0, 255, 128]] {
        let mut f = Fixture::new(&bytes, 200000, bytes.len() as u64, "data");
        let completed = f.job(CancellationToken::default()).run();
        let result = completed.complete(&mut f.store).unwrap();
        assert!(matches!(result, ToolResult::Artifact(_)));
        assert!(!result.is_error());
        assert_eq!(f.payload(&result), bytes);
        assert!(f.store.claim_tool("s", "c").is_err());
    }
}
#[test]
fn json_expansion_promotes_instead_of_truncating() {
    let bytes = vec![b'\n'; 80];
    let mut f = Fixture::new(&bytes, 4096, 80, "data");
    let completed = f.job(CancellationToken::default()).run();
    let result = completed.complete(&mut f.store).unwrap();
    assert!(matches!(result, ToolResult::Artifact(_)));
    assert_eq!(f.payload(&result), bytes);
}
#[test]
fn exact_raw_limit_is_valid_but_one_over_fails() {
    for (length, success) in [(128, true), (129, false)] {
        let mut f = Fixture::new(&[b'x'; 129], 128, length, "data");
        let completed = f.job(CancellationToken::default()).run();
        let result = completed.complete(&mut f.store).unwrap();
        assert_eq!(!result.is_error(), success);
        if success {
            assert_eq!(f.payload(&result), vec![b'x'; 128]);
        } else {
            f.no_artifacts();
        }
    }
}
#[test]
fn short_eof_returns_complete_bytes_not_requested_length() {
    let mut f = Fixture::new(b"", 4096, 1000, "data");
    let completed = f.job(CancellationToken::default()).run();
    let result = completed.complete(&mut f.store).unwrap();
    assert_eq!(f.payload(&result), b"");
    f.no_artifacts();
}
#[test]
fn cancellation_before_read_discards_reserved_output() {
    let mut f = Fixture::new(&[255; 1000], 4096, 1000, "data");
    let token = CancellationToken::default();
    let job = f.job(token.clone());
    token.cancel();
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    f.no_artifacts();
}
#[test]
fn traversal_does_not_publish_output() {
    let mut f = Fixture::new(b"data", 4096, 4, "../outside");
    std::fs::write(f.dir.path().join("outside"), b"SECRET").unwrap();
    let completed = f.job(CancellationToken::default()).run();
    let result = completed.complete(&mut f.store).unwrap();
    assert!(result.is_error());
    f.no_artifacts();
}
#[test]
fn artifact_metadata_failure_never_completes_or_replays() {
    let mut f = Fixture::new(&[255; 1000], 4096, 1000, "data");
    let completed = f.job(CancellationToken::default()).run();
    let probe = rusqlite::Connection::open(f.store.path()).unwrap();
    probe.execute_batch("CREATE TRIGGER reject_read_artifact BEFORE INSERT ON events WHEN NEW.kind='artifact.created' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    assert!(completed.complete(&mut f.store).is_err());
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Started")
    );
    assert!(f.store.claim_tool("s", "c").is_err());
}
#[test]
fn completion_failure_does_not_reauthorize_read() {
    let mut f = Fixture::new(&[255; 1000], 4096, 1000, "data");
    let completed = f.job(CancellationToken::default()).run();
    let probe = rusqlite::Connection::open(f.store.path()).unwrap();
    probe.execute_batch("CREATE TRIGGER reject_read_completion BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture'); END;").unwrap();
    assert!(completed.complete(&mut f.store).is_err());
    assert!(f.store.claim_tool("s", "c").is_err());
}

#[test]
fn expired_start_deadline_never_yields_success() {
    let mut f = Fixture::new(&[255; 1000], 4096, 1000, "data");
    f.policy = DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "fs.read".into(),
            allowed_arguments: vec![f.request.call.arguments.clone()],
            require_approval: false,
            timeout_ms: 20,
            max_output_bytes: 4096,
        }],
    })
    .unwrap();
    let job = f.job(CancellationToken::default());
    std::thread::sleep(std::time::Duration::from_millis(40));
    let result = job.run().complete(&mut f.store).unwrap();
    assert!(result.is_error());
    f.no_artifacts();
}
#[test]
fn artifact_bytes_and_completion_survive_reopen() {
    let bytes = vec![255; 1000];
    let mut f = Fixture::new(&bytes, 4096, 1000, "data");
    let completed = f.job(CancellationToken::default()).run();
    let result = completed.complete(&mut f.store).unwrap();
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
    let ToolResult::Artifact(a) = result else {
        panic!("artifact required")
    };
    let mut verified = artifacts
        .lookup(&store, "s", "c", &a.artifact_ref)
        .unwrap()
        .verify()
        .unwrap();
    let mut restored = Vec::new();
    verified.read_to_end(&mut restored).unwrap();
    assert_eq!(restored, bytes);
    assert_eq!(
        store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert!(store.claim_tool("s", "c").is_err());
}
