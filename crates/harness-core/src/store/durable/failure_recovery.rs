//! Step C test-only fault and recovery coverage. No builtin tool is executed.
use super::*;
use crate::{
    artifact::{ArtifactLimits, ArtifactStore},
    policy::{PolicyConfig, PolicyRule},
};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tool_registry::{
    call::{ToolCall, ToolCallContext},
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
    registry::ToolImplementation,
    result::{ArtifactToolResult, InlineToolResult, ToolResult},
};

// Called ONLY at #[cfg(test)] checkpoint sites around the actual SQLite commits.
pub(super) fn checkpoint(stage: &str) {
    if std::env::var("ALLOY_W3_C_CRASH_STAGE").ok().as_deref() == Some(stage) {
        std::process::exit(86);
    }
}

struct Counter {
    count: Arc<AtomicUsize>,
    marker: PathBuf,
}
impl ToolImplementation for Counter {
    fn execute(&self, invocation: &ToolInvocation) -> ToolResult {
        self.count.fetch_add(1, Ordering::SeqCst);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.marker)
            .unwrap();
        file.write_all(b"effect\n").unwrap();
        file.sync_all().unwrap();
        inline(&invocation.call.call_id)
    }
}
fn inline(call: &str) -> ToolResult {
    ToolResult::Inline(InlineToolResult {
        call_id: call.into(),
        content: serde_json::json!({"fixture":"done"}),
        is_error: false,
    })
}
fn definition() -> ToolDefinition {
    ToolDefinition {
        id: "fixture.effect".into(),
        version: "1.0.0".into(),
        description: "Counter/marker fixture; not a builtin".into(),
        input_schema: serde_json::json!({"type":"object","additionalProperties":false}),
        origin: ToolOrigin::Builtin,
        policy: ToolPolicy {
            effect: ToolEffect::Destructive,
            approval: ApprovalRequirement::Always,
            timeout_ms: 5000,
            max_output_bytes: 65536,
        },
    }
}
fn policy() -> DaemonPolicy {
    DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "fixture.effect".into(),
            allowed_arguments: vec![serde_json::Map::new()],
            require_approval: false,
            timeout_ms: 4000,
            max_output_bytes: 32768,
        }],
    })
    .unwrap()
}
fn request(call: &str) -> ToolInvocation {
    ToolInvocation {
        context: ToolCallContext {
            session_id: "s".into(),
            turn_id: "t".into(),
            generation_id: "g".into(),
        },
        call: ToolCall {
            call_id: call.into(),
            tool_id: "fixture.effect".into(),
            arguments: serde_json::Map::new(),
        },
    }
}
fn registry(def: ToolDefinition, count: &Arc<AtomicUsize>, marker: PathBuf) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            def,
            Arc::new(Counter {
                count: Arc::clone(count),
                marker,
            }),
        )
        .unwrap();
    registry
}
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    registry: ToolRegistry,
    policy: DaemonPolicy,
    count: Arc<AtomicUsize>,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (store, registry, policy, count) = open_fixture(dir.path());
        Self {
            dir,
            store,
            registry,
            policy,
            count,
        }
    }
    fn admit(&mut self) {
        self.store
            .admit_tool(&self.registry, &self.policy, request("c"))
            .unwrap();
    }
    fn ready(&mut self) {
        self.admit();
        self.store.decide_tool("s", "c", true).unwrap();
    }
    fn claim(&mut self) -> AuthorizedInvocation {
        self.ready();
        self.store.claim_tool("s", "c").unwrap()
    }
    fn started(&mut self) -> StartedInvocation {
        let claim = self.claim();
        self.store
            .start_tool(claim, &self.registry, &self.policy)
            .unwrap()
    }
    fn effect(&self, receipt: &StartedInvocation) -> ToolResult {
        Counter {
            count: Arc::clone(&self.count),
            marker: self.dir.path().join("effects"),
        }
        .execute(receipt.binding().invocation())
    }
    fn assert_effects(&self, expected: usize) {
        assert_eq!(self.count.load(Ordering::SeqCst), expected);
        let marker = std::fs::read(self.dir.path().join("effects")).unwrap_or_default();
        assert_eq!(marker, b"effect\n".repeat(expected));
    }
}
fn open_fixture(dir: &Path) -> (SessionStore, ToolRegistry, DaemonPolicy, Arc<AtomicUsize>) {
    let mut store = SessionStore::open(dir.join("events.db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let registry = registry(definition(), &count, dir.join("effects"));
    (store, registry, policy(), count)
}
#[derive(Clone, Copy, Debug)]
enum Failure {
    Statement,
    Commit,
}
const FAILURES: [Failure; 2] = [Failure::Statement, Failure::Commit];
fn deferred_tables(conn: &Connection) {
    assert_eq!(
        conn.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    conn.execute_batch("CREATE TEMP TABLE w3_c_parent(id INTEGER PRIMARY KEY);
CREATE TEMP TABLE w3_c_child(id INTEGER,parent_id INTEGER REFERENCES w3_c_parent(id) DEFERRABLE INITIALLY DEFERRED);").unwrap();
}
fn install_fault(conn: &Connection, kind: &str, failure: Failure) {
    match failure {
        Failure::Statement => conn.execute_batch(&format!("CREATE TEMP TRIGGER w3_c_fault BEFORE INSERT ON main.events WHEN NEW.kind='{kind}' BEGIN SELECT RAISE(ABORT,'w3_c_statement_failure'); END;")).unwrap(),
        Failure::Commit => {
            deferred_tables(conn);
            conn.execute_batch(&format!("CREATE TEMP TRIGGER w3_c_fault AFTER INSERT ON main.events WHEN NEW.kind='{kind}' BEGIN INSERT INTO w3_c_child VALUES(NEW.id,-1); END;")).unwrap();
        }
    }
}
fn clear_fault(conn: &Connection) {
    conn.execute_batch("DROP TRIGGER IF EXISTS temp.w3_c_fault;DROP TABLE IF EXISTS temp.w3_c_child;DROP TABLE IF EXISTS temp.w3_c_parent;").unwrap();
}
fn assert_fault<T>(result: HarnessResult<T>, failure: Failure) {
    let message = match result {
        Ok(_) => panic!("injected {failure:?} did not fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains(match failure {
            Failure::Statement => "w3_c_statement_failure",
            Failure::Commit => "FOREIGN KEY constraint failed",
        }),
        "unexpected failure: {message}"
    );
}
fn assert_unchanged(f: &Fixture, before: &[EventRecord], expected: Option<&str>) {
    assert!(
        f.store.conn.is_autocommit(),
        "failed transaction must not remain open"
    );
    assert_eq!(f.store.list_events("s").unwrap().as_slice(), before);
    assert_eq!(f.store.tool_state("s", "c").unwrap().as_deref(), expected);
}
fn rebuild(
    binding: &InvocationBinding,
    registry: &ToolRegistry,
    policy: &DaemonPolicy,
) -> InvocationBinding {
    let created = binding.snapshot()["created_at_unix_ms"].as_i64().unwrap();
    let epoch = binding.snapshot()["daemon_epoch"].as_str().unwrap();
    let decision = policy
        .evaluate(registry, binding.invocation().clone(), epoch, created)
        .unwrap();
    let rebuilt = match decision {
        PolicyDecision::AllowWithoutPrompt(b) | PolicyDecision::RequireApproval(b) => *b,
        _ => panic!("test binding was denied"),
    };
    assert_eq!(rebuilt.digest(), binding.digest());
    rebuilt
}

#[test]
fn intent_statement_and_commit_failures_publish_no_authority() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.requested", failure);
        assert_fault(
            f.store.admit_tool(&f.registry, &f.policy, request("c")),
            failure,
        );
        assert_unchanged(&f, &before, None);
        assert!(f.store.live.is_empty());
        f.assert_effects(0);
        assert!(f.store.claim_tool("s", "c").is_err());
        clear_fault(&f.store.conn);
        // No committed intent reserved the failed request identity.
        f.admit();
        assert_eq!(
            f.store.tool_state("s", "c").unwrap().as_deref(),
            Some("AwaitingApproval")
        );
        f.assert_effects(0);
    }
}
#[test]
fn denied_intent_failure_rolls_back_request_and_terminal_event() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let before = f.store.list_events("s").unwrap();
        let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
        install_fault(&f.store.conn, "tool.completed", failure);
        assert_fault(
            f.store.admit_tool(&f.registry, &deny, request("c")),
            failure,
        );
        assert_unchanged(&f, &before, None);
        assert!(f.store.live.is_empty());
        f.assert_effects(0);
    }
}
#[test]
fn allow_and_deny_statement_and_commit_failures_are_atomic() {
    for failure in FAILURES {
        for allow in [false, true] {
            let mut f = Fixture::new();
            f.admit();
            let before = f.store.list_events("s").unwrap();
            install_fault(&f.store.conn, "tool.approved", failure);
            assert_fault(f.store.decide_tool("s", "c", allow), failure);
            assert_unchanged(&f, &before, Some("AwaitingApproval"));
            f.assert_effects(0);
            assert!(f.store.claim_tool("s", "c").is_err());
            clear_fault(&f.store.conn);
            f.store.decide_tool("s", "c", allow).unwrap();
            assert_eq!(
                f.store.tool_state("s", "c").unwrap().as_deref(),
                Some(if allow { "Ready" } else { "Denied" })
            );
            assert_eq!(
                f.store
                    .list_events("s")
                    .unwrap()
                    .iter()
                    .filter(|e| e.kind == EventKind::ToolApproved)
                    .count(),
                1
            );
        }
    }
}
#[test]
fn claim_statement_and_commit_failures_return_no_handle() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        f.ready();
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.requested", failure);
        assert_fault(f.store.claim_tool("s", "c"), failure);
        assert_unchanged(&f, &before, Some("Ready"));
        f.assert_effects(0);
        assert!(f.store.live.contains_key(&("s".into(), "c".into())));
        assert!(f.store.claim_deadlines.is_empty());
        clear_fault(&f.store.conn);
        let claim = f.store.claim_tool("s", "c").unwrap();
        let started = f.store.start_tool(claim, &f.registry, &f.policy).unwrap();
        let result = f.effect(&started);
        f.store.complete_tool(started, result).unwrap();
        f.assert_effects(1);
    }
}
#[test]
fn start_statement_and_commit_failures_return_no_execution_receipt() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let claim = f.claim();
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.started", failure);
        assert_fault(f.store.start_tool(claim, &f.registry, &f.policy), failure);
        assert_unchanged(&f, &before, Some("Claimed"));
        f.assert_effects(0);
        assert!(f.store.claim_tool("s", "c").is_err());
        clear_fault(&f.store.conn);
        f.store
            .withdraw_tools("s", "failed_start_cancelled")
            .unwrap();
        f.assert_effects(0);
    }
}
#[test]
fn completion_statement_and_commit_failures_never_regrant_execution() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let started = f.started();
        let result = f.effect(&started);
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.completed", failure);
        assert_fault(f.store.complete_tool(started, result), failure);
        assert_unchanged(&f, &before, Some("Started"));
        f.assert_effects(1);
        let saved: Option<String> = f
            .store
            .conn
            .query_row(
                "SELECT result_json FROM tool_intents WHERE call_id='c'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(saved.is_none());
        assert!(f.store.claim_tool("s", "c").is_err());
        clear_fault(&f.store.conn);
        let path = f.store.path().to_owned();
        drop(f.store);
        let mut reopened = SessionStore::open(&path).unwrap();
        assert_eq!(
            reopened.tool_state("s", "c").unwrap().as_deref(),
            Some("OutcomeUnknown")
        );
        assert!(reopened.claim_tool("s", "c").is_err());
        assert_eq!(
            std::fs::read(f.dir.path().join("effects")).unwrap(),
            b"effect\n"
        );
    }
}
#[test]
fn expire_failure_is_atomic_and_expired_authority_never_escapes() {
    for failure in FAILURES {
        for phase in ["AwaitingApproval", "Ready", "Claimed"] {
            let mut f = Fixture::new();
            f.admit();
            let mut held = None;
            if phase != "AwaitingApproval" {
                f.store.decide_tool("s", "c", true).unwrap();
            }
            if phase == "Claimed" {
                held = Some(f.store.claim_tool("s", "c").unwrap());
                f.store
                    .claim_deadlines
                    .insert(("s".into(), "c".into()), Instant::now());
            } else {
                f.store
                    .live
                    .get_mut(&("s".into(), "c".into()))
                    .unwrap()
                    .deadline = Instant::now();
            }
            let before = f.store.list_events("s").unwrap();
            install_fault(&f.store.conn, "tool.completed", failure);
            assert_fault(f.store.expire_tool_intents(), failure);
            assert_unchanged(&f, &before, Some(phase));
            assert!(f.store.claim_tool("s", "c").is_err());
            assert!(f.store.decide_tool("s", "c", true).is_err());
            f.assert_effects(0);
            clear_fault(&f.store.conn);
            f.store.expire_tool_intents().unwrap();
            assert_eq!(
                f.store.tool_state("s", "c").unwrap().as_deref(),
                Some("Expired")
            );
            if let Some(claim) = held {
                assert!(f.store.start_tool(claim, &f.registry, &f.policy).is_err());
            }
            f.assert_effects(0);
        }
    }
}
#[test]
fn withdrawal_failure_does_not_falsely_acknowledge_revocation() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let claim = f.claim();
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.completed", failure);
        assert_fault(f.store.withdraw_tools("s", "cancel"), failure);
        assert_unchanged(&f, &before, Some("Claimed"));
        f.assert_effects(0);
        // Caller must handle a failed cancellation, not claim it committed.
        clear_fault(&f.store.conn);
        f.store.withdraw_tools("s", "cancel").unwrap();
        assert!(f.store.start_tool(claim, &f.registry, &f.policy).is_err());
        f.assert_effects(0);
    }
}
#[test]
fn pause_failure_rolls_back_status_and_withdrawal_together() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let claim = f.claim();
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.completed", failure);
        assert_fault(f.store.set_status("s", SessionStatus::Paused), failure);
        assert_unchanged(&f, &before, Some("Claimed"));
        assert_eq!(
            f.store.get_session("s").unwrap().unwrap().status,
            SessionStatus::Active
        );
        clear_fault(&f.store.conn);
        f.store.set_status("s", SessionStatus::Paused).unwrap();
        assert!(f.store.start_tool(claim, &f.registry, &f.policy).is_err());
        f.assert_effects(0);
    }
}
#[test]
fn veto_failure_consumes_claim_without_restoring_ready() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let claim = f.claim();
        let before = f.store.list_events("s").unwrap();
        install_fault(&f.store.conn, "tool.completed", failure);
        assert_fault(f.store.reject_tool(claim, "hook_veto"), failure);
        assert_unchanged(&f, &before, Some("Claimed"));
        assert!(f.store.claim_tool("s", "c").is_err());
        f.assert_effects(0);
        clear_fault(&f.store.conn);
        f.store
            .withdraw_tools("s", "failed_veto_cancelled")
            .unwrap();
        f.assert_effects(0);
    }
}
#[test]
fn duplicate_start_attempts_do_not_admit_a_second_effect() {
    let mut f = Fixture::new();
    let claim = f.claim();
    // Deliberately reconstruct equal private handles INSIDE cfg(test), to test
    // the database guard independently of Rust's public single-owner API.
    let duplicate = AuthorizedInvocation {
        binding: rebuild(&claim.binding, &f.registry, &f.policy),
    };
    let terminal_duplicate = AuthorizedInvocation {
        binding: rebuild(&claim.binding, &f.registry, &f.policy),
    };
    let started = f.store.start_tool(claim, &f.registry, &f.policy).unwrap();
    let result = f.effect(&started);
    let before = f.store.list_events("s").unwrap();
    assert!(
        f.store
            .start_tool(duplicate, &f.registry, &f.policy)
            .is_err()
    );
    assert_eq!(f.store.list_events("s").unwrap(), before);
    f.assert_effects(1);
    f.store.complete_tool(started, result).unwrap();
    let before = f.store.list_events("s").unwrap();
    assert!(
        f.store
            .start_tool(terminal_duplicate, &f.registry, &f.policy)
            .is_err()
    );
    assert_eq!(f.store.list_events("s").unwrap(), before);
    f.assert_effects(1);
    assert_eq!(
        before
            .iter()
            .filter(|e| e.kind == EventKind::ToolStarted)
            .count(),
        1
    );
}
#[test]
fn duplicate_completion_attempt_cannot_append_a_second_terminal_result() {
    let mut f = Fixture::new();
    let started = f.started();
    let duplicate = StartedInvocation {
        binding: rebuild(&started.binding, &f.registry, &f.policy),
        execution_started_at: started.execution_started_at,
    };
    let result = f.effect(&started);
    f.store.complete_tool(started, result.clone()).unwrap();
    let before = f.store.list_events("s").unwrap();
    assert!(f.store.complete_tool(duplicate, result).is_err());
    assert_eq!(f.store.list_events("s").unwrap(), before);
    f.assert_effects(1);
    assert_eq!(
        before
            .iter()
            .filter(|e| e.kind == EventKind::ToolCompleted)
            .count(),
        1
    );
}
#[test]
fn every_registered_definition_change_invalidates_final_start() {
    for field in 0..8 {
        let mut f = Fixture::new();
        let claim = f.claim();
        let mut changed = definition();
        match field {
            0 => changed.version = "1.0.1".into(),
            1 => changed.description.push_str(" changed"),
            2 => {
                changed.input_schema["properties"] =
                    serde_json::json!({"optional":{"type":"string"}})
            }
            3 => changed.origin = ToolOrigin::Plugin,
            4 => changed.policy.timeout_ms += 1,
            5 => changed.policy.max_output_bytes += 1,
            6 => changed.policy.effect = ToolEffect::Write,
            _ => changed.policy.approval = ApprovalRequirement::Never,
        }
        let altered = registry(changed, &f.count, f.dir.path().join("effects"));
        assert!(f.store.start_tool(claim, &altered, &f.policy).is_err());
        assert_eq!(
            f.store.tool_state("s", "c").unwrap().as_deref(),
            Some("Rejected")
        );
        assert!(
            f.store
                .list_events("s")
                .unwrap()
                .iter()
                .all(|e| e.kind != EventKind::ToolStarted)
        );
        f.assert_effects(0);
    }
}
#[test]
fn policy_change_rejection_has_zero_marker_effects() {
    let mut f = Fixture::new();
    let claim = f.claim();
    let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
    assert!(f.store.start_tool(claim, &f.registry, &deny).is_err());
    f.assert_effects(0);
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Rejected")
    );
}
#[test]
fn claimed_hook_gap_retains_no_sqlite_transaction() {
    let mut f = Fixture::new();
    let claim = f.claim();
    assert!(f.store.conn.is_autocommit());
    assert_eq!(
        f.store.get_session("s").unwrap().unwrap().status,
        SessionStatus::Active
    );
    f.store.withdraw_tools("s", "cancel_during_hook").unwrap();
    assert!(f.store.start_tool(claim, &f.registry, &f.policy).is_err());
    f.assert_effects(0);
}

#[test]
fn recovery_statement_and_commit_failures_preserve_prior_epoch_and_history() {
    for failure in FAILURES {
        let mut f = Fixture::new();
        let started = f.started();
        let _result = f.effect(&started);
        drop(started);
        let old_epoch = f.store.daemon_epoch().to_owned();
        let before = f.store.list_events("s").unwrap();
        let db = f.store.path().to_owned();
        match failure {
            Failure::Statement=>f.store.conn.execute_batch("CREATE TRIGGER w3_c_recovery_fault BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'w3_c_statement_failure'); END;").unwrap(),
            Failure::Commit=>f.store.conn.execute_batch("CREATE TABLE w3_c_recovery_parent(id INTEGER PRIMARY KEY);CREATE TABLE w3_c_recovery_child(id INTEGER,parent_id INTEGER REFERENCES w3_c_recovery_parent(id) DEFERRABLE INITIALLY DEFERRED);CREATE TRIGGER w3_c_recovery_fault AFTER INSERT ON events WHEN NEW.kind='tool.completed' BEGIN INSERT INTO w3_c_recovery_child VALUES(NEW.id,-1); END;").unwrap(),
        }
        drop(f.store);
        assert_fault(SessionStore::open(&db), failure);
        let connection = Connection::open(&db).unwrap();
        let epoch: String = connection
            .query_row("SELECT daemon_epoch FROM runtime_metadata", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(epoch, old_epoch);
        let state: String = connection
            .query_row(
                "SELECT state FROM tool_intents WHERE call_id='c'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(state, "Started");
        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, before.len() as i64);
        connection.execute_batch("DROP TRIGGER w3_c_recovery_fault;DROP TABLE IF EXISTS w3_c_recovery_child;DROP TABLE IF EXISTS w3_c_recovery_parent;").unwrap();
        drop(connection);
        let mut reopened = SessionStore::open(&db).unwrap();
        assert_ne!(reopened.daemon_epoch(), old_epoch);
        assert_eq!(
            reopened.tool_state("s", "c").unwrap().as_deref(),
            Some("OutcomeUnknown")
        );
        assert!(reopened.claim_tool("s", "c").is_err());
        let after = reopened.list_events("s").unwrap();
        assert_eq!(&after[..before.len()], before.as_slice());
        assert_eq!(
            std::fs::read(f.dir.path().join("effects")).unwrap(),
            b"effect\n"
        );
    }
}
#[test]
fn migration_commit_failure_rolls_back_ddl_version_and_raw_history() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("v1.db");
    let _owner = owned_file(&db).unwrap();
    let mut connection = Connection::open(&db).unwrap();
    connection
        .pragma_update(None, "foreign_keys", "ON")
        .unwrap();
    connection.execute_batch("CREATE TABLE meta(schema_version INTEGER NOT NULL,spec_version TEXT NOT NULL);INSERT INTO meta VALUES(1,'original');CREATE TABLE sessions(session_id TEXT PRIMARY KEY,status TEXT,created_ts INTEGER,updated_ts INTEGER,profile_id TEXT);INSERT INTO sessions VALUES('s','active',1,1,'p');CREATE TABLE events(id INTEGER PRIMARY KEY,session_id TEXT,seq INTEGER,ts INTEGER,kind TEXT,payload_json TEXT,artifact_refs TEXT);CREATE UNIQUE INDEX idx_events_session_seq ON events(session_id,seq);INSERT INTO events VALUES(19,'s',1,123,'user.message','{ \"old\": true }','[ \"legacy\" ]');").unwrap();
    deferred_tables(&connection);
    connection.execute_batch("CREATE TEMP TRIGGER w3_c_migration_fault AFTER UPDATE ON main.meta BEGIN INSERT INTO w3_c_child VALUES(1,-1); END;").unwrap();
    assert_fault(migrate(&mut connection), Failure::Commit);
    assert!(connection.is_autocommit());
    let version: i64 = connection
        .query_row("SELECT schema_version FROM meta", [], |r| r.get(0))
        .unwrap();
    assert_eq!(version, 1);
    let count:i64=connection.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name IN ('tool_intents','tool_artifacts','runtime_metadata')",[],|r|r.get(0)).unwrap();
    assert_eq!(count, 0);
    let raw: (i64, i64, i64, String, String) = connection
        .query_row(
            "SELECT id,seq,ts,payload_json,artifact_refs FROM events",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(
        raw,
        (
            19,
            1,
            123,
            "{ \"old\": true }".into(),
            "[ \"legacy\" ]".into()
        )
    );
}

#[test]
fn crash_child_probe() {
    let Some(dir) = std::env::var_os("ALLOY_W3_C_CRASH_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let (mut store, registry, policy, count) = open_fixture(&dir);
    store.admit_tool(&registry, &policy, request("c")).unwrap();
    store.decide_tool("s", "c", true).unwrap();
    let claim = store.claim_tool("s", "c").unwrap();
    let started = store.start_tool(claim, &registry, &policy).unwrap();
    let _effect = Counter {
        count,
        marker: dir.join("effects"),
    }
    .execute(started.binding().invocation());
    let artifacts =
        ArtifactStore::open(&mut store, dir.join("artifacts"), ArtifactLimits::default()).unwrap();
    let mut writer = artifacts.begin(&store, "s", "c", 7).unwrap();
    writer.write_all(b"output!").unwrap();
    let metadata = writer.finish().unwrap().commit(&mut store).unwrap();
    store
        .complete_tool(
            started,
            ToolResult::Artifact(ArtifactToolResult {
                call_id: "c".into(),
                artifact_ref: metadata.artifact_ref,
                is_error: false,
            }),
        )
        .unwrap();
    panic!("expected crash checkpoint was not reached");
}
fn run_crash(dir: &Path, stage: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "store::durable::failure_recovery::crash_child_probe",
            "--nocapture",
        ])
        .env("ALLOY_W3_C_CRASH_DIR", dir)
        .env("ALLOY_W3_C_CRASH_STAGE", stage)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "crash probe timed out: {stage}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(86),
        "{stage}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn process_exit_before_and_after_each_commit_preserves_single_use_recovery() {
    for (stage, old_state, new_state, executed) in [
        ("intent.before_commit", None, None, false),
        (
            "intent.after_commit",
            Some("AwaitingApproval"),
            Some("Withdrawn"),
            false,
        ),
        (
            "decision.before_commit",
            Some("AwaitingApproval"),
            Some("Withdrawn"),
            false,
        ),
        (
            "decision.after_commit",
            Some("Ready"),
            Some("Withdrawn"),
            false,
        ),
        (
            "claim.before_commit",
            Some("Ready"),
            Some("Withdrawn"),
            false,
        ),
        (
            "claim.after_commit",
            Some("Claimed"),
            Some("Withdrawn"),
            false,
        ),
        (
            "start.before_commit",
            Some("Claimed"),
            Some("Withdrawn"),
            false,
        ),
        (
            "start.after_commit",
            Some("Started"),
            Some("OutcomeUnknown"),
            false,
        ),
        (
            "complete.before_commit",
            Some("Started"),
            Some("OutcomeUnknown"),
            true,
        ),
        (
            "complete.after_commit",
            Some("Completed"),
            Some("Completed"),
            true,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        run_crash(dir.path(), stage);
        let db = dir.path().join("events.db");
        let connection = Connection::open(&db).unwrap();
        let raw: Option<String> = connection
            .query_row(
                "SELECT state FROM tool_intents WHERE session_id='s' AND call_id='c'",
                [],
                |r| r.get(0),
            )
            .optional()
            .unwrap();
        assert_eq!(raw.as_deref(), old_state, "{stage}");
        let epoch: String = connection
            .query_row("SELECT daemon_epoch FROM runtime_metadata", [], |r| {
                r.get(0)
            })
            .unwrap();
        let mut stmt=connection.prepare("SELECT id,session_id,seq,ts,kind,payload_json,artifact_refs FROM events ORDER BY id").unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                ))
            })
            .unwrap();
        let before = rows.collect::<Result<Vec<_>, _>>().unwrap();
        drop(stmt);
        drop(connection);
        let expected_marker = if executed {
            b"effect\n".as_slice()
        } else {
            b"".as_slice()
        };
        assert_eq!(
            std::fs::read(dir.path().join("effects")).unwrap_or_default(),
            expected_marker
        );
        let mut store = SessionStore::open(&db).unwrap();
        assert_ne!(store.daemon_epoch(), epoch);
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            new_state,
            "{stage}"
        );
        assert!(store.claim_tool("s", "c").is_err());
        let after = store.list_events("s").unwrap();
        assert!(after.len() >= before.len());
        if new_state == Some("OutcomeUnknown") {
            assert_eq!(after.last().unwrap().payload["outcome_known"], false);
            assert_eq!(after.last().unwrap().payload["reason"], "daemon_restarted");
        }
        if new_state == Some("Withdrawn") {
            assert_eq!(after.last().unwrap().payload["state"], "Withdrawn");
            assert_eq!(after.last().unwrap().payload["reason"], "daemon_restarted");
        }
        if new_state.is_none() || new_state == Some("Completed") {
            assert_eq!(after.len(), before.len());
        }
        for (event, old) in after.iter().zip(&before) {
            assert_eq!(
                (
                    event.id,
                    &event.session_id,
                    event.seq,
                    event.ts,
                    event.kind.as_str()
                ),
                (old.0, &old.1, old.2, old.3, old.4.as_str())
            );
        }
        // Compare raw historic bytes, not a reserialized approximation.
        let check = Connection::open(&db).unwrap();
        for old in &before {
            let raw: (String, String) = check
                .query_row(
                    "SELECT payload_json,artifact_refs FROM events WHERE id=?1",
                    [old.0],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(raw, (old.5.clone(), old.6.clone()));
        }
        drop(check);
        if executed {
            let artifacts = ArtifactStore::open(
                &mut store,
                dir.path().join("artifacts"),
                ArtifactLimits::default(),
            )
            .unwrap();
            let connection = Connection::open(&db).unwrap();
            let reference: String = connection
                .query_row("SELECT artifact_ref FROM tool_artifacts", [], |r| r.get(0))
                .unwrap();
            let mut verified = artifacts
                .lookup(&store, "s", "c", &reference)
                .unwrap()
                .verify()
                .unwrap();
            let mut bytes = Vec::new();
            verified.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes, b"output!");
            let result: Option<String> = connection
                .query_row(
                    "SELECT result_json FROM tool_intents WHERE call_id='c'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            if stage == "complete.after_commit" {
                assert_eq!(
                    serde_json::from_str::<Value>(&result.unwrap()).unwrap(),
                    serde_json::json!({"call_id":"c","artifact_ref":reference,"is_error":false})
                );
                assert_eq!(
                    after
                        .iter()
                        .filter(|e| e.kind == EventKind::ToolCompleted)
                        .count(),
                    1
                );
            } else {
                assert!(result.is_none());
                assert_eq!(after.last().unwrap().payload["state"], "OutcomeUnknown");
            }
        }
        if old_state.is_some() {
            let count = Arc::new(AtomicUsize::new(0));
            let registry = registry(definition(), &count, dir.path().join("effects"));
            assert!(
                store
                    .admit_tool(&registry, &policy(), request("c"))
                    .is_err()
            );
            assert_eq!(count.load(Ordering::SeqCst), 0);
        }
        assert_eq!(
            std::fs::read(dir.path().join("effects")).unwrap_or_default(),
            expected_marker,
            "recovery must not replay {stage}"
        );
    }
}

#[test]
fn post_start_pause_does_not_claim_nonexecution_or_reissue_authority() {
    let mut f = Fixture::new();
    let started = f.started();
    let _original_result = f.effect(&started);
    f.store.set_status("s", SessionStatus::Paused).unwrap();
    assert_eq!(f.store.withdraw_tools("s", "cancel").unwrap(), 0);
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Started")
    );
    f.assert_effects(1);
    let cancelled = ToolResult::Inline(InlineToolResult {
        call_id: "c".into(),
        is_error: true,
        content: serde_json::json!({"kind":"execution","message":"fixture cancellation","details":{"reason":"cancelled"}}),
    });
    f.store.complete_tool(started, cancelled).unwrap();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.assert_effects(1);
    let events = f.store.list_events("s").unwrap();
    let terminal = events
        .iter()
        .find(|e| e.kind == EventKind::ToolCompleted)
        .unwrap();
    assert_eq!(terminal.payload["result"]["is_error"], true);
}
#[test]
fn exact_file_and_byte_reservation_failures_release_capacity_without_orphans() {
    for files_limited in [false, true] {
        let mut f = Fixture::new();
        let started = f.started();
        let limits = if files_limited {
            ArtifactLimits {
                max_managed_files: 2,
                ..ArtifactLimits::default()
            }
        } else {
            ArtifactLimits {
                max_managed_bytes: 16,
                ..ArtifactLimits::default()
            }
        };
        let artifacts =
            ArtifactStore::open(&mut f.store, f.dir.path().join("artifacts"), limits).unwrap();
        let maximum = if files_limited { 0 } else { 8 };
        let writer = artifacts.begin(&f.store, "s", "c", maximum).unwrap();
        assert!(
            artifacts
                .begin(&f.store, "s", "c", if files_limited { 0 } else { 1 })
                .is_err()
        );
        drop(writer);
        let next = artifacts.begin(&f.store, "s", "c", maximum).unwrap();
        drop(next);
        assert_eq!(
            std::fs::read_dir(f.dir.path().join("artifacts/tmp"))
                .unwrap()
                .count(),
            0
        );
        assert_eq!(
            std::fs::read_dir(f.dir.path().join("artifacts/objects"))
                .unwrap()
                .count(),
            0
        );
        drop(started);
        f.assert_effects(0);
    }
}
