//! W4-01.01 real W3 transactions plus registry-owned counter/marker execution.
//! These are bridge tests, not five-builtin or cross-platform acceptance claims.
use harness_core::{
    SessionStore,
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::ReceiptExecutor,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tool_registry::{
    call::{ToolCall, ToolCallContext, ToolInvocation},
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, OutputSink},
    registry::ToolImplementation,
    result::{InlineToolResult, ToolResult},
};
type Observations = Arc<Mutex<Vec<(ToolInvocation, u64)>>>;
struct Counter {
    count: Arc<AtomicUsize>,
    observed: Observations,
    marker: PathBuf,
    wrong_id: bool,
    overflow: bool,
}
impl ToolImplementation for Counter {
    fn execute(&self, _: &ToolInvocation) -> ToolResult {
        panic!("legacy execute must never be called")
    }
    fn execute_with_context(
        &self,
        i: &ToolInvocation,
        c: &ExecutionContext,
        o: &mut dyn OutputSink,
    ) -> ToolResult {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.observed
            .lock()
            .unwrap()
            .push((i.clone(), c.max_output_bytes()));
        std::fs::write(&self.marker, b"effect").unwrap();
        if self.overflow {
            let _ = o.write_chunk(&vec![0; 4096]);
        }
        ToolResult::Inline(InlineToolResult {
            call_id: if self.wrong_id {
                "wrong".into()
            } else {
                i.call.call_id.clone()
            },
            content: json!({"ok":true}),
            is_error: false,
        })
    }
}
fn definition() -> ToolDefinition {
    ToolDefinition {
        id: "fixture.receipt".into(),
        version: "1.0.0".into(),
        description: "Receipt fixture".into(),
        input_schema: json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}),
        origin: ToolOrigin::Builtin,
        policy: ToolPolicy {
            effect: ToolEffect::Destructive,
            approval: ApprovalRequirement::Always,
            timeout_ms: 10000,
            max_output_bytes: 4096,
        },
    }
}
fn request() -> ToolInvocation {
    ToolInvocation {
        context: ToolCallContext {
            session_id: "s".into(),
            turn_id: "t".into(),
            generation_id: "g".into(),
        },
        call: ToolCall {
            call_id: "c".into(),
            tool_id: "fixture.receipt".into(),
            arguments: json!({"text":"$(literal) 雪"}).as_object().unwrap().clone(),
        },
    }
}
struct Fixture {
    dir: tempfile::TempDir,
    store: SessionStore,
    executor: ReceiptExecutor,
    policy: DaemonPolicy,
    count: Arc<AtomicUsize>,
    observed: Observations,
}
impl Fixture {
    fn new(timeout: u64, wrong_id: bool, overflow: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut executor = ReceiptExecutor::new();
        executor
            .register(
                definition(),
                Arc::new(Counter {
                    count: count.clone(),
                    observed: observed.clone(),
                    marker: dir.path().join("marker"),
                    wrong_id,
                    overflow,
                }),
            )
            .unwrap();
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "fixture.receipt".into(),
                allowed_arguments: vec![request().call.arguments],
                require_approval: false,
                timeout_ms: timeout,
                max_output_bytes: 1024,
            }],
        })
        .unwrap();
        Self {
            dir,
            store,
            executor,
            policy,
            count,
            observed,
        }
    }
    fn started(&mut self) -> harness_core::store::StartedInvocation {
        assert_eq!(
            self.store
                .admit_tool(self.executor.registry(), &self.policy, request())
                .unwrap(),
            IntentAdmission::AwaitingApproval
        );
        self.store.decide_tool("s", "c", true).unwrap();
        let claim = self.store.claim_tool("s", "c").unwrap();
        self.store
            .start_tool(claim, self.executor.registry(), &self.policy)
            .unwrap()
    }
    fn zero(&self) {
        assert_eq!(self.count.load(Ordering::SeqCst), 0);
        assert!(!self.dir.path().join("marker").exists());
    }
}
fn sink() -> (ExecutionContext, BoundedOutput) {
    let c = ExecutionContext::new(
        &definition().policy,
        10000,
        4096,
        CancellationToken::default(),
    )
    .unwrap();
    let out = BoundedOutput::new(&c, 4096).unwrap();
    (c, out)
}
#[test]
fn receipt_dispatch_preserves_invocation_limits_and_completes_once() {
    let mut f = Fixture::new(5000, false, false);
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    let probe = rusqlite::Connection::open(f.store.path()).unwrap();
    probe
        .execute(
            "UPDATE sessions SET profile_id=profile_id WHERE session_id='s'",
            [],
        )
        .unwrap();
    drop(probe);
    let (_c, mut out) = sink();
    let completion = job.run(&mut out);
    assert!(!completion.result().is_error());
    assert_eq!(f.observed.lock().unwrap().as_slice(), &[(request(), 1024)]);
    completion.complete(&mut f.store).unwrap();
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    assert_eq!(
        f.store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert!(f.store.claim_tool("s", "c").is_err());
}
#[test]
fn denial_never_produces_executable_receipt() {
    let mut f = Fixture::new(5000, false, false);
    f.store
        .admit_tool(f.executor.registry(), &f.policy, request())
        .unwrap();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.store.decide_tool("s", "c", false).unwrap();
    assert!(f.store.claim_tool("s", "c").is_err());
    f.zero();
}
#[test]
fn changed_policy_before_start_has_zero_effects() {
    let mut f = Fixture::new(5000, false, false);
    f.store
        .admit_tool(f.executor.registry(), &f.policy, request())
        .unwrap();
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
fn failed_start_transaction_returns_no_receipt_or_effect() {
    let mut f = Fixture::new(5000, false, false);
    f.store
        .admit_tool(f.executor.registry(), &f.policy, request())
        .unwrap();
    f.store.decide_tool("s", "c", true).unwrap();
    let claim = f.store.claim_tool("s", "c").unwrap();
    let probe = rusqlite::Connection::open(f.store.path()).unwrap();
    probe.execute_batch("CREATE TRIGGER fail_start BEFORE INSERT ON events WHEN NEW.kind='tool.started' BEGIN SELECT RAISE(ABORT,'fixture_start_failure'); END;").unwrap();
    assert!(
        f.store
            .start_tool(claim, f.executor.registry(), &f.policy)
            .is_err()
    );
    f.zero();
}
#[test]
fn cancellation_before_run_records_error_without_implementation() {
    let mut f = Fixture::new(5000, false, false);
    let receipt = f.started();
    let token = CancellationToken::default();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, token.clone())
        .unwrap();
    token.cancel();
    let (_c, mut out) = sink();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    f.zero();
    completed.complete(&mut f.store).unwrap();
    assert!(out.finish().is_err());
}
#[test]
fn delayed_run_does_not_restart_execution_deadline() {
    let mut f = Fixture::new(50, false, false);
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let (_c, mut out) = sink();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    if let ToolResult::Inline(r) = completed.result() {
        assert_eq!(r.content["kind"], "timeout");
    }
    f.zero();
    completed.complete(&mut f.store).unwrap();
}
#[test]
fn prepared_job_pins_daemon_ownership_without_borrowing_store() {
    let mut f = Fixture::new(5000, false, false);
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    let path = f.store.path().to_owned();
    drop(f.store);
    assert!(SessionStore::open(&path).is_err());
    drop(job);
    let reopened = SessionStore::open(path).unwrap();
    assert_eq!(
        reopened.tool_state("s", "c").unwrap().as_deref(),
        Some("OutcomeUnknown")
    );
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
}
#[test]
fn old_epoch_receipt_cannot_be_prepared_after_reopen() {
    let mut f = Fixture::new(5000, false, false);
    let receipt = f.started();
    let path = f.store.path().to_owned();
    drop(f.store);
    let reopened = SessionStore::open(path).unwrap();
    assert!(
        f.executor
            .prepare_started(&reopened, receipt, CancellationToken::default())
            .is_err()
    );
    assert_eq!(f.count.load(Ordering::SeqCst), 0);
    assert!(!f.dir.path().join("marker").exists());
}
#[test]
fn wrong_registration_cannot_execute_started_receipt() {
    let mut f = Fixture::new(5000, false, false);
    let receipt = f.started();
    let mut other = ReceiptExecutor::new();
    let mut changed = definition();
    changed.description.push_str(" changed");
    other
        .register(
            changed,
            Arc::new(Counter {
                count: f.count.clone(),
                observed: f.observed.clone(),
                marker: f.dir.path().join("marker"),
                wrong_id: false,
                overflow: false,
            }),
        )
        .unwrap();
    assert!(
        other
            .prepare_started(&f.store, receipt, CancellationToken::default())
            .is_err()
    );
    f.zero();
}
#[test]
fn result_identity_mismatch_preserves_bound_call_and_effect_status() {
    let mut f = Fixture::new(5000, true, false);
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    let (_c, mut out) = sink();
    let completed = job.run(&mut out);
    assert_eq!(completed.result().call_id(), "c");
    assert!(completed.result().is_error());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    completed.complete(&mut f.store).unwrap();
}
#[test]
fn ignored_output_overflow_cannot_be_reported_as_success() {
    let mut f = Fixture::new(5000, false, true);
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    let (_c, mut out) = sink();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    assert!(out.finish().is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    completed.complete(&mut f.store).unwrap();
}
#[test]
fn completion_failure_never_replays_effect() {
    let mut f = Fixture::new(5000, false, false);
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    let (_c, mut out) = sink();
    let completed = job.run(&mut out);
    let probe = rusqlite::Connection::open(f.store.path()).unwrap();
    probe.execute_batch("CREATE TRIGGER fail_complete BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'fixture_completion_failure'); END;").unwrap();
    assert!(completed.complete(&mut f.store).is_err());
    assert!(f.store.claim_tool("s", "c").is_err());
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    probe.execute_batch("DROP TRIGGER fail_complete;").unwrap();
    drop(probe);
    let path = f.store.path().to_owned();
    drop(f.store);
    let reopened = SessionStore::open(path).unwrap();
    assert_eq!(
        reopened.tool_state("s", "c").unwrap().as_deref(),
        Some("OutcomeUnknown")
    );
    assert_eq!(
        std::fs::read(f.dir.path().join("marker")).unwrap(),
        b"effect"
    );
}

#[test]
fn changed_arguments_are_rejected_before_any_receipt_or_effect() {
    let mut f = Fixture::new(5000, false, false);
    let mut changed = request();
    changed
        .call
        .arguments
        .insert("text".into(), json!("different"));
    assert_eq!(
        f.store
            .admit_tool(f.executor.registry(), &f.policy, changed)
            .unwrap(),
        IntentAdmission::Rejected
    );
    assert!(f.store.claim_tool("s", "c").is_err());
    f.zero();
}
struct Legacy(Arc<AtomicUsize>);
impl ToolImplementation for Legacy {
    fn execute(&self, i: &ToolInvocation) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::Inline(InlineToolResult {
            call_id: i.call.call_id.clone(),
            content: json!(null),
            is_error: false,
        })
    }
}
#[test]
fn receipt_dispatch_never_falls_back_to_legacy_execute() {
    let mut f = Fixture::new(5000, false, false);
    let mut executor = ReceiptExecutor::new();
    executor
        .register(definition(), Arc::new(Legacy(f.count.clone())))
        .unwrap();
    f.executor = executor;
    let receipt = f.started();
    let job = f
        .executor
        .prepare_started(&f.store, receipt, CancellationToken::default())
        .unwrap();
    let (_c, mut out) = sink();
    let completed = job.run(&mut out);
    assert!(completed.result().is_error());
    f.zero();
    completed.complete(&mut f.store).unwrap();
}
