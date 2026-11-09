//! W4-01.03: real registration/W3 transactions with explicit counter backends.
//! These fixtures are NOT production filesystem/shell/HTTP adapters or P2-07.
use harness_core::{
    SessionStore,
    policy::{DaemonPolicy, PolicyConfig, PolicyRule},
    store::IntentAdmission,
    tool_execution::{BuiltinImplementations, ReceiptExecutor},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tool_registry::{
    call::{ToolCall, ToolCallContext, ToolInvocation},
    definition::{ApprovalRequirement, ToolEffect, ToolOrigin},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, OutputSink},
    registry::ToolImplementation,
    result::{InlineToolResult, ToolResult},
};
const IDS: [&str; 5] = ["fs.read", "fs.list", "fs.write", "shell.run", "http.fetch"];
type Counts = Arc<Mutex<BTreeMap<&'static str, usize>>>;
struct Counter {
    id: &'static str,
    counts: Counts,
}
impl ToolImplementation for Counter {
    fn execute(&self, _: &ToolInvocation) -> ToolResult {
        panic!("legacy execution must not be used")
    }
    fn execute_with_context(
        &self,
        i: &ToolInvocation,
        c: &ExecutionContext,
        o: &mut dyn OutputSink,
    ) -> ToolResult {
        assert_eq!(i.call.tool_id, self.id);
        assert_eq!(c.max_output_bytes(), 2048);
        assert!(c.remaining().unwrap() <= std::time::Duration::from_millis(4000));
        *self.counts.lock().unwrap().get_mut(self.id).unwrap() += 1;
        o.write_chunk(self.id.as_bytes()).unwrap();
        ToolResult::Inline(InlineToolResult {
            call_id: i.call.call_id.clone(),
            content: json!({"tool":self.id,"arguments":&i.call.arguments}),
            is_error: false,
        })
    }
}
fn implementation(id: &'static str, counts: &Counts) -> Arc<dyn ToolImplementation> {
    Arc::new(Counter {
        id,
        counts: counts.clone(),
    })
}
fn bundle(counts: &Counts) -> BuiltinImplementations {
    BuiltinImplementations {
        fs_read: implementation("fs.read", counts),
        fs_list: implementation("fs.list", counts),
        fs_write: implementation("fs.write", counts),
        shell_run: implementation("shell.run", counts),
        http_fetch: implementation("http.fetch", counts),
    }
}
fn counts() -> Counts {
    Arc::new(Mutex::new(IDS.into_iter().map(|id| (id, 0)).collect()))
}
fn configured() -> (ReceiptExecutor, Counts) {
    let c = counts();
    (
        ReceiptExecutor::with_builtins(5000, 4096, bundle(&c)).unwrap(),
        c,
    )
}
fn arguments(id: &str) -> Value {
    match id {
        "fs.read" => json!({"path":"fixture.txt","offset":0,"length":8}),
        "fs.list" => json!({"path":".","max_entries":10}),
        "fs.write" => json!({"path":"new.txt","content":"exact 雪"}),
        "shell.run" => {
            json!({"executable":"/fixture/program","args":["$(literal)","space value"],"cwd":"/fixture"})
        }
        "http.fetch" => json!({"url":"https://fixture.invalid/","method":"GET"}),
        _ => panic!("unknown fixture tool"),
    }
}
fn request(id: &str, call: &str, args: Value) -> ToolInvocation {
    ToolInvocation {
        context: ToolCallContext {
            session_id: "s".into(),
            turn_id: "t".into(),
            generation_id: "g".into(),
        },
        call: ToolCall {
            call_id: call.into(),
            tool_id: id.into(),
            arguments: args.as_object().unwrap().clone(),
        },
    }
}
fn policy(prompt: bool) -> DaemonPolicy {
    DaemonPolicy::new(PolicyConfig {
        rules: IDS
            .into_iter()
            .map(|id| PolicyRule {
                tool_id: id.into(),
                allowed_arguments: vec![arguments(id).as_object().unwrap().clone()],
                require_approval: prompt,
                timeout_ms: 4000,
                max_output_bytes: 2048,
            })
            .collect(),
    })
    .unwrap()
}
fn zero(c: &Counts) {
    assert!(c.lock().unwrap().values().all(|n| *n == 0));
}
fn output() -> BoundedOutput {
    let c = ExecutionContext::new(
        &tool_registry::builtins::definitions(5000, 4096).unwrap()[0].policy,
        5000,
        4096,
        CancellationToken::default(),
    )
    .unwrap();
    BoundedOutput::new(&c, 4096).unwrap()
}
#[test]
fn registered_catalogue_has_five_exact_defaults() {
    let (executor, c) = configured();
    assert_eq!(executor.registry().len(), 5);
    let expected = [
        ("fs.read", ToolEffect::Read, ApprovalRequirement::Never),
        ("fs.list", ToolEffect::Read, ApprovalRequirement::Never),
        ("fs.write", ToolEffect::Write, ApprovalRequirement::Policy),
        (
            "shell.run",
            ToolEffect::Destructive,
            ApprovalRequirement::Always,
        ),
        ("http.fetch", ToolEffect::Read, ApprovalRequirement::Policy),
    ];
    for (id, effect, approval) in expected {
        let d = executor.registry().definition(id).unwrap();
        assert_eq!(d.policy.effect, effect);
        assert_eq!(d.policy.approval, approval);
        assert_eq!(d.policy.timeout_ms, 5000);
        assert_eq!(d.policy.max_output_bytes, 4096);
        assert!(matches!(&d.origin, ToolOrigin::Builtin));
    }
    zero(&c);
}
#[test]
fn registration_and_resolution_execute_nothing() {
    let (executor, c) = configured();
    for id in IDS {
        let i = request(id, id, arguments(id));
        let resolved = executor.registry().resolve(i.clone()).unwrap();
        assert_eq!(resolved.invocation(), &i);
    }
    zero(&c);
}
#[test]
fn each_slot_dispatches_unchanged_only_after_durable_start() {
    let (executor, c) = configured();
    let p = policy(false);
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    for id in IDS {
        let i = request(id, id, arguments(id));
        let admission = store
            .admit_tool(executor.registry(), &p, i.clone())
            .unwrap();
        if id == "shell.run" {
            assert_eq!(admission, IntentAdmission::AwaitingApproval);
            assert!(store.claim_tool("s", id).is_err());
            store.decide_tool("s", id, true).unwrap();
        } else {
            assert_eq!(admission, IntentAdmission::Ready);
        }
        assert_eq!(c.lock().unwrap()[id], 0);
        let claim = store.claim_tool("s", id).unwrap();
        let receipt = store.start_tool(claim, executor.registry(), &p).unwrap();
        let job = executor
            .prepare_started(&store, receipt, CancellationToken::default())
            .unwrap();
        let mut out = output();
        let completed = job.run(&mut out);
        assert!(!completed.result().is_error());
        if let ToolResult::Inline(r) = completed.result() {
            assert_eq!(r.content["tool"], id);
            assert_eq!(r.content["arguments"], Value::Object(i.call.arguments));
        } else {
            panic!("expected inline fixture result");
        }
        assert_eq!(out.finish().unwrap(), id.as_bytes());
        completed.complete(&mut store).unwrap();
        assert!(store.claim_tool("s", id).is_err());
    }
    assert!(c.lock().unwrap().values().all(|n| *n == 1));
}
#[test]
fn shell_default_requires_allow_and_denial_has_zero_dispatch() {
    let (executor, c) = configured();
    let p = policy(false);
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    assert_eq!(
        store
            .admit_tool(
                executor.registry(),
                &p,
                request("shell.run", "c", arguments("shell.run"))
            )
            .unwrap(),
        IntentAdmission::AwaitingApproval
    );
    assert!(store.claim_tool("s", "c").is_err());
    assert!(store.decide_tool("other-session", "c", true).is_err());
    store.decide_tool("s", "c", false).unwrap();
    assert!(store.claim_tool("s", "c").is_err());
    zero(&c);
}
#[test]
fn never_default_does_not_bypass_daemon_policy() {
    let (executor, c) = configured();
    let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    for id in ["fs.read", "fs.list"] {
        assert_eq!(
            store
                .admit_tool(executor.registry(), &deny, request(id, id, arguments(id)))
                .unwrap(),
            IntentAdmission::Rejected
        );
        assert!(store.claim_tool("s", id).is_err());
    }
    zero(&c);
}
#[test]
fn policy_can_strengthen_prompting_for_every_registered_builtin() {
    let (executor, c) = configured();
    let p = policy(true);
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    for id in IDS {
        assert_eq!(
            store
                .admit_tool(executor.registry(), &p, request(id, id, arguments(id)))
                .unwrap(),
            IntentAdmission::AwaitingApproval
        );
        assert!(store.claim_tool("s", id).is_err());
    }
    zero(&c);
}
#[test]
fn duplicate_registration_cannot_lower_existing_shell_defaults() {
    let (mut executor, c) = configured();
    let mut d = executor.registry().definition("shell.run").unwrap().clone();
    d.policy.effect = ToolEffect::Read;
    d.policy.approval = ApprovalRequirement::Never;
    assert!(matches!(
        executor.register(d, implementation("shell.run", &c)),
        Err(tool_registry::RegistryError::DuplicateId(_))
    ));
    let d = executor.registry().definition("shell.run").unwrap();
    assert_eq!(d.policy.effect, ToolEffect::Destructive);
    assert_eq!(d.policy.approval, ApprovalRequirement::Always);
    assert_eq!(executor.registry().len(), 5);
    zero(&c);
}
#[test]
fn invalid_limits_return_no_partial_catalogue_or_execution() {
    for (timeout, bytes) in [(0, 4096), (5000, 0), (0, 0)] {
        let c = counts();
        assert!(ReceiptExecutor::with_builtins(timeout, bytes, bundle(&c)).is_err());
        zero(&c);
    }
}
#[test]
fn malformed_builtin_calls_never_dispatch() {
    let (executor, c) = configured();
    let p = policy(false);
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    for id in IDS {
        assert!(
            store
                .admit_tool(executor.registry(), &p, request(id, id, json!({})))
                .is_err()
        );
        let mut extra = arguments(id);
        extra["approved"] = json!(true);
        assert!(
            store
                .admit_tool(executor.registry(), &p, request(id, id, extra))
                .is_err()
        );
    }
    assert!(
        store
            .admit_tool(
                executor.registry(),
                &p,
                request("unregistered.tool", "unknown", json!({}))
            )
            .is_err()
    );
    zero(&c);
}
