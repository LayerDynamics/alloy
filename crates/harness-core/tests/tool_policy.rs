//! Policy/binding tests and the counter-based P2-06 durable admission fixture.
//! No production builtin or external effect runs. Fixture implementation is not
//! candidate-bound Linux/macOS acceptance; retain the platform validation evidence.

use harness_core::{
    approval::{
        BindingError, CanonicalizationError, InvocationBinding, MAX_CANONICAL_BINDING_BYTES,
    },
    policy::{DaemonPolicy, PolicyConfig, PolicyDecision, PolicyDenial, PolicyError, PolicyRule},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tool_registry::{
    call::{ToolCall, ToolCallContext, ToolInvocation},
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
    registry::{ToolImplementation, ToolRegistry},
    result::{InlineToolResult, ToolResult},
};

struct CountingTool(Arc<AtomicUsize>);
impl ToolImplementation for CountingTool {
    fn execute(&self, invocation: &ToolInvocation) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::Inline(InlineToolResult {
            call_id: invocation.call.call_id.clone(),
            content: json!({"counter_fixture": true}),
            is_error: false,
        })
    }
}
fn definition(id: &str, effect: ToolEffect, approval: ApprovalRequirement) -> ToolDefinition {
    ToolDefinition {
        id: id.into(),
        version: "1.0.0".into(),
        description: "Counter-only policy fixture.".into(),
        input_schema: json!({"type":"object","properties":{"path":{"type":"string"}},
            "required":["path"],"additionalProperties":false}),
        origin: ToolOrigin::Builtin,
        policy: ToolPolicy {
            effect,
            approval,
            timeout_ms: 5000,
            max_output_bytes: 65536,
        },
    }
}
fn read_definition() -> ToolDefinition {
    definition("fs.read", ToolEffect::Read, ApprovalRequirement::Never)
}
fn registered(def: ToolDefinition) -> (ToolRegistry, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry
        .register(def, Arc::new(CountingTool(Arc::clone(&count))))
        .unwrap();
    (registry, count)
}
fn invocation(id: &str, arguments: Value) -> ToolInvocation {
    ToolInvocation {
        context: ToolCallContext {
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            generation_id: "generation-1".into(),
        },
        call: ToolCall {
            call_id: "call-1".into(),
            tool_id: id.into(),
            arguments: arguments.as_object().unwrap().clone(),
        },
    }
}
fn rule(id: &str, prompt: bool) -> PolicyRule {
    PolicyRule {
        tool_id: id.into(),
        allowed_arguments: vec![json!({"path":"fixture.txt"}).as_object().unwrap().clone()],
        require_approval: prompt,
        timeout_ms: 4000,
        max_output_bytes: 32000,
    }
}
fn policy(rule: PolicyRule) -> DaemonPolicy {
    DaemonPolicy::new(PolicyConfig { rules: vec![rule] }).unwrap()
}
fn evaluate(policy: &DaemonPolicy, registry: &ToolRegistry) -> PolicyDecision {
    policy
        .evaluate(
            registry,
            invocation("fs.read", json!({"path":"fixture.txt"})),
            "epoch-1",
            1000,
        )
        .unwrap()
}
fn binding(decision: PolicyDecision) -> InvocationBinding {
    match decision {
        PolicyDecision::AllowWithoutPrompt(binding) | PolicyDecision::RequireApproval(binding) => {
            *binding
        }
        PolicyDecision::Deny(reason) => panic!("expected prepared binding, got {reason:?}"),
    }
}
fn is_limit(error: PolicyError) -> bool {
    matches!(
        error,
        PolicyError::Canonicalization(CanonicalizationError::LimitExceeded)
            | PolicyError::Binding(BindingError::Canonicalization(
                CanonicalizationError::LimitExceeded
            ))
    )
}

#[test]
fn empty_policy_denies_without_execution() {
    let policy = DaemonPolicy::new(PolicyConfig::default()).unwrap();
    let (registry, count) = registered(read_definition());
    assert!(matches!(
        evaluate(&policy, &registry),
        PolicyDecision::Deny(PolicyDenial::NoMatchingRule)
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn exact_rule_prepares_read_without_prompt() {
    let policy = policy(rule("fs.read", false));
    let (registry, count) = registered(read_definition());
    let decision = evaluate(&policy, &registry);
    assert!(matches!(&decision, PolicyDecision::AllowWithoutPrompt(_)));
    let prepared = binding(decision);
    assert_eq!(
        prepared.snapshot()["effective_approval_requirement"],
        "never"
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn always_and_destructive_require_prompt() {
    for (effect, approval) in [
        (ToolEffect::Read, ApprovalRequirement::Always),
        (ToolEffect::Destructive, ApprovalRequirement::Never),
    ] {
        let (registry, count) = registered(definition("fixture.tool", effect, approval));
        let decision = policy(rule("fixture.tool", false))
            .evaluate(
                &registry,
                invocation("fixture.tool", json!({"path":"fixture.txt"})),
                "epoch-1",
                1000,
            )
            .unwrap();
        assert!(matches!(&decision, PolicyDecision::RequireApproval(_)));
        assert_eq!(
            binding(decision).snapshot()["effective_approval_requirement"],
            "always"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn explicit_rule_can_strengthen_approval() {
    let (registry, count) = registered(read_definition());
    assert!(matches!(
        evaluate(&policy(rule("fs.read", true)), &registry),
        PolicyDecision::RequireApproval(_)
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn policy_requirement_delegates_to_explicit_exact_rule() {
    let (registry, _) = registered(definition(
        "fs.write",
        ToolEffect::Write,
        ApprovalRequirement::Policy,
    ));
    for prompt in [false, true] {
        let decision = policy(rule("fs.write", prompt))
            .evaluate(
                &registry,
                invocation("fs.write", json!({"path":"fixture.txt"})),
                "epoch-1",
                1000,
            )
            .unwrap();
        assert_eq!(
            matches!(decision, PolicyDecision::RequireApproval(_)),
            prompt
        );
    }
}
#[test]
fn registry_validation_precedes_policy_binding() {
    let (registry, count) = registered(read_definition());
    let policy = policy(rule("fs.read", false));
    for request in [
        invocation("unknown.tool", json!({"path":"fixture.txt"})),
        invocation("fs.read", json!({"path":42})),
        invocation("fs.read", json!({})),
        invocation("fs.read", json!({"path":"fixture.txt","approved":true})),
    ] {
        assert!(matches!(
            policy.evaluate(&registry, request, "epoch-1", 1000),
            Err(PolicyError::Registry(_))
        ));
    }
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn unmatched_arguments_are_denied_without_execution() {
    let (registry, count) = registered(read_definition());
    let decision = policy(rule("fs.read", false))
        .evaluate(
            &registry,
            invocation("fs.read", json!({"path":"other.txt"})),
            "epoch-1",
            1000,
        )
        .unwrap();
    assert!(matches!(
        decision,
        PolicyDecision::Deny(PolicyDenial::ArgumentsNotAllowed)
    ));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn duplicate_and_invalid_rules_fail_closed() {
    let valid = rule("fs.read", false);
    assert!(
        DaemonPolicy::new(PolicyConfig {
            rules: vec![valid.clone(), valid.clone()]
        })
        .is_err()
    );
    for field in 0..4 {
        let mut invalid = valid.clone();
        match field {
            0 => invalid.timeout_ms = 0,
            1 => invalid.max_output_bytes = 0,
            2 => invalid.allowed_arguments.clear(),
            _ => invalid.tool_id = " fs.read ".into(),
        }
        assert!(
            DaemonPolicy::new(PolicyConfig {
                rules: vec![invalid]
            })
            .is_err()
        );
    }
}
#[test]
fn rule_limits_can_only_tighten_registration() {
    let (registry, _) = registered(read_definition());
    for (timeout, output, expected_timeout, expected_output) in
        [(9000, 90000, 5000, 65536), (100, 128, 100, 128)]
    {
        let mut candidate = rule("fs.read", false);
        candidate.timeout_ms = timeout;
        candidate.max_output_bytes = output;
        let prepared = binding(evaluate(&policy(candidate), &registry));
        assert_eq!(
            prepared.snapshot()["effective_timeout_ms"],
            expected_timeout
        );
        assert_eq!(
            prepared.snapshot()["effective_max_output_bytes"],
            expected_output
        );
    }
}
#[test]
fn policy_revision_ignores_order_but_binds_all_effective_rules() {
    let mut a = rule("fs.read", false);
    let b = rule("fixture.tool", true);
    a.allowed_arguments
        .push(json!({"path":"second.txt"}).as_object().unwrap().clone());
    let left = DaemonPolicy::new(PolicyConfig {
        rules: vec![a.clone(), b.clone()],
    })
    .unwrap();
    let mut reordered = a.clone();
    reordered.allowed_arguments.reverse();
    reordered
        .allowed_arguments
        .push(a.allowed_arguments[0].clone());
    let right = DaemonPolicy::new(PolicyConfig {
        rules: vec![b.clone(), reordered],
    })
    .unwrap();
    assert_eq!(left.revision(), right.revision());
    for field in 0..4 {
        let mut changed = a.clone();
        match field {
            0 => changed.require_approval = true,
            1 => changed.timeout_ms += 1,
            2 => changed.max_output_bytes += 1,
            _ => changed.allowed_arguments.truncate(1),
        }
        let changed = DaemonPolicy::new(PolicyConfig {
            rules: vec![changed, b.clone()],
        })
        .unwrap();
        assert_ne!(left.revision(), changed.revision());
    }
}
#[test]
fn builtin_registration_cannot_lower_frozen_floors() {
    for def in [
        definition("shell.run", ToolEffect::Read, ApprovalRequirement::Never),
        definition("fs.write", ToolEffect::Read, ApprovalRequirement::Policy),
        definition("http.fetch", ToolEffect::Read, ApprovalRequirement::Never),
    ] {
        let id = def.id.clone();
        let (registry, count) = registered(def);
        let decision = policy(rule(&id, false))
            .evaluate(
                &registry,
                invocation(&id, json!({"path":"fixture.txt"})),
                "epoch-1",
                1000,
            )
            .unwrap();
        assert!(matches!(
            decision,
            PolicyDecision::Deny(PolicyDenial::BuiltinContractMismatch)
        ));
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn valid_shell_still_requires_explicit_approval() {
    let (registry, count) = registered(definition(
        "shell.run",
        ToolEffect::Destructive,
        ApprovalRequirement::Always,
    ));
    let decision = policy(rule("shell.run", false))
        .evaluate(
            &registry,
            invocation("shell.run", json!({"path":"fixture.txt"})),
            "epoch-1",
            1000,
        )
        .unwrap();
    assert!(matches!(decision, PolicyDecision::RequireApproval(_)));
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn binding_has_every_frozen_field_and_domain_digest() {
    let (registry, _) = registered(read_definition());
    let policy = policy(rule("fs.read", false));
    let prepared = binding(evaluate(&policy, &registry));
    let frozen: Value =
        serde_json::from_str(include_str!("../../../docs/decisions/w3-contract-v1.json")).unwrap();
    let fields = frozen["binding_fields"].as_array().unwrap();
    assert_eq!(fields.len(), prepared.snapshot().as_object().unwrap().len());
    for field in fields {
        assert!(prepared.snapshot().get(field.as_str().unwrap()).is_some());
    }
    assert_eq!(prepared.snapshot()["policy_revision"], policy.revision());
    assert_eq!(prepared.snapshot()["created_at_unix_ms"], 1000);
    assert_eq!(prepared.snapshot()["expires_at_unix_ms"], 301000);
    assert_eq!(
        prepared.snapshot()["canonical_arguments"],
        "{\"path\":\"fixture.txt\"}"
    );
    let mut hash = Sha256::new();
    hash.update(b"alloy.w3.invocation.v1\0");
    hash.update(prepared.canonical_bytes());
    assert_eq!(prepared.digest(), format!("{:x}", hash.finalize()));
}
#[test]
fn binding_owns_immutable_request_and_configuration_snapshot() {
    let (registry, count) = registered(read_definition());
    let mut source = rule("fs.read", false);
    let policy = policy(source.clone());
    let mut request = invocation("fs.read", json!({"path":"fixture.txt"}));
    let prepared = binding(
        policy
            .evaluate(&registry, request.clone(), "epoch-1", 1000)
            .unwrap(),
    );
    request
        .call
        .arguments
        .insert("path".into(), json!("changed.txt"));
    source.allowed_arguments.clear();
    source.require_approval = true;
    assert_eq!(prepared.invocation().call.arguments["path"], "fixture.txt");
    assert_eq!(prepared.definition().id, "fs.read");
    assert_eq!(
        prepared.snapshot()["effective_approval_requirement"],
        "never"
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn definition_schema_version_and_origin_are_bound() {
    let policy = policy(rule("fs.read", false));
    let (registry, _) = registered(read_definition());
    let original = binding(evaluate(&policy, &registry));
    for field in 0..4 {
        let mut def = read_definition();
        match field {
            0 => def.version = "1.0.1".into(),
            1 => def.description.push_str(" revised"),
            2 => def.input_schema["required"] = json!([]),
            _ => def.origin = ToolOrigin::Plugin,
        }
        let (registry, _) = registered(def);
        let changed = binding(evaluate(&policy, &registry));
        assert_ne!(
            original.snapshot()["registered_definition_digest"],
            changed.snapshot()["registered_definition_digest"]
        );
        assert_ne!(original.digest(), changed.digest());
    }
}
#[test]
fn each_daemon_association_changes_binding_identity() {
    let (registry, _) = registered(read_definition());
    let policy = policy(rule("fs.read", false));
    let original = binding(evaluate(&policy, &registry));
    for field in 0..6 {
        let mut request = invocation("fs.read", json!({"path":"fixture.txt"}));
        let (mut epoch, mut time) = ("epoch-1", 1000);
        match field {
            0 => request.context.session_id = "session-2".into(),
            1 => request.context.turn_id = "turn-2".into(),
            2 => request.context.generation_id = "generation-2".into(),
            3 => request.call.call_id = "call-2".into(),
            4 => epoch = "epoch-2",
            _ => time = 1001,
        }
        let changed = binding(policy.evaluate(&registry, request, epoch, time).unwrap());
        assert_ne!(original.digest(), changed.digest());
    }
}
#[test]
fn changed_policy_revision_changes_existing_invocation_identity() {
    let (registry, _) = registered(read_definition());
    let original = binding(evaluate(&policy(rule("fs.read", false)), &registry));
    let changed = DaemonPolicy::new(PolicyConfig {
        rules: vec![rule("fs.read", false), rule("other.tool", true)],
    })
    .unwrap();
    let changed = binding(evaluate(&changed, &registry));
    assert_ne!(
        original.snapshot()["policy_revision"],
        changed.snapshot()["policy_revision"]
    );
    assert_ne!(original.digest(), changed.digest());
}
#[test]
fn complete_binding_ceiling_includes_envelope_and_escaping() {
    let (registry, _) = registered(read_definition());
    let policy = policy(rule("fs.read", false));
    let baseline = binding(
        policy
            .evaluate(
                &registry,
                invocation("fs.read", json!({"path":"fixture.txt"})),
                "a",
                1000,
            )
            .unwrap(),
    );
    let padding = MAX_CANONICAL_BINDING_BYTES - baseline.canonical_bytes().len() + 1;
    let exact = binding(
        policy
            .evaluate(
                &registry,
                invocation("fs.read", json!({"path":"fixture.txt"})),
                &"a".repeat(padding),
                1000,
            )
            .unwrap(),
    );
    assert_eq!(exact.canonical_bytes().len(), MAX_CANONICAL_BINDING_BYTES);
    let error = policy
        .evaluate(
            &registry,
            invocation("fs.read", json!({"path":"fixture.txt"})),
            &"a".repeat(padding + 1),
            1000,
        )
        .unwrap_err();
    assert!(is_limit(error));
    let error = policy
        .evaluate(
            &registry,
            invocation("fs.read", json!({"path":"fixture.txt"})),
            &"\"".repeat(padding),
            1000,
        )
        .unwrap_err();
    assert!(is_limit(error));
}
#[test]
fn invalid_daemon_context_and_deadline_overflow_fail_closed() {
    let (registry, count) = registered(read_definition());
    let policy = policy(rule("fs.read", false));
    for (epoch, time) in [
        ("", 1000),
        (" epoch ", 1000),
        ("epoch-1", -1),
        ("epoch-1", i64::MAX),
    ] {
        assert!(matches!(
            policy.evaluate(
                &registry,
                invocation("fs.read", json!({"path":"fixture.txt"})),
                epoch,
                time
            ),
            Err(PolicyError::Binding(_))
        ));
    }
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[test]
fn oversized_rule_arguments_fail_closed() {
    let mut candidate = rule("fs.read", false);
    candidate.allowed_arguments = vec![
        json!({"path":"x".repeat(MAX_CANONICAL_BINDING_BYTES)})
            .as_object()
            .unwrap()
            .clone(),
    ];
    assert!(matches!(
        DaemonPolicy::new(PolicyConfig {
            rules: vec![candidate]
        }),
        Err(PolicyError::Canonicalization(
            CanonicalizationError::LimitExceeded
        ))
    ));
}
#[test]
fn configuration_unknown_fields_and_invalid_types_reject() {
    let valid = serde_json::to_value(PolicyConfig {
        rules: vec![rule("fs.read", false)],
    })
    .unwrap();
    let mut extra = valid.clone();
    extra["approved"] = json!(true);
    assert!(serde_json::from_value::<PolicyConfig>(extra).is_err());
    let mut extra = valid.clone();
    extra["rules"][0]["effect"] = json!("read");
    assert!(serde_json::from_value::<PolicyConfig>(extra).is_err());
    let mut invalid = valid;
    invalid["rules"][0]["require_approval"] = json!("false");
    assert!(serde_json::from_value::<PolicyConfig>(invalid).is_err());
}
#[test]
fn registry_failure_preserves_typed_error_source() {
    use std::error::Error;
    let (registry, _) = registered(read_definition());
    let error = policy(rule("fs.read", false))
        .evaluate(
            &registry,
            invocation("fs.read", json!({"path":42})),
            "epoch-1",
            1000,
        )
        .unwrap_err();
    assert!(
        error
            .source()
            .unwrap()
            .downcast_ref::<tool_registry::RegistryError>()
            .is_some()
    );
}

#[test]
fn allowed_argument_changes_are_bound_without_policy_revision_change() {
    let (registry, count) = registered(read_definition());
    let mut candidate = rule("fs.read", false);
    candidate
        .allowed_arguments
        .push(json!({"path":"other.txt"}).as_object().unwrap().clone());
    let policy = policy(candidate);
    let first = binding(evaluate(&policy, &registry));
    let second = binding(
        policy
            .evaluate(
                &registry,
                invocation("fs.read", json!({"path":"other.txt"})),
                "epoch-1",
                1000,
            )
            .unwrap(),
    );
    assert_eq!(
        first.snapshot()["policy_revision"],
        second.snapshot()["policy_revision"]
    );
    assert_ne!(
        first.snapshot()["canonical_arguments"],
        second.snapshot()["canonical_arguments"]
    );
    assert_ne!(
        first.snapshot()["arguments_digest"],
        second.snapshot()["arguments_digest"]
    );
    assert_ne!(first.digest(), second.digest());
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

// W3 storage admission fixture only. Not W4 shell or W6 tool-loop acceptance.
#[test]
fn destructive_effect_requires_bound_approval() {
    use harness_core::store::{IntentAdmission, SessionStore};
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("approvals.db")).unwrap();
    store.create_session("session-1", "fixture").unwrap();
    let (registry, count) = registered(definition(
        "shell.run",
        ToolEffect::Destructive,
        ApprovalRequirement::Always,
    ));
    let policy = policy(rule("shell.run", false));
    let mut request = invocation("shell.run", json!({"path":"fixture.txt"}));
    assert_eq!(
        store
            .admit_tool(&registry, &policy, request.clone())
            .unwrap(),
        IntentAdmission::AwaitingApproval
    );
    assert!(store.claim_tool("session-1", "call-1").is_err());
    assert!(store.decide_tool("wrong-session", "call-1", true).is_err());
    assert_eq!(count.load(Ordering::SeqCst), 0);
    store.decide_tool("session-1", "call-1", false).unwrap();
    assert!(store.claim_tool("session-1", "call-1").is_err());
    assert!(store.decide_tool("session-1", "call-1", true).is_err());
    assert!(
        store
            .admit_tool(&registry, &policy, request.clone())
            .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    request.call.call_id = "call-2".into();
    store
        .admit_tool(&registry, &policy, request.clone())
        .unwrap();
    store.decide_tool("session-1", "call-2", true).unwrap();
    let handle = store.claim_tool("session-1", "call-2").unwrap();
    assert!(store.claim_tool("session-1", "call-2").is_err());
    let receipt = store.start_tool(handle, &registry, &policy).unwrap();
    let result = CountingTool(Arc::clone(&count)).execute(receipt.binding().invocation());
    store.complete_tool(receipt, result).unwrap();
    assert!(store.claim_tool("session-1", "call-2").is_err());
    assert!(store.decide_tool("session-1", "call-2", true).is_err());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    request.call.call_id = "call-3".into();
    request
        .call
        .arguments
        .insert("path".into(), json!("changed.txt"));
    assert_eq!(
        store.admit_tool(&registry, &policy, request).unwrap(),
        IntentAdmission::Rejected
    );
    assert!(store.claim_tool("session-1", "call-3").is_err());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    drop(store);
    let mut reopened = SessionStore::open(dir.path().join("approvals.db")).unwrap();
    assert!(reopened.claim_tool("session-1", "call-2").is_err());
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn approval_storage_boundary_five_frozen_builtin_defaults_preserve_prompt_resolution() {
    for strengthen in [false, true] {
        for (id, effect, approval) in [
            ("fs.read", ToolEffect::Read, ApprovalRequirement::Never),
            ("fs.list", ToolEffect::Read, ApprovalRequirement::Never),
            ("fs.write", ToolEffect::Write, ApprovalRequirement::Policy),
            (
                "shell.run",
                ToolEffect::Destructive,
                ApprovalRequirement::Always,
            ),
            ("http.fetch", ToolEffect::Read, ApprovalRequirement::Policy),
        ] {
            let (registry, count) = registered(definition(id, effect, approval));
            let decision = policy(rule(id, strengthen))
                .evaluate(
                    &registry,
                    invocation(id, json!({"path":"fixture.txt"})),
                    "epoch-boundary",
                    1000,
                )
                .unwrap();
            let expected = strengthen || id == "shell.run";
            assert_eq!(
                matches!(decision, PolicyDecision::RequireApproval(_)),
                expected,
                "{id}"
            );
            assert_eq!(count.load(Ordering::SeqCst), 0);
        }
    }
}
