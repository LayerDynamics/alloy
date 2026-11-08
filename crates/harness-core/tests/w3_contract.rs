//! Executable W3 contract vectors; not P2-06/P2-08 acceptance.
//! No database, real clock, registry authority, hooks, or effects are exercised.

use harness_core::approval::{
    ApprovalState, CANONICALIZATION_VERSION, CONTRACT_VERSION, CanonicalizationError,
    MAX_CANONICAL_BINDING_BYTES, Transition, canonical_json,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const FREEZE: &str = include_str!("../../../docs/decisions/w3-contract-v1.json");
const DOCUMENT: &[u8] =
    include_bytes!("../../../docs/decisions/w3-approval-and-storage-contract.md");
const CANONICAL: &str = include_str!("fixtures/w3_canonicalization_v1.json");
const TRANSITIONS: &str = include_str!("fixtures/w3_transitions_v1.json");

#[derive(Deserialize)]
struct CanonicalCase {
    id: String,
    input_json: String,
    canonical_json: String,
    sha256: String,
}
#[derive(Deserialize)]
struct BindingCase {
    input_json: String,
    canonical_json: String,
    domain_separated_sha256: String,
}
#[derive(Deserialize)]
struct CanonicalFixtures {
    contract_version: String,
    canonicalization_version: String,
    cases: Vec<CanonicalCase>,
    binding: BindingCase,
}
#[derive(Deserialize)]
struct Edge {
    from: String,
    event: String,
    to: String,
}
#[derive(Deserialize)]
struct TransitionFixtures {
    contract_version: String,
    states: Vec<String>,
    events: Vec<String>,
    allowed: Vec<Edge>,
}
fn vectors() -> CanonicalFixtures {
    serde_json::from_str(CANONICAL).expect("canonical fixture JSON")
}
fn graph() -> TransitionFixtures {
    serde_json::from_str(TRANSITIONS).expect("transition fixture JSON")
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn binding_digest(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"alloy.w3.invocation.v1\0");
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

#[test]
fn freeze_manifest_matches_normative_record() {
    let frozen: Value = serde_json::from_str(FREEZE).expect("frozen manifest");
    assert_eq!(frozen["contract_version"], CONTRACT_VERSION);
    assert_eq!(frozen["canonicalization_version"], CANONICALIZATION_VERSION);
    assert_eq!(frozen["database_schema_version"], 2);
    assert_eq!(frozen["normative_document_sha256"], digest(DOCUMENT));
    assert_eq!(
        frozen["limits"]["max_canonical_binding_bytes"].as_u64(),
        Some(MAX_CANONICAL_BINDING_BYTES as u64)
    );
    assert_eq!(frozen["limits"]["authorization_ttl_ms"], 300000);
    assert_eq!(
        frozen["binding_digest"]["domain_utf8"],
        "alloy.w3.invocation.v1"
    );
    assert_eq!(frozen["binding_digest"]["domain_separator_byte"], 0);
    let data = vectors();
    assert_eq!(data.contract_version, CONTRACT_VERSION);
    assert_eq!(data.canonicalization_version, CANONICALIZATION_VERSION);
    assert_eq!(graph().contract_version, CONTRACT_VERSION);
}

#[test]
fn canonical_fixtures_match_exact_bytes_and_sha256() {
    let data = vectors();
    assert_eq!(
        data.cases.len(),
        12,
        "fixture discovery must not be empty or shrink silently"
    );
    let mut ids = BTreeSet::new();
    for case in data.cases {
        assert!(ids.insert(case.id.clone()), "duplicate fixture ID");
        let value: Value = serde_json::from_str(&case.input_json).expect("valid vector input");
        let bytes = canonical_json(&value).expect("bounded canonical vector");
        assert_eq!(bytes, case.canonical_json.as_bytes(), "{}", case.id);
        assert_eq!(digest(&bytes), case.sha256, "{}", case.id);
    }
}

#[test]
fn canonical_objects_are_insertion_order_independent() {
    let mut left = Map::new();
    left.insert("z".into(), json!({"b":2,"a":1}));
    left.insert("a".into(), json!(3));
    let mut right = Map::new();
    right.insert("a".into(), json!(3));
    right.insert("z".into(), json!({"a":1,"b":2}));
    assert_eq!(
        canonical_json(&Value::Object(left)).unwrap(),
        canonical_json(&Value::Object(right)).unwrap()
    );
}

#[test]
fn canonical_arrays_preserve_order() {
    assert_ne!(
        canonical_json(&json!([1, 2])).unwrap(),
        canonical_json(&json!([2, 1])).unwrap()
    );
}

#[test]
fn canonical_strings_are_not_normalized() {
    assert_ne!(
        canonical_json(&json!("é")).unwrap(),
        canonical_json(&json!("e\u{301}")).unwrap()
    );
    assert_eq!(
        canonical_json(&json!(" /a/../b $(touch NEVER_RUN) ")).unwrap(),
        b"\" /a/../b $(touch NEVER_RUN) \""
    );
}

#[test]
fn canonical_numbers_preserve_integer_float_distinction() {
    let integer: Value = serde_json::from_str("1").unwrap();
    let float: Value = serde_json::from_str("1.0").unwrap();
    assert_eq!(canonical_json(&integer).unwrap(), b"1");
    assert_eq!(canonical_json(&float).unwrap(), b"1.0");
}

#[test]
fn canonical_output_accepts_exact_limit_and_rejects_one_over() {
    let exact = Value::String("a".repeat(MAX_CANONICAL_BINDING_BYTES - 2));
    assert_eq!(
        canonical_json(&exact).unwrap().len(),
        MAX_CANONICAL_BINDING_BYTES
    );
    let over = Value::String("a".repeat(MAX_CANONICAL_BINDING_BYTES - 1));
    assert!(matches!(
        canonical_json(&over),
        Err(CanonicalizationError::LimitExceeded)
    ));
}

#[test]
fn canonical_escaping_is_counted_after_encoding() {
    let value = Value::String("\n".repeat(MAX_CANONICAL_BINDING_BYTES / 2));
    assert!(matches!(
        canonical_json(&value),
        Err(CanonicalizationError::LimitExceeded)
    ));
}

#[test]
fn canonical_deep_values_use_iterative_traversal() {
    let mut value = Value::Null;
    for _ in 0..256 {
        value = Value::Array(vec![value]);
    }
    let expected = format!("{}null{}", "[".repeat(256), "]".repeat(256));
    assert_eq!(canonical_json(&value).unwrap(), expected.as_bytes());
}

#[test]
fn transition_fixture_is_exhaustive() {
    let data = graph();
    let expected_states: BTreeSet<_> = data.states.iter().map(String::as_str).collect();
    let actual_states: BTreeSet<_> = ApprovalState::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(expected_states, actual_states);
    assert_eq!(data.states.len(), ApprovalState::ALL.len());
    let expected_events: BTreeSet<_> = data.events.iter().map(String::as_str).collect();
    let actual_events: BTreeSet<_> = Transition::ALL.iter().map(|e| e.name()).collect();
    assert_eq!(expected_events, actual_events);
    assert_eq!(data.events.len(), Transition::ALL.len());
    assert_eq!(data.allowed.len(), 22);
    let mut pairs = BTreeSet::new();
    for edge in &data.allowed {
        assert!(expected_states.contains(edge.from.as_str()));
        assert!(expected_states.contains(edge.to.as_str()));
        assert!(expected_events.contains(edge.event.as_str()));
        assert!(
            pairs.insert((edge.from.as_str(), edge.event.as_str())),
            "duplicate edge"
        );
    }
    let mut visited = 0;
    for state in ApprovalState::ALL {
        for event in Transition::ALL {
            visited += 1;
            let expected = data
                .allowed
                .iter()
                .find(|edge| edge.from == state.name() && edge.event == event.name());
            match (expected, state.transition(event)) {
                (Some(edge), Ok(next)) => assert_eq!(next.name(), edge.to),
                (None, Err(error)) => {
                    assert_eq!(error.state, state);
                    assert_eq!(error.event, event);
                }
                other => panic!(
                    "{} + {} mismatch: expected edge {:?}, actual {:?}",
                    state.name(),
                    event.name(),
                    other.0.map(|e| &e.to),
                    other.1
                ),
            }
        }
    }
    assert_eq!(visited, 90);
}

#[test]
fn terminal_states_cannot_become_runnable() {
    for state in ApprovalState::ALL
        .into_iter()
        .filter(|state| state.is_terminal())
    {
        for event in Transition::ALL {
            if event == Transition::Restart {
                assert_eq!(state.transition(event), Ok(state));
            } else {
                assert!(
                    state.transition(event).is_err(),
                    "terminal state resurrected"
                );
            }
        }
    }
}

#[test]
fn approval_and_claim_do_not_start_execution() {
    let ready = ApprovalState::AwaitingApproval
        .transition(Transition::Approve)
        .unwrap();
    assert_eq!(ready, ApprovalState::Ready);
    assert!(ready.transition(Transition::Approve).is_err());
    assert!(ready.transition(Transition::Start).is_err());
    let claimed = ready.transition(Transition::Claim).unwrap();
    assert_eq!(claimed, ApprovalState::Claimed);
    assert!(claimed.transition(Transition::Claim).is_err());
    assert_eq!(
        claimed.transition(Transition::Start),
        Ok(ApprovalState::Started)
    );
    // Graph legality only: there is intentionally no executor or usable grant.
}

#[test]
fn expiry_includes_claimed_state() {
    for state in [
        ApprovalState::AwaitingApproval,
        ApprovalState::Ready,
        ApprovalState::Claimed,
    ] {
        assert_eq!(
            state.transition(Transition::Expire),
            Ok(ApprovalState::Expired)
        );
    }
    assert!(
        ApprovalState::Started
            .transition(Transition::Expire)
            .is_err()
    );
}

#[test]
fn restart_withdraws_prestart_and_marks_started_unknown() {
    let frozen: Value = serde_json::from_str(FREEZE).unwrap();
    for state in ApprovalState::ALL {
        let recovered = state.transition(Transition::Restart).unwrap();
        if state.is_terminal() {
            assert_eq!(recovered, state);
        } else {
            assert_eq!(frozen["restart"][state.name()], recovered.name());
        }
        assert!(recovered.is_terminal());
    }
    assert_eq!(frozen["restart"]["automatic_replay"], false);
}

#[test]
fn binding_fixture_domain_and_mutation_coverage() {
    let fixture = vectors().binding;
    let value: Value = serde_json::from_str(&fixture.input_json).unwrap();
    let bytes = canonical_json(&value).unwrap();
    assert_eq!(bytes, fixture.canonical_json.as_bytes());
    let baseline = binding_digest(&bytes);
    assert_eq!(baseline, fixture.domain_separated_sha256);
    assert_ne!(
        baseline,
        digest(&bytes),
        "binding domain separation omitted"
    );
    let frozen: Value = serde_json::from_str(FREEZE).unwrap();
    let fields = frozen["binding_fields"].as_array().unwrap();
    assert_eq!(fields.len(), value.as_object().unwrap().len());
    for field in fields {
        let field = field.as_str().unwrap();
        let mut changed = value.clone();
        let replacement = match &value[field] {
            Value::String(_) if field == "canonical_arguments" => json!("{\"a\":2}"),
            Value::String(original) => json!(format!("{original}-changed")),
            Value::Number(original) => json!(original.as_u64().unwrap() + 1),
            _ => panic!("missing or unexpected binding fixture field: {field}"),
        };
        changed[field] = replacement;
        assert_ne!(
            baseline,
            binding_digest(&canonical_json(&changed).unwrap()),
            "unbound field: {field}"
        );
    }
}
