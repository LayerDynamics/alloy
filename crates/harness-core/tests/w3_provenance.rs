//! Reproduce frozen canonical byte/digest vectors in Rust without rewriting them.
use harness_core::approval::{CANONICALIZATION_VERSION, CONTRACT_VERSION, canonical_json};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[test]
fn frozen_vectors_are_reproduced_by_rust() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/w3_canonicalization_v1.json")).unwrap();
    assert_eq!(fixture["contract_version"], CONTRACT_VERSION);
    assert_eq!(
        fixture["canonicalization_version"],
        CANONICALIZATION_VERSION
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    let mut ids = BTreeSet::new();
    let mut generated = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        assert!(ids.insert(id));
        let input: Value = serde_json::from_str(case["input_json"].as_str().unwrap()).unwrap();
        let bytes = canonical_json(&input).unwrap();
        let text = String::from_utf8(bytes.clone()).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(text, case["canonical_json"].as_str().unwrap(), "{id}");
        assert_eq!(digest, case["sha256"].as_str().unwrap(), "{id}");
        generated.push(json!({"id":id,"canonical_json":text,"sha256":digest}));
    }
    let input: Value =
        serde_json::from_str(fixture["binding"]["input_json"].as_str().unwrap()).unwrap();
    let bytes = canonical_json(&input).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let mut digest = Sha256::new();
    digest.update(b"alloy.w3.invocation.v1\0");
    digest.update(&bytes);
    let digest = format!("{:x}", digest.finalize());
    assert_eq!(text, fixture["binding"]["canonical_json"].as_str().unwrap());
    assert_eq!(
        digest,
        fixture["binding"]["domain_separated_sha256"]
            .as_str()
            .unwrap()
    );
    let report = json!({
        "contract_version":CONTRACT_VERSION,
        "canonicalization_version":CANONICALIZATION_VERSION,
        "method":"Rust canonical_json plus pinned serde_json/sha2; comparison against unchanged golden vectors",
        "original_creation_method":"not established by reproduction",
        "cases":generated,
        "binding":{"canonical_json":text,"domain_separated_sha256":digest}
    });
    if let Ok(path) = std::env::var("ALLOY_W3_PROVENANCE_OUTPUT") {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer_pretty(file, &report).unwrap();
    }
}
