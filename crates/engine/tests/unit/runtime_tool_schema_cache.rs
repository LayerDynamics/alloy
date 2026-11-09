use super::{schema_bound_disk_identity, tool_schema_compatible};
#[test]
fn changed_or_removed_schemas_fail_the_production_compatibility_gate() {
    let a = [1; 32];
    let b = [2; 32];
    assert!(tool_schema_compatible(None, None));
    assert!(tool_schema_compatible(Some(&a), Some(&a)));
    assert!(!tool_schema_compatible(Some(&a), Some(&b)));
    assert!(!tool_schema_compatible(Some(&a), None));
    assert!(!tool_schema_compatible(None, Some(&a)));
}
#[test]
fn disk_keys_bind_schema_identity_and_preserve_the_text_only_key() {
    let identity = serde_json::json!({"format":"fixture", "template":"fixed"});
    let a = [1; 32];
    let b = [2; 32];
    assert_eq!(
        schema_bound_disk_identity(identity.clone(), None),
        crate::tiered_kv::digest(identity.to_string().as_bytes())
    );
    let first = schema_bound_disk_identity(identity.clone(), Some(&a));
    assert_ne!(
        first,
        schema_bound_disk_identity(identity.clone(), Some(&b))
    );
    assert_ne!(first, schema_bound_disk_identity(identity.clone(), None));
    assert_eq!(first, schema_bound_disk_identity(identity, Some(&a)));
}
