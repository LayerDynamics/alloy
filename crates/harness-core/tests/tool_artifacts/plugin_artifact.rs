use super::*;
use harness_core::plugin_host::{
    Decision, HookInvocation, Limits, PluginConfig, PluginHost, sandbox::Sandbox,
};
use std::{
    path::Path,
    process::Command,
    time::{Duration, Instant},
};
use tool_registry::{execution::CancellationToken, result::ArtifactToolResult};

#[test]
fn sandboxed_after_hook_cannot_replace_artifact_or_its_bytes() {
    let (artifact_dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let bytes = b"the original artifact bytes";
    let metadata = publish(&mut store, &artifacts, bytes);
    let object = artifact_dir
        .path()
        .join("artifacts/objects")
        .join(&metadata.artifact_ref)
        .canonicalize()
        .unwrap();
    let binary_dir = tempfile::tempdir().unwrap();
    let plugin_dir = tempfile::tempdir().unwrap();
    let binary = binary_dir.path().join("fixture");
    assert!(
        Command::new("rustc")
            .args(["--edition=2024", "-o"])
            .arg(&binary)
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tool_hook_plugin.rs"))
            .status()
            .unwrap()
            .success()
    );
    let binary = binary.canonicalize().unwrap();
    let cwd = plugin_dir.path().canonicalize().unwrap();
    let config = PluginConfig {
        id: "artifact".into(),
        version: "1.0.0".into(),
        program: binary.clone(),
        args: vec![
            "artifact".into(),
            "artifact_probe".into(),
            cwd.join("events").to_str().unwrap().into(),
            object.to_str().unwrap().into(),
        ],
        cwd: cwd.clone(),
        sandbox: Sandbox::MacOs {
            read_files: vec![binary, "/dev/null".into()],
            read_trees: vec!["/usr/lib".into(), "/System".into()],
            write_trees: vec![cwd],
        },
    };
    let limits = Limits {
        message_bytes: 16384,
        diagnostic_bytes: 4096,
        annotation_bytes: 1024,
        reason_bytes: 256,
        timeout: Duration::from_secs(3),
    };
    let cancel = CancellationToken::default();
    let deadline = || Instant::now() + Duration::from_secs(10);
    let mut host = PluginHost::start_with_helper(
        vec![config],
        limits,
        deadline(),
        &cancel,
        Path::new(env!("CARGO_BIN_EXE_alloy-plugin-pipe-helper")),
    )
    .unwrap();
    let call = HookInvocation {
        invocation_id: "inv-c".into(),
        session_id: "s".into(),
        call_id: "c".into(),
        definition: tool_registry::builtins::definitions(1000, 4096)
            .unwrap()
            .remove(0),
        schema_hash: "fixture".into(),
        arguments: serde_json::Map::new(),
        approval: json!({}),
        deadline_unix_ms: 1000,
    };
    let result = ToolResult::Artifact(ArtifactToolResult {
        call_id: "c".into(),
        artifact_ref: metadata.artifact_ref.clone(),
        is_error: false,
    });
    let original = serde_json::to_value(&result).unwrap();
    assert_eq!(
        host.before(&call, deadline(), &cancel).decision,
        Decision::Allow
    );
    let receipts = host.after(&call, &result, 5, deadline(), &cancel);
    assert!(
        receipts[0]
            .diagnostic
            .as_deref()
            .unwrap()
            .contains("invalid hook response contract"),
        "{:?}",
        receipts
    );
    assert_eq!(serde_json::to_value(&result).unwrap(), original);
    let mut verified = artifacts
        .lookup(&store, "s", "c", &metadata.artifact_ref)
        .unwrap()
        .verify()
        .unwrap();
    let mut restored = Vec::new();
    verified.read_to_end(&mut restored).unwrap();
    assert_eq!(restored, bytes);
    assert_eq!(std::fs::read(object).unwrap(), bytes);
}
