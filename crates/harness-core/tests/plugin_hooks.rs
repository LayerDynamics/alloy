#![cfg(any(target_os = "linux", target_os = "macos"))]
use harness_core::plugin_host::{
    Decision, HookInvocation, Limits, PluginConfig, PluginHost, sandbox::Sandbox,
};
use serde_json::json;
use std::{
    path::Path,
    process::Command,
    sync::OnceLock,
    time::{Duration, Instant},
};
use tool_registry::{
    execution::CancellationToken,
    result::{InlineToolResult, ToolResult},
};
fn fixture() -> &'static Path {
    static FIXTURE: OnceLock<std::path::PathBuf> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            // The descriptor-limit helper is a re-exec of this test binary. Reuse
            // the parent's fixture rather than compiling and cold-launching a
            // second executable while the parallel protocol tests are running.
            if std::env::var("ALLOY_W5_FD_HELPER").as_deref() == Ok("1") {
                let bin = std::path::PathBuf::from(
                    std::env::var_os("ALLOY_W5_FD_FIXTURE")
                        .expect("descriptor helper requires the parent's compiled fixture"),
                );
                assert!(bin.is_absolute() && bin.is_file());
                return bin;
            }
            let dir = tempfile::tempdir().unwrap().keep();
            let bin = dir.join("tool-hook-fixture");
            let status = Command::new("rustc")
                .args(["--edition=2024", "-o"])
                .arg(&bin)
                .arg(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("tests/fixtures/tool_hook_plugin.rs"),
                )
                .status()
                .unwrap();
            assert!(status.success());
            // Finish the platform's first-exec work before exercising bounded
            // hook IPC. In particular, macOS can delay a freshly linked image
            // before main; that delay is unrelated to descriptor inheritance.
            let status = Command::new(&bin)
                .args(["fixture-readiness", "allow"])
                .arg(dir.join("readiness"))
                .stdin(std::process::Stdio::null())
                .status()
                .unwrap();
            assert!(status.success(), "compiled plugin fixture failed to start");
            bin
        })
        .as_path()
}
fn limits() -> Limits {
    Limits {
        message_bytes: 16384,
        diagnostic_bytes: 4096,
        annotation_bytes: 1024,
        reason_bytes: 256,
        timeout: Duration::from_millis(500),
    }
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(5)
}
fn start_host(
    configs: Vec<PluginConfig>,
    limits: Limits,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Result<PluginHost, harness_core::plugin_host::HostError> {
    PluginHost::start_with_helper(
        configs,
        limits,
        deadline,
        cancel,
        Path::new(env!("CARGO_BIN_EXE_alloy-plugin-pipe-helper")),
    )
}
fn config(dir: &Path, id: &str, mode: &str) -> PluginConfig {
    PluginConfig {
        id: id.into(),
        version: "1.0.0".into(),
        program: fixture().into(),
        args: vec![
            id.into(),
            mode.into(),
            dir.join("events").to_str().unwrap().into(),
        ],
        cwd: dir.into(),
        sandbox: Sandbox::Trusted,
    }
}
fn invocation() -> HookInvocation {
    HookInvocation {
        invocation_id: "inv-1".into(),
        session_id: "session-1".into(),
        call_id: "call-1".into(),
        definition: tool_registry::builtins::definitions(1000, 4096)
            .unwrap()
            .remove(0),
        schema_hash: "fixture-schema".into(),
        arguments: serde_json::Map::new(),
        approval: json!({"fixture_only":true}),
        deadline_unix_ms: 1000,
    }
}
#[test]
fn plugin_tool_hooks_order_veto_annotation() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let call = invocation();
    let original = serde_json::to_value(&call).unwrap();
    let mut host = start_host(
        vec![
            config(dir.path(), "a", "allow"),
            config(dir.path(), "b", "allow"),
        ],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let before = host.before(&call, deadline(), &cancel);
    assert_eq!(before.decision, Decision::Allow);
    assert_eq!(
        before
            .receipts
            .iter()
            .map(|r| r.plugin_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    let result = ToolResult::Inline(InlineToolResult {
        call_id: call.call_id.clone(),
        is_error: false,
        content: json!("original"),
    });
    let saved = serde_json::to_value(&result).unwrap();
    let after = host.after(&call, &result, 1, deadline(), &cancel);
    assert_eq!(after.len(), 2);
    assert!(
        after
            .iter()
            .all(|r| r.diagnostic.is_none() && r.annotations.len() == 1)
    );
    assert_eq!(serde_json::to_value(&call).unwrap(), original);
    assert_eq!(serde_json::to_value(&result).unwrap(), saved);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("events")).unwrap(),
        "a:initialize\nb:initialize\na:before\nb:before\na:after\nb:after\n"
    );
    drop(host);
    let other = tempfile::tempdir().unwrap();
    let mut host = start_host(
        vec![
            config(other.path(), "a", "veto"),
            config(other.path(), "b", "allow"),
        ],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let before = host.before(&call, deadline(), &cancel);
    let mut execution_count = 0;
    if before.decision == Decision::Allow {
        execution_count += 1;
    }
    assert_eq!(execution_count, 0);
    assert_eq!(before.receipts.len(), 1);
    assert!(
        !std::fs::read_to_string(other.path().join("events"))
            .unwrap()
            .contains("b:before")
    );
}
#[test]
fn plugin_protocol_timeout_and_sandbox() {
    // Protocol half of P2-11. Native Seatbelt evidence is a separate Mac test.
    let cancel = CancellationToken::default();
    let dir = tempfile::tempdir().unwrap();
    assert!(
        start_host(
            vec![config(dir.path(), "a", "version")],
            limits(),
            deadline(),
            &cancel
        )
        .is_err()
    );
    for mode in ["hang", "crash", "malformed", "flood", "mutation"] {
        let mut host = start_host(
            vec![config(dir.path(), "a", mode)],
            limits(),
            deadline(),
            &cancel,
        )
        .unwrap();
        let now = Instant::now();
        let outcome = host.before(&invocation(), deadline(), &cancel);
        assert_eq!(outcome.decision, Decision::Veto, "{mode}");
        assert!(now.elapsed() < Duration::from_secs(3), "{mode}");
        assert!(outcome.receipts[0].diagnostic.is_some());
    }
}
#[test]
fn unsolicited_output_between_calls_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut host = start_host(
        vec![config(dir.path(), "a", "unsolicited")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let marker = dir.path().join("events.a.unsolicited");
    let until = Instant::now() + Duration::from_secs(3);
    while !marker.exists() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(marker.exists(), "fixture did not send unsolicited output");
    let outcome = host.before(&invocation(), deadline(), &cancel);
    assert_eq!(outcome.decision, Decision::Veto);
    assert!(
        outcome.receipts[0]
            .diagnostic
            .as_ref()
            .unwrap()
            .contains("unsolicited")
    );
    assert!(
        !std::fs::read_to_string(dir.path().join("events"))
            .unwrap()
            .contains("a:before")
    );
}
#[test]
fn raced_preconstructed_response_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut large_limits = limits();
    large_limits.message_bytes = 512 * 1024;
    let mut host = start_host(
        vec![config(dir.path(), "a", "raced_stale")],
        large_limits,
        deadline(),
        &cancel,
    )
    .unwrap();
    let mut call = invocation();
    call.arguments
        .insert("payload".into(), json!("x".repeat(256 * 1024)));
    let outcome = host.before(&call, deadline(), &cancel);
    assert_eq!(outcome.decision, Decision::Veto);
    let diagnostic = outcome.receipts[0].diagnostic.as_deref().unwrap();
    assert!(
        diagnostic.contains("unsolicited or multiple response frames")
            || diagnostic.contains("response identity mismatch"),
        "{diagnostic}"
    );
    assert!(dir.path().join("events.a.race_started").exists());
}
#[test]
fn plugin_receives_canonical_payload_and_cannot_replace_result() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut call = invocation();
    call.arguments.insert("path".into(), json!("fixture.txt"));
    call.approval = json!({"binding":"approved", "version":1});
    let result = ToolResult::Inline(InlineToolResult {
        call_id: call.call_id.clone(),
        is_error: false,
        content: json!({"bytes":"original"}),
    });
    let mut cfg = config(dir.path(), "a", "payload");
    cfg.args.push(serde_json::to_string(&call).unwrap());
    cfg.args.push(serde_json::to_string(&result).unwrap());
    let mut host = start_host(vec![cfg], limits(), deadline(), &cancel).unwrap();
    assert_eq!(
        host.before(&call, deadline(), &cancel).decision,
        Decision::Allow
    );
    assert!(
        host.after(&call, &result, 37, deadline(), &cancel)[0]
            .diagnostic
            .is_none()
    );
    let ids = std::fs::read_to_string(dir.path().join("events.a.request_ids")).unwrap();
    let ids: Vec<_> = ids.lines().collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    assert!(
        ids.iter()
            .all(|id| id.parse::<u64>().is_ok_and(|n| n < (1u64 << 53)))
    );
    drop(host);
    let mut host = start_host(
        vec![config(dir.path(), "a", "after_mutation")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    assert_eq!(
        host.before(&call, deadline(), &cancel).decision,
        Decision::Allow
    );
    let original = serde_json::to_value(&result).unwrap();
    assert!(
        host.after(&call, &result, 37, deadline(), &cancel)[0]
            .diagnostic
            .is_some()
    );
    assert_eq!(serde_json::to_value(&result).unwrap(), original);
}
#[test]
fn cancellation_reaps_and_after_failure_cannot_mutate_original_result() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut host = start_host(
        vec![config(dir.path(), "a", "hang")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let trigger = cancel.clone();
    let worker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        trigger.cancel();
    });
    assert_eq!(
        host.before(&invocation(), deadline(), &cancel).decision,
        Decision::Veto
    );
    worker.join().unwrap();
    let cancel = CancellationToken::default();
    let mut host = start_host(
        vec![config(dir.path(), "a", "after_failure")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let call = invocation();
    assert_eq!(
        host.before(&call, deadline(), &cancel).decision,
        Decision::Allow
    );
    let result = ToolResult::Inline(InlineToolResult {
        call_id: call.call_id.clone(),
        is_error: false,
        content: json!("unchanged"),
    });
    let saved = serde_json::to_value(&result).unwrap();
    assert!(
        host.after(&call, &result, 1, deadline(), &cancel)[0]
            .diagnostic
            .is_some()
    );
    assert_eq!(serde_json::to_value(&result).unwrap(), saved);
}

#[path = "plugin_hooks/hardening.rs"]
mod hardening;
