//! Actual binary option/preflight tests. No model or daemon is started.
use std::process::{Command, Output};
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_alloyd"))
        .args(args)
        .output()
        .expect("alloyd binary")
}
fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
#[test]
fn rpc_transport_help_lists_both_explicit_options() {
    let out = run(&["--help"]);
    assert!(out.status.success());
    let text = stderr(&out);
    assert!(text.contains("--rpc-max-notification-bytes"));
    assert!(text.contains("--rpc-write-timeout-ms"));
}
#[test]
fn rpc_transport_incomplete_pair_is_rejected_before_artifact_setup() {
    for args in [
        vec!["--rpc-max-notification-bytes", "4096"],
        vec!["--rpc-write-timeout-ms", "1000"],
    ] {
        let out = run(&args);
        assert!(!out.status.success());
        assert!(stderr(&out).contains("together"));
    }
}
#[test]
fn rpc_transport_valid_pair_reaches_the_existing_artifact_guard() {
    let out = run(&[
        "--rpc-max-notification-bytes",
        "4096",
        "--rpc-write-timeout-ms",
        "1000",
    ]);
    assert!(!out.status.success());
    let text = stderr(&out);
    assert!(text.contains("--artifact PATH is required"), "{text}");
    assert!(!text.contains("unknown argument"));
}
#[test]
fn rpc_transport_zero_and_malformed_values_are_rejected_before_boot() {
    for flag in ["--rpc-max-notification-bytes", "--rpc-write-timeout-ms"] {
        for value in ["0", "-1", "nope"] {
            let out = run(&[flag, value]);
            assert!(!out.status.success());
            assert!(stderr(&out).contains("positive"));
        }
    }
}
#[test]
fn rpc_transport_duplicate_flags_are_rejected_before_boot() {
    let out = run(&[
        "--rpc-max-notification-bytes",
        "4096",
        "--rpc-max-notification-bytes",
        "8192",
    ]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("more than once"));
}
