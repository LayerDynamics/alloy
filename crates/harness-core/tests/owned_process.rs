//! Execute native macOS process-global tests in a separate test driver.
#[cfg(target_os = "macos")]
#[test]
fn macos_identity_signals_survive_exec_reaping_and_pid_mismatch() {
    use std::{path::Path, process::Command};
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = tempfile::tempdir().unwrap();
    let driver = dir.path().join("owned-process-test");
    let output = Command::new("cc")
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DALLOY_OWNED_PROCESS_TEST",
            "-I",
        ])
        .arg(root.join("native"))
        .arg(root.join("tests/fixtures/owned_process.c"))
        .arg(root.join("native/owned_process.c"))
        .arg("-o")
        .arg(&driver)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(driver)
        .arg(env!("CARGO_BIN_EXE_alloy-plugin-exec-helper"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("OWNED_IDENTITY_EXEC_REAP_AUTOREAP_PASSED")
    );
}
