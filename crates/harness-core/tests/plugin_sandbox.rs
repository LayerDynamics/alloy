#![cfg(target_os = "macos")]
use harness_core::plugin_host::sandbox::{Sandbox, profile};
use std::process::Command;
#[test]
fn plugin_protocol_timeout_and_sandbox() {
    let allowed = tempfile::tempdir().unwrap();
    let denied = tempfile::tempdir().unwrap();
    let good = allowed.path().join("good");
    let bad = denied.path().join("bad");
    std::fs::write(&good, "allowed").unwrap();
    std::fs::write(&bad, "secret").unwrap();
    // W5: isolate canonical-path access from alias traversal.
    let good = good.canonicalize().expect("canonical allowed fixture");
    let bad = bad.canonicalize().expect("canonical denied fixture");

    let policy = Sandbox::MacOs {
        read_files: vec!["/dev/null".into()],
        read_trees: vec![
            "/bin".into(),
            "/usr/lib".into(),
            "/System".into(),
            allowed.path().into(),
        ],
        write_trees: vec![],
    };
    let profile = profile(&policy).unwrap();
    let run = |path: &std::path::Path| {
        Command::new("/usr/bin/sandbox-exec")
            .args(["-p", &profile, "/bin/cat"])
            .arg(path)
            .env_clear()
            .output()
            .unwrap()
    };
    let positive = run(&good);
    eprintln!("W5 sandbox positive status: {:?}", positive.status);
    eprintln!(
        "W5 sandbox positive stdout: {:?}",
        String::from_utf8_lossy(&positive.stdout)
    );
    eprintln!(
        "W5 sandbox positive stderr: {:?}",
        String::from_utf8_lossy(&positive.stderr)
    );
    eprintln!("W5 sandbox profile:\\n{}", profile);
    eprintln!("W5 sandbox allowed file: {:?}", good);
    eprintln!("W5 sandbox denied file: {:?}", bad);

    assert!(
        positive.status.success(),
        "{}",
        String::from_utf8_lossy(&positive.stderr)
    );
    assert_eq!(positive.stdout, b"allowed");
    let negative = run(&bad);
    assert_eq!(
        negative.status.code(),
        Some(1),
        "Denied read must fail normally, not crash: status={:?}, stderr={:?}",
        negative.status,
        String::from_utf8_lossy(&negative.stderr)
    );
    assert!(negative.stdout.is_empty());
}

#[path = "plugin_sandbox/hardening.rs"]
mod hardening;
