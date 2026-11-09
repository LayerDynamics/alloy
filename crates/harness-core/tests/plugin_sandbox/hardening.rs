use super::*;

#[test]
fn plugin_alias_read_works_but_symlink_cannot_escape() {
    use std::os::unix::fs::symlink;
    let allowed = tempfile::tempdir().unwrap();
    let denied = tempfile::tempdir().unwrap();
    let good = allowed.path().join("good");
    let bad = denied.path().join("bad");
    std::fs::write(&good, b"allowed").unwrap();
    std::fs::write(&bad, b"secret").unwrap();
    let escape = allowed.path().join("escape");
    symlink(bad.canonicalize().unwrap(), &escape).unwrap();
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
    assert!(
        positive.status.success(),
        "{:?}: {}",
        positive.status,
        String::from_utf8_lossy(&positive.stderr)
    );
    assert_eq!(positive.stdout, b"allowed");
    let negative = run(&escape);
    assert_eq!(negative.status.code(), Some(1));
    assert!(negative.stdout.is_empty());
    assert_eq!(std::fs::read(bad).unwrap(), b"secret");
}
