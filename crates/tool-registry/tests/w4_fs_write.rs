#![cfg(any(target_os = "linux", target_os = "macos"))]
//! W4 create-only write tests. No invalid-UTF-8 fixtures or live dispatch.
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tool_registry::{
    builtins::fs::{WritePublication, WriteRequest, WriteScope},
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, ExecutionError, OutputSink},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let p = std::env::temp_dir().join(format!(
                "alloy-w4-write-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&p) {
                Ok(()) => return Self(p),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("fixture: {e}"),
            }
        }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn context(max: u64, token: CancellationToken) -> ExecutionContext {
    ExecutionContext::new(
        &ToolPolicy {
            effect: ToolEffect::Write,
            approval: ApprovalRequirement::Policy,
            timeout_ms: 60000,
            max_output_bytes: max,
        },
        60000,
        max,
        token,
    )
    .unwrap()
}
fn fixture(max: u64) -> (Temp, WriteScope) {
    let d = Temp::new();
    fs::create_dir(d.0.join("root")).unwrap();
    let s = WriteScope::open(d.0.join("root"), max).unwrap();
    (d, s)
}
fn create(s: &WriteScope, path: &str, content: &str) {
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    let report = s
        .write_into(
            &WriteRequest::new(path, content.into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap();
    assert_eq!(report.path, path);
    assert_eq!(report.bytes_written, content.len() as u64);
    let json: serde_json::Value = serde_json::from_slice(&out.finish().unwrap()).unwrap();
    assert_eq!(json["path"], path);
    assert_eq!(json["bytes_written"], content.len() as u64);
}
fn no_staging(d: &Temp) {
    assert!(fs::read_dir(d.0.join("root")).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".alloy-write-")
    }));
}
#[test]
fn writes_exact_utf8_and_reports_byte_length() {
    let (d, s) = fixture(4096);
    let text = "résumé 雪\n";
    create(&s, "result", text);
    assert_eq!(fs::read(d.0.join("root/result")).unwrap(), text.as_bytes());
    no_staging(&d);
}
#[test]
fn empty_content_is_a_valid_new_file() {
    let (d, s) = fixture(1);
    create(&s, "empty", "");
    assert!(fs::read(d.0.join("root/empty")).unwrap().is_empty());
    no_staging(&d);
}
#[test]
fn nested_destination_uses_existing_scoped_parent() {
    let (d, s) = fixture(10);
    fs::create_dir_all(d.0.join("root/a/b")).unwrap();
    create(&s, "a/b/file", "exact");
    assert_eq!(fs::read(d.0.join("root/a/b/file")).unwrap(), b"exact");
    assert_eq!(fs::read_dir(d.0.join("root/a/b")).unwrap().count(), 1);
}
#[test]
fn no_implicit_parent_creation() {
    let (d, s) = fixture(10);
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    let e = s
        .write_into(
            &WriteRequest::new("missing/file", "x".into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(e.publication, WritePublication::NotPublished);
    assert!(!d.0.join("root/missing").exists());
    assert!(out.finish().is_err());
    no_staging(&d);
}
#[test]
fn existing_regular_file_is_unchanged() {
    let (d, s) = fixture(10);
    fs::write(d.0.join("root/file"), b"old").unwrap();
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    let e = s
        .write_into(
            &WriteRequest::new("file", "new".into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(e.publication, WritePublication::NotPublished);
    assert_eq!(fs::read(d.0.join("root/file")).unwrap(), b"old");
    no_staging(&d);
}
#[test]
fn final_symlink_does_not_modify_outside_target() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture(10);
    fs::write(d.0.join("outside"), b"old").unwrap();
    symlink(d.0.join("outside"), d.0.join("root/link")).unwrap();
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.write_into(
            &WriteRequest::new("link", "new".into()).unwrap(),
            &c,
            &mut out
        )
        .is_err()
    );
    assert_eq!(fs::read(d.0.join("outside")).unwrap(), b"old");
    assert!(
        fs::symlink_metadata(d.0.join("root/link"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    no_staging(&d);
}
#[test]
fn symlink_parent_cannot_escape_write_root() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture(10);
    fs::create_dir(d.0.join("outside")).unwrap();
    symlink(d.0.join("outside"), d.0.join("root/link")).unwrap();
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.write_into(
            &WriteRequest::new("link/marker", "new".into()).unwrap(),
            &c,
            &mut out
        )
        .is_err()
    );
    assert!(!d.0.join("outside/marker").exists());
    no_staging(&d);
}
#[test]
fn hardlink_destination_is_unchanged() {
    let (d, s) = fixture(10);
    fs::write(d.0.join("outside"), b"old").unwrap();
    fs::hard_link(d.0.join("outside"), d.0.join("root/alias")).unwrap();
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.write_into(
            &WriteRequest::new("alias", "new".into()).unwrap(),
            &c,
            &mut out
        )
        .is_err()
    );
    assert_eq!(fs::read(d.0.join("outside")).unwrap(), b"old");
    no_staging(&d);
}
#[test]
fn directory_destination_is_unchanged() {
    let (d, s) = fixture(10);
    fs::create_dir(d.0.join("root/dir")).unwrap();
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.write_into(
            &WriteRequest::new("dir", "new".into()).unwrap(),
            &c,
            &mut out
        )
        .is_err()
    );
    assert!(d.0.join("root/dir").is_dir());
    no_staging(&d);
}
#[test]
fn invalid_paths_and_reserved_names_reject() {
    for p in [
        "",
        ".",
        "/tmp/outside",
        "../outside",
        "a/../b",
        "./a",
        "a//b",
        "a/",
        "a\0b",
        ".alloy-write-user",
        "a/.alloy-write-user/file",
    ] {
        assert!(WriteRequest::new(p, "x".into()).is_err(), "{p:?}");
    }
}
#[test]
fn input_limit_is_distinct_from_output_limit() {
    let (d, s) = fixture(3);
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.write_into(
            &WriteRequest::new("large", "1234".into()).unwrap(),
            &c,
            &mut out
        )
        .is_err()
    );
    assert!(!d.0.join("root/large").exists());
    create(&s, "exact", "123");
    no_staging(&d);
}
#[test]
fn result_limit_is_checked_before_filesystem_mutation() {
    let (d, s) = fixture(10);
    let c = context(1, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 1).unwrap();
    let e = s
        .write_into(
            &WriteRequest::new("file", "x".into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(e.publication, WritePublication::NotPublished);
    assert_eq!(fs::read_dir(d.0.join("root")).unwrap().count(), 0);
    assert!(out.finish().is_err());
}
#[test]
fn precancelled_write_does_not_create_destination() {
    let (d, s) = fixture(10);
    let t = CancellationToken::default();
    t.cancel();
    let c = context(4096, t);
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    let e = s
        .write_into(
            &WriteRequest::new("file", "x".into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(e.publication, WritePublication::NotPublished);
    assert!(e.cleanup_error.is_none());
    assert_eq!(fs::read_dir(d.0.join("root")).unwrap().count(), 0);
}
struct RejectSink {
    aborted: bool,
}
impl OutputSink for RejectSink {
    fn write_chunk(&mut self, _: &[u8]) -> Result<(), ExecutionError> {
        Err(ExecutionError::OutputLimit)
    }
    fn abort(&mut self) {
        self.aborted = true;
    }
}
#[test]
fn sink_failure_after_publication_preserves_effect_status() {
    let (d, s) = fixture(10);
    let c = context(4096, CancellationToken::default());
    let mut out = RejectSink { aborted: false };
    let e = s
        .write_into(
            &WriteRequest::new("file", "exact".into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(e.publication, WritePublication::Published);
    assert!(out.aborted);
    assert_eq!(fs::read(d.0.join("root/file")).unwrap(), b"exact");
    no_staging(&d);
}
struct CancelSink {
    token: CancellationToken,
    aborted: bool,
}
impl OutputSink for CancelSink {
    fn write_chunk(&mut self, _: &[u8]) -> Result<(), ExecutionError> {
        self.token.cancel();
        Ok(())
    }
    fn abort(&mut self) {
        self.aborted = true;
    }
}
#[test]
fn cancellation_after_publication_does_not_claim_no_effect() {
    let (d, s) = fixture(10);
    let t = CancellationToken::default();
    let c = context(4096, t.clone());
    let mut out = CancelSink {
        token: t,
        aborted: false,
    };
    let e = s
        .write_into(
            &WriteRequest::new("file", "exact".into()).unwrap(),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(e.publication, WritePublication::Published);
    assert!(out.aborted);
    assert_eq!(fs::read(d.0.join("root/file")).unwrap(), b"exact");
    no_staging(&d);
}
#[test]
fn pinned_write_root_is_not_redirected() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture(10);
    fs::rename(d.0.join("root"), d.0.join("original")).unwrap();
    fs::create_dir(d.0.join("outside")).unwrap();
    symlink(d.0.join("outside"), d.0.join("root")).unwrap();
    create(&s, "marker", "exact");
    assert_eq!(fs::read(d.0.join("original/marker")).unwrap(), b"exact");
    assert!(!d.0.join("outside/marker").exists());
}
#[test]
fn input_can_exceed_small_result_budget() {
    let (d, s) = fixture(150000);
    let text = "x".repeat(150000);
    let c = context(128, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 128).unwrap();
    s.write_into(
        &WriteRequest::new("file", text.clone()).unwrap(),
        &c,
        &mut out,
    )
    .unwrap();
    assert_eq!(fs::read(d.0.join("root/file")).unwrap(), text.as_bytes());
    assert!(out.finish().unwrap().len() < 128);
    no_staging(&d);
}
#[test]
fn zero_input_limit_is_invalid() {
    let d = Temp::new();
    assert!(WriteScope::open(&d.0, 0).is_err());
}
