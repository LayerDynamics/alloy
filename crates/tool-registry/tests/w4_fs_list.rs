#![cfg(unix)]
//! W4 fs.list tests: real local directories, no live dispatch or artifact claims.
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tool_registry::{
    builtins::fs::{ListRequest, ReadScope},
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, ExecutionError, OutputSink},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "alloy-w4-list-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
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
            effect: ToolEffect::Read,
            approval: ApprovalRequirement::Never,
            timeout_ms: 60000,
            max_output_bytes: max,
        },
        60000,
        max,
        token,
    )
    .unwrap()
}
fn fixture() -> (Temp, ReadScope) {
    let d = Temp::new();
    fs::create_dir(d.0.join("root")).unwrap();
    let s = ReadScope::open(d.0.join("root")).unwrap();
    (d, s)
}
fn list(s: &ReadScope, path: &str, count: u64, max: u64) -> Vec<u8> {
    let c = context(max, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, max as usize).unwrap();
    let report = s
        .list_into(&ListRequest::new(path, count).unwrap(), &c, &mut out)
        .unwrap();
    let bytes = out.finish().unwrap();
    assert_eq!(report.output_bytes, bytes.len() as u64);
    assert_eq!(
        report.entries,
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["entries"]
            .as_array()
            .unwrap()
            .len() as u64
    );
    bytes
}
fn names(bytes: &[u8]) -> Vec<Vec<u8>> {
    let json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    json["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| serde_json::from_value(e["name_bytes"].clone()).unwrap())
        .collect()
}
#[test]
fn empty_root_excludes_dot_entries() {
    let (_d, s) = fixture();
    assert_eq!(list(&s, ".", 1, 14), b"{\"entries\":[]}");
}
#[test]
fn raw_names_are_sorted_not_creation_order() {
    let (d, s) = fixture();
    for n in ["z", "a", "m"] {
        fs::write(d.0.join("root").join(n), b"").unwrap();
    }
    assert_eq!(
        names(&list(&s, ".", 3, 4096)),
        vec![b"a".to_vec(), b"m".to_vec(), b"z".to_vec()]
    );
}
#[test]
fn repeated_root_listings_have_independent_cursors() {
    let (d, s) = fixture();
    fs::write(d.0.join("root/file"), b"").unwrap();
    let expected = list(&s, ".", 1, 4096);
    for _ in 0..5 {
        assert_eq!(list(&s, ".", 1, 4096), expected);
    }
}
#[test]
fn nested_directory_is_scoped() {
    let (d, s) = fixture();
    fs::create_dir_all(d.0.join("root/a/b")).unwrap();
    fs::write(d.0.join("root/a/b/file"), b"").unwrap();
    assert_eq!(names(&list(&s, "a/b", 1, 4096)), vec![b"file".to_vec()]);
}
#[test]
fn entry_limit_rejects_instead_of_truncating() {
    let (d, s) = fixture();
    for n in ["a", "b"] {
        fs::write(d.0.join("root").join(n), b"").unwrap();
    }
    let c = context(4096, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.list_into(&ListRequest::new(".", 1).unwrap(), &c, &mut out)
            .is_err()
    );
    assert_eq!(out.finish(), Err(ExecutionError::IncompleteOutput));
    assert_eq!(names(&list(&s, ".", 2, 4096)).len(), 2);
}
#[test]
fn serialized_byte_limit_is_exact() {
    let (d, s) = fixture();
    fs::write(d.0.join("root/one"), b"").unwrap();
    let expected = list(&s, ".", 1, 4096);
    assert_eq!(list(&s, ".", 1, expected.len() as u64), expected);
    let max = expected.len() - 1;
    let c = context(max as u64, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, max).unwrap();
    assert!(
        s.list_into(&ListRequest::new(".", 1).unwrap(), &c, &mut out)
            .is_err()
    );
    assert!(out.finish().is_err());
}
#[test]
fn too_small_empty_envelope_aborts() {
    let (_d, s) = fixture();
    let c = context(13, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 13).unwrap();
    assert!(
        s.list_into(&ListRequest::new(".", 1).unwrap(), &c, &mut out)
            .is_err()
    );
    assert!(out.finish().is_err());
}
#[test]
fn request_paths_and_zero_count_reject() {
    for path in [
        "", "/", "/etc", "..", "../root", "a/../b", "./a", "a/./b", "a//b", "a/", "a\0b",
    ] {
        assert!(ListRequest::new(path, 1).is_err(), "{path:?}");
    }
    assert!(ListRequest::new(".", 0).is_err());
}
#[test]
fn escaped_names_roundtrip() {
    let (d, s) = fixture();
    let name = "a\"\n";
    fs::write(d.0.join("root").join(name), b"").unwrap();
    let bytes = list(&s, ".", 1, 4096);
    assert_eq!(names(&bytes), vec![name.as_bytes().to_vec()]);
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["entries"][0]["name_utf8"].as_str(), Some(name));
}

// APFS cannot create invalid UTF-8 names. The actual production encoding
// helper is tested with arbitrary bytes on every target in the library tests.
#[cfg(not(target_vendor = "apple"))]
#[test]
fn non_utf8_filesystem_names_roundtrip() {
    use std::os::unix::ffi::OsStringExt;
    let (d, s) = fixture();
    let raw = vec![b'x', 0xff];
    fs::write(
        d.0.join("root")
            .join(std::ffi::OsString::from_vec(raw.clone())),
        b"",
    )
    .unwrap();
    let bytes = list(&s, ".", 1, 4096);
    assert_eq!(names(&bytes), vec![raw]);
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["entries"][0]["name_utf8"].is_null());
}
#[test]
fn final_and_parent_symlink_targets_reject() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture();
    fs::create_dir(d.0.join("outside")).unwrap();
    fs::write(d.0.join("outside/secret"), b"unchanged").unwrap();
    symlink(d.0.join("outside"), d.0.join("root/link")).unwrap();
    symlink(&d.0, d.0.join("root/parent")).unwrap();
    for p in ["link", "parent/outside"] {
        let c = context(4096, CancellationToken::default());
        let mut out = BoundedOutput::new(&c, 4096).unwrap();
        assert!(
            s.list_into(&ListRequest::new(p, 5).unwrap(), &c, &mut out)
                .is_err()
        );
        assert!(out.finish().is_err());
    }
    assert_eq!(fs::read(d.0.join("outside/secret")).unwrap(), b"unchanged");
}
#[test]
fn child_symlink_is_a_name_not_a_followed_target() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture();
    symlink(d.0.join("does-not-exist"), d.0.join("root/link")).unwrap();
    assert_eq!(names(&list(&s, ".", 1, 4096)), vec![b"link".to_vec()]);
}
#[test]
fn regular_file_and_missing_directory_reject() {
    let (d, s) = fixture();
    fs::write(d.0.join("root/file"), b"").unwrap();
    for p in ["file", "missing"] {
        let c = context(4096, CancellationToken::default());
        let mut out = BoundedOutput::new(&c, 4096).unwrap();
        assert!(
            s.list_into(&ListRequest::new(p, 5).unwrap(), &c, &mut out)
                .is_err()
        );
        assert!(out.finish().is_err());
    }
}
#[test]
fn pinned_root_is_not_redirected() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture();
    fs::write(d.0.join("root/original"), b"").unwrap();
    fs::rename(d.0.join("root"), d.0.join("moved")).unwrap();
    fs::create_dir(d.0.join("other")).unwrap();
    fs::write(d.0.join("other/evil"), b"").unwrap();
    symlink(d.0.join("other"), d.0.join("root")).unwrap();
    assert_eq!(names(&list(&s, ".", 1, 4096)), vec![b"original".to_vec()]);
}
#[test]
fn pre_cancelled_listing_aborts_only_its_sink() {
    let (_d, s) = fixture();
    let token = CancellationToken::default();
    token.cancel();
    let c = context(4096, token);
    let mut out = BoundedOutput::new(&c, 4096).unwrap();
    assert!(
        s.list_into(&ListRequest::new(".", 1).unwrap(), &c, &mut out)
            .is_err()
    );
    assert!(out.finish().is_err());
    assert_eq!(list(&s, ".", 1, 14), b"{\"entries\":[]}");
}
struct CancelSink {
    token: CancellationToken,
    chunks: usize,
    aborted: bool,
}
impl OutputSink for CancelSink {
    fn write_chunk(&mut self, _: &[u8]) -> Result<(), ExecutionError> {
        self.chunks += 1;
        self.token.cancel();
        Ok(())
    }
    fn abort(&mut self) {
        self.aborted = true;
    }
}
#[test]
fn cancellation_during_output_aborts_prefix() {
    let (d, s) = fixture();
    fs::write(d.0.join("root/entry"), b"").unwrap();
    let token = CancellationToken::default();
    let c = context(4096, token.clone());
    let mut sink = CancelSink {
        token,
        chunks: 0,
        aborted: false,
    };
    assert!(
        s.list_into(&ListRequest::new(".", 1).unwrap(), &c, &mut sink)
            .is_err()
    );
    assert_eq!(sink.chunks, 1);
    assert!(sink.aborted);
}
struct RejectSink {
    chunks: usize,
    aborted: bool,
}
impl OutputSink for RejectSink {
    fn write_chunk(&mut self, _: &[u8]) -> Result<(), ExecutionError> {
        self.chunks += 1;
        Err(ExecutionError::OutputLimit)
    }
    fn abort(&mut self) {
        self.aborted = true;
    }
}
#[test]
fn sink_failure_aborts_listing() {
    let (_d, s) = fixture();
    let c = context(4096, CancellationToken::default());
    let mut sink = RejectSink {
        chunks: 0,
        aborted: false,
    };
    assert!(
        s.list_into(&ListRequest::new(".", 1).unwrap(), &c, &mut sink)
            .is_err()
    );
    assert_eq!(sink.chunks, 1);
    assert!(sink.aborted);
}
#[test]
fn directories_are_listed_without_recursive_expansion() {
    let (d, s) = fixture();
    fs::create_dir(d.0.join("root/child")).unwrap();
    fs::write(d.0.join("root/child/hidden"), b"").unwrap();
    assert_eq!(names(&list(&s, ".", 1, 4096)), vec![b"child".to_vec()]);
}
#[test]
fn distinct_directories_do_not_share_iterator_state() {
    let (d, s) = fixture();
    for sub in ["a", "b"] {
        fs::create_dir(d.0.join("root").join(sub)).unwrap();
        fs::write(d.0.join("root").join(sub).join(sub), b"").unwrap();
    }
    assert_eq!(names(&list(&s, "a", 1, 4096)), vec![b"a".to_vec()]);
    assert_eq!(names(&list(&s, "b", 1, 4096)), vec![b"b".to_vec()]);
    assert_eq!(names(&list(&s, "a", 1, 4096)), vec![b"a".to_vec()]);
}
