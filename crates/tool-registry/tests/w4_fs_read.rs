#![cfg(unix)]
//! Scoped file-read tests; no live model/daemon dispatch is enabled.
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tool_registry::{
    builtins::fs::{ReadRequest, ReadScope},
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, ExecutionError, OutputSink},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir();
        loop {
            let path = base.join(format!(
                "alloy-w4-read-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("fixture directory: {e}"),
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
    let dir = Temp::new();
    fs::create_dir(dir.0.join("root")).unwrap();
    fs::write(dir.0.join("root/data"), b"0123456789").unwrap();
    let scope = ReadScope::open(dir.0.join("root")).unwrap();
    (dir, scope)
}
fn collect(scope: &ReadScope, path: &str, offset: u64, length: u64, max: u64) -> (u64, Vec<u8>) {
    let context = context(max, CancellationToken::default());
    let mut sink = BoundedOutput::new(&context, max as usize).unwrap();
    let report = scope
        .read_into(
            &ReadRequest::new(path, offset, length).unwrap(),
            &context,
            &mut sink,
        )
        .unwrap();
    (report.bytes_read, sink.finish().unwrap())
}
#[test]
fn reads_exact_requested_range() {
    let (_d, s) = fixture();
    assert_eq!(collect(&s, "data", 2, 4, 4), (4, b"2345".to_vec()));
}
#[test]
fn eof_empty_and_past_eof_are_bounded() {
    let (d, s) = fixture();
    fs::write(d.0.join("root/empty"), b"").unwrap();
    assert_eq!(collect(&s, "empty", 0, 8, 8), (0, vec![]));
    assert_eq!(collect(&s, "data", 8, 8, 8), (2, b"89".to_vec()));
    assert_eq!(collect(&s, "data", 99, 8, 8), (0, vec![]));
}
#[test]
fn large_binary_payload_streams_without_text_conversion() {
    let (d, s) = fixture();
    let bytes: Vec<u8> = (0..150000).map(|n| (n % 251) as u8).collect();
    fs::write(d.0.join("root/binary"), &bytes).unwrap();
    assert_eq!(
        collect(&s, "binary", 0, bytes.len() as u64, bytes.len() as u64).1,
        bytes
    );
}
#[test]
fn invalid_paths_and_overflow_reject() {
    for path in [
        "",
        "/etc/passwd",
        "../outside",
        "a/../data",
        "./data",
        "a//data",
        "a/./data",
        "data/",
        "a\0b",
    ] {
        assert!(ReadRequest::new(path, 0, 1).is_err(), "{path:?}");
    }
    assert!(ReadRequest::new("data", 0, 0).is_err());
    assert!(ReadRequest::new("data", u64::MAX, 1).is_err());
}
#[test]
fn over_limit_request_aborts_without_output() {
    let (_d, s) = fixture();
    let c = context(4, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 4).unwrap();
    assert!(
        s.read_into(&ReadRequest::new("data", 0, 5).unwrap(), &c, &mut out)
            .is_err()
    );
    assert_eq!(out.finish(), Err(ExecutionError::IncompleteOutput));
}
#[test]
fn pre_cancel_aborts_and_other_operation_is_unaffected() {
    let (_d, s) = fixture();
    let token = CancellationToken::default();
    token.cancel();
    let c = context(4, token);
    let mut out = BoundedOutput::new(&c, 4).unwrap();
    assert!(
        s.read_into(&ReadRequest::new("data", 0, 4).unwrap(), &c, &mut out)
            .is_err()
    );
    assert!(out.finish().is_err());
    assert_eq!(collect(&s, "data", 0, 4, 4).1, b"0123");
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
fn cancellation_after_a_chunk_aborts_partial_output() {
    let (d, s) = fixture();
    fs::write(d.0.join("root/large"), vec![1; 150000]).unwrap();
    let token = CancellationToken::default();
    let c = context(150000, token.clone());
    let mut out = CancelSink {
        token,
        chunks: 0,
        aborted: false,
    };
    assert!(
        s.read_into(&ReadRequest::new("large", 0, 150000).unwrap(), &c, &mut out)
            .is_err()
    );
    assert_eq!(out.chunks, 1);
    assert!(out.aborted);
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
fn sink_failure_aborts() {
    let (_d, s) = fixture();
    let c = context(4, CancellationToken::default());
    let mut out = RejectSink { aborted: false };
    assert!(
        s.read_into(&ReadRequest::new("data", 0, 4).unwrap(), &c, &mut out)
            .is_err()
    );
    assert!(out.aborted);
}
#[test]
fn final_and_intermediate_symlinks_reject() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture();
    fs::write(d.0.join("secret"), b"outside").unwrap();
    symlink(d.0.join("secret"), d.0.join("root/link")).unwrap();
    symlink(&d.0, d.0.join("root/parent")).unwrap();
    for p in ["link", "parent/secret"] {
        let c = context(8, CancellationToken::default());
        let mut out = BoundedOutput::new(&c, 8).unwrap();
        assert!(
            s.read_into(&ReadRequest::new(p, 0, 8).unwrap(), &c, &mut out)
                .is_err()
        );
        assert!(out.finish().is_err());
    }
    assert_eq!(fs::read(d.0.join("secret")).unwrap(), b"outside");
}
#[test]
fn hardlinks_and_directories_reject() {
    let (d, s) = fixture();
    fs::hard_link(d.0.join("root/data"), d.0.join("root/alias")).unwrap();
    fs::create_dir(d.0.join("root/sub")).unwrap();
    for p in ["data", "alias", "sub"] {
        let c = context(8, CancellationToken::default());
        let mut out = BoundedOutput::new(&c, 8).unwrap();
        assert!(
            s.read_into(&ReadRequest::new(p, 0, 8).unwrap(), &c, &mut out)
                .is_err()
        );
    }
}
#[test]
fn pinned_root_is_not_redirected_by_path_replacement() {
    use std::os::unix::fs::symlink;
    let (d, s) = fixture();
    fs::rename(d.0.join("root"), d.0.join("original")).unwrap();
    fs::create_dir(d.0.join("other")).unwrap();
    fs::write(d.0.join("other/data"), b"EVIL").unwrap();
    symlink(d.0.join("other"), d.0.join("root")).unwrap();
    assert_eq!(collect(&s, "data", 0, 4, 4).1, b"0123");
}
#[test]
fn fifo_is_rejected_without_waiting_for_a_writer() {
    use std::os::unix::fs::FileTypeExt;

    let (d, s) = fixture();
    let pipe = d.0.join("root/pipe");

    #[cfg(target_vendor = "apple")]
    {
        // Fixture setup only: direct executable invocation, not a shell.
        let output = std::process::Command::new("/usr/bin/mkfifo")
            .args(["-m", "600"])
            .arg(&pipe)
            .output()
            .expect("run macOS mkfifo for the FIFO fixture");

        assert!(
            output.status.success(),
            "mkfifo failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        use rustix::fs::{Mode, mkfifoat};

        let directory = fs::File::open(d.0.join("root")).unwrap();
        mkfifoat(&directory, "pipe", Mode::RUSR | Mode::WUSR).unwrap();
    }

    assert!(
        fs::symlink_metadata(&pipe).unwrap().file_type().is_fifo(),
        "fixture must actually be a FIFO"
    );

    let c = context(8, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 8).unwrap();

    assert!(
        s.read_into(&ReadRequest::new("pipe", 0, 8).unwrap(), &c, &mut out,)
            .is_err()
    );
    assert!(out.finish().is_err());
}
