#![cfg(any(target_os = "linux", target_os = "macos"))]
//! P2-07 actual primitive effects; W3-bound approval is covered in harness-core.
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use tool_registry::{
    builtins::{
        fs::{ListRequest, ReadRequest, ReadScope, WriteRequest, WriteScope},
        http::{HttpDestination, HttpError, HttpRequest, HttpScope},
        shell::{ShellError, ShellRequest, ShellScope},
    },
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, ExecutionError},
};
mod fixture {
    include!("fixtures/http_server.rs");
}
use fixture::{Server, response};
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "alloy-p2-07-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(std::fs::canonicalize(path).unwrap()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("fixture: {error}"),
            }
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn context(ms: u64, bytes: u64) -> ExecutionContext {
    ExecutionContext::new(
        &ToolPolicy {
            effect: ToolEffect::Destructive,
            approval: ApprovalRequirement::Always,
            timeout_ms: ms,
            max_output_bytes: bytes,
        },
        ms,
        bytes,
        CancellationToken::default(),
    )
    .unwrap()
}
#[test]
fn acceptance_shell_child() {
    let args: Vec<_> = std::env::args().collect();
    let Some(mode) = args
        .windows(2)
        .find(|pair| pair[0] == "--skip")
        .map(|pair| pair[1].as_str())
    else {
        return;
    };
    match mode {
        "marker" => std::fs::write("marker", b"executed").unwrap(),
        "hang" => std::thread::sleep(Duration::from_secs(20)),
        _ => panic!("fixture mode"),
    }
}
fn request(exe: &std::path::Path, cwd: &std::path::Path, mode: &str) -> ShellRequest {
    ShellRequest::new(
        exe.to_str().unwrap(),
        vec![
            "--exact".into(),
            "acceptance_shell_child".into(),
            "--nocapture".into(),
            "--test-threads=1".into(),
            "--skip".into(),
            mode.into(),
        ],
        cwd.to_str().unwrap(),
    )
    .unwrap()
}
fn reaped(pid: u32) {
    use rustix::process::{Pid, WaitId, WaitIdOptions, waitid};
    let pid = Pid::from_raw(i32::try_from(pid).unwrap()).unwrap();
    assert!(matches!(
        waitid(
            WaitId::Pid(pid),
            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT
        ),
        Err(rustix::io::Errno::CHILD)
    ));
}
#[test]
fn builtin_scope_and_output_limits() {
    let _guard = fixture::http_timing_test_guard();
    let dir = Directory::new();
    std::fs::create_dir(dir.0.join("root")).unwrap();
    std::fs::write(dir.0.join("root/data"), b"0123456789").unwrap();
    std::fs::write(dir.0.join("outside"), b"SECRET").unwrap();
    let read = ReadScope::open(dir.0.join("root")).unwrap();
    let ctx = context(10000, 4096);
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    assert_eq!(
        read.read_into(&ReadRequest::new("data", 2, 4).unwrap(), &ctx, &mut out)
            .unwrap()
            .bytes_read,
        4
    );
    assert_eq!(out.finish().unwrap(), b"2345");
    assert!(ReadRequest::new("../outside", 0, 4).is_err());
    std::os::unix::fs::symlink(dir.0.join("outside"), dir.0.join("root/link")).unwrap();
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    assert!(
        read.read_into(&ReadRequest::new("link", 0, 4).unwrap(), &ctx, &mut out)
            .is_err()
    );
    assert!(out.finish().is_err());
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    assert_eq!(
        read.list_into(&ListRequest::new(".", 10).unwrap(), &ctx, &mut out)
            .unwrap()
            .entries,
        2
    );
    let listing: serde_json::Value = serde_json::from_slice(&out.finish().unwrap()).unwrap();
    assert_eq!(listing["entries"][0]["name_utf8"], "data");
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    assert!(
        read.list_into(&ListRequest::new(".", 1).unwrap(), &ctx, &mut out)
            .is_err()
    );
    assert!(out.finish().is_err());
    let write = WriteScope::open(dir.0.join("root"), 8).unwrap();
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    write
        .write_into(
            &WriteRequest::new("new", "exact".into()).unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    assert!(
        write
            .write_into(
                &WriteRequest::new("new", "replace".into()).unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert_eq!(std::fs::read(dir.0.join("root/new")).unwrap(), b"exact");
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    assert!(
        write
            .write_into(
                &WriteRequest::new("too-large", "123456789".into()).unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert!(!dir.0.join("root/too-large").exists());
    let exe = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
    let shell = ShellScope::new(vec![exe.clone()], vec![dir.0.join("root")], 65536).unwrap();
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    let report = shell
        .run(
            &request(&exe, &dir.0.join("root"), "marker"),
            &ctx,
            &mut out,
        )
        .unwrap();
    assert!(report.success);
    reaped(report.pid);
    assert!(dir.0.join("root/marker").exists());
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    let denied = shell
        .run(&request(&exe, &dir.0, "marker"), &ctx, &mut out)
        .unwrap_err();
    assert!(!denied.started);
    assert!(!dir.0.join("marker").exists());
    let timed = context(500, 4096);
    let mut out = BoundedOutput::new(&timed, 4096).unwrap();
    let failed = shell
        .run(
            &request(&exe, &dir.0.join("root"), "hang"),
            &timed,
            &mut out,
        )
        .unwrap_err();
    assert!(failed.started);
    assert!(matches!(
        failed.cause,
        ShellError::Execution(ExecutionError::DeadlineExceeded)
    ));
    assert!(failed.cleanup_errors.is_empty());
    reaped(failed.pid.unwrap());
    let denied = Server::new(|_, stream, _| response(stream, 200, "", b"secret"));
    let target = denied.origin.clone();
    let server = Server::new(move |line, stream, _| {
        if line.contains(" /redirect ") {
            response(stream, 302, &format!("Location: {target}/secret\r\n"), b"")
        } else {
            response(stream, 200, "", b"ok")
        }
    });
    let http = HttpScope::new(
        vec![HttpDestination::new(&server.origin, "127.0.0.1".parse().unwrap()).unwrap()],
        2,
        16,
        8192,
        Duration::from_secs(2),
    )
    .unwrap();
    let ctx = context(10000, 4096);
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    let report = http
        .fetch(
            &HttpRequest::new(&server.origin, "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    assert_eq!(report.body_bytes, 2);
    assert_eq!(report.status, 200);
    let mut out = BoundedOutput::new(&ctx, 4096).unwrap();
    let rejection = http
        .fetch(
            &HttpRequest::new(&format!("{}/redirect", server.origin), "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap_err();
    assert!(matches!(rejection.cause, HttpError::Policy(_)));
    assert_eq!(rejection.requests_attempted, 1);
    assert_eq!(server.calls.lock().unwrap().len(), 2);
    assert!(denied.calls.lock().unwrap().is_empty());
    assert!(out.finish().is_err());
    let tiny = context(10000, 8);
    let mut out = BoundedOutput::new(&tiny, 8).unwrap();
    let overflow = http
        .fetch(
            &HttpRequest::new(&server.origin, "GET").unwrap(),
            &tiny,
            &mut out,
        )
        .unwrap_err();
    assert!(matches!(
        overflow.cause,
        HttpError::Execution(ExecutionError::OutputLimit)
    ));
    assert!(out.finish().is_err());
    assert_eq!(std::fs::read(dir.0.join("outside")).unwrap(), b"SECRET");
}
