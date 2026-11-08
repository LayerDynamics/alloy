#![cfg(any(target_os = "linux", target_os = "macos"))]
//! Real owned subprocess fixtures: no model calls, shell interpolation or live dispatch.
use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use tool_registry::{
    builtins::shell::{ShellError, ShellRequest, ShellScope},
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, ExecutionError, OutputSink},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let p = std::env::temp_dir().join(format!(
                "alloy-w4-shell-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&p) {
                Ok(()) => return Self(fs::canonicalize(p).unwrap()),
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
fn executable() -> PathBuf {
    fs::canonicalize(std::env::current_exe().unwrap()).unwrap()
}
fn scope(d: &Temp) -> ShellScope {
    ShellScope::new(vec![executable()], vec![d.0.clone()], 65536).unwrap()
}
fn request(d: &Temp, mode: &str) -> ShellRequest {
    ShellRequest::new(
        executable().to_str().unwrap(),
        vec![
            "--exact".into(),
            "shell_child_fixture".into(),
            "--nocapture".into(),
            "--test-threads=1".into(),
            "--skip".into(),
            mode.into(),
        ],
        d.0.to_str().unwrap(),
    )
    .unwrap()
}
fn context(ms: u64, max: u64, t: CancellationToken) -> ExecutionContext {
    ExecutionContext::new(
        &ToolPolicy {
            effect: ToolEffect::Destructive,
            approval: ApprovalRequirement::Always,
            timeout_ms: ms,
            max_output_bytes: max,
        },
        ms,
        max,
        t,
    )
    .unwrap()
}
fn stream(bytes: &[u8], name: &str) -> Vec<u8> {
    let mut result = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if value["stream"] == name {
            result.extend(serde_json::from_value::<Vec<u8>>(value["data"].clone()).unwrap());
        }
    }
    result
}
fn execute(d: &Temp, mode: &str) -> (tool_registry::builtins::shell::ShellReport, Vec<u8>) {
    let c = context(10000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let report = scope(d).run(&request(d, mode), &c, &mut out).unwrap();
    let bytes = out.finish().unwrap();
    assert_eq!(report.output_bytes, bytes.len() as u64);
    assert_eq!(report.stdout_bytes, stream(&bytes, "stdout").len() as u64);
    assert_eq!(report.stderr_bytes, stream(&bytes, "stderr").len() as u64);
    (report, bytes)
}
fn assert_reaped(pid: u32) {
    use rustix::process::{Pid, WaitId, WaitIdOptions, waitid};
    let pid = Pid::from_raw(i32::try_from(pid).unwrap()).unwrap();
    assert!(matches!(
        waitid(
            WaitId::Pid(pid),
            WaitIdOptions::EXITED | WaitIdOptions::NOWAIT | WaitIdOptions::NOHANG
        ),
        Err(rustix::io::Errno::CHILD)
    ));
}
#[test]
fn shell_child_fixture() {
    let args: Vec<_> = std::env::args().collect();
    let Some(mode) = args
        .windows(2)
        .find(|p| p[0] == "--skip")
        .map(|p| p[1].as_str())
    else {
        return;
    };
    match mode {
        "hang" => std::thread::sleep(std::time::Duration::from_secs(60)),
        "stderr" => io::stderr().write_all(b"EXACT_STDERR").unwrap(),
        "binary" => io::stdout().write_all(&[0, 255, 10, 128]).unwrap(),
        "exit7" => std::process::exit(7),
        "stdin" => {
            let mut bytes = Vec::new();
            io::stdin().read_to_end(&mut bytes).unwrap();
            assert!(bytes.is_empty());
            io::stdout().write_all(b"STDIN_EOF").unwrap();
        }
        "env" => {
            assert!(std::env::var_os("PATH").is_none());
            io::stdout().write_all(b"ENV_CLEARED").unwrap();
        }
        "cwd" => io::stdout()
            .write_all(
                std::env::current_dir()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .as_bytes(),
            )
            .unwrap(),
        "flood" => {
            let bytes = [b'x'; 8192];
            loop {
                if io::stdout().write_all(&bytes).is_err() {
                    break;
                }
            }
        }
        _ if mode.starts_with("echo:") => io::stdout().write_all(&mode.as_bytes()[5..]).unwrap(),
        _ if mode.starts_with("marker:") => fs::write(&mode[7..], b"executed").unwrap(),
        _ => panic!("unknown fixture mode"),
    }
}
#[test]
fn exact_argv_is_not_interpreted_by_a_shell() {
    let d = Temp::new();
    let literal = "$(touch SHOULD_NOT_EXIST); * 'quoted'";
    let (r, b) = execute(&d, &format!("echo:{literal}"));
    assert!(r.success);
    assert!(String::from_utf8_lossy(&stream(&b, "stdout")).contains(literal));
    assert!(!d.0.join("SHOULD_NOT_EXIST").exists());
    assert_reaped(r.pid);
}
#[test]
fn stdout_and_stderr_remain_distinct() {
    let d = Temp::new();
    let (r, b) = execute(&d, "stderr");
    assert!(r.success);
    assert_eq!(stream(&b, "stderr"), b"EXACT_STDERR");
    assert!(!String::from_utf8_lossy(&stream(&b, "stdout")).contains("EXACT_STDERR"));
}
#[test]
fn binary_output_is_preserved() {
    let d = Temp::new();
    let (r, b) = execute(&d, "binary");
    assert!(r.success);
    assert!(
        stream(&b, "stdout")
            .windows(4)
            .any(|w| w == [0, 255, 10, 128])
    );
}
#[test]
fn stdin_is_closed() {
    let d = Temp::new();
    let (r, b) = execute(&d, "stdin");
    assert!(r.success);
    assert!(String::from_utf8_lossy(&stream(&b, "stdout")).contains("STDIN_EOF"));
}
#[test]
fn ambient_environment_is_not_inherited() {
    let d = Temp::new();
    let (r, b) = execute(&d, "env");
    assert!(r.success);
    assert!(String::from_utf8_lossy(&stream(&b, "stdout")).contains("ENV_CLEARED"));
}
#[test]
fn approved_cwd_is_used() {
    let d = Temp::new();
    let (r, b) = execute(&d, "cwd");
    assert!(r.success);
    assert!(String::from_utf8_lossy(&stream(&b, "stdout")).contains(d.0.to_str().unwrap()));
}
#[test]
fn nonzero_exit_is_not_reported_as_success() {
    let d = Temp::new();
    let (r, _) = execute(&d, "exit7");
    assert!(!r.success);
    assert_eq!(r.exit_code, Some(7));
    assert_reaped(r.pid);
}
#[test]
fn invalid_request_strings_reject() {
    assert!(ShellRequest::new("relative", vec![], "/tmp").is_err());
    assert!(ShellRequest::new("/bin/echo", vec![], "relative").is_err());
    assert!(ShellRequest::new("/bin/echo", vec!["a\0b".into()], "/tmp").is_err());
}
#[test]
fn unapproved_executable_scope_has_no_marker_effect() {
    let d = Temp::new();
    let sh = fs::canonicalize("/bin/sh").unwrap();
    let marker = d.0.join("marker");
    let req = ShellRequest::new(
        sh.to_str().unwrap(),
        vec!["-c".into(), format!("touch '{}'", marker.display())],
        d.0.to_str().unwrap(),
    )
    .unwrap();
    let c = context(10000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let e = scope(&d).run(&req, &c, &mut out).unwrap_err();
    assert!(!e.started);
    assert!(!marker.exists());
    assert!(out.finish().is_err());
}
#[test]
fn out_of_scope_cwd_never_spawns() {
    let d = Temp::new();
    let other = Temp::new();
    let req = request(
        &other,
        &format!("marker:{}", other.0.join("marker").display()),
    );
    let c = context(10000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let e = scope(&d).run(&req, &c, &mut out).unwrap_err();
    assert!(!e.started);
    assert!(!other.0.join("marker").exists());
}
#[test]
fn precancelled_invocation_never_spawns() {
    let d = Temp::new();
    let t = CancellationToken::default();
    t.cancel();
    let c = context(10000, 65536, t);
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let e = scope(&d)
        .run(
            &request(&d, &format!("marker:{}", d.0.join("marker").display())),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert!(!e.started);
    assert!(!d.0.join("marker").exists());
}
#[test]
fn deadline_kills_and_reaps_owned_child() {
    let d = Temp::new();
    let c = context(1000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let e = scope(&d)
        .run(&request(&d, "hang"), &c, &mut out)
        .unwrap_err();
    assert!(e.started);
    assert!(matches!(
        e.cause,
        ShellError::Execution(ExecutionError::DeadlineExceeded)
    ));
    assert!(e.cleanup_errors.is_empty(), "{e:?}");
    assert_reaped(e.pid.unwrap());
    assert!(out.finish().is_err());
}
#[test]
fn combined_serialized_output_limit_aborts_and_reaps() {
    let d = Temp::new();
    let c = context(10000, 512, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 512).unwrap();
    let e = scope(&d)
        .run(&request(&d, "flood"), &c, &mut out)
        .unwrap_err();
    assert!(e.started);
    assert!(matches!(
        e.cause,
        ShellError::Execution(ExecutionError::OutputLimit)
    ));
    assert!(e.cleanup_errors.is_empty(), "{e:?}");
    assert_reaped(e.pid.unwrap());
    assert!(out.finish().is_err());
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
fn running_cancel_reaps_without_cancelling_another_operation() {
    let d = Temp::new();
    let t = CancellationToken::default();
    let c = context(10000, 65536, t.clone());
    let mut out = CancelSink {
        token: t,
        aborted: false,
    };
    let e = scope(&d)
        .run(&request(&d, "hang"), &c, &mut out)
        .unwrap_err();
    assert!(e.started);
    assert!(out.aborted);
    assert!(e.cleanup_errors.is_empty(), "{e:?}");
    assert_reaped(e.pid.unwrap());
    assert!(execute(&d, "echo:other operation").0.success);
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
fn sink_failure_preserves_started_status_and_reaps() {
    let d = Temp::new();
    let c = context(10000, 65536, CancellationToken::default());
    let mut out = RejectSink { aborted: false };
    let e = scope(&d)
        .run(&request(&d, "hang"), &c, &mut out)
        .unwrap_err();
    assert!(e.started);
    assert!(out.aborted);
    assert!(e.cleanup_errors.is_empty(), "{e:?}");
    assert_reaped(e.pid.unwrap());
}
#[test]
fn configured_invocation_limit_prevents_execution() {
    let d = Temp::new();
    let s = ShellScope::new(vec![executable()], vec![d.0.clone()], 1).unwrap();
    let c = context(10000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let e = s
        .run(
            &request(&d, &format!("marker:{}", d.0.join("marker").display())),
            &c,
            &mut out,
        )
        .unwrap_err();
    assert!(!e.started);
    assert!(!d.0.join("marker").exists());
}
