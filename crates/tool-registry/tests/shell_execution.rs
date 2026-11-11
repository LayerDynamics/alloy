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
        "exit127" => std::process::exit(127),
        "exit23" => std::process::exit(23),
        "signal" => {
            unsafe extern "C" {
                fn raise(signal: i32) -> i32;
            }
            // SAFETY: isolated owned fixture sends SIGTERM to itself.
            unsafe { raise(15) };
            panic!("SIGTERM did not terminate fixture");
        }
        "descendant" => {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "shell_child_fixture",
                    "--nocapture",
                    "--skip",
                    "hang",
                ])
                .spawn()
                .unwrap();
            fs::write("descendant.pid", child.id().to_string()).unwrap();
            std::process::exit(0);
        }
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

// Run this probe in its own test process: SIGCHLD disposition is process-wide.
#[test]
fn shell_survives_external_auto_reaping() {
    const PROBE: &str = "ALLOY_SHELL_REAPER_PROBE";
    if std::env::var_os(PROBE).is_none() {
        let output = std::process::Command::new(executable())
            .args([
                "--exact",
                "shell_survives_external_auto_reaping",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PROBE, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated probe failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    unsafe extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
    }
    #[cfg(target_os = "macos")]
    let sigchld = 20;
    #[cfg(target_os = "linux")]
    let sigchld = 17;
    // SAFETY: this is a short-lived isolated subprocess; SIG_IGN is the
    // platform's conventional value 1 and affects only this process.
    assert_ne!(unsafe { signal(sigchld, 1) }, usize::MAX);
    let d = Temp::new();
    let (result, bytes) = execute(&d, "echo:REAPER_SAFE");
    assert!(result.success);
    assert!(String::from_utf8_lossy(&stream(&bytes, "stdout")).contains("REAPER_SAFE"));
}

#[test]
fn signal_exit_preserves_actual_status() {
    let d = Temp::new();
    let (r, _) = execute(&d, "signal");
    assert!(!r.success);
    assert_eq!(r.exit_code, None);
    assert_eq!(r.signal, Some(15));
}
#[test]
fn exec_failure_is_distinct_from_program_exit_127() {
    let d = Temp::new();
    let (report, _) = execute(&d, "exit127");
    assert_eq!(report.exit_code, Some(127));
    assert!(!report.success);
    let path = d.0.join("not-executable");
    fs::write(&path, "not an executable").unwrap();
    let s = ShellScope::new(vec![path.clone()], vec![d.0.clone()], 65536).unwrap();
    let r = ShellRequest::new(path.to_str().unwrap(), vec![], d.0.to_str().unwrap()).unwrap();
    let c = context(10000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&c, 65536).unwrap();
    let failure = s.run(&r, &c, &mut out).unwrap_err();
    assert!(!failure.started, "{failure:?}");
    assert!(matches!(failure.cause, ShellError::Io(_)), "{failure:?}");
}
#[test]
fn exited_leader_descendant_is_stopped_before_completion() {
    let d = Temp::new();
    assert!(execute(&d, "descendant").0.success);
    let pid = fs::read_to_string(d.0.join("descendant.pid")).unwrap();
    let result = std::process::Command::new("/bin/ps")
        .args(["-p", pid.trim(), "-o", "stat="])
        .output()
        .unwrap();
    assert!(result.stderr.is_empty());
    assert!(
        !result.status.success()
            || String::from_utf8_lossy(&result.stdout)
                .trim()
                .starts_with('Z'),
        "live descendant: {result:?}"
    );
}

// An actual host waitpid(-1) loop must collect unrelated statuses without
// being able to reap a shell target or its supervisor on macOS.
#[test]
fn shell_survives_competing_waiter_without_stealing_statuses() {
    const PROBE: &str = "ALLOY_ACTIVE_REAPER_PROBE";
    if std::env::var_os(PROBE).is_none() {
        let result = std::process::Command::new(executable())
            .args([
                "--exact",
                "shell_survives_competing_waiter_without_stealing_statuses",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PROBE, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    let stop = Arc::new(AtomicBool::new(false));
    let observed = Arc::new(Mutex::new(Vec::new()));
    let waiter = {
        let stop = stop.clone();
        let observed = observed.clone();
        std::thread::spawn(move || {
            unsafe extern "C" {
                fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
            }
            while !stop.load(Ordering::Acquire) {
                let mut status = 0;
                // SAFETY: isolated probe intentionally owns all host child reaping.
                let pid = unsafe { waitpid(-1, &mut status, 1) };
                if pid > 0 {
                    observed.lock().unwrap().push((pid as u32, status));
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        })
    };
    let d = Temp::new();
    let mut sentinels = vec![];
    for _ in 0..12 {
        let child = std::process::Command::new(executable())
            .args([
                "--exact",
                "shell_child_fixture",
                "--nocapture",
                "--skip",
                "exit23",
            ])
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        sentinels.push(child.id());
        drop(child);
        assert_eq!(execute(&d, "exit7").0.exit_code, Some(7));
    }
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !sentinels
        .iter()
        .all(|pid| observed.lock().unwrap().iter().any(|(seen, _)| seen == pid))
        && std::time::Instant::now() < until
    {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    stop.store(true, Ordering::Release);
    waiter.join().unwrap();
    let observed = observed.lock().unwrap();
    for pid in sentinels {
        assert!(
            observed.contains(&(pid, 23 << 8)),
            "unrelated child status missing/corrupted: {pid}: {observed:?}"
        );
    }
}
