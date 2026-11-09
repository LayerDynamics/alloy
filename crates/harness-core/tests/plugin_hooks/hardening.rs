use super::*;

// Plugin-host boundary tests share the parent test target's fixture/config helpers.
fn observed_pid(dir: &Path, id: &str) -> i32 {
    std::fs::read_to_string(dir.join(format!("events.{id}.pid")))
        .unwrap()
        .parse()
        .unwrap()
}
fn assert_reaped(pid: i32) {
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    }
    assert_eq!(
        // SAFETY: signal 0 only observes process existence; it sends no signal.
        unsafe { kill(pid, 0) },
        -1,
        "owned child {pid} remains alive"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(3),
        "expected ESRCH"
    );
    let mut status = 0;
    // SAFETY: status is valid, waitpid observes this exact former child.
    assert_eq!(unsafe { waitpid(pid, &mut status, 1) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(10),
        "expected ECHILD: already reaped"
    );
}
#[test]
fn plugin_failures_reap_real_children_and_stop_later_hooks() {
    for mode in [
        "hang",
        "crash",
        "malformed",
        "flood",
        "stderr_flood",
        "partial",
        "duplicate_annotation",
        "escaped_duplicate",
        "namespace",
        "long_reason",
        "long_annotation",
        "wrong_id",
        "mutation",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::default();
        let mut host = PluginHost::start(
            vec![
                config(dir.path(), "a", mode),
                config(dir.path(), "b", "allow"),
            ],
            limits(),
            deadline(),
            &cancel,
        )
        .unwrap();
        let pid = observed_pid(dir.path(), "a");
        let call = invocation();
        assert_eq!(
            host.before(&call, deadline(), &cancel).decision,
            Decision::Veto,
            "{mode}"
        );
        assert_reaped(pid);
        let log = std::fs::read_to_string(dir.path().join("events")).unwrap();
        assert!(!log.contains("b:before"), "later hook ran after {mode}");
        let before_count = log.matches("a:before").count();
        assert_eq!(
            host.before(&call, deadline(), &cancel).decision,
            Decision::Veto
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("events"))
                .unwrap()
                .matches("a:before")
                .count(),
            before_count
        );
        let b = observed_pid(dir.path(), "b");
        drop(host);
        assert_reaped(b);
    }
}
#[test]
fn plugin_registration_is_atomic_and_drop_reaps() {
    let cancel = CancellationToken::default();
    for mode in [
        "version",
        "identity",
        "plugin_version",
        "unsupported_hook",
        "duplicate_hook",
        "handshake_hang",
    ] {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            PluginHost::start(
                vec![
                    config(dir.path(), "a", "allow"),
                    config(dir.path(), "b", mode)
                ],
                limits(),
                deadline(),
                &cancel
            )
            .is_err()
        );
        assert_reaped(observed_pid(dir.path(), "a"));
        assert_reaped(observed_pid(dir.path(), "b"));
    }
    let dir = tempfile::tempdir().unwrap();
    let host = PluginHost::start(
        vec![config(dir.path(), "a", "allow")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let pid = observed_pid(dir.path(), "a");
    drop(host);
    assert_reaped(pid);
}
#[test]
fn plugin_cancellation_isolated_and_after_failure_is_warning_only() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let independent = CancellationToken::default();
    let mut first = PluginHost::start(
        vec![config(dir.path(), "a", "hang")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let mut second = PluginHost::start(
        vec![config(dir.path(), "b", "allow")],
        limits(),
        deadline(),
        &independent,
    )
    .unwrap();
    let pid = observed_pid(dir.path(), "a");
    let trigger = cancel.clone();
    let worker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        trigger.cancel();
    });
    assert_eq!(
        first.before(&invocation(), deadline(), &cancel).decision,
        Decision::Veto
    );
    worker.join().unwrap();
    assert_reaped(pid);
    assert_eq!(
        second
            .before(&invocation(), deadline(), &independent)
            .decision,
        Decision::Allow
    );
    let mut host = PluginHost::start(
        vec![
            config(dir.path(), "c", "after_failure"),
            config(dir.path(), "d", "allow"),
        ],
        limits(),
        deadline(),
        &independent,
    )
    .unwrap();
    let call = invocation();
    assert_eq!(
        host.before(&call, deadline(), &independent).decision,
        Decision::Allow
    );
    let result = ToolResult::Inline(InlineToolResult {
        call_id: call.call_id.clone(),
        is_error: true,
        content: json!({"actual":"failure"}),
    });
    let original = serde_json::to_value(&result).unwrap();
    let receipts = host.after(&call, &result, 1, deadline(), &independent);
    assert!(receipts[0].diagnostic.is_some());
    assert!(receipts[1].diagnostic.is_none());
    assert_eq!(serde_json::to_value(&result).unwrap(), original);
    assert_reaped(observed_pid(dir.path(), "c"));
}
#[test]
fn plugin_capability_subsets_fragmentation_and_real_zero_effect_marker() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut host = PluginHost::start(
        vec![
            config(dir.path(), "a", "before_only"),
            config(dir.path(), "b", "after_only"),
            config(dir.path(), "c", "fragmented"),
        ],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let call = invocation();
    assert_eq!(
        host.before(&call, deadline(), &cancel).decision,
        Decision::Allow
    );
    let result = ToolResult::Inline(InlineToolResult {
        call_id: call.call_id.clone(),
        is_error: false,
        content: json!("actual"),
    });
    assert_eq!(host.after(&call, &result, 1, deadline(), &cancel).len(), 2);
    let log = std::fs::read_to_string(dir.path().join("events")).unwrap();
    assert!(!log.contains("a:after") && !log.contains("b:before"));
    let mut veto = PluginHost::start(
        vec![config(dir.path(), "v", "veto")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let marker = dir.path().join("executed");
    let outcome = veto.before(&call, deadline(), &cancel);
    if outcome.decision == Decision::Allow {
        std::fs::write(&marker, b"effect").unwrap();
    }
    assert_eq!(outcome.decision, Decision::Veto);
    assert!(!marker.exists());
}
#[cfg(target_os = "macos")]
#[test]
fn plugin_actual_sandboxed_host_denies_undeclared_io_network_and_fork() {
    let dir = tempfile::tempdir().unwrap();
    let denied = tempfile::tempdir().unwrap();
    let good = dir.path().join("good");
    let bad = denied.path().join("bad");
    std::fs::write(&good, b"allowed").unwrap();
    std::fs::write(&bad, b"secret").unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut c = config(dir.path(), "s", "probe");
    c.cwd = c.cwd.canonicalize().unwrap();
    c.program = c.program.canonicalize().unwrap();
    c.args[2] = c.cwd.join("events").to_str().unwrap().into();
    c.args.extend([
        good.canonicalize().unwrap().to_str().unwrap().into(),
        bad.canonicalize().unwrap().to_str().unwrap().into(),
        listener.local_addr().unwrap().to_string(),
        c.program.to_str().unwrap().into(),
    ]);
    c.sandbox = Sandbox::MacOs {
        read_files: vec![
            c.program.clone(),
            "/dev/null".into(),
            good.canonicalize().unwrap(),
        ],
        read_trees: vec!["/usr/lib".into(), "/System".into()],
        write_trees: vec![c.cwd.clone()],
    };
    let cancel = CancellationToken::default();
    let mut l = limits();
    l.timeout = Duration::from_secs(3);
    let mut host = PluginHost::start(vec![c], l, deadline(), &cancel).unwrap();
    let outcome = host.before(&invocation(), deadline(), &cancel);
    assert_eq!(
        outcome.decision,
        Decision::Allow,
        "{:?}; diagnostics={:?}",
        outcome.receipts,
        host.diagnostics()
    );
    assert_eq!(
        outcome.receipts[0].annotations.get("s.probes"),
        Some(&json!(true))
    );
    assert_eq!(std::fs::read(&bad).unwrap(), b"secret");
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    let pid = observed_pid(&dir.path().canonicalize().unwrap(), "s");
    drop(host);
    assert_reaped(pid);
}

#[test]
fn plugin_inherited_descriptor_is_closed_on_exec() {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    let file = std::fs::File::open("/dev/null").unwrap();
    let fd = file.as_raw_fd();
    // SAFETY: F_SETFD modifies only this test-owned live file descriptor.
    assert_eq!(unsafe { fcntl(fd, 2, 0) }, 0);
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(dir.path(), "fd", "fd_probe");
    c.args.push(fd.to_string());
    let cancel = CancellationToken::default();
    let mut host = PluginHost::start(vec![c], limits(), deadline(), &cancel).unwrap();
    let result = host.before(&invocation(), deadline(), &cancel);
    assert_eq!(result.decision, Decision::Allow, "{:?}", result.receipts);
    assert_eq!(
        result.receipts[0].annotations.get("fd.fd_closed"),
        Some(&json!(true))
    );
}

#[test]
fn plugin_invalid_limits_and_duplicate_registration_spawn_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut too_large = limits();
    too_large.message_bytes = 8 * 1024 * 1024 + 1;
    assert!(
        PluginHost::start(
            vec![config(dir.path(), "a", "allow")],
            too_large,
            deadline(),
            &cancel
        )
        .is_err()
    );
    assert!(
        PluginHost::start(
            vec![
                config(dir.path(), "a", "allow"),
                config(dir.path(), "a", "allow")
            ],
            limits(),
            deadline(),
            &cancel
        )
        .is_err()
    );
    assert!(!dir.path().join("events").exists());
}

#[test]
fn plugin_blocked_stdin_deadline_reaps_owned_process() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut l = limits();
    l.message_bytes = 1024 * 1024;
    let mut host = PluginHost::start(
        vec![config(
            dir.path(),
            "blocked",
            "not_reading_after_initialize",
        )],
        l,
        deadline(),
        &cancel,
    )
    .unwrap();
    let pid = observed_pid(dir.path(), "blocked");
    let mut call = invocation();
    call.arguments
        .insert("large".into(), json!("x".repeat(256 * 1024)));
    let start = Instant::now();
    let result = host.before(&call, deadline(), &cancel);
    assert_eq!(result.decision, Decision::Veto);
    assert!(start.elapsed() < Duration::from_secs(3));
    assert_reaped(pid);
}

// Re-exec the Rust test harness so resource-limit changes cannot affect parallel tests.
#[test]
fn plugin_high_fd_above_lowered_limit_is_not_inherited() {
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "hardening::plugin_high_fd_isolated_helper",
            "--nocapture",
        ])
        .env("ALLOY_W5_FD_FIXTURE", fixture())
        .env("ALLOY_W5_FD_HELPER", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "helper status={:?}\nstdout={}\nstderr={}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("W5_HIGH_FD_REGRESSION_PASSED"));
}
#[test]
fn plugin_high_fd_isolated_helper() {
    if std::env::var("ALLOY_W5_FD_HELPER").as_deref() != Ok("1") {
        return;
    }
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Limit {
        current: u64,
        maximum: u64,
    }
    unsafe extern "C" {
        fn getrlimit(resource: i32, limit: *mut Limit) -> i32;
        fn setrlimit(resource: i32, limit: *const Limit) -> i32;
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    #[cfg(target_os = "linux")]
    const NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const NOFILE: i32 = 8;
    let _ = fixture();
    let dir = tempfile::tempdir().unwrap();
    let file = std::fs::File::open("/dev/null").unwrap();
    let mut original = Limit {
        current: 0,
        maximum: 0,
    };
    // SAFETY: original is correctly sized writable storage on supported 64-bit targets.
    assert_eq!(unsafe { getrlimit(NOFILE, &mut original) }, 0);
    assert!(
        original.maximum >= 256,
        "regression requires hard NOFILE >=256"
    );
    let raised = Limit {
        current: original.current.max(256),
        maximum: original.maximum,
    };
    // SAFETY: this isolated process changes only its own soft descriptor limit.
    assert_eq!(unsafe { setrlimit(NOFILE, &raised) }, 0);
    // SAFETY: F_DUPFD creates a new owned descriptor >=128 with CLOEXEC unset.
    let high = unsafe { fcntl(file.as_raw_fd(), 0, 128) };
    assert!(high >= 128);
    // SAFETY: high is newly returned by F_DUPFD and has no other Rust owner.
    let high_fd = unsafe { OwnedFd::from_raw_fd(high) };
    // SAFETY: observes flags of this live owned descriptor.
    assert_eq!(unsafe { fcntl(high, 1) } & 1, 0);
    let lowered = Limit {
        current: 64,
        maximum: original.maximum,
    };
    // SAFETY: this affects only the helper, not the main harness or runner.
    assert_eq!(unsafe { setrlimit(NOFILE, &lowered) }, 0);
    let mut config = config(dir.path(), "highfd", "fd_probe");
    config.args.push(high.to_string());
    let cancel = CancellationToken::default();
    let mut host = PluginHost::start(vec![config], limits(), deadline(), &cancel).unwrap();
    let outcome = host.before(&invocation(), deadline(), &cancel);
    assert_eq!(outcome.decision, Decision::Allow, "{:?}", outcome.receipts);
    assert_eq!(
        outcome.receipts[0].annotations.get("highfd.fd_closed"),
        Some(&json!(true))
    );
    let pid = observed_pid(dir.path(), "highfd");
    drop(host);
    assert_reaped(pid);
    drop(high_fd);
    // SAFETY: restore this helper's original limit before final harness reporting.
    assert_eq!(unsafe { setrlimit(NOFILE, &original) }, 0);
    println!("W5_HIGH_FD_REGRESSION_PASSED fd={high} soft_limit=64");
}
