
// Included in plugin_hooks.rs so fixture/config helpers are shared.
fn observed_pid(dir: &Path, id: &str) -> i32 {
    std::fs::read_to_string(dir.join(format!("events.{id}.pid"))).unwrap().parse().unwrap()
}
fn assert_reaped(pid: i32) {
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    }
    // SAFETY: signal 0 only observes process existence.
    assert_eq!(unsafe { kill(pid, 0) }, -1, "owned child {pid} remains alive");
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(3), "expected ESRCH");
    let mut status = 0;
    // SAFETY: status is valid, waitpid observes this exact former child.
    assert_eq!(unsafe { waitpid(pid, &mut status, 1) }, -1);
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(10), "expected ECHILD: already reaped");
}
#[test]
fn w5_failures_reap_real_children_and_stop_later_hooks() {
    for mode in ["hang", "crash", "malformed", "flood", "stderr_flood", "partial",
        "duplicate_annotation", "escaped_duplicate", "namespace", "long_reason",
        "long_annotation", "wrong_id", "mutation"] {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::default();
        let mut host = PluginHost::start(vec![config(dir.path(), "a", mode), config(dir.path(), "b", "allow")],
            limits(), deadline(), &cancel).unwrap();
        let pid = observed_pid(dir.path(), "a");
        let call = invocation();
        assert_eq!(host.before(&call, deadline(), &cancel).decision, Decision::Veto, "{mode}");
        assert_reaped(pid);
        let log = std::fs::read_to_string(dir.path().join("events")).unwrap();
        assert!(!log.contains("b:before"), "later hook ran after {mode}");
        let before_count = log.matches("a:before").count();
        assert_eq!(host.before(&call, deadline(), &cancel).decision, Decision::Veto);
        assert_eq!(std::fs::read_to_string(dir.path().join("events")).unwrap().matches("a:before").count(), before_count);
        let b = observed_pid(dir.path(), "b");
        drop(host);
        assert_reaped(b);
    }
}
#[test]
fn w5_registration_is_atomic_and_drop_reaps() {
    let cancel = CancellationToken::default();
    for mode in ["version", "identity", "plugin_version", "unsupported_hook", "duplicate_hook", "handshake_hang"] {
        let dir = tempfile::tempdir().unwrap();
        assert!(PluginHost::start(vec![config(dir.path(), "a", "allow"), config(dir.path(), "b", mode)],
            limits(), deadline(), &cancel).is_err());
        assert_reaped(observed_pid(dir.path(), "a"));
        assert_reaped(observed_pid(dir.path(), "b"));
    }
    let dir = tempfile::tempdir().unwrap();
    let host = PluginHost::start(vec![config(dir.path(), "a", "allow")], limits(), deadline(), &cancel).unwrap();
    let pid = observed_pid(dir.path(), "a");
    drop(host);
    assert_reaped(pid);
}
#[test]
fn w5_cancellation_isolated_and_after_failure_is_warning_only() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let independent = CancellationToken::default();
    let mut first = PluginHost::start(vec![config(dir.path(), "a", "hang")], limits(), deadline(), &cancel).unwrap();
    let mut second = PluginHost::start(vec![config(dir.path(), "b", "allow")], limits(), deadline(), &independent).unwrap();
    let pid = observed_pid(dir.path(), "a");
    let trigger = cancel.clone();
    let worker = std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(20)); trigger.cancel(); });
    assert_eq!(first.before(&invocation(), deadline(), &cancel).decision, Decision::Veto);
    worker.join().unwrap();
    assert_reaped(pid);
    assert_eq!(second.before(&invocation(), deadline(), &independent).decision, Decision::Allow);
    let mut host = PluginHost::start(vec![config(dir.path(), "c", "after_failure"), config(dir.path(), "d", "allow")],
        limits(), deadline(), &independent).unwrap();
    let call = invocation();
    assert_eq!(host.before(&call, deadline(), &independent).decision, Decision::Allow);
    let result = ToolResult::Inline(InlineToolResult { call_id: call.call_id.clone(), is_error: true, content: json!({"actual":"failure"}) });
    let original = serde_json::to_value(&result).unwrap();
    let receipts = host.after(&call, &result, 1, deadline(), &independent);
    assert!(receipts[0].diagnostic.is_some());
    assert!(receipts[1].diagnostic.is_none());
    assert_eq!(serde_json::to_value(&result).unwrap(), original);
    assert_reaped(observed_pid(dir.path(), "c"));
}
#[test]
fn w5_capability_subsets_fragmentation_and_real_zero_effect_marker() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut host = PluginHost::start(vec![config(dir.path(), "a", "before_only"), config(dir.path(), "b", "after_only"),
        config(dir.path(), "c", "fragmented")], limits(), deadline(), &cancel).unwrap();
    let call = invocation();
    assert_eq!(host.before(&call, deadline(), &cancel).decision, Decision::Allow);
    let result = ToolResult::Inline(InlineToolResult { call_id: call.call_id.clone(), is_error: false, content: json!("actual") });
    assert_eq!(host.after(&call, &result, 1, deadline(), &cancel).len(), 2);
    let log = std::fs::read_to_string(dir.path().join("events")).unwrap();
    assert!(!log.contains("a:after") && !log.contains("b:before"));
    let mut veto = PluginHost::start(vec![config(dir.path(), "v", "veto")], limits(), deadline(), &cancel).unwrap();
    let marker = dir.path().join("executed");
    let outcome = veto.before(&call, deadline(), &cancel);
    if outcome.decision == Decision::Allow { std::fs::write(&marker, b"effect").unwrap(); }
    assert_eq!(outcome.decision, Decision::Veto);
    assert!(!marker.exists());
}
#[cfg(target_os = "macos")]
#[test]
fn w5_actual_sandboxed_host_denies_undeclared_io_network_and_fork() {
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
    c.args.extend([good.canonicalize().unwrap().to_str().unwrap().into(),
        bad.canonicalize().unwrap().to_str().unwrap().into(), listener.local_addr().unwrap().to_string()]);
    c.sandbox = Sandbox::MacOs {
        read_files: vec![c.program.clone(), "/dev/null".into(), good.canonicalize().unwrap()],
        read_trees: vec!["/usr/lib".into(), "/System".into()],
        write_trees: vec![c.cwd.clone()],
    };
    let cancel = CancellationToken::default();
    let mut l = limits(); l.timeout = Duration::from_secs(3);
    let mut host = PluginHost::start(vec![c], l, deadline(), &cancel).unwrap();
    let outcome = host.before(&invocation(), deadline(), &cancel);
    assert_eq!(outcome.decision, Decision::Allow, "{:?}; diagnostics={:?}", outcome.receipts, host.diagnostics());
    assert_eq!(outcome.receipts[0].annotations.get("s.probes"), Some(&json!(true)));
    assert_eq!(std::fs::read(&bad).unwrap(), b"secret");
    assert_eq!(listener.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    let pid = observed_pid(&dir.path().canonicalize().unwrap(), "s");
    drop(host);
    assert_reaped(pid);
}

#[test]
fn w5_inherited_descriptor_is_closed_on_exec() {
    use std::os::fd::AsRawFd;
    unsafe extern "C" { fn fcntl(fd: i32, cmd: i32, ...) -> i32; }
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
    assert_eq!(result.receipts[0].annotations.get("fd.fd_closed"), Some(&json!(true)));
}

#[test]
fn w5_invalid_limits_and_duplicate_registration_spawn_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut too_large = limits(); too_large.message_bytes = 8 * 1024 * 1024 + 1;
    assert!(PluginHost::start(vec![config(dir.path(), "a", "allow")], too_large, deadline(), &cancel).is_err());
    assert!(PluginHost::start(vec![config(dir.path(), "a", "allow"), config(dir.path(), "a", "allow")],
        limits(), deadline(), &cancel).is_err());
    assert!(!dir.path().join("events").exists());
}

#[test]
fn w5_blocked_stdin_deadline_reaps_owned_process() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut l = limits(); l.message_bytes = 1024 * 1024;
    let mut host = PluginHost::start(vec![config(dir.path(), "blocked", "not_reading_after_initialize")], l, deadline(), &cancel).unwrap();
    let pid = observed_pid(dir.path(), "blocked");
    let mut call = invocation();
    call.arguments.insert("large".into(), json!("x".repeat(256 * 1024)));
    let start = Instant::now();
    let result = host.before(&call, deadline(), &cancel);
    assert_eq!(result.decision, Decision::Veto);
    assert!(start.elapsed() < Duration::from_secs(3));
    assert_reaped(pid);
}
