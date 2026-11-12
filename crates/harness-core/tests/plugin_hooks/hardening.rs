use super::*;

#[cfg(target_os = "macos")]
fn install_exec_helper(dir: &Path) {
    let link = dir.join("alloy-plugin-exec-helper");
    if !link.exists() {
        std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_alloy-plugin-exec-helper"), link).unwrap();
    }
}

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

#[cfg(target_os = "macos")]
#[test]
fn fileport_helper_failure_and_cancellation_reap_exact_child() {
    for mode in ["crash", "hang", "cancel"] {
        let dir = tempfile::tempdir().unwrap();
        install_exec_helper(dir.path());
        let marker = dir.path().join("helper.pid");
        let source = dir.path().join("helper.rs");
        let binary = dir.path().join("helper");
        let path = format!("{:?}", marker.to_str().unwrap());
        let ending = if mode == "crash" {
            "std::process::exit(73)"
        } else {
            "loop { std::thread::sleep(std::time::Duration::from_secs(1)); }"
        };
        std::fs::write(&source, format!("fn main() {{ if std::env::args().nth(1).as_deref() == Some(\"warm\") {{ return; }} std::fs::write({path}, std::process::id().to_string()).unwrap(); {ending} }}")).unwrap();
        assert!(
            Command::new("rustc")
                .args(["--edition=2024", "-o"])
                .arg(&binary)
                .arg(&source)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(&binary)
                .arg("warm")
                .status()
                .unwrap()
                .success()
        );
        let plugin = config(dir.path(), "a", "allow");
        let cancel = CancellationToken::default();
        let mut l = limits();
        l.timeout = Duration::from_secs(2);
        let trigger = if mode == "cancel" {
            let signal = cancel.clone();
            let marker = marker.clone();
            Some(std::thread::spawn(move || {
                let until = Instant::now() + Duration::from_secs(3);
                while !marker.exists() && Instant::now() < until {
                    std::thread::sleep(Duration::from_millis(5));
                }
                signal.cancel();
            }))
        } else {
            None
        };
        let until = Instant::now() + Duration::from_secs(2);
        let started = Instant::now();
        let result = PluginHost::start_with_helper(vec![plugin], l, until, &cancel, &binary);
        assert!(result.is_err(), "{mode} unexpectedly started a plugin");
        if let Some(trigger) = trigger {
            trigger.join().unwrap();
        }
        assert!(started.elapsed() < Duration::from_secs(4), "{mode}");
        let pid: i32 = std::fs::read_to_string(&marker)
            .unwrap_or_else(|e| panic!("{mode}: helper marker missing: {e}"))
            .parse()
            .unwrap();
        assert_reaped(pid);
        assert!(
            !dir.path().join("events").exists(),
            "plugin ran after helper {mode}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn fileport_rejects_malformed_helper_message() {
    let dir = tempfile::tempdir().unwrap();
    install_exec_helper(dir.path());
    let helper = dir.path().join("malformed-helper");
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fileport_malformed_helper.c");
    assert!(
        Command::new("cc")
            .arg(source)
            .arg("-o")
            .arg(&helper)
            .status()
            .unwrap()
            .success()
    );
    let plugin = config(dir.path(), "malformed", "allow");
    let cancel = CancellationToken::default();
    let error = PluginHost::start_with_helper(vec![plugin], limits(), deadline(), &cancel, &helper)
        .err()
        .expect("malformed helper reply was accepted");
    assert!(error.to_string().contains("Input/output error"), "{error}");
    assert!(!dir.path().join("events").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn fileport_rejects_wrong_sender_and_complex_messages() {
    if std::env::var_os("ALLOY_FILEPORT_RIGHTS_TEST").is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "hardening::fileport_rejects_wrong_sender_and_complex_messages",
                "--nocapture",
            ])
            .env("ALLOY_FILEPORT_RIGHTS_TEST", "1")
            .env("ALLOY_W5_FD_FIXTURE", fixture())
            .env("ALLOY_W5_FD_HELPER", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("FILEPORT_RIGHTS_PASSED"));
        return;
    }
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_port_names(
            task: u32,
            names: *mut *mut u32,
            nc: *mut u32,
            types: *mut *mut u32,
            tc: *mut u32,
        ) -> i32;
        fn mach_port_get_refs(task: u32, name: u32, right: u32, refs: *mut u32) -> i32;
        fn mach_port_allocate(task: u32, right: u32, name: *mut u32) -> i32;
        fn mach_port_mod_refs(task: u32, name: u32, right: u32, delta: i32) -> i32;
        fn vm_deallocate(task: u32, address: usize, size: usize) -> i32;
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        fn kill(pid: i32, signal: i32) -> i32;
    }
    type Inventory = (Vec<(u32, u32, [u32; 5])>, Vec<i32>);
    fn inventory() -> Inventory {
        let mut names = std::ptr::null_mut();
        let mut types = std::ptr::null_mut();
        let (mut nc, mut tc) = (0, 0);
        // SAFETY: mach_task_self_ is the current task's read-only port name.
        let task = unsafe { mach_task_self_ };
        // SAFETY: Mach returns allocated arrays and initializes both lengths.
        let status = unsafe { mach_port_names(task, &mut names, &mut nc, &mut types, &mut tc) };
        assert_eq!(status, 0);
        assert_eq!(nc, tc);
        // SAFETY: the kernel returned nc entries in each array on success.
        let name_slice = unsafe { std::slice::from_raw_parts(names, nc as usize) };
        // SAFETY: the types array also contains tc initialized entries.
        let type_slice = unsafe { std::slice::from_raw_parts(types, tc as usize) };
        let mut rights = Vec::new();
        for (&name, &kind) in name_slice.iter().zip(type_slice) {
            let mut refs = [0; 5];
            for (right, count) in refs.iter_mut().enumerate() {
                if kind & (1 << (right + 16)) != 0 {
                    // SAFETY: name and right were returned by mach_port_names.
                    let status = unsafe { mach_port_get_refs(task, name, right as u32, count) };
                    assert_eq!(status, 0);
                }
            }
            rights.push((name, kind, refs));
        }
        // SAFETY: release the two out-of-line arrays allocated by mach_port_names.
        let status = unsafe { vm_deallocate(task, names as usize, nc as usize * 4) };
        assert_eq!(status, 0);
        // SAFETY: same ownership applies to the separate types array.
        let status = unsafe { vm_deallocate(task, types as usize, tc as usize * 4) };
        assert_eq!(status, 0);
        rights.sort_unstable();
        let mut fds: Vec<i32> = std::fs::read_dir("/dev/fd")
            .unwrap()
            .map(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .parse()
                    .unwrap()
            })
            .collect();
        // The directory stream itself appears in the listing; after dropping it,
        // retain only still-open descriptors. F_GETFD only observes state.
        // SAFETY: F_GETFD accepts any integer descriptor.
        fds.retain(|fd| unsafe { fcntl(*fd, 1) } >= 0);
        fds.sort_unstable();
        (rights, fds)
    }
    let dir = tempfile::tempdir().unwrap();
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fileport_adversarial_helper.c");
    install_exec_helper(dir.path());
    for mode in 1..=3 {
        let helper = dir.path().join(format!("adversarial-helper-{mode}"));
        assert!(
            Command::new("cc")
                .arg(format!("-DTEST_MODE={mode}"))
                .arg(&source)
                .arg("-o")
                .arg(&helper)
                .status()
                .unwrap()
                .success()
        );
        let plugin = config(dir.path(), "adversarial", "allow");
        let cancel = CancellationToken::default();
        // Include the first attempt in the inventory check, not only steady state.
        let baseline = inventory();
        let mut probe = 0;
        // SAFETY: mach_task_self_ is the current task's read-only port name.
        let task = unsafe { mach_task_self_ };
        // SAFETY: allocate an actual receive right to validate the inventory.
        assert_eq!(unsafe { mach_port_allocate(task, 1, &mut probe) }, 0);
        assert_ne!(inventory().0, baseline.0, "inventory missed an added right");
        // SAFETY: probe is the receive right allocated immediately above.
        assert_eq!(unsafe { mach_port_mod_refs(task, probe, 1, -1) }, 0);
        assert_eq!(inventory(), baseline);
        for _ in 0..8 {
            for role in ["helper", "sender"] {
                let path = format!("{}.{}.pid", helper.display(), role);
                match std::fs::remove_file(path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => panic!("failed to remove prior fixture marker: {e}"),
                }
            }
            let error = PluginHost::start_with_helper(
                vec![plugin.clone()],
                limits(),
                deadline(),
                &cancel,
                &helper,
            )
            .err()
            .expect("adversarial helper reply was accepted");
            assert!(error.to_string().contains("Input/output error"), "{error}");
            let marker = |role: &str| {
                let path = format!("{}.{}.pid", helper.display(), role);
                std::fs::read_to_string(path)
                    .unwrap()
                    .trim()
                    .parse::<i32>()
                    .unwrap()
            };
            assert_reaped(marker("helper"));
            if mode == 1 {
                let sender = marker("sender");
                let until = Instant::now() + Duration::from_secs(2);
                loop {
                    // SAFETY: signal zero observes this fixture PID without signaling it.
                    if unsafe { kill(sender, 0) } == -1
                        && std::io::Error::last_os_error().raw_os_error() == Some(3)
                    {
                        break;
                    }
                    assert!(
                        Instant::now() < until,
                        "forged sender {sender} remains alive"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            assert_eq!(
                inventory(),
                baseline,
                "rights or descriptors leaked in mode {mode}"
            );
        }
        assert!(!dir.path().join("events").exists());
    }
    println!("FILEPORT_RIGHTS_PASSED");
}

#[cfg(target_os = "macos")]
#[test]
fn plugin_rejects_writable_hardlinks_and_helper_overlap_before_spawn() {
    let program_dir = tempfile::tempdir().unwrap();
    let writable = tempfile::tempdir().unwrap();
    let program = program_dir.path().join("plugin");
    std::fs::copy(fixture(), &program).unwrap();
    let alias = writable.path().join("alias");
    std::fs::hard_link(&program, &alias).unwrap();
    let mut config = config(writable.path(), "guard", "allow");
    config.program = program;
    config.sandbox = Sandbox::MacOs {
        read_files: vec![config.program.clone()],
        read_trees: vec!["/usr/lib".into(), "/System".into()],
        write_trees: vec![writable.path().into()],
    };
    let cancel = CancellationToken::default();
    let error = start_host(vec![config.clone()], limits(), deadline(), &cancel)
        .err()
        .unwrap();
    assert!(error.to_string().contains("hard-linked"), "{error}");
    assert!(!writable.path().join("events").exists());
    std::fs::remove_file(alias).unwrap();
    let helper = Path::new(env!("CARGO_BIN_EXE_alloy-plugin-pipe-helper"))
        .canonicalize()
        .unwrap();
    config.sandbox = Sandbox::MacOs {
        read_files: vec![config.program.clone()],
        read_trees: vec![],
        write_trees: vec![helper.parent().unwrap().into()],
    };
    let error = start_host(vec![config], limits(), deadline(), &cancel)
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("protected executable"),
        "{error}"
    );
    assert!(!writable.path().join("events").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn fileport_concurrent_exec_scanner() {
    if std::env::var_os("ALLOY_FILEPORT_SCANNER").is_none() {
        return;
    }
    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    for fd in 3..1024 {
        // SAFETY: F_GETFD only observes the descriptor.
        let status = unsafe { fcntl(fd, 1) };
        if status >= 0 {
            let mut path = [0u8; 1024];
            // SAFETY: Darwin F_GETPATH writes at most MAXPATHLEN bytes to this buffer.
            let path_result = unsafe { fcntl(fd, 50, path.as_mut_ptr()) };
            let end = path.iter().position(|&b| b == 0).unwrap_or(path.len());
            eprintln!(
                "inherited fd={fd} flags={status} path_result={path_result} path={}",
                String::from_utf8_lossy(&path[..end])
            );
        }
        assert!(
            status == -1 && std::io::Error::last_os_error().raw_os_error() == Some(9),
            "unexpected inherited descriptor {fd}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn fileport_concurrent_unrelated_exec_inherits_no_endpoints() {
    if std::env::var_os("ALLOY_FILEPORT_STRESS").is_none() {
        use std::os::unix::process::CommandExt;
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "hardening::fileport_concurrent_unrelated_exec_inherits_no_endpoints",
                "--nocapture",
            ])
            .env("ALLOY_FILEPORT_STRESS", "1")
            .env("ALLOY_W5_FD_FIXTURE", fixture())
            .env("ALLOY_W5_FD_HELPER", "1");
        unsafe extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        // SAFETY: this isolated child marks inherited FDs close-on-exec using
        // only async-signal-safe fcntl calls; stdio and Rust's exec pipe remain usable.
        unsafe {
            command.pre_exec(|| {
                for fd in 3..1024 {
                    let flags = fcntl(fd, 1);
                    if flags >= 0 && fcntl(fd, 2, flags | 1) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("FILEPORT_CONCURRENT_EXEC_PASSED")
        );
        return;
    }
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    let scanner = || {
        let result = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "hardening::fileport_concurrent_exec_scanner"])
            .env("ALLOY_FILEPORT_SCANNER", "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .unwrap();
        if !result.status.success() {
            eprintln!(
                "concurrent scanner: stdout={} stderr={}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
        result.status.success()
    };
    assert!(scanner(), "scanner baseline was not clean");
    let file = std::fs::File::open("/dev/null").unwrap();
    // SAFETY: F_DUPFD returns a new descriptor with CLOEXEC clear.
    let raw = unsafe { fcntl(file.as_raw_fd(), 0, 128) };
    assert!(raw >= 128);
    // SAFETY: raw is newly owned by this test.
    let inherited = unsafe { OwnedFd::from_raw_fd(raw) };
    assert!(
        !scanner(),
        "positive control failed to detect an inheritable descriptor"
    );
    drop(inherited);
    let running = std::sync::Arc::new(AtomicBool::new(true));
    let count = std::sync::Arc::new(AtomicUsize::new(0));
    let failures = std::sync::Arc::new(AtomicUsize::new(0));
    let thread = {
        let running = running.clone();
        let count = count.clone();
        let failures = failures.clone();
        std::thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                if !scanner() {
                    failures.fetch_add(1, Ordering::SeqCst);
                }
                count.fetch_add(1, Ordering::SeqCst);
            }
        })
    };
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancellationToken::default();
    let mut l = limits();
    l.timeout = Duration::from_secs(3);
    let work = (|| -> Result<(), String> {
        for _ in 0..25 {
            let mut host = start_host(
                vec![config(dir.path(), "stress", "allow")],
                l,
                deadline(),
                &cancel,
            )
            .map_err(|e| e.to_string())?;
            if host.before(&invocation(), deadline(), &cancel).decision != Decision::Allow {
                return Err("before hook failed".into());
            }
        }
        Ok(())
    })();
    running.store(false, Ordering::SeqCst);
    thread.join().unwrap();
    assert!(work.is_ok(), "{work:?}");
    assert!(count.load(Ordering::SeqCst) > 0);
    assert_eq!(
        failures.load(Ordering::SeqCst),
        0,
        "concurrent child inherited a descriptor"
    );
    println!(
        "FILEPORT_CONCURRENT_EXEC_PASSED scans={}",
        count.load(Ordering::SeqCst)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn fileport_works_when_standard_descriptors_are_closed() {
    if std::env::var_os("ALLOY_FILEPORT_CLOSED_STDIO").is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "hardening::fileport_works_when_standard_descriptors_are_closed",
                "--nocapture",
            ])
            .env("ALLOY_FILEPORT_CLOSED_STDIO", "1")
            .env("ALLOY_W5_FD_FIXTURE", fixture())
            .env("ALLOY_W5_FD_HELPER", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("FILEPORT_CLOSED_STDIO_PASSED"));
        return;
    }
    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        fn close(fd: i32) -> i32;
        fn dup2(fd: i32, target: i32) -> i32;
    }
    let dir = tempfile::tempdir().unwrap();
    let plugin = config(dir.path(), "closed", "allow");
    let cancel = CancellationToken::default();
    let mut saved = [-1; 3];
    for (fd, slot) in saved.iter_mut().enumerate() {
        // SAFETY: F_DUPFD_CLOEXEC duplicates each live stdio descriptor.
        *slot = unsafe { fcntl(fd as i32, 67, 16) };
        assert!(*slot >= 16);
    }
    for fd in 0..3 {
        // SAFETY: isolated child owns its descriptors.
        assert_eq!(unsafe { close(fd) }, 0);
    }
    let result = start_host(vec![plugin], limits(), deadline(), &cancel);
    for (fd, saved_fd) in saved.iter().enumerate() {
        // SAFETY: restore the isolated child's original standard descriptors.
        assert_eq!(unsafe { dup2(*saved_fd, fd as i32) }, fd as i32);
        // SAFETY: saved_fd is the owned duplicate and is no longer needed.
        unsafe {
            close(*saved_fd);
        }
    }
    let mut host = result.unwrap();
    assert_eq!(
        host.before(&invocation(), deadline(), &cancel).decision,
        Decision::Allow
    );
    let pid = observed_pid(dir.path(), "closed");
    drop(host);
    assert_reaped(pid);
    println!("FILEPORT_CLOSED_STDIO_PASSED");
}

#[cfg(target_os = "macos")]
#[test]
fn fileport_descriptor_exhaustion_fails_closed_without_fd_leaks() {
    if std::env::var_os("ALLOY_FILEPORT_EXHAUSTION").is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "hardening::fileport_descriptor_exhaustion_fails_closed_without_fd_leaks",
                "--nocapture",
            ])
            .env("ALLOY_FILEPORT_EXHAUSTION", "1")
            .env("ALLOY_W5_FD_FIXTURE", fixture())
            .env("ALLOY_W5_FD_HELPER", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("FILEPORT_EXHAUSTION_PASSED"));
        return;
    }
    #[repr(C)]
    struct Limit {
        current: u64,
        maximum: u64,
    }
    unsafe extern "C" {
        fn getrlimit(resource: i32, limit: *mut Limit) -> i32;
        fn setrlimit(resource: i32, limit: *const Limit) -> i32;
    }
    let dir = tempfile::tempdir().unwrap();
    let plugin = config(dir.path(), "exhausted", "allow");
    let mut original = Limit {
        current: 0,
        maximum: 0,
    };
    // SAFETY: the isolated process owns its resource limits.
    assert_eq!(unsafe { getrlimit(8, &mut original) }, 0);
    let reduced = Limit {
        current: 64,
        maximum: original.maximum,
    };
    assert!(original.maximum >= reduced.current);
    // SAFETY: only this isolated process changes its soft limit.
    assert_eq!(unsafe { setrlimit(8, &reduced) }, 0);
    let mut held = Vec::new();
    loop {
        match std::fs::File::open("/dev/null") {
            Ok(file) => held.push(file),
            Err(e) => {
                assert_eq!(e.raw_os_error(), Some(24));
                break;
            }
        }
    }
    assert!(held.len() >= 8);
    for _ in 0..4 {
        drop(held.pop());
    }
    let cancel = CancellationToken::default();
    let mut l = limits();
    l.timeout = Duration::from_secs(3);
    let result = start_host(vec![plugin], l, deadline(), &cancel);
    let error = result
        .err()
        .expect("helper unexpectedly obtained six endpoints");
    assert!(error.to_string().contains("Too many open files"), "{error}");
    let mut restored = Vec::new();
    loop {
        match std::fs::File::open("/dev/null") {
            Ok(file) => restored.push(file),
            Err(e) => {
                assert_eq!(e.raw_os_error(), Some(24));
                break;
            }
        }
    }
    assert_eq!(restored.len(), 4, "endpoint import leaked descriptors");
    drop(restored);
    drop(held);
    // SAFETY: restore the isolated process's original soft limit.
    assert_eq!(unsafe { setrlimit(8, &original) }, 0);
    assert!(!dir.path().join("events").exists());
    println!("FILEPORT_EXHAUSTION_PASSED");
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
        let mut host = start_host(
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
            start_host(
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
    let host = start_host(
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
    let mut first = start_host(
        vec![config(dir.path(), "a", "hang")],
        limits(),
        deadline(),
        &cancel,
    )
    .unwrap();
    let mut second = start_host(
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
    let mut host = start_host(
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
    let mut host = start_host(
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
    let mut veto = start_host(
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
    unsafe extern "C" {
        fn getpgrp() -> i32;
    }
    // SAFETY: getpgrp only reads this test process's group identity.
    let parent_group = unsafe { getpgrp() };
    c.cwd = c.cwd.canonicalize().unwrap();
    c.program = c.program.canonicalize().unwrap();
    c.args[2] = c.cwd.join("events").to_str().unwrap().into();
    c.args.extend([
        good.canonicalize().unwrap().to_str().unwrap().into(),
        bad.canonicalize().unwrap().to_str().unwrap().into(),
        listener.local_addr().unwrap().to_string(),
        c.program.to_str().unwrap().into(),
        c.cwd.join("forbidden-link").to_str().unwrap().into(),
        parent_group.to_string(),
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
    let mut host = start_host(vec![c], l, deadline(), &cancel).unwrap();
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
    assert!(!dir.path().join("forbidden-link").exists());
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
    let mut host = start_host(vec![c], limits(), deadline(), &cancel).unwrap();
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
        start_host(
            vec![config(dir.path(), "a", "allow")],
            too_large,
            deadline(),
            &cancel
        )
        .is_err()
    );
    assert!(
        start_host(
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
    let mut host = start_host(
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
    let mut host = start_host(vec![config], limits(), deadline(), &cancel).unwrap();
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

#[cfg(target_os = "macos")]
#[test]
fn plugin_external_reaper_owns_supervisor_statuses() {
    const MODE: &str = "ALLOY_W5_EXTERNAL_REAPER_HELPER";
    if std::env::var_os(MODE).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "hardening::plugin_external_reaper_owns_supervisor_statuses",
                "--nocapture",
            ])
            .env(MODE, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    // Compile before the process-global reaper starts so rustc retains its own
    // exit status. This test runs alone in a separate test-harness process.
    let dir = tempfile::tempdir().unwrap();
    let plugin = config(dir.path(), "external", "allow");
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop = done.clone();
    let reaper = std::thread::spawn(move || {
        unsafe extern "C" {
            fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        }
        let mut statuses = 0;
        let mut stopping_at = None;
        loop {
            // SAFETY: this isolated test owns the process-wide reaper; status
            // is deliberately ignored for supervisor children only.
            let pid = unsafe { waitpid(-1, std::ptr::null_mut(), 1) };
            if pid > 0 {
                statuses += 1;
                continue;
            }
            if stop.load(std::sync::atomic::Ordering::Acquire) {
                // The verifier deliberately remains a live host child. Drain
                // the pipe supervisor's status without waiting for ECHILD.
                let until =
                    stopping_at.get_or_insert_with(|| Instant::now() + Duration::from_secs(2));
                if statuses >= 1 || Instant::now() >= *until {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        statuses
    });
    let host = PluginHost::start_with_external_reaper(
        vec![plugin],
        limits(),
        deadline(),
        &CancellationToken::default(),
        Some(Path::new(env!("CARGO_BIN_EXE_alloy-plugin-pipe-helper"))),
    )
    .unwrap();
    let pid = observed_pid(dir.path(), "external");
    drop(host);
    done.store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(
        reaper.join().unwrap(),
        1,
        "external reaper owns the pipe supervisor; launchd owns the workload supervisor"
    );
    unsafe extern "C" {
        fn alloy_test_verifier_pid() -> i32;
        fn proc_listchildpids(parent: i32, buffer: *mut i32, size: i32) -> i32;
    }
    let mut children = [0i32; 8];
    // SAFETY: libproc fills this writable bounded buffer; this call never signals or reaps.
    assert_eq!(
        // SAFETY: the initialized PID array remains writable for the supplied size.
        unsafe {
            proc_listchildpids(
                std::process::id() as i32,
                children.as_mut_ptr(),
                std::mem::size_of_val(&children) as i32,
            )
        },
        1,
        "unexpected additional host children: {children:?}"
    );
    // SAFETY: reads the test host's retained verifier PID, not a signalling capability.
    assert_eq!(children[0], unsafe { alloy_test_verifier_pid() });
    assert_reaped(pid);
}
