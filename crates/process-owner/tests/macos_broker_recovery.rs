#![cfg(all(target_os = "macos", debug_assertions))]

#[path = "macos_broker_recovery/containment.rs"]
mod containment;

use alloy_process_owner::spawn::{Child, default_helper, spawn_shell};
use std::{
    fs,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};

unsafe extern "C" {
    fn alloy_test_broker_fault(stage: i32);
    fn alloy_test_broker_kill() -> i32;
    fn alloy_test_broker_snapshot(pid: *mut i32, identity: *mut u64, generation: *mut u64) -> i32;
    fn alloy_test_broker_malformed() -> i32;
    fn kill(pid: i32, signal: i32) -> i32;
    fn alloy_test_wait_owner_exit();
    fn alloy_test_pid_only_slot() -> i32;
    fn alloy_test_finish_stalled_owner() -> i32;
    fn alloy_test_broker_stale_generation();
    fn alloy_test_peer_authentication(expected: *const std::ffi::c_char) -> i32;
    fn proc_listchildpids(parent: i32, buffer: *mut i32, size: i32) -> i32;
    fn __error() -> *mut i32;
    fn alloy_test_shell_diagnostic() -> *const std::ffi::c_char;
    fn alloy_test_identity_unknown(pid: i32);
    fn alloy_test_shell_owner_pause(target: i32, paused: i32) -> i32;
    fn alloy_test_shell_cancellation_pending(target: i32) -> i32;
    fn alloy_test_verifier_pid() -> i32;
    fn alloy_test_verifier_kill() -> i32;
    fn alloy_test_verifier_reject_self() -> i32;
    fn alloy_test_verifier_late_allow() -> i32;
    fn alloy_test_verifier_unknown_parent() -> i32;
    fn alloy_test_verifier_alive() -> i32;
    fn alloy_test_verifier_pause() -> i32;
    fn alloy_test_ignore_block_hup();
    fn alloy_process_identity(pid: i32) -> u64;
    fn alloy_owned_signal(pid: i32, identity: u64) -> i32;
}

fn start(program: &str, arguments: &[&str], root: &Path) -> std::io::Result<Child> {
    let helper = default_helper()?;
    let mut command = Command::new(program);
    command.args(arguments).current_dir(root).env_clear();
    let (child, input, output, error) = spawn_shell(
        command,
        Some(&helper),
        Instant::now() + Duration::from_secs(60),
        &|| false,
    )
    .map_err(|error| {
        if error.raw_os_error() != Some(5) {
            return error;
        }
        // SAFETY: the test-only function returns a live thread-local C string.
        let stage =
            unsafe { std::ffi::CStr::from_ptr(alloy_test_shell_diagnostic()) }.to_string_lossy();
        std::io::Error::new(error.kind(), format!("{error}; {stage}"))
    })?;
    drop((input, output, error));
    Ok(child)
}

fn gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        // SAFETY: signal zero only observes OS existence and delivers no signal.
        if unsafe { kill(pid as i32, 0) } != 0 {
            assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(3));
            return;
        }
        assert!(Instant::now() < deadline, "owned PID {pid} still exists");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn stop(mut child: Child) {
    let pid = child.id();
    child.kill().unwrap();
    child.wait().unwrap();
    drop(child);
    gone(pid);
}

fn snapshot() -> [u64; 2] {
    let (mut pid, mut identity, mut generation) = (0, 0, [0; 2]);
    // SAFETY: these distinct writable values remain valid for the synchronous snapshot.
    assert_eq!(
        // SAFETY: these distinct output buffers remain writable through the call.
        unsafe { alloy_test_broker_snapshot(&mut pid, &mut identity, generation.as_mut_ptr()) },
        0
    );
    assert!(pid > 0 && identity > 0);
    generation
}

fn kill_broker() {
    let (mut pid, mut identity, mut generation) = (0, 0, [0; 2]);
    // SAFETY: snapshot borrows three distinct valid output buffers.
    assert_eq!(
        // SAFETY: these distinct output buffers remain writable through the call.
        unsafe { alloy_test_broker_snapshot(&mut pid, &mut identity, generation.as_mut_ptr()) },
        0
    );
    // SAFETY: the hook signals only this host's exact authenticated broker.
    assert_eq!(unsafe { alloy_test_broker_kill() }, 0);
    gone(pid as u32);
}

fn launchd_job_state() -> (String, usize) {
    let pid = std::process::id().to_string();
    unsafe extern "C" {
        fn getuid() -> u32;
    }
    // SAFETY: getuid observes the current user and takes no pointers.
    let root = format!("/private/tmp/alloy-brokers-{}", unsafe { getuid() });
    let record = fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .find(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.split('.').nth(5) == Some(pid.as_str()))
        })
        .expect("current host has no registration record");
    // ServiceManagement selects the current user's launchd context. Inspect
    // both documented user/login domain spellings, without assuming a GUI login.
    // SAFETY: getuid takes no pointers and only observes the current user.
    let uid = unsafe { getuid() };
    let output = ["user", "gui"]
        .into_iter()
        .find_map(|domain| {
            let target = format!("{domain}/{uid}/{}", record.file_name().to_str().unwrap());
            let output = Command::new("/bin/launchctl")
                .args(["print", &target])
                .output()
                .unwrap();
            output.status.success().then_some(output)
        })
        .expect("launchd inspection found neither user nor login registration");
    let text = String::from_utf8(output.stdout).unwrap();
    let field = |key: &str| {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(key))
            .unwrap_or_else(|| panic!("missing {key} in launchd state: {text}"))
    };
    assert!(
        !text.lines().any(|line| line.trim().starts_with("pid =")),
        "idle crashed broker restarted: {text}"
    );
    (
        field("state = ").to_owned(),
        field("runs = ").parse().unwrap(),
    )
}

fn isolated(name: &str, run: impl FnOnce(&Path)) {
    if std::env::var("ALLOY_BROKER_RECOVERY_CASE").as_deref() == Ok(name) {
        let root = std::env::temp_dir().join(format!(
            "alloy-broker-recovery-{name}-{}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        run(&root);
        fs::remove_dir_all(root).unwrap();
        return;
    }
    // Each failure scenario gets a real host lifetime, its own registration and
    // its own restart budget. The test driver never injects faults into another host.
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("ALLOY_BROKER_RECOVERY_CASE", name)
        .output()
        .unwrap();
    assert!(
        output.status.success()
            && String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
        "isolated {name} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn crash_case(root: &Path, stage: i32) {
    let mut independent = start("/bin/sleep", &["60"], root).unwrap();
    let before = snapshot();
    let marker = root.join("first");
    if stage <= 2 {
        // SAFETY: test-only hook, this process owns the authenticated broker.
        unsafe {
            alloy_test_broker_fault(stage);
        }
        let begun = Instant::now();
        let result = start("/bin/sh", &["-c", "printf x >> first"], root);
        assert!(
            result.is_err(),
            "injected broker death must interrupt this launch"
        );
        assert!(
            begun.elapsed() < Duration::from_secs(3),
            "lost broker reply held the global launch lock until the 60-second operation deadline"
        );
        assert!(!marker.exists(), "uncommitted operation executed");
    } else {
        let mut child = start(
            "/bin/sh",
            &["-c", "printf x >> first; exec /bin/sleep 60"],
            root,
        )
        .unwrap();
        let until = Instant::now() + Duration::from_secs(3);
        while !marker.exists() {
            assert!(Instant::now() < until);
            std::thread::sleep(Duration::from_millis(5));
        }
        struct PausedOwner(i32);
        impl Drop for PausedOwner {
            fn drop(&mut self) {
                // SAFETY: resume only the exact test-owned supervisor. This
                // guard runs before Child drop even if an assertion unwinds.
                assert_eq!(unsafe { alloy_test_shell_owner_pause(self.0, 0) }, 0);
            }
        }
        let paused = if stage == 4 {
            // SAFETY: stop the real supervisor by kernel identity and wait
            // until the OS reports SSTOP, before publishing cancellation.
            assert_eq!(
                // SAFETY: this host retains the exact supervisor for this child.
                unsafe { alloy_test_shell_owner_pause(child.id() as i32, 1) },
                0
            );
            let guard = PausedOwner(child.id() as i32);
            child.kill().unwrap();
            // SAFETY: observes both shared STOP and the real supervisor's
            // stopped OS state. Cancellation cannot finish behind this barrier.
            assert_eq!(
                // SAFETY: only observes this live child's retained control state.
                unsafe { alloy_test_shell_cancellation_pending(child.id() as i32) },
                1
            );
            Some(guard)
        } else {
            None
        };
        // SAFETY: the hook uses the broker's retained stable identity, not a raw kill PID.
        kill_broker();
        if stage == 4 {
            // SAFETY: cancellation must still be pending after broker death.
            assert_eq!(
                // SAFETY: Child still owns this independent supervisor channel.
                unsafe { alloy_test_shell_cancellation_pending(child.id() as i32) },
                1
            );
            // SAFETY: the workload still exists; cleanup has not already won.
            assert_eq!(unsafe { kill(child.id() as i32, 0) }, 0);
        }
        drop(paused);
        if stage == 3 {
            assert!(
                child.try_wait().unwrap().is_none(),
                "broker loss destroyed the supervisor control channel"
            );
        }
        stop(child);
        assert_eq!(
            fs::read(&marker).unwrap(),
            b"x",
            "command executed more than once"
        );
    }
    assert!(
        independent.try_wait().unwrap().is_none(),
        "unrelated operation was stopped by broker loss"
    );
    let mut next = start("/bin/sh", &["-c", "printf x >> next"], root).unwrap();
    assert!(next.wait().unwrap().success());
    let pid = next.id();
    drop(next);
    gone(pid);
    assert_eq!(fs::read(root.join("next")).unwrap(), b"x");
    assert_ne!(
        snapshot(),
        before,
        "recovery reused the old instance generation"
    );
    assert!(independent.try_wait().unwrap().is_none());
    if stage <= 2 {
        assert!(
            !marker.exists(),
            "interrupted command was replayed after recovery"
        );
    } else {
        assert_eq!(
            fs::read(marker).unwrap(),
            b"x",
            "started command was replayed after recovery"
        );
    }
    stop(independent);
}

#[test]
fn broker_loss_before_launch_recovers_without_replay() {
    isolated(
        "broker_loss_before_launch_recovers_without_replay",
        |root| crash_case(root, 1),
    );
}
#[test]
fn broker_loss_during_launch_releases_other_launches() {
    isolated(
        "broker_loss_during_launch_releases_other_launches",
        |root| crash_case(root, 2),
    );
}
#[test]
fn broker_loss_during_execution_preserves_control() {
    isolated("broker_loss_during_execution_preserves_control", |root| {
        crash_case(root, 3)
    });
}
#[test]
fn broker_loss_during_cancellation_preserves_cleanup() {
    isolated(
        "broker_loss_during_cancellation_preserves_cleanup",
        |root| crash_case(root, 4),
    );
}

#[test]
fn malformed_message_does_not_retire_authenticated_service() {
    isolated(
        "malformed_message_does_not_retire_authenticated_service",
        |root| {
            let mut first = start("/usr/bin/true", &[], root).unwrap();
            assert!(first.wait().unwrap().success());
            drop(first);
            let before = snapshot();
            // SAFETY: sends one malformed message only to this test host's service.
            assert_eq!(unsafe { alloy_test_broker_malformed() }, 0);
            let mut next = start("/usr/bin/true", &[], root).unwrap();
            assert!(next.wait().unwrap().success());
            drop(next);
            assert_eq!(snapshot(), before, "malformed input restarted the service");
        },
    );
}

#[test]
fn live_supervisor_capacity_survives_broker_restart() {
    isolated("live_supervisor_capacity_survives_broker_restart", |root| {
        let mut children = Vec::new();
        for _ in 0..64 {
            children.push(start("/bin/sleep", &["60"], root).unwrap());
        }
        // SAFETY: kill only this host's authenticated broker with stable identity.
        kill_broker();
        let error = start("/usr/bin/true", &[], root)
            .err()
            .expect("restart reset the outstanding ownership limit");
        assert_eq!(
            error.raw_os_error(),
            Some(16),
            "expected host ownership capacity refusal"
        );
        for child in children {
            stop(child);
        }
        let mut next = start("/usr/bin/true", &[], root).unwrap();
        assert!(next.wait().unwrap().success());
    });
}

#[test]
fn completion_published_before_supervisor_exit_keeps_real_status() {
    isolated(
        "completion_published_before_supervisor_exit_keeps_real_status",
        |root| {
            let mut child = start("/bin/sh", &["-c", "/bin/sleep 0.1; exit 7"], root).unwrap();
            // SAFETY: force the first state-read / owner-exit ordering only in this
            // isolated test host, without changing the supervisor or target status.
            unsafe {
                alloy_test_wait_owner_exit();
            }
            assert_eq!(child.wait().unwrap().code(), Some(7));
        },
    );
}

#[test]
fn repeated_confirmed_presend_cancellation_does_not_exhaust_capacity() {
    let name = "repeated_confirmed_presend_cancellation_does_not_exhaust_capacity";
    isolated(name, |root| {
        let mut first = start("/usr/bin/true", &[], root).unwrap();
        assert!(first.wait().unwrap().success());
        drop(first);
        for _ in 0..80 {
            // SAFETY: cancel at the initial send callback, after slot allocation
            // but before any Mach message can reach this host's real broker.
            unsafe { alloy_test_broker_fault(3) };
            let error = start("/bin/sh", &["-c", "printf x >> cancelled"], root)
                .err()
                .expect("pre-send cancellation executed");
            assert_eq!(error.raw_os_error(), Some(89));
        }
        assert!(!root.join("cancelled").exists());
        let mut next = start("/usr/bin/true", &[], root).unwrap();
        assert!(next.wait().unwrap().success());
    });
}

#[test]
fn unconfirmed_stopped_supervisor_stays_charged_until_proven_exit() {
    isolated(
        "unconfirmed_stopped_supervisor_stays_charged_until_proven_exit",
        |root| {
            struct Stalled;
            impl Drop for Stalled {
                fn drop(&mut self) {
                    // SAFETY: reap only the fixture created by this isolated test.
                    assert_eq!(unsafe { alloy_test_finish_stalled_owner() }, 0);
                }
            }
            // SAFETY: create a real stopped test child before identity publication.
            let pid = unsafe { alloy_test_pid_only_slot() };
            assert!(pid > 0);
            let stalled = Stalled;
            std::thread::sleep(Duration::from_millis(5100));
            let mut children = Vec::new();
            for _ in 0..63 {
                children.push(start("/bin/sleep", &["60"], root).unwrap());
            }
            kill_broker();
            let error = start("/usr/bin/true", &[], root)
                .err()
                .expect("stopped owner lost its charge");
            assert_eq!(error.raw_os_error(), Some(16));
            // SAFETY: signal zero independently confirms the stopped owner exists.
            assert_eq!(unsafe { kill(pid, 0) }, 0);
            drop(stalled);
            gone(pid as u32);
            let mut next = start("/usr/bin/true", &[], root).unwrap();
            assert!(next.wait().unwrap().success());
            for child in children {
                stop(child);
            }
        },
    );
}

#[test]
fn stale_generation_never_launches_or_replays_a_command() {
    isolated(
        "stale_generation_never_launches_or_replays_a_command",
        |root| {
            let mut independent = start("/bin/sleep", &["60"], root).unwrap();
            let before = snapshot();
            // SAFETY: corrupt only the test host's cached generation, representing a
            // request delayed across a service instance change.
            unsafe {
                alloy_test_broker_stale_generation();
            }
            assert!(start("/bin/sh", &["-c", "printf x >> stale"], root).is_err());
            assert!(!root.join("stale").exists());
            let mut next = start("/bin/sh", &["-c", "printf x >> next"], root).unwrap();
            assert!(next.wait().unwrap().success());
            drop(next);
            assert_eq!(snapshot(), before);
            assert_eq!(fs::read(root.join("next")).unwrap(), b"x");
            assert!(!root.join("stale").exists(), "stale command was replayed");
            assert!(independent.try_wait().unwrap().is_none());
            stop(independent);
        },
    );
}

#[test]
fn peer_audit_authentication_rejects_unrelated_code() {
    use std::os::unix::ffi::OsStrExt;
    let executable =
        std::ffi::CString::new(std::env::current_exe().unwrap().as_os_str().as_bytes()).unwrap();
    // SAFETY: the C routine receives a live NUL-terminated expected code path,
    // and obtains the current process audit token directly from the kernel.
    assert_eq!(
        // SAFETY: executable owns the live NUL-terminated path for this call.
        unsafe { alloy_test_peer_authentication(executable.as_ptr()) },
        0
    );
    // SAFETY: this valid literal deliberately names code different from this process.
    assert_eq!(
        // SAFETY: the literal has static storage and is NUL-terminated.
        unsafe { alloy_test_peer_authentication(c"/usr/bin/true".as_ptr()) },
        1
    );
}

#[test]
fn inconclusive_host_identity_preserves_other_live_service() {
    let name = "inconclusive_host_identity_preserves_other_live_service";
    isolated(name, |root| {
        if let Some(shared) = std::env::var_os("ALLOY_TEST_HOLDER_ROOT") {
            let shared = std::path::PathBuf::from(shared);
            let child = start("/bin/sleep", &["60"], root).unwrap();
            fs::write(shared.join("ready"), child.id().to_string()).unwrap();
            let until = Instant::now() + Duration::from_secs(15);
            while !shared.join("release").exists() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(5));
            }
            stop(child);
            return;
        }
        let mut holder = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env("ALLOY_BROKER_RECOVERY_CASE", name)
            .env("ALLOY_TEST_HOLDER_ROOT", root)
            .spawn()
            .unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        while !root.join("ready").exists() {
            assert!(Instant::now() < until, "independent host did not start");
            assert!(holder.try_wait().unwrap().is_none());
            std::thread::sleep(Duration::from_millis(5));
        }
        let target: i32 = fs::read_to_string(root.join("ready"))
            .unwrap()
            .parse()
            .unwrap();
        // SAFETY: only this test host's identity observation for the independent
        // holder is made inconclusive. The holder and its supervisor stay real.
        unsafe { alloy_test_identity_unknown(holder.id() as i32) };
        let result = start("/usr/bin/true", &[], root);
        // SAFETY: clear the test-only observation fault before cleanup.
        unsafe { alloy_test_identity_unknown(0) };
        let mut next = result.expect("inconclusive old owner blocked an unrelated registration");
        assert!(next.wait().unwrap().success());
        assert!(holder.try_wait().unwrap().is_none());
        // SAFETY: signal zero only observes that the unrelated workload survives.
        assert_eq!(unsafe { kill(target, 0) }, 0);
        fs::write(root.join("release"), b"").unwrap();
        assert!(holder.wait().unwrap().success());
        gone(target as u32);
    });
}

#[test]
fn concurrent_host_retirement_does_not_break_registration() {
    isolated(
        "concurrent_host_retirement_does_not_break_registration",
        |root| {
            let mut child = start("/usr/bin/true", &[], root).unwrap();
            assert!(child.wait().unwrap().success());
        },
    );
    if std::env::var("ALLOY_BROKER_RECOVERY_CASE").is_ok() {
        return;
    }
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                isolated(
                    "concurrent_host_retirement_does_not_break_registration",
                    |root| {
                        let mut child = start("/usr/bin/true", &[], root).unwrap();
                        assert!(child.wait().unwrap().success());
                    },
                )
            });
        }
    });
    isolated(
        "concurrent_host_retirement_does_not_break_registration",
        |root| {
            let mut child = start("/usr/bin/true", &[], root).unwrap();
            assert!(child.wait().unwrap().success());
        },
    );
}

#[test]
fn restart_budget_refuses_excess_reconnects_then_recovers() {
    isolated(
        "restart_budget_refuses_excess_reconnects_then_recovers",
        |root| {
            let mut first = start("/usr/bin/true", &[], root).unwrap();
            assert!(first.wait().unwrap().success());
            drop(first);
            let began = Instant::now();
            for _ in 0..2 {
                // SAFETY: affect only this host's authenticated service instance.
                kill_broker();
                let mut next = start("/usr/bin/true", &[], root).unwrap();
                assert!(next.wait().unwrap().success());
                drop(next);
            }
            // SAFETY: the third crash exhausts this host's three-attempt window.
            kill_broker();
            let error = start("/bin/sh", &["-c", "printf x >> refused"], root)
                .err()
                .expect("reconnect budget was not enforced");
            assert_eq!(error.raw_os_error(), Some(35));
            assert!(!root.join("refused").exists());
            let idle = launchd_job_state();
            assert_eq!(idle.0, "not running");
            // Observe launchd's real run count throughout the cooldown, so a
            // host-only throttle cannot hide an automatic service restart loop.
            while began.elapsed() < Duration::from_millis(10_100) {
                std::thread::sleep(Duration::from_millis(200));
                assert_eq!(launchd_job_state(), idle);
            }
            // The cooldown is finite: failure never poisons the host permanently.
            if let Some(remaining) = Duration::from_millis(10_100).checked_sub(began.elapsed()) {
                std::thread::sleep(remaining);
            }
            let mut next = start("/usr/bin/true", &[], root).unwrap();
            assert!(next.wait().unwrap().success());
            assert!(
                !root.join("refused").exists(),
                "budget-refused operation was replayed"
            );
        },
    );
}

#[test]
fn registration_and_workloads_create_no_additional_host_children() {
    isolated(
        "registration_and_workloads_create_no_additional_host_children",
        |root| {
            for _ in 0..5 {
                let mut child = start("/usr/bin/true", &[], root).unwrap();
                assert!(child.wait().unwrap().success());
                drop(child);
                let mut pids = [0i32; 128];
                // SAFETY: errno is this thread's live TLS cell, and the OS fills a
                // valid bounded PID buffer. This does not signal or reap a process.
                let (count, error) = unsafe {
                    *__error() = 0;
                    let count = proc_listchildpids(
                        std::process::id() as i32,
                        pids.as_mut_ptr(),
                        std::mem::size_of_val(&pids) as i32,
                    );
                    (count, *__error())
                };
                assert_eq!(error, 0, "OS child enumeration failed");
                assert_eq!(
                    count, 1,
                    "registration or ownership added host children beyond its verifier: {pids:?}"
                );
                // SAFETY: reads this isolated host's retained verifier identity.
                assert_eq!(pids[0], unsafe { alloy_test_verifier_pid() });
            }
        },
    );
}

#[test]
fn private_verifier_rejects_unrelated_code_and_death_fails_closed() {
    isolated(
        "private_verifier_rejects_unrelated_code_and_death_fails_closed",
        |root| {
            let independent = start("/bin/sleep", &["30"], root).unwrap();
            // SAFETY: sends this test executable's real kernel audit token through the private channel.
            assert_eq!(unsafe { alloy_test_verifier_reject_self() }, 1);
            // SAFETY: delays a real allow, cancels it, and submits this unrelated identity next.
            assert_eq!(unsafe { alloy_test_verifier_late_allow() }, 0);
            stop(start("/bin/sleep", &["30"], root).unwrap());
            // SAFETY: kills only the retained verifier's exact kernel process identity.
            assert_eq!(unsafe { alloy_test_verifier_kill() }, 0);
            // A fresh broker requires fresh verification; no old success may be reused.
            kill_broker();
            for _ in 0..3 {
                assert!(start("/bin/sh", &["-c", "echo started >> unverified"], root).is_err());
            }
            assert!(!root.join("unverified").exists());
            stop(independent);
        },
    );
}

#[test]
fn verifier_exits_on_inconclusive_parent_identity() {
    isolated("verifier_exits_on_inconclusive_parent_identity", |root| {
        stop(start("/bin/sleep", &["30"], root).unwrap());
        // SAFETY: injects an inconclusive parent identity observation in this host's verifier.
        assert_eq!(unsafe { alloy_test_verifier_unknown_parent() }, 0);
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            // SAFETY: observes the retained stable identity, including exited zombie state.
            if unsafe { alloy_test_verifier_alive() } == 0 {
                break;
            }
            assert!(
                Instant::now() < until,
                "verifier retained an unverified parent"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        kill_broker();
        assert!(start("/bin/sh", &["-c", "echo started >> unverified"], root).is_err());
        assert!(!root.join("unverified").exists());
    });
}

#[test]
fn stopped_verifier_exits_with_its_host_even_if_host_ignores_hup() {
    const MODE: &str = "ALLOY_STOPPED_VERIFIER_PARENT";
    if let Some(root) = std::env::var_os(MODE) {
        let root = Path::new(&root);
        // SAFETY: this dedicated probe process changes only its own signal state.
        unsafe {
            alloy_test_ignore_block_hup();
        }
        stop(start("/bin/sleep", &["30"], root).unwrap());
        // SAFETY: these reads capture this host's exact verifier identity.
        let pid = unsafe { alloy_test_verifier_pid() };
        // SAFETY: reads the live verifier's stable kernel identity.
        let identity = unsafe { alloy_process_identity(pid) };
        assert!(pid > 0 && identity > 0);
        fs::write(root.join("verifier.pid"), format!("{pid} {identity}")).unwrap();
        // SAFETY: stops only the exact verifier and waits for OS-confirmed SSTOP.
        assert_eq!(unsafe { alloy_test_verifier_pause() }, 0);
        return; // The entire probe host exits while its watchdog is stopped.
    }
    let root = std::env::temp_dir().join(format!("alloy-stopped-verifier-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "stopped_verifier_exits_with_its_host_even_if_host_ignores_hup",
            "--nocapture",
        ])
        .env(MODE, &root)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{} {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let record = fs::read_to_string(root.join("verifier.pid")).unwrap();
    let values: Vec<_> = record.split_whitespace().collect();
    let (pid, identity) = (
        values[0].parse::<i32>().unwrap(),
        values[1].parse::<u64>().unwrap(),
    );
    struct Cleanup(i32, u64);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            // SAFETY: fallback signals only the recorded kernel identity, after the assertion.
            unsafe {
                alloy_owned_signal(self.0, self.1);
            }
        }
    }
    let cleanup = Cleanup(pid, identity);
    gone(pid as u32); // Independent OS absence must precede fallback cleanup.
    drop(cleanup);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_framework_command_retains_bootstrap_access() {
    isolated(
        "native_framework_command_retains_bootstrap_access",
        |root| {
            fs::write(
                root.join("fixture.plist"),
                "<?xml version=\"1.0\"?><plist version=\"1.0\"><dict/></plist>",
            )
            .unwrap();
            let mut child =
                start("/usr/bin/plutil", &["-lint", "-s", "fixture.plist"], root).unwrap();
            assert!(child.wait().unwrap().success());
        },
    );
}
