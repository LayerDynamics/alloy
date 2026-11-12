use super::*;
use alloy_process_owner::spawn::spawn;

fn isolated(name: &str, run: impl FnOnce(&Path)) {
    super::isolated(&format!("containment::{name}"), run);
}

unsafe extern "C" {
    fn alloy_test_watcher_start_failure();
    fn alloy_test_cleaner_signal_failure(target: i32) -> i32;
    fn alloy_process_identity(pid: i32) -> u64;
    fn alloy_owned_signal(pid: i32, identity: u64) -> i32;
    fn alloy_test_cohort_revoke_barrier(target: i32, operation: i32) -> i32;
    fn alloy_test_cohort_revoker_pause(target: i32, paused: i32) -> i32;
    fn alloy_test_cohort_gate_pause(target: i32, paused: i32) -> i32;
    fn alloy_test_cohort_startup_stage(setup: i32) -> i32;
    fn alloy_test_cohort_recovery(target: i32, revoked: i32) -> i32;
    fn alloy_test_uncertain_cohort_record(target: i32) -> i32;
    fn alloy_test_cohort_fail_recovery(target: i32) -> i32;
    fn alloy_test_cohort_kill(target: i32, revoker: i32) -> i32;
    fn alloy_test_cohort_record(
        target: i32,
        path: *mut std::ffi::c_char,
        label: *mut std::ffi::c_char,
        capacity: usize,
    ) -> i32;
    fn alloy_broker_registered(label: *const std::ffi::c_char) -> i32;
}

struct PausedHolders(i32);
impl Drop for PausedHolders {
    fn drop(&mut self) {
        // SAFETY: these hooks resume only exact recorded identities in this test's owned slot.
        unsafe {
            let _ = alloy_test_cohort_gate_pause(self.0, 0);
            let _ = alloy_test_cohort_revoker_pause(self.0, 0);
        }
    }
}
fn pause_holders(target: u32) -> PausedHolders {
    let guard = PausedHolders(target as i32);
    // SAFETY: the hook stops only this invocation's exact trusted gate identity.
    assert_eq!(unsafe { alloy_test_cohort_gate_pause(guard.0, 1) }, 0);
    // SAFETY: the hook stops only this invocation's exact independent holder identity.
    assert_eq!(unsafe { alloy_test_cohort_revoker_pause(guard.0, 1) }, 0);
    guard
}

fn fixture(root: &Path) -> std::path::PathBuf {
    let executable = root.join("detached-fixture");
    let output = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/macos_detached.c"))
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    executable
}

fn records(path: &Path) -> Vec<[i32; 3]> {
    let bytes = fs::read(path).unwrap_or_default();
    assert_eq!(bytes.len() % 12, 0, "partial fixture PID record");
    bytes
        .as_chunks::<12>()
        .0
        .iter()
        .map(|row| {
            std::array::from_fn(|i| i32::from_ne_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()))
        })
        .collect()
}

fn wait_records(path: &Path, minimum: usize) {
    let until = Instant::now() + Duration::from_secs(4);
    while records(path).len() < minimum {
        assert!(
            Instant::now() < until,
            "fixture did not create {minimum} descendants"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn begin(root: &Path, mode: &str, trusted: bool) -> (Child, std::path::PathBuf) {
    let executable = fixture(root);
    let record = root.join("descendants.bin");
    let helper = default_helper().unwrap();
    let mut command = Command::new(executable);
    command
        .args([std::ffi::OsStr::new(mode), record.as_os_str()])
        .current_dir(root)
        .env_clear();
    let deadline = Instant::now() + Duration::from_secs(10);
    let (child, input, output, error) = if trusted {
        spawn(command, Some(&helper), deadline, &|| false, false)
    } else {
        spawn_shell(command, Some(&helper), deadline, &|| false)
    }
    .unwrap();
    drop((input, output, error));
    (child, record)
}

fn verify(record: &Path) {
    let all = records(record);
    assert!(!all.is_empty());
    assert!(
        all.iter().any(|p| p[1] == p[2]),
        "no independently detached session was exercised"
    );
    for item in all.into_iter().filter(|p| p[0] > 0) {
        gone(item[0] as u32);
    }
}

#[test]
fn detached_descendants_are_cleaned_after_normal_exit() {
    isolated(
        "detached_descendants_are_cleaned_after_normal_exit",
        |root| {
            let (mut child, record) = begin(root, "detach-exit", false);
            assert_eq!(child.wait().unwrap().code(), Some(23));
            drop(child);
            verify(&record);
        },
    );
}

#[test]
fn detached_descendants_are_cleaned_after_cancellation() {
    isolated(
        "detached_descendants_are_cleaned_after_cancellation",
        |root| {
            let (child, record) = begin(root, "detach-hold", false);
            wait_records(&record, 2);
            stop(child);
            verify(&record);
        },
    );
}

#[test]
fn trusted_plugin_detached_descendants_are_cleaned() {
    isolated("trusted_plugin_detached_descendants_are_cleaned", |root| {
        let (child, record) = begin(root, "detach-hold", true);
        wait_records(&record, 2);
        stop(child);
        verify(&record);
    });
}

#[test]
fn concurrent_fork_and_spawn_cannot_outlive_cancellation() {
    isolated(
        "concurrent_fork_and_spawn_cannot_outlive_cancellation",
        |root| {
            let (mut child, record) = begin(root, "storm", false);
            wait_records(&record, 8);
            // SAFETY: the test-only barrier addresses this invocation's owned shared state.
            assert_eq!(
                // SAFETY: the hook uses only this isolated test's owned identities and initialized buffers.
                unsafe { alloy_test_cohort_revoke_barrier(child.id() as i32, 1) },
                0
            );
            child.kill().unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            loop {
                let denials = records(&record);
                // SAFETY: querying the barrier only reads this invocation's atomic test state.
                let paused = unsafe { alloy_test_cohort_revoke_barrier(child.id() as i32, 2) };
                if paused == 1
                    && (0..3).all(|kind| {
                        denials
                            .iter()
                            .any(|p| p[0] < 0 && p[1] == 1 && p[2] == kind)
                    })
                {
                    break;
                }
                assert!(
                    Instant::now() < until,
                    "fork/spawn did not report all post-release denials: {denials:?}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            // SAFETY: releases the same owned test barrier; production builds omit its wait.
            assert_eq!(
                // SAFETY: the hook uses only this isolated test's owned identities and initialized buffers.
                unsafe { alloy_test_cohort_revoke_barrier(child.id() as i32, 0) },
                1
            );
            child.wait().unwrap();
            let pid = child.id();
            drop(child);
            gone(pid);
            verify(&record);
        },
    );
}

#[test]
fn independent_revoker_cleans_after_gate_death() {
    isolated("independent_revoker_cleans_after_gate_death", |root| {
        let (mut child, record) = begin(root, "storm", false);
        wait_records(&record, 8);
        // SAFETY: the hook validates the exact recorded gate identity in this host's slot.
        assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 0) }, 0);
        assert!(
            child.wait().is_err(),
            "gate death must not fabricate command success"
        );
        drop(child);
        verify(&record);
        stop(start("/bin/sleep", &["30"], root).unwrap());
    });
}

#[test]
fn gate_revokes_after_independent_revoker_death() {
    isolated("gate_revokes_after_independent_revoker_death", |root| {
        let (child, record) = begin(root, "storm", false);
        wait_records(&record, 8);
        // SAFETY: the hook validates the exact recorded revoker identity in this host's slot.
        assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 1) }, 0);
        stop(child);
        verify(&record);
        stop(start("/bin/sleep", &["30"], root).unwrap());
    });
}

#[test]
fn host_finishes_detached_cleanup_after_cleaner_death() {
    isolated(
        "host_finishes_detached_cleanup_after_cleaner_death",
        |root| {
            let (mut child, record) = begin(root, "storm", false);
            wait_records(&record, 8);
            // SAFETY: the hook validates this invocation's exact isolated cleaner identity.
            assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 2) }, 0);
            assert!(
                child.wait().is_err(),
                "cleaner loss must report interruption"
            );
            drop(child);
            verify(&record);
            stop(start("/bin/sleep", &["30"], root).unwrap());
        },
    );
}

#[test]
fn new_host_retires_crashed_cleaner_and_detached_workload() {
    const NAME: &str = "new_host_retires_crashed_cleaner_and_detached_workload";
    if let Some(root) = std::env::var_os("ALLOY_ABANDONED_COHORT_ROOT") {
        let root = std::path::PathBuf::from(root);
        let (child, record) = begin(&root, "storm", false);
        wait_records(&record, 8);
        let (mut path, mut label) = ([0i8; 1024], [0i8; 1024]);
        // SAFETY: both output arrays are writable for capacity bytes; target is this host's child.
        assert_eq!(
            // SAFETY: the hook uses only this isolated test's owned identities and initialized buffers.
            unsafe {
                alloy_test_cohort_record(
                    child.id() as i32,
                    path.as_mut_ptr(),
                    label.as_mut_ptr(),
                    path.len(),
                )
            },
            0
        );
        // SAFETY: success above wrote terminated strings into both arrays.
        let path = unsafe { std::ffi::CStr::from_ptr(path.as_ptr()) }.to_string_lossy();
        // SAFETY: success above wrote a terminated label into this array.
        let label = unsafe { std::ffi::CStr::from_ptr(label.as_ptr()) }.to_string_lossy();
        fs::write(
            root.join("abandoned-registration.txt"),
            format!("{path}\n{label}\n"),
        )
        .unwrap();
        // SAFETY: the hook signals only the exact isolated cleaner in this host's slot.
        assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 2) }, 0);
        // Deliberately bypass Child::drop: the next independent host must recover.
        std::process::exit(0);
    }
    isolated(NAME, |root| {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &format!("containment::{NAME}"), "--nocapture"])
            .env("ALLOY_ABANDONED_COHORT_ROOT", root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let registration = fs::read_to_string(root.join("abandoned-registration.txt")).unwrap();
        let mut fields = registration.lines();
        let path = fields.next().unwrap();
        let label = std::ffi::CString::new(fields.next().unwrap()).unwrap();
        // This is the parent's first Alloy invocation: ordinary registry startup
        // must recover the dead host's record before allocating its own service.
        let next = start("/bin/sleep", &["30"], root).unwrap();
        verify(&root.join("descendants.bin"));
        assert!(
            !Path::new(path).exists(),
            "stale cohort record survived new-host recovery"
        );
        // SAFETY: this label is a live terminated C string read from the owned receipt.
        assert_eq!(unsafe { alloy_broker_registered(label.as_ptr()) }, 0);
        stop(next);
    });
}

#[test]
fn try_wait_does_not_block_on_paused_revocation_after_cleaner_loss() {
    isolated(
        "try_wait_does_not_block_on_paused_revocation_after_cleaner_loss",
        |root| {
            let (mut child, record) = begin(root, "detach-hold", false);
            wait_records(&record, 2);
            let holders = pause_holders(child.id());
            // SAFETY: this kills only the exact isolated cleaner, leaving its holder paused.
            assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 2) }, 0);
            std::thread::sleep(Duration::from_millis(20));
            let before = Instant::now();
            assert!(child.try_wait().unwrap().is_none());
            assert!(
                before.elapsed() < Duration::from_millis(100),
                "try_wait blocked on cleanup"
            );
            drop(holders);
            assert!(child.wait().is_err());
            drop(child);
            verify(&record);
        },
    );
}

fn startup_loss(name: &str, stage: i32) {
    if let Some(root) = std::env::var_os("ALLOY_COHORT_STARTUP_ROOT") {
        let root = std::path::PathBuf::from(root);
        // SAFETY: selects a bounded fault point in this isolated host only.
        unsafe { alloy_test_broker_fault(stage) };
        let thread_root = root.clone();
        std::thread::spawn(move || {
            let _ = start("/bin/sleep", &["30"], &thread_root);
        });
        let until = Instant::now() + Duration::from_secs(4);
        loop {
            // SAFETY: reads an atomic debug checkpoint in this host's owned slot.
            if unsafe { alloy_test_cohort_startup_stage(i32::from(stage == 5)) } == 1 {
                break;
            }
            assert!(Instant::now() < until, "startup checkpoint not reached");
            std::thread::sleep(Duration::from_millis(1));
        }
        let (mut path, mut label) = ([0i8; 1024], [0i8; 1024]);
        assert_eq!(
            // SAFETY: both writable buffers have capacity bytes; target zero is the pending slot.
            unsafe {
                alloy_test_cohort_record(0, path.as_mut_ptr(), label.as_mut_ptr(), path.len())
            },
            0
        );
        // SAFETY: successful hook initializes both terminated C strings.
        let path = unsafe { std::ffi::CStr::from_ptr(path.as_ptr()) }.to_string_lossy();
        // SAFETY: the second initialized buffer contains the exact owned job label.
        let label = unsafe { std::ffi::CStr::from_ptr(label.as_ptr()) }.to_string_lossy();
        fs::write(
            root.join("startup-registration.txt"),
            format!("{path}\n{label}\n"),
        )
        .unwrap();
        if stage == 5 {
            // SAFETY: pending slot has no target, and both recorded trusted identities are exact.
            assert_eq!(unsafe { alloy_test_cohort_kill(0, 0) }, 0);
            // SAFETY: signals only this pending invocation's isolated cleaner.
            assert_eq!(unsafe { alloy_test_cohort_kill(0, 2) }, 0);
        }
        // Preserve the race: the broker may still be between record and submit.
        std::process::exit(0);
    }
    isolated(name, |root| {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &format!("containment::{name}"), "--nocapture"])
            .env("ALLOY_COHORT_STARTUP_ROOT", root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let registration = fs::read_to_string(root.join("startup-registration.txt")).unwrap();
        let mut fields = registration.lines();
        let path = fields.next().unwrap();
        let label = std::ffi::CString::new(fields.next().unwrap()).unwrap();
        let next = start("/bin/sleep", &["30"], root).unwrap();
        assert!(
            !Path::new(path).exists(),
            "uncertain startup record was not retired"
        );
        // SAFETY: label is the terminated exact owned job label from this scenario.
        assert_eq!(unsafe { alloy_broker_registered(label.as_ptr()) }, 0);
        stop(next);
    });
}

#[test]
fn host_loss_between_record_and_submit_cannot_leave_untracked_job() {
    startup_loss(
        "host_loss_between_record_and_submit_cannot_leave_untracked_job",
        4,
    );
}

#[test]
fn empty_cohort_without_revocation_marker_is_retired_after_setup_crash() {
    startup_loss(
        "empty_cohort_without_revocation_marker_is_retired_after_setup_crash",
        5,
    );
}

#[test]
fn indeterminate_submission_stays_charged_until_positive_resolution() {
    isolated(
        "indeterminate_submission_stays_charged_until_positive_resolution",
        |root| {
            let child = start("/bin/sleep", &["30"], root).unwrap();
            // SAFETY: this hook owns its synthetic record and child identity, and never submits its label.
            assert_eq!(
                // SAFETY: this isolated test owns the exact identities and buffers passed to the hook.
                unsafe { alloy_test_uncertain_cohort_record(child.id() as i32) },
                0
            );
            stop(child);
        },
    );
}

#[test]
fn recovery_retries_after_timed_out_revoker_resumes() {
    isolated("recovery_retries_after_timed_out_revoker_resumes", |root| {
        let (mut child, record) = begin(root, "detach-hold", false);
        wait_records(&record, 2);
        let holders = pause_holders(child.id());
        // SAFETY: signals only this invocation's isolated cleaner.
        assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 2) }, 0);
        std::thread::sleep(Duration::from_millis(20));
        assert!(child.try_wait().unwrap().is_none());
        let until = Instant::now() + Duration::from_secs(7);
        loop {
            // SAFETY: observes the atomic recovery result for this owned invocation.
            if unsafe { alloy_test_cohort_recovery(child.id() as i32, 0) } != 0 {
                break;
            }
            assert!(
                Instant::now() < until,
                "recovery worker did not reach its bounded timeout"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(holders);
        loop {
            // SAFETY: observes this invocation's published revocation acknowledgement.
            if unsafe { alloy_test_cohort_recovery(child.id() as i32, 1) } == 1 {
                break;
            }
            assert!(Instant::now() < until, "resumed holder did not revoke");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            child.try_wait().unwrap().is_none(),
            "fresh revocation must rearm failed recovery"
        );
        assert!(child.wait().is_err());
        drop(child);
        verify(&record);
    });
}

#[test]
fn dropped_control_keeps_recovery_after_holder_timeout() {
    isolated(
        "dropped_control_keeps_recovery_after_holder_timeout",
        |root| {
            let (mut child, record) = begin(root, "detach-hold", false);
            wait_records(&record, 2);
            let holders = pause_holders(child.id());
            // SAFETY: this fault hook signals only the exact isolated cleaner.
            assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 2) }, 0);
            std::thread::sleep(Duration::from_millis(20));
            assert!(child.try_wait().unwrap().is_none());
            let until = Instant::now() + Duration::from_secs(7);
            loop {
                // SAFETY: reads the attempt status in this host's retained owned slot.
                if unsafe { alloy_test_cohort_recovery(child.id() as i32, 0) } != 0 {
                    break;
                }
                assert!(
                    Instant::now() < until,
                    "bounded recovery attempt did not end"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            drop(child); // The slot and its single watcher must outlive the control.
            drop(holders); // Publishes new evidence after no caller can poll/rearm it.
            verify(&record);
            stop(start("/bin/sleep", &["30"], root).unwrap());
        },
    );
}

#[test]
fn dropped_control_recovers_a_paused_cleaner_that_dies_later() {
    isolated(
        "dropped_control_recovers_a_paused_cleaner_that_dies_later",
        |root| {
            struct Cleanup(Vec<(i32, u64)>);
            impl Drop for Cleanup {
                fn drop(&mut self) {
                    for &(pid, identity) in &self.0 {
                        // SAFETY: signals require the captured kernel identity, never a reused PID.
                        unsafe {
                            alloy_owned_signal(pid, identity);
                        }
                    }
                }
            }
            let (child, record) = begin(root, "detach-hold", false);
            wait_records(&record, 2);
            let cleanup = Cleanup(
                records(&record)
                    .into_iter()
                    .filter(|p| p[0] > 0)
                    // SAFETY: reads the kernel identity of each independently recorded live fixture task.
                    .map(|p| (p[0], unsafe { alloy_process_identity(p[0]) }))
                    .collect(),
            );
            assert!(cleanup.0.iter().all(|p| p.1 != 0));
            let target = child.id() as i32;
            // SAFETY: stops only the recorded cleaner and waits for its stopped state.
            assert_eq!(unsafe { alloy_test_shell_owner_pause(target, 1) }, 0);
            // SAFETY: injects one failed cleaner signal; the retained watcher must retry.
            assert_eq!(unsafe { alloy_test_cleaner_signal_failure(target) }, 0);
            drop(child);
            // SAFETY: models supervisor loss after the final caller has gone. A completed
            // failover may already have retired the slot, in which case ESRCH is expected.
            let result = unsafe { alloy_test_cohort_kill(target, 2) };
            assert!(result == 0 || result == 3, "late cleaner kill: {result}");
            verify(&record); // Must pass before the fallback guard performs any cleanup.
            drop(cleanup);
            stop(start("/bin/sleep", &["30"], root).unwrap());
        },
    );
}

#[test]
fn watcher_allocation_failure_prevents_workload_execution() {
    isolated(
        "watcher_allocation_failure_prevents_workload_execution",
        |root| {
            // SAFETY: the isolated process injects exactly one watcher creation failure.
            unsafe {
                alloy_test_watcher_start_failure();
            }
            let result = start("/bin/sh", &["-c", "echo started >> workload-started"], root);
            assert!(
                result.is_err(),
                "workload started without a recovery watcher"
            );
            assert!(!root.join("workload-started").exists());
            stop(start("/bin/sleep", &["30"], root).unwrap());
        },
    );
}

#[test]
fn forced_close_retries_a_failed_cleaner_signal_without_another_caller() {
    isolated(
        "forced_close_retries_a_failed_cleaner_signal_without_another_caller",
        |root| {
            let (child, record) = begin(root, "detach-hold", false);
            wait_records(&record, 2);
            let target = child.id() as i32;
            // SAFETY: both hooks address only the exact invocation's trusted cleaner.
            assert_eq!(unsafe { alloy_test_shell_owner_pause(target, 1) }, 0);
            // SAFETY: this isolated fault is consumed by the mandatory retained watcher.
            assert_eq!(unsafe { alloy_test_cleaner_signal_failure(target) }, 0);
            drop(child);
            // No post-drop kill hook or poll can rescue this invocation.
            verify(&record);
            stop(start("/bin/sleep", &["30"], root).unwrap());
        },
    );
}

#[test]
fn workload_cannot_modify_or_hardlink_recovery_metadata() {
    isolated(
        "workload_cannot_modify_or_hardlink_recovery_metadata",
        |root| {
            let (child, record) = begin(root, "protect", false);
            let (mut path, mut label) = ([0i8; 1024], [0i8; 1024]);
            assert_eq!(
                // SAFETY: the initialized arrays remain writable for the exact hook capacity.
                unsafe {
                    alloy_test_cohort_record(
                        child.id() as i32,
                        path.as_mut_ptr(),
                        label.as_mut_ptr(),
                        path.len(),
                    )
                },
                0
            );
            // SAFETY: the successful hook wrote a terminated path into this initialized buffer.
            let path = unsafe { std::ffi::CStr::from_ptr(path.as_ptr()) }.to_bytes();
            fs::write(root.join("descendants.bin.protected-path"), path).unwrap();
            wait_records(&record, 5);
            let observations = records(&record);
            for kind in [7, 8] {
                assert!(
                    observations
                        .iter()
                        .any(|p| p[0] < 0 && p[1] == 1 && p[2] == kind),
                    "metadata access was not denied: {observations:?}"
                );
            }
            assert!(
                observations
                    .iter()
                    .any(|p| p[0] < 0 && p[1] == 0 && p[2] == 9),
                "ordinary hard links were restricted: {observations:?}"
            );
            stop(child);
            verify(&record);
        },
    );
}

#[test]
fn recovery_retries_transient_failure_after_revocation_is_already_known() {
    isolated(
        "recovery_retries_transient_failure_after_revocation_is_already_known",
        |root| {
            let (mut child, record) = begin(root, "detach-hold", false);
            wait_records(&record, 2);
            // SAFETY: signals only this invocation's exact isolated cleaner.
            assert_eq!(unsafe { alloy_test_cohort_kill(child.id() as i32, 2) }, 0);
            let until = Instant::now() + Duration::from_secs(4);
            loop {
                // SAFETY: reads the holder acknowledgement from this owned slot.
                if unsafe { alloy_test_cohort_recovery(child.id() as i32, 1) } == 1 {
                    break;
                }
                assert!(Instant::now() < until, "revocation was not published");
                std::thread::sleep(Duration::from_millis(1));
            }
            // SAFETY: injects exactly one failed sweep in this invocation's recovery worker.
            assert_eq!(
                // SAFETY: this isolated test owns the exact identities and buffers passed to the hook.
                unsafe { alloy_test_cohort_fail_recovery(child.id() as i32) },
                0
            );
            assert!(child.try_wait().unwrap().is_none());
            loop {
                // SAFETY: observes the atomic first-attempt result for this owned slot.
                if unsafe { alloy_test_cohort_recovery(child.id() as i32, 0) } == 5 {
                    break;
                }
                assert!(
                    Instant::now() < until,
                    "transient failure was not exercised"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            loop {
                match child.try_wait() {
                    Ok(None) => {}
                    Err(_) => break,
                    Ok(Some(_)) => panic!("cleaner loss invented success"),
                }
                assert!(Instant::now() < until, "quiesced recovery was not retried");
                std::thread::sleep(Duration::from_millis(5));
            }
            drop(child);
            verify(&record);
        },
    );
}
