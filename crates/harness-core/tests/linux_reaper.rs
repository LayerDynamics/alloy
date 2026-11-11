#![cfg(target_os = "linux")]
use harness_core::plugin_host::{Limits, PluginConfig, PluginHost, sandbox::Sandbox};
use std::{
    path::Path,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tool_registry::execution::CancellationToken;

#[test]
fn linux_plugin_reaping_survives_competing_reapers() {
    const MODE: &str = "ALLOY_LINUX_REAPER_MODE";
    if let Ok(mode) = std::env::var(MODE) {
        unsafe extern "C" {
            fn signal(sig: i32, handler: usize) -> usize;
            fn waitpid(pid: i32, status: *mut i32, flags: i32) -> i32;
            fn kill(pid: i32, signal: i32) -> i32;
        }
        extern "C" fn caught_signal(_: i32) {}
        let abrt_handler = if mode == "ignore" {
            1
        } else {
            caught_signal as *const () as usize
        };
        // SAFETY: this isolated helper sets only its own SIGABRT disposition.
        assert_ne!(unsafe { signal(6, abrt_handler) }, usize::MAX);
        let fixture = std::env::var("ALLOY_LINUX_REAPER_FIXTURE").unwrap();
        let mut sentinel = Command::new("/bin/sleep").arg("120").spawn().unwrap();
        let sentinel_pid = sentinel.id() as i32;
        let dir = tempfile::tempdir().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = if mode == "thread" {
            let done = stop.clone();
            Some(std::thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    // SAFETY: isolated helper process intentionally owns its global reaper.
                    unsafe {
                        waitpid(-1, std::ptr::null_mut(), 1);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }))
        } else {
            // SAFETY: isolated helper process deliberately enables auto-reaping.
            assert_ne!(unsafe { signal(17, 1) }, usize::MAX);
            None
        };
        for plugin_mode in ["stop", "exit"] {
            let path = dir.path().join(format!("{plugin_mode}.pids"));
            let config = PluginConfig {
                id: "linux".into(),
                version: "1.0.0".into(),
                program: fixture.clone().into(),
                args: vec![path.to_str().unwrap().into(), plugin_mode.into()],
                cwd: dir.path().into(),
                sandbox: Sandbox::Trusted,
            };
            let limits = Limits {
                message_bytes: 16384,
                diagnostic_bytes: 4096,
                annotation_bytes: 1024,
                reason_bytes: 256,
                timeout: Duration::from_secs(3),
            };
            let host = PluginHost::start(
                vec![config],
                limits,
                Instant::now() + Duration::from_secs(8),
                &CancellationToken::default(),
            )
            .unwrap();
            let pids: Vec<u32> = std::fs::read_to_string(path)
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(pids.len(), 4);
            assert_eq!(
                pids[2],
                u32::from(mode == "ignore"),
                "SIGCHLD ignore must survive exec"
            );
            assert_eq!(
                pids[3],
                u32::from(mode == "ignore"),
                "SIGABRT ignore must survive exec while caught handler resets"
            );
            drop(host);
            // SAFETY: sentinel is our live, unreaped fixture child.
            let sentinel_alive = unsafe { kill(sentinel_pid, 0) };
            assert_eq!(sentinel_alive, 0, "unrelated sentinel was affected");
            for pid in pids.into_iter().take(2) {
                let until = Instant::now() + Duration::from_secs(3);
                loop {
                    let p = format!("/proc/{pid}/stat");
                    let live = std::fs::read_to_string(p).ok().is_some_and(|v| {
                        v.rsplit_once(") ")
                            .is_some_and(|(_, s)| !s.starts_with('Z') && !s.starts_with('X'))
                    });
                    if !live {
                        break;
                    }
                    assert!(Instant::now() < until, "fixture process {pid} remains live");
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        use std::os::unix::fs::PermissionsExt;
        let invalid = dir.path().join("missing-interpreter");
        std::fs::write(&invalid, "#!/no-such-alloy-interpreter\n").unwrap();
        std::fs::set_permissions(&invalid, std::fs::Permissions::from_mode(0o755)).unwrap();
        let config = PluginConfig {
            id: "linux".into(),
            version: "1.0.0".into(),
            program: invalid,
            args: vec![],
            cwd: dir.path().into(),
            sandbox: Sandbox::Trusted,
        };
        let limits = Limits {
            message_bytes: 16384,
            diagnostic_bytes: 4096,
            annotation_bytes: 1024,
            reason_bytes: 256,
            timeout: Duration::from_secs(1),
        };
        assert!(
            PluginHost::start(
                vec![config],
                limits,
                Instant::now() + Duration::from_secs(3),
                &CancellationToken::default()
            )
            .is_err()
        );
        // SAFETY: the still-live unrelated sentinel is the exact process spawned above.
        assert_eq!(unsafe { kill(sentinel_pid, 0) }, 0);
        stop.store(true, Ordering::Release);
        if let Some(t) = thread {
            t.join().unwrap();
        }
        let _ = sentinel.kill();
        let _ = sentinel.wait();
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let fixture = temp.path().join("fixture");
    let status = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linux_reaper_plugin.c"))
        .arg("-o")
        .arg(&fixture)
        .status()
        .unwrap();
    assert!(status.success());
    let cancel_driver = temp.path().join("cancel-driver");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let status = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(manifest.join("tests/fixtures/linux_supervisor_cancel.c"))
        .arg(manifest.join("native/linux_supervisor.c"))
        .arg("-o")
        .arg(&cancel_driver)
        .status()
        .unwrap();
    assert!(status.success());
    let output = Command::new(cancel_driver)
        .arg(temp.path().join("must-not-exist"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("LINUX_PRECOMMIT_CANCEL_ZERO_EFFECT"));
    for mode in ["thread", "ignore"] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "linux_plugin_reaping_survives_competing_reapers",
                "--nocapture",
            ])
            .env(MODE, mode)
            .env("ALLOY_LINUX_REAPER_FIXTURE", &fixture)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "mode={mode}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
