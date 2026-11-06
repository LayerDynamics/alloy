//! Installed-binary diagnostics must work without a model or daemon state.

use serde_json::{Value, json};
use std::process::{Command, Output};
use tempfile::TempDir;

fn run(directory: &TempDir, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_alloyd"))
        .args(args)
        .current_dir(directory.path())
        // The diagnostic has no need to locate a user's configuration. This
        // also catches accidental calls to normal daemon path initialization.
        .env_remove("HOME")
        .output()
        .expect("run alloyd diagnostic")
}

fn assert_no_files(directory: &TempDir) {
    assert_eq!(
        std::fs::read_dir(directory.path()).unwrap().count(),
        0,
        "installation diagnostics must not create models, sockets, databases, or configuration",
    );
}

#[test]
fn version_and_help_are_model_free() {
    let directory = TempDir::new().unwrap();
    for flag in ["--version", "-V"] {
        let output = run(&directory, &[flag]);
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("alloyd {}\n", env!("CARGO_PKG_VERSION")),
        );
        assert!(output.stderr.is_empty());
    }
    let output = run(&directory, &["--help"]);
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8(output.stderr).unwrap();
    assert!(help.contains("--version"));
    assert!(help.contains("--check-native"));
    assert_no_files(&directory);
}

#[test]
fn native_check_executes_supported_devices_without_a_model_or_configuration() {
    let directory = TempDir::new().unwrap();
    let output = run(&directory, &["--check-native"]);
    assert!(output.status.success(), "{output:?}");
    assert!(
        output.stdout.len() < 4096,
        "diagnostic output must be bounded"
    );
    assert!(output.stderr.is_empty(), "{output:?}");
    assert_eq!(
        output.stdout.iter().filter(|&&byte| byte == b'\n').count(),
        1
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(report["platform"], std::env::consts::OS);
    assert_eq!(report["architecture"], std::env::consts::ARCH);
    assert_eq!(report["ok"], true);
    assert_eq!(report["cpu"]["status"], "passed");
    assert_eq!(report["cpu"]["result"], json!([0.5, -3.0, -6.0, -1.0]));

    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        // Required hardware coverage: no skip or CPU fallback on Apple Silicon.
        assert_eq!(report["metal"]["required"], true);
        assert_eq!(report["metal"]["status"], "passed");
        assert_eq!(report["metal"]["result"], report["cpu"]["result"]);
        assert_eq!(report["shared_arrays"]["status"], "passed");
        assert_eq!(report["shared_arrays"]["input_storage_unchanged"], true);
        assert_eq!(report["shared_arrays"]["metal_output_cpu_readable"], true);
        assert_eq!(
            report["shared_arrays"]["cpu_to_metal_result"],
            json!([1.5, -5.0, -3.0, -5.0]),
        );
        assert_eq!(
            report["shared_arrays"]["metal_to_cpu_result"],
            json!([1.5, -5.0, -3.0, -5.0]),
        );
    } else {
        assert_eq!(report["metal"]["required"], false);
        assert_eq!(report["metal"]["status"], "unsupported");
        assert_eq!(report["metal"]["reason"], "requires_apple_silicon_macos");
        assert_eq!(report["shared_arrays"]["status"], "unsupported");
    }
    assert_no_files(&directory);
}

#[test]
fn diagnostics_reject_options_that_could_weaken_or_start_the_check() {
    let directory = TempDir::new().unwrap();
    for args in [
        vec!["--check-native", "--cpu"],
        vec!["--check-native", "--artifact", "missing-model"],
        vec!["--socket", "unexpected.sock", "--check-native"],
        vec!["--version", "--check-native"],
        vec!["--version", "--artifact", "missing-model"],
    ] {
        let output = run(&directory, &args);
        assert!(!output.status.success(), "{output:?}");
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("must each be used alone")
        );
        assert_no_files(&directory);
    }
}
