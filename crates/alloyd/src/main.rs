//! `alloyd` binary entrypoint.

use alloyd::{
    BootRequest, DaemonConfig, DaemonService, bind_listener, boot_qwen3, serve_listener,
};
use harness_core::{HardwareTier, TelemetryClient};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("alloyd: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut socket = None;
    let mut db = None;
    let mut artifact = None;
    let mut corpus_models = None;
    let mut profile = "qwen3-4b-4bit".to_owned();
    let mut max_output_tokens = 512usize;
    let mut admission_budget = 8 * 1024 * 1024 * 1024u64;
    let mut prefer_metal = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let mut endpoint = "https://telemetry.invalid/v1".to_owned();

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--socket" => {
                index += 1;
                socket = Some(require_value(&args, index, "--socket")?);
            }
            "--db" => {
                index += 1;
                db = Some(require_value(&args, index, "--db")?);
            }
            "--artifact" => {
                index += 1;
                artifact = Some(require_value(&args, index, "--artifact")?);
            }
            "--corpus-models" => {
                index += 1;
                corpus_models = Some(require_value(&args, index, "--corpus-models")?);
            }
            "--profile" => {
                index += 1;
                profile = require_value(&args, index, "--profile")?;
            }
            "--max-output-tokens" => {
                index += 1;
                let raw = require_value(&args, index, "--max-output-tokens")?;
                max_output_tokens = raw
                    .parse()
                    .map_err(|_| format!("invalid --max-output-tokens: {raw}"))?;
            }
            "--admission-budget-bytes" => {
                index += 1;
                let raw = require_value(&args, index, "--admission-budget-bytes")?;
                admission_budget = raw
                    .parse()
                    .map_err(|_| format!("invalid --admission-budget-bytes: {raw}"))?;
            }
            "--cpu" => prefer_metal = false,
            "--metal" => prefer_metal = true,
            "--telemetry-endpoint" => {
                index += 1;
                endpoint = require_value(&args, index, "--telemetry-endpoint")?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }

    let artifact = artifact.ok_or_else(|| {
        "--artifact PATH is required (local mlx-community Qwen3-4B-4bit directory for profile qwen3-4b-4bit)".to_owned()
    })?;

    let defaults = DaemonConfig::default_paths().map_err(|error| error.message)?;
    let config = DaemonConfig {
        socket_path: socket
            .map(PathBuf::from)
            .unwrap_or(defaults.socket_path),
        db_path: db.map(PathBuf::from).unwrap_or(defaults.db_path),
    };

    let booted = boot_qwen3(BootRequest {
        artifact_dir: PathBuf::from(artifact),
        profile_id: profile.clone(),
        corpus_models_json: corpus_models.map(PathBuf::from),
        admission_budget_bytes: admission_budget,
        prefer_metal,
    })
    .map_err(|error| error.message)?;

    let telemetry = Arc::new(TelemetryClient::new(
        env!("CARGO_PKG_VERSION"),
        HardwareTier::Unknown,
        endpoint,
    ));
    let service = Arc::new(
        DaemonService::open(
            &config.db_path,
            booted.engine,
            telemetry,
            max_output_tokens,
            booted.models_resident,
            booted.memory_reserved,
        )
        .map_err(|error| error.message)?,
    );

    let listener = bind_listener(&config.socket_path).map_err(|error| error.message)?;
    eprintln!(
        "alloyd listening on {} (db {}, profile {}, api_version 1, real Qwen3)",
        config.socket_path.display(),
        config.db_path.display(),
        booted.profile_id
    );
    serve_listener(listener, service).map_err(|error| error.message)
}

fn require_value(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    eprintln!(
        "\
alloyd — Alloy local daemon (Phase-1 B.6 subset)

USAGE:
  alloyd --artifact PATH [--profile qwen3-4b-4bit] [--socket PATH] [--db PATH]

OPTIONS:
  --artifact PATH              Required local Qwen3-4B-4bit mlx-community artifact directory
  --profile NAME               Must be qwen3-4b-4bit in Phase 1 (default)
  --corpus-models PATH         Optional bench/corpus/models.json for sha256 verification
  --socket PATH                UDS path (default: ~/.alloy/alloyd.sock)
  --db PATH                    SQLite session store (default: ~/.alloy/sessions.sqlite)
  --max-output-tokens N        Greedy decode cap (default: 512)
  --admission-budget-bytes N   Engine admission ceiling (default: 8 GiB)
  --metal / --cpu              Execution placement (default: metal on Apple Silicon, else cpu)
  --telemetry-endpoint URL     FR-35 flush URL
  -h, --help                   Show this help"
    );
}
