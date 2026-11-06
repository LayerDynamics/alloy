//! `alloyd` binary entrypoint.

use alloyd::{BootRequest, DaemonConfig, DaemonService, bind_listener, boot_qwen3, serve_listener};
use engine::memory_enforcer::ACTIVE_POLL_INTERVAL;
use engine::memory_guard::{MemoryGuardConfig, MemoryGuardTier, PressureLevel};
use harness_core::{HardwareTier, TelemetryClient};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::task::{Context, Waker};
use std::thread;

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
    let mut memory_guard_tier = MemoryGuardTier::Balanced;
    let mut memory_guard_gb: Option<f64> = None;
    let mut admission_budget_override: Option<u64> = None;
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
            "--memory-guard" => {
                index += 1;
                let raw = require_value(&args, index, "--memory-guard")?;
                memory_guard_tier =
                    MemoryGuardTier::parse(&raw).map_err(|error| error.to_string())?;
            }
            "--memory-guard-gb" => {
                index += 1;
                let raw = require_value(&args, index, "--memory-guard-gb")?;
                memory_guard_gb = Some(
                    raw.parse()
                        .map_err(|_| format!("invalid --memory-guard-gb: {raw}"))?,
                );
                memory_guard_tier = MemoryGuardTier::Custom;
            }
            "--admission-budget-bytes" => {
                index += 1;
                let raw = require_value(&args, index, "--admission-budget-bytes")?;
                admission_budget_override = Some(
                    raw.parse()
                        .map_err(|_| format!("invalid --admission-budget-bytes: {raw}"))?,
                );
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
        socket_path: socket.map(PathBuf::from).unwrap_or(defaults.socket_path),
        db_path: db.map(PathBuf::from).unwrap_or(defaults.db_path),
    };

    let memory_guard = if memory_guard_tier == MemoryGuardTier::Custom {
        let bytes = if let Some(override_bytes) = admission_budget_override {
            override_bytes
        } else {
            let gb = memory_guard_gb.ok_or_else(|| {
                "--memory-guard custom requires --memory-guard-gb N (or --admission-budget-bytes)"
                    .to_owned()
            })?;
            if gb <= 0.0 {
                return Err("--memory-guard-gb must be positive".to_owned());
            }
            (gb * 1024.0 * 1024.0 * 1024.0) as u64
        };
        MemoryGuardConfig::custom(bytes).map_err(|error| error.to_string())?
    } else {
        MemoryGuardConfig {
            tier: memory_guard_tier,
            custom_ceiling_bytes: 0,
            enabled: true,
        }
    };

    let booted = boot_qwen3(BootRequest {
        artifact_dir: PathBuf::from(artifact),
        profile_id: profile.clone(),
        corpus_models_json: corpus_models.map(PathBuf::from),
        memory_guard,
        admission_budget_override_bytes: admission_budget_override,
        prefer_metal,
    })
    .map_err(|error| error.message)?;

    let telemetry = Arc::new(TelemetryClient::new(
        env!("CARGO_PKG_VERSION"),
        HardwareTier::Unknown,
        endpoint,
    ));
    let engine_for_enforcer = booted.engine.clone();
    let memory_enforcer = booted.memory_enforcer.clone();
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

    thread::Builder::new()
        .name("alloyd-memory-enforcer".to_owned())
        .spawn(move || {
            run_memory_enforcer_loop(memory_enforcer, engine_for_enforcer);
        })
        .map_err(|error| format!("spawn memory enforcer: {error}"))?;

    let listener = bind_listener(&config.socket_path).map_err(|error| error.message)?;
    eprintln!(
        "alloyd listening on {} (db {}, profile {}, api_version 1, real Qwen3, \
         memory-guard {} hard={} GiB binding={})",
        config.socket_path.display(),
        config.db_path.display(),
        booted.profile_id,
        booted.memory_guard_tier,
        booted.memory_reserved as f64 / (1024.0 * 1024.0 * 1024.0),
        booted.ceiling_binding.as_deref().unwrap_or("none"),
    );
    serve_listener(listener, service).map_err(|error| error.message)
}

fn run_memory_enforcer_loop(
    enforcer: engine::memory_enforcer::MemoryEnforcer,
    engine: engine::api::EngineCore<engine::runtime::FamilyRuntime>,
) {
    let mut pump = engine::runtime::MemoryEnforcementPump::new(engine, enforcer);
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut previous = PressureLevel::Ok;
    let mut previous_emergency = false;
    let mut last_cancellation = None;
    loop {
        // Never block pressure observation on an occupied native worker.
        // The pump owns at most one dispatched native action.
        match pump.poll_tick(&mut context) {
            Ok(poll) => {
                let tick = poll.tick;
                match (&tick.request_execution_cancellation, poll.cancellation) {
                    (Some(target), Some(Ok(engine::types::CancelOutcome::Requested)))
                        if last_cancellation.as_ref() != Some(target) =>
                    {
                        eprintln!(
                            "alloyd: emergency cancellation requested for {} (native cleanup pending)",
                            target.request_id().as_str(),
                        );
                        last_cancellation = Some(target.clone());
                    }
                    (_, Some(Err(error))) => {
                        eprintln!("alloyd: emergency cancellation signal failed: {error}");
                    }
                    _ => {}
                }
                if tick.status.pressure != previous || tick.status.emergency != previous_emergency {
                    eprintln!(
                        "alloyd: memory pressure {} -> {} (usage={} soft={} hard={} paused={} deferred={} emergency={})",
                        previous.as_str(),
                        tick.status.pressure.as_str(),
                        tick.status.usage_bytes,
                        tick.status.soft_bytes,
                        tick.status.hard_bytes,
                        tick.status.admissions_paused,
                        tick.deferred_for_pool_drain,
                        tick.status.emergency,
                    );
                    previous = tick.status.pressure;
                    previous_emergency = tick.status.emergency;
                }
                match poll.completed {
                    Some(Ok(report)) => {
                        if let Some(error) = report.allocator_error {
                            eprintln!("alloyd: allocator drain failed: {error}");
                        }
                        if let Some(cache) = report.allocator_cache {
                            if cache.before.cache_bytes > cache.after.cache_bytes {
                                eprintln!(
                                    "alloyd: MLX pool drain {} -> {} MiB",
                                    cache.before.cache_bytes / (1024 * 1024),
                                    cache.after.cache_bytes / (1024 * 1024)
                                );
                            }
                        }
                        match report.model {
                            engine::runtime::IdleModelReclaim::Evicted(instance) => {
                                eprintln!("alloyd: pressure evicted idle model {instance:?}")
                            }
                            engine::runtime::IdleModelReclaim::NoEligibleVictim => {}
                            engine::runtime::IdleModelReclaim::NotRequested => {}
                        }
                    }
                    Some(Err(error)) => {
                        eprintln!("alloyd: native memory enforcement failed: {error}")
                    }
                    None => {}
                }
            }
            Err(error) => eprintln!("alloyd: memory pressure observation failed: {error}"),
        }
        thread::sleep(ACTIVE_POLL_INTERVAL);
    }
}

fn noop_waker() -> Waker {
    use std::task::{RawWaker, RawWakerVTable};
    fn clone(pointer: *const ()) -> RawWaker {
        RawWaker::new(pointer, &VTABLE)
    }
    fn wake(_: *const ()) {}
    fn wake_by_ref(_: *const ()) {}
    fn drop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    // SAFETY: vtable methods are no-ops; the null data pointer is never read.
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
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
  --memory-guard TIER          safe|balanced|aggressive|custom (default: balanced)
  --memory-guard-gb N          Custom ceiling in GiB (implies --memory-guard custom)
  --admission-budget-bytes N   Test override for the admission ledger (optional)
  --metal / --cpu              Execution placement (default: metal on Apple Silicon, else cpu)
  --telemetry-endpoint URL     FR-35 flush URL
  -h, --help                   Show this help"
    );
}
