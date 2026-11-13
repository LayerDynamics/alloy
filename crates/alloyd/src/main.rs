//! `alloyd` binary entrypoint.

mod native_check;
mod rpc_transport_config;

use alloyd::{
    BootRequest, DaemonConfig, DaemonService, LiveEngineMetrics, bind_listener, boot_native,
    serve_listener,
};
use engine::memory_enforcer::ACTIVE_POLL_INTERVAL;
use engine::memory_guard::{MemoryGuardConfig, MemoryGuardTier, PressureLevel};
use engine::tiered_kv::{HotCacheLimit, TieredKvConfig};
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
    // Installation diagnostics must not enter model, configuration, session,
    // telemetry, or socket setup. Keep them standalone so startup options
    // cannot accidentally change what a successful check promises.
    if args.len() == 1 {
        match args[0].as_str() {
            "--version" | "-V" => {
                println!("alloyd {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--check-native" => {
                let (passed, report) = native_check::run();
                println!("{report}");
                return if passed {
                    Ok(())
                } else {
                    Err("native check failed; see the JSON diagnostic".to_owned())
                };
            }
            _ => {}
        }
    }
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--version" | "-V" | "--check-native"))
    {
        return Err("--version and --check-native must each be used alone".to_owned());
    }
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
    let mut paged_ssd_cache_dir = None;
    let mut hot_cache_limit = HotCacheLimit::Percent(25);
    let mut hot_cache_only = false;
    let mut tools_config: Option<std::path::PathBuf> = None;
    let mut rpc_transport = rpc_transport_config::TransportArgs::default();

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--tools-config" => {
                index += 1;
                tools_config = Some(require_value(&args, index, "--tools-config")?.into());
            }
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
            "--rpc-max-notification-bytes" => {
                index += 1;
                rpc_transport.set_maximum(&require_value(
                    &args,
                    index,
                    "--rpc-max-notification-bytes",
                )?)?;
            }
            "--rpc-write-timeout-ms" => {
                index += 1;
                rpc_transport.set_timeout(&require_value(
                    &args,
                    index,
                    "--rpc-write-timeout-ms",
                )?)?;
            }
            "--cpu" => prefer_metal = false,
            "--metal" => prefer_metal = true,
            "--paged-ssd-cache-dir" => {
                index += 1;
                paged_ssd_cache_dir = Some(PathBuf::from(require_value(
                    &args,
                    index,
                    "--paged-ssd-cache-dir",
                )?));
            }
            "--hot-cache-max-size" => {
                index += 1;
                let value = require_value(&args, index, "--hot-cache-max-size")?;
                hot_cache_limit =
                    HotCacheLimit::parse(&value).map_err(|error| error.to_string())?;
            }
            "--hot-cache-only" => hot_cache_only = true,
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

    let rpc_transport_limits = rpc_transport.finish()?;

    let artifact = artifact.ok_or_else(|| {
        "--artifact PATH is required (local artifact directory for the selected frozen native profile)".to_owned()
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

    let cache_directory = paged_ssd_cache_dir.unwrap_or_else(|| {
        config
            .db_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("kv-cache")
    });
    let mut tiered_kv = TieredKvConfig::new(Some(cache_directory), hot_cache_limit);
    tiered_kv.hot_cache_only = hot_cache_only;
    let booted = boot_native(BootRequest {
        artifact_dir: PathBuf::from(artifact),
        profile_id: profile.clone(),
        corpus_models_json: corpus_models.map(PathBuf::from),
        memory_guard,
        admission_budget_override_bytes: admission_budget_override,
        prefer_metal,
        tools_enabled: tools_config.is_some(),
        tiered_kv,
    })
    .map_err(|error| error.message)?;

    let telemetry = Arc::new(TelemetryClient::new(
        env!("CARGO_PKG_VERSION"),
        HardwareTier::Unknown,
        endpoint,
    ));
    let engine_for_enforcer = booted.engine.clone();
    let memory_enforcer = booted.memory_enforcer.clone();
    let metrics = Arc::new(LiveEngineMetrics::new(
        booted.engine.clone(),
        booted.memory_enforcer.clone(),
    ));
    let mut service = DaemonService::open(
        &config.db_path,
        booted.engine,
        telemetry,
        max_output_tokens,
        booted.models_resident,
        booted.memory_reserved,
    )
    .map_err(|error| error.message)?
    .with_metrics_source(metrics);
    if let Some(path) = tools_config {
        let artifacts = service
            .open_artifacts(&config.db_path.with_extension("artifacts"))
            .map_err(|e| e.message)?;
        service = service.with_tools(alloyd::tool_config::load(&path, artifacts)?);
    }
    let service = Arc::new(rpc_transport_config::apply(service, rpc_transport_limits));
    if let Some(limits) = rpc_transport_limits {
        eprintln!(
            "alloyd: RPC slab transport max_bytes={} write_timeout_ms={} (tool policy unchanged)",
            limits.max_notification_bytes(),
            limits.write_timeout().as_millis()
        );
    }

    thread::Builder::new()
        .name("alloyd-memory-enforcer".to_owned())
        .spawn(move || {
            run_memory_enforcer_loop(memory_enforcer, engine_for_enforcer);
        })
        .map_err(|error| format!("spawn memory enforcer: {error}"))?;

    let listener = bind_listener(&config.socket_path).map_err(|error| error.message)?;
    eprintln!(
        "alloyd listening on {} (db {}, profile {}, api_version 1, native MLX, \
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
                        if let Some(cache) = report.allocator_cache
                            && cache.before.cache_bytes > cache.after.cache_bytes
                        {
                            eprintln!(
                                "alloyd: MLX pool drain {} -> {} MiB",
                                cache.before.cache_bytes / (1024 * 1024),
                                cache.after.cache_bytes / (1024 * 1024)
                            );
                        }
                        match report.model {
                            engine::runtime::IdleModelReclaim::Evicted(instance) => {
                                eprintln!("alloyd: pressure evicted idle model {instance:?}")
                            }
                            engine::runtime::IdleModelReclaim::NoEligibleVictim => {}
                            engine::runtime::IdleModelReclaim::DeferredForKv => {
                                eprintln!(
                                    "alloyd: pressure released idle KV; model weights retained"
                                );
                            }
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
  alloyd --version
  alloyd --check-native

OPTIONS:
  --artifact PATH              Required local Qwen3-4B-4bit mlx-community artifact directory
  --profile NAME               qwen3-4b-4bit (default), llama-3.1-8b-4bit or llama-3.1-8b-8bit
  --corpus-models PATH         Optional bench/corpus/models.json for sha256 verification
  --socket PATH                UDS path (default: ~/.alloy/alloyd.sock)
  --db PATH                    SQLite session store (default: ~/.alloy/sessions.sqlite)
  --tools-config PATH          Explicit scoped tools, exact-argument policy and ordered plugins
  --max-output-tokens N        Greedy decode cap (default: 512)
  --memory-guard TIER          safe|balanced|aggressive|custom (default: balanced)
  --memory-guard-gb N          Custom ceiling in GiB (implies --memory-guard custom)
  --admission-budget-bytes N   Test override for the admission ledger (optional)
  --paged-ssd-cache-dir PATH   Persistent KV blocks (default: kv-cache beside the session DB)
  --hot-cache-max-size SIZE    Idle KV cap: bytes, KiB/MiB/GiB, or budget percent (default: 25%)
  --hot-cache-only             Keep bounded RAM retention; disable the SSD tier
  --metal                     Metal transformer operations with shared CPU/Metal memory
                              Default on native Apple Silicon; no silent CPU fallback
  --cpu                       MLX CPU transformer operations (default on other platforms)
  --rpc-max-notification-bytes N  Full escaped slab notification limit, including newline
  --rpc-write-timeout-ms N     Socket per-write timeout in milliseconds
                              Supply both RPC flags or neither; tool mode defaults to
                              2 MiB notifications and a 5-second write timeout
                              These flags do not enable tools or approvals
  --telemetry-endpoint URL     FR-35 flush URL
  -V, --version               Show the daemon version without starting it
  --check-native              Check native CPU operations; on Apple Silicon also
                              require Metal and shared arrays (JSON; no model needed)
  -h, --help                   Show this help"
    );
}
