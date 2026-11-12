#[path = "support/scripted.rs"]
mod scripted;
use engine::types::{GenerationEvent, ToolContextLimits};
use harness_core::telemetry::TelemetryTransport;
use harness_core::{artifact::*, plugin_host::*, policy::*, supervisor::*, tool_execution::*, *};
use scripted::*;
use serde_json::json;
use std::{
    io::Write,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};
use tool_registry::{
    builtins::fs::{FileReadTool, FileWriteTool, ReadScope, WriteScope},
    execution::CancellationToken,
};
fn start_host(
    configs: Vec<PluginConfig>,
    limits: Limits,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Result<PluginHost, HostError> {
    PluginHost::start_with_helper(
        configs,
        limits,
        deadline,
        cancel,
        std::path::Path::new(env!("CARGO_BIN_EXE_alloy-plugin-pipe-helper")),
    )
}
fn setup(root: &std::path::Path, store: &mut SessionStore, prompt: bool) -> ToolRuntime {
    let mut executor = ReceiptExecutor::default();
    let definitions = tool_registry::builtins::definitions(1000, 65536).unwrap();
    executor
        .register(
            definitions[0].clone(),
            Arc::new(FileReadTool::new(ReadScope::open(root).unwrap(), 8192).unwrap()),
        )
        .unwrap();
    executor
        .register(
            definitions[2].clone(),
            Arc::new(FileWriteTool::new(WriteScope::open(root, 65536).unwrap())),
        )
        .unwrap();
    let rules = vec![
        PolicyRule {
            tool_id: "fs.read".into(),
            allowed_arguments: vec![
                json!({"path":"fixture.txt","offset":0,"length":4096})
                    .as_object()
                    .unwrap()
                    .clone(),
            ],
            require_approval: prompt,
            timeout_ms: 1000,
            max_output_bytes: 65536,
        },
        PolicyRule {
            tool_id: "fs.write".into(),
            allowed_arguments: vec![
                json!({"path":"marker.txt","content":"once"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ],
            require_approval: true,
            timeout_ms: 1000,
            max_output_bytes: 65536,
        },
    ];
    ToolRuntime {
        executor,
        policy: DaemonPolicy::new(PolicyConfig { rules }).unwrap(),
        artifacts: ArtifactStore::open(store, root.join("artifacts"), Default::default()).unwrap(),
        plugins: Mutex::new(
            start_host(
                vec![],
                Limits {
                    message_bytes: 16384,
                    diagnostic_bytes: 4096,
                    annotation_bytes: 1024,
                    reason_bytes: 256,
                    timeout: Duration::from_millis(200),
                },
                Instant::now() + Duration::from_secs(5),
                &CancellationToken::default(),
            )
            .unwrap(),
        ),
    }
}
fn config() -> ToolLoopConfig {
    ToolLoopConfig {
        round: ToolRoundConfig {
            context_limits: ToolContextLimits {
                max_tools: 16,
                max_history_blocks: 128,
                max_json_bytes: 1048576,
            },
            max_output_tokens: 128,
            max_rounds: 3,
            max_proposals: 8,
            max_proposal_bytes: 65536,
            max_slab_bytes: 1048576,
            max_text_bytes: 65536,
            max_warning_bytes: 4096,
            max_events: 128,
            deadline: Instant::now() + Duration::from_secs(5),
        },
        max_calls: 8,
        inline_bytes: 8192,
    }
}
struct RecordingTransport(mpsc::Sender<TelemetryPayload>);

impl TelemetryTransport for RecordingTransport {
    fn post_json(&self, endpoint: &str, body: &str) -> Result<(), String> {
        assert_eq!(endpoint, "https://telemetry.invalid");
        let payload: TelemetryPayload = serde_json::from_str(body).map_err(|e| e.to_string())?;
        payload.assert_anonymous().map_err(|e| e.to_string())?;
        self.0.send(payload).map_err(|e| e.to_string())
    }
}

fn telemetry() -> (Arc<TelemetryClient>, mpsc::Receiver<TelemetryPayload>) {
    let (sender, receiver) = mpsc::channel();
    let client = Arc::new(TelemetryClient::with_transport(
        "test",
        HardwareTier::Unknown,
        "https://telemetry.invalid",
        Box::new(RecordingTransport(sender)),
    ));
    (client, receiver)
}
#[test]
fn tool_roundtrip_overhead_under_50ms() {
    const SAMPLES: usize = 32;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("fixture.txt"), "actual fixture bytes").unwrap();
    let mut store = SessionStore::open(dir.path().join("session.db")).unwrap();
    for i in 0..SAMPLES {
        store.create_session(format!("latency-{i}"), "p").unwrap();
    }
    let runtime = setup(dir.path(), &mut store, false);
    let store = Mutex::new(store);
    let mut samples = Vec::new();
    for i in 0..SAMPLES {
        let engine = ScriptedEngine::new(vec![
            vec![
                proposal(
                    "fs.read",
                    json!({"path":"fixture.txt","offset":0,"length":4096}),
                ),
                done(),
            ],
            vec![
                GenerationEvent::TextDelta {
                    text: "read complete".into(),
                },
                done(),
            ],
        ]);
        let (telemetry, delivered) = telemetry();
        let supervisor = Supervisor::new(engine, telemetry);
        let start = Instant::now();
        let outcome = supervisor
            .run_tool_turn(
                &store,
                &runtime,
                &format!("latency-{i}"),
                "read fixture",
                config(),
                &CancellationToken::default(),
                |_| Ok(()),
                |_| Ok(()),
            )
            .unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        // Keep the real worker and turn-completion flush, but do not let DNS
        // for a deliberately invalid endpoint perturb later timing samples.
        // Delivery is verified outside the measured tool-roundtrip boundary.
        let payload = delivered.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(payload.turn_count, 1);
        assert_eq!(payload.session_count, 0);
        assert_eq!(outcome.assistant_text, "read complete");
        let contexts = supervisor.engine().contexts.lock().unwrap();
        assert!(
            format!("{:?}", contexts.last().unwrap().tool_context())
                .contains("actual fixture bytes")
        );
    }
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / SAMPLES as f64;
    // Direct stdout retains successful samples under the default test harness.
    let mut output = std::io::stdout().lock();
    writeln!(
        output,
        "P2-16 samples={SAMPLES} boundary=run_tool_turn(mock_inference,fs.read,no_approval,no_client_transport) min_ms={:.3} median_ms={:.3} p95_ms={:.3} max_ms={:.3} mean_ms={mean:.3}",
        samples[0],
        samples[SAMPLES / 2],
        samples[(SAMPLES * 95).div_ceil(100) - 1],
        samples[SAMPLES - 1]
    ).unwrap();
    writeln!(output, "P2-16 all_samples_ms={samples:?}").unwrap();
    assert!(
        samples[SAMPLES - 1] < 50.0,
        "a sampled round trip exceeded 50 ms"
    );
}
