//! W6 acceptance through the production socket handler and configuration loader.
#[path = "../../harness-core/tests/support/scripted.rs"]
mod scripted;
use alloyd::{DaemonService, RpcRequest, bind_listener, handle_connection, metrics::MetricsSource};
use harness_core::{HardwareTier, TelemetryClient};
use scripted::*;
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::{UnixListener, UnixStream},
    sync::Arc,
    thread,
    time::Duration,
};
struct Metrics;
impl MetricsSource for Metrics {
    fn snapshot(&self) -> Result<Value, alloyd::RpcError> {
        Ok(json!({"responsive":true}))
    }
}
fn connect(
    listener: &UnixListener,
    service: Arc<DaemonService<ScriptedEngine>>,
) -> (BufReader<UnixStream>, thread::JoinHandle<()>) {
    let client =
        UnixStream::connect(listener.local_addr().unwrap().as_pathname().unwrap()).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let (server, _) = listener.accept().unwrap();
    let worker = thread::spawn(move || handle_connection(server, &service).unwrap());
    (BufReader::new(client), worker)
}
fn send(reader: &mut BufReader<UnixStream>, id: u64, method: &str, params: Value) {
    let line = json!({"jsonrpc":"2.0","id":id,"api_version":1,"method":method,"params":params})
        .to_string()
        + "\n";
    reader.get_mut().write_all(line.as_bytes()).unwrap();
}
fn receive(reader: &mut BufReader<UnixStream>) -> Value {
    let mut line = String::new();
    assert!(reader.read_line(&mut line).unwrap() > 0);
    serde_json::from_str(&line).unwrap()
}
#[test]
fn approval_rpc_remains_responsive() {
    for decision in ["allow", "deny", "pause"] {
        let dir = tempfile::tempdir().unwrap();
        let engine = ScriptedEngine::new(vec![
            vec![
                proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
                done(),
            ],
            vec![
                engine::types::GenerationEvent::TextDelta {
                    text: "finished".into(),
                },
                done(),
            ],
        ]);
        let service = DaemonService::open(
            dir.path().join("db"),
            engine,
            Arc::new(TelemetryClient::new(
                "test",
                HardwareTier::Unknown,
                "https://telemetry.invalid",
            )),
            128,
            0,
            0,
        )
        .unwrap()
        .with_metrics_source(Arc::new(Metrics));
        let config = json!({"read_root":dir.path(),"write_root":dir.path(),"shell_executables":[std::fs::canonicalize("/bin/echo").unwrap()],"shell_cwd_roots":[dir.path()],"http_destinations":[{"origin":"http://fixture.invalid/","address":"127.0.0.1"}],"policy":{"rules":[{"tool_id":"fs.write","allowed_arguments":[{"path":"marker.txt","content":"once"}],"require_approval":true,"timeout_ms":1000,"max_output_bytes":65536}]},"plugins":[]});
        let path = dir.path().join("tools.json");
        std::fs::write(&path, config.to_string()).unwrap();
        let artifacts = service
            .open_artifacts(&dir.path().join("artifacts"))
            .unwrap();
        let runtime = alloyd::tool_config::load(&path, artifacts).unwrap();
        let service = Arc::new(service.with_tools(runtime));
        let listener = bind_listener(&dir.path().join("rpc.sock")).unwrap();
        let (mut stream, worker) = connect(&listener, service.clone());
        let (mut control, control_worker) = connect(&listener, service.clone());
        send(&mut control, 1, "session.create", json!({"profile":"p"}));
        let session = receive(&mut control)["result"]["session_id"]
            .as_str()
            .unwrap()
            .to_owned();
        send(
            &mut stream,
            2,
            "session.send",
            json!({"session_id":session,"content":"write marker"}),
        );
        let call = loop {
            let e = receive(&mut stream);
            assert!(e.get("error").is_none(), "{e}");
            if e["params"]["event"] == "approval_requested" {
                assert!(e["params"]["body"]["summary"].is_string());
                break e["params"]["body"]["call_id"].as_str().unwrap().to_owned();
            }
        };
        assert!(!dir.path().join("marker.txt").exists());
        for method in ["daemon.status", "daemon.metrics"] {
            send(&mut control, 3, method, json!({}));
            assert!(receive(&mut control).get("result").is_some());
        }
        // Strict params and wrong-session decisions leave the exact intent pending.
        for params in [
            json!({"session_id":session,"call_id":call}),
            json!({"session_id":session,"call_id":call,"allow":"yes"}),
            json!({"session_id":"wrong","call_id":call,"allow":true}),
            json!({"session_id":session,"call_id":call,"allow":true,"extra":true}),
        ] {
            send(&mut control, 4, "session.approve", params);
            assert!(receive(&mut control).get("error").is_some());
            assert!(!dir.path().join("marker.txt").exists());
        }
        // One active turn per session, with the control path still responsive.
        send(
            &mut control,
            5,
            "session.send",
            json!({"session_id":session,"content":"overlap"}),
        );
        assert!(receive(&mut control).get("error").is_some());
        if decision == "pause" {
            send(
                &mut control,
                6,
                "session.pause",
                json!({"session_id":session}),
            );
        } else {
            send(
                &mut control,
                6,
                "session.approve",
                json!({"session_id":session,"call_id":call,"allow":decision=="allow"}),
            );
        }
        assert!(receive(&mut control).get("result").is_some());
        let mut terminals = 0;
        loop {
            let e = receive(&mut stream);
            if e["params"]["event"] == "generation_completed"
                || e["params"]["event"] == "request_failed"
            {
                terminals += 1;
            }
            if e["id"] == 2 {
                assert_eq!(e.get("error").is_some(), decision == "pause");
                break;
            }
        }
        assert_eq!(terminals, 1);
        assert_eq!(dir.path().join("marker.txt").exists(), decision == "allow");
        send(
            &mut control,
            7,
            "session.approve",
            json!({"session_id":session,"call_id":call,"allow":true}),
        );
        assert!(receive(&mut control).get("error").is_some());
        send(
            &mut control,
            8,
            "sessions.resume",
            json!({"session_id":session}),
        );
        let history = receive(&mut control);
        let events = history["result"]["events"].as_array().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| e["kind"] == "tool.started")
                .count(),
            usize::from(decision == "allow")
        );
        drop(stream);
        drop(control);
        worker.join().unwrap();
        control_worker.join().unwrap();
    }
}
#[test]
fn approve_rejects_non_boolean_and_unknown_fields_without_tools() {
    let dir = tempfile::tempdir().unwrap();
    let service = DaemonService::open(
        dir.path().join("db"),
        ScriptedEngine::new(vec![]),
        Arc::new(TelemetryClient::new(
            "test",
            HardwareTier::Unknown,
            "https://telemetry.invalid",
        )),
        128,
        0,
        0,
    )
    .unwrap();
    for params in [
        json!({}),
        json!({"session_id":"s","call_id":"c","allow":null}),
        json!({"session_id":"s","call_id":"c","allow":true,"extra":true}),
    ] {
        let request:RpcRequest=serde_json::from_value(json!({"jsonrpc":"2.0","id":1,"api_version":1,"method":"session.approve","params":params})).unwrap();
        let mut bytes = Vec::new();
        service.handle(request, &mut bytes).unwrap();
        let result: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result["error"]["code"], -32602);
    }
}

#[test]
fn ambiguous_approval_json_is_rejected_before_authority() {
    for raw in [
        r#"{"jsonrpc":"2.0","id":1,"method":"session.approve","params":{"session_id":"s","call_id":"c","allow":false,"allow":true}}"#,
        r#"{"jsonrpc":"2.0","id":1,"method":"session.approve","params":{"session_id":"s","call_id":"c","allow":false,"\u0061llow":true}}"#,
    ] {
        assert!(alloyd::rpc::parse_request_line(raw).is_err());
    }
}

#[test]
fn configured_tool_notifications_abort_before_approval_or_execution_on_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let engine = ScriptedEngine::new(vec![vec![
        proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
        done(),
    ]]);
    let service = DaemonService::open(
        dir.path().join("db"),
        engine,
        Arc::new(TelemetryClient::new(
            "test",
            HardwareTier::Unknown,
            "https://telemetry.invalid",
        )),
        128,
        0,
        0,
    )
    .unwrap();
    let config = json!({"read_root":dir.path(),"write_root":dir.path(),"shell_executables":[std::fs::canonicalize("/bin/echo").unwrap()],"shell_cwd_roots":[dir.path()],"http_destinations":[{"origin":"http://fixture.invalid/","address":"127.0.0.1"}],"policy":{"rules":[{"tool_id":"fs.write","allowed_arguments":[{"path":"marker.txt","content":"once"}],"require_approval":true,"timeout_ms":1000,"max_output_bytes":65536}]},"plugins":[]});
    let path = dir.path().join("tools.json");
    std::fs::write(&path, config.to_string()).unwrap();
    let artifacts = service
        .open_artifacts(&dir.path().join("artifacts"))
        .unwrap();
    let service = service.with_tools(alloyd::tool_config::load(&path, artifacts).unwrap());
    let defaults = service.rpc_transport_limits().unwrap();
    assert_eq!(defaults.max_notification_bytes(), 2 * 1024 * 1024);
    assert_eq!(defaults.write_timeout(), Duration::from_secs(5));
    let service = service.with_rpc_transport_limits(
        alloyd::rpc::RpcTransportLimits::new(128, Duration::from_millis(50)).unwrap(),
    );
    let request = |id, method, params| {
        serde_json::from_value(
            json!({"jsonrpc":"2.0","id":id,"api_version":1,"method":method,"params":params}),
        )
        .unwrap()
    };
    let mut bytes = Vec::new();
    service
        .handle(
            request(1, "session.create", json!({"profile":"p"})),
            &mut bytes,
        )
        .unwrap();
    let created: Value = serde_json::from_slice(&bytes).unwrap();
    let session = created["result"]["session_id"].as_str().unwrap();
    bytes.clear();
    service
        .handle(
            request(
                2,
                "session.send",
                json!({"session_id":session,"content":"write"}),
            ),
            &mut bytes,
        )
        .unwrap();
    let lines: Vec<Value> = String::from_utf8(bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["id"], 2);
    assert!(
        lines[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("notification byte limit")
    );
    assert!(!dir.path().join("marker.txt").exists());
    let mut history = Vec::new();
    service
        .handle(
            request(3, "sessions.resume", json!({"session_id":session})),
            &mut history,
        )
        .unwrap();
    let history: Value = serde_json::from_slice(&history).unwrap();
    let events = history["result"]["events"].as_array().unwrap();
    assert!(!events.iter().any(|e| e["kind"] == "tool.started"));
    assert!(
        events
            .iter()
            .any(|e| e["kind"] == "tool.completed" && e["payload"]["state"] == "Withdrawn")
    );
}

#[test]
fn pause_running_shell_reaps_process_without_affecting_another_session() {
    pause_running_operation(false, false);
}

#[test]
fn pause_running_shell_after_finished_request_eviction() {
    pause_running_operation(true, false);
}

#[test]
fn pause_running_plugin_hook_reaps_plugin_without_tool_effects() {
    pause_running_operation(true, true);
}

fn pause_running_operation(evicted: bool, hook: bool) {
    let dir = tempfile::tempdir().unwrap();
    let cwd = std::fs::canonicalize(dir.path()).unwrap();
    let shell = std::fs::canonicalize("/bin/sh").unwrap();
    // Creation alone is not publication: the reader must never observe the
    // empty file between shell redirection and echo. Rename the complete PID.
    let args = json!({"executable":shell,"args":["-c","echo $$ > owned.pid.tmp; /bin/mv owned.pid.tmp owned.pid; exec /bin/sleep 60"],"cwd":cwd});
    let engine = ScriptedEngine::new(vec![
        vec![proposal("shell.run", args.clone()), done()],
        vec![
            engine::types::GenerationEvent::TextDelta {
                text: "other session completed".into(),
            },
            done(),
        ],
        vec![
            engine::types::GenerationEvent::TextDelta {
                text: "other session still active".into(),
            },
            done(),
        ],
    ]);
    let mut engine = engine;
    engine.cancel_not_found = evicted;
    let service = DaemonService::open(
        dir.path().join("db"),
        engine,
        Arc::new(TelemetryClient::new(
            "test",
            HardwareTier::Unknown,
            "https://telemetry.invalid",
        )),
        128,
        0,
        0,
    )
    .unwrap()
    .with_metrics_source(Arc::new(Metrics));
    let config = json!({"read_root":cwd,"write_root":cwd,"shell_executables":[shell],"shell_cwd_roots":[cwd],"http_destinations":[{"origin":"http://fixture.invalid/","address":"127.0.0.1"}],"policy":{"rules":[{"tool_id":"shell.run","allowed_arguments":[args],"require_approval":true,"timeout_ms":30000,"max_output_bytes":65536}]},"plugins":[]});
    let path = dir.path().join("tools.json");
    std::fs::write(&path, config.to_string()).unwrap();
    let mut runtime = alloyd::tool_config::load(
        &path,
        service
            .open_artifacts(&dir.path().join("artifacts"))
            .unwrap(),
    )
    .unwrap();
    let hook_log = dir.path().join("hook-events");
    if hook {
        use harness_core::plugin_host::{Limits, PluginConfig, PluginHost, sandbox::Sandbox};
        use tool_registry::execution::CancellationToken;
        let fixture = dir.path().join("hook-fixture");
        let status = std::process::Command::new("rustc")
            .args(["--edition=2024", "-o"])
            .arg(&fixture)
            .arg(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../harness-core/tests/fixtures/tool_hook_plugin.rs"),
            )
            .status()
            .unwrap();
        assert!(status.success());
        let host = PluginHost::start_with_helper(
            vec![PluginConfig {
                id: "p".into(),
                version: "1.0.0".into(),
                program: fixture,
                args: vec!["p".into(), "hang".into(), hook_log.to_str().unwrap().into()],
                cwd: cwd.clone(),
                sandbox: Sandbox::Trusted,
            }],
            Limits {
                message_bytes: 16384,
                diagnostic_bytes: 4096,
                annotation_bytes: 1024,
                reason_bytes: 256,
                timeout: Duration::from_secs(30),
            },
            std::time::Instant::now() + Duration::from_secs(30),
            &CancellationToken::default(),
            &alloy_process_owner::spawn::default_helper().unwrap(),
        )
        .unwrap();
        runtime.plugins = std::sync::Mutex::new(host);
    }
    let service = Arc::new(service.with_tools(runtime));
    let listener = bind_listener(&dir.path().join("rpc.sock")).unwrap();
    let (mut stream, worker) = connect(&listener, service.clone());
    let (mut control, control_worker) = connect(&listener, service.clone());
    send(&mut control, 1, "session.create", json!({"profile":"p"}));
    let session = receive(&mut control)["result"]["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    send(&mut control, 2, "session.create", json!({"profile":"p"}));
    let other = receive(&mut control)["result"]["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    send(
        &mut stream,
        3,
        "session.send",
        json!({"session_id":session,"content":"run shell"}),
    );
    let call = loop {
        let event = receive(&mut stream);
        assert!(event.get("error").is_none(), "{event}");
        if event["params"]["event"] == "approval_requested" {
            break event["params"]["body"]["call_id"]
                .as_str()
                .unwrap()
                .to_owned();
        }
    };
    send(
        &mut control,
        4,
        "session.approve",
        json!({"session_id":session,"call_id":call,"allow":true}),
    );
    assert!(receive(&mut control).get("result").is_some());
    let pid = if hook {
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while !std::fs::read_to_string(&hook_log)
            .unwrap_or_default()
            .contains("p:before")
        {
            assert!(std::time::Instant::now() < until, "hook did not start");
            thread::sleep(Duration::from_millis(5));
        }
        std::fs::read_to_string(format!("{}.p.pid", hook_log.display()))
            .unwrap()
            .trim()
            .to_owned()
    } else {
        loop {
            let event = receive(&mut stream);
            assert!(event.get("error").is_none(), "{event}");
            if event["params"]["event"] == "tool_call_started" {
                break;
            }
        }
        let pidfile = dir.path().join("owned.pid");
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while !pidfile.exists() {
            assert!(std::time::Instant::now() < until, "shell did not start");
            thread::sleep(Duration::from_millis(10));
        }
        std::fs::read_to_string(pidfile).unwrap().trim().to_owned()
    };
    let pid: i32 = pid
        .parse()
        .expect("fixture must publish a complete integer PID");
    assert!(pid > 1 && pid != std::process::id() as i32);
    struct Sentinel(std::process::Child);
    impl Drop for Sentinel {
        fn drop(&mut self) {
            if matches!(self.0.try_wait(), Ok(None)) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }
    use std::os::unix::process::CommandExt;
    let mut sentinel = Sentinel(
        std::process::Command::new("/bin/sleep")
            .arg("60")
            .process_group(0)
            .spawn()
            .unwrap(),
    );
    assert!(sentinel.0.try_wait().unwrap().is_none());
    send(
        &mut control,
        5,
        "session.send",
        json!({"session_id":other,"content":"independent turn"}),
    );
    let mut other_done = false;
    loop {
        let event = receive(&mut control);
        if event["params"]["event"] == "generation_completed" {
            other_done = true;
        }
        if event["id"] == 5 {
            assert!(event.get("result").is_some(), "{event}");
            break;
        }
    }
    assert!(
        other_done,
        "other session did not finish while shell was active"
    );
    send(
        &mut control,
        6,
        "session.pause",
        json!({"session_id":session}),
    );
    let paused = receive(&mut control);
    assert_eq!(paused["result"]["status"], "paused", "{paused}");
    let mut terminals = 0;
    loop {
        let event = receive(&mut stream);
        if event["params"]["event"] == "generation_completed"
            || event["params"]["event"] == "request_failed"
        {
            terminals += 1;
        }
        if event["id"] == 3 {
            assert!(event.get("error").is_some());
            break;
        }
    }
    assert_eq!(terminals, 1);
    send(
        &mut control,
        7,
        "sessions.resume",
        json!({"session_id":session}),
    );
    let paused_history = receive(&mut control);
    assert_eq!(paused_history["result"]["status"], "paused");
    if hook {
        assert!(
            !paused_history["result"]["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event["kind"] == "tool.started")
        );
    }
    send(
        &mut control,
        8,
        "sessions.resume",
        json!({"session_id":other}),
    );
    let other_history = receive(&mut control);
    assert_eq!(other_history["result"]["status"], "active");
    assert!(
        other_history["result"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["kind"] == "assistant.message"
                && event["payload"]["content"] == "other session completed")
    );
    send(
        &mut control,
        9,
        "session.send",
        json!({"session_id":other,"content":"continue after independent pause"}),
    );
    loop {
        let event = receive(&mut control);
        if event["id"] == 9 {
            assert!(event.get("result").is_some(), "{event}");
            break;
        }
    }

    if hook {
        assert!(
            !dir.path().join("owned.pid").exists(),
            "tool executed while before hook was cancelled"
        );
    }
    assert!(
        sentinel.0.try_wait().unwrap().is_none(),
        "unrelated sentinel died during pause"
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    loop {
        // SAFETY: signal zero observes existence without delivering any signal.
        if unsafe { kill(pid, 0) } != 0 {
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(3),
                "PID inspection must establish ESRCH, not an observation failure"
            );
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "paused shell {pid} survived"
        );
        thread::sleep(Duration::from_millis(10));
    }
    drop(stream);
    drop(control);
    worker.join().unwrap();
    control_worker.join().unwrap();
}

#[test]
fn projected_batch_rpc_fit_executes_and_overflow_has_no_effect() {
    for (fits, batch_overflow) in [(true, false), (false, false), (false, true)] {
        let dir = tempfile::tempdir().unwrap();
        let mut first_round = vec![proposal(
            "fs.write",
            json!({"path":"marker.txt","content":"once"}),
        )];
        if batch_overflow {
            first_round.push(proposal(
                "fs.write",
                json!({"path":"second.txt","content":"twice"}),
            ));
        }
        first_round.push(done());
        let mut engine = ScriptedEngine::new(vec![
            first_round,
            vec![
                engine::types::GenerationEvent::TextDelta {
                    text: "finished".into(),
                },
                done(),
            ],
        ]);
        // The same budget admits one call but rejects the combined pair.
        engine.projected_byte_limit = if fits || batch_overflow { 11000 } else { 1 };
        let service = DaemonService::open(
            dir.path().join("db"),
            engine,
            Arc::new(TelemetryClient::new(
                "test",
                HardwareTier::Unknown,
                "https://telemetry.invalid",
            )),
            128,
            0,
            0,
        )
        .unwrap();
        let config = json!({"read_root":dir.path(),"write_root":dir.path(),"shell_executables":[std::fs::canonicalize("/bin/echo").unwrap()],"shell_cwd_roots":[dir.path()],"http_destinations":[{"origin":"http://fixture.invalid/","address":"127.0.0.1"}],"policy":{"rules":[{"tool_id":"fs.write","allowed_arguments":[{"path":"marker.txt","content":"once"},{"path":"second.txt","content":"twice"}],"require_approval":false,"timeout_ms":1000,"max_output_bytes":65536}]},"plugins":[]});
        let path = dir.path().join("tools.json");
        std::fs::write(&path, config.to_string()).unwrap();
        let artifacts = service
            .open_artifacts(&dir.path().join("artifacts"))
            .unwrap();
        let runtime = alloyd::tool_config::load(&path, artifacts).unwrap();
        let service = Arc::new(service.with_tools(runtime));
        let listener = bind_listener(&dir.path().join("rpc.sock")).unwrap();
        let (mut stream, worker) = connect(&listener, service.clone());
        let (mut control, control_worker) = connect(&listener, service.clone());
        send(&mut control, 1, "session.create", json!({"profile":"p"}));
        let session = receive(&mut control)["result"]["session_id"]
            .as_str()
            .unwrap()
            .to_owned();
        send(
            &mut stream,
            2,
            "session.send",
            json!({"session_id":session,"content":"write marker"}),
        );
        let mut saw_start = false;
        loop {
            let event = receive(&mut stream);
            if event["params"]["event"] == "approval_requested" {
                let call = event["params"]["body"]["call_id"].as_str().unwrap();
                send(
                    &mut control,
                    3,
                    "session.approve",
                    json!({"session_id":session,"call_id":call,"allow":true}),
                );
                assert!(receive(&mut control).get("result").is_some());
            }
            saw_start |= event["params"]["event"] == "tool_call_started";
            if event["id"] == 2 {
                assert_eq!(event.get("error").is_none(), fits, "{event}");
                break;
            }
        }
        assert_eq!(saw_start, fits);
        assert_eq!(dir.path().join("marker.txt").exists(), fits);
        assert!(!dir.path().join("second.txt").exists());
        send(
            &mut control,
            4,
            "sessions.resume",
            json!({"session_id":session}),
        );
        let history = receive(&mut control);
        let events = history["result"]["events"].as_array().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| e["kind"] == "tool.started")
                .count(),
            usize::from(fits)
        );
        if !fits {
            assert!(!events.iter().any(|e| e["kind"] == "tool.requested"));
        }
        drop(stream);
        drop(control);
        worker.join().unwrap();
        control_worker.join().unwrap();
    }
}
