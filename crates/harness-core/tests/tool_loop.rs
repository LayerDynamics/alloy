#[path = "support/scripted.rs"]
mod scripted;
use engine::types::{GenerationEvent, ToolContextLimits};
use harness_core::{artifact::*, plugin_host::*, policy::*, supervisor::*, tool_execution::*, *};
use scripted::*;
use serde_json::json;
use std::{
    sync::{Arc, Mutex},
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
fn telemetry() -> Arc<TelemetryClient> {
    Arc::new(TelemetryClient::new(
        "test",
        HardwareTier::Unknown,
        "https://telemetry.invalid",
    ))
}
#[test]
fn canonical_tool_loop_persists_results() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("fixture.txt"), "actual fixture bytes").unwrap();
    let db = dir.path().join("session.db");
    let mut store = SessionStore::open(&db).unwrap();
    store.create_session("s", "p").unwrap();
    let runtime = setup(dir.path(), &mut store, false);
    let store = Mutex::new(store);
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
    let supervisor = Supervisor::new(engine, telemetry());
    let mut events = vec![];
    let outcome = supervisor
        .run_tool_turn(
            &store,
            &runtime,
            "s",
            "read fixture",
            config(),
            &CancellationToken::default(),
            |_| Ok(()),
            |e| {
                events.push(e.event);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(outcome.assistant_text, "read complete");
    assert_eq!(
        events
            .iter()
            .filter(|e| **e == "generation_completed")
            .count(),
        1
    );
    let contexts = supervisor.engine().contexts.lock().unwrap();
    let history = format!("{:?}", contexts[1].tool_context());
    assert!(history.contains("actual fixture bytes"), "{history}");
    drop(contexts);
    let history = store.lock().unwrap().list_events("s").unwrap();
    let kinds: Vec<_> = history.iter().map(|e| e.kind).collect();
    for kind in [
        EventKind::ToolStarted,
        EventKind::ToolCompleted,
        EventKind::RunCompleted,
    ] {
        assert_eq!(kinds.iter().filter(|k| **k == kind).count(), 1);
    }
    drop(runtime);
    drop(store);
    let reopened = SessionStore::open(db).unwrap();
    assert_eq!(reopened.list_events("s").unwrap(), history);
}
#[test]
fn approval_denial_protocol_errors_and_call_limit_have_zero_writes() {
    for mode in ["deny", "invalid", "unknown", "limit"] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "p").unwrap();
        let runtime = setup(dir.path(), &mut store, false);
        let store = Mutex::new(store);
        let name = if mode == "unknown" {
            "missing.tool"
        } else {
            "fs.write"
        };
        let args = if mode == "invalid" {
            json!({"path":"marker.txt"})
        } else {
            json!({"path":"marker.txt","content":"once"})
        };
        let batch = if mode == "limit" {
            vec![proposal(name, args.clone()), proposal(name, args), done()]
        } else {
            vec![proposal(name, args), done()]
        };
        let supervisor = Supervisor::new(
            ScriptedEngine::new(vec![
                batch,
                vec![
                    GenerationEvent::TextDelta {
                        text: "finished".into(),
                    },
                    done(),
                ],
            ]),
            telemetry(),
        );
        let mut cfg = config();
        if mode == "limit" {
            cfg.max_calls = 1;
        }
        let result = supervisor.run_tool_turn(
            &store,
            &runtime,
            "s",
            "write",
            cfg,
            &CancellationToken::default(),
            |_| Ok(()),
            |e| {
                if e.event == "approval_requested" {
                    store.lock().unwrap().decide_tool(
                        "s",
                        e.body["call_id"].as_str().unwrap(),
                        false,
                    )?;
                }
                Ok(())
            },
        );
        assert_eq!(result.is_err(), mode == "limit");
        assert!(!dir.path().join("marker.txt").exists());
        assert!(
            !store
                .lock()
                .unwrap()
                .list_events("s")
                .unwrap()
                .iter()
                .any(|e| e.kind == EventKind::ToolStarted)
        );
    }
}
#[test]
fn explicit_approval_executes_real_write_once_and_duplicate_decision_fails() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "p").unwrap();
    let runtime = setup(dir.path(), &mut store, false);
    let store = Mutex::new(store);
    let supervisor = Supervisor::new(
        ScriptedEngine::new(vec![
            vec![
                proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
                done(),
            ],
            vec![
                GenerationEvent::TextDelta {
                    text: "done".into(),
                },
                done(),
            ],
        ]),
        telemetry(),
    );
    supervisor
        .run_tool_turn(
            &store,
            &runtime,
            "s",
            "write",
            config(),
            &CancellationToken::default(),
            |_| Ok(()),
            |e| {
                if e.event == "approval_requested" {
                    assert!(!dir.path().join("marker.txt").exists());
                    let id = e.body["call_id"].as_str().unwrap();
                    let mut guard = store.lock().unwrap();
                    guard.decide_tool("s", id, true)?;
                    assert!(guard.decide_tool("s", id, true).is_err());
                }
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("marker.txt")).unwrap(),
        "once"
    );
}

#[test]
fn integrated_hooks_veto_before_effect_and_annotate_original_result_once() {
    let bin_dir = tempfile::tempdir().unwrap();
    let bin = bin_dir.path().join("hook");
    assert!(
        std::process::Command::new("rustc")
            .args(["--edition=2024", "-o"])
            .arg(&bin)
            .arg(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/tool_hook_plugin.rs")
            )
            .status()
            .unwrap()
            .success()
    );
    for mode in ["allow", "veto", "after_failure"] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "p").unwrap();
        let mut runtime = setup(dir.path(), &mut store, false);
        runtime.plugins = Mutex::new(
            start_host(
                vec![
                    PluginConfig {
                        id: "first".into(),
                        version: "1.0.0".into(),
                        program: bin.clone(),
                        args: vec![
                            "first".into(),
                            mode.into(),
                            dir.path().join("hooks").to_str().unwrap().into(),
                        ],
                        cwd: dir.path().into(),
                        sandbox: sandbox::Sandbox::Trusted,
                    },
                    PluginConfig {
                        id: "second".into(),
                        version: "1.0.0".into(),
                        program: bin.clone(),
                        args: vec![
                            "second".into(),
                            "allow".into(),
                            dir.path().join("hooks").to_str().unwrap().into(),
                        ],
                        cwd: dir.path().into(),
                        sandbox: sandbox::Sandbox::Trusted,
                    },
                ],
                Limits {
                    message_bytes: 65536,
                    diagnostic_bytes: 4096,
                    annotation_bytes: 1024,
                    reason_bytes: 256,
                    timeout: Duration::from_millis(500),
                },
                Instant::now() + Duration::from_secs(5),
                &CancellationToken::default(),
            )
            .unwrap(),
        );
        let store = Mutex::new(store);
        let supervisor = Supervisor::new(
            ScriptedEngine::new(vec![
                vec![
                    proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
                    done(),
                ],
                vec![
                    GenerationEvent::TextDelta {
                        text: "done".into(),
                    },
                    done(),
                ],
            ]),
            telemetry(),
        );
        supervisor
            .run_tool_turn(
                &store,
                &runtime,
                "s",
                "write",
                config(),
                &CancellationToken::default(),
                |_| Ok(()),
                |e| {
                    if e.event == "approval_requested" {
                        store.lock().unwrap().decide_tool(
                            "s",
                            e.body["call_id"].as_str().unwrap(),
                            true,
                        )?;
                    }
                    Ok(())
                },
            )
            .unwrap();
        let log = std::fs::read_to_string(dir.path().join("hooks")).unwrap();
        let expected = if mode == "veto" {
            "first:initialize\nsecond:initialize\nfirst:before\n"
        } else {
            "first:initialize\nsecond:initialize\nfirst:before\nsecond:before\nfirst:after\nsecond:after\n"
        };
        assert_eq!(log, expected);
        assert_eq!(dir.path().join("marker.txt").exists(), mode != "veto");
        let history = store.lock().unwrap().list_events("s").unwrap();
        let receipts: Vec<_> = history
            .iter()
            .filter(|e| e.payload["transition"] == "HookReceipts")
            .collect();
        assert_eq!(receipts.len(), if mode == "veto" { 1 } else { 2 });
        if mode == "after_failure" {
            assert!(receipts[1].payload["receipts"][0]["diagnostic"].is_string());
            let completed = history
                .iter()
                .find(|e| e.kind == EventKind::ToolCompleted)
                .unwrap();
            assert_eq!(completed.payload["result"]["is_error"], false);
        }
    }
}

#[test]
fn pending_approval_cancel_releases_store_and_never_replays() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "p").unwrap();
    let runtime = setup(dir.path(), &mut store, false);
    let store = Mutex::new(store);
    let supervisor = Supervisor::new(
        ScriptedEngine::new(vec![vec![
            proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
            done(),
        ]]),
        telemetry(),
    );
    let cancel = CancellationToken::default();
    let mut terminals = 0;
    let result = supervisor.run_tool_turn(
        &store,
        &runtime,
        "s",
        "write",
        config(),
        &cancel,
        |_| Ok(()),
        |e| {
            assert!(store.try_lock().is_ok());
            if e.event == "approval_requested" {
                cancel.cancel();
            }
            if e.event == "request_failed" {
                terminals += 1;
            }
            Ok(())
        },
    );
    assert!(result.is_err());
    assert_eq!(terminals, 1);
    assert!(!dir.path().join("marker.txt").exists());
    assert!(
        store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .any(|e| e.kind == EventKind::ToolCompleted && e.payload["state"] == "Withdrawn")
    );
}

#[test]
fn canonical_loop_preserves_artifact_provenance_and_does_not_read_it_into_context() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("fixture.txt"), "x".repeat(1024)).unwrap();
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "p").unwrap();
    let runtime = setup(dir.path(), &mut store, false);
    let store = Mutex::new(store);
    let supervisor = Supervisor::new(
        ScriptedEngine::new(vec![
            vec![
                proposal(
                    "fs.read",
                    json!({"path":"fixture.txt","offset":0,"length":4096}),
                ),
                done(),
            ],
            vec![
                GenerationEvent::TextDelta {
                    text: "saved".into(),
                },
                done(),
            ],
        ]),
        telemetry(),
    );
    let mut cfg = config();
    cfg.inline_bytes = 16;
    supervisor
        .run_tool_turn(
            &store,
            &runtime,
            "s",
            "read",
            cfg,
            &CancellationToken::default(),
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
    let history = store.lock().unwrap().list_events("s").unwrap();
    let created = history
        .iter()
        .find(|e| e.kind == EventKind::ArtifactCreated)
        .unwrap();
    let completed = history
        .iter()
        .find(|e| e.kind == EventKind::ToolCompleted)
        .unwrap();
    assert!(completed.payload["result"]["artifact_ref"].is_string());
    assert_eq!(
        created.payload["artifact_ref"],
        completed.payload["result"]["artifact_ref"]
    );
    let contexts = supervisor.engine().contexts.lock().unwrap();
    let rendered = format!("{:?}", contexts[1].tool_context());
    assert!(rendered.contains("artifact_ref"));
    assert!(!rendered.contains(&"x".repeat(32)));
}
#[test]
fn subsequent_turn_after_reopen_does_not_reexecute_a_completed_effect() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    {
        let mut store = SessionStore::open(&db).unwrap();
        store.create_session("s", "p").unwrap();
        let runtime = setup(dir.path(), &mut store, false);
        let store = Mutex::new(store);
        let supervisor = Supervisor::new(
            ScriptedEngine::new(vec![
                vec![
                    proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
                    done(),
                ],
                vec![
                    GenerationEvent::TextDelta {
                        text: "done".into(),
                    },
                    done(),
                ],
            ]),
            telemetry(),
        );
        supervisor
            .run_tool_turn(
                &store,
                &runtime,
                "s",
                "write",
                config(),
                &CancellationToken::default(),
                |_| Ok(()),
                |e| {
                    if e.event == "approval_requested" {
                        store.lock().unwrap().decide_tool(
                            "s",
                            e.body["call_id"].as_str().unwrap(),
                            true,
                        )?;
                    }
                    Ok(())
                },
            )
            .unwrap();
    }
    let mut reopened = SessionStore::open(db).unwrap();
    let runtime = setup(dir.path(), &mut reopened, false);
    let store = Mutex::new(reopened);
    let supervisor = Supervisor::new(
        ScriptedEngine::new(vec![vec![
            GenerationEvent::TextDelta {
                text: "resumed".into(),
            },
            done(),
        ]]),
        telemetry(),
    );
    supervisor
        .run_tool_turn(
            &store,
            &runtime,
            "s",
            "continue",
            config(),
            &CancellationToken::default(),
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("marker.txt")).unwrap(),
        "once"
    );
    assert_eq!(
        store
            .lock()
            .unwrap()
            .list_events("s")
            .unwrap()
            .iter()
            .filter(|e| e.kind == EventKind::ToolStarted)
            .count(),
        1
    );
}

#[test]
fn turn_text_limit_is_enforced_before_streaming_across_rounds() {
    for over_limit in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "p").unwrap();
        let runtime = setup(dir.path(), &mut store, false);
        let store = Mutex::new(store);
        let supervisor = Supervisor::new(
            ScriptedEngine::new(vec![
                vec![
                    GenerationEvent::TextDelta { text: "ab".into() },
                    proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
                    done(),
                ],
                vec![
                    GenerationEvent::TextDelta {
                        text: if over_limit {
                            "éx".into()
                        } else {
                            "é".into()
                        },
                    },
                    done(),
                ],
            ]),
            telemetry(),
        );
        let mut cfg = config();
        cfg.round.max_text_bytes = 4;
        let mut streamed = String::new();
        let mut terminals = 0;
        let outcome = supervisor.run_tool_turn(
            &store,
            &runtime,
            "s",
            "write",
            cfg,
            &CancellationToken::default(),
            |_| Ok(()),
            |event| {
                match event.event {
                    "approval_requested" => store.lock().unwrap().decide_tool(
                        "s",
                        event.body["call_id"].as_str().unwrap(),
                        false,
                    )?,
                    "text_delta" => streamed.push_str(event.body["text"].as_str().unwrap()),
                    "generation_completed" | "request_failed" => terminals += 1,
                    _ => {}
                }
                Ok(())
            },
        );
        assert_eq!(outcome.is_err(), over_limit);
        assert_eq!(streamed, if over_limit { "ab" } else { "abé" });
        assert!(streamed.len() <= 4);
        assert_eq!(terminals, 1);
        assert!(!dir.path().join("marker.txt").exists());
    }
}

#[test]
fn waiting_for_shared_plugin_host_honors_cancel_and_deadline_without_effects() {
    for cancel_wait in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("db")).unwrap();
        store.create_session("s", "p").unwrap();
        let runtime = setup(dir.path(), &mut store, false);
        let store = Mutex::new(store);
        let supervisor = Supervisor::new(
            ScriptedEngine::new(vec![vec![
                proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
                done(),
            ]]),
            telemetry(),
        );
        let cancel = CancellationToken::default();
        let mut cfg = config();
        if !cancel_wait {
            cfg.round.deadline = Instant::now() + Duration::from_millis(500);
        }
        std::thread::scope(|scope| {
            let host_guard = runtime.plugins.lock().unwrap();
            let (admitted_tx, admitted_rx) = std::sync::mpsc::channel();
            let (finished_tx, finished_rx) = std::sync::mpsc::channel();
            let cancel_ref = &cancel;
            let supervisor_ref = &supervisor;
            let store_ref = &store;
            let runtime_ref = &runtime;
            scope.spawn(move || {
                let result = supervisor_ref.run_tool_turn(
                    store_ref,
                    runtime_ref,
                    "s",
                    "write",
                    cfg,
                    cancel_ref,
                    |_| Ok(()),
                    |event| {
                        if event.event == "approval_requested" {
                            let call = event.body["call_id"].as_str().unwrap();
                            store_ref.lock().unwrap().decide_tool("s", call, true)?;
                            admitted_tx.send(call.to_owned()).unwrap();
                        }
                        Ok(())
                    },
                );
                finished_tx.send(result).unwrap();
            });
            let call = admitted_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            let observation_deadline = Instant::now() + Duration::from_millis(250);
            let mut claimed = false;
            while Instant::now() < observation_deadline {
                if store
                    .lock()
                    .unwrap()
                    .tool_state("s", &call)
                    .unwrap()
                    .as_deref()
                    == Some("Claimed")
                {
                    claimed = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            if cancel_wait {
                cancel.cancel();
            }
            // The owning session keeps its host lock until after the other turn
            // must terminate. An unconditional Mutex::lock fails this assertion.
            let finished = finished_rx.recv_timeout(Duration::from_secs(2));
            drop(host_guard);
            assert!(claimed, "turn did not reach the shared host wait");
            assert!(
                finished.is_ok(),
                "turn remained blocked by another session's plugin host"
            );
            assert!(finished.unwrap().is_err());
        });
        assert!(!dir.path().join("marker.txt").exists());
        assert!(
            !store
                .lock()
                .unwrap()
                .list_events("s")
                .unwrap()
                .iter()
                .any(|event| event.kind == EventKind::ToolStarted)
        );
    }
}

#[test]
fn cancelled_after_hook_host_wait_records_warning_without_changing_executed_result() {
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("hook");
    assert!(
        std::process::Command::new("rustc")
            .args(["--edition=2024", "-o"])
            .arg(&bin)
            .arg(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/tool_hook_plugin.rs")
            )
            .status()
            .unwrap()
            .success()
    );
    let mut store = SessionStore::open(dir.path().join("db")).unwrap();
    store.create_session("s", "p").unwrap();
    let mut runtime = setup(dir.path(), &mut store, false);
    runtime.plugins = Mutex::new(
        start_host(
            vec![PluginConfig {
                id: "observer".into(),
                version: "1.0.0".into(),
                program: bin,
                args: vec![
                    "observer".into(),
                    "allow".into(),
                    dir.path().join("hooks").to_str().unwrap().into(),
                ],
                cwd: dir.path().into(),
                sandbox: sandbox::Sandbox::Trusted,
            }],
            Limits {
                message_bytes: 65536,
                diagnostic_bytes: 4096,
                annotation_bytes: 1024,
                reason_bytes: 256,
                timeout: Duration::from_secs(2),
            },
            Instant::now() + Duration::from_secs(5),
            &CancellationToken::default(),
        )
        .unwrap(),
    );
    let store = Mutex::new(store);
    let supervisor = Supervisor::new(
        ScriptedEngine::new(vec![vec![
            proposal("fs.write", json!({"path":"marker.txt","content":"once"})),
            done(),
        ]]),
        telemetry(),
    );
    let cancel = CancellationToken::default();
    std::thread::scope(|scope| {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let supervisor_ref = &supervisor;
        let store_ref = &store;
        let runtime_ref = &runtime;
        let cancel_ref = &cancel;
        scope.spawn(move || {
            let outcome = supervisor_ref.run_tool_turn(
                store_ref,
                runtime_ref,
                "s",
                "write",
                config(),
                cancel_ref,
                |_| Ok(()),
                |event| {
                    if event.event == "approval_requested" {
                        store_ref.lock().unwrap().decide_tool(
                            "s",
                            event.body["call_id"].as_str().unwrap(),
                            true,
                        )?;
                    }
                    if event.event == "tool_call_started" {
                        started_tx.send(()).unwrap();
                        continue_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    }
                    Ok(())
                },
            );
            finished_tx.send(outcome).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let host_guard = runtime.plugins.lock().unwrap();
        continue_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut completed = false;
        while Instant::now() < deadline {
            if store
                .lock()
                .unwrap()
                .list_events("s")
                .unwrap()
                .iter()
                .any(|event| event.kind == EventKind::ToolCompleted)
            {
                completed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        cancel.cancel();
        let outcome = finished_rx.recv_timeout(Duration::from_secs(2));
        drop(host_guard);
        assert!(completed, "real execution did not commit its result");
        assert!(
            outcome.is_ok(),
            "cancelled after-hook wait retained another session's host"
        );
        assert!(outcome.unwrap().is_err());
    });
    assert_eq!(
        std::fs::read_to_string(dir.path().join("marker.txt")).unwrap(),
        "once"
    );
    let history = store.lock().unwrap().list_events("s").unwrap();
    let result = history
        .iter()
        .find(|event| event.kind == EventKind::ToolCompleted)
        .unwrap();
    assert_eq!(result.payload["result"]["is_error"], false);
    let failure = history
        .iter()
        .rev()
        .find(|event| event.payload["transition"] == "HookReceipts")
        .unwrap();
    assert_eq!(failure.payload["receipts"][0]["plugin_id"], "observer");
    assert_eq!(failure.payload["receipts"][0]["hook"], "tool.after_execute");
    assert!(
        failure.payload["receipts"][0]["diagnostic"]
            .as_str()
            .unwrap()
            .contains("cancelled")
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("hooks")).unwrap(),
        "observer:initialize\nobserver:before\n"
    );
}
