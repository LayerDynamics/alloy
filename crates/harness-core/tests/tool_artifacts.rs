//! W3 artifact pipeline fixtures, not real builtin execution or W6 loop evidence.
use harness_core::{
    EventKind, SessionStore,
    artifact::{ArtifactLimits, ArtifactStore, MAX_ARTIFACT_BYTES},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    io::{Cursor, Read, Write},
    sync::Arc,
};
use tool_registry::{
    call::{ToolCall, ToolCallContext, ToolInvocation},
    definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
    registry::{ToolImplementation, ToolRegistry},
    result::ToolResult,
};

struct Never;
impl ToolImplementation for Never {
    fn execute(&self, _: &ToolInvocation) -> ToolResult {
        panic!("artifact fixture must not execute a tool")
    }
}
fn start(store: &mut SessionStore, call: &str) -> harness_core::store::StartedInvocation {
    use harness_core::policy::{DaemonPolicy, PolicyConfig, PolicyRule};
    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolDefinition {
                id: "fixture.output".into(),
                version: "1.0.0".into(),
                description: "Artifact fixture".into(),
                input_schema: json!({"type":"object","additionalProperties":false}),
                origin: ToolOrigin::Builtin,
                policy: ToolPolicy {
                    effect: ToolEffect::Read,
                    approval: ApprovalRequirement::Never,
                    timeout_ms: 1000,
                    max_output_bytes: MAX_ARTIFACT_BYTES,
                },
            },
            Arc::new(Never),
        )
        .unwrap();
    let policy = DaemonPolicy::new(PolicyConfig {
        rules: vec![PolicyRule {
            tool_id: "fixture.output".into(),
            allowed_arguments: vec![serde_json::Map::new()],
            require_approval: false,
            timeout_ms: 1000,
            max_output_bytes: MAX_ARTIFACT_BYTES,
        }],
    })
    .unwrap();
    store
        .admit_tool(
            &registry,
            &policy,
            ToolInvocation {
                context: ToolCallContext {
                    session_id: "s".into(),
                    turn_id: "t".into(),
                    generation_id: "g".into(),
                },
                call: ToolCall {
                    call_id: call.into(),
                    tool_id: "fixture.output".into(),
                    arguments: serde_json::Map::new(),
                },
            },
        )
        .unwrap();
    let claim = store.claim_tool("s", call).unwrap();
    store.start_tool(claim, &registry, &policy).unwrap()
}
fn fixture(limits: ArtifactLimits) -> (tempfile::TempDir, SessionStore, ArtifactStore) {
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    store.create_session("s", "fixture").unwrap();
    drop(start(&mut store, "c"));
    let artifacts = ArtifactStore::open(&mut store, dir.path().join("artifacts"), limits).unwrap();
    (dir, store, artifacts)
}
fn publish(
    store: &mut SessionStore,
    artifacts: &ArtifactStore,
    bytes: &[u8],
) -> harness_core::artifact::ArtifactMetadata {
    let mut writer = artifacts
        .begin(store, "s", "c", bytes.len() as u64)
        .unwrap();
    writer.copy_from(&mut Cursor::new(bytes)).unwrap();
    writer.finish().unwrap().commit(store).unwrap()
}
#[test]
fn oversized_tool_result_artifact_roundtrip() {
    let (dir, mut store, artifacts, receipt) = completion_tests::fixture_started();
    let bytes: Vec<u8> = (0..1_048_699).map(|n| (n % 251) as u8).collect();
    let metadata = publish(&mut store, &artifacts, &bytes);
    assert_eq!(metadata.byte_length, bytes.len() as u64);
    assert_eq!(metadata.sha256, format!("{:x}", Sha256::digest(&bytes)));
    assert_eq!(metadata.artifact_ref.len(), 37);
    assert_eq!(metadata.preview_encoding, "base64");
    assert!(metadata.preview_bytes.len() <= 5464);
    let result = ToolResult::Artifact(tool_registry::result::ArtifactToolResult {
        call_id: "c".into(),
        artifact_ref: metadata.artifact_ref.clone(),
        is_error: false,
    });
    let expected_result = serde_json::to_value(&result).unwrap();
    store.complete_tool(receipt, result).unwrap();
    assert_eq!(
        store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    let events = store.list_events("s").unwrap();
    let created: Vec<_> = events
        .iter()
        .filter(|e| e.kind == EventKind::ArtifactCreated)
        .collect();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].artifact_refs, json!([metadata.artifact_ref]));
    let completed: Vec<_> = events
        .iter()
        .filter(|e| e.kind == EventKind::ToolCompleted)
        .collect();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].payload["result"], expected_result);
    assert_eq!(completed[0].artifact_refs, json!([metadata.artifact_ref]));
    drop(artifacts);
    drop(store);
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    let artifacts = ArtifactStore::open(
        &mut store,
        dir.path().join("artifacts"),
        ArtifactLimits::default(),
    )
    .unwrap();
    let mut verified = artifacts
        .lookup(&store, "s", "c", &metadata.artifact_ref)
        .unwrap()
        .verify()
        .unwrap();
    assert_eq!(verified.metadata(), &metadata);
    let mut restored = Vec::new();
    verified.read_to_end(&mut restored).unwrap();
    assert_eq!(restored, bytes);
    let after = store.list_events("s").unwrap();
    assert_eq!(after, events);
    assert_eq!(
        store.tool_state("s", "c").unwrap().as_deref(),
        Some("Completed")
    );
    assert!(store.claim_tool("s", "c").is_err());
    let connection = rusqlite::Connection::open(store.path()).unwrap();
    let persisted: String = connection
        .query_row(
            "SELECT result_json FROM tool_intents WHERE session_id='s' AND call_id='c'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&persisted).unwrap(),
        expected_result
    );
    assert_eq!(
        after
            .iter()
            .filter(|e| e.kind == EventKind::ArtifactCreated)
            .count(),
        1
    );
}
#[test]
fn empty_and_exact_payloads_publish_but_one_over_never_registers() {
    let (_dir, mut store, artifacts) = fixture(ArtifactLimits {
        max_artifact_bytes: 64,
        ..ArtifactLimits::default()
    });
    let empty = publish(&mut store, &artifacts, b"");
    assert_eq!(empty.byte_length, 0);
    let exact = publish(&mut store, &artifacts, &[b'x'; 64]);
    assert_eq!(exact.byte_length, 64);
    let mut writer = artifacts.begin(&store, "s", "c", 64).unwrap();
    assert!(writer.copy_from(&mut Cursor::new([0u8; 65])).is_err());
    assert!(writer.finish().is_err());
    assert_eq!(
        store
            .list_events("s")
            .unwrap()
            .iter()
            .filter(|e| e.kind == EventKind::ArtifactCreated)
            .count(),
        2
    );
}
#[test]
fn frozen_exact_16mib_payload_is_supported() {
    let (_dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let mut writer = artifacts
        .begin(&store, "s", "c", MAX_ARTIFACT_BYTES)
        .unwrap();
    let block = [b'x'; 65536];
    for _ in 0..256 {
        writer.write_all(&block).unwrap();
    }
    let m = writer.finish().unwrap().commit(&mut store).unwrap();
    assert_eq!(m.byte_length, MAX_ARTIFACT_BYTES);
}
#[test]
fn utf8_preview_trims_only_incomplete_trailing_codepoint() {
    let (_dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let mut bytes = vec![b'a'; 4095];
    bytes.extend_from_slice("\u{20ac}".as_bytes());
    let m = publish(&mut store, &artifacts, &bytes);
    assert_eq!(m.preview_encoding, "utf8");
    assert_eq!(m.preview_bytes, "a".repeat(4095));
}
#[test]
fn binary_preview_uses_exact_bounded_base64() {
    let (_dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let m = publish(&mut store, &artifacts, &[0xff, 0x00, 0x01]);
    assert_eq!(m.preview_bytes, "/wAB");
    assert_eq!(m.preview_encoding, "base64");
}
#[test]
fn lookup_requires_exact_session_call_and_valid_reference() {
    let (_dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let m = publish(&mut store, &artifacts, b"secret");
    assert!(
        artifacts
            .lookup(&store, "other", "c", &m.artifact_ref)
            .is_err()
    );
    assert!(
        artifacts
            .lookup(&store, "s", "other", &m.artifact_ref)
            .is_err()
    );
    for reference in [
        "../secret",
        "/tmp/x",
        "art1_ABC",
        "art1_00000000000000000000000000000000",
    ] {
        assert!(artifacts.lookup(&store, "s", "c", reference).is_err());
    }
}
#[test]
fn corruption_and_missing_objects_fail_integrity_checks() {
    let (dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let m = publish(&mut store, &artifacts, b"original");
    let path = dir.path().join("artifacts/objects").join(&m.artifact_ref);
    std::fs::write(&path, b"tampered").unwrap();
    assert!(
        artifacts
            .lookup(&store, "s", "c", &m.artifact_ref)
            .unwrap()
            .verify()
            .is_err()
    );
    std::fs::remove_file(path).unwrap();
    assert!(
        artifacts
            .lookup(&store, "s", "c", &m.artifact_ref)
            .unwrap()
            .verify()
            .is_err()
    );
}
#[test]
fn writer_count_and_file_byte_reservations_are_shared() {
    let (_dir, store, artifacts) = fixture(ArtifactLimits {
        max_writers: 2,
        max_managed_bytes: 16,
        max_managed_files: 4,
        ..ArtifactLimits::default()
    });
    let first = artifacts.begin(&store, "s", "c", 4).unwrap();
    let cloned = artifacts.clone();
    let second = cloned.begin(&store, "s", "c", 4).unwrap();
    assert!(artifacts.begin(&store, "s", "c", 0).is_err());
    drop(first);
    drop(second);
    assert!(artifacts.begin(&store, "s", "c", 8).is_ok());
}
#[test]
fn partial_reader_failure_cannot_publish_a_truncated_result() {
    struct Broken(bool);
    impl Read for Broken {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            if self.0 {
                Err(std::io::Error::other("injected"))
            } else {
                self.0 = true;
                b[0] = 1;
                Ok(1)
            }
        }
    }
    let (_dir, store, artifacts) = fixture(ArtifactLimits::default());
    let mut writer = artifacts.begin(&store, "s", "c", 32).unwrap();
    assert!(writer.copy_from(&mut Broken(false)).is_err());
    assert!(writer.finish().is_err());
    assert!(
        store
            .list_events("s")
            .unwrap()
            .iter()
            .all(|e| e.kind != EventKind::ArtifactCreated)
    );
}
#[test]
fn metadata_event_failure_leaves_no_valid_reference_and_reconciles_orphan() {
    let (dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let connection = rusqlite::Connection::open(store.path()).unwrap();
    connection.execute_batch("CREATE TRIGGER reject_artifact BEFORE INSERT ON events WHEN NEW.kind='artifact.created' BEGIN SELECT RAISE(ABORT,'injected'); END;").unwrap();
    let mut writer = artifacts.begin(&store, "s", "c", 4).unwrap();
    writer.write_all(b"data").unwrap();
    assert!(writer.finish().unwrap().commit(&mut store).is_err());
    let count: i64 = connection
        .query_row("SELECT COUNT(*) FROM tool_artifacts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
    assert!(artifacts.begin(&store, "s", "c", 0).is_err());
    drop(connection);
    drop(artifacts);
    drop(store);
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    let _artifacts = ArtifactStore::open(
        &mut store,
        dir.path().join("artifacts"),
        ArtifactLimits::default(),
    )
    .unwrap();
    assert_eq!(
        std::fs::read_dir(dir.path().join("artifacts/objects"))
            .unwrap()
            .count(),
        0
    );
}
#[test]
fn live_writer_retains_daemon_lease_after_session_store_drop() {
    let (dir, store, artifacts) = fixture(ArtifactLimits::default());
    let writer = artifacts.begin(&store, "s", "c", 4).unwrap();
    drop(artifacts);
    drop(store);
    assert!(SessionStore::open(dir.path().join("events.db")).is_err());
    drop(writer);
    assert!(SessionStore::open(dir.path().join("events.db")).is_ok());
}
#[test]
fn proven_orphans_and_temporaries_are_removed_on_reopen() {
    let (dir, store, artifacts) = fixture(ArtifactLimits::default());
    drop(artifacts);
    drop(store);
    std::fs::write(
        dir.path()
            .join("artifacts/tmp/t_00000000000000000000000000000000"),
        b"partial",
    )
    .unwrap();
    std::fs::write(
        dir.path()
            .join("artifacts/objects/art1_00000000000000000000000000000000"),
        b"orphan",
    )
    .unwrap();
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    let _a = ArtifactStore::open(
        &mut store,
        dir.path().join("artifacts"),
        ArtifactLimits::default(),
    )
    .unwrap();
    assert_eq!(
        std::fs::read_dir(dir.path().join("artifacts/tmp"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        std::fs::read_dir(dir.path().join("artifacts/objects"))
            .unwrap()
            .count(),
        0
    );
}
#[test]
fn failed_integrity_proof_does_not_delete_any_orphan() {
    let (dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let m = publish(&mut store, &artifacts, b"valid");
    drop(artifacts);
    drop(store);
    std::fs::remove_file(dir.path().join("artifacts/objects").join(m.artifact_ref)).unwrap();
    let orphan = dir
        .path()
        .join("artifacts/tmp/t_00000000000000000000000000000000");
    std::fs::write(&orphan, b"keep").unwrap();
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    assert!(
        ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default()
        )
        .is_err()
    );
    assert!(orphan.exists());
}
#[test]
fn root_cannot_be_rebound_to_another_database_or_location() {
    let (dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    assert!(
        ArtifactStore::open(
            &mut store,
            dir.path().join("other-root"),
            ArtifactLimits::default()
        )
        .is_err()
    );
    let mut other = SessionStore::open(dir.path().join("other.db")).unwrap();
    assert!(
        ArtifactStore::open(
            &mut other,
            dir.path().join("artifacts"),
            ArtifactLimits::default()
        )
        .is_err()
    );
    drop(artifacts);
    drop(store);
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    assert!(
        ArtifactStore::open(
            &mut store,
            dir.path().join("new-root"),
            ArtifactLimits::default()
        )
        .is_err()
    );
}
#[test]
#[cfg(unix)]
fn root_and_object_symlinks_fail_closed() {
    use std::os::unix::fs::symlink;
    let (dir, mut store, artifacts) = fixture(ArtifactLimits::default());
    let m = publish(&mut store, &artifacts, b"original");
    let object = dir.path().join("artifacts/objects").join(&m.artifact_ref);
    std::fs::remove_file(&object).unwrap();
    let outside = dir.path().join("outside");
    std::fs::write(&outside, b"original").unwrap();
    symlink(&outside, &object).unwrap();
    assert!(
        artifacts
            .lookup(&store, "s", "c", &m.artifact_ref)
            .unwrap()
            .verify()
            .is_err()
    );
    drop(artifacts);
    drop(store);
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    assert!(
        ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default()
        )
        .is_err()
    );
    assert_eq!(std::fs::read(outside).unwrap(), b"original");
}

#[test]
#[cfg(unix)]
fn configured_root_symlink_is_rejected_before_publication() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    let target = dir.path().join("target");
    std::fs::create_dir(&target).unwrap();
    let alias = dir.path().join("alias");
    symlink(&target, &alias).unwrap();
    assert!(ArtifactStore::open(&mut store, &alias, ArtifactLimits::default()).is_err());
    assert_eq!(std::fs::read_dir(target).unwrap().count(), 0);
}
#[test]
fn metadata_event_disagreement_refuses_cleanup() {
    let (dir, mut store, a) = fixture(ArtifactLimits::default());
    let m = publish(&mut store, &a, b"keep");
    drop(a);
    drop(store);
    let connection = rusqlite::Connection::open(dir.path().join("events.db")).unwrap();
    connection
        .execute(
            "DELETE FROM tool_artifacts WHERE artifact_ref=?1",
            [&m.artifact_ref],
        )
        .unwrap();
    drop(connection);
    let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
    assert!(
        ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default()
        )
        .is_err()
    );
    assert!(
        dir.path()
            .join("artifacts/objects")
            .join(m.artifact_ref)
            .exists()
    );
}

// Step B: artifact registration -> canonical ToolResult -> durable completion.
mod completion_tests {
    use super::*;
    use harness_core::store::StartedInvocation;
    use tool_registry::result::{ArtifactToolResult, InlineToolResult};

    pub(super) fn fixture_started() -> (
        tempfile::TempDir,
        SessionStore,
        ArtifactStore,
        StartedInvocation,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let receipt = start(&mut store, "c");
        let artifacts = ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        (dir, store, artifacts, receipt)
    }
    fn artifact_result(call: &str, reference: &str, is_error: bool) -> ToolResult {
        ToolResult::Artifact(ArtifactToolResult {
            call_id: call.into(),
            artifact_ref: reference.into(),
            is_error,
        })
    }
    fn completed(store: &SessionStore) -> Vec<harness_core::store::EventRecord> {
        store
            .list_events("s")
            .unwrap()
            .into_iter()
            .filter(|e| e.kind == EventKind::ToolCompleted && e.payload["state"] == "Completed")
            .collect()
    }
    fn assert_uncompleted(store: &SessionStore, call: &str) {
        assert_eq!(
            store.tool_state("s", call).unwrap().as_deref(),
            Some("Started")
        );
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        let result: Option<String> = connection
            .query_row(
                "SELECT result_json FROM tool_intents WHERE session_id='s' AND call_id=?1",
                [call],
                |r| r.get(0),
            )
            .unwrap();
        assert!(result.is_none());
        assert!(completed(store).is_empty());
        assert!(
            store
                .list_events("s")
                .unwrap()
                .iter()
                .all(|e| e.kind != EventKind::ToolCompleted)
        );
    }
    #[test]
    fn artifact_completion_preserves_both_error_flags_and_exact_wire_fields() {
        for is_error in [false, true] {
            let (_dir, mut store, a, receipt) = fixture_started();
            let m = publish(&mut store, &a, b"result");
            let result = artifact_result("c", &m.artifact_ref, is_error);
            let expected = serde_json::to_value(&result).unwrap();
            store.complete_tool(receipt, result).unwrap();
            let events = completed(&store);
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].payload["result"], expected);
            assert_eq!(events[0].artifact_refs, json!([m.artifact_ref]));
            assert_eq!(expected.as_object().unwrap().len(), 3);
            assert_eq!(
                expected,
                json!({"call_id":"c","artifact_ref":m.artifact_ref,"is_error":is_error})
            );
            let connection = rusqlite::Connection::open(store.path()).unwrap();
            let saved: String = connection
                .query_row(
                    "SELECT result_json FROM tool_intents WHERE session_id='s' AND call_id='c'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&saved).unwrap(),
                expected
            );
            assert!(store.claim_tool("s", "c").is_err());
            assert!(a.begin(&store, "s", "c", 1).is_err());
        }
    }
    #[test]
    fn inline_completion_remains_wire_compatible_and_has_no_artifact_refs() {
        let (_dir, mut store, _a, receipt) = fixture_started();
        let result = ToolResult::Inline(InlineToolResult {
            call_id: "c".into(),
            content: json!({"text":"inline"}),
            is_error: false,
        });
        let expected = serde_json::to_value(&result).unwrap();
        store.complete_tool(receipt, result).unwrap();
        let events = completed(&store);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["result"], expected);
        assert_eq!(events[0].artifact_refs, json!([]));
    }
    #[test]
    fn artifact_from_another_call_cannot_complete_this_call() {
        let (_dir, mut store, a, old_receipt) = fixture_started();
        let m = publish(&mut store, &a, b"owned by c");
        drop(old_receipt);
        let receipt = start(&mut store, "other");
        assert!(
            store
                .complete_tool(receipt, artifact_result("other", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "other");
    }
    fn start_other_session(store: &mut SessionStore) -> StartedInvocation {
        use harness_core::policy::{DaemonPolicy, PolicyConfig, PolicyRule};
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolDefinition {
                    id: "fixture.output".into(),
                    version: "1.0.0".into(),
                    description: "other session".into(),
                    input_schema: json!({"type":"object","additionalProperties":false}),
                    origin: ToolOrigin::Builtin,
                    policy: ToolPolicy {
                        effect: ToolEffect::Read,
                        approval: ApprovalRequirement::Never,
                        timeout_ms: 1000,
                        max_output_bytes: MAX_ARTIFACT_BYTES,
                    },
                },
                Arc::new(Never),
            )
            .unwrap();
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "fixture.output".into(),
                allowed_arguments: vec![serde_json::Map::new()],
                require_approval: false,
                timeout_ms: 1000,
                max_output_bytes: MAX_ARTIFACT_BYTES,
            }],
        })
        .unwrap();
        store
            .admit_tool(
                &registry,
                &policy,
                ToolInvocation {
                    context: ToolCallContext {
                        session_id: "other-session".into(),
                        turn_id: "t".into(),
                        generation_id: "g".into(),
                    },
                    call: ToolCall {
                        call_id: "c".into(),
                        tool_id: "fixture.output".into(),
                        arguments: serde_json::Map::new(),
                    },
                },
            )
            .unwrap();
        let claim = store.claim_tool("other-session", "c").unwrap();
        store.start_tool(claim, &registry, &policy).unwrap()
    }
    #[test]
    fn artifact_from_another_session_cannot_complete_same_call_id() {
        let (_dir, mut store, a, old_receipt) = fixture_started();
        let m = publish(&mut store, &a, b"session s only");
        drop(old_receipt);
        store.create_session("other-session", "fixture").unwrap();
        let receipt = start_other_session(&mut store);
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_eq!(
            store.tool_state("other-session", "c").unwrap().as_deref(),
            Some("Started")
        );
        assert!(
            store
                .list_events("other-session")
                .unwrap()
                .iter()
                .all(|e| e.kind != EventKind::ToolCompleted)
        );
    }
    #[test]
    fn unknown_artifact_reference_is_rejected() {
        let (_dir, mut store, _a, receipt) = fixture_started();
        assert!(
            store
                .complete_tool(
                    receipt,
                    artifact_result("c", "art1_00000000000000000000000000000000", false)
                )
                .is_err()
        );
        assert_uncompleted(&store, "c");
    }
    #[test]
    fn malformed_artifact_reference_is_rejected_without_completion() {
        for reference in [
            "../object",
            "/tmp/object",
            "art1_ABC",
            "art1_0000000000000000000000000000000g",
            "",
        ] {
            let (_dir, mut store, _a, receipt) = fixture_started();
            assert!(
                store
                    .complete_tool(receipt, artifact_result("c", reference, false))
                    .is_err()
            );
            assert_uncompleted(&store, "c");
        }
    }
    #[test]
    fn published_but_unregistered_object_cannot_complete_a_tool() {
        let (dir, mut store, a, receipt) = fixture_started();
        let mut writer = a.begin(&store, "s", "c", 4).unwrap();
        writer.write_all(b"data").unwrap();
        let pending = writer.finish().unwrap();
        let reference = std::fs::read_dir(dir.path().join("artifacts/objects"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .file_name()
            .into_string()
            .unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &reference, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
        drop(pending);
    }
    #[test]
    fn result_call_must_match_consumed_start_receipt() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        assert!(
            store
                .complete_tool(
                    receipt,
                    artifact_result("wrong-call", &m.artifact_ref, false)
                )
                .is_err()
        );
        assert_uncompleted(&store, "c");
    }
    #[test]
    fn completion_rejects_altered_metadata_ownership() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        let mut altered = m.clone();
        altered.session_id = "forged-session".into();
        connection
            .execute(
                "UPDATE tool_artifacts SET metadata_json=?1 WHERE artifact_ref=?2",
                rusqlite::params![serde_json::to_string(&altered).unwrap(), m.artifact_ref],
            )
            .unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
    }
    #[test]
    fn artifact_without_creation_event_cannot_complete() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        connection
            .execute("DELETE FROM events WHERE kind='artifact.created'", [])
            .unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
    }
    #[test]
    fn duplicate_creation_event_is_rejected() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        store
            .append_event(
                "s",
                EventKind::ArtifactCreated,
                serde_json::to_value(&m).unwrap(),
                json!([m.artifact_ref]),
            )
            .unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
    }
    #[test]
    fn mismatched_creation_event_reference_is_rejected() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        connection
            .execute(
                "UPDATE events SET artifact_refs='[]' WHERE kind='artifact.created'",
                [],
            )
            .unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
    }
    #[test]
    fn wrong_epoch_or_binding_digest_cannot_complete() {
        for column in ["daemon_epoch", "binding_digest"] {
            let (_dir, mut store, a, receipt) = fixture_started();
            let m = publish(&mut store, &a, b"data");
            let connection = rusqlite::Connection::open(store.path()).unwrap();
            connection.execute(&format!("UPDATE tool_intents SET {column}='changed' WHERE session_id='s' AND call_id='c'"),[]).unwrap();
            assert!(
                store
                    .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                    .is_err()
            );
            assert_uncompleted(&store, "c");
        }
    }
    #[test]
    fn terminal_projection_rejects_late_receipt_without_second_completion() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        connection
            .execute(
                "UPDATE tool_intents SET state='Completed' WHERE session_id='s' AND call_id='c'",
                [],
            )
            .unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert!(completed(&store).is_empty());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Completed")
        );
    }
    #[test]
    fn completion_update_failure_leaves_result_and_event_uncommitted() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        connection.execute_batch("CREATE TRIGGER fail_completion_update BEFORE UPDATE ON tool_intents WHEN NEW.state='Completed' BEGIN SELECT RAISE(ABORT,'injected'); END;").unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
        assert!(store.claim_tool("s", "c").is_err());
    }
    #[test]
    fn completion_event_failure_rolls_back_and_recovers_without_reexecution() {
        let (dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"preserved");
        let connection = rusqlite::Connection::open(store.path()).unwrap();
        connection.execute_batch("CREATE TRIGGER fail_completion_event BEFORE INSERT ON events WHEN NEW.kind='tool.completed' BEGIN SELECT RAISE(ABORT,'injected'); END;").unwrap();
        assert!(
            store
                .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
                .is_err()
        );
        assert_uncompleted(&store, "c");
        assert!(store.claim_tool("s", "c").is_err());
        // Remove only the injected fixture trigger so recovery can append evidence.
        connection
            .execute_batch("DROP TRIGGER fail_completion_event;")
            .unwrap();
        drop(connection);
        drop(a);
        drop(store);
        let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
        let a = ArtifactStore::open(
            &mut store,
            dir.path().join("artifacts"),
            ArtifactLimits::default(),
        )
        .unwrap();
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("OutcomeUnknown")
        );
        assert!(store.claim_tool("s", "c").is_err());
        assert!(completed(&store).is_empty());
        let mut verified = a
            .lookup(&store, "s", "c", &m.artifact_ref)
            .unwrap()
            .verify()
            .unwrap();
        let mut bytes = Vec::new();
        verified.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"preserved");
    }
    #[test]
    fn completion_keeps_metadata_out_of_the_result_envelope() {
        let (_dir, mut store, a, receipt) = fixture_started();
        let m = publish(&mut store, &a, b"data");
        store
            .complete_tool(receipt, artifact_result("c", &m.artifact_ref, false))
            .unwrap();
        let events = completed(&store);
        let result = events[0].payload["result"].as_object().unwrap();
        for field in [
            "preview_bytes",
            "preview_encoding",
            "byte_length",
            "sha256",
            "contract_version",
        ] {
            assert!(!result.contains_key(field));
        }
        assert_eq!(result.len(), 3);
    }
}
