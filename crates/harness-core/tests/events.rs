//! Named acceptance: `events_append_only`.

use harness_core::{EventKind, SessionStore};
use tempfile::tempdir;

#[test]
fn events_append_only() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sessions.sqlite");

    let mut store = SessionStore::open(&path).expect("open");
    store
        .create_session("sess-1", "profile-local")
        .expect("create");

    let first = store
        .append_event(
            "sess-1",
            EventKind::UserMessage,
            serde_json::json!({ "content": "hello" }),
            serde_json::json!([]),
        )
        .expect("user");
    let second = store
        .append_event(
            "sess-1",
            EventKind::AssistantMessage,
            serde_json::json!({ "content": "hi" }),
            serde_json::json!([]),
        )
        .expect("assistant");

    assert_eq!(first.seq, 2, "session.created is seq 1");
    assert_eq!(second.seq, 3);
    assert!(second.id > first.id, "row ids must be monotonic");

    let events = store.list_events("sess-1").expect("list");
    let seqs: Vec<i64> = events.iter().map(|event| event.seq).collect();
    assert_eq!(seqs, vec![1, 2, 3], "per-session seq must be gapless");
    assert_eq!(events[0].kind, EventKind::SessionCreated);
    assert_eq!(events[1].kind, EventKind::UserMessage);
    assert_eq!(events[2].kind, EventKind::AssistantMessage);

    drop(store);

    let reopened = SessionStore::open(&path).expect("reopen");
    let again = reopened.list_events("sess-1").expect("list after reopen");
    assert_eq!(again.len(), 3);
    assert_eq!(again[1].payload["content"], "hello");
    assert_eq!(again[2].payload["content"], "hi");
    assert_eq!(again[0].id, events[0].id);
    assert_eq!(again[2].id, events[2].id);
}
