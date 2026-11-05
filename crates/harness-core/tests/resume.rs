//! Named acceptance: `session_resume_restart` (W4 store-reopen stand-in).

use harness_core::{EventKind, SessionStatus, SessionStore};
use tempfile::tempdir;

#[test]
fn session_resume_restart() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sessions.sqlite");

    {
        let mut store = SessionStore::open(&path).expect("open");
        let session = store
            .create_session("resume-1", "profile-coder")
            .expect("create");
        assert_eq!(session.status, SessionStatus::Active);

        store
            .append_event(
                "resume-1",
                EventKind::UserMessage,
                serde_json::json!({ "content": "remember this" }),
                serde_json::json!([]),
            )
            .expect("user");
        store
            .append_event(
                "resume-1",
                EventKind::AssistantMessage,
                serde_json::json!({ "content": "remembered" }),
                serde_json::json!([]),
            )
            .expect("assistant");
        // Drop the store handle — W4 stand-in for a process restart.
    }

    let mut store = SessionStore::open(&path).expect("reopen after restart");
    let session = store
        .get_session("resume-1")
        .expect("get")
        .expect("session must survive reopen");
    assert_eq!(session.profile_id, "profile-coder");
    assert_eq!(session.status, SessionStatus::Active);

    let events = store.list_events("resume-1").expect("history");
    assert_eq!(events.len(), 3);
    assert_eq!(events[1].payload["content"], "remember this");
    assert_eq!(events[2].payload["content"], "remembered");

    // Resume continues with gapless seq after reopen.
    let next = store
        .append_event(
            "resume-1",
            EventKind::UserMessage,
            serde_json::json!({ "content": "and continue" }),
            serde_json::json!([]),
        )
        .expect("continue");
    assert_eq!(next.seq, 4);
}
