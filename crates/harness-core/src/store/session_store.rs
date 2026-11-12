//! B.4 append-only SQLite history, schema-v2 projections and exclusive ownership.

use crate::error::{HarnessError, HarnessResult};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema version understood by this crate. Readers reject newer versions.
pub const SCHEMA_VERSION: i64 = 4;

/// Spec version recorded in the meta row for forward-only migrations.
pub const SPEC_VERSION: &str = "SPEC-1/B.4/v1";

/// Closed B.4 event kind enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// Session was created.
    SessionCreated,
    /// User message appended.
    UserMessage,
    /// Assistant message appended.
    AssistantMessage,
    /// Tool call requested (unused in Phase 1 supervisor).
    ToolRequested,
    /// Tool call approved (unused in Phase 1 supervisor).
    ToolApproved,
    /// Tool execution started (unused in Phase 1 supervisor).
    ToolStarted,
    /// Tool execution completed (unused in Phase 1 supervisor).
    ToolCompleted,
    /// Skill loaded (unused in Phase 1 supervisor).
    SkillLoaded,
    /// Artifact created (unused in Phase 1 supervisor).
    ArtifactCreated,
    /// Context compacted (unused in Phase 1 supervisor).
    ContextCompacted,
    /// Model/profile changed (unused in Phase 1 supervisor).
    ModelChanged,
    /// Run started.
    RunStarted,
    /// Run paused.
    RunPaused,
    /// Run completed.
    RunCompleted,
}

impl EventKind {
    /// Wire / SQLite kind string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SessionCreated => "session.created",
            Self::UserMessage => "user.message",
            Self::AssistantMessage => "assistant.message",
            Self::ToolRequested => "tool.requested",
            Self::ToolApproved => "tool.approved",
            Self::ToolStarted => "tool.started",
            Self::ToolCompleted => "tool.completed",
            Self::SkillLoaded => "skill.loaded",
            Self::ArtifactCreated => "artifact.created",
            Self::ContextCompacted => "context.compacted",
            Self::ModelChanged => "model.changed",
            Self::RunStarted => "run.started",
            Self::RunPaused => "run.paused",
            Self::RunCompleted => "run.completed",
        }
    }

    fn parse(value: &str) -> HarnessResult<Self> {
        match value {
            "session.created" => Ok(Self::SessionCreated),
            "user.message" => Ok(Self::UserMessage),
            "assistant.message" => Ok(Self::AssistantMessage),
            "tool.requested" => Ok(Self::ToolRequested),
            "tool.approved" => Ok(Self::ToolApproved),
            "tool.started" => Ok(Self::ToolStarted),
            "tool.completed" => Ok(Self::ToolCompleted),
            "skill.loaded" => Ok(Self::SkillLoaded),
            "artifact.created" => Ok(Self::ArtifactCreated),
            "context.compacted" => Ok(Self::ContextCompacted),
            "model.changed" => Ok(Self::ModelChanged),
            "run.started" => Ok(Self::RunStarted),
            "run.paused" => Ok(Self::RunPaused),
            "run.completed" => Ok(Self::RunCompleted),
            other => Err(HarnessError::Invalid(format!(
                "unknown event kind: {other}"
            ))),
        }
    }
}

/// Lifecycle status for a session row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// Session accepts new turns.
    Active,
    /// Session is paused.
    Paused,
    /// Session finished.
    Completed,
}

impl SessionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Completed => "completed",
        }
    }

    fn parse(value: &str) -> HarnessResult<Self> {
        match value {
            "active" => Ok(Self::Active),
            "paused" => Ok(Self::Paused),
            "completed" => Ok(Self::Completed),
            other => Err(HarnessError::Invalid(format!(
                "unknown session status: {other}"
            ))),
        }
    }
}

/// Durable session metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRecord {
    /// Stable session identifier.
    pub session_id: String,
    /// Lifecycle status.
    pub status: SessionStatus,
    /// Creation time (unix epoch ms).
    pub created_ts: i64,
    /// Last update time (unix epoch ms).
    pub updated_ts: i64,
    /// Engine model profile id for this session.
    pub profile_id: String,
}

/// One append-only event row.
#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    /// Monotonic row id.
    pub id: i64,
    /// Owning session.
    pub session_id: String,
    /// Per-session gapless sequence.
    pub seq: i64,
    /// Event timestamp (unix epoch ms).
    pub ts: i64,
    /// Event kind.
    pub kind: EventKind,
    /// Kind-specific JSON payload.
    pub payload: Value,
    /// Artifact references JSON array.
    pub artifact_refs: Value,
}

/// SQLite-backed B.4 session store.
pub struct SessionStore {
    path: PathBuf,
    pub(crate) conn: Connection,
    _ownership: std::sync::Arc<durable::StoreOwnership>,
    epoch: String,
    live: std::collections::HashMap<(String, String), durable::LiveIntent>,
    claim_deadlines: std::collections::HashMap<(String, String), std::time::Instant>,
    pub(crate) artifact_state: Option<std::sync::Arc<crate::artifact::Shared>>,
}

impl SessionStore {
    /// Open/create a schema-v2 store with exclusive mutating daemon ownership.
    pub fn open(path: impl AsRef<Path>) -> HarnessResult<Self> {
        let requested = path.as_ref();
        let name = requested
            .file_name()
            .ok_or_else(|| HarnessError::Invalid("database file name required".into()))?;
        let parent = requested
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)
            .map_err(|e| HarnessError::Store(format!("create parent dir: {e}")))?;
        let parent = std::fs::canonicalize(parent)
            .map_err(|e| HarnessError::Store(format!("canonicalize parent: {e}")))?;
        let path = parent.join(name);
        if std::fs::symlink_metadata(&path)
            .is_ok_and(|m| m.file_type().is_symlink() || !m.is_file())
        {
            return Err(HarnessError::Invalid(
                "database must be a regular non-symlink file".into(),
            ));
        }
        let ownership = durable::owned_file(&path)?;
        let mut conn = Connection::open(&path)?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        durable::migrate(&mut conn)?;
        let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(HarnessError::Store("WAL unavailable".into()));
        }
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let epoch: String =
            conn.query_row("SELECT lower(hex(randomblob(16)))", [], |r| r.get(0))?;
        let mut store = Self {
            path,
            conn,
            _ownership: std::sync::Arc::new(ownership),
            epoch,
            live: std::collections::HashMap::new(),
            claim_deadlines: std::collections::HashMap::new(),
            artifact_state: None,
        };
        store.recover_tools()?;
        Ok(store)
    }

    /// Filesystem path of the SQLite database.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Create a new active session and append `session.created`.
    pub fn create_session(
        &mut self,
        session_id: impl Into<String>,
        profile_id: impl Into<String>,
    ) -> HarnessResult<SessionRecord> {
        let session_id = session_id.into();
        let profile_id = profile_id.into();
        if session_id.is_empty() {
            return Err(HarnessError::Invalid("session_id must be non-empty".into()));
        }
        if profile_id.is_empty() {
            return Err(HarnessError::Invalid("profile_id must be non-empty".into()));
        }
        let ts = now_ms()?;
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO sessions (session_id, status, created_ts, updated_ts, profile_id)
             VALUES (?1, ?2, ?3, ?3, ?4)",
            params![session_id, SessionStatus::Active.as_str(), ts, profile_id],
        )?;
        append_event_in_tx(
            &tx,
            &session_id,
            ts,
            EventKind::SessionCreated,
            serde_json::json!({ "profile_id": profile_id }),
            serde_json::json!([]),
        )?;
        tx.commit()?;
        Ok(SessionRecord {
            session_id,
            status: SessionStatus::Active,
            created_ts: ts,
            updated_ts: ts,
            profile_id,
        })
    }

    /// Update session lifecycle status and touch `updated_ts`.
    pub fn set_status(
        &mut self,
        session_id: &str,
        status: SessionStatus,
    ) -> HarnessResult<SessionRecord> {
        let ts = now_ms()?;
        let tx = self.conn.transaction()?;
        let updated = tx.execute(
            "UPDATE sessions SET status = ?1, updated_ts = ?2 WHERE session_id = ?3",
            params![status.as_str(), ts, session_id],
        )?;
        if updated == 0 {
            return Err(HarnessError::Invalid(format!(
                "unknown session_id: {session_id}"
            )));
        }
        if !matches!(status, SessionStatus::Active) {
            durable::withdraw_in_tx(&tx, session_id)?;
        }
        if matches!(status, SessionStatus::Paused) {
            append_event_in_tx(
                &tx,
                session_id,
                ts,
                EventKind::RunPaused,
                serde_json::json!({}),
                serde_json::json!([]),
            )?;
        }
        tx.commit()?;
        if !matches!(status, SessionStatus::Active) {
            self.live.retain(|(s, _), _| s != session_id);
            self.claim_deadlines.retain(|(s, _), _| s != session_id);
        }
        self.get_session(session_id)?
            .ok_or_else(|| HarnessError::Store(format!("session vanished: {session_id}")))
    }

    /// Count sessions currently marked active.
    pub fn count_active_sessions(&self) -> HarnessResult<u64> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE status = ?1",
            params![SessionStatus::Active.as_str()],
            |row| row.get(0),
        )?;
        u64::try_from(count).map_err(|_| HarnessError::Store("session count overflow".into()))
    }

    /// Return every session id currently stored.
    pub fn list_session_ids(&self) -> HarnessResult<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT session_id FROM sessions")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }

    /// Load a session row.
    pub fn get_session(&self, session_id: &str) -> HarnessResult<Option<SessionRecord>> {
        self.conn
            .query_row(
                "SELECT session_id, status, created_ts, updated_ts, profile_id
                 FROM sessions WHERE session_id = ?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?
            .map(|(session_id, status, created_ts, updated_ts, profile_id)| {
                Ok(SessionRecord {
                    session_id,
                    status: SessionStatus::parse(&status)?,
                    created_ts,
                    updated_ts,
                    profile_id,
                })
            })
            .transpose()
    }

    /// Append one event; assigns the next gapless per-session `seq`.
    pub fn append_event(
        &mut self,
        session_id: &str,
        kind: EventKind,
        payload: Value,
        artifact_refs: Value,
    ) -> HarnessResult<EventRecord> {
        if kind == EventKind::ContextCompacted {
            return Err(HarnessError::Invalid(
                "checkpoint events must use the validated transaction".into(),
            ));
        }
        if !artifact_refs.is_array() {
            return Err(HarnessError::Invalid(
                "artifact_refs must be a JSON array".into(),
            ));
        }
        let ts = now_ms()?;
        let tx = self.conn.transaction()?;
        let exists: bool = tx
            .query_row(
                "SELECT 1 FROM sessions WHERE session_id = ?1",
                params![session_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if !exists {
            return Err(HarnessError::Invalid(format!(
                "unknown session_id: {session_id}"
            )));
        }
        let record = append_event_in_tx(&tx, session_id, ts, kind, payload, artifact_refs)?;
        tx.execute(
            "UPDATE sessions SET updated_ts = ?1 WHERE session_id = ?2",
            params![ts, session_id],
        )?;
        tx.commit()?;
        Ok(record)
    }

    /// Atomically commit a validated checkpoint against an unchanged session history.
    /// `covered_through` is the final event replaced in model-visible context;
    /// original rows remain intact. A changed history or inactive session aborts.
    pub(crate) fn commit_checkpoint(
        &mut self,
        session_id: &str,
        expected_last_seq: i64,
        covered_through: i64,
        checkpoint: &crate::context::Checkpoint,
    ) -> HarnessResult<EventRecord> {
        let checkpoint_json = serde_json::to_string(checkpoint)?;
        crate::context::Checkpoint::parse_json(&checkpoint_json).map_err(HarnessError::Invalid)?;
        if covered_through < 1 || covered_through > expected_last_seq {
            return Err(HarnessError::Invalid(
                "invalid checkpoint history boundary".into(),
            ));
        }
        let ts = now_ms()?;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let state: Option<String> = tx
            .query_row(
                "SELECT status FROM sessions WHERE session_id=?1",
                [session_id],
                |r| r.get(0),
            )
            .optional()?;
        if state.as_deref() != Some("active") {
            return Err(HarnessError::Invalid(
                "checkpoint requires an active session".into(),
            ));
        }
        let last: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq),0) FROM events WHERE session_id=?1",
            [session_id],
            |r| r.get(0),
        )?;
        if last != expected_last_seq {
            return Err(HarnessError::Invalid(
                "session history changed during compaction".into(),
            ));
        }
        let boundary_event: Option<(String, String)> = tx
            .query_row(
                "SELECT kind,payload_json FROM events WHERE session_id=?1 AND seq=?2",
                params![session_id, covered_through],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let successful = match boundary_event {
            Some((kind, payload)) if kind == "run.completed" => {
                let payload: Value = serde_json::from_str(&payload)?;
                payload.get("status").and_then(Value::as_str) != Some("failed")
            }
            _ => false,
        };
        if !successful {
            return Err(HarnessError::Invalid(
                "checkpoint boundary must end a completed turn".into(),
            ));
        }
        let pending: i64 = tx.query_row(
            "SELECT COUNT(*) FROM tool_intents WHERE session_id=?1 AND state IN ('AwaitingApproval','Ready','Claimed','Started')",
            [session_id], |r| r.get(0),
        )?;
        if pending != 0 {
            return Err(HarnessError::Invalid(
                "cannot compact with an unresolved tool call".into(),
            ));
        }
        let prior: Option<(i64, String)> = tx.query_row(
            "SELECT seq,payload_json FROM events WHERE session_id=?1 AND kind='context.compacted' ORDER BY seq DESC LIMIT 1",
            [session_id], |r| Ok((r.get(0)?, r.get(1)?)),
        ).optional()?;
        let mut prior_boundary = 0;
        let mut provenance = Vec::new();
        if let Some((seq, prior)) = prior {
            let value: Value = serde_json::from_str(&prior)?;
            let boundary = value
                .get("covered_through")
                .and_then(Value::as_i64)
                .ok_or_else(|| HarnessError::Invalid("invalid prior checkpoint boundary".into()))?;
            if covered_through <= boundary {
                return Err(HarnessError::Invalid(
                    "checkpoint must advance its history boundary".into(),
                ));
            }
            prior_boundary = boundary;
            provenance.push(serde_json::json!({"seq":seq,"kind":"context.compacted","trust":"untrusted_generated_summary"}));
        }
        {
            let mut statement = tx.prepare("SELECT seq,kind FROM events WHERE session_id=?1 AND seq>?2 AND seq<=?3 ORDER BY seq")?;
            let rows = statement.query_map(
                params![session_id, prior_boundary, covered_through],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )?;
            for row in rows {
                let (seq, kind) = row?;
                let trust = match kind.as_str() {
                    "user.message" => "user_supplied",
                    "tool.completed" | "artifact.created" => "untrusted_tool_data",
                    "assistant.message" | "tool.requested" => "untrusted_generated_data",
                    _ => continue,
                };
                provenance.push(serde_json::json!({"seq":seq,"kind":kind,"trust":trust}));
            }
        }
        // Checkpoint references can also be ordinary paths or human-readable
        // identifiers. Only validated durable artifact IDs become event links.
        let mut artifact_links = Vec::new();
        for reference in &checkpoint.artifact_references {
            if reference.starts_with("art1_") {
                let owned: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM tool_artifacts WHERE artifact_ref=?1 AND session_id=?2)",
                    params![reference, session_id],
                    |r| r.get(0),
                )?;
                if !owned {
                    return Err(HarnessError::Invalid(format!(
                        "checkpoint artifact reference is not owned by this session: {reference}"
                    )));
                }
                artifact_links.push(reference);
            }
        }
        let record = append_event_in_tx(
            &tx,
            session_id,
            ts,
            EventKind::ContextCompacted,
            serde_json::json!({"covered_through":covered_through,"checkpoint":checkpoint,"source_events":provenance,"trust":"untrusted_generated_summary"}),
            serde_json::to_value(artifact_links)?,
        )?;
        tx.execute(
            "UPDATE sessions SET updated_ts=?1 WHERE session_id=?2",
            params![ts, session_id],
        )?;
        #[cfg(test)]
        checkpoint_crash_test::exit_at("before");
        tx.commit()?;
        #[cfg(test)]
        checkpoint_crash_test::exit_at("after");
        Ok(record)
    }

    /// Return all events for a session ordered by `seq` ascending.
    pub fn list_events(&self, session_id: &str) -> HarnessResult<Vec<EventRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, seq, ts, kind, payload_json, artifact_refs
             FROM events WHERE session_id = ?1 ORDER BY seq ASC",
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
            ))
        })?;

        let mut events = Vec::new();
        for row in rows {
            let (id, session_id, seq, ts, kind, payload_json, artifact_refs) = row?;
            events.push(EventRecord {
                id,
                session_id,
                seq,
                ts,
                kind: EventKind::parse(&kind)?,
                payload: serde_json::from_str(&payload_json)?,
                artifact_refs: serde_json::from_str(&artifact_refs)?,
            });
        }
        Ok(events)
    }
}

fn append_event_in_tx(
    tx: &rusqlite::Transaction<'_>,
    session_id: &str,
    ts: i64,
    kind: EventKind,
    payload: Value,
    artifact_refs: Value,
) -> HarnessResult<EventRecord> {
    let next_seq: i64 = tx.query_row(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE session_id = ?1",
        params![session_id],
        |row| row.get(0),
    )?;
    let payload_json = serde_json::to_string(&payload)?;
    let artifact_refs_json = serde_json::to_string(&artifact_refs)?;
    tx.execute(
        "INSERT INTO events (session_id, seq, ts, kind, payload_json, artifact_refs)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            session_id,
            next_seq,
            ts,
            kind.as_str(),
            payload_json,
            artifact_refs_json
        ],
    )?;
    let id = tx.last_insert_rowid();
    Ok(EventRecord {
        id,
        session_id: session_id.to_owned(),
        seq: next_seq,
        ts,
        kind,
        payload,
        artifact_refs,
    })
}

fn now_ms() -> HarnessResult<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| HarnessError::Store(format!("system clock: {error}")))?;
    i64::try_from(duration.as_millis())
        .map_err(|_| HarnessError::Store("timestamp overflow".into()))
}

#[path = "durable/mod.rs"]
mod durable;
pub use durable::{AuthorizedInvocation, IntentAdmission, StartedInvocation};

impl SessionStore {
    pub(crate) fn artifact_lease(&self) -> std::sync::Arc<dyn Send + Sync> {
        self._ownership.clone()
    }
}

pub(crate) fn commit_artifact(
    store: &mut SessionStore,
    metadata: &crate::artifact::ArtifactMetadata,
    json: &str,
    epoch: &str,
) -> HarnessResult<()> {
    let tx = store
        .conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let owner: Option<(String, String)> = tx
        .query_row(
            "SELECT state,daemon_epoch FROM tool_intents WHERE session_id=?1 AND call_id=?2",
            params![metadata.session_id, metadata.call_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    if owner
        .as_ref()
        .map(|(state, owner)| (state.as_str(), owner.as_str()))
        != Some(("Started", epoch))
        || epoch != store.epoch
    {
        return Err(HarnessError::Invalid(
            "artifact origin is not a current Started invocation".into(),
        ));
    }
    tx.execute("INSERT INTO tool_artifacts(artifact_ref,session_id,call_id,metadata_json) VALUES(?1,?2,?3,?4)",params![metadata.artifact_ref,metadata.session_id,metadata.call_id,json])?;
    append_event_in_tx(
        &tx,
        &metadata.session_id,
        metadata.created_at_unix_ms,
        EventKind::ArtifactCreated,
        serde_json::to_value(metadata)?,
        serde_json::json!([metadata.artifact_ref]),
    )?;
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod checkpoint_crash_test {
    use super::*;
    use crate::context::{Checkpoint, ContextBuilder};

    pub(super) fn exit_at(stage: &str) {
        if std::env::var("ALLOY_CHECKPOINT_CRASH_STAGE")
            .ok()
            .as_deref()
            == Some(stage)
        {
            std::process::exit(86);
        }
    }
    fn checkpoint() -> Checkpoint {
        Checkpoint {
            checkpoint_version: 1,
            goal: "goal".into(),
            constraints: vec!["retain".into()],
            decisions: vec![],
            completed_work: vec![],
            open_tasks: vec![],
            artifact_references: vec![],
            critical_identifiers: vec![],
            failures_and_lessons: vec![],
            next_action: "next".into(),
        }
    }
    #[test]
    fn checkpoint_rejects_unowned_durable_artifact_and_preserves_history() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = SessionStore::open(dir.path().join("db.sqlite")).unwrap();
        db.create_session("s", "p").unwrap();
        let boundary = db
            .append_event(
                "s",
                EventKind::RunCompleted,
                serde_json::json!({}),
                serde_json::json!([]),
            )
            .unwrap()
            .seq;
        let before = db.list_events("s").unwrap();
        let mut checkpoint = checkpoint();
        checkpoint
            .artifact_references
            .push(format!("art1_{}", "a".repeat(32)));
        let error = db
            .commit_checkpoint("s", boundary, boundary, &checkpoint)
            .unwrap_err();
        assert!(error.to_string().contains("not owned by this session"));
        assert_eq!(db.list_events("s").unwrap(), before);
        db.create_session("other", "p").unwrap();
        db.conn.execute(
            "INSERT INTO tool_artifacts(artifact_ref,session_id,call_id,metadata_json) VALUES(?1,'other','call','{}')",
            [&checkpoint.artifact_references[0]],
        ).unwrap();
        let error = db
            .commit_checkpoint("s", boundary, boundary, &checkpoint)
            .unwrap_err();
        assert!(error.to_string().contains("not owned by this session"));
        assert_eq!(db.list_events("s").unwrap(), before);
    }
    #[test]
    fn checkpoint_records_conservative_source_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = SessionStore::open(dir.path().join("db.sqlite")).unwrap();
        db.create_session("s", "p").unwrap();
        db.append_event(
            "s",
            EventKind::UserMessage,
            serde_json::json!({"content":"goal"}),
            serde_json::json!([]),
        )
        .unwrap();
        let tool_seq = db
            .append_event(
                "s",
                EventKind::ToolCompleted,
                serde_json::json!({"content":"untrusted instructions"}),
                serde_json::json!([]),
            )
            .unwrap()
            .seq;
        let boundary = db
            .append_event(
                "s",
                EventKind::RunCompleted,
                serde_json::json!({}),
                serde_json::json!([]),
            )
            .unwrap()
            .seq;
        let event = db
            .commit_checkpoint("s", boundary, boundary, &checkpoint())
            .unwrap();
        assert_eq!(event.payload["trust"], "untrusted_generated_summary");
        assert_eq!(event.payload["source_events"][0]["trust"], "user_supplied");
        assert_eq!(
            event.payload["source_events"][1]["trust"],
            "untrusted_tool_data"
        );
        assert_eq!(event.payload["source_events"][1]["seq"], tool_seq);
    }
    #[test]
    fn schema_v3_checkpoint_migrates_without_rewriting_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.sqlite");
        let mut db = SessionStore::open(&path).unwrap();
        db.create_session("s", "p").unwrap();
        let boundary = db
            .append_event(
                "s",
                EventKind::RunCompleted,
                serde_json::json!({}),
                serde_json::json!([]),
            )
            .unwrap()
            .seq;
        db.commit_checkpoint("s", boundary, boundary, &checkpoint())
            .unwrap();
        db.conn.execute("UPDATE events SET payload_json=json_remove(payload_json,'$.source_events','$.trust') WHERE kind='context.compacted'", []).unwrap();
        db.conn
            .execute("UPDATE meta SET schema_version=3", [])
            .unwrap();
        let before = db.list_events("s").unwrap();
        drop(db);
        let reopened = SessionStore::open(&path).unwrap();
        assert_eq!(reopened.list_events("s").unwrap(), before);
        let version: i64 = reopened
            .conn
            .query_row("SELECT schema_version FROM meta", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        assert!(ContextBuilder::build_snapshot("r", "p", &before, None).is_ok());
    }
    #[test]
    fn checkpoint_crash_child() {
        let Ok(path) = std::env::var("ALLOY_CHECKPOINT_CRASH_DB") else {
            return;
        };
        let mut db = SessionStore::open(path).unwrap();
        db.commit_checkpoint("s", 4, 3, &checkpoint()).unwrap();
        panic!("injected exit was not reached");
    }
    #[test]
    fn checkpoint_process_exit_before_and_after_commit_recovers_once() {
        for stage in ["before", "after"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.sqlite");
            let mut db = SessionStore::open(&path).unwrap();
            db.create_session("s", "p").unwrap();
            db.append_event(
                "s",
                EventKind::UserMessage,
                serde_json::json!({"content":"old"}),
                serde_json::json!([]),
            )
            .unwrap();
            db.append_event(
                "s",
                EventKind::RunCompleted,
                serde_json::json!({}),
                serde_json::json!([]),
            )
            .unwrap();
            db.append_event(
                "s",
                EventKind::UserMessage,
                serde_json::json!({"content":"new"}),
                serde_json::json!([]),
            )
            .unwrap();
            let original = db.list_events("s").unwrap();
            drop(db);
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "store::session_store::checkpoint_crash_test::checkpoint_crash_child",
                ])
                .env("ALLOY_CHECKPOINT_CRASH_STAGE", stage)
                .env("ALLOY_CHECKPOINT_CRASH_DB", &path)
                .status()
                .unwrap();
            assert_eq!(status.code(), Some(86));
            let mut db = SessionStore::open(&path).unwrap();
            let after = db.list_events("s").unwrap();
            assert_eq!(&after[..original.len()], original.as_slice());
            if stage == "before" {
                assert_eq!(after.len(), original.len());
                db.commit_checkpoint("s", 4, 3, &checkpoint()).unwrap();
            } else {
                assert_eq!(after.len(), original.len() + 1);
                assert!(db.commit_checkpoint("s", 4, 3, &checkpoint()).is_err());
            }
            let events = db.list_events("s").unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| e.kind == EventKind::ContextCompacted)
                    .count(),
                1
            );
            let messages = ContextBuilder::messages_from_events(&events).unwrap();
            assert!(messages[0].content.contains("retain"));
            assert_eq!(messages[1].content, "new");
        }
    }
}
