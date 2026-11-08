//! B.4 append-only SQLite history, schema-v2 projections and exclusive ownership.

use crate::error::{HarnessError, HarnessResult};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema version understood by this crate. Readers reject newer versions.
pub const SCHEMA_VERSION: i64 = 2;

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
    conn: Connection,
    _ownership: durable::StoreOwnership,
    epoch: String,
    live: std::collections::HashMap<(String, String), durable::LiveIntent>,
    claim_deadlines: std::collections::HashMap<(String, String), std::time::Instant>,
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
            _ownership: ownership,
            epoch,
            live: std::collections::HashMap::new(),
            claim_deadlines: std::collections::HashMap::new(),
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

mod durable;
pub use durable::{AuthorizedInvocation, IntentAdmission, StartedInvocation};
