//! Durable W3 authorization. No tool implementation is executed here.
use super::*;
use crate::{
    approval::{CONTRACT_VERSION, InvocationBinding},
    policy::{DaemonPolicy, PolicyDecision},
};
use std::{
    fs::File,
    time::{Duration, Instant},
};
use tool_registry::{call::ToolInvocation, registry::ToolRegistry};

/// Single-use pre-start ownership. Dropping this value never restores Ready.
#[derive(Debug)]
pub struct AuthorizedInvocation {
    binding: InvocationBinding,
}
impl AuthorizedInvocation {
    /// Immutable exact invocation for before hooks; not permission to execute.
    pub fn binding(&self) -> &InvocationBinding {
        &self.binding
    }
}
/// Single-use durable start receipt. W4 may attempt only this exact invocation.
///
/// A start receipt cannot be cloned:
/// ```compile_fail
/// use harness_core::store::StartedInvocation;
/// fn clone_receipt(receipt: &StartedInvocation) -> StartedInvocation {
///     Clone::clone(receipt)
/// }
/// ```
/// Completion consumes it; a second completion cannot reuse the same receipt:
/// ```compile_fail
/// use harness_core::store::{SessionStore, StartedInvocation};
/// use tool_registry::result::ToolResult;
/// fn reuse(store: &mut SessionStore, receipt: StartedInvocation, result: ToolResult) {
///     store.complete_tool(receipt, result.clone()).unwrap();
///     store.complete_tool(receipt, result).unwrap();
/// }
/// ```
/// Start receipts cannot be constructed by deserializing client input:
/// ```compile_fail
/// use harness_core::store::StartedInvocation;
/// let _: StartedInvocation = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug)]
pub struct StartedInvocation {
    binding: InvocationBinding,
}
impl StartedInvocation {
    /// Exact invocation whose tool.started transaction committed.
    pub fn binding(&self) -> &InvocationBinding {
        &self.binding
    }
}
/// Result of durable intent admission, never execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentAdmission {
    /// Explicit human decision required.
    AwaitingApproval,
    /// May be atomically claimed, subject to live checks.
    Ready,
    /// Policy denied and a terminal outcome was recorded.
    Rejected,
}
pub(super) struct LiveIntent {
    binding: InvocationBinding,
    deadline: Instant,
}

fn invalid(message: impl Into<String>) -> HarnessError {
    HarnessError::Invalid(message.into())
}
fn field<'a>(binding: &'a InvocationBinding, key: &str) -> HarnessResult<&'a str> {
    binding.snapshot()[key]
        .as_str()
        .ok_or_else(|| invalid(format!("missing binding {key}")))
}
fn key(binding: &InvocationBinding) -> (String, String) {
    let i = binding.invocation();
    (i.context.session_id.clone(), i.call.call_id.clone())
}
fn active(tx: &rusqlite::Transaction<'_>, session: &str) -> HarnessResult<()> {
    let status: Option<String> = tx
        .query_row(
            "SELECT status FROM sessions WHERE session_id=?1",
            [session],
            |r| r.get(0),
        )
        .optional()?;
    if status.as_deref() != Some("active") {
        return Err(invalid("session is not active"));
    }
    Ok(())
}
fn state(tx: &rusqlite::Transaction<'_>, session: &str, call: &str) -> HarnessResult<String> {
    tx.query_row(
        "SELECT state FROM tool_intents WHERE session_id=?1 AND call_id=?2",
        params![session, call],
        |r| r.get(0),
    )
    .optional()?
    .ok_or_else(|| invalid("unknown session/call"))
}
fn event(
    tx: &rusqlite::Transaction<'_>,
    session: &str,
    call: &str,
    kind: EventKind,
    payload: Value,
) -> HarnessResult<()> {
    let mut payload = payload
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("event object required"))?;
    payload.insert(
        "contract_version".into(),
        Value::String(CONTRACT_VERSION.into()),
    );
    payload.insert("call_id".into(), Value::String(call.into()));
    append_event_in_tx(
        tx,
        session,
        now_ms()?,
        kind,
        Value::Object(payload),
        serde_json::json!([]),
    )?;
    Ok(())
}
fn terminal(
    tx: &rusqlite::Transaction<'_>,
    session: &str,
    call: &str,
    next: &str,
    reason: &str,
) -> HarnessResult<()> {
    tx.execute(
        "UPDATE tool_intents SET state=?3, reason=?4 WHERE session_id=?1 AND call_id=?2",
        params![session, call, next, reason],
    )?;
    event(
        tx,
        session,
        call,
        EventKind::ToolCompleted,
        serde_json::json!({"state":next,"reason":reason,"is_error":true,"outcome_known": next != "OutcomeUnknown"}),
    )
}

// Validate only committed database provenance inside the completion transaction.
// File verification remains a separate lock-free ArtifactStore read operation.
fn committed_artifact_for_completion(
    tx: &rusqlite::Transaction<'_>,
    binding: &InvocationBinding,
    reference: &str,
) -> HarnessResult<()> {
    use crate::artifact::{ArtifactMetadata, MAX_ARTIFACT_BYTES, MAX_PREVIEW_BYTES};
    let valid_hex = |bytes: &[u8]| {
        bytes
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
    };
    if reference.len() != 37
        || !reference.starts_with("art1_")
        || !valid_hex(&reference.as_bytes()[5..])
    {
        return Err(invalid("invalid artifact reference"));
    }
    let invocation = binding.invocation();
    let session = &invocation.context.session_id;
    let call = &invocation.call.call_id;
    let json: Option<String> = tx.query_row(
        "SELECT metadata_json FROM tool_artifacts WHERE artifact_ref=?1 AND session_id=?2 AND call_id=?3",
        params![reference,session,call], |r| r.get(0),
    ).optional()?;
    let json = json.ok_or_else(|| invalid("artifact is not committed for this session/call"))?;
    if json.len() > crate::approval::MAX_CANONICAL_BINDING_BYTES {
        return Err(invalid(
            "artifact metadata exceeds storage envelope ceiling",
        ));
    }
    let metadata: ArtifactMetadata = serde_json::from_str(&json)?;
    let bound_max = binding.snapshot()["effective_max_output_bytes"]
        .as_u64()
        .ok_or_else(|| invalid("binding lacks output limit"))?;
    let preview_valid = match metadata.preview_encoding.as_str() {
        "utf8" => metadata.preview_bytes.len() <= MAX_PREVIEW_BYTES,
        "base64" => metadata.preview_bytes.len() <= 5464 && metadata.preview_bytes.is_ascii(),
        _ => false,
    };
    if metadata.artifact_ref != reference
        || &metadata.session_id != session
        || &metadata.call_id != call
        || metadata.contract_version != CONTRACT_VERSION
        || metadata.byte_length > MAX_ARTIFACT_BYTES
        || metadata.byte_length > bound_max
        || metadata.created_at_unix_ms < 0
        || metadata.sha256.len() != 64
        || !valid_hex(metadata.sha256.as_bytes())
        || !preview_valid
    {
        return Err(invalid(
            "artifact metadata does not match invocation contract",
        ));
    }
    let expected = serde_json::to_value(metadata)?;
    let mut statement = tx.prepare(
        "SELECT payload_json,artifact_refs FROM events WHERE session_id=?1 AND kind='artifact.created' ORDER BY seq",
    )?;
    let mut rows = statement.query([session])?;
    let mut matches = 0usize;
    let mut count = 0usize;
    while let Some(row) = rows.next()? {
        count += 1;
        if count > 4096 {
            return Err(invalid(
                "artifact creation history exceeds frozen file ceiling",
            ));
        }
        let payload: String = row.get(0)?;
        if payload.len() > crate::approval::MAX_CANONICAL_BINDING_BYTES {
            return Err(invalid("artifact creation payload exceeds ceiling"));
        }
        let payload: Value = serde_json::from_str(&payload)?;
        if payload["artifact_ref"].as_str() != Some(reference) {
            continue;
        }
        let refs: String = row.get(1)?;
        if refs.len() > crate::approval::MAX_CANONICAL_BINDING_BYTES {
            return Err(invalid("artifact creation references exceed ceiling"));
        }
        let refs: Value = serde_json::from_str(&refs)?;
        if payload != expected || refs != serde_json::json!([reference]) {
            return Err(invalid("artifact metadata/creation event mismatch"));
        }
        matches += 1;
    }
    if matches != 1 {
        return Err(invalid("artifact has no unique committed creation event"));
    }
    Ok(())
}

impl SessionStore {
    /// Current exclusively-owned daemon epoch. Not a client-controlled identity.
    pub fn daemon_epoch(&self) -> &str {
        &self.epoch
    }

    /// Resolve, validate, evaluate, and atomically persist a unique intent.
    /// Approval notifications may be published only after this returns success.
    pub fn admit_tool(
        &mut self,
        registry: &ToolRegistry,
        policy: &DaemonPolicy,
        invocation: ToolInvocation,
    ) -> HarnessResult<IntentAdmission> {
        self.expire_tool_intents()?;
        let session = invocation.context.session_id.clone();
        let call = invocation.call.call_id.clone();
        let created = now_ms()?;
        let clock = Instant::now();
        let decision = policy
            .evaluate(registry, invocation.clone(), &self.epoch, created)
            .map_err(|e| invalid(format!("policy preparation: {e}")))?;
        let (admission, binding, reason) = match decision {
            PolicyDecision::Deny(reason) => (
                IntentAdmission::Rejected,
                None,
                Some(format!("policy:{reason:?}")),
            ),
            PolicyDecision::AllowWithoutPrompt(b) => (IntentAdmission::Ready, Some(*b), None),
            PolicyDecision::RequireApproval(b) => {
                (IntentAdmission::AwaitingApproval, Some(*b), None)
            }
        };
        let label = match admission {
            IntentAdmission::Rejected => "Rejected",
            IntentAdmission::Ready => "Ready",
            IntentAdmission::AwaitingApproval => "AwaitingApproval",
        };
        let snapshot = match &binding {
            Some(b) => String::from_utf8(b.canonical_bytes().to_vec())
                .map_err(|_| invalid("binding encoding"))?,
            None => serde_json::to_string(&invocation)?,
        };
        if snapshot.len() > crate::approval::MAX_CANONICAL_BINDING_BYTES {
            return Err(invalid("intent exceeds binding ceiling"));
        }
        let digest = binding.as_ref().map(|b| b.digest());
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        active(&tx, &session)?;
        if binding.is_some() {
            let (all, local): (i64,i64) = tx.query_row(
                "SELECT COUNT(*), COALESCE(SUM(session_id=?1),0) FROM tool_intents WHERE state IN ('AwaitingApproval','Ready','Claimed')", [&session], |r| Ok((r.get(0)?,r.get(1)?)))?;
            if all >= 256 || local >= 32 {
                return Err(invalid("pending intent capacity exhausted"));
            }
        }
        tx.execute("INSERT INTO tool_intents(session_id,call_id,daemon_epoch,state,binding_json,binding_digest,created_ts,reason) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![session,call,self.epoch,label,snapshot,digest,created,reason])?;
        event(
            &tx,
            &session,
            &call,
            EventKind::ToolRequested,
            serde_json::json!({"state":label,"binding":binding.as_ref().map(|b| b.snapshot()),"request":invocation,"reason":reason}),
        )?;
        if admission == IntentAdmission::Rejected {
            terminal(
                &tx,
                &session,
                &call,
                "Rejected",
                reason.as_deref().unwrap_or("policy_denied"),
            )?;
        }
        #[cfg(test)]
        failure_recovery::checkpoint("intent.before_commit");
        tx.commit()?;
        #[cfg(test)]
        failure_recovery::checkpoint("intent.after_commit");
        if let Some(binding) = binding {
            self.live.insert(
                (session, call),
                LiveIntent {
                    binding,
                    deadline: clock + Duration::from_millis(300_000),
                },
            );
        }
        Ok(admission)
    }

    /// First valid human decision wins. A duplicate never adds an approval event.
    pub fn decide_tool(&mut self, session: &str, call: &str, allow: bool) -> HarnessResult<()> {
        self.expire_tool_intents()?;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        active(&tx, session)?;
        if state(&tx, session, call)? != "AwaitingApproval" {
            return Err(invalid("approval is not awaiting a decision"));
        }
        if !self.live.contains_key(&(session.into(), call.into())) {
            return Err(invalid("no live epoch authority"));
        }
        tx.execute("UPDATE tool_intents SET state=?3, decision=?4 WHERE session_id=?1 AND call_id=?2 AND daemon_epoch=?5", params![session,call,if allow {"Ready"} else {"Denied"},allow,self.epoch])?;
        event(
            &tx,
            session,
            call,
            EventKind::ToolApproved,
            serde_json::json!({"allow":allow}),
        )?;
        if !allow {
            terminal(&tx, session, call, "Denied", "human_denied")?;
        }
        #[cfg(test)]
        failure_recovery::checkpoint("decision.before_commit");
        tx.commit()?;
        #[cfg(test)]
        failure_recovery::checkpoint("decision.after_commit");
        if !allow {
            self.live.remove(&(session.into(), call.into()));
        }
        Ok(())
    }

    /// Atomically consume Ready. The returned handle cannot be cloned or decoded.
    pub fn claim_tool(&mut self, session: &str, call: &str) -> HarnessResult<AuthorizedInvocation> {
        self.expire_tool_intents()?;
        let k = (session.to_owned(), call.to_owned());
        if !self.live.contains_key(&k) {
            return Err(invalid("no unclaimed live binding"));
        }
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        active(&tx, session)?;
        if state(&tx, session, call)? != "Ready" {
            return Err(invalid("intent is not Ready"));
        }
        if tx.execute("UPDATE tool_intents SET state='Claimed' WHERE session_id=?1 AND call_id=?2 AND state='Ready' AND daemon_epoch=?3", params![session,call,self.epoch])? != 1 { return Err(invalid("claim lost")); }
        event(
            &tx,
            session,
            call,
            EventKind::ToolRequested,
            serde_json::json!({"transition":"Claimed"}),
        )?;
        #[cfg(test)]
        failure_recovery::checkpoint("claim.before_commit");
        tx.commit()?;
        #[cfg(test)]
        failure_recovery::checkpoint("claim.after_commit");
        let live = self
            .live
            .remove(&k)
            .ok_or_else(|| invalid("live binding vanished"))?;
        self.claim_deadlines.insert(k, live.deadline);
        Ok(AuthorizedInvocation {
            binding: live.binding,
        })
    }

    /// Revalidate the live policy/registration after hooks and commit tool.started.
    /// No lock is held while the caller runs hooks. A failed return permits no effect.
    pub fn start_tool(
        &mut self,
        handle: AuthorizedInvocation,
        registry: &ToolRegistry,
        policy: &DaemonPolicy,
    ) -> HarnessResult<StartedInvocation> {
        self.expire_tool_intents()?;
        let b = &handle.binding;
        let (session, call) = key(b);
        let created = b.snapshot()["created_at_unix_ms"]
            .as_i64()
            .ok_or_else(|| invalid("missing creation time"))?;
        let current = policy.evaluate(registry, b.invocation().clone(), &self.epoch, created);
        let matches = match current {
            Ok(PolicyDecision::AllowWithoutPrompt(now) | PolicyDecision::RequireApproval(now)) => {
                now.digest() == b.digest()
            }
            _ => false,
        };
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let status = state(&tx, &session, &call)?;
        if status != "Claimed" {
            return Err(invalid("intent is not Claimed"));
        }
        let owner: (String,String) = tx.query_row("SELECT daemon_epoch,binding_digest FROM tool_intents WHERE session_id=?1 AND call_id=?2",params![session,call],|r| Ok((r.get(0)?,r.get(1)?)))?;
        let session_active = active(&tx, &session).is_ok();
        let deadline_ok = self
            .claim_deadlines
            .get(&(session.clone(), call.clone()))
            .is_some_and(|d| Instant::now() < *d);
        if !matches
            || !session_active
            || !deadline_ok
            || owner.0 != self.epoch
            || owner.1 != b.digest()
            || field(b, "daemon_epoch")? != self.epoch
        {
            terminal(
                &tx,
                &session,
                &call,
                "Rejected",
                "final_authorization_failed",
            )?;
            tx.commit()?;
            self.claim_deadlines.remove(&(session, call));
            return Err(invalid("final authorization failed"));
        }
        tx.execute(
            "UPDATE tool_intents SET state='Started' WHERE session_id=?1 AND call_id=?2",
            params![session, call],
        )?;
        event(
            &tx,
            &session,
            &call,
            EventKind::ToolStarted,
            serde_json::json!({"binding_digest":b.digest()}),
        )?;
        #[cfg(test)]
        failure_recovery::checkpoint("start.before_commit");
        tx.commit()?;
        #[cfg(test)]
        failure_recovery::checkpoint("start.after_commit");
        self.claim_deadlines.remove(&(session, call));
        Ok(StartedInvocation {
            binding: handle.binding,
        })
    }

    /// Reject a claimed invocation after a hook veto/failure; no start is recorded.
    pub fn reject_tool(&mut self, handle: AuthorizedInvocation, reason: &str) -> HarnessResult<()> {
        let (session, call) = key(&handle.binding);
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        if state(&tx, &session, &call)? != "Claimed" {
            return Err(invalid("intent is not Claimed"));
        }
        let owner: (String,String) = tx.query_row("SELECT daemon_epoch,binding_digest FROM tool_intents WHERE session_id=?1 AND call_id=?2",params![session,call],|r|Ok((r.get(0)?,r.get(1)?)))?;
        if owner.0 != self.epoch || owner.1 != handle.binding.digest() {
            return Err(invalid("wrong claim ownership"));
        }
        terminal(&tx, &session, &call, "Rejected", reason)?;
        tx.commit()?;
        self.claim_deadlines.remove(&(session, call));
        Ok(())
    }

    /// Persist one canonical inline or committed artifact-backed result.
    /// Artifact validation here is bounded metadata/creation-event validation,
    /// not filesystem streaming. Use ArtifactStore lookup/verify outside store
    /// locks when consuming bytes. Failure consumes the receipt but never grants
    /// another execution; unresolved Started is conservatively recovered unknown.
    pub fn complete_tool(
        &mut self,
        handle: StartedInvocation,
        result: tool_registry::result::ToolResult,
    ) -> HarnessResult<()> {
        use tool_registry::result::ToolResult;
        let (session, call) = key(&handle.binding);
        if result.call_id() != call {
            return Err(invalid("result call identity mismatch"));
        }
        let artifact_ref = match &result {
            ToolResult::Inline(_) => None,
            ToolResult::Artifact(artifact) => Some(artifact.artifact_ref.clone()),
        };
        let value = serde_json::to_value(&result)?;
        let result_json = serde_json::to_string(&value)?;
        if result_json.len() > crate::approval::MAX_CANONICAL_BINDING_BYTES {
            return Err(invalid("tool result exceeds storage envelope ceiling"));
        }
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        if state(&tx, &session, &call)? != "Started" {
            return Err(invalid("intent is not Started"));
        }
        let owner: (String, String) = tx.query_row(
            "SELECT daemon_epoch,binding_digest FROM tool_intents WHERE session_id=?1 AND call_id=?2",
            params![session,call], |r| Ok((r.get(0)?,r.get(1)?)),
        )?;
        if owner.0 != self.epoch
            || field(&handle.binding, "daemon_epoch")? != self.epoch
            || owner.1 != handle.binding.digest()
        {
            return Err(invalid("completion receipt epoch or binding mismatch"));
        }
        let refs = match artifact_ref.as_deref() {
            Some(reference) => {
                committed_artifact_for_completion(&tx, &handle.binding, reference)?;
                serde_json::json!([reference])
            }
            None => serde_json::json!([]),
        };
        let updated = tx.execute(
            "UPDATE tool_intents SET state='Completed',result_json=?3 WHERE session_id=?1 AND call_id=?2 AND state='Started' AND daemon_epoch=?4 AND binding_digest=?5",
            params![session,call,result_json,self.epoch,handle.binding.digest()],
        )?;
        if updated != 1 {
            return Err(invalid("completion ownership was lost"));
        }
        append_event_in_tx(
            &tx,
            &session,
            now_ms()?,
            EventKind::ToolCompleted,
            serde_json::json!({"contract_version":CONTRACT_VERSION,"call_id":call,"state":"Completed","result":value}),
            refs,
        )?;
        #[cfg(test)]
        failure_recovery::checkpoint("complete.before_commit");
        tx.commit()?;
        #[cfg(test)]
        failure_recovery::checkpoint("complete.after_commit");
        Ok(())
    }

    /// Withdraw pre-start authority without claiming a Started effect did not run.
    pub fn withdraw_tools(&mut self, session: &str, reason: &str) -> HarnessResult<usize> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let calls = {
            let mut stmt = tx.prepare("SELECT call_id FROM tool_intents WHERE session_id=?1 AND state IN ('AwaitingApproval','Ready','Claimed')")?;
            let rows = stmt.query_map([session], |r| r.get::<_, String>(0))?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        for call in &calls {
            terminal(&tx, session, call, "Withdrawn", reason)?;
        }
        tx.commit()?;
        self.live.retain(|(s, _), _| s != session);
        self.claim_deadlines.retain(|(s, _), _| s != session);
        Ok(calls.len())
    }

    /// Sweep monotonic deadlines, including claims currently held by callers.
    pub fn expire_tool_intents(&mut self) -> HarnessResult<usize> {
        let now = Instant::now();
        let mut expired: Vec<_> = self
            .live
            .iter()
            .filter(|(_, v)| now >= v.deadline)
            .map(|(k, _)| k.clone())
            .collect();
        expired.extend(
            self.claim_deadlines
                .iter()
                .filter(|(_, d)| now >= **d)
                .map(|(k, _)| k.clone()),
        );
        if expired.is_empty() {
            return Ok(0);
        }
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut count = 0;
        for (session, call) in &expired {
            if matches!(
                state(&tx, session, call)?.as_str(),
                "AwaitingApproval" | "Ready" | "Claimed"
            ) {
                terminal(&tx, session, call, "Expired", "authorization_expired")?;
                count += 1;
            }
        }
        tx.commit()?;
        for k in expired {
            self.live.remove(&k);
            self.claim_deadlines.remove(&k);
        }
        Ok(count)
    }

    /// Read a projection label; possession of this string grants no authority.
    pub fn tool_state(&self, session: &str, call: &str) -> HarnessResult<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT state FROM tool_intents WHERE session_id=?1 AND call_id=?2",
                params![session, call],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub(super) fn recover_tools(&mut self) -> HarnessResult<()> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rows = {
            let mut stmt = tx.prepare("SELECT session_id,call_id,state FROM tool_intents WHERE state IN ('AwaitingApproval','Ready','Claimed','Started')")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        for (session, call, old) in rows {
            terminal(
                &tx,
                &session,
                &call,
                if old == "Started" {
                    "OutcomeUnknown"
                } else {
                    "Withdrawn"
                },
                "daemon_restarted",
            )?;
        }
        tx.execute("UPDATE runtime_metadata SET daemon_epoch=?1", [&self.epoch])?;
        tx.commit()?;
        Ok(())
    }
}

// Owns the logical daemon lock independently of duplicated OS descriptors.
// SessionStore declares this field after conn, so SQLite closes before unlock.
pub(super) struct StoreOwnership {
    file: File,
}
impl Drop for StoreOwnership {
    fn drop(&mut self) {
        // Explicitly release even if a duplicated/inherited descriptor is alive.
        // Drop cannot report errors; closing this handle remains the fallback.
        let _ = self.file.unlock();
    }
}

pub(super) fn owned_file(path: &Path) -> HarnessResult<StoreOwnership> {
    // Ownership must never lock SQLite's database, WAL, or SHM inode.
    // The trusted daemon-owned directory must not be modified by another actor.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match std::fs::symlink_metadata(path) {
            Ok(meta) if !meta.is_file() || meta.file_type().is_symlink() || meta.nlink() != 1 => {
                return Err(HarnessError::Store(
                    "database must be a regular single-link file".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(HarnessError::Store(format!(
                    "inspect database ownership path: {error}"
                )));
            }
        }
    }
    let mut name = path.as_os_str().to_os_string();
    name.push(".alloy-owner.lock");
    let lock_path = PathBuf::from(name);
    match std::fs::symlink_metadata(&lock_path) {
        Ok(meta) if !meta.is_file() || meta.file_type().is_symlink() => {
            return Err(HarnessError::Store(
                "ownership sidecar must be a regular non-symlink file".into(),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(HarnessError::Store(format!(
                "inspect ownership sidecar: {error}"
            )));
        }
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&lock_path)
        .map_err(|error| HarnessError::Store(format!("open ownership sidecar: {error}")))?;
    let held = file
        .metadata()
        .map_err(|error| HarnessError::Store(format!("inspect ownership handle: {error}")))?;
    let named = std::fs::symlink_metadata(&lock_path).map_err(|error| {
        HarnessError::Store(format!("inspect named ownership sidecar: {error}"))
    })?;
    if !held.is_file() || !named.is_file() || named.file_type().is_symlink() {
        return Err(HarnessError::Store("invalid ownership sidecar".into()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if held.nlink() != 1
            || named.nlink() != 1
            || held.dev() != named.dev()
            || held.ino() != named.ino()
        {
            return Err(HarnessError::Store(
                "ownership sidecar is aliased or changed during open".into(),
            ));
        }
    }
    file.try_lock().map_err(|error| {
        HarnessError::Store(format!(
            "store already owned or OS sidecar lock unavailable: {error}"
        ))
    })?;
    // Never unlink this sidecar on Drop: contenders must always lock one inode.
    Ok(StoreOwnership { file })
}

pub(super) fn migrate(conn: &mut Connection) -> HarnessResult<()> {
    let has_meta: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta')",
        [],
        |r| r.get(0),
    )?;
    let version = if has_meta {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM meta", [], |r| r.get(0))?;
        if count != 1 {
            return Err(invalid("meta must contain exactly one row"));
        }
        conn.query_row("SELECT schema_version FROM meta", [], |r| {
            r.get::<_, i64>(0)
        })?
    } else {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        if count != 0 {
            return Err(invalid("unversioned nonempty database"));
        }
        0
    };
    if !(0..=SCHEMA_VERSION).contains(&version) {
        return Err(HarnessError::Store(format!(
            "unsupported event log schema_version {version}; supported {SCHEMA_VERSION}"
        )));
    }
    if version == SCHEMA_VERSION {
        return Ok(());
    }
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if version == 0 {
        tx.execute_batch("CREATE TABLE meta(schema_version INTEGER NOT NULL,spec_version TEXT NOT NULL);
CREATE TABLE sessions(session_id TEXT PRIMARY KEY,status TEXT NOT NULL,created_ts INTEGER NOT NULL,updated_ts INTEGER NOT NULL,profile_id TEXT NOT NULL);
CREATE TABLE events(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,seq INTEGER NOT NULL,ts INTEGER NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL,artifact_refs TEXT NOT NULL DEFAULT '[]');
CREATE UNIQUE INDEX idx_events_session_seq ON events(session_id,seq);")?;
        tx.execute("INSERT INTO meta VALUES(1,?1)", [SPEC_VERSION])?;
    }
    tx.execute_batch("CREATE TABLE tool_intents(
session_id TEXT NOT NULL REFERENCES sessions(session_id),call_id TEXT NOT NULL,daemon_epoch TEXT NOT NULL,
state TEXT NOT NULL CHECK(state IN ('AwaitingApproval','Ready','Claimed','Started','Denied','Expired','Withdrawn','Rejected','Completed','OutcomeUnknown')),
binding_json TEXT NOT NULL,binding_digest TEXT,created_ts INTEGER NOT NULL,decision INTEGER CHECK(decision IN (0,1)),reason TEXT,result_json TEXT,
PRIMARY KEY(session_id,call_id));
CREATE INDEX idx_tool_intents_state ON tool_intents(state,session_id);
CREATE TABLE tool_artifacts(artifact_ref TEXT PRIMARY KEY,session_id TEXT NOT NULL,call_id TEXT NOT NULL,metadata_json TEXT NOT NULL);
CREATE TABLE runtime_metadata(store_id TEXT NOT NULL,daemon_epoch TEXT NOT NULL);
INSERT INTO runtime_metadata VALUES(lower(hex(randomblob(16))), '');")?;
    tx.execute("UPDATE meta SET schema_version=?1", [SCHEMA_VERSION])?;
    tx.commit()?;
    Ok(())
}

pub(super) fn withdraw_in_tx(tx: &rusqlite::Transaction<'_>, session: &str) -> HarnessResult<()> {
    let calls = {
        let mut stmt = tx.prepare("SELECT call_id FROM tool_intents WHERE session_id=?1 AND state IN ('AwaitingApproval','Ready','Claimed')")?;
        let rows = stmt.query_map([session], |r| r.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    for call in calls {
        terminal(tx, session, &call, "Withdrawn", "session_not_active")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{PolicyConfig, PolicyRule};
    use std::sync::Arc;
    use tool_registry::{
        call::{ToolCall, ToolCallContext},
        definition::{ApprovalRequirement, ToolDefinition, ToolEffect, ToolOrigin, ToolPolicy},
        registry::ToolImplementation,
        result::{InlineToolResult, ToolResult},
    };
    struct Never;
    impl ToolImplementation for Never {
        fn execute(&self, _: &ToolInvocation) -> ToolResult {
            panic!("storage may never execute")
        }
    }
    fn setup() -> (tempfile::TempDir, SessionStore, ToolRegistry, DaemonPolicy) {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        let mut registry = ToolRegistry::new();
        registry
            .register(
                ToolDefinition {
                    id: "shell.run".into(),
                    version: "1.0.0".into(),
                    description: "Fixture only".into(),
                    input_schema: serde_json::json!({"type":"object","additionalProperties":false}),
                    origin: ToolOrigin::Builtin,
                    policy: ToolPolicy {
                        effect: ToolEffect::Destructive,
                        approval: ApprovalRequirement::Always,
                        timeout_ms: 100,
                        max_output_bytes: 100,
                    },
                },
                Arc::new(Never),
            )
            .unwrap();
        let policy = DaemonPolicy::new(PolicyConfig {
            rules: vec![PolicyRule {
                tool_id: "shell.run".into(),
                allowed_arguments: vec![serde_json::Map::new()],
                require_approval: false,
                timeout_ms: 100,
                max_output_bytes: 100,
            }],
        })
        .unwrap();
        (dir, store, registry, policy)
    }
    fn request(call: &str) -> ToolInvocation {
        ToolInvocation {
            context: ToolCallContext {
                session_id: "s".into(),
                turn_id: "t".into(),
                generation_id: "g".into(),
            },
            call: ToolCall {
                call_id: call.into(),
                tool_id: "shell.run".into(),
                arguments: serde_json::Map::new(),
            },
        }
    }
    fn claimed(
        store: &mut SessionStore,
        registry: &ToolRegistry,
        policy: &DaemonPolicy,
    ) -> AuthorizedInvocation {
        store.admit_tool(registry, policy, request("c")).unwrap();
        store.decide_tool("s", "c", true).unwrap();
        store.claim_tool("s", "c").unwrap()
    }
    #[test]
    fn ownership_blocks_second_open_and_releases_on_drop() {
        let (dir, store, _, _) = setup();
        assert!(SessionStore::open(store.path()).is_err());
        drop(store);
        let _reopened = SessionStore::open(dir.path().join("events.db"))
            .expect("store must reopen immediately after logical owner drop");
    }
    #[test]
    fn ownership_child_probe() {
        if let Some(path) = std::env::var_os("ALLOY_W3_LOCK_PROBE") {
            assert!(SessionStore::open(path).is_err());
        }
    }
    #[test]
    fn ownership_blocks_another_process() {
        let (_dir, store, _, _) = setup();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "store::durable::tests::ownership_child_probe",
                "--nocapture",
            ])
            .env("ALLOY_W3_LOCK_PROBE", store.path())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }
    #[test]
    fn decisions_and_claims_are_single_use() {
        let (_dir, mut store, r, p) = setup();
        store.admit_tool(&r, &p, request("c")).unwrap();
        assert!(store.claim_tool("s", "c").is_err());
        assert!(store.decide_tool("other", "c", true).is_err());
        store.decide_tool("s", "c", true).unwrap();
        assert!(store.decide_tool("s", "c", false).is_err());
        let _claim = store.claim_tool("s", "c").unwrap();
        assert!(store.claim_tool("s", "c").is_err());
        assert_eq!(
            store
                .list_events("s")
                .unwrap()
                .iter()
                .filter(|e| e.kind == EventKind::ToolApproved)
                .count(),
            1
        );
    }
    #[test]
    fn denial_reserves_call_identity_forever() {
        let (_dir, mut store, r, p) = setup();
        store.admit_tool(&r, &p, request("c")).unwrap();
        store.decide_tool("s", "c", false).unwrap();
        assert!(store.claim_tool("s", "c").is_err());
        assert!(store.admit_tool(&r, &p, request("c")).is_err());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Denied")
        );
    }
    #[test]
    fn policy_change_rejects_final_start() {
        let (_dir, mut store, r, p) = setup();
        let handle = claimed(&mut store, &r, &p);
        let deny = DaemonPolicy::new(PolicyConfig::default()).unwrap();
        assert!(store.start_tool(handle, &r, &deny).is_err());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Rejected")
        );
    }
    #[test]
    fn hook_veto_never_records_start() {
        let (_dir, mut store, r, p) = setup();
        let h = claimed(&mut store, &r, &p);
        store.reject_tool(h, "hook_veto").unwrap();
        assert!(
            store
                .list_events("s")
                .unwrap()
                .iter()
                .all(|e| e.kind != EventKind::ToolStarted)
        );
    }
    #[test]
    fn pause_withdraws_claim_in_same_transaction() {
        let (_dir, mut store, r, p) = setup();
        let h = claimed(&mut store, &r, &p);
        store.set_status("s", SessionStatus::Paused).unwrap();
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Withdrawn")
        );
        store.set_status("s", SessionStatus::Active).unwrap();
        assert!(store.start_tool(h, &r, &p).is_err());
    }
    #[test]
    fn monotonic_expiry_includes_claimed() {
        let (_dir, mut store, r, p) = setup();
        let h = claimed(&mut store, &r, &p);
        store
            .claim_deadlines
            .insert(("s".into(), "c".into()), Instant::now());
        assert!(store.start_tool(h, &r, &p).is_err());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Expired")
        );
    }
    #[test]
    fn approval_does_not_renew_deadline() {
        let (_dir, mut store, r, p) = setup();
        store.admit_tool(&r, &p, request("c")).unwrap();
        let before = store.live[&("s".into(), "c".into())].deadline;
        store.decide_tool("s", "c", true).unwrap();
        assert_eq!(before, store.live[&("s".into(), "c".into())].deadline);
        store
            .live
            .get_mut(&("s".into(), "c".into()))
            .unwrap()
            .deadline = Instant::now();
        assert!(store.claim_tool("s", "c").is_err());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Expired")
        );
    }
    #[test]
    fn dropped_claim_is_not_reissued() {
        let (_dir, mut store, r, p) = setup();
        drop(claimed(&mut store, &r, &p));
        assert!(store.claim_tool("s", "c").is_err());
        assert_eq!(store.withdraw_tools("s", "cancel").unwrap(), 1);
    }
    #[test]
    fn recovery_withdraws_all_prestart_and_marks_started_unknown() {
        let (dir, mut store, r, p) = setup();
        let epoch = store.daemon_epoch().to_owned();
        let h = claimed(&mut store, &r, &p);
        let _started = store.start_tool(h, &r, &p).unwrap();
        // Separate calls cover waiting and ready recovery projections.
        store.admit_tool(&r, &p, request("wait")).unwrap();
        store.admit_tool(&r, &p, request("ready")).unwrap();
        store.decide_tool("s", "ready", true).unwrap();
        store.admit_tool(&r, &p, request("claim")).unwrap();
        store.decide_tool("s", "claim", true).unwrap();
        drop(store.claim_tool("s", "claim").unwrap());
        drop(store);
        let store = SessionStore::open(dir.path().join("events.db")).unwrap();
        assert_ne!(store.daemon_epoch(), epoch);
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("OutcomeUnknown")
        );
        for c in ["wait", "ready", "claim"] {
            assert_eq!(
                store.tool_state("s", c).unwrap().as_deref(),
                Some("Withdrawn")
            );
        }
    }
    #[test]
    fn completion_is_terminal_and_survives_reopen() {
        let (dir, mut store, r, p) = setup();
        let h = claimed(&mut store, &r, &p);
        let started = store.start_tool(h, &r, &p).unwrap();
        store
            .complete_tool(
                started,
                ToolResult::Inline(InlineToolResult {
                    call_id: "c".into(),
                    content: serde_json::json!({"ok":true}),
                    is_error: false,
                }),
            )
            .unwrap();
        drop(store);
        let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Completed")
        );
        assert!(store.claim_tool("s", "c").is_err());
    }
    #[test]
    fn per_session_capacity_released_by_denial() {
        let (_dir, mut store, r, p) = setup();
        for i in 0..32 {
            store.admit_tool(&r, &p, request(&format!("c{i}"))).unwrap();
        }
        assert!(store.admit_tool(&r, &p, request("overflow")).is_err());
        store.decide_tool("s", "c0", false).unwrap();
        store.admit_tool(&r, &p, request("replacement")).unwrap();
    }
    #[test]
    fn decision_event_failure_rolls_back_state() {
        let (_dir, mut store, r, p) = setup();
        store.admit_tool(&r, &p, request("c")).unwrap();
        store.conn.execute_batch("CREATE TEMP TRIGGER fail_event BEFORE INSERT ON events WHEN NEW.kind='tool.approved' BEGIN SELECT RAISE(ABORT,'injected'); END;").unwrap();
        assert!(store.decide_tool("s", "c", true).is_err());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("AwaitingApproval")
        );
        assert!(store.claim_tool("s", "c").is_err());
    }
    #[test]
    fn start_event_failure_returns_no_execution_receipt() {
        let (_dir, mut store, r, p) = setup();
        let h = claimed(&mut store, &r, &p);
        store.conn.execute_batch("CREATE TEMP TRIGGER fail_start BEFORE INSERT ON events WHEN NEW.kind='tool.started' BEGIN SELECT RAISE(ABORT,'injected'); END;").unwrap();
        assert!(store.start_tool(h, &r, &p).is_err());
        assert_eq!(
            store.tool_state("s", "c").unwrap().as_deref(),
            Some("Claimed")
        );
        assert!(store.claim_tool("s", "c").is_err());
    }
    #[test]
    fn migration_preserves_raw_historical_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE meta(schema_version INTEGER NOT NULL,spec_version TEXT NOT NULL); INSERT INTO meta VALUES(1,'old-spec');
CREATE TABLE sessions(session_id TEXT PRIMARY KEY,status TEXT NOT NULL,created_ts INTEGER NOT NULL,updated_ts INTEGER NOT NULL,profile_id TEXT NOT NULL); INSERT INTO sessions VALUES('s','active',1,1,'p');
CREATE TABLE events(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,seq INTEGER NOT NULL,ts INTEGER NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL,artifact_refs TEXT NOT NULL DEFAULT '[]');
CREATE UNIQUE INDEX idx_events_session_seq ON events(session_id,seq);
INSERT INTO events VALUES(17,'s',1,123,'user.message','{ \"old\": true }','[ \"legacy\" ]');").unwrap();
        drop(conn);
        let store = SessionStore::open(&path).unwrap();
        let raw: (i64, i64, i64, String, String) = store
            .conn
            .query_row(
                "SELECT id,seq,ts,payload_json,artifact_refs FROM events",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap();
        assert_eq!(
            raw,
            (
                17,
                1,
                123,
                "{ \"old\": true }".into(),
                "[ \"legacy\" ]".into()
            )
        );
        let v: i64 = store
            .conn
            .query_row("SELECT schema_version FROM meta", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, 2);
    }
    #[test]
    fn newer_schema_rejected_before_ddl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE meta(schema_version INTEGER NOT NULL,spec_version TEXT NOT NULL);INSERT INTO meta VALUES(99,'future');").unwrap();
        drop(conn);
        assert!(SessionStore::open(&path).is_err());
        let conn = Connection::open(&path).unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
    }
    #[test]
    fn racing_decisions_and_claims_have_one_winner() {
        let (_dir, mut store, r, p) = setup();
        store.admit_tool(&r, &p, request("c")).unwrap();
        let store = Arc::new(std::sync::Mutex::new(store));
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let decisions: Vec<_> = (0..2)
            .map(|_| {
                let s = Arc::clone(&store);
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    b.wait();
                    s.lock().unwrap().decide_tool("s", "c", true).is_ok()
                })
            })
            .collect();
        barrier.wait();
        assert_eq!(
            decisions
                .into_iter()
                .filter_map(|h| h.join().ok())
                .filter(|ok| *ok)
                .count(),
            1
        );
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let claims: Vec<_> = (0..2)
            .map(|_| {
                let s = Arc::clone(&store);
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    b.wait();
                    s.lock().unwrap().claim_tool("s", "c").is_ok()
                })
            })
            .collect();
        barrier.wait();
        assert_eq!(
            claims
                .into_iter()
                .filter_map(|h| h.join().ok())
                .filter(|ok| *ok)
                .count(),
            1
        );
    }
    #[test]
    fn per_daemon_capacity_is_combined_across_sessions() {
        let (_dir, mut store, r, p) = setup();
        for s in 0..9 {
            let sid = format!("extra{s}");
            store.create_session(&sid, "fixture").unwrap();
            for c in 0..32 {
                let mut req = request(&format!("c{c}"));
                req.context.session_id = sid.clone();
                let admitted = store.admit_tool(&r, &p, req);
                if s < 8 {
                    admitted.unwrap();
                } else {
                    assert!(admitted.is_err());
                    break;
                }
            }
        }
        store.withdraw_tools("extra0", "cancel").unwrap();
        let mut req = request("new");
        req.context.session_id = "extra8".into();
        store.admit_tool(&r, &p, req).unwrap();
    }
    #[test]
    fn failed_migration_rolls_back_all_new_tables_and_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE meta(schema_version INTEGER NOT NULL,spec_version TEXT NOT NULL);INSERT INTO meta VALUES(1,'v1');CREATE TABLE tool_artifacts(collision TEXT);").unwrap();
        drop(conn);
        assert!(SessionStore::open(&path).is_err());
        let conn = Connection::open(&path).unwrap();
        let version: i64 = conn
            .query_row("SELECT schema_version FROM meta", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 1);
        let new_tables: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='tool_intents'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(new_tables, 0);
    }

    #[test]
    fn ownership_sidecar_survives_drop_without_blocking_sqlite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sidecar.db");
        let mut name = path.as_os_str().to_os_string();
        name.push(".alloy-owner.lock");
        let lock_path = PathBuf::from(name);
        let mut store = SessionStore::open(&path).unwrap();
        store.create_session("sidecar-session", "fixture").unwrap();
        assert!(lock_path.is_file());
        // SQLite must obtain its own locks while daemon ownership is held.
        let connection = Connection::open(&path).unwrap();
        let version: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 0);
        drop(connection);
        #[cfg(unix)]
        let inode = {
            use std::os::unix::fs::MetadataExt;
            std::fs::metadata(&lock_path).unwrap().ino()
        };
        drop(store);
        assert!(lock_path.is_file());
        let reopened = SessionStore::open(&path).unwrap();
        assert!(reopened.get_session("sidecar-session").unwrap().is_some());
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(&lock_path).unwrap().ino(), inode);
        }
    }

    #[test]
    #[cfg(unix)]
    fn ownership_rejects_database_hardlink_alias() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("original.db");
        let alias = dir.path().join("alias.db");
        let store = SessionStore::open(&path).unwrap();
        std::fs::hard_link(&path, &alias).unwrap();
        assert!(SessionStore::open(&alias).is_err());
        assert!(SessionStore::open(&path).is_err());
        drop(store);
        assert!(SessionStore::open(&alias).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn ownership_rejects_sidecar_symlink() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("symlink.db");
        let target = dir.path().join("untouched");
        std::fs::write(&target, b"unchanged").unwrap();
        let mut name = path.as_os_str().to_os_string();
        name.push(".alloy-owner.lock");
        symlink(&target, PathBuf::from(name)).unwrap();
        assert!(SessionStore::open(&path).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"unchanged");
        assert!(!path.exists());
    }

    #[test]
    fn ownership_drop_unlocks_with_a_duplicate_descriptor_alive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicate.db");
        let owner = owned_file(&path).unwrap();
        let duplicate = owner.file.try_clone().unwrap();
        assert!(owned_file(&path).is_err());
        drop(owner);
        let next = owned_file(&path)
            .expect("logical owner drop must release despite a duplicate descriptor");
        assert!(owned_file(&path).is_err());
        drop(duplicate);
        assert!(owned_file(&path).is_err());
        drop(next);
        let _reopened = owned_file(&path).expect("replacement owner must release its lock on drop");
    }
}

#[cfg(test)]
mod failure_recovery;
