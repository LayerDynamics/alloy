//! Project trusted, full-session W3 history into model-facing context.
//! This module reads no artifact files and grants no execution authority.
use super::ContextBuilder;
use crate::{
    approval::CONTRACT_VERSION,
    error::{HarnessError, HarnessResult},
    store::{EventKind, EventRecord},
};
use engine::types::{
    ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId, RenderingTool,
    ToolContext, ToolContextLimits, ToolHistoryBlock,
};
use serde_json::Value;
use std::collections::BTreeMap;
use tool_registry::{
    call::ToolInvocation,
    registry::ToolRegistry,
    result::{ToolError, ToolErrorKind, ToolResult},
};

impl ContextBuilder {
    /// Build generation context from a complete, ordered durable session history.
    ///
    /// `history` must come from the owning store's `list_events(session_id)`.
    /// Current schemas are copied from validated registrations, not event/model
    /// metadata. Historical calls are data, not fresh registry resolution or
    /// authorization. All historical requests must have a terminal result before
    /// another generation is built; the loop must wait outside store locks.
    ///
    /// Text before the first call stays in the base snapshot. Calls, results and
    /// subsequent text form one ordered tail, avoiding reordering or duplication.
    /// Artifact results retain their opaque reference without opening any file.
    /// Explicit tool-context limits do not bound all transient allocations; the
    /// engine's rendering/tokenization/admission checks still apply.
    pub fn build_tool_snapshot(
        snapshot_id: impl AsRef<str>,
        profile_id: impl AsRef<str>,
        session_id: &str,
        history: &[EventRecord],
        registry: &ToolRegistry,
        limits: ToolContextLimits,
    ) -> HarnessResult<ContextSnapshot> {
        if session_id.is_empty()
            || limits.max_tools == 0
            || limits.max_history_blocks == 0
            || limits.max_json_bytes == 0
        {
            return Err(invalid(
                "session and positive explicit context limits required",
            ));
        }
        if !matches!(
            history.first().map(|event| event.kind),
            Some(EventKind::SessionCreated)
        ) {
            return Err(invalid("full history must begin with session.created"));
        }
        if registry.len() > limits.max_tools {
            return Err(invalid("registered tool count exceeds context limit"));
        }
        let tools = registry
            .definitions()
            .map(|definition| {
                RenderingTool::new(
                    definition.id.clone(),
                    definition.version.clone(),
                    definition.description.clone(),
                    definition.input_schema.clone(),
                )
                .map_err(HarnessError::from)
            })
            .collect::<HarnessResult<Vec<_>>>()?;
        let mut base = Vec::new();
        let mut tail = Vec::new();
        let mut calls = BTreeMap::new();
        let mut next_seq = 1_i64;
        let mut saw_call = false;
        for event in history {
            if event.session_id != session_id || event.seq != next_seq {
                return Err(invalid(
                    "history must be complete, gapless and session-bound",
                ));
            }
            next_seq = next_seq
                .checked_add(1)
                .ok_or_else(|| invalid("event sequence overflow"))?;
            match event.kind {
                EventKind::SkillLoaded | EventKind::ContextCompacted | EventKind::ModelChanged => {
                    return Err(invalid(
                        "skill, checkpoint, or model-change context is not yet supported; cannot safely reconstruct this session",
                    ));
                }
                EventKind::UserMessage | EventKind::AssistantMessage => {
                    let message = Message {
                        role: if event.kind == EventKind::UserMessage {
                            MessageRole::User
                        } else {
                            MessageRole::Assistant
                        },
                        content: super::message_content(event)?,
                    };
                    if saw_call {
                        push(&mut tail, ToolHistoryBlock::Text(message), limits)?;
                    } else {
                        base.push(message);
                    }
                }
                EventKind::ToolRequested => {
                    contract(event)?;
                    let call_id = string(&event.payload, "call_id")?;
                    if let Some(request) = event.payload.get("request") {
                        if event.payload.get("transition").is_some() {
                            return Err(invalid("request and transition are mutually exclusive"));
                        }
                        let invocation: ToolInvocation = serde_json::from_value(request.clone())?;
                        if invocation.context.session_id != session_id
                            || invocation.call.call_id != call_id
                            || invocation.context.turn_id.is_empty()
                            || invocation.context.turn_id.trim() != invocation.context.turn_id
                            || invocation.context.generation_id.is_empty()
                            || invocation.context.generation_id.trim()
                                != invocation.context.generation_id
                            || calls.insert(call_id.to_owned(), false).is_some()
                        {
                            return Err(invalid(
                                "duplicate or incorrectly bound historical request",
                            ));
                        }
                        saw_call = true;
                        push(
                            &mut tail,
                            ToolHistoryBlock::AssistantCall {
                                call_id: invocation.call.call_id,
                                tool_id: invocation.call.tool_id,
                                arguments: invocation.call.arguments,
                            },
                            limits,
                        )?;
                    } else {
                        if string(&event.payload, "transition")? == "HookReceipts" {
                            if !calls.contains_key(call_id) || !event.payload["receipts"].is_array()
                            {
                                return Err(invalid("orphan or malformed hook receipts"));
                            }
                            continue;
                        }
                        if string(&event.payload, "transition")? != "Claimed" {
                            return Err(invalid("unknown request transition"));
                        }
                        pending(&mut calls, call_id)?;
                    }
                }
                EventKind::ToolApproved | EventKind::ToolStarted => {
                    contract(event)?;
                    pending(&mut calls, string(&event.payload, "call_id")?)?;
                }
                EventKind::ToolCompleted => {
                    contract(event)?;
                    let call_id = string(&event.payload, "call_id")?;
                    let completed = pending(&mut calls, call_id)?;
                    let (content, is_error) = result_content(event, call_id)?;
                    *completed = true;
                    push(
                        &mut tail,
                        ToolHistoryBlock::ToolResult {
                            call_id: call_id.to_owned(),
                            content,
                            is_error,
                        },
                        limits,
                    )?;
                }
                _ => {}
            }
        }
        if calls.values().any(|completed| !completed) {
            return Err(invalid(
                "tool history has an unresolved call; do not regenerate",
            ));
        }
        // Empty bases are rejected by ContextSnapshot; do not invent a user turn.
        let snapshot_id =
            ContextSnapshotId::new(snapshot_id.as_ref()).map_err(HarnessError::from)?;
        let profile_id = ModelProfileId::new(profile_id.as_ref()).map_err(HarnessError::from)?;
        let context = ToolContext::new(tools, tail, limits).map_err(HarnessError::from)?;
        ContextSnapshot::new(snapshot_id, profile_id, base)
            .map(|snapshot| snapshot.with_tool_context(context))
            .map_err(HarnessError::from)
    }
}
fn invalid(message: &str) -> HarnessError {
    HarnessError::Invalid(message.into())
}
fn string<'a>(payload: &'a Value, field: &str) -> HarnessResult<&'a str> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("historical event missing required string field"))
}
fn contract(event: &EventRecord) -> HarnessResult<()> {
    if string(&event.payload, "contract_version")? != CONTRACT_VERSION {
        return Err(invalid("unsupported historical tool contract"));
    }
    Ok(())
}
fn pending<'a>(
    calls: &'a mut BTreeMap<String, bool>,
    call_id: &str,
) -> HarnessResult<&'a mut bool> {
    match calls.get_mut(call_id) {
        Some(completed) if !*completed => Ok(completed),
        _ => Err(invalid(
            "orphan, future or already-completed historical call",
        )),
    }
}
fn push(
    tail: &mut Vec<ToolHistoryBlock>,
    block: ToolHistoryBlock,
    limits: ToolContextLimits,
) -> HarnessResult<()> {
    if tail.len() >= limits.max_history_blocks {
        return Err(invalid("historical block count exceeds context limit"));
    }
    tail.push(block);
    Ok(())
}
fn result_content(event: &EventRecord, call_id: &str) -> HarnessResult<(String, bool)> {
    match string(&event.payload, "state")? {
        "Completed" => {
            let value = event
                .payload
                .get("result")
                .ok_or_else(|| invalid("completed call lacks result"))?;
            let result: ToolResult = serde_json::from_value(value.clone())?;
            if result.call_id() != call_id {
                return Err(invalid("historical result call identity mismatch"));
            }
            let is_error = result.is_error();
            let content = match &result {
                ToolResult::Inline(inline) => serde_json::to_string(&inline.content)?,
                ToolResult::Artifact(_) => serde_json::to_string(&result)?,
            };
            Ok((
                if is_error {
                    bounded_error_context(content)?
                } else {
                    content
                },
                is_error,
            ))
        }
        state @ ("Rejected" | "Denied" | "Withdrawn" | "Expired" | "OutcomeUnknown") => {
            if event.payload.get("result").is_some()
                || event.payload.get("is_error").and_then(Value::as_bool) != Some(true)
                || event.payload.get("outcome_known").and_then(Value::as_bool)
                    != Some(state != "OutcomeUnknown")
                || string(&event.payload, "reason")?.is_empty()
            {
                return Err(invalid("invalid terminal history marker"));
            }
            let kind = match state {
                "Rejected" | "Denied" | "Expired" => ToolErrorKind::Policy,
                _ => ToolErrorKind::Execution,
            };
            let error = ToolError {
                kind,
                message: format!("Tool call terminated: {state}"),
                details: event.payload.clone(),
            };
            Ok((bounded_error_context(serde_json::to_string(&error)?)?, true))
        }
        _ => Err(invalid("unknown terminal tool state")),
    }
}

/// Keep the full result in its append-only event, but make unusually large
/// errors explicit bounded previews in model context. The preview is data,
/// never authority, and callers can inspect the unmodified session event.
fn bounded_error_context(content: String) -> HarnessResult<String> {
    const PREVIEW_SOURCE_BYTES: usize = 96;
    if content.len() <= super::MODEL_ERROR_CONTEXT_BYTES {
        return Ok(content);
    }
    let mut end = PREVIEW_SOURCE_BYTES.min(content.len());
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    let preview = serde_json::json!({
        "context_projection": "truncated_error_preview",
        "preview": &content[..end],
        "full_result": "retained_in_session_history",
    });
    let serialized = serde_json::to_string(&preview)?;
    if serialized.len() > super::MODEL_ERROR_CONTEXT_BYTES {
        return Err(invalid("bounded error preview exceeds context ceiling"));
    }
    Ok(serialized)
}
