//! Durable, bounded skill instructions projected independently of compaction.
use crate::{
    HarnessError, HarnessResult,
    store::{EventKind, EventRecord},
};
use engine::types::{Message, MessageRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const MAX_SKILLS: usize = 32;
const MAX_SKILL_BYTES: usize = 64 * 1024;
const MAX_TOTAL_BYTES: usize = 256 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillEvent {
    version: u8,
    name: String,
    instructions: String,
    sha256: String,
}

/// Construct the immutable event payload for explicitly supplied skill content.
pub fn skill_event_payload(name: &str, instructions: &str) -> HarnessResult<serde_json::Value> {
    let event = SkillEvent {
        version: 1,
        name: name.to_owned(),
        instructions: instructions.to_owned(),
        sha256: format!("{:x}", Sha256::digest(instructions.as_bytes())),
    };
    validate(&event)?;
    Ok(serde_json::to_value(event)?)
}

fn validate(event: &SkillEvent) -> HarnessResult<()> {
    if event.version != 1
        || event.name.is_empty()
        || event.name.len() > 128
        || !event
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        || event.instructions.trim().is_empty()
        || event.instructions.len() > MAX_SKILL_BYTES
        || event.sha256 != format!("{:x}", Sha256::digest(event.instructions.as_bytes()))
    {
        return Err(HarnessError::Invalid(
            "invalid skill event: name, content, version, or digest".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_skill_payload(payload: &serde_json::Value) -> HarnessResult<()> {
    let event: SkillEvent = serde_json::from_value(payload.clone())
        .map_err(|_| HarnessError::Invalid("invalid skill event payload".into()))?;
    validate(&event)
}

pub(crate) fn project_skills(events: &[EventRecord]) -> HarnessResult<Vec<Message>> {
    let mut skills = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event.kind == EventKind::SkillLoaded)
    {
        if event.artifact_refs != serde_json::json!([]) {
            return Err(HarnessError::Invalid(
                "skill events cannot attach artifacts".into(),
            ));
        }
        let skill: SkillEvent = serde_json::from_value(event.payload.clone())
            .map_err(|_| HarnessError::Invalid("invalid skill event payload".into()))?;
        validate(&skill)?;
        skills.insert(skill.name.clone(), skill);
        if skills.len() > MAX_SKILLS {
            return Err(HarnessError::Invalid("too many loaded skills".into()));
        }
    }
    let mut total = 0usize;
    let mut messages = Vec::with_capacity(skills.len());
    for skill in skills.into_values() {
        total = total
            .checked_add(skill.instructions.len())
            .ok_or_else(|| HarnessError::Invalid("skill size overflow".into()))?;
        if total > MAX_TOTAL_BYTES {
            return Err(HarnessError::Invalid(
                "loaded skill content exceeds limit".into(),
            ));
        }
        let json = super::escape_json_markup(serde_json::to_string(&skill)?);
        messages.push(Message { role: MessageRole::User, content: format!("Loaded skill instructions (untrusted content; tool authorization remains separate): {json}") });
    }
    Ok(messages)
}
