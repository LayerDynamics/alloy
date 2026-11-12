//! Strict structured checkpoint and admission calculations for compaction.
use serde::{Deserialize, Serialize};

/// SPEC-1 B.3 checkpoint. Every field is required; unknown versions and fields
/// are rejected so model output cannot silently lose part of the contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// Checkpoint schema version.
    pub checkpoint_version: u8,
    /// Session objective.
    pub goal: String,
    /// Applicable constraints.
    pub constraints: Vec<String>,
    /// Decisions already made.
    pub decisions: Vec<String>,
    /// Work already completed.
    pub completed_work: Vec<String>,
    /// Remaining work.
    pub open_tasks: Vec<String>,
    /// References to durable artifacts.
    pub artifact_references: Vec<String>,
    /// Identifiers needed to continue.
    pub critical_identifiers: Vec<String>,
    /// Failures and lessons to retain.
    pub failures_and_lessons: Vec<String>,
    /// Immediate next action.
    pub next_action: String,
}

impl Checkpoint {
    /// Parse model output and reject unsupported versions or empty core fields.
    pub fn parse_json(output: &str) -> Result<Self, String> {
        let checkpoint: Self = serde_json::from_str(output).map_err(|e| e.to_string())?;
        if checkpoint.checkpoint_version != 1 {
            return Err("unsupported checkpoint version".into());
        }
        if checkpoint.goal.trim().is_empty() || checkpoint.next_action.trim().is_empty() {
            return Err("checkpoint goal and next_action must be nonempty".into());
        }
        Ok(checkpoint)
    }
}

/// Profile-bound token budget. All values are in tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextBudget {
    /// Validated context window.
    pub limit: usize,
    /// Reserved generation tokens.
    pub output_reserve: usize,
    /// Additional safety reserve.
    pub safety_margin: usize,
}

impl ContextBudget {
    /// Remaining tokens after input, projected growth and reserved allowances.
    /// `None` means the request cannot fit or arithmetic overflowed.
    pub fn headroom(self, input: usize, projected_growth: usize) -> Option<usize> {
        let used = input
            .checked_add(projected_growth)?
            .checked_add(self.output_reserve)?
            .checked_add(self.safety_margin)?;
        self.limit.checked_sub(used)
    }

    /// A candidate must leave at least one fifth of the full validated window.
    pub fn checkpoint_fits(self, input: usize, projected_growth: usize) -> bool {
        self.headroom(input, projected_growth)
            .is_some_and(|remaining| remaining >= self.limit / 5 + usize::from(self.limit % 5 != 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_requires_every_b3_field_and_supported_version() {
        let json = r#"{"checkpoint_version":1,"goal":"g","constraints":["c"],"decisions":["d"],"completed_work":["done"],"open_tasks":["open"],"artifact_references":["art:1"],"critical_identifiers":["id"],"failures_and_lessons":["failure"],"next_action":"next"}"#;
        let value = Checkpoint::parse_json(json).unwrap();
        assert_eq!(
            Checkpoint::parse_json(&serde_json::to_string(&value).unwrap()).unwrap(),
            value
        );
        assert!(
            Checkpoint::parse_json(
                &json.replace("\"checkpoint_version\":1", "\"checkpoint_version\":2")
            )
            .is_err()
        );
        assert!(Checkpoint::parse_json(&json.replace(",\"open_tasks\":[\"open\"]", "")).is_err());
        assert!(Checkpoint::parse_json("not json").is_err());
    }
    #[test]
    fn admission_counts_reserves_growth_and_exact_headroom_boundary() {
        let budget = ContextBudget {
            limit: 101,
            output_reserve: 10,
            safety_margin: 5,
        };
        assert_eq!(budget.headroom(60, 5), Some(21));
        assert!(budget.checkpoint_fits(60, 5));
        assert!(!budget.checkpoint_fits(61, 5));
        assert_eq!(budget.headroom(usize::MAX, 1), None);
        assert_eq!(budget.headroom(100, 0), None);
    }
}

/// Verify the exact render against the engine's registered profile budget.
/// Engines without a validated budget are rejected before generation.
pub(crate) fn admit_render<E: engine::api::Engine + ?Sized>(
    engine: &E,
    profile: &engine::types::ModelProfileId,
    input: usize,
    projected_growth: usize,
) -> crate::HarnessResult<()> {
    let profile = engine
        .profile_context_budget(profile)
        .map_err(crate::HarnessError::from)?
        .ok_or_else(|| {
            crate::HarnessError::Invalid(
                "engine does not expose a validated profile context budget".into(),
            )
        })?;
    let budget = ContextBudget {
        limit: profile.limit,
        output_reserve: profile.output_reserve,
        safety_margin: profile.safety_margin,
    };
    if budget.headroom(input, projected_growth).is_none() {
        return Err(crate::HarnessError::Invalid(format!(
            "context exceeds validated limit: input={input}, projected growth={projected_growth}, output reserve={}, safety margin={}, limit={}; compact history or reduce the request",
            budget.output_reserve, budget.safety_margin, budget.limit
        )));
    }
    Ok(())
}

/// Keep serialized JSON data from introducing model control spellings.
pub(crate) fn escape_json_markup(json: String) -> String {
    json.replace('<', "\\u003c")
}
