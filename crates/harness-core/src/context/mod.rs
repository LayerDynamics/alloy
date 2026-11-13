//! Context builder: session history → `ContextSnapshot` → engine render_and_count.

mod builder;
mod checkpoint;
mod skills;

pub use builder::*;
pub use checkpoint::{Checkpoint, ContextBudget};
pub(crate) use checkpoint::{admit_render, escape_json_markup};
pub use skills::skill_event_payload;
pub(crate) use skills::{project_skills, validate_skill_payload};
