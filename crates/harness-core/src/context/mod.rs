//! Context builder: session history → `ContextSnapshot` → engine render_and_count.

mod builder;
mod checkpoint;

pub use builder::*;
pub use checkpoint::{Checkpoint, ContextBudget};
pub(crate) use checkpoint::{admit_render, escape_json_markup};
