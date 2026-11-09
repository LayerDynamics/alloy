//! Shared types for the real engine API.
//!
//! Request types validate structural invariants. They do not prove that an
//! artifact is verified, a profile is supported, or a request is admitted.
//! Those checks belong to the registry and concrete engine implementation.

mod contracts;

pub use contracts::*;
