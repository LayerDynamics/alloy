//! W4 builtin definitions and scoped execution primitives.
//! Limits are explicit daemon inputs; no new default time/byte limits are frozen.
//! Schema validity is not filesystem/network resource authorization.

mod definitions;

pub use definitions::*;
