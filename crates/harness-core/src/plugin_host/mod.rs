//! W5 supervised two-hook host. This module does not authorize/start tools.
//! W6 must invoke before hooks after approval but BEFORE durable tool.started;
//! invoke after hooks only for attempted execution, retaining the original result.

mod host;

pub use host::*;
