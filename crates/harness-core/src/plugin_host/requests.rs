//! Borrow immutable inputs directly; do not materialize a second JSON value tree.
use super::protocol::{HookInvocation, VERSION};
use serde::Serialize;
use tool_registry::result::ToolResult;
#[derive(Serialize)]
struct Before<'a> {
    protocol_version: u32,
    invocation: &'a HookInvocation,
}
#[derive(Serialize)]
struct After<'a> {
    protocol_version: u32,
    invocation: &'a HookInvocation,
    result: &'a ToolResult,
    duration_ms: u64,
}
pub(super) fn before(call: &HookInvocation) -> impl Serialize + '_ {
    Before {
        protocol_version: VERSION,
        invocation: call,
    }
}
pub(super) fn after<'a>(
    call: &'a HookInvocation,
    result: &'a ToolResult,
    duration_ms: u64,
) -> impl Serialize + 'a {
    After {
        protocol_version: VERSION,
        invocation: call,
        result,
        duration_ms,
    }
}
