//! Legacy W1 implementations cannot be implicitly executed via W4 context.
use std::sync::atomic::{AtomicUsize, Ordering};
use tool_registry::{
    call::{ToolCall, ToolCallContext, ToolInvocation},
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext},
    registry::ToolImplementation,
    result::{InlineToolResult, ToolResult},
};
struct Legacy(AtomicUsize);
impl ToolImplementation for Legacy {
    fn execute(&self, call: &ToolInvocation) -> ToolResult {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolResult::Inline(InlineToolResult {
            call_id: call.call.call_id.clone(),
            content: serde_json::json!(null),
            is_error: false,
        })
    }
}
#[test]
fn legacy_context_call_fails_closed_without_execution() {
    let implementation = Legacy(AtomicUsize::new(0));
    let request = ToolInvocation {
        context: ToolCallContext {
            session_id: "s".into(),
            turn_id: "t".into(),
            generation_id: "g".into(),
        },
        call: ToolCall {
            call_id: "c".into(),
            tool_id: "fixture".into(),
            arguments: serde_json::Map::new(),
        },
    };
    let policy = ToolPolicy {
        effect: ToolEffect::Read,
        approval: ApprovalRequirement::Never,
        timeout_ms: 1000,
        max_output_bytes: 64,
    };
    let context = ExecutionContext::new(&policy, 1000, 64, CancellationToken::default()).unwrap();
    let mut output = BoundedOutput::new(&context, 64).unwrap();
    let result = implementation.execute_with_context(&request, &context, &mut output);
    assert!(result.is_error());
    assert_eq!(result.call_id(), "c");
    assert_eq!(implementation.0.load(Ordering::SeqCst), 0);
    assert!(output.finish().unwrap().is_empty());
}
