//! Single-use W3-authorized create-only writes with explicit publication status.
//! No store borrow crosses execution. Completion failure retains the actual tool
//! result and consumes authority; callers must never retry the external effect.
use super::*;
use tool_registry::builtins::fs::{ReadError, WriteFailure, WritePublication};
use tool_registry::execution::BoundedOutput;

impl ReceiptExecutor {
    /// Verify and consume an fs.write start receipt. No caller arguments, approval
    /// flags or replacement limits are accepted. Preparation performs no write.
    pub fn prepare_file_write(
        &self,
        store: &SessionStore,
        receipt: StartedInvocation,
        cancellation: CancellationToken,
    ) -> HarnessResult<FileWriteExecution> {
        if receipt.binding().invocation().call.tool_id != "fs.write" {
            return Err(HarnessError::Invalid("fs.write receipt required".into()));
        }
        Ok(FileWriteExecution {
            job: self.prepare_started(store, receipt, cancellation)?,
        })
    }
}
/// Single-use write retaining the exact registered implementation and receipt.
#[must_use]
pub struct FileWriteExecution {
    job: StartedExecution,
}
/// A completed attempt, including publication outcome. Not execution authority.
#[must_use]
pub struct FileWriteCompletion {
    receipt: StartedInvocation,
    result: ToolResult,
    context: Option<ExecutionContext>,
    lease: Arc<dyn Send + Sync>,
}
/// Persistence failed after an attempt. The original outcome remains available;
/// neither this value nor its result permits another write or completion retry.
#[derive(Debug)]
pub struct WriteCompletionFailure {
    /// Underlying W3 persistence error, which may have an uncertain commit outcome.
    pub error: HarnessError,
    /// Actual attempted-write result, preserving published/unknown status.
    pub result: ToolResult,
}
impl std::fmt::Display for WriteCompletionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "write completion persistence failed: {}", self.error)
    }
}
impl std::error::Error for WriteCompletionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
fn publication_name(value: WritePublication) -> &'static str {
    match value {
        WritePublication::NotPublished => "not_published",
        WritePublication::Unknown => "unknown",
        WritePublication::Published => "published",
    }
}
fn failure_result(
    call: &str,
    kind: ToolErrorKind,
    reason: &str,
    publication: WritePublication,
    cleanup_failed: bool,
) -> ToolResult {
    ToolResult::Inline(InlineToolResult {
        call_id: call.into(),
        is_error: true,
        content: serde_json::json!({"kind":kind,"message":"Scoped file write failed or could not deliver its complete result",
            "details":{"reason":reason,"publication":publication_name(publication),
                "destination_may_have_been_published":publication!=WritePublication::NotPublished,
                "staging_cleanup_failed":cleanup_failed}}),
    })
}
fn execution_failure(
    call: &str,
    error: ExecutionError,
    publication: WritePublication,
) -> ToolResult {
    let (kind, reason) = match error {
        ExecutionError::Cancelled => (ToolErrorKind::Execution, "cancelled"),
        ExecutionError::DeadlineExceeded => (ToolErrorKind::Timeout, "deadline_exceeded"),
        ExecutionError::InvalidLimits => (ToolErrorKind::Execution, "invalid_execution_limits"),
        ExecutionError::OutputLimit => (ToolErrorKind::Execution, "output_limit"),
        ExecutionError::AllocationFailed => (ToolErrorKind::Execution, "allocation_failed"),
        ExecutionError::IncompleteOutput => (ToolErrorKind::Execution, "incomplete_output"),
    };
    failure_result(call, kind, reason, publication, false)
}
fn primitive_failure(call: &str, failure: WriteFailure) -> ToolResult {
    let result = match failure.cause {
        ReadError::Execution(error) => execution_failure(call, error, failure.publication),
        ReadError::Policy(_) => failure_result(
            call,
            ToolErrorKind::Policy,
            "write_scope_or_input_rejected",
            failure.publication,
            false,
        ),
        ReadError::Io(_) => failure_result(
            call,
            ToolErrorKind::Execution,
            "file_write_failed",
            failure.publication,
            false,
        ),
    };
    match result {
        ToolResult::Inline(mut inline) => {
            inline.content["details"]["staging_cleanup_failed"] =
                serde_json::json!(failure.cleanup_error.is_some());
            ToolResult::Inline(inline)
        }
        result => result,
    }
}
fn success_content(path: &str, bytes: u64) -> serde_json::Value {
    serde_json::json!({"path":path,"bytes_written":bytes,"publication":"published"})
}
impl FileWriteExecution {
    /// Attempt the bound invocation once. Preflight successful-result size before
    /// effects; afterward typed primitive publication status remains authoritative.
    /// Cancellation or output errors after publication never imply non-execution.
    pub fn run(self) -> FileWriteCompletion {
        let StartedExecution {
            receipt,
            implementation,
            context,
            lease,
        } = self.job;
        let call = receipt.binding().invocation().call.call_id.clone();
        let context = match context {
            Ok(context) => context,
            Err(error) => {
                return FileWriteCompletion {
                    receipt,
                    result: execution_failure(&call, error, WritePublication::NotPublished),
                    context: None,
                    lease,
                };
            }
        };
        let invocation = receipt.binding().invocation();
        let result = (|| {
            if let Err(error) = context.check() {
                return execution_failure(&call, error, WritePublication::NotPublished);
            }
            let args = &invocation.call.arguments;
            let (Some(path), Some(content)) = (
                args.get("path").and_then(|v| v.as_str()),
                args.get("content").and_then(|v| v.as_str()),
            ) else {
                return failure_result(
                    &call,
                    ToolErrorKind::Protocol,
                    "invalid_write_arguments",
                    WritePublication::NotPublished,
                    false,
                );
            };
            let expected = success_content(path, content.len() as u64);
            let expected_result = ToolResult::Inline(InlineToolResult {
                call_id: call.clone(),
                content: expected.clone(),
                is_error: false,
            });
            if !fits(&expected, context.max_output_bytes())
                || !fits(
                    &expected_result,
                    crate::approval::MAX_CANONICAL_BINDING_BYTES as u64,
                )
            {
                return execution_failure(
                    &call,
                    ExecutionError::OutputLimit,
                    WritePublication::NotPublished,
                );
            }
            // The primitive's output is just path/byte-count metadata. Bound its
            // collector to that exact representation, never to an unbounded body.
            let primitive_content =
                serde_json::json!({"path":path,"bytes_written":content.len() as u64});
            let capacity = match serde_json::to_vec(&primitive_content) {
                Ok(bytes) => bytes.len(),
                Err(_) => {
                    return execution_failure(
                        &call,
                        ExecutionError::IncompleteOutput,
                        WritePublication::NotPublished,
                    );
                }
            };
            let mut output = match BoundedOutput::new(&context, capacity) {
                Ok(output) => output,
                Err(error) => {
                    return execution_failure(&call, error, WritePublication::NotPublished);
                }
            };
            let report = match implementation.execute_file_write(invocation, &context, &mut output)
            {
                Ok(report) => report,
                Err(failure) => {
                    output.abort();
                    return primitive_failure(&call, failure);
                }
            };
            // Ok is a publication claim. Never downgrade it to NotPublished even
            // if a registered backend returns malformed output or wrong metadata.
            if report.path != path || report.bytes_written != content.len() as u64 {
                output.abort();
                return failure_result(
                    &call,
                    ToolErrorKind::Protocol,
                    "write_report_mismatch",
                    WritePublication::Unknown,
                    false,
                );
            }
            let bytes = match output.finish() {
                Ok(bytes) => bytes,
                Err(error) => return execution_failure(&call, error, WritePublication::Published),
            };
            if serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .as_ref()
                != Some(&primitive_content)
            {
                return failure_result(
                    &call,
                    ToolErrorKind::Protocol,
                    "write_output_mismatch",
                    WritePublication::Published,
                    false,
                );
            }
            if let Err(error) = context.check() {
                return execution_failure(&call, error, WritePublication::Published);
            }
            expected_result
        })();
        FileWriteCompletion {
            receipt,
            result,
            context: Some(context),
            lease,
        }
    }
}
impl FileWriteCompletion {
    /// Inspect or clone the outcome without recovering execution authority.
    pub fn result(&self) -> &ToolResult {
        &self.result
    }
    /// Consume the original receipt in W3 completion. Persistence failure returns
    /// the actual outcome alongside the error; no replay/automatic retry is offered.
    /// W3 recovery, not this in-memory value, decides an uncertain commit's state.
    pub fn complete(
        self,
        store: &mut SessionStore,
    ) -> Result<ToolResult, Box<WriteCompletionFailure>> {
        let Self {
            receipt,
            mut result,
            context,
            lease,
        } = self;
        if !result.is_error()
            && let Some(error) = context.as_ref().and_then(|c| c.check().err())
        {
            result = execution_failure(result.call_id(), error, WritePublication::Published);
        }
        let saved = store.complete_tool(receipt, result.clone());
        drop(lease);
        match saved {
            Ok(()) => Ok(result),
            Err(error) => Err(Box::new(WriteCompletionFailure { error, result })),
        }
    }
}
