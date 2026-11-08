//! W3-bound HTTP execution with bounded inline/artifact output.
//! Streaming retains no store borrow. Completion never replays an external effect.
//! Destination pins and cooperative cancellation bounds remain those of HttpScope.
use super::*;
use crate::artifact::{ArtifactStore, ArtifactWriter, PendingArtifact};
use tool_registry::{
    builtins::http::{HttpError, HttpFailure, HttpReport},
    result::ArtifactToolResult,
};

impl ReceiptExecutor {
    /// Consume a verified http.fetch receipt and reserve bounded output before
    /// network work. The binding must retain read/policy defaults; no replacement
    /// arguments, client approval flags, or caller execution limits are accepted.
    pub fn prepare_http_fetch(
        &self,
        store: &SessionStore,
        receipt: StartedInvocation,
        artifacts: &ArtifactStore,
        cancellation: CancellationToken,
        inline_bytes: usize,
    ) -> HarnessResult<HttpExecution> {
        use tool_registry::definition::{ApprovalRequirement, ToolEffect};
        let binding = receipt.binding();
        if inline_bytes == 0
            || binding.invocation().call.tool_id != "http.fetch"
            || binding.definition().policy.effect != ToolEffect::Read
            || binding.definition().policy.approval != ApprovalRequirement::Policy
        {
            return Err(HarnessError::Invalid(
                "read/policy http.fetch receipt and positive inline threshold required".into(),
            ));
        }
        let job = self.prepare_started(store, receipt, cancellation)?;
        let invocation = job.receipt.binding().invocation();
        let writer = match &job.context {
            Ok(context) => Some(artifacts.begin(
                store,
                &invocation.context.session_id,
                &invocation.call.call_id,
                context.max_output_bytes(),
            )?),
            Err(_) => None,
        };
        Ok(HttpExecution {
            job,
            writer,
            inline_bytes,
        })
    }
}
/// Single-use approved HTTP attempt holding the exact registered implementation.
#[must_use]
pub struct HttpExecution {
    job: StartedExecution,
    writer: Option<ArtifactWriter>,
    inline_bytes: usize,
}
/// Original receipt paired with its actual HTTP outcome and collected output.
#[must_use]
pub struct HttpCompletion {
    receipt: StartedInvocation,
    payload: HttpPayload,
    outcome: serde_json::Value,
    context: Option<ExecutionContext>,
    lease: Arc<dyn Send + Sync>,
}
enum HttpPayload {
    Result(ToolResult),
    Artifact {
        pending: Box<PendingArtifact>,
        is_error: bool,
    },
}
/// Persistence failed; execution authority remains consumed. Inspect the outcome
/// even if artifact registration failed before a usable result reference existed.
#[derive(Debug)]
pub struct HttpCompletionFailure {
    /// Original persistence error, not evidence of whether its commit took effect.
    pub error: HarnessError,
    /// Actual attempted-exchange/response metadata from this one attempt.
    pub outcome: serde_json::Value,
    /// Result when available; None means artifact registration did not return a
    /// usable reference. It never means the request was not attempted.
    pub result: Option<ToolResult>,
}
impl std::fmt::Display for HttpCompletionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP completion persistence failed: {}", self.error)
    }
}
impl std::error::Error for HttpCompletionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
struct HttpOutput<'a> {
    writer: Option<ArtifactWriter>,
    context: &'a ExecutionContext,
    inline: Option<Vec<u8>>,
    threshold: usize,
    bytes: u64,
    failure: Option<ExecutionError>,
}
impl HttpOutput<'_> {
    fn fail(&mut self, error: ExecutionError) -> ExecutionError {
        self.failure.get_or_insert(error);
        self.writer.take();
        self.inline.take();
        error
    }
}
impl OutputSink for HttpOutput<'_> {
    fn write_chunk(&mut self, bytes: &[u8]) -> Result<(), ExecutionError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if let Err(error) = self.context.check() {
            return Err(self.fail(error));
        }
        if bytes.len() as u64 > self.context.max_output_bytes().saturating_sub(self.bytes) {
            return Err(self.fail(ExecutionError::OutputLimit));
        }
        let Some(writer) = self.writer.as_mut() else {
            return Err(self.fail(ExecutionError::IncompleteOutput));
        };
        if writer.write_all(bytes).is_err() {
            return Err(self.fail(ExecutionError::IncompleteOutput));
        }
        if let Some(inline) = &mut self.inline {
            if bytes.len() > self.threshold.saturating_sub(inline.len()) {
                self.inline = None;
            } else {
                if inline.try_reserve(bytes.len()).is_err() {
                    return Err(self.fail(ExecutionError::AllocationFailed));
                }
                inline.extend_from_slice(bytes);
            }
        }
        self.bytes += bytes.len() as u64;
        if let Err(error) = self.context.check() {
            return Err(self.fail(error));
        }
        Ok(())
    }
    fn abort(&mut self) {
        self.fail(ExecutionError::IncompleteOutput);
    }
}
fn no_request() -> serde_json::Value {
    serde_json::json!({"requests_attempted":0})
}
fn report_outcome(report: &HttpReport) -> serde_json::Value {
    serde_json::json!({"requests_attempted":report.requests_attempted,"status":report.status,
        "final_url_preview":report.final_url.chars().take(512).collect::<String>(),
        "final_url_truncated":report.final_url.chars().count()>512,
        "body_bytes":report.body_bytes,"output_bytes":report.output_bytes})
}
fn error_result(
    call: &str,
    kind: ToolErrorKind,
    reason: &str,
    outcome: &serde_json::Value,
) -> ToolResult {
    ToolResult::Inline(InlineToolResult {
        call_id: call.into(),
        is_error: true,
        content: serde_json::json!({"kind":kind,"message":"HTTP invocation failed or could not deliver complete output",
            "details":{"reason":reason,"outcome":outcome}}),
    })
}
fn execution_error(call: &str, error: ExecutionError, outcome: &serde_json::Value) -> ToolResult {
    let (kind, reason) = match error {
        ExecutionError::Cancelled => (ToolErrorKind::Execution, "cancelled"),
        ExecutionError::DeadlineExceeded => (ToolErrorKind::Timeout, "deadline_exceeded"),
        ExecutionError::InvalidLimits => (ToolErrorKind::Execution, "invalid_execution_limits"),
        ExecutionError::OutputLimit => (ToolErrorKind::Execution, "output_limit"),
        ExecutionError::AllocationFailed => (ToolErrorKind::Execution, "allocation_failed"),
        ExecutionError::IncompleteOutput => (ToolErrorKind::Execution, "incomplete_output"),
    };
    error_result(call, kind, reason, outcome)
}
fn primitive_error(call: &str, failure: HttpFailure) -> (ToolResult, serde_json::Value) {
    let outcome = serde_json::json!({"requests_attempted":failure.requests_attempted});
    let result = match failure.cause {
        HttpError::Execution(error) => execution_error(call, error, &outcome),
        HttpError::Policy(_) => error_result(
            call,
            ToolErrorKind::Policy,
            "http_scope_or_response_rejected",
            &outcome,
        ),
        HttpError::ExchangeTimeout => error_result(
            call,
            ToolErrorKind::Timeout,
            "http_exchange_timeout",
            &outcome,
        ),
        HttpError::Transport => error_result(
            call,
            ToolErrorKind::Execution,
            "http_transport_failed",
            &outcome,
        ),
    };
    (result, outcome)
}
impl HttpExecution {
    /// Execute the approved fetch once, collecting response/body NDJSON frames.
    /// Small output is inline; oversized complete output uses W3 artifacts.
    /// HTTP 4xx/5xx responses retain complete output but set the tool error flag.
    pub fn run(self) -> HttpCompletion {
        let Self {
            job,
            writer,
            inline_bytes,
        } = self;
        let StartedExecution {
            receipt,
            implementation,
            context,
            lease,
        } = job;
        let call = receipt.binding().invocation().call.call_id.clone();
        let context = match context {
            Ok(context) => context,
            Err(error) => {
                let outcome = no_request();
                return HttpCompletion {
                    receipt,
                    payload: HttpPayload::Result(execution_error(&call, error, &outcome)),
                    outcome,
                    context: None,
                    lease,
                };
            }
        };
        let mut output = HttpOutput {
            writer,
            context: &context,
            inline: Some(Vec::new()),
            threshold: inline_bytes,
            bytes: 0,
            failure: None,
        };
        let (payload, outcome) = match context.check() {
            Err(error) => {
                output.abort();
                let outcome = no_request();
                (
                    HttpPayload::Result(execution_error(&call, error, &outcome)),
                    outcome,
                )
            }
            Ok(()) => match implementation.execute_http_fetch(
                receipt.binding().invocation(),
                &context,
                &mut output,
            ) {
                Err(failure) => {
                    output.abort();
                    let (result, outcome) = primitive_error(&call, failure);
                    (HttpPayload::Result(result), outcome)
                }
                Ok(report) => {
                    let outcome = report_outcome(&report);
                    let failure = output
                        .failure
                        .or_else(|| context.check().err())
                        .or_else(|| {
                            (report.output_bytes != output.bytes)
                                .then_some(ExecutionError::IncompleteOutput)
                        });
                    let payload = if let Some(error) = failure {
                        output.abort();
                        HttpPayload::Result(execution_error(&call, error, &outcome))
                    } else {
                        let inline=output.inline.take().and_then(|bytes|String::from_utf8(bytes).ok()).map(|ndjson|
                            ToolResult::Inline(InlineToolResult {call_id:call.clone(),is_error:report.status >= 400,
                                content:serde_json::json!({"format":"http.ndjson.v1","outcome":outcome,"ndjson":ndjson})}));
                        let cap = context.max_output_bytes().min(inline_bytes as u64);
                        let inline = inline.filter(|result| match result {
                            ToolResult::Inline(inline) => {
                                fits(&inline.content, cap)
                                    && fits(
                                        result,
                                        crate::approval::MAX_CANONICAL_BINDING_BYTES as u64,
                                    )
                            }
                            _ => false,
                        });
                        if let Some(result) = inline {
                            output.writer.take();
                            HttpPayload::Result(result)
                        } else {
                            match output.writer.take().map(ArtifactWriter::finish) {
                                Some(Ok(pending)) => HttpPayload::Artifact {
                                    pending: Box::new(pending),
                                    is_error: report.status >= 400,
                                },
                                _ => HttpPayload::Result(error_result(
                                    &call,
                                    ToolErrorKind::Execution,
                                    "artifact_finalization_failed",
                                    &outcome,
                                )),
                            }
                        }
                    };
                    (payload, outcome)
                }
            },
        };
        drop(output);
        HttpCompletion {
            receipt,
            payload,
            outcome,
            context: Some(context),
            lease,
        }
    }
}
impl HttpCompletion {
    /// Inspect actual HTTP metadata without exposing a pending artifact reference.
    pub fn outcome(&self) -> &serde_json::Value {
        &self.outcome
    }
    /// Commit complete output and persist the result using the original receipt.
    /// Any persistence failure returns the HTTP outcome and never offers replay.
    pub fn complete(
        self,
        store: &mut SessionStore,
    ) -> Result<ToolResult, Box<HttpCompletionFailure>> {
        let Self {
            receipt,
            payload,
            outcome,
            context,
            lease,
        } = self;
        let call = receipt.binding().invocation().call.call_id.clone();
        let result = match payload {
            HttpPayload::Result(result) => result,
            HttpPayload::Artifact { pending, is_error } => {
                if let Some(error) = context.as_ref().and_then(|c| c.check().err()) {
                    drop(pending);
                    execution_error(&call, error, &outcome)
                } else {
                    let metadata = match (*pending).commit(store) {
                        Ok(metadata) => metadata,
                        Err(error) => {
                            return Err(Box::new(HttpCompletionFailure {
                                error,
                                outcome,
                                result: None,
                            }));
                        }
                    };
                    ToolResult::Artifact(ArtifactToolResult {
                        call_id: call.clone(),
                        artifact_ref: metadata.artifact_ref,
                        is_error,
                    })
                }
            }
        };
        let result = if !result.is_error() {
            if let Some(error) = context.as_ref().and_then(|c| c.check().err()) {
                execution_error(&call, error, &outcome)
            } else {
                result
            }
        } else {
            result
        };
        let saved = store.complete_tool(receipt, result.clone());
        drop(lease);
        match saved {
            Ok(()) => Ok(result),
            Err(error) => Err(Box::new(HttpCompletionFailure {
                error,
                outcome,
                result: Some(result),
            })),
        }
    }
}
