//! Receipt-owned deterministic directory listings with bounded W3 output.
//! Preparation borrows the store briefly. Streaming and file synchronization do
//! not retain a SessionStore borrow. Completion never retries the listing.
use super::*;
use crate::artifact::{ArtifactStore, ArtifactWriter, PendingArtifact};
use tool_registry::builtins::fs::ReadError;
use tool_registry::result::ArtifactToolResult;

impl ReceiptExecutor {
    /// Consume an fs.list receipt and reserve its output before any read occurs.
    /// inline_bytes is trusted daemon configuration, not an increased output cap.
    /// Other registered backends must explicitly support canonical fs.list streaming.
    pub fn prepare_file_list(
        &self,
        store: &SessionStore,
        receipt: StartedInvocation,
        artifacts: &ArtifactStore,
        cancellation: CancellationToken,
        inline_bytes: usize,
    ) -> HarnessResult<FileListExecution> {
        if inline_bytes == 0 || receipt.binding().invocation().call.tool_id != "fs.list" {
            return Err(HarnessError::Invalid(
                "fs.list and positive inline threshold required".into(),
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
        Ok(FileListExecution {
            job,
            writer,
            inline_bytes,
        })
    }
}

/// Single-use listing retaining the registered implementation and receipt lease.
#[must_use]
pub struct FileListExecution {
    job: StartedExecution,
    writer: Option<ArtifactWriter>,
    inline_bytes: usize,
}
/// Complete listing output or a failed result; no uncommitted reference escapes.
#[must_use]
pub struct FileListCompletion {
    receipt: StartedInvocation,
    payload: ListPayload,
    context: Option<ExecutionContext>,
    lease: Arc<dyn Send + Sync>,
}
enum ListPayload {
    Result(ToolResult),
    Artifact(PendingArtifact),
}
struct ListOutput<'a> {
    writer: Option<ArtifactWriter>,
    context: &'a ExecutionContext,
    inline: Option<Vec<u8>>,
    inline_bytes: usize,
    bytes: u64,
    failure: Option<ExecutionError>,
}
impl ListOutput<'_> {
    fn fail(&mut self, error: ExecutionError) -> ExecutionError {
        self.failure.get_or_insert(error);
        self.writer.take();
        self.inline.take();
        error
    }
}
impl OutputSink for ListOutput<'_> {
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
            if bytes.len() > self.inline_bytes.saturating_sub(inline.len()) {
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
fn list_error(call: &str, error: ReadError) -> ToolResult {
    match error {
        ReadError::Execution(error) => context_error(call, error, true),
        ReadError::Policy(_) => control_error(
            call,
            ToolErrorKind::Policy,
            "list_scope_or_backend_rejected",
            true,
        ),
        ReadError::Io(_) => control_error(
            call,
            ToolErrorKind::Execution,
            "directory_listing_failed",
            true,
        ),
    }
}
impl FileListExecution {
    /// List exactly once with bounded inline staging. Artifacts contain the exact
    /// deterministic JSON emitted by the scoped primitive. Inline content uses
    /// that same JSON value and must fit both inline and approved byte ceilings.
    pub fn run(self) -> FileListCompletion {
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
                return FileListCompletion {
                    receipt,
                    payload: ListPayload::Result(context_error(&call, error, false)),
                    context: None,
                    lease,
                };
            }
        };
        let mut output = ListOutput {
            writer,
            context: &context,
            inline: Some(Vec::new()),
            inline_bytes,
            bytes: 0,
            failure: None,
        };
        let payload = match context.check() {
            Err(error) => {
                output.abort();
                ListPayload::Result(context_error(&call, error, false))
            }
            Ok(()) => {
                let invocation = receipt.binding().invocation();
                match implementation.execute_file_list(invocation, &context, &mut output) {
                    Err(error) => {
                        output.abort();
                        ListPayload::Result(list_error(&call, error))
                    }
                    Ok(report) => {
                        let failure =
                            output
                                .failure
                                .or_else(|| context.check().err())
                                .or_else(|| {
                                    (report.output_bytes != output.bytes)
                                        .then_some(ExecutionError::IncompleteOutput)
                                });
                        if let Some(error) = failure {
                            output.abort();
                            ListPayload::Result(context_error(&call, error, true))
                        } else {
                            let inline = output.inline.take().and_then(|bytes| {
                                serde_json::from_slice::<serde_json::Value>(&bytes).ok()
                            });
                            let cap = context.max_output_bytes().min(inline_bytes as u64);
                            if let Some(content) = inline.filter(|value| {
                                fits(value, cap)
                                    && fits(
                                        &ToolResult::Inline(InlineToolResult {
                                            call_id: call.clone(),
                                            content: value.clone(),
                                            is_error: false,
                                        }),
                                        crate::approval::MAX_CANONICAL_BINDING_BYTES as u64,
                                    )
                            }) {
                                output.writer.take();
                                ListPayload::Result(ToolResult::Inline(InlineToolResult {
                                    call_id: call.clone(),
                                    content,
                                    is_error: false,
                                }))
                            } else {
                                match output.writer.take().map(ArtifactWriter::finish) {
                                    Some(Ok(pending)) => ListPayload::Artifact(pending),
                                    _ => ListPayload::Result(control_error(
                                        &call,
                                        ToolErrorKind::Execution,
                                        "artifact_finalization_failed",
                                        true,
                                    )),
                                }
                            }
                        }
                    }
                }
            }
        };
        drop(output);
        FileListCompletion {
            receipt,
            payload,
            context: Some(context),
            lease,
        }
    }
}
impl FileListCompletion {
    /// Commit any complete artifact through W3, then consume the original start
    /// receipt in canonical completion. Failure never grants another listing.
    /// Publication/completion failures leave W3 recovery authoritative; no valid
    /// uncommitted reference is returned. The caller releases enclosing locks
    /// before run; only these short persistence operations borrow the store.
    pub fn complete(self, store: &mut SessionStore) -> HarnessResult<ToolResult> {
        let Self {
            receipt,
            payload,
            context,
            lease,
        } = self;
        let call = receipt.binding().invocation().call.call_id.clone();
        let result = match payload {
            ListPayload::Result(result) => result,
            ListPayload::Artifact(pending) => {
                if let Some(error) = context.as_ref().and_then(|c| c.check().err()) {
                    drop(pending);
                    context_error(&call, error, true)
                } else {
                    let metadata = pending.commit(store)?;
                    ToolResult::Artifact(ArtifactToolResult {
                        call_id: call.clone(),
                        artifact_ref: metadata.artifact_ref,
                        is_error: false,
                    })
                }
            }
        };
        let result = if !result.is_error() {
            if let Some(error) = context.as_ref().and_then(|c| c.check().err()) {
                context_error(&call, error, true)
            } else {
                result
            }
        } else {
            result
        };
        store.complete_tool(receipt, result.clone())?;
        drop(lease);
        Ok(result)
    }
}
