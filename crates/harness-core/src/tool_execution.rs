//! W4 receipt-owned dispatch at the existing W3 harness boundary.
//!
//! This module does not admit, approve, claim or start an invocation. It consumes
//! an existing durable receipt. Registering retains the SAME implementation Arc
//! in W1 and this private dispatch index; no executable handle is exposed.
//! No SessionStore borrow is retained across run. Callers must also release any
//! enclosing session/store/native-slot locks. W6 configuration publication fences
//! and W5 hooks remain separate obligations, not implemented here.

use crate::{HarnessError, HarnessResult, SessionStore, store::StartedInvocation};
/// Receipt-owned raw file-read output and canonical artifact completion.
#[cfg(unix)]
pub mod file_read;

/// Receipt-owned deterministic directory listing and canonical output.
#[cfg(unix)]
pub mod file_list;

/// Receipt-owned create-only writes preserving publication and persistence outcomes.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod file_write;

/// Receipt-owned destructive shell execution and bounded result persistence.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod shell_run;

/// Receipt-owned policy-governed HTTP requests and bounded results.
pub mod http_fetch;

use serde::Serialize;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::Arc,
};
use tool_registry::{
    RegistryError,
    call::ToolInvocation,
    definition::ToolDefinition,
    execution::{CancellationToken, ExecutionContext, ExecutionError, OutputSink},
    registry::{ToolImplementation, ToolRegistry},
    result::{InlineToolResult, ToolErrorKind, ToolResult},
};

/// Required daemon-supplied executable implementations for the builtin set.
/// No field is optional and no placeholder/default implementation is supplied.
/// The caller must configure resource-scoped implementations; this bundle alone
/// neither authorizes calls nor establishes filesystem/network scope enforcement.
/// ```compile_fail
/// use harness_core::tool_execution::BuiltinImplementations;
/// use std::sync::Arc;
/// use tool_registry::registry::ToolImplementation;
/// fn incomplete(read: Arc<dyn ToolImplementation>) {
///     let _ = BuiltinImplementations { fs_read: read };
/// }
/// ```
pub struct BuiltinImplementations {
    /// Implementation assigned exclusively to the fs.read registration.
    pub fs_read: Arc<dyn ToolImplementation>,
    /// Implementation assigned exclusively to the fs.list registration.
    pub fs_list: Arc<dyn ToolImplementation>,
    /// Implementation assigned exclusively to the fs.write registration.
    pub fs_write: Arc<dyn ToolImplementation>,
    /// Implementation assigned exclusively to the shell.run registration.
    pub shell_run: Arc<dyn ToolImplementation>,
    /// Implementation assigned exclusively to the http.fetch registration.
    pub http_fetch: Arc<dyn ToolImplementation>,
}

/// Daemon-owned registration facade with a private index of the same W1 Arcs.
/// Existing IDs cannot be replaced; registration never invokes an implementation.
#[derive(Default)]
pub struct ReceiptExecutor {
    registry: ToolRegistry,
    implementations: BTreeMap<String, Arc<dyn ToolImplementation>>,
}
impl ReceiptExecutor {
    /// Build a complete five-entry executable catalogue using the existing
    /// builtin definition/schema factory and its exact effect/approval defaults.
    /// Limits are trusted daemon inputs, not new implicit defaults. This returns
    /// either the whole catalogue or an error; it never mutates a live executor.
    /// Registration performs no execution and grants no W3 admission/approval.
    /// Callers must supply all five real backends before operational use; this
    /// constructor does not implement any missing backend or wire W6 dispatch.
    pub fn with_builtins(
        timeout_ms: u64,
        max_output_bytes: u64,
        implementations: BuiltinImplementations,
    ) -> Result<Self, RegistryError> {
        let mut definitions: BTreeMap<_, _> =
            tool_registry::builtins::definitions(timeout_ms, max_output_bytes)?
                .into_iter()
                .map(|definition| (definition.id.clone(), definition))
                .collect();
        if definitions.len() != 5 {
            return Err(RegistryError::Validation(
                tool_registry::ValidationError::InvalidSchema(
                    "builtin definition set must contain five unique entries".into(),
                ),
            ));
        }
        let entries = [
            ("fs.read", implementations.fs_read),
            ("fs.list", implementations.fs_list),
            ("fs.write", implementations.fs_write),
            ("shell.run", implementations.shell_run),
            ("http.fetch", implementations.http_fetch),
        ];
        let mut executor = Self::new();
        for (id, implementation) in entries {
            let definition = definitions
                .remove(id)
                .ok_or_else(|| RegistryError::UnknownTool(id.into()))?;
            executor.register(definition, implementation)?;
        }
        Ok(executor)
    }

    /// Construct an empty facade; this does not register builtin placeholders.
    pub fn new() -> Self {
        Self::default()
    }
    /// Register one exact definition/implementation in W1 and the private index.
    pub fn register(
        &mut self,
        definition: ToolDefinition,
        implementation: Arc<dyn ToolImplementation>,
    ) -> Result<(), RegistryError> {
        let id = definition.id.clone();
        self.registry
            .register(definition, Arc::clone(&implementation))?;
        self.implementations.insert(id, implementation);
        Ok(())
    }
    /// Read-only registry for existing W3 admission/final-start validation.
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }
    /// Consume a current durable receipt and detach a single-use execution job.
    /// No replacement invocation, client approval flag or caller limit is accepted.
    /// Preparation failure permits zero implementation invocations; its consumed
    /// Started receipt is not restored to runnable authority and recovers unknown.
    pub fn prepare_started(
        &self,
        store: &SessionStore,
        receipt: StartedInvocation,
        cancellation: CancellationToken,
    ) -> HarnessResult<StartedExecution> {
        store.verify_started_execution(&receipt)?;
        let binding = receipt.binding();
        let resolved = self
            .registry
            .resolve(binding.invocation().clone())
            .map_err(|e| HarnessError::Invalid(format!("execution registry: {e}")))?;
        if serde_json::to_value(resolved.definition())?
            != serde_json::to_value(binding.definition())?
        {
            return Err(HarnessError::Invalid(
                "execution registration differs from started binding".into(),
            ));
        }
        let implementation = self
            .implementations
            .get(&binding.invocation().call.tool_id)
            .cloned()
            .ok_or_else(|| HarnessError::Invalid("missing registered implementation".into()))?;
        let field = |name: &str| {
            binding.snapshot()[name]
                .as_u64()
                .ok_or_else(|| HarnessError::Invalid(format!("missing bound {name}")))
        };
        let context = ExecutionContext::from_start(
            &binding.definition().policy,
            field("effective_timeout_ms")?,
            field("effective_max_output_bytes")?,
            cancellation,
            receipt.execution_started_at(),
        );
        Ok(StartedExecution {
            receipt,
            implementation,
            context,
            lease: store.artifact_lease(),
        })
    }
}

/// Single-use execution job; no constructor, Clone or Deserialize is provided.
/// Its lease retains exclusive daemon database ownership, not a store/SQL lock.
/// Dropping a job never regrants execution; unresolved Started recovers unknown.
/// ```compile_fail
/// use harness_core::tool_execution::StartedExecution;
/// use tool_registry::execution::OutputSink;
/// fn twice(job: StartedExecution, output: &mut dyn OutputSink) {
///     let _first = job.run(output);
///     let _second = job.run(output);
/// }
/// ```
#[must_use]
pub struct StartedExecution {
    receipt: StartedInvocation,
    implementation: Arc<dyn ToolImplementation>,
    context: Result<ExecutionContext, ExecutionError>,
    lease: Arc<dyn Send + Sync>,
}
impl StartedExecution {
    /// Invoke only the receipt's unchanged call, once, without a store borrow.
    /// Context-aware dispatch never falls back to the W1 legacy execute method.
    /// Panics leave Started unresolved; they do not grant another attempt.
    pub fn run(self, output: &mut dyn OutputSink) -> ExecutionCompletion {
        let Self {
            receipt,
            implementation,
            context,
            lease,
        } = self;
        let invocation = receipt.binding().invocation();
        let result = match context {
            Ok(context) => invoke(implementation.as_ref(), invocation, &context, output),
            Err(error) => {
                output.abort();
                context_error(&invocation.call.call_id, error, false)
            }
        };
        ExecutionCompletion {
            receipt,
            result,
            lease,
        }
    }
}

/// Original receipt paired with its result. This cannot be executed or cloned.
/// Persist completion once; failure must not cause replay of the external effect.
#[must_use]
pub struct ExecutionCompletion {
    receipt: StartedInvocation,
    result: ToolResult,
    lease: Arc<dyn Send + Sync>,
}
impl ExecutionCompletion {
    /// Inspect the result without mutating its call identity or recovering authority.
    pub fn result(&self) -> &ToolResult {
        &self.result
    }
    /// Use the existing W3 canonical completion transaction, consuming the receipt.
    /// Artifact results still require W3 committed provenance. Streaming artifact
    /// finalization and builtin adapters are NOT implemented by this generic bridge.
    pub fn complete(self, store: &mut SessionStore) -> HarnessResult<()> {
        let Self {
            receipt,
            result,
            lease,
        } = self;
        let saved = store.complete_tool(receipt, result);
        drop(lease);
        saved
    }
}
fn control_error(call: &str, kind: ToolErrorKind, reason: &str, invoked: bool) -> ToolResult {
    ToolResult::Inline(InlineToolResult {
        call_id: call.into(),
        is_error: true,
        content: serde_json::json!({"kind":kind,"message":"Tool execution boundary rejected or failed the invocation",
            "details":{"reason":reason,"implementation_invoked":invoked,"effects_may_have_occurred":invoked}}),
    })
}
fn context_error(call: &str, error: ExecutionError, invoked: bool) -> ToolResult {
    let (kind, reason) = match error {
        ExecutionError::Cancelled => (ToolErrorKind::Execution, "cancelled"),
        ExecutionError::DeadlineExceeded => (ToolErrorKind::Timeout, "deadline_exceeded"),
        ExecutionError::OutputLimit => (ToolErrorKind::Execution, "output_limit"),
        ExecutionError::InvalidLimits => (ToolErrorKind::Execution, "invalid_execution_limits"),
        ExecutionError::AllocationFailed => (ToolErrorKind::Execution, "allocation_failed"),
        ExecutionError::IncompleteOutput => (ToolErrorKind::Execution, "incomplete_output"),
    };
    control_error(call, kind, reason, invoked)
}
struct LengthLimit {
    bytes: u64,
    maximum: u64,
}
impl Write for LengthLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.maximum.saturating_sub(self.bytes) {
            return Err(io::Error::other("serialized output limit"));
        }
        self.bytes += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn fits<T: Serialize>(value: &T, maximum: u64) -> bool {
    serde_json::to_writer(&mut LengthLimit { bytes: 0, maximum }, value).is_ok()
}
struct CheckedOutput<'a> {
    output: &'a mut dyn OutputSink,
    context: &'a ExecutionContext,
    bytes: u64,
    failure: Option<ExecutionError>,
}
impl CheckedOutput<'_> {
    fn fail(&mut self, error: ExecutionError) -> ExecutionError {
        self.failure.get_or_insert(error);
        self.output.abort();
        error
    }
}
impl OutputSink for CheckedOutput<'_> {
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
        if let Err(error) = self.output.write_chunk(bytes) {
            return Err(self.fail(error));
        }
        self.bytes += bytes.len() as u64;
        Ok(())
    }
    fn abort(&mut self) {
        self.fail(ExecutionError::IncompleteOutput);
    }
}
fn invoke(
    implementation: &dyn ToolImplementation,
    invocation: &ToolInvocation,
    context: &ExecutionContext,
    output: &mut dyn OutputSink,
) -> ToolResult {
    let call = &invocation.call.call_id;
    if let Err(error) = context.check() {
        output.abort();
        return context_error(call, error, false);
    }
    let mut checked = CheckedOutput {
        output,
        context,
        bytes: 0,
        failure: None,
    };
    let result = implementation.execute_with_context(invocation, context, &mut checked);
    if result.call_id() != call {
        checked.abort();
        return control_error(call, ToolErrorKind::Protocol, "result_call_mismatch", true);
    }
    let payload_fits = match &result {
        ToolResult::Inline(inline) => fits(&inline.content, context.max_output_bytes()),
        ToolResult::Artifact(_) => true,
    };
    if !payload_fits || !fits(&result, crate::approval::MAX_CANONICAL_BINDING_BYTES as u64) {
        checked.abort();
        return context_error(call, ExecutionError::OutputLimit, true);
    }
    if let Some(error) = checked.failure.or_else(|| context.check().err()) {
        checked.abort();
        // Keep an already-failed tool's bounded, correctly linked actual result:
        // e.g. a write may have published before cancellation. Never turn that
        // into a false assertion of no effect. Control errors are fixed metadata,
        // not permission to accept additional successful output payload bytes.
        if result.is_error() {
            return result;
        }
        return context_error(call, error, true);
    }
    result
}
