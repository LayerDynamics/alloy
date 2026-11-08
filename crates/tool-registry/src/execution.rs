//! W4 execution context primitives, not authorization or an executor.
//!
//! The harness must supply a successful durable-start receipt and current trusted
//! registration before invoking production code. Constructing a context grants
//! no permission. Cancellation is cooperative here; each builtin must also stop
//! and release its owned processes, sockets and handles.

use crate::definition::ToolPolicy;
use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Operation-local cancellation. Clones affect only this operation's state.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    /// Request cancellation; no child process is implicitly killed by this flag.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Whether this operation has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Internal execution failure, not a new serialized tool-error category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    /// A limit is zero, exceeds registration, or overflows the clock.
    InvalidLimits,
    /// The operation's owner requested cancellation.
    Cancelled,
    /// The monotonic execution deadline has been reached.
    DeadlineExceeded,
    /// A bounded output sink cannot accept the complete output.
    OutputLimit,
    /// Memory reservation failed.
    AllocationFailed,
    /// A previous write failure makes the output incomplete.
    IncompleteOutput,
}
impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ExecutionError {}

/// Trusted runtime inputs; no Clone, deserialization, or execution authority.
#[derive(Debug)]
pub struct ExecutionContext {
    cancellation: CancellationToken,
    deadline: Instant,
    max_output_bytes: u64,
}
impl ExecutionContext {
    /// Start the execution clock using effective limits that cannot widen the
    /// supplied registration. The harness must obtain these values from the
    /// immutable approved binding, not from model/client input or fresh defaults.
    pub fn new(
        registration: &ToolPolicy,
        timeout_ms: u64,
        max_output_bytes: u64,
        cancellation: CancellationToken,
    ) -> Result<Self, ExecutionError> {
        if timeout_ms == 0
            || max_output_bytes == 0
            || timeout_ms > registration.timeout_ms
            || max_output_bytes > registration.max_output_bytes
        {
            return Err(ExecutionError::InvalidLimits);
        }
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(timeout_ms))
            .ok_or(ExecutionError::InvalidLimits)?;
        Ok(Self {
            cancellation,
            deadline,
            max_output_bytes,
        })
    }
    /// Check the operation-local cancellation flag and deadline.
    pub fn check(&self) -> Result<(), ExecutionError> {
        self.check_at(Instant::now())
    }
    fn check_at(&self, now: Instant) -> Result<(), ExecutionError> {
        if self.cancellation.is_cancelled() {
            return Err(ExecutionError::Cancelled);
        }
        if now >= self.deadline {
            return Err(ExecutionError::DeadlineExceeded);
        }
        Ok(())
    }
    /// Remaining time, or a failure if already cancelled/expired.
    pub fn remaining(&self) -> Result<Duration, ExecutionError> {
        let now = Instant::now();
        self.check_at(now)?;
        Ok(self.deadline.saturating_duration_since(now))
    }
    /// Effective accepted-output ceiling, distinct from any inline threshold.
    pub fn max_output_bytes(&self) -> u64 {
        self.max_output_bytes
    }
    /// The fixed monotonic execution deadline; callers cannot renew it.
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

/// Streaming sink contract implemented above the registry for W3 artifacts.
/// A write failure must prevent treating any accepted prefix as complete output.
/// No SessionStore borrow should be retained by a sink while streaming.
pub trait OutputSink {
    /// Accept all bytes or fail; partial output is never complete success.
    fn write_chunk(&mut self, bytes: &[u8]) -> Result<(), ExecutionError>;
    /// Invalidate partial output after any producer or sink failure.
    fn abort(&mut self);
}

/// Small bounded collector for fixtures/inline staging, not the artifact store.
/// Production artifact promotion must not allocate an entire unbounded result.
#[derive(Debug)]
pub struct BoundedOutput {
    bytes: Vec<u8>,
    maximum: usize,
    failed: bool,
}
impl BoundedOutput {
    /// Create a collector with a caller-selected ceiling, capped by the binding.
    pub fn new(context: &ExecutionContext, maximum: usize) -> Result<Self, ExecutionError> {
        if maximum == 0 || maximum as u128 > u128::from(context.max_output_bytes()) {
            return Err(ExecutionError::InvalidLimits);
        }
        Ok(Self {
            bytes: Vec::new(),
            maximum,
            failed: false,
        })
    }
    /// Expose bytes only if every write completed within the ceiling.
    pub fn finish(self) -> Result<Vec<u8>, ExecutionError> {
        if self.failed {
            Err(ExecutionError::IncompleteOutput)
        } else {
            Ok(self.bytes)
        }
    }
}
impl OutputSink for BoundedOutput {
    fn abort(&mut self) {
        self.failed = true;
    }
    fn write_chunk(&mut self, bytes: &[u8]) -> Result<(), ExecutionError> {
        if self.failed {
            return Err(ExecutionError::IncompleteOutput);
        }
        if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
            self.failed = true;
            return Err(ExecutionError::OutputLimit);
        }
        if self.bytes.try_reserve(bytes.len()).is_err() {
            self.failed = true;
            return Err(ExecutionError::AllocationFailed);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::{ApprovalRequirement, ToolEffect};
    fn policy() -> ToolPolicy {
        ToolPolicy {
            effect: ToolEffect::Read,
            approval: ApprovalRequirement::Never,
            timeout_ms: 1000,
            max_output_bytes: 8,
        }
    }
    #[test]
    fn effective_limits_cannot_widen_registration() {
        for (time, bytes) in [(0, 1), (1, 0), (1001, 1), (1, 9)] {
            assert_eq!(
                ExecutionContext::new(&policy(), time, bytes, CancellationToken::default())
                    .unwrap_err(),
                ExecutionError::InvalidLimits
            );
        }
        assert!(ExecutionContext::new(&policy(), 1000, 8, CancellationToken::default()).is_ok());
    }
    #[test]
    fn cancellation_is_shared_only_with_the_owned_operation() {
        let a = CancellationToken::default();
        let b = CancellationToken::default();
        let context = ExecutionContext::new(&policy(), 1000, 8, a.clone()).unwrap();
        a.cancel();
        assert_eq!(context.check(), Err(ExecutionError::Cancelled));
        assert!(!b.is_cancelled());
    }
    #[test]
    fn equality_with_deadline_is_expired_without_sleeping() {
        let context =
            ExecutionContext::new(&policy(), 1000, 8, CancellationToken::default()).unwrap();
        assert_eq!(
            context.check_at(context.deadline()),
            Err(ExecutionError::DeadlineExceeded)
        );
    }
    #[test]
    fn exact_limit_succeeds_but_overflow_poisoning_is_terminal() {
        let context =
            ExecutionContext::new(&policy(), 1000, 8, CancellationToken::default()).unwrap();
        let mut output = BoundedOutput::new(&context, 8).unwrap();
        output.write_chunk(b"12345678").unwrap();
        assert_eq!(output.finish().unwrap(), b"12345678");
        let mut output = BoundedOutput::new(&context, 8).unwrap();
        output.write_chunk(b"1234").unwrap();
        assert_eq!(
            output.write_chunk(b"56789"),
            Err(ExecutionError::OutputLimit)
        );
        assert_eq!(
            output.write_chunk(b""),
            Err(ExecutionError::IncompleteOutput)
        );
        assert_eq!(output.finish(), Err(ExecutionError::IncompleteOutput));
    }
}
