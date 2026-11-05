//! Family-independent text-generation execution.
//!
//! This module runs actual model execution through GenerationSession.
//! It provides:
//!
//! - A worker thread for blocking model operations.
//! - Scheduler-owned execution lifetime.
//! - Cooperative cancellation.
//! - Bounded text-event buffering.
//! - Cancellation-aware producer backpressure.
//! - Validated, monotonic token-usage accounting.
//! - Terminal-event arbitration through ExecutionContext::finish.
//! - Rust panic containment around family execution and cleanup.
//!
//! Native process crashes are not recoverable through catch_unwind.
//!
//! A concrete family adapter must implement GenerationSession using its
//! actual model, cache, token selection, and incremental tokenizer decoder.
//! This module does not supply a simulated or fallback implementation.

use crate::{
    api::ExecutionContext,
    cancellation::{CancellationToken, Cancelled},
    error::EngineError,
    scheduler::ExecutionOutcome,
    types::{
        FinishReason, Generation, GenerationEvent, GenerationEventStream, GenerationOptions,
        RenderedContext, RequestId, Usage,
    },
};
use std::{
    collections::VecDeque,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard},
    task::{Context, Poll, Wake, Waker},
    thread::{self, Thread},
};

/// One actual token-selection and model-execution step.
///
/// The family adapter must update its real KV state and usage consistently
/// before returning this value.
///
/// Special tokens may legitimately produce an empty text delta.
#[derive(Debug)]
pub struct TokenStep {
    /// Actual selected token ID.
    pub token_id: u32,

    /// Incrementally decoded text newly available after this token.
    ///
    /// Do not decode each token independently if that violates the
    /// tokenizer's byte or Unicode semantics.
    pub text_delta: String,

    /// Whether this token satisfies the profile's EOS condition.
    pub end_of_sequence: bool,
}

/// Actual family-owned execution state for one generation.
///
/// Implement this interface with the concrete model loader/execution/cache
/// path. Merely implementing the trait does not establish conformance.
///
/// The session should own request-local resources and memory reservations
/// that must be released before the scheduler execution slot is released.
pub trait GenerationSession: Send + 'static {
    /// Perform actual prefill using the supplied rendered token sequence.
    ///
    /// Validate and honor the requested decoding configuration. Unsupported
    /// options must return an explicit error rather than silently changing
    /// the decoding mode.
    ///
    /// Check cancellation at safe prefill interruption points. Do not
    /// rerender the input with different template or tokenizer settings.
    ///
    /// On success, usage must report all supplied input tokens and zero
    /// newly generated tokens.
    fn prefill(
        &mut self,
        context: &RenderedContext,
        options: &GenerationOptions,
        cancellation: &CancellationToken,
    ) -> Result<(), EngineError>;

    /// Select and process one actual next token.
    ///
    /// Honor the supplied decoding options and maintain actual family-owned
    /// KV state. Check cancellation at safe interruption points.
    ///
    /// On success, output usage must increase by exactly one token.
    /// EOS tokens count as generated tokens even when omitted from text.
    fn decode_next(
        &mut self,
        options: &GenerationOptions,
        cancellation: &CancellationToken,
    ) -> Result<TokenStep, EngineError>;

    /// Finish the incremental tokenizer decoder after normal generation.
    ///
    /// Return only text not already emitted. This must not execute another
    /// model step or increase output-token usage.
    ///
    /// A decoding error must be reported rather than silently inventing
    /// replacement output.
    fn finish_text(&mut self) -> Result<String, EngineError>;

    /// Return usage measured by actual execution.
    ///
    /// This must remain available after an execution error so partial usage
    /// can be reported. Input usage may be partial when prefill fails.
    ///
    /// Counts must not reset or move backward during the session.
    /// This method must not perform additional model execution.
    fn usage(&self) -> Usage;
}

/// Explicit buffering policy for one generation.
///
/// The text budget covers String capacities retained in the event queue.
/// A producer may additionally retain one pending delta, bounded by
/// max_text_delta_bytes.
///
/// Queue headers, terminal diagnostics, tokenizer/model temporaries, and
/// allocator overhead must be accounted for separately by admission.
#[derive(Clone, Copy, Debug)]
pub struct GenerationLimits {
    event_capacity: usize,
    max_buffered_text_bytes: usize,
    max_text_delta_bytes: usize,
}

impl GenerationLimits {
    /// Construct an explicit, structurally valid buffering policy.
    ///
    /// No default queue or memory policy is selected implicitly.
    pub fn new(
        event_capacity: usize,
        max_buffered_text_bytes: usize,
        max_text_delta_bytes: usize,
    ) -> Result<Self, EngineError> {
        if event_capacity == 0 {
            return Err(EngineError::invalid_request(
                "event_capacity",
                "must be greater than zero",
            ));
        }

        if max_buffered_text_bytes == 0 {
            return Err(EngineError::invalid_request(
                "max_buffered_text_bytes",
                "must be greater than zero",
            ));
        }

        if max_text_delta_bytes == 0 || max_text_delta_bytes > max_buffered_text_bytes {
            return Err(EngineError::invalid_request(
                "max_text_delta_bytes",
                "must be positive and no greater than the queued-text budget",
            ));
        }

        event_capacity
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(|| {
                EngineError::invalid_request(
                    "event_capacity",
                    "queue-header size arithmetic overflow",
                )
            })?;

        Ok(Self {
            event_capacity,
            max_buffered_text_bytes,
            max_text_delta_bytes,
        })
    }

    /// Maximum number of queued nonterminal text events.
    pub fn event_capacity(&self) -> usize {
        self.event_capacity
    }

    /// Maximum aggregate String capacity retained by queued text.
    pub fn max_buffered_text_bytes(&self) -> usize {
        self.max_buffered_text_bytes
    }

    /// Maximum capacity of a single emitted or pending text delta.
    pub fn max_text_delta_bytes(&self) -> usize {
        self.max_text_delta_bytes
    }
}

/// Last validated usage and whether accounting has become inconsistent.
///
/// An inconsistent report never replaces the last validated counts.
/// Once inconsistency is detected, later reports cannot restore trust
/// in the session's accounting for this generation.
#[derive(Debug, Default)]
struct UsageTracker {
    latest: Usage,
    inconsistent: bool,
}

impl UsageTracker {
    fn reject(&mut self, message: &'static str) -> EngineError {
        self.inconsistent = true;

        EngineError::Internal {
            message: message.to_owned(),
        }
    }

    fn record_prefill(
        &mut self,
        reported: Usage,
        expected_input_tokens: usize,
    ) -> Result<(), EngineError> {
        if self.inconsistent {
            return Err(
                self.reject("cannot accept prefill usage after an accounting inconsistency")
            );
        }

        if reported.input_tokens != expected_input_tokens || reported.output_tokens != 0 {
            return Err(self.reject("family prefill usage does not match the rendered input"));
        }

        self.latest = reported;
        Ok(())
    }

    fn record_decode(
        &mut self,
        reported: Usage,
        expected_input_tokens: usize,
        expected_output_tokens: usize,
    ) -> Result<(), EngineError> {
        if self.inconsistent {
            return Err(self.reject("cannot accept decode usage after an accounting inconsistency"));
        }

        if reported.input_tokens != expected_input_tokens
            || reported.output_tokens != expected_output_tokens
        {
            return Err(self.reject("family decode usage did not advance by one token"));
        }

        let next_output_tokens = self
            .latest
            .output_tokens
            .checked_add(1)
            .ok_or_else(|| self.reject("validated output usage arithmetic overflow"))?;

        if reported.input_tokens != self.latest.input_tokens
            || reported.output_tokens != next_output_tokens
        {
            return Err(self.reject("family decode usage disagrees with validated progress"));
        }

        self.latest = reported;
        Ok(())
    }

    fn validate_text_finish(&mut self, reported: Usage) -> Result<(), EngineError> {
        if self.inconsistent {
            return Err(
                self.reject("cannot validate text completion after an accounting inconsistency")
            );
        }

        if reported != self.latest {
            return Err(self.reject("finishing text changed model token usage"));
        }

        Ok(())
    }

    /// Validate a final report without losing already validated progress.
    ///
    /// Failed execution may report additional bounded work completed
    /// before the error was returned. Successful execution may not add
    /// unobserved progress after the driver's validated completion state.
    fn reconcile_final(
        &mut self,
        reported: Usage,
        input_limit: usize,
        output_limit: usize,
        execution_succeeded: bool,
    ) -> Result<(), EngineError> {
        if self.inconsistent {
            return Err(
                self.reject("final usage cannot replace previously inconsistent accounting")
            );
        }

        if reported.input_tokens > input_limit || reported.output_tokens > output_limit {
            return Err(self.reject("family reported usage outside admitted bounds"));
        }

        if reported.input_tokens < self.latest.input_tokens
            || reported.output_tokens < self.latest.output_tokens
        {
            return Err(self.reject("family final usage moved backward from validated progress"));
        }

        if execution_succeeded && reported != self.latest {
            return Err(self.reject("successful final usage does not match validated completion"));
        }

        self.latest = reported;
        Ok(())
    }
}

struct Shared {
    state: Mutex<ChannelState>,
    limits: GenerationLimits,
}

struct ChannelState {
    text: VecDeque<String>,
    buffered_text_bytes: usize,
    terminal: Option<GenerationEvent>,
    worker_finished: bool,
    receiver_dropped: bool,
    consumer_waker: Option<Waker>,
    worker_thread: Option<Thread>,
}

struct Producer {
    shared: Arc<Shared>,
    request_id: RequestId,
    cancellation: CancellationToken,
    cancelled: Pin<Box<Cancelled>>,
    cancellation_waker: Waker,
}

struct EventStream {
    shared: Arc<Shared>,
    exhausted: bool,
}

struct WorkerWake {
    thread: Thread,
}

impl Wake for WorkerWake {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

/// Start an actual family-owned generation session.
///
/// The supplied ExecutionContext must already own the scheduler slot.
/// The session must own any real memory reservation required for execution.
///
/// This function does not perform approximate rendering, substitute model
/// output, or select a model family.
///
/// Startup allocation/thread-creation failures return directly. Model
/// execution failures are emitted through the returned event stream.
pub fn start_generation(
    execution: ExecutionContext,
    rendered: RenderedContext,
    options: GenerationOptions,
    session: Box<dyn GenerationSession>,
    limits: GenerationLimits,
) -> Result<Generation, EngineError> {
    execution.check_cancelled()?;

    if rendered.token_ids().is_empty() {
        return Err(EngineError::invalid_request(
            "token_ids",
            "generation requires nonempty rendered input",
        ));
    }

    let request_id = execution.request_id().clone();

    let requested_queue_bytes = limits
        .event_capacity
        .checked_mul(std::mem::size_of::<String>())
        .ok_or_else(|| EngineError::Internal {
            message: "generation queue size overflow".to_owned(),
        })?;

    let mut text = VecDeque::new();
    text.try_reserve_exact(limits.event_capacity)
        .map_err(|_| EngineError::AllocationFailed {
            operation: "allocate generation event queue",
            requested_bytes: u64::try_from(requested_queue_bytes).ok(),
        })?;

    let shared = Arc::new(Shared {
        state: Mutex::new(ChannelState {
            text,
            buffered_text_bytes: 0,
            terminal: None,
            worker_finished: false,
            receiver_dropped: false,
            consumer_waker: None,
            worker_thread: None,
        }),
        limits,
    });

    let worker_shared = Arc::clone(&shared);
    let worker_name = format!("engine-{}", request_id.as_str());

    thread::Builder::new()
        .name(worker_name)
        .spawn(move || {
            run_worker(worker_shared, execution, rendered, options, session);
        })
        .map_err(|source| EngineError::ExecutionFailed {
            stage: "spawn generation worker",
            message: source.to_string(),
        })?;

    Ok(Generation {
        request_id,
        events: Box::pin(EventStream {
            shared,
            exhausted: false,
        }),
    })
}

fn run_worker(
    shared: Arc<Shared>,
    execution: ExecutionContext,
    rendered: RenderedContext,
    options: GenerationOptions,
    mut session: Box<dyn GenerationSession>,
) {
    let request_id = execution.request_id().clone();
    let cancellation = execution.cancellation_token();

    let cancellation_waker = Waker::from(Arc::new(WorkerWake {
        thread: thread::current(),
    }));

    {
        let mut state = lock(&shared.state);
        state.worker_thread = Some(thread::current());
    }

    let mut producer = Producer {
        shared,
        request_id,
        cancelled: Box::pin(cancellation.cancelled()),
        cancellation,
        cancellation_waker,
    };

    let mut usage = UsageTracker::default();

    let execution_result = catch_unwind(AssertUnwindSafe(|| {
        drive_generation(
            session.as_mut(),
            &rendered,
            &options,
            &mut producer,
            &mut usage,
        )
    }));

    let mut result = match execution_result {
        Ok(result) => result,
        Err(payload) => Err(EngineError::ExecutionFailed {
            stage: "generation worker",
            message: panic_message(payload),
        }),
    };

    // An earlier accounting inconsistency already invalidated subsequent
    // reports. Preserve that error and the last validated counts rather
    // than asking the same session to overwrite them.
    if !usage.inconsistent {
        let execution_succeeded = result.is_ok();

        match catch_unwind(AssertUnwindSafe(|| session.usage())) {
            Ok(reported) => {
                if let Err(error) = usage.reconcile_final(
                    reported,
                    rendered.token_count(),
                    options.max_output_tokens(),
                    execution_succeeded,
                ) {
                    result = Err(error);
                }
            }
            Err(payload) => {
                // The last validated counts remain unchanged.
                result = Err(EngineError::ExecutionFailed {
                    stage: "read generation usage",
                    message: panic_message(payload),
                });
            }
        }
    }

    // Session-owned model workspaces and admission guards must be released
    // before execution ownership is returned to the scheduler.
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(session))) {
        result = Err(EngineError::ExecutionFailed {
            stage: "release generation session",
            message: panic_message(payload),
        });
    }

    let latest_usage = usage.latest;

    let outcome = match result {
        Ok(finish_reason) => ExecutionOutcome::Completed {
            finish_reason,
            usage: latest_usage,
        },
        Err(error) => ExecutionOutcome::Failed {
            error,
            partial_usage: latest_usage,
        },
    };

    let terminal = match execution.finish(outcome) {
        Ok(event) => event,
        Err(error) => GenerationEvent::RequestFailed {
            error,
            partial_usage: latest_usage,
        },
    };

    // Terminal delivery has its own single reserved slot. It cannot block
    // the worker behind a full text queue after model execution has stopped.
    producer.publish_terminal(terminal);
}

fn drive_generation(
    session: &mut dyn GenerationSession,
    rendered: &RenderedContext,
    options: &GenerationOptions,
    producer: &mut Producer,
    usage: &mut UsageTracker,
) -> Result<FinishReason, EngineError> {
    producer.check_running()?;

    session.prefill(rendered, options, &producer.cancellation)?;

    // Record successful prefill before checking cancellation so validated
    // completed work remains available if cancellation arrived meanwhile.
    usage.record_prefill(session.usage(), rendered.token_count())?;

    producer.check_running()?;

    let mut finish_reason = FinishReason::OutputLimit;

    for index in 0..options.max_output_tokens() {
        producer.check_running()?;

        let step = session.decode_next(options, &producer.cancellation)?;

        let expected_output_tokens = index.checked_add(1).ok_or_else(|| EngineError::Internal {
            message: "generation token counter overflow".to_owned(),
        })?;

        usage.record_decode(
            session.usage(),
            rendered.token_count(),
            expected_output_tokens,
        )?;

        producer.send_text(step.text_delta)?;

        if step.end_of_sequence {
            finish_reason = FinishReason::EndOfSequence;
            break;
        }
    }

    producer.check_running()?;

    let remaining_text = session.finish_text()?;

    usage.validate_text_finish(session.usage())?;

    producer.send_text(remaining_text)?;
    producer.check_running()?;

    Ok(finish_reason)
}

impl Producer {
    fn cancelled_error(&self) -> EngineError {
        EngineError::Cancelled {
            request_id: self.request_id.as_str().to_owned(),
        }
    }

    fn check_running(&mut self) -> Result<(), EngineError> {
        let mut context = Context::from_waker(&self.cancellation_waker);

        if self.cancelled.as_mut().poll(&mut context).is_ready() {
            return Err(self.cancelled_error());
        }

        if lock(&self.shared.state).receiver_dropped {
            return Err(self.cancelled_error());
        }

        Ok(())
    }

    fn send_text(&mut self, text: String) -> Result<(), EngineError> {
        self.check_running()?;

        if text.is_empty() {
            return Ok(());
        }

        // Count retained allocation capacity, not only the visible length.
        let charged_bytes = text.capacity();

        if charged_bytes > self.shared.limits.max_text_delta_bytes {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "buffer generation text delta",
                required_bytes: bytes_as_u64(charged_bytes)?,
                available_bytes: bytes_as_u64(self.shared.limits.max_text_delta_bytes)?,
            });
        }

        let mut pending = Some(text);

        loop {
            self.check_running()?;

            let mut consumer_to_wake = None;
            let mut sent = false;

            {
                let mut state = lock(&self.shared.state);

                if state.receiver_dropped {
                    return Err(self.cancelled_error());
                }

                let new_bytes = state
                    .buffered_text_bytes
                    .checked_add(charged_bytes)
                    .ok_or_else(|| EngineError::Internal {
                        message: "queued text size arithmetic overflow".to_owned(),
                    })?;

                let has_event_space = state.text.len() < self.shared.limits.event_capacity;
                let has_byte_space = new_bytes <= self.shared.limits.max_buffered_text_bytes;

                if has_event_space && has_byte_space {
                    let text = pending.take().ok_or_else(|| EngineError::Internal {
                        message: "generation producer lost its pending delta".to_owned(),
                    })?;

                    state.text.push_back(text);
                    state.buffered_text_bytes = new_bytes;
                    consumer_to_wake = state.consumer_waker.take();
                    sent = true;
                }
            }

            if let Some(waker) = consumer_to_wake {
                waker.wake();
            }

            if sent {
                return Ok(());
            }

            // Queue consumption, stream disposal, or cancellation unparks
            // this thread. An unpark arriving before park is retained.
            thread::park();
        }
    }

    fn publish_terminal(self, terminal: GenerationEvent) {
        let consumer_to_wake;
        let mut discarded_terminal = Some(terminal);

        {
            let mut state = lock(&self.shared.state);

            state.worker_finished = true;
            state.worker_thread = None;

            if !state.receiver_dropped {
                state.terminal = discarded_terminal.take();
            }

            consumer_to_wake = state.consumer_waker.take();
        }

        // Drop terminal diagnostics and invoke wakers outside the mutex.
        drop(discarded_terminal);

        if let Some(waker) = consumer_to_wake {
            waker.wake();
        }
    }
}

impl GenerationEventStream for EventStream {
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        let this = self.get_mut();

        if this.exhausted {
            return Poll::Ready(None);
        }

        // Clone outside the channel mutex because a custom RawWaker
        // implementation may execute code during cloning.
        //
        // Keep the replacement in an Option so the pending branch can
        // transfer it into channel state while every branch still uses
        // the same cleanup and producer-notification path.
        let mut replacement_waker = Some(cx.waker().clone());

        let result;
        let producer_to_wake;
        let old_waker;

        {
            let mut state = lock(&this.shared.state);

            if let Some(text) = state.text.pop_front() {
                state.buffered_text_bytes -= text.capacity();

                // Consuming queued text frees both an event position
                // and its charged text capacity. Resume a producer
                // that may be parked waiting for either resource.
                producer_to_wake = state.worker_thread.clone();
                old_waker = state.consumer_waker.take();

                result = Poll::Ready(Some(GenerationEvent::TextDelta { text }));
            } else if let Some(terminal) = state.terminal.take() {
                // Terminal publication occurs after execution stops;
                // no producer needs to resume to deliver this event.
                producer_to_wake = None;
                old_waker = state.consumer_waker.take();

                this.exhausted = true;
                result = Poll::Ready(Some(terminal));
            } else if state.worker_finished {
                producer_to_wake = None;
                old_waker = state.consumer_waker.take();

                this.exhausted = true;
                result = Poll::Ready(None);
            } else {
                // No queued item was consumed, so no producer capacity
                // was freed. Register the consumer and leave producer
                // notification disabled for this poll.
                producer_to_wake = None;

                old_waker = std::mem::replace(&mut state.consumer_waker, replacement_waker.take());

                result = Poll::Pending;
            }
        }

        // All branches reach this cleanup path. Waker destruction and
        // thread notification happen only after releasing the mutex.
        drop(replacement_waker);
        drop(old_waker);

        // Read the notification decision from every branch, including
        // Poll::Pending. Only actual queue consumption selects a worker.
        if let Some(worker) = producer_to_wake {
            worker.unpark();
        }

        result
    }
}

impl Drop for EventStream {
    fn drop(&mut self) {
        let worker;
        let old_waker;
        let queued;
        let terminal;

        {
            let mut state = lock(&self.shared.state);

            state.receiver_dropped = true;
            state.buffered_text_bytes = 0;

            queued = std::mem::take(&mut state.text);
            terminal = state.terminal.take();
            old_waker = state.consumer_waker.take();
            worker = state.worker_thread.clone();
        }

        drop(queued);
        drop(terminal);
        drop(old_waker);

        if let Some(worker) = worker {
            worker.unpark();
        }
    }
}

fn bytes_as_u64(bytes: usize) -> Result<u64, EngineError> {
    u64::try_from(bytes).map_err(|_| EngineError::Internal {
        message: "buffer size cannot be represented as u64".to_owned(),
    })
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("Rust panic: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("Rust panic: {message}")
    } else {
        "Rust panic with a non-string payload".to_owned()
    }
}

/// Recover channel bookkeeping after poisoning.
///
/// No family execution or user waker invocation occurs while this mutex
/// is held. The protected state is queue and notification bookkeeping.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    fn observed_usage(input_tokens: usize, output_tokens: usize) -> UsageTracker {
        let mut tracker = UsageTracker::default();

        tracker
            .record_prefill(
                Usage {
                    input_tokens,
                    output_tokens: 0,
                },
                input_tokens,
            )
            .expect("valid prefill accounting");

        for output in 1..=output_tokens {
            tracker
                .record_decode(
                    Usage {
                        input_tokens,
                        output_tokens: output,
                    },
                    input_tokens,
                    output,
                )
                .expect("valid decode accounting");
        }

        tracker
    }

    #[test]
    fn final_output_regression_preserves_validated_usage() {
        let mut tracker = observed_usage(3, 12);
        let before = tracker.latest;

        let error = tracker
            .reconcile_final(
                Usage {
                    input_tokens: 3,
                    output_tokens: 0,
                },
                3,
                20,
                false,
            )
            .expect_err("output usage must not move backward");

        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);
    }

    #[test]
    fn final_input_regression_preserves_validated_usage() {
        let mut tracker = observed_usage(8, 2);
        let before = tracker.latest;

        let error = tracker
            .reconcile_final(
                Usage {
                    input_tokens: 7,
                    output_tokens: 2,
                },
                8,
                20,
                false,
            )
            .expect_err("input usage must not move backward");

        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);
    }

    #[test]
    fn successful_final_usage_must_match_completion_exactly() {
        let mut tracker = observed_usage(3, 12);
        let before = tracker.latest;

        let error = tracker
            .reconcile_final(
                Usage {
                    input_tokens: 3,
                    output_tokens: 13,
                },
                3,
                20,
                true,
            )
            .expect_err("success cannot add unvalidated progress");

        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);
    }

    #[test]
    fn successful_matching_final_usage_is_accepted() {
        let mut tracker = observed_usage(3, 12);
        let expected = tracker.latest;

        tracker
            .reconcile_final(expected, 3, 20, true)
            .expect("matching successful report");

        assert_eq!(tracker.latest, expected);
        assert!(!tracker.inconsistent);
    }

    #[test]
    fn failed_execution_can_report_additional_bounded_progress() {
        let mut tracker = observed_usage(3, 12);

        let reported = Usage {
            input_tokens: 3,
            output_tokens: 13,
        };

        tracker
            .reconcile_final(reported, 3, 20, false)
            .expect("additional work before failure");

        assert_eq!(tracker.latest, reported);
        assert!(!tracker.inconsistent);
    }

    #[test]
    fn failed_prefill_can_report_partial_input_usage() {
        let mut tracker = UsageTracker::default();

        let reported = Usage {
            input_tokens: 2,
            output_tokens: 0,
        };

        tracker
            .reconcile_final(reported, 8, 20, false)
            .expect("partial failed prefill accounting");

        assert_eq!(tracker.latest, reported);
        assert!(!tracker.inconsistent);
    }

    #[test]
    fn final_usage_outside_input_bounds_is_rejected() {
        let mut tracker = observed_usage(3, 2);
        let before = tracker.latest;

        let error = tracker
            .reconcile_final(
                Usage {
                    input_tokens: 4,
                    output_tokens: 2,
                },
                3,
                20,
                false,
            )
            .expect_err("input usage exceeds admitted input");

        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);
    }

    #[test]
    fn final_usage_outside_output_bounds_is_rejected() {
        let mut tracker = observed_usage(3, 2);
        let before = tracker.latest;

        let error = tracker
            .reconcile_final(
                Usage {
                    input_tokens: 3,
                    output_tokens: 21,
                },
                3,
                20,
                false,
            )
            .expect_err("output usage exceeds admitted output");

        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);
    }

    #[test]
    fn inconsistent_decode_cannot_be_overwritten_by_final_report() {
        let mut tracker = observed_usage(3, 2);
        let before = tracker.latest;

        tracker
            .record_decode(
                Usage {
                    input_tokens: 3,
                    output_tokens: 7,
                },
                3,
                3,
            )
            .expect_err("decode must advance by exactly one");

        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);

        tracker
            .reconcile_final(
                Usage {
                    input_tokens: 3,
                    output_tokens: 7,
                },
                3,
                20,
                false,
            )
            .expect_err("later report cannot restore accounting trust");

        assert_eq!(tracker.latest, before);
    }

    #[test]
    fn text_finish_cannot_change_token_usage() {
        let mut tracker = observed_usage(3, 12);
        let before = tracker.latest;

        tracker
            .validate_text_finish(Usage {
                input_tokens: 3,
                output_tokens: 13,
            })
            .expect_err("text finish must not generate another token");

        assert_eq!(tracker.latest, before);
        assert!(tracker.inconsistent);
    }

    #[test]
    fn matching_text_finish_preserves_validated_usage() {
        let mut tracker = observed_usage(3, 12);
        let before = tracker.latest;

        tracker
            .validate_text_finish(before)
            .expect("unchanged usage");

        assert_eq!(tracker.latest, before);
        assert!(!tracker.inconsistent);
    }

    #[test]
    fn invalid_prefill_does_not_replace_initial_usage() {
        let mut tracker = UsageTracker::default();

        tracker
            .record_prefill(
                Usage {
                    input_tokens: 3,
                    output_tokens: 1,
                },
                3,
            )
            .expect_err("prefill must not report generated tokens");

        assert_eq!(tracker.latest, Usage::default());
        assert!(tracker.inconsistent);
    }
}
