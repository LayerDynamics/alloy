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

    let mut latest_usage = Usage::default();

    let execution_result = catch_unwind(AssertUnwindSafe(|| {
        drive_generation(
            session.as_mut(),
            &rendered,
            &options,
            &mut producer,
            &mut latest_usage,
        )
    }));

    let mut result = match execution_result {
        Ok(result) => result,
        Err(payload) => Err(EngineError::ExecutionFailed {
            stage: "generation worker",
            message: panic_message(payload),
        }),
    };

    // Read actual partial usage even after a normal execution error.
    // Preserve the last successfully observed usage if this method panics.
    match catch_unwind(AssertUnwindSafe(|| session.usage())) {
        Ok(usage) => {
            if usage.input_tokens > rendered.token_count()
                || usage.output_tokens > options.max_output_tokens()
            {
                result = Err(EngineError::Internal {
                    message: "family reported usage outside admitted bounds".to_owned(),
                });
            } else {
                latest_usage = usage;
            }
        }
        Err(payload) => {
            result = Err(EngineError::ExecutionFailed {
                stage: "read generation usage",
                message: panic_message(payload),
            });
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
    latest_usage: &mut Usage,
) -> Result<FinishReason, EngineError> {
    producer.check_running()?;

    session.prefill(rendered, options, &producer.cancellation)?;

    producer.check_running()?;

    let usage = session.usage();
    if usage.input_tokens != rendered.token_count() || usage.output_tokens != 0 {
        return Err(EngineError::Internal {
            message: "family prefill usage does not match the rendered input".to_owned(),
        });
    }
    *latest_usage = usage;

    let mut finish_reason = FinishReason::OutputLimit;

    for index in 0..options.max_output_tokens() {
        producer.check_running()?;

        let step = session.decode_next(options, &producer.cancellation)?;

        let expected_output_tokens = index.checked_add(1).ok_or_else(|| EngineError::Internal {
            message: "generation token counter overflow".to_owned(),
        })?;

        let usage = session.usage();
        if usage.input_tokens != rendered.token_count()
            || usage.output_tokens != expected_output_tokens
        {
            return Err(EngineError::Internal {
                message: "family decode usage did not advance by one token".to_owned(),
            });
        }
        *latest_usage = usage;

        producer.send_text(step.text_delta)?;

        if step.end_of_sequence {
            finish_reason = FinishReason::EndOfSequence;
            break;
        }
    }

    producer.check_running()?;

    let remaining_text = session.finish_text()?;

    let usage = session.usage();
    if usage != *latest_usage {
        return Err(EngineError::Internal {
            message: "finishing text changed model token usage".to_owned(),
        });
    }

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
