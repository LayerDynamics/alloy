//! Phase-1 supervisor skeleton: user → render → generate → persist.

use crate::context::ContextBuilder;
use crate::error::{HarnessError, HarnessResult};
use crate::store::{EventKind, SessionStatus, SessionStore};
use crate::telemetry::TelemetryClient;
use engine::api::Engine;
use engine::types::{
    CancelOutcome, GenerationEvent, GenerationOptions, GenerationRequest, MessageRole,
    ModelProfileId, RequestId,
};
use std::future::{Future, poll_fn};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Wake, Waker};
use std::thread;
use std::time::{Duration, Instant};

#[path = "compaction.rs"]
mod compaction;
#[path = "tool_loop.rs"]
mod tool_loop;
#[path = "tool_round.rs"]
mod tool_round;
pub use tool_loop::{ToolLoopConfig, ToolLoopEvent, ToolRuntime};
pub use tool_round::{RoundProposals, ToolRoundConfig, ToolRoundOutcome, ToolRoundRequest};

const TURN_TIMEOUT: Duration = Duration::from_secs(120);
// Large native prefills can exceed the ordinary inter-event timeout. Keep a
// bounded first-token deadline; control operations and decode retain 120s.
const LONG_PREFILL_TIMEOUT: Duration = Duration::from_secs(600);

/// Result of one Phase-1 chat turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    /// Session that ran the turn.
    pub session_id: String,
    /// Assistant text concatenated from stream deltas.
    pub assistant_text: String,
    /// Exact input token count from render_and_count.
    pub input_tokens: usize,
    /// Request id used for the generation (for cancel coordination).
    pub request_id: RequestId,
}

/// Single-session turn loop over engine + store + telemetry.
pub struct Supervisor<E: Engine> {
    engine: E,
    telemetry: Arc<TelemetryClient>,
}

impl<E: Engine> Supervisor<E> {
    /// Bind an engine handle and telemetry client.
    pub fn new(engine: E, telemetry: Arc<TelemetryClient>) -> Self {
        Self { engine, telemetry }
    }

    /// Borrow the engine (cancel / capabilities).
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Create a session row and record telemetry.
    pub fn create_session(
        &self,
        store: &Mutex<SessionStore>,
        session_id: &str,
        profile_id: &str,
    ) -> HarnessResult<()> {
        let mut guard = lock_store(store)?;
        guard.create_session(session_id, profile_id)?;
        self.telemetry.record_session();
        Ok(())
    }

    /// Pause a live session: cancel in-flight generation first, then flip status.
    ///
    /// Only `active` → `paused` (idempotent if already paused). Cancel errors
    /// propagate and leave status unchanged.
    pub fn pause_session(
        &self,
        store: &Mutex<SessionStore>,
        session_id: &str,
        inflight_request: Option<&RequestId>,
    ) -> HarnessResult<()> {
        if let Some(request_id) = inflight_request {
            block_on(self.engine.cancel(request_id.clone()))?.map_err(HarnessError::from)?;
        }
        let mut guard = lock_store(store)?;
        let session = guard
            .get_session(session_id)?
            .ok_or_else(|| HarnessError::Invalid(format!("unknown session: {session_id}")))?;
        match session.status {
            SessionStatus::Paused => Ok(()),
            SessionStatus::Active => {
                guard.set_status(session_id, SessionStatus::Paused)?;
                Ok(())
            }
            SessionStatus::Completed => Err(HarnessError::Invalid(format!(
                "cannot pause completed session: {session_id}"
            ))),
        }
    }

    /// Resume a paused live session (only `paused` → `active`).
    pub fn resume_session(
        &self,
        store: &Mutex<SessionStore>,
        session_id: &str,
    ) -> HarnessResult<()> {
        let mut guard = lock_store(store)?;
        let session = guard
            .get_session(session_id)?
            .ok_or_else(|| HarnessError::Invalid(format!("unknown session: {session_id}")))?;
        match session.status {
            SessionStatus::Paused => {
                guard.set_status(session_id, SessionStatus::Active)?;
                Ok(())
            }
            SessionStatus::Active => Err(HarnessError::Invalid(format!(
                "cannot resume active session (already active): {session_id}"
            ))),
            SessionStatus::Completed => Err(HarnessError::Invalid(format!(
                "cannot resume completed session: {session_id}"
            ))),
        }
    }

    /// Request cancellation of an active generation.
    pub fn cancel(&self, request_id: RequestId) -> HarnessResult<CancelOutcome> {
        block_on(self.engine.cancel(request_id))?.map_err(HarnessError::from)
    }

    /// Run one user turn, buffering the assistant text until completion.
    pub fn run_turn(
        &self,
        store: &Mutex<SessionStore>,
        session_id: &str,
        user_text: &str,
        max_output_tokens: usize,
    ) -> HarnessResult<TurnOutcome> {
        self.run_turn_streaming(
            store,
            session_id,
            user_text,
            max_output_tokens,
            |_| Ok(()),
            |_| Ok(()),
        )
    }

    /// Run one user turn with streaming callbacks.
    ///
    /// Store locks are taken only around durable reads/writes so a concurrent
    /// `pause_session` can cancel mid-generation.
    pub fn run_turn_streaming<S, F>(
        &self,
        store: &Mutex<SessionStore>,
        session_id: &str,
        user_text: &str,
        max_output_tokens: usize,
        on_started: S,
        on_event: F,
    ) -> HarnessResult<TurnOutcome>
    where
        S: FnMut(&RequestId) -> HarnessResult<()>,
        F: FnMut(&GenerationEvent) -> HarnessResult<()>,
    {
        self.run_turn_streaming_with_lifecycle(
            store,
            session_id,
            user_text,
            max_output_tokens,
            on_started,
            |_| Ok(()),
            on_event,
        )
    }

    /// Run a text turn with callbacks for registering and clearing compaction requests.
    #[allow(clippy::too_many_arguments)]
    pub fn run_turn_streaming_with_lifecycle<S, D, F>(
        &self,
        store: &Mutex<SessionStore>,
        session_id: &str,
        user_text: &str,
        max_output_tokens: usize,
        mut on_started: S,
        mut on_finished: D,
        mut on_event: F,
    ) -> HarnessResult<TurnOutcome>
    where
        S: FnMut(&RequestId) -> HarnessResult<()>,
        D: FnMut(&RequestId) -> HarnessResult<()>,
        F: FnMut(&GenerationEvent) -> HarnessResult<()>,
    {
        if user_text.is_empty() {
            return Err(HarnessError::Invalid("user text must be non-empty".into()));
        }

        let (profile_id, request_id, history, turn_seq) = {
            let mut guard = lock_store(store)?;
            let session = guard
                .get_session(session_id)?
                .ok_or_else(|| HarnessError::Invalid(format!("unknown session: {session_id}")))?;
            if session.status != SessionStatus::Active {
                return Err(HarnessError::Invalid(format!(
                    "session is not active: {session_id}"
                )));
            }

            guard.append_event(
                session_id,
                EventKind::UserMessage,
                serde_json::json!({ "content": user_text, "role": MessageRole::User.as_str() }),
                serde_json::json!([]),
            )?;
            let turn_started = guard.append_event(
                session_id,
                EventKind::RunStarted,
                serde_json::json!({}),
                serde_json::json!([]),
            )?;

            let history = guard.list_events(session_id)?;
            let request_id = round_request_id(session_id, turn_started.seq, 0)?;
            (
                session.profile_id.clone(),
                request_id,
                history,
                turn_started.seq,
            )
        };

        let result = (|| {
            let mut snapshot =
                ContextBuilder::build_snapshot(request_id.as_str(), &profile_id, &history, None)?;
            let mut rendered = ContextBuilder::render_and_count(&self.engine, snapshot.clone())?;
            if let Some(limits) = self
                .engine
                .profile_context_budget(snapshot.profile_id())
                .map_err(HarnessError::from)?
            {
                if max_output_tokens > limits.output_reserve {
                    return Err(HarnessError::Invalid(format!(
                        "output token limit {max_output_tokens} exceeds validated profile reserve {}; reduce the requested limit",
                        limits.output_reserve
                    )));
                }
                let budget = crate::context::ContextBudget {
                    limit: limits.limit,
                    output_reserve: limits.output_reserve,
                    safety_margin: limits.safety_margin,
                };
                if budget.headroom(rendered.token_count(), 0).is_none() {
                    compaction::compact(
                        &self.engine,
                        store,
                        session_id,
                        &history,
                        &request_id,
                        &compaction::Mode::Text,
                        budget,
                        0,
                        Instant::now() + TURN_TIMEOUT,
                        &|| ensure_session_active(store, session_id),
                        &mut on_started,
                        &mut on_finished,
                    )?;
                    let history = lock_store(store)?.list_events(session_id)?;
                    snapshot = ContextBuilder::build_snapshot(
                        request_id.as_str(),
                        &profile_id,
                        &history,
                        None,
                    )?;
                    rendered = ContextBuilder::render_and_count(&self.engine, snapshot.clone())?;
                }
            }
            crate::context::admit_render(
                &self.engine,
                snapshot.profile_id(),
                rendered.token_count(),
                0,
            )?;
            // Re-check Active after render so a concurrent pause aborts before generate.
            ensure_session_active(store, session_id)?;
            let profile = ModelProfileId::new(profile_id.clone())
                .map_err(|error| HarnessError::Invalid(format!("profile_id: {error}")))?;
            let options =
                GenerationOptions::greedy(max_output_tokens).map_err(HarnessError::from)?;
            on_started(&request_id)?;
            // Pause may have raced between ensure and on_started; re-check once registered.
            ensure_session_active(store, session_id)?;
            let request = GenerationRequest::new(request_id.clone(), profile, snapshot, options)
                .and_then(|request| request.with_expected_render(rendered.clone()))
                .map_err(HarnessError::from)?;

            let first_token_deadline = Instant::now()
                + if rendered.token_count() > 8192 {
                    LONG_PREFILL_TIMEOUT
                } else {
                    TURN_TIMEOUT
                };
            let generation = block_on_until(self.engine.generate(request), first_token_deadline)?
                .map_err(HarnessError::from)?;
            let mut stream = generation.into_events();
            let mut assistant = String::new();
            let mut completed = false;

            let mut first_token_pending = true;
            while let Some(event) = block_on_until(
                poll_fn(|cx| stream.as_mut().poll_next(cx)),
                if first_token_pending {
                    first_token_deadline
                } else {
                    Instant::now() + TURN_TIMEOUT
                },
            )? {
                if matches!(&event, GenerationEvent::TextDelta { .. }) {
                    first_token_pending = false;
                }
                // The text-only supervisor must not forward tool proposals to
                // callbacks or persist this turn as successfully completed.
                if matches!(
                    &event,
                    GenerationEvent::ToolCallProposed { .. }
                        | GenerationEvent::ToolCallBatchProposed { .. }
                        | GenerationEvent::ToolCallDelta { .. }
                ) {
                    return Err(HarnessError::Engine(
                        "tool-call proposals require a tool-aware supervisor".into(),
                    ));
                }
                on_event(&event)?;
                match event {
                    GenerationEvent::TextDelta { text } => assistant.push_str(&text),
                    GenerationEvent::GenerationCompleted { .. } => {
                        completed = true;
                        break;
                    }
                    GenerationEvent::RequestFailed { error, .. } => {
                        return Err(error.into());
                    }
                    GenerationEvent::ToolCallProposed { .. }
                    | GenerationEvent::ToolCallBatchProposed { .. }
                    | GenerationEvent::ToolCallDelta { .. } => {
                        return Err(HarnessError::Engine(
                            "tool-call proposals require a tool-aware supervisor".into(),
                        ));
                    }
                    GenerationEvent::RuntimeWarning { .. } => {}
                }
            }

            if !completed {
                return Err(HarnessError::Engine(
                    "generation stream ended before successful terminal event".into(),
                ));
            }
            if assistant.is_empty() {
                return Err(HarnessError::Engine(
                    "generation completed without assistant text".into(),
                ));
            }

            {
                let mut guard = lock_store(store)?;
                let session = guard.get_session(session_id)?.ok_or_else(|| {
                    HarnessError::Invalid(format!("unknown session: {session_id}"))
                })?;
                if session.status != SessionStatus::Active {
                    return Err(HarnessError::Engine(format!(
                        "generation cancelled: session is not active: {session_id}"
                    )));
                }
                guard.append_event(
                    session_id,
                    EventKind::AssistantMessage,
                    serde_json::json!({
                        "content": assistant,
                        "role": MessageRole::Assistant.as_str()
                    }),
                    serde_json::json!([]),
                )?;
                guard.append_event(
                    session_id,
                    EventKind::RunCompleted,
                    serde_json::json!({ "input_tokens": rendered.token_count() }),
                    serde_json::json!([]),
                )?;
            }

            self.telemetry.record_turn();
            self.telemetry.request_flush();

            Ok(TurnOutcome {
                session_id: session_id.to_owned(),
                assistant_text: assistant,
                input_tokens: rendered.token_count(),
                request_id,
            })
        })();
        if let Err(error) = &result {
            let mut guard = lock_store(store)?;
            let events = guard.list_events(session_id)?;
            if !events.iter().any(|event| {
                event.seq > turn_seq
                    && matches!(event.kind, EventKind::RunCompleted | EventKind::RunPaused)
            }) {
                guard.append_event(
                    session_id,
                    EventKind::RunCompleted,
                    serde_json::json!({"status":"failed","error":error.to_string()}),
                    serde_json::json!([]),
                )?;
            }
        }
        result
    }
}

fn lock_store(
    store: &Mutex<SessionStore>,
) -> HarnessResult<std::sync::MutexGuard<'_, SessionStore>> {
    store
        .lock()
        .map_err(|_| HarnessError::Store("store lock poisoned".into()))
}

fn ensure_session_active(store: &Mutex<SessionStore>, session_id: &str) -> HarnessResult<()> {
    let guard = lock_store(store)?;
    let session = guard
        .get_session(session_id)?
        .ok_or_else(|| HarnessError::Invalid(format!("unknown session: {session_id}")))?;
    if session.status != SessionStatus::Active {
        return Err(HarnessError::Invalid(format!(
            "session is not active: {session_id}"
        )));
    }
    Ok(())
}

struct Notify(std::sync::mpsc::Sender<()>);

impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.send(());
    }
}

fn block_on<T>(future: impl Future<Output = T>) -> HarnessResult<T> {
    block_on_until(future, Instant::now() + TURN_TIMEOUT)
}

fn block_on_until<T>(future: impl Future<Output = T>, deadline: Instant) -> HarnessResult<T> {
    let mut future = Box::pin(future);
    let (sender, receiver) = std::sync::mpsc::channel();
    let waker = Waker::from(Arc::new(Notify(sender)));
    let mut context = TaskContext::from_waker(&waker);

    loop {
        if Instant::now() >= deadline {
            return Err(HarnessError::Engine(
                "timed out waiting for engine future".into(),
            ));
        }
        let polled = future.as_mut().poll(&mut context);
        if Instant::now() >= deadline {
            return Err(HarnessError::Engine(
                "timed out waiting for engine future".into(),
            ));
        }
        match polled {
            Poll::Ready(value) => return Ok(value),
            Poll::Pending => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(HarnessError::Engine(
                        "timed out waiting for engine future".into(),
                    ));
                }
                match receiver.recv_timeout(remaining) {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        thread::yield_now();
                    }
                }
            }
        }
    }
}

// This is tracing/cancellation identity, never a KV cache identity or authority.
fn round_request_id(session_id: &str, turn_seq: i64, round: u64) -> HarnessResult<RequestId> {
    use sha2::{Digest, Sha256};
    if turn_seq <= 0 {
        return Err(HarnessError::Invalid(
            "turn sequence must be positive".into(),
        ));
    }
    let length = u64::try_from(session_id.len())
        .map_err(|_| HarnessError::Invalid("session identity length overflow".into()))?;
    let mut digest = Sha256::new();
    digest.update(b"alloy.w6.generation.v1\0");
    digest.update(length.to_be_bytes());
    digest.update(session_id.as_bytes());
    digest.update(turn_seq.to_be_bytes());
    digest.update(round.to_be_bytes());
    RequestId::new(format!("gen-{:x}", digest.finalize())).map_err(HarnessError::from)
}

#[cfg(test)]
mod tool_loop_request_identity_tests {
    use super::*;
    #[test]
    fn round_identity_binds_session_durable_sequence_and_round() {
        let first = round_request_id("session", 2, 0).unwrap();
        for other in [
            round_request_id("other", 2, 0).unwrap(),
            round_request_id("session", 3, 0).unwrap(),
            round_request_id("session", 2, 1).unwrap(),
        ] {
            assert_ne!(first, other);
        }
        assert_eq!(first, round_request_id("session", 2, 0).unwrap());
        assert!(round_request_id("session", 0, 0).is_err());
        assert!(round_request_id("session", -1, 0).is_err());
        assert_eq!(
            round_request_id(&"long / unicode-é".repeat(100), 2, u64::MAX)
                .unwrap()
                .as_str()
                .len(),
            68
        );
    }
}

#[cfg(test)]
mod prefill_deadline_tests {
    use super::*;
    #[test]
    fn first_token_wait_does_not_renew_deadline_on_wake() {
        let deadline = Instant::now() + Duration::from_millis(5);
        let mut polls = 0;
        let result: HarnessResult<()> = block_on_until(
            poll_fn(|cx| {
                polls += 1;
                cx.waker().wake_by_ref();
                Poll::Pending
            }),
            deadline,
        );
        assert!(polls > 1);
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }
    #[test]
    fn ready_future_cannot_bypass_expired_deadline() {
        assert!(block_on_until(std::future::ready(7), Instant::now()).is_err());
    }
    #[test]
    fn completed_future_returns_within_explicit_prefill_deadline() {
        assert_eq!(
            block_on_until(std::future::ready(7), Instant::now() + LONG_PREFILL_TIMEOUT).unwrap(),
            7
        );
    }
}
