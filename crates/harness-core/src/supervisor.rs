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
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Wake, Waker};
use std::thread;
use std::time::{Duration, Instant};

const TURN_TIMEOUT: Duration = Duration::from_secs(120);

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
            block_on(self.engine.cancel(request_id.clone()))?
                .map_err(|error| HarnessError::Engine(error.to_string()))?;
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
        block_on(self.engine.cancel(request_id))?
            .map_err(|error| HarnessError::Engine(error.to_string()))
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
        mut on_started: S,
        mut on_event: F,
    ) -> HarnessResult<TurnOutcome>
    where
        S: FnMut(&RequestId) -> HarnessResult<()>,
        F: FnMut(&GenerationEvent) -> HarnessResult<()>,
    {
        if user_text.is_empty() {
            return Err(HarnessError::Invalid("user text must be non-empty".into()));
        }

        let (profile_id, snapshot, input_tokens_hint) = {
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
            guard.append_event(
                session_id,
                EventKind::RunStarted,
                serde_json::json!({}),
                serde_json::json!([]),
            )?;

            let history = guard.list_events(session_id)?;
            let snapshot = ContextBuilder::build_snapshot(
                format!(
                    "{session_id}-turn-{}",
                    history.last().map(|event| event.seq).unwrap_or(0)
                ),
                &session.profile_id,
                &history,
                None,
            )?;
            (session.profile_id.clone(), snapshot, history.len())
        };
        let _ = input_tokens_hint;

        let rendered = ContextBuilder::render_and_count(&self.engine, snapshot.clone())?;
        // Re-check Active after render so a concurrent pause aborts before generate.
        ensure_session_active(store, session_id)?;
        let profile = ModelProfileId::new(profile_id.clone())
            .map_err(|error| HarnessError::Invalid(format!("profile_id: {error}")))?;
        let options = GenerationOptions::greedy(max_output_tokens)
            .map_err(|error| HarnessError::Engine(error.to_string()))?;
        let request_id = RequestId::new(format!("{session_id}-req-{}", rendered.token_count()))
            .map_err(|error| HarnessError::Invalid(format!("request_id: {error}")))?;
        on_started(&request_id)?;
        // Pause may have raced between ensure and on_started; re-check once registered.
        ensure_session_active(store, session_id)?;
        let request = GenerationRequest::new(request_id.clone(), profile, snapshot, options)
            .map_err(|error| HarnessError::Engine(error.to_string()))?;

        let generation = block_on(self.engine.generate(request))?
            .map_err(|error| HarnessError::Engine(error.to_string()))?;
        let mut stream = generation.into_events();
        let mut assistant = String::new();
        let mut failed: Option<String> = None;
        let mut cancelled = false;

        while let Some(event) = next_event(&mut stream)? {
            on_event(&event)?;
            match &event {
                GenerationEvent::TextDelta { text } => assistant.push_str(text),
                GenerationEvent::GenerationCompleted { .. } => break,
                GenerationEvent::RequestFailed { error, .. } => {
                    let message = error.to_string();
                    if message.to_ascii_lowercase().contains("cancel") {
                        cancelled = true;
                    }
                    failed = Some(message);
                    break;
                }
                GenerationEvent::ToolCallDelta { .. } | GenerationEvent::RuntimeWarning { .. } => {}
            }
        }

        if cancelled {
            return Err(HarnessError::Engine(
                failed.unwrap_or_else(|| "generation cancelled".into()),
            ));
        }
        if let Some(message) = failed {
            return Err(HarnessError::Engine(message));
        }
        if assistant.is_empty() {
            return Err(HarnessError::Engine(
                "generation completed without assistant text".into(),
            ));
        }

        {
            let mut guard = lock_store(store)?;
            let session = guard
                .get_session(session_id)?
                .ok_or_else(|| HarnessError::Invalid(format!("unknown session: {session_id}")))?;
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
        let _ = self.telemetry.flush();

        Ok(TurnOutcome {
            session_id: session_id.to_owned(),
            assistant_text: assistant,
            input_tokens: rendered.token_count(),
            request_id,
        })
    }
}

fn lock_store(store: &Mutex<SessionStore>) -> HarnessResult<std::sync::MutexGuard<'_, SessionStore>> {
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
    let mut future = Box::pin(future);
    let (sender, receiver) = std::sync::mpsc::channel();
    let waker = Waker::from(Arc::new(Notify(sender)));
    let mut context = TaskContext::from_waker(&waker);
    let deadline = Instant::now() + TURN_TIMEOUT;

    loop {
        match future.as_mut().poll(&mut context) {
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

fn next_event(
    stream: &mut Pin<Box<dyn engine::types::GenerationEventStream>>,
) -> HarnessResult<Option<GenerationEvent>> {
    block_on(poll_fn(|cx| stream.as_mut().poll_next(cx)))
}
