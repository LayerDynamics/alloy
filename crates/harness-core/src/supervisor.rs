//! Phase-1 supervisor skeleton: user → render → generate → persist.

use crate::context::ContextBuilder;
use crate::error::{HarnessError, HarnessResult};
use crate::store::{EventKind, SessionStore};
use crate::telemetry::TelemetryClient;
use engine::api::Engine;
use engine::types::{
    GenerationEvent, GenerationOptions, GenerationRequest, MessageRole, ModelProfileId, RequestId,
};
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::Arc;
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

    /// Create a session row and record telemetry.
    pub fn create_session(
        &self,
        store: &mut SessionStore,
        session_id: &str,
        profile_id: &str,
    ) -> HarnessResult<()> {
        store.create_session(session_id, profile_id)?;
        self.telemetry.record_session();
        Ok(())
    }

    /// Run one user turn: persist → render → generate → persist assistant.
    ///
    /// Telemetry flush failures are ignored after recording the turn counter so
    /// they never fail the session path.
    pub fn run_turn(
        &self,
        store: &mut SessionStore,
        session_id: &str,
        user_text: &str,
        max_output_tokens: usize,
    ) -> HarnessResult<TurnOutcome> {
        if user_text.is_empty() {
            return Err(HarnessError::Invalid("user text must be non-empty".into()));
        }
        let session = store
            .get_session(session_id)?
            .ok_or_else(|| HarnessError::Invalid(format!("unknown session: {session_id}")))?;

        store.append_event(
            session_id,
            EventKind::UserMessage,
            serde_json::json!({ "content": user_text, "role": MessageRole::User.as_str() }),
            serde_json::json!([]),
        )?;
        store.append_event(
            session_id,
            EventKind::RunStarted,
            serde_json::json!({}),
            serde_json::json!([]),
        )?;

        let history = store.list_events(session_id)?;
        let snapshot = ContextBuilder::build_snapshot(
            format!(
                "{session_id}-turn-{}",
                history.last().map(|event| event.seq).unwrap_or(0)
            ),
            &session.profile_id,
            &history,
            None,
        )?;

        let rendered = ContextBuilder::render_and_count(&self.engine, snapshot.clone())?;
        let profile = ModelProfileId::new(session.profile_id.clone())
            .map_err(|error| HarnessError::Invalid(format!("profile_id: {error}")))?;
        let options = GenerationOptions::greedy(max_output_tokens)
            .map_err(|error| HarnessError::Engine(error.to_string()))?;
        let request_id = RequestId::new(format!("{session_id}-req-{}", rendered.token_count()))
            .map_err(|error| HarnessError::Invalid(format!("request_id: {error}")))?;
        let request = GenerationRequest::new(request_id, profile, snapshot, options)
            .map_err(|error| HarnessError::Engine(error.to_string()))?;

        let generation = block_on(self.engine.generate(request))
            .map_err(|error| HarnessError::Engine(error.to_string()))?;
        let mut stream = generation.into_events();
        let mut assistant = String::new();
        let mut failed: Option<String> = None;

        while let Some(event) = next_event(&mut stream) {
            match event {
                GenerationEvent::TextDelta { text } => assistant.push_str(&text),
                GenerationEvent::GenerationCompleted { .. } => break,
                GenerationEvent::RequestFailed { error, .. } => {
                    failed = Some(error.to_string());
                    break;
                }
                GenerationEvent::ToolCallDelta { .. } | GenerationEvent::RuntimeWarning { .. } => {}
            }
        }

        if let Some(message) = failed {
            return Err(HarnessError::Engine(message));
        }
        if assistant.is_empty() {
            return Err(HarnessError::Engine(
                "generation completed without assistant text".into(),
            ));
        }

        store.append_event(
            session_id,
            EventKind::AssistantMessage,
            serde_json::json!({
                "content": assistant,
                "role": MessageRole::Assistant.as_str()
            }),
            serde_json::json!([]),
        )?;
        store.append_event(
            session_id,
            EventKind::RunCompleted,
            serde_json::json!({ "input_tokens": rendered.token_count() }),
            serde_json::json!([]),
        )?;

        self.telemetry.record_turn();
        // Failure-isolated: ignore flush errors.
        let _ = self.telemetry.flush();

        Ok(TurnOutcome {
            session_id: session_id.to_owned(),
            assistant_text: assistant,
            input_tokens: rendered.token_count(),
        })
    }
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

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut future = Box::pin(future);
    let (sender, receiver) = std::sync::mpsc::channel();
    let waker = Waker::from(Arc::new(Notify(sender)));
    let mut context = TaskContext::from_waker(&waker);
    let deadline = Instant::now() + TURN_TIMEOUT;

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    panic!("timed out waiting for engine future");
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
) -> Option<GenerationEvent> {
    block_on(poll_fn(|cx| stream.as_mut().poll_next(cx)))
}
