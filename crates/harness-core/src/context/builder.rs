//! Context builder: session history → `ContextSnapshot` → engine render_and_count.

use crate::error::{HarnessError, HarnessResult};
use crate::store::{EventKind, EventRecord};
use engine::api::Engine;
use engine::types::{
    ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId, RenderedContext,
};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll, Wake, Waker};
use std::thread;
use std::time::{Duration, Instant};

#[path = "tool_history.rs"]
mod tool_history;

const RENDER_TIMEOUT: Duration = Duration::from_secs(30);

/// Builds engine context snapshots from durable session events.
#[derive(Debug, Default)]
pub struct ContextBuilder;

impl ContextBuilder {
    /// Reconstruct chat messages from append-only session events.
    ///
    /// Text-only reconstruction rejects tool lifecycle history; use
    /// `build_tool_snapshot` to preserve linked tool context instead.
    pub fn messages_from_events(events: &[EventRecord]) -> HarnessResult<Vec<Message>> {
        let mut messages = Vec::new();
        for event in events {
            let role = match event.kind {
                EventKind::UserMessage => MessageRole::User,
                EventKind::AssistantMessage => MessageRole::Assistant,
                EventKind::ToolRequested
                | EventKind::ToolApproved
                | EventKind::ToolStarted
                | EventKind::ToolCompleted => {
                    return Err(HarnessError::Invalid(
                        "tool events require build_tool_snapshot".into(),
                    ));
                }
                _ => continue,
            };
            let content = event
                .payload
                .get("content")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    HarnessError::Invalid(format!(
                        "event seq {} missing string payload.content",
                        event.seq
                    ))
                })?;
            messages.push(Message {
                role,
                content: content.to_owned(),
            });
        }
        Ok(messages)
    }

    /// Build a nonempty snapshot for `profile_id`, optionally appending a new user turn.
    pub fn build_snapshot(
        snapshot_id: impl AsRef<str>,
        profile_id: impl AsRef<str>,
        history: &[EventRecord],
        new_user_message: Option<&str>,
    ) -> HarnessResult<ContextSnapshot> {
        let mut messages = Self::messages_from_events(history)?;
        if let Some(content) = new_user_message {
            if content.is_empty() {
                return Err(HarnessError::Invalid(
                    "new user message must be non-empty".into(),
                ));
            }
            messages.push(Message {
                role: MessageRole::User,
                content: content.to_owned(),
            });
        }
        if messages.is_empty() {
            return Err(HarnessError::Invalid(
                "context snapshot requires at least one message".into(),
            ));
        }
        let snapshot_id = ContextSnapshotId::new(snapshot_id.as_ref())
            .map_err(|error| HarnessError::Invalid(format!("snapshot_id: {error}")))?;
        let profile_id = ModelProfileId::new(profile_id.as_ref())
            .map_err(|error| HarnessError::Invalid(format!("profile_id: {error}")))?;
        ContextSnapshot::new(snapshot_id, profile_id, messages).map_err(HarnessError::from)
    }

    /// Render and tokenize through the engine contract (exact HF counts).
    pub fn render_and_count<E: Engine + ?Sized>(
        engine: &E,
        snapshot: ContextSnapshot,
    ) -> HarnessResult<RenderedContext> {
        block_on(engine.render_and_count(snapshot))?.map_err(HarnessError::from)
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

fn block_on<T>(future: impl Future<Output = T>) -> Result<T, HarnessError> {
    let mut future = Box::pin(future);
    let (sender, receiver) = std::sync::mpsc::channel();
    let waker = Waker::from(Arc::new(Notify(sender)));
    let mut context = TaskContext::from_waker(&waker);
    let deadline = Instant::now() + RENDER_TIMEOUT;

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
