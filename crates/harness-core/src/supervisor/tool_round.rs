//! One bounded, opt-in generation round. Proposals remain untrusted data.
//! This is not admission, approval, execution or a completed user-turn loop.
use super::{Notify, Supervisor, ensure_session_active, lock_store, round_request_id};
use crate::{
    ContextBuilder, EventKind, EventRecord, HarnessError, HarnessResult, SessionStatus,
    SessionStore,
};
use engine::{
    api::Engine,
    parser_allocation::FrozenParserAllocation,
    types::{
        FinishReason, GenerationEvent, GenerationOptions, GenerationRequest, GenerationStream,
        RequestId, ToolCallProposal, ToolContextLimits, Usage,
    },
};
use std::{
    future::{Future, poll_fn},
    io::{self, Write},
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};
use tool_registry::registry::ToolRegistry;

/// Caller-owned durable turn and next serial round. Never take these from model output.
#[derive(Clone, Copy, Debug)]
pub struct ToolRoundRequest<'a> {
    /// Session owned by the calling turn coordinator.
    pub session_id: &'a str,
    /// Sequence of that session's latest still-open run.started event.
    pub turn_seq: i64,
    /// Zero-based round; the coordinator must advance it and never reuse an ID.
    pub round: u64,
}
/// Explicit acceptance ceilings, not defaults, memory reservations or authority.
#[derive(Clone, Copy, Debug)]
pub struct ToolRoundConfig {
    /// Existing model-facing schema/history limits, also checked by the engine.
    pub context_limits: ToolContextLimits,
    /// Existing engine generation output-token allowance.
    pub max_output_tokens: usize,
    /// Exclusive upper bound on the zero-based round index.
    pub max_rounds: u64,
    /// Maximum retained complete proposals.
    pub max_proposals: usize,
    /// Maximum combined UTF-8 name and serialized argument bytes.
    pub max_proposal_bytes: usize,
    /// Maximum retained slab reservation, including scratch, not only used payload.
    pub max_slab_bytes: u64,
    /// Maximum buffered assistant UTF-8 bytes.
    pub max_text_bytes: usize,
    /// Maximum cumulative warning code/message UTF-8 bytes.
    pub max_warning_bytes: usize,
    /// Maximum polled generation events, including warnings and terminal.
    pub max_events: usize,
    /// Absolute caller-owned deadline; never renewed between events or rounds.
    pub deadline: Instant,
}
/// Owned untrusted proposals. Retaining a slab retains its actual reservation.
/// No variant is a W3 authorization or W4 execution receipt.
#[derive(Debug)]
pub enum RoundProposals {
    /// Successful generation with no proposed calls.
    None,
    /// Compatibility proposals moved from the engine without copying their trees.
    Legacy(Vec<ToolCallProposal>),
    /// The original frozen allocation, including its admission reservation.
    Slab(FrozenParserAllocation),
}
impl RoundProposals {
    /// Number of complete proposals. This does not resolve or authorize them.
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Legacy(calls) => calls.len(),
            Self::Slab(owner) => owner.proposal_count(),
        }
    }
    /// Whether the generation proposed no calls.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
/// Successful engine-round outcome, NOT a terminal user-turn outcome.
#[derive(Debug)]
pub struct ToolRoundOutcome {
    /// Exact generated/cancellable request identity for this round.
    pub request_id: RequestId,
    /// Text emitted before proposals, under the configured byte ceiling.
    pub assistant_text: String,
    /// Exact input token count from the existing engine rendering path.
    pub input_tokens: usize,
    /// Observed successful engine stop reason.
    pub finish_reason: FinishReason,
    /// Usage from that successful terminal event.
    pub usage: Usage,
    /// Untrusted proposals to validate/admit later; possession permits no effects.
    pub proposals: RoundProposals,
}
impl<E: Engine> Supervisor<E> {
    /// Generate one round from canonical durable tool context without executing calls.
    ///
    /// The caller must own the session's single active turn, advance round indices
    /// without reuse, and retain its absolute turn deadline. This method checks
    /// the latest open durable turn but does not itself reserve the daemon's slot.
    /// No store lock is held during rendering, engine polling or either callback.
    /// `on_event` receives only text/warnings, never proposals or engine terminals.
    /// The outer coordinator owns the one terminal user-turn outcome.
    ///
    /// Limits/deadline are checked around cooperative polls and callbacks; arbitrary
    /// blocking engine code/callbacks cannot be forcibly preempted by this API.
    /// Errors drop retained proposals/streams and attempt cancellation of this exact
    /// request. That best-effort request is not proof of native/process reaping.
    pub fn generate_tool_round<S, F>(
        &self,
        store: &Mutex<SessionStore>,
        registry: &ToolRegistry,
        request: ToolRoundRequest<'_>,
        config: ToolRoundConfig,
        mut on_started: S,
        mut on_event: F,
    ) -> HarnessResult<ToolRoundOutcome>
    where
        S: FnMut(&RequestId) -> HarnessResult<()>,
        F: FnMut(&GenerationEvent) -> HarnessResult<()>,
    {
        validate_config(config, request)?;
        let (profile_id, history) = read_open_turn(store, request)?;
        let request_id = round_request_id(request.session_id, request.turn_seq, request.round)?;
        let snapshot = ContextBuilder::build_tool_snapshot(
            request_id.as_str(),
            &profile_id,
            request.session_id,
            &history,
            registry,
            config.context_limits,
        )?;
        let check = || check_live(store, request.session_id, config.deadline);
        let rendered = wait_checked(
            self.engine.render_and_count(snapshot.clone()),
            config.deadline,
            &check,
        )?
        .map_err(HarnessError::from)?;
        check()?;
        let options =
            GenerationOptions::greedy(config.max_output_tokens).map_err(HarnessError::from)?;
        let generation_request = GenerationRequest::new(
            request_id.clone(),
            snapshot.profile_id().clone(),
            snapshot,
            options,
        )
        .map_err(HarnessError::from)?;
        // Re-check after registering the exact cancellation ID; no lock crosses the callback.
        let result = (|| {
            on_started(&request_id)?;
            check()?;
            let generation = wait_checked(
                self.engine.generate(generation_request),
                config.deadline,
                &check,
            )?
            .map_err(HarnessError::from)?;
            if generation.request_id() != &request_id {
                return Err(invalid("engine returned a different generation identity"));
            }
            let stream = generation.into_events();
            let collected = collect(stream, config, &check, &mut on_event)?;
            // A pause/resume or terminal/new turn must not publish old proposals.
            read_open_turn(store, request)?;
            check()?;
            Ok(ToolRoundOutcome {
                request_id: request_id.clone(),
                assistant_text: collected.text,
                input_tokens: rendered.token_count(),
                finish_reason: collected.finish_reason,
                usage: collected.usage,
                proposals: collected.proposals,
            })
        })();
        if result.is_err() {
            // The stream/collected owner has already dropped. Preserve the primary error.
            // This bounded best-effort cancel targets only our expected ID, never a foreign handle ID.
            let cleanup_deadline = Instant::now()
                .checked_add(Duration::from_millis(100))
                .unwrap_or_else(Instant::now);
            let _ = wait_checked(self.engine.cancel(request_id), cleanup_deadline, &|| Ok(()));
        }
        result
    }
}
fn invalid(message: &str) -> HarnessError {
    HarnessError::Engine(message.into())
}
fn validate_config(config: ToolRoundConfig, request: ToolRoundRequest<'_>) -> HarnessResult<()> {
    if [
        config.max_output_tokens,
        config.max_proposals,
        config.max_proposal_bytes,
        config.max_text_bytes,
        config.max_warning_bytes,
        config.max_events,
        config.context_limits.max_tools,
        config.context_limits.max_history_blocks,
        config.context_limits.max_json_bytes,
    ]
    .contains(&0)
        || config.max_slab_bytes == 0
        || config.max_rounds == 0
        || request.round >= config.max_rounds
        || request.turn_seq <= 0
    {
        return Err(invalid(
            "positive explicit limits and an in-range round are required",
        ));
    }
    if Instant::now() >= config.deadline {
        return Err(invalid("tool round deadline expired"));
    }
    Ok(())
}
fn read_open_turn(
    store: &Mutex<SessionStore>,
    request: ToolRoundRequest<'_>,
) -> HarnessResult<(String, Vec<EventRecord>)> {
    let guard = lock_store(store)?;
    let session = guard
        .get_session(request.session_id)?
        .ok_or_else(|| invalid("unknown session"))?;
    if session.status != SessionStatus::Active {
        return Err(invalid("session is not active"));
    }
    let history = guard.list_events(request.session_id)?;
    let latest = history
        .iter()
        .rev()
        .find(|event| event.kind == EventKind::RunStarted);
    if latest.map(|event| event.seq) != Some(request.turn_seq)
        || history.iter().any(|event| {
            event.seq > request.turn_seq
                && matches!(event.kind, EventKind::RunCompleted | EventKind::RunPaused)
        })
    {
        return Err(invalid(
            "round must belong to the latest still-open durable turn",
        ));
    }
    Ok((session.profile_id, history))
}
fn check_live(store: &Mutex<SessionStore>, session: &str, deadline: Instant) -> HarnessResult<()> {
    if Instant::now() >= deadline {
        return Err(invalid("tool round deadline expired"));
    }
    ensure_session_active(store, session)
}
fn wait_checked<T>(
    future: impl Future<Output = T>,
    deadline: Instant,
    check: &impl Fn() -> HarnessResult<()>,
) -> HarnessResult<T> {
    let mut future = Box::pin(future);
    let (sender, receiver) = std::sync::mpsc::channel();
    let waker = Waker::from(Arc::new(Notify(sender)));
    let mut context = Context::from_waker(&waker);
    loop {
        check()?;
        if Instant::now() >= deadline {
            return Err(invalid("tool round deadline expired"));
        }
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => {
                check()?;
                if Instant::now() >= deadline {
                    return Err(invalid("tool round deadline expired"));
                }
                return Ok(value);
            }
            Poll::Pending => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(invalid("tool round deadline expired"));
                }
                // Recheck pause even if a broken producer never wakes us.
                let _ = receiver.recv_timeout(remaining.min(Duration::from_millis(20)));
            }
        }
    }
}
struct Collected {
    text: String,
    proposals: RoundProposals,
    finish_reason: FinishReason,
    usage: Usage,
}
fn collect(
    mut stream: GenerationStream,
    config: ToolRoundConfig,
    check: &impl Fn() -> HarnessResult<()>,
    on_event: &mut impl FnMut(&GenerationEvent) -> HarnessResult<()>,
) -> HarnessResult<Collected> {
    let mut text = String::new();
    let mut proposals = RoundProposals::None;
    let mut proposal_bytes = 0usize;
    let mut warning_bytes = 0usize;
    let mut event_count = 0usize;
    loop {
        let event = wait_checked(
            poll_fn(|cx| stream.as_mut().poll_next(cx)),
            config.deadline,
            check,
        )?
        .ok_or_else(|| invalid("generation stream ended before successful terminal event"))?;
        event_count = add(
            event_count,
            1,
            config.max_events,
            "generation event count exceeded",
        )?;
        match event {
            GenerationEvent::TextDelta { text: delta } => {
                if !proposals.is_empty() {
                    return Err(invalid(
                        "text after complete proposals violates event ordering",
                    ));
                }
                add(
                    text.len(),
                    delta.len(),
                    config.max_text_bytes,
                    "assistant byte limit exceeded",
                )?;
                text.try_reserve_exact(delta.len())
                    .map_err(|_| invalid("assistant buffer allocation failed"))?;
                let event = GenerationEvent::TextDelta { text: delta };
                on_event(&event)?;
                check()?;
                if let GenerationEvent::TextDelta { text: delta } = event {
                    text.push_str(&delta);
                }
            }
            GenerationEvent::RuntimeWarning { code, message } => {
                warning_bytes = add(
                    warning_bytes,
                    code.len(),
                    config.max_warning_bytes,
                    "warning byte limit exceeded",
                )?;
                warning_bytes = add(
                    warning_bytes,
                    message.len(),
                    config.max_warning_bytes,
                    "warning byte limit exceeded",
                )?;
                on_event(&GenerationEvent::RuntimeWarning { code, message })?;
                check()?;
            }
            GenerationEvent::ToolCallDelta { .. } => {
                return Err(invalid("raw tool deltas are not complete proposals"));
            }
            GenerationEvent::ToolCallProposed { proposal } => {
                if matches!(proposals, RoundProposals::Slab(_)) {
                    return Err(invalid("mixed legacy/slab proposal delivery"));
                }
                add(
                    proposals.len(),
                    1,
                    config.max_proposals,
                    "proposal count exceeded",
                )?;
                proposal_bytes = add(
                    proposal_bytes,
                    proposal.name().len(),
                    config.max_proposal_bytes,
                    "proposal byte limit exceeded",
                )?;
                let mut counter = Counter {
                    used: proposal_bytes,
                    limit: config.max_proposal_bytes,
                };
                serde_json::to_writer(&mut counter, proposal.arguments())
                    .map_err(|_| invalid("proposal byte limit exceeded"))?;
                proposal_bytes = counter.used;
                if matches!(proposals, RoundProposals::None) {
                    proposals = RoundProposals::Legacy(Vec::new());
                }
                if let RoundProposals::Legacy(calls) = &mut proposals {
                    calls
                        .try_reserve_exact(1)
                        .map_err(|_| invalid("proposal descriptor allocation failed"))?;
                    calls.push(proposal);
                }
            }
            GenerationEvent::ToolCallBatchProposed { proposals: owner } => {
                if !matches!(proposals, RoundProposals::None) {
                    return Err(invalid("duplicate or mixed slab proposal delivery"));
                }
                if owner.proposal_count() == 0
                    || owner.proposal_count() > config.max_proposals
                    || owner.reserved_bytes() > config.max_slab_bytes
                {
                    return Err(invalid("slab count or retained reservation exceeds limit"));
                }
                for index in 0..owner.proposal_count() {
                    let view = owner
                        .proposal(index)
                        .ok_or_else(|| invalid("missing slab proposal descriptor"))?;
                    proposal_bytes = add(
                        proposal_bytes,
                        view.name.len(),
                        config.max_proposal_bytes,
                        "proposal byte limit exceeded",
                    )?;
                    proposal_bytes = add(
                        proposal_bytes,
                        view.arguments_json.len(),
                        config.max_proposal_bytes,
                        "proposal byte limit exceeded",
                    )?;
                }
                // Move the original owner; never decode/clone slab arguments here.
                proposals = RoundProposals::Slab(owner);
            }
            GenerationEvent::RequestFailed { error, .. } => return Err(error.into()),
            GenerationEvent::GenerationCompleted {
                finish_reason,
                usage,
            } => {
                check()?;
                if text.is_empty() && proposals.is_empty() {
                    return Err(invalid("empty successful generation"));
                }
                if !proposals.is_empty() && finish_reason != FinishReason::EndOfSequence {
                    return Err(invalid(
                        "output-limited proposals cannot advance to admission",
                    ));
                }
                return Ok(Collected {
                    text,
                    proposals,
                    finish_reason,
                    usage,
                });
            }
        }
    }
}
fn add(used: usize, amount: usize, limit: usize, message: &str) -> HarnessResult<usize> {
    used.checked_add(amount)
        .filter(|sum| *sum <= limit)
        .ok_or_else(|| invalid(message))
}
struct Counter {
    used: usize,
    limit: usize,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.used = self
            .used
            .checked_add(bytes.len())
            .filter(|sum| *sum <= self.limit)
            .ok_or_else(|| io::Error::other("proposal byte limit exceeded"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
