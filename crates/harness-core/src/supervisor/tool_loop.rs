//! Bounded canonical W6 coordinator. Only durable W3 receipts reach W4 dispatch.
use super::*;
use crate::{
    artifact::ArtifactStore,
    plugin_host::{Decision, HookInvocation, HookReceipt, PluginHost},
    policy::DaemonPolicy,
    store::IntentAdmission,
    tool_execution::ReceiptExecutor,
};
use serde_json::{Value, json};
use tool_registry::{
    call::{ToolCall, ToolCallContext, ToolInvocation},
    execution::CancellationToken,
    result::{InlineToolResult, ToolResult},
};

/// Immutable execution catalogue and policy, with separately owned plugin processes.
pub struct ToolRuntime {
    /// Exact registry and its existing scoped W4 implementations.
    pub executor: ReceiptExecutor,
    /// Daemon-owned exact-argument policy.
    pub policy: DaemonPolicy,
    /// Durable bounded output storage.
    pub artifacts: ArtifactStore,
    /// Ordered W5 plugin host. Its mutex never encloses store or execution work.
    pub plugins: Mutex<PluginHost>,
}
/// Limits for a whole user turn; the round deadline is shared, never renewed.
#[derive(Clone, Copy)]
pub struct ToolLoopConfig {
    /// Existing generation and context ceilings.
    pub round: ToolRoundConfig,
    /// Total calls across all rounds.
    pub max_calls: usize,
    /// Inline output threshold, independent of approved output ceiling.
    pub inline_bytes: usize,
}
/// Streaming supervisor event. Engine round terminals never escape as turn terminals.
pub struct ToolLoopEvent {
    /// Existing wire event name.
    pub event: &'static str,
    /// Structured data, never authority.
    pub body: Value,
}
impl<E: Engine> Supervisor<E> {
    /// Run a complete user turn. The caller owns the session's exclusive turn slot.
    /// Store/session/native locks must not enclose this call; waits and effects borrow none.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tool_turn<S, F>(
        &self,
        store: &Mutex<SessionStore>,
        runtime: &ToolRuntime,
        session: &str,
        user: &str,
        config: ToolLoopConfig,
        cancel: &CancellationToken,
        started: S,
        emit: F,
    ) -> HarnessResult<TurnOutcome>
    where
        S: FnMut(&RequestId) -> HarnessResult<()>,
        F: FnMut(ToolLoopEvent) -> HarnessResult<()>,
    {
        self.run_tool_turn_with_lifecycle(
            store,
            runtime,
            session,
            user,
            config,
            cancel,
            started,
            |_| Ok(()),
            emit,
        )
    }

    /// Run a turn while notifying the caller when each generation has finished.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tool_turn_with_lifecycle<S, D, F>(
        &self,
        store: &Mutex<SessionStore>,
        runtime: &ToolRuntime,
        session: &str,
        user: &str,
        config: ToolLoopConfig,
        cancel: &CancellationToken,
        mut started: S,
        mut finished: D,
        mut emit: F,
    ) -> HarnessResult<TurnOutcome>
    where
        S: FnMut(&RequestId) -> HarnessResult<()>,
        D: FnMut(&RequestId) -> HarnessResult<()>,
        F: FnMut(ToolLoopEvent) -> HarnessResult<()>,
    {
        if user.is_empty() || config.max_calls == 0 || config.inline_bytes == 0 {
            return Err(HarnessError::Invalid(
                "nonempty user and positive turn limits required".into(),
            ));
        }
        let turn = {
            let mut guard = lock_store(store)?;
            ensure_active_locked(&guard, session)?;
            guard.append_event(
                session,
                EventKind::UserMessage,
                json!({"content":user,"role":"user"}),
                json!([]),
            )?;
            guard
                .append_event(
                    session,
                    EventKind::RunStarted,
                    json!({"mode":"tools"}),
                    json!([]),
                )?
                .seq
        };
        let mut calls = 0usize;
        let mut text = String::new();
        let mut streamed_bytes = 0usize;
        let mut total_usage = engine::types::Usage::default();
        let mut finish_reason = engine::types::FinishReason::EndOfSequence;
        let result = (|| {
            for round in 0..config.round.max_rounds {
                live(store, session, cancel, config.round.deadline)?;
                let outcome = self.generate_tool_round_cancellable(
                    store,
                    runtime.executor.registry(),
                    ToolRoundRequest {
                        session_id: session,
                        turn_seq: turn,
                        round,
                    },
                    config.round,
                    Some(cancel),
                    |id| {
                        live(store, session, cancel, config.round.deadline)?;
                        started(id)
                    },
                    |event| {
                        live(store, session, cancel, config.round.deadline)?;
                        let (name, body) = match event {
                            GenerationEvent::TextDelta { text } => {
                                let next = streamed_bytes
                                    .checked_add(text.len())
                                    .filter(|bytes| *bytes <= config.round.max_text_bytes)
                                    .ok_or_else(|| {
                                        HarnessError::Engine("turn text limit exceeded".into())
                                    })?;
                                streamed_bytes = next;
                                ("text_delta", json!({"text":text}))
                            }
                            GenerationEvent::RuntimeWarning { code, message } => {
                                ("runtime_warning", json!({"code":code,"message":message}))
                            }
                            _ => {
                                return Err(HarnessError::Engine(
                                    "unexpected round callback".into(),
                                ));
                            }
                        };
                        emit(ToolLoopEvent { event: name, body })
                    },
                )?;
                finished(&outcome.request_id)?;
                total_usage.input_tokens = total_usage
                    .input_tokens
                    .checked_add(outcome.usage.input_tokens)
                    .ok_or_else(|| HarnessError::Engine("usage overflow".into()))?;
                total_usage.output_tokens = total_usage
                    .output_tokens
                    .checked_add(outcome.usage.output_tokens)
                    .ok_or_else(|| HarnessError::Engine("usage overflow".into()))?;
                finish_reason = outcome.finish_reason;
                if outcome.assistant_text.len()
                    > config.round.max_text_bytes.saturating_sub(text.len())
                {
                    return Err(HarnessError::Engine("turn text limit exceeded".into()));
                }
                text.push_str(&outcome.assistant_text);
                if outcome.proposals.is_empty() {
                    live(store, session, cancel, config.round.deadline)?;
                    let mut guard = lock_store(store)?;
                    ensure_active_locked(&guard, session)?;
                    guard.append_event(
                        session,
                        EventKind::AssistantMessage,
                        json!({"content":outcome.assistant_text,"role":"assistant"}),
                        json!([]),
                    )?;
                    guard.append_event(
                        session,
                        EventKind::RunCompleted,
                        json!({"input_tokens":outcome.input_tokens,"status":"completed"}),
                        json!([]),
                    )?;
                    return Ok(TurnOutcome {
                        session_id: session.into(),
                        assistant_text: text.clone(),
                        input_tokens: outcome.input_tokens,
                        request_id: outcome.request_id,
                    });
                }
                if outcome.proposals.len() > config.max_calls.saturating_sub(calls) {
                    return Err(HarnessError::Engine("turn call limit exceeded".into()));
                }
                // Persist text before its calls, preserving exactly the streamed ordering.
                if !outcome.assistant_text.is_empty() {
                    lock_store(store)?.append_event(
                        session,
                        EventKind::AssistantMessage,
                        json!({"content":outcome.assistant_text,"role":"assistant"}),
                        json!([]),
                    )?;
                }
                let proposed: Vec<(String, serde_json::Map<String, Value>)> =
                    match outcome.proposals {
                        RoundProposals::None => Vec::new(),
                        RoundProposals::Legacy(values) => values
                            .into_iter()
                            .map(|v| (v.name().into(), v.arguments().clone()))
                            .collect(),
                        RoundProposals::Slab(owner) => {
                            let mut values = Vec::new();
                            for i in 0..owner.proposal_count() {
                                let view = owner.proposal(i).ok_or_else(|| {
                                    HarnessError::Invalid("missing slab descriptor".into())
                                })?;
                                values.push((
                                    view.name.into(),
                                    serde_json::from_str(view.arguments_json)?,
                                ));
                            }
                            values
                        }
                    };
                // Check the complete proposed batch before persisting any tool
                // intent or invoking a hook/implementation. The projected
                // history is independent of authorization and cannot execute.
                let admitted_inline_bytes = self.preflight_tool_batch(
                    store,
                    runtime,
                    session,
                    turn,
                    &outcome.request_id,
                    calls,
                    &proposed,
                    config,
                    cancel,
                )?;
                for (name, arguments) in proposed {
                    live(store, session, cancel, config.round.deadline)?;
                    let call_id = format!("{}-call-{calls}", outcome.request_id.as_str());
                    calls += 1;
                    let invocation = ToolInvocation {
                        context: ToolCallContext {
                            session_id: session.into(),
                            turn_id: format!("turn-{turn}"),
                            generation_id: outcome.request_id.as_str().into(),
                        },
                        call: ToolCall {
                            call_id: call_id.clone(),
                            tool_id: name,
                            arguments,
                        },
                    };
                    // Invalid calls become linked canonical protocol results, never executable intents.
                    if runtime
                        .executor
                        .registry()
                        .resolve(invocation.clone())
                        .is_err()
                    {
                        let result = ToolResult::Inline(InlineToolResult {
                            call_id: call_id.clone(),
                            is_error: true,
                            content: json!({"kind":"protocol","message":"Unknown tool or invalid arguments"}),
                        });
                        let mut guard = lock_store(store)?;
                        guard.append_event(session,EventKind::ToolRequested,json!({"contract_version":crate::approval::CONTRACT_VERSION,"call_id":call_id,"request":invocation,"state":"ProtocolRejected"}),json!([]))?;
                        guard.append_event(session,EventKind::ToolCompleted,json!({"contract_version":crate::approval::CONTRACT_VERSION,"call_id":call_id,"state":"Completed","result":result}),json!([]))?;
                        drop(guard);
                        emit(ToolLoopEvent {
                            event: "tool_result",
                            body: tool_result_body(&call_id, &result),
                        })?;
                        continue;
                    }
                    let admission = lock_store(store)?.admit_tool(
                        runtime.executor.registry(),
                        &runtime.policy,
                        invocation.clone(),
                    )?;
                    emit(ToolLoopEvent {
                        event: "tool_requested",
                        body: json!({"call_id":call_id,"tool_id":invocation.call.tool_id,"arguments":invocation.call.arguments}),
                    })?;
                    if admission == IntentAdmission::AwaitingApproval {
                        let definition = runtime
                            .executor
                            .registry()
                            .resolve(invocation.clone())
                            .map_err(|e| HarnessError::Invalid(e.to_string()))?
                            .definition()
                            .clone();
                        emit(ToolLoopEvent {
                            event: "approval_requested",
                            body: json!({"call_id":call_id,"tool_id":invocation.call.tool_id,"arguments":invocation.call.arguments,"effect":definition.policy.effect,"summary":summary(&json!({"tool_id":invocation.call.tool_id,"arguments":invocation.call.arguments,"effect":definition.policy.effect}))}),
                        })?;
                        loop {
                            live(store, session, cancel, config.round.deadline)?;
                            let mut guard = lock_store(store)?;
                            guard.expire_tool_intents()?;
                            let state = guard.tool_state(session, &call_id)?;
                            drop(guard);
                            if state.as_deref() != Some("AwaitingApproval") {
                                break;
                            }
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                    let state = lock_store(store)?.tool_state(session, &call_id)?;
                    if state.as_deref() != Some("Ready") {
                        emit_terminal_tool(store, session, &call_id, &mut emit)?;
                        continue;
                    }
                    let handle = lock_store(store)?.claim_tool(session, &call_id)?;
                    let b = handle.binding();
                    let timeout = b.snapshot()["effective_timeout_ms"]
                        .as_u64()
                        .ok_or_else(|| HarnessError::Invalid("missing timeout".into()))?;
                    let deadline = config
                        .round
                        .deadline
                        .min(Instant::now() + Duration::from_millis(timeout));
                    let hook = HookInvocation {
                        invocation_id: b.digest().into(),
                        session_id: session.into(),
                        call_id: call_id.clone(),
                        definition: b.definition().clone(),
                        schema_hash: {
                            use sha2::{Digest, Sha256};
                            format!(
                                "{:x}",
                                Sha256::digest(
                                    crate::approval::canonical_json(&b.definition().input_schema)
                                        .map_err(|e| HarnessError::Invalid(e.to_string()))?
                                )
                            )
                        },
                        arguments: invocation.call.arguments.clone(),
                        approval: b.snapshot().clone(),
                        deadline_unix_ms: b.snapshot()["expires_at_unix_ms"].as_u64().unwrap_or(0),
                    };
                    let (before, mut after_unavailable) = {
                        let mut host = lock_plugin_host(&runtime.plugins, deadline, cancel)?;
                        let after_unavailable = host.after_dispatch_failure_receipts(
                            &hook,
                            "after-hook dispatch unavailable",
                        );
                        (host.before(&hook, deadline, cancel), after_unavailable)
                    };
                    record_hooks(store, session, &call_id, &before.receipts)?;
                    if before.decision == Decision::Veto {
                        lock_store(store)?.reject_tool(handle, "before_hook_veto_or_failure")?;
                        emit_terminal_tool(store, session, &call_id, &mut emit)?;
                        continue;
                    }
                    live(store, session, cancel, config.round.deadline)?;
                    let receipt = lock_store(store)?.start_tool(
                        handle,
                        runtime.executor.registry(),
                        &runtime.policy,
                    )?;
                    emit(ToolLoopEvent {
                        event: "tool_call_started",
                        body: json!({"call_id":call_id,"tool_id":invocation.call.tool_id}),
                    })?;
                    let clock = Instant::now();
                    let result = dispatch(
                        runtime,
                        store,
                        receipt,
                        cancel.clone(),
                        admitted_inline_bytes,
                        config.round.deadline,
                    )?;
                    // After hooks annotate the persisted original result and can never retry it.
                    let after = if after_unavailable.is_empty() {
                        Vec::new()
                    } else {
                        match lock_plugin_host(&runtime.plugins, config.round.deadline, cancel) {
                            Ok(mut host) => host.after(
                                &hook,
                                &result,
                                clock.elapsed().as_millis().min(u64::MAX as u128) as u64,
                                config.round.deadline,
                                cancel,
                            ),
                            Err(error) => {
                                for receipt in &mut after_unavailable {
                                    receipt.diagnostic = Some(error.to_string());
                                }
                                after_unavailable
                            }
                        }
                    };
                    record_hooks(store, session, &call_id, &after)?;
                    emit(ToolLoopEvent {
                        event: "tool_result",
                        body: {
                            let mut body = tool_result_body(&call_id, &result);
                            body["hooks"] = hook_values(&after);
                            body
                        },
                    })?;
                }
            }
            Err(HarnessError::Engine("tool round limit exceeded".into()))
        })();
        match &result {
            Ok(_) => {
                self.telemetry.record_turn();
                self.telemetry.request_flush();
                emit(ToolLoopEvent {
                    event: "generation_completed",
                    body: json!({"finish_reason":match finish_reason{engine::types::FinishReason::EndOfSequence=>"end_of_sequence",engine::types::FinishReason::OutputLimit=>"output_limit"},"usage":{"input_tokens":total_usage.input_tokens,"output_tokens":total_usage.output_tokens}}),
                })?;
            }
            Err(error) => {
                cancel.cancel();
                let mut guard = lock_store(store)?;
                guard.withdraw_tools(session, "turn_failed_or_cancelled")?;
                let history = guard.list_events(session)?;
                if !history.iter().any(|e| {
                    e.seq > turn && matches!(e.kind, EventKind::RunCompleted | EventKind::RunPaused)
                }) {
                    guard.append_event(
                        session,
                        EventKind::RunCompleted,
                        json!({"status":"failed","error":error.to_string()}),
                        json!([]),
                    )?;
                }
                drop(guard);
                let _ = emit(ToolLoopEvent {
                    event: "request_failed",
                    body: json!({"message":error.to_string()}),
                });
            }
        }
        result
    }
}

impl<E: Engine> Supervisor<E> {
    #[allow(clippy::too_many_arguments)]
    fn preflight_tool_batch(
        &self,
        store: &Mutex<SessionStore>,
        runtime: &ToolRuntime,
        session: &str,
        turn: i64,
        request_id: &RequestId,
        prior_calls: usize,
        proposed: &[(String, serde_json::Map<String, Value>)],
        config: ToolLoopConfig,
        cancel: &CancellationToken,
    ) -> HarnessResult<usize> {
        use crate::policy::PolicyDecision;
        use engine::types::ToolHistoryBlock;
        live(store, session, cancel, config.round.deadline)?;
        let (profile, history, epoch) = {
            let guard = lock_store(store)?;
            let profile = guard
                .get_session(session)?
                .ok_or_else(|| HarnessError::Invalid("unknown session".into()))?
                .profile_id;
            (
                profile,
                guard.list_events(session)?,
                guard.daemon_epoch().to_owned(),
            )
        };
        let snapshot = ContextBuilder::build_tool_snapshot(
            request_id.as_str(),
            &profile,
            session,
            &history,
            runtime.executor.registry(),
            config.round.context_limits,
        )?;
        let mut blocks = Vec::with_capacity(proposed.len().saturating_mul(2));
        let mut result_limits = Vec::with_capacity(proposed.len());
        for (index, (name, arguments)) in proposed.iter().enumerate() {
            let number = prior_calls
                .checked_add(index)
                .ok_or_else(|| HarnessError::Engine("projected call count overflow".into()))?;
            let call_id = format!("{}-call-{number}", request_id.as_str());
            let invocation = ToolInvocation {
                context: ToolCallContext {
                    session_id: session.into(),
                    turn_id: format!("turn-{turn}"),
                    generation_id: request_id.as_str().into(),
                },
                call: ToolCall {
                    call_id: call_id.clone(),
                    tool_id: name.clone(),
                    arguments: arguments.clone(),
                },
            };
            // Error and artifact representations need space even for a denied
            // call or a small policy output cap. Accepted inline JSON is bounded
            // by the immutable policy binding's effective output limit.
            let output_limit =
                match runtime
                    .policy
                    .evaluate(runtime.executor.registry(), invocation, &epoch, 0)
                {
                    Ok(
                        PolicyDecision::AllowWithoutPrompt(binding)
                        | PolicyDecision::RequireApproval(binding),
                    ) => binding.snapshot()["effective_max_output_bytes"]
                        .as_u64()
                        .ok_or_else(|| {
                            HarnessError::Invalid("missing projected output limit".into())
                        })?,
                    _ => 0,
                };
            let output_limit = usize::try_from(output_limit).map_err(|_| {
                HarnessError::Engine("projected tool output limit is too large".into())
            })?;
            // Streaming adapters move output larger than the inline threshold
            // to the artifact store. fs.write returns bounded metadata based on
            // its already-known path rather than its input content.
            let streaming = matches!(
                name.as_str(),
                "fs.read" | "fs.list" | "http.fetch" | "shell.run"
            );
            let inline_limit = match name.as_str() {
                "fs.read" | "fs.list" | "http.fetch" | "shell.run" => output_limit,
                "fs.write" => {
                    let path = arguments.get("path").and_then(Value::as_str).unwrap_or("");
                    serde_json::to_vec(path)?
                        .len()
                        .checked_add(128)
                        .ok_or_else(|| {
                            HarnessError::Engine("projected write result overflow".into())
                        })?
                        .min(output_limit)
                }
                _ => output_limit,
            };
            // All current built-in control errors and durable terminal markers
            // fit this allowance, including their bounded call identifier.
            result_limits.push((streaming, inline_limit));
            blocks.push(ToolHistoryBlock::AssistantCall {
                call_id: call_id.clone(),
                tool_id: name.clone(),
                arguments: arguments.clone(),
            });
            blocks.push(ToolHistoryBlock::ToolResult {
                call_id,
                content: String::new(),
                is_error: false,
            });
        }
        let original = snapshot
            .tool_context()
            .ok_or_else(|| HarnessError::Invalid("missing tool context".into()))?;
        let projected = original
            .with_appended_history(blocks, config.round.context_limits)
            .map_err(HarnessError::from)?;
        let base_json_bytes = projected.input_bytes();
        let snapshot = snapshot.with_tool_context(projected);
        let rendered = block_on(self.engine.render_and_count(snapshot.clone()))?
            .map_err(HarnessError::from)?;
        let base_message_bytes = snapshot
            .messages()
            .iter()
            .try_fold(0usize, |n, message| n.checked_add(message.content.len()))
            .ok_or_else(|| HarnessError::Engine("projected input overflow".into()))?;
        let fits = |inline_bytes: usize| -> HarnessResult<bool> {
            let extra = result_limits
                .iter()
                .try_fold(0usize, |total, (streaming, limit)| {
                    let limit = if *streaming {
                        (*limit).min(inline_bytes)
                    } else {
                        *limit
                    };
                    // Results are compact JSON. Each additional quoting layer at
                    // most doubles its bytes; control characters are already escaped.
                    total.checked_add(limit.max(1024).checked_mul(2)?)
                })
                .ok_or_else(|| HarnessError::Engine("projected tool batch overflow".into()))?;
            let json_bytes = extra
                .checked_mul(2)
                .and_then(|n| n.checked_add(base_json_bytes))
                .ok_or_else(|| HarnessError::Engine("projected tool context overflow".into()))?;
            if json_bytes > config.round.context_limits.max_json_bytes {
                return Ok(false);
            }
            let render_bytes = rendered.text().len().checked_add(extra).ok_or_else(|| {
                HarnessError::Engine("projected rendered context overflow".into())
            })?;
            let input_bytes = json_bytes
                .checked_add(base_message_bytes)
                .ok_or_else(|| HarnessError::Engine("projected input overflow".into()))?;
            match block_on(self.engine.check_projected_rendered_bytes(
                snapshot.profile_id().clone(),
                render_bytes.max(input_bytes),
            ))? {
                Ok(()) => Ok(true),
                Err(
                    engine::error::EngineError::ContextLimitExceeded { .. }
                    | engine::error::EngineError::MemoryLimitExceeded {
                        reason: engine::error::MemoryLimitReason::BufferLimit,
                        ..
                    },
                ) => Ok(false),
                Err(error) => Err(HarnessError::Engine(format!(
                    "cannot verify projected tool results: {error}"
                ))),
            }
        };
        // Reduce only the inline threshold when necessary. The adapters retain
        // larger successful outputs as artifacts, so no result bytes are lost.
        if !fits(1)? {
            return Err(HarnessError::Engine("projected tool results exceed the profile context; start a shorter session or reduce the batch".into()));
        }
        let (mut low, mut high) = (1usize, config.inline_bytes);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if fits(middle)? {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        live(store, session, cancel, config.round.deadline)?;
        Ok(low)
    }
}
// Plugin processes are shared, but waiting for their host must not retain another
// session's turn past cancellation/deadline. Once acquired, W5 checks those same
// controls before IPC, including when an already-cancelled after hook is attempted.
fn lock_plugin_host<'a>(
    plugins: &'a Mutex<PluginHost>,
    deadline: Instant,
    cancel: &CancellationToken,
) -> HarnessResult<std::sync::MutexGuard<'a, PluginHost>> {
    loop {
        match plugins.try_lock() {
            Ok(host) => return Ok(host),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(HarnessError::Engine("plugin lock poisoned".into()));
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
        if cancel.is_cancelled() || Instant::now() >= deadline {
            return Err(HarnessError::Engine(
                "plugin host wait cancelled or deadline expired".into(),
            ));
        }
        thread::sleep(
            Duration::from_millis(5).min(deadline.saturating_duration_since(Instant::now())),
        );
    }
}

fn ensure_active_locked(store: &SessionStore, session: &str) -> HarnessResult<()> {
    if store
        .get_session(session)?
        .is_none_or(|s| s.status != SessionStatus::Active)
    {
        return Err(HarnessError::Invalid("session is not active".into()));
    }
    Ok(())
}
fn live(
    store: &Mutex<SessionStore>,
    session: &str,
    cancel: &CancellationToken,
    deadline: Instant,
) -> HarnessResult<()> {
    if cancel.is_cancelled() || Instant::now() >= deadline {
        return Err(HarnessError::Engine(
            "tool turn cancelled or deadline expired".into(),
        ));
    }
    ensure_active_locked(&*lock_store(store)?, session)
}
fn hook_values(receipts: &[HookReceipt]) -> Value {
    json!(receipts.iter().map(|r|json!({"plugin_id":r.plugin_id,"plugin_version":r.plugin_version,"invocation_id":r.invocation_id,"hook":r.hook,"annotations":r.annotations,"diagnostic":r.diagnostic})).collect::<Vec<_>>())
}
fn record_hooks(
    store: &Mutex<SessionStore>,
    session: &str,
    call: &str,
    receipts: &[HookReceipt],
) -> HarnessResult<()> {
    if !receipts.is_empty() {
        lock_store(store)?.append_event(session,EventKind::ToolRequested,json!({"contract_version":crate::approval::CONTRACT_VERSION,"call_id":call,"transition":"HookReceipts","receipts":hook_values(receipts)}),json!([]))?;
    }
    Ok(())
}
fn emit_terminal_tool<F: FnMut(ToolLoopEvent) -> HarnessResult<()>>(
    store: &Mutex<SessionStore>,
    session: &str,
    call: &str,
    emit: &mut F,
) -> HarnessResult<()> {
    let terminal = lock_store(store)?
        .list_events(session)?
        .into_iter()
        .rev()
        .find(|e| e.kind == EventKind::ToolCompleted && e.payload["call_id"] == call)
        .ok_or_else(|| HarnessError::Invalid("missing terminal tool result".into()))?;
    emit(ToolLoopEvent {
        event: "tool_result",
        body: {
            let mut body = terminal.payload;
            body["is_error"] = json!(true);
            body["summary"] = json!(summary(&body));
            body
        },
    })
}
fn dispatch(
    runtime: &ToolRuntime,
    store: &Mutex<SessionStore>,
    receipt: crate::store::StartedInvocation,
    cancel: CancellationToken,
    inline: usize,
    deadline: Instant,
) -> HarnessResult<ToolResult> {
    let (finished, watch) = std::sync::mpsc::channel();
    let cancellation = cancel.clone();
    thread::scope(|scope| {
        scope.spawn(move || {
            if matches!(
                watch.recv_timeout(deadline.saturating_duration_since(Instant::now())),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                cancellation.cancel();
            }
        });
        let result = (|| {
            let executor = &runtime.executor;
            let name = receipt.binding().invocation().call.tool_id.clone();
            // Preparation and final persistence are short store operations. run borrows no store.
            macro_rules! io_tool {
                ($prepare:ident) => {{
                    let job = executor.$prepare(
                        &*lock_store(store)?,
                        receipt,
                        &runtime.artifacts,
                        cancel,
                        inline,
                    )?;
                    let completion = job.run();
                    completion.complete(&mut *lock_store(store)?)
                }};
            }
            match name.as_str() {
                "fs.read" => io_tool!(prepare_file_read),
                "fs.list" => io_tool!(prepare_file_list),
                "shell.run" => io_tool!(prepare_shell_run).map_err(|e| e.error),
                "http.fetch" => io_tool!(prepare_http_fetch).map_err(|e| e.error),
                "fs.write" => {
                    let job = executor.prepare_file_write(&*lock_store(store)?, receipt, cancel)?;
                    let completion = job.run();
                    completion
                        .complete(&mut *lock_store(store)?)
                        .map_err(|e| e.error)
                }
                _ => Err(HarnessError::Invalid(
                    "W6 dispatch requires a registered builtin adapter".into(),
                )),
            }
        })();
        let _ = finished.send(());
        result
    })
}

fn summary(value: &Value) -> String {
    value.to_string().chars().take(1024).collect()
}
fn tool_result_body(call: &str, result: &ToolResult) -> Value {
    json!({"call_id":call,"is_error":result.is_error(),"summary":summary(&serde_json::to_value(result).expect("ToolResult serialization")),"result":result})
}
