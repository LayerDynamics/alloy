//! Phase-1 / issue #4 named acceptance tests that need crate-private types.

#![cfg(test)]

use crate::{
    api::{ENGINE_API_VERSION, Engine, EngineCore, EngineFuture, EngineRuntime, ExecutionContext},
    error::EngineError,
    registry::{FamilyRegistration, FamilyVersion, ModelProfile, Registry},
    scheduler::{ExecutionOutcome, Scheduler},
    types::{
        ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, ContextSnapshotId,
        DecodingCapability, ExecutionBackend, FinishReason, Generation, GenerationEvent,
        GenerationEventStream, GenerationOptions, GenerationRequest, MemoryBreakdown,
        MemoryEstimate, Message, MessageRole, ModelProfileId, ReleaseOutcome, RenderedContext,
        RendererIdentity, RequestId, SessionStateId, TokenizerId, Usage, WeightRepresentation,
    },
};
use std::{
    collections::VecDeque,
    future::{Future, poll_fn},
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

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
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + TEST_TIMEOUT;

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    panic!("timed out waiting for future");
                }
                let _ = receiver.recv_timeout(remaining);
            }
        }
    }
}

fn next_event(stream: &mut Pin<Box<dyn GenerationEventStream>>) -> Option<GenerationEvent> {
    block_on(poll_fn(|cx| stream.as_mut().poll_next(cx)))
}

struct MockStream {
    deltas: VecDeque<&'static str>,
    execution: Option<ExecutionContext>,
    output_tokens: usize,
}

impl GenerationEventStream for MockStream {
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        let this = self.get_mut();
        let Some(execution) = this.execution.as_ref() else {
            return Poll::Ready(None);
        };

        if let Err(error) = execution.check_cancelled() {
            let execution = this.execution.take().expect("execution");
            let terminal = execution
                .finish(ExecutionOutcome::Failed {
                    error,
                    partial_usage: Usage {
                        input_tokens: 1,
                        output_tokens: this.output_tokens,
                    },
                })
                .expect("cancel terminal");
            return Poll::Ready(Some(terminal));
        }

        if let Some(text) = this.deltas.pop_front() {
            this.output_tokens = this.output_tokens.saturating_add(1);
            return Poll::Ready(Some(GenerationEvent::TextDelta {
                text: text.to_owned(),
            }));
        }

        let execution = this.execution.take().expect("execution");
        let terminal = execution
            .finish(ExecutionOutcome::Completed {
                finish_reason: FinishReason::OutputLimit,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: this.output_tokens,
                },
            })
            .expect("completed terminal");
        Poll::Ready(Some(terminal))
    }
}

impl Drop for MockStream {
    fn drop(&mut self) {
        if let Some(execution) = self.execution.take() {
            let _ = execution.finish(ExecutionOutcome::Failed {
                error: EngineError::Cancelled {
                    request_id: "mock-drop".to_owned(),
                },
                partial_usage: Usage {
                    input_tokens: 1,
                    output_tokens: self.output_tokens,
                },
            });
        }
    }
}

struct MockRuntime {
    capabilities: Capabilities,
}

impl EngineRuntime for MockRuntime {
    fn render_and_count<'a>(
        &'a self,
        registry: &'a Registry,
        context: ContextSnapshot,
    ) -> EngineFuture<'a, RenderedContext> {
        Box::pin(async move {
            let artifact = registry.resolve_profile(context.profile_id())?.artifact();
            Ok(RenderedContext {
                snapshot_id: context.snapshot_id().clone(),
                profile_id: context.profile_id().clone(),
                artifact_id: artifact.id().clone(),
                text: "hello".to_owned(),
                token_ids: vec![1],
                render_hash: [2; 32],
                template_hash: [3; 32],
                tool_schema_hash: None,
                tokenizer_id: TokenizerId::new("mock-tokenizer").expect("tokenizer"),
                renderer: RendererIdentity {
                    implementation: "mock".to_owned(),
                    version: "0".to_owned(),
                },
            })
        })
    }

    fn estimate_memory<'a>(
        &'a self,
        _registry: &'a Registry,
        _profile: ModelProfileId,
        _input_tokens: usize,
    ) -> EngineFuture<'a, MemoryEstimate> {
        Box::pin(async move {
            MemoryEstimate::new(
                MemoryBreakdown {
                    weights_bytes: 1,
                    kv_bytes: 1,
                    temporary_bytes: 1,
                    output_bytes: 1,
                    draft_bytes: 0,
                    headroom_bytes: 1,
                },
                1_000_000,
                Vec::new(),
            )
        })
    }

    fn start_generation(
        &self,
        _registry: Arc<Registry>,
        request: GenerationRequest,
        _rendered: RenderedContext,
        execution: ExecutionContext,
    ) -> EngineFuture<'_, Generation> {
        let request_id = request.request_id().clone();
        Box::pin(async move {
            Ok(Generation {
                request_id,
                events: Box::pin(MockStream {
                    deltas: VecDeque::from(["one", "two", "three"]),
                    execution: Some(execution),
                    output_tokens: 0,
                }),
            })
        })
    }

    fn release_session_state(&self, _state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        Box::pin(async move { Ok(ReleaseOutcome::AlreadyAbsent) })
    }

    fn capabilities<'a>(
        &'a self,
        _registry: &'a Registry,
        _artifact_id: ArtifactId,
    ) -> EngineFuture<'a, Capabilities> {
        let capabilities = self.capabilities.clone();
        Box::pin(async move { Ok(capabilities) })
    }
}

fn mock_engine() -> (EngineCore<MockRuntime>, PathBuf, ModelProfileId, ArtifactId) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let stamp = NEXT.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("alloy-w3-mock-{}-{}", std::process::id(), stamp));
    std::fs::create_dir_all(&directory).expect("temp artifact dir");
    std::fs::write(
        directory.join("config.json"),
        br#"{"architectures":["MockForCausalLM"],"max_position_embeddings":128}"#,
    )
    .expect("config");

    let capabilities = Capabilities {
        task: "text-generation".to_owned(),
        family: "mock-family".to_owned(),
        backend: ExecutionBackend::HostCpu,
        weight_representations: vec![WeightRepresentation::F32],
        cache_modes: Vec::new(),
        context_limit_tokens: 128,
        supports_streaming: true,
        supports_incremental_kv: false,
        supports_prefix_reuse: false,
        decoding_modes: vec![DecodingCapability::Greedy],
    };

    let family = FamilyRegistration::new(
        FamilyVersion::new(1, 0, 0),
        vec!["MockForCausalLM".to_owned()],
        capabilities.clone(),
    )
    .expect("family");

    let artifact_id = ArtifactId::new(format!("mock-artifact-{stamp}")).expect("artifact");
    let profile_id = ModelProfileId::new(format!("mock-profile-{stamp}")).expect("profile");

    let mut registry = Registry::new();
    registry.register_family(family).expect("register family");
    registry
        .register_artifact(artifact_id.clone(), &directory)
        .expect("register artifact");
    registry
        .register_profile(
            ModelProfile::new(
                profile_id.clone(),
                artifact_id.clone(),
                128,
                8,
                8,
                GenerationOptions::greedy(4).expect("options"),
            )
            .expect("profile"),
        )
        .expect("register profile");

    let runtime = MockRuntime { capabilities };

    let engine = EngineCore::new(registry, Scheduler::new(8).expect("scheduler"), runtime);
    (engine, directory, profile_id, artifact_id)
}

fn request(id: &str, profile: &ModelProfileId) -> GenerationRequest {
    GenerationRequest::new(
        RequestId::new(id).expect("request"),
        profile.clone(),
        ContextSnapshot::new(
            ContextSnapshotId::new(format!("{id}-snap")).expect("snapshot"),
            profile.clone(),
            vec![Message {
                role: MessageRole::User,
                content: "hello".to_owned(),
            }],
        )
        .expect("context"),
        GenerationOptions::greedy(4).expect("options"),
    )
    .expect("generation request")
}

#[test]
fn engine_traits_documented() {
    // Public engine traits compile under `#![deny(missing_docs)]` in lib.rs.
    // This named acceptance test keeps the Phase-1 checklist discoverable.
    assert_eq!(ENGINE_API_VERSION, 1);
    let (engine, directory, _, artifact) = mock_engine();
    assert_eq!(engine.api_version(), ENGINE_API_VERSION);
    let capabilities = block_on(engine.capabilities(artifact)).expect("capabilities");
    assert_eq!(capabilities.family, "mock-family");
    assert!(capabilities.supports_streaming);
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn stream_events_order() {
    let (engine, directory, profile, _artifact) = mock_engine();

    let generation =
        block_on(engine.generate(request("stream-order", &profile))).expect("generate");
    let request_id = generation.request_id().clone();
    let mut events = generation.into_events();

    assert!(matches!(
        next_event(&mut events),
        Some(GenerationEvent::TextDelta { text }) if text == "one"
    ));
    assert!(matches!(
        next_event(&mut events),
        Some(GenerationEvent::TextDelta { text }) if text == "two"
    ));
    assert!(matches!(
        next_event(&mut events),
        Some(GenerationEvent::TextDelta { text }) if text == "three"
    ));
    assert!(matches!(
        next_event(&mut events),
        Some(GenerationEvent::GenerationCompleted {
            finish_reason: FinishReason::OutputLimit,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 3,
            },
        })
    ));
    assert!(next_event(&mut events).is_none());
    assert_eq!(
        block_on(engine.cancel(request_id)).expect("cancel finished"),
        CancelOutcome::AlreadyFinished,
    );

    let generation = block_on(engine.generate(request("stream-cancel", &profile)))
        .expect("generate cancel path");
    let request_id = generation.request_id().clone();
    let mut events = generation.into_events();

    assert!(matches!(
        next_event(&mut events),
        Some(GenerationEvent::TextDelta { text }) if text == "one"
    ));
    assert_eq!(
        block_on(engine.cancel(request_id.clone())).expect("cancel active"),
        CancelOutcome::Requested,
    );
    assert!(matches!(
        next_event(&mut events),
        Some(GenerationEvent::RequestFailed { .. })
    ));

    drop(events);
    let _ = std::fs::remove_dir_all(directory);
    // Give the worker drop path a beat before process teardown.
    thread::sleep(Duration::from_millis(10));
}
