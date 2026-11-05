//! Supervisor skeleton: one turn persists events; telemetry failure is isolated.

use engine::api::{ENGINE_API_VERSION, Engine, EngineFuture};
use engine::rendering::{
    PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions, render_and_count,
};
use engine::types::{
    ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, DecodingCapability,
    ExecutionBackend, FinishReason, Generation, GenerationEvent, GenerationEventStream,
    GenerationRequest, MemoryBreakdown, MemoryEstimate, ModelProfileId, ReleaseOutcome,
    SessionStateId, Usage, WeightRepresentation,
};
use harness_core::telemetry::TelemetryTransport;
use harness_core::{EventKind, HardwareTier, SessionStore, Supervisor, TelemetryClient};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tempfile::tempdir;

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

struct FailTransport;

impl TelemetryTransport for FailTransport {
    fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
        Err("forced offline".into())
    }
}

struct ReplyEngine {
    tokenizer: PinnedTokenizer,
    artifact: ArtifactId,
}

impl ReplyEngine {
    fn new() -> Self {
        let source = serde_json::to_vec(&json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "WhitespaceSplit" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
                "unk_token": "[UNK]"
            }
        }))
        .expect("tokenizer");
        Self {
            tokenizer: PinnedTokenizer::from_pinned_json(&source, digest(&source), 16 * 1024)
                .expect("pin"),
            artifact: ArtifactId::new("supervisor-fixture").expect("artifact"),
        }
    }
}

impl Engine for ReplyEngine {
    fn api_version(&self) -> u32 {
        ENGINE_API_VERSION
    }

    fn render_and_count(
        &self,
        context: ContextSnapshot,
    ) -> EngineFuture<'_, engine::types::RenderedContext> {
        let profile = context.profile_id().clone();
        // MiniJinja may not support messages[-1]; use last message content via join.
        let template = "{% for message in messages %}{% if loop.last %}{{ message.content }}{% endif %}{% endfor %}";
        let binding = RenderingBinding {
            profile_id: &profile,
            artifact_id: &self.artifact,
            template,
            template_sha256: digest(template.as_bytes()),
            artifact_manifest_sha256: [1; 32],
        };
        let options = RenderingOptions {
            add_generation_prompt: false,
            enable_thinking: None,
            add_special_tokens: false,
            keep_trailing_newline: true,
            trim_blocks: false,
            lstrip_blocks: false,
            reserved_spellings: &[],
        };
        let limits = RenderingLimits::new(16, 4096, 4096, 4096, 128).expect("limits");
        let result = render_and_count(&context, &binding, &self.tokenizer, &options, limits);
        Box::pin(async move { result })
    }

    fn estimate_memory(
        &self,
        _profile: ModelProfileId,
        _input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        Box::pin(async {
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

    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let request_id = request.request_id().clone();
        Box::pin(async move {
            Ok(Generation::from_stream(
                request_id,
                Box::pin(StaticStream {
                    events: VecDeque::from([
                        GenerationEvent::TextDelta {
                            text: "hi there".to_owned(),
                        },
                        GenerationEvent::GenerationCompleted {
                            finish_reason: FinishReason::EndOfSequence,
                            usage: Usage {
                                input_tokens: 1,
                                output_tokens: 2,
                            },
                        },
                    ]),
                }),
            ))
        })
    }

    fn cancel(&self, _request_id: engine::types::RequestId) -> EngineFuture<'_, CancelOutcome> {
        Box::pin(async { Ok(CancelOutcome::AlreadyFinished) })
    }

    fn release_session_state(&self, _state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        Box::pin(async { Ok(ReleaseOutcome::AlreadyAbsent) })
    }

    fn capabilities(&self, _artifact_id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        Box::pin(async {
            Ok(Capabilities {
                task: "text-generation".to_owned(),
                family: "fixture".to_owned(),
                backend: ExecutionBackend::HostCpu,
                weight_representations: vec![WeightRepresentation::F32],
                cache_modes: Vec::new(),
                context_limit_tokens: 128,
                supports_streaming: true,
                supports_incremental_kv: false,
                supports_prefix_reuse: false,
                decoding_modes: vec![DecodingCapability::Greedy],
            })
        })
    }
}

struct StaticStream {
    events: VecDeque<GenerationEvent>,
}

impl GenerationEventStream for StaticStream {
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        Poll::Ready(self.get_mut().events.pop_front())
    }
}

#[test]
fn supervisor_turn_persists_and_isolates_telemetry() {
    let dir = tempdir().expect("tempdir");
    let mut store = SessionStore::open(dir.path().join("s.sqlite")).expect("store");
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram36Gb,
        "https://127.0.0.1:1/drop",
        Box::new(FailTransport),
    ));
    let supervisor = Supervisor::new(ReplyEngine::new(), Arc::clone(&telemetry));
    supervisor
        .create_session(&mut store, "sup-1", "harness-profile")
        .expect("create");

    let outcome = supervisor
        .run_turn(&mut store, "sup-1", "hello", 16)
        .expect("turn must succeed even when telemetry flush fails");
    assert_eq!(outcome.assistant_text, "hi there");
    assert!(outcome.input_tokens > 0);

    let events = store.list_events("sup-1").expect("events");
    let kinds: Vec<_> = events.iter().map(|event| event.kind).collect();
    assert!(kinds.contains(&EventKind::UserMessage));
    assert!(kinds.contains(&EventKind::AssistantMessage));
    assert!(kinds.contains(&EventKind::RunStarted));
    assert!(kinds.contains(&EventKind::RunCompleted));
    assert_eq!(
        events
            .iter()
            .find(|event| event.kind == EventKind::AssistantMessage)
            .expect("assistant")
            .payload["content"],
        "hi there"
    );
}
