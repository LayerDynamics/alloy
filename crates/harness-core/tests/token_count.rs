//! Named acceptance: `token_count_exact`.

use engine::api::{ENGINE_API_VERSION, Engine, EngineFuture};
use engine::rendering::{
    PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions, render_and_count,
};
use engine::types::{
    ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, DecodingCapability, ExecutionBackend,
    FinishReason, Generation, GenerationEvent, GenerationEventStream, GenerationRequest,
    MemoryBreakdown, MemoryEstimate, ModelProfileId, ReleaseOutcome, SessionStateId, Usage,
    WeightRepresentation,
};
use harness_core::{ContextBuilder, EventKind, SessionStore};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use tempfile::tempdir;

const CONTENT_TEMPLATE: &str = "{{ messages[0].content }}";

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn tokenizer_source() -> Vec<u8> {
    serde_json::to_vec(&json!({
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
            "vocab": {
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
                "assistant": 3
            },
            "unk_token": "[UNK]"
        }
    }))
    .expect("tokenizer JSON")
}

struct ExactRenderEngine {
    tokenizer: PinnedTokenizer,
    template: &'static str,
    artifact: ArtifactId,
}

impl ExactRenderEngine {
    fn new() -> Self {
        let source = tokenizer_source();
        let tokenizer = PinnedTokenizer::from_pinned_json(&source, digest(&source), 16 * 1024)
            .expect("tokenizer");
        Self {
            tokenizer,
            template: CONTENT_TEMPLATE,
            artifact: ArtifactId::new("harness-token-fixture").expect("artifact"),
        }
    }
}

impl Engine for ExactRenderEngine {
    fn api_version(&self) -> u32 {
        ENGINE_API_VERSION
    }

    fn render_and_count(
        &self,
        context: ContextSnapshot,
    ) -> EngineFuture<'_, engine::types::RenderedContext> {
        let profile = context.profile_id().clone();
        let binding = RenderingBinding {
            profile_id: &profile,
            artifact_id: &self.artifact,
            template: self.template,
            template_sha256: digest(self.template.as_bytes()),
            artifact_manifest_sha256: [9; 32],
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
        let limits = RenderingLimits::new(8, 4096, 4096, 4096, 128).expect("limits");
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
                            text: "ok".to_owned(),
                        },
                        GenerationEvent::GenerationCompleted {
                            finish_reason: FinishReason::EndOfSequence,
                            usage: Usage {
                                input_tokens: 2,
                                output_tokens: 1,
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

#[test]
fn unsupported_skill_and_checkpoint_events_are_not_silently_dropped() {
    for kind in [
        EventKind::SkillLoaded,
        EventKind::ContextCompacted,
        EventKind::ModelChanged,
    ] {
        let dir = tempdir().expect("tempdir");
        let mut store = SessionStore::open(dir.path().join("s.sqlite")).expect("store");
        store
            .create_session("session", "harness-profile")
            .expect("session");
        store
            .append_event(
                "session",
                EventKind::UserMessage,
                json!({"content":"hello"}),
                json!([]),
            )
            .expect("message");
        store
            .append_event("session", kind, json!({}), json!([]))
            .expect("event");
        let history = store.list_events("session").expect("history");
        let error = ContextBuilder::build_snapshot("snapshot", "harness-profile", &history, None)
            .expect_err("must not ignore model-visible context");
        assert!(error.to_string().contains("model-change context"));
    }
}

#[test]
fn text_snapshot_preserves_message_artifact_references() {
    let dir = tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("s.sqlite")).unwrap();
    store.create_session("session", "harness-profile").unwrap();
    store
        .append_event(
            "session",
            EventKind::UserMessage,
            json!({"content":"hello"}),
            json!(["artifact"]),
        )
        .unwrap();
    let events = store.list_events("session").unwrap();
    let snapshot =
        ContextBuilder::build_snapshot("snapshot", "harness-profile", &events, None).unwrap();
    assert!(snapshot.messages()[0].content.contains("[\"artifact\"]"));
}

impl GenerationEventStream for StaticStream {
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        Poll::Ready(self.get_mut().events.pop_front())
    }
}

#[test]
fn token_count_exact() {
    let dir = tempdir().expect("tempdir");
    let mut store = SessionStore::open(dir.path().join("s.sqlite")).expect("store");
    store
        .create_session("tok-1", "harness-profile")
        .expect("session");
    store
        .append_event(
            "tok-1",
            EventKind::UserMessage,
            json!({ "content": "hello world" }),
            json!([]),
        )
        .expect("user");

    let events = store.list_events("tok-1").expect("events");
    let snapshot = ContextBuilder::build_snapshot("tok-snap", "harness-profile", &events, None)
        .expect("snapshot");

    let engine = ExactRenderEngine::new();
    let rendered = ContextBuilder::render_and_count(&engine, snapshot).expect("render");

    // Exact HF WordLevel counts for the fixed fixture vocabulary.
    assert_eq!(rendered.text(), "hello world");
    assert_eq!(rendered.token_ids(), &[1, 2]);
    assert_eq!(rendered.token_count(), 2);

    // Independent check against the same tokenizer JSON.
    let source = tokenizer_source();
    let tokenizer = tokenizers::Tokenizer::from_bytes(&source).expect("hf tokenizer");
    let encoding = tokenizer.encode("hello world", false).expect("encode");
    assert_eq!(encoding.get_ids(), rendered.token_ids());
}
