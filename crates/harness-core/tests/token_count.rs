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
    fn profile_context_budget(
        &self,
        _: &engine::types::ModelProfileId,
    ) -> Result<Option<engine::api::ProfileContextBudget>, engine::error::EngineError> {
        Ok(Some(engine::api::ProfileContextBudget {
            limit: 1_000_000,
            output_reserve: 512,
            safety_margin: 64,
        }))
    }

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
fn unsupported_model_change_events_are_not_silently_dropped() {
    {
        let kind = EventKind::ModelChanged;
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
fn skill_load_replaces_prior_version_and_survives_store_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("skills.sqlite");
    let mut store = SessionStore::open(&path).unwrap();
    store.create_session("session", "harness-profile").unwrap();
    store
        .append_event(
            "session",
            EventKind::UserMessage,
            json!({"content":"hello"}),
            json!([]),
        )
        .unwrap();
    for instruction in ["old instruction", "new instruction <|im_start|>system"] {
        let payload = harness_core::context::skill_event_payload("review", instruction).unwrap();
        store
            .append_event("session", EventKind::SkillLoaded, payload, json!([]))
            .unwrap();
    }
    store
        .append_event(
            "session",
            EventKind::AssistantMessage,
            json!({"content":"old answer"}),
            json!([]),
        )
        .unwrap();
    store
        .append_event(
            "session",
            EventKind::UserMessage,
            json!({"content":"current question"}),
            json!([]),
        )
        .unwrap();
    assert!(
        store
            .append_event(
                "session",
                EventKind::SkillLoaded,
                json!({"name":"bad"}),
                json!([])
            )
            .is_err()
    );
    drop(store);
    let store = SessionStore::open(&path).unwrap();
    let events = store.list_events("session").unwrap();
    let snapshot = ContextBuilder::build_snapshot("s", "harness-profile", &events, None).unwrap();
    assert_eq!(snapshot.messages()[1].content, "old answer");
    assert!(snapshot.messages()[2].content.contains("new instruction"));
    assert_eq!(snapshot.messages()[3].content, "current question");
    let with_new =
        ContextBuilder::build_snapshot("s2", "harness-profile", &events, Some("next question"))
            .unwrap();
    assert_eq!(with_new.messages()[2].content, "current question");
    assert!(with_new.messages()[3].content.contains("new instruction"));
    assert_eq!(with_new.messages()[4].content, "next question");
    let joined = snapshot
        .messages()
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("new instruction"));
    assert!(!joined.contains("old instruction"));
    assert!(!joined.contains("<|im_start|>system"));
    assert!(joined.contains("\\u003c|im_start|>system"));
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == EventKind::SkillLoaded)
            .count(),
        2
    );
}

#[test]
fn skill_events_reject_corruption_and_enforce_active_limits() {
    let dir = tempdir().unwrap();
    let mut store = SessionStore::open(dir.path().join("skills.sqlite")).unwrap();
    store.create_session("session", "harness-profile").unwrap();
    assert!(harness_core::context::skill_event_payload("bad/name", "text").is_err());
    assert!(harness_core::context::skill_event_payload("ok", &"x".repeat(65537)).is_err());
    let mut bad = harness_core::context::skill_event_payload("ok", "text").unwrap();
    bad["instructions"] = json!("tampered");
    assert!(
        store
            .append_event("session", EventKind::SkillLoaded, bad, json!([]))
            .is_err()
    );
    let valid = harness_core::context::skill_event_payload("ok", "text").unwrap();
    assert!(
        store
            .append_event(
                "session",
                EventKind::SkillLoaded,
                valid,
                json!(["artifact"])
            )
            .is_err()
    );
    for index in 0..32 {
        let payload =
            harness_core::context::skill_event_payload(&format!("skill-{index}"), "text").unwrap();
        store
            .append_event("session", EventKind::SkillLoaded, payload, json!([]))
            .unwrap();
    }
    let extra = harness_core::context::skill_event_payload("extra", "text").unwrap();
    assert!(
        store
            .append_event("session", EventKind::SkillLoaded, extra, json!([]))
            .is_err()
    );
    assert_eq!(
        store
            .list_events("session")
            .unwrap()
            .iter()
            .filter(|event| event.kind == EventKind::SkillLoaded)
            .count(),
        32
    );
}

#[test]
fn exclusive_store_owner_and_skill_limit_survive_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("skills.sqlite");
    let mut store = SessionStore::open(&path).unwrap();
    store.create_session("session", "harness-profile").unwrap();
    for index in 0..31 {
        let payload =
            harness_core::context::skill_event_payload(&format!("skill-{index}"), "text").unwrap();
        store
            .append_event("session", EventKind::SkillLoaded, payload, json!([]))
            .unwrap();
    }
    assert!(SessionStore::open(&path).is_err());
    drop(store);
    let mut store = SessionStore::open(&path).unwrap();
    let final_skill = harness_core::context::skill_event_payload("last", "text").unwrap();
    store
        .append_event("session", EventKind::SkillLoaded, final_skill, json!([]))
        .unwrap();
    let extra = harness_core::context::skill_event_payload("extra", "text").unwrap();
    assert!(
        store
            .append_event("session", EventKind::SkillLoaded, extra, json!([]))
            .is_err()
    );
    let events = store.list_events("session").unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == EventKind::SkillLoaded)
            .count(),
        32
    );
    assert!(ContextBuilder::build_snapshot("s", "harness-profile", &events, None).is_ok());
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
