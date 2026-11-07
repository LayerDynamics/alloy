//! Pause cancels before status flip; resume is paused→active only.

use engine::api::{ENGINE_API_VERSION, Engine, EngineFuture};
use engine::error::EngineError;
use engine::rendering::{
    PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions, render_and_count,
};
use engine::types::{
    ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, DecodingCapability, ExecutionBackend,
    FinishReason, Generation, GenerationEvent, GenerationEventStream, GenerationRequest,
    MemoryBreakdown, MemoryEstimate, ModelProfileId, ReleaseOutcome, RequestId, SessionStateId,
    Usage, WeightRepresentation,
};
use harness_core::{HardwareTier, SessionStatus, SessionStore, Supervisor, TelemetryClient};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tempfile::tempdir;

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

struct OrderEngine {
    tokenizer: PinnedTokenizer,
    artifact: ArtifactId,
    log: Arc<Mutex<Vec<&'static str>>>,
    fail_cancel: bool,
}

impl OrderEngine {
    fn new(log: Arc<Mutex<Vec<&'static str>>>, fail_cancel: bool) -> Self {
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
            artifact: ArtifactId::new("pause-order-fixture").expect("artifact"),
            log,
            fail_cancel,
        }
    }
}

impl Engine for OrderEngine {
    fn api_version(&self) -> u32 {
        ENGINE_API_VERSION
    }

    fn render_and_count(
        &self,
        context: ContextSnapshot,
    ) -> EngineFuture<'_, engine::types::RenderedContext> {
        let profile = context.profile_id().clone();
        let template = "{% for message in messages %}{% if loop.last %}{{ message.content }}{% endif %}{% endfor %}";
        let binding = RenderingBinding {
            profile_id: &profile,
            artifact_id: &self.artifact,
            template,
            template_sha256: digest(template.as_bytes()),
            artifact_manifest_sha256: [2; 32],
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
        let limits = RenderingLimits::new(32, 8192, 8192, 8192, 256).expect("limits");
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
                                input_tokens: 1,
                                output_tokens: 1,
                            },
                        },
                    ]),
                }),
            ))
        })
    }

    fn cancel(&self, _request_id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        self.log.lock().expect("log").push("cancel");
        let fail = self.fail_cancel;
        Box::pin(async move {
            if fail {
                Err(EngineError::Internal {
                    message: "forced cancel failure".into(),
                })
            } else {
                Ok(CancelOutcome::Requested)
            }
        })
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

fn telemetry() -> Arc<TelemetryClient> {
    Arc::new(TelemetryClient::new(
        "0.1.0-test",
        HardwareTier::Ram36Gb,
        "https://127.0.0.1:1/drop",
    ))
}

#[test]
fn pause_invokes_cancel_before_status_flip() {
    let dir = tempdir().expect("tempdir");
    let store = Mutex::new(SessionStore::open(dir.path().join("s.sqlite")).expect("store"));
    let log = Arc::new(Mutex::new(Vec::new()));
    let supervisor = Supervisor::new(OrderEngine::new(Arc::clone(&log), false), telemetry());
    supervisor
        .create_session(&store, "p-1", "profile")
        .expect("create");

    let request_id = RequestId::new("p-1-req").expect("id");
    supervisor
        .pause_session(&store, "p-1", Some(&request_id))
        .expect("pause");

    assert_eq!(log.lock().expect("log").as_slice(), ["cancel"]);
    let status = store
        .lock()
        .expect("lock")
        .get_session("p-1")
        .expect("get")
        .expect("row")
        .status;
    assert_eq!(status, SessionStatus::Paused);
}

#[test]
fn pause_keeps_active_when_cancel_fails() {
    let dir = tempdir().expect("tempdir");
    let store = Mutex::new(SessionStore::open(dir.path().join("s.sqlite")).expect("store"));
    let log = Arc::new(Mutex::new(Vec::new()));
    let supervisor = Supervisor::new(OrderEngine::new(Arc::clone(&log), true), telemetry());
    supervisor
        .create_session(&store, "p-2", "profile")
        .expect("create");

    let request_id = RequestId::new("p-2-req").expect("id");
    let error = supervisor
        .pause_session(&store, "p-2", Some(&request_id))
        .expect_err("cancel failure must fail pause");
    assert!(error.to_string().contains("forced cancel failure"));
    assert_eq!(log.lock().expect("log").as_slice(), ["cancel"]);

    let status = store
        .lock()
        .expect("lock")
        .get_session("p-2")
        .expect("get")
        .expect("row")
        .status;
    assert_eq!(status, SessionStatus::Active);
}

#[test]
fn resume_rejects_active_and_completed() {
    let dir = tempdir().expect("tempdir");
    let store = Mutex::new(SessionStore::open(dir.path().join("s.sqlite")).expect("store"));
    let supervisor = Supervisor::new(
        OrderEngine::new(Arc::new(Mutex::new(Vec::new())), false),
        telemetry(),
    );
    supervisor
        .create_session(&store, "r-1", "profile")
        .expect("create");

    let active_err = supervisor
        .resume_session(&store, "r-1")
        .expect_err("active resume must fail");
    assert!(active_err.to_string().contains("already active"));

    store
        .lock()
        .expect("lock")
        .set_status("r-1", SessionStatus::Completed)
        .expect("complete");
    let completed_err = supervisor
        .resume_session(&store, "r-1")
        .expect_err("completed resume must fail");
    assert!(completed_err.to_string().contains("completed"));

    store
        .lock()
        .expect("lock")
        .set_status("r-1", SessionStatus::Paused)
        .expect("pause");
    supervisor
        .resume_session(&store, "r-1")
        .expect("paused→active");
    let status = store
        .lock()
        .expect("lock")
        .get_session("r-1")
        .expect("get")
        .expect("row")
        .status;
    assert_eq!(status, SessionStatus::Active);
}
