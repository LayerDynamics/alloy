//! Supervisor skeleton: one turn persists events; telemetry failure is isolated.

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
use harness_core::telemetry::TelemetryTransport;
use harness_core::{EventKind, HardwareTier, SessionStore, Supervisor, TelemetryClient};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

struct FailTransport(SyncSender<()>);

impl TelemetryTransport for FailTransport {
    fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
        let _ = self.0.try_send(());
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
    let store = Mutex::new(SessionStore::open(dir.path().join("s.sqlite")).expect("store"));
    let (attempted, received) = mpsc::sync_channel(1);
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram36Gb,
        "https://127.0.0.1:1/drop",
        Box::new(FailTransport(attempted)),
    ));
    let supervisor = Supervisor::new(ReplyEngine::new(), Arc::clone(&telemetry));
    supervisor
        .create_session(&store, "sup-1", "harness-profile")
        .expect("create");

    let outcome = supervisor
        .run_turn(&store, "sup-1", "hello", 16)
        .expect("turn must succeed even when telemetry flush fails");
    received
        .recv_timeout(Duration::from_secs(5))
        .expect("offline POST attempted");
    telemetry
        .flush()
        .expect("worker settles after isolated failure");
    assert_eq!(outcome.assistant_text, "hi there");
    assert!(outcome.input_tokens > 0);

    let events = store
        .lock()
        .expect("lock")
        .list_events("sup-1")
        .expect("events");
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

struct GatedTransport {
    entered: SyncSender<()>,
    release: Mutex<Receiver<()>>,
    calls: Arc<AtomicUsize>,
}

impl TelemetryTransport for GatedTransport {
    fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let _ = self.entered.send(());
            let _ = self.release.lock().unwrap().recv();
        }
        Err("forced offline after gate".to_owned())
    }
}

struct ReleaseOnDrop(SyncSender<()>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        let _ = self.0.try_send(());
    }
}

#[test]
fn supervisor_completes_two_turns_while_telemetry_transport_is_blocked() {
    let dir = tempdir().expect("tempdir");
    let store = Arc::new(Mutex::new(
        SessionStore::open(dir.path().join("s.sqlite")).unwrap(),
    ));
    let (entered, blocked) = mpsc::sync_channel(1);
    let (release, gate) = mpsc::sync_channel(1);
    let release = ReleaseOnDrop(release);
    let calls = Arc::new(AtomicUsize::new(0));
    let telemetry = Arc::new(TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram36Gb,
        "https://telemetry.example.test/v1",
        Box::new(GatedTransport {
            entered,
            release: Mutex::new(gate),
            calls: Arc::clone(&calls),
        }),
    ));
    let supervisor = Supervisor::new(ReplyEngine::new(), Arc::clone(&telemetry));
    supervisor
        .create_session(&store, "blocked-telemetry", "harness-profile")
        .unwrap();
    telemetry.request_flush();
    blocked
        .recv_timeout(Duration::from_secs(5))
        .expect("telemetry worker blocked");

    let turn_store = Arc::clone(&store);
    let (returned, results) = mpsc::sync_channel(1);
    let turns = thread::spawn(move || {
        let outcomes: Vec<_> = ["hello", "world"]
            .into_iter()
            .map(|prompt| supervisor.run_turn(&turn_store, "blocked-telemetry", prompt, 16))
            .collect();
        let _ = returned.send(outcomes);
    });
    let outcomes = results
        .recv_timeout(Duration::from_secs(5))
        .expect("both turns return while telemetry remains blocked");
    for outcome in outcomes {
        assert_eq!(outcome.expect("turn unaffected").assistant_text, "hi there");
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "one in-flight telemetry POST"
    );
    assert_eq!(telemetry.snapshot().unwrap().turn_count, 2);
    let events = store
        .lock()
        .unwrap()
        .list_events("blocked-telemetry")
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == EventKind::AssistantMessage)
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == EventKind::RunCompleted)
            .count(),
        2
    );
    turns.join().unwrap();
    drop(release);
}

include!("w6_turn_boundary.inc");
include!("w6_tool_round.inc");
