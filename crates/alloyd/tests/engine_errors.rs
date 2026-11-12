//! Structured engine errors survive rendering, startup and streamed failure.

use alloyd::{API_VERSION, DaemonService, FixtureEngine, RpcRequest};
use engine::api::{Engine, EngineFuture};
use engine::error::{EngineError, MemoryLimitReason, ProtectedResource};
use engine::family::ModelInstanceId;
use engine::memory_guard::{CeilingBinding, CeilingBreakdown};
use engine::types::{
    ArtifactId, CancelOutcome, Capabilities, ContextSnapshot, Generation, GenerationEvent,
    GenerationEventStream, GenerationRequest, MemoryEstimate, ModelProfileId, ReleaseOutcome,
    RenderedContext, RequestId, SessionStateId, Usage,
};
use harness_core::telemetry::TelemetryTransport;
use harness_core::{EventKind, HardwareTier, SessionStore, TelemetryClient};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tempfile::tempdir;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureAt {
    Render,
    Generate,
    Stream,
}

struct DiagnosticEngine {
    inner: FixtureEngine,
    at: FailureAt,
    error: Mutex<Option<EngineError>>,
}

impl DiagnosticEngine {
    fn take_error(&self) -> EngineError {
        self.error
            .lock()
            .unwrap()
            .take()
            .expect("one engine attempt")
    }
}

impl Engine for DiagnosticEngine {
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
        self.inner.api_version()
    }

    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext> {
        if self.at == FailureAt::Render {
            let error = self.take_error();
            return Box::pin(async move { Err(error) });
        }
        self.inner.render_and_count(context)
    }

    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        self.inner.estimate_memory(profile, input_tokens)
    }

    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let error = self.take_error();
        if self.at == FailureAt::Generate {
            return Box::pin(async move { Err(error) });
        }
        let request_id = request.request_id().clone();
        Box::pin(async move {
            Ok(Generation::from_stream(
                request_id,
                Box::pin(FailedStream(VecDeque::from([
                    GenerationEvent::TextDelta {
                        text: "partial answer".to_owned(),
                    },
                    GenerationEvent::RequestFailed {
                        error,
                        partial_usage: Usage {
                            input_tokens: 1,
                            output_tokens: 2,
                        },
                    },
                ]))),
            ))
        })
    }

    fn cancel(&self, request_id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        self.inner.cancel(request_id)
    }

    fn release_session_state(&self, state_id: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        self.inner.release_session_state(state_id)
    }

    fn capabilities(&self, artifact_id: ArtifactId) -> EngineFuture<'_, Capabilities> {
        self.inner.capabilities(artifact_id)
    }
}

struct FailedStream(VecDeque<GenerationEvent>);

impl GenerationEventStream for FailedStream {
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        Poll::Ready(self.get_mut().0.pop_front())
    }
}

struct NoopTransport;

impl TelemetryTransport for NoopTransport {
    fn post_json(&self, _endpoint: &str, _body: &str) -> Result<(), String> {
        Ok(())
    }
}

fn failures() -> Vec<EngineError> {
    let artifact_id = ArtifactId::new("protected-model").unwrap();
    let instance_id = ModelInstanceId::new();
    vec![
        EngineError::Busy {
            active_request_id: Some("active-request".to_owned()),
            message: "native worker owns the active request".to_owned(),
        },
        EngineError::ProtectedResources {
            operation: "generation startup",
            resources: vec![
                ProtectedResource::ModelPin {
                    artifact_id: artifact_id.clone(),
                    instance_id: instance_id.clone(),
                    handles: NonZeroUsize::new(2).unwrap(),
                },
                ProtectedResource::SessionStateLease {
                    artifact_id,
                    instance_id,
                    state_id: SessionStateId::from_digest([7; 32]),
                    handles: NonZeroUsize::new(1).unwrap(),
                },
            ],
        },
        EngineError::MemoryLimitExceeded {
            operation: "generation startup",
            reason: MemoryLimitReason::ReservationBudget { budget_bytes: 100 },
            required_bytes: 200,
            available_bytes: 50,
        },
        EngineError::MemoryLimitExceeded {
            operation: "startup native demand",
            reason: MemoryLimitReason::AllocatorCeiling { ceiling_bytes: 100 },
            required_bytes: 200,
            available_bytes: 100,
        },
        EngineError::MemoryLimitExceeded {
            operation: "generation startup",
            reason: MemoryLimitReason::AdmissionsPaused,
            required_bytes: 1,
            available_bytes: 100,
        },
        EngineError::MemoryLimitExceeded {
            operation: "planned host startup",
            reason: MemoryLimitReason::HostPreflight {
                artifact_id: ArtifactId::new("startup-model").unwrap(),
                ceiling: CeilingBreakdown {
                    reserve_bytes: 20,
                    static_bytes: 180,
                    dynamic_bytes: 100,
                    metal_cap_bytes: 150,
                    hard_limit_bytes: 100,
                    binding: Some(CeilingBinding::Dynamic),
                },
            },
            required_bytes: 200,
            available_bytes: 100,
        },
        EngineError::MemoryLimitExceeded {
            operation: "load buffer",
            reason: MemoryLimitReason::BufferLimit,
            required_bytes: 200,
            available_bytes: 100,
        },
        EngineError::MemoryLimitExceeded {
            operation: "generation estimate",
            reason: MemoryLimitReason::AdvisoryEstimate,
            required_bytes: 200,
            available_bytes: 100,
        },
        EngineError::Cancelled {
            request_id: "cancelled-request".to_owned(),
        },
    ]
}

fn invoke(service: &DaemonService<DiagnosticEngine>, method: &str, params: Value) -> Vec<Value> {
    let request: RpcRequest = serde_json::from_value(json!({
        "jsonrpc": "2.0", "id": 1, "api_version": API_VERSION,
        "method": method, "params": params,
    }))
    .unwrap();
    let mut output = Vec::new();
    service.handle(request, &mut output).unwrap();
    output
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

fn assert_failure_survives(at: FailureAt) {
    for error in failures() {
        let expected = error.diagnostic();
        let dir = tempdir().unwrap();
        let db = dir.path().join("session.sqlite");
        let telemetry = Arc::new(TelemetryClient::with_transport(
            "test",
            HardwareTier::Ram36Gb,
            "https://example.invalid",
            Box::new(NoopTransport),
        ));
        let service = DaemonService::open(
            &db,
            DiagnosticEngine {
                inner: FixtureEngine::new("unused successful reply"),
                at,
                error: Mutex::new(Some(error)),
            },
            telemetry,
            32,
            0,
            0,
        )
        .unwrap();
        let created = invoke(
            &service,
            "session.create",
            json!({ "profile": "fixture-profile" }),
        );
        let session_id = created[0]["result"]["session_id"].as_str().unwrap();
        let lines = invoke(
            &service,
            "session.send",
            json!({ "session_id": session_id, "content": "hello" }),
        );
        let response = lines.last().unwrap();
        assert_eq!(response["error"]["code"], -32000);
        assert_eq!(response["error"]["data"]["kind"], "engine_failed");
        assert_eq!(response["error"]["message"], expected["message"]);
        assert_eq!(response["error"]["data"]["engine_error"], expected);
        assert!(
            response.get("result").is_none(),
            "refusal cannot become success"
        );

        let terminals: Vec<_> = lines
            .iter()
            .filter(|line| line["params"]["event"] == "request_failed")
            .collect();
        if at == FailureAt::Stream {
            assert_eq!(terminals.len(), 1);
            let body = &terminals[0]["params"]["body"];
            assert_eq!(
                body["engine_error"],
                response["error"]["data"]["engine_error"]
            );
            assert_eq!(body["message"], response["error"]["message"]);
            assert_eq!(
                body["partial_usage"],
                json!({ "input_tokens": 1, "output_tokens": 2 })
            );
            assert!(
                lines
                    .iter()
                    .any(|line| line["params"]["event"] == "text_delta")
            );
        } else {
            assert!(
                terminals.is_empty(),
                "pre-stream failures use the final response"
            );
        }
        assert!(
            invoke(&service, "daemon.status", json!({}))[0]
                .get("result")
                .is_some(),
            "refusal leaves the daemon reachable"
        );
        assert!(
            SessionStore::open(&db).is_err(),
            "live daemon must retain exclusive mutating store ownership"
        );
        drop(service);
        let store = SessionStore::open(&db).unwrap();
        let events = store.list_events(session_id).unwrap();
        assert!(
            !events.iter().any(|event| matches!(
                event.kind,
                EventKind::AssistantMessage | EventKind::RunCompleted
            )),
            "failed partial text cannot be committed as a successful turn"
        );
    }
}

#[test]
fn structured_render_failure_reaches_rpc_unchanged() {
    assert_failure_survives(FailureAt::Render);
}

#[test]
fn structured_startup_failure_reaches_rpc_unchanged() {
    assert_failure_survives(FailureAt::Generate);
}

#[test]
fn structured_stream_failure_matches_final_rpc_and_preserves_partial_usage() {
    assert_failure_survives(FailureAt::Stream);
}
