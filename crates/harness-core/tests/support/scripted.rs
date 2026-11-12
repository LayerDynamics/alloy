use engine::{
    api::{ENGINE_API_VERSION, Engine, EngineFuture},
    rendering::{
        PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions, render_and_count,
    },
    types::*,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
};
pub struct ScriptedEngine {
    tokenizer: PinnedTokenizer,
    pub batches: Mutex<VecDeque<Vec<GenerationEvent>>>,
    pub contexts: Mutex<Vec<ContextSnapshot>>,
    pub cancel_not_found: bool,
    pub cancel_not_found_id: Option<String>,
    pub budget: Option<engine::api::ProfileContextBudget>,
    pub projected_byte_limit: usize,
    pub projected_pending: bool,
}
impl ScriptedEngine {
    pub fn new(batches: Vec<Vec<GenerationEvent>>) -> Self {
        let bytes=serde_json::to_vec(&json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}})).unwrap();
        Self {
            tokenizer: PinnedTokenizer::from_pinned_json(
                &bytes,
                Sha256::digest(&bytes).into(),
                16384,
            )
            .unwrap(),
            batches: Mutex::new(batches.into()),
            contexts: Mutex::new(Vec::new()),
            cancel_not_found: false,
            cancel_not_found_id: None,
            budget: Some(engine::api::ProfileContextBudget {
                limit: 1_000_000,
                output_reserve: 512,
                safety_margin: 64,
            }),
            projected_byte_limit: usize::MAX,
            projected_pending: false,
        }
    }
}
pub fn done() -> GenerationEvent {
    GenerationEvent::GenerationCompleted {
        finish_reason: FinishReason::EndOfSequence,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    }
}
pub fn proposal(name: &str, args: serde_json::Value) -> GenerationEvent {
    GenerationEvent::ToolCallProposed {
        proposal: ToolCallProposal::new(name, args.as_object().unwrap().clone()).unwrap(),
    }
}
impl Engine for ScriptedEngine {
    fn profile_context_budget(
        &self,
        _: &ModelProfileId,
    ) -> Result<Option<engine::api::ProfileContextBudget>, engine::error::EngineError> {
        Ok(self.budget)
    }
    fn api_version(&self) -> u32 {
        ENGINE_API_VERSION
    }
    fn check_projected_tool_render<'a>(
        &'a self,
        projection: engine::api::ProjectedToolRender<'a>,
    ) -> EngineFuture<'a, ()> {
        let limit = self.projected_byte_limit;
        let bytes = projection.input_bytes;
        if self.projected_pending {
            return Box::pin(std::future::pending());
        }
        Box::pin(async move {
            if bytes > limit {
                Err(engine::error::EngineError::ContextLimitExceeded {
                    input_tokens: bytes,
                    output_reserve_tokens: 0,
                    safety_margin_tokens: 0,
                    limit_tokens: limit,
                })
            } else {
                Ok(())
            }
        })
    }
    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext> {
        self.contexts.lock().unwrap().push(context.clone());
        let profile = context.profile_id().clone();
        let artifact = ArtifactId::new("scripted").unwrap();
        let template = "{% for message in messages %}{{ message.role }} {% if message.kind is defined and message.kind == 'call' %}{{ message.call_json }}{% else %}{{ message.content }}{% endif %}{% endfor %} {{ tools_json | default('') }}";
        let binding = RenderingBinding {
            profile_id: &profile,
            artifact_id: &artifact,
            template,
            template_sha256: Sha256::digest(template.as_bytes()).into(),
            artifact_manifest_sha256: [1; 32],
        };
        let options = RenderingOptions {
            add_generation_prompt: false,
            enable_thinking: None,
            add_special_tokens: false,
            keep_trailing_newline: true,
            trim_blocks: false,
            lstrip_blocks: false,
            reserved_spellings: &[
                "<|",
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
            ],
        };
        let result = render_and_count(
            &context,
            &binding,
            &self.tokenizer,
            &options,
            RenderingLimits::new(
                1024,
                4 * 1024 * 1024,
                4 * 1024 * 1024,
                4 * 1024 * 1024,
                32768,
            )
            .unwrap(),
        );
        Box::pin(async move { result })
    }
    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let events = self
            .batches
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected extra generation");
        let id = request.request_id().clone();
        Box::pin(async move { Ok(Generation::from_stream(id, Box::pin(Events(events.into())))) })
    }
    fn cancel(&self, id: RequestId) -> EngineFuture<'_, CancelOutcome> {
        let missing = self.cancel_not_found;
        let missing_id = self.cancel_not_found_id.clone();
        Box::pin(async move {
            if missing {
                Err(engine::error::EngineError::NotFound {
                    resource: "request",
                    id: missing_id.unwrap_or_else(|| id.as_str().to_owned()),
                })
            } else {
                Ok(CancelOutcome::Requested)
            }
        })
    }
    fn estimate_memory(&self, _: ModelProfileId, _: usize) -> EngineFuture<'_, MemoryEstimate> {
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
                10000000,
                Vec::new(),
            )
        })
    }
    fn release_session_state(&self, _: SessionStateId) -> EngineFuture<'_, ReleaseOutcome> {
        Box::pin(async { Ok(ReleaseOutcome::AlreadyAbsent) })
    }
    fn capabilities(&self, _: ArtifactId) -> EngineFuture<'_, Capabilities> {
        Box::pin(async {
            Ok(Capabilities {
                task: "text-generation".into(),
                family: "scripted".into(),
                backend: ExecutionBackend::HostCpu,
                weight_representations: vec![WeightRepresentation::F32],
                cache_modes: vec![],
                context_limit_tokens: 32768,
                supports_streaming: true,
                supports_incremental_kv: false,
                supports_prefix_reuse: false,
                decoding_modes: vec![DecodingCapability::Greedy],
            })
        })
    }
}
struct Events(VecDeque<GenerationEvent>);
impl GenerationEventStream for Events {
    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        Poll::Ready(self.get_mut().0.pop_front())
    }
}
