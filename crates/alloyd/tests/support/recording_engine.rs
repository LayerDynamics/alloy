//! Observe real requests without replacing rendering, generation, or cache state.
use engine::{
    api::{Engine, EngineCore, EngineFuture, ProfileContextBudget, ProjectedToolRender},
    error::EngineError,
    runtime::FamilyRuntime,
    tiered_kv::TieredKvSnapshot,
    types::*,
};
use std::{
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
};

struct CheckpointTrace {
    inner: GenerationStream,
    request_id: RequestId,
    text: String,
    label: &'static str,
}

impl GenerationEventStream for CheckpointTrace {
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>> {
        let result = self.inner.as_mut().poll_next(cx);
        match &result {
            Poll::Ready(Some(GenerationEvent::TextDelta { text })) => self.text.push_str(text),
            Poll::Ready(Some(
                GenerationEvent::GenerationCompleted { .. } | GenerationEvent::RequestFailed { .. },
            )) => {
                eprintln!(
                    "{} id={} output={:?} terminal={result:?}",
                    self.label,
                    self.request_id.as_str(),
                    self.text
                );
            }
            _ => {}
        }
        result
    }
}

pub struct RecordingEngine {
    pub inner: EngineCore<FamilyRuntime>,
    pub requests: Mutex<Vec<(GenerationRequest, TieredKvSnapshot)>>,
}

impl RecordingEngine {
    pub fn new(inner: EngineCore<FamilyRuntime>) -> Self {
        Self {
            inner,
            requests: Mutex::new(Vec::new()),
        }
    }
    pub fn runtime(&self) -> &FamilyRuntime {
        self.inner.runtime()
    }
}

impl Engine for RecordingEngine {
    fn api_version(&self) -> u32 {
        self.inner.api_version()
    }
    fn profile_context_budget(
        &self,
        profile: &ModelProfileId,
    ) -> Result<Option<ProfileContextBudget>, EngineError> {
        self.inner.profile_context_budget(profile)
    }
    fn render_and_count(&self, context: ContextSnapshot) -> EngineFuture<'_, RenderedContext> {
        self.inner.render_and_count(context)
    }
    fn check_projected_tool_render<'a>(
        &'a self,
        projection: ProjectedToolRender<'a>,
    ) -> EngineFuture<'a, ()> {
        self.inner.check_projected_tool_render(projection)
    }
    fn check_projected_context(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
        growth_tokens: usize,
    ) -> EngineFuture<'_, ()> {
        self.inner
            .check_projected_context(profile, input_tokens, growth_tokens)
    }
    fn estimate_memory(
        &self,
        profile: ModelProfileId,
        input_tokens: usize,
    ) -> EngineFuture<'_, MemoryEstimate> {
        self.inner.estimate_memory(profile, input_tokens)
    }
    fn generate(&self, request: GenerationRequest) -> EngineFuture<'_, Generation> {
        let trace_all = std::env::var_os("ALLOY_TEST_TRACE_REQUESTS").is_some();
        if trace_all {
            let rendered = request.expected_render().expect("counted native request");
            eprintln!(
                "native_request={}",
                serde_json::json!({
                    "id": request.request_id().as_str(),
                    "profile": request.profile_id().as_str(),
                    "tokens": rendered.token_ids(),
                    "text": rendered.text(),
                })
            );
        }
        self.requests
            .lock()
            .unwrap()
            .push((request.clone(), self.runtime().tiered_kv_snapshot()));
        Box::pin(async move {
            let generation = self.inner.generate(request).await?;
            let id = generation.request_id().clone();
            let checkpoint = id.as_str().contains(":compact:") || id.as_str().contains(":verify:");
            if trace_all || checkpoint {
                Ok(Generation::from_stream(
                    id.clone(),
                    Box::pin(CheckpointTrace {
                        inner: generation.into_events(),
                        request_id: id,
                        text: String::new(),
                        label: if checkpoint {
                            "checkpoint_trace"
                        } else {
                            "native_trace"
                        },
                    }),
                ))
            } else {
                Ok(generation)
            }
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
