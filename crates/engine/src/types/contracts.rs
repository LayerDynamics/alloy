//! Shared types for the real engine API.
//!
//! Request types validate structural invariants. They do not prove that an
//! artifact is verified, a profile is supported, or a request is admitted.
//! Those checks belong to the registry and concrete engine implementation.

use crate::error::EngineError;
#[path = "tool_context.rs"]
mod tool_context;
use std::{
    fmt,
    num::NonZeroUsize,
    pin::Pin,
    task::{Context, Poll},
};
pub use tool_context::{RenderingTool, ToolContext, ToolContextLimits, ToolHistoryBlock};

/// Maximum byte length of a textual engine identifier.
pub const MAX_IDENTIFIER_BYTES: usize = 128;

/// Validate a registry identifier, not a filesystem path.
fn validate_identifier(value: &str, field: &'static str) -> Result<(), EngineError> {
    if value.is_empty() || value.len() > MAX_IDENTIFIER_BYTES {
        return Err(EngineError::invalid_request(
            field,
            format!(
                "identifier must contain between 1 and \
                 {MAX_IDENTIFIER_BYTES} bytes"
            ),
        ));
    }

    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(EngineError::invalid_request(
            field,
            "identifier may contain only ASCII letters, digits, '-', '_', '.', ':'",
        ));
    }

    Ok(())
}

macro_rules! identifier {
    ($name:ident, $description:literal, $field:literal) => {
        #[doc = $description]
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Construct a structurally valid identifier.
            ///
            /// This does not verify that the identified resource exists.
            pub fn new(value: impl Into<String>) -> Result<Self, EngineError> {
                let value = value.into();
                validate_identifier(&value, $field)?;
                Ok(Self(value))
            }

            /// Borrow the identifier's string representation.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Consume the identifier and return its string representation.
            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl TryFrom<String> for $name {
            type Error = EngineError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = EngineError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

identifier!(
    RequestId,
    "Identifier of one generation attempt.",
    "request_id"
);

identifier!(
    ModelProfileId,
    "Registry identifier of a model profile, not a filesystem path.",
    "profile_id"
);

identifier!(
    ArtifactId,
    "Registry identifier of an artifact; verification is a separate requirement.",
    "artifact_id"
);

identifier!(
    ContextSnapshotId,
    "Tracing identity of an immutable context snapshot, not a cache key.",
    "snapshot_id"
);

identifier!(
    TokenizerId,
    "Registry identity of the exact tokenizer configuration.",
    "tokenizer_id"
);

identifier!(
    CacheRepresentationId,
    "Stable identity of a family cache layout and its semantics.",
    "cache_representation"
);

/// Identity of engine-owned session state.
///
/// The digest must be derived using Appendix B.1's canonical BLAKE3
/// serialization before it is used as a cache-reuse identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionStateId([u8; 32]);

impl SessionStateId {
    /// Wrap an already-computed canonical identity digest.
    ///
    /// This method does not perform hashing or prove cache compatibility.
    /// Concrete engine implementations must not trust an arbitrary digest
    /// as evidence that reusable state exists.
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Borrow the identity digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for SessionStateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Role supported by the initial text-only context contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
    /// Instructions supplied by the harness.
    System,
    /// User-provided input.
    User,
    /// Prior assistant output.
    Assistant,
}

impl MessageRole {
    /// Return the role spelling supplied to a family renderer.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One structured text message.
///
/// Family-specific control-token restrictions are enforced by the
/// renderer before a message is included in model-visible input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// Structural message role.
    pub role: MessageRole,
    /// Text body, not pre-rendered conversation syntax.
    pub content: String,
}

/// Immutable input to profile-specific rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextSnapshot {
    snapshot_id: ContextSnapshotId,
    profile_id: ModelProfileId,
    messages: Vec<Message>,
    tool_context: Option<ToolContext>,
}

impl ContextSnapshot {
    /// Construct a nonempty text-context snapshot.
    pub fn new(
        snapshot_id: ContextSnapshotId,
        profile_id: ModelProfileId,
        messages: Vec<Message>,
    ) -> Result<Self, EngineError> {
        if messages.is_empty() {
            return Err(EngineError::invalid_request(
                "messages",
                "context snapshot must contain at least one message",
            ));
        }

        Ok(Self {
            snapshot_id,
            profile_id,
            messages,
            tool_context: None,
        })
    }

    /// Return the snapshot's tracing identifier.
    pub fn snapshot_id(&self) -> &ContextSnapshotId {
        &self.snapshot_id
    }

    /// Return the profile against which this snapshot must be rendered.
    pub fn profile_id(&self) -> &ModelProfileId {
        &self.profile_id
    }

    /// Borrow the ordered message sequence.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Attach immutable model-facing schemas and an ordered historical tail.
    /// Callers source metadata from validated registrations and bind historical
    /// IDs before construction. This does not authorize new effects.
    pub fn with_tool_context(mut self, context: ToolContext) -> Self {
        self.tool_context = Some(context);
        self
    }

    /// Borrow optional model-facing tool schemas and linked history.
    pub fn tool_context(&self) -> Option<&ToolContext> {
        self.tool_context.as_ref()
    }
}

/// Exact renderer implementation identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RendererIdentity {
    /// Renderer implementation name, such as `minijinja`.
    pub implementation: String,
    /// Exact renderer version.
    pub version: String,
}

/// Engine-produced rendering and tokenization result.
///
/// Fields are crate-visible so the real renderer can construct this
/// value. External callers cannot manufacture a verified rendering
/// through a public struct literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedContext {
    pub(crate) snapshot_id: ContextSnapshotId,
    pub(crate) profile_id: ModelProfileId,
    pub(crate) artifact_id: ArtifactId,
    pub(crate) text: String,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) render_hash: [u8; 32],
    pub(crate) template_hash: [u8; 32],
    pub(crate) tool_schema_hash: Option<[u8; 32]>,
    pub(crate) tokenizer_id: TokenizerId,
    pub(crate) renderer: RendererIdentity,
}

impl RenderedContext {
    /// Return the source snapshot identifier.
    pub fn snapshot_id(&self) -> &ContextSnapshotId {
        &self.snapshot_id
    }

    /// Return the profile used for rendering.
    pub fn profile_id(&self) -> &ModelProfileId {
        &self.profile_id
    }

    /// Return the verified artifact selected by the profile.
    pub fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    /// Borrow the exact rendered text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Borrow the exact tokenizer output.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Derive the exact token count from the token sequence.
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    /// Borrow the rendered-input digest.
    ///
    /// Its algorithm and included identity inputs must be pinned by the
    /// rendering implementation before compatibility is claimed.
    pub fn render_hash(&self) -> &[u8; 32] {
        &self.render_hash
    }

    /// Borrow the SHA-256 digest of the exact template source bytes.
    pub fn template_hash(&self) -> &[u8; 32] {
        &self.template_hash
    }

    /// Model-visible tool schema identity, independent of output token collisions.
    /// None preserves text-only compatibility and identity behavior.
    pub fn tool_schema_hash(&self) -> Option<&[u8; 32]> {
        self.tool_schema_hash.as_ref()
    }

    /// Return the tokenizer identity.
    pub fn tokenizer_id(&self) -> &TokenizerId {
        &self.tokenizer_id
    }

    /// Return the exact renderer identity.
    pub fn renderer(&self) -> &RendererIdentity {
        &self.renderer
    }
}

/// Requested token-selection behavior.
///
/// A structurally valid mode may still be unsupported by a particular
/// backend. The concrete engine must reject unsupported modes explicitly.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodingMode {
    /// Choose the highest-logit token deterministically.
    Greedy,
    /// Sample from a temperature-adjusted distribution.
    Sampling {
        /// Positive, finite sampling temperature.
        temperature: f32,
        /// Optional nucleus probability in `(0, 1]`.
        top_p: Option<f32>,
        /// Optional positive top-k limit.
        top_k: Option<NonZeroUsize>,
    },
}

/// Validated structural options for one generation.
///
/// Profile-specific context, output, and capability limits are checked
/// by the engine during admission.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationOptions {
    max_output_tokens: NonZeroUsize,
    decoding: DecodingMode,
    seed: Option<u64>,
}

impl GenerationOptions {
    /// Construct structurally valid generation options.
    ///
    /// A seed is only meaningful for sampling and is rejected for greedy
    /// decoding rather than silently ignored.
    pub fn new(
        max_output_tokens: usize,
        decoding: DecodingMode,
        seed: Option<u64>,
    ) -> Result<Self, EngineError> {
        let max_output_tokens = NonZeroUsize::new(max_output_tokens).ok_or_else(|| {
            EngineError::invalid_request("max_output_tokens", "must be greater than zero")
        })?;

        match &decoding {
            DecodingMode::Greedy => {
                if seed.is_some() {
                    return Err(EngineError::invalid_request(
                        "seed",
                        "a seed is not applicable to greedy decoding",
                    ));
                }
            }

            DecodingMode::Sampling {
                temperature, top_p, ..
            } => {
                if !temperature.is_finite() || *temperature <= 0.0 {
                    return Err(EngineError::invalid_request(
                        "temperature",
                        "must be finite and greater than zero",
                    ));
                }

                if let Some(probability) = top_p
                    && (!probability.is_finite() || *probability <= 0.0 || *probability > 1.0)
                {
                    return Err(EngineError::invalid_request(
                        "top_p",
                        "must be finite and within (0, 1]",
                    ));
                }
            }
        }

        Ok(Self {
            max_output_tokens,
            decoding,
            seed,
        })
    }

    /// Construct a greedy-generation request with an output-token limit.
    pub fn greedy(max_output_tokens: usize) -> Result<Self, EngineError> {
        Self::new(max_output_tokens, DecodingMode::Greedy, None)
    }

    /// Return the maximum number of newly generated tokens.
    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens.get()
    }

    /// Borrow the requested decoding mode.
    pub fn decoding(&self) -> &DecodingMode {
        &self.decoding
    }

    /// Return the optional sampling seed.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }
}

/// Structurally validated generation request.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    request_id: RequestId,
    profile_id: ModelProfileId,
    context: ContextSnapshot,
    options: GenerationOptions,
    expected_render: Option<RenderedContext>,
}

impl GenerationRequest {
    /// Construct a request whose profile agrees with its context snapshot.
    pub fn new(
        request_id: RequestId,
        profile_id: ModelProfileId,
        context: ContextSnapshot,
        options: GenerationOptions,
    ) -> Result<Self, EngineError> {
        if context.profile_id() != &profile_id {
            return Err(EngineError::invalid_request(
                "profile_id",
                "request profile does not match the context snapshot",
            ));
        }

        Ok(Self {
            request_id,
            profile_id,
            context,
            options,
            expected_render: None,
        })
    }

    /// Require generation to use precisely the previously counted rendering.
    /// The engine independently verifies this value against its current render.
    pub fn with_expected_render(mut self, rendered: RenderedContext) -> Result<Self, EngineError> {
        if rendered.snapshot_id() != self.context.snapshot_id()
            || rendered.profile_id() != &self.profile_id
        {
            return Err(EngineError::invalid_request(
                "expected_render",
                "rendered context does not match the request snapshot and profile",
            ));
        }
        self.expected_render = Some(rendered);
        Ok(self)
    }

    /// Previously counted rendering, if exact render verification is required.
    pub fn expected_render(&self) -> Option<&RenderedContext> {
        self.expected_render.as_ref()
    }

    /// Return the request identifier.
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// Return the selected profile.
    pub fn profile_id(&self) -> &ModelProfileId {
        &self.profile_id
    }

    /// Borrow the context to render.
    pub fn context(&self) -> &ContextSnapshot {
        &self.context
    }

    /// Borrow the generation options.
    pub fn options(&self) -> &GenerationOptions {
        &self.options
    }
}

/// Components of a memory reservation.
///
/// The concrete backend must account for its actual representations,
/// including any dequantized weights or host-side execution buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryBreakdown {
    /// Resident model-weight storage.
    pub weights_bytes: u64,
    /// KV state reserved for the admitted context.
    pub kv_bytes: u64,
    /// Prefill/decode workspaces and temporary representations.
    pub temporary_bytes: u64,
    /// Bounded output and event-buffer storage.
    pub output_bytes: u64,
    /// Draft-model storage, or zero when drafting is not used.
    pub draft_bytes: u64,
    /// Additional reserved operating headroom.
    pub headroom_bytes: u64,
}

impl MemoryBreakdown {
    /// Compute the complete reservation using checked arithmetic.
    pub fn total_bytes(&self) -> Result<u64, EngineError> {
        [
            self.weights_bytes,
            self.kv_bytes,
            self.temporary_bytes,
            self.output_bytes,
            self.draft_bytes,
            self.headroom_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| EngineError::Internal {
                    message: "memory reservation arithmetic overflow".to_owned(),
                })
        })
    }
}

/// An explicit action a caller may choose when admission does not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegradationOption {
    /// Release idle engine-owned session state.
    ReleaseIdleSessionState,
    /// Unload an idle model.
    UnloadIdleModel,
    /// Retry with a smaller prefill chunk.
    ReducePrefillChunk,
    /// Retry with a smaller permitted generation-output allowance.
    ReduceOutputLimit,
}

/// Advisory memory estimate, not an acquired reservation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEstimate {
    breakdown: MemoryBreakdown,
    reserved_bytes: u64,
    available_bytes: u64,
    degradation_options: Vec<DegradationOption>,
}

impl MemoryEstimate {
    /// Construct an estimate with a checked total.
    ///
    /// `available_bytes` must come from the engine's effective budget.
    /// Creating this value does not acquire memory or authorize execution.
    pub fn new(
        breakdown: MemoryBreakdown,
        available_bytes: u64,
        degradation_options: Vec<DegradationOption>,
    ) -> Result<Self, EngineError> {
        let reserved_bytes = breakdown.total_bytes()?;

        Ok(Self {
            breakdown,
            reserved_bytes,
            available_bytes,
            degradation_options,
        })
    }

    /// Return the individual reservation categories.
    pub fn breakdown(&self) -> &MemoryBreakdown {
        &self.breakdown
    }

    /// Return the checked reservation total.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    /// Return the applicable available budget.
    pub fn available_bytes(&self) -> u64 {
        self.available_bytes
    }

    /// Whether the estimate fits the budget captured by this estimate.
    pub fn fits(&self) -> bool {
        self.reserved_bytes <= self.available_bytes
    }

    /// Borrow the available explicit degradation actions.
    pub fn degradation_options(&self) -> &[DegradationOption] {
        &self.degradation_options
    }
}

/// Token usage measured during actual execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Number of input tokens actually supplied to execution.
    pub input_tokens: usize,
    /// Number of newly generated token IDs.
    pub output_tokens: usize,
}

impl Usage {
    /// Return the total token count using checked arithmetic.
    pub fn total_tokens(&self) -> Result<usize, EngineError> {
        self.input_tokens
            .checked_add(self.output_tokens)
            .ok_or_else(|| EngineError::Internal {
                message: "token usage arithmetic overflow".to_owned(),
            })
    }
}

/// Why a successful generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The profile's end-of-sequence condition was reached.
    EndOfSequence,
    /// The admitted output-token limit was reached.
    OutputLimit,
}

/// Outcome of a cancellation operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Cancellation was accepted for an active request.
    Requested,
    /// The request had already reached its terminal state.
    AlreadyFinished,
}

/// Outcome of releasing engine-owned session state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// State was released.
    Released,
    /// No state with this identity remained.
    AlreadyAbsent,
}

/// Event emitted by actual generation.
///
/// An active stream emits exactly one terminal event while its consumer
/// remains connected. No events may follow that terminal event.
#[derive(Debug)]
pub enum GenerationEvent {
    /// Incrementally detokenized model output.
    TextDelta {
        /// Newly decoded text, not the accumulated response.
        text: String,
    },

    /// Incremental structured tool-call content.
    ///
    /// Implementations without tool-call support must not emit this event.
    ToolCallDelta {
        /// Stable identifier of the call being assembled.
        call_id: String,
        /// Incremental call payload.
        delta: String,
    },

    /// Complete untrusted tool-call proposal from a finished generation.
    ///
    /// A producer publishes proposals only after the family protocol parser
    /// has accepted the whole generation and the engine has reached a
    /// `GenerationCompleted` outcome. Proposals follow every text delta and
    /// precede that terminal event. A consumer that observes any other
    /// terminal event must discard proposals it already received.
    ///
    /// The proposal has no daemon-issued call ID, registration, approval,
    /// or permission to execute.
    ToolCallProposed {
        /// Name and arguments exactly as proposed by the model.
        proposal: ToolCallProposal,
    },

    /// One owned batch of validated, untrusted slab-backed proposals.
    ///
    /// The backing reservation remains alive until this batch is dropped.
    /// The producer withholds this event until successful generation
    /// finalization. Consumers must still observe the successful terminal
    /// event before executing effects, and must apply registry/policy checks.
    ToolCallBatchProposed {
        /// Immutable proposal views and their backing allocation owner.
        proposals: crate::parser_allocation::FrozenParserAllocation,
    },

    /// Nonterminal execution warning.
    RuntimeWarning {
        /// Stable machine-readable warning code.
        code: String,
        /// Human-readable explanation.
        message: String,
    },

    /// Successful terminal event.
    GenerationCompleted {
        /// Why generation stopped.
        finish_reason: FinishReason,
        /// Measured token usage.
        usage: Usage,
    },

    /// Failed or cancelled terminal event.
    RequestFailed {
        /// Structured execution failure.
        error: EngineError,
        /// Usage measured before failure.
        partial_usage: Usage,
    },
}

impl GenerationEvent {
    /// Whether this event terminates generation.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::GenerationCompleted { .. } | Self::RequestFailed { .. }
        )
    }
}

/// Runtime-independent asynchronous event source.
///
/// Returning `Poll::Pending` requires registering the supplied waker.
/// After returning `Poll::Ready(None)`, the source remains exhausted.
///
/// The concrete implementation must provide bounded buffering and stop
/// actual execution when its consumer is dropped.
pub trait GenerationEventStream: Send {
    /// Poll for the next event produced by actual model execution.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<GenerationEvent>>;
}

/// Owned asynchronous generation-event stream.
pub type GenerationStream = Pin<Box<dyn GenerationEventStream>>;

/// Handle returned after a real generation request is accepted.
///
/// Only the engine implementation constructs this handle.
pub struct Generation {
    pub(crate) request_id: RequestId,
    pub(crate) events: GenerationStream,
}

impl Generation {
    /// Construct a generation handle from an accepted request and event stream.
    ///
    /// Engine runtimes and in-process adapters (including harness tests) use
    /// this to return a live stream after admission succeeds.
    pub fn from_stream(request_id: RequestId, events: GenerationStream) -> Self {
        Self { request_id, events }
    }

    /// Return the accepted request identifier.
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// Consume the handle and take ownership of its event stream.
    pub fn into_events(self) -> GenerationStream {
        self.events
    }
}

impl fmt::Debug for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Generation")
            .field("request_id", &self.request_id)
            .finish_non_exhaustive()
    }
}

/// Actual execution backend, not merely the process's default device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionBackend {
    /// Host-side CPU execution outside MLX operations.
    HostCpu,
    /// Execution through MLX CPU operations.
    MlxCpu,
    /// Execution through MLX Metal operations.
    MlxMetal,
    /// Explicitly selected external backend.
    External {
        /// Registered backend name.
        name: String,
    },
}

/// Representation actually retained for model weights.
///
/// Affine4 means packed uint32 codes + scales/biases stay resident and
/// execute via `mlx_quantized_matmul`. An artifact that is expanded to
/// host/device f32 storage is reported as `F32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRepresentation {
    /// IEEE 754 single-precision storage.
    F32,
    /// IEEE 754 half-precision storage.
    F16,
    /// Bfloat16 storage.
    Bf16,
    /// Resident affine four-bit storage.
    Affine4 {
        /// Number of weights sharing quantization parameters.
        group_size: NonZeroUsize,
    },
}

/// KV retention behavior supported by an implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheKind {
    /// Retain the full admitted prefix.
    Full,
    /// Retain a bounded rolling window.
    Rotating {
        /// Number of retained token positions.
        window_tokens: NonZeroUsize,
    },
}

/// Supported KV layout and retention behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheMode {
    /// Stable representation identity used by cache compatibility checks.
    pub representation: CacheRepresentationId,
    /// Retention behavior of that representation.
    pub kind: CacheKind,
}

/// Token-selection modes actually implemented by a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodingCapability {
    /// Highest-logit token selection.
    Greedy,
    /// Temperature-adjusted sampling.
    Sampling,
}

/// Capabilities of an artifact through its selected family/backend.
///
/// This is descriptive metadata, not authorization to execute. Admission
/// and profile validation still apply to every generation request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Supported task, such as `text-generation`.
    pub task: String,
    /// Registered family name.
    pub family: String,
    /// Actual execution backend.
    pub backend: ExecutionBackend,
    /// Weight representations supported by this implementation.
    pub weight_representations: Vec<WeightRepresentation>,
    /// Supported cache layouts and retention modes.
    pub cache_modes: Vec<CacheMode>,
    /// Validated model-visible context limit.
    pub context_limit_tokens: usize,
    /// Whether actual generation produces incremental events.
    pub supports_streaming: bool,
    /// Whether token-by-token execution retains KV state.
    pub supports_incremental_kv: bool,
    /// Whether this implementation can reuse compatible prefixes with an enabled
    /// retention path. This does not guarantee a hit or sufficient capacity;
    /// quotas, memory pressure, eviction, and the actual prompt still apply.
    pub supports_prefix_reuse: bool,
    /// Implemented decoding modes.
    pub decoding_modes: Vec<DecodingCapability>,
}

// W2_UNTRUSTED_GENERATION_OUTPUT_V1

/// A model-proposed tool name and object arguments, not an invocation.
///
/// This value has no daemon call/session identity, registry version,
/// implementation, approval, or policy. Construction does not resolve a
/// registration, validate its schema, or authorize an effect.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallProposal {
    name: String,
    arguments: serde_json::Map<String, serde_json::Value>,
}

impl ToolCallProposal {
    /// Check name structure while retaining arguments exactly as data.
    /// Parser and runtime boundaries must separately enforce byte/count limits.
    pub fn new(
        name: impl Into<String>,
        arguments: serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, EngineError> {
        let name = name.into();
        if name.is_empty() || name.trim() != name || name.chars().any(char::is_control) {
            return Err(EngineError::invalid_request(
                "tool_call.name",
                "tool name must be nonempty, unpadded, and contain no control characters",
            ));
        }
        Ok(Self { name, arguments })
    }

    /// Untrusted name awaiting resolution against the registry.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Untrusted object awaiting registered-schema validation.
    pub fn arguments(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.arguments
    }

    /// Transfer the proposed name and arguments without assigning identity.
    pub fn into_parts(self) -> (String, serde_json::Map<String, serde_json::Value>) {
        (self.name, self.arguments)
    }
}

/// Family-independent parsed output, before daemon invocation binding.
///
/// A complete call proposal is not an incremental JSON delta or permission
/// to execute. A producer must apply its protocol-completion and failure
/// rules before publishing a proposal.
#[derive(Clone, Debug, PartialEq)]
pub enum GenerationOutputBlock {
    /// Assistant text with family control syntax already handled.
    Text(String),
    /// Complete untrusted name/arguments, with no daemon-issued call ID.
    ToolCall(ToolCallProposal),
}

#[cfg(test)]
#[path = "../../tests/unit/types_output_contract.rs"]
mod output_contract_tests;

#[cfg(test)]
#[path = "../../tests/unit/types_proposal_event.rs"]
mod proposal_event_tests;
