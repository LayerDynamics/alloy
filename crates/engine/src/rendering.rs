//! Shared exact text rendering and tokenization.
//!
//! This module provides:
//!
//! - SHA-256 verification of supplied template and tokenizer source bytes.
//! - Explicit message, input, template, rendered-output, and token limits.
//! - Strict MiniJinja rendering with explicitly selected whitespace options.
//! - Family-selected reserved-token spelling rejection.
//! - Exact Hugging Face tokenizer output.
//! - Versioned render-input identity hashing.
//!
//! Rendering is not model loading, artifact verification for all checkpoint
//! files, execution admission, or proof that reusable KV state exists.
//!
//! The caller must obtain expected digests from its trusted artifact policy.
//! Hashing arbitrary input and accepting that same hash as trusted does not
//! establish artifact authenticity.
//!
//! Limits bound accepted inputs and retained rendered output. They do not
//! constitute a hard limit on all allocations or CPU work performed inside
//! MiniJinja or the tokenizer. Those libraries' metadata, intermediate
//! allocations, and execution costs require appropriate external budgeting.
//!
//! Only trusted, pinned templates should be supplied. No filesystem loader
//! or dynamic template source is configured.
//!
//! Render hashes are distinct from canonical SessionStateId values.
//! Do not substitute a render hash for cache_identity's session-state key.

use crate::{
    error::{EngineError, MemoryLimitReason},
    types::{
        ArtifactId, ContextSnapshot, ModelProfileId, RenderedContext, RendererIdentity, TokenizerId,
    },
};
use minijinja::{
    AutoEscape, Environment, Error as TemplateError, ErrorKind as TemplateErrorKind,
    UndefinedBehavior,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use tokenizers::Tokenizer;

/// Exact MiniJinja version pinned by the workspace.
///
/// Update this value together with the workspace dependency pin and the
/// renderer compatibility decision.
pub const RENDERER_VERSION: &str = "2.12.0";

/// Exact tokenizer library version pinned by the engine manifest.
///
/// Update this value together with the dependency pin and tokenizer
/// compatibility decision.
pub const TOKENIZER_LIBRARY_VERSION: &str = "0.23.2";

/// Version of this module's render-hash encoding.
///
/// This is not the canonical SessionStateId encoding version.
pub const RENDER_HASH_VERSION: u32 = 1;

/// Explicit resource limits for one rendering operation.
///
/// No implicit production policy is selected by this type.
#[derive(Clone, Copy, Debug)]
pub struct RenderingLimits {
    max_messages: usize,
    max_input_bytes: usize,
    max_template_bytes: usize,
    max_rendered_bytes: usize,
    max_tokens: usize,
}

impl RenderingLimits {
    /// Construct explicit positive rendering limits.
    pub fn new(
        max_messages: usize,
        max_input_bytes: usize,
        max_template_bytes: usize,
        max_rendered_bytes: usize,
        max_tokens: usize,
    ) -> Result<Self, EngineError> {
        if [
            max_messages,
            max_input_bytes,
            max_template_bytes,
            max_rendered_bytes,
            max_tokens,
        ]
        .contains(&0)
        {
            return Err(EngineError::invalid_request(
                "rendering_limits",
                "all rendering limits must be greater than zero",
            ));
        }

        max_messages
            .checked_mul(std::mem::size_of::<Value>())
            .ok_or_else(|| {
                EngineError::invalid_request(
                    "max_messages",
                    "message-value header size arithmetic overflow",
                )
            })?;

        Ok(Self {
            max_messages,
            max_input_bytes,
            max_template_bytes,
            max_rendered_bytes,
            max_tokens,
        })
    }

    /// Maximum accepted message count.
    pub fn max_messages(&self) -> usize {
        self.max_messages
    }

    /// Maximum cumulative message-content bytes.
    pub fn max_input_bytes(&self) -> usize {
        self.max_input_bytes
    }

    /// Maximum template source bytes.
    pub fn max_template_bytes(&self) -> usize {
        self.max_template_bytes
    }

    /// Maximum retained rendered UTF-8 bytes.
    pub fn max_rendered_bytes(&self) -> usize {
        self.max_rendered_bytes
    }

    /// Maximum accepted tokenizer output length.
    ///
    /// Tokenization itself may allocate its result before this check.
    /// This does not silently truncate the input to this length.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

/// Explicit template and tokenization settings for a profile.
///
/// There is deliberately no Default implementation selecting a model's
/// generation-prompt or thinking-mode policy.
#[derive(Clone, Copy, Debug)]
pub struct RenderingOptions<'a> {
    /// Supply add_generation_prompt to the template.
    pub add_generation_prompt: bool,

    /// Supply enable_thinking when Some; omit the variable when None.
    ///
    /// The family adapter must select a value supported by its template.
    pub enable_thinking: Option<bool>,

    /// Whether the tokenizer should apply its configured special-token
    /// post-processing.
    ///
    /// Templates that already insert structural special tokens commonly
    /// require false. The selected family must decide explicitly.
    pub add_special_tokens: bool,

    /// Preserve a trailing newline in the template source.
    pub keep_trailing_newline: bool,

    /// Remove the first newline after a template block.
    pub trim_blocks: bool,

    /// Strip leading whitespace before template blocks.
    pub lstrip_blocks: bool,

    /// Structural spellings forbidden inside message content.
    ///
    /// The family adapter selects this list. Rejection is explicit;
    /// content is never silently escaped, removed, or rewritten.
    pub reserved_spellings: &'a [&'a str],
}

/// Trusted profile/artifact/template binding supplied by a backend.
///
/// Construction does not itself verify every artifact file. The caller
/// remains responsible for the verified artifact and manifest policy.
#[derive(Clone, Copy, Debug)]
pub struct RenderingBinding<'a> {
    /// Profile selected by the input snapshot.
    pub profile_id: &'a ModelProfileId,

    /// Artifact selected by that profile.
    pub artifact_id: &'a ArtifactId,

    /// Exact trusted template source.
    pub template: &'a str,

    /// Trusted SHA-256 of the exact template source bytes.
    pub template_sha256: [u8; 32],

    /// Trusted fingerprint of the selected artifact manifest.
    ///
    /// Its construction must be pinned by the artifact implementation.
    /// It must change when relevant pinned artifact identities change.
    pub artifact_manifest_sha256: [u8; 32],
}

/// Tokenizer parsed from source bytes checked against a trusted digest.
///
/// The tokenizer is exposed immutably so callers cannot silently change
/// truncation, padding, normalization, or post-processing on this instance
/// after deriving its identity.
pub struct PinnedTokenizer {
    tokenizer: Tokenizer,
    identity: TokenizerId,
    source_sha256: [u8; 32],
}

impl PinnedTokenizer {
    /// Verify bounded tokenizer.json bytes and parse the actual tokenizer.
    ///
    /// The caller supplies the already-read source buffer and its trusted
    /// expected digest. This method performs no filesystem reads.
    ///
    /// Configured truncation and padding are rejected because this engine
    /// path requires exact, untruncated model-visible input counts.
    pub fn from_pinned_json(
        source: &[u8],
        expected_sha256: [u8; 32],
        max_source_bytes: usize,
    ) -> Result<Self, EngineError> {
        if max_source_bytes == 0 {
            return Err(EngineError::invalid_request(
                "max_source_bytes",
                "tokenizer source limit must be greater than zero",
            ));
        }

        if source.len() > max_source_bytes {
            return Err(memory_limit(
                "read tokenizer source",
                source.len(),
                max_source_bytes,
            )?);
        }

        let observed = sha256(source);

        if observed != expected_sha256 {
            return Err(EngineError::ArtifactValidationFailed {
                artifact_id: None,
                message: "tokenizer source SHA-256 does not match its trusted pin".to_owned(),
            });
        }

        let tokenizer =
            Tokenizer::from_bytes(source).map_err(|source| EngineError::CheckpointFormat {
                format: "tokenizer.json",
                path: None,
                source,
            })?;

        if tokenizer.get_truncation().is_some() {
            return Err(EngineError::unsupported(
                "tokenizer truncation",
                "exact rendering does not permit configured truncation",
            ));
        }

        if tokenizer.get_padding().is_some() {
            return Err(EngineError::unsupported(
                "tokenizer padding",
                "exact rendering does not permit configured padding",
            ));
        }

        let mut hash = Sha256::new();
        hash.update(b"alloy.tokenizer.hf.v1\0");
        hash.update(TOKENIZER_LIBRARY_VERSION.as_bytes());
        hash.update(observed);

        let identity = TokenizerId::new(format!("sha256:{:x}", hash.finalize(),))?;

        Ok(Self {
            tokenizer,
            identity,
            source_sha256: observed,
        })
    }

    /// Borrow the actual tokenizer without permitting mutation.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Identity derived from the pinned source and tokenizer library version.
    pub fn identity(&self) -> &TokenizerId {
        &self.identity
    }

    /// SHA-256 of the actual tokenizer source bytes parsed.
    pub fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }
}

/// Bound the token count of any UTF-8 rendering of at most `rendered_bytes`
/// bytes for the pinned Qwen3-4B corpus tokenizer, without special-token
/// post-processing. This deliberately does not admit a request: callers must
/// bound the complete rendered text, including every projected result.
///
/// The pinned tokenizer uses the tokenizers 0.23.2 NFC tables, regex splitting
/// and ByteLevel BPE without a prefix space. NFC can be bounded by eight UTF-8
/// bytes per source byte: canonical decomposition produces at most two
/// scalars per input byte for the pinned Unicode tables (exhaustively checked
/// in `projected_context`), composition does not increase scalar count, and
/// each scalar is at most four bytes. ByteLevel BPE cannot emit more tokens
/// than its normalized input bytes; nonempty added tokens do not increase that
/// bound. Other tokenizer sources must provide their own verified bound.
pub fn qwen3_projected_token_upper_bound(
    tokenizer: &PinnedTokenizer,
    rendered_bytes: usize,
) -> Result<usize, EngineError> {
    const PIN: [u8; 32] = [
        0xae, 0xb1, 0x33, 0x07, 0xa7, 0x1a, 0xcd, 0x8f, 0xe8, 0x18, 0x61, 0xd9, 0x4a, 0xd5, 0x4a,
        0xb6, 0x89, 0xdf, 0x77, 0x33, 0x18, 0x80, 0x9e, 0xed, 0x3c, 0xbe, 0x79, 0x4b, 0x44, 0x92,
        0xda, 0xe4,
    ];
    if tokenizer.source_sha256() != &PIN {
        return Err(EngineError::unsupported(
            "projected tokenizer bound",
            "the selected tokenizer has no verified projected-input bound",
        ));
    }
    rendered_bytes.checked_mul(8).ok_or_else(|| {
        EngineError::invalid_request("rendered_bytes", "projected token bound overflow")
    })
}

/// Bound a Qwen tool-template skeleton after replacing its last result
/// placeholders with bounded results. The pinned tokenizer recognizes each
/// `<|im_start|>` as a non-normalized, non-stripping added token, so BPE and
/// NFC processing cannot cross those message boundaries. Existing segments
/// are charged at the exact skeleton count; each changing segment is charged
/// again at the conservative byte bound. The deliberate double charge avoids
/// subtracting tokens attributed to a segment.
pub fn qwen3_projected_tool_token_upper_bound(
    tokenizer: &PinnedTokenizer,
    rendered: &RenderedContext,
    result_count: usize,
    additional_bytes: usize,
    ascii_escape_bytes: usize,
) -> Result<usize, EngineError> {
    const START: &str = "<|im_start|>";
    const RESULT: &str = "<|im_start|>tool\n<tool_response>";
    if result_count == 0 {
        return Err(EngineError::invalid_request(
            "projection",
            "a projected result is required",
        ));
    }
    let text = rendered.text();
    let mut segment_bytes = 0usize;
    let mut count = 0usize;
    for (start, _) in text.rmatch_indices(RESULT).take(result_count) {
        let next = text[start + RESULT.len()..]
            .find(START)
            .map(|offset| start + RESULT.len() + offset)
            .ok_or_else(|| {
                EngineError::invalid_request(
                    "projection",
                    "result placeholder has no following message boundary",
                )
            })?;
        let segment = &text[start..next];
        if !segment.contains("\"content\":\"\"") || !segment.contains("</tool_response><|im_end|>")
        {
            return Err(EngineError::invalid_request(
                "projection",
                "result placeholder is not empty",
            ));
        }
        segment_bytes = segment_bytes.checked_add(segment.len()).ok_or_else(|| {
            EngineError::invalid_request("projection", "result segment byte overflow")
        })?;
        count += 1;
    }
    if count != result_count {
        return Err(EngineError::invalid_request(
            "projection",
            "missing result placeholders",
        ));
    }
    let variable_bytes = segment_bytes
        .checked_add(
            additional_bytes
                .checked_sub(ascii_escape_bytes)
                .ok_or_else(|| {
                    EngineError::invalid_request("projection", "ASCII escape bound exceeds total")
                })?,
        )
        .ok_or_else(|| EngineError::invalid_request("projection", "result byte bound overflow"))?;
    rendered
        .token_count()
        .checked_add(qwen3_projected_token_upper_bound(
            tokenizer,
            variable_bytes,
        )?)
        .and_then(|n| n.checked_add(ascii_escape_bytes))
        .ok_or_else(|| EngineError::invalid_request("projection", "token bound overflow"))
}

/// Render exact model-visible text from a structured text snapshot.
///
/// Uses the supplied pinned template. No fallback template is selected.
///
/// Template variables supplied by this implementation are:
///
/// - messages: ordered objects with role and content.
/// - add_generation_prompt: the explicit profile-selected flag.
/// - enable_thinking: present only when explicitly configured.
///
/// Unsupported variables, filters, includes, or functions return errors
/// rather than silently producing a substituted prompt.
pub fn render_chat(
    context: &ContextSnapshot,
    binding: &RenderingBinding<'_>,
    options: &RenderingOptions<'_>,
    limits: RenderingLimits,
) -> Result<String, EngineError> {
    validate_render_input(context, binding, options, limits)?;

    let header_bytes = context
        .messages()
        .len()
        .checked_mul(std::mem::size_of::<Value>())
        .ok_or_else(|| internal("message-value header size overflow"))?;

    let mut messages = Vec::new();

    messages
        .try_reserve_exact(context.messages().len())
        .map_err(|_| EngineError::AllocationFailed {
            operation: "allocate template message values",
            requested_bytes: u64::try_from(header_bytes).ok(),
        })?;

    for message in context.messages() {
        let mut value = Map::new();

        value.insert(
            "role".to_owned(),
            Value::String(message.role.as_str().to_owned()),
        );

        value.insert("content".to_owned(), Value::String(message.content.clone()));

        value.insert("kind".to_owned(), Value::String("text".to_owned()));
        messages.push(Value::Object(value));
    }
    if let Some(tools) = context.tool_context() {
        messages.extend(tools.messages());
    }

    let mut variables = Map::new();
    if let Some(tools) = context.tool_context() {
        variables.insert("tools_json".to_owned(), Value::String(tools.tools_json()?));
    }

    variables.insert("messages".to_owned(), Value::Array(messages));

    variables.insert(
        "add_generation_prompt".to_owned(),
        Value::Bool(options.add_generation_prompt),
    );

    if let Some(enabled) = options.enable_thinking {
        variables.insert("enable_thinking".to_owned(), Value::Bool(enabled));
    }

    let mut environment = Environment::new();

    environment.set_undefined_behavior(UndefinedBehavior::Strict);

    environment.set_auto_escape_callback(|_| AutoEscape::None);

    environment.set_keep_trailing_newline(options.keep_trailing_newline);

    environment.set_trim_blocks(options.trim_blocks);
    environment.set_lstrip_blocks(options.lstrip_blocks);

    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, TemplateError> {
            Err(TemplateError::new(
                TemplateErrorKind::InvalidOperation,
                message,
            ))
        },
    );

    environment
        .add_template("chat.txt", binding.template)
        .map_err(|error| execution_error("compile chat template", error.to_string()))?;

    let template = environment
        .get_template("chat.txt")
        .map_err(|error| execution_error("resolve chat template", error.to_string()))?;

    let mut writer = BoundedWriter::new(limits.max_rendered_bytes);

    let rendering = template.render_to_write(Value::Object(variables), &mut writer);

    // Inspect the recorded writer failure rather than parsing a template
    // error's human-readable text.
    if let Some(failure) = writer.failure.take() {
        return Err(failure.into_engine_error(limits.max_rendered_bytes));
    }

    rendering.map_err(|error| execution_error("render chat template", error.to_string()))?;

    String::from_utf8(writer.bytes)
        .map_err(|error| execution_error("decode rendered template UTF-8", error.to_string()))
}

/// Render and tokenize exact model-visible input.
///
/// Token IDs come directly from the supplied actual tokenizer. Their
/// count is derived from the returned sequence, not estimated from text.
///
/// This does not reserve execution memory or apply the profile's output
/// reserve and safety margin. The engine must still perform those checks.
pub fn render_and_count(
    context: &ContextSnapshot,
    binding: &RenderingBinding<'_>,
    tokenizer: &PinnedTokenizer,
    options: &RenderingOptions<'_>,
    limits: RenderingLimits,
) -> Result<RenderedContext, EngineError> {
    let text = render_chat(context, binding, options, limits)?;

    let encoding = tokenizer
        .tokenizer
        .encode(text.as_str(), options.add_special_tokens)
        .map_err(|error| execution_error("tokenize rendered context", error.to_string()))?;

    let ids = encoding.get_ids();

    if ids.is_empty() {
        return Err(EngineError::invalid_request(
            "token_ids",
            "rendered context produced no input tokens",
        ));
    }

    if ids.len() > limits.max_tokens {
        return Err(EngineError::ContextLimitExceeded {
            input_tokens: ids.len(),
            output_reserve_tokens: 0,
            safety_margin_tokens: 0,
            limit_tokens: limits.max_tokens,
        });
    }

    let token_bytes = ids
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| internal("rendered token storage size overflow"))?;

    let mut token_ids = Vec::new();

    token_ids
        .try_reserve_exact(ids.len())
        .map_err(|_| EngineError::AllocationFailed {
            operation: "allocate rendered token IDs",
            requested_bytes: u64::try_from(token_bytes).ok(),
        })?;

    token_ids.extend_from_slice(ids);

    let render_hash = compute_render_hash(context, binding, tokenizer, options, &text, &token_ids)?;

    Ok(RenderedContext {
        snapshot_id: context.snapshot_id().clone(),
        profile_id: context.profile_id().clone(),
        artifact_id: binding.artifact_id.clone(),
        text,
        token_ids,
        render_hash,
        template_hash: binding.template_sha256,
        tool_schema_hash: context.tool_context().map(|tools| *tools.schema_hash()),
        tokenizer_id: tokenizer.identity.clone(),
        renderer: RendererIdentity {
            implementation: "minijinja".to_owned(),
            version: RENDERER_VERSION.to_owned(),
        },
    })
}

fn validate_render_input(
    context: &ContextSnapshot,
    binding: &RenderingBinding<'_>,
    options: &RenderingOptions<'_>,
    limits: RenderingLimits,
) -> Result<(), EngineError> {
    if context.profile_id() != binding.profile_id {
        return Err(EngineError::invalid_request(
            "profile_id",
            "rendering binding does not match the context profile",
        ));
    }

    if binding.template.len() > limits.max_template_bytes {
        return Err(memory_limit(
            "compile chat template",
            binding.template.len(),
            limits.max_template_bytes,
        )?);
    }

    if sha256(binding.template.as_bytes()) != binding.template_sha256 {
        return Err(EngineError::ArtifactValidationFailed {
            artifact_id: Some(binding.artifact_id.as_str().to_owned()),
            message: "chat template SHA-256 does not match its trusted pin".to_owned(),
        });
    }

    let message_count = context
        .messages()
        .len()
        .checked_add(
            context
                .tool_context()
                .map_or(0, |tools| tools.history_len()),
        )
        .ok_or_else(|| internal("tool message count overflow"))?;
    if message_count > limits.max_messages {
        return Err(EngineError::invalid_request(
            "messages",
            "message count exceeds the rendering limit",
        ));
    }

    for spelling in options.reserved_spellings {
        if spelling.is_empty() {
            return Err(EngineError::invalid_request(
                "reserved_spellings",
                "reserved structural spellings must not be empty",
            ));
        }
    }

    let mut input_bytes = context
        .tool_context()
        .map_or(0, |tools| tools.input_bytes());
    if input_bytes > limits.max_input_bytes {
        return Err(memory_limit(
            "accept tool rendering input",
            input_bytes,
            limits.max_input_bytes,
        )?);
    }
    if let Some(tools) = context.tool_context() {
        tools.validate_spellings(options.reserved_spellings)?;
    }

    for message in context.messages() {
        input_bytes = input_bytes
            .checked_add(message.content.len())
            .ok_or_else(|| internal("message-content byte count overflow"))?;

        if input_bytes > limits.max_input_bytes {
            return Err(memory_limit(
                "accept rendering input",
                input_bytes,
                limits.max_input_bytes,
            )?);
        }

        if options
            .reserved_spellings
            .iter()
            .any(|spelling| message.content.contains(*spelling))
        {
            return Err(EngineError::invalid_request(
                "message.content",
                "message contains a reserved structural token spelling",
            ));
        }
    }

    Ok(())
}

fn compute_render_hash(
    context: &ContextSnapshot,
    binding: &RenderingBinding<'_>,
    tokenizer: &PinnedTokenizer,
    options: &RenderingOptions<'_>,
    text: &str,
    token_ids: &[u32],
) -> Result<[u8; 32], EngineError> {
    let mut hash = Sha256::new();

    hash.update(b"alloy.engine.render.v1\0");
    hash.update(RENDER_HASH_VERSION.to_le_bytes());
    if let Some(tools) = context.tool_context() {
        hash_field(&mut hash, b"alloy.render.tool-context.v1")?;
        hash_field(&mut hash, tools.schema_hash())?;
    }

    for field in [
        context.profile_id().as_str().as_bytes(),
        binding.artifact_id.as_str().as_bytes(),
        tokenizer.identity.as_str().as_bytes(),
        RENDERER_VERSION.as_bytes(),
        TOKENIZER_LIBRARY_VERSION.as_bytes(),
        text.as_bytes(),
    ] {
        hash_field(&mut hash, field)?;
    }

    hash_field(&mut hash, &binding.template_sha256)?;

    hash_field(&mut hash, &binding.artifact_manifest_sha256)?;

    hash_field(&mut hash, &tokenizer.source_sha256)?;

    let thinking = match options.enable_thinking {
        None => 0_u8,
        Some(false) => 1_u8,
        Some(true) => 2_u8,
    };

    let flags = [
        u8::from(options.add_generation_prompt),
        thinking,
        u8::from(options.add_special_tokens),
        u8::from(options.keep_trailing_newline),
        u8::from(options.trim_blocks),
        u8::from(options.lstrip_blocks),
        u8::from(tokenizer.tokenizer.get_encode_special_tokens()),
    ];

    hash_field(&mut hash, &flags)?;

    hash.update(as_u64(token_ids.len())?.to_le_bytes());

    for token in token_ids {
        hash.update(token.to_le_bytes());
    }

    Ok(hash.finalize().into())
}

fn hash_field(hash: &mut Sha256, bytes: &[u8]) -> Result<(), EngineError> {
    hash.update(as_u64(bytes.len())?.to_le_bytes());
    hash.update(bytes);
    Ok(())
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn as_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value).map_err(|_| internal("rendering size cannot be represented as u64"))
}

fn memory_limit(
    operation: &'static str,
    required: usize,
    available: usize,
) -> Result<EngineError, EngineError> {
    Ok(EngineError::MemoryLimitExceeded {
        operation,
        required_bytes: as_u64(required)?,
        available_bytes: as_u64(available)?,
        reason: MemoryLimitReason::BufferLimit,
    })
}

fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        message: message.to_owned(),
    }
}

fn execution_error(stage: &'static str, message: String) -> EngineError {
    EngineError::ExecutionFailed { stage, message }
}

enum WriterFailure {
    Limit { required: usize },
    SizeOverflow,
    Allocation { requested: usize },
}

impl WriterFailure {
    fn into_engine_error(self, limit: usize) -> EngineError {
        match self {
            Self::Limit { required } => {
                match memory_limit("render chat template output", required, limit) {
                    Ok(error) | Err(error) => error,
                }
            }

            Self::SizeOverflow => internal("rendered output byte count overflow"),

            Self::Allocation { requested } => EngineError::AllocationFailed {
                operation: "allocate rendered template output",
                requested_bytes: u64::try_from(requested).ok(),
            },
        }
    }
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
    failure: Option<WriterFailure>,
}

impl BoundedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            failure: None,
        }
    }

    fn fail(&mut self, failure: WriterFailure) -> io::Error {
        if self.failure.is_none() {
            self.failure = Some(failure);
        }

        io::Error::other("bounded template output failed")
    }
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other(
                "bounded template output previously failed",
            ));
        }

        let required = match self.bytes.len().checked_add(bytes.len()) {
            Some(required) => required,
            None => {
                return Err(self.fail(WriterFailure::SizeOverflow));
            }
        };

        if required > self.limit {
            return Err(self.fail(WriterFailure::Limit { required }));
        }

        if self.bytes.try_reserve_exact(bytes.len()).is_err() {
            return Err(self.fail(WriterFailure::Allocation {
                requested: required,
            }));
        }

        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.failure.is_some() {
            return Err(io::Error::other(
                "bounded template output previously failed",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/unit/rendering_tests.rs"]
mod tests;
