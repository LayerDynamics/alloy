use minijinja::{Environment, UndefinedBehavior, context};
use sha2::{Digest, Sha256};

/// Embedded, versioned text-only, non-thinking chat serialization.
/// Tool calls and multimodal messages are deliberately unsupported.
pub const CHAT_TEMPLATE: &str = include_str!("../fixture/chat_template.jinja");
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateError {
    InvalidMessages,
    /// Message content contains a tokenizer control spelling or reserved delimiter.
    ReservedControlToken,
    Render,
}
impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMessages => write!(f, "unsupported role or empty conversation"),
            Self::ReservedControlToken => {
                write!(f, "message body contains reserved Qwen control syntax")
            }
            Self::Render => write!(f, "chat template rendering failed"),
        }
    }
}
impl std::error::Error for TemplateError {}
/// Hash the exact versioned template bytes, independently of message content.
pub fn template_hash() -> String {
    format!("{:x}", Sha256::digest(CHAT_TEMPLATE.as_bytes()))
}
/// Reject all <|... special-token syntax, including unknown future spellings,
/// and the reasoning/tool delimiters reserved by this text-only interface.
fn contains_reserved_control(content: &str) -> bool {
    content.contains("<|")
        || [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ]
        .iter()
        .any(|token| content.contains(token))
}
/// Render only validated messages. This prevents token-delimiter turn forgery;
/// it is not a general defense against semantic prompt injection.
pub fn render_chat(messages: &[(&str, &str)]) -> Result<String, TemplateError> {
    if messages.is_empty()
        || messages
            .iter()
            .any(|(role, _)| !matches!(*role, "system" | "user" | "assistant"))
    {
        return Err(TemplateError::InvalidMessages);
    }
    if messages
        .iter()
        .any(|(_, content)| contains_reserved_control(content))
    {
        return Err(TemplateError::ReservedControlToken);
    }
    let messages: Vec<_> = messages
        .iter()
        .map(|(role, content)| context! {role=>*role,content=>*content})
        .collect();
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment
        .add_template("qwen3-text", CHAT_TEMPLATE)
        .map_err(|_| TemplateError::Render)?;
    environment
        .get_template("qwen3-text")
        .map_err(|_| TemplateError::Render)?
        .render(context! {messages=>messages})
        .map_err(|_| TemplateError::Render)
}
