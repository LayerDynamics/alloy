use minijinja::{Environment, UndefinedBehavior, context};
use sha2::{Digest, Sha256};

/// Embedded, versioned text-only, non-thinking chat serialization.
/// Tool calls and multimodal messages are deliberately unsupported.
pub const CHAT_TEMPLATE: &str = include_str!("../fixture/chat_template.jinja");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateError {
    InvalidMessages,
    Render,
}
impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMessages => write!(f, "unsupported role or empty conversation"),
            Self::Render => write!(f, "chat template rendering failed"),
        }
    }
}
impl std::error::Error for TemplateError {}

/// Identity is computed from the exact template bytes, not rendered messages.
pub fn template_hash() -> String {
    format!("{:x}", Sha256::digest(CHAT_TEMPLATE.as_bytes()))
}

/// Render the supported text-only template with an assistant generation prompt.
pub fn render_chat(messages: &[(&str, &str)]) -> Result<String, TemplateError> {
    if messages.is_empty()
        || messages
            .iter()
            .any(|(role, _)| !matches!(*role, "system" | "user" | "assistant"))
    {
        return Err(TemplateError::InvalidMessages);
    }
    let messages: Vec<_> = messages
        .iter()
        .map(|(role, content)| {
            context! { role => *role, content => *content }
        })
        .collect();
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment
        .add_template("qwen3-text", CHAT_TEMPLATE)
        .map_err(|_| TemplateError::Render)?;
    environment
        .get_template("qwen3-text")
        .map_err(|_| TemplateError::Render)?
        .render(context! { messages => messages })
        .map_err(|_| TemplateError::Render)
}
