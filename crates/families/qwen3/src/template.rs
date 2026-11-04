/// Failure modes of chat-template rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateError {
    /// The conversation did not fit the template schema.
    InvalidMessages,
    /// Rendering failed inside the template engine.
    Render,
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::InvalidMessages => write!(f, "messages do not fit the template schema"),
            TemplateError::Render => write!(f, "template rendering failed"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Render a chat through the family's embedded Qwen3 chat template with
/// minijinja. Taking that dependency requires pinning the exact version
/// and amending DR-0001 (issue #10) — do it in the implementation commit,
/// not a mixed one.
pub fn render_chat(messages: &[(&str, &str)]) -> Result<String, TemplateError> {
    let _ = messages;
    todo!("minijinja chat-template rendering (issue #3; pin per issue #10)")
}
