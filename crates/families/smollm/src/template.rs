//! Deterministic non-thinking text and XML-tool profiles for SmolLM3.
pub const CHAT_TEMPLATE: &str = include_str!("../fixture/chat_template.jinja");
pub const TOOL_CHAT_TEMPLATE: &str = include_str!("../fixture/tool_chat_template.jinja");
pub const END_OF_TURN: u32 = 128012;
