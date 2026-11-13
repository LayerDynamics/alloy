//! Versioned Llama text and XML tool-proposal templates over the shared renderer.
pub const CHAT_TEMPLATE: &str = include_str!("../fixture/chat_template.jinja");
pub const TOOL_CHAT_TEMPLATE: &str = include_str!("../fixture/tool_chat_template.jinja");
pub const BOS: u32 = 128000;
pub const END_OF_TEXT: u32 = 128001;
pub const END_OF_MESSAGE: u32 = 128008;
pub const TOOL_HANDOFF: u32 = 128008;
pub const END_OF_TURN: u32 = 128009;
