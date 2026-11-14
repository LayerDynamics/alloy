//! Bounded Qwen tool-envelope parsing in caller-admitted storage.
//! Success-path JSON validation uses no Map, Value, String, or Vec storage.
//! The bounded recursive control flow is capped at 128 levels; heap scratch,
//! keys, nodes, pending bytes, and retained proposals reside in the slab.

mod parser;
mod xml;
mod xml_stream;

pub use xml::{XmlParameterType, XmlToolSchema, commit_xml_tool_call};
pub use xml_stream::XmlToolParser;

pub(crate) use parser::*;
