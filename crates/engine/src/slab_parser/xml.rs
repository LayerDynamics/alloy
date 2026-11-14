//! Trained Qwen3.5/Coder function/parameter grammar in fixed parser storage.
//! Schema resolution belongs to the caller; conversion never guesses a type.
use super::parser::{commit_json_call, validate_json_parameter};
use crate::{error::EngineError, parser_allocation::ParserAllocation};

/// Declared JSON value kind for one parameter of a registered tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XmlParameterType {
    /// Literal text, including numeric-looking text and the word `null`.
    String,
    /// A signed or unsigned JSON integer representable in 64 bits.
    Integer,
    /// A finite JSON number.
    Number,
    /// The literal `true` or `false`, case insensitive.
    Boolean,
    /// A JSON object, with duplicate keys rejected recursively.
    Object,
    /// A JSON array, with duplicate object keys rejected recursively.
    Array,
    /// The literal `null`, case insensitive, permitted by the schema.
    Null,
}

/// Read-only lookup in the exact advertised tool schemas for this generation.
/// Implementations must reject unsupported schema constructs rather than guess.
pub trait XmlToolSchema {
    /// Reject an unknown tool, including a call with no parameters.
    fn validate_tool(&self, name: &str) -> Result<(), EngineError>;
    /// Resolve a parameter's declared type; raw text permits explicit nullable
    /// unions to select `Null`. This is conversion, not full schema validation.
    fn parameter_type(
        &self,
        tool: &str,
        parameter: &str,
        raw: &str,
    ) -> Result<XmlParameterType, EngineError>;
}

fn bad(message: &'static str) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "XML tool parsing",
        message: message.into(),
    }
}

/// Validate and retain one complete trained XML function body.
///
/// The caller writes `call_bytes` bytes into `allocation.workspaces().call`;
/// outer `<tool_call>` delimiters are excluded. The resolver must obtain the
/// parameter type from the exact generation's advertised schema and reject
/// unknown tools, parameters or unsupported types. It must not infer types
/// from model output. A nullable union can resolve `Null` only when its schema
/// permits it; the raw value is supplied for that decision.
///
/// All conversion/validation scratch uses the admitted slab. Successful
/// conversion performs no heap allocation apart from any work in the resolver.
/// One template-added newline at each value boundary is removed; other text
/// whitespace and Unicode are preserved. JSON strings are escaped, not parsed
/// as Python literals. No XML entity decoding is applied to this trained syntax.
///
/// A failure publishes no proposal for this call. The caller must poison and
/// discard the entire generation on failure, including previously retained
/// proposals. Scratch bytes may change. Freezing is not execution authorization.
pub fn commit_xml_tool_call(
    allocation: &mut ParserAllocation,
    call_bytes: usize,
    schema: &impl XmlToolSchema,
) -> Result<(), EngineError> {
    let converted = {
        let w = allocation.workspaces();
        let body = w
            .call
            .get(..call_bytes)
            .ok_or_else(|| bad("call capacity exceeded"))?;
        let input = std::str::from_utf8(body).map_err(|_| bad("invalid UTF-8 in XML call"))?;
        let mut cursor = Cursor { rest: input };
        cursor.ws();
        let name = cursor.name("<function=")?;
        schema.validate_tool(name)?;
        let mut out = Writer {
            bytes: w.decoded,
            used: 0,
        };
        out.write(b"{\"name\":")?;
        out.string(name)?;
        out.write(b",\"arguments\":{")?;
        let mut count = 0;
        loop {
            cursor.ws();
            if cursor.take("</function>") {
                break;
            }
            let parameter = cursor.name("<parameter=")?;
            let end = cursor
                .rest
                .find("</parameter>")
                .ok_or_else(|| bad("incomplete parameter"))?;
            let raw = &cursor.rest[..end];
            cursor.rest = &cursor.rest[end + "</parameter>".len()..];
            // Exactly the publisher's added line boundaries, not trim().
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            if [
                "<tool_call",
                "</tool_call",
                "<function=",
                "</function>",
                "<parameter=",
                "<think>",
                "</think>",
                "<|",
            ]
            .iter()
            .any(|s| raw.contains(s))
            {
                return Err(bad("nested or reserved protocol syntax in parameter"));
            }
            let kind = schema.parameter_type(name, parameter, raw)?;
            if count != 0 {
                out.write(b",")?;
            }
            out.string(parameter)?;
            out.write(b":")?;
            match kind {
                XmlParameterType::String => out.string(raw)?,
                XmlParameterType::Integer => {
                    if raw.parse::<i64>().is_err() && raw.parse::<u64>().is_err() {
                        return Err(bad("invalid or out-of-range integer parameter"));
                    }
                    out.write(raw.as_bytes())?;
                }
                XmlParameterType::Number => {
                    if !raw.parse::<f64>().is_ok_and(f64::is_finite) {
                        return Err(bad("invalid or non-finite number parameter"));
                    }
                    out.write(raw.as_bytes())?;
                }
                XmlParameterType::Boolean => {
                    out.write(if raw.eq_ignore_ascii_case("true") {
                        b"true"
                    } else if raw.eq_ignore_ascii_case("false") {
                        b"false"
                    } else {
                        return Err(bad("invalid boolean parameter"));
                    })?;
                }
                XmlParameterType::Null => {
                    if !raw.eq_ignore_ascii_case("null") {
                        return Err(bad("invalid null parameter"));
                    }
                    out.write(b"null")?;
                }
                XmlParameterType::Object | XmlParameterType::Array => {
                    let raw = raw.trim_matches([' ', '\n', '\r', '\t']);
                    let expected = if kind == XmlParameterType::Object {
                        '{'
                    } else {
                        '['
                    };
                    if !raw.starts_with(expected) {
                        return Err(bad("parameter JSON kind mismatch"));
                    }
                    validate_json_parameter(
                        raw.as_bytes(),
                        &mut out.bytes[out.used..],
                        w.stack,
                        w.keys,
                        w.nodes,
                        if expected == '{' { 1 } else { 2 },
                    )?;
                    out.write(raw.as_bytes())?;
                }
            }
            count += 1;
        }
        cursor.ws();
        if !cursor.rest.is_empty() {
            return Err(bad("trailing XML call data"));
        }
        out.write(b"}}")?;
        if out.used > w.call.len() {
            return Err(bad("converted call capacity exceeded"));
        }
        w.call[..out.used].copy_from_slice(&out.bytes[..out.used]);
        out.used
    };
    // Reuse the exact JSON grammar, recursive duplicate-key checks, capacity
    // checks and transactional proposal publication used by the JSON protocol.
    commit_json_call(allocation, converted)
}

struct Cursor<'a> {
    rest: &'a str,
}
impl<'a> Cursor<'a> {
    fn ws(&mut self) {
        self.rest = self.rest.trim_start_matches([' ', '\n', '\r', '\t']);
    }
    fn take(&mut self, expected: &str) -> bool {
        if let Some(rest) = self.rest.strip_prefix(expected) {
            self.rest = rest;
            true
        } else {
            false
        }
    }
    fn name(&mut self, prefix: &str) -> Result<&'a str, EngineError> {
        if !self.take(prefix) {
            return Err(bad("expected XML function or parameter"));
        }
        let end = self
            .rest
            .find('>')
            .ok_or_else(|| bad("incomplete XML name"))?;
        let name = &self.rest[..end];
        if name.is_empty()
            || name.trim() != name
            || name
                .chars()
                .any(|c| c.is_control() || matches!(c, '<' | '>' | '='))
        {
            return Err(bad("invalid XML name"));
        }
        self.rest = &self.rest[end + 1..];
        Ok(name)
    }
}
struct Writer<'a> {
    bytes: &'a mut [u8],
    used: usize,
}
impl Writer<'_> {
    fn write(&mut self, input: &[u8]) -> Result<(), EngineError> {
        let end = self
            .used
            .checked_add(input.len())
            .ok_or_else(|| bad("conversion size overflow"))?;
        self.bytes
            .get_mut(self.used..end)
            .ok_or_else(|| bad("conversion scratch capacity exceeded"))?
            .copy_from_slice(input);
        self.used = end;
        Ok(())
    }
    fn string(&mut self, input: &str) -> Result<(), EngineError> {
        self.write(b"\"")?;
        for byte in input.bytes() {
            match byte {
                b'"' | b'\\' => self.write(&[b'\\', byte])?,
                0..=31 => {
                    const HEX: &[u8; 16] = b"0123456789abcdef";
                    self.write(&[
                        b'\\',
                        b'u',
                        b'0',
                        b'0',
                        HEX[usize::from(byte >> 4)],
                        HEX[usize::from(byte & 15)],
                    ])?;
                }
                _ => self.write(&[byte])?,
            }
        }
        self.write(b"\"")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen35_xml_tools.rs"]
mod tests;
