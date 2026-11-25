//! Gemma 4 emits tool controls and string delimiters as tokenizer special IDs.
use super::{ministral3::Fragment, parser::validate_json_parameter};
use crate::{
    error::EngineError,
    parser_allocation::{FrozenParserAllocation, ParserAllocation},
};

const CALL: u32 = 48;
const CLOSE: u32 = 49;
const RESPONSE: u32 = 50;
const QUOTE: u32 = 52;

#[derive(Clone, Copy, Debug)]
enum Phase {
    Text,
    Call { quoted: bool },
    AfterCall,
    End,
    Failed,
}

fn invalid(reason: &'static str) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "Gemma 4 tool parsing",
        message: reason.into(),
    }
}

/// Write within the admitted slab, including when a JSON escape expands its input.
fn append(out: &mut [u8], used: &mut usize, bytes: &[u8]) -> Result<(), EngineError> {
    let end = used
        .checked_add(bytes.len())
        .ok_or_else(|| invalid("length overflow"))?;
    out.get_mut(*used..end)
        .ok_or_else(|| invalid("tool workspace exhausted"))?
        .copy_from_slice(bytes);
    *used = end;
    Ok(())
}

fn string_fragment(out: &mut [u8], used: &mut usize, text: &str) -> Result<(), EngineError> {
    const HEX: &[u8] = b"0123456789abcdef";
    for b in text.bytes() {
        match b {
            b'"' | b'\\' => append(out, used, &[b'\\', b])?,
            0..=31 => append(
                out,
                used,
                &[
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX[(b >> 4) as usize],
                    HEX[(b & 15) as usize],
                ],
            )?,
            _ => append(out, used, &[b])?,
        }
    }
    Ok(())
}

/// Gemma uses bare object keys. The quoted portions here have already been
/// JSON-escaped from real QUOTE token boundaries; text cannot synthesize one.
fn json_arguments(input: &[u8], out: &mut [u8]) -> Result<usize, EngineError> {
    let mut used = 0;
    let mut i = 0;
    while i < input.len() {
        let start = i;
        let b = input[i];
        i += 1;
        if b == b'"' {
            loop {
                match input.get(i) {
                    Some(b'\\') => i += 2,
                    Some(b'"') => {
                        i += 1;
                        break;
                    }
                    Some(_) => i += 1,
                    None => return Err(invalid("incomplete quoted value")),
                }
            }
        } else if !delimiter(b) {
            while i < input.len() && !delimiter(input[i]) {
                i += 1;
            }
            let word = &input[start..i];
            let following = input[i..].iter().find(|b| !b.is_ascii_whitespace());
            if following == Some(&b':') {
                append(out, &mut used, b"\"")?;
                append(out, &mut used, word)?;
                append(out, &mut used, b"\"")?;
                continue;
            }
            if word == b"None" {
                append(out, &mut used, b"null")?;
                continue;
            }
        }
        append(out, &mut used, &input[start..i])?;
    }
    Ok(used)
}

fn delimiter(b: u8) -> bool {
    b.is_ascii_whitespace() || matches!(b, b'{' | b'}' | b'[' | b']' | b',' | b':' | b'"')
}

pub(crate) struct Gemma4ToolParser {
    storage: ParserAllocation,
    phase: Phase,
    used: usize,
    remaining: usize,
}

impl Gemma4ToolParser {
    pub(crate) fn new(storage: ParserAllocation, output_bytes: usize) -> Self {
        Self {
            storage,
            phase: Phase::Text,
            used: 0,
            remaining: output_bytes,
        }
    }

    pub(crate) fn push(&mut self, fragment: Fragment<'_>) -> Result<bool, EngineError> {
        use Fragment::{Special, Text};
        let phase = std::mem::replace(&mut self.phase, Phase::Failed);
        let size = match fragment {
            Text(t) => t.len(),
            Special(CALL) => "<|tool_call>".len(),
            Special(CLOSE) => "<tool_call|>".len(),
            Special(QUOTE) => "<|\"|>".len(),
            Special(RESPONSE | 1 | 106) => 0,
            Special(_) => return Err(invalid("unexpected assistant control token")),
        };
        self.remaining = self
            .remaining
            .checked_sub(size)
            .ok_or_else(|| invalid("output budget exhausted"))?;
        let visible = matches!((phase, &fragment), (Phase::Text, Text(_)));
        self.phase = match (phase, fragment) {
            (Phase::Text, Text(_)) => Phase::Text,
            (Phase::Text, Special(1 | 106)) | (Phase::AfterCall, Special(RESPONSE | 1 | 106)) => {
                Phase::End
            }
            (Phase::Text | Phase::AfterCall, Special(CALL)) => {
                self.used = 0;
                Phase::Call { quoted: false }
            }
            (Phase::Call { quoted }, Special(QUOTE)) => {
                append(self.storage.workspaces().call, &mut self.used, b"\"")?;
                Phase::Call { quoted: !quoted }
            }
            (Phase::Call { quoted }, Text(text)) => {
                let out = self.storage.workspaces().call;
                if quoted {
                    string_fragment(out, &mut self.used, text)?;
                } else {
                    if text.contains(['"', '\\']) {
                        return Err(invalid("expected a Gemma string delimiter"));
                    }
                    append(out, &mut self.used, text.as_bytes())?;
                }
                phase
            }
            (Phase::Call { quoted: false }, Special(CLOSE)) => {
                self.commit()?;
                Phase::AfterCall
            }
            (phase, token) => {
                let input = match token {
                    Special(id) => format!("control ID {id}"),
                    Text(s) if s.trim().is_empty() => format!("{} whitespace bytes", s.len()),
                    Text(s) => format!("{} text bytes", s.len()),
                };
                return Err(EngineError::ExecutionFailed {
                    stage: "Gemma 4 tool parsing",
                    message: format!("unexpected {input} in {phase:?}"),
                });
            }
        };
        Ok(visible)
    }

    fn commit(&mut self) -> Result<(), EngineError> {
        let w = self.storage.workspaces();
        let prefix = b"call:";
        let raw = w.call[..self.used]
            .strip_prefix(prefix)
            .ok_or_else(|| invalid("missing call prefix"))?;
        let name_len = raw
            .iter()
            .position(|b| *b == b'{')
            .ok_or_else(|| invalid("missing argument object"))?;
        let name = &raw[..name_len];
        if name.is_empty()
            || !name
                .iter()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(b))
        {
            return Err(invalid("invalid tool name"));
        }
        let args = json_arguments(&raw[name_len..], w.decoded)?;
        let start = prefix.len() + name_len;
        let end = start
            .checked_add(args)
            .ok_or_else(|| invalid("argument size overflow"))?;
        w.call
            .get_mut(start..end)
            .ok_or_else(|| invalid("expanded arguments exceed call capacity"))?
            .copy_from_slice(&w.decoded[..args]);
        validate_json_parameter(&w.call[start..end], w.decoded, w.stack, w.keys, w.nodes, 1)?;
        w.decoded
            .get_mut(..name_len)
            .ok_or_else(|| invalid("name exceeds decoded capacity"))?
            .copy_from_slice(&w.call[prefix.len()..start]);
        self.storage
            .commit_validated_call(name_len, 0..name_len, end, start..end)
            .map_err(|e| EngineError::invalid_request("Gemma 4 call", e.to_string()))?;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<FrozenParserAllocation, EngineError> {
        match self.phase {
            Phase::Text | Phase::End => Ok(self.storage.freeze()),
            _ => Err(invalid("unterminated tool call or missing response marker")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Fragment::{Special, Text},
        *,
    };
    use crate::{admission::AdmissionController, parser_allocation::ParserAllocationLimits};

    fn parser(max: usize) -> Gemma4ToolParser {
        let limits = ParserAllocationLimits {
            carry_bytes: 64,
            call_bytes: 256,
            decoded_bytes: 256,
            depth: 8,
            keys: 16,
            nodes: 32,
            proposal_bytes: 512,
            calls: 2,
        };
        let size = ParserAllocation::required_bytes(limits).unwrap();
        let controller = AdmissionController::new(size).unwrap();
        let mut grant = controller.reserve_grant("Gemma 4 test", &[size]).unwrap();
        Gemma4ToolParser::new(
            ParserAllocation::from_grant(&mut grant, limits).unwrap(),
            max,
        )
    }

    #[test]
    fn gemma4_tool_fragments_and_typed_arguments() {
        let literal = "雪 {x},\\\" \u{0} \n <tool_call|>";
        for split in (0..=literal.len()).filter(|&i| literal.is_char_boundary(i)) {
            let mut p = parser(1024);
            assert!(p.push(Text("<|tool_call>")).unwrap());
            for f in [
                Special(CALL),
                Text("call:fs."),
                Text("write{path:"),
                Special(QUOTE),
                Text(&literal[..split]),
                Text(&literal[split..]),
                Special(QUOTE),
                Text(",value:{a:[true,2,-1.5,None]}}"),
                Special(CLOSE),
                Special(CALL),
                Text("call:fs.read{}"),
                Special(CLOSE),
                Special(RESPONSE),
            ] {
                assert!(!p.push(f).unwrap());
            }
            let batch = p.finish().unwrap();
            assert_eq!(batch.proposal_count(), 2);
            let first = batch.proposal(0).unwrap();
            assert_eq!(first.name, "fs.write");
            let args: serde_json::Value = serde_json::from_str(first.arguments_json).unwrap();
            assert_eq!(
                args,
                serde_json::json!({"path":literal,"value":{"a":[true,2,-1.5,null]}})
            );
            assert_eq!(batch.proposal(1).unwrap().name, "fs.read");
        }
    }

    #[test]
    fn gemma4_tool_rejects_invalid_batches() {
        // A fully closed call may be followed by any publisher-declared EOS.
        // An output length cutoff, or EOS inside the call, must still discard it.
        for eos in [1, 106, RESPONSE] {
            let mut p = parser(512);
            for f in [
                Special(CALL),
                Text("call:example{}"),
                Special(CLOSE),
                Special(eos),
            ] {
                p.push(f).unwrap();
            }
            assert_eq!(p.finish().unwrap().proposal_count(), 1);
            let mut p = parser(512);
            p.push(Special(CALL)).unwrap();
            p.push(Text("call:example{}")).unwrap();
            assert!(p.push(Special(eos)).is_err());
            assert!(p.finish().is_err());
        }
        for bad in [
            "{}",
            "call:{}",
            "call:fs.read[]",
            "call:fs.read{a:1,a:2}",
            "call:a{a:foo}",
            "call:a{a:1}extra",
            "call:a{a:[[]}",
            "call:a{x:\"fake\"}",
        ] {
            let mut p = parser(512);
            for f in [
                Special(CALL),
                Text("call:ok{}"),
                Special(CLOSE),
                Special(CALL),
            ] {
                p.push(f).unwrap();
            }
            assert!(
                p.push(Text(bad))
                    .and_then(|_| p.push(Special(CLOSE)))
                    .is_err(),
                "{bad}"
            );
            assert!(p.finish().is_err());
        }
        for incomplete in [
            vec![Special(CALL)],
            vec![Special(CALL), Text("call:ok{}"), Special(CLOSE)],
            vec![
                Special(CALL),
                Text("call:ok{v:"),
                Special(QUOTE),
                Text("value"),
            ],
        ] {
            let mut p = parser(512);
            for f in incomplete {
                p.push(f).unwrap();
            }
            assert!(p.finish().is_err());
        }
        for id in [QUOTE, CLOSE, RESPONSE, 98, 105] {
            let mut p = parser(512);
            assert!(p.push(Special(id)).is_err());
            assert!(p.finish().is_err());
        }
        let mut p = parser(2);
        assert!(p.push(Special(CALL)).is_err());
        let mut p = parser(2048);
        p.push(Special(CALL)).unwrap();
        assert!(p.push(Text(&"x".repeat(257))).is_err());
        assert!(p.finish().is_err());
    }
}
