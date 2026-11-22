//! Ministral 3 separates tool names and argument objects with special token IDs.
use super::parser::validate_json_parameter;
use crate::{
    error::EngineError,
    parser_allocation::{FrozenParserAllocation, ParserAllocation},
};

pub(crate) const CALL: u32 = 9;
pub(crate) const ARGS: u32 = 32;
pub(crate) const EOS: u32 = 2;

#[derive(Clone, Copy)]
enum Phase {
    Text,
    Name,
    Arguments { name_len: usize },
    End,
    Failed,
}

pub(crate) enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

fn invalid(message: impl Into<String>) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "Ministral tool parsing",
        message: message.into(),
    }
}

pub(crate) struct MinistralToolParser {
    storage: ParserAllocation,
    phase: Phase,
    used: usize,
    remaining: usize,
}

impl MinistralToolParser {
    pub(crate) fn new(storage: ParserAllocation, max_output_bytes: usize) -> Self {
        Self {
            storage,
            phase: Phase::Text,
            used: 0,
            remaining: max_output_bytes,
        }
    }

    /// Return true only when the input is visible assistant text. Enter Failed
    /// before fallible work so a rejected suffix invalidates the entire batch.
    pub(crate) fn push(&mut self, fragment: Fragment<'_>) -> Result<bool, EngineError> {
        use Fragment::{Special, Text};
        let phase = std::mem::replace(&mut self.phase, Phase::Failed);
        let size = match fragment {
            Text(text) => text.len(),
            Special(CALL) => "[TOOL_CALLS]".len(),
            Special(ARGS) => "[ARGS]".len(),
            Special(EOS) => 0,
            Special(_) => return Err(invalid("unexpected special token in assistant output")),
        };
        self.remaining = self
            .remaining
            .checked_sub(size)
            .ok_or_else(|| invalid("output byte budget exceeded"))?;
        let visible = matches!((phase, &fragment), (Phase::Text, Text(_)));
        self.phase = match (phase, fragment) {
            (Phase::Text, Text(_)) => Phase::Text,
            (Phase::Text, Special(CALL)) => Phase::Name,
            (Phase::Text, Special(EOS)) => Phase::End,
            (Phase::Name | Phase::Arguments { .. }, Text(text)) => {
                let end = self
                    .used
                    .checked_add(text.len())
                    .ok_or_else(|| invalid("call length overflow"))?;
                self.storage
                    .workspaces()
                    .call
                    .get_mut(self.used..end)
                    .ok_or_else(|| invalid("call byte capacity exceeded"))?
                    .copy_from_slice(text.as_bytes());
                self.used = end;
                phase
            }
            (Phase::Name, Special(ARGS)) if self.used > 0 => Phase::Arguments {
                name_len: self.used,
            },
            (Phase::Arguments { name_len }, Special(token @ (CALL | EOS))) => {
                self.commit(name_len)?;
                self.used = 0;
                if token == EOS {
                    Phase::End
                } else {
                    Phase::Name
                }
            }
            _ => return Err(invalid("unexpected or incomplete tool-call sequence")),
        };
        Ok(visible)
    }

    fn commit(&mut self, name_len: usize) -> Result<(), EngineError> {
        let w = self.storage.workspaces();
        let (name, args) = w.call[..self.used].split_at(name_len);
        // JsonCheck uses kind 1 for objects. Preserve the original JSON bytes
        // for registry schema validation and the policy decision.
        validate_json_parameter(args, w.decoded, w.stack, w.keys, w.nodes, 1)?;
        w.decoded
            .get_mut(..name_len)
            .ok_or_else(|| invalid("name capacity exceeded"))?
            .copy_from_slice(name);
        self.storage
            .commit_validated_call(name_len, 0..name_len, self.used, name_len..self.used)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<FrozenParserAllocation, EngineError> {
        match self.phase {
            Phase::Text | Phase::End => Ok(self.storage.freeze()),
            _ => Err(invalid("failed or unterminated tool-call batch")),
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

    fn parser(max_output: usize) -> MinistralToolParser {
        let limits = ParserAllocationLimits {
            carry_bytes: 1,
            call_bytes: 128,
            decoded_bytes: 128,
            depth: 6,
            keys: 8,
            nodes: 16,
            proposal_bytes: 192,
            calls: 2,
        };
        let bytes = ParserAllocation::required_bytes(limits).unwrap();
        let controller = AdmissionController::new(bytes).unwrap();
        let mut grant = controller
            .reserve_grant("Ministral test", &[bytes])
            .unwrap();
        MinistralToolParser::new(
            ParserAllocation::from_grant(&mut grant, limits).unwrap(),
            max_output,
        )
    }

    #[test]
    fn ministral_protocol_preserves_text_and_sequential_calls_at_every_split() {
        let args = r#"{"literal":"[ARGS]雪[TOOL_CALLS]","n":[2,true]}"#;
        for split in (0..=args.len()).filter(|&i| args.is_char_boundary(i)) {
            let mut p = parser(512);
            assert!(p.push(Text("[TOOL_CALLS]fs.read[ARGS]{}")).unwrap());
            for fragment in [
                Special(CALL),
                Text("fs."),
                Text("write"),
                Special(ARGS),
                Text(&args[..split]),
                Text(&args[split..]),
                Special(CALL),
                Text("fs.read"),
                Special(ARGS),
                Text("{}"),
                Special(EOS),
            ] {
                assert!(!p.push(fragment).unwrap());
            }
            let batch = p.finish().unwrap();
            assert_eq!(batch.proposal_count(), 2);
            let first = batch.proposal(0).unwrap();
            assert_eq!(first.name, "fs.write");
            assert_eq!(first.arguments_json, args);
            assert_eq!(batch.proposal(1).unwrap().name, "fs.read");
        }
    }

    #[test]
    fn ministral_protocol_discards_malformed_or_truncated_batches() {
        for bad in [
            "",
            "[]",
            "null",
            "{}{}",
            r#"{"a":0,"a":1}"#,
            r#"{"a":1e999}"#,
        ] {
            let mut p = parser(512);
            for f in [
                Special(CALL),
                Text("first"),
                Special(ARGS),
                Text("{}"),
                Special(CALL),
                Text("second"),
                Special(ARGS),
                Text(bad),
            ] {
                p.push(f).unwrap();
            }
            assert!(p.push(Special(EOS)).is_err());
            assert!(p.finish().is_err());
        }
        let mut p = parser(512);
        for f in [Special(CALL), Text("fs.read"), Special(ARGS), Text("{}")] {
            p.push(f).unwrap();
        }
        assert!(p.finish().is_err());
        for token in [ARGS, 7, 999] {
            let mut p = parser(512);
            assert!(p.push(Special(token)).is_err());
            assert!(p.finish().is_err());
        }
    }

    #[test]
    fn ministral_protocol_enforces_capacity_before_publication() {
        let mut p = parser(11);
        assert!(p.push(Special(CALL)).is_err());
        assert!(p.finish().is_err());
        let mut p = parser(512);
        p.push(Special(CALL)).unwrap();
        assert!(p.push(Text(&"x".repeat(129))).is_err());
        assert!(p.finish().is_err());
        let mut p = parser(512);
        for _ in 0..3 {
            for f in [Special(CALL), Text("read"), Special(ARGS), Text("{}")] {
                p.push(f).unwrap();
            }
        }
        assert!(p.push(Special(EOS)).is_err());
        assert!(p.finish().is_err());
    }
}
