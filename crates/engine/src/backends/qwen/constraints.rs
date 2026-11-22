//! Raw catalogue uses the same verified decoder as native text emission.
use super::{ArtifactData, execution_error};
use crate::{
    error::EngineError,
    family::ExecutionControl,
    structured_output::{JsonConstraint, JsonOutputSchema},
};

pub(super) fn build(
    data: &ArtifactData,
    schema: JsonOutputSchema,
    control: &ExecutionControl,
) -> Result<JsonConstraint, EngineError> {
    JsonConstraint::new(schema, data.config.vocabulary(), &data.eos, control, |id| {
        if data.special_tokens.contains(&id) {
            return Ok(None);
        }
        let Some(spelling) = data.tokenizer.tokenizer().id_to_token(id) else {
            return Ok(None);
        };
        // Bound transformed scratch before allocating it; ByteLevel uses one
        // byte per character, Gemma replacement never increases byte length.
        if spelling.len() > 4096 {
            return Err(execution_error(
                "structured output",
                "token spelling exceeds constraint bound",
            ));
        }
        let bytes = if data.gemma_decoder {
            match spelling
                .strip_prefix("<0x")
                .and_then(|s| s.strip_suffix('>'))
                .filter(|s| s.len() == 2 && s.is_ascii())
                .and_then(|s| u8::from_str_radix(s, 16).ok())
            {
                Some(byte) => vec![byte],
                None => spelling.replace('▁', " ").into_bytes(),
            }
        } else {
            spelling
                .chars()
                .map(|ch| {
                    data.byte_decoder.get(&ch).copied().ok_or_else(|| {
                        execution_error(
                            "structured output",
                            "token is outside verified ByteLevel alphabet",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Some(bytes))
    })
}
