use super::TextDecoder;
use crate::{error::EngineError, types::GenerationOutputBlock};

struct TextOnlyDecoder {
    pushes: usize,
    finishes: usize,
}

impl TextDecoder for TextOnlyDecoder {
    fn push(&mut self, token: u32) -> Result<String, EngineError> {
        self.pushes += 1;
        match token {
            0 => Ok(String::new()),
            1 => Ok("résumé 雪".to_owned()),
            _ => Err(EngineError::invalid_request(
                "token",
                "unsupported test token",
            )),
        }
    }

    fn finish(&mut self) -> Result<String, EngineError> {
        self.finishes += 1;
        Ok(" tail".to_owned())
    }
}

#[test]
fn text_decoder_defaults_preserve_text_and_skip_empty_deltas() {
    let mut decoder = TextOnlyDecoder {
        pushes: 0,
        finishes: 0,
    };
    assert!(decoder.push_blocks(0).expect("empty delta").is_empty());
    assert_eq!(
        decoder.push_blocks(1).expect("text delta"),
        vec![GenerationOutputBlock::Text("résumé 雪".to_owned())],
    );
    assert_eq!(decoder.pushes, 2);
    assert_eq!(decoder.finishes, 0);
}

#[test]
fn text_decoder_defaults_preserve_finish_output() {
    let mut decoder = TextOnlyDecoder {
        pushes: 0,
        finishes: 0,
    };
    assert_eq!(
        decoder.finish_blocks().expect("finish"),
        vec![GenerationOutputBlock::Text(" tail".to_owned())],
    );
    assert_eq!(decoder.pushes, 0);
    assert_eq!(decoder.finishes, 1);
}

#[test]
fn structured_decoder_methods_work_through_trait_objects() {
    let mut decoder: Box<dyn TextDecoder> = Box::new(TextOnlyDecoder {
        pushes: 0,
        finishes: 0,
    });
    assert_eq!(
        decoder.push_blocks(1).expect("trait-object push"),
        vec![GenerationOutputBlock::Text("résumé 雪".to_owned())],
    );
    assert!(decoder.push_blocks(99).is_err());
}
