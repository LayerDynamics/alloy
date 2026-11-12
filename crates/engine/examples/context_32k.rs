//! Reproducible render/count measurement; see Phase 3 plan for boundary.
use engine::{
    rendering::{
        PinnedTokenizer, RenderingBinding, RenderingLimits, RenderingOptions,
        qwen3_projected_token_upper_bound, render_and_count,
    },
    types::{ArtifactId, ContextSnapshot, ContextSnapshotId, Message, MessageRole, ModelProfileId},
};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, time::Instant};

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let tokenizer_path = std::env::args()
        .nth(1)
        .ok_or("usage: context_32k TOKENIZER_JSON")?;
    let source = fs::read(tokenizer_path)?;
    let tokenizer_hash = digest(&source);
    assert_eq!(
        hex(&tokenizer_hash),
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer pin"
    );
    let tokenizer = PinnedTokenizer::from_pinned_json(&source, tokenizer_hash, source.len())?;
    for text in ["", "ASCII and spaces", "Ǖ\u{344}é中🦀", "<|im_start|>\n"] {
        let tokens = tokenizer
            .tokenizer()
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .len();
        let bound = qwen3_projected_token_upper_bound(&tokenizer, text.len())?;
        assert!(tokens <= bound, "projected bound failed for {text:?}");
    }
    let document = fs::read("bench/corpus/fixtures/documents/document-32k.txt")?;
    assert_eq!(
        hex(&digest(&document)),
        "393bc22d01056d7e6f5fbba865071725867679aa00521f84780802c660cb0c75",
        "corpus pin"
    );
    let profile = ModelProfileId::new("qwen3-4b-4bit")?;
    let artifact = ArtifactId::new("qwen3-4b-4bit")?;
    let context = ContextSnapshot::new(
        ContextSnapshotId::new("32k-measurement")?,
        profile.clone(),
        vec![Message {
            role: MessageRole::User,
            content: String::from_utf8(document)?,
        }],
    )?;
    let template = qwen3::template::CHAT_TEMPLATE;
    let binding = RenderingBinding {
        profile_id: &profile,
        artifact_id: &artifact,
        template,
        template_sha256: digest(template.as_bytes()),
        artifact_manifest_sha256: [0; 32],
    };
    let options = RenderingOptions {
        add_generation_prompt: true,
        enable_thinking: Some(false),
        add_special_tokens: false,
        keep_trailing_newline: false,
        trim_blocks: false,
        lstrip_blocks: false,
        reserved_spellings: &[
            "<|",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ],
    };
    let limits = RenderingLimits::new(8, 1_000_000, 100_000, 1_000_000, 65_536)?;
    let run = || render_and_count(&context, &binding, &tokenizer, &options, limits);
    let warm = run()?;
    assert!(
        warm.token_count() >= 32768 && warm.token_count() < 33000,
        "unexpected token count: {}",
        warm.token_count()
    );
    println!(
        "tokenizer_sha256={} corpus_sha256={} template_sha256={} rendered_tokens={} render_hash={}",
        hex(&tokenizer_hash),
        hex(&digest(context.messages()[0].content.as_bytes())),
        hex(&binding.template_sha256),
        warm.token_count(),
        hex(warm.render_hash())
    );
    let mut max = 0u128;
    for iteration in 0..10 {
        let start = Instant::now();
        let rendered = run()?;
        let micros = start.elapsed().as_micros();
        assert_eq!(rendered.token_ids(), warm.token_ids());
        max = max.max(micros);
        println!("iteration={iteration} elapsed_us={micros}");
    }
    println!(
        "max_elapsed_us={max} boundary=render_and_count_including_hash_and_tokenization_excluding_artifact_load"
    );
    assert!(max < 150_000, "32k render/count exceeds 150 ms");
    Ok(())
}
