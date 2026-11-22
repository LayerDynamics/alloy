//! Optional raw output for the M3 numerical report. Each file belongs to one run.
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub fn record(
    profile: &Value,
    case: usize,
    step: usize,
    prompt: &[u32],
    executed_suffix: &[u32],
    logits: &[f32],
) {
    let Some(dir) = std::env::var_os("ALLOY_M3_NUMERICAL_OUTPUT") else {
        return;
    };
    let dir = PathBuf::from(dir);
    assert!(
        dir.is_absolute() && dir.is_dir(),
        "supply a dedicated absolute output directory"
    );
    let path = dir.join(format!("case-{case}-step-{step}.json"));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut out = BufWriter::new(file);
    // Widen before serialization: shortest-roundtrip f32 decimals can change
    // near-zero relative error when read back into the f64 report evaluator.
    let row = json!({
        "profile": profile["id"],
        "reference_sha256": profile["reference"]["sha256"],
        "case": case, "step": step,
        "tokens": [prompt, executed_suffix].concat(),
        "logits": logits.iter().copied().map(f64::from).collect::<Vec<_>>()
    });
    serde_json::to_writer(&mut out, &row).unwrap();
    out.flush().unwrap();
    out.get_ref().sync_all().unwrap();
}
