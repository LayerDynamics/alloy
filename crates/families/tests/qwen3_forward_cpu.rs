use qwen3::config::Qwen3Config;
use qwen3::fixture::generate_fixture;
use qwen3::model::Qwen3Model;

/// FR-8 tolerances from the execution plan.
const ABS_TOL: f32 = 2e-2;
const REL_TOL: f32 = 1e-2;

/// Fixed token sequence; must match TOKENS in gen_reference.py.
const TOKENS: [u32; 8] = [3, 41, 7, 200, 128, 95, 12, 250];

const REFERENCE: &str = include_str!("../fixture/reference_logits.txt");

fn load_reference() -> Vec<Vec<f32>> {
    let first = REFERENCE.lines().next().expect("reference file not empty");
    if first.starts_with("PLACEHOLDER") {
        panic!(
            "reference logits not generated yet: run \
             `python3 crates/families/qwen3/fixture/gen_reference.py` and commit the result"
        );
    }
    REFERENCE
        .lines()
        .map(|l| l.split_whitespace().map(|v| v.parse::<f32>().expect("f32 in reference")).collect())
        .collect()
}

#[test]
fn qwen3_forward_cpu() {
    let config = Qwen3Config::tiny_fixture();
    let weights = generate_fixture(&config);
    let model = Qwen3Model::load(config, weights).expect("fixture weights match config");

    let logits = model.forward(&TOKENS).expect("forward");
    let reference = load_reference();

    assert_eq!(logits.len(), TOKENS.len(), "one logit row per token");
    for (pos, (got, want)) in logits.iter().zip(reference.iter()).enumerate() {
        assert_eq!(got.len(), want.len(), "vocab size mismatch at position {pos}");
        for (g, w) in got.iter().zip(want.iter()) {
            let diff = (g - w).abs();
            assert!(
                diff <= ABS_TOL || diff <= REL_TOL * w.abs(),
                "position {pos}: got {g}, reference {w}"
            );
        }
    }
}