use qwen3::{
    config::Qwen3Config,
    fixture::generate_fixture,
    model::{ModelError, Qwen3Model},
};

const TOKENS: [u32; 8] = [3, 41, 7, 200, 128, 95, 12, 250];

const ABS_TOLERANCE: f32 = 2e-2;
const REL_TOLERANCE: f32 = 1e-2;

fn tiny_model() -> Qwen3Model {
    let config = Qwen3Config::tiny_fixture();
    let weights = generate_fixture(&config);

    Qwen3Model::load(config, weights).expect("valid fixture must load")
}

#[test]
fn tiny_fixture_loads() {
    let _model = tiny_model();
}

#[test]
fn loader_rejects_invalid_embedding_shape() {
    let config = Qwen3Config::tiny_fixture();
    let expected = config.vocab_size * config.hidden_size;
    let mut weights = generate_fixture(&config);

    weights.embed_tokens.as_f32_mut().expect("f32").pop();

    let result = Qwen3Model::load(config, weights);

    assert!(matches!(
        result,
        Err(ModelError::ShapeMismatch {
            tensor: "embed_tokens",
            expected: actual_expected,
            got,
        }) if actual_expected == expected && got == expected - 1
    ));
}

#[test]
fn loader_rejects_missing_layer() {
    let config = Qwen3Config::tiny_fixture();
    let expected = config.num_layers;
    let mut weights = generate_fixture(&config);

    weights.layers.pop();

    let result = Qwen3Model::load(config, weights);

    assert!(matches!(
        result,
        Err(ModelError::ShapeMismatch {
            tensor: "layers",
            expected: actual_expected,
            got,
        }) if actual_expected == expected && got == expected - 1
    ));
}

#[test]
fn loader_rejects_invalid_q_norm_shape() {
    let config = Qwen3Config::tiny_fixture();
    let expected = config.head_dim;
    let mut weights = generate_fixture(&config);

    weights.layers[0].q_norm.pop();

    let result = Qwen3Model::load(config, weights);

    assert!(matches!(
        result,
        Err(ModelError::ShapeMismatch {
            tensor: "q_norm",
            expected: actual_expected,
            got,
        }) if actual_expected == expected && got == expected - 1
    ));
}

#[test]
fn loader_rejects_invalid_k_norm_shape() {
    let config = Qwen3Config::tiny_fixture();
    let expected = config.head_dim;
    let mut weights = generate_fixture(&config);

    weights.layers[0].k_norm.pop();

    let result = Qwen3Model::load(config, weights);

    assert!(matches!(
        result,
        Err(ModelError::ShapeMismatch {
            tensor: "k_norm",
            expected: actual_expected,
            got,
        }) if actual_expected == expected && got == expected - 1
    ));
}

#[test]
fn qwen3_forward_cpu() {
    let config = Qwen3Config::tiny_fixture();
    let reference = include_str!("../fixture/reference_logits.txt");

    assert!(
        !reference.contains("PLACEHOLDER"),
        "reference_logits.txt must contain independently generated logits"
    );

    let expected: Vec<Vec<f32>> = reference
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(position, line)| {
            line.split_whitespace()
                .enumerate()
                .map(|(token, value)| {
                    value.parse::<f32>().unwrap_or_else(|error| {
                        panic!(
                            "invalid reference logit at position {position}, \
                             token {token}: {value:?}: {error}"
                        )
                    })
                })
                .collect()
        })
        .collect();

    assert_eq!(
        expected.len(),
        TOKENS.len(),
        "reference must contain one row per input token"
    );

    for (position, row) in expected.iter().enumerate() {
        assert_eq!(
            row.len(),
            config.vocab_size,
            "reference row {position} has the wrong vocabulary size"
        );

        for (token, value) in row.iter().enumerate() {
            assert!(
                value.is_finite(),
                "non-finite reference logit at position {position}, token {token}"
            );
        }
    }

    let actual = tiny_model()
        .forward(&TOKENS)
        .expect("forward pass must succeed");

    assert_eq!(
        actual.len(),
        expected.len(),
        "forward must return one row per input token"
    );

    for (position, (actual_row, expected_row)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual_row.len(),
            config.vocab_size,
            "output row {position} has the wrong vocabulary size"
        );

        for (token, (&got, &want)) in actual_row.iter().zip(expected_row).enumerate() {
            assert!(
                got.is_finite(),
                "non-finite output logit at position {position}, token {token}"
            );

            let difference = (got - want).abs();
            let tolerance = ABS_TOLERANCE + REL_TOLERANCE * want.abs();

            assert!(
                difference <= tolerance,
                "position {position}, token {token}: \
                 got {got}, expected {want}, \
                 absolute difference {difference}, tolerance {tolerance}"
            );
        }
    }
}
