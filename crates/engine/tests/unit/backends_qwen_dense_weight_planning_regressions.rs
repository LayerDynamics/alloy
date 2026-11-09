use super::*;

#[test]
fn dense_fixture_payload_and_host_storage_fit_the_dense_bounds() {
    for tied in [true, false] {
        let mut config = Qwen3Config::tiny_fixture();
        config.vocab_size = 258;
        config.tie_embeddings = tied;

        let weights = qwen3::fixture::generate_fixture(&config);
        let measured = weight_memory(&weights).expect("fixture accounting");
        let headers = product(
            as_u64(weights.layers.len()).expect("layer count"),
            as_u64(std::mem::size_of::<LayerWeights>()).expect("layer size"),
        )
        .expect("headers");
        let payload = measured.used_bytes() - headers;

        assert_eq!(
            dense_weight_payload_bound(&config).expect("dense payload"),
            payload,
        );
        assert!(
            dense_decoded_weight_bound(&config).expect("dense host bound")
                >= measured.allocated_bytes(),
        );
    }
}
