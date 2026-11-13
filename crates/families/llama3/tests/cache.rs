use inference_infra::decoder::{
    config::{DecoderConfig, DecoderFamily},
    fixture::generate_fixture,
    mlx_model::{MlxDecoderError, MlxDecoderModel},
};
use mlx_ffi::{Device, DeviceType, Stream};

#[test]
fn llama_evaluated_layer_cancellation_preserves_committed_prefix() {
    let config = DecoderConfig {
        family: DecoderFamily::Llama3,
        rms_eps: 1e-5,
        ..DecoderConfig::tiny_fixture()
    };
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    let model =
        MlxDecoderModel::from_weights(config.clone(), generate_fixture(&config), 16 << 20, &stream)
            .unwrap();
    let mut cache = model.new_cache(16, 16 << 20, &stream).unwrap();
    model
        .prefill(&[3, 41], &mut cache, 16 << 20, &stream)
        .unwrap();
    let mut checks = 0;
    let cancelled = model.prefill_cancellable(&[7, 12], &mut cache, 16 << 20, &stream, || {
        checks += 1;
        checks == 3
    });
    assert!(matches!(cancelled, Err(MlxDecoderError::Cancelled)));
    assert_eq!(cache.len(), 2);
    let resumed = model
        .prefill(&[7, 12], &mut cache, 16 << 20, &stream)
        .unwrap();
    let mut cold = model.new_cache(16, 16 << 20, &stream).unwrap();
    model
        .prefill(&[3, 41], &mut cold, 16 << 20, &stream)
        .unwrap();
    let expected = model
        .prefill(&[7, 12], &mut cold, 16 << 20, &stream)
        .unwrap();
    assert_eq!(resumed.last_row(), expected.last_row());
    assert_eq!(cache.len(), 4);
}
