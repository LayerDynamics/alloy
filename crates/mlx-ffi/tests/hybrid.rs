//! Native recurrent primitives against independent scalar equations, not model qualification.
use mlx_ffi::{Array, Device, DeviceType, Stream, ops};

fn streams() -> Vec<Stream> {
    let mut kinds = vec![DeviceType::Cpu];
    if cfg!(target_os = "macos") {
        kinds.push(DeviceType::Gpu);
    }
    kinds
        .into_iter()
        .map(|k| Stream::new(&Device::new(k, 0).unwrap()).unwrap())
        .collect()
}
fn array(values: &[f32], shape: &[i32]) -> Array {
    Array::from_f32_slice(values, shape).unwrap()
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (f64::from(a) - b).abs() < 0.00002,
            "index {i}: {a} vs {b}"
        );
    }
}
#[test]
fn gated_delta_matches_scalar_and_chunked_state() {
    const B: usize = 2;
    const T: usize = 5;
    const HK: usize = 2;
    const HV: usize = 4;
    const D: usize = 128;
    let values = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 17 + seed) % 37) as f32 - 18.0) / 100.0)
            .collect()
    };
    let q = values(B * T * HK * D, 1);
    let k = values(B * T * HK * D, 7);
    let v = values(B * T * HV * D, 13);
    let g: Vec<_> = (0..B * T * HV)
        .map(|i| 0.7 + (i % 4) as f32 * 0.05)
        .collect();
    let beta: Vec<_> = (0..B * T * HV)
        .map(|i| 0.2 + (i % 3) as f32 * 0.1)
        .collect();
    let initial = values(B * HV * D * D, 9);
    let mut state: Vec<f64> = initial.iter().map(|&x| f64::from(x)).collect();
    let mut expected = vec![0f64; B * T * HV * D];
    for b in 0..B {
        for t in 0..T {
            for h in 0..HV {
                for dv in 0..D {
                    let base = ((b * HV + h) * D + dv) * D;
                    let key_base = ((b * T + t) * HK + h / (HV / HK)) * D;
                    let gate = (b * T + t) * HV + h;
                    let value = ((b * T + t) * HV + h) * D + dv;
                    let mut memory = 0f64;
                    for dk in 0..D {
                        state[base + dk] *= f64::from(g[gate]);
                        memory += state[base + dk] * f64::from(k[key_base + dk]);
                    }
                    let delta = (f64::from(v[value]) - memory) * f64::from(beta[gate]);
                    for dk in 0..D {
                        state[base + dk] += f64::from(k[key_base + dk]) * delta;
                        expected[value] += state[base + dk] * f64::from(q[key_base + dk]);
                    }
                }
            }
        }
    }
    for stream in streams() {
        let q = array(&q, &[B as i32, T as i32, HK as i32, D as i32]);
        let k = array(&k, &[B as i32, T as i32, HK as i32, D as i32]);
        let v = array(&v, &[B as i32, T as i32, HV as i32, D as i32]);
        let g = array(&g, &[B as i32, T as i32, HV as i32]);
        let beta = array(&beta, &[B as i32, T as i32, HV as i32]);
        let initial = array(&initial, &[B as i32, HV as i32, D as i32, D as i32]);
        let (output, last) =
            ops::gated_delta_update(&q, &k, &v, &g, &beta, Some(&initial), &stream).unwrap();
        close(&ops::to_vec_f32(&output, &stream).unwrap(), &expected);
        close(&ops::to_vec_f32(&last, &stream).unwrap(), &state);
        let slice = |a: &Array, start, stop| {
            let mut lo = vec![0; a.shape().len()];
            let mut hi = a.shape();
            lo[1] = start;
            hi[1] = stop;
            ops::slice(a, &lo, &hi, &stream).unwrap()
        };
        for split in 1..T {
            let (_, prefix) = ops::gated_delta_update(
                &slice(&q, 0, split),
                &slice(&k, 0, split),
                &slice(&v, 0, split),
                &slice(&g, 0, split),
                &slice(&beta, 0, split),
                Some(&initial),
                &stream,
            )
            .unwrap();
            let (suffix, end) = ops::gated_delta_update(
                &slice(&q, split, T),
                &slice(&k, split, T),
                &slice(&v, split, T),
                &slice(&g, split, T),
                &slice(&beta, split, T),
                Some(&prefix),
                &stream,
            )
            .unwrap();
            let expected_suffix: Vec<_> = (0..B)
                .flat_map(|b| {
                    expected[(b * T + split) * HV * D..(b + 1) * T * HV * D]
                        .iter()
                        .copied()
                })
                .collect();
            close(
                &ops::to_vec_f32(&suffix, &stream).unwrap(),
                &expected_suffix,
            );
            close(&ops::to_vec_f32(&end, &stream).unwrap(), &state);
        }
        let zero = array(
            &vec![0.; B * HV * D * D],
            &[B as i32, HV as i32, D as i32, D as i32],
        );
        let (_, none) = ops::gated_delta_update(&q, &k, &v, &g, &beta, None, &stream).unwrap();
        let (_, explicit) =
            ops::gated_delta_update(&q, &k, &v, &g, &beta, Some(&zero), &stream).unwrap();
        assert_eq!(
            ops::to_vec_f32(&none, &stream).unwrap(),
            ops::to_vec_f32(&explicit, &stream).unwrap()
        );
        // Functional update must not mutate the original recurrent state.
        close(
            &ops::to_vec_f32(&initial, &stream).unwrap(),
            &values(B * HV * D * D, 9)
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>(),
        );
    }
}
#[test]
fn causal_depthwise_convolution_and_stable_gates() {
    for stream in streams() {
        let x = array(&[1., 10., 2., 20., 3., 30., 4., 40.], &[1, 4, 2]);
        let w = array(&[1., 2., 3., -1., 0., 1.], &[2, 3, 1]);
        let out = ops::depthwise_conv1d(&x, &w, &stream).unwrap();
        close(
            &ops::to_vec_f32(&out, &stream).unwrap(),
            &[14., 20., 20., 20.],
        );
        let x = array(&[-1000., -10., 0., 10., 1000.], &[5]);
        let out = ops::softplus(&x, &stream).unwrap();
        close(
            &ops::to_vec_f32(&out, &stream).unwrap(),
            &[
                0.,
                (-10f64).exp().ln_1p(),
                2f64.ln(),
                10. + (-10f64).exp().ln_1p(),
                1000.,
            ],
        );
        let out = ops::sigmoid(&x, &stream).unwrap();
        close(
            &ops::to_vec_f32(&out, &stream).unwrap(),
            &[
                0.,
                1. / (1. + 10f64.exp()),
                0.5,
                1. / (1. + (-10f64).exp()),
                1.,
            ],
        );
        let out = ops::exp(&array(&[-1., 0., 1.], &[3]), &stream).unwrap();
        close(
            &ops::to_vec_f32(&out, &stream).unwrap(),
            &[(-1f64).exp(), 1., 1f64.exp()],
        );
    }
}
#[test]
fn hybrid_operations_reject_invalid_shapes() {
    let stream = Stream::new(&Device::new(DeviceType::Cpu, 0).unwrap()).unwrap();
    let q = array(&[0.; 8], &[1, 1, 2, 4]);
    let v = array(&[0.; 12], &[1, 1, 3, 4]);
    let gate = array(&[1.; 3], &[1, 1, 3]);
    assert!(ops::gated_delta_update(&q, &q, &v, &gate, &gate, None, &stream).is_err());
    let gate = array(&[1.; 2], &[1, 1, 2]);
    let bad_state = array(&[0.; 4], &[4]);
    assert!(ops::gated_delta_update(&q, &q, &q, &gate, &gate, Some(&bad_state), &stream).is_err());
    assert!(ops::gated_delta_update(&q, &q, &q, &bad_state, &gate, None, &stream).is_err());
    assert!(ops::gated_delta_update(&bad_state, &q, &q, &gate, &gate, None, &stream).is_err());
    assert!(ops::depthwise_conv1d(&q, &q, &stream).is_err());
    assert!(
        ops::depthwise_conv1d(
            &array(&[0.; 4], &[1, 2, 2]),
            &array(&[0.; 6], &[2, 3, 1]),
            &stream
        )
        .is_err()
    );
    assert!(
        ops::depthwise_conv1d(
            &array(&[0.; 8], &[1, 4, 2]),
            &array(&[0.; 6], &[1, 3, 2]),
            &stream
        )
        .is_err()
    );
}
