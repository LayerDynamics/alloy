//! Isolated process for Metal allocator peaks (the counters are process-wide).
#[path = "support/ministral3.rs"]
mod ministral3;

use inference_infra::decoder::mlx_model::MlxDecoderModel;
use mlx_ffi::{Stream, memory};
use serde_json::{Value, json};
use std::process::Command;

#[test]
#[ignore = "requires a pinned Ministral checkpoint, Metal and a 36 GiB Mac"]
fn ministral3_native_memory_tiers() {
    let name = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let (manifest, expected_weights) = match name.as_str() {
        "ministral-3-3b-instruct-2512-4bit" => (
            include_str!("fixtures/ministral3_profile.json"),
            2_143_678_464,
        ),
        "ministral-3-3b-instruct-2512-8bit" => (
            include_str!("fixtures/ministral3_8bit_profile.json"),
            3_858_100_224,
        ),
        other => panic!("unknown Ministral memory profile {other}"),
    };
    let profile: Value = serde_json::from_str(manifest).unwrap();
    assert_eq!(profile["id"], name);
    let physical = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .unwrap();
    assert!(physical.status.success());
    let physical: usize = std::str::from_utf8(&physical.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(
        physical,
        36 << 30,
        "qualification requires the real 36 GiB tier"
    );
    let prompt: Vec<u32> = serde_json::from_value(profile["prompt_tokens"][0].clone()).unwrap();
    let initial_limit = memory::memory_limit_bytes().unwrap();
    for (budget, length, chunk) in [(8 << 30, 512, 64), (physical, 32_736, 32)] {
        // MLX treats its limit as a guideline. Check the measured high-water mark too.
        memory::set_memory_limit_bytes(initial_limit.min(budget)).unwrap();
        memory::clear_cache().unwrap();
        memory::reset_peak_memory().unwrap();
        let (model, stream) = ministral3::load_pinned_model(&profile, expected_weights);
        let load_peak = memory::peak_memory_bytes().unwrap();
        assert!(
            load_peak <= budget,
            "native load peak {load_peak} exceeded {budget}"
        );
        let tokens: Vec<_> = prompt.iter().copied().cycle().take(length).collect();
        let sample = measure_generation(&model, &stream, budget, &tokens, chunk);
        println!(
            "ministral_memory={}",
            json!({
                "profile": name, "budget_bytes": budget, "host_bytes": physical,
                "load_allocator_peak_bytes": load_peak, "generation": sample
            })
        );
        drop(model);
        stream.synchronize().unwrap();
        memory::set_memory_limit_bytes(initial_limit).unwrap();
    }
}

fn measure_generation(
    model: &MlxDecoderModel,
    stream: &Stream,
    budget: usize,
    tokens: &[u32],
    chunk: usize,
) -> Value {
    let weights = model.weight_buffer_bytes().unwrap();
    let kv_limit = (budget as u64).checked_sub(weights).unwrap();
    let mut cache = model.new_cache(tokens.len() + 1, kv_limit, stream).unwrap();
    stream.synchronize().unwrap();
    let kv = cache.buffer_bytes().unwrap();
    let baseline = memory::active_memory_bytes().unwrap();
    assert!(kv > 0 && weights + kv <= budget as u64);
    assert!(
        memory::peak_memory_bytes().unwrap() <= budget,
        "KV allocation exceeded budget"
    );
    memory::reset_peak_memory().unwrap();
    let mut sampled = None;
    for part in tokens.chunks(chunk) {
        let logits = model.prefill(part, &mut cache, 1 << 27, stream).unwrap();
        sampled = logits
            .values()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| u32::try_from(id).unwrap());
    }
    let logits = model
        .decode(sampled.unwrap(), &mut cache, 1 << 27, stream)
        .unwrap();
    assert_eq!(logits.values().len(), 131_072);
    assert!(logits.values().iter().all(|x| x.is_finite()));
    drop(logits);
    stream.synchronize().unwrap();
    assert_eq!(cache.len(), tokens.len() + 1);
    assert_eq!(cache.buffer_bytes().unwrap(), kv);
    let settled = memory::snapshot().unwrap();
    assert!(
        settled.peak_bytes <= budget,
        "inference exceeded budget: {settled:?}"
    );
    // Exclude any net persistent growth from the observed transient high-water mark.
    let transient = settled
        .peak_bytes
        .checked_sub(baseline.max(settled.active_bytes))
        .unwrap();
    assert!(transient > 0, "no transient native storage was observed");
    json!({
        "tokens": cache.len(), "chunk": chunk, "weight_bytes": weights,
        "committed_kv_bytes": kv, "allocator_baseline_active_bytes": baseline,
        "allocator_settled_active_bytes": settled.active_bytes,
        "allocator_peak_bytes": settled.peak_bytes,
        "transient_above_both_endpoints_bytes": transient,
        "allocator_guideline_bytes": settled.memory_limit_bytes
    })
}
