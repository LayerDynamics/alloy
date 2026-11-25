//! Isolated allocator measurements on the same hybrid decoder used for generation.
#[path = "support/qwen35.rs"]
mod qwen35;

use inference_infra::decoder::qwen35::model::{ForwardOptions, MlxQwen35Model, Qwen35Cache};
use mlx_ffi::{Device, DeviceType, Stream, memory};
use serde_json::{Value, json};
use std::process::Command;

fn forward(
    model: &MlxQwen35Model,
    stream: &Stream,
    cache: &mut Qwen35Cache,
    tokens: &[u32],
) -> u32 {
    let end = cache.len() + tokens.len();
    let plan = model
        .config()
        .step_payload_plan(tokens.len(), end, false)
        .unwrap();
    let options = ForwardOptions {
        all_logits: false,
        output_limit: plan.output_bytes,
        work_limit: plan.work_bytes,
    };
    let logits = model
        .forward(tokens, cache, options, stream, &|| false)
        .unwrap();
    assert_eq!(cache.len(), end);
    assert_eq!(logits.values().len(), model.config().vocabulary);
    assert!(logits.values().iter().all(|x| x.is_finite()));
    let (id, _) = logits
        .values()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    id.try_into().unwrap()
}

#[test]
#[ignore = "requires pinned Ornith or upstream Qwen3.5 on a 36 GiB Metal host"]
fn qwen35_native_memory_tiers() {
    let name = std::env::var("ALLOY_REAL_PROFILE").unwrap();
    let (manifest, artifact_env) = match name.as_str() {
        "ornith-1.5-9b-4bit" => (
            include_str!("fixtures/ornith_1_5_9b_profile.json"),
            "ALLOY_ORNITH_ARTIFACT",
        ),
        "qwen3.5-0.8b-4bit" => (
            include_str!("fixtures/qwen35_0_8b_profile.json"),
            "ALLOY_QWEN35_SMALL_ARTIFACT",
        ),
        other => panic!("unknown hybrid memory profile {other}"),
    };
    let profile: Value = serde_json::from_str(manifest).unwrap();
    assert_eq!(profile["id"], name);
    let host = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .unwrap();
    assert!(host.status.success());
    let physical: usize = std::str::from_utf8(&host.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(physical, 36 << 30);
    let prompt: Vec<u32> = serde_json::from_value(profile["prompt_tokens"][0].clone()).unwrap();
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let initial_limit = memory::memory_limit_bytes().unwrap();
    for (tier, length) in [(8 << 30, 512), (physical, 32_736)] {
        memory::set_memory_limit_bytes(initial_limit.min(tier)).unwrap();
        memory::clear_cache().unwrap();
        memory::reset_peak_memory().unwrap();
        let (config, weights) = qwen35::load(&profile, artifact_env);
        let model = MlxQwen35Model::from_weights(config, weights, tier as u64, &stream).unwrap();
        stream.synchronize().unwrap();
        let weights = model.weight_buffer_bytes().unwrap();
        assert!(weights >= model.weight_payload_bytes());
        let load_peak = memory::peak_memory_bytes().unwrap();
        assert!(load_peak <= tier);
        let plan = model.config().cache_payload_plan(length + 1).unwrap();
        let allowance = (tier as u64).checked_sub(weights).unwrap();
        assert!(plan.total_overlap_bytes <= allowance);
        let mut cache = model.new_cache(length + 1, allowance, &stream).unwrap();
        let (empty_full, empty_recurrent) = cache.component_buffer_bytes().unwrap();
        let logical_full = plan.full.payload_plan().unwrap().resident_bytes;
        assert!(empty_full >= logical_full);
        assert_eq!(empty_recurrent, 0);
        let baseline = memory::snapshot().unwrap();
        assert!(baseline.peak_bytes <= tier);
        memory::reset_peak_memory().unwrap();
        let tokens: Vec<_> = prompt.iter().copied().cycle().take(length).collect();
        let mut next = None;
        for part in tokens.chunks(32) {
            next = Some(forward(&model, &stream, &mut cache, part));
        }
        forward(&model, &stream, &mut cache, &[next.unwrap()]);
        let (full, recurrent) = cache.component_buffer_bytes().unwrap();
        let total = cache.buffer_bytes().unwrap();
        assert_eq!(total, full + recurrent);
        assert!(
            full >= logical_full && recurrent >= plan.recurrent_overlap_bytes / 2 && recurrent > 0
        );
        assert!(total <= allowance);
        let settled = memory::snapshot().unwrap();
        let transient = settled
            .peak_bytes
            .checked_sub(settled.active_bytes.max(baseline.active_bytes))
            .unwrap();
        println!(
            "qwen35_memory={}",
            json!({
                "profile": name, "physical_bytes": physical, "budget_bytes": tier, "chunk": 32,
                "tokens": cache.len(), "weight_bytes": weights, "empty_full_kv_bytes": empty_full,
                "committed_full_kv_bytes": full, "committed_recurrent_bytes": recurrent,
                "load_peak_bytes": load_peak, "initial_cache_peak_bytes": baseline.peak_bytes,
                "allocator_peak_bytes": settled.peak_bytes, "allocator_baseline_active_bytes": baseline.active_bytes,
                "allocator_settled_active_bytes": settled.active_bytes, "allocator_guideline_bytes": settled.memory_limit_bytes,
                "transient_above_both_endpoints_bytes": transient
            })
        );
        assert!(settled.peak_bytes <= tier && transient > 0, "{settled:?}");
        assert_eq!(cache.len(), length + 1);
    }
    memory::set_memory_limit_bytes(initial_limit).unwrap();
}
