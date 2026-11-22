//! Isolated Metal process: MLX allocator counters and limits are process-wide.
#[path = "support/gemma4.rs"]
mod gemma4;

use inference_infra::decoder::gemma4::model::{Gemma4Cache, Gemma4Model};
use inference_infra::decoder::mlx_model::MlxLogits;
use mlx_ffi::{Device, DeviceType, Stream, memory};
use serde_json::json;
use std::process::Command;

fn step(model: &Gemma4Model, stream: &Stream, cache: &mut Gemma4Cache, ids: &[u32]) -> MlxLogits {
    let end = cache.len() + ids.len();
    let workspace = model.config().workspace_bytes(ids.len(), end).unwrap();
    let logits = model
        .step(ids, cache, workspace, stream, &mut || false)
        .unwrap();
    assert_eq!(cache.len(), end);
    assert_eq!(cache.retained_tokens(), end.min(512));
    assert_eq!(logits.values().len(), 262_144);
    assert!(logits.values().iter().all(|v| v.is_finite()));
    logits
}

fn greedy(logits: &MlxLogits) -> u32 {
    let (id, _) = logits
        .values()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    id.try_into().unwrap()
}

#[test]
#[ignore = "requires pinned ALLOY_GEMMA4_ARTIFACT, Metal and a 36 GiB Mac"]
fn gemma4_native_memory_tiers() {
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
    let profile = gemma4::profile();
    let prompt: Vec<u32> = serde_json::from_value(profile["prompt_tokens"][0].clone()).unwrap();
    let stream = Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap();
    let initial_limit = memory::memory_limit_bytes().unwrap();
    for (budget, length) in [(8 << 30, 512), (physical, 32_736)] {
        memory::set_memory_limit_bytes(initial_limit.min(budget)).unwrap();
        memory::clear_cache().unwrap();
        memory::reset_peak_memory().unwrap();
        let model = gemma4::load(&profile, &stream);
        stream.synchronize().unwrap();
        let weights = model.weight_buffer_bytes();
        assert!(weights >= model.config().weight_payload_plan().unwrap().payload_bytes);
        let load_peak = memory::peak_memory_bytes().unwrap();
        assert!(load_peak <= budget);
        let mut tokens: Vec<_> = prompt.iter().copied().cycle().take(length).collect();
        let kv_limit = model.config().cache_overlap_bound(length + 1, 32).unwrap();
        let mut cache = model.new_cache(length + 1, kv_limit).unwrap();
        assert_eq!(cache.buffer_bytes().unwrap(), 0);
        let baseline = memory::active_memory_bytes().unwrap();
        memory::reset_peak_memory().unwrap();
        let mut parts = tokens.chunks_exact(32);
        let last = parts.next_back().unwrap();
        assert!(parts.remainder().is_empty());
        for part in parts {
            step(&model, &stream, &mut cache, part);
        }
        let next = greedy(&step(&model, &stream, &mut cache, last));
        tokens.push(next);
        let cached = step(&model, &stream, &mut cache, &[next]);
        let allocated = cache.buffer_bytes().unwrap();
        // Twenty local and four global KV owners with two f32 key/value heads.
        let logical = (20 * 512 * 256 + 4 * tokens.len() * 512) as u64 * 2 * 2 * 4;
        assert_eq!(model.config().cache_payload(tokens.len()).unwrap(), logical);
        assert!(
            (logical..=kv_limit).contains(&allocated),
            "KV {allocated}, logical {logical}, overlap {kv_limit}"
        );
        let settled = memory::snapshot().unwrap();
        assert!(
            settled.peak_bytes <= budget,
            "generation exceeded tier: {settled:?}"
        );
        let transient = settled
            .peak_bytes
            .checked_sub(baseline.max(settled.active_bytes))
            .unwrap();
        assert!(transient > 0);
        println!(
            "gemma4_memory={}",
            json!({
                "profile": profile["id"], "budget_bytes": budget, "physical_bytes": physical,
                "tokens": tokens.len(), "chunk": 32, "weight_bytes": weights,
                "load_allocator_peak_bytes": load_peak, "empty_kv_bytes": 0,
                "logical_kv_bytes": logical, "committed_kv_bytes": allocated,
                "shortest_retained_owner_tokens": cache.retained_tokens(),
                "allocator_baseline_active_bytes": baseline, "allocator_settled_active_bytes": settled.active_bytes,
                "allocator_peak_bytes": settled.peak_bytes, "allocator_guideline_bytes": settled.memory_limit_bytes,
                "transient_above_both_endpoints_bytes": transient
            })
        );
        drop(cache);
        if length == 512 {
            let n = tokens.len();
            let mut cold_cache = model
                .new_cache(n, model.config().cache_overlap_bound(n, n).unwrap())
                .unwrap();
            let cold = step(&model, &stream, &mut cold_cache, &tokens);
            let max_abs = cached
                .values()
                .iter()
                .zip(cold.values())
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            println!(
                "gemma4_window_replay tokens={n} cached={} cold={} max_abs={max_abs}",
                greedy(&cached),
                greedy(&cold)
            );
            assert_eq!(greedy(&cached), greedy(&cold));
            assert!(max_abs < 0.02, "window-boundary logits differ by {max_abs}");
            assert!(memory::peak_memory_bytes().unwrap() <= budget);
        }
        drop(model);
        stream.synchronize().unwrap();
    }
    memory::set_memory_limit_bytes(initial_limit).unwrap();
}
