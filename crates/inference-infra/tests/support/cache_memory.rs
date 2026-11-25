//! Run in a single-test process because MLX allocator counters are process-wide.
use inference_infra::decoder::mlx_model::MlxLogits;
use mlx_ffi::{Device, DeviceType, Stream, memory};
use serde_json::{Value, json};
use std::process::Command;

pub struct Tier<'a> {
    profile: &'a str,
    physical: usize,
    pub budget: usize,
    pub tokens: Vec<u32>,
    pub stream: Stream,
}

pub fn run(profile: &Value, mut exercise: impl FnMut(&Tier<'_>)) {
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
    assert!(!prompt.is_empty());
    let initial_limit = memory::memory_limit_bytes().unwrap();
    for (budget, length) in [(8 << 30, 512), (physical, 32_736)] {
        memory::set_memory_limit_bytes(initial_limit.min(budget)).unwrap();
        memory::clear_cache().unwrap();
        memory::reset_peak_memory().unwrap();
        let tier = Tier {
            profile: profile["id"].as_str().unwrap(),
            budget,
            physical,
            tokens: prompt.iter().copied().cycle().take(length).collect(),
            stream: Stream::new(&Device::new(DeviceType::Gpu, 0).unwrap()).unwrap(),
        };
        exercise(&tier);
        tier.stream.synchronize().unwrap();
    }
    memory::set_memory_limit_bytes(initial_limit).unwrap();
}

impl Tier<'_> {
    pub fn load_peak(&self) -> usize {
        self.stream.synchronize().unwrap();
        let peak = memory::peak_memory_bytes().unwrap();
        assert!(
            peak <= self.budget,
            "native load peak {peak} exceeds {}",
            self.budget
        );
        peak
    }

    pub fn measure<C>(
        &self,
        load_peak: usize,
        weights: u64,
        vocabulary: usize,
        mut cache: C,
        mut forward: impl FnMut(&[u32], &mut C) -> MlxLogits,
        usage: impl Fn(&C) -> (usize, u64),
    ) -> (u64, u64) {
        self.stream.synchronize().unwrap();
        let (length, initial_kv) = usage(&cache);
        assert_eq!(length, 0);
        let baseline = memory::snapshot().unwrap();
        assert!(baseline.peak_bytes <= self.budget);
        memory::reset_peak_memory().unwrap();
        let mut next = 0;
        for chunk in self.tokens.chunks(32) {
            let logits = forward(chunk, &mut cache);
            assert_eq!(logits.values().len(), vocabulary);
            assert!(logits.values().iter().all(|x| x.is_finite()));
            next = logits
                .values()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
                .try_into()
                .unwrap();
        }
        let logits = forward(&[next], &mut cache);
        assert_eq!(logits.values().len(), vocabulary);
        assert!(logits.values().iter().all(|x| x.is_finite()));
        drop(logits);
        self.stream.synchronize().unwrap();
        let (tokens, kv) = usage(&cache);
        assert_eq!(tokens, self.tokens.len() + 1);
        assert!(weights > 0 && kv > 0 && weights + kv <= self.budget as u64);
        let settled = memory::snapshot().unwrap();
        let transient = settled
            .peak_bytes
            .checked_sub(baseline.active_bytes.max(settled.active_bytes))
            .unwrap();
        println!(
            "native_cache_memory={}",
            json!({
                "profile": self.profile, "tokens": tokens, "physical_bytes": self.physical, "budget_bytes": self.budget,
                "weight_bytes":weights, "initial_kv_bytes":initial_kv, "committed_kv_bytes":kv,
                "load_peak_bytes":load_peak, "initial_cache_peak_bytes":baseline.peak_bytes,
                "allocator_baseline_active_bytes":baseline.active_bytes, "allocator_settled_active_bytes":settled.active_bytes,
                "allocator_peak_bytes":settled.peak_bytes,"transient_above_both_endpoints_bytes":transient
            })
        );
        assert!(
            settled.peak_bytes <= self.budget && transient > 0,
            "{settled:?}"
        );
        (initial_kv, kv)
    }
}
