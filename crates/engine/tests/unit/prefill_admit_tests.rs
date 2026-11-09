use super::*;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("nonzero")
}

#[test]
fn disabled_admit_line_keeps_requested_chunk() {
    let decision = select_prefill_chunk(PrefillAdmitRequest {
        requested_chunk_tokens: nz(1024),
        min_chunk_tokens: nz(DEFAULT_PREFILL_MIN_CHUNK_TOKENS),
        usage_bytes: 10_000,
        admit_line_bytes: 0,
        workspace_bytes_at_requested: 8_000,
        binding: None,
    })
    .expect("disabled");
    assert_eq!(decision.chunk_tokens.get(), 1024);
    assert!(!decision.shrunk);
}

#[test]
fn shrinks_until_peak_fits_admit_line() {
    // usage 7 + workspace(1024)=8 → peak 15 > admit 12
    // half 512 → workspace 4 → peak 11 <= 12
    let decision = select_prefill_chunk(PrefillAdmitRequest {
        requested_chunk_tokens: nz(1024),
        min_chunk_tokens: nz(DEFAULT_PREFILL_MIN_CHUNK_TOKENS),
        usage_bytes: 7,
        admit_line_bytes: 12,
        workspace_bytes_at_requested: 8,
        binding: Some("dynamic"),
    })
    .expect("shrink");
    assert_eq!(decision.chunk_tokens.get(), 512);
    assert_eq!(decision.workspace_bytes, 4);
    assert!(decision.shrunk);
}

#[test]
fn rejects_when_min_chunk_still_over_admit_line() {
    let error = select_prefill_chunk(PrefillAdmitRequest {
        requested_chunk_tokens: nz(512),
        min_chunk_tokens: nz(256),
        usage_bytes: 10,
        admit_line_bytes: 11,
        workspace_bytes_at_requested: 8, // at 256 → workspace 4 → peak 14 > 11
        binding: Some("metal_cap"),
    })
    .expect_err("must reject");
    match error {
        EngineError::MemoryLimitExceeded {
            operation,
            required_bytes,
            available_bytes,
            reason,
        } => {
            assert_eq!(operation, "prefill admit line (binding=metal_cap)");
            assert_eq!(required_bytes, 14);
            assert_eq!(available_bytes, 11);
            assert_eq!(
                reason,
                MemoryLimitReason::AllocatorCeiling { ceiling_bytes: 11 }
            );
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn min_above_requested_uses_requested_as_floor() {
    let decision = select_prefill_chunk(PrefillAdmitRequest {
        requested_chunk_tokens: nz(128),
        min_chunk_tokens: nz(256),
        usage_bytes: 0,
        admit_line_bytes: 100,
        workspace_bytes_at_requested: 50,
        binding: None,
    })
    .expect("fits");
    assert_eq!(decision.chunk_tokens.get(), 128);
    assert!(!decision.shrunk);
}
