use super::*;
use crate::error::ErrorKind;

fn observed_usage(input_tokens: usize, output_tokens: usize) -> UsageTracker {
    let mut tracker = UsageTracker::default();

    tracker
        .record_prefill(
            Usage {
                input_tokens,
                output_tokens: 0,
            },
            input_tokens,
        )
        .expect("valid prefill accounting");

    for output in 1..=output_tokens {
        tracker
            .record_decode(
                Usage {
                    input_tokens,
                    output_tokens: output,
                },
                input_tokens,
                output,
            )
            .expect("valid decode accounting");
    }

    tracker
}

#[test]
fn final_output_regression_preserves_validated_usage() {
    let mut tracker = observed_usage(3, 12);
    let before = tracker.latest;

    let error = tracker
        .reconcile_final(
            Usage {
                input_tokens: 3,
                output_tokens: 0,
            },
            3,
            20,
            false,
        )
        .expect_err("output usage must not move backward");

    assert_eq!(error.kind(), ErrorKind::Internal);
    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);
}

#[test]
fn final_input_regression_preserves_validated_usage() {
    let mut tracker = observed_usage(8, 2);
    let before = tracker.latest;

    let error = tracker
        .reconcile_final(
            Usage {
                input_tokens: 7,
                output_tokens: 2,
            },
            8,
            20,
            false,
        )
        .expect_err("input usage must not move backward");

    assert_eq!(error.kind(), ErrorKind::Internal);
    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);
}

#[test]
fn successful_final_usage_must_match_completion_exactly() {
    let mut tracker = observed_usage(3, 12);
    let before = tracker.latest;

    let error = tracker
        .reconcile_final(
            Usage {
                input_tokens: 3,
                output_tokens: 13,
            },
            3,
            20,
            true,
        )
        .expect_err("success cannot add unvalidated progress");

    assert_eq!(error.kind(), ErrorKind::Internal);
    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);
}

#[test]
fn successful_matching_final_usage_is_accepted() {
    let mut tracker = observed_usage(3, 12);
    let expected = tracker.latest;

    tracker
        .reconcile_final(expected, 3, 20, true)
        .expect("matching successful report");

    assert_eq!(tracker.latest, expected);
    assert!(!tracker.inconsistent);
}

#[test]
fn failed_execution_can_report_additional_bounded_progress() {
    let mut tracker = observed_usage(3, 12);

    let reported = Usage {
        input_tokens: 3,
        output_tokens: 13,
    };

    tracker
        .reconcile_final(reported, 3, 20, false)
        .expect("additional work before failure");

    assert_eq!(tracker.latest, reported);
    assert!(!tracker.inconsistent);
}

#[test]
fn failed_prefill_can_report_partial_input_usage() {
    let mut tracker = UsageTracker::default();

    let reported = Usage {
        input_tokens: 2,
        output_tokens: 0,
    };

    tracker
        .reconcile_final(reported, 8, 20, false)
        .expect("partial failed prefill accounting");

    assert_eq!(tracker.latest, reported);
    assert!(!tracker.inconsistent);
}

#[test]
fn final_usage_outside_input_bounds_is_rejected() {
    let mut tracker = observed_usage(3, 2);
    let before = tracker.latest;

    let error = tracker
        .reconcile_final(
            Usage {
                input_tokens: 4,
                output_tokens: 2,
            },
            3,
            20,
            false,
        )
        .expect_err("input usage exceeds admitted input");

    assert_eq!(error.kind(), ErrorKind::Internal);
    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);
}

#[test]
fn final_usage_outside_output_bounds_is_rejected() {
    let mut tracker = observed_usage(3, 2);
    let before = tracker.latest;

    let error = tracker
        .reconcile_final(
            Usage {
                input_tokens: 3,
                output_tokens: 21,
            },
            3,
            20,
            false,
        )
        .expect_err("output usage exceeds admitted output");

    assert_eq!(error.kind(), ErrorKind::Internal);
    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);
}

#[test]
fn inconsistent_decode_cannot_be_overwritten_by_final_report() {
    let mut tracker = observed_usage(3, 2);
    let before = tracker.latest;

    tracker
        .record_decode(
            Usage {
                input_tokens: 3,
                output_tokens: 7,
            },
            3,
            3,
        )
        .expect_err("decode must advance by exactly one");

    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);

    tracker
        .reconcile_final(
            Usage {
                input_tokens: 3,
                output_tokens: 7,
            },
            3,
            20,
            false,
        )
        .expect_err("later report cannot restore accounting trust");

    assert_eq!(tracker.latest, before);
}

#[test]
fn text_finish_cannot_change_token_usage() {
    let mut tracker = observed_usage(3, 12);
    let before = tracker.latest;

    tracker
        .validate_text_finish(Usage {
            input_tokens: 3,
            output_tokens: 13,
        })
        .expect_err("text finish must not generate another token");

    assert_eq!(tracker.latest, before);
    assert!(tracker.inconsistent);
}

#[test]
fn matching_text_finish_preserves_validated_usage() {
    let mut tracker = observed_usage(3, 12);
    let before = tracker.latest;

    tracker
        .validate_text_finish(before)
        .expect("unchanged usage");

    assert_eq!(tracker.latest, before);
    assert!(!tracker.inconsistent);
}

#[test]
fn invalid_prefill_does_not_replace_initial_usage() {
    let mut tracker = UsageTracker::default();

    tracker
        .record_prefill(
            Usage {
                input_tokens: 3,
                output_tokens: 1,
            },
            3,
        )
        .expect_err("prefill must not report generated tokens");

    assert_eq!(tracker.latest, Usage::default());
    assert!(tracker.inconsistent);
}
