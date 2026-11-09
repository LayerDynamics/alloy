use super::*;
use crate::error::ErrorKind;
use std::{sync::Barrier, thread};

fn id(value: &str) -> RequestId {
    RequestId::new(value).expect("valid request ID")
}

fn completed() -> ExecutionOutcome {
    ExecutionOutcome::Completed {
        finish_reason: FinishReason::OutputLimit,
        usage: Usage {
            input_tokens: 3,
            output_tokens: 2,
        },
    }
}

#[test]
fn only_one_request_owns_execution() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let first = scheduler.try_acquire(id("first")).expect("permit");

    assert!(scheduler.is_busy().expect("busy state"));

    let error = scheduler
        .try_acquire(id("second"))
        .expect_err("second request must not acquire the slot");

    assert_eq!(error.kind(), ErrorKind::Busy);
    assert_eq!(
        scheduler.active_request_id().expect("active request"),
        Some(id("first"))
    );

    first.finish(completed()).expect("finish");

    assert!(!scheduler.is_busy().expect("idle state"));
    let second = scheduler
        .try_acquire(id("second"))
        .expect("slot available after execution finishes");

    second.finish(completed()).expect("finish second");
}

#[test]
fn duplicate_active_request_id_is_rejected() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let permit = scheduler.try_acquire(id("same")).expect("permit");

    let error = scheduler
        .try_acquire(id("same"))
        .expect_err("duplicate active ID");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);

    permit.finish(completed()).expect("finish");
}

#[test]
fn cancellation_keeps_execution_slot_owned() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let request = id("cancel-me");
    let permit = scheduler.try_acquire(request.clone()).expect("permit");
    let token = permit.cancellation_token();

    assert_eq!(
        scheduler.cancel(&request).expect("cancel"),
        CancelOutcome::Requested
    );
    assert!(token.is_cancelled());

    assert_eq!(
        scheduler.request_state(&request).expect("state"),
        RequestState::CancellationRequested
    );
    assert!(scheduler.is_busy().expect("still busy"));

    let error = scheduler
        .try_acquire(id("other"))
        .expect_err("cancelled work still owns the slot");

    assert_eq!(error.kind(), ErrorKind::Busy);

    let terminal = permit.finish(completed()).expect("finish");

    match terminal {
        GenerationEvent::RequestFailed {
            error,
            partial_usage,
        } => {
            assert_eq!(error.kind(), ErrorKind::Cancelled);
            assert_eq!(partial_usage.output_tokens, 2);
        }
        other => panic!("expected cancelled terminal event, got {other:?}"),
    }

    assert!(!scheduler.is_busy().expect("released"));
}

#[test]
fn cancellation_target_signals_without_releasing_execution() {
    let scheduler = Scheduler::new(2).expect("scheduler");
    assert!(
        scheduler
            .active_cancellation_target()
            .expect("idle")
            .is_none()
    );
    let permit = scheduler.try_acquire(id("target")).expect("permit");
    let token = permit.cancellation_token();
    let target = scheduler
        .active_cancellation_target()
        .expect("snapshot")
        .expect("active target");
    assert_eq!(target.request_id(), &id("target"));

    assert_eq!(
        target.request_cancel().expect("cancel"),
        CancelOutcome::Requested
    );
    assert_eq!(
        target.request_cancel().expect("repeat"),
        CancelOutcome::Requested
    );
    assert!(token.is_cancelled());
    assert_eq!(
        scheduler.request_state(target.request_id()).expect("state"),
        RequestState::CancellationRequested
    );
    assert_eq!(
        scheduler.active_cancellation_target().expect("same target"),
        Some(target.clone())
    );
    assert_eq!(
        scheduler
            .try_acquire(id("replacement"))
            .expect_err("still owned")
            .kind(),
        ErrorKind::Busy
    );

    let terminal = permit.finish(completed()).expect("finish after cleanup");
    assert!(matches!(
        terminal,
        GenerationEvent::RequestFailed {
            error: EngineError::Cancelled { .. },
            ..
        }
    ));
    assert_eq!(
        target.request_cancel().expect("stale"),
        CancelOutcome::AlreadyFinished
    );
    assert!(!scheduler.is_busy().expect("released"));
}

#[test]
fn stale_cancellation_target_cannot_cancel_replacement_or_reused_id() {
    let scheduler = Scheduler::new(1).expect("scheduler");
    let first = scheduler.try_acquire(id("original")).expect("first permit");
    let target = scheduler
        .active_cancellation_target()
        .expect("snapshot")
        .expect("target");
    first.finish(completed()).expect("finish first");

    let replacement = scheduler
        .try_acquire(id("replacement"))
        .expect("replacement");
    let replacement_token = replacement.cancellation_token();
    assert_eq!(
        target.request_cancel().expect("stale target"),
        CancelOutcome::AlreadyFinished
    );
    assert!(!replacement_token.is_cancelled());
    assert!(matches!(
        replacement.finish(completed()).expect("finish replacement"),
        GenerationEvent::GenerationCompleted { .. }
    ));

    // A bounded history eventually permits the same public ID again.
    let reused = scheduler.try_acquire(id("original")).expect("reused ID");
    let reused_token = reused.cancellation_token();
    let current = scheduler
        .active_cancellation_target()
        .expect("snapshot")
        .expect("target");
    assert_ne!(target, current);
    assert_eq!(target.request_id(), current.request_id());
    assert_eq!(
        target.request_cancel().expect("expired target"),
        CancelOutcome::AlreadyFinished
    );
    assert!(!reused_token.is_cancelled());
    assert_eq!(
        scheduler
            .request_state(current.request_id())
            .expect("running"),
        RequestState::Running
    );
    assert!(matches!(
        reused.finish(completed()).expect("finish reused"),
        GenerationEvent::GenerationCompleted { .. }
    ));
}

#[test]
fn cancellation_targets_preserve_scheduler_identity() {
    let first = Scheduler::new(1).expect("first scheduler");
    let second = Scheduler::new(1).expect("second scheduler");
    let first_permit = first.try_acquire(id("same")).expect("first permit");
    let second_permit = second.try_acquire(id("same")).expect("second permit");
    let first_target = first
        .active_cancellation_target()
        .expect("snapshot")
        .expect("target");
    let second_target = second
        .active_cancellation_target()
        .expect("snapshot")
        .expect("target");

    assert_ne!(first_target, second_target);
    first_target.request_cancel().expect("cancel first");
    assert!(first_permit.cancellation_token().is_cancelled());
    assert!(!second_permit.cancellation_token().is_cancelled());
    drop(first_target);
    drop(second_target);
    assert!(first.is_busy().expect("target does not own permit"));
    assert!(second.is_busy().expect("target does not own permit"));
    first_permit.finish(completed()).expect("finish first");
    assert!(matches!(
        second_permit.finish(completed()).expect("finish second"),
        GenerationEvent::GenerationCompleted { .. }
    ));
}

#[test]
fn cancellation_target_and_completion_have_one_serialized_winner() {
    for _ in 0..32 {
        let scheduler = Scheduler::new(1).expect("scheduler");
        let permit = scheduler.try_acquire(id("race-target")).expect("permit");
        let target = scheduler
            .active_cancellation_target()
            .expect("snapshot")
            .expect("target");
        let barrier = Arc::new(Barrier::new(2));
        let cancel_barrier = Arc::clone(&barrier);
        let cancel_worker = thread::spawn(move || {
            cancel_barrier.wait();
            target.request_cancel().expect("cancel")
        });
        let finish_worker = thread::spawn(move || {
            barrier.wait();
            permit.finish(completed()).expect("finish")
        });
        let cancellation = cancel_worker.join().expect("cancel worker");
        let terminal = finish_worker.join().expect("finish worker");
        match cancellation {
            CancelOutcome::Requested => assert!(matches!(
                terminal,
                GenerationEvent::RequestFailed {
                    error: EngineError::Cancelled { .. },
                    ..
                }
            )),
            CancelOutcome::AlreadyFinished => assert!(matches!(
                terminal,
                GenerationEvent::GenerationCompleted { .. }
            )),
        }
        assert!(!scheduler.is_busy().expect("released"));
    }
}

#[test]
fn completion_before_cancellation_is_already_finished() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let request = id("done");

    let terminal = scheduler
        .try_acquire(request.clone())
        .expect("permit")
        .finish(completed())
        .expect("finish");

    assert!(matches!(
        terminal,
        GenerationEvent::GenerationCompleted { .. }
    ));

    assert_eq!(
        scheduler.cancel(&request).expect("cancel completed"),
        CancelOutcome::AlreadyFinished
    );
}

#[test]
fn repeated_active_cancellation_is_safe() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let request = id("repeat-cancel");
    let permit = scheduler.try_acquire(request.clone()).expect("permit");

    assert_eq!(
        scheduler.cancel(&request).expect("first"),
        CancelOutcome::Requested
    );
    assert_eq!(
        scheduler.cancel(&request).expect("second"),
        CancelOutcome::Requested
    );

    permit.finish(completed()).expect("finish");
}

#[test]
fn unknown_request_is_not_found() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let error = scheduler
        .cancel(&id("unknown"))
        .expect_err("unknown request");

    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[test]
fn abandoned_permit_releases_slot_and_signals_cancellation() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let request = id("abandoned");
    let permit = scheduler.try_acquire(request.clone()).expect("permit");
    let token = permit.cancellation_token();

    drop(permit);

    assert!(token.is_cancelled());
    assert!(!scheduler.is_busy().expect("released"));
    assert_eq!(
        scheduler.request_state(&request).expect("history"),
        RequestState::Finished
    );
}

#[test]
fn completed_history_is_bounded() {
    let scheduler = Scheduler::new(2).expect("scheduler");

    for name in ["one", "two", "three"] {
        scheduler
            .try_acquire(id(name))
            .expect("permit")
            .finish(completed())
            .expect("finish");
    }

    assert_eq!(
        scheduler.retained_finished_count().expect("history size"),
        2
    );

    let error = scheduler
        .cancel(&id("one"))
        .expect_err("oldest completed ID was evicted");

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        scheduler.cancel(&id("three")).expect("recent history"),
        CancelOutcome::AlreadyFinished
    );
}

#[test]
fn recently_finished_id_cannot_be_reused() {
    let scheduler = Scheduler::new(8).expect("scheduler");
    let request = id("finished");

    scheduler
        .try_acquire(request.clone())
        .expect("permit")
        .finish(completed())
        .expect("finish");

    let error = scheduler
        .try_acquire(request)
        .expect_err("request attempt IDs must be fresh");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
}

#[test]
fn execution_failure_is_preserved_without_cancellation() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let terminal = scheduler
        .try_acquire(id("execution-failure"))
        .expect("permit")
        .finish(ExecutionOutcome::Failed {
            error: EngineError::ExecutionFailed {
                stage: "decode",
                message: "non-finite logits".to_owned(),
            },
            partial_usage: Usage {
                input_tokens: 3,
                output_tokens: 1,
            },
        })
        .expect("finish");

    match terminal {
        GenerationEvent::RequestFailed {
            error,
            partial_usage,
        } => {
            assert_eq!(error.kind(), ErrorKind::ExecutionFailed);
            assert_eq!(partial_usage.output_tokens, 1);
        }
        other => panic!("expected execution failure, got {other:?}"),
    }
}

#[test]
fn cancellation_and_completion_have_one_serialized_winner() {
    for _ in 0..32 {
        let scheduler = Scheduler::new(2).expect("scheduler");
        let request = id("race");
        let permit = scheduler.try_acquire(request.clone()).expect("permit");

        let barrier = Arc::new(Barrier::new(2));

        let cancel_scheduler = scheduler.clone();
        let cancel_request = request.clone();
        let cancel_barrier = barrier.clone();

        let cancel_worker = thread::spawn(move || {
            cancel_barrier.wait();
            cancel_scheduler.cancel(&cancel_request).expect("cancel")
        });

        let finish_worker = thread::spawn(move || {
            barrier.wait();
            permit.finish(completed()).expect("finish")
        });

        let cancel_outcome = cancel_worker.join().expect("cancel worker");
        let terminal = finish_worker.join().expect("finish worker");

        match cancel_outcome {
            CancelOutcome::Requested => match terminal {
                GenerationEvent::RequestFailed { error, .. } => {
                    assert_eq!(error.kind(), ErrorKind::Cancelled);
                }
                other => {
                    panic!("accepted cancellation must win: {other:?}");
                }
            },

            CancelOutcome::AlreadyFinished => {
                assert!(matches!(
                    terminal,
                    GenerationEvent::GenerationCompleted { .. }
                ));
            }
        }

        assert!(!scheduler.is_busy().expect("released"));
    }
}

#[test]
fn zero_history_capacity_is_rejected() {
    let error = Scheduler::new(0).expect_err("invalid history capacity");
    assert_eq!(error.kind(), ErrorKind::InvalidRequest);
}
