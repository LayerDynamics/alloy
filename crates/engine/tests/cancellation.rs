//! Public-API cancellation integration tests.
//!
//! Exercises actual cancellation sources, futures, scheduler permits,
//! family execution control, and cross-thread wakeups.
//!
//! Scheduler-permit tests do not perform model inference. They verify
//! control-plane ownership before or outside actual model execution.
//! Real generation cancellation is tested in tests/generation.rs.

use engine::{
    cancellation::{CancellationSource, CancellationToken},
    error::ErrorKind,
    family::ExecutionControl,
    scheduler::{RequestState, Scheduler},
    types::{CancelOutcome, RequestId},
};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Wake, Waker},
    thread,
    time::Duration,
};

#[derive(Default)]
struct WakeCounter {
    calls: AtomicUsize,
}

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}

struct ThreadWake {
    thread: thread::Thread,
}

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

fn request_id(value: &str) -> RequestId {
    RequestId::new(value).expect("request ID")
}

#[test]
fn cancellation_handles_can_cross_threads() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<CancellationSource>();
    assert_send_sync::<CancellationToken>();
    assert_send_sync::<ExecutionControl>();
}

#[test]
fn source_clones_and_token_clones_share_one_signal() {
    let source = CancellationSource::new();
    let other_source = source.clone();

    let first = source.token();
    let second = first.clone();

    assert!(!first.is_cancelled());
    assert!(!second.is_cancelled());

    assert!(other_source.request_cancel());
    assert!(!source.request_cancel());

    assert!(source.is_cancelled());
    assert!(other_source.is_cancelled());
    assert!(first.is_cancelled());
    assert!(second.is_cancelled());
}

#[test]
fn dropping_a_source_does_not_request_cancellation() {
    let token = {
        let source = CancellationSource::new();
        source.token()
    };

    assert!(!token.is_cancelled());

    let mut future = token.cancelled();
    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    let mut context = Context::from_waker(&waker);

    assert!(Pin::new(&mut future).poll(&mut context).is_pending());

    drop(future);

    assert_eq!(counter.calls.load(Ordering::SeqCst), 0,);
}

#[test]
fn cancellation_before_registration_is_immediately_observed() {
    let source = CancellationSource::new();

    assert!(source.request_cancel());

    let token = source.token();
    let mut future = token.cancelled();

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    let mut context = Context::from_waker(&waker);

    assert!(Pin::new(&mut future).poll(&mut context).is_ready());

    assert_eq!(counter.calls.load(Ordering::SeqCst), 0,);
}

#[test]
fn synchronous_check_preserves_request_identity() {
    let source = CancellationSource::new();
    let token = source.token();
    let request = request_id("typed-cancellation");

    token.check_cancelled(&request).expect("not cancelled");

    source.request_cancel();

    let error = token
        .check_cancelled(&request)
        .expect_err("typed cancellation");

    assert_eq!(error.kind(), ErrorKind::Cancelled,);

    match error {
        engine::error::EngineError::Cancelled { request_id } => {
            assert_eq!(request_id, request.as_str(),);
        }

        other => {
            panic!("unexpected cancellation error: {other:?}");
        }
    }
}

#[test]
fn repolling_updates_the_waiter_to_the_latest_waker() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let first_counter = Arc::new(WakeCounter::default());
    let second_counter = Arc::new(WakeCounter::default());

    let first_waker = Waker::from(first_counter.clone());
    let second_waker = Waker::from(second_counter.clone());

    {
        let mut context = Context::from_waker(&first_waker);

        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    {
        let mut context = Context::from_waker(&second_waker);

        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    source.request_cancel();

    assert_eq!(first_counter.calls.load(Ordering::SeqCst), 0,);

    assert_eq!(second_counter.calls.load(Ordering::SeqCst), 1,);
}

#[test]
fn equivalent_repolls_do_not_duplicate_notifications() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    let mut context = Context::from_waker(&waker);

    for _ in 0..16 {
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    source.request_cancel();

    assert_eq!(counter.calls.load(Ordering::SeqCst), 1,);

    for _ in 0..4 {
        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
    }

    assert_eq!(counter.calls.load(Ordering::SeqCst), 1,);
}

#[test]
fn every_registered_waiter_is_notified() {
    let source = CancellationSource::new();
    let token = source.token();

    let mut first = token.cancelled();
    let mut second = token.cancelled();

    let first_counter = Arc::new(WakeCounter::default());
    let second_counter = Arc::new(WakeCounter::default());

    let first_waker = Waker::from(first_counter.clone());
    let second_waker = Waker::from(second_counter.clone());

    let mut first_context = Context::from_waker(&first_waker);
    let mut second_context = Context::from_waker(&second_waker);

    assert!(Pin::new(&mut first).poll(&mut first_context).is_pending());

    assert!(Pin::new(&mut second).poll(&mut second_context).is_pending());

    source.request_cancel();

    assert_eq!(first_counter.calls.load(Ordering::SeqCst), 1,);

    assert_eq!(second_counter.calls.load(Ordering::SeqCst), 1,);

    assert!(Pin::new(&mut first).poll(&mut first_context).is_ready());

    assert!(Pin::new(&mut second).poll(&mut second_context).is_ready());
}

#[test]
fn dropping_a_pending_future_removes_its_notification() {
    let source = CancellationSource::new();
    let token = source.token();

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());

    {
        let mut future = token.cancelled();
        let mut context = Context::from_waker(&waker);

        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    assert!(!token.is_cancelled());

    source.request_cancel();

    assert_eq!(counter.calls.load(Ordering::SeqCst), 0,);
}

#[test]
fn cancellation_wakes_an_actual_waiting_thread() {
    let source = CancellationSource::new();
    let token = source.token();

    let (registered_tx, registered_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();

    let worker = thread::spawn(move || {
        let waker = Waker::from(Arc::new(ThreadWake {
            thread: thread::current(),
        }));

        let mut context = Context::from_waker(&waker);

        let mut future = token.cancelled();

        assert!(Pin::new(&mut future).poll(&mut context).is_pending());

        registered_tx.send(()).expect("registration signal");

        loop {
            thread::park();

            if Pin::new(&mut future).poll(&mut context).is_ready() {
                break;
            }
        }

        finished_tx.send(()).expect("completion signal");
    });

    registered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("waiter registration");

    source.request_cancel();

    finished_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("cancellation wakeup");

    worker.join().expect("waiting thread");
}

#[test]
fn scheduler_cancellation_reaches_family_control_without_releasing_slot() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let request = request_id("scheduled-family-control");

    let permit = scheduler
        .try_acquire(request.clone())
        .expect("execution ownership");

    let control = ExecutionControl::new(request.clone(), permit.cancellation_token());

    assert!(!control.is_cancelled());

    assert_eq!(
        scheduler.cancel(&request).expect("cancel"),
        CancelOutcome::Requested,
    );

    assert!(control.check_cancelled().is_err());

    assert_eq!(
        scheduler.request_state(&request).expect("request state"),
        RequestState::CancellationRequested,
    );

    assert!(scheduler.is_busy().expect("slot still owned"));

    let error = scheduler
        .try_acquire(request_id("competing-request"))
        .expect_err("cancelled execution still owns slot");

    assert_eq!(error.kind(), ErrorKind::Busy,);

    // This test owns no model work. Relinquishing the permit therefore
    // completes the control-plane cleanup boundary.
    drop(permit);

    assert!(!scheduler.is_busy().expect("slot released"));

    assert_eq!(
        scheduler.cancel(&request).expect("finished request"),
        CancelOutcome::AlreadyFinished,
    );
}

#[test]
fn repeated_scheduler_cancellation_is_safe() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let request = request_id("repeated-cancellation");

    let permit = scheduler.try_acquire(request.clone()).expect("permit");

    for _ in 0..4 {
        assert_eq!(
            scheduler.cancel(&request).expect("cancel"),
            CancelOutcome::Requested,
        );
    }

    assert!(permit.cancellation_token().is_cancelled());

    assert!(scheduler.is_busy().expect("owned slot"));

    drop(permit);
}

#[test]
fn cancelling_unknown_request_does_not_affect_active_request() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let active = request_id("active-request");

    let permit = scheduler.try_acquire(active.clone()).expect("permit");

    let token = permit.cancellation_token();

    let error = scheduler
        .cancel(&request_id("unknown-request"))
        .expect_err("unknown request");

    assert_eq!(error.kind(), ErrorKind::NotFound,);

    assert!(!token.is_cancelled());

    assert_eq!(
        scheduler.active_request_id().expect("active request"),
        Some(active),
    );

    drop(permit);
}

#[test]
fn duplicate_request_rejection_does_not_cancel_the_original() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let request = request_id("duplicate-request");

    let permit = scheduler
        .try_acquire(request.clone())
        .expect("original permit");

    let token = permit.cancellation_token();

    let error = scheduler
        .try_acquire(request)
        .expect_err("duplicate active request");

    assert_eq!(error.kind(), ErrorKind::InvalidRequest,);

    assert!(!token.is_cancelled());

    drop(permit);
}

#[test]
fn cancelling_finished_request_does_not_cancel_a_new_request() {
    let scheduler = Scheduler::new(8).expect("scheduler");

    let old_request = request_id("old-request");

    let old_permit = scheduler
        .try_acquire(old_request.clone())
        .expect("old permit");

    drop(old_permit);

    let new_request = request_id("new-request");

    let new_permit = scheduler
        .try_acquire(new_request.clone())
        .expect("new permit");

    let new_token = new_permit.cancellation_token();

    assert_eq!(
        scheduler.cancel(&old_request).expect("old request status"),
        CancelOutcome::AlreadyFinished,
    );

    assert!(!new_token.is_cancelled());

    assert_eq!(
        scheduler.active_request_id().expect("active request"),
        Some(new_request),
    );

    drop(new_permit);
}

#[test]
fn separate_sources_remain_independent() {
    let first = CancellationSource::new();
    let second = CancellationSource::new();

    first.request_cancel();

    assert!(first.token().is_cancelled());
    assert!(!second.token().is_cancelled());
}
