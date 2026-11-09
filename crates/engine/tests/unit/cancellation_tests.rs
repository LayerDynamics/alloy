use super::*;
use crate::error::ErrorKind;
use std::{
    sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::Wake,
    thread,
    time::Duration,
};

#[derive(Default)]
struct WakeCounter {
    count: AtomicUsize,
}

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.count.fetch_add(1, Ordering::SeqCst);
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

#[derive(Default)]
struct CallbackChecks {
    wakes: AtomicUsize,
    drops: AtomicUsize,
    locked_callbacks: AtomicUsize,
}

struct LockCheckingWake {
    shared: Weak<Shared>,
    waiter: Weak<Waiter>,
    checks: Arc<CallbackChecks>,
}

impl LockCheckingWake {
    fn inspect_locks(&self) {
        let registration_locked = self
            .shared
            .upgrade()
            .is_some_and(|shared| shared.waiters.try_lock().is_err());

        let waiter_locked = self
            .waiter
            .upgrade()
            .is_some_and(|waiter| waiter.waker.try_lock().is_err());

        if registration_locked || waiter_locked {
            self.checks.locked_callbacks.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Wake for LockCheckingWake {
    fn wake(self: Arc<Self>) {
        self.inspect_locks();
        self.checks.wakes.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.inspect_locks();
        self.checks.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for LockCheckingWake {
    fn drop(&mut self) {
        self.inspect_locks();
        self.checks.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn checking_waker(
    source: &CancellationSource,
    future: &Cancelled,
    checks: &Arc<CallbackChecks>,
) -> Waker {
    Waker::from(Arc::new(LockCheckingWake {
        shared: Arc::downgrade(&source.inner),
        waiter: Arc::downgrade(&future.waiter),
        checks: Arc::clone(checks),
    }))
}

#[test]
fn fresh_source_and_token_are_not_cancelled() {
    let source = CancellationSource::new();
    let token = source.token();

    assert!(!source.is_cancelled());
    assert!(!token.is_cancelled());
}

#[test]
fn cancellation_is_shared_and_idempotent() {
    let source = CancellationSource::new();
    let other_source = source.clone();
    let first = source.token();
    let second = first.clone();

    assert!(source.request_cancel());
    assert!(!other_source.request_cancel());

    assert!(source.is_cancelled());
    assert!(other_source.is_cancelled());
    assert!(first.is_cancelled());
    assert!(second.is_cancelled());
}

#[test]
fn synchronous_check_returns_typed_cancellation() {
    let source = CancellationSource::new();
    let token = source.token();
    let request_id = RequestId::new("request-1").expect("request ID");

    token.check_cancelled(&request_id).expect("not cancelled");

    source.request_cancel();

    let error = token
        .check_cancelled(&request_id)
        .expect_err("must report cancellation");

    assert_eq!(error.kind(), ErrorKind::Cancelled);

    match error {
        EngineError::Cancelled {
            request_id: returned,
        } => assert_eq!(returned, request_id.as_str()),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn cancellation_before_poll_is_immediately_ready() {
    let source = CancellationSource::new();
    let token = source.token();

    source.request_cancel();

    let mut future = token.cancelled();
    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter);
    let mut context = Context::from_waker(&waker);

    assert!(Pin::new(&mut future).poll(&mut context).is_ready());
    assert!(lock(&source.inner.waiters).is_empty());
}

#[test]
fn cancellation_wakes_all_registered_waiters() {
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

    assert!(source.request_cancel());

    assert_eq!(first_counter.count.load(Ordering::SeqCst), 1);
    assert_eq!(second_counter.count.load(Ordering::SeqCst), 1);
    assert!(lock(&source.inner.waiters).is_empty());

    assert!(Pin::new(&mut first).poll(&mut first_context).is_ready());
    assert!(Pin::new(&mut second).poll(&mut second_context).is_ready());
}

#[test]
fn repoll_updates_the_registered_waker() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let old_counter = Arc::new(WakeCounter::default());
    let new_counter = Arc::new(WakeCounter::default());

    let old_waker = Waker::from(old_counter.clone());
    let new_waker = Waker::from(new_counter.clone());

    let mut old_context = Context::from_waker(&old_waker);
    let mut new_context = Context::from_waker(&new_waker);

    assert!(Pin::new(&mut future).poll(&mut old_context).is_pending());
    assert!(Pin::new(&mut future).poll(&mut new_context).is_pending());

    assert_eq!(lock(&source.inner.waiters).len(), 1);

    source.request_cancel();

    assert_eq!(old_counter.count.load(Ordering::SeqCst), 0);
    assert_eq!(new_counter.count.load(Ordering::SeqCst), 1);
}

#[test]
fn equivalent_waker_repoll_does_not_duplicate_registration() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    let mut context = Context::from_waker(&waker);

    for _ in 0..8 {
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
        assert_eq!(lock(&source.inner.waiters).len(), 1);
    }

    assert!(source.request_cancel());

    assert_eq!(counter.count.load(Ordering::SeqCst), 1);
    assert!(lock(&source.inner.waiters).is_empty());
}

#[test]
fn completed_future_remains_ready_without_registration() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    let mut context = Context::from_waker(&waker);

    assert!(Pin::new(&mut future).poll(&mut context).is_pending());

    source.request_cancel();

    for _ in 0..8 {
        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
        assert!(lock(&source.inner.waiters).is_empty());
    }

    assert_eq!(counter.count.load(Ordering::SeqCst), 1);
}

#[test]
fn dropping_a_waiter_unregisters_it_without_cancelling() {
    let source = CancellationSource::new();
    let token = source.token();

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    let mut context = Context::from_waker(&waker);

    {
        let mut future = token.cancelled();

        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
        assert_eq!(lock(&source.inner.waiters).len(), 1);
    }

    assert!(lock(&source.inner.waiters).is_empty());
    assert!(!token.is_cancelled());

    source.request_cancel();

    assert_eq!(counter.count.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_wake_callback_runs_outside_internal_locks() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let checks = Arc::new(CallbackChecks::default());
    let waker = checking_waker(&source, &future, &checks);

    {
        let mut context = Context::from_waker(&waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    assert!(source.request_cancel());

    assert_eq!(checks.wakes.load(Ordering::SeqCst), 1);
    assert_eq!(checks.locked_callbacks.load(Ordering::SeqCst), 0);

    drop(waker);

    assert_eq!(checks.drops.load(Ordering::SeqCst), 1);
    assert_eq!(checks.locked_callbacks.load(Ordering::SeqCst), 0);
}

#[test]
fn replaced_waker_is_destroyed_outside_internal_locks() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let checks = Arc::new(CallbackChecks::default());
    let old_waker = checking_waker(&source, &future, &checks);

    {
        let mut context = Context::from_waker(&old_waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    // Leave the registered slot as the last owner of this callback.
    drop(old_waker);

    assert_eq!(checks.drops.load(Ordering::SeqCst), 0);

    let replacement_counter = Arc::new(WakeCounter::default());
    let replacement_waker = Waker::from(replacement_counter.clone());
    let mut replacement_context = Context::from_waker(&replacement_waker);

    assert!(
        Pin::new(&mut future)
            .poll(&mut replacement_context)
            .is_pending()
    );

    assert_eq!(checks.drops.load(Ordering::SeqCst), 1);
    assert_eq!(checks.locked_callbacks.load(Ordering::SeqCst), 0);

    source.request_cancel();

    assert_eq!(replacement_counter.count.load(Ordering::SeqCst), 1);
}

#[test]
fn unregistering_waiter_destroys_waker_outside_internal_locks() {
    let source = CancellationSource::new();
    let token = source.token();
    let mut future = token.cancelled();

    let checks = Arc::new(CallbackChecks::default());
    let waker = checking_waker(&source, &future, &checks);

    {
        let mut context = Context::from_waker(&waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    }

    drop(waker);
    assert_eq!(checks.drops.load(Ordering::SeqCst), 0);

    drop(future);

    assert_eq!(checks.drops.load(Ordering::SeqCst), 1);
    assert_eq!(checks.locked_callbacks.load(Ordering::SeqCst), 0);
    assert!(lock(&source.inner.waiters).is_empty());
    assert!(!source.is_cancelled());
}

#[test]
fn cancellation_racing_registration_is_not_lost() {
    for _ in 0..32 {
        let source = CancellationSource::new();
        let token = source.token();
        let mut future = token.cancelled();

        let counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(counter.clone());
        let mut context = Context::from_waker(&waker);

        let barrier = Arc::new(Barrier::new(2));
        let cancel_barrier = Arc::clone(&barrier);
        let cancel_source = source.clone();

        let worker = thread::spawn(move || {
            cancel_barrier.wait();
            cancel_source.request_cancel()
        });

        barrier.wait();

        let first_poll = Pin::new(&mut future).poll(&mut context);

        assert!(worker.join().expect("cancellation worker"));

        if first_poll.is_pending() {
            assert_eq!(counter.count.load(Ordering::SeqCst), 1);
        }

        assert!(Pin::new(&mut future).poll(&mut context).is_ready());
        assert!(source.is_cancelled());
        assert!(lock(&source.inner.waiters).is_empty());
    }
}

#[test]
fn cancellation_wakes_a_waiting_thread() {
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
        .expect("waiter must register");

    source.request_cancel();

    finished_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("cancellation must wake the waiter");

    worker.join().expect("worker must complete");
}

#[test]
fn separate_requests_have_independent_cancellation() {
    let first = CancellationSource::new();
    let second = CancellationSource::new();

    first.request_cancel();

    assert!(first.token().is_cancelled());
    assert!(!second.token().is_cancelled());
}
