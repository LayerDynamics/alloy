//! Cooperative cancellation for actual engine execution.
//!
//! The scheduler owns a CancellationSource. Execution code receives a
//! CancellationToken and checks it before prefill, between prefill chunks,
//! between decode steps, and while waiting for output-buffer capacity.
//!
//! Cancellation is monotonic: a cancelled source cannot be reset.
//! Create a new source for every generation attempt.
//!
//! Request lifecycle and terminal-event arbitration remain the scheduler's
//! responsibility. Requesting cancellation does not prove that execution
//! has stopped or that an in-flight native operation was interrupted.

use crate::{error::EngineError, types::RequestId};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex, MutexGuard, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};

/// Authority to request cancellation for one generation attempt.
///
/// Clones share the same cancellation state. Dropping a source does not
/// request cancellation automatically.
#[derive(Clone, Debug)]
pub struct CancellationSource {
    inner: Arc<Shared>,
}

/// Observation handle passed to real execution and asynchronous waits.
///
/// A token can observe cancellation but cannot request it.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    inner: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    requested: AtomicBool,
    waiters: Mutex<Vec<Weak<Waiter>>>,
}

#[derive(Debug)]
struct Waiter {
    waker: Mutex<Option<Waker>>,
}

/// Future that completes when its token observes cancellation.
///
/// Dropping this future unregisters its waiter. Dropping the future does
/// not cancel the generation.
#[derive(Debug)]
#[must_use = "a cancellation future must be awaited or polled to register a waiter"]
pub struct Cancelled {
    inner: Arc<Shared>,
    waiter: Arc<Waiter>,
    registered: bool,
}

impl CancellationSource {
    /// Construct a fresh, non-cancelled source.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Shared {
                requested: AtomicBool::new(false),
                waiters: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Create an observation token for this generation attempt.
    pub fn token(&self) -> CancellationToken {
        CancellationToken {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Request cancellation and wake registered asynchronous waiters.
    ///
    /// Returns true only for the first transition to cancelled.
    /// Subsequent calls return false.
    ///
    /// This return value describes the cancellation flag transition,
    /// not whether the generation was active or already finished. The
    /// scheduler must arbitrate that lifecycle separately.
    ///
    /// Wakers are invoked without holding internal mutexes.
    pub fn request_cancel(&self) -> bool {
        if self.inner.requested.swap(true, Ordering::AcqRel) {
            return false;
        }

        let waiters = {
            let mut registrations = lock(&self.inner.waiters);
            std::mem::take(&mut *registrations)
        };

        for registration in waiters {
            let Some(waiter) = registration.upgrade() else {
                continue;
            };

            let waker = {
                let mut slot = lock(&waiter.waker);
                slot.take()
            };

            if let Some(waker) = waker {
                waker.wake();
            }
        }

        true
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.requested.load(Ordering::Acquire)
    }
}

impl Default for CancellationSource {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    /// Whether cancellation has been requested.
    ///
    /// Execution code should check this at safe interruption points.
    pub fn is_cancelled(&self) -> bool {
        self.inner.requested.load(Ordering::Acquire)
    }

    /// Return a structured cancellation error when cancellation is requested.
    ///
    /// This method does not release reservations or emit a terminal event.
    /// Those actions occur when the execution owner finishes unwinding
    /// request-local work.
    pub fn check_cancelled(&self, request_id: &RequestId) -> Result<(), EngineError> {
        if self.is_cancelled() {
            Err(EngineError::Cancelled {
                request_id: request_id.as_str().to_owned(),
            })
        } else {
            Ok(())
        }
    }

    /// Wait asynchronously until cancellation is requested.
    ///
    /// The returned future also works when cancellation was requested
    /// before the future was created or first polled.
    pub fn cancelled(&self) -> Cancelled {
        Cancelled {
            inner: Arc::clone(&self.inner),
            waiter: Arc::new(Waiter {
                waker: Mutex::new(None),
            }),
            registered: false,
        }
    }
}

impl Future for Cancelled {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        if this.inner.requested.load(Ordering::Acquire) {
            let old_waker = {
                let mut slot = lock(&this.waiter.waker);
                slot.take()
            };
            drop(old_waker);
            return Poll::Ready(());
        }

        let inner = Arc::clone(&this.inner);

        let (ready, old_waker) = {
            let mut registrations = lock(&inner.waiters);

            // Recheck while holding the registration lock. This closes
            // the race between observing a clear flag and registering
            // a waker after cancellation has already started.
            if inner.requested.load(Ordering::Acquire) {
                let old_waker = {
                    let mut slot = lock(&this.waiter.waker);
                    slot.take()
                };

                (true, old_waker)
            } else {
                if !this.registered {
                    registrations.retain(|entry| entry.strong_count() > 0);
                    registrations.push(Arc::downgrade(&this.waiter));
                    this.registered = true;
                }

                let old_waker = {
                    let mut slot = lock(&this.waiter.waker);

                    if slot
                        .as_ref()
                        .is_some_and(|existing| existing.will_wake(cx.waker()))
                    {
                        None
                    } else {
                        slot.replace(cx.waker().clone())
                    }
                };

                (false, old_waker)
            }
        };

        // A custom waker's destructor may run arbitrary code. Drop it
        // after releasing both internal locks.
        drop(old_waker);

        if ready {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for Cancelled {
    fn drop(&mut self) {
        if self.registered {
            let own_pointer = Arc::as_ptr(&self.waiter);

            let mut registrations = lock(&self.inner.waiters);
            registrations.retain(|entry| entry.strong_count() > 0 && entry.as_ptr() != own_pointer);
        }

        let old_waker = {
            let mut slot = lock(&self.waiter.waker);
            slot.take()
        };

        drop(old_waker);
    }
}

/// Recover the cancellation bookkeeping after mutex poisoning.
///
/// Protected data consists only of waiter registrations and optional
/// wakers. Cancellation remains monotonic through the atomic flag.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use std::{
        sync::{
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

        source.request_cancel();

        assert_eq!(old_counter.count.load(Ordering::SeqCst), 0);
        assert_eq!(new_counter.count.load(Ordering::SeqCst), 1);
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
}
