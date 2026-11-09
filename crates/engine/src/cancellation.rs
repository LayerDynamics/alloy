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
//!
//! Waker cloning, destruction, and invocation occur outside internal
//! cancellation mutexes.

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
///
/// Repeated pending polls update the registered waker when necessary
/// without registering the same future more than once.
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
    /// Registered wakers are removed from internal state and invoked
    /// without holding either internal mutex.
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

        // A custom waker clone callback may execute arbitrary code.
        // Clone before acquiring either internal mutex.
        //
        // Cancellation may occur during cloning, so the flag must still
        // be rechecked under the registration lock below.
        let mut replacement_waker = Some(cx.waker().clone());
        let inner = Arc::clone(&this.inner);

        let (ready, old_waker) = {
            let mut registrations = lock(&inner.waiters);

            // Serialize registration against request_cancel taking the
            // registration list. If cancellation has already started,
            // do not install a new waiter or replacement waker.
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
                        // Keep the equivalent stored waker. The unused
                        // replacement is dropped after both locks exit.
                        None
                    } else {
                        // Move the already-cloned replacement into the
                        // slot without cloning or dropping a waker here.
                        std::mem::replace(&mut *slot, replacement_waker.take())
                    }
                };

                (false, old_waker)
            }
        };

        // Either value may invoke a custom destructor. Both mutex guards
        // have been released before either value is dropped.
        drop(old_waker);
        drop(replacement_waker);

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

            {
                let mut registrations = lock(&self.inner.waiters);

                registrations
                    .retain(|entry| entry.strong_count() > 0 && entry.as_ptr() != own_pointer);
            }
        }

        let old_waker = {
            let mut slot = lock(&self.waiter.waker);
            slot.take()
        };

        drop(old_waker);
    }
}

/// Recover cancellation bookkeeping after mutex poisoning.
///
/// Protected data consists only of waiter registrations and optional
/// wakers. Cancellation remains monotonic through the atomic flag.
///
/// This recovery policy is specific to cancellation notification. It must
/// not be copied to admission or scheduler accounting without separately
/// establishing that their invariants remain trustworthy.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/cancellation_tests.rs"]
mod tests;
