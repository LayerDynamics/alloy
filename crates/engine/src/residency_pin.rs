//! Explicit, instance-bound model residency pins.
//!
//! The owner belongs to the native executor. Handles may cross threads;
//! dropping the final handle removes protection without a worker command.
//!
//! This module does not own model buffers or admission reservations.
//! Runtime integration must check pin protection at every model-destruction
//! path and serialize acquisition with eviction authorization.

use crate::family::ModelInstanceId;
use std::{
    num::NonZeroUsize,
    sync::{Arc, Weak},
};

/// Why an explicit model pin could not be acquired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinAcquireError {
    /// The caller addressed a different loaded instance.
    InstanceChanged,
    /// The executor has already claimed destruction of this instance.
    Evicting,
}

/// Caller-owned protection for one particular loaded model instance.
///
/// Cloning extends the same protection lifetime. Dropping a handle releases
/// only its ownership share; the owner remains protected until all handles
/// for that protection have been dropped.
///
/// The identity token does not itself own or retain native model buffers.
#[derive(Clone, Debug)]
pub struct ModelResidencyPin {
    instance: ModelInstanceId,
    _protection: Arc<()>,
}

impl ModelResidencyPin {
    /// Exact native load protected by this handle.
    pub fn instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }
}

/// Executor-owned source of explicit pins for one loaded model.
///
/// This value must move with its resident through publication or pool moves.
/// It must not be recreated to clear existing protection.
///
/// `try_pin` and the runtime's eviction claim must run on the same serialized
/// executor. The runtime, not this primitive, owns the evicting state.
#[derive(Debug)]
pub struct ModelPinOwner {
    instance: ModelInstanceId,
    protection: Weak<()>,
}

impl ModelPinOwner {
    /// Bind pin ownership to a freshly loaded native instance.
    pub fn new(instance: ModelInstanceId) -> Self {
        Self {
            instance,
            protection: Weak::new(),
        }
    }

    /// Exact loaded instance associated with this owner.
    pub fn instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }

    /// Whether at least one explicit pin handle is alive.
    ///
    /// A concurrent final drop may make a protected result conservative.
    /// New acquisition must still be serialized with eviction on the worker.
    pub fn is_pinned(&self) -> bool {
        self.live_handles().is_some()
    }

    /// Snapshot the live handle count without acquiring another pin.
    ///
    /// Concurrent drops can make this observation conservative. New
    /// acquisition remains serialized with native ownership operations.
    pub(crate) fn live_handles(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.protection.strong_count())
    }

    /// Observe handle lifetime without acquiring residency protection.
    pub(crate) fn protection_observer(&self) -> Weak<()> {
        self.protection.clone()
    }

    /// Acquire protection after executor-side identity and lifecycle checks.
    ///
    /// The `evicting` value must come from current executor-owned state,
    /// not a previously published metadata snapshot.
    pub fn try_pin(
        &mut self,
        expected: &ModelInstanceId,
        evicting: bool,
    ) -> Result<ModelResidencyPin, PinAcquireError> {
        if !self.instance.same_instance(expected) {
            return Err(PinAcquireError::InstanceChanged);
        }
        if evicting {
            return Err(PinAcquireError::Evicting);
        }

        let protection = match self.protection.upgrade() {
            Some(protection) => protection,
            None => {
                let protection = Arc::new(());
                self.protection = Arc::downgrade(&protection);
                protection
            }
        };

        Ok(ModelResidencyPin {
            instance: self.instance.clone(),
            _protection: protection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_handle_drop_removes_protection() {
        let instance = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(instance.clone());
        assert!(!owner.is_pinned());

        let first = owner.try_pin(&instance, false).expect("first pin");
        let second = owner.try_pin(&instance, false).expect("second pin");
        let cloned = first.clone();

        drop(first);
        drop(second);
        assert!(owner.is_pinned());
        drop(cloned);
        assert!(!owner.is_pinned());
    }

    #[test]
    fn pin_binds_to_native_instance_not_artifact_identity() {
        let original = ModelInstanceId::new();
        let replacement = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(replacement);

        assert_eq!(
            owner.try_pin(&original, false).expect_err("stale instance"),
            PinAcquireError::InstanceChanged,
        );
        assert!(!owner.is_pinned());
    }

    #[test]
    fn evicting_instance_rejects_new_acquisition() {
        let instance = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(instance.clone());

        assert_eq!(
            owner.try_pin(&instance, true).expect_err("evicting"),
            PinAcquireError::Evicting,
        );
        assert!(!owner.is_pinned());
    }

    #[test]
    fn handle_reports_the_exact_instance() {
        let instance = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(instance.clone());
        let pin = owner.try_pin(&instance, false).expect("pin");
        assert!(pin.instance_id().same_instance(&instance));
    }

    #[test]
    fn owner_move_preserves_live_protection() {
        let instance = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(instance.clone());
        let pin = owner.try_pin(&instance, false).expect("pin");

        let moved_owner = owner;
        assert!(moved_owner.is_pinned());
        drop(pin);
        assert!(!moved_owner.is_pinned());
    }

    #[test]
    fn acquisition_after_final_drop_creates_fresh_protection() {
        let instance = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(instance.clone());

        drop(owner.try_pin(&instance, false).expect("first pin"));
        assert!(!owner.is_pinned());

        let pin = owner.try_pin(&instance, false).expect("new pin");
        assert!(owner.is_pinned());
        drop(pin);
        assert!(!owner.is_pinned());
    }

    #[test]
    fn handle_can_be_dropped_on_another_thread() {
        let instance = ModelInstanceId::new();
        let mut owner = ModelPinOwner::new(instance.clone());
        let pin = owner.try_pin(&instance, false).expect("pin");

        std::thread::spawn(move || drop(pin))
            .join()
            .expect("thread");
        assert!(!owner.is_pinned());
    }
}
