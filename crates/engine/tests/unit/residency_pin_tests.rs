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
