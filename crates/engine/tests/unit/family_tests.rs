use super::*;
use crate::{cancellation::CancellationSource, scheduler::Scheduler};

#[test]
fn family_control_observes_the_original_source() {
    let source = CancellationSource::new();
    let id = RequestId::new("family-control").expect("ID");
    let control = ExecutionControl::new(id, source.token());
    assert!(!control.is_cancelled());
    source.request_cancel();
    assert!(control.check_cancelled().is_err());
}

#[test]
fn family_control_observes_scheduler_cancellation() {
    let scheduler = Scheduler::new(2).expect("scheduler");
    let id = RequestId::new("scheduled-family-control").expect("ID");
    let permit = scheduler.try_acquire(id.clone()).expect("permit");
    let control = ExecutionControl::new(id.clone(), permit.cancellation_token());
    scheduler.cancel(&id).expect("cancel");
    assert!(control.is_cancelled());
    assert!(scheduler.is_busy().expect("slot remains owned"));
    drop(permit);
    assert!(!scheduler.is_busy().expect("slot released"));
}
