//! Explicit paired RPC transport options. This module does not enable tools.
use alloyd::{DaemonService, rpc::RpcTransportLimits};
use engine::api::Engine;
use std::time::Duration;

#[derive(Default)]
pub(super) struct TransportArgs {
    maximum: Option<usize>,
    timeout_ms: Option<u64>,
}
impl TransportArgs {
    pub(super) fn set_maximum(&mut self, raw: &str) -> Result<(), String> {
        if self.maximum.is_some() {
            return Err("--rpc-max-notification-bytes specified more than once".into());
        }
        digits(raw, "--rpc-max-notification-bytes")?;
        let value = raw.parse::<usize>().map_err(|_| {
            "--rpc-max-notification-bytes exceeds the supported integer range".to_owned()
        })?;
        if value == 0 {
            return Err("--rpc-max-notification-bytes must be positive".into());
        }
        self.maximum = Some(value);
        Ok(())
    }
    pub(super) fn set_timeout(&mut self, raw: &str) -> Result<(), String> {
        if self.timeout_ms.is_some() {
            return Err("--rpc-write-timeout-ms specified more than once".into());
        }
        digits(raw, "--rpc-write-timeout-ms")?;
        let value = raw
            .parse::<u64>()
            .map_err(|_| "--rpc-write-timeout-ms exceeds the supported integer range".to_owned())?;
        if value == 0 {
            return Err("--rpc-write-timeout-ms must be positive".into());
        }
        self.timeout_ms = Some(value);
        Ok(())
    }
    pub(super) fn finish(self) -> Result<Option<RpcTransportLimits>, String> {
        match (self.maximum, self.timeout_ms) {
            (None, None) => Ok(None),
            (Some(maximum), Some(timeout)) => {
                RpcTransportLimits::new(maximum, Duration::from_millis(timeout))
                    .map(Some)
                    .map_err(|error| error.message)
            }
            _ => Err("use --rpc-max-notification-bytes and --rpc-write-timeout-ms together".into()),
        }
    }
}
fn digits(raw: &str, flag: &str) -> Result<(), String> {
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("{flag} requires a positive decimal integer"));
    }
    Ok(())
}

// Called by the actual daemon constructor after options have been validated.
pub(super) fn apply<E: Engine>(
    service: DaemonService<E>,
    policy: Option<RpcTransportLimits>,
) -> DaemonService<E> {
    match policy {
        Some(limits) => service.with_rpc_transport_limits(limits),
        None => service,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloyd::FixtureEngine;
    use harness_core::{HardwareTier, TelemetryClient};
    use std::{
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "alloy-rpc-policy-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("create test directory: {e}"),
                }
            }
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn absent_options_preserve_unconfigured_transport() {
        assert!(TransportArgs::default().finish().expect("none").is_none());
    }
    #[test]
    fn paired_options_preserve_exact_values() {
        let mut a = TransportArgs::default();
        a.set_maximum("65537").expect("bytes");
        a.set_timeout("1234").expect("timeout");
        let p = a.finish().expect("policy").expect("configured");
        assert_eq!(p.max_notification_bytes(), 65537);
        assert_eq!(p.write_timeout(), Duration::from_millis(1234));
    }
    #[test]
    fn missing_partner_is_rejected() {
        let mut a = TransportArgs::default();
        a.set_maximum("1").expect("bytes");
        assert!(a.finish().is_err());
        let mut a = TransportArgs::default();
        a.set_timeout("1").expect("timeout");
        assert!(a.finish().is_err());
    }
    #[test]
    fn zero_values_are_rejected() {
        assert!(TransportArgs::default().set_maximum("0").is_err());
        assert!(TransportArgs::default().set_timeout("0").is_err());
    }
    #[test]
    fn malformed_values_are_rejected() {
        for raw in ["", "-1", "+1", " 1", "1.5", "nan"] {
            assert!(TransportArgs::default().set_maximum(raw).is_err());
            assert!(TransportArgs::default().set_timeout(raw).is_err());
        }
    }
    #[test]
    fn overflowing_values_are_rejected() {
        let raw = "999999999999999999999999999999999999999999999";
        assert!(TransportArgs::default().set_maximum(raw).is_err());
        assert!(TransportArgs::default().set_timeout(raw).is_err());
    }
    #[test]
    fn duplicate_options_are_rejected_without_overwriting() {
        let mut a = TransportArgs::default();
        a.set_maximum("10").expect("first");
        assert!(a.set_maximum("20").is_err());
        a.set_timeout("30").expect("first timeout");
        assert!(a.set_timeout("40").is_err());
        let p = a.finish().expect("policy").expect("configured");
        assert_eq!(p.max_notification_bytes(), 10);
        assert_eq!(p.write_timeout(), Duration::from_millis(30));
    }
    #[test]
    fn actual_service_configuration_helper_applies_only_the_supplied_policy() {
        for configured in [false, true] {
            let directory = Directory::new();
            let telemetry = Arc::new(TelemetryClient::new(
                "test",
                HardwareTier::Unknown,
                "https://telemetry.invalid/v1",
            ));
            let service = DaemonService::open(
                directory.0.join("sessions.sqlite"),
                FixtureEngine::new("text"),
                telemetry,
                4,
                0,
                0,
            )
            .expect("service");
            let policy = if configured {
                Some(RpcTransportLimits::new(4096, Duration::from_millis(777)).expect("policy"))
            } else {
                None
            };
            let service = apply(service, policy);
            match service.rpc_transport_limits() {
                Some(p) => {
                    assert!(configured);
                    assert_eq!(p.max_notification_bytes(), 4096);
                    assert_eq!(p.write_timeout(), Duration::from_millis(777));
                }
                None => assert!(!configured),
            }
            drop(service);
        }
    }
}
