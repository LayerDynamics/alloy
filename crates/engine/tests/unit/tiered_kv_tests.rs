use super::*;
#[test]
fn storage_owner_prevents_live_cleanup_and_restart_removes_abandoned_writes() {
    let root = std::env::temp_dir().join(format!(
        "alloy-kv-owner-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let _cleanup = TemporaryDirectory(Some(root.clone()));
    let admission = AdmissionController::new(32 * 1024 * 1024).unwrap();
    let mut config = TieredKvConfig::new(Some(root.clone()), HotCacheLimit::Percent(25));
    config.transfer_bytes = 9 * 1024 * 1024;
    let mut owner = TieredKvStore::open(config.clone(), &admission).unwrap();
    let abandoned = root.join(format!(".tmp-kv-v1-{}", "a".repeat(64)));
    fs::create_dir(&abandoned).unwrap();
    fs::write(abandoned.join("partial.safetensors"), [1, 2, 3]).unwrap();
    assert!(TieredKvStore::open(config.clone(), &admission).is_err());
    assert!(
        abandoned.exists(),
        "another live owner cannot remove a writer's staging directory"
    );
    assert_eq!(
        admission.snapshot().unwrap().reserved_bytes(),
        config.transfer_bytes
    );
    config.hot_limit = HotCacheLimit::Bytes(0);
    assert!(owner.update_limits(&config).unwrap());
    assert_eq!(
        admission.snapshot().unwrap().reserved_bytes(),
        config.transfer_bytes
    );
    drop(owner);
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
    let unrelated = root.join("caller-owned-directory");
    fs::create_dir(&unrelated).unwrap();
    let incomplete = root.join(format!("kv-v1-{}", "b".repeat(64)));
    fs::create_dir(&incomplete).unwrap();
    let reopened = TieredKvStore::open(config, &admission).unwrap();
    assert!(!abandoned.exists());
    assert!(!incomplete.exists());
    assert!(unrelated.exists());
    drop(reopened);
    assert_eq!(admission.snapshot().unwrap().reserved_bytes(), 0);
}

#[test]
fn hot_limit_parses_and_tracks_live_budget() {
    assert_eq!(HotCacheLimit::parse("25%").unwrap().bytes(400), 100);
    assert_eq!(
        HotCacheLimit::parse("2MiB").unwrap(),
        HotCacheLimit::Bytes(2 << 20)
    );
    assert_eq!(HotCacheLimit::parse("0").unwrap().bytes(100), 0);
    for bad in ["-1", "101%", "NaN", "18446744073709551615GiB", "2.5%"] {
        assert!(HotCacheLimit::parse(bad).is_err(), "{bad}");
    }
}
#[test]
fn prefix_identity_binds_tokens_and_model() {
    assert_ne!(token_digest("a", &[1, 2]), token_digest("a", &[1, 3]));
    assert_ne!(token_digest("a", &[1, 2]), token_digest("b", &[1, 2]));
    assert!(!valid_entry_name("kv-v1-../../outside"));
}
