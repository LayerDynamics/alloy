//! Per-host launchd registration through the user-domain ServiceManagement API.
//!
//! The native service authenticates the host and gives each restart a fresh
//! generation. Registration is idempotent and retained across caller cancellation.
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::Mutex,
};

struct Registration {
    launcher: PathBuf,
    name: String,
    definition: Vec<u8>,
    bootstrapped: bool,
}

static REGISTRATION: Mutex<Option<Registration>> = Mutex::new(None);

pub(crate) fn protected_registry_path() -> PathBuf {
    unsafe extern "C" {
        fn getuid() -> u32;
    }
    // SAFETY: getuid takes no pointers and only observes the current identity.
    PathBuf::from(format!("/private/tmp/alloy-brokers-{}", unsafe {
        getuid()
    }))
}

fn registered(name: &str) -> io::Result<bool> {
    unsafe extern "C" {
        fn alloy_broker_registered(name: *const std::ffi::c_char) -> i32;
    }
    let name = std::ffi::CString::new(name)
        .map_err(|_| io::Error::other("invalid broker service name"))?;
    // SAFETY: the synchronous call borrows this NUL-terminated name.
    match unsafe { alloy_broker_registered(name.as_ptr()) } {
        0 => Ok(false),
        1 => Ok(true),
        code => Err(io::Error::new(
            io::Error::from_raw_os_error(-code).kind(),
            format!("broker registry job lookup failed: OS error {}", -code),
        )),
    }
}

fn service_job(name: &str, definition: Option<&[u8]>) -> io::Result<()> {
    unsafe extern "C" {
        fn alloy_launchd_job(
            name: *const std::ffi::c_char,
            plist: *const std::ffi::c_void,
            size: usize,
            diagnostic: *mut std::ffi::c_char,
            capacity: usize,
        ) -> i32;
    }
    let name = std::ffi::CString::new(name)
        .map_err(|_| io::Error::other("invalid broker service name"))?;
    let (bytes, size) = definition.map_or((std::ptr::null(), 0), |bytes| {
        (bytes.as_ptr().cast(), bytes.len())
    });
    let mut diagnostic = [0u8; 2048];
    // SAFETY: all inputs and the writable output buffer remain valid for this
    // synchronous user-domain operation; no authorization or root domain is used.
    let code = unsafe {
        alloy_launchd_job(
            name.as_ptr(),
            bytes,
            size,
            diagnostic.as_mut_ptr().cast(),
            diagnostic.len(),
        )
    };
    if code == 0 {
        return Ok(());
    }
    let message = std::ffi::CStr::from_bytes_until_nul(&diagnostic)
        .unwrap_or(c"ServiceManagement failed")
        .to_string_lossy();
    Err(io::Error::new(
        io::Error::from_raw_os_error(code).kind(),
        format!("broker service: {message}"),
    ))
}

fn registry(stopped: &impl Fn() -> bool) -> io::Result<(PathBuf, fs::File)> {
    unsafe extern "C" {
        fn getuid() -> u32;
        fn alloy_owned_alive(pid: i32, identity: u64) -> i32;
    }
    // SAFETY: getuid does not borrow any memory.
    let uid = unsafe { getuid() };
    // Ignore caller-controlled TMPDIR: all embedding hosts for this user share
    // the same finite registration budget, including inactive crashed services.
    let root = protected_registry_path();
    match fs::DirBuilder::new().mode(0o700).create(&root) {
        Ok(()) => (),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error),
    }
    let metadata = fs::symlink_metadata(&root)?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.permissions().mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "unsafe broker registry",
        ));
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(0x100 /* O_NOFOLLOW */)
        .open(root.join("registry.lock"))?;
    let metadata = lock.metadata()?;
    if !metadata.is_file() || metadata.uid() != uid || metadata.permissions().mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "unsafe broker registry lock",
        ));
    }
    unsafe extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    loop {
        // SAFETY: the File owns this descriptor until after registration publication.
        if unsafe {
            flock(lock.as_raw_fd(), 2 | 4 /* LOCK_EX | LOCK_NB */)
        } == 0
        {
            break;
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::WouldBlock {
            return Err(error);
        }
        if stopped() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "broker registry interrupted",
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let mut retained = 0;
    for entry in fs::read_dir(&root)? {
        if stopped() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "broker registry interrupted",
            ));
        }
        let entry = entry?;
        if entry.file_name() == "registry.lock" {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path())?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| io::Error::other("invalid broker registration name"))?;
        if !metadata.is_dir()
            || metadata.uid() != uid
            || metadata.permissions().mode() & 0o077 != 0
            || !name.starts_with(&format!("com.layerdynamics.alloy.broker.{uid}."))
            || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '.')
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "unsafe broker registration",
            ));
        }
        // Identity is in the atomically created directory name as well as its
        // record, so a creator crash during file publication is collectible.
        let fields: Vec<_> = name.split('.').collect();
        if fields.len() != 8 {
            return Err(io::Error::other("invalid broker owner name"));
        }
        let pid: i32 = fields[5]
            .parse()
            .map_err(|_| io::Error::other("invalid broker owner PID"))?;
        let identity: u64 = fields[6]
            .parse()
            .map_err(|_| io::Error::other("invalid broker owner identity"))?;
        if pid <= 0 || identity == 0 {
            return Err(io::Error::other("invalid broker owner identity"));
        }
        // SAFETY: this reads stable kernel identity and never signals a PID.
        let alive = unsafe { alloy_owned_alive(pid, identity) };
        // Inconclusive identity is not proof of death. Preserve this service
        // and charge the registration budget without blocking other hosts
        // while there is capacity. Never remove an unverified live owner.
        if alive != 0 {
            retained += 1;
            continue;
        }
        unsafe extern "C" {
            fn alloy_cohort_records_retire(directory: *const std::ffi::c_char) -> i32;
        }
        use std::os::unix::ffi::OsStrExt;
        let directory = std::ffi::CString::new(entry.path().as_os_str().as_bytes())
            .map_err(|_| io::Error::other("invalid cohort record directory"))?;
        // SAFETY: the NUL-terminated directory remains live; the registry lock
        // is held, and this host's original stable identity was proven dead.
        let cleanup = unsafe { alloy_cohort_records_retire(directory.as_ptr()) };
        if cleanup != 0 {
            // Unconfirmed cleanup remains charged; it does not prevent a new
            // host while the bounded shared registration budget has capacity.
            retained += 1;
            continue;
        }
        if registered(&name)? {
            let removal = service_job(&name, None);
            if let Err(error) = removal
                && registered(&name)?
            {
                return Err(error);
            }
            let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while registered(&name)? {
                if stopped() || std::time::Instant::now() >= until {
                    return Err(io::Error::other("stale broker retirement incomplete"));
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        // Exact owned files only. Unexpected contents are preserved as an error.
        for file in ["service.plist", "owner"] {
            match fs::remove_file(entry.path().join(file)) {
                Ok(()) => (),
                Err(error) if error.kind() == io::ErrorKind::NotFound => (),
                Err(error) => return Err(error),
            }
        }
        fs::remove_dir(entry.path())?;
    }
    if retained >= 64 {
        return Err(io::Error::other("broker registration capacity exhausted"));
    }
    Ok((root, lock))
}

fn xml(value: &str) -> io::Result<String> {
    if value.chars().any(char::is_control) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "control character in broker configuration",
        ));
    }
    Ok(value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;"))
}

fn plist(
    launcher: &Path,
    name: &str,
    directory: &Path,
    parent: u32,
    identity: u64,
) -> io::Result<String> {
    let path = |p: &Path| {
        p.to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "broker path is not UTF-8"))
            .and_then(xml)
    };
    let arguments = [
        "/usr/bin/env".into(),
        "-i".into(),
        path(launcher)?,
        "--alloy-shell-service-v2".into(),
        path(launcher)?,
        parent.to_string(),
        identity.to_string(),
        xml(name)?,
        path(directory)?,
    ];
    let arguments = arguments
        .iter()
        .map(|a| format!("<string>{a}</string>"))
        .collect::<String>();
    Ok(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict>\n<key>Label</key><string>{name}</string>\n<key>ProgramArguments</key><array>{arguments}</array>\n<key>LimitLoadToSessionType</key><array><string>Aqua</string><string>Background</string></array>\n<key>MachServices</key><dict><key>{name}</key><dict><key>ResetAtClose</key><true/></dict></dict>\n<key>KeepAlive</key><false/>\n<key>RunAtLoad</key><true/>\n<key>ThrottleInterval</key><integer>2</integer>\n<key>ProcessType</key><string>Standard</string>\n</dict></plist>\n"
    ))
}

fn create(launcher: &Path, root: &Path) -> io::Result<Registration> {
    unsafe extern "C" {
        fn arc4random() -> u32;
        fn getuid() -> u32;
        fn alloy_process_identity(pid: i32) -> u64;
    }
    let parent = std::process::id();
    // SAFETY: these functions take no pointers and read the current process identity.
    let (random, uid, identity) = unsafe {
        (
            ((arc4random() as u64) << 32) | arc4random() as u64,
            getuid(),
            alloy_process_identity(parent as i32),
        )
    };
    if identity == 0 {
        return Err(io::Error::other("broker host identity unavailable"));
    }
    let name = format!("com.layerdynamics.alloy.broker.{uid}.{parent}.{identity}.{random:016x}");
    let directory = root.join(&name);
    let definition = plist(launcher, &name, &directory, parent, identity)?.into_bytes();
    fs::DirBuilder::new().mode(0o700).create(&directory)?;
    let file = directory.join("service.plist");
    let result = (|| {
        let mut owner = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.join("owner"))?;
        writeln!(owner, "{parent}\n{identity}\n{name}\nservice-management")?;
        owner.sync_all()?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&file)?;
        file.write_all(&definition)?;
        file.sync_all()
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&file);
        let _ = fs::remove_file(directory.join("owner"));
        let _ = fs::remove_dir(&directory);
        return Err(error);
    }
    Ok(Registration {
        launcher: launcher.into(),
        name,
        definition,
        bootstrapped: false,
    })
}

/// Submit the in-memory definition through the public ServiceManagement API.
/// A cancelled registration may have succeeded; inspect its launchd job record
/// before resubmission. No workload or administrative subprocess is replayed.
pub(crate) fn prepare(launcher: &Path, stopped: impl Fn() -> bool) -> io::Result<String> {
    let mut guard = loop {
        match REGISTRATION.try_lock() {
            Ok(guard) => break guard,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(io::Error::other("broker registration lock poisoned"));
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                if stopped() {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "broker registration interrupted",
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    };
    if guard.is_none() {
        let (root, _registry_lock) = registry(&stopped)?;
        *guard = Some(create(launcher, &root)?);
    }
    let record = guard.as_mut().expect("created registration");
    if record.launcher != launcher {
        return Err(io::Error::other(
            "broker helper changed within a host lifetime",
        ));
    }
    if stopped() {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "broker registration interrupted",
        ));
    }
    if record.bootstrapped && !registered(&record.name)? {
        record.bootstrapped = false;
    }
    if !record.bootstrapped {
        if !registered(&record.name)? {
            service_job(&record.name, Some(&record.definition))?;
        }
        let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !registered(&record.name)? {
            if stopped() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "broker registration interrupted",
                ));
            }
            if std::time::Instant::now() >= until {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "broker registration not advertised",
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        record.bootstrapped = true;
    }
    Ok(record.name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plugin_write_grant_cannot_cover_broker_registration() {
        let root = protected_registry_path();
        match fs::DirBuilder::new().mode(0o700).create(&root) {
            Ok(()) => (),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
            Err(error) => panic!("registry setup failed: {error}"),
        }
        let helper = crate::spawn::default_helper().unwrap();
        let config = crate::spawn::AuditConfig {
            program: "/usr/bin/true".into(),
            write_trees: vec![root],
        };
        let error = crate::spawn::prepare_helper(
            &[config],
            Some(&helper),
            std::time::Instant::now() + std::time::Duration::from_secs(3),
            &|| false,
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("broker registry"));
    }
    #[test]
    fn service_definition_preserves_literal_paths_and_restart_boundary() {
        let text = plist(
            Path::new("/a&b/<helper>"),
            "com.example.test",
            Path::new("/tmp/a'b"),
            42,
            123,
        )
        .unwrap();
        assert!(text.contains("/a&amp;b/&lt;helper&gt;"));
        assert!(text.contains("/tmp/a&apos;b"));
        assert!(text.contains("<key>ResetAtClose</key><true/>"));
        assert!(text.contains("--alloy-shell-service-v2"));
        assert!(
            plist(
                Path::new("/bad\npath"),
                "com.example.test",
                Path::new("/tmp/x"),
                42,
                123
            )
            .is_err()
        );
    }
}
