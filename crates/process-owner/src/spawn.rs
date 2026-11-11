//! Close-by-default supervised process spawn, independent of descriptor allocation limits.

#[cfg(target_os = "macos")]
pub use macos::Child;
#[cfg(target_os = "linux")]
pub struct Child {
    control: std::os::unix::net::UnixStream,
    pidfd: Option<std::os::fd::OwnedFd>,
    target_pid: u32,
    shell: bool,
    completion: Vec<u8>,
    status: Option<std::process::ExitStatus>,
}
#[cfg(target_os = "linux")]
static DEFERRED_WAITERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(target_os = "linux")]
fn defer_supervisor_wait(fd: std::os::fd::OwnedFd) -> std::io::Result<()> {
    use std::sync::atomic::Ordering;
    DEFERRED_WAITERS
        .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < 64).then_some(n + 1)
        })
        .map_err(|_| {
            std::io::Error::other("cleanup not established: deferred waiter capacity exhausted")
        })?;
    struct Permit;
    impl Drop for Permit {
        fn drop(&mut self) {
            DEFERRED_WAITERS.fetch_sub(1, Ordering::AcqRel);
        }
    }
    let permit = Permit;
    std::thread::Builder::new()
        .name("alloy-pidfd-reaper".into())
        .spawn(move || {
            use std::os::fd::AsRawFd;
            unsafe extern "C" {
                fn alloy_linux_reap_pidfd_blocking(fd: i32);
            }
            let _permit = permit;
            // SAFETY: fd holds the exact supervisor identity until the blocking wait completes.
            unsafe { alloy_linux_reap_pidfd_blocking(fd.as_raw_fd()) };
        })
        .map(|_| ())
        .map_err(|e| {
            std::io::Error::other(format!(
                "cleanup not established: deferred waiter could not start: {e}"
            ))
        })
}
#[cfg(target_os = "linux")]
impl Child {
    pub fn kill(&mut self) -> std::io::Result<()> {
        self.control.shutdown(std::net::Shutdown::Write)
    }
    pub fn id(&self) -> u32 {
        self.target_pid
    }
    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        use std::io::Read;
        use std::os::unix::process::ExitStatusExt;
        if let Some(status) = self.status {
            return Ok(Some(status));
        }
        let expected = if self.shell { 5 } else { 1 };
        self.control.set_nonblocking(true)?;
        while self.completion.len() < expected {
            let mut bytes = [0u8; 5];
            match self
                .control
                .read(&mut bytes[..expected - self.completion.len()])
            {
                Ok(0) => return Err(std::io::Error::other("owner ended without completion")),
                Ok(n) => self.completion.extend_from_slice(&bytes[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        if self.completion[0] != b'D' {
            return Err(std::io::Error::other("supervisor cleanup not established"));
        }
        let mut raw = [0u8; 4];
        if self.shell {
            raw.copy_from_slice(&self.completion[1..]);
        }
        let status = std::process::ExitStatus::from_raw(i32::from_le_bytes(raw));
        self.status = Some(status);
        Ok(Some(status))
    }
    pub fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn alloy_linux_wait_pidfd(fd: i32) -> i32;
        }
        let code = match self.pidfd.as_ref() {
            // SAFETY: descriptor pins the exact supervisor, independent of host reapers.
            Some(fd) => unsafe { alloy_linux_wait_pidfd(fd.as_raw_fd()) },
            None => return Err(std::io::Error::other("supervisor wait already deferred")),
        };
        if code != 0 {
            if let Some(fd) = self.pidfd.take() {
                defer_supervisor_wait(fd)?;
            }
            return Err(std::io::Error::from_raw_os_error(code));
        }
        self.try_wait()?
            .ok_or_else(|| std::io::Error::other("missing supervisor completion"))
    }
}
use std::process::{ChildStderr, ChildStdin, ChildStdout};
use std::{path::Path, time::Instant};

/// Executables and writable trees whose identities must be protected at launch.
#[derive(Clone, Debug)]
pub struct AuditConfig {
    /// Configured executable path.
    pub program: std::path::PathBuf,
    /// Trees writable by the supervised process.
    pub write_trees: Vec<std::path::PathBuf>,
}

pub type Spawned = (Child, ChildStdin, ChildStdout, ChildStderr);
#[cfg(target_os = "linux")]
pub fn prepare_helper(
    _configs: &[AuditConfig],
    helper: Option<&Path>,
    _deadline: Instant,
    _cancel: &dyn Fn() -> bool,
) -> std::io::Result<Option<std::path::PathBuf>> {
    Ok(helper.map(Path::to_owned))
}
#[cfg(target_os = "macos")]
pub fn prepare_helper(
    configs: &[AuditConfig],
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &dyn Fn() -> bool,
) -> std::io::Result<Option<std::path::PathBuf>> {
    if configs.is_empty() {
        return Ok(None);
    }
    macos::prepare_helper(configs, helper, deadline, cancel).map(Some)
}
/// Launch one validated command with an isolated, identity-preserving owner.
pub fn spawn(
    command: std::process::Command,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &dyn Fn() -> bool,
    external_reaper: bool,
) -> std::io::Result<Spawned> {
    platform(command, helper, deadline, cancel, external_reaper)
}

/// Launch was committed but the exec acknowledgement was lost. Effects may exist.
#[derive(Debug)]
pub struct CommittedLaunchError(pub std::io::Error);
impl std::fmt::Display for CommittedLaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "committed launch: {}", self.0)
    }
}
impl std::error::Error for CommittedLaunchError {}

/// Installed helper path; development builds also carry freshly compiled native helpers.
pub fn default_helper() -> std::io::Result<std::path::PathBuf> {
    #[cfg(all(target_os = "macos", debug_assertions))]
    {
        Ok(std::path::PathBuf::from(env!("ALLOY_DEV_HELPER")))
    }
    #[cfg(not(all(target_os = "macos", debug_assertions)))]
    {
        let exe = std::env::current_exe()?;
        Ok(exe
            .parent()
            .ok_or_else(|| std::io::Error::other("executable parent missing"))?
            .join("alloy-plugin-pipe-helper"))
    }
}

/// Launch a shell command through the same isolated owner as plugins.
pub fn spawn_shell(
    command: std::process::Command,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &dyn Fn() -> bool,
) -> std::io::Result<Spawned> {
    #[cfg(target_os = "linux")]
    {
        linux_platform(command, helper, deadline, cancel, false, true)
    }
    #[cfg(target_os = "macos")]
    {
        macos::spawn_shell(command, helper, deadline, cancel)
    }
}
#[cfg(target_os = "linux")]
fn platform(
    command: std::process::Command,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &dyn Fn() -> bool,
    external_reaper: bool,
) -> std::io::Result<Spawned> {
    linux_platform(command, helper, deadline, cancel, external_reaper, false)
}
#[cfg(target_os = "linux")]
fn linux_platform(
    cmd: std::process::Command,
    _helper: Option<&Path>,
    deadline: Instant,
    cancel: &dyn Fn() -> bool,
    _external_reaper: bool,
    shell: bool,
) -> std::io::Result<Spawned> {
    use std::{
        ffi::CString,
        io,
        os::{
            fd::{AsRawFd, FromRawFd, OwnedFd},
            unix::{ffi::OsStrExt, net::UnixStream},
        },
    };
    unsafe extern "C" {
        fn alloy_linux_launch(
            program: *const std::ffi::c_char,
            argv: *const *const std::ffi::c_char,
            cwd: *const std::ffi::c_char,
            result: *mut i32,
        ) -> i32;
        fn alloy_linux_wait_pidfd(fd: i32) -> i32;
    }
    fn cstring(value: &std::ffi::OsStr) -> io::Result<CString> {
        CString::new(value.as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in plugin argument"))
    }
    let program = cstring(cmd.get_program())?;
    let cwd = cstring(
        cmd.get_current_dir()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "plugin cwd missing"))?
            .as_os_str(),
    )?;
    let mut args = vec![program.clone()];
    for arg in cmd.get_args() {
        args.push(cstring(arg)?);
    }
    let mut argv: Vec<*const std::ffi::c_char> = args.iter().map(|a| a.as_ptr()).collect();
    argv.push(std::ptr::null());
    if cancel() {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "plugin launch cancelled",
        ));
    }
    if Instant::now() >= deadline {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "plugin launch deadline",
        ));
    }
    let mut fds = [-1; 5];
    // SAFETY: all C strings and argv remain alive; launch atomically returns
    // unique owned fds or reports an error. The child executes only C syscalls.
    let code = unsafe {
        alloy_linux_launch(
            program.as_ptr(),
            argv.as_ptr(),
            cwd.as_ptr(),
            fds.as_mut_ptr(),
        )
    };
    if fds[4] < 0 {
        return Err(io::Error::from_raw_os_error(code));
    }
    // SAFETY: successful clone returns five distinct owned descriptors.
    let input: ChildStdin = unsafe { OwnedFd::from_raw_fd(fds[0]) }.into();
    // SAFETY: the output pipe is a distinct owned descriptor from successful clone.
    let output: ChildStdout = unsafe { OwnedFd::from_raw_fd(fds[1]) }.into();
    // SAFETY: the diagnostic pipe is a distinct owned descriptor from successful clone.
    let error: ChildStderr = unsafe { OwnedFd::from_raw_fd(fds[2]) }.into();
    // SAFETY: the control socket is a distinct owned descriptor from successful clone.
    let mut parent = unsafe { UnixStream::from_raw_fd(fds[3]) };
    // SAFETY: CLONE_PIDFD returned the exact supervisor's distinct owned descriptor.
    let pidfd = unsafe { OwnedFd::from_raw_fd(fds[4]) };
    let setup = (|| -> io::Result<()> {
        use std::io::Write;
        if code != 0 {
            return Err(io::Error::from_raw_os_error(code));
        }
        parent.set_read_timeout(Some(std::time::Duration::from_secs(3)))?;
        if cancel() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "plugin launch cancelled",
            ));
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "plugin launch deadline",
            ));
        }
        if shell {
            parent.set_nonblocking(true)?;
        }
        parent.write_all(if shell { b"Q" } else { b"R" })
    })();
    if let Err(error) = setup {
        let _ = parent.shutdown(std::net::Shutdown::Write);
        // SAFETY: kernel-pinned supervisor identity; ECHILD is harmless under an external reaper.
        let waited = unsafe { alloy_linux_wait_pidfd(pidfd.as_raw_fd()) };
        if waited != 0
            && let Err(deferred) = defer_supervisor_wait(pidfd)
        {
            return Err(io::Error::other(format!("{error}; {deferred}")));
        }
        return Err(error);
    }
    let mut target_pid = 0;
    if shell {
        use std::io::Read;
        let mut response = [0u8; 5];
        let read_ack = (|| {
            let mut used = 0;
            while used < response.len() {
                if cancel() || Instant::now() >= deadline {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "committed launch cancelled or expired",
                    ));
                }
                match parent.read(&mut response[used..]) {
                    Ok(0) => {
                        return Err(std::io::Error::other(
                            "owner ended before launch acknowledgement",
                        ));
                    }
                    Ok(n) => used += n,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1))
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        })();
        if let Err(error) = read_ack {
            let _ = parent.shutdown(std::net::Shutdown::Write);
            let cleanup = defer_supervisor_wait(pidfd);
            return Err(std::io::Error::other(CommittedLaunchError(match cleanup {
                Ok(()) => error,
                Err(cleanup) => std::io::Error::other(format!("{error}; {cleanup}")),
            })));
        }
        let value = i32::from_le_bytes(response[1..].try_into().expect("four bytes"));
        if response[0] != b'S' || value <= 1 {
            let _ = parent.shutdown(std::net::Shutdown::Write);
            let cleanup = defer_supervisor_wait(pidfd);
            let error = if response[0] == b'F' && value > 0 {
                std::io::Error::from_raw_os_error(value)
            } else {
                std::io::Error::other(CommittedLaunchError(std::io::Error::other(
                    "invalid launch acknowledgement",
                )))
            };
            return Err(match cleanup {
                Ok(()) => error,
                Err(cleanup) => std::io::Error::other(CommittedLaunchError(std::io::Error::other(
                    format!("{error}; {cleanup}"),
                ))),
            });
        }
        target_pid = value as u32;
    }
    Ok((
        Child {
            control: parent,
            pidfd: Some(pidfd),
            target_pid,
            shell,
            completion: Vec::new(),
            status: None,
        },
        input,
        output,
        error,
    ))
}
#[cfg(target_os = "macos")]
fn platform(
    cmd: std::process::Command,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &dyn Fn() -> bool,
    external_reaper: bool,
) -> std::io::Result<Spawned> {
    macos::spawn(cmd, helper, deadline, cancel, external_reaper)
}

#[cfg(target_os = "macos")]
mod macos {
    use super::Spawned;
    use std::{
        ffi::{CString, c_char, c_void},
        io,
        os::{
            fd::{AsRawFd, FromRawFd, OwnedFd},
            unix::{ffi::OsStrExt, fs::MetadataExt, process::ExitStatusExt},
        },
        process::{Command, ExitStatus},
    };
    type Opaque = *mut c_void;
    const POSIX_SPAWN_SETSID: i16 = 0x0400;
    const POSIX_SPAWN_CLOEXEC_DEFAULT: i16 = 0x4000;
    #[link(name = "System")]
    unsafe extern "C" {
        fn alloy_fileport_pipes(
            launcher: *const c_char,
            helper: *const c_char,
            stop: extern "C" fn(*mut c_void) -> i32,
            context: *mut c_void,
            reap_owner: i32,
            out: *mut i32,
        ) -> i32;
        fn alloy_owned_stop(control: Opaque) -> i32;
        fn alloy_owned_wait(control: Opaque, status: *mut i32, nohang: i32) -> i32;
        fn alloy_owned_close(control: Opaque);
        fn posix_spawnattr_init(attr: *mut Opaque) -> i32;
        fn posix_spawnattr_destroy(attr: *mut Opaque) -> i32;
        fn posix_spawnattr_setflags(attr: *mut Opaque, flags: i16) -> i32;
        fn posix_spawn_file_actions_init(actions: *mut Opaque) -> i32;
        fn posix_spawn_file_actions_destroy(actions: *mut Opaque) -> i32;
        fn posix_spawn_file_actions_adddup2(actions: *mut Opaque, fd: i32, target: i32) -> i32;
        fn posix_spawn_file_actions_addclose(actions: *mut Opaque, fd: i32) -> i32;
        fn posix_spawn_file_actions_addchdir_np(actions: *mut Opaque, path: *const c_char) -> i32;
        fn alloy_owned_spawn(
            launcher: *const c_char,
            path: *const c_char,
            actions: *mut Opaque,
            attr: *mut Opaque,
            argv: *const *mut c_char,
            envp: *const *mut c_char,
            stop: extern "C" fn(*mut c_void) -> i32,
            context: *mut c_void,
            reap_owner: i32,
            pid: *mut i32,
            identity: *mut u64,
            control: *mut Opaque,
        ) -> i32;
    }
    fn checked(code: i32) -> io::Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(code))
        }
    }
    fn cstring(value: &std::ffi::OsStr) -> io::Result<CString> {
        CString::new(value.as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in spawn configuration"))
    }
    pub(super) fn prepare_helper(
        configs: &[super::AuditConfig],
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &dyn Fn() -> bool,
    ) -> io::Result<std::path::PathBuf> {
        let helper = match helper {
            Some(path) if path.is_absolute() => path.to_owned(),
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "helper path must be absolute",
                ));
            }
            None => std::env::current_exe()?
                .parent()
                .ok_or_else(|| io::Error::other("executable has no parent"))?
                .join("alloy-plugin-pipe-helper"),
        }
        .canonicalize()?;
        if !helper.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid pipe helper",
            ));
        }
        let launcher = launcher_path(&helper)?;
        let mut protected = vec![helper.clone(), launcher];
        for config in configs {
            protected.push(config.program.canonicalize()?);
        }
        let mut visited = 0usize;
        for config in configs {
            for path in &config.write_trees {
                let root = path.canonicalize()?;
                if protected.iter().any(|p| p.starts_with(&root)) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "plugin writable tree includes a protected executable",
                    ));
                }
                if crate::macos_service::protected_registry_path().starts_with(&root) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "plugin writable tree includes the broker registry",
                    ));
                }
                let mut pending = vec![root];
                while let Some(directory) = pending.pop() {
                    if cancel() || std::time::Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "plugin writable-tree audit cancelled or expired",
                        ));
                    }
                    for entry in std::fs::read_dir(directory)? {
                        if cancel() || std::time::Instant::now() >= deadline {
                            return Err(io::Error::new(
                                io::ErrorKind::Interrupted,
                                "plugin writable-tree audit cancelled or expired",
                            ));
                        }
                        visited += 1;
                        if visited > 16384 {
                            return Err(io::Error::other(
                                "plugin writable-tree audit exceeds entry limit",
                            ));
                        }
                        let path = entry?.path();
                        let metadata = std::fs::symlink_metadata(&path)?;
                        if metadata.file_type().is_symlink() {
                            continue;
                        }
                        if metadata.is_dir() {
                            pending.push(path);
                        } else if metadata.is_file() && metadata.nlink() != 1 {
                            return Err(io::Error::new(
                                io::ErrorKind::PermissionDenied,
                                "plugin writable tree contains a hard-linked file",
                            ));
                        }
                    }
                }
            }
        }
        Ok(helper)
    }
    struct Attr(Opaque);
    impl Attr {
        fn new() -> io::Result<Self> {
            let mut value = std::ptr::null_mut();
            // SAFETY: init receives a valid pointer to an uninitialized opaque handle.
            checked(unsafe { posix_spawnattr_init(&mut value) })?;
            Ok(Self(value))
        }
    }
    impl Drop for Attr {
        fn drop(&mut self) {
            // SAFETY: this owns one successfully initialized opaque attribute handle.
            unsafe {
                posix_spawnattr_destroy(&mut self.0);
            }
        }
    }
    struct Actions(Opaque);
    impl Actions {
        fn new() -> io::Result<Self> {
            let mut value = std::ptr::null_mut();
            // SAFETY: init receives a valid pointer to an uninitialized opaque handle.
            checked(unsafe { posix_spawn_file_actions_init(&mut value) })?;
            Ok(Self(value))
        }
    }
    impl Drop for Actions {
        fn drop(&mut self) {
            // SAFETY: this owns one successfully initialized opaque file-actions handle.
            unsafe {
                posix_spawn_file_actions_destroy(&mut self.0);
            }
        }
    }
    fn pipes(
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &dyn Fn() -> bool,
        external_reaper: bool,
    ) -> io::Result<[OwnedFd; 6]> {
        let selected = match helper {
            Some(path) if path.is_absolute() => path.to_owned(),
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "helper path must be absolute",
                ));
            }
            None => std::env::current_exe()?
                .parent()
                .ok_or_else(|| io::Error::other("executable has no parent"))?
                .join("alloy-plugin-pipe-helper"),
        };
        let selected = selected.canonicalize()?;
        if !selected.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid pipe helper",
            ));
        }
        let selected_path = selected;
        let selected = cstring(selected_path.as_os_str())?;
        let launcher = cstring(launcher_path(&selected_path)?.as_os_str())?;
        let mut state = Stop { deadline, cancel };
        let mut raw = [-1; 6];
        // SAFETY: selected is NUL-terminated; callback/context and six output slots remain valid.
        let code = unsafe {
            alloy_fileport_pipes(
                launcher.as_ptr(),
                selected.as_ptr(),
                stopped,
                (&mut state as *mut Stop<'_>).cast(),
                i32::from(!external_reaper),
                raw.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(io::Error::from_raw_os_error(code));
        }
        // SAFETY: the C function returns six distinct, owned CLOEXEC descriptors on success.
        Ok(raw.map(|fd| unsafe { OwnedFd::from_raw_fd(fd) }))
    }
    fn launcher_path(helper: &std::path::Path) -> io::Result<std::path::PathBuf> {
        let path = helper
            .with_file_name("alloy-plugin-exec-helper")
            .canonicalize()?;
        if !path.is_file() {
            return Err(io::Error::other("invalid exec helper"));
        }
        Ok(path)
    }
    struct Stop<'a> {
        deadline: std::time::Instant,
        cancel: &'a dyn Fn() -> bool,
    }
    extern "C" fn stopped(context: *mut c_void) -> i32 {
        // SAFETY: the synchronous C call retains this live stack value only for its duration.
        let state = unsafe { &*(context.cast::<Stop<'_>>()) };
        i32::from((state.cancel)() || std::time::Instant::now() >= state.deadline)
    }
    pub struct Child {
        control: Opaque,
        status: Option<ExitStatus>,
        target_pid: u32,
    }
    impl Child {
        pub fn id(&self) -> u32 {
            self.target_pid
        }
        pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
            if let Some(status) = self.status {
                return Ok(Some(status));
            }
            let mut raw = 0;
            // SAFETY: the owned native record is valid through Drop.
            let code = unsafe { alloy_owned_wait(self.control, &mut raw, 1) };
            if code == 35 {
                return Ok(None);
            } // Darwin EAGAIN
            checked(code)?;
            let status = ExitStatus::from_raw(raw);
            self.status = Some(status);
            Ok(Some(status))
        }
        pub fn kill(&mut self) -> io::Result<()> {
            // SAFETY: this uniquely owned record remains valid until Drop.
            checked(unsafe { alloy_owned_stop(self.control) })
        }
        pub fn wait(&mut self) -> io::Result<ExitStatus> {
            if let Some(status) = self.status {
                return Ok(status);
            }
            let mut raw = 0;
            // SAFETY: control owns a live native supervisor record; raw is writable.
            checked(unsafe { alloy_owned_wait(self.control, &mut raw, 0) })?;
            let status = ExitStatus::from_raw(raw);
            self.status = Some(status);
            Ok(status)
        }
    }
    // SAFETY: the uniquely owned native record is accessed only through &mut self.
    unsafe impl Send for Child {}
    impl Drop for Child {
        fn drop(&mut self) {
            // SAFETY: exactly one Child owns this native record.
            unsafe { alloy_owned_close(self.control) };
        }
    }
    pub(super) fn spawn_shell(
        cmd: Command,
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &dyn Fn() -> bool,
    ) -> io::Result<Spawned> {
        unsafe extern "C" {
            fn alloy_shell_spawn(
                launcher: *const c_char,
                service: *const c_char,
                program: *const c_char,
                cwd: *const c_char,
                argv: *const *mut c_char,
                stop: unsafe extern "C" fn(*mut c_void) -> i32,
                context: *mut c_void,
                pid: *mut i32,
                control: *mut Opaque,
                fds: *mut i32,
            ) -> i32;
        }
        let program = cstring(cmd.get_program())?;
        let cwd = cstring(
            cmd.get_current_dir()
                .ok_or_else(|| io::Error::other("shell cwd missing"))?
                .as_os_str(),
        )?;
        let launcher_path =
            launcher_path(helper.ok_or_else(|| io::Error::other("missing helper"))?)?;
        let service = crate::macos_service::prepare(&launcher_path, || {
            cancel() || std::time::Instant::now() >= deadline
        })?;
        let service = cstring(std::ffi::OsStr::new(&service))?;
        let launcher = cstring(launcher_path.as_os_str())?;
        let mut args = vec![program.clone()];
        for arg in cmd.get_args() {
            args.push(cstring(arg)?);
        }
        let mut argv: Vec<*mut c_char> = args.iter().map(|arg| arg.as_ptr().cast_mut()).collect();
        argv.push(std::ptr::null_mut());
        let mut state = Stop { deadline, cancel };
        let mut pid = 0;
        let mut control = std::ptr::null_mut();
        let mut fds = [-1; 3];
        // SAFETY: null-terminated C strings and callback state remain live for this synchronous call.
        checked(unsafe {
            alloy_shell_spawn(
                launcher.as_ptr(),
                service.as_ptr(),
                program.as_ptr(),
                cwd.as_ptr(),
                argv.as_ptr(),
                stopped,
                (&mut state as *mut Stop<'_>).cast(),
                &mut pid,
                &mut control,
                fds.as_mut_ptr(),
            )
        })?;
        // SAFETY: success transfers three distinct, close-on-exec descriptors and one native control record.
        let owned = unsafe { fds.map(|fd| OwnedFd::from_raw_fd(fd)) };
        let [input, output, error] = owned;
        Ok((
            Child {
                control,
                status: None,
                target_pid: pid as u32,
            },
            input.into(),
            output.into(),
            error.into(),
        ))
    }
    pub(super) fn spawn(
        cmd: Command,
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &dyn Fn() -> bool,
        external_reaper: bool,
    ) -> io::Result<Spawned> {
        spawn_impl(cmd, helper, deadline, cancel, external_reaper, false)
    }
    pub(super) fn spawn_impl(
        cmd: Command,
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &dyn Fn() -> bool,
        external_reaper: bool,
        shell: bool,
    ) -> io::Result<Spawned> {
        let program = cstring(cmd.get_program())?;
        let mut args = vec![program.clone()];
        for arg in cmd.get_args() {
            args.push(cstring(arg)?);
        }
        let cwd = cmd
            .get_current_dir()
            .map(|p| cstring(p.as_os_str()))
            .transpose()?;
        let mut argv: Vec<*mut c_char> = args.iter().map(|a| a.as_ptr().cast_mut()).collect();
        argv.push(std::ptr::null_mut());
        // sandbox::command requires an empty environment; never inherit environ.
        let environment = [std::ptr::null_mut::<c_char>()];
        let [
            input_read,
            input_write,
            output_read,
            output_write,
            error_read,
            error_write,
        ] = pipes(helper, deadline, cancel, external_reaper)?;
        let mut attr = Attr::new()?;
        let mut actions = Actions::new()?;
        // SAFETY: opaque handles are initialized, C strings/FDs remain live through
        // spawn, and every C return is checked before continuing. Only 0/1/2 are
        // inherited. Explicit source-end closes avoid duplicate pipe capabilities.
        unsafe {
            checked(posix_spawnattr_setflags(
                &mut attr.0,
                POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSID,
            ))?;
            for (source, target) in [
                (input_read.as_raw_fd(), 0),
                (output_write.as_raw_fd(), 1),
                (error_write.as_raw_fd(), 2),
            ] {
                if shell && target == 0 {
                    unsafe extern "C" {
                        fn posix_spawn_file_actions_addopen(
                            actions: *mut Opaque,
                            fd: i32,
                            path: *const c_char,
                            flags: i32,
                            mode: u16,
                        ) -> i32;
                    }
                    checked(posix_spawn_file_actions_addopen(
                        &mut actions.0,
                        0,
                        c"/dev/null".as_ptr(),
                        0,
                        0,
                    ))?;
                } else {
                    checked(posix_spawn_file_actions_adddup2(
                        &mut actions.0,
                        source,
                        target,
                    ))?;
                }
            }
            for fd in [
                input_read.as_raw_fd(),
                input_write.as_raw_fd(),
                output_read.as_raw_fd(),
                output_write.as_raw_fd(),
                error_read.as_raw_fd(),
                error_write.as_raw_fd(),
            ] {
                checked(posix_spawn_file_actions_addclose(&mut actions.0, fd))?;
            }
            if let Some(cwd) = &cwd {
                checked(posix_spawn_file_actions_addchdir_np(
                    &mut actions.0,
                    cwd.as_ptr(),
                ))?;
            }
        }
        let launcher = cstring(
            launcher_path(helper.ok_or_else(|| io::Error::other("missing helper"))?)?.as_os_str(),
        )?;
        let mut state = Stop { deadline, cancel };
        let mut identity = 0;
        let mut control = std::ptr::null_mut();
        let mut pid = 0;
        // SAFETY: argv/envp are null-terminated and all referenced storage and
        // initialized spawn objects remain valid until the call returns.
        checked(unsafe {
            alloy_owned_spawn(
                launcher.as_ptr(),
                program.as_ptr(),
                &mut actions.0,
                &mut attr.0,
                argv.as_ptr(),
                environment.as_ptr(),
                stopped,
                (&mut state as *mut Stop<'_>).cast(),
                i32::from(!external_reaper),
                &mut pid,
                &mut identity,
                &mut control,
            )
        })?;
        let child = Child {
            control,
            status: None,
            target_pid: pid as u32,
        };
        drop((input_read, output_write, error_write));
        Ok((
            child,
            input_write.into(),
            output_read.into(),
            error_read.into(),
        ))
    }
}

#[cfg(all(test, target_os = "linux"))]
mod linux_supervisor_tests {
    use super::Child;
    use std::{
        os::fd::{FromRawFd, OwnedFd},
        os::unix::net::UnixStream,
        process::Command,
        time::{Duration, Instant},
    };
    #[test]
    fn partial_completion_frame_does_not_block_or_lose_status() {
        use std::{io::Write, os::unix::process::ExitStatusExt};
        let (control, mut peer) = UnixStream::pair().unwrap();
        let mut owned = Child {
            control,
            pidfd: None,
            target_pid: 2,
            shell: true,
            completion: vec![],
            status: None,
        };
        peer.write_all(b"D").unwrap();
        let before = Instant::now();
        assert!(owned.try_wait().unwrap().is_none());
        assert!(before.elapsed() < Duration::from_millis(50));
        peer.write_all(&1792i32.to_le_bytes()[..2]).unwrap();
        assert!(owned.try_wait().unwrap().is_none());
        peer.write_all(&1792i32.to_le_bytes()[2..]).unwrap();
        assert_eq!(owned.try_wait().unwrap().unwrap().into_raw(), 1792);
    }
    #[test]
    fn deferred_wait_reaps_exact_supervisor_after_timeout() {
        unsafe extern "C" {
            fn syscall(number: isize, ...) -> isize;
        }
        let child = Command::new("/bin/sleep").arg("4").spawn().unwrap();
        let pid = child.id();
        // SAFETY: this isolated test opens the still-live child created above.
        let fd = unsafe { syscall(434, pid, 0) } as i32;
        assert!(fd >= 0, "pidfd_open: {}", std::io::Error::last_os_error());
        // SAFETY: pidfd_open returned a new descriptor.
        let pidfd = unsafe { OwnedFd::from_raw_fd(fd) };
        let (control, _peer) = UnixStream::pair().unwrap();
        let mut owned = Child {
            control,
            pidfd: Some(pidfd),
            target_pid: pid,
            shell: false,
            completion: Vec::new(),
            status: None,
        };
        assert!(
            owned.wait().is_err(),
            "a live child must not report completed cleanup"
        );
        drop(owned);
        drop(child);
        let until = Instant::now() + Duration::from_secs(5);
        while std::path::Path::new(&format!("/proc/{pid}")).exists() {
            assert!(Instant::now() < until, "deferred waiter did not reap child");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod shell_broker_tests {
    use super::*;
    use std::{
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };
    #[test]
    fn launchd_service_runs_native_child() {
        let helper = default_helper().unwrap();
        let mut command = Command::new("/usr/bin/true");
        command.current_dir("/private/tmp");
        let (mut child, input, _, _) = spawn_shell(
            command,
            Some(&helper),
            Instant::now() + Duration::from_secs(5),
            &|| false,
        )
        .unwrap();
        drop(input);
        assert!(child.wait().unwrap().success());
    }
    #[test]
    fn cancelled_launch_does_not_execute_late_or_disable_next_launch() {
        let helper = default_helper().unwrap();
        let root = std::env::temp_dir().join(format!("alloy-broker-cancel-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let mut bootstrap = Command::new("/usr/bin/true");
        bootstrap.current_dir(&root);
        assert!(
            spawn_shell(
                bootstrap,
                Some(&helper),
                Instant::now() + Duration::from_secs(5),
                &|| true
            )
            .is_err()
        );
        std::thread::sleep(Duration::from_millis(5100));
        let mut cancelled = 0;
        for limit in [1, 2, 3, 5, 8, 13, 21, 34] {
            let marker = root.join(format!("marker-{limit}"));
            let mut cmd = Command::new("/bin/sh");
            cmd.args(["-c", &format!("printf executed > {}", marker.display())])
                .current_dir(&root);
            let calls = AtomicUsize::new(0);
            let result = spawn_shell(
                cmd,
                Some(&helper),
                Instant::now() + Duration::from_secs(5),
                &|| calls.fetch_add(1, Ordering::Relaxed) >= limit,
            );
            match result {
                Ok((mut child, input, _, _)) => {
                    drop(input);
                    child.wait().unwrap();
                }
                Err(_) => {
                    cancelled += 1;
                    std::thread::sleep(Duration::from_millis(30));
                    assert!(!marker.exists(), "cancelled launch executed later");
                }
            }
            let mut next = Command::new("/usr/bin/true");
            next.current_dir(&root);
            let (mut child, input, _, _) = spawn_shell(
                next,
                Some(&helper),
                Instant::now() + Duration::from_secs(5),
                &|| false,
            )
            .unwrap();
            drop(input);
            assert!(child.wait().unwrap().success());
        }
        assert!(cancelled > 0);
        std::fs::remove_dir_all(root).unwrap();
    }
}
