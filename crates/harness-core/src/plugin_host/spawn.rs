//! Close-by-default plugin spawn, independent of descriptor allocation limits.
use super::{HostError, PluginConfig, sandbox};
#[cfg(target_os = "macos")]
pub(super) use macos::Child;
#[cfg(target_os = "linux")]
pub(super) use std::process::Child;
use std::process::{ChildStderr, ChildStdin, ChildStdout};
use std::{path::Path, time::Instant};
use tool_registry::execution::CancellationToken;
type Spawned = (Child, ChildStdin, ChildStdout, ChildStderr);
#[cfg(target_os = "linux")]
pub(super) fn prepare_helper(
    _configs: &[PluginConfig],
    helper: Option<&Path>,
    _deadline: Instant,
    _cancel: &CancellationToken,
) -> Result<Option<std::path::PathBuf>, HostError> {
    Ok(helper.map(Path::to_owned))
}
#[cfg(target_os = "macos")]
pub(super) fn prepare_helper(
    configs: &[PluginConfig],
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Result<Option<std::path::PathBuf>, HostError> {
    if configs.is_empty() {
        return Ok(None);
    }
    macos::prepare_helper(configs, helper, deadline, cancel)
        .map(Some)
        .map_err(HostError::io)
}
pub(super) fn spawn(
    config: &PluginConfig,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &CancellationToken,
    external_reaper: bool,
) -> Result<Spawned, HostError> {
    let command = sandbox::command(config)?;
    platform(command, helper, deadline, cancel, external_reaper).map_err(HostError::io)
}
#[cfg(target_os = "linux")]
fn platform(
    mut cmd: std::process::Command,
    _helper: Option<&Path>,
    _deadline: Instant,
    _cancel: &CancellationToken,
    _external_reaper: bool,
) -> std::io::Result<Spawned> {
    use std::{io, os::unix::process::CommandExt, process::Stdio};
    unsafe extern "C" {
        fn close_range(first: u32, last: u32, flags: i32) -> i32;
    }
    const CLOSE_RANGE_CLOEXEC: i32 = 1 << 2;
    // SAFETY: child callback invokes only close_range and errno observation;
    // CLOEXEC keeps Rust's exec-error pipe open until exec. No allocation limit
    // is used, so inherited descriptors above a lowered RLIMIT are covered.
    unsafe {
        cmd.pre_exec(|| {
            if close_range(3, u32::MAX, CLOSE_RANGE_CLOEXEC) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()?;
    let input = child.stdin.take().expect("configured piped stdin");
    let output = child.stdout.take().expect("configured piped stdout");
    let error = child.stderr.take().expect("configured piped stderr");
    Ok((child, input, output, error))
}
#[cfg(target_os = "macos")]
fn platform(
    cmd: std::process::Command,
    helper: Option<&Path>,
    deadline: Instant,
    cancel: &CancellationToken,
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
        configs: &[super::PluginConfig],
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &tool_registry::execution::CancellationToken,
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
            let super::sandbox::Sandbox::MacOs { write_trees, .. } = &config.sandbox else {
                continue;
            };
            for path in write_trees {
                let root = path.canonicalize()?;
                if protected.iter().any(|p| p.starts_with(&root)) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "plugin writable tree includes a protected executable",
                    ));
                }
                let mut pending = vec![root];
                while let Some(directory) = pending.pop() {
                    if cancel.is_cancelled() || std::time::Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "plugin writable-tree audit cancelled or expired",
                        ));
                    }
                    for entry in std::fs::read_dir(directory)? {
                        if cancel.is_cancelled() || std::time::Instant::now() >= deadline {
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
        cancel: &tool_registry::execution::CancellationToken,
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
        cancel: &'a tool_registry::execution::CancellationToken,
    }
    extern "C" fn stopped(context: *mut c_void) -> i32 {
        // SAFETY: the synchronous C call retains this live stack value only for its duration.
        let state = unsafe { &*(context.cast::<Stop<'_>>()) };
        i32::from(state.cancel.is_cancelled() || std::time::Instant::now() >= state.deadline)
    }
    pub(crate) struct Child {
        control: Opaque,
        status: Option<ExitStatus>,
    }
    impl Child {
        pub(crate) fn kill(&mut self) -> io::Result<()> {
            // SAFETY: this uniquely owned record remains valid until Drop.
            checked(unsafe { alloy_owned_stop(self.control) })
        }
        pub(crate) fn wait(&mut self) -> io::Result<ExitStatus> {
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
    pub(super) fn spawn(
        cmd: Command,
        helper: Option<&std::path::Path>,
        deadline: std::time::Instant,
        cancel: &tool_registry::execution::CancellationToken,
        external_reaper: bool,
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
                checked(posix_spawn_file_actions_adddup2(
                    &mut actions.0,
                    source,
                    target,
                ))?;
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
