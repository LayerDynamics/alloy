//! Close-by-default plugin spawn, independent of descriptor allocation limits.
use super::{HostError, PluginConfig, sandbox};
#[cfg(target_os = "macos")]
pub(super) use macos::Child;
#[cfg(target_os = "linux")]
pub(super) use std::process::Child;
use std::process::{ChildStderr, ChildStdin, ChildStdout};
type Spawned = (Child, ChildStdin, ChildStdout, ChildStderr);
pub(super) fn spawn(config: &PluginConfig) -> Result<Spawned, HostError> {
    let command = sandbox::command(config)?;
    platform(command).map_err(HostError::io)
}
#[cfg(target_os = "linux")]
fn platform(mut cmd: std::process::Command) -> std::io::Result<Spawned> {
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
fn platform(cmd: std::process::Command) -> std::io::Result<Spawned> {
    macos::spawn(cmd)
}

#[cfg(target_os = "macos")]
mod macos {
    use super::Spawned;
    use std::{
        ffi::{CString, c_char, c_void},
        io,
        os::{
            fd::{AsRawFd, FromRawFd, OwnedFd},
            unix::{ffi::OsStrExt, process::ExitStatusExt},
        },
        process::{Command, ExitStatus},
    };
    type Opaque = *mut c_void;
    const POSIX_SPAWN_SETPGROUP: i16 = 0x0002;
    const POSIX_SPAWN_CLOEXEC_DEFAULT: i16 = 0x4000;
    #[link(name = "System")]
    unsafe extern "C" {
        fn pipe(fds: *mut i32) -> i32;
        fn fcntl(fd: i32, command: i32, ...) -> i32;
        fn kill(pid: i32, signal: i32) -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        fn posix_spawnattr_init(attr: *mut Opaque) -> i32;
        fn posix_spawnattr_destroy(attr: *mut Opaque) -> i32;
        fn posix_spawnattr_setflags(attr: *mut Opaque, flags: i16) -> i32;
        fn posix_spawnattr_setpgroup(attr: *mut Opaque, group: i32) -> i32;
        fn posix_spawn_file_actions_init(actions: *mut Opaque) -> i32;
        fn posix_spawn_file_actions_destroy(actions: *mut Opaque) -> i32;
        fn posix_spawn_file_actions_adddup2(actions: *mut Opaque, fd: i32, target: i32) -> i32;
        fn posix_spawn_file_actions_addclose(actions: *mut Opaque, fd: i32) -> i32;
        fn posix_spawn_file_actions_addchdir_np(actions: *mut Opaque, path: *const c_char) -> i32;
        fn posix_spawn(
            pid: *mut i32,
            path: *const c_char,
            actions: *const Opaque,
            attr: *const Opaque,
            argv: *const *mut c_char,
            envp: *const *mut c_char,
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
    fn pipe_pair() -> io::Result<(OwnedFd, OwnedFd)> {
        let mut raw = [-1; 2];
        // SAFETY: raw contains two writable ints; success creates owned descriptors.
        if unsafe { pipe(raw.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: successful pipe produced two distinct, newly owned descriptors.
        let pair = unsafe { (OwnedFd::from_raw_fd(raw[0]), OwnedFd::from_raw_fd(raw[1])) };
        fn prepare(fd: OwnedFd) -> io::Result<OwnedFd> {
            let owned = if fd.as_raw_fd() < 3 {
                // SAFETY: F_DUPFD duplicates this live descriptor to an available slot >=3.
                let copied = unsafe { fcntl(fd.as_raw_fd(), 0, 3) };
                if copied < 0 {
                    return Err(io::Error::last_os_error());
                }
                // SAFETY: copied is a new owned descriptor not owned elsewhere.
                unsafe { OwnedFd::from_raw_fd(copied) }
            } else {
                fd
            };
            // SAFETY: F_SETFD changes flags on this owned descriptor only.
            if unsafe { fcntl(owned.as_raw_fd(), 2, 1) } < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(owned)
        }
        Ok((prepare(pair.0)?, prepare(pair.1)?))
    }
    pub(crate) struct Child {
        pid: i32,
        status: Option<ExitStatus>,
    }
    impl Child {
        pub(crate) fn id(&self) -> u32 {
            self.pid as u32
        }
        pub(crate) fn kill(&mut self) -> io::Result<()> {
            // SAFETY: targets the exact still-owned, unreaped direct child.
            if unsafe { kill(self.pid, 9) } == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }
        pub(crate) fn wait(&mut self) -> io::Result<ExitStatus> {
            if let Some(status) = self.status {
                return Ok(status);
            }
            loop {
                let mut raw = 0;
                // SAFETY: raw is writable; only this exact direct child is reaped.
                let result = unsafe { waitpid(self.pid, &mut raw, 0) };
                if result == self.pid {
                    let status = ExitStatus::from_raw(raw);
                    self.status = Some(status);
                    return Ok(status);
                }
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            }
        }
    }
    pub(super) fn spawn(cmd: Command) -> io::Result<Spawned> {
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
        let (input_read, input_write) = pipe_pair()?;
        let (output_read, output_write) = pipe_pair()?;
        let (error_read, error_write) = pipe_pair()?;
        let mut attr = Attr::new()?;
        let mut actions = Actions::new()?;
        // SAFETY: opaque handles are initialized, C strings/FDs remain live through
        // spawn, and every C return is checked before continuing. Only 0/1/2 are
        // inherited. Explicit source-end closes avoid duplicate pipe capabilities.
        unsafe {
            checked(posix_spawnattr_setflags(
                &mut attr.0,
                POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETPGROUP,
            ))?;
            checked(posix_spawnattr_setpgroup(&mut attr.0, 0))?;
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
        let mut pid = 0;
        // SAFETY: argv/envp are null-terminated and all referenced storage and
        // initialized spawn objects remain valid until the call returns.
        checked(unsafe {
            posix_spawn(
                &mut pid,
                program.as_ptr(),
                &actions.0,
                &attr.0,
                argv.as_ptr(),
                environment.as_ptr(),
            )
        })?;
        let child = Child { pid, status: None };
        drop((input_read, output_write, error_write));
        Ok((
            child,
            input_write.into(),
            output_read.into(),
            error_read.into(),
        ))
    }
}
